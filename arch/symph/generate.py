import torch
from tqdm import tqdm
from arch.model.event_decoder import EventDecoderWithLMHead
from arch.model.cross_attn_provider import CrossAttnProviderForInfer
from arch.model.hidden_with_mask import HiddenWithMask, LastHiddenWithMask
from arch.symph.conditions import HarmCondition, NextBarCondition
from arch.symph.modules.cross_attn_provider import CrossAttnProviderForInferHarm, CrossAttnProviderForInfer2S, CrossAttnProviderForInfer3S, CrossAttnProviderForInferMel
from arch.symph.modules.track_manager import TrackManager
from arch.symph.vocab import *
from arch.symph.model import MusicModel3D
from arch.symph.sampling import *
from arch.vocab import *
from arch.config import *
from arch.symph.config import *

# Set by entry points (e.g. arch/symph/generator.py, Results/batch_filter_gen.py):
# mask out the piano instrument family (programs 0-4) during generation
FORBID_PIANO = False

# Sampling parameters for the metadata heads (track ID, instrument)
META_TEMPERATURE = 0.8
META_TOP_P = 0.95
# Sampling parameter for the music event head
EVENT_TOP_P = 0.99

def generate_bars(
    model: MusicModel3D, device,
    harmo_cond: HarmCondition,
    bar_num=BAR_NUM, track_num=TRACK_NUM, event_num=EVENT_NUM,
    desc=None
):
    B, D = harmo_cond.bar_meta_emb.shape[0], model.hidden_size
    L = model.event_decoder.config.n_layer

    this_bar_feature = harmo_cond.last_bar_feature
    past_key_values = harmo_cond.past_key_values

    track_manager = TrackManager(B, track_num, device=device)
    prev_bar_hidden = HiddenWithMask.zeros((B, track_num, event_num, D), L + 1, device=device)
    active_pitch_buffer = torch.zeros((B, DUR_NUM, PITCH_NUM), dtype=torch.float, device=device)

    events = torch.full((B, bar_num, track_num, event_num), PAD_EVENT, dtype=torch.long, device=device)

    for bar_id in tqdm(range(bar_num), desc=desc, mininterval=TQDM_MIN_INTERVAL):
        output = model.meta_predictor.bar_decoder.forward(
            inputs_embeds=this_bar_feature.unsqueeze(-2),
            past_key_values=past_key_values if bar_id == 0 else output.past_key_values, use_cache=True,
        )
        bar_dec_out = output.last_hidden_state[:, -1]
        bar_context = model.meta_predictor.bar_proj.forward(bar_dec_out)
        bar_meta_emb = harmo_cond.bar_meta_emb[:, bar_id]
        all_bar_context = bar_context + harmo_cond.harmo_bar_context[:, bar_id]
        harmo_hidden = harmo_cond.harmo_hidden[:, bar_id]
        bar_len = harmo_cond.bar_len[:, bar_id]
        harmo_beat_mask = harmo_cond.harmo_beat_mask[:, bar_id]

        bar_events, this_bar_hidden, this_bar_track_feature = generate_bar_tracks(
            model,
            NextBarCondition(
                track_manager, bar_meta_emb, all_bar_context, harmo_hidden,
                prev_bar_hidden,
                bar_len, harmo_beat_mask,
            ),
            active_pitch_buffer,
            track_num=track_num, event_num=event_num
        )

        this_bar_feature = model.hier_encoder.encode_tracks(
            this_bar_track_feature.hidden, this_bar_track_feature.mask, is_harmo=False
        )

        prev_bar_hidden = this_bar_hidden

        events[:, bar_id] = bar_events

        if DEBUG_GEN:
            debug_events(bar_events[0])
            break

    track_id_data, inst_data = track_manager.pack_track_data_tensor()
    return events, track_id_data, inst_data

def generate_bar_tracks(
    model: MusicModel3D,
    cond: NextBarCondition,
    active_pitch_buffer,
    track_num=TRACK_NUM, event_num=EVENT_NUM,
    prepare_next_bar=True,
):
    B, D = cond.all_bar_context.shape
    L = model.event_decoder.config.n_layer
    device = cond.all_bar_context.device

    track_manager = cond.track_manager
    track_manager.init_bar()
    existing_track_mask = torch.zeros((B, META_TRACK_VOCAB_SIZE), dtype=torch.bool, device=device)

    this_track_feature = torch.zeros(B, D, device=device)
    active_batch = torch.ones(B, dtype=torch.bool, device=device)
    prev_track_hidden = HiddenWithMask.zeros((B, event_num, D), L + 1, device=device)

    if prepare_next_bar:
        this_bar_hidden = HiddenWithMask.zeros((B, track_num, event_num, D), L + 1, device=device)
        this_bar_track_feature = LastHiddenWithMask.zeros((B, track_num, D), device=device)
    bar_events = torch.full((B, track_num, event_num), PAD_EVENT, dtype=torch.long, device=device)

    for tid in range(track_num):
        output = model.meta_predictor.track_decoder.forward(
            inputs_embeds=(this_track_feature + cond.all_bar_context).unsqueeze(-2),
            past_key_values=None if tid == 0 else output.past_key_values, use_cache=True,
        )
        track_dec_out = output.last_hidden_state[:, -1]

        track_id_logits = model.meta_predictor.track_id_head.forward(track_dec_out)
        # Forbid already-generated track_ids in the prediction
        track_id_logits.masked_fill_(existing_track_mask, -1e9)
        track_id = sample_from_logits(track_id_logits, temperature=META_TEMPERATURE, top_p=META_TOP_P)
        active_batch &= (track_id != META_END_OF_BAR)
        if not active_batch.any():
            break
        existing_track_mask.scatter_(1, track_id.unsqueeze(-1), True)
        inst_logits = model.meta_predictor.new_inst_head(track_dec_out)
        if FORBID_PIANO:
            inst_logits[:, :5] = -1e9
        new_inst = sample_from_logits(inst_logits, temperature=META_TEMPERATURE, top_p=META_TOP_P)
        track_manager.update_track(track_id, new_inst, active_batch)
        inst = track_manager.bar_inst[:, tid]

        track_id_emb = model.meta_predictor.track_id_embedding.forward(track_id)
        inst_emb = model.meta_predictor.inst_embedding.forward(inst)
        track_meta_emb = cond.bar_meta_emb + track_id_emb + inst_emb

        track_prev_bar_hidden = HiddenWithMask.zeros((B, event_num, D), L + 1, device=device)
        prev_tid = track_manager.get_track_prev_map(track_id)
        has_prev = (prev_tid != -1)
        if has_prev.any():
            track_prev_bar_hidden[has_prev] = cond.prev_bar_hidden[has_prev, prev_tid[has_prev]]

        track_context = track_meta_emb + model.track_proj.forward(track_dec_out)

        if model.cross_attn_stream == 3:
            cross_attn_provider = CrossAttnProviderForInfer3S(track_prev_bar_hidden, prev_track_hidden, cond.harmo_hidden)
        elif model.cross_attn_stream == 2:
            cross_attn_provider = CrossAttnProviderForInfer2S(track_prev_bar_hidden, cond.harmo_hidden)
        elif model.cross_attn_stream == "harmony":
            cross_attn_provider = CrossAttnProviderForInferHarm(cond.harmo_hidden)
        elif model.cross_attn_stream == "melody":
            cross_attn_provider = CrossAttnProviderForInferMel(track_prev_bar_hidden)

        this_track_events, this_track_hidden = generate_track_events(
            model.event_decoder, track_context, cross_attn_provider, active_batch,
            inst, cond.bar_len, cond.harmo_beat_mask, active_pitch_buffer,
            event_num=event_num,
        )

        this_track_feature = model.hier_encoder.encode_events(this_track_events, this_track_hidden.mask, track_meta_emb, is_harmo=False)

        prev_track_hidden = this_track_hidden

        if prepare_next_bar:
            this_bar_hidden[:, tid] = this_track_hidden
            this_bar_track_feature[:, tid] = LastHiddenWithMask(this_track_feature, active_batch)
        bar_events[active_batch, tid] = this_track_events[active_batch]

    track_manager.update_bar()
    can_add_end = track_manager.tid_counter < track_num
    if can_add_end.any():
        can_add_batch_id = torch.where(can_add_end)[0]
        target_tid = track_manager.tid_counter[can_add_end]
        track_manager.bar_track_id[can_add_batch_id, target_tid] = META_END_OF_BAR
        if prepare_next_bar:
            # We manually encode the END_OF_BAR event
            end_of_bar_event = torch.full((B, 1), END_OF_BAR, dtype=torch.long, device=device)
            mask = None
            end_of_bar_feature = model.hier_encoder.encode_events(end_of_bar_event, mask, cond.bar_meta_emb, is_harmo=False)
            this_bar_track_feature.hidden[can_add_batch_id, target_tid] = end_of_bar_feature[can_add_end]
            this_bar_track_feature.mask[can_add_batch_id, target_tid] = True
        bar_events[can_add_batch_id, target_tid, 0] = END_OF_BAR

    if prepare_next_bar:
        if DissonanceConstrainer.enabled:
            # NOTE: We can generate all batches in the same bar length
            # to make it vectorizable; for now we keep the slow loop
            for b in range(B):
                shift = cond.bar_len[b].item()
                if shift < DUR_NUM:
                    active_pitch_buffer[b, :DUR_NUM - shift] = active_pitch_buffer[b, shift:].clone()
                    active_pitch_buffer[b, DUR_NUM - shift:] = 0.0
                else:
                    active_pitch_buffer[b] = 0.0
        return bar_events, this_bar_hidden, this_bar_track_feature
    return bar_events

def generate_track_events(
    decoder: EventDecoderWithLMHead,
    track_context: torch.FloatTensor,
    cross_attn_provider: CrossAttnProviderForInfer,
    parent_active_batch,
    inst, bar_len, harmo_beat_mask, active_pitch_buffer,
    event_num=EVENT_NUM,
):
    B, D = track_context.shape
    L = decoder.config.n_layer
    device = track_context.device

    # Input
    active_idx = torch.where(parent_active_batch)[0]
    this_e_embed = torch.zeros(len(active_idx), D, device=device)
    track_context = track_context[active_idx]
    cross_attn_provider.active_idx = active_idx

    # Constraint
    inst_range_constraint = InstRangeConstrainer(inst[active_idx])
    pos_constraint = PosConstrainer(bar_len[active_idx])
    diss_constraint = DissonanceConstrainer(active_idx, harmo_beat_mask, active_pitch_buffer)

    if DEBUG_DISS_CONSTRAINER:
        print("New Track")

    # Output
    track_events = torch.full((B, event_num), PAD_EVENT, dtype=torch.long, device=device)
    this_track_hidden = HiddenWithMask.zeros((B, event_num, D), L + 1, device=device)

    for eid in range(event_num):
        inputs_embeds = (this_e_embed + track_context).unsqueeze(-2)
        output = decoder.forward(
            inputs_embeds,
            cross_attn_provider=cross_attn_provider,
            output_hidden_states=True,
            past_key_values=None if eid == 0 else output.past_key_values, use_cache=True,
        )

        logits = output.logits[:, -1]

        inst_range_constraint.apply_(logits)
        pos_constraint.apply_(logits)
        diss_constraint.apply_(logits)
        this_event = sample_from_logits(logits, top_p=EVENT_TOP_P)
        pos_constraint.update_(this_event)
        diss_constraint.update_(this_event)

        if DEBUG_DISS_CONSTRAINER:
            print(event_vocab.idx2word[this_event[0].item()])

        track_events[active_idx, eid] = this_event
        this_track_hidden[active_idx, eid: eid + 1] = HiddenWithMask(output.hidden_states, (this_event != PAD_EVENT).unsqueeze(-1))

        finished = (this_event == END_EVENT)
        if finished.any():
            keep = ~finished
            active_idx = active_idx[keep]
            if len(active_idx) == 0:
                break
            this_event = this_event[keep]
            track_context = track_context[keep]
            cross_attn_provider.active_idx = active_idx
            output.past_key_values.batch_select_indices(keep)
            inst_range_constraint.batch_select_indices(keep)
            pos_constraint.batch_select_indices(keep)
            diss_constraint.batch_select_indices(keep)

        this_e_embed = decoder.transformer.wte(this_event)

    return track_events, this_track_hidden

def debug_events(bar_events):
    words = event_vocab.idx_seq2word_seq([event for track_events in bar_events for event in track_events])
    with open("tmp.txt", "w") as f:
        f.write("\n".join(words))
    print("Debug: generated words saved in tmp.txt")
    __import__("pdb").set_trace()
