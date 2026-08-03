import torch
from arch.config import HARMO_LAST_LAYER
from arch.model.hidden_with_mask import LastHiddenWithMask
from arch.model.ops import create_harmo_beat_mask, create_hier_attn_masks, right_shift_pad
from arch.symph.conditions import HarmCondition
from arch.symph.data_tensor import MusicTensorConverter3D
from arch.symph.generate import generate_bars
from arch.symph.generator import SymphonyGenerator
from arch.symph.model import MusicModel3D
from arch.vocab import *
from data_prep.data_structure import BarStruct

def prepare_harmony_cond(model: MusicModel3D, hier_harmo_events: torch.LongTensor, bar_len):
    B, Bars, _, HarmEvents = hier_harmo_events.shape
    device = hier_harmo_events.device
    meta_pred = model.meta_predictor

    harmo_events = hier_harmo_events.flatten(0, -2)
    harmo_masks = create_hier_attn_masks(hier_harmo_events)

    bar_meta_emb = meta_pred.bar_len_embedding.forward(bar_len)
    harmo_track_id_emb = meta_pred.track_id_embedding.forward(
        torch.full((1, 1, 1), META_HARMO_TRACK_ID, dtype=torch.long, device=device)
    )
    hier_harmo_track_meta_emb = bar_meta_emb.unsqueeze(-2) + harmo_track_id_emb
    harmo_track_meta_emb = hier_harmo_track_meta_emb.flatten(0, -2)

    _, harmo_bar_feature = model.hier_encoder.forward(
        harmo_events, harmo_masks, harmo_track_meta_emb, is_harmo=True
    )

    bar_dec_outputs = meta_pred.bar_decoder.forward(
        inputs_embeds=right_shift_pad(harmo_bar_feature),
        use_cache=True # We need to save past_key_values as part of condition
    )
    harmo_bar_dec_out = bar_dec_outputs.last_hidden_state.view(B * Bars, -1) # from (B, Bars, -1)
    harmo_bar_context = meta_pred.harmo_bar_proj.forward(harmo_bar_dec_out)
    harmo_track_dec_out = meta_pred.track_decoder.forward(
        inputs_embeds=harmo_bar_context.unsqueeze(-2)
    ).last_hidden_state.view(B * Bars, -1) # from (B * Bars, 1, -1)

    harmo_track_context = harmo_track_meta_emb + model.harmo_track_proj.forward(harmo_track_dec_out)
    harmo_dec_outputs = model.harmony_stage(
        harmo_events, harmo_masks, harmo_track_context
    )

    harmo_hidden = LastHiddenWithMask(
        harmo_dec_outputs.hidden_states[HARMO_LAST_LAYER], harmo_masks.event_mask
    ).view(B, Bars, HarmEvents, -1)
    return HarmCondition(
        last_bar_feature=harmo_bar_feature[:, -1],
        past_key_values=bar_dec_outputs.past_key_values,
        bar_meta_emb=bar_meta_emb,
        harmo_bar_context=harmo_bar_context.view(B, Bars, -1),
        harmo_hidden=harmo_hidden,
    )

class SymphonyGenerator3D(SymphonyGenerator):
    converter: MusicTensorConverter3D

    def generate_given_harmony(self, model: MusicModel3D, all_harmo_list: list[list[BarStruct]], group_size: int):
        device = next(model.parameters()).device

        harmo_events_list = []
        bar_lens_list = []
        for harmo_bars in all_harmo_list:
            _, harmo_events, bar_len, _, _, _, _ = self.converter.pack_data_tensor(harmo_bars)
            harmo_events_list.append(harmo_events)
            bar_lens_list.append(bar_len)
        harmo_events_batch = torch.stack(harmo_events_list).to(device)
        bar_len_batch = torch.stack(bar_lens_list).to(device)

        harmo_cond = prepare_harmony_cond(model, harmo_events_batch, bar_len_batch)
        harmo_cond.bar_len = bar_len_batch
        harmo_cond.harmo_beat_mask = create_harmo_beat_mask(all_harmo_list).to(device)
        harmo_cond = harmo_cond.expand(group_size)

        events, track_id_data, inst_data = generate_bars(
            model, device,
            harmo_cond,
            desc=f"Generating for {len(all_harmo_list)} conditions"
        )

        return events, harmo_cond.bar_len, track_id_data, inst_data
