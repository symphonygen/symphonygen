import json
import torch
import torch.nn as nn
from transformers import BertConfig, GPT2Config
from arch.base_class import ModelBase
from arch.config import *
from arch.model.event_decoder import EventDecoderWithLMHead
from arch.model.hidden_with_mask import HiddenWithMask, LastHiddenWithMask
from arch.model.ops import create_hier_attn_masks, right_shift_pad
from arch.symph.config import *
from arch.symph.modules.cross_attn_provider import *
from arch.symph.modules.hier_encoder import HierarchicalEncoder, HierAttnMasks
from arch.symph.modules.meta_predictor import MetaPredictor
from arch.symph.vocab import *
from arch.vocab import *
from data_prep.data_structure import *

class MusicModel3D(ModelBase):
    def __init__(
        self,
        bert_config: BertConfig,
        meta_gpt_config: GPT2Config,
        event_gpt_config: GPT2Config,
        harmo_event_gpt_config: GPT2Config,
        shape=(BAR_NUM, TRACK_NUM, EVENT_NUM, HARMO_EVENT_NUM),
        cross_attn_stream=CROSS_ATTN_STREAM,
        meta_loss_factor=0.05,
        harmo_loss_factor=0.5,
    ):
        super().__init__()
        self.hidden_size = event_gpt_config.n_embd

        self.hier_encoder = HierarchicalEncoder(bert_config)
        self.meta_predictor = MetaPredictor(meta_gpt_config)

        self.event_decoder = EventDecoderWithLMHead(event_gpt_config)
        self.harmo_event_decoder = EventDecoderWithLMHead(harmo_event_gpt_config)
        # NOTE: Shared event embedding
        self.harmo_event_decoder.transformer.wte.weight = self.event_decoder.transformer.wte.weight

        self.track_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.harmo_track_proj = nn.Linear(self.hidden_size, self.hidden_size)

        bar_num, track_num, event_num, harmo_event_num = shape
        self.harmo_cross_attn_provider = HarmCrossAttnProvider((bar_num, track_num, harmo_event_num, self.hidden_size))
        self.cross_attn_stream = cross_attn_stream
        if self.cross_attn_stream == 3:
            self.cross_attn_provider = CrossAttnProvider3S((bar_num, track_num, event_num, self.hidden_size), harmo_event_num)
        elif self.cross_attn_stream == 2:
            self.cross_attn_provider = CrossAttnProvider2S((bar_num, track_num, event_num, self.hidden_size), harmo_event_num)
        elif self.cross_attn_stream == "harmony":
            self.cross_attn_provider = CrossAttnProviderHarm((bar_num, track_num, event_num, self.hidden_size), harmo_event_num)
        elif self.cross_attn_stream == "melody":
            self.cross_attn_provider = CrossAttnProviderMel((bar_num, track_num, event_num, self.hidden_size))
        else:
            raise NotImplementedError

        self.meta_loss_factor = meta_loss_factor
        self.harmo_loss_factor = harmo_loss_factor

    def forward(
        self,
        hier_events: torch.LongTensor,
        hier_harmo_events: torch.LongTensor, # [B, Bars, 1, HarmEvents]
        bar_len, track_id, inst, new_inst,
        track_prev_map: torch.LongTensor,
        return_hidden_states=False,
        return_outputs=False,
    ):
        return_loss = not return_hidden_states

        events = hier_events.flatten(0, -2)
        harmo_events = hier_harmo_events.flatten(0, -2)
        masks = create_hier_attn_masks(hier_events)
        harmo_masks = create_hier_attn_masks(hier_harmo_events)
        if return_loss:
            # the ending track is not needed for event prediction, but for meta prediction
            event_labels = events.clone()
            event_labels[event_labels == END_OF_BAR] = PAD_EVENT

        # XXX: Here track_meta_emb already includes bar_meta_emb
        hier_bar_meta_emb, hier_harmo_track_meta_emb, hier_track_meta_emb = self.meta_predictor.get_meta_emb(bar_len, track_id, inst)
        harmo_track_meta_emb = hier_harmo_track_meta_emb.flatten(0, -2)
        track_meta_emb = hier_track_meta_emb.flatten(0, -2)

        meta_outputs = self.meta_stage(
            events, masks, harmo_events, harmo_masks,
            track_meta_emb, harmo_track_meta_emb,
            labels=(bar_len, track_id, new_inst) if return_loss else None,
        )
        harmo_track_context = harmo_track_meta_emb + self.harmo_track_proj.forward(meta_outputs.harmo_track_dec_out)
        harmo_dec_outputs = self.harmony_stage(
            harmo_events, harmo_masks,
            harmo_track_context,
            labels=harmo_events if return_loss else None,
        )
        track_context = track_meta_emb + self.track_proj.forward(meta_outputs.track_dec_out)
        harmo_hidden = LastHiddenWithMask(harmo_dec_outputs.hidden_states[HARMO_LAST_LAYER], harmo_masks.event_mask)
        event_dec_outputs = self.symphony_stage(
            events, masks,
            track_context,
            track_prev_map,
            harmo_hidden,
            labels=event_labels if return_loss else None,
            output_hidden_states=return_hidden_states,
        )

        if return_hidden_states:
            hier_all_bar_context = meta_outputs.all_bar_context.view(*hier_bar_meta_emb.shape)
            hier_harmo_hidden = harmo_hidden.view(*hier_harmo_events.shape, -1)
            hier_event_hidden = HiddenWithMask(event_dec_outputs.hidden_states, masks.event_mask).view(*hier_events.shape, -1)
            return HierarchicalMusicModelHiddenStates(hier_bar_meta_emb, hier_all_bar_context, hier_harmo_hidden, hier_event_hidden)
        if return_outputs:
            return meta_outputs, event_dec_outputs, event_labels

        loss = (
            event_dec_outputs.loss +
            self.meta_loss_factor * meta_outputs.loss +
            self.harmo_loss_factor * harmo_dec_outputs.loss
        )
        return {
            "loss": loss,
            "metrics": {
                "music_loss": event_dec_outputs.loss,
                "meta_loss": meta_outputs.loss,
                "harmony_loss": harmo_dec_outputs.loss
            }
        }

    def meta_stage(self,
        events, masks, harmo_events, harmo_masks,
        track_meta_emb, harmo_track_meta_emb,
        labels=None,
    ):
        track_feature, bar_feature = self.hier_encoder.forward(events, masks, track_meta_emb, is_harmo=False)
        _, harmo_bar_feature = self.hier_encoder.forward(harmo_events, harmo_masks, harmo_track_meta_emb, is_harmo=True)
        meta_outputs = self.meta_predictor.forward(
            harmo_bar_feature, bar_feature, track_feature, masks,
            labels=labels,
        )
        return meta_outputs

    def harmony_stage(
        self,
        harmo_events: torch.LongTensor, harmo_masks: HierAttnMasks,
        harmo_track_context: torch.FloatTensor,
        labels=None,
    ):
        harmo_e_embed = self.harmo_event_decoder.transformer.wte(harmo_events)
        inputs_embeds = right_shift_pad(harmo_e_embed) + harmo_track_context.unsqueeze(-2)
        self.harmo_cross_attn_provider.prepare(harmo_masks.hier_event_mask)
        attention_mask = harmo_masks.event_mask
        harmo_dec_outputs = self.harmo_event_decoder.forward(
            inputs_embeds,
            cross_attn_provider=self.harmo_cross_attn_provider,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        return harmo_dec_outputs

    def symphony_stage(
        self,
        events: torch.LongTensor, masks: HierAttnMasks,
        track_context: torch.FloatTensor,
        track_prev_map: torch.LongTensor,
        harmo_hidden: LastHiddenWithMask,
        labels=None,
        output_hidden_states=False,
    ):
        e_embed = self.event_decoder.transformer.wte(events)
        # XXX: The last event's hidden state won't be cross-attended due to the right shift,
        # usually the last event is end event, which is fine,
        # but when the sequence is truncated, it will impact the model a little bit.
        inputs_embeds = right_shift_pad(e_embed) + track_context.unsqueeze(-2)
        if self.cross_attn_stream == 3:
            self.cross_attn_provider.prepare(masks.hier_event_mask, track_prev_map, harmo_hidden)
        elif self.cross_attn_stream == 2:
            self.cross_attn_provider.prepare(masks.hier_event_mask, track_prev_map, harmo_hidden)
        elif self.cross_attn_stream == "harmony":
            self.cross_attn_provider.prepare(harmo_hidden)
        elif self.cross_attn_stream == "melody":
            self.cross_attn_provider.prepare(masks.hier_event_mask, track_prev_map)
        attention_mask = masks.event_mask
        event_dec_outputs = self.event_decoder.forward(
            inputs_embeds,
            cross_attn_provider=self.cross_attn_provider,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
        )
        return event_dec_outputs

    def submodules(self):
        return {
            'Dec_Event': self.event_decoder.transformer,
            'Dec_Harm_Event': self.harmo_event_decoder.transformer,
            'Enc_Event': self.hier_encoder.event_encoder,
            'Enc_Track': self.hier_encoder.track_encoder,
            'Dec_Bar': self.meta_predictor.bar_decoder,
            'Dec_Track': self.meta_predictor.track_decoder,
        }

    @classmethod
    def from_config_dict(cls, config: dict, device='auto'):
        if device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {device}")

        cross_attn_stream = config.get('cross_attn_stream', CROSS_ATTN_STREAM)
        bert_config = BertConfig.from_dict(config['bert_config'])
        meta_gpt_config = GPT2Config.from_dict(config['meta_gpt_config'])
        event_gpt_config = GPT2Config.from_dict(config['event_gpt_config'])
        harmo_event_gpt_config = GPT2Config.from_dict(config['harmo_event_gpt_config'])

        print(f"Cross-Attn Stream: {cross_attn_stream}")
        print(f"Size: {SMALL_SIZE=}")

        model = cls(
            bert_config, meta_gpt_config, event_gpt_config, harmo_event_gpt_config,
            cross_attn_stream=cross_attn_stream,
        )
        if device is not None:
            model = model.to(device)
        return model

    def dump_json_config(self, config_json: str):
        config_dict = {
            'cross_attn_stream': self.cross_attn_stream,
            'bert_config': self.hier_encoder.event_encoder.config.to_dict(),
            'meta_gpt_config': self.meta_predictor.bar_decoder.config.to_dict(),
            'event_gpt_config': self.event_decoder.config.to_dict(),
            'harmo_event_gpt_config': self.harmo_event_decoder.config.to_dict()
        }
        with open(config_json, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def from_python_config(
        cls,
        device='auto',
        hidden_size=HIDDEN_SIZE,
        encoder_layers=ENCODER_LAYERS,
        bar_track_decoder_layers=BAR_TRACK_DECODER_LAYERS,
        dec_layers=EVENT_DECODER_LAYERS,
        harmo_dec_layers=HARMO_EVENT_DECODER_LAYERS,
        heads=HEADS,
        shape=(BAR_NUM, TRACK_NUM, EVENT_NUM, HARMO_EVENT_NUM),
        cross_attn_stream=CROSS_ATTN_STREAM,
        meta_loss_factor=0.05,
        harmo_loss_factor=0.5,
    ):
        if device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {device}")

        bar_num, track_num, event_num, harmo_event_num = shape

        bert_config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=encoder_layers,
            num_attention_heads=heads,
            intermediate_size=hidden_size * 4,
            vocab_size=event_vocab.size,
            max_position_embeddings=max(event_num, harmo_event_num),
            type_vocab_size=2,
            pad_token_id=PAD_EVENT
        )

        meta_gpt_config = GPT2Config(
            n_embd=hidden_size,
            n_layer=bar_track_decoder_layers,
            n_head=heads,
            n_inner=hidden_size * 4,
            vocab_size=0,
            n_positions=max(2 * bar_num, track_num)
        )

        event_gpt_config = GPT2Config(
            n_embd=hidden_size,
            n_layer=dec_layers,
            n_head=heads,
            n_inner=hidden_size * 4,
            vocab_size=event_vocab.size,
            n_positions=event_num,
            pad_token_id=PAD_EVENT,
            add_cross_attention=True
        )

        harmo_event_gpt_config = GPT2Config(
            n_embd=hidden_size,
            n_layer=harmo_dec_layers,
            n_head=heads,
            n_inner=hidden_size * 4,
            vocab_size=event_vocab.size,
            n_positions=harmo_event_num,
            pad_token_id=PAD_EVENT,
            add_cross_attention=True
        )

        print(f"Cross-Attn Stream: {cross_attn_stream}")
        print(f"Size: {SMALL_SIZE=}")

        model = cls(
            bert_config, meta_gpt_config, event_gpt_config, harmo_event_gpt_config,
            shape=shape,
            cross_attn_stream=cross_attn_stream,
            meta_loss_factor=meta_loss_factor,
            harmo_loss_factor=harmo_loss_factor,
        )
        if device is not None:
            model = model.to(device)
        return model

@dataclass
class HierarchicalMusicModelHiddenStates:
    bar_meta_emb: torch.FloatTensor
    all_bar_context: torch.FloatTensor
    harmo_hidden: LastHiddenWithMask
    event_hidden: HiddenWithMask
