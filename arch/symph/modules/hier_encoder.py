import torch
from torch import nn
from transformers import BertConfig, BertModel
from arch.vocab import *
from arch.model.ops import mean_pooling, HierAttnMasks

class HierarchicalEncoder(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()

        self.event_encoder = BertModel(config, add_pooling_layer=False)
        self.track_encoder = BertModel(config, add_pooling_layer=False)

        self.track_encoder.embeddings.word_embeddings = nn.Identity()

    def forward(self, events: torch.LongTensor, masks: HierAttnMasks, track_meta_emb: torch.FloatTensor, is_harmo: bool):
        B, Bars, Tracks, _ = masks.hier_event_mask.shape
        track_feature = self.encode_events(
            events, masks.event_mask, track_meta_emb, is_harmo=is_harmo
        ).view(B * Bars, Tracks, -1) # from (B * Bars * Tracks, -1)
        bar_feature = self.encode_tracks(
            track_feature, masks.track_mask, is_harmo=is_harmo
        ).view(B, Bars, -1) # from (B * Bars, -1)
        return track_feature, bar_feature

    def encode_events(self, events: torch.LongTensor, event_mask: torch.BoolTensor | None, track_meta_emb: torch.FloatTensor, is_harmo: bool):
        e_embed = self.event_encoder.embeddings.forward(input_ids=events)
        event_enc_out = self.event_encoder.forward(
            inputs_embeds=e_embed + track_meta_emb.unsqueeze(-2),
            token_type_ids=(torch.zeros if is_harmo else torch.ones)(*events.shape, dtype=torch.long, device=events.device),
            attention_mask=event_mask
        ).last_hidden_state
        track_feature = mean_pooling(event_enc_out, event_mask)
        return track_feature

    def encode_tracks(self, track_feature: torch.FloatTensor, track_mask: torch.BoolTensor, is_harmo: bool):
        track_enc_out = self.track_encoder.forward(
            inputs_embeds=track_feature,
            token_type_ids=(torch.zeros if is_harmo else torch.ones)(*track_feature.shape[:-1], dtype=torch.long, device=track_feature.device),
            attention_mask=track_mask
        ).last_hidden_state
        bar_feature = mean_pooling(track_enc_out, track_mask)
        return bar_feature
