import torch
from torch import nn
from dataclasses import dataclass
from transformers import GPT2Config, GPT2Model
from arch.model.ops import create_meta_predictors, right_shift_pad, HierAttnMasks
from arch.vocab import *

class MetaPredictor(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()

        self.hidden_size = config.n_embd

        self.bar_len_embedding, self.bar_len_head, self.bar_len_criterion = create_meta_predictors(
            META_BAR_LEN_VOCAB_SIZE, self.hidden_size, META_PAD_BAR_LEN
        )
        self.track_id_embedding, self.track_id_head, self.track_id_criterion = create_meta_predictors(
            META_TRACK_VOCAB_SIZE, self.hidden_size, META_PAD_TRACK
        )
        self.inst_embedding, self.new_inst_head, self.new_inst_criterion = create_meta_predictors(
            META_INST_VOCAB_SIZE, self.hidden_size, META_PAD_INST
        )

        self.bar_decoder = GPT2Model(config)
        self.track_decoder = GPT2Model(config)

        self.bar_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.harmo_bar_proj = nn.Linear(self.hidden_size, self.hidden_size)

        self.bar_decoder.wte = nn.Identity()
        self.track_decoder.wte = nn.Identity()

    def get_meta_emb(self, bar_len: torch.LongTensor, track_id, inst):
        device = bar_len.device
        bar_meta_emb = self.bar_len_embedding.forward(bar_len)

        harmo_track_id_emb = self.track_id_embedding.forward(
            torch.full((1, 1, 1), META_HARMO_TRACK_ID, dtype=torch.long, device=device)
        )
        harmo_track_meta_emb = bar_meta_emb.unsqueeze(-2) + harmo_track_id_emb

        track_id_emb = self.track_id_embedding.forward(track_id)
        inst_emb = self.inst_embedding.forward(inst)
        track_meta_emb = bar_meta_emb.unsqueeze(-2) + track_id_emb + inst_emb

        return bar_meta_emb, harmo_track_meta_emb, track_meta_emb

    def forward(
        self,
        harmo_bar_feature: torch.FloatTensor,
        bar_feature: torch.FloatTensor,
        track_feature: torch.FloatTensor,
        masks: HierAttnMasks,
        labels: list[torch.LongTensor] | None = None,
    ):
        B, Bars, Tracks, _ = masks.hier_event_mask.shape

        # --- Bar Level GPT Decoder ---
        bar_dec_out = self.bar_decoder.forward(
            inputs_embeds=right_shift_pad(torch.cat((harmo_bar_feature, bar_feature), dim=1))
            # NOTE: If song is short and trailing bars are empty, their bar feature will be zero
        ).last_hidden_state
        harmo_bar_dec_out = bar_dec_out[:, :Bars].reshape(B * Bars, -1)
        bar_dec_out = bar_dec_out[:, Bars:].reshape(B * Bars, -1)

        # --- Track Level GPT Decoder ---
        bar_context = self.bar_proj.forward(bar_dec_out)
        harmo_bar_context = self.harmo_bar_proj.forward(harmo_bar_dec_out)
        all_bar_context = bar_context + harmo_bar_context
        track_dec_out = self.track_decoder.forward(
            inputs_embeds=right_shift_pad(track_feature) + all_bar_context.unsqueeze(-2),
            attention_mask=masks.track_mask
        ).last_hidden_state.view(B * Bars * Tracks, -1) # from (B * Bars, Tracks, -1)
        harmo_track_dec_out = self.track_decoder.forward(
            inputs_embeds=harmo_bar_context.unsqueeze(-2) # Although only one track, this stabilizes training
        ).last_hidden_state.view(B * Bars, -1) # from (B * Bars, 1, -1)

        loss = None
        if labels is not None:
            bar_len, track_id, new_inst = labels
            bar_len_logits = self.bar_len_head.forward(harmo_bar_dec_out)
            track_id_logits = self.track_id_head.forward(track_dec_out)
            new_inst_logits = self.new_inst_head.forward(track_dec_out)
            bar_len_loss = self.bar_len_criterion.forward(bar_len_logits.flatten(0, -2), bar_len.flatten())
            track_id_loss = self.track_id_criterion.forward(track_id_logits.flatten(0, -2), track_id.flatten())
            new_inst_loss = self.new_inst_criterion.forward(new_inst_logits.flatten(0, -2), new_inst.flatten())
            new_inst_loss.nan_to_num_() # May happen when data is empty
            loss = bar_len_loss + track_id_loss + new_inst_loss

        return MetaPredictorOutput(
            harmo_track_dec_out=harmo_track_dec_out,
            track_dec_out=track_dec_out,
            all_bar_context=all_bar_context,
            loss=loss,
            bar_len_logits=bar_len_logits,
            track_id_logits=track_id_logits,
            new_inst_logits=new_inst_logits,
        )

@dataclass
class MetaPredictorOutput:
    harmo_track_dec_out: torch.FloatTensor | None = None
    track_dec_out: torch.FloatTensor | None = None
    all_bar_context: torch.FloatTensor | None = None
    loss: torch.FloatTensor | None = None
    bar_len_logits: torch.FloatTensor | None = None
    track_id_logits: torch.FloatTensor | None = None
    new_inst_logits: torch.FloatTensor | None = None
