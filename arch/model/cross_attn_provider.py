import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
from arch.model.hidden_with_mask import HiddenWithMask

class CrossAttnProvider:
    # We hardcode the possible name of masks to be friendly to torch.compile
    track_prev_bar_mask: torch.BoolTensor | None = None
    prev_track_mask: torch.BoolTensor | None = None
    prev_bar_mask: torch.BoolTensor | None = None
    harmo_event_mask: torch.BoolTensor | None = None

    def __init__(self):
        raise NotImplementedError

    def forward(self, hidden_states: torch.FloatTensor, layer_idx: int):
        raise NotImplementedError

class CrossAttnProviderForInfer(CrossAttnProvider):
    hiddens: list[HiddenWithMask]
    active_idx: torch.LongTensor | None = None

    def forward(self, _, layer_idx: int):
        hidden, mask = self.retrieve(layer_idx)
        if self.active_idx is not None:
            hidden = hidden[self.active_idx, ...]
            if mask is not None:
                mask = mask[self.active_idx, ...]
        return hidden, mask

    def retrieve(self, layer_idx: int):
        raise NotImplementedError

def prepare_masks(provider: CrossAttnProvider, attn_implementation, dtype, tgt_len):
    """ Transform mask to [Batch, 1, TgtLen, SrcLen] (SDPA requirement) """
    if attn_implementation != "sdpa":
        raise ValueError(f"{attn_implementation=} is not tested")
    if isinstance(provider, CrossAttnProviderForInfer):
        for hidden in provider.hiddens:
            if hidden.mask is None or hidden.mask.ndim == 4: # Already prepared
                continue
            hidden.mask = _prepare_4d_attention_mask_for_sdpa(mask=hidden.mask, dtype=dtype, tgt_len=tgt_len)
    else:
        if provider.track_prev_bar_mask is not None:
            provider.track_prev_bar_mask = _prepare_4d_attention_mask_for_sdpa(mask=provider.track_prev_bar_mask, dtype=dtype, tgt_len=tgt_len)
        if provider.prev_track_mask is not None:
            provider.prev_track_mask = _prepare_4d_attention_mask_for_sdpa(mask=provider.prev_track_mask, dtype=dtype, tgt_len=tgt_len)
        if provider.prev_bar_mask is not None:
            provider.prev_bar_mask = _prepare_4d_attention_mask_for_sdpa(mask=provider.prev_bar_mask, dtype=dtype, tgt_len=tgt_len)
        if provider.harmo_event_mask is not None:
            provider.harmo_event_mask = _prepare_4d_attention_mask_for_sdpa(mask=provider.harmo_event_mask, dtype=dtype, tgt_len=tgt_len)
