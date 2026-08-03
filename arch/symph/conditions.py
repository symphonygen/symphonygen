import torch
from dataclasses import dataclass
from transformers import DynamicCache
from arch.model.hidden_with_mask import HiddenWithMask, LastHiddenWithMask
from arch.symph.modules.track_manager import TrackManager

@dataclass
class HarmCondition:
    last_bar_feature: torch.FloatTensor | None = None
    past_key_values: DynamicCache | None = None # For bar decoder

    bar_meta_emb: torch.FloatTensor | None = None
    harmo_bar_context: torch.FloatTensor | None = None
    harmo_hidden: LastHiddenWithMask | None = None

    # Used in sampling constraints
    bar_len: torch.LongTensor | None = None
    harmo_beat_mask: torch.BoolTensor | None = None

    def expand(self, group_size):
        for layer in self.past_key_values.layers:
            layer.keys = layer.keys.repeat_interleave(group_size, dim=0)
            layer.values = layer.values.repeat_interleave(group_size, dim=0)

        return HarmCondition(
            last_bar_feature=self.last_bar_feature.repeat_interleave(group_size, dim=0),
            past_key_values=self.past_key_values,

            bar_meta_emb=self.bar_meta_emb.repeat_interleave(group_size, dim=0),
            harmo_bar_context=self.harmo_bar_context.repeat_interleave(group_size, dim=0),
            harmo_hidden=self.harmo_hidden.repeat_interleave(group_size, dim=0),

            bar_len=self.bar_len.repeat_interleave(group_size, dim=0),
            harmo_beat_mask=self.harmo_beat_mask.repeat_interleave(group_size, dim=0),
        )

@dataclass
class NextBarCondition:
    track_manager: TrackManager | None # Stores information of past tracks
    bar_meta_emb: torch.FloatTensor
    all_bar_context: torch.FloatTensor
    harmo_hidden: LastHiddenWithMask | None

    prev_bar_hidden: HiddenWithMask

    # Used in sampling constraints
    bar_len: torch.LongTensor
    harmo_beat_mask: torch.BoolTensor
