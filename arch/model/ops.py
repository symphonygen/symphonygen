from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from arch.config import *
from arch.vocab import *
from data_prep.data_structure import *

EPS = 1e-9

def mean_pooling(x: torch.FloatTensor, mask: torch.BoolTensor | None):
    if mask is None:
        return torch.mean(x, dim=1)
    mask = mask.unsqueeze(-1)
    x_masked = x.masked_fill(~mask, 0.0)

    sum_x = torch.sum(x_masked, dim=1)
    sum_mask = torch.clamp(mask.sum(dim=1), min=EPS)
    return sum_x / sum_mask

def right_shift_pad(x: torch.FloatTensor):
    assert x.ndim == 3, "Input x must be a [Batch, SeqLen, Hidden] tensor"
    return F.pad(x[:, :-1, :], (0, 0, 1, 0), value=0.0)

def create_hier_attn_masks(hier_events: torch.LongTensor):
    """ Create 3D hierarchical attention masks at bar, track, and event level. """
    hier_event_mask = (hier_events != 0)
    hier_track_mask = hier_event_mask.any(dim=-1)
    event_mask = hier_event_mask.flatten(0, -2)
    track_mask = hier_track_mask.flatten(0, -2)
    bar_mask = hier_track_mask.any(dim=-1)
    return HierAttnMasks(hier_event_mask, event_mask, track_mask, bar_mask)

def create_meta_predictors(vocab_size, hidden_size, padding_idx):
    """ Factory function for Meta Predictor heads. """
    embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
    head = nn.Linear(hidden_size, vocab_size)
    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)
    return embedding, head, criterion

def create_harmo_beat_mask(all_harmo_list: list[list[BarStruct]]):
    """
    Create compact harmony mask for beat-quantized harmony skeletons.
    NOTE: Assumes the beat duration is divisible by quarter note.
    """
    harmo_beat_mask = torch.zeros((len(all_harmo_list), BAR_NUM, BEAT_NUM, PITCH_NUM), dtype=torch.bool)
    for b, harmo_bars in enumerate(all_harmo_list):
        for bid, bar in enumerate(harmo_bars):
            if bar.harmony_track:
                for note in bar.harmony_track.notes:
                    harmo_beat_mask[b, bid, note.pos // QUARTER, note.pitch] = True
    return harmo_beat_mask

def to_list(*tensors):
    return tuple(
        tensor.tolist() if isinstance(tensor, torch.Tensor) else tensor
        for tensor in tensors
    )

@dataclass
class HierAttnMasks:
    hier_event_mask: torch.BoolTensor
    event_mask: torch.BoolTensor
    track_mask: torch.BoolTensor
    bar_mask: torch.BoolTensor
