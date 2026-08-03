import torch
from dataclasses import dataclass

@dataclass
class LastHiddenWithMask:
    hidden: torch.FloatTensor
    mask: torch.BoolTensor

    @classmethod
    def zeros(cls, shape, device):
        hidden = torch.zeros(*shape, device=device)
        mask = torch.zeros(*shape[:-1], dtype=torch.bool, device=device)
        return cls(hidden, mask)

    @property
    def shape(self):
        return self.hidden.shape

    @property
    def ndim(self):
        return self.hidden.ndim

    def __getitem__(self, idx):
        return LastHiddenWithMask(self.hidden[idx], self.mask[idx])

    def __setitem__(self, idx, small_cache: "LastHiddenWithMask"):
        self.hidden[idx] = small_cache.hidden
        self.mask[idx] = small_cache.mask

    def pad_where(self, idx):
        self.hidden[idx] = 0
        self.mask[idx] = True # XXX: Set mask to all True because all False will throw error

    def flatten(self, start_dim, end_dim):
        return LastHiddenWithMask(self.hidden.flatten(start_dim, end_dim), self.mask.flatten(start_dim, end_dim + 1))

    def view(self, *shape):
        return LastHiddenWithMask(self.hidden.view(*shape), self.mask.view(*shape[:-1]))

    def expand(self, *shape):
        return LastHiddenWithMask(self.hidden.expand(*shape), self.mask.expand(*shape[:-1]))

    def repeat(self, *repeats):
        return LastHiddenWithMask(self.hidden.repeat(*repeats), self.mask.repeat(*repeats[:-1]))

    def repeat_interleave(self, *args, **kwargs):
        return LastHiddenWithMask(self.hidden.repeat_interleave(*args, **kwargs), self.mask.repeat_interleave(*args, **kwargs))

@dataclass
class HiddenWithMask:
    hidden: list[torch.FloatTensor]
    mask: torch.BoolTensor

    @classmethod
    def zeros(cls, shape, num_layers, device):
        hidden = [torch.zeros(*shape, device=device) for _ in range(num_layers)]
        mask = torch.zeros(*shape[:-1], dtype=torch.bool, device=device)
        return cls(hidden, mask)

    @property
    def shape(self):
        return self.hidden[0].shape

    @property
    def ndim(self):
        return self.hidden[0].ndim

    def __getitem__(self, idx):
        return HiddenWithMask([hidden[idx] for hidden in self.hidden], self.mask[idx])

    def __setitem__(self, idx, small_cache: "HiddenWithMask"):
        for lid, hidden in enumerate(small_cache.hidden):
            self.hidden[lid][idx] = hidden
        self.mask[idx] = small_cache.mask

    def pad_where(self, idx):
        for hidden in self.hidden:
            hidden[idx] = 0
        self.mask[idx] = True # XXX: Set mask to all True because all False will throw error
        return self

    def flatten(self, start_dim, end_dim):
        return HiddenWithMask([hidden.flatten(start_dim, end_dim) for hidden in self.hidden], self.mask.flatten(start_dim, end_dim + 1))

    def view(self, *shape):
        return HiddenWithMask([hidden.view(*shape) for hidden in self.hidden], self.mask.view(*shape[:-1]))

    def expand(self, *shape):
        return HiddenWithMask([hidden.expand(*shape) for hidden in self.hidden], self.mask.expand(*shape[:-1]))

    def repeat(self, *repeats):
        return HiddenWithMask([hidden.repeat(*repeats) for hidden in self.hidden], self.mask.repeat(*repeats[:-1]))

    def repeat_interleave(self, *args, **kwargs):
        return HiddenWithMask([hidden.repeat_interleave(*args, **kwargs) for hidden in self.hidden], self.mask.repeat_interleave(*args, **kwargs))
