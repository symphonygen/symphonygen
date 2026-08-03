import torch
import torch.nn.functional as F
from arch.model.hidden_with_mask import HiddenWithMask, LastHiddenWithMask
from arch.model.cross_attn_provider import CrossAttnProvider, CrossAttnProviderForInfer

class HarmCrossAttnProvider(CrossAttnProvider):
    def __init__(self, shape):
        self.shape = shape

    def prepare(self, hier_event_mask):
        self.track_prev_bar_mask = F.pad(
            hier_event_mask[:, :-1, :, :], (0, 0, 0, 0, 1, 0), value=True
        ).flatten(0, -2)
        self.prev_bar_mask = None
        self.harmo_event_mask = None

    def forward(self, hidden_states: torch.FloatTensor, layer_idx: int):
        Bars, _, HarmEvents, D = self.shape
        hier_hidden_states = hidden_states.view(-1, Bars, 1, HarmEvents, D)
        hier_hidden_states = F.pad(hier_hidden_states[:, :-1, :, :, :], (0, 0, 0, 0, 0, 0, 1, 0), value=0.0)
        encoder_hidden_states = hier_hidden_states.flatten(0, -3)
        encoder_attention_mask = self.track_prev_bar_mask
        return encoder_hidden_states, encoder_attention_mask

class CrossAttnProvider3S(CrossAttnProvider):
    def __init__(self, shape, HarmEvents):
        self.shape = shape
        self.HarmEvents = HarmEvents

    def prepare(
        self,
        hier_event_mask: torch.BoolTensor,
        track_prev_map,
        harmo_hidden: LastHiddenWithMask
    ):
        B, *_ = hier_event_mask.shape
        Bars, Tracks, _, D = self.shape
        HarmEvents = self.HarmEvents
        device = hier_event_mask.device
        self._batch_idx = torch.arange(B, device=device).view(B, 1, 1)
        self.track_prev_map = track_prev_map
        self._invalid_track_prev = (track_prev_map == -1)

        self._prev_bar_idx = torch.arange(-1, Bars - 1, device=device).view(1, Bars, 1)
        track_prev_bar_mask = hier_event_mask[self._batch_idx, self._prev_bar_idx, track_prev_map]
        track_prev_bar_mask[self._invalid_track_prev] = True
        self.track_prev_bar_mask = track_prev_bar_mask.flatten(0, -2)

        self.prev_track_mask = F.pad(hier_event_mask[:, :, :-1, :], (0, 0, 1, 0), value=True).flatten(0, -2)

        # TODO: Currently repeating tensors to match B * Bars * Tracks.
        # In the future, one can modify cross attention to use 5D broadcasting to avoid this.
        harmo_hidden = harmo_hidden.view(B, Bars, 1, HarmEvents, D).repeat(1, 1, Tracks, 1, 1).flatten(0, -3)
        self.harmo_hidden = harmo_hidden.hidden
        self.harmo_event_mask = harmo_hidden.mask

        self.masks = (self.track_prev_bar_mask, self.prev_track_mask, self.harmo_event_mask)

    def forward(self, hidden_states: torch.FloatTensor, layer_idx: int):
        if layer_idx % 3 == 0:
            hier_hidden = hidden_states.view(-1, *self.shape)
            hier_hidden = hier_hidden[self._batch_idx, self._prev_bar_idx, self.track_prev_map]
            hier_hidden[self._invalid_track_prev] = 0
            hidden = hier_hidden.flatten(0, -3)
            mask = self.track_prev_bar_mask
        elif layer_idx % 3 == 1:
            hier_hidden = hidden_states.view(-1, *self.shape)
            hier_hidden = F.pad(hier_hidden[:, :, :-1, :, :], (0, 0, 0, 0, 1, 0), value=0.0)
            hidden = hier_hidden.flatten(0, -3)
            mask = self.prev_track_mask
        else:
            hidden = self.harmo_hidden
            mask = self.harmo_event_mask
        return hidden, mask

class CrossAttnProvider2S(CrossAttnProvider):
    def __init__(self, shape, HarmEvents):
        self.shape = shape
        self.HarmEvents = HarmEvents

    def prepare(
        self,
        hier_event_mask: torch.BoolTensor,
        track_prev_map,
        harmo_hidden: LastHiddenWithMask
    ):
        B, _, _, _ = hier_event_mask.shape
        Bars, Tracks, _, D = self.shape
        HarmEvents = self.HarmEvents
        device = hier_event_mask.device
        self._batch_idx = torch.arange(B, device=device).view(B, 1, 1)
        self.track_prev_map = track_prev_map
        self._invalid_track_prev = (track_prev_map == -1)

        self._prev_bar_idx = torch.arange(-1, Bars - 1, device=device).view(1, Bars, 1)
        track_prev_bar_mask = hier_event_mask[self._batch_idx, self._prev_bar_idx, track_prev_map]
        track_prev_bar_mask[self._invalid_track_prev] = True
        self.track_prev_bar_mask = track_prev_bar_mask.flatten(0, -2)

        # TODO: Currently repeating tensors to match B * Bars * Tracks.
        # In the future, one can modify cross attention to use 5D broadcasting to avoid this.
        harmo_hidden = harmo_hidden.view(B, Bars, 1, HarmEvents, D).repeat(1, 1, Tracks, 1, 1).flatten(0, -3)
        self.harmo_hidden = harmo_hidden.hidden
        self.harmo_event_mask = harmo_hidden.mask

        self.masks = (self.track_prev_bar_mask, self.harmo_event_mask)

    def forward(self, hidden_states: torch.FloatTensor, layer_idx: int):
        if layer_idx % 2 == 0:
            hier_hidden = hidden_states.view(-1, *self.shape)
            hier_hidden = hier_hidden[self._batch_idx, self._prev_bar_idx, self.track_prev_map]
            hier_hidden[self._invalid_track_prev] = 0
            hidden = hier_hidden.flatten(0, -3)
            mask = self.track_prev_bar_mask
        else:
            hidden = self.harmo_hidden
            mask = self.harmo_event_mask
        return hidden, mask

class CrossAttnProviderHarm(CrossAttnProvider):
    def __init__(self, shape, HarmEvents):
        self.shape = shape
        self.HarmEvents = HarmEvents

    def prepare(self, harmo_hidden: LastHiddenWithMask):
        Bars, Tracks, _, D = self.shape
        HarmEvents = self.HarmEvents

        harmo_hidden = harmo_hidden.view(-1, Bars, 1, HarmEvents, D).repeat(1, 1, Tracks, 1, 1).flatten(0, -3)
        self.harmo_hidden = harmo_hidden.hidden
        self.harmo_event_mask = harmo_hidden.mask

        self.masks = (self.harmo_event_mask,)

    def forward(self, hidden_states: torch.FloatTensor, layer_idx: int):
        hidden = self.harmo_hidden
        mask = self.harmo_event_mask
        return hidden, mask

class CrossAttnProviderMel(CrossAttnProvider):
    def __init__(self, shape):
        self.shape = shape

    def prepare(
        self,
        hier_event_mask: torch.BoolTensor,
        track_prev_map,
    ):
        B, Bars, _, _ = hier_event_mask.shape
        device = hier_event_mask.device
        self._batch_idx = torch.arange(B, device=device).view(B, 1, 1)
        self.track_prev_map = track_prev_map
        self._invalid_track_prev = (track_prev_map == -1)

        self._prev_bar_idx = torch.arange(-1, Bars - 1, device=device).view(1, Bars, 1)
        track_prev_bar_mask = hier_event_mask[self._batch_idx, self._prev_bar_idx, track_prev_map]
        track_prev_bar_mask[self._invalid_track_prev] = True
        self.track_prev_bar_mask = track_prev_bar_mask.flatten(0, -2)

        self.masks = (self.track_prev_bar_mask,)

    def forward(self, hidden_states: torch.FloatTensor, layer_idx: int):
        hier_hidden = hidden_states.view(-1, *self.shape)
        hier_hidden = hier_hidden[self._batch_idx, self._prev_bar_idx, self.track_prev_map]
        hier_hidden[self._invalid_track_prev] = 0
        hidden = hier_hidden.flatten(0, -3)
        mask = self.track_prev_bar_mask
        return hidden, mask

class CrossAttnProviderForInfer3S(CrossAttnProviderForInfer):
    def __init__(self, track_prev_bar_hidden: HiddenWithMask, prev_track_hidden: HiddenWithMask, harmo_hidden: HiddenWithMask):
        if harmo_hidden.ndim == 4:
            harmo_hidden = harmo_hidden[:, 0]
        self.track_prev_bar_hidden = track_prev_bar_hidden
        self.prev_track_hidden = prev_track_hidden
        self.harmo_hidden = harmo_hidden
        self.hiddens = [self.track_prev_bar_hidden, self.prev_track_hidden, self.harmo_hidden]

    def retrieve(self, layer_idx: int):
        if layer_idx % 3 == 0:
            hidden = self.track_prev_bar_hidden.hidden[layer_idx]
            mask = self.track_prev_bar_hidden.mask
        elif layer_idx % 3 == 1:
            hidden = self.prev_track_hidden.hidden[layer_idx]
            mask = self.prev_track_hidden.mask
        else:
            hidden = self.harmo_hidden.hidden
            mask = self.harmo_hidden.mask
        return hidden, mask

class CrossAttnProviderForInfer2S(CrossAttnProviderForInfer):
    def __init__(self, prev_bar_hidden: HiddenWithMask, harmo_hidden: LastHiddenWithMask):
        if harmo_hidden.ndim == 4:
            harmo_hidden = harmo_hidden[:, 0]
        self.prev_bar_hidden = prev_bar_hidden
        self.harmo_hidden = harmo_hidden
        self.hiddens = [self.prev_bar_hidden, self.harmo_hidden]

    def retrieve(self, layer_idx: int):
        if layer_idx % 2 == 0:
            hidden = self.prev_bar_hidden.hidden[layer_idx]
            mask = self.prev_bar_hidden.mask
        else:
            hidden = self.harmo_hidden.hidden
            mask = self.harmo_hidden.mask
        return hidden, mask

class CrossAttnProviderForInferHarm(CrossAttnProviderForInfer):
    def __init__(self, harmo_hidden: LastHiddenWithMask):
        if harmo_hidden.ndim == 4:
            harmo_hidden = harmo_hidden[:, 0]
        self.harmo_hidden = harmo_hidden
        self.hiddens = [self.harmo_hidden]

    def retrieve(self, layer_idx: int):
        hidden = self.harmo_hidden.hidden
        mask = self.harmo_hidden.mask
        return hidden, mask

class CrossAttnProviderForInferMel(CrossAttnProviderForInfer):
    def __init__(self, prev_bar_hidden: HiddenWithMask):
        self.prev_bar_hidden = prev_bar_hidden
        self.hiddens = [self.prev_bar_hidden]

    def retrieve(self, layer_idx: int):
        hidden = self.prev_bar_hidden.hidden[layer_idx]
        mask = self.prev_bar_hidden.mask
        return hidden, mask

class NextBarCrossAttnProvider3S(CrossAttnProvider):
    def __init__(
        self,
        hier_event_mask: torch.BoolTensor,
        track_prev_bar_hidden: HiddenWithMask,
        harmo_hidden: LastHiddenWithMask,
    ):
        _, Tracks, Events, D = track_prev_bar_hidden.shape
        B_Groups, _, HarmEvents, _ = harmo_hidden.shape
        self.shape = (Tracks, Events, D)

        track_prev_bar_hidden = track_prev_bar_hidden.flatten(0, -3)
        self.track_prev_bar_hidden = track_prev_bar_hidden.hidden
        self.track_prev_bar_mask = track_prev_bar_hidden.mask

        self.prev_track_mask = F.pad(hier_event_mask[:, :, :-1, :], (0, 0, 1, 0), value=True).flatten(0, -2)

        harmo_hidden = harmo_hidden.view(B_Groups, 1, HarmEvents, -1).repeat(1, Tracks, 1, 1).flatten(0, -3)
        self.harmo_hidden = harmo_hidden.hidden
        self.harmo_event_mask = harmo_hidden.mask

    def forward(self, hidden_states: torch.FloatTensor, layer_idx: int):
        if layer_idx % 3 == 0:
            hidden = self.track_prev_bar_hidden[layer_idx]
            mask = self.track_prev_bar_mask
        elif layer_idx % 3 == 1:
            hier_hidden = hidden_states.view(-1, *self.shape)
            hier_hidden = F.pad(hier_hidden[:, :-1, :, :], (0, 0, 0, 0, 1, 0), value=0.0)
            hidden = hier_hidden.flatten(0, -3)
            mask = self.prev_track_mask
        else:
            hidden = self.harmo_hidden
            mask = self.harmo_event_mask
        return hidden, mask

class NextBarCrossAttnProvider2S(CrossAttnProvider):
    def __init__(
        self,
        track_prev_bar_hidden: HiddenWithMask,
        harmo_hidden: LastHiddenWithMask,
    ):
        _, Tracks, Events, D = track_prev_bar_hidden.shape
        B_Groups, _, HarmEvents, _ = harmo_hidden.shape
        self.shape = (Tracks, Events, D)

        track_prev_bar_hidden = track_prev_bar_hidden.flatten(0, -3)
        self.track_prev_bar_hidden = track_prev_bar_hidden.hidden
        self.track_prev_bar_mask = track_prev_bar_hidden.mask

        harmo_hidden = harmo_hidden.view(B_Groups, 1, HarmEvents, -1).repeat(1, Tracks, 1, 1).flatten(0, -3)
        self.harmo_hidden = harmo_hidden.hidden
        self.harmo_event_mask = harmo_hidden.mask

    def forward(self, hidden_states: torch.FloatTensor, layer_idx: int):
        if layer_idx % 2 == 0:
            hidden = self.track_prev_bar_hidden[layer_idx]
            mask = self.track_prev_bar_mask
        else:
            hidden = self.harmo_hidden
            mask = self.harmo_event_mask
        return hidden, mask
