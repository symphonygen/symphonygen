import torch
from arch.data_tensor import *
from arch.harmo.vocab import *
from arch.harmo.config import *
from data_prep.data_structure import *

class HarmonyTensorConverter(DataTensorConverter):
    def __init__(self, harmo_event_num=HARMO_EVENT_NUM):
        self.harmo_event_num = harmo_event_num

    def pack_data_tensor(self, bars: list[BarStruct]):
        harmo_events_data = torch.full((self.harmo_event_num,), PAD_EVENT, dtype=torch.long)

        harmo_events = [START]
        for bar in bars:
            if len(harmo_events) >= self.harmo_event_num:
                break

            harmo_events.append(pack_bar_len(bar.duration_q))
            if bar.harmony_track:
                harmo_track_events = pack_track(bar.harmony_track, include_end=True)
                harmo_events.extend(harmo_track_events)

        harmo_events.append(END_EVENT)
        harmo_events = harmo_events[:self.harmo_event_num]

        harmo_events_data[:len(harmo_events)] = torch.tensor(harmo_events)
        return harmo_events_data

    def unpack_data_tensor(self, harmo_events):
        if isinstance(harmo_events, torch.Tensor):
            harmo_events = harmo_events.tolist()
        if not harmo_events or harmo_events[0] != START:
            raise ValueError(f"Invalid start event: {harmo_events[0]}")

        bars = []
        bar_indices = [i for i, event in enumerate(harmo_events) if is_vocab_bar_len(event)]
        for i, start_idx in enumerate(bar_indices):
            end_idx = bar_indices[i + 1] if i + 1 < len(bar_indices) else len(harmo_events)
            bar_tokens = harmo_events[start_idx: end_idx]

            # The first token is the bar length
            bar_len = unpack_bar_len(bar_tokens[0])
            bar = BarStruct(duration_q=bar_len)
            notes = unpack_track(bar_tokens[1:], pos_limit=bar_len)
            if notes:
                bar.harmony_track = TrackStruct(META_HARMO_TRACK_ID, META_HARMO_INST, notes=notes)
            bars.append(bar)
        return bars
