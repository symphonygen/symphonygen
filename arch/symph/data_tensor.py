import torch
from arch.config import *
from arch.data_tensor import DataTensorConverter, pack_track, unpack_track
from arch.harmo.config import *
from arch.harmo.vocab import *
from arch.model.ops import to_list
from arch.symph.config import *
from arch.symph.vocab import *
from arch.vocab import *
from data_prep.data_structure import *

class MusicTensorConverter3D(DataTensorConverter):
    def __init__(
        self,
        bar_num=BAR_NUM, track_num=TRACK_NUM, event_num=EVENT_NUM,
        harmo_event_num=HARMO_EVENT_NUM,
        print_warnings=False,
    ):
        self.bar_num = bar_num
        self.track_num = track_num
        self.event_num = event_num
        self.harmo_event_num = harmo_event_num
        self.print_warnings = print_warnings

    def get_track_prev_map(self, bars: list[BarStruct]):
        track_prev_map = torch.full((self.bar_num, self.track_num), -1, dtype=torch.long)

        for bid, (prev_bar, bar) in enumerate(zip(bars[:-1], bars[1:])):
            prev_track_id_to_tid = {
                track.track_id: tid
                for tid, track in enumerate(prev_bar.tracks[:self.track_num])
            }
            for tid, track in enumerate(bar.tracks[:self.track_num]):
                if track.track_id in prev_track_id_to_tid:
                    track_prev_map[bid, tid] = prev_track_id_to_tid[track.track_id]

        return track_prev_map

    def pack_data_tensor(self, bars: list[BarStruct]):
        if len(bars) > self.bar_num:
            raise RuntimeError(f"{len(bars)=} exceeds {self.bar_num=}")

        events = torch.full((self.bar_num, self.track_num, self.event_num), PAD_EVENT, dtype=torch.long)
        harmo_events = torch.full((self.bar_num, 1, self.harmo_event_num), PAD_EVENT, dtype=torch.long)
        bar_len = torch.full((self.bar_num,), META_PAD_BAR_LEN, dtype=torch.long)
        track_id = torch.full((self.bar_num, self.track_num), META_PAD_TRACK, dtype=torch.long)
        inst = torch.full((self.bar_num, self.track_num), META_PAD_INST, dtype=torch.long)
        new_inst = torch.full((self.bar_num, self.track_num), META_PAD_INST, dtype=torch.long)
        track_prev_map = self.get_track_prev_map(bars)

        existing_tracks = set()
        for bid, bar in enumerate(bars):
            if bar.harmony_track:
                harmo_track_events = pack_track(bar.harmony_track, event_limit=self.harmo_event_num, include_end=True)
                harmo_events[bid, 0, :len(harmo_track_events)] = torch.tensor(harmo_track_events)
            else:
                harmo_events[bid, 0, 0] = END_EVENT

            for tid, track in enumerate(bar.tracks[:self.track_num]):
                track_events = pack_track(track, event_limit=self.event_num, include_end=True)
                events[bid, tid, :len(track_events)] = torch.tensor(track_events)
                track_id[bid, tid] = track.track_id
                inst[bid, tid] = track.program
                if track.track_id not in existing_tracks:
                    new_inst[bid, tid] = track.program
                    existing_tracks.add(track.track_id)

            if len(bar.tracks) < self.track_num:
                events[bid, len(bar.tracks), 0] = END_OF_BAR
                track_id[bid, len(bar.tracks)] = META_END_OF_BAR
            bar_len[bid] = bar.duration_q

        return events, harmo_events, bar_len, track_id, inst, new_inst, track_prev_map

    def unpack_data_tensor(
        self,
        events, bar_len, track_id, inst,
    ):
        if not len(events) == len(bar_len) == len(track_id) == len(inst):
            raise ValueError(f"First dimension of tensors must be equal: {len(events)=}, {len(bar_len)=}, {len(track_id)=}, {len(inst)=}")
        events, bar_len, track_id, inst = to_list(events, bar_len, track_id, inst)

        bars = []
        for bid in range(len(events)):
            bar = BarStruct(duration_q=bar_len[bid])

            for tid, track_events in enumerate(events[bid]):
                if track_events[0] in (END_OF_BAR, PAD_EVENT):
                    break
                notes, warnings = unpack_track(track_events, pos_limit=bar.duration_q, return_warnings=True)
                if self.print_warnings:
                    for warning in warnings:
                        print(f"Warning: {warning} at bar {bid}, track {tid}")
                if inst[bid][tid] == META_PAD_INST:
                    raise ValueError(f"Instrument unassigned at bar {bid}, track {tid}")
                track = TrackStruct(track_id[bid][tid], inst[bid][tid], notes=notes)
                bar.tracks.append(track)
            bars.append(bar)

        return bars
