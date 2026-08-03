import sys
import itertools
from data_prep.data_structure import *
from arch.vocab import *

class DataTensorConverter:
    def pack_data_tensor(self):
        raise NotImplementedError

    def unpack_data_tensor(self):
        raise NotImplementedError

def pack_track(track: TrackStruct, event_limit=sys.maxsize, include_end=False):
    track_events = []

    _last_note_end = None
    for pos, pos_group in itertools.groupby(track.notes, key=lambda n: n.pos):
        if len(track_events) >= event_limit: break

        if pos == _last_note_end:
            # Combine next position token with last duration token if their ends meet
            track_events[-1] = pack_legato(unpack_dur(track_events[-1]))
        elif pos > 0:
            # Omit position token at downbeat
            track_events.append(pack_pos(pos))

        for dur, dur_group in itertools.groupby(pos_group, key=lambda n: n.duration_q):
            if len(track_events) >= event_limit: break

            for note in dur_group:
                track_events.append(pack_pitch(note.pitch))
            track_events.append(pack_dur(dur))
        _last_note_end = pos + dur

    if include_end:
        track_events.append(END_EVENT)
    track_events = track_events[:event_limit]
    return track_events

def unpack_track(track_events, pos_limit=sys.maxsize, return_warnings=False):
    cur_pos = 0
    pending_notes: List[MusicNote] = [] # Wait for duration token to arrive
    dur = None
    notes = []
    warnings = []
    for event in track_events:
        if event == END_EVENT or event == PAD_EVENT:
            break
        if is_vocab_pos(event):
            cur_pos = unpack_pos(event)
            if cur_pos >= pos_limit:
                raise ValueError(f"Position out of range: POS of {cur_pos} exceeds {pos_limit}")
        elif is_vocab_pitch(event):
            pending_notes.append(MusicNote(pitch=unpack_pitch(event), pos=cur_pos, duration_q=None))
        elif is_vocab_dur(event):
            dur = unpack_dur(event)
        elif is_vocab_legato(event):
            dur = unpack_legato(event)
            cur_pos += dur
            if cur_pos >= pos_limit:
                raise ValueError(f"Position out of range: LEGATO of {dur} shifts position to {cur_pos}, exceeds {pos_limit}")
        if dur is not None:
            for note in pending_notes:
                note.duration_q = dur
            notes.extend(pending_notes)
            pending_notes = []
            dur = None
    if not notes:
        warnings += ["No notes found in a track"]
    if pending_notes:
        warnings += ["Pending notes without duration, discarded"]
    if return_warnings:
        return notes, warnings
    return notes
