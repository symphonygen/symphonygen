from dataclasses import dataclass, field
import numpy as np

@dataclass
class MidiRawNote:
    tick: int
    pitch: int
    program: int
    is_drum: bool = False
    track_id: int | None = None
    duration: int | None = None

@dataclass
class MusicNote:
    pos: int # Unit: 32nd notes
    duration_q: int # Unit: 32nd notes
    pitch: int

    @property
    def end_pos(self):
        return self.pos + self.duration_q # May exceeds the bar boundary

@dataclass
class TrackStruct:
    track_id: int # Track ID, according to the order of tracks in the MIDI
    program: int # MIDI program number
    notes: list[MusicNote] = field(default_factory=list)

@dataclass
class BarStruct:
    duration_q: int # Unit: 32nd notes
    tracks: list[TrackStruct] = field(default_factory=list)

    # Harmonic analysis
    harmony_track: TrackStruct | None = None # Stores beat-quantized harmony skeleton
    _beat_dur_q: int | None = None
    _sliced_notes: dict[int, list[MusicNote]] | None = None

    # Melodic analysis
    _skyline: np.ndarray | None = None
    _popular_dur: dict[int, int] | None = None
    _track_is_high: dict[int, list[bool]] | None = None

def NOTE_ORDER(note: MusicNote):
    return (note.pos, -note.duration_q, note.pitch)
