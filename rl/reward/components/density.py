from data_prep.data_structure import BarStruct
from arch.vocab import QUARTER

def eval_harmony_density(bar: BarStruct, max_notes_per_beat):
    """
    Harmony density of a bar: num_of_notes / max_num_of_notes
    """
    if bar.harmony_track is None:
        return 0.0
    num_beats = max(1, bar.duration_q // QUARTER)
    density = len(bar.harmony_track.notes) / (max_notes_per_beat * num_beats)
    return density
