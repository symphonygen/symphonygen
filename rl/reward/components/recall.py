import itertools
from data_prep.data_structure import BarStruct, MusicNote

def eval_harmony_precision(bars: list[BarStruct], recon_outline: list[list[MusicNote]]):
    tot_score = 0
    num_pos = 0

    for bar, recon in zip(bars, recon_outline):
        if bar.harmony_track is None:
            continue

        ref_pos_map = get_pitches_by_pos(bar.harmony_track.notes)
        pos_map = get_pitches_by_pos(recon)
        for pos in ref_pos_map:
            ref_pitches = set(ref_pos_map[pos])
            pitches = set(pos_map.get(pos, []))
            if not pitches:
                continue

            intersection = len(pitches & ref_pitches)
            precision = intersection / len(pitches)
            tot_score += precision
            num_pos += 1

    return tot_score / num_pos if num_pos else 0

def eval_harmony_recall(bars: list[BarStruct], recon_outline: list[list[MusicNote]]):
    tot_score = 0
    num_pos = 0

    for bar, recon in zip(bars, recon_outline):
        if bar.harmony_track is None:
            continue

        ref_pos_map = get_pitches_by_pos(bar.harmony_track.notes)
        pos_map = get_pitches_by_pos(recon)
        for pos in ref_pos_map:
            ref_pitches = set(ref_pos_map[pos])
            pitches = set(pos_map.get(pos, []))
            if not ref_pitches:
                continue

            intersection = len(pitches & ref_pitches)
            recall = intersection / len(ref_pitches)
            tot_score += recall
            num_pos += 1

    return tot_score / num_pos if num_pos else 0

def get_pitches_by_pos(notes: list[MusicNote]):
    return {
        pos: [note.pitch for note in pos_group]
        for pos, pos_group in itertools.groupby(notes, key=lambda x: x.pos)
    }
