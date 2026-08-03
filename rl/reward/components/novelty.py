import numpy as np
import itertools
from data_prep.data_structure import *

def eval_harmony_bar_repeat_ratio(bars: list[BarStruct], loop_periods: list[int], similar_threshold=0.9):
    """
    Bar repeat ratio of loops of certain periods (e.g. 1, 2, and 4)
    """
    bar_num = len(bars)

    matches = {d: np.zeros(bar_num) for d in loop_periods}

    for bid in range(bar_num):
        cur_h = bars[bid].harmony_track

        if cur_h and cur_h.notes:
            for dist in loop_periods:
                if bid >= dist:
                    prev_h = bars[bid - dist].harmony_track
                    if prev_h and track_similar(cur_h.notes, prev_h.notes, threshold=similar_threshold):
                        matches[dist][bid] = 1

    repeat_mask = get_repeat_coverage(matches)
    longest_repeat = longest_run_len(repeat_mask)
    return (longest_repeat + int(np.sum(repeat_mask))) / bar_num if bar_num else 0

def eval_harmony_beat_repeat_ratio(bars: list[BarStruct], similar_threshold=0.9):
    """
    (longest_beat_repeat + beat_repeat) / num_beats
    """
    beats = []
    for bar in bars:
        if bar.harmony_track is None:
            continue
        for _, pos_group in itertools.groupby(bar.harmony_track.notes, key=lambda note: note.pos):
            beats.append(list(pos_group))

    match = np.zeros(len(beats), dtype=bool)

    for beat_id, (prev_beat, beat) in enumerate(zip(beats, beats[1:])):
        if track_similar(prev_beat, beat, threshold=similar_threshold):
            match[beat_id + 1] = True

    longest_repeat = longest_run_len(match)
    return (longest_repeat + int(np.sum(match))) / len(beats) if beats else 0

def track_similar(t1: list[MusicNote], t2: list[MusicNote], threshold):
    n1 = set((note.pos, note.pitch) for note in t1)
    n2 = set((note.pos, note.pitch) for note in t2)
    return len(n1 & n2) / len(n1 | n2) >= threshold

def get_repeat_coverage(s):
    """
    How many bars are in a repeat pattern of period (1, 2, 4)?
    Not including the first appearance
    """
    length = len(s[1])
    # Initialize a mask of False for every bar in the track
    covered_mask = np.zeros(length, dtype=bool)

    # 1. One-bar repeats (AA): Covers 1 bar
    # s[1] is already a mask of where B_i == B_{i-1}
    covered_mask |= (s[1] == 1)

    # 2. Two-bar repeats (ABAB): Covers 2 bars
    # Find where a 2-bar block matches the previous 2-bar block
    if 2 in s:
        two_match = (s[2][1:] == 1) & (s[2][:-1] == 1)

        # If index 'i' is a match, it covers bars i and i-1
        # We shift the mask to cover both positions
        covered_mask[1:][two_match] = True   # Covers 'i'
        covered_mask[:-1][two_match] = True  # Covers 'i-1'

    # 3. Four-bar repeats (ABCDABCD): Covers 4 bars
    if 4 in s:
        four_match = (s[4][3:] == 1) & (s[4][2:-1] == 1) & (s[4][1:-2] == 1) & (s[4][:-3] == 1)

        # If index 'i' is a match, it covers i, i-1, i-2, i-3
        for offset in range(4):
            # We apply the mask to the current and 3 preceding positions
            covered_mask[offset: offset + len(four_match)][four_match] = True

    return covered_mask

def longest_run_len(arr):
    if not np.any(arr):
        return 0
    # Pad with False to handle runs at the start/end
    bounded = np.concatenate(([False], arr, [False]))
    # Find indices where the value changes
    diffs = np.diff(bounded.view(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return np.max(ends - starts)
