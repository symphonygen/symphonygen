import itertools
import numpy as np
from collections import defaultdict
from arch.vocab import QUARTER
from rl.reward.utils_ import dedupe_sorted
from data_prep.data_structure import BarStruct, MusicNote, TrackStruct

# --- Utility Functions ---

MAIN_DURS = [
    # QUARTER * 4,
    QUARTER * 2,
    QUARTER,
    QUARTER // 2,
    QUARTER // 4,
    QUARTER // 8
]

def get_dur_class(dur, main_durs=MAIN_DURS):
    return min(main_durs, key=lambda x: abs(x - dur))

def analyze_popular_dur_(bars: list[BarStruct], debug_break=None):
    for bid, bar in enumerate(bars):
        bar._popular_dur = {}
        for tid, track in enumerate(bar.tracks):
            if (bid + 1, tid + 1) == debug_break:
                __import__("pdb").set_trace()

            note_groups = itertools.groupby(track.notes, key=lambda x: x.pos)
            group_info = []
            for pos, pos_group in note_groups:
                highest_note = max(pos_group, key=lambda x: x.pitch)
                group_info.append((pos, highest_note))
            last_end_pos = group_info[-1][1].end_pos
            group_info: list[tuple[int, MusicNote]] = list(dedupe_sorted(group_info, sig=lambda x: x[1].pitch))

            note_durs = [note.duration_q for _, note in group_info]
            pos = [pos for pos, _ in group_info] + [last_end_pos]
            gap_durs = [pos[i] - pos[i - 1] for i in range(1, len(pos))]
            perceived_durs = [
                gap_dur if gap_dur <= QUARTER else note_dur
                for gap_dur, note_dur in zip(gap_durs, note_durs)
            ]
            dur_classes = [get_dur_class(dur) for dur in perceived_durs]
            votes = defaultdict(int)
            for dur, dur_class in zip(perceived_durs, dur_classes):
                votes[dur_class] += dur
            votes = {k: v for k, v in votes.items() if v >= bar.duration_q // 2}
            if votes:
                bar._popular_dur[track.track_id] = max(votes, key=votes.get)

def analyze_skyline_(bar: BarStruct):
    bar._skyline = np.array([-1] * bar.duration_q)
    for track_notes in bar._sliced_notes.values():
        for note in track_notes:
            bar._skyline[note.pos: note.pos + note.duration_q] = np.maximum(bar._skyline[note.pos: note.pos + note.duration_q], note.pitch)

def analyze_potential_melody_track_(bars: list[BarStruct]):
    for bar in bars:
        analyze_skyline_(bar)
        bar._track_is_high = {}
        for track in bar.tracks:
            bar._track_is_high[track.track_id] = any(note.pitch == bar._skyline[note.pos] for note in track.notes)

def analyze_melodic_(bars: list[BarStruct]):
    analyze_popular_dur_(bars)
    analyze_potential_melody_track_(bars)

# --- Evaluation Functions ---

def eval_melody_moving(
    bars: list[BarStruct],
    return_loc = False,
    debug_break = None,
):
    moving: list[int | None] = [None] * len(bars)
    for bid, (bar, next_bar) in enumerate(zip(bars, bars[1:])):
        for tid, track in enumerate(bar.tracks):
            if (bid + 1, tid + 1) == debug_break:
                __import__("pdb").set_trace()

            if track.track_id not in bar._popular_dur or track.track_id not in next_bar._popular_dur:
                continue
            if not bar._track_is_high[track.track_id] or not next_bar._track_is_high[track.track_id]:
                continue

            if bar._popular_dur[track.track_id] != next_bar._popular_dur[track.track_id]:
                moving[bid] = tid
                break

    moving_locs = [(bid, tid) for bid, tid in enumerate(moving) if tid is not None]
    return moving_locs if return_loc else len(moving_locs)

def _peek_track_next_bar(bars: list[BarStruct], bid: int, track_id):
    next_bar_first_note, next_is_mono = None, True
    if bid + 1 < len(bars):
        track_next_bar = next((t for t in bars[bid + 1].tracks if t.track_id == track_id), None)
        if track_next_bar:
            first_n = track_next_bar.notes[0]
            # Check if first note is at the downbeat
            if first_n.pos == 0:
                next_bar_first_note = first_n
                # Check if next bar's start is mono (2nd note must be after 1st)
                if len(track_next_bar.notes) > 1:
                    next_is_mono = track_next_bar.notes[1].pos > first_n.pos
    return next_bar_first_note, next_is_mono

def eval_ornament(
    bars: list[BarStruct],
    ornament_dur = (QUARTER // 4, QUARTER // 2),
    resolution_min_dur = (QUARTER // 2, QUARTER),
    ornament_seq_len = (4, 4),
    ornament_uniq_pitch = (3, 3),
    debug_break = None,
    return_loc = False,
):
    ornament_locs = []
    for bid, bar in enumerate(bars):
        found_in_bar = False
        for tid, track in enumerate(bar.tracks):
            if (bid + 1, tid + 1) == debug_break:
                __import__("pdb").set_trace()
            notes = track.notes

            # 1. PEEK LOGIC: Look for the same track in the next bar
            next_bar_first_note, next_is_mono = _peek_track_next_bar(bars, bid, track.track_id)

            # 2. CONSTRUCT DATA WINDOW
            # If next_note exists, we append it to our local window for calculation
            eval_notes = list(notes) + [next_bar_first_note] * bool(next_bar_first_note)

            # 3. CALCULATE GAPS
            # Gap for internal notes: next.pos - current.pos
            # Gap for the bridge: (bar_length - current.pos) + next.pos
            gaps = []
            for i in range(len(eval_notes) - 1):
                if i < len(notes) - 1:
                    gaps.append(eval_notes[i + 1].pos - eval_notes[i].pos)
                else:
                    # This is the bridge gap to the next bar
                    gaps.append((bar.duration_q - eval_notes[i].pos) + eval_notes[i + 1].pos)

            is_mono = all(g > 0 for g in gaps) and next_is_mono
            if not is_mono:
                continue # For simplicity, assume the ornament track is mono

            # 4. CHECK LOGIC
            # The leading notes must be short, and the resolution must be long
            for o_dur, r_min_dur, o_seq_len, o_uniq_pitch in zip(ornament_dur, resolution_min_dur, ornament_seq_len, ornament_uniq_pitch):
                for i in range(len(eval_notes) - o_seq_len + 1):
                    last_note: MusicNote = eval_notes[i + o_seq_len - 1]

                    last_note_is_high = ((bars[bid + 1] if last_note == next_bar_first_note else bar)._skyline[last_note.pos] == last_note.pitch)
                    if not last_note_is_high:
                        # This ensures the ornament is clearly perceived, although might miss some cases
                        continue

                    # The "body" of the ornament (all notes except the last resolution)
                    body_gaps = gaps[i: i + o_seq_len - 1]
                    # The "resolution" gap (the gap after the last ornament note)
                    res_gap = gaps[i + o_seq_len - 1] if (i + o_seq_len - 1) < len(gaps) else None
                    # Final note duration
                    last_note_dur = last_note.duration_q

                    # Might miss some triplet ornaments
                    all_short = all(max(gap, QUARTER // 4) == o_dur for gap in body_gaps)
                    resolves_long = (res_gap is not None and res_gap >= r_min_dur) or (res_gap is None and last_note_dur >= r_min_dur)

                    pitches = [eval_notes[i + k].pitch for k in range(o_seq_len)]
                    has_enough_variation = len(set(pitches)) >= o_uniq_pitch

                    if all_short and resolves_long and has_enough_variation:
                        found_in_bar = True
                        found_o_dur = o_dur
                        break
                if found_in_bar:
                    break

            if found_in_bar:
                ornament_locs.append((bid, tid, found_o_dur))
                break # For simplicity, count once per bar

    return ornament_locs if return_loc else len(ornament_locs)

if __name__ == "__main__":
    def make_test_bars(notes):
        bar = BarStruct(duration_q=32, tracks=[TrackStruct(track_id=0, program=0, notes=notes)])
        bar._sliced_notes = {0: notes}
        return [bar]

    notes = [
        MusicNote(pos=0, pitch=60, duration_q=4),
        MusicNote(pos=4, pitch=59, duration_q=4),
        MusicNote(pos=8, pitch=60, duration_q=4),
        MusicNote(pos=12, pitch=62, duration_q=8),
    ]
    bars = make_test_bars(notes)
    analyze_melodic_(bars)
    print(eval_ornament(bars))
