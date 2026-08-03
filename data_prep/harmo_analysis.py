import itertools
import numpy as np
from functools import lru_cache
from dataclasses import replace
from miditoolkit import TimeSignature
from collections import defaultdict
from data_prep.data_structure import BarStruct, MusicNote, NOTE_ORDER

CHORD_TEMPLATES = {
    "maj"  : [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    "min"  : [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    "aug"  : [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    "dim"  : [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    "dim7" : [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    "hdim7": [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    "min7" : [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    "dom7" : [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    "maj7" : [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
}

PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

class HarmonySkeletonAnalyzer:
    def __init__(
        self,
        short_dur_threshold = 1,
        dp_penalty_lambda = 3.0,
        triad_match_strict = False,
        print_info = False,
    ):
        self.short_dur_threshold = short_dur_threshold
        self.dp_penalty_lambda = dp_penalty_lambda
        self.print_info = print_info

        # Fetch shared data (Calculated once, shared by all instances)
        self.chord_data, self.W_norm, self.W_binary = self._get_shared_chord_data()

        # Seventh chords need all 4 pitch classes present to match
        # Triads need all 3 pitch classes present if strict, otherwise only 1
        self.required_pc_cnt = [
            4 if chord_type.endswith('7') else (3 if triad_match_strict else 1)
            for _, chord_type in self.chord_data
        ]

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_shared_chord_data():
        chord_data: list[str, str] = [] # Stores (root, chord_type)
        W_norm = []

        for chord_type, template in CHORD_TEMPLATES.items():
            template = np.array(template, dtype=float)
            for root_idx, root in enumerate(PITCH_CLASS_NAMES):
                chord_data.append((root, chord_type))

                transposed = np.roll(template, root_idx)
                norm = np.linalg.norm(transposed)
                normalized_template = transposed / norm if norm > 0 else transposed
                W_norm.append(normalized_template)

        W_norm = np.array(W_norm)
        W_binary = (W_norm > 0).astype(int)
        return chord_data, W_norm, W_binary

    def filter_short_notes(self, notes: list[MusicNote]) -> list[MusicNote]:
        pitch_durations = defaultdict(int)
        for n in notes:
            pitch_durations[n.pitch] += n.duration_q
        return [n for n in notes if pitch_durations[n.pitch] >= self.short_dur_threshold]

    def match_chord_template(self, notes: list[MusicNote], return_score=False):
        UNMATCHED = (("N", "N"), 0.0) if return_score else ("N", "N")
        if not notes:
            return UNMATCHED

        pc_duration = np.zeros(12)
        for n in notes:
            pc_duration[n.pitch % 12] += n.duration_q
        norm = np.linalg.norm(pc_duration)
        pc_duration_norm = pc_duration / norm if norm > 0 else pc_duration

        scores = self.W_norm @ pc_duration_norm

        # Check Chord Completeness
        pc_existence = (pc_duration > 0).astype(int)
        present_pc = self.W_binary @ pc_existence
        valid_mask = (present_pc >= self.required_pc_cnt)

        if not np.any(valid_mask):
            return UNMATCHED
        scores[~valid_mask] = -1.0

        best_idx = np.argmax(scores) # Will first try to match 'maj', then 'min', etc

        if scores[best_idx] <= 0:
            return UNMATCHED

        if return_score:
            return self.chord_data[best_idx], scores[best_idx]
        return self.chord_data[best_idx]

    def get_notes_by_beat(self, notes: list[MusicNote], bar_dur_q: int, beat_dur_q: int) -> list[list[MusicNote]]:
        num_beats = bar_dur_q // beat_dur_q # Division remainder should occur rarely
        notes_by_beat = [[] for _ in range(num_beats)]

        for note in notes:
            note_start = note.pos
            note_end = note.end_pos

            # Calculate which beats this note touches
            first_beat = note_start // beat_dur_q
            last_beat = (note_end - 1) // beat_dur_q

            for beat in range(first_beat, min(last_beat + 1, num_beats)):
                beat_start = beat * beat_dur_q
                beat_end = (beat + 1) * beat_dur_q

                chunk_start = max(note_start, beat_start)
                chunk_end = min(note_end, beat_end)
                chunk_dur = chunk_end - chunk_start
                if chunk_dur > 0:
                    notes_by_beat[beat].append(replace(note, pos=chunk_start, duration_q=chunk_dur))

        return notes_by_beat

    def select_representative_pitches(self, notes: list[MusicNote], chord_pc: set[int], close_interval: int = 2):
        """
        Use Dynamic Programming to select a subset of pitches that best represent the chord.
        DP is based on maximizing the total duration under the constraint of no non-harmonic close intervals.
        By default, pitches that belong to the template chord are included.
        If any chord tone is missing, receives penalty of its duration times lambda.
        """
        if not notes:
            return ()

        # Pre-compute total duration per pitch
        pitch_durations = defaultdict(int)
        for note in notes:
            pitch_durations[note.pitch] += note.duration_q

        sorted_pitches = sorted(pitch_durations.keys())
        n = len(sorted_pitches)

        # DP Tables
        dp = np.zeros(n, dtype=int)
        path = np.full(n, -1, dtype=int)

        # Pre-compute the prefix sums of the penalties
        missing_chord_tone_penalty = np.zeros(n, dtype=int)
        for i, pitch in enumerate(sorted_pitches):
            if pitch % 12 in chord_pc:
                missing_chord_tone_penalty[i] = pitch_durations[pitch]
        penalty_prefix_sum = np.cumsum(missing_chord_tone_penalty) * self.dp_penalty_lambda

        # DP Execution: dp[i] is the maximal score of including i, up to i
        for i in range(n):
            cur_pitch = sorted_pitches[i]
            cur_dur = pitch_durations[cur_pitch]

            # Trivial case: only include pitch i
            penalty = penalty_prefix_sum[i - 1] if i else 0
            dp[i] = cur_dur - penalty

            # Transition: dp[j] -> dp[i]
            for j in range(i):
                prev_pitch = sorted_pitches[j]

                # Constraint: Any two included pitches must be apart, unless both pitches belong to the template chord
                is_apart = (cur_pitch - prev_pitch > close_interval)
                is_both_chordal = cur_pitch % 12 in chord_pc and prev_pitch % 12 in chord_pc

                if is_apart or is_both_chordal:
                    # New score = Score at j + Current duration - Penalty for missing chord tones between j and i
                    penalty = penalty_prefix_sum[i - 1] - penalty_prefix_sum[j]
                    new_score = dp[j] + cur_dur - penalty

                    if new_score > dp[i]:
                        dp[i] = new_score
                        path[i] = j

        # Backtrack through the path to get the DP solution
        idx = np.argmax(dp)
        selected_pitches = []
        while idx != -1:
            selected_pitches.append(sorted_pitches[idx])
            idx = path[idx]

        return tuple(sorted(selected_pitches))

    def analyze_bar(self, notes: list[MusicNote], bar_dur_q: int, beat_dur_q: int) -> list[MusicNote]:
        if not notes:
            return []

        notes_by_beat = self.get_notes_by_beat(notes, bar_dur_q, beat_dur_q)
        harmo_outline = []
        for beat_id, notes_in_beat in enumerate(notes_by_beat):
            # Filter short notes (may be artifacts of quantization and slicing)
            notes_in_beat = self.filter_short_notes(notes_in_beat)
            if not notes_in_beat:
                continue

            root, chord_type = self.match_chord_template(notes_in_beat)
            chord_pc = get_chord_pc(root, chord_type)
            selected_pitches = self.select_representative_pitches(notes_in_beat, chord_pc)
            if not selected_pitches:
                continue

            chord_pos = beat_id * beat_dur_q
            for pitch in selected_pitches:
                harmo_outline.append(MusicNote(chord_pos, beat_dur_q, pitch))

        harmo_outline.sort(key=NOTE_ORDER)
        return harmo_outline

    def purify_harmony_(self, bars: list[BarStruct]):
        """
        Prune the harmony into pure template chords.
        Recommended to set `triad_match_strict` to True for harmony purification.
        """
        _orig_num_notes = 0
        _pure_num_notes = 0

        for bar in bars:
            if not bar.harmony_track or not bar.harmony_track.notes:
                continue

            _orig_num_notes += len(bar.harmony_track.notes)
            bar.harmony_track.notes.sort(key=lambda n: n.pos)
            purified_notes = []

            for _, pos_group in itertools.groupby(bar.harmony_track.notes, key=lambda note: note.pos):
                pos_group = list(pos_group)
                root, chord_type = self.match_chord_template(pos_group)
                if root == "N":
                    # Not enough pitch classes to form a chord, keep the notes intact
                    purified_notes.extend(pos_group)
                    continue

                chord_pc = get_chord_pc(root, chord_type)
                for note in pos_group:
                    if note.pitch % 12 in chord_pc:
                        purified_notes.append(note)

            bar.harmony_track.notes = purified_notes
            _pure_num_notes += len(purified_notes)

        if self.print_info:
            ratio = _pure_num_notes / _orig_num_notes if _orig_num_notes else 0
            print(f"Purified {_pure_num_notes} notes out of {_orig_num_notes}, ratio: {ratio:.1%}")

@lru_cache(maxsize=1)
def CHORD_PC_SETS():
    chord_pc_sets = {}
    for root_idx, root in enumerate(PITCH_CLASS_NAMES):
        for chord_type, template in CHORD_TEMPLATES.items():
            indices = [i for i, val in enumerate(template) if val == 1]
            shifted = set((idx + root_idx) % 12 for idx in indices)
            chord_pc_sets[f"{root}:{chord_type}"] = shifted
    return chord_pc_sets

def get_chord_pc(root, chord_type):
    chord_key = f"{root}:{chord_type}"
    return CHORD_PC_SETS().get(chord_key, set())

def get_beats_per_bar(time_sig: TimeSignature) -> int:
    """ Get number of beats in a bar for a time signature (e.g. 4 for 4/4, 2 for 6/8) """
    # Logic to distinguish compound meters like 6/8 and simple meters like 4/4
    is_compound = (
        time_sig.numerator % 3 == 0 and time_sig.numerator > 3
        and time_sig.numerator * 8 >= 3 * time_sig.denominator # Excludes 3/8 from compound meter
    )

    if is_compound:
        beat_per_bar = time_sig.numerator // 3
    elif time_sig.denominator >= 8:
        # This is rare, for simplicity, treat the whole bar as one beat (e.g. 5/8)
        beat_per_bar = 1
    else:
        beat_per_bar = time_sig.numerator

    return beat_per_bar
