from collections import defaultdict
from dataclasses import replace
import itertools
import os
from pathlib import Path
from typing import Dict, List
from miditoolkit import MidiFile
from miditoolkit.midi.containers import (
    Instrument as MidiInst,
    Note as MidiNote,
    TimeSignature,
)
from arch.config import *
from arch.vocab import *
from data_prep.data_structure import *
from data_prep.vocab import *
from data_prep.harmo_analysis import HarmonySkeletonAnalyzer, get_beats_per_bar
from data_prep.merge_tracks import merge_tracks_
from utils.remove_harmo_outline import POSSIBLE_HARMO_OUTLINE_TRACK_NAMES

def extract_and_slice_notes(midi_obj: MidiFile, max_dur_beat: int = 8) -> List[MidiRawNote]:
    """ Slice notes that are longer than `max_dur_beat` beats """
    raw_notes: List[MidiRawNote] = []
    max_dur = max_dur_beat * midi_obj.ticks_per_beat

    for track_id, inst in enumerate(midi_obj.instruments):
        program = META_DRUM_INST if inst.is_drum else int(inst.program) # from np.int8
        for note in inst.notes:
            total_dur = note.end - note.start
            cur_start = note.start
            while total_dur > 0:
                chunk_dur = min(total_dur, max_dur)
                raw_notes.append(MidiRawNote(
                    cur_start,
                    note.pitch,
                    program,
                    is_drum=inst.is_drum,
                    track_id=track_id,
                    duration=chunk_dur
                ))
                total_dur -= chunk_dur
                cur_start += chunk_dur

    raw_notes.sort(key=lambda e: e.tick)
    return raw_notes

def quantize_raw_note(note: MidiRawNote, base_tick: int, grid: float):
    rel_tick = round((note.tick - base_tick) / grid)
    duration_q = max(1, round(note.duration / grid))
    return rel_tick, duration_q

def quantize_and_structure_notes(midi_obj: MidiFile, raw_notes: List[MidiRawNote]) -> List[BarStruct]:
    raw_notes.sort(key=lambda e: e.tick)
    sigs = sorted(midi_obj.time_signature_changes, key=lambda s: s.time) or [TimeSignature(4, 4, 0)]

    # NOTE: `max_tick` is the last note's **onset**
    max_tick = raw_notes[-1].tick if raw_notes else 0
    num_notes = len(raw_notes)
    eid = 0
    _32_tick = midi_obj.ticks_per_beat / 8 # MidiFile's ticks_per_beat is actually ticks_per_quarter

    bars = []
    for i, sig in enumerate(sigs):
        ticks_per_bar = int(midi_obj.ticks_per_beat * 4 * sig.numerator / sig.denominator)

        # Quantize bar duration
        bar_dur_q = max(1, round(ticks_per_bar / _32_tick))

        # Determine beat duration
        beats_per_bar = get_beats_per_bar(sig)
        beat_dur_q = max(1, bar_dur_q // beats_per_bar)

        start_tick = sig.time
        end_tick = sigs[i + 1].time if i + 1 < len(sigs) else (max_tick + 1)
        for bar_start in range(start_tick, end_tick, ticks_per_bar):
            bar_end = min(bar_start + ticks_per_bar, end_tick)
            cur_bar = BarStruct(duration_q=bar_dur_q)
            cur_bar._beat_dur_q = beat_dur_q

            # Group notes in the bar by track
            tracks: Dict[int, TrackStruct] = {}
            while eid < num_notes and raw_notes[eid].tick < bar_end:
                e = raw_notes[eid]
                pos, dur_q = quantize_raw_note(e, base_tick=bar_start, grid=_32_tick)
                pos = max(0, min(pos, bar_dur_q - 1))

                if e.track_id not in tracks:
                    tracks[e.track_id] = TrackStruct(e.track_id, e.program)

                note = MusicNote(pos, dur_q, e.pitch)
                tracks[e.track_id].notes.append(note)
                eid += 1

            for track in tracks.values():
                track.notes.sort(key=NOTE_ORDER)
            # Sort tracks by track ID
            cur_bar.tracks = list(sorted(tracks.values(), key=lambda t: t.track_id))
            bars.append(cur_bar)

    return bars

# --- Harmonic Outline ---

def bar_slice_notes_(bars: List[BarStruct]):
    """ Slice notes by bar boundary, to prepare for harmony analysis """
    for bar in bars:
        bar._sliced_notes = defaultdict(list)

    for b_id, bar in enumerate(bars):
        for track in bar.tracks:
            for note in track.notes:
                # The duration in the current bar
                dur_in_cur_bar = min(bar.duration_q - note.pos, note.duration_q)
                bar._sliced_notes[track.track_id].append(replace(note, duration_q=dur_in_cur_bar))

                # The remaining duration
                remaining_dur = note.duration_q - dur_in_cur_bar
                next_b_id = b_id + 1
                while remaining_dur > 0 and next_b_id < len(bars):
                    next_bar = bars[next_b_id]
                    chunk_dur = min(remaining_dur, next_bar.duration_q)

                    next_bar._sliced_notes[track.track_id].append(replace(note, pos=0, duration_q=chunk_dur))

                    remaining_dur -= chunk_dur
                    next_b_id += 1

def analyze_harmo_outline(
    bars: List[BarStruct],
    inplace=False,
    derive_beat_dur=False, # Use the note duration of existing harmony track as beat duration (for re-analysis)
):
    PERCUSSION_INSTS = (
        META_DRUM_INST,
        47, # Timpani
    )
    percussion_tracks = set(
        track.track_id
        for bar in bars for track in bar.tracks
        if track.program in PERCUSSION_INSTS
    )

    harmo_analyzer = HarmonySkeletonAnalyzer()
    if not inplace:
        all_outline_notes = []
    for bar in bars:
        if bar._sliced_notes is None:
            raise RuntimeError(f"Need to run {bar_slice_notes_.__name__} before harmony analysis")

        sliced_notes = [
            note
            for track_id, track_notes in bar._sliced_notes.items()
            if track_id not in percussion_tracks
            for note in track_notes
        ]

        if derive_beat_dur and bar.harmony_track:
            beat_dur = bar.harmony_track.notes[0].duration_q
        else:
            beat_dur = bar._beat_dur_q
        if beat_dur is None and sliced_notes:
            raise RuntimeError("Unknown beat duration, unable to analyze harmony")

        outline_notes = harmo_analyzer.analyze_bar(sliced_notes, bar.duration_q, beat_dur)

        if inplace:
            bar.harmony_track = TrackStruct(META_HARMO_TRACK_ID, META_HARMO_INST, notes=outline_notes)
        else:
            all_outline_notes.append(outline_notes)

    if not inplace:
        return all_outline_notes

# --- Conversion Pipeline ---

def separate_harmo_track_(bars: List[BarStruct]):
    for bar in bars:
        for i, track in enumerate(bar.tracks):
            if track.program == META_HARMO_INST:
                bar.harmony_track = bar.tracks.pop(i)
                bar.harmony_track.track_id = META_HARMO_TRACK_ID
                break

def serialize_song(bars: List[BarStruct]) -> str:
    """
    Serialize songs into short human-readable strings.
    Do not confuse the tokens used for serialization with tokens used by the model.
    """
    tokens = []

    def serialize_track(track: TrackStruct):
        if not track.notes:
            return
        # Track header: ID and Program
        tokens.extend([trk2str(track.track_id), ins2str(track.program)])
        for rel_tick, pos_group in itertools.groupby(track.notes, key=lambda x: x.pos):
            tokens.append(pos2str(rel_tick))
            for note in pos_group:
                # Only need Pitch and Duration here
                tokens.extend([pit2str(note.pitch), dur2str(note.duration_q)])

    for bar in bars:
        tokens.append(met2str(bar.duration_q))
        if bar.harmony_track:
            serialize_track(bar.harmony_track)
        for track in bar.tracks:
            serialize_track(track)

    return ''.join(tokens)

def deserialize_song(tokens: str | List[str]) -> List[BarStruct]:
    if isinstance(tokens, str):
        # Divide by every two chars
        it = iter(tokens)
        tokens = [''.join(two_chars) for two_chars in zip(it, it)]

    bars: List[BarStruct] = []
    i = 0
    while i < len(tokens):
        bar_dur = str2met(tokens[i])
        cur_bar = BarStruct(duration_q=bar_dur)
        i += 1

        while i < len(tokens) and is_trk(tokens[i]):
            track_id = str2trk(tokens[i])
            program = str2ins(tokens[i + 1])
            i += 2

            cur_track = TrackStruct(track_id, program)
            while i < len(tokens) and is_pos(tokens[i]):
                pos = str2pos(tokens[i])
                i += 1
                while i + 1 < len(tokens) and is_pitch(tokens[i]):
                    pitch = str2pit(tokens[i])
                    dur = str2dur(tokens[i + 1])
                    cur_track.notes.append(MusicNote(pos, dur, pitch))
                    i += 2
            cur_bar.tracks.append(cur_track)
        bars.append(cur_bar)

    separate_harmo_track_(bars)
    return bars

def preprocess_midi(
    midi_path_or_obj: str | Path | MidiFile,
    analyze_harmo=False, # Will overwrite the existing harmony track
    analyze_harmo_if_not_exist=False, # Will not overwrite the existing harmony track
    serialize=False
):
    midi_obj = MidiFile(midi_path_or_obj) if isinstance(midi_path_or_obj, (str, Path)) else midi_path_or_obj

    harmo_exist = False
    for inst in midi_obj.instruments:
        if inst.name in POSSIBLE_HARMO_OUTLINE_TRACK_NAMES:
            inst.program = META_HARMO_INST
            harmo_exist = True

    merge_tracks_(midi_obj)
    raw_notes = extract_and_slice_notes(midi_obj)
    bars = quantize_and_structure_notes(midi_obj, raw_notes)

    separate_harmo_track_(bars)
    if analyze_harmo or analyze_harmo_if_not_exist and not harmo_exist:
        bar_slice_notes_(bars)
        analyze_harmo_outline(bars, inplace=True)

    if serialize:
        return serialize_song(bars)
    return bars

def export_midi(bars: List[BarStruct], output_path: str, include_harmo=True, harmo_is_cond=False):
    midi_obj = MidiFile()
    _32_tick = midi_obj.ticks_per_beat // 8
    VELOCITY = 100
    cur_bar_start = 0
    last_sig = (0, 4)
    midi_insts: Dict[int, MidiInst] = {}
    if include_harmo:
        harmo_inst = MidiInst(
            program=0, is_drum=False,
            name="Reference Outline" if harmo_is_cond else "Harmonic Outline"
        )

    def extend_inst(inst: MidiInst, track: TrackStruct):
        for note in track.notes:
            start_tick = cur_bar_start + note.pos * _32_tick
            end_tick = start_tick + note.duration_q * _32_tick
            inst.notes.append(MidiNote(VELOCITY, note.pitch, start_tick, end_tick))

    for bar in bars:
        if not bar.duration_q:
            raise ValueError("Bar has no duration")

        # Determine time signature: for simplicity, assume denominator is 4th or 8th note
        if bar.duration_q % 8 == 0:
            numerator = bar.duration_q // 8
            denominator = 4
        elif bar.duration_q % 4 == 0:
            numerator = bar.duration_q // 4
            denominator = 8
            print(f"Warning: {bar.duration_q=} is not a multiple of 4th notes, using {denominator=}")
        else:
            raise NotImplementedError(f"{bar.duration_q=} is not a multiple of 8th notes, unsupported")
        if (numerator, denominator) != last_sig:
            midi_obj.time_signature_changes.append(
                TimeSignature(numerator, denominator, cur_bar_start)
            )
            last_sig = (numerator, denominator)

        if include_harmo and bar.harmony_track:
            extend_inst(harmo_inst, bar.harmony_track)
        for track in bar.tracks:
            track_id = track.track_id
            if track_id not in midi_insts:
                is_drum = (track.program == META_DRUM_INST)
                program = 0 if is_drum else track.program
                midi_insts[track_id] = MidiInst(program, is_drum=is_drum, name=f"Track {track_id}")
            extend_inst(midi_insts[track_id], track)

        cur_bar_start += bar.duration_q * _32_tick

    if include_harmo:
        midi_obj.instruments.append(harmo_inst)
    # Sort instruments by track ID
    for track_id in sorted(midi_insts.keys()):
        midi_obj.instruments.append(midi_insts[track_id])

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    midi_obj.dump(output_path)
