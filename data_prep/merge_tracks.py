from miditoolkit.midi.parser import MidiFile
from arch.vocab import MIDI_TRACK_LIMIT

def merge_tracks_(midi_obj: MidiFile, track_limit=MIDI_TRACK_LIMIT):
    """ Merge tracks until under track limit, while preserving the original track order """
    for inst in midi_obj.instruments:
        inst.remove_notes_with_no_duration()

    inst_list = midi_obj.instruments
    if len(inst_list) <= track_limit:
        return

    # Capture original order
    for i, inst in enumerate(midi_obj.instruments):
        inst._orig_idx = i

    # Identify what tracks to keep based on priority
    # (Drum first, then highest note count first)
    priority_list = sorted(inst_list, key=lambda x: (not x.is_drum, -len(x.notes)))

    kept = priority_list[:track_limit]
    to_merge = priority_list[track_limit:]

    # Try to merge tracks of the same program
    for extra in to_merge:
        for target in kept:
            if extra.program == target.program or extra.is_drum == target.is_drum == True:
                target.notes.extend(extra.notes)
                break

    midi_obj.instruments = kept

    # Restore original order
    midi_obj.instruments.sort(key=lambda x: x._orig_idx)
