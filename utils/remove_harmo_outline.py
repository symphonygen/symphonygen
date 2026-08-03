import mido
from pathlib import Path
from arch.vocab import META_HARMO_TRACK_ID
from utils.my_multiprocess import my_batch_convert

POSSIBLE_HARMO_OUTLINE_TRACK_NAMES = {
    f'Track {META_HARMO_TRACK_ID}',
    'Reference Outline',
    'Harmonic Outline'
}

def midi_remove_harmo_outline(midi_path: Path, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        midi_obj = mido.MidiFile(midi_path)
        for i, track in enumerate(midi_obj.tracks):
            if track.name in POSSIBLE_HARMO_OUTLINE_TRACK_NAMES:
                midi_obj.tracks.pop(i)
                midi_obj.save(output_path)
                return output_path
    except Exception as e:
        print(f"Error processing {midi_path}: {e}")
    return midi_path

if __name__ == '__main__':
    import sys
    my_batch_convert(midi_remove_harmo_outline, sys.argv[1], sys.argv[1], ".mid", ".no_harmo.mid")
