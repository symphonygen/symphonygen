from pathlib import Path
import numpy as np
from rl.lib.clamp3 import CLAMP3Encoder
from utils.midi2audio import process_convert_midi_to_audio
from utils.my_multiprocess import my_batch_convert
from utils.parseargs import parseargs

class CLAMP3Extractor:
    _model: CLAMP3Encoder | None = None

    @staticmethod
    def init():
        if CLAMP3Extractor._model is None:
            CLAMP3Extractor._model = CLAMP3Encoder()

    @staticmethod
    def extract(audio_path: Path, output_path: Path=None):
        CLAMP3Extractor.init()
        audio_emb = CLAMP3Extractor._model.encode_audio(audio_path)
        if output_path is None:
            return audio_emb
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, audio_emb)

def main(input_folder: str, output_folder: str=None):
    output_folder = output_folder or input_folder
    my_batch_convert(process_convert_midi_to_audio, input_folder, output_folder, ".mid", ".mp3", check_empty=True)
    my_batch_convert(CLAMP3Extractor.extract, input_folder, output_folder, ".mp3", ".npy", serial=True)

if __name__ == "__main__":
    parseargs(main)
