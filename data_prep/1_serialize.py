import os
import random
from tqdm import tqdm
from arch.config import *
from data_prep.main import preprocess_midi
from utils.parseargs import parseargs

def main(
    chunk_idx: int = 0,
    tot_chunks: int = 1,
    midi_dir: str = MIDI_DATASET_DIR,
    processed_dir: str = PROCESSED_DIR,
):
    os.makedirs(f"{processed_dir}/chunks", exist_ok=True)
    os.makedirs(f"{processed_dir}/errors", exist_ok=True)

    output_file = f"{processed_dir}/chunks/{chunk_idx + 1}_of_{tot_chunks}.txt"
    error_file = f"{processed_dir}/errors/{chunk_idx + 1}_of_{tot_chunks}.log"
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists, skipped.")
        return

    files = [os.path.join(r, f) for r, _, fs in os.walk(midi_dir) for f in fs if f.lower().endswith((".mid", ".midi"))]
    files.sort()
    random.seed(42)
    random.shuffle(files)

    # Split files into chunks for multiprocessing
    if tot_chunks > len(files):
        raise ValueError(f"{tot_chunks=} exceeds {len(files)=}")
    chunk_size = len(files) // tot_chunks
    start_idx = chunk_idx * chunk_size
    # The last chunk absorbs the division remainder
    end_idx = start_idx + chunk_size if chunk_idx < tot_chunks - 1 else len(files)
    print(f"Chunk {chunk_idx + 1}/{tot_chunks}: from {start_idx} to {end_idx}")

    all_data = []
    errors = []
    for path in tqdm(
        files[start_idx: end_idx],
        desc=f"Processing Chunk {chunk_idx + 1}/{tot_chunks}",
        mininterval=TQDM_MIN_INTERVAL
    ):
        try:
            tokens = preprocess_midi(path, analyze_harmo=True, serialize=True)
            all_data.append(tokens)
        except Exception as e:
            errors.append(f"Error in {path}: {e}")
            continue
    with open(output_file, "w") as f:
        f.write("\n".join(all_data))
    with open(error_file, "w") as f:
        f.write("\n".join(errors))

if __name__ == "__main__":
    parseargs(main)
