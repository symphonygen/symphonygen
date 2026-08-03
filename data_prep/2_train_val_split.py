import os
import glob
from tqdm import tqdm
from arch.config import *
from utils.parseargs import parseargs

def main(
    processed_dir: str=PROCESSED_DIR,
    split_ratio: float=0.9,
    num_chunks: int=None
):
    """ Combine chunk files while creating train/val splits """
    chunk_pattern = os.path.join(processed_dir, 'chunks/*_of_{}.txt'.format(num_chunks or '*'))
    chunk_files = sorted(glob.glob(chunk_pattern))
    num_chunks = len(chunk_files)
    split_index = int(num_chunks * split_ratio)
    for split in ('train', 'val'):
        selected_files = (chunk_files[:split_index] if split == 'train' else chunk_files[split_index:])
        output_path = os.path.join(processed_dir, f'{split}_data.txt')
        with open(output_path, 'w') as outfile:
            for chunk_file in tqdm(selected_files, desc=f'Combining {split} chunks', mininterval=TQDM_MIN_INTERVAL):
                with open(chunk_file) as infile:
                    outfile.write(infile.read())
                outfile.write('\n')

        print(f'Combined {len(selected_files)} chunks into {output_path}')

    # Combine error logs
    error_pattern = os.path.join(processed_dir, 'errors/*_of_{}.log'.format(num_chunks or '*'))
    error_files = sorted(glob.glob(error_pattern))

    error_path = os.path.join(processed_dir, 'all_errors.log')
    with open(error_path, 'w') as outfile:
        for error_file in error_files:
            with open(error_file) as infile:
                outfile.write(infile.read())
            outfile.write('\n')

    print(f'Combined {len(error_files)} error logs into {error_path}')

if __name__ == '__main__':
    parseargs(main)
