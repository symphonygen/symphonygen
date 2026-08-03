import pickle
from arch.config import *
from data_prep.vocab import is_met_unsafe

def index_bar_offset(token_path: str) -> list[list[int]]:
    """
    Returns the index map, where index[i] contains all bar start offsets for song i.
    The last element of each inner list is the offset of the end of the song.
    """
    print(f"Generating bar index for {token_path}...")

    index = []
    with open(token_path, 'rb') as f:
        # Read raw bytes to ensure exact offsets
        while True:
            start_of_song_offset = f.tell()
            line = f.readline()

            if not line:
                break
            line_str = line.decode('utf-8').rstrip('\n')
            if not line_str:
                # We can skip empty songs
                continue

            # This list stores the absolute byte offset of every bar start
            bar_offsets = []

            for i in range(0, len(line_str), 2):
                if is_met_unsafe(line_str[i]):
                    offset = start_of_song_offset + i # Assume string is ASCII
                    bar_offsets.append(offset)

            # Add the end of each song as the final boundary
            end_of_song_offset = start_of_song_offset + len(line_str)
            bar_offsets.append(end_of_song_offset)

            index.append(bar_offsets)

            if len(index) % 1000 == 0:
                print(f"Indexed {len(index)} songs.")

    output_path = bar_index_path(token_path)
    with open(output_path, 'wb') as f:
        pickle.dump(index, f)
    print(f"Bar index saved to {output_path}")

if __name__ == "__main__":
    for split in ('train', 'val'):
        index_bar_offset(os.path.join(PROCESSED_DIR, f'{split}_data.txt'))
