#!/bin/bash

set -e  # Exit on any error

# multiprocessing for MIDI preprocessing
NUM_PROCESSES=$(nproc)
MEMORY_DIVISOR=4  # How much to divide the work to control memory usage
# NUM_CHUNKS=$((NUM_PROCESSES * MEMORY_DIVISOR))
NUM_CHUNKS="${NUM_CHUNKS:-80}" # Fixed number of chunks for reproducibility

echo "Processing $NUM_CHUNKS chunks using $NUM_PROCESSES parallel slots..."

for ((chunk_index = 0; chunk_index < NUM_CHUNKS; chunk_index++)); do
    # Launch process in background
    python3 data_prep/1_serialize.py -c "$chunk_index" -t "$NUM_CHUNKS" &

    # Limit concurrency: if we hit the process limit, wait for one to finish
    if [[ $(jobs -r -p | wc -l) -ge $NUM_PROCESSES ]]; then
        wait -n
    fi
done

# Wait for the last batch to finish
wait

python3 data_prep/2_train_val_split.py -n "$NUM_CHUNKS"
python3 data_prep/3_index_bar.py
