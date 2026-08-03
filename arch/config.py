import os

SIMPLIFY_HARMONY_DUR = True
HARMO_LAST_LAYER = -2 # Use the second last layer instead of the last layer, for historical reasons

# The working directory of the project holding intermediate or temporary data
WORK_DIR = os.environ.get("WORK_DIR", "data/")

# The result directory holding important assets (checkpoints, reference audio, MuseScore)
ASSET_DIR = os.environ.get("ASSET_DIR", "asset/")

MIDI_DATASET_DIR = f"{WORK_DIR}/SymphonyNet_Dataset"
if SIMPLIFY_HARMONY_DUR:
    PREPROCESS_VER = "simplified"
else:
    PREPROCESS_VER = "rhythmic"
PROCESSED_DIR = f"{WORK_DIR}/{PREPROCESS_VER}_harmo_outline/"
TMP_DIR = f"{WORK_DIR}/tmp/"
os.makedirs(TMP_DIR, exist_ok=True)

TRAIN_DATA_PATH = f"{WORK_DIR}/{PREPROCESS_VER}_harmo_outline/train_data.txt"
VAL_DATA_PATH = f"{WORK_DIR}/{PREPROCESS_VER}_harmo_outline/val_data.txt"
def bar_index_path(token_path: str):
    return token_path.rsplit('.', 1)[0] + '_bar_index.pkl'

BAR_NUM = 32
BAR_STRIDE = BAR_NUM // 2
BEAT_NUM = 4

import sys
if sys.stdout.isatty():
    TQDM_MIN_INTERVAL = 0.1
else:
    TQDM_MIN_INTERVAL = 30

DEBUG = False
DEBUG_PRETRAIN = False
DEBUG_SCHEDULER = False
DEBUG_GEN = False
DEBUG_GRPO = False
DEBUG_REWARD = False
DEBUG_FINAL_EVAL = False
DEBUG_DISS_CONSTRAINER = False
DEBUG_ANY = DEBUG or DEBUG_PRETRAIN or DEBUG_SCHEDULER or DEBUG_GEN or DEBUG_GRPO or DEBUG_REWARD or DEBUG_FINAL_EVAL or DEBUG_DISS_CONSTRAINER

NO_WRITER = DEBUG_ANY
NO_SAVE_LOAD_MODEL = DEBUG_ANY
NO_COMPILE_MODEL = DEBUG_ANY

if DEBUG:
    print(f"{DEBUG=}")
if DEBUG_PRETRAIN:
    print(f"{DEBUG_PRETRAIN=}")
if DEBUG_SCHEDULER:
    print(f"{DEBUG_SCHEDULER=}")
if DEBUG_GEN:
    print(f"{DEBUG_GEN=}")
if DEBUG_GRPO:
    print(f"{DEBUG_GRPO=}")
if DEBUG_REWARD:
    print(f"{DEBUG_REWARD=}")

if NO_WRITER:
    print(f"{NO_WRITER=}")
if NO_SAVE_LOAD_MODEL:
    print(f"{NO_SAVE_LOAD_MODEL=}")
if NO_COMPILE_MODEL:
    print(f"{NO_COMPILE_MODEL=}")
