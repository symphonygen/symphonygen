from arch.config import ASSET_DIR

HARMO_MODEL_PATH = f"{ASSET_DIR}/stage_one_finetuned.pt"
PRETRAIN_MODEL_PATH = f"{ASSET_DIR}/stage_two_pretrained.pt"
# Reinforced with the pure CLaMP 3 reward (subjective test round 1)
GRPO_MODEL_PATH = f"{ASSET_DIR}/grpo_clamp_epoch_10.pt"
# Reinforced with the composite CLaMP 3 + track density reward (subjective test round 2)
GRPO_MODEL_PATH_TRACK = f"{ASSET_DIR}/grpo_clamp+track_epoch_6.pt"

MODEL_CONFIGS = [
    # {"label": "pretrained", "path": PRETRAIN_MODEL_PATH, "diss_HN": 0, "diss_NN": 0, "register_decay": False},
    {"label": "reinforced", "path": GRPO_MODEL_PATH, "diss_HN": 1, "diss_NN": 10, "register_decay": False},
    {"label": "reinforced+track", "path": GRPO_MODEL_PATH_TRACK, "diss_HN": 1, "diss_NN": 10, "register_decay": True},
]

for model_config in MODEL_CONFIGS:
    model_config["label"] += f"_HN_{model_config['diss_HN']}_NN_{model_config['diss_NN']}"

TRACK_THRESHOLD = 6

FORBID_PIANO = True

# Additional skeleton curation for the round-1 subjective test:
# generated skeletons must open with a major or minor chord (paper Sec. 4.4).
# Set to None to disable the start-chord filter.
START_CHORDS = ["maj", "min"]

# Number of best songs (by composite reward) copied to <midi_dir>/best_songs/
# after evaluation; 0 disables the copy
PICK_BEST_NUM = 0
