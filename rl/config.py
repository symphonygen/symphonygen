import numpy as np
from arch.config import ASSET_DIR

CLAMP_MODEL_PATH = f"{ASSET_DIR}/weights_clamp3.pth"

REF_FEATURE_DIR = f"{ASSET_DIR}/clamp3_ref/your_ref_audio/"

GROUP_PROMPT_SIZE = 16
GROUP_SIZE = 32

GRPO_LOSS_CONFIG = {
    "head_names": ("event", "track_id", "inst"),
    "loss_scales": (1.0, 5.0, 5.0),
    "kl_betas": (0.03, 0.01, 0.01),
}

HARMO_FILTER_LOG_PROB_LOW = 0.25
HARMO_FILTER_LOG_PROB_HIGH = 0.5

MAX_HARMO_BATCH_SIZE = 128
MAX_BATCH_SIZE = 256

RULES_FOR_REWARD = {
    "track": {
        "low": 8,
        "high": 20,
        "low_max_diff": 4,
        "high_max_diff": 12,

        # 6 or 26 tracks = 0.9
        "pre_tanh_scale": 0.5 / np.arctanh(0.9),
    },
}

RULES_FOR_HARMO_FILTER = {
    "harmony_bar_repeat": {
        "loop_periods": [1, 2, 4],
        "low": 0.0,
        "high": 0.15,
        "low_max_diff": 0.0,
        "high_max_diff": 0.1,
    },
    "harmony_beat_repeat": {
        "low": 0.0,
        "high": 0.15,
        "low_max_diff": 0.0,
        "high_max_diff": 0.1,
    },
    "harmony_density": {
        "max_notes_per_beat": 16,
        "low": 0.4,
        "high": 1.0,
        "low_max_diff": 0.2,
        "high_max_diff": 0.2,
    },
    "harmony_cadence": {
        "chord_match_threshold": 0.8,
        "num_threshold_32_bar": 4,
    },

    "weights": {
        "default": 1.0,
    },
    "threshold": 3.8,
}

RULES_FOR_EVAL = {
    "dissonance": True,
    "recall": True,
    "moving": True,
    "ornament": True,
}
