from typing import Iterable, TypeVar, Callable
from pathlib import Path
import numpy as np
from arch.config import DEBUG_FINAL_EVAL

_typT = TypeVar("_typT")
_typK = TypeVar("_typK")

def load_ref_features(ref_folder, recursive=True):
    ref_root = Path(ref_folder).resolve()
    ref_files = (ref_root.rglob if recursive else ref_root.glob)("*.npy")
    ref_features = np.stack([np.load(f).flatten() for f in ref_files])
    return ref_features

def get_cosine_sim(features, target):
    f_norm = features / np.linalg.norm(features, axis=1, keepdims=True)
    t_norm = target / np.linalg.norm(target)
    return np.dot(f_norm, t_norm)

def compute_linear_plateau_reward(score, config, eps=1e-8):
    low = config["low"]
    high = config["high"]
    low_max_diff = config["low_max_diff"]
    high_max_diff = config["high_max_diff"]

    # If inside the range, perfect reward
    if low - eps < score < high + eps:
        return 1.0

    if score < low:
        diff = low - score
        max_diff = low_max_diff
    else:
        diff = score - high
        max_diff = high_max_diff

    if DEBUG_FINAL_EVAL and max_diff == 0:
        print(f"{score=} {low=} {high=}")
        __import__("pdb").set_trace()

    return max(0.0, 1.0 - (diff / max_diff))

def weighted_tot_score(score_dict: dict[str, float], weights: dict[str, float]):
    return sum(
        weights.get(k, weights["default"]) * r
        for k, r in score_dict.items()
        if not k.startswith("_")
    )

def dedupe_sorted(items: Iterable[_typT], sig: Callable[[_typT], _typK] = None):
    """ Dedupe a sorted iterable to eliminate continuous identical values """
    it = iter(items)
    try:
        last_item = next(it)
    except StopIteration:
        return

    last_sig = sig(last_item) if sig else last_item
    yield last_item

    for item in it:
        current_sig = sig(item) if sig else item
        if current_sig != last_sig:
            yield item
            last_sig = current_sig
