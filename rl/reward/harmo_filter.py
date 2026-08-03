import random
from rl.config import RULES_FOR_HARMO_FILTER
from rl.reward.components.chord import eval_harmony_cadence_num
from rl.reward.components.density import eval_harmony_density
from rl.reward.components.novelty import eval_harmony_bar_repeat_ratio, eval_harmony_beat_repeat_ratio
from rl.reward.utils_ import compute_linear_plateau_reward, weighted_tot_score
from arch.config import DEBUG_REWARD
from data_prep.data_structure import BarStruct

def filter_harmony_by_rule(
    all_bars_list: list[list[BarStruct]],
    num_requested: int | None = None,
    config: dict = RULES_FOR_HARMO_FILTER,
):
    weights = config["weights"]
    threshold = config["threshold"]
    scores = [compute_harmony_score(bars, config) for bars in all_bars_list]
    scored = []
    for bars, score_dict in zip(all_bars_list, scores):
        tot_score = weighted_tot_score(score_dict, weights)
        scored.append((bars, tot_score))

    good = [
        item for item, score in scored
        if score >= threshold
    ]
    print(f"{len(good)} of {len(scored)} harmonies have total score >= {threshold}")
    if DEBUG_REWARD:
        print(f"Scores: {[round(score, 2) for _, score in scored]}")

    if num_requested and len(good) > num_requested:
        good = random.sample(good, num_requested)
    return good

def compute_harmony_score(bars: list[BarStruct], config: dict):
    score_dict = {}

    harmo_den_config = config.get("harmony_density")
    if harmo_den_config:
        tot_harmo_den_score = 0
        for bar in bars:
            harmo_density_score = eval_harmony_density(bar, max_notes_per_beat=harmo_den_config["max_notes_per_beat"])
            tot_harmo_den_score += compute_linear_plateau_reward(harmo_density_score, harmo_den_config)
        score_dict["harmony_density"] = tot_harmo_den_score / len(bars)

    harmo_bar_novel_config = config.get("harmony_bar_repeat")
    if harmo_bar_novel_config:
        harmo_repeat_ratio = eval_harmony_bar_repeat_ratio(bars, harmo_bar_novel_config["loop_periods"])
        score_dict["harmony_bar_repeat"] = compute_linear_plateau_reward(harmo_repeat_ratio, harmo_bar_novel_config)

    harmo_beat_novel_config = config.get("harmony_beat_repeat")
    if harmo_beat_novel_config:
        harmo_beat_repeat_ratio = eval_harmony_beat_repeat_ratio(bars)
        score_dict["harmony_beat_repeat"] = compute_linear_plateau_reward(harmo_beat_repeat_ratio, harmo_beat_novel_config)

    harmo_cad_config = config.get("harmony_cadence")
    if harmo_cad_config:
        harmo_cadence_num = eval_harmony_cadence_num(bars, harmo_cad_config["chord_match_threshold"])
        score_dict["harmony_cadence"] = 1.0 if harmo_cadence_num >= harmo_cad_config["num_threshold_32_bar"] else 0.0

    return score_dict
