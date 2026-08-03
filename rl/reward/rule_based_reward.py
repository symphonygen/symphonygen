from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from rl.config import RULES_FOR_REWARD, RULES_FOR_EVAL
from rl.base_class import Rewarder
from rl.reward.components.dissonance import eval_dissonance
from rl.reward.components.melodic import analyze_melodic_, eval_melody_moving, eval_ornament
from rl.reward.components.recall import eval_harmony_precision, eval_harmony_recall
from rl.reward.utils_ import compute_linear_plateau_reward
from arch.config import DEBUG_FINAL_EVAL, DEBUG_REWARD
from data_prep.data_structure import BarStruct
from data_prep.main import analyze_harmo_outline, bar_slice_notes_, deserialize_song, preprocess_midi, serialize_song
from utils.conventions import TOKENS_JSON
from utils.my_multiprocess import my_multiprocess

# Keys of `RULES_FOR_EVAL` that `compute_eval_metrics` understands
EVAL_METRIC_KEYS = ("dissonance", "moving", "ornament", "precision", "recall")

def compute_reward(rel_name: str, song_tokens: str, config: dict):
    bars = deserialize_song(song_tokens)

    reward_dict = {}

    track_config = config.get("track")
    if track_config:
        total_trk = 0
        for bar in bars:
            trk_score = compute_linear_plateau_reward(len(bar.tracks), track_config)
            trk_score = np.tanh(trk_score / track_config["pre_tanh_scale"])
            total_trk += trk_score
        reward_dict["track"] = total_trk / len(bars) if bars else 0
        reward_dict["_raw_track"] = sum(len(bar.tracks) for bar in bars) / len(bars) if bars else 0

    if any(key in config for key in EVAL_METRIC_KEYS):
        reward_dict |= compute_eval_metrics(bars, config)

    if DEBUG_REWARD:
        print(f"{rel_name=}: {reward_dict=}")

    return rel_name, {k: float(v) for k, v in reward_dict.items()}

def compute_eval_metrics(bars: list[BarStruct], config: dict):
    bar_slice_notes_(bars)
    analyze_melodic_(bars)

    metric_dict = {}

    if "dissonance" in config:
        diss_HN, diss_NN = eval_dissonance(bars)
        metric_dict["dissonance_HN"] = diss_HN
        metric_dict["dissonance_NN"] = diss_NN

    if "moving" in config:
        mov_score = eval_melody_moving(bars) / len(bars) if bars else 0
        metric_dict["moving"] = mov_score

    if "ornament" in config:
        orn_score = eval_ornament(bars) / len(bars) if bars else 0
        metric_dict["ornament"] = orn_score

    if "precision" in config or "recall" in config:
        recon_outline = analyze_harmo_outline(bars, derive_beat_dur=True)

        prc_score = eval_harmony_precision(bars, recon_outline)
        metric_dict["precision"] = prc_score

        rec_score = eval_harmony_recall(bars, recon_outline)
        metric_dict["recall"] = rec_score

    return metric_dict

class RuleBasedReward(Rewarder):
    def __init__(self, rules_for_eval=False):
        self.config = {
            "name": "rule_based",
            **RULES_FOR_REWARD,
        }
        if rules_for_eval:
            self.config |= RULES_FOR_EVAL

    def compute_reward_dict(self, rollout_root: str | Path, rel_names: list[str | Path], missing_ok=False):
        rollout_root = Path(rollout_root)
        all_tokens = load_or_serialize_songs(rollout_root, rel_names)

        tasks = []
        missing = []
        for rel_name in rel_names:
            rel_name = Path(rel_name)
            cond = str(rel_name.parent)
            song = rel_name.name

            try:
                song_tokens = all_tokens[cond][song]
            except KeyError:
                missing.append(str(rel_name))
                continue
            tasks.append((str(rel_name), song_tokens, self.config))

        if missing:
            msg = (f"No tokens for {len(missing)}/{len(rel_names)} songs "
                   f"(failed generation/preprocessing?), e.g. {missing[:3]}")
            if not missing_ok:
                raise KeyError(msg)
            print(f"Warning: {msg}")

        rewards = my_multiprocess(compute_reward, tasks, serial=DEBUG_FINAL_EVAL)

        return dict(rewards or [])

def load_or_serialize_songs(rollout_root: Path, rel_names: list[str | Path]) -> dict[str, dict[str, str]]:
    """
    tokens.json caches the serialized songs; the GRPO rollout writes it directly
    so it contains the exact Reference Outline the songs were conditioned on.
    Songs missing from the cache are serialized from their MIDI files.
    """
    tokens_path = rollout_root / TOKENS_JSON
    all_tokens = defaultdict(dict)
    if tokens_path.exists():
        with open(tokens_path) as f:
            for cond, group_tokens in json.load(f).items():
                all_tokens[cond] |= group_tokens

    todo = [
        rel_name for rel_name in map(Path, rel_names)
        if str(rel_name.parent) not in all_tokens
        or rel_name.name not in all_tokens[str(rel_name.parent)]
    ]
    if not todo:
        return all_tokens

    print(f"{len(todo)} songs missing from {tokens_path}, serializing...")
    for rel_name in tqdm(todo):
        midi_path = rollout_root / f"{rel_name}.mid"
        if not midi_path.exists():
            continue
        try:
            bars = preprocess_midi(midi_path, analyze_harmo_if_not_exist=True)
        except ValueError as e:
            print(f"{rel_name} skipped: {e}")
            continue
        all_tokens[str(rel_name.parent)][rel_name.name] = serialize_song(bars)
    with open(tokens_path, "w") as f:
        json.dump(all_tokens, f)
    return all_tokens
