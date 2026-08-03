from collections import defaultdict
import json
from pathlib import Path
import random
import shutil
import mido
import numpy as np
import pandas as pd
from tqdm import tqdm
from Results.config import PICK_BEST_NUM
from rl.base_class import Rewarder
from rl.reward.clamp3_reward import Clamp3Reward
from rl.reward.composite_reward import CompositeReward
from rl.reward.rule_based_reward import RuleBasedReward
from utils.conventions import (
    ADVANTAGES_JSON, BEST_SONGS_DIR, NO_HARMO_SUFFIXES, REWARDS_JSON, REWARDS_TSV,
    cond_of_rel_name,
)
from utils.distributed import dist_gather_dict, is_main_process
from utils.parseargs import parseargs
from data_prep.main import preprocess_midi

EPS = 1e-8

OBJECTIVE_TSV = Path(__file__).resolve().parent / "objective.tsv"

def compute_save_advantages(rewarder: Rewarder, midi_dir, rel_names, expected_group_size=None):
    """
    Computes rewards for `rel_names`, groups them by condition directory and
    normalizes the advantages within each group.

    expected_group_size: when set (GRPO training), warn about incomplete
        groups and raise if a group is larger than expected — an oversized
        group means songs of different conditions were mixed together.
    """
    midi_root = Path(midi_dir).resolve()

    reward_dict = rewarder.compute_reward_dict(midi_root, rel_names, missing_ok=True)
    if not reward_dict:
        raise ValueError(f"No rewards could be computed for {len(rel_names)} songs under {midi_root}")
    sub_reward_keys = [k for k in next(iter(reward_dict.values())) if k != 'reward']

    grouped_reward_dict = {} # {cond_name: {reward_key: [r for song]}}
    for rel_name, sub_rewards in reward_dict.items():
        cond_name = cond_of_rel_name(rel_name).replace("/", "_")
        group: dict[str, list] = grouped_reward_dict.setdefault(cond_name, {"rel_names": []})
        group["rel_names"].append(rel_name)
        for k, r in sub_rewards.items():
            group.setdefault(k, []).append(r)

    tot_reward_to_json = {}
    advantage_to_json = {}
    sub_rewards_to_json = {k: {} for k in sub_reward_keys}

    for cond_name, group in grouped_reward_dict.items():
        group_size = len(group["rel_names"])
        if expected_group_size is not None and group_size != expected_group_size:
            msg = f"Group {cond_name} has {group_size} songs, expected {expected_group_size}"
            if group_size > expected_group_size:
                # More songs than the group size can only mean the layout is corrupt
                raise ValueError(msg)
            print(f"Warning: {msg}; advantages are normalized over the partial group")

        rewards = np.array(group["reward"])
        mean, std = np.mean(rewards), np.std(rewards)
        advantages = (rewards - mean) / (std + EPS)

        tot_reward_to_json.update(dict(zip(group["rel_names"], rewards, strict=True)))
        advantage_to_json.update(dict(zip(group["rel_names"], advantages, strict=True)))
        for k in sub_reward_keys:
            sub_rewards = np.array(group[k])
            sub_rewards_to_json[k].update(dict(zip(group["rel_names"], sub_rewards, strict=True)))

    tot_reward_to_json = dist_gather_dict(tot_reward_to_json)
    advantage_to_json = dist_gather_dict(advantage_to_json)
    metrics = {"mean_reward": np.mean(list(tot_reward_to_json.values()))}
    for k in sub_reward_keys:
        sub_rewards_to_json[k] = dist_gather_dict(sub_rewards_to_json[k])
        metrics[f"{k}_mean_reward"] = np.mean(list(sub_rewards_to_json[k].values()))

    if is_main_process():
        save_eval_results(
            midi_root,
            tot_reward_to_json, advantage_to_json,
            sub_reward_keys, sub_rewards_to_json,
        )

    return metrics

def save_eval_results(
    midi_root: Path,
    rewards: dict[str, float], advantages: dict[str, float],
    sub_reward_keys: list[str], sub_rewards: dict[str, dict[str, float]],
    pick_best_num=PICK_BEST_NUM,
    best_each_cond=True,
):
    # 1. Restructure sub-rewards
    rewards_to_json = {}
    for rel_name, reward in rewards.items():
        rewards_to_json[rel_name] = {
            "reward": reward,
            "advantage": advantages.get(rel_name, None)
        }
        for k in sub_reward_keys:
            rewards_to_json[rel_name][k] = sub_rewards[k].get(rel_name, None)

    if best_each_cond:
        cond_groups = defaultdict(list)
        for rel_name in rewards:
            cond_groups[cond_of_rel_name(rel_name)].append(rel_name)

        # Pick the best song for each condition, then pick the best song overall
        best_of_each_cond = []
        for songs in cond_groups.values():
            best_in_group = max(songs, key=lambda song: rewards[song])
            best_of_each_cond.append(best_in_group)

        best_songs = sorted(best_of_each_cond, key=rewards.get, reverse=True)[:pick_best_num]
    else:
        best_songs = sorted(rewards, key=rewards.get, reverse=True)[:pick_best_num]

    # 2. Save JSON
    best_song = best_songs[0] if best_songs else None
    best_rewards = rewards_to_json.get(best_song, {})
    with open(midi_root / ADVANTAGES_JSON, "w") as f:
        json.dump({
            "rewards": rewards,
            "advantages": advantages,
            "best": best_song,
            "best_rewards": best_rewards,
        }, f, indent=4)
    with open(midi_root / REWARDS_JSON, "w") as f:
        json.dump(rewards_to_json, f, indent=4)

    # 3. Copy Best MIDIs
    if best_songs:
        save_dir = midi_root / BEST_SONGS_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        for i, song in enumerate(best_songs):
            src = midi_root / f"{song}.mid"
            if src.exists():
                clean_name = song.replace('/', '_')
                dest = save_dir / f"{i}_{clean_name}.mid"
                shutil.copy(src, dest)

    report_eval_results(midi_root, sub_reward_keys, rewards_to_json)

def report_eval_results(
    midi_root: Path,
    sub_reward_keys: list[str],
    rewards_to_json: dict[str, dict[str, float]],
):
    rows = []
    for rel_name in sorted(rewards_to_json):
        cond = cond_of_rel_name(rel_name)
        row = {"cond": cond, "rel_name": rel_name, **rewards_to_json[rel_name]}
        rows.append(row)
    df_samples = pd.DataFrame(rows)

    tsv_path = midi_root / REWARDS_TSV
    df_samples.to_csv(tsv_path, sep='\t', index=False)

    means = {}
    stds = {}
    for k in sub_reward_keys:
        if f"_raw_{k}" in sub_reward_keys:
            continue
        raw_k = k.removeprefix("_raw_") # Report raw metrics
        means[raw_k] = df_samples[k].mean()
        stds[raw_k + "_std"] = df_samples[k].std()

    summary_df = pd.DataFrame([{"method": midi_root.name} | means | stds])

    # Align columns by name with the existing file: different runs may report
    # different metric sets (e.g. --baseline drops precision/recall)
    if OBJECTIVE_TSV.exists():
        existing_df = pd.read_csv(OBJECTIVE_TSV, sep='\t')
        summary_df = pd.concat([existing_df, summary_df], ignore_index=True)
    summary_df.to_csv(OBJECTIVE_TSV, sep='\t', index=False)

def main(midi_dir: str, baseline: bool=False, sample: bool=False):
    midi_root = Path(midi_dir).resolve()
    rewarder = CompositeReward(Clamp3Reward(persist_audio=True), RuleBasedReward(rules_for_eval=True))
    if baseline:
        # Baselines have no input harmony skeleton to compare against;
        # dissonance is measured relative to the analyzed skeleton instead
        rewarder.rule_based_reward.config.pop("precision", None)
        rewarder.rule_based_reward.config.pop("recall", None)

    rel_names = []
    for midi_path in tqdm(sorted(midi_root.rglob("*.mid"))):
        if midi_path.suffixes == NO_HARMO_SUFFIXES:
            continue
        if baseline:
            midi_obj = mido.MidiFile(midi_path)
            if len(midi_obj.tracks) < 2:
                continue
            try:
                preprocess_midi(midi_path, serialize=True)
            except Exception as e:
                print(f"Error processing {midi_path}: {e}")
                continue
        elif midi_path.parent.name == BEST_SONGS_DIR:
            continue
        rel_names.append(midi_path.relative_to(midi_root).with_suffix(""))

    if sample and len(rel_names) > 256:
        random.seed(42)
        rel_names = random.sample(rel_names, 256)
    print(f"Found {len(rel_names)} MIDI files supported for evaluation.")

    compute_save_advantages(rewarder, midi_root, rel_names)

if __name__ == "__main__":
    parseargs(main)
