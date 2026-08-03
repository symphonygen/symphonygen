"""
Regression tests for the post-training pipeline's cross-module contracts
(the conventions in utils/conventions.py and the code that must agree on them).

Run from the repository root:  python -m pytest tests/
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from utils.conventions import (
    cond_dir_name, cond_of_rel_name, song_file_stem, song_key, song_rel_name, split_song_key,
)

def test_song_key_round_trip():
    for group_size in (1, 2, 32):
        for cond_id in (0, 3, 17):
            for song_id in range(group_size):
                key = song_key(cond_id, song_id, group_size)
                assert split_song_key(key, group_size) == (cond_id, song_id)

def test_cond_of_rel_name_matches_layout():
    rel_name = song_rel_name(3, 0)
    assert rel_name == f"{cond_dir_name(3)}/{song_file_stem(0)}"
    assert cond_of_rel_name(rel_name) == cond_dir_name(3)
    assert cond_of_rel_name(Path(rel_name)) == cond_dir_name(3)

def test_generator_chunking_covers_all_conditions(monkeypatch):
    """
    SymphonyGenerator.run chunks the conditions to fit MAX_BATCH_SIZE songs per
    GPU pass. Every (condition, song) pair must appear exactly once under the
    song-key convention, with the condition's own harmony attached — this is
    the invariant whose violation silently corrupted GRPO rollouts.
    """
    from arch.symph import generator as generator_mod
    from arch.symph.generator import SymphonyGenerator

    group_size = 4
    num_conds = 11
    # Force multiple chunks: 2 conditions (8 songs) per GPU pass
    monkeypatch.setattr(generator_mod, "MAX_BATCH_SIZE", 8)

    class StubGenerator(SymphonyGenerator):
        def __init__(self):
            pass  # No converter needed

        print_warning = False

        def prep_harmony_bars(self, cond, analyze_harmo, bar_offset):
            return cond, None

        def generate_given_harmony(self, model, all_harmo_list, group_size):
            # Batch is condition-major: each condition's songs are consecutive
            return [
                (cond, song_id)
                for cond in all_harmo_list
                for song_id in range(group_size)
            ]

        def unpack_group(self, song_data):
            return dict(enumerate(song_data))

        def postprocess(self, bars, harmo_bars):
            cond, song_id = bars
            assert cond == harmo_bars, (
                f"Song of condition {cond} was attributed to condition {harmo_bars}"
            )

    conditions = [[f"harmo_{c}"] for c in range(num_conds)]
    all_song_dict = StubGenerator().run(None, conditions, group_size=group_size)

    assert len(all_song_dict) == num_conds * group_size
    for key, (cond, song_id) in all_song_dict.items():
        cond_id, song_id_from_key = split_song_key(key, group_size)
        assert conditions[cond_id] == cond
        assert song_id_from_key == song_id

def test_rollout_rel_names_match_generator_keys():
    """ The rollout's directory layout must agree with the generator's keys. """
    group_size = 3
    for b1 in range(2):  # local condition index
        for b2 in range(group_size):
            key = song_key(b1, b2, group_size)
            assert split_song_key(key, group_size) == (b1, b2)
            rel_name = song_rel_name(b1, b2)
            assert cond_of_rel_name(rel_name) == cond_dir_name(b1)

def test_skeleton_filters_run_before_purify(monkeypatch):
    """
    Paper Sec. 4.2: skeletons that pass the filters are pruned into pure
    template chords — the filters must see the skeletons as sampled.
    """
    from arch.harmo import generator as gen_mod

    calls = []

    class StubGenerator(gen_mod.HarmonyGenerator):
        def __init__(self):
            self.print_info = False

        def run(self, model, **kwargs):
            return {0: ["bars0"], 1: ["bars1"]}

        def filter_by_log_prob(self, model, batch):
            calls.append("log_prob")
            return batch

    class StubAnalyzer:
        def __init__(self, **kwargs):
            pass

        def purify_harmony_(self, bars):
            calls.append("purify")

    monkeypatch.setattr(gen_mod, "filter_harmony_by_rule",
                        lambda batch: (calls.append("rule"), batch)[1])
    monkeypatch.setattr(gen_mod, "HarmonySkeletonAnalyzer", StubAnalyzer)

    harmo = StubGenerator().generate_until_fulfilled(None, 2, filter=True, purify=True)
    assert len(harmo) == 2
    assert calls.index("rule") < calls.index("log_prob") < calls.index("purify")

def test_resume_accepts_packed_checkpoint(tmp_path, monkeypatch):
    """
    README step 6: resuming/finetuning must accept the released packed
    checkpoints ({"config": ..., "state_dict": ...}, see utils/pack_ckpt.py),
    including state dicts saved from compiled models (`_orig_mod.` prefixes).
    """
    import torch
    from arch.base_class import ModelBase
    from utils import save_load

    class TinyModel(ModelBase):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

    model = TinyModel()
    packed = {
        "config": {"hidden": 2},
        "state_dict": {f"_orig_mod.{k}": v for k, v in model.state_dict().items()},
    }
    ckpt_path = tmp_path / "packed.pt"
    torch.save(packed, ckpt_path)

    monkeypatch.setattr(save_load, "NO_WRITER", True)
    fresh = TinyModel()
    ckpt_dir, start_epoch, global_step, _, writer = save_load.load_or_create_ckpt(
        fresh, ckpt_path=str(ckpt_path),
    )
    assert (ckpt_dir, start_epoch, global_step, writer) == (str(tmp_path), 0, 0, None)
    for k, v in model.state_dict().items():
        assert torch.equal(fresh.state_dict()[k], v)

def test_composite_reward_ignores_eval_metrics():
    """ Eval-only metrics (e.g. dissonance) must never leak into the reward. """
    from rl.reward.utils_ import weighted_tot_score

    # The weight scheme CompositeReward builds with rule_weight=0.2
    weights = {"clamp3": 1.0, "track": 0.2, "default": 0.0}
    rule_rewards = {
        "track": 0.5,
        "_raw_track": 12.0,       # raw metrics never scored
        "dissonance_HN": 0.9,     # eval metric: must not add reward
        "dissonance_NN": 0.4,
        "moving": 0.3,
        "precision": 0.99,
    }
    assert weighted_tot_score(rule_rewards, weights) == pytest.approx(0.2 * 0.5)

def test_advantage_grouping_and_group_size_guard(tmp_path, monkeypatch):
    from Results import evaluate

    # save_eval_results appends a summary row; keep it away from the real TSV
    monkeypatch.setattr(evaluate, "OBJECTIVE_TSV", tmp_path / "objective.tsv")

    class StubRewarder:
        def compute_reward_dict(self, root, rel_names, missing_ok=False):
            return {
                str(n): {"reward": float(i), "clamp3": float(i)}
                for i, n in enumerate(rel_names)
            }

    group_size = 4
    rel_names = [song_rel_name(c, s) for c in range(3) for s in range(group_size)]
    metrics = evaluate.compute_save_advantages(
        StubRewarder(), tmp_path, rel_names, expected_group_size=group_size,
    )
    assert metrics["mean_reward"] == pytest.approx(np.mean(range(len(rel_names))))
    # Advantages are normalized within each condition group
    import json
    advantages = json.loads((tmp_path / "advantages.json").read_text())["advantages"]
    for c in range(3):
        group = [advantages[song_rel_name(c, s)] for s in range(group_size)]
        assert np.mean(group) == pytest.approx(0.0, abs=1e-6)
        assert max(group) > 0 > min(group)

    # Oversized groups (mixed conditions) must raise
    with pytest.raises(ValueError):
        evaluate.compute_save_advantages(
            StubRewarder(), tmp_path, rel_names, expected_group_size=group_size - 1,
        )

def test_objective_tsv_appends_align_columns(tmp_path, monkeypatch):
    """ Rows with different metric sets must align by column name. """
    from Results import evaluate

    tsv = tmp_path / "objective.tsv"
    monkeypatch.setattr(evaluate, "OBJECTIVE_TSV", tsv)

    (tmp_path / "ours").mkdir()
    evaluate.report_eval_results(
        tmp_path / "ours", ["clamp3", "precision", "recall"],
        {"cond_harmony_0/song_0": {"reward": 1.0, "advantage": 0.0,
                                   "clamp3": 0.7, "precision": 0.9, "recall": 0.8}},
    )
    (tmp_path / "baseline").mkdir()
    evaluate.report_eval_results(
        tmp_path / "baseline", ["clamp3"],  # baseline: no precision/recall
        {"cond_harmony_0/song_0": {"reward": 1.0, "advantage": 0.0, "clamp3": 0.4}},
    )

    df = pd.read_csv(tsv, sep='\t')
    assert list(df["method"]) == ["ours", "baseline"]
    assert df.loc[df.method == "ours", "precision"].item() == pytest.approx(0.9)
    assert df.loc[df.method == "ours", "clamp3"].item() == pytest.approx(0.7)
    assert df.loc[df.method == "baseline", "clamp3"].item() == pytest.approx(0.4)
    assert pd.isna(df.loc[df.method == "baseline", "precision"].item())

def test_rule_based_eval_metrics_gated_by_config():
    """ Eval metrics must follow rules_for_eval, independent of any global flag. """
    from rl.reward.rule_based_reward import EVAL_METRIC_KEYS, RuleBasedReward

    training_config = RuleBasedReward(rules_for_eval=False).config
    assert not any(key in training_config for key in EVAL_METRIC_KEYS)

    eval_config = RuleBasedReward(rules_for_eval=True).config
    assert any(key in eval_config for key in EVAL_METRIC_KEYS)
    # The four metrics reported in the paper's Table 1
    for key in ("dissonance", "recall", "moving", "ornament"):
        assert key in eval_config

def test_clamp3_cache_is_incremental(tmp_path, monkeypatch):
    """ New songs must be computed even when a cache from a prior run exists. """
    import json
    from rl.reward.clamp3_reward import Clamp3Reward
    from utils.conventions import CLAMP3_CACHE_JSON

    rewarder = Clamp3Reward.__new__(Clamp3Reward)  # Skip reference loading
    rewarder.persist_audio = True
    computed = []

    def fake_compute(root, rel_names):
        computed.extend(map(str, rel_names))
        return {str(n): 0.5 for n in rel_names}

    monkeypatch.setattr(rewarder, "compute_rewards", fake_compute)

    (tmp_path / CLAMP3_CACHE_JSON).write_text(json.dumps({"cond_harmony_0/song_0": 0.9}))
    ret = rewarder.compute_reward_dict(
        tmp_path, ["cond_harmony_0/song_0", "cond_harmony_0/song_1"],
    )
    assert computed == ["cond_harmony_0/song_1"]  # Only the missing song
    assert ret == {"cond_harmony_0/song_0": 0.9, "cond_harmony_0/song_1": 0.5}
    # The merged cache is persisted
    merged = json.loads((tmp_path / CLAMP3_CACHE_JSON).read_text())
    assert merged == ret

    # Missing rewards: warn with missing_ok, raise without
    monkeypatch.setattr(rewarder, "compute_rewards", lambda root, rel_names: {})
    with pytest.raises(KeyError):
        rewarder.compute_reward_dict(tmp_path, ["cond_harmony_9/song_9"])
    ret = rewarder.compute_reward_dict(tmp_path, ["cond_harmony_9/song_9"], missing_ok=True)
    assert ret == {}
