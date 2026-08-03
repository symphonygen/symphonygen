# SymphonyGen: 3D Hierarchical Orchestral Generation with Controllable Harmony Skeleton

[![arXiv](https://img.shields.io/badge/arXiv-2604.25498-b31b1b.svg)](https://arxiv.org/abs/2604.25498)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/SymphonyGen/SymphonyGen)
[![Demo Page](https://img.shields.io/badge/%F0%9F%8C%90%20Demo-symphonygen.github.io-green)](https://symphonygen.github.io)

This repository contains the official implementation of **SymphonyGen**, a 3D hierarchical framework for contemporary cinematic orchestration. SymphonyGen decomposes symphonic scores along the Bar, Track, and Event axes with a cascading decoder architecture, conditions generation on a beat-quantized multi-voice **harmony skeleton** ("short-score" conditioning), refines the model with **GRPO** using a cross-modal audio-perceptual reward (CLaMP 3), and suppresses tonal clashes at inference time with **dissonance-averse sampling**.

- [**🌐 Project Demo Page**](https://symphonygen.github.io) — audio examples for both listening-test rounds (composition and re-orchestration), along with the baseline excerpts (dataset, SymphonyNet, NotaGen, METEOR) used in the subjective tests.
- [**🤗 Released checkpoints**](https://huggingface.co/SymphonyGen/SymphonyGen) — all four released models (harmony skeleton model, pretrained 3D model, and both GRPO variants).
- [**📄 Paper (arXiv)**](https://arxiv.org/abs/2604.25498) — accepted at **ISMIR 2026**.

## Repository Layout

```
arch/           Core architecture and pretraining
  harmo/          Standalone 1D harmony skeleton model
  symph/          3D hierarchical symphony model (pretraining, GRPO entry, generation)
  model/          Shared Transformer modules (event decoder, cross-attention, ops)
rl/             GRPO training code, rewards and metrics
  grpo/           GRPO trainer, rollout, loss
  reward/         CLaMP 3 reward, rule-based rewards/metrics, harmony filters
  lib/            Third-party code (CLaMP 3 dependencies, adapted from sanderwood/clamp3)
data_prep/      Data preparation and preprocessing (MIDI parsing, harmony analysis)
Results/        Steps to reproduce the main results (batch generation, evaluation)
utils/          Shared utilities (distributed, multiprocessing, MIDI-to-audio, ...)
tests/          Regression tests for the post-training pipeline conventions
                  (see utils/conventions.py for the shared layout and key conventions)
  smoke/          Tiny-scale end-to-end smoke tests for every README stage
                  (tests/smoke/run_smoke.sh)
```

## Getting Started

### 1. Environment

```bash
pip install -r requirements.txt

# Run all commands from the repository root with the root on PYTHONPATH
export PYTHONPATH=$(pwd)

# (Optional) sanity-check the post-training pipeline conventions
python -m pytest tests/

# (Optional) after completing steps 2-4 below, smoke-test every README stage
# end-to-end at a tiny scale (a few dozen MIDI files, a few optimizer steps)
bash tests/smoke/run_smoke.sh
```

### 2. Directories and checkpoints

Two directories configure all paths (see `arch/config.py`); both can be overridden by environment variables:

- `ASSET_DIR` (default `asset/`): important assets such as checkpoints and reference audio
- `WORK_DIR` (default `data/`): intermediate or temporary data

Download the released checkpoints from [🤗 SymphonyGen/SymphonyGen](https://huggingface.co/SymphonyGen/SymphonyGen) and the remaining assets into the following layout:

```
$ASSET_DIR/
    clamp3_ref/your_ref_audio/                     # your reference audio set for the CLaMP 3 reward
                                                   #   (mp3/wav/...; encoded to .npy on first use)
    weights_clamp3.pth                             # download and rename the CLaMP 3 pretrained model
                                                   #   (from "CLaMP 3: Universal Music Information
                                                   #    Retrieval Across Unaligned Modalities and
                                                   #    Unseen Languages")
    MuseScore-3.6.2.548021370-x86_64.AppImage      # MuseScore 3.6.2 for MIDI-to-audio rendering
    stage_one_finetuned.pt                         # harmony skeleton model  (l_12_h_768)
    stage_two_pretrained.pt                        # pretrained 3D model    (l_33_h_512, 2-stream)
    grpo_clamp_epoch_10.pt                         # GRPO with the pure CLaMP 3 reward
    grpo_clamp+track_epoch_6.pt                    # GRPO with the CLaMP 3 + track density reward
$WORK_DIR/
    SymphonyNet_Dataset/                           # MIDI dataset (for training / dataset harmony)
    simplified_harmo_outline/                      # preprocessed data (created in step 4)
```

The released checkpoints are packed: each `.pt` file bundles the model config together with the weights (see `utils/pack_ckpt.py`). Checkpoints saved during training instead consist of a weights file plus a `config.json` dumped in the same directory; without the `config.json` the default Python config is used. `ModelBase.from_pretrained` loads both formats.

### 3. Headless MuseScore (for audio rendering)

Audio rendering (used by the CLaMP 3 reward and `--export_audio`) requires MuseScore. On a headless server, extract the AppImage and install its dependencies:

```bash
bash utils/headless_musescore.sh
```

### 4. Preprocess the data

Serializes the MIDI dataset, analyzes the harmony skeleton for each song, creates the train/val splits and the bar index maps:

```bash
bash data_prep/run_preprocess.sh
```

### 5. Pretrain

```bash
# Harmony skeleton model (1D)
torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) arch/harmo/1_pretrain.py

# Symphony model (3D)
torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) arch/symph/1_pretrain.py
```

Checkpoints and TensorBoard logs are written to `$WORK_DIR/{harmony,stage_two_3d}_ckpt/<timestamp>/`. Pass a checkpoint path as the first positional argument to resume training.

### 6. (Optional) Finetune on a subset

Finetuning is a resumed pretraining run on a curated subset: point the data paths (`arch/config.py`) at your subset, rebuild its bar index with `data_prep/3_index_bar.py`, and resume from the pretrained checkpoint:

```bash
torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) arch/symph/1_pretrain.py $ASSET_DIR/stage_two_pretrained.pt
```

### 7. Reinforcement learning (GRPO)

```bash
torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) arch/symph/2_reinforce.py \
    $ASSET_DIR/stage_one_finetuned.pt \
    $ASSET_DIR/stage_two_pretrained.pt
```

The reward is the CLaMP 3 audio-embedding similarity against the centroid of your reference set (`$ASSET_DIR/clamp3_ref/your_ref_audio/`, encoded on first use), combined by default with a track density reward (see `rl/config.py`). The CLaMP 3 audio encoder also downloads the MERT feature extractor (`m-a-p/MERT-v1-95M`) from the Hugging Face Hub on first use. Two model variants are released:

- `grpo_clamp_epoch_10.pt` ("reinforced"): pure CLaMP 3 reward — train with `--track_reward 0`
- `grpo_clamp+track_epoch_6.pt` ("reinforced+track"): composite CLaMP 3 + track density reward (default)

### 8. Generation

Generate symphonies conditioned on harmony-skeleton MIDI files (or analyze the harmony from any MIDI with `--analyze_harmo`):

```bash
python arch/symph/generator.py <model.pt> <cond_midi_path_or_dir> \
    [--analyze_harmo] [--group_size 2] [--bar_offset 0] [--save_dir .] \
    [--export_audio] [--forbid_piano] \
    [--disable_dissonance_averse] [--hn_weight 1.0] [--nn_weight 10.0] \
    [--register_decay 1] [--vis_dissonance]
```

- `--forbid_piano`: mask out the piano instrument family during generation
- `--hn_weight` / `--nn_weight`: dissonance-averse sampling weights (λ_hn, λ_nn); `--disable_dissonance_averse` turns the constraint off
- `--register_decay 0/1`: register-dependent decay of the dissonance matrix (Low Interval Limit); use `1` with the `reinforced+track` variant and `0` with `reinforced`
- `--vis_dissonance`: **visualize dissonance-averse sampling** — saves plots of the original vs. adjusted pitch logits (with active harmonic / non-harmonic tones highlighted) to `<save_dir>/dissonance_vis/`

Harmony skeletons can also be generated standalone:

```bash
python arch/harmo/generator.py $ASSET_DIR/stage_one_finetuned.pt --batch_size 4 --save_dir harmony_out
```

### 9. Reproduce the main results

Batch inference with harmony skeleton filters (density, repetition, log-probability, cadence, start-chord), followed by automatic evaluation:

```bash
python Results/batch_filter_gen.py <exp_dir> [--use_dataset_harmo] [--num_harmo 24] [--group_size 1] \
    [--skip_eval] [--zip_mp3]
```

- By default, harmony skeletons are sampled from the harmony model and filtered; `--use_dataset_harmo` instead analyzes and filters skeletons from the validation set (the second-round setting in the paper).
- Each model's outputs are evaluated right after generation (pass `--skip_eval` to generate only); `--zip_mp3` bundles the rendered audio per model.
- Model checkpoints, dissonance-averse settings, the start-chord filter, and best-song picking are configured in `Results/config.py` (including `FORBID_PIANO`, `START_CHORDS`, `PICK_BEST_NUM`).

Evaluate the objective metrics (Track Density, Harmony Precision/Recall, Dissonance, Melodic Movement/Ornament) on any folder of generated MIDI files:

```bash
python Results/evaluate.py <midi_dir> [--baseline] [--sample]
```

A summary row per method is appended to `Results/objective.tsv`.

> **Reproducibility disclaimer:** the objective metrics depend on the exact implementation of the harmony analysis and of the harmony-skeleton filters (see `DIFF.md` for where the released implementations differ from or refine the paper's description). As a result, you might see reproduced objective evaluation numbers that differ from the paper's tables. The trends, however, should be the same.

## Citation

The paper has been accepted at ISMIR 2026. If you find this work useful, please cite:

```bibtex
@inproceedings{symphonygen2026,
  title     = {SymphonyGen: 3D Hierarchical Orchestral Generation with Controllable Harmony Skeleton},
  author    = {He, Xuzheng and Nan, Nan and Wang, Zhilin and Kang, Ziyue and Mo, Zhuoru and Li, Ao and Pan, Yu and Li, Xiaobing and Yu, Feng and Guan, Xiaohong},
  booktitle = {Proceedings of the 27th International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2026}
}
```

