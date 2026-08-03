#!/bin/bash
# ---------------------------------------------------------------------------
# SymphonyGen release smoke tests: run every README stage end-to-end at a tiny
# scale (a few dozen MIDI files, a few optimizer steps) to verify the released
# pipeline works on a GPU server.
#
# Usage (from anywhere; the script cd's to the repository root):
#     bash tests/smoke/run_smoke.sh [stage ...]
#
# Stages (default: all, in this order):
#     unit            python -m pytest tests/
#     data            README step 4 - tiny-dataset preprocessing (run_preprocess.sh)
#     pretrain_harmo  README step 5 - harmony model, debug size, 2 tiny epochs
#     pretrain_symph  README step 5 - symphony model, debug size, 2 tiny epochs
#     finetune        README step 6 - resume the full model from the released packed ckpt
#     gen             README step 8 - harmony + symphony generation CLIs (incl. audio export)
#     grpo            README step 7 - one tiny GRPO epoch (rollout / reward / training)
#     results         README step 9 - batch generation with skeleton filters + evaluation
#
# Requirements: README steps 1-3 completed ($ASSET_DIR checkpoints and MuseScore,
# dataset at $WORK_DIR/SymphonyNet_Dataset, dependencies installed).
#
# Environment:
#     WORK_DIR / ASSET_DIR  as in the README; the source dataset is read from
#                           $WORK_DIR/SymphonyNet_Dataset (override: SRC_DATASET)
#     SMOKE_WORK_DIR        scratch dir, default $WORK_DIR/smoke_work; the
#                           pipeline stages run with WORK_DIR pointed here
#     NUM_SMOKE_MIDIS       size of the tiny dataset (default 64)
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

REAL_WORK_DIR="${WORK_DIR:-data}"
export ASSET_DIR="${ASSET_DIR:-asset/}"
SRC_DATASET="${SRC_DATASET:-$REAL_WORK_DIR/SymphonyNet_Dataset}"
SMOKE_WORK_DIR="${SMOKE_WORK_DIR:-$REAL_WORK_DIR/smoke_work}"
NUM_SMOKE_MIDIS="${NUM_SMOKE_MIDIS:-64}"

# All pipeline stages below read and write the isolated smoke work dir
export WORK_DIR="$SMOKE_WORK_DIR"
mkdir -p "$SMOKE_WORK_DIR"

PY=python3
log() { echo; echo "===== [smoke] $* ====="; }

# Evaluation appends summary rows to the released Results/objective.tsv;
# keep it pristine no matter how the run ends.
OBJ_TSV="Results/objective.tsv"
OBJ_TSV_BAK="$SMOKE_WORK_DIR/objective.tsv.bak"
restore_objective_tsv() {
    if [ -f "$OBJ_TSV_BAK" ]; then
        mv "$OBJ_TSV_BAK" "$OBJ_TSV"
        echo "[smoke] Restored $OBJ_TSV"
    fi
}
trap restore_objective_tsv EXIT

stage_unit() {
    log "unit: regression tests (pytest tests/)"
    $PY -m pytest tests/ -q
}

stage_data() {
    log "data: tiny-dataset preprocessing (README step 4)"
    rm -rf "$SMOKE_WORK_DIR/SymphonyNet_Dataset" "$SMOKE_WORK_DIR/simplified_harmo_outline"
    mkdir -p "$SMOKE_WORK_DIR/SymphonyNet_Dataset"

    # Deterministic spread over the sorted dataset listing
    ALL_LIST="$SMOKE_WORK_DIR/all_midis.txt"
    find "$SRC_DATASET" -type f \( -iname '*.mid' -o -iname '*.midi' \) | sort > "$ALL_LIST"
    TOTAL=$(wc -l < "$ALL_LIST")
    if [ "$TOTAL" -lt 8 ]; then echo "Source dataset too small: $TOTAL files"; exit 1; fi
    STRIDE=$(( TOTAL / NUM_SMOKE_MIDIS ))
    if [ "$STRIDE" -lt 1 ]; then STRIDE=1; fi
    i=0
    while IFS= read -r f; do
        cp "$f" "$SMOKE_WORK_DIR/SymphonyNet_Dataset/${i}_$(basename "$f")"
        i=$((i + 1))
    done < <(awk -v s="$STRIDE" -v n="$NUM_SMOKE_MIDIS" 'NR % s == 1 && cnt < n { print; cnt++ }' "$ALL_LIST")
    echo "Copied $i MIDI files into the tiny dataset"

    NUM_CHUNKS=4 bash data_prep/run_preprocess.sh

    # Sanity: splits exist, are non-empty, and agree with their bar indices
    $PY - <<'EOF'
import pickle
from arch.config import TRAIN_DATA_PATH, VAL_DATA_PATH, bar_index_path
for path in (TRAIN_DATA_PATH, VAL_DATA_PATH):
    num_songs = sum(1 for line in open(path) if line.strip())
    with open(bar_index_path(path), 'rb') as f:
        index = pickle.load(f)
    assert num_songs > 0, f"{path} is empty"
    assert len(index) == num_songs, (path, num_songs, len(index))
    print(f"{path}: {num_songs} songs, bar index OK")
EOF
}

stage_pretrain_harmo() {
    log "pretrain_harmo: 2 debug-size epochs (README step 5)"
    $PY tests/smoke/smoke_pretrain.py harmo
    ls "$SMOKE_WORK_DIR"/harmony_ckpt/*/epoch_2.pt > /dev/null
}

stage_pretrain_symph() {
    log "pretrain_symph: 2 debug-size epochs (README step 5)"
    $PY tests/smoke/smoke_pretrain.py symph
    ls "$SMOKE_WORK_DIR"/stage_two_3d_ckpt/*/epoch_2.pt > /dev/null
}

stage_finetune() {
    log "finetune: resume the full model from the released packed checkpoint (README step 6)"
    # Resume from a copy so checkpoints/logs of the run stay inside the smoke dir
    rm -rf "$SMOKE_WORK_DIR/finetune"
    mkdir -p "$SMOKE_WORK_DIR/finetune"
    cp "$ASSET_DIR/stage_two_pretrained.pt" "$SMOKE_WORK_DIR/finetune/"
    $PY tests/smoke/smoke_pretrain.py symph -n 1 -r "$SMOKE_WORK_DIR/finetune/stage_two_pretrained.pt"
    ls "$SMOKE_WORK_DIR"/finetune/epoch_1.pt > /dev/null
}

stage_gen() {
    log "gen: harmony + symphony generation CLIs (README step 8)"
    GEN_DIR="$SMOKE_WORK_DIR/gen"
    rm -rf "$GEN_DIR"
    mkdir -p "$GEN_DIR"

    # 8a. Standalone harmony skeleton generation
    $PY arch/harmo/generator.py "$ASSET_DIR/stage_one_finetuned.pt" \
        --num_batches 1 --batch_size 2 --save_dir "$GEN_DIR/harmony_out"
    ls "$GEN_DIR"/harmony_out/harmony_*.mid > /dev/null

    # 8b. Symphony generation conditioned on the generated skeletons, with
    #     dissonance-averse sampling visualization and audio export
    $PY arch/symph/generator.py "$ASSET_DIR/grpo_clamp+track_epoch_6.pt" "$GEN_DIR/harmony_out" \
        --group_size 1 --save_dir "$GEN_DIR/songs" \
        --forbid_piano --vis_dissonance --export_audio
    ls "$GEN_DIR"/songs/cond_*/song_0.mid > /dev/null
    ls "$GEN_DIR"/songs/cond_*/mp3/*.mp3 > /dev/null

    # 8c. Re-orchestration: analyze the skeleton from a dataset MIDI
    #     ("reinforced" variant: pure CLaMP reward, no register decay).
    #     A few dataset MIDIs may fail preprocessing; try a couple.
    ok=0
    for src in "$SMOKE_WORK_DIR"/SymphonyNet_Dataset/*; do
        rm -rf "$GEN_DIR/cond_src" "$GEN_DIR/reorch"
        mkdir -p "$GEN_DIR/cond_src"
        cp "$src" "$GEN_DIR/cond_src/reorch_source.mid"
        if $PY arch/symph/generator.py "$ASSET_DIR/grpo_clamp_epoch_10.pt" "$GEN_DIR/cond_src" \
            --analyze_harmo --group_size 1 --register_decay 0 --save_dir "$GEN_DIR/reorch"
        then ok=1; break; fi
        echo "[smoke] Source $src failed, trying the next one"
    done
    [ "$ok" = 1 ]
    ls "$GEN_DIR"/reorch/cond_*/song_0.mid > /dev/null
}

stage_grpo() {
    log "grpo: one tiny GRPO epoch (README step 7)"
    $PY tests/smoke/smoke_grpo.py \
        "$ASSET_DIR/stage_one_finetuned.pt" "$ASSET_DIR/stage_two_pretrained.pt"
    GRPO_DIR=$(ls -td "$SMOKE_WORK_DIR"/grpo_ckpt/*/ | head -1)
    ls "$GRPO_DIR"/epoch_1.pt \
       "$GRPO_DIR"/rollout/epoch_0/advantages.json \
       "$GRPO_DIR"/rollout/epoch_1/advantages.json > /dev/null
}

stage_results() {
    log "results: batch generation with skeleton filters + evaluation (README step 9)"
    if [ -f "$OBJ_TSV" ] && [ ! -f "$OBJ_TSV_BAK" ]; then cp "$OBJ_TSV" "$OBJ_TSV_BAK"; fi

    # Round-1 setting: model-generated skeletons + start-chord curation (generation only)
    rm -rf "$SMOKE_WORK_DIR/exp_gen"
    $PY Results/batch_filter_gen.py "$SMOKE_WORK_DIR/exp_gen" \
        --num_harmo 2 --group_size 1 --skip_eval
    ls "$SMOKE_WORK_DIR"/exp_gen/harmony/harmony_*.mid > /dev/null
    ls "$SMOKE_WORK_DIR"/exp_gen/*/cond_harmony_0/song_0.mid > /dev/null

    # Round-2 setting: analyzed + pruned dataset skeletons, full evaluation, audio zip
    rm -rf "$SMOKE_WORK_DIR/exp_reorch"
    $PY Results/batch_filter_gen.py "$SMOKE_WORK_DIR/exp_reorch" \
        --num_harmo 2 --group_size 1 --use_dataset_harmo --zip_mp3
    ls "$SMOKE_WORK_DIR"/exp_reorch/*/rewards.tsv > /dev/null
    ls "$SMOKE_WORK_DIR"/exp_reorch/*.zip > /dev/null

    # Standalone objective evaluation of a baseline folder (dataset excerpts)
    rm -rf "$SMOKE_WORK_DIR/baseline_midis"
    mkdir -p "$SMOKE_WORK_DIR/baseline_midis"
    find "$SMOKE_WORK_DIR/SymphonyNet_Dataset" -type f | head -5 | while IFS= read -r f; do
        cp "$f" "$SMOKE_WORK_DIR/baseline_midis/"
    done
    $PY Results/evaluate.py "$SMOKE_WORK_DIR/baseline_midis" --baseline
    ls "$SMOKE_WORK_DIR"/baseline_midis/rewards.tsv > /dev/null
}

STAGES=("$@")
if [ ${#STAGES[@]} -eq 0 ]; then
    STAGES=(unit data pretrain_harmo pretrain_symph finetune gen grpo results)
fi
for s in "${STAGES[@]}"; do
    "stage_$s"
done
log "ALL SMOKE STAGES PASSED: ${STAGES[*]}"
