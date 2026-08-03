"""
Tiny-scale GRPO smoke driver (see tests/smoke/run_smoke.sh).

Runs the real entry (arch/symph/2_reinforce.py) with the built-in DEBUG_GRPO
sizes (prompt_size=2, group_size=2) for one epoch, covering the full loop:
skeleton sampling + filters -> symphony rollout -> MuseScore audio rendering ->
CLaMP 3 reward -> group advantages -> one GRPO training epoch -> final rollout.

Usage (repository root on PYTHONPATH, WORK_DIR pointing at the smoke work dir):
    python3 tests/smoke/smoke_grpo.py \
        $ASSET_DIR/stage_one_finetuned.pt $ASSET_DIR/stage_two_pretrained.pt
"""
import argparse
import importlib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("harmo_ckpt_path")
    parser.add_argument("ckpt_path")
    parser.add_argument("-n", "--num_epochs", type=int, default=1)
    parser.add_argument("-t", "--track_reward", type=int, default=1)
    args = parser.parse_args()

    # Patch the config BEFORE any other project import: downstream modules bind
    # these flags at import time (`from arch.config import *`).
    import arch.config as config
    config.DEBUG_GRPO = True        # prompt_size=2, group_size=2
    config.NO_COMPILE_MODEL = True  # compiling four full models is far too slow for smoke

    reinforce = importlib.import_module("arch.symph.2_reinforce")
    reinforce.main(
        args.harmo_ckpt_path, args.ckpt_path,
        num_epochs=args.num_epochs,
        track_reward=args.track_reward,
    )

if __name__ == "__main__":
    main()
