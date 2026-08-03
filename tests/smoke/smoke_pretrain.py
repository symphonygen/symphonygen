"""
Tiny-scale pretraining smoke driver (see tests/smoke/run_smoke.sh).

Runs the real entry scripts (arch/harmo/1_pretrain.py, arch/symph/1_pretrain.py)
at smoke scale. Without --resume, DEBUG_PRETRAIN builds the debug-size model and
takes one optimizer step per epoch. With --resume, the full-size model is
resumed from a checkpoint (the README step-6 finetuning flow, incl. the packed
release checkpoints) and trained for real steps on the tiny smoke dataset.

Usage (repository root on PYTHONPATH, WORK_DIR pointing at the smoke work dir):
    python3 tests/smoke/smoke_pretrain.py harmo
    python3 tests/smoke/smoke_pretrain.py symph
    python3 tests/smoke/smoke_pretrain.py symph -n 1 -r <ckpt.pt>
"""
import argparse
import importlib
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_typ", choices=("harmo", "symph"))
    parser.add_argument("-r", "--resume", default=None,
                        help="checkpoint to resume from (README step 6); uses the full-size model")
    parser.add_argument("-n", "--num_epochs", type=int, default=2)
    args = parser.parse_args()

    # Patch the config BEFORE any other project import: downstream modules bind
    # these flags at import time (`from arch.config import *`).
    import arch.config as config
    if not args.resume:
        config.DEBUG_PRETRAIN = True  # debug-size model, one optimizer step per epoch
    config.NO_COMPILE_MODEL = True    # torch.compile takes minutes; irrelevant for smoke

    from arch import ddp_trainer
    real_main = ddp_trainer.main

    def tiny_main(model, train_dataset, val_dataset, **kwargs):
        kwargs.update(
            num_epochs=args.num_epochs,
            batch_size=4,
            workers=2,
            warmup_steps_per_epoch=2,
        )
        return real_main(model, train_dataset, val_dataset, **kwargs)

    ddp_trainer.main = tiny_main

    entry = importlib.import_module(
        "arch.harmo.1_pretrain" if args.model_typ == "harmo" else "arch.symph.1_pretrain")
    sys.argv = [sys.argv[0]] + ([args.resume] if args.resume else [])
    entry.main()

if __name__ == "__main__":
    main()
