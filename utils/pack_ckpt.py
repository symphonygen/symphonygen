"""Pack a raw state-dict checkpoint and its sidecar config.json into a single .pt file.

The packed file holds {"config": <dict from config.json>, "state_dict": <weights>}
and is loaded transparently by ModelBase.from_pretrained (arch/base_class.py).

Usage:
    python utils/pack_ckpt.py <ckpt_root>

For every <ckpt_root>/**/<name>.pt with a config.json next to it, writes
<ckpt_root>/<subdir_parts>_<name>.pt, e.g. grpo/clamp/epoch_10.pt -> grpo_clamp_epoch_10.pt.
"""
import json
import sys
from pathlib import Path
import torch

def pack_one(weight_path: Path, config_path: Path, out_path: Path):
    state_dict = torch.load(weight_path, map_location="cpu")
    if isinstance(state_dict, dict) and "config" in state_dict and "state_dict" in state_dict:
        print(f"Skipping {weight_path}: already packed")
        return
    with open(config_path) as f:
        config = json.load(f)
    torch.save({"config": config, "state_dict": state_dict}, out_path)
    print(f"Packed {weight_path} + {config_path} -> {out_path}")

def pack_tree(ckpt_root: Path):
    for weight_path in sorted(ckpt_root.rglob("*.pt")):
        if weight_path.parent == ckpt_root:
            continue
        config_path = weight_path.parent / "config.json"
        if not config_path.exists():
            print(f"Skipping {weight_path}: no config.json")
            continue
        rel = weight_path.relative_to(ckpt_root)
        out_path = ckpt_root / "_".join(rel.with_suffix("").parts)
        pack_one(weight_path, config_path, out_path.with_suffix(".pt"))

if __name__ == "__main__":
    pack_tree(Path(sys.argv[1]))
