from pathlib import Path
from rl.grpo.rollout import GRPODataset
from rl.grpo.ddp_trainer import GRPOTrainer
from rl.reward.clamp3_reward import Clamp3Reward
from rl.reward.composite_reward import CompositeReward
from rl.reward.rule_based_reward import RuleBasedReward
from arch.config import *
from arch.harmo.data_tensor import HarmonyTensorConverter
from arch.harmo.generator import HarmonyGenerator
from arch.harmo.model import HarmonyGPT
from arch.symph.cond_gen import SymphonyGenerator3D
from arch.symph.data_tensor import MusicTensorConverter3D
from arch.symph.rollout import SymphonyRollout3D
from arch.symph.model import MusicModel3D
from arch.symph.vocab import *
from utils.parseargs import parseargs

def freeze_unused_params(model: MusicModel3D):
    for layer_idx in range(HARMO_LAST_LAYER + 1, 0):
        model.harmo_event_decoder.transformer.h[layer_idx].requires_grad_(False)
    if HARMO_LAST_LAYER < -1:
        model.harmo_event_decoder.transformer.ln_f.requires_grad_(False)
    model.harmo_event_decoder.lm_head.requires_grad_(False)
    model.meta_predictor.bar_len_head.requires_grad_(False)

def main(
    harmo_ckpt_path: str, ckpt_path: str,
    ref_path: str=None,
    num_epochs: int=15,
    start_epoch: int=0,
    first_rollout_dir: str=None,
    track_reward: int=1, # Set to 0 to train with the pure CLaMP 3 reward
):
    harmo_converter = HarmonyTensorConverter()
    converter = MusicTensorConverter3D()

    harmo_config_name = "harmo_config" if Path(harmo_ckpt_path).parent == Path(ckpt_path).parent else "config"
    harmo = HarmonyGPT.from_pretrained(harmo_ckpt_path, harmo_config_name, device=None)
    model, old = MusicModel3D.init_copies(ckpt_path, num=2, device=None)
    ref = MusicModel3D.from_pretrained(ref_path or ckpt_path, typ="ref", device=None)
    freeze_unused_params(model)

    grpo_dataset = GRPODataset(converter)

    harmo_generator = HarmonyGenerator(harmo_converter)
    generator = SymphonyGenerator3D(converter)
    rollout = SymphonyRollout3D(harmo_generator, generator)

    rewarder = CompositeReward(
        # Rendered rollout audio goes to a temp dir; training is strict about
        # missing rewards only at the group level (see compute_save_advantages)
        Clamp3Reward(persist_audio=False), RuleBasedReward(),
        rule_weight=0.2 if track_reward else 0.0,
    )

    GRPOTrainer(
        harmo,
        model, old, ref,
        grpo_dataset,
        rollout, rewarder,
    ).run(
        num_epochs=num_epochs,
        start_epoch=start_epoch,
        first_rollout_dir=first_rollout_dir
    )

if __name__ == '__main__':
    parseargs(main)
