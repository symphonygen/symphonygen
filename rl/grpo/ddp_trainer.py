import contextlib
import json
from pathlib import Path
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ConstantLR
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from rl.config import (
    GROUP_PROMPT_SIZE, GROUP_SIZE, GRPO_LOSS_CONFIG,
    HARMO_FILTER_LOG_PROB_LOW, HARMO_FILTER_LOG_PROB_HIGH,
    REF_FEATURE_DIR, RULES_FOR_HARMO_FILTER,
)
from Results.evaluate import compute_save_advantages
from rl.grpo.rollout import GRPODataset
from rl.base_class import GRPORollout, Rewarder
from rl.grpo.loss import GRPO_loss_tuple
from arch.config import *
from utils.conventions import ADVANTAGES_JSON
from arch.base_class import ModelBase
from utils.distributed import *
from utils.log_loss import *
from utils.save_load import load_or_create_ckpt, save_model

class GRPOTrainer:
    def __init__(
        self,
        harmo_model: ModelBase,
        model: ModelBase, old_model: ModelBase, ref_model: ModelBase,
        grpo_dataset: GRPODataset,
        rollout: GRPORollout,
        rewarder: Rewarder,
        learning_rate=4e-5,
        prompt_size=2 if DEBUG_GRPO else GROUP_PROMPT_SIZE,
        group_size=2 if DEBUG_GRPO else GROUP_SIZE,
    ):
        self.harmo = harmo_model
        self.model, self.old, self.ref = model, old_model, ref_model
        self.grpo_dataset = grpo_dataset
        self.rollout = rollout
        self.rewarder = rewarder

        self.learning_rate = learning_rate
        self.group_size = group_size
        self.prompt_size = prompt_size

        setup_distributed()
        self.device = get_device()

        self.ckpt_dir, _, self.global_step, _, self.writer = load_or_create_ckpt(
            self.model, typ="grpo",
        )
        self.ckpt_root = Path(self.ckpt_dir)
        if is_main_process():
            self.harmo.dump_json_config(f"{self.ckpt_dir}/harmo_config.json")
            with open(self.ckpt_root / 'exp_config.json', 'w') as f:
                json.dump({
                    "sys_argv": sys.argv,
                    "learning_rate": learning_rate,
                    "grpo_config": {
                        "prompt_size": prompt_size,
                        "group_size": group_size,
                        **GRPO_LOSS_CONFIG,
                    },
                    "ref_feature_dir": REF_FEATURE_DIR,
                    "reward_config": self.rewarder.config,
                    "harmony_filter_rule": RULES_FOR_HARMO_FILTER,
                    "harmony_filter_log_prob": [HARMO_FILTER_LOG_PROB_LOW, HARMO_FILTER_LOG_PROB_HIGH],
                }, f, indent=4)
            if DEBUG_REWARD:
                print(f"{rewarder.config=}")

    def run(
        self,
        num_epochs=15,
        start_epoch=0,
        first_rollout_dir=None,
    ):
        self.setup_models()

        stop = torch.zeros(1).to(self.device)
        stop_sign = os.path.join(self.ckpt_dir, 'stop.txt')

        for epoch in range(start_epoch, num_epochs + 1):
            epoch1 = epoch + 1
            if is_main_process():
                print(f"\n{epoch1}-th Epoch, Total {num_epochs}")

            if is_distributed:
                self.old.module.load_state_dict(self.model.module.state_dict())
            else:
                self.old.load_state_dict(self.model.state_dict())

            self.run_rollout(
                epoch,
                use_rollout_dir=(first_rollout_dir if epoch == start_epoch else None)
            )
            if epoch == num_epochs:
                # After the last epoch, we perform a rollout to see
                # the final average rewards of the model
                break

            grad_accu_steps = max(1, self.prompt_size * self.group_size // world_size)

            sampler = DistributedSampler(self.grpo_dataset, shuffle=True) if is_distributed else None
            train_loader = DataLoader(
                self.grpo_dataset,
                batch_size=1,
                shuffle=sampler is None, sampler=sampler, num_workers=0 if DEBUG_GRPO else 4,
                pin_memory=True, drop_last=False
            )
            self.train_epoch(
                self.model, self.old, self.ref,
                train_loader, self.optimizer, self.scheduler,
                grad_accu_steps=grad_accu_steps,
            )
            torch.cuda.empty_cache()

            if is_main_process():
                save_model(self.model, epoch, self.global_step, self.ckpt_dir, f'epoch_{epoch1}')

            if os.path.exists(stop_sign):
                print("User's stop sign detected")
                stop += 1
            if is_distributed:
                dist.all_reduce(stop, op=dist.ReduceOp.SUM)
            if stop > 0:
                if is_main_process() and os.path.exists(stop_sign):
                    os.remove(stop_sign)
                break

        if is_main_process() and self.writer is not None:
            self.writer.close()

        cleanup_distributed()

    def run_rollout(self, epoch, use_rollout_dir=None):
        prompt_size = self.prompt_size
        group_size = self.group_size

        if use_rollout_dir and (Path(use_rollout_dir) / "tokens.json").exists():
            print(f"Use existing first rollout in {use_rollout_dir}")
            rollout_root = Path(use_rollout_dir)
        else:
            rollout_root = self.ckpt_root / "rollout" / f"epoch_{epoch}"
            self.rollout.run(
                self.harmo, self.old,
                rollout_root,
                prompt_size=prompt_size,
                group_size=group_size,
            )
            torch.cuda.empty_cache()

        if DEBUG_GRPO and (rollout_root / ADVANTAGES_JSON).exists():
            print(f"Debug: use existing advantages in {rollout_root}")
        else:
            rel_names = self.rollout.get_rel_names(prompt_size, group_size)
            metrics = compute_save_advantages(
                self.rewarder, rollout_root, rel_names,
                expected_group_size=group_size,
            )
            if is_main_process():
                log_rewards(metrics, self.writer, epoch)
            barrier()

        self.grpo_dataset.load_rollouts(rollout_root)

    def train_epoch(
        self,
        model: ModelBase, old: ModelBase, ref: ModelBase,
        dataloader, optimizer, scheduler,
        grad_accu_steps=1, log_scalar_interval=1,
    ):
        device = self.device
        optimizer.zero_grad()

        interval_loss = LossDictAverager()
        orig_model = model.module if isinstance(model, DDP) else model

        for i, (batch, advantages) in enumerate(tqdm(dataloader, desc=f"Training", mininterval=TQDM_MIN_INTERVAL, disable=not is_main_process())):
            batch = [batch.to(device, non_blocking=True)] if isinstance(batch, torch.Tensor) else [item.to(device, non_blocking=True) for item in batch]
            advantages = advantages.to(device)

            is_last_accumulation_step = ((i + 1) % grad_accu_steps == 0 or (i + 1) == len(dataloader))
            # Use no_sync context to prevent premature gradient reduction
            context = model.no_sync() if is_distributed and not is_last_accumulation_step else contextlib.nullcontext()

            with context:
                with torch.amp.autocast(device.type, dtype=torch.bfloat16):
                    with torch.no_grad():
                        _, _, ref_log_dist = self.rollout.compute_completion_log_probs(ref, batch)
                        old_log_probs, _, _ = self.rollout.compute_completion_log_probs(old, batch)
                    policy_log_probs, log_probs_mask, policy_log_dist = self.rollout.compute_completion_log_probs(model, batch)
                    outputs = GRPO_loss_tuple(
                        policy_log_probs, old_log_probs,
                        policy_log_dist, ref_log_dist,
                        log_probs_mask,
                        advantages,
                        **GRPO_LOSS_CONFIG
                    )
                    loss = outputs["loss"] / grad_accu_steps
                check_loss_nan(loss)
                loss.backward()
            if DEBUG_GRPO and is_main_process() and i == 0:
                check_unused_parameters(orig_model)

            interval_loss.add(outputs)

            if is_last_accumulation_step:
                self.global_step += 1

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                if is_main_process() and self.global_step % log_scalar_interval == 0:
                    # TODO: Average the scalars across rank before logging
                    module_grad_norms = orig_model.get_module_grads()
                    log_scalars(
                        interval_loss.average(), optimizer, grad_norm, self.writer, self.global_step,
                        module_grad_norms=module_grad_norms
                    )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

    def setup_models(self):
        self.all_models: list[ModelBase] = [self.harmo, self.model, self.old, self.ref]

        for m in self.all_models:
            m.to(self.device)
        for m in self.all_models:
            m.eval()

        if is_main_process():
            print(f"Distributed: {is_distributed}, World Size: {world_size}")
            print(f"Harmony Params: {sum(p.numel() for p in self.harmo.parameters()):,}")
            print(f"Symphony Params: {sum(p.numel() for p in self.model.parameters()):,}")

        if not NO_COMPILE_MODEL:
            if is_main_process():
                print("Compiling all models...")
            for i in range(len(self.all_models)):
                self.all_models[i] = torch.compile(self.all_models[i], fullgraph=True)

        if is_distributed:
            for i in range(len(self.all_models)):
                self.all_models[i] = DDP(self.all_models[i], device_ids=[local_rank])

        self.harmo, self.model, self.old, self.ref = self.all_models

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=sys.maxsize)

def log_rewards(metrics: dict, writer: SummaryWriter, epoch):
    if writer is not None:
        for key, value in metrics.items():
            writer.add_scalar(f"Reward/{key}", value, epoch)
        with open(os.path.join(writer.log_dir, "reward.log"), "a") as f:
            f.write(f"rollout {epoch} by {'pretrained' if epoch == 0 else f'epoch_{epoch}.pt'} model:\n")
            for key, value in metrics.items():
                f.write(f"  {key}: {value:.4f}\n")
    print(f"Mean Reward: {metrics['mean_reward']:.4f}")
