import os
import torch
from torch import nn
import contextlib
from tqdm import tqdm
import math
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from arch.config import *
from utils.save_load import load_or_create_ckpt, save_model
from utils.log_loss import check_unused_parameters, check_loss_nan, log_scalars, LossDictAverager
from utils.distributed import *

def cosine_with_decay_restarts_and_warmup(optimizer, steps_per_epoch, warmup_steps, min_lr_ratio=0.1):
    def lr_lambda(cur_step):
        epoch1 = cur_step // steps_per_epoch + 1
        # 1. Calculate the decay for the current "restart" cycle (the peak height)
        this_peak_decay = 1 / math.sqrt(epoch1)
        last_peak_decay = 1 / math.sqrt(max(1, epoch1 - 1))

        # 2. Identify position within the current epoch
        step_in_cycle = cur_step % steps_per_epoch

        if DEBUG_SCHEDULER:
            print(f"{cur_step=}, {epoch1=}, {step_in_cycle=}")

        # 3. Warmup Phase (Interpolating from last_peak_decay * min_lr_ratio to this_peak_decay)
        if step_in_cycle < warmup_steps:
            alpha = float(step_in_cycle) / float(max(1, warmup_steps))
            warmup_scale = last_peak_decay * min_lr_ratio + (this_peak_decay - last_peak_decay * min_lr_ratio) * alpha
            return warmup_scale

        # 4. Cosine Phase (Interpolating from this_peak_decay to this_peak_decay * min_lr_ratio)
        progress = float(step_in_cycle - warmup_steps) / float(
            max(1, steps_per_epoch - warmup_steps - 1)
        )
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        cosine_scale = this_peak_decay * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor)

        return cosine_scale

    return LambdaLR(optimizer, lr_lambda)

def debug_log_scheduler(optimizer: torch.optim.Optimizer, scheduler: LambdaLR):
    print(f"Current LR: {optimizer.param_groups[0]['lr']:.4e}")

def main(
    model: nn.Module, train_dataset, val_dataset,
    exp_typ="pretrain",
    ckpt_resume_path=None,

    batch_size = 16,
    grad_accu_steps = 2,
    workers = 4,

    learning_rate = 1e-4,
    warmup_steps_per_epoch = 4 if DEBUG_SCHEDULER else 1000,
    log_scalar_interval = 1 if DEBUG_SCHEDULER else 10,

    num_epochs = 20,
    train_collapsed_factor = 1.5,
    patience_limit = 2
):
    setup_distributed()
    device = get_device()
    model.to(device)

    if is_main_process():
        print(f"Distributed training: {is_distributed}, World size: {world_size}")
        print(f"Rank: {rank}, Local rank: {local_rank}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")

    # Data
    if DEBUG_SCHEDULER:
        train_dataset = Subset(train_dataset, range(batch_size * 8))
        val_dataset = Subset(val_dataset, range(batch_size))

    per_gpu_batch_size = max(1, batch_size // (grad_accu_steps * world_size))

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None

    train_loader = DataLoader(
        train_dataset, batch_size=per_gpu_batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler, num_workers=0 if DEBUG else workers,
        pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=per_gpu_batch_size,
        shuffle=False, sampler=val_sampler, num_workers=0 if DEBUG else max(1, workers // 2),
        pin_memory=True
    )

    # Optimizer
    steps_per_epoch = (len(train_loader) + grad_accu_steps - 1) // grad_accu_steps

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = cosine_with_decay_restarts_and_warmup(optimizer, steps_per_epoch, warmup_steps_per_epoch)
    ckpt_dir, start_epoch, global_step, best_val_loss, writer = load_or_create_ckpt(
        model, optimizer=optimizer, scheduler=scheduler, ckpt_path=ckpt_resume_path, typ=exp_typ
    )

    # Model
    if not NO_COMPILE_MODEL:
        if is_main_process():
            print("Compiling model...")
        model = torch.compile(model, fullgraph=True)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank])

    # Train
    patience_counter = 0
    stop = torch.zeros(1).to(device)
    stop_sign = os.path.join(ckpt_dir, 'stop.txt')

    for epoch in range(start_epoch, num_epochs):
        epoch1 = epoch + 1
        if is_main_process():
            print(f"\n{epoch1}-th Epoch, Total {num_epochs}")

        if is_distributed:
            train_sampler.set_epoch(epoch)

        old_global_step = global_step
        global_step = train_epoch(
            model, train_loader, optimizer, scheduler, device, writer, global_step,
            grad_accu_steps, log_scalar_interval
        )
        if is_main_process():
            if ckpt_dir.startswith(ASSET_DIR):
                ckpt_work_dir = WORK_DIR + ckpt_dir.removeprefix(ASSET_DIR)
            else:
                ckpt_work_dir = ckpt_dir
            save_model(
                model, epoch, global_step,
                ckpt_work_dir, f'epoch_{epoch1}',
                optimizer=optimizer, scheduler=scheduler,
            )
        if not DEBUG_PRETRAIN and (global_step - old_global_step) % steps_per_epoch != 0:
            raise RuntimeError(f"Corrupted: computed steps per epoch {steps_per_epoch} != actual {global_step - old_global_step}")

        val_loss = validate_epoch(model, val_loader, device)

        if is_main_process():
            print(f"Validation Loss: {val_loss:.4f}")
            with open(os.path.join(ckpt_dir, f"val_loss.log"), "a") as f:
                f.write(f"{epoch1=} {global_step=} {val_loss=}\n")
            if writer is not None:
                writer.add_scalar('Loss/val_total', val_loss, global_step)

            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(model, epoch, global_step, ckpt_dir, "best_model", best_val_loss=best_val_loss)
                patience_counter = 0
            elif val_loss < train_collapsed_factor * best_val_loss:
                patience_counter += 1
                print(f"Validation loss does not improve, patience: {patience_counter}/{patience_limit}")
            else:
                print("Validation loss skyrockets, training collapsed")
                stop += 1

            if patience_counter >= patience_limit:
                print(f"Early stopping triggered after {patience_counter} epochs without improvement")
                stop += 1
            if os.path.exists(stop_sign):
                print("User's stop sign detected")
                stop += 1

        if is_distributed:
            dist.all_reduce(stop, op=dist.ReduceOp.SUM)
        if stop > 0:
            if is_main_process() and os.path.exists(stop_sign):
                os.remove(stop_sign)
            break

    if is_main_process() and writer is not None:
        writer.close()

    cleanup_distributed()

def train_epoch(
    model,
    dataloader, optimizer, scheduler, device,
    writer, global_step,
    grad_accu_steps, log_scalar_interval
):
    model.train()
    optimizer.zero_grad()

    interval_loss = LossDictAverager()
    orig_model = model.module if isinstance(model, DDP) else model

    for i, batch in enumerate(tqdm(dataloader, desc="Training", mininterval=TQDM_MIN_INTERVAL, disable=not is_main_process())):
        batch = [batch.to(device, non_blocking=True)] if isinstance(batch, torch.Tensor) else [item.to(device, non_blocking=True) for item in batch]

        is_last_accumulation_step = (i + 1) % grad_accu_steps == 0 or (i + 1) == len(dataloader)
        # Use no_sync context to prevent premature gradient reduction
        context = model.no_sync() if is_distributed and not is_last_accumulation_step else contextlib.nullcontext()

        with context:
            with torch.amp.autocast(device.type, dtype=torch.bfloat16):
                outputs = model(*batch)
                loss = outputs['loss'] / grad_accu_steps
            check_loss_nan(loss)
            loss.backward()
        if DEBUG_PRETRAIN and is_main_process() and i == 0:
            check_unused_parameters(orig_model)

        interval_loss.add(outputs)
        if is_last_accumulation_step:
            global_step += 1

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if is_main_process() and global_step % log_scalar_interval == 0:
                # TODO: Average the scalars across rank before logging
                module_grad_norms = orig_model.get_module_grads()
                log_scalars(
                    interval_loss.average(), optimizer, grad_norm, writer, global_step,
                    module_grad_norms=module_grad_norms
                )
            if is_main_process() and DEBUG_SCHEDULER:
                debug_log_scheduler(optimizer, scheduler)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if DEBUG_PRETRAIN:
                break

    return global_step

def validate_epoch(model, dataloader, device):
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    num_batches = torch.tensor(0, device=device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", mininterval=TQDM_MIN_INTERVAL, disable=not is_main_process()):
            batch = [batch.to(device, non_blocking=True)] if isinstance(batch, torch.Tensor) else [item.to(device, non_blocking=True) for item in batch]

            with torch.amp.autocast(device.type, dtype=torch.bfloat16):
                outputs = model(*batch)
                loss = outputs['loss']
            check_loss_nan(loss)

            total_loss += loss
            num_batches += 1
            if DEBUG_PRETRAIN:
                break

    if is_distributed:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        total_loss /= world_size

    avg_loss = (total_loss / num_batches).item()
    return avg_loss
