import os
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime
from arch.base_class import ModelBase
from arch.config import *
from utils.distributed import *
from utils.parseargs import parseargs

def save_model(
    model: ModelBase, epoch, global_step, ckpt_dir, save_name,
    optimizer=None, scheduler=None, best_val_loss=None,
):
    if NO_SAVE_LOAD_MODEL:
        return

    orig_model = model.module if isinstance(model, DDP) else model
    if hasattr(orig_model, "_orig_mod"): # Unwrap compiled model
        orig_model = orig_model._orig_mod

    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(
        orig_model.state_dict(),
        os.path.join(ckpt_dir, f"{save_name}.pt")
    )
    if optimizer is not None and scheduler is not None:
        torch.save({
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'global_step': global_step,
            'best_val_loss': best_val_loss,
        }, os.path.join(ckpt_dir, f'{save_name}_resume_info.pt'))
    print(f"Saved {save_name}.pt")

def load_or_create_ckpt(
    model: ModelBase,
    optimizer=None, scheduler=None,
    ckpt_path=None, typ="unsorted",
    device='cpu'
):
    if ckpt_path:
        ckpt_path = Path(ckpt_path)
        ckpt_dir = str(ckpt_path.parent)
        if is_main_process():
            print(f"--- Resuming from {ckpt_path} ---")

        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and "config" in ckpt and "state_dict" in ckpt:
            # Packed release checkpoint (see utils/pack_ckpt.py)
            ckpt = ckpt["state_dict"]
        model.load_state_dict(model.clean_ckpt_params(ckpt))

        resume_info_path = ckpt_path.parent / f'{ckpt_path.stem}_resume_info.pt'
        if resume_info_path.exists():
            resume_info = torch.load(resume_info_path, map_location=device)
            optimizer.load_state_dict(resume_info['optimizer_state_dict'])
            scheduler.load_state_dict(resume_info['scheduler_state_dict'])
            start_epoch = resume_info['epoch'] + 1
            global_step = resume_info['global_step']
            best_val_loss = resume_info.get('best_val_loss', None) or float('inf')
        else:
            if is_main_process():
                print(f"--- No resume info found at {resume_info_path} ---")
            start_epoch = 0
            global_step = 0
            best_val_loss = float('inf')
    else:
        if is_main_process():
            ckpt_dir = create_ckpt_dir(typ)
            model.dump_json_config(f"{ckpt_dir}/config.json")
        else:
            ckpt_dir = None
        if is_distributed:
            to_sync = [ckpt_dir]
            dist.broadcast_object_list(to_sync, src=0)
            ckpt_dir = to_sync[0]
            dist.barrier()

        start_epoch = 0
        global_step = 0
        best_val_loss = float('inf')

    if is_main_process() and not NO_WRITER:
        writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, 'logs'))
    else:
        writer = None
    return ckpt_dir, start_epoch, global_step, best_val_loss, writer

def create_ckpt_dir(typ: str):
    print(f"--- Starting {typ.upper()} ---")
    if NO_SAVE_LOAD_MODEL:
        timestamp = datetime.now().strftime("%Y%m%d")
        ckpt_dir = f"{typ}_ckpt/{timestamp}"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_dir = f"{WORK_DIR}/{typ}_ckpt/{timestamp}"
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Created checkpoint directory: {ckpt_dir}")
    return ckpt_dir

def rename_state_dict_keys(model_path: str, key: str, new_key: str):
    state_dict = torch.load(model_path)
    for k in list(state_dict.keys()):
        if k.startswith(key):
            state_dict[new_key + k[len(key):]] = state_dict.pop(k)
            print(f"Renamed key {k} to {new_key + k[len(key):]}")
    torch.save(state_dict, model_path)

if __name__ == "__main__":
    parseargs(rename_state_dict_keys)
