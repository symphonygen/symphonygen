import os
import torch
import torch.distributed as dist

is_distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
world_size = int(os.environ.get('WORLD_SIZE', 1))
rank = int(os.environ.get('RANK', 0))
local_rank = int(os.environ.get('LOCAL_RANK', 0))
is_multi_gpu = is_distributed and world_size > 1

def get_device():
    return torch.device(f'cuda:{local_rank}' if is_multi_gpu else 'cuda' if torch.cuda.is_available() else 'cpu')

def setup_distributed():
    if not is_distributed:
        print('Not using distributed mode')
        return
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl', rank=rank, world_size=world_size,
        device_id=get_device()
    )

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def barrier():
    if dist.is_initialized():
        dist.barrier()

def dist_gather_dict(this_rank_dict):
    if not dist.is_initialized():
        return this_rank_dict

    gathered_list = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_list, this_rank_dict)

    all_ranks_combined = {}
    for rank_dict in gathered_list:
        for key, value in rank_dict.items():
            if key in all_ranks_combined:
                raise KeyError(f"Duplicate key '{key}' found during gather")
            all_ranks_combined[key] = value

    return all_ranks_combined
