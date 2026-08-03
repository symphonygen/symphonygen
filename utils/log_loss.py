import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import torch.distributed as dist

def log_scalars(loss, optimizer, grad_norm, writer: SummaryWriter | None, global_step, module_grad_norms=None):
    if writer is None:
        return
    writer.add_scalar('Loss/loss', loss["loss"], global_step)
    for key, value in loss.get("metrics", {}).items():
        writer.add_scalar(f'Loss/Metrics/{key}', value, global_step)
    writer.add_scalar('Params/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
    writer.add_scalar('GradNorms/Total', grad_norm, global_step)
    if module_grad_norms is not None:
        for name, module_norm in module_grad_norms.items():
            writer.add_scalar(f'GradNorms/{name}', module_norm, global_step)

class LossDictAverager:
    def __init__(self):
        self.count = self.loss = 0
        self.metrics = defaultdict(float)

    def add(self, outputs):
        self.count += 1
        loss = outputs["loss"]
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        self.loss += loss
        for key, value in outputs.get("metrics", {}).items():
            if value is not None:
                if isinstance(value, torch.Tensor):
                    value = value.item()
                self.metrics[key] += value

    def average(self):
        loss = self.loss / self.count
        metrics = {key: value / self.count for key, value in self.metrics.items()}
        self.count = self.loss = 0
        self.metrics.clear()
        return {"loss": loss, "metrics": metrics}

def check_unused_parameters(model: nn.Module):
    unused = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            unused.append(name)

    if unused:
        print(f"Found {len(unused)} unused parameters:")
        for name in unused:
            print(f" - {model.__class__.__name__}.{name}")
        raise ValueError("Some parameters require gradients but are unused.")
    else:
        print("All parameters are being updated.")

def check_loss_nan(loss):
    is_nan = torch.isnan(loss).any().to(dtype=torch.float, device=loss.device)
    if dist.is_initialized():
        dist.all_reduce(is_nan, op=dist.ReduceOp.MAX)
    if is_nan.item() > 0:
        raise ValueError("Loss is NaN.")
