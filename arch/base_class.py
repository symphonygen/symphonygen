import json
import os
import torch
from torch import nn
from arch.config import DEBUG_GEN
from arch.data_tensor import DataTensorConverter
from data_prep.data_structure import BarStruct

class ModelBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @classmethod
    def from_pretrained(cls, model_path, config_name="config", typ="model", device='auto'):
        if model_path == "debug":
            model = cls.from_python_config(device=device)
            print("Debugging generation, model checkpoint is not loaded")
        else:
            print(f"Loading {typ} from {model_path}")
            ckpt = torch.load(model_path, map_location='cpu')
            if isinstance(ckpt, dict) and "config" in ckpt and "state_dict" in ckpt:
                # Packed checkpoint: config and weights in a single file
                model = cls.from_config_dict(ckpt["config"], device=device)
                ckpt = ckpt["state_dict"]
            else:
                config_path = f"{os.path.dirname(model_path)}/{config_name}.json"
                if os.path.exists(config_path):
                    model = cls.from_json_config(config_path, device=device)
                else:
                    model = cls.from_python_config(device=device)
            ckpt = model.clean_ckpt_params(ckpt)
            model.load_state_dict(ckpt)
        return model

    @classmethod
    def from_python_config(cls, device) -> "ModelBase":
        raise NotImplementedError

    @classmethod
    def from_config_dict(cls, config: dict, device) -> "ModelBase":
        raise NotImplementedError

    @classmethod
    def from_json_config(cls, config_json: str, device) -> "ModelBase":
        with open(config_json) as f:
            config: dict = json.load(f)
        return cls.from_config_dict(config, device=device)

    @classmethod
    def dump_json_config(cls, config_json: str):
        raise NotImplementedError

    def submodules(self) -> dict[str, nn.Module]:
        return {}

    def get_module_grads(self):
        module_grad_norms = {}
        for name, module in self.submodules().items():
            if module is None:
                continue
            grads = [p.grad.detach().data.abs() for p in module.parameters() if p.grad is not None]
            if grads:
                module_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in grads]), 2)
                module_grad_norms[name] = module_norm
            else:
                raise ValueError(f"No gradients found for {name}")
        return module_grad_norms

    @staticmethod
    def clean_ckpt_params(state_dict):
        return {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    @classmethod
    def init_copies(cls, ckpt_path: str="debug", config_name="config", num: int=1, device='auto'):
        if device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {device}")

        if ckpt_path == "debug":
            print("Debugging, model checkpoint is not loaded")
            models = [cls.from_python_config(device) for _ in range(num)]
        else:
            models = [cls.from_pretrained(ckpt_path, config_name=config_name, device=device) for _ in range(num)]

        return models

class SamplingConstrainer:
    def apply_(self, logits):
        pass

    def update_(self, this_event):
        pass

    def batch_select_indices(self, indices):
        pass

class Generator:
    def __init__(
        self,
        converter: DataTensorConverter,
        print_info=False,
        print_warning=False,
    ):
        self.converter = converter
        self.print_info = print_info
        self.print_warning = print_warning

    def run(self) -> dict[int, list[BarStruct]]:
        raise NotImplementedError

    def unpack_group(self, gen_data: torch.Tensor | tuple[torch.Tensor]):
        if not isinstance(gen_data, tuple):
            gen_data = (gen_data,)

        gen_bars_dict = {}
        for b, one_gen_data in enumerate(zip(*gen_data)):
            try:
                gen_bars: list[BarStruct] = self.converter.unpack_data_tensor(*one_gen_data)
            except Exception as e:
                if DEBUG_GEN:
                    raise
                print(f"Error unpacking song {b}: {e}")
                continue
            gen_bars_dict[b] = gen_bars

        return gen_bars_dict
