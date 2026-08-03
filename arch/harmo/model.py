import json
import torch
from dataclasses import dataclass
from transformers import GPT2Config, GPT2LMHeadModel
from arch.base_class import ModelBase
from arch.harmo.vocab import *
from arch.harmo.config import *
from utils.distributed import *

@dataclass
class HarmonyGPTOutput:
    logits: torch.FloatTensor | None = None
    loss: torch.FloatTensor | None = None
    past_key_values: tuple | None = None
    hidden_states: tuple | None = None

class HarmonyGPT(ModelBase):
    def __init__(self, config: GPT2Config):
        super().__init__()

        self.config = config
        self.hidden_size = config.n_embd
        self.transformer = GPT2LMHeadModel(config)
        self.transformer.loss_type = "ForCausalLM"

    def forward(self, events: torch.LongTensor, return_outputs=False):
        attention_mask = (events != PAD_EVENT)
        outputs = self.transformer.forward(
            input_ids=events,
            attention_mask=attention_mask,
            labels=events,
        )
        if return_outputs:
            return outputs
        return {
            'loss': outputs.loss,
        }

    def dump_json_config(self, config_json: str):
        config_dict = {
            'hidden_size': self.hidden_size,
            'num_layers': self.config.n_layer,
            'num_heads': self.config.n_head,
            'vocab_size': self.config.vocab_size,
            'max_position_embeddings': self.config.n_positions,
            'pad_token_id': self.config.pad_token_id
        }
        with open(config_json, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def from_config_dict(cls, config: dict, device='auto'):
        model = cls.from_python_config(
            device=device,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            vocab_size=config['vocab_size'],
            max_position_embeddings=config['max_position_embeddings'],
            pad_token_id=config['pad_token_id']
        )
        return model

    @classmethod
    def from_python_config(
        cls,
        device='auto',
        hidden_size=HARMO_HIDDEN_SIZE,
        num_layers=HARMO_LAYERS,
        num_heads=HARMO_HEADS,
        vocab_size=harmo_vocab.size,
        max_position_embeddings=HARMO_EVENT_NUM,
        pad_token_id=PAD_EVENT
    ):
        if device == 'auto':
            device = get_device()
            print(f"Using device: {device}")
        config = GPT2Config(
            n_embd=hidden_size,
            n_layer=num_layers,
            n_head=num_heads,
            n_inner=hidden_size * 4,
            vocab_size=vocab_size,
            n_positions=max_position_embeddings,
            pad_token_id=pad_token_id,
        )
        model = cls(config)
        if device is not None:
            model.to(device)
        return model
