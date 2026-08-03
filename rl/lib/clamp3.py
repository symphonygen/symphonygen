""" Adapted from CLaMP 3 repo: https://github.com/sanderwood/clamp3 """

import os
import torch
from torch import nn
from transformers import BertConfig, BertModel, PreTrainedModel
from rl.config import CLAMP_MODEL_PATH
from rl.lib.extract_mert import get_mert_features
from rl.lib.hf_pretrains import HuBERTFeature

CLAMP3_HIDDEN_SIZE = 768
AUDIO_HIDDEN_SIZE = 768
AUDIO_NUM_LAYERS = 12
MAX_AUDIO_WINDOWS = 128 # One window is 5 sec ("sliding_window_size_in_sec" in extract_mert)

def clean_clamp3_weights():
    x = torch.load("weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth")
    del x["optimizer"]
    del x["lr_sched"]
    x["model"] = {
        k: v for k, v in x["model"].items() if not k.startswith('text_') and not k.startswith('symbolic_')
    }
    torch.save(x, "weights_clamp3.pth")

class CLaMP3Model(PreTrainedModel):
    def __init__(self, audio_config: BertConfig, hidden_size=CLAMP3_HIDDEN_SIZE):
        super().__init__(audio_config)
        self.audio_model = BertModel(audio_config)
        self.audio_proj = nn.Linear(audio_config.hidden_size, hidden_size)

    def avg_pooling(self, input_features: torch.Tensor, input_masks: torch.Tensor):
        input_masks = input_masks.unsqueeze(-1).to(self.device)
        input_features = input_features * input_masks
        avg_pool = input_features.sum(dim=1) / input_masks.sum(dim=1)
        return avg_pool

    def get_audio_features(self, inputs_embeds, mask, get_global=True):
        audio_features = self.audio_model(inputs_embeds=inputs_embeds, attention_mask=mask)['last_hidden_state']
        if get_global:
            audio_features = self.avg_pooling(audio_features, mask)
        audio_features = self.audio_proj(audio_features)
        return audio_features

class CLAMP3Encoder:
    def __init__(self, model_path: str=CLAMP_MODEL_PATH, device: str='auto'):
        self.model_path = model_path
        if device == 'auto':
            if torch.cuda.is_available():
                local_rank = int(os.environ.get('LOCAL_RANK', 0))
                # torch.distributed.is_initialized() handles the distributed check
                device = torch.device(f'cuda:{local_rank}' if torch.distributed.is_initialized() else 'cuda')
            else:
                device = torch.device('cpu')
        self.device = device
        self.load_model()

    def load_model(self):
        audio_config = BertConfig(
            vocab_size=1,
            hidden_size=AUDIO_HIDDEN_SIZE,
            num_hidden_layers=AUDIO_NUM_LAYERS,
            num_attention_heads=AUDIO_HIDDEN_SIZE // 64,
            intermediate_size=AUDIO_HIDDEN_SIZE * 4,
            max_position_embeddings=MAX_AUDIO_WINDOWS
        )
        self.model = CLaMP3Model(audio_config=audio_config).to(self.device).eval()
        self.hubert = HuBERTFeature("m-a-p/MERT-v1-95M", 24000, force_half=False, processor_normalize=True).to(self.device).eval()
        checkpoint = torch.load(self.model_path, map_location="cpu")
        self.model.load_state_dict(checkpoint['model'])

    def encode_audio(self, audio_path: str, get_global=True):
        with torch.no_grad():
            # MERT feature extraction: Every 5 seconds
            mert_feat = get_mert_features(self.hubert, audio_path, self.device)
            mert_feat = mert_feat.mean(dim=0) # Average across MERT Layers

            pad = torch.zeros((1, mert_feat.size(-1)), device=self.device)
            mert_feat = torch.cat((pad, mert_feat, pad), dim=0)

            # Get features from the CLaMP3 model
            features = self.run_short_audio(mert_feat, get_global=get_global)

            return features.cpu().detach().numpy()

    def run_short_audio(self, input_data: torch.Tensor, get_global=True):
        # Input_data shape: (SeqLen, Hidden)
        seq_len = input_data.size(0)

        if seq_len > MAX_AUDIO_WINDOWS:
            seq_len = MAX_AUDIO_WINDOWS
            input_data = input_data[:MAX_AUDIO_WINDOWS]
            print(f"Input data too long (max {MAX_AUDIO_WINDOWS} windows), clipped")

        input_data = input_data.unsqueeze(0)
        mask = torch.ones(1, seq_len, device=self.device)
        features: torch.Tensor = self.model.get_audio_features(
            inputs_embeds=input_data,
            mask=mask,
            get_global=get_global
        )
        if get_global:
            features = features.squeeze(0)

        # If get_global=True, returns (CLAMP3_HIDDEN_SIZE,)
        # If get_global=False, returns (SeqLen, CLAMP3_HIDDEN_SIZE)
        return features

    def run_long_audio(self, input_data, L=MAX_AUDIO_WINDOWS):
        segs: list[torch.Tensor] = [] # One segment is 128 windows, one window is 5 seconds
        for i in range(0, len(input_data), L):
            segs.append(input_data[i: i + L])
        segs[-1] = input_data[-L:] # A little overlap here

        last_hidden_list = []

        for seg in segs:
            pad = torch.zeros((MAX_AUDIO_WINDOWS - seg.size(0), AUDIO_HIDDEN_SIZE), device=self.device)
            seg = torch.cat((seg, pad), dim=0)
            mask = torch.tensor([1] * seg.size(0) + [0] * (L - seg.size(0)), device=self.device)
            last_hidden = self.model.get_audio_features(
                inputs_embeds=seg.unsqueeze(0),
                mask=mask.unsqueeze(0),
            )
            last_hidden_list.append(last_hidden)

        full_chunk_cnt = len(input_data) // L
        remain_chunk_len = len(input_data) % L
        if remain_chunk_len == 0:
            weights = torch.tensor([L] * full_chunk_cnt, device=self.device).view(-1, 1)
        else:
            weights = torch.tensor([L] * full_chunk_cnt + [remain_chunk_len], device=self.device).view(-1, 1)

        hidden = torch.concat(last_hidden_list, 0)
        hidden = hidden * weights
        feature = hidden.sum(dim=0) / weights.sum()
        return feature.squeeze(0)
