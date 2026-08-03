import torch
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor
from rl.lib.music_hubert import MusicHubertModel

class AudioBERTFeature(nn.Module):
    def __init__(
            self,
            pre_trained_folder,
            sample_rate,
            force_half=False,
            disable_backprop=True,
            processor_normalize=True,
        ):
        super().__init__()

    @torch.no_grad()
    def process_wav(self, waveform):
        # Return the same shape
        return self.processor(
            waveform,
            return_tensors="pt",
            sampling_rate=self.sample_rate,
            padding=True).input_values[0]

    def forward(self, input_values, layer=-1, reduction="mean"):
        if not self.force_half:
            out = self.model(input_values, output_hidden_states=True).hidden_states
        else:
            out = self.model(input_values.half(), output_hidden_states=True).hidden_states
            out = [o.float() for o in out]

        if layer != None:
            out = out[layer] # [B, T, H]
        else:
            out = torch.stack(out) # [L, B, T, H]
        if reduction == "mean":
            return out.mean(-2)
        elif reduction == "max":
            return out.max(-2)[0]
        elif reduction == "none":
            return out
        else:
            raise NotImplementedError

    def sliding_window_forward(self, input_values, window_size_in_sample, stride_in_sample, layer=-1, reduction="mean", allow_non_full_window=True):
        B, T = input_values.shape
        out = []
        for i in range(0, T-window_size_in_sample+1, stride_in_sample):
            out.append(self.forward(input_values[:, i:i+window_size_in_sample], layer=layer, reduction=reduction))
        if allow_non_full_window and T % stride_in_sample != 0:
            out.append(self.forward(input_values[:, -window_size_in_sample:], layer=layer, reduction=reduction))
        return torch.stack(out)

class HuBERTFeature(AudioBERTFeature):
    def __init__(
        self,
        pre_trained_folder,
        sample_rate,
        force_half=False,
        disable_backprop=True,
        processor_normalize=True,
    ):
        super().__init__(pre_trained_folder, sample_rate, force_half, disable_backprop, processor_normalize)
        self.sample_rate = sample_rate
        self.processor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=sample_rate,
            padding_value=0.0,
            return_attention_mask=True,
            do_normalize=processor_normalize,
        )

        print(f'Loading HuBERT model from {pre_trained_folder}')
        self.model = MusicHubertModel.from_pretrained(pre_trained_folder)

        self.force_half = force_half
        if disable_backprop:
            self.model.eval()
            if self.force_half:
                self.model.half()

            for param in self.model.parameters():
                param.requires_grad = False
