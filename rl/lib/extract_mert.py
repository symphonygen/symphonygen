import torch
import torchaudio
from rl.lib.hf_pretrains import HuBERTFeature

target_sr = 24000
sliding_window_size_in_sec = 5
sliding_window_overlap_in_percent = 0.0
layer = None
reduction = 'mean'

def get_mert_features(feature_extractor: HuBERTFeature, audio_file, device):
    try:
        waveform = load_audio(audio_file, target_sr=target_sr, device=device)
    except Exception as e:
        print(f"Failed to load audio {audio_file}: {e}")
        return None
    wav = feature_extractor.process_wav(waveform)
    wav = wav.to(device)
    if sliding_window_size_in_sec:
        assert sliding_window_size_in_sec > 0, "sliding_window_size_in_sec must be positive"
        overlap_in_sec = sliding_window_size_in_sec * sliding_window_overlap_in_percent / 100
        wavs = []
        for i in range(0, wav.shape[-1], int(target_sr * (sliding_window_size_in_sec - overlap_in_sec))):
            wavs.append(wav[:, i: i + int(target_sr * sliding_window_size_in_sec)])
        if wavs[-1].shape[-1] < target_sr * 1:
            wavs = wavs[:-1]
        features = []
        for wav_chunk in wavs:
            features.append(feature_extractor(wav_chunk, layer=layer, reduction=reduction))
        features = torch.cat(features, dim=1)
    else:
        features = feature_extractor(wav, layer=layer, reduction=reduction)
    return features

def load_audio(
    file_path,
    target_sr,
    is_mono=True,
    is_normalize=False,
    device=torch.device('cpu')
):
    """Load audio file and convert to target sample rate.

    Args:
        file_path (str): path to audio file
        target_sr (int): target sample rate, if not equal to sample rate of audio file, resample to target_sr
        is_mono (bool, optional): convert to mono. Defaults to True.
        is_normalize (bool, optional): normalize to [-1, 1]. Defaults to False.
        device (torch.device, optional): device to use for resampling. Defaults to torch.device('cpu').

    Returns:
        torch.Tensor: waveform of shape (1, n_sample)
    """
    # TODO: Deal with target_depth
    waveform, sample_rate = torchaudio.load(file_path)
    if waveform.shape[0] > 1:
        if is_mono:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

    if is_normalize:
        waveform = waveform / waveform.abs().max()

    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = waveform.to(device)
        resampler = resampler.to(device)
        waveform = resampler(waveform)

    return waveform
