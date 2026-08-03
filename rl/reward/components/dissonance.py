import torch
import numpy as np
from arch.vocab import PITCH_NUM, QUARTER
from utils.distributed import get_device
from data_prep.data_structure import BarStruct, TrackStruct
from data_prep.main import bar_slice_notes_

class DissonanceMatrix:
    W_no_register_decay = None # Used for evaluation
    W_tensor_no_register_decay = None
    W_tensor = None # Used for dense orchestration

    @staticmethod
    def init(device='auto'):
        if DissonanceMatrix.W_tensor is not None:
            return
        if device == 'auto':
            device = get_device()

        base_weights = np.array([0.0, 1.0, 0.7, 0.3, 0.2, 0.4, 0.95, 0.1, 0.4, 0.3, 0.7, 0.9])
        p = np.arange(PITCH_NUM)
        diff = np.abs(p[:, np.newaxis] - p[np.newaxis, :])
        mean_pitch = (p[:, np.newaxis] + p[np.newaxis, :]) / 2.0
        W = base_weights[diff % 12]
        register_decay = np.exp(-mean_pitch / 24.0) + 0.5

        DissonanceMatrix.W_no_register_decay = W
        DissonanceMatrix.W_tensor_no_register_decay = torch.tensor(W, dtype=torch.float32, device=device)
        DissonanceMatrix.W_tensor = torch.tensor(
            W * register_decay,
            dtype=torch.float32, device=device,
        )

def eval_dissonance(bars: list[BarStruct], eps=1e-6):
    if not bars:
        return 0.0, 0.0

    DissonanceMatrix.init(device='cpu')
    HN_all = []
    NN_all = []

    for bar in bars:
        T = bar.duration_q
        if not bar.tracks:
            HN_all.append(np.zeros(T))
            NN_all.append(np.zeros(T))
            continue

        harmo = create_harmo_skeleton_mask(bar.harmony_track, T)
        M = create_music_tensor(bar)

        # Separate Harmonic (H) and Non-Harmonic (N)
        H = M * harmo
        N = M * (1 - harmo)

        # Normalize at each specific tick
        energy_H = np.linalg.norm(H, axis=1, keepdims=True)
        energy_N = np.linalg.norm(N, axis=1, keepdims=True)
        H_norm = np.where(energy_H > eps, H / (energy_H + eps), 0)
        N_norm = np.where(energy_N > eps, N / (energy_N + eps), 0)

        # Calculate dissonance: Sum(A_i * W_ij * B_j) at each time step
        HN_diss = np.einsum('ti,ij,tj->t', H_norm, DissonanceMatrix.W_no_register_decay, N_norm)
        NN_diss = np.einsum('ti,ij,tj->t', N_norm, DissonanceMatrix.W_no_register_decay, N_norm) / 2

        HN_all.append(HN_diss)
        NN_all.append(NN_diss)

    HN_all = np.concatenate(HN_all)
    NN_all = np.concatenate(NN_all)

    return float(np.mean(HN_all)), float(np.mean(NN_all))

def create_harmo_skeleton_mask(harmo_track: TrackStruct | None, bar_duration: int):
    harmo_skeleton_mask = np.zeros((bar_duration, PITCH_NUM), dtype=np.float32)
    if not harmo_track:
        return harmo_skeleton_mask

    for note in harmo_track.notes:
        start = note.pos
        end = min(note.end_pos, bar_duration)

        # Quantize to beat to fill every tick
        start = round(start / QUARTER) * QUARTER
        end = round(end / QUARTER) * QUARTER

        if start < bar_duration:
            harmo_skeleton_mask[start: end, note.pitch] = 1.0

    return harmo_skeleton_mask

def create_music_tensor(bar: BarStruct):
    if bar._sliced_notes is None:
        raise RuntimeError(f"Need to run {bar_slice_notes_.__name__} before harmony analysis")

    T = bar.duration_q
    M = np.zeros((T, PITCH_NUM))
    for track_notes in bar._sliced_notes.values():
        for note in track_notes:
            # Technically not needed to clamp the value, because notes are already sliced
            start, end = max(0, min(note.pos, T)), max(0, min(note.end_pos, T))
            if start < end:
                M[start: end, note.pitch] += 1.0
    return M
