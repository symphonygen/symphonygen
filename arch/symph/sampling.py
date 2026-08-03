import torch
import torch.nn.functional as F
import numpy as np
from rl.reward.components.dissonance import DissonanceMatrix
from arch.config import DEBUG_DISS_CONSTRAINER
from arch.base_class import SamplingConstrainer
from arch.vocab import *

EPS = 1e-5
INF = 1e9

def sample_from_logits(logits, temperature=None, top_p=None, top_k=None):
    """
    logits: Shape (batch, vocab_size)
    temperature: Controls randomness (lower = more deterministic)
    top_k: Keep only top k tokens with highest probability
    top_p: Keep the top tokens whose cumulative probability >= top_p
    """
    if temperature is not None:
        logits = logits / max(temperature, EPS)

    if top_k is not None:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')

    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to keep the first token that exceeds top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter mask back to original logits shape
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')

    # Sample from the filtered distribution
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token.squeeze(-1)

class InstRangeConstrainer(SamplingConstrainer):
    def __init__(self, inst: torch.LongTensor):
        device = inst.device
        pitch_values = torch.arange(PITCH_NUM, device=device).unsqueeze(0)
        inst_range = INST_RANGE_STATS.to(device)[inst].unsqueeze(-1)
        lower_range = inst_range[:, 0]
        upper_range = inst_range[:, 1]

        # [B, 1] vs [1, PITCH_NUM]
        self.pitch_mask = (pitch_values < lower_range) | (upper_range < pitch_values)

    def apply_(self, logits: torch.Tensor):
        logits[:, PITCH_SLICE].masked_fill_(self.pitch_mask, -INF)

    def batch_select_indices(self, indices: torch.LongTensor):
        self.pitch_mask = self.pitch_mask[indices]

class PosConstrainer(SamplingConstrainer):
    def __init__(self, pos_limit: torch.LongTensor):
        device = pos_limit.device

        pos_values = torch.arange(POS_NUM, device=device)
        self.legato_values = torch.arange(1, DUR_NUM + 1, device=device)

        self.cur_pos = torch.zeros(len(pos_limit), dtype=torch.long, device=device)
        self.pos_limit = pos_limit

        # [B, 1] vs [1, POS_NUM]
        self.pos_mask = (pos_values.unsqueeze(0) >= pos_limit.unsqueeze(-1))

    def apply_(self, logits: torch.Tensor):
        logits[:, POS_SLICE].masked_fill_(self.pos_mask, -INF)

        # [B, 1] vs [1, DUR_NUM]
        legato_mask = (self.legato_values.unsqueeze(0) >= (self.pos_limit - self.cur_pos).unsqueeze(-1))
        logits[:, LEGATO_SLICE].masked_fill_(legato_mask, -INF)

    def update_(self, this_event: torch.Tensor):
        is_pos = is_vocab_pos(this_event)
        if is_pos.any():
            self.cur_pos[is_pos] = unpack_pos(this_event[is_pos])

        is_legato = is_vocab_legato(this_event)
        if is_legato.any():
            # LEGATO increments position by duration
            self.cur_pos[is_legato] += unpack_legato(this_event[is_legato])

    def batch_select_indices(self, indices: torch.LongTensor):
        self.cur_pos = self.cur_pos[indices]
        self.pos_limit = self.pos_limit[indices]
        self.pos_mask = self.pos_mask[indices]

class DissonanceConstrainer(SamplingConstrainer):
    """
    NOTE: Assumes the beat duration is divisible by quarter note.
    Still some dissonances left, such as:
    two notes that are harmonic now but non-harmonic next, and duration is long.
    """
    # Not enabled during training
    enabled = DEBUG_DISS_CONSTRAINER

    HN_weight = 1.0
    NN_weight = 10.0

    # Apply the register-dependent decay to W (Low Interval Limit);
    # activated for the model variant trained with the track density reward
    register_decay = True

    # Visualize how the logits are shifted by the dissonance penalty
    # (see vis_diss_logit_shift); set by entry points
    visualize = DEBUG_DISS_CONSTRAINER
    vis_dir = "."
    vis_max_plots = 16

    def __init__(
        self,
        active_idx: torch.LongTensor,
        harmo_beat_mask: torch.BoolTensor,
        active_pitch_buffer: torch.LongTensor,
    ):
        if not self.enabled:
            return

        device = active_idx.device
        B = len(active_idx)

        self.active_idx = active_idx
        self.harmo_beat_mask = harmo_beat_mask[active_idx] # [B, Beats, Pitches]
        self.active_pitch_buffer = active_pitch_buffer # Updates in-place, [ParentB, Pos, Pitches]

        self.cur_pos = torch.zeros(B, dtype=torch.long, device=device)
        self.cur_beat = torch.zeros(B, dtype=torch.long, device=device)

        self.cur_harmo = self.harmo_beat_mask[:, 0] # [B, Pitches]

        self.pending_pitch = torch.zeros(B, PITCH_NUM, dtype=torch.bool, device=device)
        self.pos_values = torch.arange(DUR_NUM, device=device)

        DissonanceMatrix.init(device)
        self.W = DissonanceMatrix.W_tensor if self.register_decay else DissonanceMatrix.W_tensor_no_register_decay

    def apply_(self, logits: torch.Tensor):
        """
        H_penalty: If candidate is harmonic, it only clashes with non-harmonic notes;
        N_penalty: If candidate is non-harmonic, it clashes with both harmonic and non-harmonic notes.
        """
        if not self.enabled:
            return

        harmo = self.cur_harmo

        # Include pending pitches that haven't received a duration token yet
        pending_H = (self.pending_pitch & harmo).float()
        pending_N = (self.pending_pitch & ~harmo).float()

        M = self.active_pitch_buffer[self.active_idx, self.cur_pos]
        H = M * harmo.float() + pending_H
        N = M * (1 - harmo.float()) + pending_N

        H_penalty = torch.matmul(self.HN_weight * N, self.W) # [B, Pitches]
        N_penalty = torch.matmul(self.HN_weight * H + self.NN_weight * N, self.W) # [B, Pitches]

        # Route the penalty based on whether the candidate pitch is S or N
        penalty = torch.where(harmo, H_penalty, N_penalty)

        orig = logits[:, PITCH_SLICE]
        if self.visualize:
            orig = orig.clone()
        adjusted = add_renormalize(orig, -penalty)
        logits[:, PITCH_SLICE] = adjusted

        # Only visualize the interesting steps with at least two active non-harmonic tones
        if self.visualize and self.vis_max_plots > 0 and len(torch.where(N[0])[0]) >= 2:
            DissonanceConstrainer.vis_max_plots -= 1
            self.vis_diss_logit_shift(orig, adjusted, H, N)

    def update_(self, this_event: torch.Tensor):
        if not self.enabled:
            return

        is_pitch = is_vocab_pitch(this_event)
        if is_pitch.any():
            self.pending_pitch[is_pitch, unpack_pitch(this_event[is_pitch])] = True

        is_dur = is_vocab_dur(this_event)
        is_legato = is_vocab_legato(this_event)
        is_pos = is_vocab_pos(this_event)

        dur_completed = is_dur | is_legato
        if dur_completed.any():
            dur = torch.zeros_like(this_event)
            dur[is_dur] = unpack_dur(this_event[is_dur])
            dur[is_legato] = unpack_legato(this_event[is_legato])

            # Note starts at cur_pos and lasts for 'dur'
            note_dur_mask = (
                (self.cur_pos[dur_completed].unsqueeze(-1) <= self.pos_values.unsqueeze(0)) &
                (self.pos_values.unsqueeze(0) < (self.cur_pos + dur)[dur_completed].unsqueeze(-1))
            )

            # note_dur_mask: [B, Pos], pending_pitch: [B, Pitch], cur_harmo: [B, Pitch]
            note_mask = note_dur_mask.unsqueeze(-1) & self.pending_pitch[dur_completed].unsqueeze(1)

            # Update energy
            self.active_pitch_buffer[self.active_idx[dur_completed]] += note_mask.float()

            if DEBUG_DISS_CONSTRAINER:
                print(f"Draw pitch {self.pending_pitch[0].cpu().numpy().nonzero()[0]} on {self.cur_pos[0].item()} to {(self.cur_pos[0] + dur[0]).item() - 1}")

            self.pending_pitch[dur_completed] = False
            self.cur_pos[is_legato] += dur[is_legato]

        if is_pos.any():
            self.cur_pos[is_pos] = unpack_pos(this_event[is_pos])

        pos_changed = is_pos | is_legato
        if pos_changed.any():
            self.cur_beat = self.cur_pos // QUARTER
            batch_idx = torch.arange(len(this_event), device=this_event.device)
            self.cur_harmo = self.harmo_beat_mask[batch_idx, self.cur_beat]

    def batch_select_indices(self, indices: torch.LongTensor):
        if not self.enabled:
            return
        self.active_idx = self.active_idx[indices]
        self.harmo_beat_mask = self.harmo_beat_mask[indices]
        self.cur_pos = self.cur_pos[indices]
        self.cur_beat = self.cur_beat[indices]
        self.cur_harmo = self.cur_harmo[indices]
        self.pending_pitch = self.pending_pitch[indices]

    def vis_diss_logit_shift(self, orig, adjusted, H, N):
        import os
        from matplotlib import pyplot as plt

        o = orig[0].detach().cpu().numpy()
        a = adjusted[0].detach().cpu().numpy()

        # Mask out large negative values (masked logits) for better visibility
        o[o <= -1e4] = np.nan
        a[a <= -1e4] = np.nan
        indices = np.arange(len(o))

        plt.figure(figsize=(10, 5))

        # 1. Highlight the background for active sets
        # Get integer indices for active masks
        H_indices = H[0].cpu().numpy().nonzero()[0]
        N_indices = N[0].cpu().numpy().nonzero()[0]
        other_harmo_indices = (self.cur_harmo & (H == 0))[0].cpu().numpy().nonzero()[0]

        # 1. Active Harmonic (Strong Green)
        for idx in H_indices:
            plt.axvspan(idx - 0.5, idx + 0.5, color='green', alpha=0.2,
                        label='Active Harmonic' if idx == H_indices[0] else "")

        # 2. Planned/Other Harmonic (Light Blue/Cyan)
        # This shows notes that belong to the chord but aren't currently "active"
        for idx in other_harmo_indices:
            plt.axvspan(idx - 0.5, idx + 0.5, color='cyan', alpha=0.15,
                        label='Harmony Condition' if idx == other_harmo_indices[0] else "")

        # 3. Non-Harmonic / Penalized (Red)
        for idx in N_indices:
            plt.axvspan(idx - 0.5, idx + 0.5, color='red', alpha=0.1,
                        label='Active Non-Harmonic' if idx == N_indices[0] else "")

        # 2. Plot the Logits
        plt.step(indices, o, label='Original Logits', where='mid', alpha=0.9, color='blue', linewidth=1.5)
        plt.step(indices, a, label='Adjusted Logits', where='mid', linestyle='--', alpha=0.9, color='orange', linewidth=1.5)

        # Formatting
        plt.xlabel("Pitch Index")
        plt.ylabel("Logit Value")
        plt.legend(loc='best')
        plt.xticks(np.arange(0, 128, 12))
        plt.grid(True, which='both', axis='x', alpha=0.2) # Show grid for the ticks
        plt.grid(True, axis='y', alpha=0.3)

        os.makedirs(self.vis_dir, exist_ok=True)
        plot_id = DissonanceConstrainer.vis_max_plots
        save_path = os.path.join(self.vis_dir, f"logit_shift_{plot_id}_pos_{self.cur_pos[0].item()}.png")
        plt.savefig(save_path, dpi=240)
        plt.close()

        print(f"Saved {save_path}")
        print(f"Position: {self.cur_pos[0].item()}")
        print(f"Active H: {H_indices}")
        print(f"Active N: {N_indices}")
        if DEBUG_DISS_CONSTRAINER:
            __import__("pdb").set_trace()

def add_renormalize(logit_slice: torch.Tensor, adjust: torch.Tensor):
    orig_lse = torch.logsumexp(logit_slice, dim=-1, keepdim=True)
    new_pitch_slice = logit_slice + adjust
    new_lse = torch.logsumexp(new_pitch_slice, dim=-1, keepdim=True)
    return new_pitch_slice - new_lse + orig_lse

# 0.5% and 99.5% percentiles of pitch ranges for each instrument
INST_RANGE_STATS = torch.tensor([
    [28, 93],
    [33, 96],
    [35, 96],
    [26, 96],
    [35, 94],
    [35, 89],
    [35, 88],
    [34, 91],
    [35, 106],
    [36, 108],
    [31, 108],
    [38, 92],
    [38, 91],
    [35, 99],
    [31, 89],
    [36, 90],
    [24, 86],
    [35, 91],
    [31, 90],
    [33, 90],
    [31, 87],
    [43, 90],
    [34, 89],
    [40, 88],
    [35, 84],
    [35, 84],
    [33, 84],
    [36, 84],
    [34, 84],
    [33, 84],
    [32, 82],
    [24, 86],
    [25, 75],
    [24, 69],
    [25, 69],
    [24, 74],
    [24, 88],
    [26, 81],
    [23, 91],
    [24, 96],
    [41, 93],
    [43, 83],
    [36, 77],
    [25, 70],
    [29, 92],
    [29, 87],
    [39, 95],
    [32, 65],
    [28, 91],
    [29, 91],
    [32, 92],
    [32, 89],
    [40, 85],
    [41, 85],
    [39, 87],
    [33, 97],
    [48, 85],
    [1, 81],
    [27, 79],
    [36, 90],
    [41, 78],
    [31, 90],
    [34, 92],
    [34, 88],
    [43, 91],
    [50, 82],
    [44, 77],
    [33, 74],
    [57, 88],
    [46, 87],
    [28, 71],
    [38, 88],
    [63, 105],
    [59, 94],
    [43, 112],
    [46, 92],
    [47, 96],
    [52, 110],
    [48, 98],
    [42, 98],
    [29, 99],
    [27, 94],
    [35, 80],
    [36, 84],
    [38, 95],
    [36, 89],
    [38, 87],
    [27, 103],
    [24, 102],
    [29, 90],
    [28, 96],
    [24, 93],
    [35, 115],
    [41, 102],
    [38, 99],
    [24, 121],
    [32, 96],
    [36, 100],
    [40, 100],
    [35, 95],
    [34, 100],
    [29, 91],
    [33, 90],
    [34, 98],
    [29, 94],
    [40, 83],
    [36, 98],
    [37, 98],
    [36, 111],
    [33, 95],
    [43, 90],
    [25, 91],
    [41, 96],
    [29, 90],
    [36, 89],
    [41, 107],
    [11, 101],
    [12, 80],
    [29, 95],
    [14, 111],
    [24, 103],
    [37, 101],
    [0, 105],
    [36, 98],
    [19, 93],
    [35, 88],
    [21, 107],
    [13, 118],

    # 128: Drum
    [0, 127],
    # 129: Harmonic Outline
    [0, 127],
], dtype=torch.long)
