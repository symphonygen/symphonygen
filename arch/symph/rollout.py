import torch
from rl.grpo.loss import get_log_probs
from rl.grpo.rollout import SymphonyRollout
from arch.symph.model import MusicModel3D
from arch.vocab import *

class SymphonyRollout3D(SymphonyRollout):
    @staticmethod
    def compute_completion_log_probs(
        model: MusicModel3D, batch: torch.Tensor | tuple[torch.Tensor, ...],
    ):
        events, _, _, track_id, _, new_inst, _ = batch
        B, Bars, Tracks, Events = events.shape

        meta_pred_outputs, event_dec_outputs, event_labels = model(*batch, return_outputs=True)

        track_id_probs, track_id_prob_mask, track_id_dist = get_log_probs(
            meta_pred_outputs.track_id_logits.view(B, Bars * Tracks, -1), track_id.view(B, -1), META_PAD_TRACK,
            return_full_dist=True,
        )
        new_inst_probs, new_inst_prob_mask, new_inst_dist = get_log_probs(
            meta_pred_outputs.new_inst_logits.view(B, Bars * Tracks, -1), new_inst.view(B, -1), META_PAD_INST,
            return_full_dist=True,
        )
        event_probs, event_prob_mask, event_dist = get_log_probs(
            event_dec_outputs.logits.view(B, Bars * Tracks * Events, -1), event_labels.view(B, -1), PAD_EVENT,
            return_full_dist=True,
        )

        return (
            (event_probs, track_id_probs, new_inst_probs),
            (event_prob_mask, track_id_prob_mask, new_inst_prob_mask),
            (event_dist, track_id_dist, new_inst_dist),
        )
