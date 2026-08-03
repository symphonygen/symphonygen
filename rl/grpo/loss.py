import torch
import torch.nn.functional as F

def GRPO_loss_tuple(
    policy_log_probs,    # [Head] of [Batch, SeqLen]
    old_log_probs,
    policy_log_dist,     # [Head] of [Batch, SeqLen, Vocab]
    ref_log_dist,
    log_probs_mask,      # [Head] of [Batch, SeqLen]
    advantages,    # [Batch]
    loss_scales,   # [Head]
    kl_betas,      # [Head]
    head_names,    # [Head]
    epsilon=0.2,
):
    advantages = advantages.unsqueeze(-1)
    loss = 0.0
    metrics = {}

    for i, (p_lp, o_lp, p_dist, r_dist, mask) in enumerate(zip(
        policy_log_probs, old_log_probs, policy_log_dist, ref_log_dist, log_probs_mask
    )):
        # 1. Policy Loss (Clipped Surrogate Objective)
        ratio = torch.exp(p_lp - o_lp)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages

        # Calculate mean loss only over non-masked tokens for this specific head
        m_sum = mask.sum() + 1e-8
        policy_loss = -(torch.min(surr1, surr2) * mask).sum() / m_sum

        # 2. KL Divergence (Relative to Reference Model)
        # Using separate beta for each head type
        kl_div = (full_symmetric_kl_loss(r_dist, p_dist, mask, pre_norm=True) * mask).sum() / m_sum

        # 3. Weighted Combination
        head_loss = (policy_loss * loss_scales[i]) + (kl_betas[i] * kl_div)
        loss += head_loss

        # Logging metrics for debugging
        metrics[f"{head_names[i]}_policy_loss"] = policy_loss.detach()
        metrics[f"{head_names[i]}_kl_loss"] = kl_div.detach()

    return {
        "loss": loss,
        "metrics": metrics
    }

def full_symmetric_kl_loss(ref_log_dist, policy_log_dist, mask, pre_norm=False):
    """ dist: [Batch, SeqLen, Vocab] """
    log_ratio = ref_log_dist - policy_log_dist

    # Forward KL: Punishes "forgetting" what the base model knew.
    kl_forward = torch.sum(torch.exp(ref_log_dist) * log_ratio, dim=-1)
    # Reverse KL: Punishes "hallucinating" or choosing tokens the base model hated (e.g. wrong format).
    kl_reverse = torch.sum(torch.exp(policy_log_dist) * (-log_ratio), dim=-1)

    kl_per_token = (kl_forward + kl_reverse) / 2
    if not pre_norm:
        kl_loss = (kl_per_token * mask).sum() / (mask.sum() + 1e-8)
        return kl_loss
    return kl_per_token

def get_log_probs(
    logits: torch.FloatTensor, labels: torch.LongTensor, ignore_index: int,
    sum_log_probs=False,
    return_full_dist=False,
):
    """ logits: [Batch, SeqLen, Vocab], labels: [Batch, SeqLen] """
    log_dist = F.log_softmax(logits, dim=-1)
    mask = (labels != ignore_index)
    log_probs = log_dist.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1) * mask

    if sum_log_probs:
        log_probs = log_probs.sum(dim=-1)
    if return_full_dist:
        return log_probs, mask, log_dist
    return log_probs, mask
