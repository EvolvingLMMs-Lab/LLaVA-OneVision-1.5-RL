import torch


def _compute_sequence_level_ratio_and_advantages(
    log_ratio: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute sequence-level geometric mean ratios and average advantages per sequence (GSPO).

    Args:
        log_ratio: Log of probability ratios (logprobs - proximal_logprobs)
        advantages: Per-token advantages
        loss_mask: Boolean mask indicating valid tokens
        cu_seqlens: Cumulative sequence lengths. Required for 1D tensors (packed format).
            Shape: [batch_size + 1], where cu_seqlens[i] marks the start of sequence i.
            For a single sequence, use cu_seqlens=torch.tensor([0, seq_len]).

    Returns:
        ratio: Sequence-level importance sampling ratios (broadcast to all tokens)
        advantages: Sequence-averaged advantages (broadcast to all tokens)
            Note: We use mean instead of sum to keep gradient magnitude independent
            of sequence length. When multiplied by ratio and summed over tokens,
            this gives the correct total gradient contribution per sequence.
    """
    # Handle both 1D (packed) and 2D (padded) tensor shapes
    if log_ratio.ndim == 1:
        # For 1D tensors (packed format), cu_seqlens is required
        if cu_seqlens is None:
            raise ValueError(
                "cu_seqlens is required for 1D tensors (packed format). "
                "In AReaL, 1D tensors are produced by pack_tensor_dict() and always have cu_seqlens. "
                "For a single sequence, use cu_seqlens=torch.tensor([0, seq_len], dtype=torch.int32)."
            )

        # Packed sequences: use cu_seqlens boundaries
        batch_size = cu_seqlens.shape[0] - 1
        seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        # Create sequence index for each token: [0,0,0,1,1,2,2,2,2,...]
        sequence_idx = torch.arange(
            batch_size, device=log_ratio.device
        ).repeat_interleave(seq_lengths)

        # Use scatter_add for vectorized summation per sequence (faster than Python loop)
        masked_log_ratio = torch.where(loss_mask, log_ratio, 0.0)
        log_ratio_sum_per_seq = torch.zeros(
            batch_size, device=log_ratio.device, dtype=log_ratio.dtype
        ).scatter_add_(0, sequence_idx, masked_log_ratio)

        masked_advantages = torch.where(loss_mask, advantages, 0.0)
        advantages_sum_per_seq = torch.zeros(
            batch_size, device=advantages.device, dtype=advantages.dtype
        ).scatter_add_(0, sequence_idx, masked_advantages)

        valid_count_per_seq = (
            torch.zeros(batch_size, device=loss_mask.device, dtype=torch.int32)
            .scatter_add_(0, sequence_idx, loss_mask.int())
            .clamp(min=1)
        )

        # Compute sequence-level means
        log_ratio_mean_per_seq = log_ratio_sum_per_seq / valid_count_per_seq.to(
            log_ratio.dtype
        )
        adv_mean_per_seq = advantages_sum_per_seq / valid_count_per_seq.to(
            advantages.dtype
        )

        # Broadcast sequence-level values back to token-level
        ratio = torch.exp(log_ratio_mean_per_seq)[sequence_idx]
        ratio = torch.where(loss_mask, ratio, 0.0)

        advantages = adv_mean_per_seq[sequence_idx]
        advantages = torch.where(loss_mask, advantages, 0.0)
    else:
        # For 2D tensors (padded sequences)
        # Input shape: [batch_size, seq_len]
        # Compute mean log ratio over sequence length for each sample
        seq_log_ratio_mean = torch.where(loss_mask, log_ratio, 0.0).sum(dim=1) / (
            loss_mask.sum(dim=1).clamp(min=1)
        )
        # Broadcast back to original shape: each sequence gets its own geometric mean ratio
        ratio = torch.exp(seq_log_ratio_mean.unsqueeze(1).expand_as(log_ratio))
        # Apply mask
        ratio = torch.where(loss_mask, ratio, 0.0)

        # Average token advantages per sequence
        # This ensures gradient magnitude is independent of sequence length
        seq_lengths = loss_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        advantages = (advantages.sum(dim=-1, keepdim=True) / seq_lengths).expand_as(
            log_ratio
        )

    return ratio, advantages


def ppo_actor_loss_fn(
    logprobs: torch.Tensor,
    proximal_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    loss_mask: torch.Tensor,
    eps_clip_higher: float | None = None,
    c_clip: float | None = None,
    behav_imp_weight_cap: float | None = None,
    importance_sampling_level: str = "token",
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    When decoupled loss is disabled:
    1. if recompute logp, both old_logprobs and proximal_logprobs are recomputed logp;
    2. if no recomputation, both old_logp and proximal_logprobs are produced by the inference backend.

    When decoupled loss is enabled, proximal_logprobs is the recomputed logp,
    old_logprobs is produced by the inference engine.

    Args:
        importance_sampling_level: Level at which to compute importance sampling ratios.
            - 'token': Per-token ratios
            - 'sequence': Sequence-level geometric mean of per-token ratios (GSPO)
        cu_seqlens: Cumulative sequence lengths for packed sequences (1D tensors).
            Required when inputs are 1D and importance_sampling_level='sequence'.
            Shape: [batch_size + 1], where cu_seqlens[i] marks the start of sequence i.
            Not needed for 2D padded inputs (sequences identified by batch dimension).
    """
    loss_mask_count = loss_mask.count_nonzero() or 1

    if importance_sampling_level == "sequence":
        # GSPO: Compute sequence-level geometric mean of probability ratios
        log_ratio = logprobs - proximal_logprobs
        ratio, advantages = _compute_sequence_level_ratio_and_advantages(
            log_ratio, advantages, loss_mask, cu_seqlens
        )
    elif importance_sampling_level == "token":
        # Standard PPO: per-token ratio
        ratio = torch.where(loss_mask, torch.exp(logprobs - proximal_logprobs), 0)
    else:
        raise ValueError(
            f"Invalid importance_sampling_level: {importance_sampling_level}. "
            "Must be 'token' or 'sequence'."
        )

    clipped_ratio = torch.clamp(
        ratio,
        1.0 - eps_clip,
        1.0 + (eps_clip if eps_clip_higher is None else eps_clip_higher),
    )

    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    clip_mask = pg_loss1.detach() < pg_loss2.detach()
    pg_loss = torch.max(pg_loss1, pg_loss2)
    if c_clip is not None:
        assert c_clip > 1.0, c_clip
        pg_loss3 = torch.sign(advantages) * c_clip * advantages
        dual_clip_mask = pg_loss3.detach() < pg_loss.detach()
        pg_loss = torch.min(pg_loss, pg_loss3)
    else:
        dual_clip_mask = torch.zeros_like(clip_mask)
    behav_kl = proximal_logprobs - old_logprobs
    behav_imp_weight = behav_kl.exp()
    behav_mask = (
        (behav_imp_weight <= behav_imp_weight_cap).logical_and(loss_mask)
        if behav_imp_weight_cap is not None
        else loss_mask
    )
    behav_kl = torch.where(behav_mask, behav_kl, 0.0)
    behav_imp_weight = torch.where(behav_mask, behav_imp_weight, 0.0)
    pg_loss = pg_loss * behav_imp_weight
    logging_loss = pg_loss.detach()
    pg_loss = torch.where(loss_mask, pg_loss, 0).sum() / loss_mask_count
    clip_mask.logical_and_(loss_mask)
    dual_clip_mask.logical_and_(loss_mask)
    stat = dict(
        loss=logging_loss,
        importance_weight=ratio.detach(),
        approx_kl=(logprobs - proximal_logprobs).detach(),
        clip_mask=clip_mask,
        dual_clip_mask=dual_clip_mask,
    )
    if proximal_logprobs is not None:
        stat["behave_imp_weight"] = behav_imp_weight
        stat["behave_approx_kl"] = behav_kl
        stat["behave_mask"] = behav_mask
    return pg_loss, stat