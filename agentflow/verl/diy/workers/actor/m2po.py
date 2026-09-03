import torch
import verl.utils.torch_functional as verl_F


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss



@torch.no_grad()
def get_ratio_stats(ratio: torch.Tensor,
                    advantages: torch.Tensor,
                    response_mask: torch.Tensor,
                    log_prob: torch.Tensor,
                    old_log_prob: torch.Tensor,
                    bins=(0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0),
                    eps: float = 1e-12,
                    tol: float = 1e-6):
    """
    Summarize ratio distribution for three advantage conditions (pos, neg, nonzero).
    Keeps the (0.8, 1.0) bin AND adds an explicit eq_1.0 bin.

    Final bin order (len=9):
        (-inf, 0.2], (0.2, 0.5], (0.5, 0.8], (0.8, 1.0), eq_1.0,
        (1.0, 1.2], (1.2, 1.5], (1.5, 2.0], (2.0, +inf)

    Returns a dict with keys like:
        ratio_pos/inf_0.2, ..., ratio_pos/gt_2.0  (fractions in [0,1])
        ratio_pos/avg (mean of ratio over masked & condition tokens)
    """
    mask = response_mask.bool()
    finite = torch.isfinite(ratio)
    mask = mask & finite

    edges = torch.tensor(bins, device=ratio.device, dtype=ratio.dtype)


    # bucketize indices for 8 original bins:
    # 0:(-inf,0.2], 1:(0.2,0.5], 2:(0.5,0.8], 3:(0.8,1.0], 4:(1.0,1.2], 5:(1.2,1.5], 6:(1.5,2.0], 7:(2.0,+inf)

    ## 英伟达 bucket size, 昇腾不支持
    # bin_idx = torch.bucketize(ratio, edges, right=True)

    if ratio.device.type == "npu":
        bin_idx = torch.bucketize(ratio.cpu(), edges.cpu(), right=True).to(ratio.device)
    else:
        bin_idx = torch.bucketize(ratio, edges, right=True)
    


    def compute_for(cond: torch.Tensor):
        m = mask & cond
        # 9 bins now (insert eq_1.0 at index 4)
        counts = torch.zeros(len(bins) + 2, device=ratio.device, dtype=torch.float32)

        if m.any():
            eq1_mask = (torch.abs(ratio - 1.0) <= tol) & m
            not_eq1_mask = m & (~eq1_mask)

            if not_eq1_mask.any():
                idx = bin_idx[not_eq1_mask].reshape(-1).long()
                # shift indices >= 4 (i.e., > 1.0 side) by +1 to make room for eq_1.0 at index 4
                shift = (idx >= 4).long()
                idx = idx + shift
                counts.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))

            # put exact-1.0 counts at index 4
            counts[4] = eq1_mask.sum()

        total = counts.sum()
        frac = counts / (total + eps)

        # average ratio under this condition (masked)
        if m.any():
            avg = ratio[m].sum() / (m.sum() + eps)
        else:
            avg = torch.tensor(0.0, device=ratio.device, dtype=torch.float32)

        return frac, avg

    results = {}
    conditions = {
        "pos": advantages > 0,
        "neg": advantages < 0,
        "nonzero": advantages != 0
    }

    bin_names = [
        f"inf_{bins[0]}", f"{bins[0]}_{bins[1]}", f"{bins[1]}_{bins[2]}", f"{bins[2]}_{bins[3]}",
        "eq_1.0",
        f"{bins[3]}_{bins[4]}", f"{bins[4]}_{bins[5]}", f"{bins[5]}_{bins[6]}", f"gt_{bins[-1]}"
    ]

    for cond_name, cond_mask in conditions.items():
        frac, avg = compute_for(cond_mask)
        for i, bn in enumerate(bin_names):
            results[f"ratio_{cond_name}/{bn}"] = frac[i].item()
        results[f"ratio_{cond_name}/avg"] = float(avg.item())

    # ---- append: conditional KL means ----
    negative_approx_kl = log_prob - old_log_prob    # = log(ratio)
    approx_kl = -negative_approx_kl                 # PPO-style approx KL ≥ 0

    base_mask = response_mask.bool() & torch.isfinite(ratio) \
                & torch.isfinite(log_prob) & torch.isfinite(old_log_prob)

    m_neg_r_lt_1 = base_mask & (advantages < 0) & (ratio < (1.0 - tol))
    m_pos_r_gt_1 = base_mask & (advantages > 0) & (ratio > (1.0 + tol))

    def _mean_where(x: torch.Tensor, m: torch.Tensor):
        n = m.sum()
        if n.item() == 0:
            return torch.tensor(0.0, device=x.device, dtype=torch.float32)
        return x[m].sum() / (n + eps)

    results["kl_neg_r_lt_1/mean"] = float(_mean_where(approx_kl, m_neg_r_lt_1).item())
    results["kl_pos_r_gt_1/mean"] = float(_mean_where(approx_kl, m_pos_r_gt_1).item())

    # optional diagnostics
    total_tokens = int(mask.sum().item())
    results["kl_neg_r_lt_1/count"] = int(m_neg_r_lt_1.sum().item())
    results["kl_pos_r_gt_1/count"] = int(m_pos_r_gt_1.sum().item())
    results["kl_neg_r_lt_1/frac_tokens"] = float((m_neg_r_lt_1.sum() / (mask.sum() + eps)).item()) if total_tokens > 0 else 0.0
    results["kl_pos_r_gt_1/frac_tokens"] = float((m_pos_r_gt_1.sum() / (mask.sum() + eps)).item()) if total_tokens > 0 else 0.0

    # ---- append: KL stats (flat with kl_stats/ prefix) ----

    conds = {
        "pos": advantages > 0,
        "neg": advantages < 0,
        "nonzero": advantages != 0,
    }

    def _mean_where(x: torch.Tensor, m: torch.Tensor):
        n = m.sum()
        if n.item() == 0:
            return torch.tensor(0.0, device=x.device, dtype=torch.float32)
        return x[m].sum() / (n + eps)

    for name, cmask in conds.items():
        m = base_mask & cmask
        results[f"kl_stats/{name}_abs_mean"]    = float(_mean_where(negative_approx_kl.abs(), m).item())
        results[f"kl_stats/{name}_sq_mean"]     = float(_mean_where(negative_approx_kl.pow(2), m).item())
        results[f"kl_stats/{name}_signed_mean"] = float(_mean_where(-negative_approx_kl, m).item())
        # results[f"kl_stats/{name}_approx_mean"] = float(_mean_where(approx_kl, m).item())

    return results



# we need it
def _get_trust_region_tokens_delta_sq(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
):
    mask = response_mask.bool()
    adv_example = advantages[:,0]
    pos_adv_mask = adv_example > 1e-12
    neg_adv_mask = adv_example < -1e-12

    delta = (old_log_prob - log_prob)             # Δ = log p_old - log p_new
    ratio = torch.exp(-delta)                     # r = exp(log_new - log_old)

    pos_adv_response_mask = mask[pos_adv_mask]
    neg_adv_response_mask = mask[neg_adv_mask]

    pos_adv_ratio = ratio[pos_adv_mask]
    neg_adv_ratio = ratio[neg_adv_mask]

    pos_adv_r_gt_1_mask = pos_adv_ratio > 1.0 + 1e-12
    neg_adv_r_lt_1_mask = neg_adv_ratio < 1.0 - 1e-12

    delta_sq = delta.pow(2)
    pos_adv_harm_tokens_delta_sq = delta_sq[pos_adv_mask][pos_adv_r_gt_1_mask & pos_adv_response_mask]
    neg_adv_harm_tokens_delta_sq = delta_sq[neg_adv_mask][neg_adv_r_lt_1_mask & neg_adv_response_mask]

    tr_tokens_delta_sq = torch.cat([pos_adv_harm_tokens_delta_sq, neg_adv_harm_tokens_delta_sq])

    print("#mask:", mask, "adv_example:", adv_example, "ratio:", ratio, "tr_tokens_delta_sq:", tr_tokens_delta_sq, "delta:", delta, "delta_sq:", delta)

    return tr_tokens_delta_sq


# we need it
def _solve_tau_from_sorted_delta2(sorted_delta2: torch.Tensor, target_sum: float) -> float:
    """
    Given sorted ascending values v_i = Δ_i^2 (i=0..n-1) and a target sum S,
    find τ^2 such that sum_i min(v_i, τ^2) = S.
    This uses a single pass over breakpoints without binary search.

    Returns:
        tau (float): sqrt(τ^2). If S >= sum(v_i), returns +inf (no clipping needed).
                     If S <= 0, returns 0.0 (clip everything to 0).
    """

    if sorted_delta2.numel() == 0:
        return 100000

    total = float(sorted_delta2.sum().item())
    if target_sum >= total - 1e-12: # no clipping needed
        return 100000
    if target_sum <= 1e-12: # clip everything to 0
        return 0.0

    csum = torch.cumsum(sorted_delta2, dim=0)  # prefix sums
    n = sorted_delta2.numel()

    for k in range(0,n):
        left_sum = float(csum[k].item())
        rest = n - k - 1
        m2 = sorted_delta2[k].item() - 1e-12
        if m2 * rest + left_sum >= target_sum - 1e-12:
            print(f"================")
            print(f"n: {n}, k: {k}, left_sum: {left_sum}, target_sum: {target_sum}")
            print(f"sorted_delta2[{k}]: {sorted_delta2[k].item()}")
            print(f"{list(zip(sorted_delta2[k-5:k+5].tolist(), csum[k-5:k+5].tolist()))}")
            print((sorted_delta2 == 0).float().mean())
            if k == 0:
                return 0.0, csum[-1].item() / n
            else:
                M2_after = (sorted_delta2[k-1].item() * (rest + 1) + float(csum[k-1].item())) / n
                tau = float(sorted_delta2[k-1].item() - 1e-12) ** 0.5
                print(f"tau: {tau}, k-1: {k-1}")
                return tau, M2_after

    return 100000

# we need it
def kpo_clip_harmful_tokens(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    KL2_budget: float = None
):
    """
    Decide global clip scalars (clip_low, clip_high) under an M2 budget.

    Policy:
      - Consider only harmful tokens: (A>0 & r>1) or (A<0 & r<1), where r = exp(log_new - log_old).
      - Sort harmful tokens by delta^2 = (log p_old - log p_new)^2 ascending.
      - Find a single threshold τ so that capping |delta| at τ across harmful tokens
        yields overall M2 <= KL2_budget.
      - Map τ to two global ratio bounds:
            clip_low  = exp(-τ)  (applies to adv<0 & r<1)
            clip_high = exp(+τ)  (applies to adv>0 & r>1)
      - Non-harmful quadrants are not constrained by these bounds.

    Returns:
      clip_low  (float): lower clamp for tokens with (adv<0 & r<1)
      clip_high (float): upper clamp for tokens with (adv>0 & r>1)
    """
    assert KL2_budget is not None, "KL2_budget must be set."

    tr_tokens_delta_sq = _get_trust_region_tokens_delta_sq(old_log_prob, log_prob, advantages, response_mask)
    token_num = tr_tokens_delta_sq.numel()

    if token_num == 0: # no clipping needed
        print(f"#M2_now: {0.0}")
        return 0.0, 100000, 0.0, 0.0

    target_total = KL2_budget * float(token_num)
    M2_now = float(tr_tokens_delta_sq.sum().detach().item() / token_num)
    print(f"#1-M2_now: {M2_now}")

    if M2_now <= KL2_budget + 1e-12:
        # No clipping needed -> effectively no constraint
        return 0.0, 100000, M2_now, M2_now

    print(f"#tr-M2_now: {M2_now}")
    print(f"#KL2_budget: {KL2_budget}")

    # import pdb; pdb.set_trace()

    sorted_delta2, _ = torch.sort(tr_tokens_delta_sq)  # ascending
    tau, M2_after = _solve_tau_from_sorted_delta2(sorted_delta2, target_total)

    # Map |Δ|<=τ to ratio bounds per quadrant
    clip_low = float(torch.exp(torch.tensor(-tau)).item())   # applies to (adv<0, r<1)
    clip_high = float(torch.exp(torch.tensor(+tau)).item())  # applies to (adv>0, r>1)

    return clip_low, clip_high, M2_now, M2_after



def compute_m2po_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    M2_budget: float = None,
    miniclip_low: float = 0.3,
    miniclip_high: float = 0.5,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute policy loss under an M2 (KL^2) budget using per-token clipping bounds.

    Steps:
      1) Get per-token (clip_low, clip_high) from kpo_clip.
      2) Compute ratio and apply element-wise clamp.
      3) Compute surrogate loss -A * ratio_clipped and aggregate.

    Returns:
      pg_loss:       aggregated policy loss
      stats:         dict with basic diagnostics (M2 before/after, fractions)
      clip_low/high: the per-token bounds actually used
    """

    clip_low, clip_high, M2_data, M2_after = kpo_clip_harmful_tokens(old_log_prob, log_prob, advantages, response_mask, M2_budget)

    clip_low = 1 - clip_low
    clip_high = clip_high - 1
    print(f"clip_low: {clip_low}, clip_high: {clip_high}")
    if miniclip_low is not None and clip_low < miniclip_low:
        clip_low = miniclip_low
    if miniclip_high is not None and clip_high > miniclip_high:
        clip_high = miniclip_high

    # ratio = exp(log_new - log_old)
    ratio = torch.exp(log_prob - old_log_prob)
    ppo_kl = verl_F.masked_mean(-(log_prob - old_log_prob), response_mask)

    ratio_stats = get_ratio_stats(ratio, advantages, response_mask, log_prob, old_log_prob)

    ##### clip
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_low, 1 + clip_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    ### vtrace -----------
    capped_ratio = torch.clamp(ratio, min=0.5, max=2.0)
    capped_ratio = capped_ratio / capped_ratio.mean()
    surrogate = -advantages * capped_ratio
    clip_pg_losses_1 = torch.maximum(clip_pg_losses1, surrogate)
    ## add ----------------

    pg_loss = agg_loss(loss_mat=clip_pg_losses1, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)


    ratio_stats["m2po/clip_low"] = clip_low
    ratio_stats["m2po/clip_high"] = clip_high
    ratio_stats["m2po/M2"] = M2_data
    ratio_stats["m2po/M2_after"] = M2_after
    ratio_stats["m2po/M2_budget"] = M2_budget

    return pg_loss, pg_clipfrac, ppo_kl, (ppo_kl - ppo_kl), ratio_stats



def compute_vtrace_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    M2_budget: float = None,
    miniclip_low: float = 0.3,
    miniclip_high: float = 0.5,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute policy loss under an M2 (KL^2) budget using per-token clipping bounds.

    Steps:
      1) Get per-token (clip_low, clip_high) from kpo_clip.
      2) Compute ratio and apply element-wise clamp.
      3) Compute surrogate loss -A * ratio_clipped and aggregate.

    Returns:
      pg_loss:       aggregated policy loss
      stats:         dict with basic diagnostics (M2 before/after, fractions)
      clip_low/high: the per-token bounds actually used
    """

    ## 重要性采样比率
    ratio = torch.exp(log_prob - old_log_prob)

    ## 截断比率
    capped_ratio = torch.clamp(ratio, min=0.5, max=2.0)

    ## 归一化
    capped_ratio = capped_ratio / capped_ratio.mean()

    # 计算损失
    surrogate = -capped_ratio * advantages
    clipped_surrogate = -torch.clamp(capped_ratio, 1-miniclip_low, 1+miniclip_high) * advantages

    ## ppo kl
    ppo_kl = verl_F.masked_mean(-(log_prob - old_log_prob), response_mask)

    ratio_stats = get_ratio_stats(ratio, advantages, response_mask, log_prob, old_log_prob)

    clip_pg_loss = torch.maximum(surrogate, clipped_surrogate) # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
 
    pg_clipfrac = verl_F.masked_mean(torch.gt(clipped_surrogate, surrogate).float(), response_mask)

    pg_loss = agg_loss(loss_mat=clip_pg_loss, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, (ppo_kl - ppo_kl), ratio_stats