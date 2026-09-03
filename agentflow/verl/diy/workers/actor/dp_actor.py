# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os
from collections import defaultdict

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor
import torch.nn.functional as F

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty

from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

from .m2po import compute_m2po_policy_loss, compute_vtrace_policy_loss

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        ## leanrable parameters
        self.h = nn.Parameter(torch.tensor(0.0, device=get_device_name()))

        ## 添加学习参数
        if self.actor_optimizer is not None:
            base_lr = self.actor_optimizer.param_groups[0]['lr']
            h_lr = base_lr * 100.0
            self.actor_optimizer.add_param_group({
                'params': [self.h],
                'lr': h_lr,
            })
            if torch.distributed.get_rank() == 0:
                print(f"ABPO hardness h registered to optimizer, lr={h_lr}")


        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    def _split_model_inputs(self, pair_model_inputs):
        org_select_key = [
            "attention_mask",
            "input_ids",
            "old_log_probs",
            "position_ids",
            "responses",
            "response_mask",
            "token_level_scores"
        ]

        if self.config.use_kl_loss:
            org_select_key += ["ref_log_prob"]
        
        
        pair_a = {key : pair_model_inputs[key + "_a"] for key in org_select_key}
        pair_b = {key : pair_model_inputs[key + "_b"] for key in org_select_key}
        return pair_a, pair_b




    # def _ppo_loss(self, old_log_prob, log_prob, advantages, response_mask, loss_agg_mode, rollout_is_weights, max_response_length):
        
    #     loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
    #     # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

    #     # NOTE: Both mismatch diagnostic metrics (PPL, KL, etc.) and IS weight metrics
    #     # are computed centrally in ray_trainer.py for consistency and efficiency.
    #     # This ensures metrics are computed uniformly across all batches at the trainer level
    #     # and avoids redundant computation across workers and micro-batches.

    #     # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
    #     # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
    #     policy_loss_fn = get_policy_loss_fn(loss_mode)

    #     # clip advantages (避免梯度爆炸)
    #     advantages = torch.clamp(advantages, min=-5.0, max=5.0)

    #     # Compute policy loss (all functions return 4 values)
    #     pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
    #         old_log_prob=old_log_prob,
    #         log_prob=log_prob,
    #         advantages=advantages,
    #         response_mask=response_mask,
    #         loss_agg_mode=loss_agg_mode,
    #         config=self.config,
    #         rollout_is_weights=rollout_is_weights,
    #     )
    #     return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


    ### DR.GRPO
    def _ppo_loss(self, old_log_prob, log_prob, advantages, response_mask, loss_agg_mode, rollout_is_weights, max_response_length=1024):
        ## ================= DR.GRPO loss ========================================
        config = self.config
        clip_ratio = config.clip_ratio  # Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
        clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
        clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
        clip_ratio_c = config.get(  # Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            "clip_ratio_c", 3.0
        )

        cliprange = clip_ratio
        cliprange_low = clip_ratio_low
        cliprange_high = clip_ratio_high

        assert clip_ratio_c > 1.0, (
            "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
            + f" but get the value: {clip_ratio_c}."
        )

        negative_approx_kl = log_prob - old_log_prob
        # Clamp negative_approx_kl for stability
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl)
        ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

        pg_losses1 = -advantages * ratio
        if cliprange_low is None:
            cliprange_low = cliprange
        if cliprange_high is None:
            cliprange_high = cliprange
        pg_losses2 = -advantages * torch.clamp(
            ratio, 1 - cliprange_low, 1 + cliprange_high
        )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
        clip_pg_losses1 = torch.maximum(
            pg_losses1, pg_losses2
        )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
        pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

        pg_losses3 = -advantages * clip_ratio_c
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_clipfrac_lower = verl_F.masked_mean(
            torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
        )

        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

        # Apply rollout importance sampling weights if provided
        if rollout_is_weights is not None:
            pg_losses = pg_losses * rollout_is_weights
        
        # ===== Dr.GRPO 核心修改：固定长度归一化 =====
        ## "max_response_length must be provided for Dr.GRPO mode"
        # 对每个 response 的 loss 求和，除以固定 max_response_length，再对 batch 平均
        # 等价于: (1/G) * sum_i [ (1/MAX) * sum_t loss_{i,t} ]
        seq_loss = (pg_losses * response_mask).sum(dim=-1)  # (batch_size,)
        pg_loss = (seq_loss / max_response_length).mean()
        ## ==========================================================

        return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


    def _hspo_loss(self, inputs, log_probs, beta=1.0, gamma=0.25, sft_weight=0.05, 
               buffer=0.5, max_response_length=1024):
        """
        HSPO v3 Stable: 去除错误熵项，平滑窗口梯度，抑制难度正反馈
        """
        pair_a, pair_b = inputs
        log_prob_a, log_prob_b = log_probs
        response_mask_a = pair_a["response_mask"]
        response_mask_b = pair_b["response_mask"]

        # 1. 序列聚合 + 长度归一化
        len_a = response_mask_a.sum(dim=-1).clamp(min=1)
        len_b = response_mask_b.sum(dim=-1).clamp(min=1)
        logP_a = (log_prob_a * response_mask_a).sum(dim=-1) / len_a
        logP_b = (log_prob_b * response_mask_b).sum(dim=-1) / len_b
        delta = logP_a - logP_b

        # ========== 动态 Margin（核心修改）==========
        length_ratio = len_a / len_b.clamp(min=1.0)
        adaptive_gamma = gamma * torch.clamp(length_ratio, min=1.0, max=3.0)
        # ============================================

        hard_loss = torch.relu(adaptive_gamma - delta)
        raw_logits = (delta - adaptive_gamma) / beta
        soft_loss = -F.logsigmoid(raw_logits)
        
        open_mask = torch.sigmoid((delta - adaptive_gamma) * 2.0)
        close_mask = torch.sigmoid((adaptive_gamma + buffer - delta) * 2.0)
        soft_mask = open_mask * close_mask
        soft_loss = soft_loss * soft_mask

        with torch.no_grad():
            difficulty_weight = 1.0 + 0.2 * torch.sigmoid(-delta * 2.0)
        
        core_loss = (hard_loss + 0.5 * soft_loss) * difficulty_weight
        sft_loss = -logP_a.clamp(min=-50).mean()
        loss = core_loss.mean() + sft_weight * sft_loss
        
        with torch.no_grad():
            acc = (delta > 0).float().mean()
            violation = (delta < adaptive_gamma).float().mean()
            soft_grad_mag = torch.sigmoid(-raw_logits).mean() / beta
            prob_a = torch.exp(logP_a)
            prob_b = torch.exp(logP_b)
        
        metrics = {
            "dpo_loss": loss.item(),
            "slic_loss": hard_loss.mean().item(),
            "simpo_loss": soft_loss.mean().item(),
            "sft_loss": sft_loss.item(),
            "abpo/violation_rate": violation.item(),
            "abpo/soft_grad_mag": soft_grad_mag.item(),
            "abpo/delta_mean": delta.mean().item(),
            "abpo/delta_std": delta.std().item(),
            "rewards/accuracy": acc.item(),
            "rewards/chosen": prob_a.mean().item(),
            "rewards/rejected": prob_b.mean().item(),
            "rewards/margin": (prob_a.mean() - prob_b.mean()).item(),
            "length/chosen": len_a.float().mean().item(),
            "length/rejected": len_b.float().mean().item(),
            "adaptive_gamma": adaptive_gamma.mean().item(),
            "length_ratio": length_ratio.mean().item(),
        }
        return loss, metrics

    def _abpo_loss(self, inputs, log_probs, beta=1.0, gamma=0.25, logits_cap=10.0, 
               sft_weight=0.05, max_response_length=1024):
        """
        ABPO: Adaptive Boundary Preference Optimization
        """
        pair_a, pair_b = inputs             # a=chosen, b=rejected
        log_prob_a, log_prob_b = log_probs  # 当前策略的 log π(y|x)
        response_mask_a = pair_a["response_mask"]
        response_mask_b = pair_b["response_mask"]
        
        logP_a = (log_prob_a * response_mask_a).sum(dim=-1)
        logP_b = (log_prob_b * response_mask_b).sum(dim=-1)
        len_a = response_mask_a.sum(dim=-1).clamp(min=1)
        len_b = response_mask_b.sum(dim=-1).clamp(min=1)
        logP_a = logP_a / len_a
        logP_b = logP_b / len_b

        # ========== 动态 Margin（核心修改）==========
        length_ratio = len_a / len_b.clamp(min=1.0)
        adaptive_gamma = gamma * torch.clamp(length_ratio, min=1.0, max=3.0)
        # ============================================

        h_raw = torch.sigmoid(self.h)
        h = 0.25 + 0.6 * h_raw
        
        delta = logP_a - logP_b
        
        # 所有用到 gamma 的地方替换为 adaptive_gamma
        slic_loss = torch.relu(adaptive_gamma - delta)
        
        temperature = beta / (1.0 + 4 * h)
        raw_logits = (delta - adaptive_gamma) / temperature
        logits = raw_logits.clamp(min=-logits_cap, max=logits_cap)
        simpo_loss = -F.logsigmoid(logits)
        
        margin_distance = torch.abs(delta - adaptive_gamma)
        violation_boost = torch.sigmoid((adaptive_gamma - delta) * 0.3)
        harness_weight = h * torch.exp(-margin_distance * 2) + 0.3 * violation_boost
        harness_weight = torch.clamp(harness_weight, 0, 1)
        
        satisfication = torch.sigmoid((delta - adaptive_gamma) * 0.3)
        simpo_weight = (1 - harness_weight) * (1 - satisfication)
        
        loss = harness_weight * slic_loss + simpo_weight * simpo_loss
        loss = loss.mean()
        
        sft_loss = -logP_a.clamp(min=-50).mean()
        loss = loss + sft_weight * sft_loss
        
        with torch.no_grad():
            prob_a = torch.exp(logP_a)
            prob_b = torch.exp(logP_b)
            acc = ((logP_a - logP_b) > 0).float().mean()
        
        metrics = {
            "dpo_loss": loss.item(),
            "hardness": h.item(),
            "h_grad": self.h.grad.item() if self.h.grad is not None else 0.0,
            "h_raw": h_raw.item(),
            "temperature": temperature.mean().item(),
            "sft_loss": sft_loss.item(),
            "slic_loss": slic_loss.mean().item(),
            "simpo_loss": simpo_loss.mean().item(),
            "rewards/chosen": prob_a.mean().item(),
            "rewards/rejected": prob_b.mean().item(),
            "rewards/margin": (prob_a.mean() - prob_b.mean()).item(),
            "rewards/accuracy": acc.item(),
            "length/chosen": len_a.float().mean().item(),
            "length/rejected": len_b.float().mean().item(),
            # 新增监控
            "adaptive_gamma": adaptive_gamma.mean().item(),
            "length_ratio": length_ratio.mean().item(),
        }
        return loss, metrics

    def _simpo_loss(self, inputs, log_probs, beta=1.0, gamma=0.25, logits_cap=10.0, sft_weight=0.05, max_response_length=1024):
        """
        SimPO: Simple Preference Optimization
        参考: https://arxiv.org/abs/2405.14734

        beta:  # 2.0 比较大，梯度比较平缓，降低梯度强度，让优化更"柔和", 更大的 β  让 sigmoid((s_a - s_b - γ)/β) 的梯度更平缓，模型不会疯狂试图拉开差距，从而减少撞墙概率。
        gamma: # 0.3, 训练升起到一半后停下
        logits_cap: 10, 梯度容易截断
        sft_weight: 0.02 过大容易锚定不动
        
        与 SLIC 的关键差异:
        - 使用 Sigmoid (soft) 替代 Hinge Loss (hard)，梯度更平滑
        - 无需 SFT 损失项，纯偏好优化
        - 无需 ref model，计算极简
        - 对 stale 数据鲁棒性优于 DPO
        - 修复: 添加 beta 温度参数防止过度优化，添加 logits 裁剪防止梯度消失
        """
        pair_a, pair_b = inputs             # a=chosen, b=rejected
        log_prob_a, log_prob_b = log_probs  # 当前策略的 log π(y|x)
        response_mask_a = pair_a["response_mask"]
        response_mask_b = pair_b["response_mask"]

        # 1. 序列聚合 + 长度归一化（SimPO 必须归一化，避免长度博弈）
        logP_a = (log_prob_a * response_mask_a).sum(dim=-1)
        logP_b = (log_prob_b * response_mask_b).sum(dim=-1)
        len_a = response_mask_a.sum(dim=-1).clamp(min=1)
        len_b = response_mask_b.sum(dim=-1).clamp(min=1)

        # SimPO 核心: per-token 平均 log prob（与 SLIC 的 use_length_norm=True 相同）
        logP_a = logP_a / len_a
        logP_b = logP_b / len_b

        # ========== 动态 Margin（核心修改）==========
        length_ratio = len_a / len_b.clamp(min=1.0)
        adaptive_gamma = gamma * torch.clamp(length_ratio, min=1.0, max=3.0)
        # ============================================

        raw_logits = (logP_a - logP_b - adaptive_gamma) / beta
        logits = raw_logits.clamp(min=-logits_cap, max=logits_cap)
        simpo_loss = -F.logsigmoid(logits).mean()

        sft_loss = -logP_a.clamp(min=-50).mean()
        loss = simpo_loss + sft_weight * sft_loss
        
        with torch.no_grad():
            prob_a = torch.exp(logP_a)
            prob_b = torch.exp(logP_b)
            acc = ((logP_a - logP_b) > 0).float().mean()
        
        metrics = {
            "dpo_loss": loss.item(),
            "sft_loss": sft_loss.item(),
            "simpo_loss": simpo_loss.item(),
            "rewards/chosen": prob_a.mean().item(),
            "rewards/rejected": prob_b.mean().item(),
            "rewards/margin": (prob_a.mean() - prob_b.mean()).item(),
            "rewards/accuracy": acc.item(),
            "length/chosen": len_a.float().mean().item(),
            "length/rejected": len_b.float().mean().item(),
            "adaptive_gamma": adaptive_gamma.mean().item(),
            "length_ratio": length_ratio.mean().item(),
        }
        return loss, metrics
    

    def _slic_loss(self, inputs, log_probs, group_baseline, beta=0.1, max_response_length=1024):
        """
        SLiC: 组内 baseline 修正版
        - calibration 在 per-token 层级比较，与长度解耦
        - SFT 强化累积超额质量 corrected_a，长而扎实的序列获得更多正向梯度
        """
        pair_a, pair_b = inputs
        log_prob_a, log_prob_b = log_probs
        response_mask_a = pair_a["response_mask"]
        response_mask_b = pair_b["response_mask"]

        # 累积 logP
        logP_a = (log_prob_a * response_mask_a).sum(dim=-1)
        logP_b = (log_prob_b * response_mask_b).sum(dim=-1)
        len_a = response_mask_a.sum(dim=-1).float().clamp(min=1)
        len_b = response_mask_b.sum(dim=-1).float().clamp(min=1)

        # per-token 平均
        per_token_a = logP_a / len_a
        per_token_b = logP_b / len_b

        # baseline: 组内平均 per-token logP（从 non_tensor_batch 来，batch-wise）
        baseline = group_baseline.to(per_token_a.device)

        # 累积超额质量 = 实际累积 logP - 长度预期的累积 logP
        corrected_a = logP_a - len_a * baseline
        corrected_b = logP_b - len_b * baseline

        # ========== Calibration：per-token 层级比较，baseline 抵消，与长度无关 ==========
        delta = per_token_a - per_token_b
        
        # 动态 margin：如果 chosen 的 per-token 质量远高于 rejected，margin 略增
        # 注意：delta 与 margin 同量纲（per-token），不会出现量纲淹没
        quality_gap = delta
        adaptive_margin = beta * torch.clamp(1.0 + 0.3 * quality_gap, min=1.0, max=2.0)
        
        calibration_loss = torch.relu(adaptive_margin - delta).mean()
        # ================================================================================

        # ========== SFT：强化"累积超额质量"高的 chosen ==========
        # 长序列如果推导扎实，corrected_a 会显著为正；短序列即使 per-token 高，累积值有限
        # 保底：如果 baseline 异常（>=0，说明未正确传递），fallback 到标准 per-token SFT
        if baseline.mean() >= 0:
            sft_target = per_token_a
        else:
            sft_target = corrected_a
        
        sft_loss = -sft_target.clamp(min=-50).mean()
        # =======================================================

        sft_weight = 0.05
        loss = calibration_loss + sft_weight * sft_loss
        
        with torch.no_grad():
            prob_a = torch.exp(logP_a)
            prob_b = torch.exp(logP_b)
            acc = (delta > 0).float().mean()
        
        metrics = {
            "dpo_loss": loss.item(),
            "sft_loss": sft_loss.item(),
            "calibration_loss": calibration_loss.item(),
            "rewards/chosen": prob_a.mean().item(),
            "rewards/rejected": prob_b.mean().item(),
            "rewards/margin": (prob_a.mean() - prob_b.mean()).item(),
            "rewards/accuracy": acc.item(),
            "length/chosen": len_a.float().mean().item(),
            "length/rejected": len_b.float().mean().item(),
            "adaptive_margin": adaptive_margin.mean().item(),
            "quality_gap": quality_gap.mean().item(),
            "group_baseline": baseline.mean().item(),
            "corrected_a": corrected_a.mean().item(),
        }
        return loss, metrics
        

    def _kto_loss(self, inputs, log_probs, use_relative_advantage=False):
        """
        KTO 损失实现
        输入假设：
        - pair_a: desirable (好答案, label=1)
        - pair_b: undesirable (坏答案, label=0)
        不需要是同一个 prompt！
        """
        pair_a, pair_b = inputs
        log_prob_a, log_prob_b = log_probs
        ref_log_prob_a, ref_log_prob_b = pair_a["ref_log_prob"], pair_b["ref_log_prob"]
        response_mask_a, response_mask_b = pair_a["response_mask"], pair_b["response_mask"]

        beta = 0.1
        lambda_d = 1.0    # 好样本权重（通常保持 1.0）
        lambda_u = 2.0    # 坏样本权重（如果坏样本少，可设为 1.0-2.0 提高惩罚）
        
        # ========== 处理 Desirable 样本（pair_a，好答案） ==========
        log_prob_valid_a = (log_prob_a * response_mask_a).sum(dim=-1)
        ref_log_prob_valid_a = (ref_log_prob_a * response_mask_a).sum(dim=-1)
        len_a = response_mask_a.sum(dim=-1).clamp(min=1)
        
        # 隐式奖励 r = beta * log(pi/ref) （per-token 平均）
        r_desirable = (log_prob_valid_a / len_a - ref_log_prob_valid_a / len_a)

        # ========== 处理 Undesirable 样本（pair_b，坏答案） ==========
        log_prob_valid_b = (log_prob_b * response_mask_b).sum(dim=-1)
        ref_log_prob_valid_b = (ref_log_prob_b * response_mask_b).sum(dim=-1)
        len_b = response_mask_b.sum(dim=-1).clamp(min=1)
        
        r_undesirable = (log_prob_valid_b / len_b - ref_log_prob_valid_b / len_b)

        # ========== 相对优势归一化（可选，处理数据不平衡） ==========
        if use_relative_advantage:
            # 计算整个 batch（好+坏）的均值作为中性参考点 z_ref
            all_rewards = torch.cat([r_desirable, r_undesirable])
            z_ref = all_rewards.detach().mean()
        else:
            z_ref = 0.0  # 固定中性点

        # ========== KTO 非对称损失 ==========
        # 好样本：鼓励 r > z_ref （即 softplus(z_ref - r) 惩罚 r 太小）
        # 等价于 -log(sigmoid(r - z_ref))
        loss_desirable = F.softplus(beta * (z_ref - r_desirable)).mean()
        
        # 坏样本：鼓励 r < z_ref （即 softplus(r - z_ref) 惩罚 r 太大）
        # 等价于 -log(sigmoid(z_ref - r))
        loss_undesirable = F.softplus(lambda_u * beta * (r_undesirable - z_ref)).mean()

        # 加权组合（KTO 通常对坏样本给更高权重，因为通常坏样本少但更重要）
        loss = (loss_desirable + loss_undesirable)
        
        # 可选：添加 KL 正则（防止模型偏离 ref 太远）
        # loss += 0.01 * (r_desirable.mean().abs() + r_undesirable.mean().abs())

        # ========== 计算辅助指标 ==========
        with torch.no_grad():
            # 好样本中，reward 高于 z_ref 的比例（应该接近 1.0）
            acc_desirable = (r_desirable > z_ref).float().mean()
            
            # 坏样本中，reward 低于 z_ref 的比例（应该接近 1.0）
            acc_undesirable = (r_undesirable < z_ref).float().mean()
            
            # 平均奖励（用于监控）
            avg_len_a = len_a.float().mean()
            avg_len_b = len_b.float().mean()


        ## KTO: original metrics
        # metrics = {
        #     "kto_loss": loss.item(),
        #     "kto/loss_desirable": loss_desirable.item(),
        #     "kto/loss_undesirable": loss_undesirable.item(),
        #     "kto/reward_desirable": r_desirable.mean().item(),
        #     "kto/reward_undesirable": r_undesirable.mean().item(),
        #     "kto/z_ref": z_ref.item() if isinstance(z_ref, torch.Tensor) else z_ref,
        #     "kto/accuracy_desirable": acc_desirable.item(),      # 好样本应该 > z_ref
        #     "kto/accuracy_undesirable": acc_undesirable.item(),  # 坏样本应该 < z_ref
        #     "kto/reward_margin": (r_desirable.mean() - r_undesirable.mean()).item(),
        #     "length/chosen": avg_len_a.item(),
        #     "length/rejected": avg_len_b.item(),
        #     "kto/relative_advantage": float(use_relative_advantage),
        # }


        metrics = {
            "dpo_loss": loss.item(),
            "dpo_loss/relative_advantage": float(use_relative_advantage),
            "dpo_loss/batch_mean": z_ref.item() if isinstance(z_ref, torch.Tensor) else z_ref,  # 归一化前基线

            ## rewards chosen
            "rewards/chosen": r_desirable.mean().item(),       # 好样本应该 > z_ref
            "rewards/rejected": r_undesirable.mean().item(),   # 坏样本应该 < z_ref
            "rewards/margin": (r_desirable.mean() - r_undesirable.mean()).item(),
            "rewards/accuracy": (acc_desirable.item() + acc_undesirable.item()) * 0.5,
    
  
            "length/chosen": avg_len_a.item(),
            "length/rejected": avg_len_b.item(),            
        }

        return loss, metrics



    def _dpo_loss(self, inputs, log_probs, use_relative_advantage=True):
        ## 假设 A优于B
        pair_a, pair_b = inputs
        log_prob_a, log_prob_b = log_probs

        # old_log_prob_a, old_log_prob_b = pair_a["old_log_probs"], pair_b["old_log_probs"]
        ref_log_prob_a, ref_log_prob_b = pair_a["ref_log_prob"], pair_b["ref_log_prob"]
        response_mask_a, response_mask_b = pair_a["response_mask"], pair_b["response_mask"]

        beta = 0.5
        label_smoothing = 0.0
        loss_type = "ipo"
        

        # 序列聚合+防止除零
        log_prob_valid_a = (log_prob_a * response_mask_a).sum(dim=-1)
        log_prob_valid_b = (log_prob_b * response_mask_b).sum(dim=-1)
        ref_log_prob_valid_a = (ref_log_prob_a * response_mask_a).sum(dim=-1)
        ref_log_prob_valid_b = (ref_log_prob_b * response_mask_b).sum(dim=-1)

        # 长度归一化版本（解决短文本偏好失效问题）
        len_a = response_mask_a.sum(dim=-1).clamp(min=1)
        len_b = response_mask_b.sum(dim=-1).clamp(min=1)

        # 现在 policy_logratios 是 [batch]，代表每个样本的完整 log ratio
        policy_chosen_logratio = (log_prob_valid_a - ref_log_prob_valid_a) / len_a
        policy_rejected_logratio = (log_prob_valid_b - ref_log_prob_valid_b) / len_b


        if use_relative_advantage:
            # 计算整个 batch 的所有样本均值（包括 chosen 和 rejected）
            all_ratios = torch.cat([policy_chosen_logratio, policy_rejected_logratio])
            batch_mean = all_ratios.detach().mean()
            policy_chosen_logratio = policy_chosen_logratio - batch_mean
            policy_rejected_logratio = policy_rejected_logratio - batch_mean
            logits = beta * (policy_chosen_logratio - policy_rejected_logratio)
        else:
            # 标准 DPO（原有逻辑）
            logits = beta * (policy_chosen_logratio  - policy_rejected_logratio)  # 关键修复：减去 ref_logratios
        


        if loss_type == "sigmoid":
            losses = -F.logsigmoid(logits)
        elif loss_type == "hinge":
            losses = torch.relu(1 - logits)
        elif loss_type == "ipo":
            log_ratio_diff = policy_chosen_logratio - policy_rejected_logratio
            losses = (log_ratio_diff - 1/(2*beta)) ** 2
        elif loss_type == "js":
            # JS 散度的一般形式（示例）
            losses = F.softplus(-logits)  # 等价于 -F.logsigmoid(logits)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # label smoothing
        if label_smoothing > 0 and loss_type == "sigmoid":
            losses = (1 - label_smoothing) * losses - label_smoothing * F.logsigmoid(-logits)

        loss = losses.mean()

        # 计算辅助指标（用于 logging）
        with torch.no_grad():
            if use_relative_advantage:
                # 相对优势下的奖励（已经是相对值）
                chosen_rewards = beta * policy_chosen_logratio
                rejected_rewards = beta * policy_rejected_logratio
            else:
                # 隐式奖励：r = beta * log(pi/ref)
                chosen_rewards = beta * (log_prob_valid_a - ref_log_prob_valid_a) / len_a
                rejected_rewards = beta * (log_prob_valid_b - ref_log_prob_valid_b) / len_b
            
            reward_margin = chosen_rewards - rejected_rewards
            accuracy = (reward_margin > 0).float().mean()
            # 记录长度以监控"短文本偏好"效果
            avg_len_a = len_a.float().mean()
            avg_len_b = len_b.float().mean()
            # per-token 平均 log prob 用于监控
            avg_logp_a = (log_prob_valid_a / len_a).mean()
            avg_ref_logp_a = (ref_log_prob_valid_a / len_a).mean()

            
        metrics = {
            "rewards/chosen": chosen_rewards.mean().item(),
            "rewards/rejected": rejected_rewards.mean().item(),
            "rewards/margin": reward_margin.mean().item(),
            "rewards/accuracy": accuracy.item(),
            # metrics 中的 logps 也需要 mask 并归一化
            "logps/policy_chosen": avg_logp_a.item(),
            "logps/reference_chosen": avg_ref_logp_a.item(),
            # 长度监控（针对你之前的"越训越长"问题）
            "length/chosen": avg_len_a.item(),
            "length/rejected": avg_len_b.item(),
            "dpo_loss": loss.item(),
            "dpo_loss/relative_advantage": float(use_relative_advantage),
            "dpo_loss/batch_mean": batch_mean.item() if use_relative_advantage else 0.0,  # 归一化前基线
            "dpo_loss/logits_std": logits.std().item(),  # batch 内离散程度
        }
        
        return loss, metrics

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys


    @GPUMemoryLogger(role="dp actor by dpo", logger=logger)
    def update_policy_by_dpo(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        dpo_type = data.meta_info["dpo_type"]
        max_response_length = data.meta_info["max_response_length"]

        ## mini batch update
        mini_batches = data.split(self.config.ppo_mini_batch_size)


        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    pair_model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    pair_a, pair_b = self._split_model_inputs(pair_model_inputs)

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = pair_a["response_mask"].shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation
                    # Start: all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    
                    # entropy_a, log_prob_a = self._forward_micro_batch(
                    #     pair_a, temperature=temperature, calculate_entropy=calculate_entropy
                    # )
                    # entropy_b, log_prob_b = self._forward_micro_batch(
                    #     pair_b, temperature=temperature, calculate_entropy=calculate_entropy
                    # )
                    # print(">>> Use DPO Loss")

                    # ===== 优化1: pair_a + pair_b 在 batch 维拼接，只做一次 forward =====
                    concat_keys = ["input_ids", "attention_mask", "position_ids", "responses", "response_mask"]
                    concat_inputs = {}
                    for key in concat_keys:
                        if key in pair_a and key in pair_b:
                            concat_inputs[key] = torch.cat([pair_a[key], pair_b[key]], dim=0)
                    
                    # 统一 forward（batch_size 翻倍，但只算一次）
                    entropy_concat, log_probs_concat = self._forward_micro_batch(
                        concat_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                    )

                    ## 拆分回原pair
                    bs_a = pair_a["responses"].shape[0]
                    log_prob_a = log_probs_concat[:bs_a]
                    log_prob_b = log_probs_concat[bs_a:]

                    ## FFN-ABPO-1-loss
                    if dpo_type == "abpo":
                        dpo_loss, dpo_metrics = self._abpo_loss(
                            inputs=(pair_a, pair_b),
                            log_probs=(log_prob_a, log_prob_b),
                            max_response_length=max_response_length)
                    elif dpo_type == "hspo":
                        dpo_loss, dpo_metrics = self._hspo_loss(
                            inputs=(pair_a, pair_b),
                            log_probs=(log_prob_a, log_prob_b),
                            max_response_length=max_response_length)
                    elif dpo_type == "slic":
                        # 从 non_tensor_batch 读取 group_baseline（DataProto.concat 会正确拼接）
                        group_baselines = micro_batch.non_tensor_batch.get("group_baseline", None)
                        if group_baselines is not None:
                            # shape: (bsz, 1) 或 (bsz,) -> 转 tensor 并 squeeze
                            group_baseline_tensor = torch.from_numpy(group_baselines).to(
                                log_prob_a.device, dtype=log_prob_a.dtype
                            ).squeeze(-1)
                        else:
                            group_baseline_tensor = torch.zeros(log_prob_a.shape[0], device=log_prob_a.device)

                        dpo_loss, dpo_metrics = self._slic_loss(
                            inputs=(pair_a, pair_b),
                            log_probs=(log_prob_a, log_prob_b),
                            group_baseline=group_baseline_tensor,
                            max_response_length=max_response_length)
                    elif dpo_type == "simpo":
                        dpo_loss, dpo_metrics = self._simpo_loss(
                            inputs=(pair_a, pair_b),
                            log_probs=(log_prob_a, log_prob_b),
                            max_response_length=max_response_length)
                    else:
                        raise ValueError(f"Miss dpo type:{dpo_type}")

                    ## add loss scale factor
                    dpo_loss = dpo_loss * loss_scale_factor * 0.001  ## add scale factor (magic number)
                    dpo_loss.backward()
                    append_to_dict(metrics, dpo_metrics)
                    
                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor_dpo/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
    

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        max_response_length = data.meta_info["max_response_length"]

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    print(">>> Use PPO Loss")
                    if on_policy:
                        old_log_prob = log_prob.detach()
                    else:
                        old_log_prob = model_inputs["old_log_probs"]
                    
                    # Extract pre-computed rollout importance sampling weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = self._ppo_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob, 
                        advantages=advantages, 
                        response_mask=response_mask, 
                        loss_agg_mode=loss_agg_mode, 
                        rollout_is_weights=rollout_is_weights,
                        max_response_length=max_response_length
                        )


                    # ## M2PO: t00620714
                    # ratio_stats = None
                    # if self.config.get("use_m2po_loss", False):
                    #     print(">>> Use M2PO Loss")
                    #     M2_budget = self.config.get("M2_budget", None)
                    #     miniclip_low = self.config.get("miniclip_low", None)
                    #     miniclip_high = self.config.get("miniclip_high", None)
                    #     pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, ratio_stats = compute_m2po_policy_loss(
                    #         old_log_prob=old_log_prob,
                    #         log_prob=log_prob,
                    #         advantages=advantages,
                    #         response_mask=response_mask,
                    #         M2_budget=M2_budget,
                    #         # miniclip_low=miniclip_low,
                    #         # miniclip_high=miniclip_high,
                    #         loss_agg_mode=loss_agg_mode,
                    #     )
                    # if ratio_stats is not None:
                    #     append_to_dict(metrics, ratio_stats)
               
                        
                        

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
