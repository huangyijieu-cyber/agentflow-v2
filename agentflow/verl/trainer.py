import random
from contextlib import contextmanager
from copy import deepcopy
from typing import Dict, Tuple

import torch
from omegaconf import OmegaConf
from pprint import pprint
from tqdm import tqdm

from codetiming import Timer
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
# from verl.trainer.ppo.ray_trainer import (
from .diy.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    AdvantageEstimator,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking

from .daemon import AgentModeDaemon

import os
import json
import uuid
from collections import defaultdict, deque

import time
import gc

import numpy as np
import random
random.seed(42)
np.random.seed(24)

@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


## clean npu cache
def npu_safe_step(func):
    """装饰器：包装训练步骤，带 NPU 错误恢复"""
    def wrapper(self, *args, **kwargs):
        max_retries = 2
        for attempt in range(max_retries):
            try:
                return func(self, *args, **kwargs)
            except RuntimeError as e:
                if "ACL" in str(e) or "rtMemcpy" in str(e):
                    print(f"[WARNING] NPU clean error (attempt {attempt+1}/{max_retries}): {e}")
                    # 强制清理
                    if hasattr(torch, 'npu') and torch.npu.is_available():
                        torch.npu.synchronize()
                    torch.npu.empty_cache()
                    gc.collect()
                    if attempt < max_retries - 1:
                        continue  # 重试
                raise
        return func(self, *args, **kwargs)
    return wrapper


class AgentFlowTrainer(RayPPOTrainer):
    """
    Specialized PPO trainer for agent-based reinforcement learning.

    This trainer is designed specifically for scenarios where the model interacts with
    external environments, tools, or APIs through an AgentFlowServer. It simplifies
    the training loop by removing the complex conditional logic present in the original
    RayPPOTrainer and focusing on the agent mode workflow.

    Key differences from RayPPOTrainer:
    1. Uses AgentModeDaemon for server communication
    2. Simplified data flow without pop/union operations
    3. Direct batch processing through agent daemon
    4. Streamlined validation using agent_mode validation
    """

    ## default setting
    _stale_step_counter = 0

    @npu_safe_step
    def _validate(self):
        assert len(self.val_dataloader) == 1, "Please set val_batch_size to None for better throughput."

        # no empty check dataloader
        try:
            test_data = next(iter(self.val_dataloader))
        except StopIteration:
            raise ValueError("Validation dataloader is empty. Check your validation dataset.")

        # no empty check key
        print(f"Validation data keys: {test_data.keys()}")
        for key, value in test_data.items():
            if isinstance(value, list):
                print(f"Validation data {key} length: {len(value)}")
                if len(value) == 0:
                    print(f"Warning: Empty data in {key}")
            elif isinstance(value, torch.Tensor):
                print(f"Validation data {key} shape: {value.shape}")
                if value.numel() == 0:
                    print(f"Warning: Empty tensor in {key}")
            else:
                print(f"Validation data {key} type: {type(value)}")

        # no empty check
        if not test_data or all((isinstance(v, list) and len(v) == 0) or (isinstance(v, torch.Tensor) and v.numel() == 0) for v in test_data.values()):
            raise ValueError("Validation data is empty. Check your validation dataset.")

        test_batch = DataProto.from_single_dict(test_data)
        # test_batch.non_tensor_batch["step"] = np.ones_like(test_batch.non_tensor_batch["question"]) * self.global_steps
        self.async_rollout_manager.wake_up()
        self.agent_mode_daemon.set_up_data_and_server(
            test_batch.non_tensor_batch,
            self.async_rollout_manager.server_addresses,
            is_train=False,
        )

        # whether persisting queueing 
        if self.agent_mode_daemon._total_tasks_queued == 0:
            raise ValueError("No validation tasks were queued. Check data preparation.")

        self.agent_mode_daemon.run_until_all_finished()

        # Check if we have any completed rollouts, with more detailed error reporting
        completed_count = len(self.agent_mode_daemon._completed_rollouts)
        valid_count = len([r for r in self.agent_mode_daemon._completed_rollouts.values()
                          if r.triplets and len(r.triplets) > 0])
        original_count = self.agent_mode_daemon._total_tasks_queued

        completion_rate = completed_count / original_count if original_count > 0 else 0
        print(f"Validation summary: {completed_count}/{original_count} total rollouts ({completion_rate:.1%}), {valid_count} valid rollouts")

        # More lenient validation acceptance
        if completed_count == 0:
            raise ValueError("No validation tasks completed. Check server and agent execution.")

        # Accept partial results if we have some reasonable completion
        min_acceptable_rate = 0.1  # Accept if at least 10% completed
        if completion_rate < min_acceptable_rate:
            raise ValueError(f"Insufficient validation completion: {completion_rate:.1%} < {min_acceptable_rate:.1%}. "
                           f"Only {completed_count}/{original_count} tasks completed.")

        if valid_count == 0:
            print("Warning: No valid validation rollouts (all have empty triplets), using fallback metrics")
        else:
            print(f"Validation proceeding with {valid_count} valid rollouts ({valid_count/completed_count:.1%} of completed)")

        test_metrics = self.agent_mode_daemon.get_test_metrics()

        self.agent_mode_daemon.clear_data_and_server()
        self.async_rollout_manager.sleep()
        return test_metrics


    def _clean_npu_cache(self):
        # clean cache
        if hasattr(torch, 'npu') and torch.npu.is_available():
            try:
                torch.npu.synchronize()
            except Exception as e:
                print(f"[WARNING] NPU sync failed: {e}")
            torch.npu.empty_cache()
        gc.collect()

    ## score=0, score=1, classify(统一划分)
    def _trans_to_dpo_batch(self, history_batch_list: list):
        if not history_batch_list:
            print("## No DPO data with history")
            return None

        k_partitions = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        split_select_keys = [
            "input_ids", "attention_mask", "old_log_probs", "position_ids",
            "prompts", "response_mask", "responses", "token_level_scores",
        ]
        if self.use_reference_policy:
            split_select_keys += ["ref_log_prob"]

        non_tensor_select_keys = ["data_id_list", "uid", "traj_id_list", "traj_uid"]
        non_tensor_split_select_keys = ["turn_index_list", "rollout_id_list"]

        proto_list = []
        
        for history_batch in history_batch_list:
            token_level_scores = history_batch.batch["token_level_scores"]
            response_mask = history_batch.batch["response_mask"]
            uids = history_batch.non_tensor_batch["uid"]
            turn_indices = history_batch.non_tensor_batch["turn_index_list"]
            old_log_probs = history_batch.batch["old_log_probs"]

            scores = token_level_scores.sum(dim=-1)
            lengths = response_mask.sum(dim=-1)
            bsz = scores.shape[0]

            id2score = defaultdict(list)
            id2index = defaultdict(list)
            id2length = defaultdict(list)
            id2logP = defaultdict(list)

            with torch.no_grad():
                for i in range(bsz):
                    uid = str(uids[i])
                    turn = str(turn_indices[i])
                    ## replay collection
                    replay_collection = self.config.data.get("replay_collection", "uid-turn")
                    if replay_collection == "uid-turn":
                        key_index = f"{uid}::turn_{turn}"
                    elif replay_collection == "cross":
                        key_index = "0"
                    elif replay_collection == "turn":
                        key_index = f"0::trun_{turn}"
                    else:
                        raise ValueError(f"replay collection: {key_index} is invalid!")
                    
                    # 累积 logP（用于 baseline）
                    logP = (old_log_probs[i] * response_mask[i]).sum().item()
                    
                    id2score[key_index].append(scores[i])
                    id2length[key_index].append(lengths[i])
                    id2index[key_index].append(i)
                    id2logP[key_index].append(logP)

                # 计算每个组的 baseline：组内平均 per-token logP
                id2baseline = {}
                for idx in id2score.keys():
                    group_logPs = id2logP[idx]
                    group_lengths = [max(l.item(), 1) for l in id2length[idx]]
                    per_tokens = [lp / l for lp, l in zip(group_logPs, group_lengths)]
                    id2baseline[idx] = float(sum(per_tokens) / max(len(per_tokens), 1))

                # usage = defaultdict(int)
                # MAX_USE = 4

                for idx in id2score.keys():
                    group_size = len(id2score[idx])
                    if group_size <= 1:
                        continue

                    pos_samples = []
                    neg_samples = []
                    
                    for i in range(group_size):
                        score = id2score[idx][i].item()
                        length = id2length[idx][i].item()
                        index = id2index[idx][i]
                        
                        if length == 0:
                            continue
                            
                        if score > 0:
                            pos_samples.append((index, score, length))
                        else:
                            neg_samples.append((index, score, length))

                    if not pos_samples or not neg_samples:
                        continue

                    valid_pairs = []
                    # MAX_NEG_PER_POS = 5
                    random.shuffle(neg_samples)
                    
                    for p_idx, _, p_len in pos_samples:
                        # paired = 0
                        for n_idx, _, n_len in neg_samples:
                            # if paired >= MAX_NEG_PER_POS:
                            #     break
                            if max(p_len, n_len) / min(p_len, n_len) > 1.8:
                                continue
                            # if usage[p_idx] >= MAX_USE or usage[n_idx] >= MAX_USE:
                            #     continue
                            valid_pairs.append((p_idx, n_idx, p_len, n_len))
                            # usage[p_idx] += 1
                            # usage[n_idx] += 1
                            # paired += 1

                    if not valid_pairs:
                        continue

                    # ========== 采样权重：长度接近优先（与任务无关）==========
                    weights = []
                    for _, _, p_len, n_len in valid_pairs:
                        length_diff = abs(p_len - n_len)
                        # 长度越接近，权重越高；彻底消除"短 chosen 优先"的系统性偏见
                        w = 1.0 / (1.0 + length_diff / 64.0)
                        weights.append(max(w, 0.1))
                    
                    total_w = sum(weights)
                    weights = [w / total_w for w in weights]
                    # =======================================================

                    # max_pairs = min(len(valid_pairs), max(bsz, 64))

                    SOFT_CAP = 512  # 可按显存调整，建议 256~1024
                    max_pairs = min(len(valid_pairs), SOFT_CAP)

                    if len(valid_pairs) <= max_pairs:
                        selected = list(range(len(valid_pairs)))
                    else:
                        selected = np.random.choice(
                            len(valid_pairs), size=max_pairs, replace=False, p=weights
                        )

                    baseline = id2baseline[idx]
                    
                    for s in selected:
                        p_idx, n_idx, _, _ = valid_pairs[s]
                        proto = DataProto.from_single_dict(             
                            {key + "_a": history_batch.batch[key][p_idx: p_idx+1, :] for key in split_select_keys} 
                            | {key + "_b": history_batch.batch[key][n_idx: n_idx+1, :] for key in split_select_keys} 
                            | {key: history_batch.non_tensor_batch[key][p_idx: p_idx+1] for key in non_tensor_select_keys} 
                            | {key + "_a": history_batch.non_tensor_batch[key][p_idx: p_idx+1] for key in non_tensor_split_select_keys}
                            | {key + "_b": history_batch.non_tensor_batch[key][n_idx: n_idx+1] for key in non_tensor_split_select_keys}
                        )
                        # ========== 关键：baseline 和 group_size 放入 non_tensor_batch ==========
                        # DataProto.concat 会正确拼接 non_tensor_batch，不会丢失 per-sample 标量
                        proto.non_tensor_batch["group_baseline"] = np.array([baseline], dtype=np.float32)
                        proto.non_tensor_batch["group_size"] = np.array([group_size], dtype=np.int32)
                        # =====================================================================
                        proto_list.append(proto)

        if not proto_list:
            print("## DPO pair is not found!")
            return None

        remainder = len(proto_list) % k_partitions
        if remainder != 0:
            extra = random.choices(proto_list, k=k_partitions - remainder)
            proto_list.extend(extra)

        print(f">>> Obtain paired #batch: {len(proto_list)}")

        temperature = history_batch_list[0].meta_info.get("temperature", 0.7)
        for proto in proto_list:
            proto.meta_info["temperature"] = temperature
            
        return proto_list

    
    def _sample_dpo_data(self, dpo_queue: deque, bs: int):
        """
           从环形缓冲中随机有放回采样 bs 个单样本，拼接成 DataProto。
           只在需要训练 DPO 时才做一次 concat，避免每 step 全量拷贝。
        """
        n_size = len(dpo_queue)
        if n_size == 0:
          return None
        
        # # 随机回放机制：
        # indices = [random.randint(0, n_size - 1) for _ in range(bs)]
        # selected = [dpo_queue[i] for i in indices]

        # ## 新增“新鲜度经验”回放
        # # 索引 i 越接近 n_size-1（越新），权重越高
        # weights = np.exp(0.1 * np.arange(n_size))  # 指数加权
        # weights = weights / weights.sum()
        # indices = np.random.choice(n_size, size=bs, replace=True, p=weights)
        # selected = [dpo_queue[i] for i in indices]

        ## 保证至少都训一遍
        if bs <= n_size:
            indices = random.sample(range(n_size), bs)
        else:
            # buffer 不够，先全取一遍，剩余有放回补齐
            indices = random.sample(range(n_size), n_size)
            indices += random.choices(range(n_size), k=bs - n_size)
            random.shuffle(indices)

        selected = [dpo_queue[i] for i in indices]
        
        sample_data = DataProto.concat(selected)
        return sample_data



    @npu_safe_step
    def _train_step(self, batch_dict: dict) -> dict:
        # Isolate in a separate method to automatically recycle the variables before validation.
        batch: DataProto = DataProto.from_single_dict(batch_dict)
        metrics = {}
        timing_raw = {}

        algorithm = self.algorithm
        dpo_types = ["abpo", "slic", "simpo", "hspo"]

        if "+" in algorithm:
            parts = algorithm.split("+")
            assert len(parts) == 2, (
                f"Algorithm format error: '{algorithm}'. "
                f"Combined algorithms must contain exactly one '+' separator (e.g., 'slic+abpo'), "
                f"but found {len(parts)} parts: {parts}."
            )

        # data key check & no empty check
        print(f"Training data keys: {batch_dict.keys()}")
        for key, value in batch_dict.items():
            if isinstance(value, list):
                print(f"Training data {key} length: {len(value)}")
                if len(value) == 0:
                    print(f"Warning: Empty data in {key}")
            elif isinstance(value, torch.Tensor):
                print(f"Training data {key} shape: {value.shape}")
                if value.numel() == 0:
                    print(f"Warning: Empty tensor in {key}")
            else:
                print(f"Training data {key} type: {type(value)}")

        # ensure no empty
        if not batch_dict or all((isinstance(v, list) and len(v) == 0) or (isinstance(v, torch.Tensor) and v.numel() == 0) for v in batch_dict.values()):
            raise ValueError("Training data is empty. Check your training dataset.")

        with _timer("step", timing_raw):
            # When agent mode is enabled, we read the batch as it is.
            gen_batch = batch

            # generate a batch
            with _timer("gen", timing_raw), torch.no_grad():
                # gen_batch.non_tensor_batch["step"] = np.ones_like(gen_batch.non_tensor_batch["question"]) * self.global_steps
                self.async_rollout_manager.wake_up()
                self.agent_mode_daemon.set_up_data_and_server(
                    gen_batch.non_tensor_batch, self.async_rollout_manager.server_addresses
                )

                if self.agent_mode_daemon._total_tasks_queued == 0:
                    raise ValueError("No training tasks were queued. Check data preparation.")

                self.agent_mode_daemon.run_until_all_finished()

                if len(self.agent_mode_daemon._completed_rollouts) == 0:
                    raise ValueError("No training tasks completed. Check server and agent execution.")

                batch, agent_metrics = self.agent_mode_daemon.get_train_data_batch(
                    max_prompt_length=self.config.data.max_prompt_length,
                    max_response_length=self.config.data.max_response_length,
                    device=gen_batch.batch["fake_ids"].device,
                )
                metrics.update(agent_metrics)
                self.agent_mode_daemon.clear_data_and_server()
                self.async_rollout_manager.sleep()

            ## 2. REMAX advantage
            if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                with _timer("gen_max", timing_raw), torch.no_grad():
                    gen_baseline_batch = deepcopy(gen_batch)
                    gen_baseline_batch.meta_info["do_sample"] = False
                    gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                    batch = batch.union(gen_baseline_output)
                    reward_baseline_tensor = self.reward_fn(batch)
                    reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                    batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                    batch.batch["reward_baselines"] = reward_baseline_tensor

                    del gen_baseline_batch, gen_baseline_output

            # uid is used for algorithm like GRPO, should be aligned to data id
            with torch.no_grad():
                batch.non_tensor_batch["uid"] = batch.non_tensor_batch["data_id_list"]
                batch.non_tensor_batch["traj_uid"] = batch.non_tensor_batch["traj_id_list"]
                batch.batch["response_mask"] = compute_response_mask(batch)

            with _timer("reward", timing_raw), torch.no_grad():
                # compute reward model score
                if self.use_rm:
                    reward_tensor = self.rm_wg.compute_rm_score(batch)
                    batch = batch.union(reward_tensor)

                reward_extra_infos_dict = {}

            # for agent mode, pad the lengths to calculate old log prob, ref, and values
            batch, pad_size = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size)

            # recompute old_log_probs
            with _timer("old_log_prob", timing_raw), torch.no_grad():
                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                entropys = old_log_prob.batch["entropys"]
                response_masks = batch.batch["response_mask"]
                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                metrics.update(old_log_prob_metrics)
                old_log_prob.batch.pop("entropys")
                batch = batch.union(old_log_prob)

            if self.use_reference_policy:
                print(f"use_reference_policy: {self.use_reference_policy}")
                # compute reference log_prob
                with _timer("ref", timing_raw), torch.no_grad():
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                    print("ref_log_prob:", ref_log_prob)
                    batch = batch.union(ref_log_prob)

            # compute values
            if self.use_critic:
                with _timer("values", timing_raw), torch.no_grad():
                    values = self.critic_wg.compute_values(batch)
                    batch = batch.union(values)

            # for agent mode, unpad to calculate adv
            # it is important, as adv should be based on the raw traces
            batch = unpad_dataproto(batch, pad_size=pad_size)
            found = any(k in algorithm for k in dpo_types)
            if found:
                ## add stale data memory
                print('Append to stale data queue.')
                # 延迟初始化环形缓冲（避免类变量共享问题）
                if not hasattr(self, '_dpo_ring_buffer') or self._dpo_ring_buffer is None:
                    replay_buffer_size = self.config.data.get("replay_buffer_size", 8192)
                    self._dpo_ring_buffer = deque(maxlen=replay_buffer_size)

                new_dpo_pairs = self._trans_to_dpo_batch([batch])
                if new_dpo_pairs is not None:
                    # 逐个样本入队；超长的自动从左侧弹出，无大 tensor concat/截断拷贝
                    self._dpo_ring_buffer.extend(new_dpo_pairs)
                    print(f"DPO ring buffer size: {len(self._dpo_ring_buffer)} / {self._dpo_ring_buffer.maxlen}")
                self._stale_step_counter += 1
                stale_iteration = self.config.data.get("stale_iteration", 0)
                if self._stale_step_counter > stale_iteration:
                    print(f"Fetch ({self._stale_step_counter}) > stale_iteration({stale_iteration}) stale data from queue. ")
                    self._stale_step_counter -= 1
                else:
                    print(f'stale data queue size {self._stale_step_counter} <= {stale_iteration}, continue')
                    return None

            with _timer("adv", timing_raw), torch.no_grad():
                # if agent_mode is enabled, there is already token_level_scores
                # token_level_scores is not needed to compute here

                # compute rewards. apply_kl_penalty if available
                if self.config.algorithm.use_kl_in_reward:
                    batch, kl_metrics = apply_kl_penalty(
                        batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                    )
                    metrics.update(kl_metrics)
                else:
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                # compute advantages, executed on the driver process

                norm_adv_by_std_in_grpo = self.config.algorithm.get(
                    "norm_adv_by_std_in_grpo", True
                )  # GRPO adv normalization factor

                batch = compute_advantage(
                    batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=self.config.actor_rollout_ref.rollout.n,
                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                    config=self.config.algorithm,
                )

            # after advantages are assinged, we begin to drop (1) long prompt (2) floor to ppo minisize
            keep_indices = (~batch.batch["is_drop_mask"]).nonzero(as_tuple=True)[0]
            metrics["agent_mode/n_dropped_sample_because_of_prompt"] = (
                batch.batch["is_drop_mask"].shape[0] - keep_indices.shape[0]
            )
            batch = batch[keep_indices]
            # next, round to minibatch size
            mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            n_transition = len(batch)

            random_indices = list(range(n_transition))
            random.shuffle(random_indices)
            batch.reorder(torch.tensor(random_indices).type(torch.int32))
            n_remained_transition = n_transition // mini_batch_size * mini_batch_size
            batch = batch[list(range(n_remained_transition))]
            metrics["agent_mode/n_dropped_sample_because_of_mini_batch"] = n_transition - n_remained_transition

            n_transition = len(batch)
            # make sure divisible by k_partitions for seqlen_balancing
            k_partitions = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes  # n_gpus_per_node * nnodes 需要完全发放
            n_remained_transition = n_transition // k_partitions * k_partitions
            if n_remained_transition != n_transition:
                batch = batch[list(range(n_remained_transition))]
            metrics["agent_mode/n_dropped_sample_because_of_gpu_partitions"] = n_transition - n_remained_transition


            # compute global_valid tokens
            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

            # Agent mode note: Change the order of balance batch;
            #     1. first calculate advantage
            #     2. then drop the samples (too long prompt & floor to ppo minisize)
            #     3. balance
            # balance the number of valid tokens on each dp rank.
            # Note that this breaks the order of data inside the batch.
            # Please take care when you implement group based adv computation such as GRPO and rloo
            if self.config.trainer.balance_batch:
                # print("### balance_batch:", batch)
                self._balance_batch(batch, metrics=metrics)

            # update critic
            if self.use_critic:
                with _timer("update_critic", timing_raw):
                    critic_output = self.critic_wg.update_critic(batch)
                critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                metrics.update(critic_output_metrics)

            # implement critic warmup
            if self.config.trainer.critic_warmup <= self.global_steps:
                ### FFN-GRPO

                if "grpo" in algorithm:
                    # update actor with grpo
                    print(f"{time.strftime('%Y-%m-%d %H-%M-%S')}: >>> Begin GRPO")
                    with _timer("update_actor", timing_raw):
                        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                        batch.meta_info["max_response_length"] = self.config.data.max_response_length
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)
                    print(f"{time.strftime('%Y-%m-%d %H-%M-%S')}: >>> End GRPO")

                # update actor by dpo
                dpo_type = next((k for k in dpo_types if k in algorithm), None)
                if dpo_type is not None:
                    print(f"{time.strftime('%Y-%m-%d %H-%M-%S')}: >>> Begin DPO")
                    
                    with _timer("update_actor_by_dpo", timing_raw):
                        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable

                        if hasattr(self, '_dpo_ring_buffer') and len(self._dpo_ring_buffer) > 0:
                            batch_size = len(batch)
                            history_length = len(self._dpo_ring_buffer)
                            
                            # 保守策略：DPO batch不超过GRPO batch，且硬上限128（版本2优点）
                            # n_ratio = max(history_length // batch_size, 1)
                            # n_ratio = min(n_ratio, 2)  # 不放大，防止DPO主导训练
                            # n_batch = n_ratio * batch_size

                            ## batch size (full)
                            n_batch = history_length // k_partitions * k_partitions
                            
                            print(f"## update dpo ({n_batch}) bs from ({history_length})")
                            sample_batch = self._sample_dpo_data(self._dpo_ring_buffer, bs=n_batch)
                            
                            if sample_batch is not None and len(sample_batch) > 0:
                                print(f"{time.strftime('%Y-%m-%d %H-%M-%S')}: >>> Middle-2 DPO")
                                sample_batch.meta_info["dpo_type"] = dpo_type
                                sample_batch.meta_info["max_response_length"] = self.config.data.max_response_length
                                
                                # 建议：在DPO前也做长度检查，过滤极端长短样本
                                # 可在 update_actor_by_dpo 内部或这里加
                                dpo_output = self.actor_rollout_wg.update_actor_by_dpo(sample_batch)
                                dpo_output_metrics = reduce_metrics(dpo_output.meta_info["metrics"])
                                metrics.update(dpo_output_metrics)
                            else:
                                print("## DPO skipped this step (insufficient fresh pairs).")
                        else:
                            print("## DPO skipped: ring buffer empty.")
                            
                    print(f"{time.strftime('%Y-%m-%d %H-%M-%S')}: >>> END DPO")

            

        # compute training metrics
        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

        n_gpus = self.resource_pool_manager.get_n_gpus()
        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
        

        # 释放目标(pytorch, 释放训练过程)
        print(f"Free pytorch memory!")
        del batch
        self.agent_mode_daemon.clear_data_and_server()
        return metrics

    def _dump_rollout_data(self, inputs, outputs, scores, reward_extra_infos_dict, metrics, dump_path, is_train, batch, data_ids=None, ground_truths=None):
        data_type = 'train' if is_train else 'val'
        current_time = time.strftime("%Y%m%d_%H%M%S")
        step_dir = os.path.join(dump_path, data_type, f"step_{self.global_steps}_{current_time}")
        os.makedirs(step_dir, exist_ok=True)

        if data_ids is None:
            data_ids = batch.non_tensor_batch.get("data_id_list", [str(uuid.uuid4()) for _ in inputs])
        else:
            data_ids = data_ids[:len(inputs)] + [str(uuid.uuid4()) for _ in range(len(inputs) - len(data_ids))]

        question_groups = defaultdict(list)
        all_metrics = metrics.copy()

        for i, (input_text, output_text, score) in enumerate(zip(inputs, outputs, scores)):
            data_id = data_ids[i] if i < len(data_ids) else str(uuid.uuid4())

            record = {
                "query_index": i,
                "data_id": data_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "input": input_text,
                "output": output_text,
                "score": score,
                "metrics": {},
                "extra_info": {}
            }

            if ground_truths and i < len(ground_truths):
                record["ground_truth"] = ground_truths[i]

            for metric_name, metric_value in all_metrics.items():
                if isinstance(metric_value, (list, tuple)) and i < len(metric_value):
                    record["metrics"][metric_name] = metric_value[i]
                else:
                    record["metrics"][f"global_{metric_name}"] = metric_value

            if reward_extra_infos_dict:
                for key, values in reward_extra_infos_dict.items():
                    if i < len(values):
                        record["extra_info"][key] = values[i]

            question_groups[data_id].append(record)

        for data_id, records in question_groups.items():
            json_path = os.path.join(step_dir, f"query_{data_id}.json")

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)

        print(f"Successfully saved rollout data to {step_dir}")

    def fit(self):
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        assert self.async_rollout_mode, "If agent mode is enabled, async server must be enabled"
        self.agent_mode_daemon = AgentModeDaemon(
            self.config.agentflow.port,
            self.config.actor_rollout_ref.rollout.n,
            train_information={
                "model": self.config.actor_rollout_ref.model.path,
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
            },
            tokenizer=self.tokenizer,
            mini_batch_size=self.config.actor_rollout_ref.actor.ppo_mini_batch_size,
            pad_token_id=self.tokenizer.pad_token_id,
            enable_rollout_validation=self.config.agentflow.get("enable_rollout_validation", True),
            max_empty_retries=self.config.agentflow.get("max_empty_retries", 2),
        )
        self.agent_mode_daemon.start()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            self._clean_npu_cache()
            with torch.no_grad():
                val_metrics = self._validate()
            self._clean_npu_cache()
            assert val_metrics, f"{val_metrics}"
            print(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                is_last_step = self.global_steps >= self.total_training_steps

                # train step
                self._clean_npu_cache()
                metrics = self._train_step(batch_dict)
                self._clean_npu_cache()
                ## skip stale 
                if metrics is None:
                    print("stale data is not collnected enough, skip this batch.")
                    continue
                
                

                # validate
                # ### is_last_step close
                # is_last_step = False
                # timing_raw = {}
                # metrics = {}
                if (self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with _timer("validate", timing_raw), torch.no_grad():
                        self._clean_npu_cache()
                        val_metrics: dict = self._validate()
                        self._clean_npu_cache()

                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with _timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

                # step metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )

                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()

                    # This exit logic is to ensure a robust CI.
                    pprint(f"Flush the logger...")
                    del logger  # Make sure the loggers are flushed and closed properly
                    pprint(f"Training finished at step {self.global_steps}.")
                    return

                progress_bar.update(1)
                self.global_steps += 1