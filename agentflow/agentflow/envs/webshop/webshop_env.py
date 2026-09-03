import os
import gym
import re
import numpy as np

from agentflow.envs.webshop.webshop_template import WEBSHOP_TEMPLATE_NO_HIS, WEBSHOP_TEMPLATE
from agentflow.envs.memory import SimpleMemory

def upper_path(path):
    upper_path = os.path.dirname(path)
    return upper_path

## file path
file_path = os.path.abspath(__file__)
for _ in range(5):
    file_path = upper_path(file_path)
file_path = os.path.join(file_path, "data/WebShop-master")

import sys
sys.path.append(file_path)

# ==================== 多进程并发核心修改：索引隔离 ====================
import shutil
import tempfile

# 每个 Worker 进程分配一个私有索引目录
_pid = os.getpid()
_private_index_dir = f"/tmp/webshop_index_worker_{_pid}"

def _find_original_index():
    """自动探测原始 Lucene 索引路径"""
    # 优先从环境变量读取（如果你知道确切路径，可提前 export）
    env_path = os.environ.get("WEBSHOP_INDEX_DIR")
    if env_path and os.path.exists(env_path):
        return env_path
    
    # 常见路径探测（根据你的实际部署调整）
    candidates = [
        os.path.join(file_path, "indexes"),                    # 数据目录下
        os.path.join(file_path, "index"),                      # 数据目录下
        os.path.join(file_path, "web_agent_site", "indexes"),  # 包内路径
        "/tmp/webshop_default_index",
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.isdir(c):
            # 验证是否为有效 Lucene 索引（包含 segments 文件）
            if any(f.startswith('segments') for f in os.listdir(c)):
                return c
    return None

# 只在进程首次 import 时复制一次
if not os.path.exists(_private_index_dir):
    _src_index = _find_original_index()
    if _src_index:
        print(f"[Worker {_pid}] Copying WebShop index from {_src_index} -> {_private_index_dir} ...")
        shutil.copytree(_src_index, _private_index_dir)
        print(f"[Worker {_pid}] Index ready. All Lucene operations are isolated.")
    else:
        print(f"[Worker {_pid}] ERROR: Cannot find WebShop Lucene index.")
        print(f"[Worker {_pid}] Please set export WEBSHOP_INDEX_DIR=/path/to/index before running.")

# Monkey-patch：让 WebShop 的 SimServer 使用私有索引
import web_agent_site.engine.engine as _engine_module
from pyserini.search.lucene import LuceneSearcher

_original_init_search_engine = _engine_module.init_search_engine

def _isolated_init_search_engine(num_products=None):
    """每个 Worker 进程启动独立的 LuceneSearcher，mmap 自己的索引副本"""
    if os.path.exists(_private_index_dir):
        return LuceneSearcher(_private_index_dir)
    # 如果复制失败，回退到原始行为（会报错，方便定位路径问题）
    return _original_init_search_engine(num_products)

_engine_module.init_search_engine = _isolated_init_search_engine
print(f"[Worker {_pid}] Lucene index isolation enabled. Multi-process safe.")
# ==================== Monkey-patch end ====================

# # ==================== 多进程并发核心修改：绕过 Lucene ====================
# import web_agent_site.engine.engine as _engine_module

# # 停用词表，避免 "Find", "me", "with" 这类词把匹配拖空
# _STOPWORDS = {
#     'find', 'me', 'with', 'and', 'for', 'a', 'an', 'the', 'in', 'of', 'to', 
#     'my', 'i', 'want', 'need', 'looking', 'is', 'are', 'it', 'this', 'that',
#     'lower', 'than', 'dollars', 'price', 'color', 'size', 'daily', 'wear'
# }

# def _mock_init_search_engine(num_products=None):
#     """Mock 搜索引擎，不启动 JVM，不持有任何文件句柄"""
#     return {"type": "mock", "num_products": num_products}

# def _mock_get_top_n_product_from_keywords(keywords, search_engine, all_products, product_item_dict, top_n=10):
#     """
#     纯 Python 内存搜索，替代 Lucene BM25。
#     关键修复：
#       1. 过滤停用词，避免长查询中无关词导致匹配失败
#       2. 确保返回的 ASIN 必须存在于 product_item_dict 中
#       3. 如果匹配结果为空，兜底返回所有有效商品，防止 episode 卡死
#     """
#     # 收集所有有效 ASIN（确保在 product_item_dict 中存在）
#     all_valid_asins = []
#     for p in all_products:
#         asin = p.get('asin') or p.get('id')
#         if asin and asin in product_item_dict:
#             all_valid_asins.append(asin)
    
#     # 如果没有关键词，直接返回前 top_n 个有效商品
#     if not keywords:
#         return all_valid_asins[:top_n]
    
#     # 清洗关键词：去停用词、去标点、去空、长度>2
#     keyword_set = set()
#     for k in keywords:
#         k_clean = k.lower().strip().strip(',').strip('.').strip("'").strip('"')
#         if k_clean and k_clean not in _STOPWORDS and len(k_clean) > 2:
#             keyword_set.add(k_clean)
    
#     # 如果清洗后没有有效关键词，兜底返回所有商品
#     if not keyword_set:
#         return all_valid_asins[:top_n]
    
#     # 逐商品匹配
#     scored = []
#     for product in all_products:
#         asin = product.get('asin') or product.get('id')
#         if not asin or asin not in product_item_dict:
#             continue  # 只保留在 product_item_dict 中存在的 ASIN
        
#         name = product.get('name', '') or product.get('Title', '') or ''
#         desc = product.get('description', '') or product.get('Description', '') or ''
#         text = f"{name} {desc}".lower()
        
#         score = 0
#         for kw in keyword_set:
#             if kw in text:
#                 score += 1
#             # 标题匹配权重更高
#             if kw in name.lower():
#                 score += 3
        
#         if score > 0:
#             scored.append((score, asin))
    
#     # 如果关键词匹配结果为空，必须兜底，否则 episode 会卡死
#     if not scored:
#         print(f"[WebShopEnv MockSearch] No keyword match for {keywords}, fallback to all {len(all_valid_asins)} products.")
#         return all_valid_asins[:top_n]
    
#     # 按分数降序，去重，取 top_n
#     scored.sort(key=lambda x: x[0], reverse=True)
#     seen = set()
#     result = []
#     for _, asin in scored:
#         if asin not in seen:
#             seen.add(asin)
#             result.append(asin)
#         if len(result) >= top_n:
#             break
    
#     # 最终兜底：如果去重后结果为空（理论上不会），返回全量
#     if not result:
#         return all_valid_asins[:top_n]
    
#     print(f"[WebShopEnv MockSearch] keywords={keywords} -> matched {len(result)} products.")
#     return result

# # 执行替换
# _engine_module.init_search_engine = _mock_init_search_engine
# _engine_module.get_top_n_product_from_keywords = _mock_get_top_n_product_from_keywords
# print("[WebShopEnv] Lucene/Pyserini mocked with pure-Python search for multi-process safety.")
# # ==================== Monkey-patch end ====================


# 原来的 import 保持不变
# from pyserini.search.lucene import LuceneSearcher
# from web_agent_site.envs import WebAgentTextEnv

from web_agent_site.utils import DEBUG_PROD_SIZE


class WebShopEnv:
    def __init__(self, history_length=4, max_step=15):
        # ====== 延迟导入：确保 JVM 只在 worker 进程内启动 ======
        self.history_length = history_length
        self.max_step = max_step
        self.env = None  ## 延迟初始化
        self.seed = 42
        self._rng = np.random.RandomState(self.seed)
        print("#### Params #####")
        print(f"history length: {self.history_length}")
        print(f"max step: {self.max_step}")

    def _extract_task(self, text_obs: str):
        parts = text_obs.split(" [SEP] ")
        assert parts[1]=='Instruction:'
        task = parts[2]
        return task


    def _format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions


    def _build_prompt(self, observation, init):
        avail_actions = self.env.get_available_actions()

        if not init and self.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                self.history_length,
                obs_key="text_obs",
                action_key="action"
            )

        available_actions = self._format_avail_actions(avail_actions)
        reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

        if init or self.history_length <= 0:
            prompt = WEBSHOP_TEMPLATE_NO_HIS.format(
                task_description=self.task,
                current_observation=observation,
                available_actions=reformatted_available_actions
            )
        else:
            prompt = WEBSHOP_TEMPLATE.format(
                task_description=self.task,
                step_count=len(self.memory[0]),
                history_length=valid_lens[0],
                action_history=memory_contexts[0],
                current_step=len(self.memory[0]) + 1,
                current_observation=observation,
                available_actions=reformatted_available_actions
            )
            if len(prompt) > 8192: ## todo: fixed here!
                    print(f"Warning len(prompt)={len(prompt)} is too long")
                    prompt = WEBSHOP_TEMPLATE_NO_HIS.format(
                        task_description=self.task,
                        current_observation=observation,
                        available_actions=reformatted_available_actions
                    )
        return prompt

    
    def _format_obs(self, observation):
        ## instruction
        parts = observation.split(" [SEP] ")
        try:
            index = parts.index(self.task)
            reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
        except:
            reformatted_obs = observation
        return reformatted_obs


    def _parse_action(self, response):
        original_str = response  # keep the original string
        action = response.lower()

        # Attempt to extract the substring within <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = action.find(start_tag)
        end_idx = action.find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                # If we can't find a valid <action>...</action> block, mark as invalid
                action = action[-20:]
            else:
                # Extract just the content between the tags
                extracted_action = action[start_idx + len(start_tag):end_idx]
                action = extracted_action

        except:
            # randomly choose an action from the action list if illegal
            action = action[-20:]

        return action

    
    def reset(self, session=None, is_train=None):
        #  ====  init environment ====================
        if self.env is None:
            from web_agent_site.envs import WebAgentTextEnv
            self.env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=DEBUG_PROD_SIZE)
        
        # ## goal of dataset
        # goals = self.env.server.goals
        # if is_train:
        #     goal_idxs = range(500, len(goals))
        # else:
        #     goal_idxs = range(500)

        # if session is None:
        #     idx = self._rng.choice(goal_idxs, size=1, replace=False)
        #     idx = idx.tolist()[0]
        #     session = idx

        print("session:", session)
        obs, info = self.env.reset(session=session)

        if info is None:
            info = dict()
        info['available_actions'] = self.env.get_available_actions()
        info['won'] = False

        self.task = self._extract_task(obs)
        format_obs = self._format_obs(obs)
        self.pre_text_obs = format_obs
        obs = self._build_prompt(format_obs, init=True)
        
        ## info anchor
        info["anchor"] = format_obs

        ## 清空恢复数据
        self.step_count = 0
        self.memory = SimpleMemory()
        self.memory.reset(batch_size=1)
        return obs, info


    def step(self, response):
        print(f"[web_env.py]-[step]-response: {response}") 
        self.step_count += 1
        action = self._parse_action(response)
        print(f"[web_env.py]-[step]-action: {action}") 

        result = self.env.step(action)
        if len(result) == 5:
            next_observation, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            next_observation, reward, done, info = result
        
        next_format_obs = self._format_obs(next_observation)
        self.memory.store({"text_obs": [self.pre_text_obs], "action": [action]})
        self.pre_text_obs = next_format_obs
        next_observation = self._build_prompt(next_format_obs, init=False)
        
        ## task score
        if info is None:
            info = dict()
        info["task_score"] = reward
        info["available_actions"] = self.env.get_available_actions()
        info["anchor"] = next_format_obs


        # ## 动作解析失败，直接终止，reward=0，不累积错误
        # if action is None:
        #     print(f"[web_env.py]-[step] 动作解析失败，强制终止 (step={self.step_count})")
        #     obs = self._parse_observation(self.observation) if self.observation else ""
        #     return obs, 0.0, True, info
        # ## ==================================================
        if self.step_count >= self.max_step:
            done = True

        ## 确定选用规则成功率做法
        if done and reward == 1.0:
            info["won"] = True
            reward = 1.0
        else:
            info["won"] = False
            reward = 0.0
        
        return next_observation, reward, done, info


    def close(self):
        self.env.close()
        




if __name__ == "__main__":
    env = WebShopEnv()
    env.reset()
    print(f"#goals: {len(env.env.server.goals)}")
