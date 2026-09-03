import multiprocessing as mp
import time
import traceback
import sys
import os

# ============ 配置区 ============
BASE_URL = "http://127.0.0.1:19998/v1"
MODEL = "Qwen3-30B-A3B-Instruct-2507"
API_KEY = "EMPTY"
NUM_WORKERS = 10          # 并发 Worker 数
MAX_STEPS_PER_EP = 15    # 每个 episode 最大步数
STEP_TIMEOUT = 120       # 单步最大等待时间（秒）
TOTAL_TIMEOUT = 600      # 单个 Worker 总超时（秒）

# 强制使用 spawn 启动，避免 fork 带来的共享状态/文件描述符问题
mp.set_start_method("spawn", force=True)


class LLMClient:
    def __init__(self, base_url="http://127.0.0.1:19998/v1", model="Qwen3-30B-A3B-Instruct-2507", api_key="EMPTY"):
        import requests
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        self.model = model
        self.api_type = self._detect_api_type()
        print(f"[Worker-{os.getpid()}] LLMClient: model={self.model}, api_type={self.api_type}")

    def _detect_api_type(self):
        import requests
        for api_type, endpoint in [("chat", "/chat/completions"), ("completion", "/completions")]:
            try:
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 5
                } if api_type == "chat" else {
                    "model": self.model, "prompt": "hi", "max_tokens": 5
                }
                r = requests.post(f"{self.base_url}{endpoint}", headers=self.headers, json=payload, timeout=10)
                if r.status_code == 200:
                    return api_type
            except Exception:
                pass
        print(f"[Worker-{os.getpid()}] 警告: 无法探测接口，默认使用 chat/completions")
        return "chat"

    def chat(self, prompt, temperature=0.3, max_tokens=1024):
        import requests
        if self.api_type == "chat":
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            try:
                resp = requests.post(f"{self.base_url}/chat/completions", headers=self.headers, json=payload, timeout=STEP_TIMEOUT)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"[Worker-{os.getpid()}] Chat API 失败: {e}")
                return ""
        else:
            payload = {"model": self.model, "prompt": prompt, "temperature": temperature, "max_tokens": max_tokens}
            try:
                resp = requests.post(f"{self.base_url}/completions", headers=self.headers, json=payload, timeout=STEP_TIMEOUT)
                resp.raise_for_status()
                return resp.json()["choices"][0]["text"]
            except Exception as e:
                print(f"[Worker-{os.getpid()}] Completion API 失败: {e}")
                return ""


class LLMPolicy:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def forward(self, obs):
        response = self.llm_client.chat(obs)
        return response


def run_episode(worker_id):
    """单个 Worker 运行一个完整 episode"""
    pid = os.getpid()
    print(f"\n[Worker-{worker_id}] PID={pid} 启动")
    
    # 每个 Worker 独立创建 Env 和 LLMClient
    try:
        from agentflow.envs.webshop.webshop_env import WebShopEnv
        env = WebShopEnv()
    except Exception as e:
        print(f"[Worker-{worker_id}] Env 创建失败: {e}")
        traceback.print_exc()
        return {"worker_id": worker_id, "status": "env_error", "error": str(e)}
    
    llm_client = LLMClient(base_url=BASE_URL, model=MODEL, api_key=API_KEY)
    policy = LLMPolicy(llm_client=llm_client)
    
    try:
        obs, info = env.reset()
        print(f"[Worker-{worker_id}] Env reset 成功")
    except Exception as e:
        print(f"[Worker-{worker_id}] Env reset 失败: {e}")
        traceback.print_exc()
        env.close()
        return {"worker_id": worker_id, "status": "reset_error", "error": str(e)}
    
    step = 0
    history = []
    reward = 0
    done = False
    
    try:
        while step < MAX_STEPS_PER_EP:
            step += 1
            print(f"[Worker-{worker_id}] ====== Step {step} ======")
            
            # 记录单步开始时间
            t_start = time.time()
            
            # LLM 生成 action
            action = policy.forward(obs)
            t_action = time.time()
            print(f"[Worker-{worker_id}] Action 生成耗时: {t_action - t_start:.2f}s")
            
            # Env step
            obs, reward, done, info = env.step(action)
            t_step = time.time()
            print(f"[Worker-{worker_id}] Env step 耗时: {t_step - t_action:.2f}s, reward={reward}, done={done}")
            
            history.append({
                "step": step,
                "action": action[:200],  # 截断避免日志过长
                "reward": reward,
                "done": done
            })
            
            if done:
                print(f"[Worker-{worker_id}] Episode 结束，Final reward={reward}")
                break
                
    except Exception as e:
        print(f"[Worker-{worker_id}] Step {step} 运行时异常: {e}")
        traceback.print_exc()
        env.close()
        return {
            "worker_id": worker_id,
            "status": "step_error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "steps": step,
            "history": history
        }
    
    finally:
        try:
            env.close()
            print(f"[Worker-{worker_id}] Env 已关闭")
        except Exception as e:
            print(f"[Worker-{worker_id}] Env close 异常: {e}")
    
    return {
        "worker_id": worker_id,
        "status": "success",
        "steps": step,
        "reward": reward,
        "done": done,
        "history": history
    }


if __name__ == '__main__':
    print(f"主进程 PID={os.getpid()}，启动 {NUM_WORKERS} 个 Worker...")
    
    # 使用 Pool 并发
    with mp.Pool(processes=NUM_WORKERS) as pool:
        # 异步提交所有任务
        async_results = [
            pool.apply_async(run_episode, (wid,)) 
            for wid in range(NUM_WORKERS)
        ]
        
        # 收集结果，带总超时
        results = []
        for i, res in enumerate(async_results):
            try:
                result = res.get(timeout=TOTAL_TIMEOUT)
                results.append(result)
                print(f"\n[Main] Worker-{i} 完成，status={result['status']}")
            except Exception as e:
                print(f"\n[Main] Worker-{i} 超时或异常: {e}")
                results.append({
                    "worker_id": i,
                    "status": "timeout",
                    "error": str(e)
                })
    
    # ============ 结果汇总 ============
    print("\n" + "="*60)
    print("汇总结果:")
    success_count = 0
    for r in results:
        wid = r["worker_id"]
        status = r["status"]
        if status == "success":
            success_count += 1
            print(f"  Worker-{wid}: ✅ 成功 | steps={r['steps']} | reward={r['reward']} | done={r['done']}")
        else:
            print(f"  Worker-{wid}: ❌ 失败 | status={status} | error={r.get('error', 'N/A')[:100]}")
    
    print(f"\n成功率: {success_count}/{NUM_WORKERS}")
    print("="*60)