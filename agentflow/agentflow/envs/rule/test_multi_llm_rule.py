import re
import requests
import pandas as pd
import concurrent.futures
import time
from tqdm import tqdm

from agentflow.envs.rule.rule_env import RuleEnv


class LLMClient:
    def __init__(self, base_url="http://127.0.0.1:19998/v1", model="Qwen3-30B-A3B-Instruct-2507", api_key="EMPTY"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        self.model = model
        self.api_type = self._detect_api_type()
        print(f"[green]LLMClient: model={self.model}, api_type={self.api_type}[/green]")

    def _detect_api_type(self):
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
                    print(f"[green]探测到接口: {endpoint}[/green]")
                    return api_type
            except:
                pass
        print("[red]警告: 无法探测接口，默认使用 chat/completions[/red]")
        return "chat"

    def chat(self, prompt, temperature=0.3, max_tokens=4096):
        if self.api_type == "chat":
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            try:
                resp = requests.post(f"{self.base_url}/chat/completions", headers=self.headers, json=payload, timeout=1200)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"[red]Chat API 失败: {e}[/red]")
                return ""
        else:
            payload = {"model": self.model, "prompt": prompt, "temperature": temperature, "max_tokens": max_tokens}
            try:
                resp = requests.post(f"{self.base_url}/completions", headers=self.headers, json=payload, timeout=1200)
                resp.raise_for_status()
                return resp.json()["choices"][0]["text"]
            except Exception as e:
                print(f"[red]Completion API 失败: {e}[/red]")
                return ""


class LLMPolicy:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def forward(self, obs):
        # print(f"[cyan]{'='*60}[/cyan]")
        response = self.llm_client.chat(obs)
        # print(f"[green]LLM Response:[/green]")
        # print(response)
        # print(f"[green]{'='*60}[/green]")
        return response


def extract_answer(text):
    """从 LLM 响应中提取 <answer> 标签内的内容"""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def process_single_item(item, llm_client):
    """处理单条数据"""
    item_id = item.get("id", "")
    question = item.get("question", "")
    extra_info = item.get("extra_info", {})
    ground_truth = str(extra_info.get("ground_truth", "")).strip()

    policy = LLMPolicy(llm_client=llm_client)

    env = RuleEnv()
    obs, info = env.reset(question=question, answer=ground_truth)
    step = 0
    while True:
        action = policy.forward(obs)
        obs, reward, done, info = env.step(action)
        step += 1
        if done:
            break

    result = {
        "id": item_id,
        "question": question,
        "ground_truth": ground_truth,
        "predicted": extract_answer(action),
        "reward": reward,
        "steps": step,
        "answer": action,
        "source": item.get("source", ""),
        "idx": extra_info.get("idx", -1),
    }
    return result


def load_parquet_data(parquet_path):
    """加载 parquet 文件，返回 list[dict]"""
    df = pd.read_parquet(parquet_path)
    records = df.to_dict(orient="records")
    print(f"[blue]加载完成: {len(records)} 条数据[/blue]")
    return records


def run_concurrent(parquet_path, base_url, model, api_key, max_workers=8, max_tokens=4096):
    """并发运行评测"""
    # 1. 加载数据
    records = load_parquet_data(parquet_path)

    # 2. 初始化共享客户端（每个线程复用同一个 client）
    llm_client = LLMClient(base_url=base_url, model=model, api_key=api_key)
    results = []

    # 3. 并发执行
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_item, item, llm_client): item 
            for item in records
        }

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(records), desc="Processing"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                item = futures[future]
                print(f"[red]处理失败 id={item.get('id')}: {e}[/red]")
                results.append({
                    "id": item.get("id", ""),
                    "question": item.get("question", ""),
                    "ground_truth": str(item.get("extra_info", {}).get("ground_truth", "")),
                    "predicted": "",
                    "reward": 0.0,
                    "steps": 0,
                    "error": str(e),
                })

    # 4. 统计结果
    total = len(results)
    correct = sum(1 for r in results if r.get("reward", 0) > 0)
    accuracy = correct / total if total > 0 else 0

    print(f"\n[bold green]评测完成! 总计: {total}, 正确: {correct}, 准确率: {accuracy:.2%}[/bold green]")

    return results


if __name__ == '__main__':
    PARQUET_PATH = "/home/ma-user/work/tangzhentao/code-rl/agent-rl-train/AI4EDA/AgentFlow-distributed-GRPO/data/val/aime24_only_10.parquet"  # 请替换为实际路径

    results = run_concurrent(
        parquet_path=PARQUET_PATH,
        base_url="http://127.0.0.1:19998/v1",
        model="Qwen3-30B-A3B-Instruct-2507",
        api_key="EMPTY",
        max_workers=8,       # 并发数，根据服务端承载能力调整
        max_tokens=4096,     # 每次推理最大 token 数
    )

    # 保存结果（可选）
    import json
    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("[blue]结果已保存到 eval_results.json[/blue]")