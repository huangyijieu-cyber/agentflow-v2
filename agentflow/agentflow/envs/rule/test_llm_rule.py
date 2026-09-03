import re
import requests

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
                resp = requests.post(f"{self.base_url}/chat/completions", headers=self.headers, json=payload, timeout=120)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"[red]Chat API 失败: {e}[/red]")
                return ""
        else:
            payload = {"model": self.model, "prompt": prompt, "temperature": temperature, "max_tokens": max_tokens}
            try:
                resp = requests.post(f"{self.base_url}/completions", headers=self.headers, json=payload, timeout=120)
                resp.raise_for_status()
                return resp.json()["choices"][0]["text"]
            except Exception as e:
                print(f"[red]Completion API 失败: {e}[/red]")
                return ""


class LLMPolicy:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def forward(self, obs):
        print(f"[cyan]{'='*60}[/cyan]")

        response = self.llm_client.chat(obs)
        print(f"[green]LLM Response:[/green]")
        print(response)
        print(f"[green]{'='*60}[/green]")
        return response


if __name__ == '__main__':
    llm_client = LLMClient(
        base_url="http://127.0.0.1:19998/v1",
        model="Qwen3-30B-A3B-Instruct-2507",
        api_key="EMPTY"
    )

    env = RuleEnv()
    
    question = """Which film has the director who is older than the other, Let'S Make Money or Short Term 12?"""
    answer = "Let'S Make Money"


    total_reward = list()
    total_observation = list()

    for question_id in range(1):
        obs, info = env.reset(question=question, answer=answer)
        print(f"[blue]Env type: {type(env)}[/blue]")
        
        policy = LLMPolicy(llm_client=llm_client)
        step = 0
        total_observation.append(obs)
        while True:
            print("1-Observation:", obs)
            action = policy.forward(obs)
            print(f'2-Response: "{action}"')
            obs, reward, done, info = env.step(action)
            print(f'3-Reward = {reward}, Done = {done}')

            if done:
                break
        total_reward.append(reward)

    print(f"[bold green]Task finished! Total reward: {total_reward}[/bold green]")
    print(f"total observation:", len(total_observation))
    print(f"set(total):", len(set(total_observation)))
            
