import os
import re

class RuleEnv:
    def __init__(self):
        # ====== 延迟导入：确保 JVM 只在 worker 进程内启动 ======
        self.sufix_prompt = "When ready, output the final answer enclosed in <answer> and </answer> tags. Do not generate any content after the </answer> tag"

    def _parse_action(self, response):
        if not response:
            print("[**ERROR**] LLM 返回空响应，强制终止")
            return None

        # 清理响应：去除可能的换行符和多余空格
        action = response.strip()
        
        # 去除可能的引号
        if action.startswith('"') and action.endswith('"'):
            action = action[1:-1]
        if action.startswith("'") and action.endswith("'"):
            action = action[1:-1]

        end_idx = action.rfind("</answer>")
        if end_idx == -1:
            action = None
        else:
            # 在 </answer> 之前找最后一个 <answer>
            start_idx = action.rfind('<answer>', 0, end_idx)
            if start_idx == -1:
                action = None
            else:
                action = action[start_idx + len('<answer>'):end_idx].strip()

        print(f"[**DEBUG**] 调试: LLM 提取到 action={action}, type:{type(action)}")
        return action
            

    
    def reset(self, question: str, answer: str):
        # ======================================
        self.problem = question
        self.answer = answer
        info = dict()
        info["anchor"] = question

        if not self.sufix_prompt in self.problem:
            question = self.problem + "\n\n" + self.sufix_prompt

        return question, info


    def step(self, response):
        action = self._parse_action(response)

        reward = 0.0
        if action is not None:
            # 统一转为字符串并规范化
            action_str = str(action).strip()
            answer_str = str(self.answer).strip()
            
            # 可选：进一步规范化（去多余空格、统一大小写）
            # action_str = re.sub(r'\s+', ' ', action_str).lower()
            # answer_str = re.sub(r'\s+', ' ', answer_str).lower()
            
            if action_str == answer_str:
                reward = 1.0
       
        next_observation = None
        done = True
        info = dict()
        info["anchor"] = None
        print(f"action:{action} | answer:{self.answer} | reward:{reward} | ")
        
        return next_observation, reward, done, info


        





