import yaml
import os
os.environ["AGENTOPS_API_KEY"] = ""  # 清空 API key 禁用监控
os.environ["AGENTOPS_AUTO_INIT"] = "false"
import re
import json
from pydantic import BaseModel
from agentflow.engine.openai import ChatOpenAI



def load_and_set_env_from_yaml(config_file_path: str):
    # --- Parse YAML configuration ---
    print(f"Parsing YAML configuration from '{config_file_path}'...")
    try:
        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at '{config_file_path}'")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)

    # --- Set Environment Variables ---
    if 'env' in config:
        print("Setting environment variables...")
        for key, value in config['env'].items():
            # Use os.environ to set variables.
            # Convert non-string values to string.
            str_val = str(value)            
            start_prefix = "${env:"
            if str_val.startswith(start_prefix) and "," in str_val:
                env_variable = str_val.split(",")[0][len(start_prefix):]
                env_default = str_val.split(",")[1][:-len("}")]
                org_val = str_val
                if env_variable in os.environ:
                    str_val = os.environ[env_variable]
                    print(f"  Exported {key}={org_val} -> {str_val}")
                else:
                    str_val = env_default
                    print(f"  Exported {key}={org_val}")
            else:
                print(f"  Exported {key}={str_val}")


            os.environ[key] = str_val
            config['env'][key] = str_val
    return config
            


## config initialization
file_path = os.path.dirname(os.path.abspath(__file__))
config_file_path = f'{file_path}/config.yaml'
load_and_set_env_from_yaml(config_file_path)

try:
    # llm_scorer_engine = ChatOpenAI(
    #     model_string=AI_MODEL,   # (o:"gpt-4o")
    #     is_multimodal=False, 
    #     enable_cache=True
    # )

    ## local setting
    LOCAL_MODEL = os.environ["SCORE_MODEL_NAME"]
    LOCAL_BASE_URL = os.environ["SCORE_MODEL_URL"]
    LOCAL_API_KEY = "NONE"

    print("LOCAL_MODEL:", LOCAL_MODEL)
    print("LOCAL_BASE_URL:", LOCAL_BASE_URL)

    llm_scorer_engine = ChatOpenAI(
        model_string=LOCAL_MODEL,
        base_url=LOCAL_BASE_URL,
        api_key=LOCAL_API_KEY, 
        is_multimodal=False, 
        use_cache=False
    )
    
    print(f"\nLLM Scorer engine '{llm_scorer_engine.model_string}' initialized successfully.\n")
except Exception as e:
    print(f"Failed to initialize LLM Scorer engine: {e}")
    llm_scorer_engine = None



    



def _try_load(s: str):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None

def extract_json_from_markdown(text: str) -> dict:
    """从 markdown 代码块或混合格式文本中提取 JSON"""
    if not text:
        return None

    # 1. 提取 Markdown 代码块
    pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(pattern, text)

    if match:
        json_str = match.group(1).strip()
    else:
        json_str = text.strip()
    
    if not json_str:
        return None

    # 2. 快速拒绝：完全没有 JSON 结构特征
    if not re.search(r'[\{\[]', json_str):
        return None

    # 3. 尝试直接解析
    result = _try_load(json_str)
    if result is not None:
        return result

    # 4. 处理 "前缀文字 + JSON" 的混合格式
    if not match:
        # 贪婪匹配：假设只有一个顶层 JSON 对象/数组
        obj_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', json_str)
        if obj_match:
            candidate = obj_match.group(1).strip()
            result = _try_load(candidate)
            if result is not None:
                return result

    # 5. 修复非法转义序列
    try:
        # 先保护 \uXXXX
        protected = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: f'__U{m.group(1)}__', json_str)
        
        legal_escapes = ['\\\\', '\\"', '\\/', '\\b', '\\f', '\\n', '\\r', '\\t']
        placeholders = [f'__ESC{i}__' for i in range(len(legal_escapes))]
        
        for esc, ph in zip(legal_escapes, placeholders):
            protected = protected.replace(esc, ph)
        
        # 转义剩余的裸反斜杠
        protected = protected.replace('\\', '\\\\')
        
        # 恢复合法转义
        for ph, esc in zip(placeholders, legal_escapes):
            protected = protected.replace(ph, esc)
        
        # 恢复 \uXXXX
        protected = re.sub(r'__U([0-9a-fA-F]{4})__', r'\\u\1', protected)
        
        result = _try_load(protected)
        if result is not None:
            return result
    except Exception:
        pass

    # 6. 最终尝试：移除所有非法转义，保留字符本身
    try:
        cleaned = re.sub(r'\\([^"\\/bfnrtu])', r'\1', json_str)
        return json.loads(cleaned)
    except Exception:
        return None





class AnswerVerification(BaseModel):
    analysis: str
    true_false: bool



def compute_score(question: str,  groundtruth: str, answer_extracted: str,) -> bool:
    """
    Uses gpt-4o to determine if the extracted answer matches the groundtruth.
    
    Args:
        question: The full question text, including options.
        answer_extracted: The answer provided by the model being evaluated.
        groundtruth: The correct answer label (e.g., "A").

    Returns:
        A boolean indicating whether the answer is correct.
    """
    if llm_scorer_engine is None:
        raise RuntimeError("LLM Scorer engine is not available.")
        
    
    ## chat-model prompt
    query_prompt = f"""
You are a precise evaluator. Determine if the Model Response is equivalent to the Ground Truth.

**Instructions:**
1.  **Extract:** Isolate the final answer from the Model Response, ignoring reasoning. Look for `\boxed{{...}}` or concluding statements.
2.  **Normalize & Compare:** The extracted answer and Ground Truth must be equivalent after normalization:
    - **Math:** Mathematically identical (e.g., `\\frac{{1}}{{2}}` == `0.5`).
    - **Numbers/Text:** Ignore formatting, case, and currency/units (e.g., `1,000` == `1000`).
    - **MCQ:** Match option content (e.g., "Paris") or number (e.g., `3rd` option) to the correct letter.
3.  **Verdict:** "True" only for semantically or mathematically equivalent answers.

**Inputs:**
Question: {question}
Model Response: {answer_extracted}
Ground Truth: {groundtruth}

**Output Format Requirements (STRICT):**
You MUST respond with a valid JSON object and NOTHING else. Do not include markdown code blocks, explanations, or any text outside the JSON.

Required JSON structure:
```json
{{
    "analysis": "Brief analysis of the comparison",
    "true_false": true
}}
"""


# #     ## GPT-4o prompt
#     query_prompt = f"""
# You are a precise evaluator. Determine if the Model Response is equivalent to the Ground Truth.

# **Instructions:**
# 1.  **Extract:** Isolate the final answer from the Model Response, ignoring reasoning. Look for `\boxed{{...}}` or concluding statements.
# 2.  **Normalize & Compare:** The extracted answer and Ground Truth must be equivalent after normalization:
#     - **Math:** Mathematically identical (e.g., `\\frac{{1}}{{2}}` == `0.5`).
#     - **Numbers/Text:** Ignore formatting, case, and currency/units (e.g., `1,000` == `1000`).
#     - **MCQ:** Match option content (e.g., "Paris") or number (e.g., `3rd` option) to the correct letter.
# 3.  **Verdict:** "True" only for semantically or mathematically equivalent answers.

# **Inputs:**
# Question: {question}
# Model Response: {answer_extracted}
# Ground Truth: {groundtruth}

# **Format:**
# <analysis>: Brief analysis of the comparison.
# <true_false>: "True" or "False".
# """

    verification_result = llm_scorer_engine(query_prompt, response_format=AnswerVerification)

    if hasattr(verification_result, "true_false"):
        return verification_result.true_false
    else:
        print("verification_result:", verification_result)
        ## judger output parsing
        judger_response = extract_json_from_markdown(verification_result)

        ## failed to load then it is forced to False.
        if judger_response is None:
            return False
        
        if "true_false" in judger_response.keys():
            final_response = judger_response["true_false"]
            # print("final_response:", final_response)
            return final_response

        return False
        
    



def eval(question: str, groundtruth: any, answer_extracted: any, val: bool = False) -> float:
    """
    Evaluates if the extracted answer is correct by calling an LLM judge (gpt-4o).
    It strip(), and matches the final answer.
    """
    question_str = str(question)
    groundtruth_str = str(groundtruth)
    answer_extracted_str = str(answer_extracted)

    is_correct = compute_score(question_str, answer_extracted_str, groundtruth_str)
    
    return 1.0 if is_correct else 0.0

async def main():
    # ==============================================================================
    # ==============================================================================
    print("--- Running Simple Case ---")
    simple_question = "What is the capital of France?\nA) Berlin\nB) Madrid\nC) Paris\nD) Rome"
    simple_groundtruth = "C"
    simple_model_answer = "The correct answer is C."
    score1 = eval(simple_question, simple_groundtruth, simple_model_answer)
    print(f"Question: {simple_question}")
    print(f"Model Answer: '{simple_model_answer}'")
    print(f"Ground Truth: '{simple_groundtruth}'")
    print(f"==> Score: {score1}\n") # 1.0

    # ==============================================================================
    # ==============================================================================
    print("--- Running Case with LaTeX Formula ---")
    latex_question = r"""
Calculate the definite integral of $f(x) = 2x$ from $x=1$ to $x=3$.
A) 4
B) 6
C) 8
D) 10
"""
    latex_groundtruth = "C"
    latex_model_answer = r"""
To solve this, we need to compute the integral $\int_{1}^{3} 2x \,dx$.
The antiderivative of $2x$ is $x^2$. 
Using the Fundamental Theorem of Calculus, we evaluate this at the bounds:
$F(b) - F(a) = 3^2 - 1^2 = 9 - 1 = 8$.
"""
    score2 = eval(latex_question, latex_groundtruth, latex_model_answer)
    print(f"Question: {latex_question.strip()}")
    print(f"Model Answer: '{latex_model_answer.strip()}'")
    print(f"Ground Truth: '{latex_groundtruth}'")
    print(f"==> Score: {score2}\n") # 1.0

    # ==============================================================================
    # ==============================================================================
    print("--- Running Case with Multiple Intermediate Answers ---")
    multi_answer_question = """
A project has two phases. Phase 1 costs $5,000 and takes 3 months. Phase 2 costs $8,000 and takes 4 months. What is the total duration of the project?
A) $13,000
B) 4 months
C) 7 months
D) $5,000
"""
    multi_answer_groundtruth = "C"
    multi_answer_model_response = """
Let's analyze the problem.
The cost of Phase 1 is $5,000 and the duration is 3 months.
The cost of Phase 2 is $8,000 and the duration is 4 months.
The total cost would be $5,000 + $8,000 = $13,000.
The question asks for the total duration, which is 3 months + 4 months = 7 months.
Therefore, the final answer is 7 months. This matches option C.
"""
    score3 = eval(multi_answer_question, multi_answer_groundtruth, multi_answer_model_response)
    print(f"Question: {multi_answer_question.strip()}")
    print(f"Model Answer: '{multi_answer_model_response.strip()}'")
    print(f"Ground Truth: '{multi_answer_groundtruth}'")
    print(f"==> Score: {score3}\n") # 1.0


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())