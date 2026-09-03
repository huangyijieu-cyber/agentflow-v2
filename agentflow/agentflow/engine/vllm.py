# Reference: https://github.com/zou-group/textgrad/blob/main/textgrad/engine/openai.py

try:
    import vllm
except ImportError:
    raise ImportError("If you'd like to use VLLM models, please install the vllm package by running `pip install vllm`.")

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("If you'd like to use VLLM models, please install the openai package by running `pip install openai`.")

import os
import json
import base64
import platformdirs
from typing import List, Union
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)
from .base import EngineLM, CachedEngine
from openai import APIConnectionError, APITimeoutError

class ChatVLLM(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string="Qwen/Qwen2.5-VL-3B-Instruct",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool=False,
        use_cache: bool=True,
        base_url=None,
        api_key=None,
        check_model: bool=True,
        **kwargs):
        """
        :param model_string:
        :param system_prompt:
        :param is_multimodal:
        """

        self.model_string = model_string
        self.use_cache = use_cache
        self.system_prompt = system_prompt
        self.is_multimodal = is_multimodal

        if self.use_cache:
            root = platformdirs.user_cache_dir("agentflow")
            cache_path = os.path.join(root, f"cache_vllm_{self.model_string}.db")
            self.image_cache_dir = os.path.join(root, "image_cache")
            os.makedirs(self.image_cache_dir, exist_ok=True)
            super().__init__(cache_path=cache_path)
        
        self.base_url = base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        self.api_key = api_key or os.environ.get("VLLM_API_KEY", "dummy-token")

        try:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=1200.0
            )
        except Exception as e:
            raise ValueError(f"Failed to connect to VLLM server at {self.base_url}. Please ensure the server is running and try again.")

    
    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt=None, **kwargs):
        try:
            if isinstance(content, str):
                return self._generate_text(content, system_prompt=system_prompt, **kwargs)
            
            elif isinstance(content, list):
                if all(isinstance(item, str) for item in content):
                    full_text = "\n".join(content)
                    return self._generate_text(full_text, system_prompt=system_prompt, **kwargs)

                elif any(isinstance(item, bytes) for item in content):
                    if not self.is_multimodal:
                        raise NotImplementedError(
                            f"Multimodal generation is only supported for {self.model_string}. "
                            "Consider using a multimodal model like 'gpt-4o'."
                        )
                    return self._generate_multimodal(content, system_prompt=system_prompt, **kwargs)

                else:
                    raise ValueError("Unsupported content in list: only str or bytes are allowed.")
        except (APIConnectionError, APITimeoutError) as e:
            # 3 次重试后还是连不上 / 超时，打印日志，返回空字符串
            print(f"Traceback [ChatVLLM] Serving unreachable after retries: {e}")
            return ""  # ← 调用方拿到 str，不会 TypeError
                
        except Exception as e:
            print(f"Error in generate method: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {e.args}")
            import traceback
            print(f"Traceback [ChatVLLM]:")
            traceback.print_exc()
            return ""
            # return {
            #     "error": type(e).__name__,
            #     "message": str(e),
            #     "details": getattr(e, 'args', None)
            # }
    
    @retry(
            wait=wait_random_exponential(min=1, max=10), 
            stop=stop_after_attempt(3), 
            retry=retry_if_exception_type((APIConnectionError, APITimeoutError)),
            reraise=True,  # 3 次还失败就把异常抛上去
    )
    def _generate_text(
        self, prompt, system_prompt=None, max_tokens=2048, top_p=0.99, response_format=None, **kwargs
    ):

        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        if self.use_cache:
            cache_key = sys_prompt_arg + prompt
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none
        
        ## setting
        # temperature = kwargs.get("temperature", 0.7)
        # if temperature == 0.0:
        #     top_p = 1.0


        # ## Qwen-Family Parameters (推理可以开这个，但训练开这个容易报错top_k和top_p超出限制，尤其是top_k需要调成-1，top_p为1.0)
        # temperature = 0.7
        # top_p = 0.8
        # top_k = 20
        # min_p = 0
        # presence_penalty = 0


        ## fixed parameters
        temperature = 0.7
        top_p = 1.0
        presence_penalty = 0
        print(f"post by _generate_text, temperature: {temperature}")

        
        if response_format is not None:
            response_format_arg = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "strict": True,
                    "schema": response_format.model_json_schema(),
                },
            }
        else:
            response_format_arg = None
        # ## Chat models without structured outputs (without stream)
        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=[
                {"role": "system", "content": sys_prompt_arg},
                {"role": "user", "content": prompt},
            ],
            # frequency_penalty=kwargs.get("frequency_penalty", 1.2),
            # stop=None,
            presence_penalty=presence_penalty,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            # top_k=top_k,  ## 都不存在入参
            # min_p=min_p,  ## 都不存在入参
            stream=False,
            # extra_body={
            #     "top_k": top_k,
            #     "min_p": min_p
            # }
            response_format=response_format_arg,
        )
        response = response.choices[0].message.content
        
        

        # # 提交为流式
        # full_response = ""
        # response = self.client.chat.completions.create(
        #     model=self.model_string,
        #     messages=[
        #         {"role": "system", "content": sys_prompt_arg},
        #         {"role": "user", "content": prompt},
        #     ],
        #     temperature=temperature,
        #     max_tokens=max_tokens,
        #     top_p=top_p,
        #     stream=True,
        # )

        # for chunk in response:
        #     if chunk.choices[0].delta.content is not None:
        #         content = chunk.choices[0].delta.content
        #         full_response += content
        # response = full_response



        if self.use_cache:
            self._save_cache(cache_key, response)
        return response

    def __call__(self, prompt, **kwargs):
        print(f"call vllm, kwargs:{kwargs}")
        return self.generate(prompt, **kwargs)

    def _format_content(self, content: List[Union[str, bytes]]) -> List[dict]:
        formatted_content = []
        for item in content:
            if isinstance(item, bytes):
                base64_image = base64.b64encode(item).decode('utf-8')
                formatted_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
            elif isinstance(item, str):
                formatted_content.append({
                    "type": "text",
                    "text": item
                })
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")
        return formatted_content

    def _generate_multimodal(
        self, content: List[Union[str, bytes]], system_prompt=None, temperature=0, max_tokens=2048, top_p=0.99, response_format=None
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        formatted_content = self._format_content(content)

        if self.use_cache:
            cache_key = sys_prompt_arg + json.dumps(formatted_content)
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none

        ## 提交为非流式
        print("post by _generate_multimodal")
        full_response = ""
        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=[
                {"role": "system", "content": sys_prompt_arg},
                {"role": "user", "content": formatted_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=False,
        )
        response_text = response.choices[0].message.content

        ## 提交为流式
        # full_response = ""
        # response = self.client.chat.completions.create(
        #     model=self.model_string,
        #     messages=[
        #         {"role": "system", "content": sys_prompt_arg},
        #         {"role": "user", "content": formatted_content},
        #     ],
        #     temperature=temperature,
        #     max_tokens=max_tokens,
        #     top_p=top_p,
        #     stream=True,
        # )
        # response_text = response.choices[0].message.content

        # for chunk in response:
        #     if chunk.choices[0].delta.content is not None:
        #         content = chunk.choices[0].delta.content
        #         full_response += content
        # response_text = full_response

        if self.use_cache:
            self._save_cache(cache_key, response_text)
        return response_text
