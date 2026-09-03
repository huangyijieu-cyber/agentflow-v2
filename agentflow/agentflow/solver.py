import argparse
import time
import json
from datetime import datetime
from typing import Optional
import os

from copy import deepcopy

from agentflow.models.initializer import Initializer
from agentflow.models.planner import Planner
from agentflow.models.verifier import Verifier
from agentflow.models.memory import Memory
from agentflow.models.executor import Executor
from agentflow.models.utils import make_json_serializable_truncated


def log_info(message):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}")
    

class Solver:
    def __init__(
        self,
        planner,
        verifier,
        memory,
        executor,
        output_types: str = "base,final,direct",
        max_steps: int = 10,
        max_time: int = 300,
        max_tokens: int = 4000,
        root_cache_dir: str = "cache",
        verbose: bool = True,
        temperature: float = .0,
        task: str = "qa"
    ):
        self.planner = planner
        self.verifier = verifier
        self.memory = memory
        self.executor = executor
        self.max_steps = max_steps
        self.max_time = max_time
        self.max_tokens = max_tokens
        self.root_cache_dir = root_cache_dir

        self.output_types = output_types.lower().split(',')
        self.temperature  = temperature
        assert all(output_type in ["base", "final", "direct"] for output_type in self.output_types), "Invalid output type. Supported types are 'base', 'final', 'direct'."
        self.verbose = verbose
        self.task = task

        if task == "webshop":
            from agentflow.envs.webshop.webshop_env import WebShopEnv
            self.env = WebShopEnv()
        elif task == "alfworld":
            from agentflow.envs.alfworld.alfworld_env import AlfWorldEnv
            self.env = AlfWorldEnv()
        elif task == "rule":
            from agentflow.envs.rule.rule_env import RuleEnv
            self.env = RuleEnv()
        else:
            self.env = None
            
    

    def solve(self, question: str, image_path: Optional[str] = None, ground_truth: Optional[str] = None):
        """
        Solve a single problem from the benchmark dataset.
        
        Args:
            index (int): Index of the problem to solve
        """
        reward = None ## placeholder

        if self.task == "qa":
            # Update cache directory for the executor
            self.executor.set_query_cache_dir(self.root_cache_dir)

            # Initialize json_data with basic problem information
            json_data = {
                "query": question,
                "image": image_path,
                "anchor": None,
            }
            if self.verbose:
                log_info(f"\n==> 🔍 Received Query: {question}")
                if image_path:
                    log_info(f"\n==> 🖼️ Received Image: {image_path}")

            # Generate base response if requested
            if 'base' in self.output_types:
                base_response = self.planner.generate_base_response(question, image_path, self.max_tokens)
                json_data["base_response"] = base_response
                if self.verbose:
                    log_info(f"\n==> 📝 Base Response from LLM:\n{base_response}")

            # If only base response is needed, save and return
            if set(self.output_types) == {'base'}:
                ## update logs
                json_data.update({"planner_logs": deepcopy(self.planner.logs)})
                self.planner.logs.clear()
                return json_data, reward
        
            # Continue with query analysis and tool execution if final or direct responses are needed
            if {'final', 'direct'} & set(self.output_types):
                if self.verbose:
                    log_info(f"\n==> 🐙 Reasoning Steps from AgentFlow (Deep Thinking...)")

                # [1] Analyze query
                query_start_time = time.time()
                query_analysis = self.planner.analyze_query(question, image_path)
                json_data["query_analysis"] = query_analysis
                if self.verbose:
                    log_info(f"\n==> 🔍 Step 0: Query Analysis\n")
                    log_info(f"{query_analysis}")
                    log_info(f"[Time]: {round(time.time() - query_start_time, 2)}s")

                # Main execution loop
                step_count = 0
                action_times = []
                while step_count < self.max_steps and (time.time() - query_start_time) < self.max_time:
                    step_count += 1
                    step_start_time = time.time()

                    # [2] Generate next step
                    local_start_time = time.time()
                    next_step = self.planner.generate_next_step(
                        question, 
                        image_path, 
                        query_analysis, 
                        self.memory, 
                        step_count, 
                        self.max_steps,
                        json_data
                    )
                    context, sub_goal, tool_name = self.planner.extract_context_subgoal_and_tool(next_step)
                    if self.verbose:
                        log_info(f"\n==> 🎯 Step {step_count}: Action Prediction ({tool_name})\n")
                        log_info(f"[Context]: {context}\n[Sub Goal]: {sub_goal}\n[Tool]: {tool_name}")
                        log_info(f"[Time]: {round(time.time() - local_start_time, 2)}s")

                    if tool_name is None or tool_name not in self.planner.available_tools:
                        log_info(f"\n==> 🚫 Error: Tool '{tool_name}' is not available or not found.")
                        command = "No command was generated because the tool was not found."
                        result = "No result was generated because the tool was not found."

                    else:
                        # [3] Generate the tool command
                        local_start_time = time.time()
                        tool_command = self.executor.generate_tool_command(
                            question, 
                            image_path, 
                            context, 
                            sub_goal, 
                            tool_name, 
                            self.planner.toolbox_metadata[tool_name],
                            step_count,
                            json_data
                        )
                        analysis, explanation, command = self.executor.extract_explanation_and_command(tool_command)
                        if self.verbose:
                            log_info(f"\n==> 📝 Step {step_count}: Command Generation ({tool_name})\n")
                            log_info(f"[Analysis]: {analysis}\n[Explanation]: {explanation}\n[Command]: {command}")
                            log_info(f"[Time]: {round(time.time() - local_start_time, 2)}s")
                        
                        # [4] Execute the tool command
                        local_start_time = time.time()
                        result = self.executor.execute_tool_command(tool_name, command)
                        result = make_json_serializable_truncated(result) # Convert to JSON serializable format
                        json_data[f"tool_result_{step_count}"] = result

                        if self.verbose:
                            log_info(f"\n==> 🛠️ Step {step_count}: Command Execution ({tool_name})\n")
                            log_info(f"[Result]:\n{json.dumps(result, indent=4)}")
                            log_info(f"[Time]: {round(time.time() - local_start_time, 2)}s")
                    
                    # Track execution time for the current step
                    execution_time_step = round(time.time() - step_start_time, 2)
                    action_times.append(execution_time_step)

                    # Update memory
                    self.memory.add_action(step_count, tool_name, sub_goal, command, result)
                    memory_actions = self.memory.get_all_actions()

                    # [5] Verify memory (context verification)
                    local_start_time = time.time()
                    stop_verification = self.verifier.verificate_context(
                        question,
                        image_path,
                        query_analysis,
                        self.memory,
                        step_count,
                        json_data
                    )
                    context_verification, conclusion = self.verifier.extract_conclusion(stop_verification)
                    if self.verbose:
                        conclusion_emoji = "✅" if conclusion == 'STOP' else "🛑"
                        log_info(f"\n==> 🤖 Step {step_count}: Context Verification\n")
                        log_info(f"[Analysis]: {context_verification}\n[Conclusion]: {conclusion} {conclusion_emoji}")
                        log_info(f"[Time]: {round(time.time() - local_start_time, 2)}s")
                    
                    # Break the loop if the context is verified
                    if conclusion == 'STOP':
                        break

                # Add memory and statistics to json_data
                json_data.update({
                    "memory": memory_actions,
                    "step_count": step_count,
                    "execution_time": round(time.time() - query_start_time, 2),
                })

                # Generate final output if requested
                if 'final' in self.output_types:
                    final_output = self.planner.generate_final_output(question, image_path, self.memory)
                    json_data["final_output"] = final_output
                    log_info(f"\n==> 🐙 Detailed Solution:\n\n{final_output}")

                # Generate direct output if requested
                if 'direct' in self.output_types:
                    direct_output = self.planner.generate_direct_output(question, image_path, self.memory)
                    json_data["direct_output"] = direct_output
                    log_info(f"\n==> 🐙 Final Answer:\n\n{direct_output}")

                log_info(f"\n[Total Time]: {round(time.time() - query_start_time, 2)}s")
                log_info(f"\n==> ✅ Query Solved!")

            ## update logs
            json_data.update({"planner_logs": deepcopy(self.planner.logs)})
            self.planner.logs.clear()
            return json_data, reward

        

        
        elif self.task in ["webshop", "rule", "alfworld"]:
            print(f"Task: [{self.task}]")

            if self.task == "rule":
                ## environment (task)
                print(f"Question:\n{question} \n\n ground_truth:{ground_truth}")
                if isinstance(ground_truth, list) and len(ground_truth) == 1:
                    ground_truth = ground_truth[0]
                
                question, info = self.env.reset(question=question, answer=ground_truth)
            
            elif self.task == "alfworld":
                raw_alfworld_info = question
                if "val_seen_" in raw_alfworld_info:
                    task_type = raw_alfworld_info.split("val_seen_")[1].split("_alfworld")[0]
                    question_id = int(raw_alfworld_info.split("alfworld_")[1])
                    data_type = "val_seen"
                elif "val_unseen_" in raw_alfworld_info:
                    task_type = raw_alfworld_info.split("val_unseen_")[1].split("_alfworld")[0]
                    question_id = int(raw_alfworld_info.split("alfworld_")[1])
                    data_type = "val_unseen"
                elif "train_" in raw_alfworld_info:
                    task_type = raw_alfworld_info.split("train_")[1].split("_alfworld")[0]
                    question_id = int(raw_alfworld_info.split("alfworld_")[1])
                    data_type = "train"
                else:
                    raise ValueError(f"Invalid alfworld data: {raw_alfworld_info}")
                
                question, info = self.env.reset(task_type=task_type, question_id=question_id, data_type=data_type)
                
            elif self.task == "webshop":
                print("Solver Session:\n", question)
                question, info = self.env.reset(session=question)
            
            else:
                raise ValueError(f"Invalid task: {self.task}")

            # Initialize json_data with basic problem information
            json_data = {
                "query": question,
                "image": image_path,
                "anchor": [info["anchor"]] if info["anchor"] is not None else None
            }
            if self.verbose:
                log_info(f"\n==> 🔍 Received Query: {question}")
                if image_path:
                    log_info(f"\n==> 🖼️ Received Image: {image_path}")

            # Generate base response if requested
            if 'base' in self.output_types:
                step_count = 0
                query_start_time = time.time()
                while True:
                    step_count += 1
                    # print(f"Query:\n{question}")
                    base_response = self.planner.generate_base_response(question, image_path, self.max_tokens)
                    print(f"Action:\n{base_response}")
                    question, reward, done, info = self.env.step(base_response)
                    print(f'Reward = {reward}, Done = {done}')
                    if info["anchor"] is not None:
                        json_data["anchor"].append(info["anchor"])
                    
        
                    if done:
                        print(f"[bold green]Task finished! Final reward: {reward}[/bold green]")
                        break

                json_data.update({
                        "step_count": step_count,
                        "execution_time": round(time.time() - query_start_time, 2),
                    })

                json_data["base_response"] = base_response
                if self.verbose:
                    log_info(f"\n==> 📝 Base Response from LLM:\n{base_response}")

            # If only base response is needed, save and return
            if set(self.output_types) == {'base'}:
                ## update logs
                json_data.update({"planner_logs": deepcopy(self.planner.logs)})
                self.planner.logs.clear()
                return json_data, reward
        
            # # Continue with query analysis and tool execution if final or direct responses are needed
            # if {'final', 'direct'} & set(self.output_types):
            #     if self.verbose:
            #         log_info(f"\n==> 🐙 Reasoning Steps from AgentFlow (Deep Thinking...)")

            #     # [1] Analyze query
            #     query_start_time = time.time()
            #     query_analysis = self.planner.analyze_query(question, image_path)
            #     json_data["query_analysis"] = query_analysis
            #     if self.verbose:
            #         log_info(f"\n==> 🔍 Step 0: Query Analysis\n")
            #         log_info(f"{query_analysis}")
            #         log_info(f"[Time]: {round(time.time() - query_start_time, 2)}s")

            #     # Main execution loop
            #     step_count = 0
            #     action_times = []
            #     while step_count < self.max_steps and (time.time() - query_start_time) < self.max_time:
            #         step_count += 1
            #         step_start_time = time.time()

            #         # [2] Generate next step
            #         local_start_time = time.time()
            #         next_step = self.planner.generate_next_step(
            #             question, 
            #             image_path, 
            #             query_analysis, 
            #             self.memory, 
            #             step_count, 
            #             self.max_steps,
            #             json_data
            #         )
            #         context, sub_goal, tool_name = self.planner.extract_context_subgoal_and_tool(next_step)
            #         if self.verbose:
            #             log_info(f"\n==> 🎯 Step {step_count}: Action Prediction ({tool_name})\n")
            #             log_info(f"[Context]: {context}\n[Sub Goal]: {sub_goal}\n[Tool]: {tool_name}")
            #             log_info(f"[Time]: {round(time.time() - local_start_time, 2)}s")

            #         if tool_name is None or tool_name not in self.planner.available_tools:
            #             log_info(f"\n==> 🚫 Error: Tool '{tool_name}' is not available or not found.")
            #             command = "No command was generated because the tool was not found."
            #             result = "No result was generated because the tool was not found."

            #         else:
            #             # [3] Generate the tool command
            #             local_start_time = time.time()
            #             tool_command = self.executor.generate_tool_command(
            #                 question, 
            #                 image_path, 
            #                 context, 
            #                 sub_goal, 
            #                 tool_name, 
            #                 self.planner.toolbox_metadata[tool_name],
            #                 step_count,
            #                 json_data
            #             )
            #             analysis, explanation, command = self.executor.extract_explanation_and_command(tool_command)
            #             if self.verbose:
            #                 log_info(f"\n==> 📝 Step {step_count}: Command Generation ({tool_name})\n")
            #                 log_info(f"[Analysis]: {analysis}\n[Explanation]: {explanation}\n[Command]: {command}")
            #                 log_info(f"[Time]: {round(time.time() - local_start_time, 2)}s")
                        
            #             # [4] Execute the tool command
            #             local_start_time = time.time()
            #             result = self.executor.execute_tool_command(tool_name, command)
            #             result = make_json_serializable_truncated(result) # Convert to JSON serializable format
            #             json_data[f"tool_result_{step_count}"] = result

            #             if self.verbose:
            #                 log_info(f"\n==> 🛠️ Step {step_count}: Command Execution ({tool_name})\n")
            #                 log_info(f"[Result]:\n{json.dumps(result, indent=4)}")
            #                 log_info(f"[Time]: {round(time.time() - local_start_time, 2)}s")
                    
            #         # Track execution time for the current step
            #         execution_time_step = round(time.time() - step_start_time, 2)
            #         action_times.append(execution_time_step)

            #         # Update memory
            #         self.memory.add_action(step_count, tool_name, sub_goal, command, result)
            #         memory_actions = self.memory.get_actions()

            #         # [5] Verify memory (context verification)
            #         local_start_time = time.time()
            #         stop_verification = self.verifier.verificate_context(
            #             question,
            #             image_path,
            #             query_analysis,
            #             self.memory,
            #             step_count,
            #             json_data
            #         )
            #         context_verification, conclusion = self.verifier.extract_conclusion(stop_verification)
            #         if self.verbose:
            #             conclusion_emoji = "✅" if conclusion == 'STOP' else "🛑"
            #             log_info(f"\n==> 🤖 Step {step_count}: Context Verification\n")
            #             log_info(f"[Analysis]: {context_verification}\n[Conclusion]: {conclusion} {conclusion_emoji}")
            #             log_info(f"[Time]: {round(time.time() - local_start_time, 2)}s")
                    
            #         # Break the loop if the context is verified
            #         if conclusion == 'STOP':
            #             break

            #     # Add memory and statistics to json_data
            #     json_data.update({
            #         "memory": memory_actions,
            #         "step_count": step_count,
            #         "execution_time": round(time.time() - query_start_time, 2),
            #     })

            #     # Generate final output if requested
            #     if 'final' in self.output_types:
            #         final_output = self.planner.generate_final_output(question, image_path, self.memory)
            #         json_data["final_output"] = final_output
            #         log_info(f"\n==> 🐙 Detailed Solution:\n\n{final_output}")

            #     # Generate direct output if requested
            #     if 'direct' in self.output_types:
            #         direct_output = self.planner.generate_direct_output(question, image_path, self.memory)
            #         json_data["direct_output"] = direct_output
            #         log_info(f"\n==> 🐙 Final Answer:\n\n{direct_output}")

            #     log_info(f"\n[Total Time]: {round(time.time() - query_start_time, 2)}s")
            #     log_info(f"\n==> ✅ Query Solved!")

            # ## update logs
            # json_data.update({"planner_logs": deepcopy(self.planner.logs)})
            # self.planner.logs.clear()
            # return json_data

def construct_solver(llm_engine_name : str = "gpt-4o",
                     enabled_tools : list[str] = ["all"],
                     tool_engine: list[str] = ["Default"],
                     model_engine: list[str] = ["trainable", "gpt-4o", "gpt-4o", "gpt-4o"],  # [planner_main, planner_fixed, verifier, executor]
                     output_types : str = "final,direct",
                     max_steps : int = 10,
                     max_time : int = 300,
                     max_tokens : int = 4000,
                     root_cache_dir : str = "solver_cache",
                     verbose : bool = True,
                     vllm_config_path : str = None,
                     base_url : str = None,
                     temperature: float = 0.0,
                     task: str = "qa"
                     ):

    # Parse model_engine configuration
    # Format: [planner_main, planner_fixed, verifier, executor]
    # "trainable" means use llm_engine_name (the trainable model)
    planner_main_engine = llm_engine_name if model_engine[0] == "trainable" else model_engine[0]
    planner_fixed_engine = llm_engine_name if model_engine[1] == "trainable" else model_engine[1]
    verifier_engine = llm_engine_name if model_engine[2] == "trainable" else model_engine[2]
    executor_engine = llm_engine_name if model_engine[3] == "trainable" else model_engine[3]

    verifier_is_train = True if model_engine[2] == "trainable" else False
    executor_is_train = True if model_engine[3] == "trainable" else False

    # Instantiate Initializer
    initializer = Initializer(
        enabled_tools=enabled_tools,
        tool_engine=tool_engine,
        model_string=llm_engine_name,
        verbose=verbose,
        vllm_config_path=vllm_config_path,
    )
    
    print("llm_engine_name:", llm_engine_name)
    print("planner_main_engine:", planner_main_engine)
    print("planner_fixed_engine:", planner_fixed_engine)
    print("verifier_engine:", verifier_engine)
    print("executor_engine:", executor_engine)

    # Instantiate Planner
    planner = Planner(
        llm_engine_name=planner_main_engine,
        llm_engine_fixed_name=planner_fixed_engine,
        toolbox_metadata=initializer.toolbox_metadata,
        available_tools=initializer.available_tools,
        verbose=verbose,
        base_url=base_url,
        temperature=temperature
    )


    # Instantiate Verifier
    verifier = Verifier(
        llm_engine_name=verifier_engine,
        llm_engine_fixed_name=planner_fixed_engine,
        toolbox_metadata=initializer.toolbox_metadata,
        available_tools=initializer.available_tools,
        verbose=verbose,
        base_url=base_url if verifier_engine == llm_engine_name else None,
        temperature=temperature if verifier_is_train else 0.0
    )

    # Instantiate Memory
    memory = Memory()

    # Instantiate Executor with tool instances cache
    executor = Executor(
        llm_engine_name=executor_engine,
        root_cache_dir=root_cache_dir,
        verbose=verbose,
        base_url=base_url if executor_engine == llm_engine_name else None,  # Only use base_url for trainable model
        temperature=temperature if executor_is_train else 0.0,
        tool_instances_cache=initializer.tool_instances_cache  # Pass the cached tool instances
    )

    # Instantiate Solver
    solver = Solver(
        planner=planner,
        verifier=verifier,
        memory=memory,
        executor=executor,
        output_types=output_types,
        max_steps=max_steps,
        max_time=max_time,
        max_tokens=max_tokens,
        root_cache_dir=root_cache_dir,
        verbose=verbose,
        temperature=temperature,
        task=task
    )
    return solver

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the agentflow demo with specified parameters.")
    parser.add_argument("--llm_engine_name", default="gpt-4o", help="LLM engine name.")
    parser.add_argument(
        "--output_types",
        default="base,final,direct",
        help="Comma-separated list of required outputs (base,final,direct)"
    )
    parser.add_argument("--enabled_tools", default="Base_Generator_Tool", help="List of enabled tools.")
    parser.add_argument("--root_cache_dir", default="solver_cache", help="Path to solver cache directory.")
    parser.add_argument("--max_tokens", type=int, default=4000, help="Maximum tokens for LLM generation.")
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum number of steps to execute.")
    parser.add_argument("--max_time", type=int, default=300, help="Maximum time allowed in seconds.")
    parser.add_argument("--verbose", type=bool, default=True, help="Enable verbose output.")
    return parser.parse_args()
    
def main(args):
    ## gpt-4o-mini
    # tool_engine=["gpt-4o-mini","gpt-4o-mini","Default","Default"]
    # enabled_tools = ["Base_Generator_Tool","Python_Coder_Tool","Google_Search_Tool","Wikipedia_Search_Tool"]


   
    ## qwen (local)
    tool_engine = ["vllm-Qwen3-30B-A3B-Instruct-2507","vllm-Qwen3-30B-A3B-Instruct-2507", "vllm-Qwen3-30B-A3B-Instruct-2507", "vllm-Qwen3-30B-A3B-Instruct-2507", "Default"]
    enabled_tools = ["Base_Generator_Tool", "Python_Coder_Tool", "Wikipedia_Search_Tool", "Web_Search_Tool", "Brave_Search_Tool"]
    model_engine = ["trainable", "vllm-Qwen3-30B-A3B-Instruct-2507","vllm-Qwen3-30B-A3B-Instruct-2507","vllm-Qwen3-30B-A3B-Instruct-2507"]  # [planner_main, planner_fixed, verifier, executor] , trainable is local setting.
    
    llm_engine_name = "vllm-Qwen3-30B-A3B-Instruct-2507" # "vllm-Qwen3-4B-Instruct-2507"
    base_url = "http://127.0.0.1:19998/v1"  ## vllm-based url (local planner)

    ## test different types
    args.output_types = "final"
    task = "qa"
    

    solver = construct_solver(
        llm_engine_name=llm_engine_name,
        enabled_tools=enabled_tools,
        tool_engine=tool_engine,
        model_engine=model_engine,
        output_types=args.output_types,
        max_steps=args.max_steps,
        max_time=args.max_time,
        max_tokens=args.max_tokens,
        base_url=base_url,
        verbose=args.verbose,
        # temperature=0.7
        temperature=0,
        task=task
    )
    
    # Solve the task or problem
    # query = "What is the capital of France?"
    ## webshop
    # query = None

    ## enconomic
    query = """【客观题】某寿险模型中，设人的死亡年龄为连续型随机变量 X，分布函数为 F(x)=x/(1+x)，x≥0。现已知一位目前年龄为 25 岁的人已存活至今（作为条件事件），在忽略截尾与竞争风险的前提下，其在 35.50 岁至 42.80 岁之间死亡的条件概率为（ ）。
A：0.1187
B：0.1215
C：0.1162
D：0.1199"""
    
    # query = "1+1=?"
    
    ## alfworld
    # query = "val_seen_pick_alfworld_0"
    # query = "Jen enters a lottery by picking $4$ distinct numbers from $S=\\{1,2,3,\\cdots,9,10\\}.$ $4$ numbers are randomly chosen from $S.$ She wins a prize if at least two of her numbers were $2$ of the randomly chosen numbers, and wins the grand prize if all four of her numbers were the randomly chosen numbers. The probability of her winning the grand prize given that she won a prize is $\\tfrac{m}{n}$ where $m$ and $n$ are relatively prime positive integers. Find $m+n$.\n\nWhen ready, output the final answer enclosed in <answer> and </answer> tags. Do not generate any content after the </answer> tag."

    print("### 1st time ###")
    result = solver.solve(question=query)
    # with open("output_1.txt", "w", encoding="utf-8") as f:
    #     f.write(str(result))
    
    # print("======================== 2nd time ==============================")
    # ## v2-time
    # solver = construct_solver(
    #     llm_engine_name=llm_engine_name,
    #     enabled_tools=enabled_tools,
    #     tool_engine=tool_engine,
    #     model_engine=model_engine,
    #     output_types=args.output_types,
    #     max_steps=args.max_steps,
    #     max_time=args.max_time,
    #     max_tokens=args.max_tokens,
    #     base_url=base_url,
    #     verbose=args.verbose,
    #     # temperature=0.7
    #     temperature=0
    # )

    # result = solver.solve(query)
    # with open("output_2.txt", "w", encoding="utf-8") as f:
    #     f.write(str(result))
    


    # print("===== total result ======")
    # print(f"result: {result}")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
