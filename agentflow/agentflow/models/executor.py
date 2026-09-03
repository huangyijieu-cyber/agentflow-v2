import importlib
import json
import os
import re
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from agentflow.engine.factory import create_llm_engine
from agentflow.models.formatters import ToolCommand
from agentflow.models.utils import robust_json_loads
## t00620714 (modified here)
# Tool name mapping: Static fallback mapping (long external names to internal)
TOOL_NAME_MAPPING_LONG = {
    "Generalist_Solution_Generator_Tool": {
        "class_name": "Base_Generator_Tool",
        "dir_name": "base_generator"
    },
    # "Ground_Google_Search_Tool": {
    #     "class_name": "Google_Search_Tool",
    #     "dir_name": "google_search"
    # },
    "Python_Code_Generator_Tool": {
        "class_name": "Python_Coder_Tool",
        "dir_name": "python_coder"
    },
    "Web_RAG_Search_Tool": {
        "class_name": "Web_Search_Tool",
        "dir_name": "web_search"
    },
    "Wikipedia_RAG_Search_Tool": {
        "class_name": "Wikipedia_Search_Tool",
        "dir_name": "wikipedia_search"
    },
    "Yibu_Brave_Search_Tool": {
        "class_name": "Brave_Search_Tool",
        "dir_name": "brave_search"
    }
}

# Short to long mapping for fallback
TOOL_NAME_MAPPING_SHORT = {
    "Base_Generator_Tool": "Generalist_Solution_Generator_Tool",
    # "Google_Search_Tool": "Ground_Google_Search_Tool",
    "Python_Coder_Tool": "Python_Code_Generator_Tool",
    "Web_Search_Tool": "Web_RAG_Search_Tool",
    "Wikipedia_Search_Tool": "Wikipedia_RAG_Search_Tool",
    "Brave_Search_Tool": "Yibu_Brave_Search_Tool"
}



class Executor:
    def __init__(self, llm_engine_name: str, root_cache_dir: str = "solver_cache",  num_threads: int = 1, max_time: int = 300,
    max_output_length: int = 100000, verbose: bool = False, base_url: str = None, check_model: bool = True, temperature: float = .0,
    tool_instances_cache: dict = None):
        self.llm_engine_name = llm_engine_name
        self.root_cache_dir = root_cache_dir
        self.num_threads = num_threads
        self.max_time = max_time
        self.max_output_length = max_output_length
        self.verbose = verbose
        self.base_url = base_url
        self.check_model = check_model
        self.temperature  = temperature

        # Store the tool instances cache
        self.tool_instances_cache = tool_instances_cache if tool_instances_cache is not None else {}

        if base_url is not None:
            self.llm_generate_tool_command = create_llm_engine(model_string=self.llm_engine_name, is_multimodal=False, base_url=self.base_url, temperature = self.temperature)
        else:
            self.llm_generate_tool_command = create_llm_engine(model_string=self.llm_engine_name, is_multimodal=False, temperature = self.temperature)
    
    def set_query_cache_dir(self, query_cache_dir):
        if query_cache_dir:
            self.query_cache_dir = query_cache_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.query_cache_dir = os.path.join(self.root_cache_dir, timestamp)
        os.makedirs(self.query_cache_dir, exist_ok=True)

    def generate_tool_command(self, question: str, image: str, context: str, sub_goal: str, tool_name: str, tool_metadata: Dict[str, Any], step_count: int = 0, json_data: Any = None) -> Any:
        prompt_generate_tool_command = f"""
        Task: Generate a precise command to execute the selected tool.

Context:
- **Query:** {question}
- **Sub-Goal:** {sub_goal}
- **Tool Name:** {tool_name}
- **Tool Metadata:** {tool_metadata}
- **Relevant Data:** {context}

Instructions:
1.  Analyze the tool's required parameters from its metadata.
2.  Construct valid Python code that addresses the sub-goal using the provided context and data.
3.  The command must include at least one call to `tool.execute()`.
4.  Each `tool.execute()` call must be assigned to a variable named **`execution`**.
5.  Please give the exact numbers and parameters should be used in the `tool.execute()` call.

Output Format:
Present your response in the following structured format. Do not include any extra text or explanations.

Generated Command:
```python
<command>
```

Example1:
Generated Command:
```python
execution = tool.execute(query="Summarize the following porblom:"Isaac has 100 toys, masa gets ...., how much are their together?")
```

Example2:
Generated Command:
```python
execution = tool.execute(query=["Methanol", "function of hyperbola", "Fermat's Last Theorem"])
```
"""

        tool_command = self.llm_generate_tool_command(prompt_generate_tool_command, response_format=ToolCommand, temperature=self.temperature, usage_by="[executor] generate tool command")
        if json_data is not None:
            json_data[f"tool_commander_{step_count}_prompt"] = prompt_generate_tool_command
            json_data[f"tool_commander_{step_count}_response"] = str(tool_command)

        return tool_command

    def extract_explanation_and_command(self, response: Any) -> tuple:
        def normalize_code(code: str) -> str:
            # Remove leading/trailing whitespace and triple backticks if present
            return re.sub(r'^```python\s*', '', code).rstrip('```').strip()

        analysis = "No analysis found."
        explanation = "No explanation found."
        command = "No command found."

        if isinstance(response, str):
            # Attempt to parse as JSON first
            try:
                response_dict = robust_json_loads(response)
                response_obj = ToolCommand(**response_dict)
                analysis = response_obj.analysis.strip()
                explanation = response_obj.explanation.strip()
                command = response_obj.command.strip()
            except Exception as e:
                print(f"Failed to parse response as JSON in Executor: {str(e)}")
                # Fall back to regex parsing on string
                try:
                    # Extract analysis
                    analysis_pattern = r"Analysis:(.*?)Command Explanation"
                    analysis_match = re.search(analysis_pattern, response, re.DOTALL | re.IGNORECASE)
                    analysis = analysis_match.group(1).strip() if analysis_match else "No analysis found."

                    # Extract explanation
                    explanation_pattern = r"Command Explanation:(.*?)Generated Command"
                    explanation_match = re.search(explanation_pattern, response, re.DOTALL | re.IGNORECASE)
                    explanation = explanation_match.group(1).strip() if explanation_match else "No explanation found."

                    # Extract command using "Generated Command:" prefix
                    command_pattern = r"Generated Command:.*?```python\n(.*?)```"
                    command_match = re.search(command_pattern, response, re.DOTALL | re.IGNORECASE)
                    if command_match:
                        command = command_match.group(1).strip()
                    else:
                        # Fallback: Extract ANY ```python ... ``` block (even without prefix)
                        loose_command_pattern = r"```python\s*\n(.*?)```"
                        loose_match = re.findall(loose_command_pattern, response, re.DOTALL | re.IGNORECASE)
                        if loose_match:
                            # Take the last or most complete one? Or first meaningful?
                            # Here we take the longest one as heuristic
                            command = max(loose_match, key=lambda x: len(x.strip())).strip()
                        else:
                            command = "No command found."
                except Exception as e:
                    print(f"Error during regex parsing: {str(e)}")
                    analysis = "Parsing error."
                    explanation = "Parsing error."
                    command = "No command found."
        elif isinstance(response, ToolCommand):
            analysis = response.analysis.strip()
            explanation = response.explanation.strip()
            command = response.command.strip()
        else:
            command = "Invalid response type."

        # Final normalization
        command = normalize_code(command)

        return analysis, explanation, command

    def execute_tool_command(self, tool_name: str, command: str) -> Any:
        """
        Execute a tool command with timeout protection. If execution exceeds max_time seconds,
        the function will be interrupted and return a timeout message.

        Args:
            tool_name (str): Name of the tool to execute
            command (str): Command string containing tool.execute() calls

        Returns:
            Any: List of execution results or error message
        """

        def split_commands(command: str) -> List[str]:
            # Use regex to find all tool.execute() commands and their surrounding code
            pattern = r'.*?execution\s*=\s*tool\.execute\([^\n]*\)\s*(?:\n|$)'
            blocks = re.findall(pattern, command, re.DOTALL)
            return [block.strip() for block in blocks if block.strip()]

        def execute_with_timeout(block: str, local_context: dict) -> Optional[str]:
            """
            Execute a code block with timeout protection using threading.
            This works in any thread, unlike signal.alarm() which only works in the main thread.
            
            Uses a cancellation event to allow cooperative cancellation and reduce memory leaks.
            """
            import threading
            
            result_container = {'result': None, 'exception': None, 'completed': False}
            cancel_event = threading.Event()
            
            def target():
                try:
                    # Inject cancel_event into the execution context for cooperative cancellation
                    local_context['_cancel_event'] = cancel_event
                    exec(block, globals(), local_context)
                    result_container['result'] = local_context.get('execution')
                    result_container['completed'] = True
                except Exception as e:
                    result_container['exception'] = e
                    result_container['completed'] = True
            
            # Start execution in a daemon thread
            exec_thread = threading.Thread(target=target, name=f"ToolExec-{id(block)}")
            exec_thread.daemon = True
            exec_thread.start()
            
            # Wait for completion or timeout
            exec_thread.join(timeout=self.max_time)
            
            if not result_container['completed']:
                # Timeout occurred - signal cancellation
                cancel_event.set()
                
                # Give it a brief moment to notice the cancellation
                exec_thread.join(timeout=0.5)
                
                # Clean up references to help GC
                result_container.clear()
                local_context.pop('_cancel_event', None)
                
                return f"Execution timed out after {self.max_time} seconds"
            elif result_container['exception']:
                raise result_container['exception']
            else:
                return result_container['result']


        # Try to get tool from cache first
        tool = None

        # Check if tool is in cache (tool_name could be the external long name)
        if tool_name in self.tool_instances_cache:
            tool = self.tool_instances_cache[tool_name]
            print(f"Using cached tool instance for: {tool_name}")
        else:
            # Fallback: Import the tool module and instantiate it
            print(f"Warning: Tool '{tool_name}' not found in cache, instantiating with default parameters")

            # tool_name could be either short or long name
            # First check if it's a long name
            if tool_name in TOOL_NAME_MAPPING_LONG:
                dir_name = TOOL_NAME_MAPPING_LONG[tool_name]["dir_name"]
                class_name = TOOL_NAME_MAPPING_LONG[tool_name]["class_name"]
            # Then check if it's a short name (convert to long, then get internal)
            elif tool_name in TOOL_NAME_MAPPING_SHORT:
                long_name = TOOL_NAME_MAPPING_SHORT[tool_name]
                if long_name in TOOL_NAME_MAPPING_LONG:
                    dir_name = TOOL_NAME_MAPPING_LONG[long_name]["dir_name"]
                    class_name = TOOL_NAME_MAPPING_LONG[long_name]["class_name"]
                else:
                    # Shouldn't happen, but fallback
                    dir_name = tool_name.lower().replace('_tool', '')
                    class_name = tool_name
            else:
                # Fallback to original behavior for unmapped tools
                dir_name = tool_name.lower().replace('_tool', '')
                class_name = tool_name

            module_name = f"tools.{dir_name}.tool"

            try:
                # Dynamically import the module
                module = importlib.import_module(module_name)

                # Get the tool class
                tool_class = getattr(module, class_name)

                tool = tool_class()

            except Exception as e:
                return f"Error importing tool '{tool_name}': {str(e)}"

        if tool is None:
            return f"Error: Could not get tool instance for '{tool_name}'"

        try:
            # Set the custom output directory
            tool.set_custom_output_dir(self.query_cache_dir)

            # Split the command into blocks, execute each one and store execution results
            command_blocks = split_commands(command)
            executions = []

            for block in command_blocks:
                # Create a local context to safely execute the block
                local_context = {'tool': tool}

                # Execute the block with timeout protection
                result = execute_with_timeout(block, local_context)

                if result is not None:
                    executions.append(result)
                else:
                    executions.append(f"No execution captured from block: {block}")

            # Return all the execution results
            return executions
        except Exception as e:
            return f"Error in execute_tool_command: {str(e)}"



if __name__ == "__main__":
    from agentflow.models.initializer import Initializer
    from agentflow.models.utils import make_json_serializable_truncated

    tool_engine = ["vllm-Qwen3-30B-A3B-Instruct-2507","vllm-Qwen3-30B-A3B-Instruct-2507"]
    enabled_tools = ["Base_Generator_Tool","Python_Coder_Tool"]
    llm_engine_name = "vllm-Qwen3-4B-Instruct-2507"
    verbose = True
    vllm_config_path = None

     # Instantiate Initializer
    initializer = Initializer(
        enabled_tools=enabled_tools,
        tool_engine=tool_engine,
        model_string=llm_engine_name,
        verbose=verbose,
        vllm_config_path=vllm_config_path,
    )

    executor = Executor(
        llm_engine_name="vllm-Qwen3-30B-A3B-Instruct-2507",
        root_cache_dir="solver_cache",
        verbose=True,
        base_url=None, ## 用内置url
        temperature=0,
        tool_instances_cache=initializer.tool_instances_cache, # Pass the cache tool instance
    )
    executor.set_query_cache_dir("solver_cache")
    
    question = "Jen enters a lottery by picking $4$ distinct numbers from $S=\\{1,2,3,\\cdots,9,10\\}.$ $4$ numbers are randomly chosen from $S.$ She wins a prize if at least two of her numbers were $2$ of the randomly chosen numbers, and wins the grand prize if all four of her numbers were the randomly chosen numbers. The probability of her winning the grand prize given that she won a prize is $\\tfrac{m}{n}$ where $m$ and $n$ are relatively prime positive integers. Find $m+n$.\n\nWhen ready, output the final answer enclosed in <answer> and </answer> tags. Do not generate any content after the </answer> tag."
    sub_goal = "Compute the exact value of the conditional probability $ \\frac{P(\\text{grand prize} \\mid \\text{prize})} $ using binomial coefficients, simplify the resulting fraction to lowest terms, and return $ m+n $."
    tool_name = "Python_Code_Generator_Tool"
    tool_metadata = "{'tool_name': 'Python_Code_Generator_Tool', 'tool_description': 'A tool that generates and executes simple Python code snippets for basic arithmetical calculations and math-related problems. The generated code runs in a highly restricted environment with only basic mathematical operations available.', 'tool_version': '1.0.0', 'input_types': {'query': 'str - A clear, specific description of the arithmetic calculation or math problem to be solved, including any necessary numerical inputs.'}, 'output_type': 'dict - A dictionary containing the generated code, calculation result, and any error messages.', 'demo_commands': [{'command': 'execution = tool.execute(query=\"Find the sum of prime numbers up to 50\")', 'description': 'Generate a Python code snippet to find the sum of prime numbers up to 50.'}, {'command': 'query=\"Given the list [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], calculate the sum of squares of odd numbers\"\\nexecution = tool.execute(query=query)', 'description': 'Generate a Python function for a specific mathematical operation on a given list of numbers.'}], 'user_metadata': {'limitations': \"\\nThe Python_Code_Generator_Tool has several limitations:\\n1. Restricted to basic Python arithmetic operations and built-in mathematical functions.\\n2. Cannot use any external libraries or modules, including those in the Python standard library.\\n3. Limited to simple mathematical calculations and problems.\\n4. Cannot perform any string processing, data structure manipulation, or complex algorithms.\\n5. No access to any system resources, file operations, or network requests.\\n6. Cannot use 'import' statements.\\n7. All calculations must be self-contained within a single function or script.\\n8. Input must be provided directly in the query string.\\n9. Output is limited to numerical results or simple lists/tuples of numbers.\\n10. Output should be kept to a single numerical result or a simple list/tuple of numbers.\\n11. DO NOT generate loop output.\\n\", 'best_practices': '\\nFor optimal results with the Python_Code_Generator_Tool:\\n1. Provide clear and specific queries that describe the desired mathematical calculation.\\n2. Include all necessary numerical inputs directly in the query string.\\n3. Keep tasks focused on basic arithmetic, algebraic calculations, or simple mathematical algorithms.\\n4. Ensure all required numerical data is included in the query.\\n5. Verify that the query only involves mathematical operations and does not require any data processing or complex algorithms.\\n6. Review generated code to ensure it only uses basic Python arithmetic operations and built-in math functions.\\n'}, 'require_llm_engine': True}"
    context = "- Total numbers in set $ S = \\{1, 2, \\dots, 10\\} $  \n- Jen picks 4 distinct numbers.  \n- 4 numbers are drawn randomly from $ S $.  \n- We need to compute:  \n  $$\n  P(\\text{grand prize} \\mid \\text{prize}) = \\frac{P(\\text{4 matches})}{P(\\text{at least 2 matches})}\n  $$  \n  where:  \n  - $ P(\\text{4 matches}) = \\frac{\\binom{4}{4} \\binom{6}{0}}{\\binom{10}{4}} $  \n  - $ P(\\text{at least 2 matches}) = \\frac{\\binom{4}{2} \\binom{6}{2} + \\binom{4}{3} \\binom{6}{1} + \\binom{4}{4} \\binom{6}{0}}{\\binom{10}{4}} $  \n- We need to compute these values numerically and simplify the resulting fraction $ \\frac{m}{n} $ with $ \\gcd(m,n)=1 $, then compute $ m+n $."
    step_count = 0
    image_path = None


    ## step 2 for tool command execution
    for idx in range(2):
        tool_command = executor.generate_tool_command(
            question,
            image_path,
            context,
            sub_goal,
            tool_name,
            tool_metadata,
            step_count,
        )
        print(f" {idx} step: tool_command:", tool_command)
        analysis, explanation, command = executor.extract_explanation_and_command(tool_command)
        result = executor.execute_tool_command(tool_name, command)
        result = make_json_serializable_truncated(result) # Convert to JSON serializable format
        print(f" {idx} step: result:", result)