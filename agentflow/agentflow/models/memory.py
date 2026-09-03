from typing import Dict, Any, List, Union, Optional
import os

class Memory:

    def __init__(self):
        self.query: Optional[str] = None
        self.files: List[Dict[str, str]] = []
        self.actions: Dict[str, Dict[str, Any]] = {}
        self._init_file_types()

    def set_query(self, query: str) -> None:
        if not isinstance(query, str):
            raise TypeError("Query must be a string")
        self.query = query

    def _init_file_types(self):
        self.file_types = {
            'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp'],
            'text': ['.txt', '.md'],
            'document': ['.pdf', '.doc', '.docx'],
            'code': ['.py', '.js', '.java', '.cpp', '.h'],
            'data': ['.json', '.csv', '.xml'],
            'spreadsheet': ['.xlsx', '.xls'],
            'presentation': ['.ppt', '.pptx'],
        }
        self.file_type_descriptions = {
            'image': "An image file ({ext} format) provided as context for the query",
            'text': "A text file ({ext} format) containing additional information related to the query",
            'document': "A document ({ext} format) with content relevant to the query",
            'code': "A source code file ({ext} format) potentially related to the query",
            'data': "A data file ({ext} format) containing structured data pertinent to the query",
            'spreadsheet': "A spreadsheet file ({ext} format) with tabular data relevant to the query",
            'presentation': "A presentation file ({ext} format) with slides related to the query",
        }

    def _get_default_description(self, file_name: str) -> str:
        _, ext = os.path.splitext(file_name)
        ext = ext.lower()

        for file_type, extensions in self.file_types.items():
            if ext in extensions:
                return self.file_type_descriptions[file_type].format(ext=ext[1:])

        return f"A file with {ext[1:]} extension, provided as context for the query"
    
    def add_file(self, file_name: Union[str, List[str]], description: Union[str, List[str], None] = None) -> None:
        if isinstance(file_name, str):
            file_name = [file_name]
        
        if description is None:
            description = [self._get_default_description(fname) for fname in file_name]
        elif isinstance(description, str):
            description = [description]
        
        if len(file_name) != len(description):
            raise ValueError("The number of files and descriptions must match.")
        
        for fname, desc in zip(file_name, description):
            self.files.append({
                'file_name': fname,
                'description': desc
            })

    def add_action(self, step_count: int, tool_name: str, sub_goal: str, command: str, result: Any) -> None:
        action = {
            'tool_name': tool_name,
            'sub_goal': sub_goal,
            'command': command,
            'result': result,
        }
        step_name = f"Action Step {step_count}"
        self.actions[step_name] = action
    
    def clear(self) -> None:
        """Clear all accumulated actions (and file list).

        Called at the start of each task's solve() to prevent cross-task memory
        accumulation: without this, a shared solver instance keeps appending every
        tool result across tasks, which blows up the prompt input_tokens and causes
        vLLM context-limit 400 errors (see memory accumulation analysis).
        """
        self.actions = {}
        self.files = []

    def get_actions(self, max_steps: int = 3, max_result_chars: int = 2000,
                    max_command_chars: int = 500) -> Dict[str, Dict[str, Any]]:
        """Return actions with bounded size to keep the prompt's input_tokens in check.

        The raw actions dict grows unboundedly: each step stores the full tool result
        (e.g. long Brave/Web_RAG snippets). Embedding all of it into every
        Planner/Verifier prompt eventually exceeds the model's context window
        (input_tokens + max_tokens > max_model_len -> vLLM 400).

        Strategy (bounded memory view):
          - keep only the most recent `max_steps` actions (older steps are usually
            less relevant to the current sub-goal);
          - truncate each step's `result` to `max_result_chars`;
          - truncate `command` to `max_command_chars`.

        Returns a dict with the same shape as before so existing
        `{memory.get_actions()}` f-string usage keeps working.
        """
        if not self.actions:
            return {}
        # Keep the most recent max_steps actions (dict preserves insertion order).
        steps = list(self.actions.items())[-max_steps:]
        truncated = {}
        for step_name, action in steps:
            item = dict(action)
            try:
                result_str = str(item.get("result", ""))
                if len(result_str) > max_result_chars:
                    item["result"] = result_str[:max_result_chars] + "...[truncated]"
            except Exception:
                pass
            try:
                cmd_str = str(item.get("command", ""))
                if len(cmd_str) > max_command_chars:
                    item["command"] = cmd_str[:max_command_chars] + "...[truncated]"
            except Exception:
                pass
            truncated[step_name] = item
        return truncated

    def get_all_actions(self) -> Dict[str, Dict[str, Any]]:
        """Return the FULL (untruncated) actions dict, for trace/logging purposes.

        Keeps complete per-step results for offline analysis (rollout_data jsonl),
        independent of the size-bounded view used inside the prompt (get_actions).
        """
        return self.actions
        
    def get_query(self) -> Optional[str]:
        return self.query

    def get_files(self) -> List[Dict[str, str]]:
        return self.files
    
    # def get_actions(self) -> Dict[str, Dict[str, Any]]:
    #     return self.actions
    