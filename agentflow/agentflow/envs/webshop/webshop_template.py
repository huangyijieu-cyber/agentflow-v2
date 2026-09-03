
# ----- no thinking ---------------------------------
# --------------------- WebShop --------------------- #
WEBSHOP_TEMPLATE_NO_HIS = """You are an expert autonomous agent operating in the WebShop e-commerce environment. 
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are: 
[
{available_actions}
].

Rules:
1. Choose exactly one admissible action from the list above that best advances the shopping goal.
2. For the "search" action, replace the placeholder "<your query>" with a specific, relevant search query based on the task description and current observation. Do not output the literal text "<your query>".
3. Respond ONLY with the chosen action enclosed in <action> </action> tags. Do not output any reasoning, explanation, or <think> blocks.
"""

WEBSHOP_TEMPLATE = """You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are: 
[
{available_actions}
].

Rules:
1. Choose exactly one admissible action from the list above that best advances the shopping goal.
2. For the "search" action, replace the placeholder "<your query>" with a specific, relevant search query based on the task description and current observation. Do not output the literal text "<your query>".
3. Respond ONLY with the chosen action enclosed in <action> </action> tags. Do not output any reasoning, explanation, or <think> blocks.
"""



# ## ------ thinking ---------------------------------------------
# WEBSHOP_TEMPLATE_NO_HIS = """
# You are an expert autonomous agent operating in the WebShop e‑commerce environment. 
# Your task is to: {task_description}.
# Your current observation is: {current_observation}.
# Your admissible actions of the current situation are: 
# [
# {available_actions}
# ].

# Now it's your turn to take one action for the current step.
# You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. 
# Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
# """

# WEBSHOP_TEMPLATE = """
# You are an expert autonomous agent operating in the WebShop e‑commerce environment.
# Your task is to: {task_description}.
# Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
# You are now at step {current_step} and your current observation is: {current_observation}.
# Your admissible actions of the current situation are: 
# [
# {available_actions}
# ].

# Now it's your turn to take one action for the current step.
# You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. 
# Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
# """