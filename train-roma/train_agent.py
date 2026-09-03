import yaml
import os
os.environ["AGENTOPS_API_KEY"] = ""  # 清空 API key 禁用监控
os.environ["AGENTOPS_AUTO_INIT"] = "false"
import subprocess
import sys
import argparse

from agentflow import Trainer, LitAgent, NamedResources, LLM, reward, configure_logger, DevTaskLoader

import logging
configure_logger(logging.INFO)


def main():
    """
    Main function to parse YAML config, set environment variables,
    and run the training script.
    """
    # Define the path to the YAML configuration file
    from utils import load_and_set_env_from_yaml
    config_file_path = "train-roma/config.yaml"
    config = load_and_set_env_from_yaml(config_file_path)
    

    # --- Construct Python Command Arguments ---
    # Start with the core command parts
    command = ["python", "-m", "agentflow.verl"]

    # Use argparse to handle user-provided command-line overrides
    # This allows users to pass args like `python run_training.py data.train_batch_size=16`
    parser = argparse.ArgumentParser(description="Run training script with YAML config.")
    # Add a catch-all argument for user overrides
    parser.add_argument('overrides', nargs='*', default=[])
    args, unknown = parser.parse_known_args()

    # Get arguments from YAML and format them as 'key=value'
    if 'python_args' in config:
        print("Constructing Python command arguments...")
        for key, value in config['python_args'].items():
            # Support referencing environment variables in the YAML file
            # e.g., ${TRAIN_DATA_DIR}
            if isinstance(value, str):
                # Use os.path.expandvars to replace ${VAR} with its value
                expanded_value = os.path.expandvars(value)
                command.append(f"{key}={expanded_value}")
            else:
                command.append(f"{key}={value}")

    # Add any user-provided overrides to the command
    command.extend(unknown)
    
    # --- Execute the command ---
    print("\nStarting training script with the following command:")
    print(" ".join([str(item) for item in command]))
    print("-" * 50)

    try:
        # Use subprocess.run to execute the command.
        # env=os.environ passes all currently set environment variables.
        # check=True will raise an exception if the command returns a non-zero exit code.
        subprocess.run(command, check=True, env=os.environ)
    except subprocess.CalledProcessError as e:
        print(f"Error: The training script failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print(f"Error: The command '{command[0]}' was not found. "
              "Please make sure python is in your PATH.")
        sys.exit(1)

if __name__ == "__main__":
    main()