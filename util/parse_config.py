import yaml
import sys
import argparse
from typing import List, Any, Tuple
import os

def convert_value(v):
    if not isinstance(v, str):
        return v
    # 尝试解析列表
    if v.strip().startswith('[') and v.strip().endswith(']'):
        try:
            return eval(v)
        except:
            pass
    # 尝试转int
    try:
        return int(v)
    except ValueError:
        pass
    # 尝试转float
    try:
        return float(v)
    except ValueError:
        pass
    # 保持字符串
    return v


def get_values_from_yaml(config_path: str, keys: List[str], config=None) -> Tuple[Any, ...]:
    """
    Parses a YAML file and returns the values for a list of specified keys.
    It first searches in the 'python_args' section, then falls back to 'env'.

    Args:
        config_path (str): The path to the YAML configuration file.
        keys (List[str]): A list of keys to retrieve values for.

    Returns:
        Tuple[Any, ...]: A tuple of the values corresponding to the keys.
                         None is used for keys that are not found.
    """
    if config is None:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: Config file not found at '{config_path}'", file=sys.stderr)
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}", file=sys.stderr)
            sys.exit(1)

    ## values
    values = []
    for key in keys:
        value = None
        try:
            # First, try to get the value from the 'python_args' section.
            value = config['python_args'].get(key)
            if value is not None:
                ## 解析 环境变量
                if isinstance(value, str):
                    if value.startswith("${") and value.endswith("}"):
                        value = config["env"].get(value[len("${"): -len("}")])
                value = convert_value(value)
                values.append(value)
                continue
        except KeyError:
            # If 'python_args' section doesn't exist, ignore and proceed to the next step.
            pass

        try:
            # If not found in 'python_args', try to get the value from the 'env' section.
            value = config['env'].get(key)
            if value is not None:
                value = convert_value(value)
                values.append(value)
                continue
        except KeyError:
            # If 'env' section doesn't exist, ignore.
            pass

        # If the key was not found in either section, append None.
        if value is None:
            print(f"Warning: Key '{key}' not found in either 'python_args' or 'env' section.", file=sys.stderr)
            values.append(None)
    
    return tuple(values)

# The `main` function remains unchanged as it handles command-line parsing and calls `get_values_from_yaml`.
def main():
    """
    Main function to handle command-line arguments and run the parser.
    """
    parser = argparse.ArgumentParser(
        description="Retrieve values for specified keys from a YAML config file."
    )
    parser.add_argument(
        '-c', '--config', 
        type=str, 
        default='config.yaml',
        help="Path to the YAML configuration file. Defaults to 'config.yaml'."
    )
    parser.add_argument(
        'keys', 
        nargs='+', 
        help="A space-separated list of keys to retrieve from the YAML file."
    )
    
    args = parser.parse_args()
    
    result_tuple = get_values_from_yaml(args.config, args.keys)
    
    # Print the resulting tuple to standard output
    print(result_tuple)

if __name__ == "__main__":
    main()