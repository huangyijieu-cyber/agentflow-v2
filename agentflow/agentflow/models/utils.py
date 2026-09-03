import json


def robust_json_loads(text: str):
    """Robustly parse a JSON object from a string that may contain extra text
    before/after the JSON payload (e.g. model prose or trailing explanation).

    Defensive fallback chain:
      1) whole string is valid JSON  -> parse directly (fast path)
      2) slice from first '{' to last '}' and parse (handles prefix/suffix)
      3) raw_decode the first valid JSON object, ignoring trailing garbage
    Raises json.JSONDecodeError if no valid JSON object can be found.
    """
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")
    text = text.strip()

    # 1) Whole string is valid JSON -> return directly
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2) Try extracting from the first '{' to the last '}'
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 3) raw_decode the first valid JSON object, tolerate trailing garbage
    idx = text.find('{')
    if idx != -1:
        try:
            decoder = json.JSONDecoder()
            obj, _ = decoder.raw_decode(text[idx:])
            return obj
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("No valid JSON object found in response", text, 0)
    

def make_json_serializable(obj):
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {make_json_serializable(key): make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(element) for element in obj]
    elif hasattr(obj, '__dict__'):
        return make_json_serializable(obj.__dict__)
    else:
        return str(obj)
    

def make_json_serializable_truncated(obj, max_length: int = 100000):
    if isinstance(obj, (int, float, bool, type(None))):
        if isinstance(obj, (int, float)) and len(str(obj)) > max_length:
            return str(obj)[:max_length - 3] + "..."
        return obj
    elif isinstance(obj, str):
        return obj if len(obj) <= max_length else obj[:max_length - 3] + "..."
    elif isinstance(obj, dict):
        return {make_json_serializable_truncated(key, max_length): make_json_serializable_truncated(value, max_length) 
                for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable_truncated(element, max_length) for element in obj]
    elif hasattr(obj, '__dict__'):
        return make_json_serializable_truncated(obj.__dict__, max_length)
    else:
        result = str(obj)
        return result if len(result) <= max_length else result[:max_length - 3] + "..."
    