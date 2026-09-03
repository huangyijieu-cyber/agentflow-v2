import json
import ast

file_list = ["output_1.txt", "output_2.txt"]

for file in file_list:
    with open(file, "r", encoding="utf-8") as f:
        lines = [ast.literal_eval(line.strip()) for line in f if line.strip()]
    
    result = lines
  
    with open(file.replace(".txt", ".json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

