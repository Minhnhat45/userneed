import json

test_path = "./data/test_list_27_11_2025.json"
suffix = "_".join(test_path.split('_')[-3:])
print(f"./data/qwen3_infer_{suffix}")