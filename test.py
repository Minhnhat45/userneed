import json
from utils import get_article_data, build_input_data
from tqdm import tqdm
results = "./data/qwen3_infer_27_11_2025.json"
with open(results, "r", encoding="utf-8") as f:
    data = json.load(f)

out_data = data.copy()

for key in tqdm(data.keys()):
    for item, item_out in zip(data[key], out_data[key]):
        article_id = item["article_id"]
        api_data = get_article_data(int(article_id))
        item_out["data"] = build_input_data(api_data)
        item_out["url"] = api_data["data"]["share_url"] if api_data else None

with open("./data/qwen3_infer_27_11_2025_with_data_1.json", "w", encoding="utf-8") as out_f:
    json.dump(out_data, out_f, ensure_ascii=False, indent=4)
