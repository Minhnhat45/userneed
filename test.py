import json

with open("./data/test_list.json", "r", encoding="utf-8") as f:
    test_list = json.load(f)

for key in test_list["articles_id"].keys():
    print(test_list["articles_id"][key])