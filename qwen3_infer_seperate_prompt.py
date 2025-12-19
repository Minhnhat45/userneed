import requests
import json
import sys
import logging
import logging.config
import yaml
from tqdm import tqdm
from g4f import Client
import ast
from openai import OpenAI
from utils import get_article_data, build_input_data
client = Client()

global local_infer
local_infer = True
with open(f"./logger/config.yaml", "r") as stream:
    log_config = yaml.safe_load(stream)
    # print("config=", json.dumps(config, indent=2, ensure_ascii=False))
    logging.config.dictConfig(log_config)


def build_prompt(article_text: str) -> str:
    return f"""Bạn là một trợ lý biên tập thông minh.  
Nhiệm vụ của bạn là phân tích một bài báo và:

1. **Phân loại** bài viết theo **duy nhất 1** trong 8 nhu cầu người dùng Smartocto 2.0:

   **Know – fact driven**
   - `Update me` – Cập nhật thông tin, tin tức mới, số liệu, diễn biến mới.
   - `Keep me engaged` – Tin tức/diễn biến nối tiếp, giúp tôi theo dõi câu chuyện đang quan tâm.

   **Understand – context driven**
   - `Educate me` – Giải thích, hướng dẫn, kiến thức, how-to, phân tích dễ hiểu.
   - `Give me perspective` – Phân tích, bình luận, so sánh nhiều góc nhìn, bối cảnh sâu.

   **Feel – emotion driven**
   - `Inspire me` – Truyền cảm hứng, động lực, câu chuyện vượt khó, thành công.
   - `Divert me` – Giải trí, thư giãn, hài hước, đời sống nhẹ nhàng.

   **Do – action driven**
   - `Help me` – Hướng dẫn hành động cụ thể, mẹo, lời khuyên thực tế.
   - `Connect me` – Kết nối cộng đồng, kêu gọi tham gia, tương tác, sự kiện.

- Không trả về các điểm số nào khác ngoài "user_need".

2. **Chấm điểm** bài viết theo 3 chỉ số:

   - `I1`: Tác động cảm xúc (Emotional Impact)
   - `I3`: Khả năng tạo tranh luận xã hội (Public Discourse Potential)
   - `I4`: Liên quan đến thay đổi chính sách/xã hội (Policy or Social Change Relevance)

### QUY TẮC RẤT QUAN TRỌNG:
- **Chỉ được dùng các giá trị sau cho từng chỉ số**:
  - `I1 ∈ {{1, 3, 5, 7, 9}}`
  - `I3 ∈ {{1, 3, 5, 7, 9}}`
  - `I4 ∈ {{1, 3, 5, 7, 9}}`
- **KHÔNG được sử dụng bất kỳ giá trị nào khác** (không dùng 2, 4, 6, 8, 10).
- Nếu đánh giá nằm giữa hai mức, hãy chọn **mức gần nhất** theo mô tả dưới đây.

#### I1 – Tác động cảm xúc
- `1`: Không có yếu tố cảm xúc.
- `3`: Có yếu tố cảm xúc nhẹ, không đáng chú ý.
- `5`: Có chút đồng cảm, cảm xúc hiện diện nhưng không sâu sắc.
- `7`: Gợi cảm xúc rõ rệt ở người đọc.
- `9`: Gây cảm xúc mạnh mẽ, dễ lan truyền, có thể dẫn tới hành động cộng đồng.

#### I3 – Khả năng tạo tranh luận xã hội
- `1`: Không tạo tranh luận.
- `3`: Có khả năng được chia sẻ nhưng không gây tranh luận.
- `5`: Có thể tạo bình luận cá nhân, không thành làn sóng.
- `7`: Có thể kích hoạt thảo luận trong một cộng đồng cụ thể.
- `9`: Dễ trở thành chủ đề nóng, tranh cãi rộng rãi trên mạng/xã hội.

#### I4 – Liên quan đến thay đổi chính sách/xã hội
- `1`: Rất Thấp/Không Liên Quan: Nội dung chủ yếu là giải trí, tin tức nhẹ, hoặc sở thích cá nhân, không có bất kỳ liên kết rõ ràng nào đến chính sách hoặc vấn đề xã hội lớn.
- `3`: Liên Quan Thấp: Nội dung là tin tức tiêu dùng/địa phương, có liên kết gián tiếp hoặc nhỏ đến chính sách hoặc xã hội.
- `5`: Liên Quan Trung Bình: Nội dung đề cập đến các vấn đề có tầm quan trọng công cộng nhưng chưa đi sâu vào khía cạnh chính sách hoặc cải cách.
- `7`: Liên Quan Cao: Nội dung phân tích hoặc thảo luận trực tiếp về chính sách công hiện hành, đề xuất luật mới, hoặc xu hướng xã hội lớn.
- `9`: Rất Cao/Tác Động Trực Tiếp: Nội dung là nghiên cứu chuyên sâu, báo cáo điều tra, hoặc phân tích chính sách có tiềm năng cao nhất để thúc đẩy hành động hoặc thay đổi quan điểm chính sách/xã hội.
---

### ĐẦU RA:
- Chỉ trả lời **duy nhất** dưới dạng **JSON hợp lệ**, **không thêm bất kỳ chữ nào khác**.
- Cấu trúc JSON:
{{
  "user_need": "một trong: Keep me engaged, Educate me, Give me perspective, Inspire me, Divert me, Help me, Connect me, Update me.",
  "I1": 1 hoặc 3 hoặc 5 hoặc 7 hoặc 9,
  "I3": 1 hoặc 3 hoặc 5 hoặc 7 hoặc 9,
  "I4": 1 hoặc 3 hoặc 5 hoặc 7 hoặc 9
}}
- Trước khi trả lời, hãy **tự kiểm tra lại**:
  - `user_need` có đúng một trong 8 giá trị cho phép hay không.
  - `I1`, `I3`, `I4` có nằm trong tập {{1, 3, 5, 7, 9}} hay không.
  - Nếu bất kỳ giá trị nào **không hợp lệ**, hãy **sửa lại** cho hợp lệ rồi mới xuất JSON.

---

### Văn bản:
{article_text}
"""
# - Do văn bản là bài báo nên luôn có yếu tố cập nhật thông tin, không được quá lạm dụng "Update me" hoặc "Educate me".
def build_user_need_prompt(article_text: str) -> str:
    return f"""
Bạn là một trợ lý biên tập thông minh.

**Nhiệm vụ:** 
- Đọc văn bản bài báo và trả về 1 JSON gồm 1 nhãn user_need (duy nhất 1 trong 8 nhãn Smartocto 2.0)

**Yêu cầu:**
- Bạn PHẢI phân loại TUÂN THỦ điều kiện gate. Không được chọn “Update me” hoặc “Educate me” như phương án an toàn.
- 8 nhãn Smartocto 2.0: 
   - “Update me” – Cập nhật thông tin, tin tức mới, số liệu, diễn biến mới.  
   - “Keep me engaged” – Tin tức/diễn biến nối tiếp, giúp tôi theo dõi câu chuyện đang quan tâm.

   - “Educate me” – Giải thích, hướng dẫn, kiến thức, how-to, phân tích dễ hiểu.  
   - “Give me perspective” – Phân tích, bình luận, so sánh nhiều góc nhìn, bối cảnh sâu.
   - “Inspire me” – Truyền cảm hứng, động lực, câu chuyện vượt khó, thành công.  
   - “Divert me” – Giải trí, thư giãn, hài hước, đời sống nhẹ nhàng.

   - “Help me” – Hướng dẫn hành động cụ thể, mẹo, lời khuyên thực tế.  
   - “Connect me” – Kết nối cộng đồng, kêu gọi tham gia, tương tác, sự kiện.

**Áp dụng GATE để chống bias:**

GATE 1 (Update me KHÔNG phải mặc định):
- Chỉ được chọn `Update me` khi bài viết chủ yếu là “tin mới/diễn biến/số liệu” và KHÔNG thỏa mạnh bất kỳ điều kiện nào trong GATE 2–5 dưới đây.
- Nếu bài có hướng dẫn, quan điểm, cảm xúc, giải trí, hoặc kêu gọi hành động rõ rệt thì KHÔNG được chọn Update me.

GATE 2 (Educate me chỉ khi có GIẢI THÍCH/HƯỚNG DẪN THỰC SỰ):
- Chỉ chọn Educate me khi trọng tâm là giải thích kiến thức, cơ chế, bối cảnh nền, how-to, hướng dẫn (có cấu trúc “là gì / vì sao / như thế nào / bước 1-2-3”).
- Nếu bài chỉ “cung cấp thông tin mới” mà không giải thích sâu → nghiêng về Know (Update/Keep engaged), không phải Educate.

GATE 3 (Give me perspective ưu tiên khi có PHÂN TÍCH/QUAN ĐIỂM):
- Chọn Give me perspective khi có bình luận, đánh giá, so sánh nhiều góc nhìn, phân tích nguyên nhân-hệ quả, phản biện, nhận định xu hướng (thường có giọng điệu “theo chuyên gia / nhìn từ / tranh luận / tác động / hệ lụy / vì vậy”).
- Nếu có phân tích là trọng tâm → KHÔNG chọn Educate/Update.

GATE 4 (Feel group ưu tiên khi mục tiêu là CẢM XÚC/THƯ GIÃN):
- Inspire me: câu chuyện truyền cảm hứng, vượt khó, thành tựu, nhân vật/đời sống tạo động lực.
- Divert me: giải trí, hài hước, đời sống nhẹ, thư giãn, showbiz, thời trang, mẹo vui (không phải hướng dẫn hành động nghiêm túc).
Nếu cảm xúc/giải trí là trọng tâm → KHÔNG chọn Update/Educate.

GATE 5 (Do group ưu tiên khi có KÊU GỌI HÀNH ĐỘNG/KẾT NỐI):
- Help me: lời khuyên thực tế, mẹo, khuyến nghị, checklist hành động cho người đọc (nên/không nên/làm thế nào để…).
- Connect me: kêu gọi tham gia, đăng ký, sự kiện, cộng đồng, tương tác, kết nối nguồn lực/địa điểm/đầu mối.
Nếu có CTA rõ → KHÔNG chọn Update/Educate.

**QUY TẮC khi phân vân:**

1. Nếu có hành động cụ thể → ưu tiên Help/Connect.
2. Nếu có phân tích, quan điểm hoặc đa góc nhìn → ưu tiên Give me perspective.
3. Nếu có giải thích/how-to có cấu trúc → Educate me.
4. Nếu là diễn biến tiếp theo của một câu chuyện đang theo dõi (phiên tòa tiếp theo, cập nhật vụ việc đã có trước đó, tập tiếp theo) → Keep me engaged.
5. Nếu không rơi vào các trường hợp trên và chủ yếu là tin mới → Update me.
---
### ĐẦU RA:
- Chỉ trả lời **duy nhất** dưới dạng **JSON hợp lệ**, **không thêm bất kỳ chữ nào khác**.
- Cấu trúc JSON:
{{
  "user_need": "một trong: trong 8 nhãn Smartocto 2.0"
}}

### Văn bản:
{article_text}
"""


def build_scoring_prompt(article_text: str) -> str:
    split_token = "\n2."
    full_prompt = build_prompt(article_text)
    if split_token not in full_prompt:
        raise ValueError("Split token for section 2 not found in prompt template.")

    _, scoring_section = full_prompt.split(split_token, 1)
    scoring_intro = f"2.{scoring_section}"
    output_anchor = "\n### ĐẦU RA:"
    if output_anchor in scoring_intro:
        scoring_intro = scoring_intro.split(output_anchor, 1)[0]

    return f"""{scoring_intro.rstrip()}
---
### ĐẦU RA:
- Chỉ trả lời **duy nhất** dưới dạng **JSON hợp lệ**, **không thêm bất kỳ chữ nào khác**.
- Cấu trúc JSON:
{{
  "I1": 1 hoặc 3 hoặc 5 hoặc 7 hoặc 9,
  "I3": 1 hoặc 3 hoặc 5 hoặc 7 hoặc 9,
  "I4": 1 hoặc 3 hoặc 5 hoặc 7 hoặc 9
}}
- Trước khi trả lời, hãy **tự kiểm tra lại**:
  - `I1`, `I3`, `I4` có nằm trong tập {{1, 3, 5, 7, 9}} hay không.
  - Nếu bất kỳ giá trị nào **không hợp lệ**, hãy **sửa lại** cho hợp lệ rồi mới xuất JSON.
---
### Văn bản:
{article_text}
"""


def build_prompts(article_text: str):
    return build_user_need_prompt(article_text), build_scoring_prompt(article_text)


def norm_output_open_ai(answer):
    norm_answer = ast.literal_eval(answer)

    return json.loads(norm_answer) 

def query_local(prompt: str) -> str:
    # infer using OLLAMA local server
    OLLAMA = False
    if OLLAMA:
        OLLAMA_URL = "http://localhost:11434/api/generate"
        MODEL_NAME = "qwen3:14b-q4_K_M"
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "seed": 4545,
                "max_tokens": 2000,
                "temperature": 0.1,
                "top_p": 0.95
            }
        }
        # print(f"Temperature used: {payload['options']['temperature']}")
        response = requests.post(OLLAMA_URL, json=payload)
        
        response.raise_for_status()
        return response.json()["response"]
    else:
        # infer using vLLM local server
        # print("Using OpenAI client for local inference...")
        model_name = "Qwen/Qwen3-14B-AWQ"
        base_url = "http://157.10.188.151:8808/v1"
        api_key = "123456"
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        system_prompt = """
        Bạn là trợ lý biên tập thông minh, nhiệm vụ của bạn là phân loại bài viết vào đúng 1 user need trong 8 nhóm Smartocto 2.0 và chấm ba chỉ số I1, I3, I4. Các giá trị I1/I3/I4 bắt buộc phải thuộc tập {1, 3, 5, 7, 9} và phải chọn mức gần nhất theo mô tả chuẩn. Luôn trả lời bằng **Định dạng JSON**, không thêm bất kỳ chữ nào ngoài JSON, với các trường: user_need, I1, I3, I4. Trước khi xuất kết quả, phải tự kiểm tra tất cả giá trị đều hợp lệ và đúng danh sách cho phép.
        """
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            seed=4545,
            temperature=0.1,
            max_tokens=2048,
            top_p=0.95
        )
        answer = response.choices[0].message.content
        import pdb
        # pdb.set_trace()
        return answer

def query_model(prompt: str) -> str:
    if local_infer:
        return query_local(prompt)
    else:
        return query_g4f(prompt)
    

def query_g4f(prompt: str) -> str:
    # print("Using G4F client for inference...")
    response = client.chat.completions.create(
        model="Qwen/Qwen3-14B",
        messages=[{"role": "user", "content": prompt}],
        web_search=False,
        api_key="PQEWyuBNCu0IkDoiPb2VsQyp3bx0euCy",
        provider="DeepInfra",
        temperature=0.1,
        # stream=True
    )
    
    return response.choices[0].message.content

def parse_json_output(raw_output: str):
    try:
        # Find the first JSON-like block
        start = raw_output.find("{")
        end = raw_output.rfind("}") + 1
        json_str = raw_output[start:end]
        parsed = json.loads(json_str)
        return parsed
    except Exception as e:
        print("⚠️  Failed to parse JSON:", e)
        print("Raw output was:")
        print(raw_output)
        return None

def run_user_need_and_scoring(context: str):
    user_need_prompt, scoring_prompt = build_prompts(context)
    import pdb
    
    # pdb.set_trace()
    user_need_raw = query_model(user_need_prompt)
    user_need_result = parse_json_output(user_need_raw)

    scoring_raw = query_model(scoring_prompt)
    scoring_result = parse_json_output(scoring_raw)

    # pdb.set_trace()

    merged_result = {}
    if isinstance(user_need_result, dict) and "user_need" in user_need_result:
        merged_result["user_need"] = user_need_result["user_need"]
    if isinstance(scoring_result, dict):
        for key in ("I1", "I3", "I4"):
            if key in scoring_result:
                merged_result[key] = scoring_result[key]
        if "user_need" not in merged_result and "user_need" in scoring_result:
            merged_result["user_need"] = scoring_result["user_need"]

    logging.info(
        f"Context: {repr(context)} ==> USER_NEED: {user_need_result} | SCORES: {scoring_result}"
    )

    return merged_result, {
        "user_need_raw": user_need_raw,
        "scoring_raw": scoring_raw,
        "user_need_response": user_need_result,
        "scoring_response": scoring_result,
    }

def single_query(article_id: int):
    api_data = get_article_data(int(article_id))
    context = build_input_data(api_data)
    # print(f"Context used: {context}")
    result, _ = run_user_need_and_scoring(context)
    print(json.dumps(result, indent=2, ensure_ascii=False))

def infer_test_file(test_path: str):
    output_dict = {}
    with open(test_path, "r") as f:
        articles = json.load(f)
        for key in articles["articles_id"].keys():
            print(key)
            output_dict.setdefault(key, [])
            for id in tqdm(articles["articles_id"][key]):

                api_data = get_article_data(int(id))
                context = build_input_data(api_data)

                url = api_data["data"]["share_url"] if api_data else None
                result, _ = run_user_need_and_scoring(context)
                output_dict[key].append({
                    "article_id": id,
                    "response": result,
                    "data": context,
                    "url": url
                })
                print(json.dumps(result, indent=2, ensure_ascii=False))
    suffix = "_".join(test_path.split('_')[-3:])
    with open(f"./data/qwen3_infer_{suffix}", "w", encoding="utf-8") as out_f:
        json.dump(output_dict, out_f, ensure_ascii=False, indent=4)
if __name__ == "__main__":
    # article_id = 4986601
    # single_query(article_id=article_id)
    
    test_path = "./data/test_list_12_12_2025_qc_selected.json"
    infer_test_file(test_path)
