import requests
import json
import sys
import logging
import logging.config
import yaml
OLLAMA_URL = "https://4d80090eef09.ngrok-free.app/api/generate"
MODEL_NAME = "qwen3:14b-q4_K_M"

with open(f"./logger/config.yaml", "r") as stream:
    log_config = yaml.safe_load(stream)
    # print("config=", json.dumps(config, indent=2, ensure_ascii=False))
    logging.config.dictConfig(log_config)


def build_prompt(article_text: str) -> str:
    return f"""Bạn là một trợ lý biên tập thông minh.  
Nhiệm vụ của bạn là phân tích một bài báo/ngữ liệu và:

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
- `7`: Gợi cảm xúc rõ rệt ở người đọc (tò mò, phấn khích, thương cảm…).  
- `9`: Gây cảm xúc mạnh mẽ, dễ lan truyền, có thể dẫn tới hành động cộng đồng.

#### I3 – Khả năng tạo tranh luận xã hội
- `1`: Không tạo tranh luận.  
- `3`: Có khả năng được chia sẻ nhưng không gây tranh luận.  
- `5`: Có thể tạo bình luận cá nhân, không thành làn sóng.  
- `7`: Có thể kích hoạt thảo luận trong một cộng đồng cụ thể.  
- `9`: Dễ trở thành chủ đề nóng, tranh cãi rộng rãi trên mạng/xã hội.

#### I4 – Liên quan đến thay đổi chính sách/xã hội
- `1`: Rất thấp/Không liên quan (giải trí, tin nhẹ, sở thích cá nhân…).  
- `3`: Liên quan thấp (tin tiêu dùng/địa phương, liên hệ gián tiếp đến chính sách/xã hội).  
- `5`: Liên quan trung bình (đề cập vấn đề công cộng nhưng chưa đi sâu chính sách/cải cách).  
- `7`: Liên quan cao (phân tích/thảo luận trực tiếp về chính sách, luật, xu hướng xã hội lớn).  
- `9`: Rất cao (nghiên cứu chuyên sâu, điều tra, phân tích chính sách có tiềm năng thúc đẩy thay đổi).

---

### ĐẦU RA:
- Chỉ trả lời **duy nhất** dưới dạng **JSON hợp lệ**, **không thêm bất kỳ chữ nào khác**.  
- Cấu trúc JSON:
{{
  "user_need": "một trong: Update me, Keep me engaged, Educate me, Give me perspective, Inspire me, Divert me, Help me, Connect me",
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

def query_ollama(prompt: str) -> str:
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
    print(f"Temperature used: {payload['options']['temperature']}")
    response = requests.post(OLLAMA_URL, json=payload)
    
    response.raise_for_status()
    return response.json()["response"]

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
from utils import strip_html_tags_regex, build_input_data
if __name__ == "__main__":
    article_id = sys.argv[1]
    context = build_input_data(int(article_id))
    
    # with open(sys.argv[1], "r", encoding="utf-8") as f:
    #     metadata = json.load(f)
    #     title = metadata["data"]["title"]
    #     content = strip_html_tags_regex(metadata["data"]["content"])
    #     article_text = title + "\n\n" + content
    print(f"Context used: {context}")
    prompt = build_prompt(context)
    # print(f"📝 Prompt: \n{prompt}")
    raw_output = query_ollama(prompt)
    try:
        result = parse_json_output(raw_output)
        logging.info(f"Context: {repr(context)} ==> RESPONSE: {result}")
    except:
        logging.error(f"Context: {repr(context)}", exc_info=True)

    # print("\n✅ LLM Prediction:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
