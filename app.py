from pathlib import Path
from typing import Optional, Tuple

import logging
import logging.config
import requests
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from qwen3_infer import build_prompt, query_model, parse_json_output, local_infer
from utils import build_input_data, get_article_data

LOG_CONFIG_PATH = Path(__file__).parent / "logger" / "config.yaml"


def setup_logging() -> None:
    if logging.getLogger().handlers:
        return

    if LOG_CONFIG_PATH.exists():
        with open(LOG_CONFIG_PATH, "r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)


setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Qwen3 Inference API",
    description="Expose the qwen3 inference pipeline over HTTP.",
    version="1.0.0",
)


class InferRequest(BaseModel):
    article_id: Optional[int] = None
    text: Optional[str] = None



def run_inference(article_text: str) -> Tuple[dict, str]:
    if not article_text:
        raise ValueError("Article text is empty.")

    prompt = build_prompt(article_text)
    raw_output = query_model(prompt)
    parsed = parse_json_output(raw_output)

    if parsed is None:
        raise ValueError("Model response could not be parsed as JSON.")

    return parsed, raw_output


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/infer")
def infer(request: InferRequest):
    try:
        if request.article_id is not None:
            api_data = get_article_data(request.article_id)
            if not api_data:
                raise HTTPException(
                    status_code=404,
                    detail="Article not found or upstream returned no data.",
                )
            article_text = build_input_data(api_data)
            if not article_text:
                raise HTTPException(
                    status_code=404,
                    detail="Article content is empty.",
                )
            source = "article"
        else:
            text = (request.text or "").strip()
            if not text:
                raise HTTPException(
                    status_code=400,
                    detail="Provided text is empty.",
                )
            article_text = text
            source = "text"

        parsed, raw_output = run_inference(article_text)

    except HTTPException:
        raise
    except requests.exceptions.RequestException as exc:
        logger.exception("Network error during inference")
        raise HTTPException(
            status_code=502,
            detail="Network error while contacting upstream services.",
        ) from exc
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "source": source,
        "article_id": request.article_id,
        "result": parsed,
        "raw_response": raw_output,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
