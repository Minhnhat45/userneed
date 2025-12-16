"""Evaluate model outputs against ground truth using Smartocto rules.

This script compares the model inference dump at ``data/qwen3_infer_2025_qc_selected.json``
with the corresponding ground truth file ``data/qwen3_infer_2025_qc_selected_ground_truth.json``.
It implements the scoring rubric provided in the task description:

- User need: exact match = 2, same group = 1, different group = 0.
- Impact metrics (I1, I3, I4): exact = 2, adjacent level = 1, otherwise = 0.
- score_emotion = (score_I1 + score_I3 + score_I4) / 6
- final_score = score_userneed + score_emotion

Run ``python evaluate.py`` to see the aggregate scores, or provide custom paths via CLI flags.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping


USER_NEED_GROUPS: Dict[str, str] = {
    "Update me": "Know",
    "Keep me engaged": "Know",
    "Educate me": "Understand",
    "Give me perspective": "Understand",
    "Inspire me": "Feel",
    "Divert me": "Feel",
    "Help me": "Do",
    "Connect me": "Do",
}

IMPACT_LEVELS = (1, 3, 5, 7, 9)
IMPACT_INDEX = {value: idx for idx, value in enumerate(IMPACT_LEVELS)}


def load_json(path: Path) -> MutableMapping[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def flatten_articles(dataset: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Flatten category -> items structure into an article_id keyed mapping."""

    index: Dict[int, Dict[str, Any]] = {}
    for category, items in dataset.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            if "article_id" not in item or "response" not in item:
                continue

            article_id = int(item["article_id"])
            if article_id in index:
                raise ValueError(f"Duplicate article_id detected: {article_id}")

            index[article_id] = {
                "category": category,
                "response": item["response"],
                "raw": item,
            }

    return index


def user_need_group(user_need: str) -> str:
    try:
        return USER_NEED_GROUPS[user_need]
    except KeyError as exc:
        raise ValueError(f"Unknown user need label: {user_need!r}") from exc


def score_user_need(model_userneed: str, gt_userneed: str) -> int:
    if model_userneed == gt_userneed:
        return 2

    model_group = user_need_group(model_userneed)
    gt_group = user_need_group(gt_userneed)

    return 1 if model_group == gt_group else 0


def score_impact_metric(model_value: int, gt_value: int) -> int:
    if model_value not in IMPACT_INDEX:
        raise ValueError(f"Invalid model impact value: {model_value!r}")
    if gt_value not in IMPACT_INDEX:
        raise ValueError(f"Invalid ground truth impact value: {gt_value!r}")

    distance = abs(IMPACT_INDEX[model_value] - IMPACT_INDEX[gt_value])
    if distance == 0:
        return 2
    if distance == 1:
        return 1
    return 0


def evaluate_response(model_response: Mapping[str, Any], gt_response: Mapping[str, Any]) -> Dict[str, float]:
    model_need = model_response["user_need"]
    gt_need = gt_response["user_need"]
    score_need = score_user_need(model_need, gt_need)

    score_i1 = score_impact_metric(int(model_response["I1"]), int(gt_response["I1"]))
    score_i3 = score_impact_metric(int(model_response["I3"]), int(gt_response["I3"]))
    score_i4 = score_impact_metric(int(model_response["I4"]), int(gt_response["I4"]))

    score_emotion = (score_i1 + score_i3 + score_i4) / 6
    final_score = score_need + score_emotion

    return {
        "score_userneed": score_need,
        "score_I1": score_i1,
        "score_I3": score_i3,
        "score_I4": score_i4,
        "score_emotion": score_emotion,
        "final_score": final_score,
    }


def evaluate_dataset(model_dataset: Mapping[str, Any], gt_dataset: Mapping[str, Any]) -> Dict[str, Any]:
    model_index = flatten_articles(model_dataset)
    gt_index = flatten_articles(gt_dataset)

    common_ids = sorted(set(model_index).intersection(gt_index))
    missing_in_model = sorted(set(gt_index) - set(model_index))
    missing_in_gt = sorted(set(model_index) - set(gt_index))

    results = []
    totals = {
        "score_userneed": 0.0,
        "score_I1": 0.0,
        "score_I3": 0.0,
        "score_I4": 0.0,
        "score_emotion": 0.0,
        "final_score": 0.0,
    }

    for article_id in common_ids:
        model_resp = model_index[article_id]["response"]
        gt_resp = gt_index[article_id]["response"]
        scores = evaluate_response(model_resp, gt_resp)

        results.append({
            "article_id": article_id,
            "category": model_index[article_id]["category"],
            "model_response": model_resp,
            "ground_truth": gt_resp,
            **scores,
        })

        for key in totals:
            totals[key] += scores[key]

    count = len(common_ids)
    averages = {key: (totals[key] / count) if count else 0.0 for key in totals}

    summary = {
        "evaluated": count,
        "missing_in_predictions": missing_in_model,
        "missing_in_ground_truth": missing_in_gt,
        "averages": averages,
        "totals": totals,
    }

    return {"summary": summary, "results": results}


def format_score(value: float, maximum: float) -> str:
    percent = (value / maximum * 100) if maximum else 0.0
    return f"{value:.3f} / {maximum:g} | {percent:.0f}%"


def print_summary(evaluation: Mapping[str, Any]) -> None:
    summary = evaluation["summary"]
    averages = summary["averages"]

    print(f"Evaluated articles: {summary['evaluated']}")
    if summary["missing_in_predictions"]:
        print(f"Missing in predictions: {summary['missing_in_predictions']}")
    if summary["missing_in_ground_truth"]:
        print(f"Missing in ground truth: {summary['missing_in_ground_truth']}")

    print("\nAverage scores:")
    print(f"  user_need      : {format_score(averages['score_userneed'], 2)}")
    print(f"  I1             : {format_score(averages['score_I1'], 2)}")
    print(f"  I3             : {format_score(averages['score_I3'], 2)}")
    print(f"  I4             : {format_score(averages['score_I4'], 2)}")
    print(f"  score_emotion  : {format_score(averages['score_emotion'], 1)}")
    print(f"  final_score    : {format_score(averages['final_score'], 3)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model outputs against ground truth.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("./data/qwen3_infer_2025_qc_selected.json"),
        help="Path to the model prediction JSON file.",
    )
    parser.add_argument(
        "--ground-truth",
        dest="ground_truth",
        type=Path,
        default=Path("./data/qwen3_infer_2025_qc_selected_ground_truth.json"),
        help="Path to the ground truth JSON file.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Optional path to write detailed per-article scores as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_dataset = load_json(args.predictions)
    gt_dataset = load_json(args.ground_truth)

    evaluation = evaluate_dataset(model_dataset, gt_dataset)
    print_summary(evaluation)

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        with args.save.open("w", encoding="utf-8") as f:
            json.dump(evaluation, f, ensure_ascii=False, indent=2)
        print(f"\nDetailed results written to {args.save}")


if __name__ == "__main__":
    main()
