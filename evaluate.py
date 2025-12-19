"""Evaluate model outputs against dual human scores using Smartocto rules.

This script compares the model inference dump at ``data/qwen3_infer_2025_qc_selected.json``
against two sets of human annotations: ``scores_trung.json`` and ``scores_thao.json``.
A prediction is treated as correct when it matches either annotator for a field.
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
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Tuple, Union


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
DEFAULT_GROUND_TRUTHS = (Path("scores_trung.json"), Path("scores_thao.json"))


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


def evaluate_response(model_response: Mapping[str, Any], gt_responses: Iterable[Mapping[str, Any]]) -> Tuple[Mapping[str, Any], Dict[str, float]]:
    gt_list: List[Mapping[str, Any]] = list(gt_responses)
    if not gt_list:
        raise ValueError("No ground truth responses provided for evaluation.")

    model_need = model_response["user_need"]

    best_need_score = -1
    best_need_value = None
    for gt in gt_list:
        candidate_need = gt["user_need"]
        score = score_user_need(model_need, candidate_need)
        if score > best_need_score:
            best_need_score = score
            best_need_value = candidate_need

    def best_impact(metric: str) -> Tuple[int, int]:
        best_score = -1
        best_value = None
        model_value = int(model_response[metric])
        for gt in gt_list:
            candidate_value = int(gt[metric])
            score = score_impact_metric(model_value, candidate_value)
            if score > best_score:
                best_score = score
                best_value = candidate_value
        if best_score < 0 or best_value is None:
            raise ValueError(f"No ground truth values found for metric {metric}")
        return best_score, best_value

    score_i1, best_i1_value = best_impact("I1")
    score_i3, best_i3_value = best_impact("I3")
    score_i4, best_i4_value = best_impact("I4")

    score_emotion = (score_i1 + score_i3 + score_i4) / 6
    score_need = best_need_score
    final_score = score_need + score_emotion

    resolved_ground_truth = {
        "user_need": best_need_value,
        "I1": best_i1_value,
        "I3": best_i3_value,
        "I4": best_i4_value,
    }

    scores = {
        "score_userneed": score_need,
        "score_I1": score_i1,
        "score_I3": score_i3,
        "score_I4": score_i4,
        "score_emotion": score_emotion,
        "final_score": final_score,
    }

    return resolved_ground_truth, scores


def evaluate_dataset(model_dataset: Mapping[str, Any], gt_datasets: Union[Iterable[Mapping[str, Any]], Mapping[str, Any]]) -> Dict[str, Any]:
    if isinstance(gt_datasets, Mapping):
        gt_sources = [gt_datasets]
    else:
        gt_sources = list(gt_datasets)
    if not gt_sources:
        raise ValueError("At least one ground truth dataset is required.")

    model_index = flatten_articles(model_dataset)
    gt_indices = [flatten_articles(gt) for gt in gt_sources]

    gt_union_ids = set()
    for index in gt_indices:
        gt_union_ids.update(index.keys())

    common_ids = sorted(set(model_index).intersection(gt_union_ids))
    missing_in_model = sorted(gt_union_ids - set(model_index))
    missing_in_gt = sorted(set(model_index) - gt_union_ids)

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
        gt_candidates = [idx[article_id]["response"] for idx in gt_indices if article_id in idx]
        resolved_gt, scores = evaluate_response(model_resp, gt_candidates)

        results.append({
            "article_id": article_id,
            "category": model_index[article_id]["category"],
            "model_response": model_resp,
            "ground_truth": resolved_gt,
            "ground_truth_candidates": gt_candidates,
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
    parser = argparse.ArgumentParser(description="Evaluate model outputs against dual ground truth files.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("./data/qwen3_infer_2025_qc_selected.json"),
        help="Path to the model prediction JSON file.",
    )
    parser.add_argument(
        "--ground-truths",
        dest="ground_truths",
        type=Path,
        nargs="+",
        default=list(DEFAULT_GROUND_TRUTHS),
        help="One or more ground truth JSON files to use (defaults to scores_trung.json and scores_thao.json).",
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
    gt_datasets = [load_json(path) for path in args.ground_truths]

    evaluation = evaluate_dataset(model_dataset, gt_datasets)
    print_summary(evaluation)

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        with args.save.open("w", encoding="utf-8") as f:
            json.dump(evaluation, f, ensure_ascii=False, indent=2)
        print(f"\nDetailed results written to {args.save}")


if __name__ == "__main__":
    main()
