"""
Analyze prediction vs ground truth alignment.

This script surfaces the strongest matches, worst mismatches, common
user-need pairings, and confusion matrices between the model outputs
and the human scores.
"""

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence, Tuple

from evaluate import (
    IMPACT_LEVELS,
    USER_NEED_GROUPS,
    evaluate_dataset,
    load_json,
    user_need_group,
)
import matplotlib.pyplot as plt


DEFAULT_PREDICTIONS = Path("data/qwen3_infer_2025_qc_selected.json")
DEFAULT_GROUND_TRUTH = Path("scores_trung.json")


def format_case(entry: Mapping) -> str:
    model_resp = entry["model_response"]
    gt_resp = entry["ground_truth"]
    return (
        f"{entry['article_id']}: final_score={entry['final_score']:.3f} | "
        f"user_need model={model_resp['user_need']}, human={gt_resp['user_need']} | "
        f"impacts model={{'I1': {int(model_resp['I1'])}, 'I3': {int(model_resp['I3'])}, 'I4': {int(model_resp['I4'])}}}, "
        f"human={{'I1': {int(gt_resp['I1'])}, 'I3': {int(gt_resp['I3'])}, 'I4': {int(gt_resp['I4'])}}}"
    )


def top_and_bottom(results: List[Mapping], n: int) -> Tuple[List[Mapping], List[Mapping]]:
    ordered = sorted(results, key=lambda r: r["final_score"])
    worst = ordered[:n]
    best = list(reversed(ordered[-n:]))
    return best, worst


def most_common_pairs(results: Iterable[Mapping], n: int) -> List[Tuple[Tuple[str, str], int]]:
    counter: Counter = Counter()
    for entry in results:
        model_need = entry["model_response"]["user_need"]
        gt_need = entry["ground_truth"]["user_need"]
        counter[(model_need, gt_need)] += 1
    return counter.most_common(n)


def build_confusion(
    results: Iterable[Mapping],
    labels: Sequence[str],
    model_value_fn,
    gt_value_fn,
) -> List[List[int]]:
    size = len(labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    matrix = [[0 for _ in range(size)] for _ in range(size)]

    for entry in results:
        model_val = model_value_fn(entry)
        gt_val = gt_value_fn(entry)
        if model_val not in label_to_idx or gt_val not in label_to_idx:
            continue
        matrix[label_to_idx[gt_val]][label_to_idx[model_val]] += 1
    return matrix


def plot_matrix(title: str, labels: Sequence[str], matrix: List[List[int]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Thao")
    ax.set_ylabel("Trung")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # Annotate cells with counts
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, matrix[i][j], ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze best/worst matches between predictions and ground truth.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS, help="Path to model prediction JSON.")
    parser.add_argument("--ground-truth", dest="ground_truth", type=Path, default=DEFAULT_GROUND_TRUTH, help="Path to ground truth JSON.")
    parser.add_argument("--top", type=int, default=5, help="Number of top and bottom cases to show.")
    parser.add_argument("--pairs", type=int, default=5, help="Number of most common user_need pairings to show.")
    parser.add_argument("--out-dir", type=Path, default=Path("plots-trung"), help="Directory to write confusion matrix images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    preds = load_json(args.predictions)
    gt = load_json(args.ground_truth)

    evaluation = evaluate_dataset(preds, gt)
    summary = evaluation["summary"]
    results = evaluation["results"]

    best, worst = top_and_bottom(results, args.top)
    pairs = most_common_pairs(results, args.pairs)

    print(f"Evaluated articles: {summary['evaluated']}")
    print(f"Missing in predictions: {summary['missing_in_predictions']}")
    print(f"Missing in ground truth: {summary['missing_in_ground_truth']}")

    print(f"\nTop {args.top} matches:")
    for entry in best:
        print(f"  - {format_case(entry)}")

    print(f"\nWorst {args.top} mismatches:")
    for entry in worst:
        print(f"  - {format_case(entry)}")

    print(f"\nMost common user_need pairings (model, human):")
    for (model_need, human_need), count in pairs:
        print(f"  - {model_need} -> {human_need}: {count}")

    # Confusion matrices
    user_need_labels = list(USER_NEED_GROUPS.keys())
    group_labels = ["Know", "Understand", "Feel", "Do"]
    impact_labels = [str(level) for level in IMPACT_LEVELS]

    user_need_conf = build_confusion(
        results,
        user_need_labels,
        model_value_fn=lambda e: e["model_response"]["user_need"],
        gt_value_fn=lambda e: e["ground_truth"]["user_need"],
    )
    group_conf = build_confusion(
        results,
        group_labels,
        model_value_fn=lambda e: user_need_group(e["model_response"]["user_need"]),
        gt_value_fn=lambda e: user_need_group(e["ground_truth"]["user_need"]),
    )
    impact_conf = {
        key: build_confusion(
            results,
            impact_labels,
            model_value_fn=lambda e, k=key: str(int(e["model_response"][k])),
            gt_value_fn=lambda e, k=key: str(int(e["ground_truth"][k])),
        )
        for key in ("I1", "I3", "I4")
    }

    plot_matrix("User need confusion (8x8)", user_need_labels, user_need_conf, args.out_dir / "user_need_confusion.png")
    plot_matrix("Group confusion (4x4)", group_labels, group_conf, args.out_dir / "group_confusion.png")
    for key, matrix in impact_conf.items():
        plot_matrix(f"{key} impact confusion (5x5)", impact_labels, matrix, args.out_dir / f"{key}_confusion.png")
    print(f"\nConfusion matrices saved to {args.out_dir}")


if __name__ == "__main__":
    main()
