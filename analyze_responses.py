"""Summarize response values in a qwen3 inference JSON dump."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def iter_responses(data: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield each response dict from the nested category -> items structure."""
    for items in data.values():
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict) and isinstance(item.get("response"), dict):
                yield item["response"]


def analyze_responses(data: Dict[str, Any]) -> Tuple[int, Dict[str, Counter], Counter]:
    field_counts: Dict[str, Counter] = defaultdict(Counter)
    combo_counts: Counter = Counter()
    total = 0

    for resp in iter_responses(data):
        total += 1
        combo_counts[tuple(sorted(resp.items()))] += 1
        for key, value in resp.items():
            field_counts[key][value] += 1

    return total, field_counts, combo_counts


def format_combo(combo: Tuple[Tuple[str, Any], ...]) -> str:
    return "{" + ", ".join(f"{k}: {v!r}" for k, v in combo) + "}"


def main() -> None:

    json_path = Path("./data/qwen3_infer_27_11_2025.json")

    data = load_json(json_path)
    total, field_counts, combo_counts = analyze_responses(data)

    print(f"Total response rows: {total}")
    print("\nCounts per response field:")
    for field in sorted(field_counts):
        for value, count in field_counts[field].most_common():
            print(f"  {field}={value}: {count}")

    print("\nCounts per full response payload:")
    for combo, count in combo_counts.most_common():
        print(f"  {format_combo(combo)}: {count}")


if __name__ == "__main__":
    main()
