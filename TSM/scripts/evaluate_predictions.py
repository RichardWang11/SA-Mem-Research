#!/usr/bin/env python3
"""Deterministic diagnostic EM/token-F1 and official-evaluator export."""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path


def normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, answer: str) -> float:
    predicted = normalize(prediction).split()
    gold = normalize(answer).split()
    if not predicted or not gold:
        return float(predicted == gold)
    overlap = sum((Counter(predicted) & Counter(gold)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def flatten(payload: dict) -> list[dict]:
    rows = []
    for key, value in payload.items():
        if key == "config_info" or not isinstance(value, list):
            continue
        rows.extend(item for item in value if isinstance(item, dict) and "question_id" in item)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--export-jsonl", type=Path)
    args = parser.parse_args()
    rows = flatten(json.loads(args.results.read_text(encoding="utf-8")))
    if not rows:
        raise SystemExit("no prediction rows found")

    scores = []
    exports = []
    for row in rows:
        prediction = str(row.get("hypothesis", row.get("output_text", "")))
        answer = str(row.get("answer", ""))
        scores.append(
            {
                "question_id": row["question_id"],
                "exact_match": float(normalize(prediction) == normalize(answer)),
                "token_f1": token_f1(prediction, answer),
            }
        )
        exports.append({"question_id": row["question_id"], "prediction": prediction})

    report = {
        "metric_scope": "diagnostic_only_not_official_benchmark_score",
        "count": len(scores),
        "exact_match": sum(item["exact_match"] for item in scores) / len(scores),
        "token_f1": sum(item["token_f1"] for item in scores) / len(scores),
        "per_question": scores,
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    if args.export_jsonl:
        args.export_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.export_jsonl.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in exports),
            encoding="utf-8",
        )
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
