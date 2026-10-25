#!/usr/bin/env python3
"""Create an auditable LoCoMo query manifest without shrinking the memory corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_evidence_ids(values: list[Any]) -> list[str]:
    """Expand the one upstream multi-message evidence string into message IDs."""
    result = []
    for value in values:
        result.extend(part.strip() for part in str(value).split(";") if part.strip())
    return result


def dialogue_index(item: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    conversation = item.get("conversation", {})
    session_keys = sorted(
        (key for key in conversation if key.startswith("session_") and not key.endswith("_date_time")),
        key=lambda key: int(key.split("_")[1]),
    )
    for session_key in session_keys:
        for message in conversation.get(session_key, []):
            dialogue_id = message.get("dia_id")
            if not dialogue_id:
                continue
            if dialogue_id in index:
                raise ValueError(f"duplicate dialogue ID {dialogue_id} in {item.get('sample_id')}")
            index[dialogue_id] = {
                "dialogue_id": dialogue_id,
                "session": session_key,
                "speaker": message.get("speaker"),
                "text": message.get("text", ""),
            }
    return index


def build_qa_index(dataset: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in dataset:
        sample_id = item.get("sample_id")
        if not sample_id:
            raise ValueError("LoCoMo item has no sample_id")
        evidence_index = dialogue_index(item)
        for qa_index, qa in enumerate(item.get("qa", [])):
            qid = f"{sample_id}_{qa_index}"
            if qid in result:
                raise ValueError(f"duplicate generated qid {qid}")
            evidence_raw = qa.get("evidence", []) or []
            evidence = normalize_evidence_ids(evidence_raw)
            evidence_records = []
            unresolved_evidence_ids = []
            for evidence_id in evidence:
                if evidence_id not in evidence_index:
                    unresolved_evidence_ids.append(evidence_id)
                    continue
                evidence_records.append(evidence_index[evidence_id])
            result[qid] = {
                "qid": qid,
                "conversation_id": sample_id,
                "qa_index": qa_index,
                "question": qa.get("question", ""),
                "answer": qa.get("answer"),
                "category": qa.get("category"),
                "gold_evidence_raw": evidence_raw,
                "gold_evidence_ids": evidence,
                "gold_dialogue_evidence": evidence_records,
                "unresolved_gold_evidence_ids": unresolved_evidence_ids,
                "mapping_status": "mapping_unknown" if unresolved_evidence_ids else "dialogue_id_validated",
                "conversation_message_count": len(evidence_index),
                "conversation_session_count": sum(
                    key.startswith("session_") and not key.endswith("_date_time")
                    for key in item.get("conversation", {})
                ),
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    annotations = list(csv.DictReader(args.annotations.open(encoding="utf-8-sig", newline="")))
    annotation_by_qid = {row["qid"]: row for row in annotations if row.get("dataset") == "locomo"}
    if len(annotation_by_qid) != 1986:
        raise ValueError(f"expected 1986 unique LoCoMo annotations, got {len(annotation_by_qid)}")

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    qa_by_qid = build_qa_index(dataset)
    if set(annotation_by_qid) != set(qa_by_qid):
        missing = sorted(set(annotation_by_qid) - set(qa_by_qid))
        extra = sorted(set(qa_by_qid) - set(annotation_by_qid))
        raise ValueError(f"qid mismatch: missing={missing[:5]} extra={extra[:5]}")

    selected = []
    for qid, annotation in annotation_by_qid.items():
        if annotation.get("jev_label") != "explicit_temporal_filter":
            continue
        record = dict(qa_by_qid[qid])
        record.update(
            {
                "dataset": annotation.get("dataset"),
                "scope": annotation.get("scope"),
                "official_type": annotation.get("official_type"),
                "main_subset": as_bool(annotation.get("main_subset", "false")),
                "reference_date": annotation.get("reference_date") or None,
                "jev_label": annotation.get("jev_label"),
                "jev_selected_probability": annotation.get("jev_selected_probability"),
                "tsm_active": as_bool(annotation.get("tsm_active", "false")),
                "review_flags": annotation.get("review_flags", ""),
            }
        )
        selected.append(record)

    selected.sort(key=lambda record: (record["conversation_id"], record["qa_index"]))
    if len(selected) != 321:
        raise ValueError(f"expected 321 explicit temporal filter queries, got {len(selected)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "locomo_explicit_temporal_queries.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    filter_path = args.out_dir / "locomo_explicit_temporal_qids.json"
    filter_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "locomo",
                "selection": {"jev_label": "explicit_temporal_filter"},
                "selected_qids": [record["qid"] for record in selected],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "dataset_path": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "annotation_path": str(args.annotations.resolve()),
        "annotation_sha256": sha256(args.annotations),
        "full_locomo_questions": len(qa_by_qid),
        "selected_questions": len(selected),
        "selected_main_subset": sum(record["main_subset"] for record in selected),
        "selected_adversarial": sum(not record["main_subset"] for record in selected),
        "selected_tsm_active": sum(record["tsm_active"] for record in selected),
        "official_type_counts": dict(Counter(record["official_type"] for record in selected)),
        "conversation_counts": dict(Counter(record["conversation_id"] for record in selected)),
        "mapping_status": {
            "dialogue_evidence_ids_validated": sum(
                record["mapping_status"] == "dialogue_id_validated" for record in selected
            ),
            "mapping_unknown": sum(
                record["mapping_status"] == "mapping_unknown" for record in selected
            ),
            "unresolved_evidence_references": sum(
                len(record["unresolved_gold_evidence_ids"]) for record in selected
            ),
        },
        "manifest": str(manifest_path.resolve()),
        "filter": str(filter_path.resolve()),
    }
    (args.out_dir / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
