#!/usr/bin/env python3
"""Offline acceptance audit for the completed TSM LoCoMo run.

This script only reads the TSM run, dataset, annotations, and source checkout.
It writes auditable artifacts into the acceptance directory and does not modify
the experiment output.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--dataset', type=Path, required=True)
parser.add_argument('--config', type=Path, required=True)
parser.add_argument('--cache-date', default='20261021')
parser.add_argument('--annotations', type=Path, default=Path(__file__).resolve().parents[1]/'selection/annotations_locomo1986.csv')
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
ROOT=args.out.resolve()
RESEARCH=Path(__file__).resolve().parents[1]
TSM=RESEARCH/'TSM'
RUN=args.run.resolve()
DATASET=args.dataset.resolve()
ANNOTATIONS=args.annotations.resolve()
result_files=list((RUN/'answers/Locomo').glob('*results.json'))
if len(result_files)!=1: raise SystemExit('Expected exactly one QA result file')
RESULTS=result_files[0]
TRACE_DIR=RUN/'request_trace'
CACHE_ROOT=RUN/'cache/locomo10'/args.cache_date
CONFIG=args.config.resolve()



def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_text(*args: str, cwd: Path = TSM) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def norm(text: str) -> str:
    text = str(text).replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"\s+", " ", text.lower()).strip()
    return text


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def result_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for value in payload.values() if isinstance(value, list) for row in value]


def load_source(dataset: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    samples = {item["sample_id"]: item for item in dataset}
    qa_by_qid: dict[str, dict[str, Any]] = {}
    for item in dataset:
        for idx, qa in enumerate(item.get("qa", [])):
            qa_by_qid[f"{item['sample_id']}_{idx}"] = qa
    return samples, qa_by_qid


def dialogue_index(sample: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    conversation = sample["conversation"]
    for key, messages in conversation.items():
        if not key.startswith("session_") or key.endswith("_date_time"):
            continue
        date = conversation.get(f"{key}_date_time")
        for message in messages:
            if isinstance(message, dict) and message.get("dia_id"):
                out[message["dia_id"]] = {
                    "dia_id": message["dia_id"],
                    "speaker": message.get("speaker"),
                    "text": message.get("text", ""),
                    "session": key,
                    "date": date,
                }
    return out


def find_leaf_hits(text: str, tree: dict[str, Any]) -> list[str]:
    needle = norm(text)
    return [leaf_id for leaf_id, node in tree.items() if needle and needle in norm(node.get("text", ""))]


def find_edge_hits(text: str, text_edges: dict[str, Any]) -> list[str]:
    needle = norm(text)
    return [edge_id for edge_id, edge in text_edges.items() if needle and needle in norm(edge.get("text", ""))]


def map_gold(
    row: dict[str, Any],
    qa: dict[str, Any],
    sample: dict[str, Any],
    tree: dict[str, Any],
    text_edges: dict[str, Any],
) -> dict[str, Any]:
    dia = dialogue_index(sample)
    evidence_ids = list(qa.get("evidence", []))
    records = []
    for evidence_id in evidence_ids:
        message = dia.get(evidence_id)
        if not message:
            records.append({"dia_id": evidence_id, "status": "mapping_unknown"})
            continue
        leaf_ids = find_leaf_hits(message["text"], tree)
        edge_ids = find_edge_hits(message["text"], text_edges)
        records.append({
            "dia_id": evidence_id,
            "session": message["session"],
            "date": message["date"],
            "speaker": message["speaker"],
            "text": message["text"],
            "leaf_ids": leaf_ids,
            "graphiti_edge_uuids": edge_ids,
            "status": (
                "mapped_to_leaf" if leaf_ids else
                "source_present_graphiti_missing_dense_leaf" if edge_ids else
                "mapping_unknown"
            ),
        })

    all_leaf_ids = sorted({leaf for record in records for leaf in record.get("leaf_ids", [])})
    all_edge_ids = sorted({edge for record in records for edge in record.get("graphiti_edge_uuids", [])})
    statuses = {record.get("status") for record in records}
    if not records:
        status = "no_gold_evidence"
    elif "mapping_unknown" in statuses:
        status = "mapping_unknown"
    elif "source_present_graphiti_missing_dense_leaf" in statuses:
        status = "source_present_graphiti_missing_dense_leaf"
    else:
        status = "mapped"

    trace = row.get("retrieval_trace") or {}
    dense20 = list(dict.fromkeys(trace.get("dense_top20_ids", [])))
    dense100 = list(dict.fromkeys(trace.get("dense_top100_ids", [])))
    final_context = set(trace.get("final_context_ids", []))
    mapped = bool(all_leaf_ids) and status == "mapped"
    return {
        "qid": row["question_id"],
        "question": row.get("question"),
        "official_type": qa.get("category"),
        "gold_answer": row.get("answer"),
        "gold_evidence_ids": evidence_ids,
        "mapping_status": status,
        "gold_leaf_ids": all_leaf_ids,
        "gold_graphiti_edge_uuids": all_edge_ids,
        "evidence_records": records,
        "dense_top20_unique_ids": dense20,
        "dense_top100_unique_ids": dense100,
        "final_context_ids": sorted(final_context),
        "gold_any_in_dense20": bool(set(all_leaf_ids) & set(dense20)) if mapped else None,
        "gold_all_in_dense20": bool(set(all_leaf_ids) <= set(dense20)) if mapped else None,
        "gold_any_in_dense100": bool(set(all_leaf_ids) & set(dense100)) if mapped else None,
        "gold_all_in_dense100": bool(set(all_leaf_ids) <= set(dense100)) if mapped else None,
        "gold_any_in_final_context": bool(set(all_leaf_ids) & final_context) if mapped else None,
        "gold_all_in_final_context": bool(set(all_leaf_ids) <= final_context) if mapped else None,
        "parser_hit": trace.get("tsm_parser_hit"),
        "parsed_time_ranges": trace.get("parsed_time_ranges", []),
        "graphiti_matched_edge_count": len(trace.get("graphiti_matched_edges", [])),
        "temporal_matched_chunk_ids": trace.get("temporal_matched_chunk_ids", []),
    }


def trace_audit() -> dict[str, Any]:
    status = Counter()
    stages = Counter()
    errors = []
    token_totals = defaultdict(int)
    trace_count = 0
    for path in TRACE_DIR.glob("*.json"):
        try:
            data = read_json(path)
        except Exception as exc:
            errors.append({"file": path.name, "kind": "invalid_json", "error": str(exc)})
            continue
        trace_count += 1
        status[data.get("status", "missing")] += 1
        stages[data.get("stage", "missing")] += 1
        if data.get("status") != "completed":
            errors.append({"file": path.name, "status": data.get("status"), "stage": data.get("stage"), "error": data.get("error")})
        usage = (data.get("response") or {}).get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, (int, float)):
                token_totals[f"{data.get('stage', 'missing')}.{key}"] += value
        if "total_tokens" not in usage and isinstance(usage.get("prompt_tokens"), (int, float)):
            token_totals[f"{data.get('stage', 'missing')}.total_tokens_derived"] += usage["prompt_tokens"] + usage.get("completion_tokens", 0)
    return {"trace_files": trace_count, "status": dict(status), "stages": dict(stages), "non_completed": errors, "token_totals": dict(token_totals)}


def main() -> None:
    if ROOT.exists():
        raise SystemExit("Output directory exists; choose a new audit output")
    ROOT.joinpath("artifacts").mkdir(parents=True, exist_ok=True)
    dataset = read_json(DATASET)
    result_payload = read_json(RESULTS)
    rows = result_rows(result_payload)
    samples, qa_by_qid = load_source(dataset)
    annotations = [row for row in csv.DictReader(ANNOTATIONS.open(newline="", encoding="utf-8")) if row["dataset"] == "locomo"]
    annotation_by_qid = {row["qid"]: row for row in annotations}
    selected_annotations = [row for row in annotations if row["jev_label"] == "explicit_temporal_filter"]
    selected_ids = {row["qid"] for row in selected_annotations}
    result_ids = {row["question_id"] for row in rows}

    selection = {
        "annotation_locomo_rows": len(annotations),
        "annotation_explicit_rows": len(selected_annotations),
        "result_rows": len(rows),
        "result_unique_qids": len(result_ids),
        "missing_result_qids": sorted(selected_ids - result_ids),
        "extra_result_qids": sorted(result_ids - selected_ids),
        "main_subset": sum(annotation_by_qid[qid]["main_subset"] == "True" for qid in result_ids),
        "adversarial_subset": sum(annotation_by_qid[qid]["main_subset"] == "False" for qid in result_ids),
        "annotation_tsm_active": sum(annotation_by_qid[qid]["tsm_active"] == "True" for qid in result_ids),
    }

    mapping_rows = []
    cache_audit = {}
    for row in rows:
        qid = row["question_id"]
        sample_id = qid.rsplit("_", 1)[0]
        qa = qa_by_qid.get(qid, {})
        sample = samples[sample_id]
        sample_index = next(i for i, item in enumerate(dataset) if item["sample_id"] == sample_id)
        cache = CACHE_ROOT / f"item_{sample_index}_all_sessions"
        tree = read_json(cache / "tree.json")
        text_edges = read_json(cache / "text_edge_mapping.json")
        mapping_rows.append(map_gold(row, qa, sample, tree, text_edges))
        tree_texts = [norm(node.get("text", "")) for node in tree.values()]
        edge_texts = [norm(edge.get("text", "")) for edge in text_edges.values()]
        edge_text_in_tree = sum(bool(text and any(text in leaf for leaf in tree_texts)) for text in edge_texts)
        cache_audit[sample_id] = {
            "item_index": sample_index,
            "cache": str(cache),
            "tree_leaf_count": len(tree),
            "text_edge_count": len(text_edges),
            "graphiti_text_edges_present_in_dense_tree": edge_text_in_tree,
            "graphiti_to_dense_tree_coverage": edge_text_in_tree / len(text_edges) if text_edges else None,
            "required_files": {name: (cache / name).exists() for name in ["tree.json", "graph.json", "index.json", "appearance_count.json", "text_edge_mapping.json", "snapshots.json", "dense_index.faiss"]},
        }

    mapping_counts = Counter(row["mapping_status"] for row in mapping_rows)
    def subset(rows_: list[dict[str, Any]], predicate) -> dict[str, Any]:
        xs = [r for r in rows_ if predicate(r)]
        mapped = [r for r in xs if r["mapping_status"] == "mapped"]
        out = {"n": len(xs), "mapped_n": len(mapped), "mapping_status": dict(Counter(r["mapping_status"] for r in xs))}
        for key in ["gold_any_in_dense20", "gold_all_in_dense20", "gold_any_in_dense100", "gold_all_in_dense100", "gold_any_in_final_context", "gold_all_in_final_context"]:
            ys = [r[key] for r in mapped if r[key] is not None]
            out[key] = {"count": sum(bool(v) for v in ys), "denominator": len(ys), "rate": sum(bool(v) for v in ys) / len(ys) if ys else None}
        out["parser_hit"] = {"count": sum(r["parser_hit"] is True for r in xs), "rate": sum(r["parser_hit"] is True for r in xs) / len(xs) if xs else None}
        out["nonempty_graphiti_edges"] = {"count": sum(r["graphiti_matched_edge_count"] > 0 for r in xs), "rate": sum(r["graphiti_matched_edge_count"] > 0 for r in xs) / len(xs) if xs else None}
        return out

    metrics = {
        "all": subset(mapping_rows, lambda r: True),
        "main": subset(mapping_rows, lambda r: annotation_by_qid[r["qid"]]["main_subset"] == "True"),
        "adversarial": subset(mapping_rows, lambda r: annotation_by_qid[r["qid"]]["main_subset"] == "False"),
    }
    for category in sorted({str(qa_by_qid[r["question_id"]].get("category")) for r in rows}):
        metrics[f"official_type_{category}"] = subset(mapping_rows, lambda r, c=category: str(qa_by_qid[r["qid"]].get("category")) == c)

    trace = trace_audit()
    memory_coverages = [value["graphiti_to_dense_tree_coverage"] for value in cache_audit.values()]
    memory_coverage = {
        "conversation_count": len(memory_coverages),
        "min": min(memory_coverages) if memory_coverages else None,
        "max": max(memory_coverages) if memory_coverages else None,
        "mean": sum(memory_coverages) / len(memory_coverages) if memory_coverages else None,
        "all_full_coverage": bool(memory_coverages) and all(value == 1.0 for value in memory_coverages),
    }
    config_info = {
        "run_id": RUN.name,
        "dataset": str(DATASET),
        "dataset_sha256": sha256(DATASET),
        "config": str(CONFIG),
        "config_sha256": sha256(CONFIG),
        "result_sha256": sha256(RESULTS),
        "code_commit": read_json(RESEARCH/"manifests/source_snapshot.json")["research_head"],
        "code_status": "see manifests/source_snapshot.json; source copy exported without TSM .git",
        "completed_at_from_log": None,
        "llm_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "llm_base_url": "https://api.siliconflow.cn/v1",
        "embedding_model": "text-embedding-3-small",
        "embedding_base_url": "https://api.relayrouter.ai/v1",
        "max_chunk_setting": 20,
        "trace_dense_k": 100,
        "batch_size": 1,
        "parallel_api_calls": 1,
        "graphiti_max_tokens": 16384,
        "manifest_scope": "Historical batch1 audit helper; model/config constants describe the archived full run, not arbitrary profiles.",
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection": selection,
        "cache_audit": cache_audit,
        "mapping_counts": dict(mapping_counts),
        "dense_tree_memory_coverage": memory_coverage,
        "retrieval_metrics": metrics,
        "trace_audit": trace,
        "acceptance_gates": {
            "selection_and_completion": len(result_ids) == 321 and not (selected_ids - result_ids) and not (result_ids - selected_ids),
            "trace_presence": len(mapping_rows) == 321 and all(bool(r.get("parser_hit") is not None) for r in mapping_rows),
            "full_memory_in_dense_tree": memory_coverage["all_full_coverage"],
            "gold_mapping_complete": mapping_counts.get("mapped", 0) == 321,
            "official_qa_score": False,
            "token_cost_accounting": False,
        },
        "notes": [
            "Dense Top-100 arrays contain padding duplicates; metrics use unique IDs preserving rank.",
            "The included evaluate_predictions.py is diagnostic only and is not used as official LoCoMo accuracy.",
            "Three graphiti traces have local schema-validation errors; each is followed by a completed retry and is retained in trace_audit.",
        ],
    }

    (ROOT / "artifacts/selection_audit.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n")
    with (ROOT / "artifacts/gold_mapping.jsonl").open("w", encoding="utf-8") as f:
        for row in mapping_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (ROOT / "artifacts/cache_audit.json").write_text(json.dumps(cache_audit, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "artifacts/trace_audit.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "artifacts/manifest.json").write_text(json.dumps(config_info, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "artifacts/acceptance_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
