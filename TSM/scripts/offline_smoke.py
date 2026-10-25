#!/usr/bin/env python3
"""Deterministic CPU/no-network smoke for core TSM data structures."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extract_graph import build_graph, load_nlp, naive_extract_graph
from provider.public_embedder import PublicEmbedder
from snapshot_selector import SnapshotSelector
from temporal_filter import TemporalFilter
from temporal_utils import TemporalConstraint, TemporalRange


def normalized_result() -> dict:
    texts = [
        "On 2025-01-04, Rowan discussed cedar trees with Mika.",
        "On 2025-02-09, Mika planned a fictional museum visit with Rowan.",
    ]
    embedder = PublicEmbedder(backend="hash", dimension=32)
    embeddings = asyncio.run(embedder.create_batch(texts))
    embedding_bytes = json.dumps(embeddings, separators=(",", ":")).encode("utf-8")

    nlp = load_nlp("en")
    extracted = [naive_extract_graph(text, nlp) for text in texts]
    edges = []
    nouns = set()
    for item in extracted:
        nouns.update(item["nouns"])
        edges.extend(
            (source, target, count)
            for (source, target), count in item["cooccurrence"].items()
        )
    graph = build_graph(edges)

    snapshots = [
        {"snapshot_id": "jan", "valid_from": "2025-01-01", "valid_to": "2025-02-01"},
        {"snapshot_id": "feb", "valid_from": "2025-02-01", "valid_to": "2025-03-01"},
    ]
    constraint = TemporalConstraint(
        ranges=[
            TemporalRange(
                start=datetime(2025, 2, 1, tzinfo=timezone.utc),
                end=datetime(2025, 2, 28, 23, 59, 59, tzinfo=timezone.utc),
                phrase="February 2025",
                granularity="month",
            )
        ],
        phrases=["February 2025"],
    )
    selected = SnapshotSelector(snapshots).select_by_constraint(constraint)
    temporal_map = {
        "leaf_0": {"text": texts[0], "reference_time": "2025-01-04T00:00:00Z"},
        "leaf_1": {"text": texts[1], "reference_time": "2025-02-09T00:00:00Z"},
    }
    filtered = TemporalFilter(
        external_text_map=temporal_map,
        keep_untimed_chunks=False,
    ).apply_constraint({"dense": ["leaf_0", "leaf_1"]}, constraint)

    return {
        "embedding_sha256": hashlib.sha256(embedding_bytes).hexdigest(),
        "embedding_shape": [len(embeddings), len(embeddings[0])],
        "entity_count": len(nouns),
        "graph_edge_count": graph.number_of_edges(),
        "graph_node_count": graph.number_of_nodes(),
        "selected_snapshots": [item["snapshot_id"] for item in selected],
        "temporally_filtered_chunks": filtered,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = normalized_result()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
