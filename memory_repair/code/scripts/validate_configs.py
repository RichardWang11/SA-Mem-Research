#!/usr/bin/env python3
"""Validate the public 2 x 5 experiment matrix and its mechanism contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"longmemeval": "LongMemEval", "locomo": "Locomo"}
METHODS = {"tsm", "no_temporal", "no_persona", "traditional", "pure_rag"}
ALLOWED_ENV_NAMES = {
    "OPENAI_API_KEY",
    "TSM_LLM_BASE_URL",
    "TSM_EMBEDDING_BASE_URL",
    "NEO4J_URI",
    "NEO4J_USER",
    "NEO4J_PASSWORD",
    "NEO4J_DATABASE",
}
CONTRACTS = {
    "tsm": {
        "entrypoint": "main_time.py",
        "extraction": "graphiti",
        "clustering": True,
        "temporal_filter": True,
        "chunks_only": False,
    },
    "no_temporal": {
        "entrypoint": "main.py",
        "extraction": "graphiti",
        "clustering": True,
        "temporal_filter": False,
        "chunks_only": False,
    },
    "no_persona": {
        "entrypoint": "main.py",
        "extraction": "graphiti",
        "clustering": True,
        "temporal_filter": True,
        "chunks_only": True,
    },
    "traditional": {
        "entrypoint": "main.py",
        "extraction": "none",
        "clustering": False,
        "temporal_filter": False,
        "chunks_only": False,
    },
    "pure_rag": {
        "entrypoint": "main.py",
        "extraction": "none",
        "clustering": False,
        "temporal_filter": False,
        "chunks_only": True,
    },
}


def _walk(value: Any, path: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield child_path, key, child
            yield from _walk(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}[{index}]")


def validate_config(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"{path}: cannot parse YAML: {exc}"]
    if not isinstance(config, dict):
        return [f"{path}: top level must be a mapping"]

    experiment = config.get("experiment", {})
    method = experiment.get("method")
    contract = CONTRACTS.get(method)
    if contract is None:
        errors.append(f"{path}: unknown experiment.method={method!r}")
        return errors

    observed = {
        "entrypoint": experiment.get("entrypoint"),
        "extraction": config.get("extraction", {}).get("method"),
        "clustering": config.get("entity_clustering", {}).get("enable"),
        "temporal_filter": config.get("retriever", {}).get("kwargs", {}).get(
            "enable_temporal_filter"
        ),
        "chunks_only": config.get("retriever", {}).get("kwargs", {}).get("chunks_only"),
    }
    for key, expected in contract.items():
        if observed.get(key) != expected:
            errors.append(
                f"{path}: {method}.{key} must be {expected!r}, got {observed.get(key)!r}"
            )

    dataset_name = config.get("dataset", {}).get("dataset_name")
    expected_dataset = DATASETS.get(path.parent.name)
    if dataset_name != expected_dataset:
        errors.append(
            f"{path}: dataset_name must be {expected_dataset!r}, got {dataset_name!r}"
        )
    if path.stem != method:
        errors.append(f"{path}: filename must match method {method!r}")

    if config.get("split_method") != "round_based":
        errors.append(f"{path}: split_method must be round_based")
    if config.get("temporal", {}).get("enable") is not False:
        errors.append(
            f"{path}: temporal.enable is a legacy extraction switch and must remain false"
        )
    if config.get("llm", {}).get("temperature") != 0:
        errors.append(f"{path}: llm.temperature must be 0")

    for field_path, key, value in _walk(config):
        lower_key = str(key).lower()
        if lower_key in {"password", "api_key", "app_id", "app_ids"}:
            errors.append(f"{path}: secret-like field is forbidden: {field_path}")
        if lower_key.endswith("_env"):
            if value not in ALLOWED_ENV_NAMES:
                errors.append(f"{path}: unapproved environment name {value!r} at {field_path}")
        if isinstance(value, str) and value.startswith(("/home/", "/mnt/", "/Users/")):
            errors.append(f"{path}: absolute machine-specific path at {field_path}")

    return errors


def validate_matrix(root: Path = ROOT) -> tuple[list[Path], list[str]]:
    paths = sorted((root / "configs").glob("*/*.yaml"))
    errors: list[str] = []
    expected = {
        root / "configs" / dataset / f"{method}.yaml"
        for dataset in DATASETS
        for method in METHODS
    }
    actual = set(paths)
    for missing in sorted(expected - actual):
        errors.append(f"missing config: {missing}")
    for extra in sorted(actual - expected):
        errors.append(f"unexpected config: {extra}")
    for path in paths:
        errors.extend(validate_config(path))
    return paths, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args()
    paths, errors = validate_matrix()
    result = {"config_count": len(paths), "error_count": len(errors), "errors": errors}
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"validated {len(paths)} configs; errors={len(errors)}")
        for error in errors:
            print(f"ERROR: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
