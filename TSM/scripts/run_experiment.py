#!/usr/bin/env python3
"""Run one declared experiment with a provenance manifest and transmission guard."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_configs import validate_config


BENCHMARKS = {"LongMemEval", "Locomo"}
TRACKED_DEPENDENCIES = [
    "numpy",
    "PyYAML",
    "tqdm",
    "faiss-cpu",
    "neo4j",
    "graphiti-core",
    "networkx",
    "openai",
    "scikit-learn",
    "spacy",
    "nest-asyncio",
    "python-dateutil",
    "dateparser",
    "umap-learn",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dependency_versions() -> dict[str, str]:
    result = {}
    for name in TRACKED_DEPENDENCIES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "missing"
    return result


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def redacted(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in {"password", "api_key", "token", "secret"}:
                cleaned[key] = "<redacted>"
            else:
                cleaned[key] = redacted(child)
        return cleaned
    if isinstance(value, list):
        return [redacted(item) for item in value]
    return value


def endpoint_summary(config: dict) -> dict[str, str]:
    def safe_url(value: str) -> str:
        parsed = urlsplit(value)
        if parsed.username is None and parsed.password is None:
            return value
        hostname = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port else ""
        return urlunsplit((parsed.scheme, f"<redacted>@{hostname}{port}", parsed.path, parsed.query, parsed.fragment))

    llm_endpoint = os.getenv("TSM_LLM_BASE_URL", "https://api.openai.com/v1")
    embedding_endpoint = os.getenv(
        "TSM_EMBEDDING_BASE_URL",
        llm_endpoint,
    )
    return {
        "llm_endpoint": safe_url(llm_endpoint),
        "llm_model": str(config["llm"]["api_model"]),
        "embedding_endpoint": safe_url(embedding_endpoint),
        "embedding_model": str(config["embedding"]["model"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--acknowledge-data-transmission",
        action="store_true",
        help="confirm that benchmark content may be sent to the shown LLM and embedding endpoints",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--query-only", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    errors = validate_config(config_path)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    endpoints = endpoint_summary(config)
    print(json.dumps(endpoints, indent=2, sort_keys=True))

    dataset_name = config["dataset"]["dataset_name"]
    if (
        dataset_name in BENCHMARKS
        and not args.acknowledge_data_transmission
        and not args.dry_run
    ):
        print(
            "Refusing to send benchmark content. Inspect the endpoint/model summary, then rerun "
            "with --acknowledge-data-transmission after explicit approval.",
            file=sys.stderr,
        )
        return 3

    dataset_path = (ROOT / config["dataset"]["dataset_path"]).resolve()
    if not dataset_path.is_file():
        print(f"dataset file not found: {dataset_path}", file=sys.stderr)
        return 4

    experiment_name = config["experiment"]["name"]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + f"_{experiment_name}"
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    resolved = dict(config)
    resolved["paths"] = {
        "log_path": str(run_dir / "logs"),
        "answer_path": str(run_dir / "answers"),
        "cache_path": str(run_dir / "cache"),
    }
    resolved["use_improved_paths"] = False
    resolved_path = run_dir / "config.resolved.yaml"
    resolved_path.write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    command = [
        sys.executable,
        config["experiment"]["entrypoint"],
        "--config",
        str(resolved_path),
    ]
    if args.force_rebuild:
        command.append("--force-rebuild")
    if args.query_only:
        command.append("--query-only")

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "started_at_utc": utc_now(),
        "status": "dry_run" if args.dry_run else "running",
        "git_commit": git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": dependency_versions(),
        "experiment": redacted(config),
        "source_config": str(config_path.relative_to(ROOT)),
        "source_config_sha256": sha256(config_path),
        "dataset": {
            "path": str(dataset_path.relative_to(ROOT)),
            "bytes": dataset_path.stat().st_size,
            "sha256": sha256(dataset_path),
        },
        "endpoints": endpoints,
        "credentials": {
            "openai_api_key_present": bool(os.getenv("OPENAI_API_KEY")),
            "neo4j_password_present": bool(os.getenv("NEO4J_PASSWORD")),
        },
        "data_transmission_acknowledged": args.acknowledge_data_transmission,
        "command": command,
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"manifest={manifest_path}")

    if args.dry_run:
        return 0
    completed = subprocess.run(command, cwd=ROOT, check=False)
    manifest["finished_at_utc"] = utc_now()
    manifest["exit_code"] = completed.returncode
    manifest["status"] = "completed" if completed.returncode == 0 else "failed"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
