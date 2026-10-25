#!/usr/bin/env python3
"""Run or inspect the complete 2 x 5 public experiment matrix."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["all", "longmemeval", "locomo"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--acknowledge-data-transmission", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()
    datasets = ["longmemeval", "locomo"] if args.dataset == "all" else [args.dataset]
    methods = ["tsm", "no_temporal", "no_persona", "traditional", "pure_rag"]

    for dataset in datasets:
        for method in methods:
            command = [
                sys.executable,
                "scripts/run_experiment.py",
                "--config",
                f"configs/{dataset}/{method}.yaml",
            ]
            if args.dry_run:
                command.append("--dry-run")
            if args.acknowledge_data_transmission:
                command.append("--acknowledge-data-transmission")
            if args.force_rebuild:
                command.append("--force-rebuild")
            print(f"running {dataset}/{method}", flush=True)
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
