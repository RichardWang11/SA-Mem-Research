#!/usr/bin/env python3
"""Download public benchmark files and verify their release checksums."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "longmemeval": {
        "target": ROOT / "data/longmemeval/longmemeval_s.json",
        "sha256": "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
        "urls": [
            "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_s_cleaned.json",
            "https://hf-mirror.com/datasets/xiaowu0162/longmemeval-cleaned/resolve/98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_s_cleaned.json",
        ],
    },
    "locomo": {
        "target": ROOT / "data/locomo/locomo10.json",
        "sha256": "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
        "urls": [
            "https://raw.githubusercontent.com/snap-research/locomo/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/data/locomo10.json",
            "https://api.github.com/repos/snap-research/locomo/contents/data/locomo10.json?ref=3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376",
        ],
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str, force: bool = False) -> None:
    spec = DATASETS[name]
    target: Path = spec["target"]
    expected: str = spec["sha256"]
    if target.is_file() and sha256(target) == expected and not force:
        print(f"{name}: already verified at {target}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in spec["urls"]:
        fd, temporary_name = tempfile.mkstemp(prefix=f"tsm-{name}-", suffix=".json")
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            print(f"{name}: downloading official source {url}")
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "TSM/0.1.0",
                    "Accept": "application/vnd.github.raw+json",
                },
            )
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open(
                "wb"
            ) as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            observed = sha256(temporary)
            if observed != expected:
                raise RuntimeError(
                    f"checksum mismatch for {name}: expected {expected}, got {observed}"
                )
            temporary.replace(target)
            print(f"{name}: verified sha256={observed}")
            return
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            last_error = exc
        finally:
            temporary.unlink(missing_ok=True)
    raise RuntimeError(f"could not download {name}: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["all", *DATASETS])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in names:
        download(name, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
