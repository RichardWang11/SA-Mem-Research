#!/usr/bin/env python3
"""Fail on common internal infrastructure, credential, or generated-file leaks."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", ".pytest_cache", "outputs", "runs"}
PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "OpenAI-style secret": re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    "machine path": re.compile(r"/(?:home|mnt|Users)/[^\s'\"]+"),
    "private IPv4": re.compile(
        r"\b(?:10\.(?:\d{1,3}\.){2}\d{1,3}|192\.168\.(?:\d{1,3}\.)\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.(?:\d{1,3}\.)\d{1,3})\b"
    ),
    "internal backend marker": re.compile(r"\b(?:mt_api|app_ids?|vllm_api_base)\b", re.I),
    "internal host marker": re.compile(r"\b(?:hadoop-ai-search|dolphinfs|sankuai)\b", re.I),
}


def release_paths() -> list[Path]:
    """Return files intended for release, excluding gitignored local artifacts."""
    if (ROOT / ".git").is_dir():
        completed = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
        )
        return [ROOT / item.decode("utf-8") for item in completed.stdout.split(b"\0") if item]
    return [path for path in ROOT.rglob("*") if path.is_file()]


def main() -> int:
    findings = []
    for path in sorted(release_paths()):
        if path.resolve() in {
            Path(__file__).resolve(),
            (ROOT / "scripts/validate_configs.py").resolve(),
        }:
            continue
        if any(part in SKIP_PARTS for part in path.relative_to(ROOT).parts):
            continue
        if path.is_dir():
            continue
        if "__pycache__" in path.relative_to(ROOT).parts or path.suffix in {".pyc", ".pyo"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for label, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append(f"{label}: {path.relative_to(ROOT)}:{line_number}")
    print(f"scanned release; findings={len(findings)}")
    for finding in findings:
        print(f"ERROR: {finding}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
