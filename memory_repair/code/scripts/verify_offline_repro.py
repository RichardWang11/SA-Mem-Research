#!/usr/bin/env python3
"""Run the offline smoke twice and require byte-identical normalized output."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="tsm-repro-") as directory:
        outputs = [Path(directory) / "run1.json", Path(directory) / "run2.json"]
        for output in outputs:
            subprocess.run(
                [sys.executable, "scripts/offline_smoke.py", "--output", str(output)],
                cwd=ROOT,
                check=True,
                stdout=subprocess.DEVNULL,
            )
        first = outputs[0].read_bytes()
        second = outputs[1].read_bytes()
        if first != second:
            print("offline smoke outputs differ", file=sys.stderr)
            return 1
        digest = hashlib.sha256(first).hexdigest()
        print(f"offline_smoke_sha256={digest}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
