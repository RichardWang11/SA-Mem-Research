#!/usr/bin/env python3
"""Run a fully synthetic pure-RAG pipeline against a local fake chat endpoint."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class FakeOpenAIHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 - inherited HTTP method name
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        payload = {
            "id": "synthetic-chat-1",
            "object": "chat.completion",
            "created": 0,
            "model": "synthetic-chat-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "A cedar tree."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002
        return


def find_result(root: Path) -> dict:
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        for value in payload.values():
            if isinstance(value, list) and any(
                isinstance(item, dict) and item.get("question_id") == "synthetic-long-001"
                for item in value
            ):
                return payload
    raise RuntimeError("synthetic result JSON was not produced")


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="tsm-synthetic-e2e-") as directory:
            work = Path(directory)
            config = yaml.safe_load(
                (ROOT / "configs/longmemeval/pure_rag.yaml").read_text(encoding="utf-8")
            )
            config["dataset"]["dataset_path"] = str(
                ROOT / "data/smoke/longmemeval_synthetic.json"
            )
            config["llm"]["api_model"] = "synthetic-chat-model"
            config["llm"]["backend_model"] = "synthetic-chat-model"
            config["llm"]["base_url"] = f"http://127.0.0.1:{server.server_port}/v1"
            config["embedding"] = {
                "backend": "hash",
                "model": "synthetic-hash-32",
                "dimension": 32,
                "batch_size": 8,
            }
            config["paths"] = {
                "log_path": str(work / "logs"),
                "answer_path": str(work / "answers"),
                "cache_path": str(work / "cache"),
            }
            config["parallel_api_calls"] = 1
            config["batch_size"] = 1
            config["llm_parallel"]["max_parallel_llm"] = 1
            config_path = work / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
            )
            env = os.environ.copy()
            env["OPENAI_API_KEY"] = "synthetic-local-key"
            env["NO_PROXY"] = "127.0.0.1,localhost"
            completed = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "--config",
                    str(config_path),
                    "--force-rebuild",
                ],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=120,
            )
            if completed.returncode != 0:
                print(completed.stdout, file=sys.stderr)
                return completed.returncode
            payload = find_result(work)
            rows = [
                item
                for value in payload.values()
                if isinstance(value, list)
                for item in value
                if isinstance(item, dict) and item.get("question_id") == "synthetic-long-001"
            ]
            if len(rows) != 1 or rows[0].get("hypothesis") != "A cedar tree.":
                print(f"unexpected synthetic result: {rows}", file=sys.stderr)
                return 1
            semantic_rows = [
                {
                    "question_id": row["question_id"],
                    "answer": row.get("answer"),
                    "hypothesis": row.get("hypothesis"),
                }
                for row in rows
            ]
            digest = hashlib.sha256(
                json.dumps(semantic_rows, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()
            print("synthetic_e2e_questions=1")
            print(f"synthetic_e2e_result_sha256={digest}")
            return 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
