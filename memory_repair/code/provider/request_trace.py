"""Optional request/response tracing for research API diagnostics.

Tracing is enabled only when ``TSM_LLM_TRACE_DIR`` is set. Each request is
written before the network call and updated after it, so an interrupted call
still leaves its complete input on disk. Credentials are never included.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any


_lock = threading.Lock()
_sequence = 0


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _jsonable(vars(value))
        except Exception:
            pass
    return str(value)


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


class RequestTrace:
    def __init__(self, path: Path | None, base_payload: dict[str, Any] | None = None):
        self.path = path
        self.base_payload = base_payload or {}
        self.started = time.time()

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def finish(self, payload: dict[str, Any]) -> None:
        if self.path is None:
            return
        payload = {
            **self.base_payload,
            **payload,
            "status": "completed",
            "elapsed_seconds": round(time.time() - self.started, 4),
        }
        _write(self.path, payload)

    def fail(self, error: BaseException, payload: dict[str, Any] | None = None) -> None:
        if self.path is None:
            return
        _write(
            self.path,
            {
                **self.base_payload,
                **(payload or {}),
                "status": "error",
                "elapsed_seconds": round(time.time() - self.started, 4),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )


def start_request(kind: str, payload: dict[str, Any]) -> RequestTrace:
    root_value = os.getenv("TSM_LLM_TRACE_DIR", "").strip()
    if not root_value:
        return RequestTrace(None)

    root = Path(root_value)
    root.mkdir(parents=True, exist_ok=True)
    global _sequence
    with _lock:
        _sequence += 1
        sequence = _sequence
    safe_kind = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in kind)
    path = root / f"{sequence:06d}_{safe_kind}.json"
    trace = RequestTrace(path, payload)
    _write(
        path,
        {
            **payload,
            "status": "pending",
            "started_at_unix": trace.started,
            "trace_kind": kind,
        },
    )
    return trace


def jsonable(value: Any) -> Any:
    """Expose the safe serializer to provider adapters."""
    return _jsonable(value)
