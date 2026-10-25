"""Offline-safe token and latency accounting for TSM."""

from __future__ import annotations

import functools
import inspect
import json
import re
from collections import deque
from datetime import datetime
from threading import Lock, RLock
from typing import Any, Callable


def _message_text(messages) -> str:
    if isinstance(messages, str):
        return messages
    parts = []
    for message in messages:
        if isinstance(message, dict):
            parts.append(str(message.get("content", "")))
        else:
            parts.append(str(message))
    return "\n".join(parts)


def _count_tokens(messages, tokenizer=None) -> int:
    text = _message_text(messages)
    if tokenizer is not None:
        try:
            if hasattr(tokenizer, "encode"):
                encoded = tokenizer.encode(text)
                return len(encoded.ids if hasattr(encoded, "ids") else encoded)
        except Exception:
            pass
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        # Deterministic fallback for environments without tokenizer extras.
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


class CostState:
    def __init__(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_time: float = 0.0,
        window_size: int = 100_000,
        total_count: int = 0,
        histories=None,
    ):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_time = total_time
        self.total_count = total_count
        self.histories = deque(histories or [], maxlen=window_size)
        self._lock = RLock()

    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens

    def to_dict(self):
        with self._lock:
            count = max(self.total_count, 1)
            return {
                "total_count": self.total_count,
                "total_tokens": self.total_tokens,
                "average_input_tokens": self.input_tokens / count,
                "average_output_tokens": self.output_tokens / count,
                "average_tokens_per_call": self.total_tokens / count,
                "average_time_per_call": self.total_time / count,
                "histories": list(self.histories),
                "total_time": self.total_time,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "window_size": self.histories.maxlen,
            }

    def update(self, input_tokens, output_tokens, total_time, histories):
        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.total_time += total_time
            self.total_count += len(histories)
            self.histories.extend(histories)

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data):
        allowed = {
            "input_tokens",
            "output_tokens",
            "total_time",
            "window_size",
            "total_count",
            "histories",
        }
        return cls(**{key: value for key, value in data.items() if key in allowed})

    @classmethod
    def from_json(cls, value):
        return cls.from_dict(json.loads(value))


class CostStateManager:
    _states: dict[str, CostState | dict[str, CostState]] = {}
    _tokenizers: dict[str, Any] = {}
    _lock = Lock()

    def __init__(self):
        raise OSError("CostStateManager exposes class methods only")

    @classmethod
    def register(cls, model, state=None, tokenizer=None, exist_ok=False):
        with cls._lock:
            if model in cls._states and not exist_ok:
                raise ValueError(f"Model {model} is already registered")
            cls._states[model] = state or CostState()
            cls._tokenizers[model] = tokenizer

    @classmethod
    def get(cls, model):
        with cls._lock:
            if model not in cls._states:
                raise KeyError(model)
            return cls._states[model]

    @classmethod
    def update(cls, model, input_output_pair, **kwargs):
        input_dict = input_output_pair["input"]
        output_dict = input_output_pair["output"]
        elapsed = float(input_output_pair["elapsed"])
        operation = input_dict.get("metadata", {}).get("op_type")

        with cls._lock:
            state = cls._states.get(model)
            if state is None:
                raise KeyError(model)
            if operation:
                if isinstance(state, CostState):
                    state = {}
                    cls._states[model] = state
                state = state.setdefault(operation, CostState())
            elif isinstance(state, dict):
                raise ValueError("Operation type is required for this model")
            tokenizer = cls._tokenizers.get(model)

        input_tokens = _count_tokens(input_dict["messages"], tokenizer)
        output_tokens = _count_tokens(output_dict["messages"], tokenizer)
        input_dict["input_tokens"] = input_tokens
        output_dict["output_tokens"] = output_tokens
        state.update(input_tokens, output_tokens, elapsed, [input_output_pair])

    @classmethod
    def reset(cls):
        with cls._lock:
            cls._states.clear()
            cls._tokenizers.clear()

    @classmethod
    def save_to_json_file(cls, filename):
        with cls._lock:
            output = {
                model: (
                    state.to_dict()
                    if isinstance(state, CostState)
                    else {key: value.to_dict() for key, value in state.items()}
                )
                for model, state in cls._states.items()
            }
        with open(f"{filename}.json", "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2, ensure_ascii=False, sort_keys=True)


def token_monitor(extract_model_name, extract_input_dict, extract_output_dict) -> Callable:
    """Decorate a sync or async call and record tokens plus elapsed time."""

    def decorator(func):
        async def _record_async(*args, **kwargs):
            model, metadata = extract_model_name(*args, **kwargs)
            input_dict = extract_input_dict(*args, **kwargs)
            started = datetime.now().astimezone()
            result = await func(*args, **kwargs)
            finished = datetime.now().astimezone()
            CostStateManager.update(
                model,
                {
                    "input": input_dict,
                    "output": extract_output_dict(result),
                    "elapsed": (finished - started).total_seconds(),
                },
                **metadata,
            )
            return result

        def _record_sync(*args, **kwargs):
            model, metadata = extract_model_name(*args, **kwargs)
            input_dict = extract_input_dict(*args, **kwargs)
            started = datetime.now().astimezone()
            result = func(*args, **kwargs)
            finished = datetime.now().astimezone()
            CostStateManager.update(
                model,
                {
                    "input": input_dict,
                    "output": extract_output_dict(result),
                    "elapsed": (finished - started).total_seconds(),
                },
                **metadata,
            )
            return result

        wrapper = _record_async if inspect.iscoroutinefunction(func) else _record_sync
        return functools.wraps(func)(wrapper)

    return decorator
