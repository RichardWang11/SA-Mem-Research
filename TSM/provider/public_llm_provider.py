"""OpenAI-compatible LLM backend for the public TSM release."""

from __future__ import annotations

import os
import time
from typing import Any

from openai import OpenAI

from .base_provider import BaseLLMProvider
from .request_trace import jsonable, start_request


DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAICompatibleProvider(BaseLLMProvider):
    """Thread-safe OpenAI-compatible chat client with bounded retries."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        timeout: float = 120,
        retry: int = 3,
        retry_interval: float = 2,
    ) -> None:
        self.api_key_env = api_key_env
        self.api_key = api_key or os.getenv(api_key_env)
        self.base_url = base_url or os.getenv("TSM_LLM_BASE_URL") or DEFAULT_BASE_URL
        self.timeout = timeout
        self.retry = max(1, retry)
        self.retry_interval = max(0.0, retry_interval)

        # Local OpenAI-compatible servers commonly ignore the key but the SDK requires one.
        if not self.api_key and self.base_url.startswith(("http://localhost", "http://127.0.0.1")):
            self.api_key = "local-not-required"
        if not self.api_key:
            raise ValueError(
                f"Missing API key. Set {self.api_key_env} or pass api_key explicitly."
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OpenAICompatibleProvider":
        return cls(
            api_key_env=config.get("api_key_env", "OPENAI_API_KEY"),
            base_url=config.get("base_url"),
            timeout=float(config.get("timeout", 120)),
            retry=int(config.get("max_retries", 3)),
        )

    def chat_completion(
        self,
        messages,
        model,
        tools=None,
        tool_choice=None,
        stream=False,
        max_tokens=8192,
        op_type=None,
        **kwargs,
    ):
        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "max_tokens": max_tokens,
        }
        if tools is not None:
            params["tools"] = tools
        if tool_choice is not None:
            params["tool_choice"] = tool_choice
        params.update(kwargs)

        trace = start_request(
            "tsm_chat_completion",
            {
                "stage": op_type or "unspecified",
                "endpoint": self.base_url,
                "model": model,
                "request": {
                    "messages": jsonable(messages),
                    "max_tokens": max_tokens,
                    "stream": stream,
                    "has_tools": tools is not None,
                    "tool_choice": jsonable(tool_choice),
                    "extra": jsonable(kwargs),
                },
            },
        )

        last_error: Exception | None = None
        for attempt in range(self.retry):
            try:
                response = self.client.chat.completions.create(**params)
                trace.finish(
                    {
                        "attempt": attempt + 1,
                        "response": {
                            "id": getattr(response, "id", None),
                            "model": getattr(response, "model", None),
                            "choices": jsonable(getattr(response, "choices", None)),
                            "usage": jsonable(getattr(response, "usage", None)),
                        },
                    }
                )
                return response
            except Exception as exc:  # the OpenAI SDK exposes provider-specific subclasses
                last_error = exc
                if attempt + 1 < self.retry:
                    time.sleep(self.retry_interval * (attempt + 1))
        assert last_error is not None
        trace.fail(last_error, {"attempts": self.retry})
        raise last_error


def call_llm(
    messages,
    model,
    temperature=0,
    max_tokens=1024,
    stop=None,
    return_usage=False,
    tools=None,
    return_full_response=False,
    max_retries=3,
    return_time=False,
    op_type=None,
    *,
    base_url=None,
    api_key_env="OPENAI_API_KEY",
    timeout=120,
    **kwargs,
):
    """Compatibility helper for legacy TSM call sites, backed only by public APIs."""
    provider = OpenAICompatibleProvider(
        api_key_env=api_key_env,
        base_url=base_url,
        timeout=timeout,
        retry=max_retries,
    )
    started = time.perf_counter()
    request_kwargs: dict[str, Any] = {"temperature": temperature}
    if stop is not None:
        request_kwargs["stop"] = stop
    response = provider.chat_completion(
        messages=messages,
        model=model,
        tools=tools,
        max_tokens=max_tokens,
        op_type=op_type,
        **request_kwargs,
    )
    elapsed = time.perf_counter() - started

    if return_full_response:
        result: Any = response.choices[0].message
    else:
        content = response.choices[0].message.content or ""
        result = (content, response.usage) if return_usage else content
    return (result, elapsed) if return_time else result
