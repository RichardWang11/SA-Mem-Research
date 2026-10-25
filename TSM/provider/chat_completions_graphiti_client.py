"""Graphiti client adapter for providers exposing Chat Completions only.

Graphiti's current OpenAIClient uses the newer Responses API for structured
outputs. SiliconFlow documents the OpenAI-compatible Chat Completions API
instead, so this adapter keeps the Graphiti client contract while requesting a
JSON object from ``/chat/completions``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from graphiti_core.llm_client.openai_client import OpenAIClient


class ChatCompletionsGraphitiClient(OpenAIClient):
    """Use OpenAI-compatible Chat Completions for Graphiti structured calls."""

    async def _create_structured_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel],
        reasoning: str | None = None,
        verbosity: str | None = None,
    ) -> Any:
        # SiliconFlow supports JSON mode, while its documented API does not
        # expose the Responses API or OpenAI's beta parse endpoint. Include the
        # schema in the prompt so JSON mode still has the expected shape.
        schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
        prompt = (
            "\n\nReturn only one valid JSON object. It must conform to this JSON schema:\n"
            f"{schema}"
        )
        chat_messages = list(messages)
        if chat_messages:
            last = dict(chat_messages[-1])
            content = last.get("content") or ""
            last["content"] = f"{content}{prompt}"
            chat_messages[-1] = last  # type: ignore[assignment]

        response = await self.client.chat.completions.create(
            model=model,
            messages=chat_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        finish_reason = getattr(response.choices[0], "finish_reason", None)
        status = "completed" if finish_reason in (None, "stop") else "incomplete"
        incomplete_details = (
            {"reason": "max_tokens"} if finish_reason == "length" else None
        )
        # BaseOpenAIClient._handle_structured_response consumes output_text.
        # Preserve provider metadata so the diagnostic trace can distinguish a
        # valid JSON response from a Chat Completions length cutoff.
        return SimpleNamespace(
            output_text=content,
            status=status,
            incomplete_details=incomplete_details,
            usage=getattr(response, "usage", None),
            id=getattr(response, "id", None),
        )
