"""Public and deterministic embedding backends for TSM."""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
from collections.abc import Iterable
from typing import Any

from openai import AsyncOpenAI

from .request_trace import jsonable, start_request

try:
    from graphiti_core.embedder.client import EmbedderClient
except ImportError:  # keeps the offline smoke independent of Graphiti/Neo4j
    class EmbedderClient:  # type: ignore[no-redef]
        pass


DEFAULT_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


class PublicEmbedder(EmbedderClient):
    """OpenAI-compatible embeddings plus a deterministic offline smoke backend."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        dimension: int | None = None,
        backend: str = "openai_compatible",
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 120,
        **_: Any,
    ) -> None:
        self.model = model
        self.dimension = int(dimension or DEFAULT_DIMENSIONS.get(model, 1536))
        self.backend = backend
        self.base_url = base_url or os.getenv("TSM_EMBEDDING_BASE_URL")
        self.api_key_env = api_key_env
        self.api_key = api_key or os.getenv(api_key_env)
        self.timeout = timeout
        self.client: AsyncOpenAI | None = None

        if self.backend not in {"openai_compatible", "hash"}:
            raise ValueError(f"Unsupported embedding backend: {self.backend}")
        if self.backend == "openai_compatible":
            if not self.api_key and (self.base_url or "").startswith(
                ("http://localhost", "http://127.0.0.1")
            ):
                self.api_key = "local-not-required"
            if not self.api_key:
                raise ValueError(
                    f"Missing embedding API key. Set {self.api_key_env} or use backend=hash for smoke tests."
                )
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "PublicEmbedder":
        config = config or {}
        return cls(
            model=config.get("model", "text-embedding-3-small"),
            dimension=config.get("dimension"),
            backend=config.get("backend", "openai_compatible"),
            base_url=config.get("base_url"),
            api_key_env=config.get("api_key_env", "OPENAI_API_KEY"),
            timeout=float(config.get("timeout", 120)),
        )

    def _hash_embedding(self, text: str) -> list[float]:
        values: list[float] = []
        counter = 0
        while len(values) < self.dimension:
            digest = hashlib.sha256(f"tsm-v1\0{counter}\0{text}".encode("utf-8")).digest()
            values.extend((byte - 127.5) / 127.5 for byte in digest)
            counter += 1
        values = values[: self.dimension]
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]

    async def create(
        self,
        input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]],
    ) -> list[float]:
        if isinstance(input_data, str):
            text = input_data
        elif isinstance(input_data, list) and input_data and isinstance(input_data[0], str):
            text = str(input_data[0])
        else:
            text = str(list(input_data))
        if self.backend == "hash":
            return self._hash_embedding(text)
        assert self.client is not None
        trace = start_request(
            "embedding",
            {
                "stage": "embedding",
                "endpoint": self.base_url,
                "model": self.model,
                "request": {"input": jsonable(input_data)},
            },
        )
        try:
            response = await self.client.embeddings.create(input=input_data, model=self.model)
            trace.finish(
                {
                    "response": {
                        "count": len(response.data),
                        "dimension": len(response.data[0].embedding) if response.data else 0,
                        "usage": jsonable(getattr(response, "usage", None)),
                    }
                }
            )
            return response.data[0].embedding[: self.dimension]
        except Exception as exc:
            trace.fail(exc)
            raise

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        if self.backend == "hash":
            return [self._hash_embedding(text) for text in input_data_list]
        assert self.client is not None
        trace = start_request(
            "embedding",
            {
                "stage": "embedding",
                "endpoint": self.base_url,
                "model": self.model,
                "request": {"input": jsonable(input_data_list)},
            },
        )
        try:
            response = await self.client.embeddings.create(input=input_data_list, model=self.model)
            trace.finish(
                {
                    "response": {
                        "count": len(response.data),
                        "dimension": len(response.data[0].embedding) if response.data else 0,
                        "usage": jsonable(getattr(response, "usage", None)),
                    }
                }
            )
            return [item.embedding[: self.dimension] for item in response.data]
        except Exception as exc:
            trace.fail(exc)
            raise

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await self.create_batch(texts)

    def encode(self, texts, batch_size=16, **_: Any):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            result = asyncio.run(self.create_batch(items))
        else:
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(self.create_batch(items))
            finally:
                loop.close()
        try:
            import numpy as np

            array = np.asarray(result, dtype="float32")
            return array[0] if single else array
        except ImportError:
            return result[0] if single else result

    def eval(self) -> "PublicEmbedder":
        return self
