"""Graphiti adapter for the public TSM release.

The adapter depends only on Graphiti's public API, Neo4j, and the
OpenAI-compatible providers in ``provider/``. It preserves the tuple contract
used by the original E2GraphRAG pipeline.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import networkx as nx

from provider.public_embedder import PublicEmbedder

logger = logging.getLogger(__name__)

try:
    from graphiti_core import Graphiti
    from graphiti_core.cross_encoder.client import CrossEncoderClient
    from graphiti_core.driver.neo4j_driver import Neo4jDriver
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_client import OpenAIClient
    from graphiti_core.nodes import EpisodeType
    from graphiti_core.utils.bulk_utils import RawEpisode
    from openai.resources.responses.responses import _type_to_text_format_param
    from provider.chat_completions_graphiti_client import ChatCompletionsGraphitiClient
    from provider.request_trace import jsonable, start_request

    class _TraceGraphitiMixin:
        async def _create_structured_completion(
            self,
            model,
            messages,
            temperature,
            max_tokens,
            response_model,
            reasoning=None,
            verbosity=None,
        ):
            trace = start_request(
                "graphiti_structured",
                {
                    "stage": "graphiti",
                    "endpoint": getattr(getattr(self, "config", None), "base_url", None),
                    "model": model,
                    "request": {
                        "messages": jsonable(messages),
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                        "response_model": getattr(response_model, "__name__", str(response_model)),
                        "reasoning": reasoning,
                        "verbosity": verbosity,
                    },
                },
            )
            trace_written = False
            try:
                response = await super()._create_structured_completion(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_model=response_model,
                    reasoning=reasoning,
                    verbosity=verbosity,
                )
                saved_response = {
                    "status": getattr(response, "status", None),
                    "incomplete_details": jsonable(
                        getattr(response, "incomplete_details", None)
                    ),
                    "output_text": getattr(response, "output_text", None),
                    "usage": jsonable(getattr(response, "usage", None)),
                    "id": getattr(response, "id", None),
                }

                # Chat Completions returns the raw text before TSM's base
                # client calls json.loads(). Validate here so malformed or
                # truncated output is saved together with the raw response.
                parsed_output = json.loads(saved_response.get("output_text") or "")
                response_model.model_validate(parsed_output)

                trace.finish({"response": saved_response})
                trace_written = True
                return response
            except Exception as exc:
                if not trace_written:
                    payload = locals().get("saved_response")
                    trace.fail(
                        exc,
                        {
                            "response": payload,
                            "parse_stage": "local_structured_validation"
                            if payload is not None
                            else "request",
                        },
                    )
                    trace_written = True
                raise

    class TraceOpenAIClient(OpenAIClient):
        """Responses client that preserves raw output when parsing fails.

        ``AsyncResponses.parse`` applies its Pydantic parser inside the SDK
        before returning the response object. When that parser sees truncated
        JSON, the normal TSM trace only receives the validation exception and
        loses the raw output, usage, and incomplete metadata. Use the same
        Responses request through ``create`` and validate locally so the raw
        response can be saved before re-raising the same kind of error.
        """

        async def _create_structured_completion(
            self,
            model,
            messages,
            temperature,
            max_tokens,
            response_model,
            reasoning=None,
            verbosity=None,
        ):
            trace = start_request(
                "graphiti_structured",
                {
                    "stage": "graphiti",
                    "endpoint": getattr(getattr(self, "config", None), "base_url", None),
                    "model": model,
                    "request": {
                        "messages": jsonable(messages),
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                        "response_model": getattr(response_model, "__name__", str(response_model)),
                        "reasoning": reasoning,
                        "verbosity": verbosity,
                    },
                },
            )
            trace_written = False

            def response_payload(response):
                return {
                    "status": getattr(response, "status", None),
                    "incomplete_details": jsonable(
                        getattr(response, "incomplete_details", None)
                    ),
                    "output_text": getattr(response, "output_text", None),
                    "usage": jsonable(getattr(response, "usage", None)),
                    "id": getattr(response, "id", None),
                }

            try:
                response = await self.client.responses.create(
                    model=model,
                    input=messages,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    text={"format": _type_to_text_format_param(response_model)},
                    reasoning={"effort": reasoning} if reasoning is not None else None,
                )
                saved_response = response_payload(response)

                # Validate before returning. BaseOpenAIClient will parse the
                # same JSON again, while this first pass ensures that failures
                # retain the raw text and provider metadata in the trace.
                raw_output = saved_response.get("output_text") or ""
                parsed_output = json.loads(raw_output)
                response_model.model_validate(parsed_output)

                trace.finish({"response": saved_response})
                trace_written = True
                return response
            except Exception as exc:
                if not trace_written:
                    payload = locals().get("saved_response")
                    trace.fail(
                        exc,
                        {
                            "response": payload,
                            "parse_stage": "local_structured_validation"
                            if payload is not None
                            else "request",
                        },
                    )
                    trace_written = True
                raise

    class TraceChatCompletionsGraphitiClient(_TraceGraphitiMixin, ChatCompletionsGraphitiClient):
        pass

    GRAPHITI_AVAILABLE = True
except ImportError:
    Graphiti = None  # type: ignore[assignment]
    CrossEncoderClient = object  # type: ignore[assignment,misc]
    GRAPHITI_AVAILABLE = False


class StableOrderCrossEncoder(CrossEncoderClient):
    """Keep Graphiti candidates in their input order without a reranker API."""

    async def rank(self, query: str, passages: list[str], **kwargs) -> list[dict]:
        return [
            {"corpus_id": index, "score": 1.0 - index * 1e-6, "text": passage}
            for index, passage in enumerate(passages)
        ]


class GraphitiExtractor:
    """Build a Graphiti graph and expose the legacy TSM retrieval structures."""

    def __init__(self, config_bundle):
        if not GRAPHITI_AVAILABLE:
            raise ImportError("graphiti-core is required for extraction.method=graphiti")

        self.config_bundle = config_bundle
        self.language = getattr(config_bundle, "language", "en")
        self.graphiti_config = getattr(config_bundle, "graphiti_config", {}) or {}
        neo4j = self.graphiti_config.get("neo4j", self.graphiti_config)

        self.neo4j_uri = os.getenv(
            neo4j.get("uri_env", "NEO4J_URI"), neo4j.get("uri", "bolt://localhost:7687")
        )
        self.neo4j_user = os.getenv(
            neo4j.get("user_env", "NEO4J_USER"), neo4j.get("user", "neo4j")
        )
        self.neo4j_password = os.getenv(
            neo4j.get("password_env", "NEO4J_PASSWORD"), neo4j.get("password", "")
        )
        self.neo4j_database = os.getenv(
            neo4j.get("database_env", "NEO4J_DATABASE"), neo4j.get("database", "neo4j")
        )
        if not self.neo4j_password:
            raise ValueError("Missing Neo4j password. Set NEO4J_PASSWORD.")

        self.llm_api_config = config_bundle.llm_api
        self.embedding_config = getattr(config_bundle, "embedding_config", {}) or {}
        self.batch_size = max(1, int(getattr(config_bundle, "batch_size", 10)))
        self.reset_group = bool(self.graphiti_config.get("reset_group", True))
        self.graphiti = None
        self.initialized = False

    async def _initialize_graphiti(self):
        if self.initialized:
            return

        api_key_env = self.llm_api_config.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.getenv(api_key_env)
        base_url = self.llm_api_config.get("base_url") or os.getenv("TSM_LLM_BASE_URL")
        if not api_key and (base_url or "").startswith(
            ("http://localhost", "http://127.0.0.1")
        ):
            api_key = "local-not-required"
        if not api_key:
            raise ValueError(f"Missing API key. Set {api_key_env}.")

        backend_model = self.llm_api_config.get(
            "backend_model", self.llm_api_config.get("model", "gpt-4o-mini")
        )
        graphiti_max_tokens = int(
            self.llm_api_config.get(
                "graphiti_max_tokens", self.llm_api_config.get("max_tokens", 8192)
            )
        )
        llm_config = LLMConfig(
            api_key=api_key,
            model=backend_model,
            small_model=backend_model,
            base_url=base_url,
            temperature=self.llm_api_config.get("temperature", 0),
            max_tokens=graphiti_max_tokens,
        )
        # graphiti-core defaults to sending reasoning.effort/verbosity. The
        # configured gpt-4o-mini-compatible RelayRouter endpoint rejects those
        # optional fields, while the TSM config already fixes temperature=0.
        client_class = (
            TraceChatCompletionsGraphitiClient
            if self.llm_api_config.get("api_protocol") == "chat_completions"
            else TraceOpenAIClient
        )
        llm_client = client_class(
            config=llm_config,
            cache=False,
            max_tokens=graphiti_max_tokens,
            reasoning=None,
            verbosity=None,
        )
        embedder = PublicEmbedder.from_config(self.embedding_config)
        cross_encoder = StableOrderCrossEncoder()

        if self.neo4j_database != "neo4j":
            driver = Neo4jDriver(
                uri=self.neo4j_uri,
                user=self.neo4j_user,
                password=self.neo4j_password,
                database=self.neo4j_database,
            )
            self.graphiti = Graphiti(
                self.neo4j_uri,
                self.neo4j_user,
                self.neo4j_password,
                graph_driver=driver,
                llm_client=llm_client,
                embedder=embedder,
                cross_encoder=cross_encoder,
            )
        else:
            self.graphiti = Graphiti(
                self.neo4j_uri,
                self.neo4j_user,
                self.neo4j_password,
                llm_client=llm_client,
                embedder=embedder,
                cross_encoder=cross_encoder,
            )
        await self.graphiti.build_indices_and_constraints()
        self.initialized = True

    async def clear_group_data(self, group_id: str) -> None:
        if not self.initialized:
            await self._initialize_graphiti()
        await self.graphiti.driver.execute_query(
            "MATCH (n) WHERE n.group_id = $group_id DETACH DELETE n",
            group_id=group_id,
        )

    @staticmethod
    def _reference_time(text: str, index: int) -> datetime:
        patterns = (
            r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b",
            r"\b(\d{4})/(\d{1,2})/(\d{1,2})\b",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return datetime(*(int(value) for value in match.groups()), tzinfo=timezone.utc)
                except ValueError:
                    pass
        return datetime(2000, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=index)

    @staticmethod
    def _cache_paths(cache_folder: str) -> dict[str, Path]:
        root = Path(cache_folder)
        return {
            "graph": root / "graph.json",
            "index": root / "index.json",
            "count": root / "appearance_count.json",
            "mapping": root / "text_edge_mapping.json",
        }

    def _load_cache(self, cache_folder: str):
        paths = self._cache_paths(cache_folder)
        if not all(path.is_file() for path in paths.values()):
            return None
        with paths["graph"].open(encoding="utf-8") as handle:
            edges = json.load(handle)
        graph = nx.Graph()
        for source, target, attributes in edges:
            graph.add_edge(source, target, **attributes)
        with paths["index"].open(encoding="utf-8") as handle:
            index = json.load(handle)
        with paths["count"].open(encoding="utf-8") as handle:
            counts = json.load(handle)
        with paths["mapping"].open(encoding="utf-8") as handle:
            mapping = json.load(handle)
        return graph, index, counts, mapping

    def _save_cache(self, result, cache_folder: str) -> None:
        graph, index, counts, mapping = result
        paths = self._cache_paths(cache_folder)
        paths["graph"].parent.mkdir(parents=True, exist_ok=True)
        edges = [
            [str(source), str(target), dict(sorted(attributes.items()))]
            for source, target, attributes in sorted(
                graph.edges(data=True), key=lambda item: (str(item[0]), str(item[1]))
            )
        ]
        payloads = {
            "graph": edges,
            "index": {key: sorted(value) for key, value in sorted(index.items())},
            "count": dict(sorted(counts.items())),
            "mapping": dict(sorted(mapping.items())),
        }
        for key, payload in payloads.items():
            with paths[key].open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")

    @staticmethod
    def _records(result) -> list[Any]:
        return list(getattr(result, "records", []) or [])

    async def _read_compatible_graph(self, group_id: str, text_chunks: list[str]):
        episodes_result = await self.graphiti.driver.execute_query(
            """
            MATCH (e:Episodic)
            WHERE e.group_id = $group_id
            RETURN e.uuid AS uuid, e.name AS name, e.content AS content, e.valid_at AS valid_at
            ORDER BY e.name
            """,
            group_id=group_id,
        )
        episodes = self._records(episodes_result)
        episode_by_name = {record.get("name"): record for record in episodes}

        mention_result = await self.graphiti.driver.execute_query(
            """
            MATCH (e:Episodic)-[:MENTIONS]->(n:Entity)
            WHERE e.group_id = $group_id AND n.group_id = $group_id
            RETURN e.uuid AS episode_uuid, n.name AS entity_name
            ORDER BY entity_name, episode_uuid
            """,
            group_id=group_id,
        )
        mentions = self._records(mention_result)

        edge_result = await self.graphiti.driver.execute_query(
            """
            MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
            WHERE r.group_id = $group_id
            RETURN a.name AS source, b.name AS target, r.name AS relation,
                   r.fact AS fact, r.episodes AS episodes
            ORDER BY source, target, relation
            """,
            group_id=group_id,
        )
        edge_records = self._records(edge_result)

        graph = nx.Graph()
        index: dict[str, list[str]] = defaultdict(list)
        counts: dict[str, int] = defaultdict(int)
        edges_by_episode: dict[str, list[list[Any]]] = defaultdict(list)

        for record in mentions:
            entity = str(record.get("entity_name") or "").strip()
            episode_uuid = str(record.get("episode_uuid") or "").strip()
            if entity and episode_uuid:
                graph.add_node(entity)
                if episode_uuid not in index[entity]:
                    index[entity].append(episode_uuid)
                counts[entity] += 1

        for record in edge_records:
            source = str(record.get("source") or "").strip()
            target = str(record.get("target") or "").strip()
            if not source or not target:
                continue
            attributes = {
                "weight": 1.0,
                "relation": str(record.get("relation") or "related_to"),
                "fact": str(record.get("fact") or ""),
            }
            graph.add_edge(source, target, **attributes)
            episode_ids = record.get("episodes") or []
            if isinstance(episode_ids, str):
                episode_ids = [episode_ids]
            for episode_uuid in episode_ids:
                edges_by_episode[str(episode_uuid)].append([source, target, 1.0])

        mapping: dict[str, dict[str, Any]] = {}
        for position, text in enumerate(text_chunks):
            name = f"chunk_{position}"
            record = episode_by_name.get(name)
            episode_uuid = str(record.get("uuid")) if record else f"leaf_{position}"
            valid_at = record.get("valid_at") if record else self._reference_time(text, position)
            if hasattr(valid_at, "isoformat"):
                valid_at = valid_at.isoformat()
            mapping[episode_uuid] = {
                "text": text,
                "uuid": episode_uuid,
                "reference_time": str(valid_at),
                "edges": edges_by_episode.get(episode_uuid, []),
            }

        return graph, dict(index), dict(counts), mapping

    async def extract_graph_compatible(
        self,
        text_chunks: list[str],
        cache_folder: str,
        language: str = "en",
        group_id: str | None = None,
        original_text_for_db: list | None = None,
    ):
        started = time.perf_counter()
        cached = self._load_cache(cache_folder)
        if cached is not None:
            return cached, -1

        group_id = group_id or self.graphiti_config.get("group_id", "tsm_default")
        await self._initialize_graphiti()
        try:
            if self.reset_group:
                await self.clear_group_data(group_id)

            episodes = []
            for index, text in enumerate(text_chunks):
                # Graphiti 0.19 treats a non-null RawEpisode UUID as an
                # existing node lookup. Leave it unset for new episodes; the
                # stable chunk identity used by TSM is the episode name.
                episodes.append(
                    RawEpisode(
                        name=f"chunk_{index}",
                        content=text,
                        source=EpisodeType.text,
                        source_description="TSM memory chunk",
                        reference_time=self._reference_time(text, index),
                    )
                )

            for start in range(0, len(episodes), self.batch_size):
                await self.graphiti.add_episode_bulk(
                    episodes[start : start + self.batch_size], group_id=group_id
                )

            result = await self._read_compatible_graph(group_id, text_chunks)
            self._save_cache(result, cache_folder)
            return result, time.perf_counter() - started
        finally:
            if self.graphiti is not None:
                await self.graphiti.close()
                self.graphiti = None
                self.initialized = False


def init_async_worker() -> None:
    """Compatibility hook retained for the multiprocessing wrapper."""


def _run_coroutine(coroutine):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coroutine).result()


def extract_graph_graphiti(args):
    """Synchronous wrapper matching ``process_utils.extract_graph_task``."""
    group_id = None
    original_text_for_db = None
    if len(args) == 6:
        text, cache_folder, language, _, _, config_bundle = args
    elif len(args) == 7:
        text, cache_folder, language, _, _, config_bundle, group_id = args
    elif len(args) == 8:
        text, cache_folder, language, _, _, config_bundle, group_id, original_text_for_db = args
    else:
        raise ValueError(f"Expected 6, 7, or 8 arguments, received {len(args)}")

    extractor = GraphitiExtractor(config_bundle)
    return _run_coroutine(
        extractor.extract_graph_compatible(
            text, cache_folder, language, group_id, original_text_for_db
        )
    )
