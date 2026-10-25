"""
Refactored query processing module.

This module provides a cleaner, more modular implementation of the Retriever class
using separate components for different retrieval strategies.
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import dateparser
import faiss
import networkx as nx
import spacy
try:
    import torch
except ImportError:  # Used only for optional CUDA cache cleanup.
    torch = None
from neo4j import GraphDatabase

# Import modular components
from dense_retriever import DenseRetriever, FAISSIndexBuilder, run_async_embedding
from graph_utils import GraphRetriever, OccurrenceRanker
from snapshot_selector import SnapshotSelector
from temporal_filter import TemporalFilter
from temporal_utils import TemporalConstraint, extract_temporal_constraints, parse_iso_datetime

# Import existing modules
from extract_graph import naive_extract_graph

logger = logging.getLogger(__name__)


class RefactoredRetriever:
    """
    Refactored retriever with modular components.

    This class orchestrates different retrieval strategies using specialized
    components for dense retrieval, graph-based retrieval, temporal filtering,
    and snapshot selection.
    """

    def __init__(self, cache_tree, G: nx.Graph, index, appearance_count: Dict[str, int], nlp: spacy.Language, **kwargs):
        """
        Initialize refactored retriever.

        Args:
            cache_tree: Summary tree of the document
            G: Graph of the document
            index: Entity to chunks index
            appearance_count: Appearance count of entities in chunks
            nlp: Spacy model
            **kwargs: Additional configuration parameters
        """
        # Core data structures
        self.cache_tree = cache_tree
        self.G = G
        self.index = index
        self.appearance_count = appearance_count
        self.nlp = nlp

        # Build collapsed tree for dense retrieval
        self.collapse_tree, self.collapse_tree_ids = self._collapse_tree(cache_tree)

        # Configuration
        self.device = kwargs.get("device", "cuda:0")
        self.merge_num = kwargs.get("merge_num", 5)
        self.min_count = kwargs.get("min_count", 2)
        self.temporal_filter_enabled = kwargs.get("enable_temporal_filter", True)
        self.keep_untimed_chunks = kwargs.get("keep_untimed_chunks", True)
        self.keep_untimed_edges = kwargs.get("keep_untimed_edges", True)
        self.external_text_map = kwargs.get("external_text_map", {})

        # Graphiti configuration
        self.graphiti_config = kwargs.get("graphiti_config", {})
        self.graphiti_group_id = kwargs.get("graphiti_group_id")
        self._graphiti_database = kwargs.get("graphiti_database")
        self._graphiti_sync_driver = kwargs.get("graphiti_sync_driver")
        self._owns_graphiti_driver = False

        # Initialize Graphiti driver if needed
        self._setup_graphiti_driver()

        # Load snapshots
        self.snapshots = self._load_snapshots(kwargs)

        # Initialize modular components
        self._setup_components(kwargs)

        # State tracking
        self.last_temporal_edges = []
        self._last_temporal_constraint: Optional[TemporalConstraint] = None
        self._last_temporal_order: Optional[str] = None
        self._last_reference_date: Optional[datetime] = None
        self.trace_enabled = bool(kwargs.get("trace_enabled", False))
        self.trace_dense_k = int(kwargs.get("trace_dense_k", 100))
        self.last_retrieval_trace: Dict[str, Any] = {}

        logger.info("RefactoredRetriever initialized successfully")

    def _setup_graphiti_driver(self):
        """Setup Graphiti Neo4j driver if needed."""
        if self._graphiti_sync_driver is None and self.graphiti_config:
            neo4j_config = self.graphiti_config.get("neo4j", self.graphiti_config)
            uri = os.getenv(
                neo4j_config.get("uri_env", "NEO4J_URI"),
                neo4j_config.get("uri", "bolt://localhost:7687"),
            )
            user = os.getenv(
                neo4j_config.get("user_env", "NEO4J_USER"),
                neo4j_config.get("user", "neo4j"),
            )
            password = os.getenv(
                neo4j_config.get("password_env", "NEO4J_PASSWORD"),
                neo4j_config.get("password", ""),
            )
            database = os.getenv(
                neo4j_config.get("database_env", "NEO4J_DATABASE"),
                neo4j_config.get("database", "neo4j"),
            )

            if uri:
                try:
                    self._graphiti_sync_driver = GraphDatabase.driver(
                        uri,
                        auth=(user or "", password or ""),
                        max_connection_pool_size=50,
                    )
                    self._owns_graphiti_driver = True
                    self._graphiti_database = self._graphiti_database or database
                    logger.info("Graphiti Neo4j driver initialized")
                except Exception as exc:
                    logger.warning(f"Failed to initialize Graphiti driver: {exc}")
                    self._graphiti_sync_driver = None

        if self._graphiti_database is None:
            neo4j_config = self.graphiti_config.get("neo4j", self.graphiti_config)
            self._graphiti_database = (
                os.getenv(
                    neo4j_config.get("database_env", "NEO4J_DATABASE"),
                    neo4j_config.get("database", "neo4j"),
                )
                if neo4j_config
                else None
            )

    def _load_snapshots(self, kwargs) -> List[Dict]:
        """Load snapshots from configuration or cache folder."""
        snapshots = kwargs.get("snapshots", [])
        if not snapshots:
            cache_folder = kwargs.get("cache_folder")
            if cache_folder:
                snapshots_path = os.path.join(cache_folder, "snapshots.json")
                if os.path.exists(snapshots_path):
                    try:
                        with open(snapshots_path, 'r', encoding='utf-8') as f:
                            snapshots = json.load(f)
                        logger.info(f"Loaded {len(snapshots)} snapshots")
                    except Exception as e:
                        logger.warning(f"Failed to load snapshots: {e}")
                        snapshots = []
        return snapshots

    def _setup_components(self, kwargs):
        """Initialize modular components."""
        # Snapshot selector
        self.snapshot_selector = SnapshotSelector(self.snapshots)

        # Temporal filter
        self.temporal_filter = TemporalFilter(
            external_text_map=self.external_text_map,
            cache_tree=self.cache_tree,
            keep_untimed_chunks=self.keep_untimed_chunks,
            keep_untimed_edges=self.keep_untimed_edges,
            graphiti_sync_driver=self._graphiti_sync_driver,
            graphiti_database=self._graphiti_database,
            graphiti_group_id=self.graphiti_group_id,
        )

        # Graph retriever
        self.graph_retriever = GraphRetriever(
            graph=self.G,
            entity_index=self.index,
            appearance_count=self.appearance_count,
            cache_tree=self.cache_tree,
        )

        # Occurrence ranker
        self.occurrence_ranker = OccurrenceRanker(
            appearance_count=self.appearance_count,
            cache_tree=self.cache_tree,
        )

        # Dense retriever (if embedder available)
        self.dense_retriever = None
        self.embedder = None
        self.faiss_index = None

        if kwargs.get("prebuilt_faiss_index") is not None:
            self._setup_prebuilt_dense_retriever(kwargs)
        elif kwargs.get("embedder", kwargs.get("tokenizer")) is not None:
            self._setup_dense_retriever(kwargs)
        else:
            logger.info("No embedder provided, dense retrieval disabled")

    def _setup_prebuilt_dense_retriever(self, kwargs):
        """Setup dense retriever with prebuilt FAISS index."""
        prebuilt_index = kwargs.get("prebuilt_faiss_index")
        self.prebuilt_doc_ids = kwargs.get("prebuilt_doc_ids", [])

        # Setup GPU acceleration if enabled
        use_gpu_faiss = kwargs.get("use_gpu_faiss", True)
        gpu_id = kwargs.get("gpu_id", 0)

        if use_gpu_faiss and hasattr(faiss, 'StandardGpuResources'):
            try:
                is_gpu_index = hasattr(prebuilt_index, 'device') or str(type(prebuilt_index)).find('Gpu') != -1
                if not is_gpu_index:
                    gpu_res = faiss.StandardGpuResources()
                    self.faiss_index = faiss.index_cpu_to_gpu(gpu_res, gpu_id, prebuilt_index)
                    logger.info(f"Prebuilt FAISS index moved to GPU {gpu_id}")
                else:
                    self.faiss_index = prebuilt_index
                    logger.info("Prebuilt FAISS index already on GPU")
            except Exception as e:
                logger.warning(f"GPU transfer failed, using CPU: {e}")
                self.faiss_index = prebuilt_index
        else:
            self.faiss_index = prebuilt_index
            logger.info("Using prebuilt CPU FAISS index")

        # Setup embedder for query encoding
        self._setup_embedder(kwargs)
        if self.embedder:
            self.dense_retriever = DenseRetriever(
                embedder=self.embedder,
                faiss_index=self.faiss_index,
                collapse_tree_ids=self.collapse_tree_ids,
                use_gpu_faiss=use_gpu_faiss,
                gpu_id=gpu_id,
                external_text_map=self.external_text_map,
                cache_tree=self.cache_tree,
            )

    def _setup_dense_retriever(self, kwargs):
        """Setup dense retriever with runtime index building."""
        self._setup_embedder(kwargs)
        if self.embedder:
            # Build FAISS index
            use_gpu_faiss = kwargs.get("use_gpu_faiss", True)
            gpu_id = kwargs.get("gpu_id", 0)
            embedding_batch_size = kwargs.get("embedding_batch_size", 1024)

            index_builder = FAISSIndexBuilder(
                embedder=self.embedder,
                use_gpu=use_gpu_faiss,
                gpu_id=gpu_id,
                embedding_batch_size=embedding_batch_size,
            )

            # Build index from collapse_tree
            self.faiss_index = run_async_embedding(
                index_builder.build_index(self.collapse_tree)
            )

            if self.faiss_index:
                self.dense_retriever = DenseRetriever(
                    embedder=self.embedder,
                    faiss_index=self.faiss_index,
                    collapse_tree_ids=self.collapse_tree_ids,
                    use_gpu_faiss=use_gpu_faiss,
                    gpu_id=gpu_id,
                    external_text_map=self.external_text_map,
                    cache_tree=self.cache_tree,
                )
                logger.info("Dense retriever setup completed")
            else:
                logger.warning("Failed to build FAISS index")

    def _setup_embedder(self, kwargs):
        """Setup embedder for query encoding."""
        if kwargs.get("embedder", kwargs.get("tokenizer")) is not None:
            try:
                from provider.public_embedder import PublicEmbedder
                embedding_config = kwargs.get("embedding_config", {})
                self.embedder = PublicEmbedder.from_config(embedding_config)
                logger.info("Public embedder initialized")
            except ImportError as e:
                logger.error(f"Failed to import public embedder: {e}")
                self.embedder = None

    def _collapse_tree(self, cache_tree: Dict[str, Dict]) -> Tuple[List[str], List[str]]:
        """Collapse tree structure for dense retrieval."""
        collapsed_tree = []
        collapsed_tree_ids = []
        for key, value in cache_tree.items():
            collapsed_tree.append(value["text"])
            collapsed_tree_ids.append(key)
        return collapsed_tree, collapsed_tree_ids

    def update(self, cache_tree, G, index, appearance_count):
        """Update retriever with new data structures."""
        self.cache_tree = cache_tree
        self.collapse_tree, self.collapse_tree_ids = self._collapse_tree(cache_tree)
        self.G = G
        self.index = index
        self.appearance_count = appearance_count

        # Update components
        self.graph_retriever = GraphRetriever(
            graph=G,
            entity_index=index,
            appearance_count=appearance_count,
            cache_tree=cache_tree,
        )

        self.occurrence_ranker = OccurrenceRanker(
            appearance_count=appearance_count,
            cache_tree=cache_tree,
        )

        # Rebuild FAISS index if needed
        if self.embedder and not hasattr(self, 'prebuilt_doc_ids'):
            if hasattr(self.embedder, 'embed_texts'):
                # Rebuild the index with the configured public embedder.
                index_builder = FAISSIndexBuilder(embedder=self.embedder)
                self.faiss_index = run_async_embedding(
                    index_builder.build_index(self.collapse_tree)
                )
                if self.dense_retriever:
                    self.dense_retriever.faiss_index = self.faiss_index
                    self.dense_retriever.collapse_tree_ids = self.collapse_tree_ids

    def query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Process query using modular retrieval components.

        Args:
            query: Query string
            **kwargs: Query parameters

        Returns:
            Dictionary containing chunks and metadata
        """
        start_time = time.time()
        logger.info(f"=== Refactored Retriever Query Started ===")
        logger.info(f"Query: {query[:50]}...")
        logger.info(f"Parameters: {kwargs}")

        try:
            # Step 1: Extract entities (currently disabled in favor of dense retrieval)
            entities = []  # Skip entity extraction for now
            logger.info("Entity extraction skipped, using dense retrieval")

            # Step 2: Parse temporal information
            temporal_info = self._parse_temporal_information(query, kwargs)
            temporal_constraint = temporal_info["constraint"]
            temporal_order = temporal_info["order"]
            reference_date = temporal_info["reference_date"]

            self.last_retrieval_trace = {
                "question": query,
                "tsm_parser_hit": bool(
                    temporal_constraint is not None and not temporal_constraint.is_empty()
                ),
                "parsed_time_ranges": [
                    {
                        "start": item.start.isoformat(),
                        "end": item.end.isoformat(),
                        "phrase": item.phrase,
                        "granularity": item.granularity,
                    }
                    for item in (temporal_constraint.ranges if temporal_constraint else [])
                ],
                "temporal_order": temporal_order,
                "reference_date": reference_date.isoformat() if reference_date else None,
            }

            # Step 3: Select snapshots
            selected_snapshots = self._select_snapshots(temporal_constraint, reference_date)

            # Step 4: Perform retrieval
            if len(entities) == 0 and self.dense_retriever:
                # Use dense retrieval
                result = self._perform_dense_retrieval(
                    query, kwargs, temporal_constraint, temporal_order, selected_snapshots
                )
            else:
                # Use graph-based retrieval (fallback)
                result = self._perform_graph_retrieval(
                    query, entities, kwargs, temporal_constraint, temporal_order, selected_snapshots
                )

            # Step 5: Add metadata
            if kwargs.get("info", True):
                result.update(self._build_metadata(entities, selected_snapshots, temporal_constraint))

            total_time = time.time() - start_time
            logger.info(f"Query completed in {total_time:.3f}s")
            logger.info(f"=== Refactored Retriever Query Ended ===")

            return result

        except Exception as exc:
            logger.error(f"Query processing failed: {exc}", exc_info=True)
            return {"chunks": "", "error": str(exc)}

    def _parse_temporal_information(self, query: str, kwargs: Dict) -> Dict[str, Any]:
        """Parse temporal information from query and parameters."""
        # Parse reference date
        reference_input = (
            kwargs.get("question_date") or
            kwargs.get("reference_time") or
            kwargs.get("reference_date")
        )

        reference_date: Optional[datetime] = None
        if reference_input:
            reference_date = parse_iso_datetime(reference_input)
            if reference_date is None:
                parsed_reference = dateparser.parse(reference_input)
                if parsed_reference:
                    reference_date = parse_iso_datetime(parsed_reference.isoformat())

        # Parse temporal constraint
        temporal_constraint: Optional[TemporalConstraint] = None
        temporal_order: Optional[str] = None

        if self.temporal_filter_enabled:
            temporal_constraint = extract_temporal_constraints(
                query.split("\n")[0],
                nlp=self.nlp,
                reference_date=reference_date,
            )

            # Extract reference date from constraint if not provided
            if reference_date is None and temporal_constraint and not temporal_constraint.is_empty():
                if temporal_constraint.ranges:
                    reference_date = temporal_constraint.ranges[0].start

            temporal_order = self._infer_temporal_order(query)

        return {
            "constraint": temporal_constraint,
            "order": temporal_order,
            "reference_date": reference_date,
        }

    def _infer_temporal_order(self, text: str) -> str:
        """Infer temporal order from query text."""
        lowered = (text or "").lower()
        if any(token in lowered for token in ("first", "earliest", "beginning", "start", "initial")):
            return "asc"
        if any(token in lowered for token in ("last", "latest", "recent", "most recent", "newest", "final")):
            return "desc"
        return "asc"

    def _select_snapshots(
        self,
        temporal_constraint: Optional[TemporalConstraint],
        reference_date: Optional[datetime],
    ) -> List[Dict]:
        """Select appropriate snapshots based on temporal information."""
        if not self.snapshots:
            return []

        selected_snapshots = []
        snapshot_selection_reason = "none"

        # Select by temporal constraint if available
        if temporal_constraint is not None and not temporal_constraint.is_empty():
            selected_snapshots = self.snapshot_selector.select_by_constraint(temporal_constraint)
            snapshot_selection_reason = "temporal_constraint"

            # Fallback if no snapshots match constraint
            if not selected_snapshots:
                if reference_date:
                    fallback = self.snapshot_selector.select_latest_before(reference_date)
                    if fallback:
                        selected_snapshots = [fallback]
                        snapshot_selection_reason = "fallback_latest_before_question_date"

                if not selected_snapshots:
                    fallback = self.snapshot_selector.select_latest()
                    if fallback:
                        selected_snapshots = [fallback]
                        snapshot_selection_reason = "fallback_global_latest"
        else:
            # Select by reference date or latest
            if reference_date:
                latest_before = self.snapshot_selector.select_latest_before(reference_date)
                if latest_before:
                    selected_snapshots = [latest_before]
                    snapshot_selection_reason = "latest_before_question_date"

            if not selected_snapshots:
                latest = self.snapshot_selector.select_latest()
                if latest:
                    selected_snapshots = [latest]
                    snapshot_selection_reason = "global_latest"

        logger.info(
            f"Snapshot selection: reason={snapshot_selection_reason}, "
            f"selected={len(selected_snapshots)}"
        )

        return selected_snapshots

    def _perform_dense_retrieval(
        self,
        query: str,
        kwargs: Dict,
        temporal_constraint: Optional[TemporalConstraint],
        temporal_order: Optional[str],
        selected_snapshots: List[Dict],
    ) -> Dict[str, Any]:
        """Perform dense retrieval using FAISS."""
        logger.info("Performing dense retrieval")

        # Get chunks from dense retriever
        max_chunks = kwargs.get("max_chunk_setting", 25)
        if self.trace_enabled:
            chunk_ids, dense_trace = run_async_embedding(
                self.dense_retriever.retrieve_with_trace(
                    query,
                    max_chunks,
                    self.trace_dense_k,
                    selected_snapshots,
                )
            )
            self.last_retrieval_trace.update(dense_trace)
        else:
            chunk_ids = run_async_embedding(
                self.dense_retriever.retrieve(
                    query,
                    max_chunks,
                    selected_snapshots,
                )
            )

        # Apply temporal filtering
        if self.temporal_filter_enabled:
            chunk_ids = self.temporal_filter.apply_constraint(
                chunk_ids,
                temporal_constraint,
                self._last_reference_date,
                temporal_order,
            )

        # Format results
        chunks_str, summary_info = self._format_results_with_summaries(chunk_ids, selected_snapshots)

        result = {"chunks": chunks_str}
        result.update(summary_info)
        if self.trace_enabled:
            self.last_retrieval_trace.update(
                {
                    "graphiti_matched_edges": self._json_safe(
                        self.temporal_filter.last_temporal_edges
                    ),
                    "temporal_matched_chunk_ids": sorted(
                        self.temporal_filter.temporal_matched_chunk_ids
                    ),
                    "final_context_ids": [
                        chunk_id
                        for chunk_ids_for_key in chunk_ids.values()
                        for chunk_id in chunk_ids_for_key
                    ],
                }
            )
            result["retrieval_trace"] = self._json_safe(self.last_retrieval_trace)

        logger.info(f"Dense retrieval completed: {self._count_chunks(chunk_ids)} chunks")
        return result

    def _perform_graph_retrieval(
        self,
        query: str,
        entities: List[str],
        kwargs: Dict,
        temporal_constraint: Optional[TemporalConstraint],
        temporal_order: Optional[str],
        selected_snapshots: List[Dict],
    ) -> Dict[str, Any]:
        """Perform graph-based retrieval (fallback method)."""
        logger.info("Performing graph-based retrieval")

        # Local retrieval
        shortest_path_k = kwargs.get("shortest_path_k", 4)
        chunk_ids = self.graph_retriever.local_retrieval(entities, shortest_path_k)

        # Apply temporal filtering
        if self.temporal_filter_enabled:
            chunk_ids = self.temporal_filter.apply_constraint(
                chunk_ids,
                temporal_constraint,
                self._last_reference_date,
                temporal_order,
            )

        # Format results
        chunks_str = self._format_results(chunk_ids)

        result = {"chunks": chunks_str}
        logger.info(f"Graph retrieval completed: {self._count_chunks(chunk_ids)} chunks")
        return result

    def _format_results_with_summaries(
        self,
        chunk_ids: Dict[str, List[str]],
        selected_snapshots: List[Dict],
    ) -> Tuple[str, Dict[str, Any]]:
        """Format results including snapshot summaries."""
        # Separate chunks and summary nodes
        chunk_only_ids = {}
        retrieved_summary_node_ids = set()

        # Collect summary node IDs from snapshots
        selected_summary_node_ids = set()
        summary_node_to_text = {}
        summary_node_to_info = {}

        if selected_snapshots:
            for snapshot in selected_snapshots:
                snapshot_id = snapshot.get("snapshot_id", "unknown")
                valid_from = snapshot.get("valid_from", "")
                valid_to = snapshot.get("valid_to", "")

                # Collect persona summaries
                for idx, persona in enumerate(snapshot.get("persona_summaries", [])):
                    node_id = persona.get("node_id")
                    persona_text = persona.get("text", "")
                    if node_id and persona_text:
                        selected_summary_node_ids.add(node_id)
                        clean_text = persona_text.replace("###", "").replace("**", "").strip()
                        summary_node_to_text[node_id] = clean_text
                        summary_node_to_info[node_id] = {
                            "type": "persona",
                            "snapshot_id": snapshot_id,
                            "valid_from": valid_from,
                            "valid_to": valid_to,
                            "index": idx + 1
                        }

                # Collect topic summaries
                for idx, topic in enumerate(snapshot.get("topic_summaries", [])):
                    node_id = topic.get("node_id")
                    topic_text = topic.get("text", "")
                    if node_id and topic_text:
                        selected_summary_node_ids.add(node_id)
                        clean_text = topic_text.replace("###", "").replace("**", "").strip()
                        summary_node_to_text[node_id] = clean_text
                        summary_node_to_info[node_id] = {
                            "type": "topic",
                            "snapshot_id": snapshot_id,
                            "valid_from": valid_from,
                            "valid_to": valid_to,
                            "index": idx + 1
                        }

        # Separate chunks and summary nodes
        for key, chunk_list in chunk_ids.items():
            chunk_only_list = []
            for cid in chunk_list:
                if cid in selected_summary_node_ids:
                    retrieved_summary_node_ids.add(cid)
                else:
                    chunk_only_list.append(cid)
            if chunk_only_list:
                chunk_only_ids[key] = chunk_only_list

        # Format chunks
        chunks_str = self._format_results(chunk_only_ids) if chunk_only_ids else ""

        # Add retrieved summaries
        retrieved_summary_texts = []
        for node_id in retrieved_summary_node_ids:
            if node_id in summary_node_to_text:
                info = summary_node_to_info[node_id]
                text = summary_node_to_text[node_id]
                retrieved_summary_texts.append(
                    f"[Snapshot {info['snapshot_id']} ({info['valid_from']} ~ {info['valid_to']}) - "
                    f"{info['type'].capitalize()} Summary {info['index']}]\n{text}"
                )

        # Combine summaries and chunks
        if retrieved_summary_texts:
            summary_str = "\n\n".join(retrieved_summary_texts)
            if chunks_str:
                chunks_str = f"--- Retrieved Chunks ---\n{chunks_str}\n\n--- Snapshot Summaries ---\n{summary_str}"
            else:
                chunks_str = summary_str

        summary_info = {
            "retrieved_summary_count": len(retrieved_summary_node_ids),
            "selected_snapshots": selected_snapshots,
        }

        return chunks_str, summary_info

    def _format_results(self, chunk_ids: Dict[str, List[str]]) -> str:
        """Format chunk results into text."""
        result_texts = []
        for key, chunks in chunk_ids.items():
            chunk_texts = []
            for chunk_id in chunks:
                chunk_text = self._get_chunk_text(chunk_id)
                chunk_texts.append(chunk_text)
            if chunk_texts:
                result_texts.append('\n\n'.join(chunk_texts))
        return '\n\n'.join(result_texts)

    def _get_chunk_text(self, chunk_id: str) -> str:
        """Get text content for a chunk ID."""
        # Try cache_tree first
        if isinstance(self.cache_tree, dict) and chunk_id in self.cache_tree:
            try:
                return self.cache_tree[chunk_id].get("text", "")
            except Exception:
                pass

        # Try external_text_map
        if isinstance(self.external_text_map, dict):
            entry = self.external_text_map.get(chunk_id)
            if isinstance(entry, dict):
                return entry.get("text", "")
            elif isinstance(entry, str):
                return entry

        return f"[text missing for {chunk_id}]"

    def _build_metadata(
        self,
        entities: List[str],
        selected_snapshots: List[Dict],
        temporal_constraint: Optional[TemporalConstraint],
    ) -> Dict[str, Any]:
        """Build metadata for query results."""
        metadata = {
            "entities": entities,
            "retrieval_type": "Refactored Dense Retrieval",
            "temporal_constraint_applied": temporal_constraint is not None and not temporal_constraint.is_empty(),
            "snapshot_count": len(selected_snapshots),
        }

        # Add snapshot information
        if selected_snapshots:
            metadata["persona_snapshots"] = [
                {
                    "snapshot_id": s.get("snapshot_id"),
                    "valid_from": s.get("valid_from"),
                    "valid_to": s.get("valid_to"),
                    "persona_summaries": s.get("persona_summaries", []),
                }
                for s in selected_snapshots
            ]
            metadata["topic_snapshots"] = [
                {
                    "snapshot_id": s.get("snapshot_id"),
                    "valid_from": s.get("valid_from"),
                    "valid_to": s.get("valid_to"),
                    "topic_summaries": s.get("topic_summaries", []),
                }
                for s in selected_snapshots
            ]

        return metadata

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Convert Neo4j temporal values in optional traces to JSON-safe values."""
        if isinstance(value, dict):
            return {str(key): RefactoredRetriever._json_safe(child) for key, child in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [RefactoredRetriever._json_safe(child) for child in value]
        if hasattr(value, "iso_format"):
            return value.iso_format()
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return value

    @staticmethod
    def _count_chunks(chunk_map: Dict[str, List[str]]) -> int:
        """Count total chunks in chunk map."""
        return sum(len(chunk_ids) for chunk_ids in chunk_map.values())

    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'embedder'):
                del self.embedder
            if hasattr(self, 'faiss_index'):
                del self.faiss_index
            if getattr(self, '_owns_graphiti_driver', False) and getattr(self, '_graphiti_sync_driver', None):
                try:
                    self._graphiti_sync_driver.close()
                except Exception as exc:
                    logger.info(f"Failed to close Graphiti driver: {exc}")
            if torch is not None:
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# For backward compatibility, alias the original class name
Retriever = RefactoredRetriever
