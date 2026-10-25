"""
Dense retrieval utilities using FAISS and embeddings.

This module handles dense vector-based retrieval using FAISS indexes
and public embedding services.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import faiss
import numpy as np
import nest_asyncio

from temporal_utils import TemporalConstraint

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Handles dense vector-based retrieval using FAISS."""

    def __init__(
        self,
        embedder,
        faiss_index,
        collapse_tree_ids: List[str],
        use_gpu_faiss: bool = True,
        gpu_id: int = 0,
        external_text_map: Optional[Dict] = None,
        cache_tree: Optional[Dict] = None,
        chunks_only: bool = False,
    ):
        """
        Initialize dense retriever.

        Args:
            embedder: Embedding model
            faiss_index: FAISS index for similarity search
            collapse_tree_ids: List of document IDs corresponding to index
            use_gpu_faiss: Whether to use GPU acceleration
            gpu_id: GPU device ID
            external_text_map: External text mapping for chunks
            cache_tree: Cache tree for chunk text lookup
        """
        self.embedder = embedder
        self.faiss_index = faiss_index
        self.collapse_tree_ids = collapse_tree_ids
        self.use_gpu_faiss = use_gpu_faiss
        self.gpu_id = gpu_id
        self.external_text_map = external_text_map or {}
        self.cache_tree = cache_tree or {}
        self.chunks_only = chunks_only

    async def retrieve(
        self,
        query: str,
        k: int,
        selected_snapshots: Optional[List[Dict]] = None,
    ) -> Dict[str, List[str]]:
        """
        Perform dense retrieval for query.

        Args:
            query: Query string
            k: Number of results to retrieve
            selected_snapshots: Selected snapshots for summary nodes

        Returns:
            Dictionary mapping empty string to list of chunk IDs
        """
        if self.faiss_index is None:
            logger.warning("FAISS index not available, returning empty results")
            return {"": []}

        logger.info(f"Starting dense retrieval, query: {query}, k: {k}")

        # Encode query
        query_embed = await self._encode_query(query)
        if query_embed is None:
            logger.warning("Failed to encode query, returning empty results")
            return {"": []}

        # Search FAISS index
        candidate_chunk_ids = self._search_faiss(query_embed, k)

        # Collect selected summary node IDs
        selected_summary_node_ids = self._collect_summary_node_ids(selected_snapshots)

        # Filter results
        filtered_chunk_ids = self._filter_candidates(
            candidate_chunk_ids, selected_summary_node_ids, selected_snapshots
        )

        logger.info(f"Dense retrieval completed: {len(filtered_chunk_ids)} results")
        return {"": filtered_chunk_ids}

    async def retrieve_with_trace(
        self,
        query: str,
        k: int,
        trace_k: int,
        selected_snapshots: Optional[List[Dict]] = None,
    ) -> tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """Retrieve with an optional wider FAISS audit, preserving baseline output.

        The first return value is exactly the result produced by ``retrieve`` with
        ``k``: the raw FAISS list is sliced before candidate filtering. The second
        return value exposes the filtered candidate IDs at ``trace_k``. This keeps
        tracing observational while avoiding a second query embedding call.
        """
        if self.faiss_index is None:
            return {"": []}, {"dense_top20_ids": [], "dense_top100_ids": []}

        query_embed = await self._encode_query(query)
        if query_embed is None:
            return {"": []}, {"dense_top20_ids": [], "dense_top100_ids": []}

        search_k = max(int(k), int(trace_k))
        raw_candidates = self._search_faiss(query_embed, search_k)
        selected_summary_node_ids = self._collect_summary_node_ids(selected_snapshots)

        baseline_candidates = self._filter_candidates(
            raw_candidates[:k], selected_summary_node_ids, selected_snapshots
        )
        traced_candidates = self._filter_candidates(
            raw_candidates[:trace_k], selected_summary_node_ids, selected_snapshots
        )
        trace = {
            "dense_top20_ids": baseline_candidates,
            "dense_top100_ids": traced_candidates,
        }
        return {"": baseline_candidates}, trace

    async def _encode_query(self, query: str) -> Optional[np.ndarray]:
        """Encode query using embedder."""
        try:
            query_embed = await self.embedder.create(query)
            if isinstance(query_embed, list) and len(query_embed) > 0:
                query_embed = np.array(query_embed, dtype='float32')
            else:
                logger.warning(f"Invalid embedding returned: {query_embed}")
                # Provide default random vector
                dimension = int(getattr(self.embedder, "dimension", 1536))
                query_embed = np.array([0.0] * dimension, dtype='float32')

            # Ensure proper shape and type
            query_embed = np.ascontiguousarray(query_embed.reshape(1, -1).astype('float32'))
            return query_embed

        except Exception as exc:
            logger.error(f"Failed to encode query: {exc}")
            return None

    def _search_faiss(self, query_embed: np.ndarray, k: int) -> List[str]:
        """Search FAISS index and return candidate chunk IDs."""
        try:
            _, candidate_indices = self.faiss_index.search(query_embed, k=k)
            candidate_indices = candidate_indices[0]  # Remove batch dimension

            candidate_chunk_ids = [self.collapse_tree_ids[i] for i in candidate_indices]
            logger.info(f"FAISS search returned {len(candidate_chunk_ids)} candidates")

            # Log first few results for debugging
            for i, chunk_id in enumerate(candidate_chunk_ids[:3]):
                chunk_content = self._get_chunk_content(chunk_id)
                faiss_index = candidate_indices[i]
                logger.info(
                    f"Candidate {i}: FAISS index={faiss_index}, "
                    f"chunk_id={chunk_id}, content={chunk_content[:100]}..."
                )

            return candidate_chunk_ids

        except Exception as exc:
            logger.error(f"FAISS search failed: {exc}")
            return []

    def _collect_summary_node_ids(self, selected_snapshots: Optional[List[Dict]]) -> Set[str]:
        """Collect summary node IDs from selected snapshots."""
        selected_summary_node_ids: Set[str] = set()

        if not selected_snapshots:
            return selected_summary_node_ids

        for snapshot in selected_snapshots:
            # Collect persona summary node IDs
            persona_summaries = snapshot.get("persona_summaries", [])
            for persona in persona_summaries:
                node_id = persona.get("node_id")
                if node_id:
                    selected_summary_node_ids.add(node_id)

            # Collect topic summary node IDs
            topic_summaries = snapshot.get("topic_summaries", [])
            for topic in topic_summaries:
                node_id = topic.get("node_id")
                if node_id:
                    selected_summary_node_ids.add(node_id)

        if selected_summary_node_ids:
            logger.info(f"Collected {len(selected_summary_node_ids)} summary node IDs from snapshots")

        return selected_summary_node_ids

    def _filter_candidates(
        self,
        candidate_chunk_ids: List[str],
        selected_summary_node_ids: Set[str],
        selected_snapshots: Optional[List[Dict]],
    ) -> List[str]:
        """Filter candidate chunk IDs."""
        filtered_chunk_ids: List[str] = []

        for cid in candidate_chunk_ids:
            if isinstance(cid, str):
                # Keep leaf_* and Graphiti UUID chunks
                if cid.startswith("leaf_") or self._is_graphiti_chunk_id(cid):
                    filtered_chunk_ids.append(cid)
                # Keep selected summary nodes if snapshots provided
                elif selected_snapshots and cid in selected_summary_node_ids:
                    filtered_chunk_ids.append(cid)
                    logger.info(f"Retrieved selected summary node: {cid}")

        return filtered_chunk_ids

    def _is_graphiti_chunk_id(self, chunk_id: str) -> bool:
        """Check if chunk ID is a Graphiti UUID."""
        return isinstance(chunk_id, str) and chunk_id.count('-') == 4

    def _get_chunk_content(self, chunk_id: str) -> str:
        """Get text content for a chunk ID."""
        # 消融实验：如果启用chunks_only，跳过summary_开头的persona节点
        if self.chunks_only and chunk_id.startswith("summary_"):
            return ""

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


class FAISSIndexBuilder:
    """Builds FAISS indexes for dense retrieval."""

    def __init__(
        self,
        embedder,
        use_gpu: bool = True,
        gpu_id: int = 0,
        embedding_batch_size: int = 1024,
    ):
        """
        Initialize FAISS index builder.

        Args:
            embedder: Embedding model
            use_gpu: Whether to use GPU acceleration
            gpu_id: GPU device ID
            embedding_batch_size: Batch size for embedding
        """
        self.embedder = embedder
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.embedding_batch_size = embedding_batch_size

    async def build_index(self, documents: List[str]) -> Optional[faiss.Index]:
        """
        Build FAISS index from documents.

        Args:
            documents: List of document texts

        Returns:
            Built FAISS index or None if failed
        """
        if not documents:
            logger.warning("No documents to embed")
            return None

        logger.info(f"Building FAISS index for {len(documents)} documents")

        # Generate embeddings
        doc_embeds = await self._embed_documents(documents)
        if doc_embeds is None:
            return None

        # Create FAISS index
        return self._create_faiss_index(doc_embeds)

    async def _embed_documents(self, documents: List[str]) -> Optional[np.ndarray]:
        """Embed documents in batches."""
        try:
            # Check if embedder has async methods
            if hasattr(self.embedder, 'embed_texts'):
                # Use the public async embedding method.
                results = []
                for i in range(0, len(documents), self.embedding_batch_size):
                    batch = documents[i:i+self.embedding_batch_size]
                    batch_embeds = await self.embedder.embed_texts(batch)
                    results.extend(batch_embeds)

                # Ensure float32 dtype and contiguous memory for FAISS
                doc_embeds = np.ascontiguousarray(np.array(results, dtype='float32'))

            else:
                # SentenceTransformer - use sync encode method
                doc_embeds = self.embedder.encode(documents, batch_size=16)

            # Validate embeddings
            if doc_embeds.ndim != 2:
                raise ValueError(f"doc_embeds should be 2D array, got shape: {doc_embeds.shape}")
            if doc_embeds.shape[0] == 0:
                raise ValueError("doc_embeds is empty")
            if doc_embeds.shape[1] == 0:
                raise ValueError("doc_embeds has 0 dimensions")

            logger.info(f"Generated embeddings: shape={doc_embeds.shape}, dtype={doc_embeds.dtype}")
            return doc_embeds

        except Exception as exc:
            logger.error(f"Failed to embed documents: {exc}")
            return None

    def _create_faiss_index(self, doc_embeds: np.ndarray) -> Optional[faiss.Index]:
        """Create FAISS index from embeddings."""
        try:
            # Create CPU index
            vector_database = faiss.IndexFlatIP(doc_embeds.shape[1])

            # Try to move to GPU if enabled
            if self.use_gpu and hasattr(faiss, 'StandardGpuResources'):
                try:
                    gpu_res = faiss.StandardGpuResources()
                    vector_database = faiss.index_cpu_to_gpu(gpu_res, self.gpu_id, vector_database)
                    logger.info(
                        f"FAISS index moved to GPU {self.gpu_id}, "
                        f"dimensions: {doc_embeds.shape[1]}, vectors: {doc_embeds.shape[0]}"
                    )
                except Exception as e:
                    logger.warning(f"GPU index creation failed, using CPU: {e}")
                    vector_database = faiss.IndexFlatIP(doc_embeds.shape[1])
            else:
                logger.info(
                    f"Using CPU FAISS index, "
                    f"dimensions: {doc_embeds.shape[1]}, vectors: {doc_embeds.shape[0]}"
                )

            # Add vectors to index
            vector_database.add(doc_embeds)
            return vector_database

        except Exception as exc:
            logger.error(f"Failed to create FAISS index: {exc}")
            return None


def run_async_embedding(coro):
    """Run async embedding function, handling event loop properly."""
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        # Apply nest_asyncio to allow nested event loops
        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop, create a new one
        return asyncio.run(coro)
