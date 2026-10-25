"""
Graph utilities for entity-based retrieval.

This module handles graph-based operations including shortest path finding,
entity mapping, and chunk validation.
"""

import logging
import re
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Set, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


class GraphRetriever:
    """Handles graph-based retrieval operations."""

    def __init__(
        self,
        graph: nx.Graph,
        entity_index: Dict[str, List[str]],
        appearance_count: Dict[str, Dict[str, int]],
        cache_tree: Dict[str, Dict],
    ):
        """
        Initialize graph retriever.

        Args:
            graph: NetworkX graph of entities
            entity_index: Mapping from entities to chunk IDs
            appearance_count: Entity appearance counts per chunk
            cache_tree: Cache tree for chunk hierarchy
        """
        self.graph = graph
        self.entity_index = entity_index
        self.appearance_count = appearance_count
        self.cache_tree = cache_tree
        self.inverse_index = self._build_inverse_index()

    def _build_inverse_index(self) -> Dict[str, List[str]]:
        """Build inverse index: chunk_id -> entities."""
        inverse_index = {}
        for entity, chunk_ids in self.entity_index.items():
            for chunk_id in chunk_ids:
                inverse_index.setdefault(chunk_id, []).append(entity)
        return inverse_index

    def local_retrieval(
        self,
        entities: List[str],
        shortest_path_k: int = 4,
    ) -> Dict[str, List[str]]:
        """
        Perform local retrieval using graph shortest paths.

        Args:
            entities: List of entities to search for
            shortest_path_k: Maximum path length for entity connections

        Returns:
            Dictionary mapping entity combinations to chunk IDs
        """
        logger.info(f"Starting local retrieval, entities: {entities}, shortest_path_k: {shortest_path_k}")

        # Find shortest path pairs
        shortest_path_pairs = self._find_shortest_paths(entities, shortest_path_k)
        logger.info(f"Graph filter returned path pairs: {shortest_path_pairs}")

        # Map entities to chunks
        init_chunk_ids = self._map_entities_to_chunks(shortest_path_pairs)
        logger.info(f"Entity mapping returned chunk_ids: {init_chunk_ids}")

        # Merge and process chunks
        neighbor_nodes = self._merge_entity_keys(init_chunk_ids)
        logger.info(f"Merged neighbor_nodes: {neighbor_nodes}")

        chunk_count = sum(len(chunks) for chunks in neighbor_nodes.values())
        logger.info(f"Local retrieval final result chunk count: {chunk_count}")
        return neighbor_nodes

    def _find_shortest_paths(self, entities: List[str], k: int) -> List[Tuple[str, str]]:
        """Find entity pairs connected by short paths in the graph."""
        shortest_path_pairs = []
        logger.info(f"Checking entities in graph: {entities}")

        for head, tail in combinations(entities, 2):
            head_in_graph = head in self.graph.nodes()
            tail_in_graph = tail in self.graph.nodes()
            logger.info(f"Entity pair ({head}, {tail}): head_in_graph={head_in_graph}, tail_in_graph={tail_in_graph}")

            if head_in_graph and tail_in_graph:
                try:
                    shortest_path = nx.shortest_path(self.graph, head, tail)
                    logger.info(f"Found path: {head} -> {tail}, length: {len(shortest_path)}, path: {shortest_path}")
                except nx.NetworkXNoPath:
                    logger.info(f"No path found: {head} -> {tail}")
                    continue

                if len(shortest_path) <= k:
                    shortest_path_pairs.append((head, tail))
                    logger.info(f"Added path pair: ({head}, {tail})")

        logger.info(f"Final path pairs count: {len(shortest_path_pairs)}")
        return shortest_path_pairs

    def _map_entities_to_chunks(self, entity_pairs: List[Tuple[str, str]]) -> Dict[str, List[str]]:
        """Map entity pairs to chunk IDs."""
        logger.info(f"Mapping entities to chunks, input: {entity_pairs}")
        chunk_ids = {}

        for entity_pair in entity_pairs:
            if isinstance(entity_pair, tuple):
                chunk_ids_set = set()
                entity_key = "_".join(entity_pair)
                logger.info(f"Processing entity tuple: {entity_pair}, key: {entity_key}")

                for entity in entity_pair:
                    if entity in self.entity_index:
                        logger.info(f"Entity {entity} found in index, has {len(self.entity_index[entity])} chunks")
                        if not chunk_ids_set:
                            chunk_ids_set = set(self.entity_index[entity])
                        else:
                            chunk_ids_set = chunk_ids_set & set(self.entity_index[entity])
                            logger.info(f"Intersection resulted in {len(chunk_ids_set)} chunks")
                    else:
                        logger.info(f"Entity {entity} not found in index")

                chunk_ids[entity_key] = sorted(list(chunk_ids_set))
                logger.info(f"Entity tuple {entity_key} final chunk count: {len(chunk_ids[entity_key])}")

            elif isinstance(entity_pair, str):
                if entity_pair in self.entity_index:
                    chunk_ids[entity_pair] = self.entity_index[entity_pair]
                    logger.info(f"Found single entity {entity_pair} chunks: {len(self.entity_index[entity_pair])}")
                else:
                    logger.info(f"Single entity {entity_pair} not found in index")

        logger.info(f"Entity mapping final result: {chunk_ids}")
        return chunk_ids

    def _merge_entity_keys(self, neighbor_nodes: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Merge chunks with different entity keys."""
        logger.info(f"Merging entity keys, input: {neighbor_nodes}")
        chunks_to_keys = defaultdict(set)

        for key, chunk_lists in neighbor_nodes.items():
            logger.info(f"Processing key {key}, contains {len(chunk_lists)} chunks")
            for chunk in chunk_lists:
                chunks_to_keys[chunk].add(key)

        logger.info(f"Chunks to keys mapping: {dict(chunks_to_keys)}")
        merged_result = {}

        for chunk, keys in chunks_to_keys.items():
            # Create new key
            if len(keys) > 1:
                all_entities = set()
                for key in keys:
                    all_entities.update(key.split("_"))
                new_key = "_".join(sorted(all_entities))
                logger.info(f"Merged multiple keys {keys} into {new_key}")
            else:
                new_key = keys.pop()
                logger.info(f"Using single key: {new_key}")

            # Add chunk to new key
            merged_result.setdefault(new_key, []).append(chunk)

        logger.info(f"Merged keys final result: {merged_result}")
        return merged_result

    def detect_neighbor_nodes(self, keys: Set[str], chunk_id: str) -> List[str]:
        """Detect neighboring nodes of a chunk based on sequential IDs."""
        m = re.search(r"(\d+)$", chunk_id)
        if not m:
            # If chunk_id doesn't end with a number, can't infer neighbors
            return [chunk_id]

        int_chunk_id = int(m.group(1))
        front = True
        back = True
        neighbor_nodes = [chunk_id]
        front_int_chunk_id = int_chunk_id
        back_int_chunk_id = int_chunk_id

        while front or back:
            if front:
                front_int_chunk_id = front_int_chunk_id - 1
                if front_int_chunk_id < 0 or set(self.inverse_index.get(f"leaf_{front_int_chunk_id}", [])) & keys != keys:
                    front = False
                str_chunk_id = "leaf_{}".format(front_int_chunk_id)
                append = True
                for key in keys:
                    if str_chunk_id not in self.entity_index.get(key, []):
                        append = False
                        front = False
                        break
                if append:
                    neighbor_nodes.append(str_chunk_id)

            if back:
                back_int_chunk_id += 1
                if set(self.inverse_index.get(f"leaf_{back_int_chunk_id}", [])) & keys != keys:
                    back = False
                str_chunk_id = "leaf_{}".format(back_int_chunk_id)
                append = True
                for key in keys:
                    if str_chunk_id not in self.entity_index.get(key, []):
                        append = False
                        back = False
                        break
                if append:
                    neighbor_nodes.append(str_chunk_id)

        return neighbor_nodes

    def validate_by_checking_father_chunks(
        self,
        init_chunk_ids: Dict[str, List[str]],
        min_count: int = 2
    ) -> Dict[str, List[str]]:
        """Validate chunks by checking parent nodes in tree structure."""
        valid_child_ids = {}
        for key, chunk_ids in init_chunk_ids.items():
            father_nodes = {}
            for chunk_id in chunk_ids:
                father_chunk_id = self.cache_tree[chunk_id]["parent"]
                father_nodes.setdefault(father_chunk_id, []).append(chunk_id)
            valid_leaf_nodes = [child_id_list for _, child_id_list in father_nodes.items()
                               if len(child_id_list) >= min_count]

            valid_child_ids[key] = []
            if len(valid_leaf_nodes) > 0:
                for leaf_node in valid_leaf_nodes:
                    valid_child_ids[key].extend(leaf_node)

        return valid_child_ids

    def entityaware_filter(
        self,
        candidate_chunks: Dict[str, List[str]],
        entities: List[str],
        max_chunks: int = 25,
    ) -> Dict[str, List[str]]:
        """
        Filter chunks based on entity-aware criteria.

        Filtering rules:
        1. Chunks with more different entities have higher priority
        2. Chunks with longer neighbor sequences have higher priority
        3. Chunks with higher entity appearance counts have higher priority
        """
        chunks_info = []
        for key, value in candidate_chunks.items():
            for chunk_id in value:
                key_count = len(key.split("_"))
                set_key = set(key.split("_"))
                neighbor_nodes_count = len(self.detect_neighbor_nodes(keys=set_key, chunk_id=chunk_id))
                entity_count = 0
                for key_entity in key.split('_'):
                    entity_count += self.appearance_count.get(chunk_id, {}).get(key_entity, 0)

                chunk_id_info = {
                    "chunk_id": chunk_id,
                    "key_count": key_count,
                    "neighbor_nodes_count": neighbor_nodes_count,
                    "entity_count": entity_count
                }
                chunks_info.append(chunk_id_info)

        # Sort by criteria
        sorted_chunks_info = sorted(
            chunks_info,
            key=lambda x: (x["key_count"], x["neighbor_nodes_count"], x["entity_count"]),
            reverse=True
        )

        # Get top chunks
        top_chunks = sorted_chunks_info[:max_chunks]
        top_chunk_ids = [chunk["chunk_id"] for chunk in top_chunks]

        # Build filtered result
        filtered_res = {}
        for chunk_id in top_chunk_ids:
            for entity in entities:
                if chunk_id in self.entity_index.get(entity, []):
                    filtered_res.setdefault(entity, []).append(chunk_id)

        filtered_res = self._merge_entity_keys(filtered_res)
        logger.info(f"Entity-aware filter result: {filtered_res}")
        return filtered_res


class OccurrenceRanker:
    """Ranks chunks by entity occurrence counts."""

    def __init__(
        self,
        appearance_count: Dict[str, Dict[str, int]],
        cache_tree: Dict[str, Dict],
    ):
        """
        Initialize occurrence ranker.

        Args:
            appearance_count: Entity appearance counts per chunk
            cache_tree: Cache tree for chunk hierarchy
        """
        self.appearance_count = appearance_count
        self.cache_tree = cache_tree

    def rank_chunks(
        self,
        candidate_chunk_ids: List[str],
        entities: List[str],
        max_chunks: int = 25,
    ) -> List[str]:
        """
        Rank chunks by entity occurrence counts.

        Args:
            candidate_chunk_ids: List of candidate chunk IDs
            entities: List of entities to count
            max_chunks: Maximum number of chunks to return

        Returns:
            List of ranked chunk IDs
        """
        logger.info(
            f"Occurrence ranking input: {len(candidate_chunk_ids)} candidates, "
            f"entities={entities}, max_chunks={max_chunks}"
        )

        chunk_counts = []
        for i, chunk_id in enumerate(candidate_chunk_ids):
            if not chunk_id.startswith("leaf_"):
                count = self._check_children(chunk_id, entities)
                chunk_counts.append(count)
                logger.info(f"Chunk {i} ({chunk_id}): non-leaf node, count={count}")
                continue

            chunk_appearance_stat = self.appearance_count.get(chunk_id, {})
            logger.info(f"Chunk {i} ({chunk_id}): appearance_count={chunk_appearance_stat}")

            this_chunk_count = 0
            for entity in entities:
                entity_count = chunk_appearance_stat.get(entity, 0)
                this_chunk_count += entity_count
                if entity_count > 0:
                    logger.info(f"Chunk {i} ({chunk_id}): entity {entity} appears {entity_count} times")

            chunk_counts.append(this_chunk_count)
            logger.info(f"Chunk {i} ({chunk_id}): total count={this_chunk_count}")

        # Filter and sort by count
        import numpy as np
        chunk_counts = np.array(chunk_counts)
        nonzero_indices = np.nonzero(chunk_counts)[0]
        logger.info(f"Nonzero indices count: {len(nonzero_indices)}")

        if len(nonzero_indices) == 0:
            logger.info(f"All chunks have zero count, returning first {max_chunks}")
            return candidate_chunk_ids[:max_chunks]

        argsorted_chunk_ids = np.argsort(chunk_counts)[::-1]
        filtered_chunk_ids = [candidate_chunk_ids[i] for i in argsorted_chunk_ids if chunk_counts[i] > 0]
        logger.info(f"Filtered chunk count: {len(filtered_chunk_ids)}")

        if len(filtered_chunk_ids) > max_chunks:
            filtered_chunk_ids = filtered_chunk_ids[:max_chunks]
            logger.info(f"Truncated to {max_chunks} chunks")

        return filtered_chunk_ids

    def _check_children(self, chunk_id: str, entities: List[str], visited=None) -> int:
        """Recursively check children nodes for entity counts."""
        if visited is None:
            visited = set()

        if chunk_id in visited:
            return 0

        visited.add(chunk_id)
        entity_count = 0

        children = self.cache_tree.get(chunk_id, {}).get("children", [])
        if children is None:
            children = []

        for child in children:
            if not child.startswith("leaf_"):
                entity_count += self._check_children(child, entities, visited)
            else:
                chunk_appearance_stat = self.appearance_count.get(child, {})
                for entity in entities:
                    entity_count += chunk_appearance_stat.get(entity, 0)

        return entity_count