"""
Temporal filtering utilities for chunk retrieval.

This module handles filtering and ranking of chunks based on temporal constraints
and Graphiti edge information.
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from temporal_utils import TemporalConstraint, parse_iso_datetime

logger = logging.getLogger(__name__)


class TemporalFilter:
    """Handles temporal filtering and ranking of chunks."""

    def __init__(
        self,
        external_text_map: Optional[Dict] = None,
        cache_tree: Optional[Dict] = None,
        keep_untimed_chunks: bool = True,
        keep_untimed_edges: bool = True,
        graphiti_sync_driver=None,
        graphiti_database: Optional[str] = None,
        graphiti_group_id: Optional[str] = None,
    ):
        """
        Initialize temporal filter.

        Args:
            external_text_map: External text mapping for chunks
            cache_tree: Cache tree for chunk text lookup
            keep_untimed_chunks: Whether to keep chunks without time info
            keep_untimed_edges: Whether to keep edges without time info
            graphiti_sync_driver: Neo4j driver for Graphiti queries
            graphiti_database: Graphiti database name
            graphiti_group_id: Group ID for data isolation
        """
        self.external_text_map = external_text_map or {}
        self.cache_tree = cache_tree or {}
        self.keep_untimed_chunks = keep_untimed_chunks
        self.keep_untimed_edges = keep_untimed_edges
        self.graphiti_sync_driver = graphiti_sync_driver
        self.graphiti_database = graphiti_database
        self.graphiti_group_id = graphiti_group_id

        # State tracking
        self.last_temporal_edges: List[Dict[str, Any]] = []
        self.temporal_matched_chunk_ids: Set[str] = set()

    def apply_constraint(
        self,
        chunk_map: Dict[str, List[str]],
        constraint: Optional[TemporalConstraint],
        reference_date: Optional[datetime] = None,
        sort_order: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """
        Apply temporal constraint to chunk map.

        Args:
            chunk_map: Mapping of keys to chunk IDs
            constraint: Temporal constraint to apply
            reference_date: Reference date for filtering
            sort_order: Sort order for results

        Returns:
            Filtered and possibly reordered chunk map
        """
        if not isinstance(chunk_map, dict):
            return chunk_map

        # Reset per-query state so an audit trace cannot inherit matches from the
        # preceding question when a local fallback is used.
        self.last_temporal_edges = []
        self.temporal_matched_chunk_ids = set()

        # No constraint: return as-is or apply reference date filtering
        if constraint is None or constraint.is_empty():
            return chunk_map

        # Try Graphiti-based filtering first
        graphiti_result = self._apply_graphiti_filter(
            chunk_map, constraint, reference_date, sort_order
        )

        if graphiti_result is not None:
            return graphiti_result

        # Fallback to local filtering
        return self._apply_local_filter(chunk_map, constraint, reference_date)

    def _apply_graphiti_filter(
        self,
        chunk_map: Dict[str, List[str]],
        constraint: TemporalConstraint,
        reference_date: Optional[datetime],
        sort_order: Optional[str],
    ) -> Optional[Dict[str, List[str]]]:
        """Apply Graphiti-based temporal filtering."""
        if not self._graphiti_available():
            logger.info("Graphiti not available, falling back to local filtering")
            return None

        if not chunk_map:
            logger.info("Empty chunk map, falling back to local filtering")
            return None

        # Collect chunk texts for matching
        chunk_texts, chunk_id_to_text = self._collect_chunk_texts(chunk_map)

        logger.info(
            f"Graphiti temporal filtering: collected {len(chunk_texts)} chunk texts, "
            f"{len(chunk_id_to_text)} chunk IDs, querying edges with time constraints"
        )

        # Query edges with time constraints
        edges = self._fetch_temporal_edges(constraint, reference_date)

        if not edges:
            self.temporal_matched_chunk_ids = set()
            self.last_temporal_edges = []
            logger.info("No temporal edges found, returning original chunk map")
            return chunk_map

        logger.info(f"Found {len(edges)} temporal edges, matching to chunks")

        # Match edges to chunks
        matched_items, chunk_edge_map = self._match_edges_to_chunks(
            edges, chunk_texts, chunk_id_to_text
        )

        # Update state
        self.temporal_matched_chunk_ids = set()
        if matched_items:
            for item in matched_items:
                matched_chunk_ids = item.get("matched_chunk_ids", [])
                self.temporal_matched_chunk_ids.update(matched_chunk_ids)
            self.last_temporal_edges = [item["edge"] for item in matched_items]
        else:
            self.last_temporal_edges = []

        logger.info(
            f"Graphiti temporal reranking: "
            f"original {self._count_chunks(chunk_map)} chunks, "
            f"matched {len(matched_items)} edges, "
            f"matched chunk IDs: {len(self.temporal_matched_chunk_ids)}"
        )

        # Reorder chunks by temporal match
        return self._reorder_by_temporal_match(chunk_map)

    def _apply_local_filter(
        self,
        chunk_map: Dict[str, List[str]],
        constraint: TemporalConstraint,
        reference_date: Optional[datetime],
    ) -> Dict[str, List[str]]:
        """Apply local temporal filtering."""
        filtered: Dict[str, List[str]] = {}
        removed = 0
        original = 0

        for key, chunk_ids in chunk_map.items():
            filtered_ids: List[str] = []
            for chunk_id in chunk_ids:
                original += 1
                if self._chunk_matches_constraint(chunk_id, constraint, reference_date):
                    filtered_ids.append(chunk_id)
                else:
                    removed += 1
            if filtered_ids:
                filtered[key] = filtered_ids

        logger.info(
            f"Local temporal filtering: checked {original} chunks, "
            f"filtered {removed}, remaining {self._count_chunks(filtered)}"
        )
        return filtered

    def _graphiti_available(self) -> bool:
        """Check if Graphiti is available for queries."""
        return (
            self.graphiti_sync_driver is not None
            and isinstance(self.graphiti_group_id, str)
            and self.graphiti_group_id != ''
        )

    def _collect_chunk_texts(
        self, chunk_map: Dict[str, List[str]]
    ) -> Tuple[Set[str], Dict[str, str]]:
        """Collect chunk texts for matching."""
        chunk_texts = set()
        chunk_id_to_text = {}

        for ids in chunk_map.values():
            for cid in ids:
                chunk_text = self._get_chunk_text(cid)
                if chunk_text:
                    chunk_texts.add(chunk_text)
                    chunk_id_to_text[cid] = chunk_text

        return chunk_texts, chunk_id_to_text

    def _get_chunk_text(self, chunk_id: str) -> Optional[str]:
        """Get text content for a chunk ID."""
        # Try cache_tree first
        if isinstance(self.cache_tree, dict) and chunk_id in self.cache_tree:
            try:
                chunk_text = self.cache_tree[chunk_id].get("text", "").strip()
                if chunk_text:
                    return chunk_text
            except Exception as exc:
                logger.debug(f"Failed to get chunk text from cache_tree {chunk_id}: {exc}")

        # Try external_text_map
        if isinstance(self.external_text_map, dict):
            external_entry = self.external_text_map.get(chunk_id)
            if isinstance(external_entry, dict):
                chunk_text = external_entry.get("text", "").strip()
            elif isinstance(external_entry, str):
                chunk_text = external_entry.strip()
            else:
                chunk_text = None

            if chunk_text:
                return chunk_text

        return None

    def _fetch_temporal_edges(
        self,
        constraint: TemporalConstraint,
        reference_date: Optional[datetime],
    ) -> List[Dict[str, Any]]:
        """Fetch edges from Graphiti with temporal constraints."""
        if not self._graphiti_available():
            return []

        # Build time constraint conditions
        time_conditions = []
        query_params = {"group_id": self.graphiti_group_id}

        if constraint and not constraint.is_empty():
            # Build range conditions for explicit time constraints
            range_conditions = []
            for i, temp_range in enumerate(constraint.ranges):
                param_start = f"constraint_start_{i}"
                param_end = f"constraint_end_{i}"
                query_params[param_start] = temp_range.start.isoformat()
                query_params[param_end] = temp_range.end.isoformat()

                range_conditions.append(
                    f"(e.valid_at IS NOT NULL AND e.valid_at >= datetime(${param_start}) AND e.valid_at <= datetime(${param_end})) OR "
                    f"(e.valid_at IS NULL AND e.created_at IS NOT NULL AND e.created_at >= datetime(${param_start}) AND e.created_at <= datetime(${param_end}))"
                )

            if range_conditions:
                time_condition = "(" + " OR ".join(range_conditions) + ")"
                if self.keep_untimed_edges:
                    time_conditions.append(f"({time_condition} OR (e.valid_at IS NULL AND e.created_at IS NULL))")
                else:
                    time_conditions.append(time_condition)

        elif reference_date is not None:
            # Filter by reference date
            param_ref_date = "reference_date"
            query_params[param_ref_date] = reference_date.isoformat()
            if self.keep_untimed_edges:
                time_conditions.append(
                    f"(e.valid_at IS NULL AND e.created_at IS NULL) OR "
                    f"(e.valid_at IS NOT NULL AND e.valid_at <= datetime(${param_ref_date})) OR "
                    f"(e.valid_at IS NULL AND e.created_at IS NOT NULL AND e.created_at <= datetime(${param_ref_date}))"
                )
            else:
                time_conditions.append(
                    f"(e.valid_at IS NOT NULL AND e.valid_at <= datetime(${param_ref_date})) OR "
                    f"(e.valid_at IS NULL AND e.created_at IS NOT NULL AND e.created_at <= datetime(${param_ref_date}))"
                )

        # Build WHERE clause
        where_clauses = ["e.group_id = $group_id"]
        if time_conditions:
            where_clauses.append("(" + " OR ".join(time_conditions) + ")")

        where_clause = " AND ".join(where_clauses)

        # Query edges with episode details
        query = f"""
        MATCH (src)-[e:RELATES_TO]->(dst)
        WHERE {where_clause}
        WITH e, src, dst, e.episodes AS episode_uuids
        UNWIND episode_uuids AS ep_uuid
        OPTIONAL MATCH (ep:Episodic {{uuid: ep_uuid}})
        WHERE ep.group_id = $group_id
        WITH e, src, dst, collect({{
            uuid: ep_uuid,
            name: ep.name,
            content: ep.content,
            created_at: ep.created_at,
            valid_at: ep.valid_at
        }}) AS episode_details
        RETURN
            e.uuid AS edge_uuid,
            e.fact AS fact,
            e.name AS relation,
            e.created_at AS created_at,
            e.valid_at AS valid_at,
            e.invalid_at AS invalid_at,
            e.episodes AS episodes,
            episode_details AS episode_details,
            src.uuid AS source_uuid,
            src.name AS source_name,
            dst.uuid AS target_uuid,
            dst.name AS target_name
        """

        records = self._run_graphiti_query(query, **query_params)
        constraint_info = "有时间约束" if constraint and not constraint.is_empty() else ("有问题日期" if reference_date else "无时间限制")
        logger.info(f"Edge query: found {len(records)} edges ({constraint_info}) with episode details")
        return records

    def _run_graphiti_query(self, query: str, **params) -> List[Dict[str, Any]]:
        """Run a Graphiti query and return results."""
        if not self.graphiti_sync_driver:
            return []

        session_kwargs: Dict[str, Any] = {}
        if self.graphiti_database:
            session_kwargs["database"] = self.graphiti_database

        try:
            with self.graphiti_sync_driver.session(**session_kwargs) as session:
                result = session.run(query, **params)
                return [record.data() for record in result]
        except Exception as exc:
            logger.warning(f"Graphiti query failed, falling back to local filtering: {exc}")
            return []

    def _match_edges_to_chunks(
        self,
        edges: List[Dict[str, Any]],
        chunk_texts: Set[str],
        chunk_id_to_text: Dict[str, str],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        """
        Match edges to chunks using episode names.

        Args:
            edges: List of edge records
            chunk_texts: Set of chunk texts (for backward compatibility)
            chunk_id_to_text: Mapping of chunk ID to text

        Returns:
            Tuple of (matched_items, chunk_edge_map)
        """
        logger.info(
            f"Matching edges to chunks (via episode names): "
            f"input edges={len(edges)}, chunk_id_to_text={len(chunk_id_to_text)}"
        )

        if not edges:
            return [], {}

        chunk_edge_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        matched_items: List[Dict[str, Any]] = []

        stats = {"total": len(edges), "matched": 0, "no_match": 0}

        for edge in edges:
            edge_uuid = edge.get("edge_uuid", edge.get("uuid", "unknown"))
            episode_details = edge.get("episode_details", [])

            matched_chunk_ids = []

            for ep_detail in episode_details:
                if not isinstance(ep_detail, dict):
                    continue

                ep_name = ep_detail.get("name", "")
                if not ep_name:
                    # Fallback to text matching for backward compatibility
                    ep_content = ep_detail.get("content", "")
                    if ep_content:
                        ep_content_normalized = ep_content.strip()
                        if ep_content_normalized in chunk_texts:
                            for cid, ctext in chunk_id_to_text.items():
                                if ctext == ep_content_normalized:
                                    if cid not in matched_chunk_ids:
                                        matched_chunk_ids.append(cid)
                    continue

                # Match by episode name: chunk_{i} -> leaf_{i}
                if ep_name.startswith("chunk_"):
                    try:
                        chunk_index = int(ep_name.replace("chunk_", ""))
                        chunk_id = f"leaf_{chunk_index}"

                        if chunk_id in chunk_id_to_text:
                            if chunk_id not in matched_chunk_ids:
                                matched_chunk_ids.append(chunk_id)
                                logger.debug(f"Matched episode {ep_name} -> chunk_id {chunk_id}")
                        else:
                            logger.debug(f"Episode {ep_name} -> chunk_id {chunk_id} not found")
                    except ValueError:
                        logger.debug(f"Could not extract index from episode name {ep_name}")
                        continue

            if not matched_chunk_ids:
                stats["no_match"] += 1
                logger.debug(f"Edge {edge_uuid}: no matching chunks found")
                continue

            stats["matched"] += 1
            logger.debug(f"Edge {edge_uuid}: found {len(matched_chunk_ids)} matching chunks")

            # Create matched item
            item = {"edge": edge, "matched_chunk_ids": matched_chunk_ids}
            matched_items.append(item)

            # Build chunk-to-edge mapping
            for chunk_id in matched_chunk_ids:
                chunk_edge_map[chunk_id].append(item)

        logger.info(
            f"Edge matching stats: total={stats['total']}, "
            f"matched={stats['matched']}, no_match={stats['no_match']}, "
            f"matched chunks={len(chunk_edge_map)}"
        )

        return matched_items, chunk_edge_map

    def _reorder_by_temporal_match(
        self, chunk_map: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """Reorder chunks to put temporally matched ones first."""
        if not isinstance(chunk_map, dict) or not self.temporal_matched_chunk_ids:
            return chunk_map

        reordered_chunk_map: Dict[str, List[str]] = {}

        for key, chunk_ids in chunk_map.items():
            # Separate matched and unmatched chunks
            matched_chunks = []
            unmatched_chunks = []

            for chunk_id in chunk_ids:
                if chunk_id in self.temporal_matched_chunk_ids:
                    matched_chunks.append(chunk_id)
                else:
                    unmatched_chunks.append(chunk_id)

            # Put matched chunks first
            reordered_chunk_map[key] = matched_chunks + unmatched_chunks

        logger.info(f"Reordered chunks: {len(self.temporal_matched_chunk_ids)} temporal matches moved to front")
        return reordered_chunk_map

    def _chunk_matches_constraint(
        self,
        chunk_id: str,
        constraint: Optional[TemporalConstraint],
        reference_date: Optional[datetime] = None,
    ) -> bool:
        """Check if chunk matches temporal constraint."""
        chunk_datetimes = self._collect_chunk_datetimes(chunk_id)

        # Apply explicit temporal constraint
        if constraint is not None and not constraint.is_empty():
            if not chunk_datetimes:
                if not self.keep_untimed_chunks:
                    logger.info(f"Temporal filtering: chunk {chunk_id} has no time info, filtered")
                    return False
                logger.info(f"Temporal filtering: chunk {chunk_id} has no time info, kept")
                return True

            for dt in chunk_datetimes:
                if constraint.matches(dt):
                    logger.info(f"Temporal filtering: chunk {chunk_id} matches time {dt.isoformat()}, kept")
                    return True

            logger.info(f"Temporal filtering: chunk {chunk_id} doesn't match constraint, filtered")
            return False

        # Apply reference date filtering
        if reference_date is not None:
            if not chunk_datetimes:
                if not self.keep_untimed_chunks:
                    logger.info(f"Temporal filtering (before ref date): chunk {chunk_id} has no time info, filtered")
                    return False
                logger.info(f"Temporal filtering (before ref date): chunk {chunk_id} has no time info, kept")
                return True

            # Check if any time is before reference date
            for dt in chunk_datetimes:
                if dt <= reference_date:
                    logger.info(
                        f"Temporal filtering (before ref date): chunk {chunk_id} time {dt.isoformat()} "
                        f"before reference {reference_date.isoformat()}, kept"
                    )
                    return True

            logger.info(
                f"Temporal filtering (before ref date): chunk {chunk_id} all times after "
                f"reference {reference_date.isoformat()}, filtered"
            )
            return False

        # No constraints
        return True

    def _collect_chunk_datetimes(self, chunk_id: str) -> List[datetime]:
        """Collect all datetime values from a chunk."""
        datetimes: List[datetime] = []
        seen: Set[str] = set()

        def add_datetime(value: Optional[str]):
            dt = parse_iso_datetime(value)
            if dt is None:
                return
            iso = dt.isoformat()
            if iso not in seen:
                seen.add(iso)
                datetimes.append(dt)

        if isinstance(self.external_text_map, dict):
            external_entry = self.external_text_map.get(chunk_id)
        else:
            external_entry = None

        potential_sources: List = []
        if isinstance(external_entry, dict):
            potential_sources.append(external_entry)
            metadata = external_entry.get("metadata")
            if isinstance(metadata, dict):
                potential_sources.append(metadata)

        for source in potential_sources:
            if not isinstance(source, dict):
                continue

            for key in ("reference_time", "valid_at", "invalid_at", "time", "start_time", "end_time"):
                value = source.get(key)
                if isinstance(value, str):
                    add_datetime(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            add_datetime(item)

            temporal_edges = source.get("temporal_edges")
            if isinstance(temporal_edges, list):
                for edge in temporal_edges:
                    if not isinstance(edge, dict):
                        continue
                    for key in ("time", "valid_at", "invalid_at", "reference_time", "start_time", "end_time"):
                        val = edge.get(key)
                        if isinstance(val, str):
                            add_datetime(val)

        return datetimes

    @staticmethod
    def _count_chunks(chunk_map: Dict[str, List[str]]) -> int:
        """Count total chunks in chunk map."""
        return sum(len(chunk_ids) for chunk_ids in chunk_map.values())
