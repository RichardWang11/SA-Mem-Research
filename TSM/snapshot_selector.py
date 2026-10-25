"""
Snapshot selection utilities for temporal retrieval.

This module handles the selection of appropriate snapshots based on temporal constraints
and reference dates.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

from temporal_utils import TemporalConstraint, parse_iso_datetime

logger = logging.getLogger(__name__)


class SnapshotSelector:
    """Handles snapshot selection based on temporal constraints."""

    def __init__(self, snapshots: List[Dict]):
        """
        Initialize snapshot selector.

        Args:
            snapshots: List of snapshot dictionaries
        """
        self.snapshots = snapshots

    def _get_snapshot_bounds(self, snapshot: Dict) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Parse snapshot time window.

        Args:
            snapshot: Snapshot dictionary

        Returns:
            Tuple of (start_time, end_time), where None values indicate unbounded
        """
        start = parse_iso_datetime(snapshot.get("valid_from"))
        end = parse_iso_datetime(snapshot.get("valid_to"))
        return start, end

    def _snapshot_overlaps_range(
        self,
        snapshot: Dict,
        range_start: datetime,
        range_end: datetime,
    ) -> bool:
        """
        Check if snapshot window overlaps with given range.

        Args:
            snapshot: Snapshot dictionary
            range_start: Start of time range
            range_end: End of time range

        Returns:
            True if snapshot overlaps with range
        """
        snap_start, snap_end = self._get_snapshot_bounds(snapshot)

        # Missing valid_from: treat as global snapshot
        if snap_start is None:
            return True

        # Check overlap for half-open [snap_start, snap_end) with closed [range_start, range_end]
        if snap_start > range_end:
            return False
        if snap_end is None:
            return True
        return snap_end > range_start

    def _snapshot_matches_by_granularity(
        self,
        snapshot: Dict,
        range_start: datetime,
        range_end: datetime,
        granularity: str,
    ) -> bool:
        """
        Check if snapshot matches time range based on granularity.

        Args:
            snapshot: Snapshot dictionary
            range_start: Start of time range
            range_end: End of time range
            granularity: Time granularity (day, week, month, quarter, year)

        Returns:
            True if snapshot matches based on granularity
        """
        snap_start, snap_end = self._get_snapshot_bounds(snapshot)

        # Missing valid_from: treat as global snapshot
        if snap_start is None:
            return True

        granularity_order = {"day": 1, "week": 2, "month": 3, "quarter": 4, "year": 5}
        granularity_level = granularity_order.get(granularity, 0)
        month_level = granularity_order["month"]

        if granularity_level <= month_level:
            return self._match_by_month(snap_start, snap_end, range_start, range_end, granularity)
        elif granularity == "quarter":
            return self._match_by_quarter(snap_start, snap_end, range_start, range_end)
        elif granularity == "year":
            return self._match_by_year(snap_start, snap_end, range_start, range_end)
        else:
            return self._snapshot_overlaps_range(snapshot, range_start, range_end)

    def _match_by_month(
        self,
        snap_start: datetime,
        snap_end: Optional[datetime],
        range_start: datetime,
        range_end: datetime,
        granularity: str,
    ) -> bool:
        """Match snapshots by month for day/week/month granularities."""
        snap_start_month = (snap_start.year, snap_start.month)

        if snap_end:
            if snap_end.month == 1:
                snap_end_month = (snap_end.year - 1, 12)
            else:
                snap_end_month = (snap_end.year, snap_end.month - 1)
        else:
            snap_end_month = None

        range_start_month = (range_start.year, range_start.month)
        range_end_month = (range_end.year, range_end.month)

        matches = snap_start_month <= range_end_month and (
            snap_end_month is None or snap_end_month >= range_start_month
        )

        logger.debug(
            f"Month matching (granularity={granularity}): snapshot "
            f"({snap_start_month}~{snap_end_month}) vs "
            f"query ({range_start_month}~{range_end_month}), result={matches}"
        )
        return matches

    def _match_by_quarter(
        self,
        snap_start: datetime,
        snap_end: Optional[datetime],
        range_start: datetime,
        range_end: datetime,
    ) -> bool:
        """Match snapshots by quarter."""
        def get_quarter(dt: datetime) -> Tuple[int, int]:
            return (dt.year, (dt.month - 1) // 3 + 1)

        snap_start_quarter = get_quarter(snap_start)

        if snap_end:
            if snap_end.month == 1:
                snap_end_quarter = (snap_end.year - 1, 4)
            else:
                prev_month = snap_end.month - 1
                snap_end_quarter = (snap_end.year, (prev_month - 1) // 3 + 1)
        else:
            snap_end_quarter = None

        range_start_quarter = get_quarter(range_start)
        range_end_quarter = get_quarter(range_end)

        matches = snap_start_quarter <= range_end_quarter and (
            snap_end_quarter is None or snap_end_quarter >= range_start_quarter
        )

        logger.debug(
            f"Quarter matching: snapshot ({snap_start_quarter}~{snap_end_quarter}) vs "
            f"query ({range_start_quarter}~{range_end_quarter}), result={matches}"
        )
        return matches

    def _match_by_year(
        self,
        snap_start: datetime,
        snap_end: Optional[datetime],
        range_start: datetime,
        range_end: datetime,
    ) -> bool:
        """Match snapshots by year."""
        snap_start_year = snap_start.year
        snap_end_year = snap_end.year if snap_end else None
        range_start_year = range_start.year
        range_end_year = range_end.year

        matches = snap_start_year <= range_end_year and (
            snap_end_year is None or snap_end_year >= range_start_year
        )

        logger.debug(
            f"Year matching: snapshot ({snap_start_year}~{snap_end_year}) vs "
            f"query ({range_start_year}~{range_end_year}), result={matches}"
        )
        return matches

    def select_by_constraint(
        self,
        temporal_constraint: Optional[TemporalConstraint],
    ) -> List[Dict]:
        """
        Select snapshots based on temporal constraint.

        Args:
            temporal_constraint: Temporal constraint to match against

        Returns:
            List of selected snapshots, sorted by valid_from
        """
        if not self.snapshots:
            return []
        if temporal_constraint is None or temporal_constraint.is_empty():
            return []

        selected: List[Dict] = []
        seen_ids: Set[str] = set()

        logger.debug(
            f"Starting snapshot selection: {len(self.snapshots)} snapshots, "
            f"{len(temporal_constraint.ranges)} time ranges"
        )

        for snapshot in self.snapshots:
            snapshot_id = str(snapshot.get("snapshot_id", id(snapshot)))
            if snapshot_id in seen_ids:
                continue

            # Check if any range matches
            for r in temporal_constraint.ranges:
                if self._snapshot_matches_by_granularity(snapshot, r.start, r.end, r.granularity):
                    logger.debug(
                        f"Snapshot matched: {snapshot_id} "
                        f"(granularity={r.granularity}, range={r.start.date()}~{r.end.date()})"
                    )
                    selected.append(snapshot)
                    seen_ids.add(snapshot_id)
                    break
            else:
                # Log debug info for unmatched snapshots
                snap_start, snap_end = self._get_snapshot_bounds(snapshot)
                logger.debug(
                    f"Snapshot unmatched: {snapshot_id} "
                    f"({snap_start.date() if snap_start else None}~{snap_end.date() if snap_end else None})"
                )

        # Sort by valid_from (None treated as earliest)
        def sort_key(s: Dict):
            s_start, _ = self._get_snapshot_bounds(s)
            return s_start or datetime.min.replace(tzinfo=timezone.utc)

        selected.sort(key=sort_key)
        return selected

    def select_latest_before(self, cutoff_time: datetime) -> Optional[Dict]:
        """
        Select latest snapshot before cutoff time.

        Args:
            cutoff_time: Cutoff time for selection

        Returns:
            Latest snapshot before cutoff, or earliest if none qualify
        """
        if not self.snapshots:
            return None

        candidates: List[Tuple[datetime, Dict]] = []
        all_with_start: List[Tuple[datetime, Dict]] = []

        for s in self.snapshots:
            start, _ = self._get_snapshot_bounds(s)
            start_dt = start or datetime.min.replace(tzinfo=timezone.utc)
            all_with_start.append((start_dt, s))
            if start_dt <= cutoff_time:
                candidates.append((start_dt, s))

        if candidates:
            # Return latest among candidates
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        # Return earliest if cutoff is before all snapshots
        all_with_start.sort(key=lambda x: x[0])
        return all_with_start[0][1]

    def select_latest(self) -> Optional[Dict]:
        """
        Select globally latest snapshot.

        Returns:
            Latest snapshot by valid_from
        """
        if not self.snapshots:
            return None

        best = None
        best_start = None
        for s in self.snapshots:
            start, _ = self._get_snapshot_bounds(s)
            start_dt = start or datetime.min.replace(tzinfo=timezone.utc)
            if best is None or start_dt > best_start:
                best = s
                best_start = start_dt
        return best

    def select_by_time(
        self,
        query_time: Optional[datetime],
        snapshot_type: str = "both"
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Select snapshot by query time.

        Args:
            query_time: Time to match against, None for latest
            snapshot_type: Type to select ("persona", "topic", or "both")

        Returns:
            Tuple of (persona_snapshot, topic_snapshot)
        """
        if not self.snapshots:
            logger.warning("No snapshots available")
            return None, None

        if query_time is None:
            logger.info("No query time specified, selecting latest snapshot")
            latest_snapshot = self._select_latest_snapshot()
            return self._return_by_type(latest_snapshot, snapshot_type)

        # Find snapshots containing query_time
        matching_snapshots = self._find_matching_snapshots(query_time)

        if not matching_snapshots:
            logger.warning(f"No snapshot found for time {query_time.isoformat()}, using latest")
            latest_snapshot = self._select_latest_snapshot()
            return self._return_by_type(latest_snapshot, snapshot_type)

        # Select closest match
        matching_snapshots.sort(key=lambda x: abs((x[1] - query_time).total_seconds()))
        selected_snapshot = matching_snapshots[0][0]

        logger.info(
            f"Selected snapshot: valid_from={selected_snapshot.get('valid_from')}, "
            f"valid_to={selected_snapshot.get('valid_to')}"
        )

        return self._return_by_type(selected_snapshot, snapshot_type)

    def _select_latest_snapshot(self) -> Dict:
        """Select latest snapshot by valid_to."""
        _dt_min = datetime.min.replace(tzinfo=timezone.utc)
        _dt_max = datetime.max.replace(tzinfo=timezone.utc)

        def latest_key(s: Dict):
            valid_to = s.get("valid_to")
            if valid_to is None:
                return _dt_max
            dt = parse_iso_datetime(valid_to)
            return dt or _dt_min

        return max(self.snapshots, key=latest_key)

    def _find_matching_snapshots(self, query_time: datetime) -> List[Tuple[Dict, datetime]]:
        """Find snapshots that contain the query time."""
        matching_snapshots = []
        for snapshot in self.snapshots:
            valid_from = parse_iso_datetime(snapshot.get("valid_from"))
            valid_to = parse_iso_datetime(snapshot.get("valid_to"))

            if valid_from is None:
                continue

            # Check if query_time is in time window
            if valid_to is None:
                # valid_to is None means current version
                if query_time >= valid_from:
                    matching_snapshots.append((snapshot, valid_from))
            else:
                if valid_from <= query_time < valid_to:
                    matching_snapshots.append((snapshot, valid_from))

        return matching_snapshots

    def _return_by_type(
        self,
        snapshot: Dict,
        snapshot_type: str
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Return snapshot tuple based on type."""
        if snapshot_type == "persona":
            return snapshot, None
        elif snapshot_type == "topic":
            return None, snapshot
        else:
            return snapshot, snapshot