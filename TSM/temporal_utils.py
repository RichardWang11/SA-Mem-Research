"""
Temporal parsing utilities for question-driven retrieval with optional time filtering.

This module centralizes the logic for extracting temporal constraints from natural
language queries. The implementation draws inspiration from `DEEP_parser.py`, but
packages the behaviour into reusable helpers that can be consumed by the retriever.
"""

from __future__ import annotations

import re
from calendar import monthrange
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional, Sequence

import dateparser
# search_dates已被移除，因为误判率太高（如"do", "did", "me"等被误判为日期）
import logging

logger = logging.getLogger(__name__)

try:
    import spacy
    from spacy.language import Language as SpacyLanguage
except ImportError:  # pragma: no cover - spaCy optional in some environments
    spacy = None  # type: ignore
    SpacyLanguage = None  # type: ignore


# -- Data structures -----------------------------------------------------------------


@dataclass
class TemporalRange:
    """Represents an inclusive temporal window inferred from the question."""

    start: datetime
    end: datetime
    phrase: str
    granularity: str = "day"

    def contains(self, dt: datetime) -> bool:
        """Return True if the datetime falls within the window."""
        return self.start <= dt <= self.end


@dataclass
class TemporalConstraint:
    """Structured temporal hint extracted from a query."""

    ranges: List[TemporalRange] = field(default_factory=list)
    phrases: List[str] = field(default_factory=list)
    reference_date: Optional[datetime] = None

    def matches(self, dt: datetime) -> bool:
        """Return True when the provided datetime satisfies any inferred range."""
        for temp_range in self.ranges:
            if temp_range.contains(dt):
                return True
        return False

    def is_empty(self) -> bool:
        return len(self.ranges) == 0


# -- Helper functions ----------------------------------------------------------------


def _ensure_datetime_timezone(dt: datetime) -> datetime:
    """Force timezone-aware UTC datetimes for consistent comparisons."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _month_bounds(dt: datetime) -> tuple[datetime, datetime]:
    # 保存原始时区的年月信息，避免时区转换导致的月份错误
    # 如果dt已经是UTC，直接使用；否则使用原始时区的年月
    if dt.tzinfo is not None and dt.tzinfo != timezone.utc:
        # 使用原始时区的年月（不转换时区）
        year = dt.year
        month = dt.month
    else:
        # UTC时间或无时区信息，直接使用
        year = dt.year
        month = dt.month

    # 在UTC时区创建月份边界
    first_day = datetime(year, month, 1, 0, 0, 0, 0, tzinfo=timezone.utc)
    days_in_month = monthrange(year, month)[1]
    last_day = datetime(year, month, days_in_month, 23, 59, 59, 999999, tzinfo=timezone.utc)
    return first_day, last_day


def _year_bounds(dt: datetime) -> tuple[datetime, datetime]:
    start = dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    end = dt.replace(month=12, day=31, hour=23, minute=59, second=59, microsecond=999999)
    return _ensure_datetime_timezone(start), _ensure_datetime_timezone(end)


def _week_bounds(dt: datetime) -> tuple[datetime, datetime]:
    # 改为前后5天，而不是完整的一周（周一到周日）
    start = dt - timedelta(days=5)
    start = start.replace(hour=0, minute=0, second=0, microsecond=0)
    end = dt + timedelta(days=5, hours=23, minutes=59, seconds=59, microseconds=999999)
    return _ensure_datetime_timezone(start), _ensure_datetime_timezone(end)


def _day_bounds(dt: datetime) -> tuple[datetime, datetime]:
    start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    end = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
    return _ensure_datetime_timezone(start), _ensure_datetime_timezone(end)


def _detect_granularity(phrase: str) -> str:
    phrase_lower = phrase.lower()

    if re.search(r"\bquarter\b|\bq[1-4]\b", phrase_lower):
        return "quarter"
    # 支持 year/years，且避免“带月份名”误判为年粒度
    if re.search(r"\byears?\b|\b\d{4}\b", phrase_lower) and not re.search(r"\bjan|\bfeb|\bmar|\bapr|\bmay|\bjune|\bjul|\baug|\bsep|\boct|\bnov|\bdec", phrase_lower):
        return "year"
    # 支持 month/months 以及月份名
    if re.search(r"\bmonths?\b|\bjan|\bfeb|\bmar|\bapr|\bmay|\bjune|\bjul|\baug|\bsep|\boct|\bnov|\bdec", phrase_lower):
        return "month"
    # 支持 week/weeks（用包含判断也可以，但这里显式对齐 month/year 的处理）
    if re.search(r"\bweeks?\b", phrase_lower):
        return "week"
    if "day" in phrase_lower or re.search(r"\b\d{1,2}(st|nd|rd|th)\b", phrase_lower):
        return "day"
    return "day"


def _granularity_bounds(granularity: str, dt: datetime) -> tuple[datetime, datetime]:
    if granularity == "month":
        return _month_bounds(dt)
    if granularity == "year":
        return _year_bounds(dt)
    if granularity == "week":
        return _week_bounds(dt)
    if granularity == "quarter":
        quarter = (dt.month - 1) // 3
        start_month = quarter * 3 + 1
        start = dt.replace(month=start_month, day=1, hour=0, minute=0, second=0, microsecond=0)
        end_month = start_month + 2
        end_base = dt.replace(month=end_month, day=1, hour=23, minute=59, second=59, microsecond=999999)
        end = end_base.replace(day=monthrange(end_base.year, end_base.month)[1])
        return _ensure_datetime_timezone(start), _ensure_datetime_timezone(end)
    return _day_bounds(dt)


def parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse ISO-like strings into timezone-aware UTC datetimes."""
    if not value:
        return None

    # 处理非字符串类型（如Neo4j DateTime对象）
    if not isinstance(value, str):
        if hasattr(value, 'iso_format'):
            # Neo4j DateTime对象
            value = value.iso_format()
        elif hasattr(value, 'isoformat'):
            # Python datetime对象
            value = value.isoformat()
        else:
            # 其他类型，尝试转换为字符串
            value = str(value)

    cleaned = value.strip()
    if cleaned == "":
        return None

    # Handle trailing 'Z'
    cleaned = cleaned.replace("Z", "+00:00")

    try:
        dt = datetime.fromisoformat(cleaned)
        return _ensure_datetime_timezone(dt)
    except ValueError:
        pass

    try:
        dt = dateparser.parse(cleaned)
        if dt is None:
            return None
        return _ensure_datetime_timezone(dt)
    except (TypeError, ValueError):
        return None


def _collect_spacy_entities(question: str, nlp: Optional[SpacyLanguage]) -> Sequence[str]:
    if nlp is None:
        if spacy is None:  # pragma: no cover
            return []
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(question)
    phrases: list[str] = []

    for ent in doc.ents:
        if ent.label_ in {"DATE", "TIME", "ORDINAL"}:
            phrases.append(ent.text)

    return phrases



def extract_temporal_constraints(
    question: str,
    *,
    nlp: Optional[SpacyLanguage] = None,
    reference_date: Optional[datetime] = None,
) -> Optional[TemporalConstraint]:
    """
    Extract temporal constraints from a natural language question.

    Parameters
    ----------
    question:
        The user question.
    nlp:
        Optional spaCy language model to reuse for entity extraction.
    reference_date:
        Optional reference datetime (e.g., question creation time) used for resolving
        relative expressions like "next Monday".

    Returns
    -------
    TemporalConstraint or None
        Parsed temporal ranges. Returns None if no time hints are detected.
    """

    question = question.strip()
    if not question:
        return None

    phrases = _collect_spacy_entities(question, nlp)
    # 确定实际使用的参考时间
    actual_reference_date = reference_date or datetime.now(timezone.utc)

    ranges: list[TemporalRange] = []

    # 优先处理spaCy提取的实体
    for phrase in phrases:
        parsed_dt = dateparser.parse(
            phrase,
            settings={
                "RELATIVE_BASE": actual_reference_date,
                "RETURN_AS_TIMEZONE_AWARE": True,
            },
        )
        if parsed_dt:
            logger.debug(f"DEV版本时间提取：dateparser成功解析 '{phrase}' -> {parsed_dt}")
        if parsed_dt:
            granularity = _detect_granularity(phrase)
            # 在转换为UTC之前计算边界，以保留原始时区的年月信息
            start, end = _granularity_bounds(granularity, parsed_dt)
            # 确保结果都是UTC时区
            start = _ensure_datetime_timezone(start)
            end = _ensure_datetime_timezone(end)
            logger.info(
                f"DEV版本时间提取：从spaCy实体解析 - 短语: '{phrase}', "
                f"日期: {parsed_dt.isoformat()}, 粒度: {granularity}, "
                f"范围: {start.isoformat()} ~ {end.isoformat()}"
            )
            ranges.append(TemporalRange(start=start, end=end, phrase=phrase, granularity=granularity))

    if not ranges:
        logger.info("DEV版本时间提取：未找到明确的时间表达式，返回None（将使用问题日期作为上限过滤）")
        return None

    constraint = TemporalConstraint(
        ranges=ranges,
        phrases=phrases if phrases else [r.phrase for r in ranges],
        reference_date=_ensure_datetime_timezone(reference_date) if reference_date else None,
    )
    return constraint


__all__ = [
    "TemporalConstraint",
    "TemporalRange",
    "extract_temporal_constraints",
    "parse_iso_datetime",
]
