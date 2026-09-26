"""Time-interval primitives for merging, masking, and overlap computation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True, order=True)
class Interval:
    start_sec: float
    end_sec: float
    reason: str = ""

    def __post_init__(self) -> None:
        if not np.isfinite(self.start_sec) or not np.isfinite(self.end_sec):
            raise ValueError("Interval endpoints must be finite")
        if self.end_sec <= self.start_sec:
            raise ValueError("Interval end must be greater than start")


def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    ordered = sorted(intervals, key=lambda item: (item.start_sec, item.end_sec))
    if not ordered:
        return []
    merged: list[Interval] = [ordered[0]]
    for item in ordered[1:]:
        previous = merged[-1]
        if item.start_sec <= previous.end_sec:
            reasons = sorted(filter(None, {previous.reason, item.reason}))
            merged[-1] = Interval(
                previous.start_sec,
                max(previous.end_sec, item.end_sec),
                "+".join(reasons),
            )
        else:
            merged.append(item)
    return merged


def timestamps_in_intervals(times: np.ndarray, intervals: Iterable[Interval]) -> np.ndarray:
    values = np.asarray(times, dtype=float)
    result = np.zeros(values.shape, dtype=bool)
    for interval in merge_intervals(intervals):
        result |= (values >= interval.start_sec) & (values < interval.end_sec)
    return result


def overlap_duration(start_sec: float, end_sec: float, intervals: Iterable[Interval]) -> float:
    total = 0.0
    for interval in merge_intervals(intervals):
        total += max(0.0, min(end_sec, interval.end_sec) - max(start_sec, interval.start_sec))
    return min(max(total, 0.0), max(end_sec - start_sec, 0.0))


def overlap_reasons(start_sec: float, end_sec: float, intervals: Iterable[Interval]) -> list[str]:
    reasons: set[str] = set()
    for interval in intervals:
        if min(end_sec, interval.end_sec) > max(start_sec, interval.start_sec) and interval.reason:
            reasons.add(interval.reason)
    return sorted(reasons)

