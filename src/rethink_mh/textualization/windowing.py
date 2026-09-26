"""Deterministic time-windowing and slicing helpers for feature arrays."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimeWindow:
    index: int
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


def make_windows(
    duration_sec: float,
    window_sec: float,
    stride_sec: float,
    minimum_tail_sec: float,
) -> list[TimeWindow]:
    """Create deterministic half-open windows and retain a sufficiently long tail."""
    if not np.isfinite(duration_sec) or duration_sec <= 0:
        return []
    windows: list[TimeWindow] = []
    start = 0.0
    index = 0
    epsilon = 1e-8
    while start < duration_sec - epsilon:
        end = min(start + window_sec, duration_sec)
        if end - start + epsilon < minimum_tail_sec:
            break
        windows.append(TimeWindow(index=index, start_sec=start, end_sec=end))
        index += 1
        start += stride_sec
    return windows


def slice_bounds(times: np.ndarray, start_sec: float, end_sec: float) -> tuple[int, int]:
    """Return indices for a half-open [start, end) interval in sorted timestamps."""
    if times.ndim != 1:
        raise ValueError("times must be one-dimensional")
    left = int(np.searchsorted(times, start_sec, side="left"))
    right = int(np.searchsorted(times, end_sec, side="left"))
    return left, right


def estimate_sampling_rate(times: np.ndarray, fallback: float) -> float:
    finite = np.asarray(times, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 3:
        return float(fallback)
    deltas = np.diff(finite)
    deltas = deltas[(deltas > 1e-6) & np.isfinite(deltas)]
    if deltas.size == 0:
        return float(fallback)
    rate = 1.0 / float(np.median(deltas))
    return rate if np.isfinite(rate) and rate > 0 else float(fallback)


def expected_count(duration_sec: float, sampling_rate_hz: float) -> int:
    return max(1, int(round(duration_sec * sampling_rate_hz)))

