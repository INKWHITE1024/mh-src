"""Fail-closed PHQ-8 binary targets derived from released scale totals."""

from __future__ import annotations

import csv
import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


PHQ8_SCORE_THRESHOLD = 10.0
_DATASET_COLUMNS = {
    "daic_woz": {
        "score": "PHQ8_Score",
        "released_binary": "PHQ8_Binary",
    },
    "e_daic": {
        "score": "PHQ_Score",
        "released_binary": "PHQ_Binary",
    },
}
_ID_COLUMNS = ("Participant_ID", "participant_ID", "session_id", "id")


def _resolve_column(
    fieldnames: Sequence[str], requested: str | None, candidates: Sequence[str]
) -> str:
    by_folded = {name.strip().casefold(): name for name in fieldnames}
    if requested is not None:
        resolved = by_folded.get(requested.strip().casefold())
        if resolved is None:
            raise ValueError(f"Column {requested!r} is absent")
        return resolved
    for candidate in candidates:
        resolved = by_folded.get(candidate.casefold())
        if resolved is not None:
            return resolved
    raise ValueError(f"None of the required columns are present: {tuple(candidates)!r}")


def _normalize_identifier(raw: str) -> str:
    value = raw.strip()
    if not value:
        raise ValueError("participant identifier cannot be empty")
    try:
        numeric = float(value)
    except ValueError:
        return value
    if math.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    return value


def validate_phq8_score_policy(
    dataset: str, label_column: str, label_threshold: float | None
) -> None:
    """Reject an experiment unless it explicitly selects score-derived labels."""

    try:
        expected = _DATASET_COLUMNS[dataset]["score"]
    except KeyError as error:
        raise ValueError(f"Unsupported PHQ-8 dataset {dataset!r}") from error
    if label_column.strip().casefold() != expected.casefold():
        raise ValueError(
            f"{dataset} must use {expected} as the target source; released binary "
            "labels are audit fields only"
        )
    if (
        label_threshold is None
        or not math.isfinite(float(label_threshold))
        or float(label_threshold) != PHQ8_SCORE_THRESHOLD
    ):
        raise ValueError(
            f"{dataset} must derive the binary target with {expected} >= 10"
        )


def validate_safe_label_source(
    label_column: str, label_threshold: float | None
) -> str | None:
    """Fail closed for known PHQ-8 columns while allowing unrelated tasks."""

    folded = label_column.strip().casefold()
    for dataset, columns in _DATASET_COLUMNS.items():
        if folded == columns["released_binary"].casefold():
            raise ValueError(
                f"{columns['released_binary']} contains known label errors; "
                "released binary columns are audit fields only. Use the PHQ-8 "
                "score column with threshold 10"
            )
        if folded == columns["score"].casefold():
            validate_phq8_score_policy(dataset, label_column, label_threshold)
            return dataset
    return None


def audit_phq8_label_file(
    path: Path,
    *,
    dataset: str,
    id_column: str | None = None,
) -> dict[str, Any]:
    """Audit released binary values against the score-derived outcome label."""

    try:
        columns = _DATASET_COLUMNS[dataset]
    except KeyError as error:
        raise ValueError(f"Unsupported PHQ-8 dataset {dataset!r}") from error
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Label file has no header: {path}")
        resolved_id = _resolve_column(reader.fieldnames, id_column, _ID_COLUMNS)
        resolved_score = _resolve_column(reader.fieldnames, columns["score"], ())
        resolved_released = _resolve_column(
            reader.fieldnames, columns["released_binary"], ()
        )
        seen: set[str] = set()
        derived_counts: Counter[int] = Counter()
        released_counts: Counter[int] = Counter()
        mismatches: list[dict[str, Any]] = []
        row_count = 0
        for line_number, row in enumerate(reader, 2):
            row_count += 1
            session_id = _normalize_identifier(str(row.get(resolved_id, "")))
            if session_id in seen:
                raise ValueError(
                    f"Duplicate participant {session_id!r} at {path}:{line_number}"
                )
            seen.add(session_id)
            try:
                score = float(str(row.get(resolved_score, "")).strip())
                released_value = float(
                    str(row.get(resolved_released, "")).strip()
                )
            except ValueError as error:
                raise ValueError(
                    f"Invalid PHQ-8 label data at {path}:{line_number}"
                ) from error
            if (
                not math.isfinite(score)
                or not score.is_integer()
                or not 0.0 <= score <= 24.0
            ):
                raise ValueError(
                    f"PHQ-8 score must be an integer in [0, 24] at "
                    f"{path}:{line_number}"
                )
            if (
                not math.isfinite(released_value)
                or not released_value.is_integer()
                or int(released_value) not in {0, 1}
            ):
                raise ValueError(
                    f"Released PHQ-8 binary must be 0 or 1 at {path}:{line_number}"
                )
            released = int(released_value)
            derived = int(score >= PHQ8_SCORE_THRESHOLD)
            released_counts[released] += 1
            derived_counts[derived] += 1
            if released != derived:
                mismatches.append(
                    {
                        "session_id": session_id,
                        "score": int(score),
                        "released_binary": released,
                        "derived_binary": derived,
                    }
                )
    if row_count == 0:
        raise ValueError(f"Label file is empty: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "row_count": row_count,
        "score_column": resolved_score,
        "released_binary_column": resolved_released,
        "threshold_rule": f"{resolved_score} >= 10",
        "released_binary_used_as_target": False,
        "derived_class_counts": {
            "0": derived_counts[0],
            "1": derived_counts[1],
        },
        "released_class_counts": {
            "0": released_counts[0],
            "1": released_counts[1],
        },
        "mismatch_count": len(mismatches),
        "mismatch_ids": [item["session_id"] for item in mismatches],
        "mismatches": mismatches,
    }


def build_phq8_label_audit(
    paths: Sequence[Path],
    *,
    dataset: str,
    id_column: str | None = None,
) -> dict[str, Any]:
    """Combine per-file audits into summary provenance for one experiment."""

    if not paths:
        raise ValueError("At least one accessed PHQ-8 label file is required")
    files = [
        audit_phq8_label_file(path, dataset=dataset, id_column=id_column)
        for path in paths
    ]
    mismatch_ids = [
        session_id
        for audit in files
        for session_id in audit["mismatch_ids"]
    ]
    if len(mismatch_ids) != len(set(mismatch_ids)):
        raise ValueError("PHQ-8 participant IDs overlap across accessed label files")
    return {
        "policy": "phq8_score_ge_10",
        "operator": ">=",
        "threshold": int(PHQ8_SCORE_THRESHOLD),
        "target_source": _DATASET_COLUMNS[dataset]["score"],
        "released_binary_column": _DATASET_COLUMNS[dataset]["released_binary"],
        "released_binary_used_as_target": False,
        "mismatch_count": sum(audit["mismatch_count"] for audit in files),
        "mismatch_ids": mismatch_ids,
        "files": files,
    }
