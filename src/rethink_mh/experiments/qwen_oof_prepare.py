"""Create label-free split files for strict fold-local Evidence compilation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

from rethink_mh.textualization.references import session_ids_digest

from .phq8_labels import audit_phq8_label_file, validate_phq8_score_policy
from .qwen_label_baseline import participant_stratified_fold
from .text_baseline import SessionRecord


_ID_COLUMNS = ("Participant_ID", "participant_ID", "session_id", "id")


def _resolve_column(
    fieldnames: Sequence[str], requested: str | None, candidates: Sequence[str]
) -> str:
    by_folded = {name.strip().casefold(): name for name in fieldnames}
    if requested is not None:
        match = by_folded.get(requested.strip().casefold())
        if match is None:
            raise ValueError(f"Column {requested!r} is absent")
        return match
    for candidate in candidates:
        match = by_folded.get(candidate.casefold())
        if match is not None:
            return match
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


def read_labeled_participants(
    path: Path,
    *,
    label_column: str,
    label_threshold: float | None,
    id_column: str | None,
) -> list[SessionRecord]:
    """Read only participant IDs and the binary label used for stratification."""

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Label file has no header: {path}")
        resolved_id = _resolve_column(reader.fieldnames, id_column, _ID_COLUMNS)
        resolved_label = _resolve_column(reader.fieldnames, label_column, ())
        records: list[SessionRecord] = []
        seen: set[str] = set()
        for line_number, row in enumerate(reader, 2):
            session_id = _normalize_identifier(str(row.get(resolved_id, "")))
            if session_id in seen:
                raise ValueError(
                    f"Duplicate participant {session_id!r} at {path}:{line_number}"
                )
            seen.add(session_id)
            raw_label = str(row.get(resolved_label, "")).strip()
            try:
                numeric_label = float(raw_label)
                label = int(numeric_label)
            except ValueError as error:
                raise ValueError(
                    f"Invalid binary label at {path}:{line_number}"
                ) from error
            if label_threshold is None:
                if label not in {0, 1} or numeric_label != label:
                    raise ValueError(f"Invalid binary label at {path}:{line_number}")
            else:
                label = int(numeric_label >= label_threshold)
            records.append(SessionRecord(session_id, "train", "", label))
    if not records:
        raise ValueError(f"Label file is empty: {path}")
    return records


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_identifiers(records: Sequence[SessionRecord]) -> str:
    return hashlib.sha256(
        "\n".join(record.session_id for record in records).encode("utf-8")
    ).hexdigest()


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def prepare_fold_assignment(
    *,
    dataset: str,
    train_labels: Path,
    label_column: str,
    label_threshold: float | None,
    id_column: str | None,
    folds: int,
    fold_index: int,
    seed: int,
    output_csv: Path,
    output_manifest: Path,
) -> dict[str, Any]:
    """Write an ID-only fit/holdout file matching the Qwen fold constructor."""

    if dataset not in {"daic_woz", "e_daic"}:
        raise ValueError(f"Unsupported dataset {dataset!r}")
    validate_phq8_score_policy(dataset, label_column, label_threshold)
    if output_csv.exists() or output_manifest.exists():
        raise FileExistsError("refusing to overwrite an existing fold assignment")
    records = read_labeled_participants(
        train_labels,
        label_column=label_column,
        label_threshold=label_threshold,
        id_column=id_column,
    )
    fit, holdout = participant_stratified_fold(
        records,
        folds=folds,
        fold_index=fold_index,
        seed=seed,
    )
    fit_ids = {record.session_id for record in fit}
    rows = ["session_id,split"]
    for record in records:
        role = "fit" if record.session_id in fit_ids else "holdout"
        rows.append(f"{record.session_id},{role}")
    assignment_text = "\n".join(rows) + "\n"
    _write_atomic(output_csv, assignment_text)
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "dataset": dataset,
        "seed": seed,
        "folds": folds,
        "fold_index": fold_index,
        "participant_count": len(records),
        "fit_count": len(fit),
        "holdout_count": len(holdout),
        "fit_class_counts": {
            str(label): sum(record.label == label for record in fit)
            for label in (0, 1)
        },
        "holdout_class_counts": {
            str(label): sum(record.label == label for record in holdout)
            for label in (0, 1)
        },
        "fit_ids_sha256": _sha256_identifiers(fit),
        "holdout_ids_sha256": _sha256_identifiers(holdout),
        "reference_session_ids_sha256": session_ids_digest(fit_ids),
        "assignment_csv_sha256": hashlib.sha256(
            assignment_text.encode("utf-8")
        ).hexdigest(),
        "source_label_file_sha256": _sha256_file(train_labels),
        "partitioning_label_access": True,
        "partitioning_label_column": label_column,
        "partitioning_label_threshold": label_threshold,
        "label_policy": audit_phq8_label_file(
            train_labels, dataset=dataset, id_column=id_column
        ),
        "evidence_compiler_input_columns": ["session_id", "split"],
        "evidence_compiler_label_access": False,
    }
    _write_atomic(
        output_manifest,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare an ID-only strict OOF fit/holdout assignment."
    )
    parser.add_argument("--dataset", required=True, choices=("daic_woz", "e_daic"))
    parser.add_argument("--train-labels", required=True, type=Path)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--label-threshold", required=True, type=float)
    parser.add_argument("--id-column")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-index", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = prepare_fold_assignment(
        dataset=args.dataset,
        train_labels=args.train_labels,
        label_column=args.label_column,
        label_threshold=args.label_threshold,
        id_column=args.id_column,
        folds=args.folds,
        fold_index=args.fold_index,
        seed=args.seed,
        output_csv=args.output_csv,
        output_manifest=args.output_manifest,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
