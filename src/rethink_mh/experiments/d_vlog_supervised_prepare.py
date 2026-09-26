"""Prepare audited D-Vlog supervised train/valid labels and label-free test IDs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_SPLITS = ("train", "valid", "test")
OFFICIAL_COUNTS = {"train": 647, "valid": 102, "test": 212}
SOURCE_VALUE_MAPPING = {"depression": 1, "normal": 0}


def natural_session_key(value: str) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value)
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ids_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8")


def prepare_d_vlog_supervised_splits(
    source: Path,
    output_dir: Path,
    *,
    expected_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Create train/valid targets while keeping the official test physically unlabeled."""

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError("D-Vlog label file is empty") from error
        normalized = {
            name.strip().casefold(): index for index, name in enumerate(header)
        }
        required = {"index", "label", "fold"}
        if not required <= set(normalized):
            raise ValueError("D-Vlog labels require index, label, and fold columns")
        id_index = normalized["index"]
        label_index = normalized["label"]
        fold_index = normalized["fold"]
        maximum_index = max(id_index, label_index, fold_index)
        ids_by_split: dict[str, list[str]] = {
            split: [] for split in EXPECTED_SPLITS
        }
        targets_by_split: dict[str, dict[str, int]] = {
            "train": {},
            "valid": {},
        }
        seen: set[str] = set()
        for line_number, row in enumerate(reader, 2):
            if maximum_index >= len(row):
                raise ValueError(f"short row at {source}:{line_number}")
            session_id = row[id_index].strip()
            split = row[fold_index].strip().casefold()
            if not session_id or session_id in seen:
                raise ValueError(
                    f"invalid or duplicate ID at {source}:{line_number}"
                )
            if split not in ids_by_split:
                raise ValueError(f"unknown split {split!r} at {source}:{line_number}")
            seen.add(session_id)
            ids_by_split[split].append(session_id)
            if split in targets_by_split:
                source_label = row[label_index].strip().casefold()
                if source_label not in SOURCE_VALUE_MAPPING:
                    raise ValueError(
                        f"unknown D-Vlog label {source_label!r} at "
                        f"{source}:{line_number}"
                    )
                targets_by_split[split][session_id] = SOURCE_VALUE_MAPPING[
                    source_label
                ]

    if not all(ids_by_split.values()):
        raise ValueError("every D-Vlog split must be non-empty")
    observed_counts = {
        split: len(ids_by_split[split]) for split in EXPECTED_SPLITS
    }
    if expected_counts is not None and observed_counts != dict(expected_counts):
        raise ValueError(
            f"unexpected D-Vlog split counts: {observed_counts}; "
            f"expected {dict(expected_counts)}"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        files: dict[str, dict[str, Any]] = {}
        for split in ("train", "valid"):
            ids = sorted(ids_by_split[split], key=natural_session_key)
            filename = f"{split}_labels.csv"
            path = staging / filename
            _write_csv(
                path,
                ("session_id", "target", "split"),
                [
                    (session_id, targets_by_split[split][session_id], split)
                    for session_id in ids
                ],
            )
            counts = Counter(targets_by_split[split].values())
            files[split] = {
                "path": str((output_dir / filename).resolve()),
                "count": len(ids),
                "sha256": _sha256(path),
                "ids_sha256": _ids_sha256(ids),
                "columns": ["session_id", "target", "split"],
                "class_counts": {"0": counts[0], "1": counts[1]},
            }
        test_ids = sorted(ids_by_split["test"], key=natural_session_key)
        test_path = staging / "test_ids.csv"
        _write_csv(
            test_path,
            ("session_id", "split"),
            [(session_id, "test") for session_id in test_ids],
        )
        files["test"] = {
            "path": str((output_dir / "test_ids.csv").resolve()),
            "count": len(test_ids),
            "sha256": _sha256(test_path),
            "ids_sha256": _ids_sha256(test_ids),
            "columns": ["session_id", "split"],
            "label_free": True,
        }
        manifest: dict[str, Any] = {
            "schema_version": "1.0.0",
            "dataset": "d_vlog",
            "source_label_file": str(source.resolve()),
            "source_label_file_sha256": _sha256(source),
            "source_columns_present": header,
            "source_columns_consumed_for_membership": [
                header[id_index],
                header[fold_index],
            ],
            "source_columns_consumed_for_train_valid_targets": [
                header[label_index]
            ],
            "test_label_values_consumed": False,
            "label_policy": {
                "policy": "d_vlog_released_current_depression",
                "task_id": "d_vlog_current_depression",
                "source_label_column": header[label_index],
                "source_value_mapping": dict(SOURCE_VALUE_MAPPING),
                "target_column": "target",
                "test_targets_exported": False,
            },
            "files": files,
            "split_counts": observed_counts,
            "total_count": len(seen),
        }
        _write_json(staging / "manifest.json", manifest)
        if output_dir.exists():
            raise FileExistsError(f"refusing to overwrite {output_dir}")
        os.replace(staging, output_dir)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--allow-nonstandard-counts",
        action="store_true",
        help="disable the official 647/102/212 count assertion (tests only)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = prepare_d_vlog_supervised_splits(
        args.source,
        args.output_dir,
        expected_counts=None if args.allow_nonstandard_counts else OFFICIAL_COUNTS,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
