"""Dataset-wide integrity audit for compiled native Evidence stores."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from rethink_mh.rethinking.retrieval import EvidenceStore

from .text_baseline import _write_json


_SEGMENT_PATTERN = re.compile(r"^SEGMENT (S[0-9]+)\b", re.MULTILINE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nonempty_line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def audit_session(session_dir: str | Path) -> dict[str, Any]:
    """Verify hashes, counts, ID mappings, and label/privacy invariants for one session."""

    started = time.time()
    directory = Path(session_dir)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing session manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("session_id")) != directory.name:
        raise ValueError(f"Session ID does not match directory name: {directory}")
    if manifest.get("protocol_version") != "native":
        raise ValueError(f"Unexpected protocol version in {manifest_path}")
    if manifest.get("label_fields_read") != []:
        raise ValueError(f"Compiler accessed label fields in {manifest_path}")
    if manifest.get("transcript_content_exposed") is not False:
        raise ValueError(f"Transcript exposure invariant failed in {manifest_path}")

    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"Manifest has no file hashes: {manifest_path}")
    verified_hashes: dict[str, str] = {}
    for name, metadata in sorted(files.items()):
        path = directory / name
        expected = metadata.get("sha256") if isinstance(metadata, dict) else None
        if not path.is_file() or not isinstance(expected, str):
            raise ValueError(f"Invalid manifest file record {name!r} in {manifest_path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {path}")
        verified_hashes[name] = actual

    store = EvidenceStore.from_session_dir(directory)
    segment_path = directory / "segment_evidence.native.txt"
    session_path = directory / "session_input.native.txt"
    segment_ids = tuple(_SEGMENT_PATTERN.findall(segment_path.read_text(encoding="utf-8")))
    session_segment_ids = tuple(
        _SEGMENT_PATTERN.findall(session_path.read_text(encoding="utf-8"))
    )
    if segment_ids != store.available_segment_ids:
        raise ValueError(f"Segment text/index mismatch in {directory}")
    if session_segment_ids != segment_ids:
        raise ValueError(f"Session/segment text mismatch in {directory}")
    if len(segment_ids) != int(manifest["segment_count"]):
        raise ValueError(f"Segment count mismatch in {directory}")

    unit_count = _nonempty_line_count(directory / "evidence_units.jsonl")
    atomic_count = _nonempty_line_count(directory / "evidence_index.native.jsonl")
    if unit_count != int(manifest["unit_count"]):
        raise ValueError(f"Canonical unit count mismatch in {directory}")
    if atomic_count != int(manifest["atomic_unit_count"]):
        raise ValueError(f"Atomic index count mismatch in {directory}")

    return {
        "session_id": directory.name,
        "dataset_key": manifest.get("dataset_key"),
        "reference_id": manifest.get("reference_id"),
        "unit_count": unit_count,
        "atomic_unit_count": atomic_count,
        "segment_count": len(segment_ids),
        "verified_file_count": len(verified_hashes),
        "elapsed_seconds": time.time() - started,
        "passed": True,
    }


def _audit_one(directory: str) -> dict[str, Any]:
    try:
        return audit_session(directory)
    except Exception as exc:  # noqa: BLE001 - audit must report every corrupt session
        return {
            "session_id": Path(directory).name,
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def run_integrity_audit(
    evidence_root: Path, *, workers: int, output: Path
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    directories = sorted(
        (path for path in evidence_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    if not directories:
        raise ValueError(f"No session directories found under {evidence_root}")

    started = time.time()
    payloads = [str(path) for path in directories]
    if workers == 1:
        reports = [_audit_one(path) for path in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            reports = list(executor.map(_audit_one, payloads))
    failed = [report for report in reports if not report["passed"]]
    passed = [report for report in reports if report["passed"]]
    summary = {
        "audit": "evidence_v2_dataset_integrity",
        "evidence_root": str(evidence_root.resolve()),
        "all_passed": not failed,
        "session_count": len(reports),
        "passed_count": len(passed),
        "failed_count": len(failed),
        "unit_count": sum(int(report["unit_count"]) for report in passed),
        "atomic_unit_count": sum(
            int(report["atomic_unit_count"]) for report in passed
        ),
        "segment_count": sum(int(report["segment_count"]) for report in passed),
        "elapsed_seconds": time.time() - started,
        "reports": reports,
    }
    _write_json(output, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify every compiled session, manifest hash, and retrieval index."
    )
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_integrity_audit(
        args.evidence_root, workers=args.workers, output=args.output
    )
    printable = {key: value for key, value in summary.items() if key != "reports"}
    print(json.dumps(printable, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["all_passed"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
