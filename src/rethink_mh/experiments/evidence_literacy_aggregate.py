"""Aggregate five outer-fold literacy mechanics checks into reviewer routes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evidence_literacy import PROTOCOL_VERSION
from .evidence_literacy_evaluate import EVALUATION_PROTOCOL_VERSION


ROUTE_PROTOCOL_VERSION = "evidence-literacy-reviewer-route"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_ids(values: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be one JSON object: {path}")
    return value


def _assignment(path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["session_id", "split"]:
                raise ValueError(
                    "assignment columns must be exactly session_id,split"
                )
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise ValueError(f"cannot read assignment {path}: {error}") from error
    fit: list[str] = []
    holdout: list[str] = []
    seen: set[str] = set()
    for line_number, row in enumerate(rows, 2):
        session_id = str(row.get("session_id", "")).strip()
        split = str(row.get("split", "")).strip()
        if not session_id or split not in {"fit", "holdout"}:
            raise ValueError(f"invalid assignment row {path}:{line_number}")
        if session_id in seen:
            raise ValueError(f"duplicate assignment session {session_id}: {path}")
        seen.add(session_id)
        (fit if split == "fit" else holdout).append(session_id)
    if not fit or not holdout:
        raise ValueError(f"assignment must contain fit and holdout sessions: {path}")
    return tuple(sorted(fit)), tuple(sorted(holdout))


@dataclass(frozen=True, slots=True)
class FoldLiteracyResult:
    fold_index: int
    assignment_path: Path
    summary_path: Path

    def __post_init__(self) -> None:
        if (
            isinstance(self.fold_index, bool)
            or not isinstance(self.fold_index, int)
            or self.fold_index < 0
        ):
            raise ValueError("fold_index must be a non-negative integer")


def _validated_route(
    result: FoldLiteracyResult,
    *,
    dataset: str,
    seed: int,
) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...], str]:
    fit, holdout = _assignment(result.assignment_path)
    summary = _read_json(result.summary_path, "literacy held-out summary")
    boundaries = summary.get("fit_boundaries")
    metrics = summary.get("metrics")
    gate = summary.get("gate")
    if any(
        (
            summary.get("evaluation_protocol_version")
            != EVALUATION_PROTOCOL_VERSION,
            summary.get("training_protocol_version") != PROTOCOL_VERSION,
            summary.get("dataset") != dataset,
            summary.get("partition") != "holdout",
            int(summary.get("session_count", -1)) != len(holdout),
            not isinstance(boundaries, Mapping),
            not isinstance(metrics, Mapping),
            not isinstance(gate, Mapping),
        )
    ):
        raise ValueError(
            f"invalid fold-{result.fold_index} literacy summary: "
            f"{result.summary_path}"
        )
    assert isinstance(boundaries, Mapping)
    assert isinstance(gate, Mapping)
    if any(
        (
            boundaries.get("sample_targets_accessed") is not False,
            boundaries.get("annotator_uses_outcomes") is not False,
            boundaries.get("label_fields_in_model_text") is not False,
            boundaries.get("heldout_from_literacy_training") is not True,
            boundaries.get("contract_repairs_allowed") is not False,
            gate.get("eligible") is not True,
        )
    ):
        raise ValueError(
            f"unsafe fold-{result.fold_index} literacy boundary: "
            f"{result.summary_path}"
        )

    mechanics_passed = gate.get("passed") is True
    model_path = Path(str(summary.get("model_path", ""))).expanduser()
    if not str(model_path):
        raise ValueError("literacy summary omitted model_path")
    if mechanics_passed:
        adapter_path = Path(str(summary.get("adapter_path", ""))).expanduser()
        required = (
            adapter_path / "adapter_config.json",
            adapter_path / "adapter_model.safetensors",
        )
        if any(not path.is_file() for path in required):
            raise FileNotFoundError(
                f"fold-{result.fold_index} passed but adapter is incomplete: "
                f"{adapter_path}"
            )
        reviewer_kind = "literacy_adapter"
        routed_adapter: str | None = str(adapter_path.resolve())
        fallback_reason: str | None = None
    else:
        reviewer_kind = "base_qwen_fallback"
        routed_adapter = None
        fallback_reason = "heldout_query_or_selection_mechanics_gate_failed"

    route = {
        "fold_index": result.fold_index,
        "seed": seed,
        "reviewer_kind": reviewer_kind,
        "mechanics_passed": mechanics_passed,
        "model_path": str(model_path.resolve()),
        "adapter_path": routed_adapter,
        "fallback_reason": fallback_reason,
        "fit_count": len(fit),
        "holdout_count": len(holdout),
        "fit_ids_sha256": _sha256_ids(fit),
        "holdout_ids_sha256": _sha256_ids(holdout),
        "assignment_path": str(result.assignment_path.resolve()),
        "assignment_sha256": _sha256(result.assignment_path),
        "heldout_summary_path": str(result.summary_path.resolve()),
        "heldout_summary_sha256": _sha256(result.summary_path),
        "query_metrics": metrics.get("evidence_query"),
        "selection_metrics": metrics.get("atomic_selection"),
        "scientific_performance_gate": False,
    }
    return route, fit, holdout, str(model_path.resolve())


def aggregate_reviewer_routes(
    *,
    dataset: str,
    seed: int,
    fold_results: Sequence[FoldLiteracyResult],
    output_path: Path,
) -> dict[str, Any]:
    """Validate complete OOF coverage and publish fold-local reviewer routes."""

    if not fold_results:
        raise ValueError("at least one fold result is required")
    expected_indices = tuple(range(len(fold_results)))
    observed_indices = tuple(sorted(result.fold_index for result in fold_results))
    if observed_indices != expected_indices:
        raise ValueError(
            f"fold indices must be contiguous {expected_indices}, got {observed_indices}"
        )
    if len(observed_indices) != len(set(observed_indices)):
        raise ValueError("fold results contain duplicate fold indices")

    routes: list[dict[str, Any]] = []
    cohort: set[str] | None = None
    covered: set[str] = set()
    model_paths: set[str] = set()
    for result in sorted(fold_results, key=lambda item: item.fold_index):
        route, fit, holdout, model_path = _validated_route(
            result,
            dataset=dataset,
            seed=seed,
        )
        assignment_cohort = set(fit) | set(holdout)
        if cohort is None:
            cohort = assignment_cohort
        elif assignment_cohort != cohort:
            raise ValueError("fold assignments do not describe one shared cohort")
        overlap = covered & set(holdout)
        if overlap:
            raise ValueError(
                f"outer-fold holdout participants overlap: {sorted(overlap)}"
            )
        covered.update(holdout)
        routes.append(route)
        model_paths.add(model_path)
    assert cohort is not None
    if covered != cohort:
        missing = sorted(cohort - covered)
        extra = sorted(covered - cohort)
        raise ValueError(
            f"outer-fold holdouts do not cover the cohort; missing={missing}, "
            f"extra={extra}"
        )
    if len(model_paths) != 1:
        raise ValueError("fold literacy results use different base model paths")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite reviewer routes: {output_path}")

    payload = {
        "schema_version": "1.0.0",
        "route_protocol_version": ROUTE_PROTOCOL_VERSION,
        "training_protocol_version": PROTOCOL_VERSION,
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "dataset": dataset,
        "seed": seed,
        "fold_count": len(routes),
        "session_count": len(cohort),
        "session_ids_sha256": _sha256_ids(tuple(sorted(cohort))),
        "mechanics_passed_fold_count": sum(
            route["mechanics_passed"] for route in routes
        ),
        "fallback_fold_count": sum(
            route["reviewer_kind"] == "base_qwen_fallback" for route in routes
        ),
        "routes": routes,
        "fit_boundaries": {
            "sample_targets_accessed": False,
            "test_participants_accessed": False,
            "failed_mechanics_blocks_full_loop": False,
            "failed_mechanics_uses_base_reviewer_fallback": True,
            "scientific_performance_selected_here": False,
        },
        "interpretation_boundary": (
            "This artifact selects an executable reviewer implementation per "
            "outer fold. It does not select a scientific result and does not "
            "decide whether the complete loop is beneficial."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return payload


def _fold_result(value: str) -> tuple[int, Path, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "fold result must be FOLD=ASSIGNMENT=SUMMARY"
        )
    try:
        fold_index = int(parts[0])
    except ValueError as error:
        raise argparse.ArgumentTypeError("fold index must be numeric") from error
    return fold_index, Path(parts[1]), Path(parts[2])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate OOF Evidence Literacy mechanics into reviewer routes."
    )
    parser.add_argument("--dataset", default="daic_woz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fold-result",
        action="append",
        required=True,
        type=_fold_result,
        metavar="FOLD=ASSIGNMENT=SUMMARY",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = aggregate_reviewer_routes(
        dataset=args.dataset,
        seed=args.seed,
        fold_results=tuple(
            FoldLiteracyResult(
                fold_index=fold_index,
                assignment_path=assignment,
                summary_path=summary,
            )
            for fold_index, assignment, summary in args.fold_result
        ),
        output_path=args.output,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
