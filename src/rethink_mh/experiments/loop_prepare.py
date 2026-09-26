"""Prepare physically separated inference and outcome views for full-loop OOF."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rethink_mh.textualization.protocols.native import (
    PROTOCOL_VERSION as EVIDENCE_PROTOCOL_VERSION,
)

from .evidence_literacy_aggregate import ROUTE_PROTOCOL_VERSION


PREPARATION_PROTOCOL_VERSION = "loop-prepare"
_FORBIDDEN_INFERENCE_KEYS = frozenset(
    {
        "label",
        "target",
        "outcome",
        "ground_truth",
        "gold",
        "correct",
        "transition",
        "utility",
        "v2_probability",
        "v2_predicted_label",
    }
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be one JSON object: {path}")
    return value


def _read_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read {description} {path}: {error}") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise ValueError(f"blank line in {description}: {path}:{line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid JSON in {description}: {path}:{line_number}"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(
                f"{description} row must be an object: {path}:{line_number}"
            )
        output.append(value)
    if not output:
        raise ValueError(f"{description} is empty: {path}")
    return output


def _probability(value: Any, path: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{path} must be a finite probability")
    return float(value)


def _binary(value: Any, path: str) -> int:
    if isinstance(value, bool) or value not in {0, 1}:
        raise ValueError(f"{path} must be binary")
    return int(value)


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
        raise ValueError(f"assignment lacks fit or holdout sessions: {path}")
    return tuple(sorted(fit)), tuple(sorted(holdout))


def _forbidden_keys(value: Any, path: str = "root") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).casefold()
            if any(token in normalized for token in _FORBIDDEN_INFERENCE_KEYS):
                found.append(f"{path}.{key}")
            found.extend(_forbidden_keys(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_forbidden_keys(item, f"{path}[{index}]"))
    return found


@dataclass(frozen=True, slots=True)
class LoopPreparationRequest:
    dataset: str
    paired_predictions_path: Path
    comparison_summary_path: Path
    reviewer_routes_path: Path
    source_native_run_root: Path
    output_dir: Path
    seed: int = 42
    expected_fold_count: int = 5
    expected_session_count: int = 107

    def __post_init__(self) -> None:
        if not self.dataset:
            raise ValueError("dataset must be non-empty")
        for name in ("seed", "expected_fold_count", "expected_session_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


def _validated_comparison(
    path: Path,
    *,
    dataset: str,
    expected_session_count: int,
) -> tuple[dict[str, Any], float, float]:
    summary = _read_json(path, "frozen matched-OOF comparison")
    boundaries = summary.get("fit_boundaries")
    candidate = summary.get("candidate")
    reference = summary.get("reference")
    if any(
        (
            summary.get("experiment")
            != "matched_oof_comparison",
            summary.get("dataset") != dataset,
            summary.get("participant_count") != expected_session_count,
            summary.get("probability_rule")
            != "participant_mean_across_seed_oof_aggregates",
            not isinstance(boundaries, Mapping),
            not isinstance(candidate, Mapping),
            not isinstance(reference, Mapping),
        )
    ):
        raise ValueError("comparison summary is not the frozen matched-OOF artifact")
    assert isinstance(boundaries, Mapping)
    if any(
        (
            boundaries.get("train_oof_only") is not True,
            boundaries.get("dev_accessed") is not False,
            boundaries.get("test_accessed") is not False,
            boundaries.get("reference_retrained") is not False,
            boundaries.get("reference_recompiled") is not False,
        )
    ):
        raise ValueError("comparison summary has unsafe fit boundaries")
    metrics = candidate.get("metrics")
    reference_metrics = reference.get("metrics")
    if not isinstance(metrics, Mapping) or not isinstance(
        reference_metrics,
        Mapping,
    ):
        raise ValueError("comparison summary omitted arm metrics")
    threshold = _probability(metrics.get("threshold"), "candidate.metrics.threshold")
    reference_threshold = _probability(
        reference_metrics.get("threshold"),
        "reference.metrics.threshold",
    )
    return summary, threshold, reference_threshold


def _fold_map(
    request: LoopPreparationRequest,
    routes: Mapping[str, Any],
) -> tuple[dict[str, int], dict[int, dict[str, Any]], list[dict[str, Any]]]:
    raw_routes = routes.get("routes")
    if (
        routes.get("route_protocol_version") != ROUTE_PROTOCOL_VERSION
        or routes.get("dataset") != request.dataset
        or routes.get("seed") != request.seed
        or routes.get("fold_count") != request.expected_fold_count
        or routes.get("session_count") != request.expected_session_count
        or not isinstance(raw_routes, list)
        or routes.get("fit_boundaries", {}).get(
            "failed_mechanics_blocks_full_loop"
        )
        is not False
    ):
        raise ValueError("reviewer route is incomplete or unsafe")
    route_by_fold: dict[int, dict[str, Any]] = {}
    for raw in raw_routes:
        if not isinstance(raw, dict):
            raise ValueError("reviewer route rows must be objects")
        fold = int(raw.get("fold_index", -1))
        if fold in route_by_fold:
            raise ValueError(f"duplicate reviewer route fold {fold}")
        route_by_fold[fold] = raw
    if set(route_by_fold) != set(range(request.expected_fold_count)):
        raise ValueError("reviewer route omitted an outer fold")

    membership: dict[str, int] = {}
    assignment_audits: list[dict[str, Any]] = []
    cohort: set[str] | None = None
    for fold in range(request.expected_fold_count):
        assignment_path = (
            request.source_native_run_root
            / "artifacts"
            / "native_from_scratch"
            / request.dataset
            / f"seed{request.seed}"
            / f"fold{fold}"
            / "assignment.csv"
        )
        fit, holdout = _assignment(assignment_path)
        current_cohort = set(fit) | set(holdout)
        if cohort is None:
            cohort = current_cohort
        elif current_cohort != cohort:
            raise ValueError("outer assignments describe different cohorts")
        overlap = set(holdout) & set(membership)
        if overlap:
            raise ValueError(f"outer holdout membership overlaps: {sorted(overlap)}")
        membership.update({session_id: fold for session_id in holdout})
        route = route_by_fold[fold]
        if (
            route.get("assignment_sha256") != _sha256(assignment_path)
            or route.get("holdout_count") != len(holdout)
        ):
            raise ValueError(f"reviewer route/assignment mismatch for fold {fold}")
        assignment_audits.append(
            {
                "fold_index": fold,
                "assignment_path": str(assignment_path.resolve()),
                "assignment_sha256": _sha256(assignment_path),
                "fit_count": len(fit),
                "holdout_count": len(holdout),
            }
        )
    assert cohort is not None
    if set(membership) != cohort or len(cohort) != request.expected_session_count:
        raise ValueError("outer holdouts do not cover the expected cohort")
    return membership, route_by_fold, assignment_audits


def _evidence_identity(
    request: LoopPreparationRequest,
    session_id: str,
    fold: int,
) -> dict[str, Any]:
    fold_root = (
        request.source_native_run_root
        / "artifacts"
        / "native_from_scratch"
        / request.dataset
        / f"seed{request.seed}"
        / f"fold{fold}"
    )
    ready = _read_json(fold_root / "READY.json", "fold readiness")
    session_dir = (
        fold_root
        / "evidence"
        / EVIDENCE_PROTOCOL_VERSION
        / request.dataset
        / session_id
    )
    manifest_path = session_dir / "manifest.json"
    manifest = _read_json(manifest_path, "Evidence manifest")
    training_summary_path = (
        request.source_native_run_root
        / "outputs"
        / "qwen_native_from_scratch"
        / request.dataset
        / f"seed{request.seed}"
        / f"fold{fold}"
        / "summary.json"
    )
    training_summary = _read_json(
        training_summary_path,
        "fold training summary",
    )
    transcript = training_summary.get("transcript")
    protocol = manifest.get("protocol_metadata")
    if any(
        (
            ready.get("fold_index") != fold,
            ready.get("evidence_protocol") != EVIDENCE_PROTOCOL_VERSION,
            ready.get("label_boundaries", {}).get("dev_or_test_access") is not False,
            manifest.get("protocol_version") != EVIDENCE_PROTOCOL_VERSION,
            manifest.get("dataset_key") != request.dataset,
            str(manifest.get("session_id")) != session_id,
            manifest.get("label_fields_read") != [],
            manifest.get("transcript_content_exposed") is not False,
            not isinstance(protocol, Mapping),
            not isinstance(transcript, Mapping),
        )
    ):
        raise ValueError(
            f"unsafe fold-local Evidence identity for session {session_id}"
        )
    assert isinstance(protocol, Mapping)
    assert isinstance(transcript, Mapping)
    fingerprint = str(protocol.get("bundle_sha256", "")).strip()
    if len(fingerprint) != 64:
        raise ValueError(f"Evidence bundle omitted fingerprint for {session_id}")
    transcript_root = Path(str(transcript.get("root", ""))).expanduser()
    transcript_dir = transcript_root / session_id
    transcript_manifest_path = transcript_dir / "transcript_manifest.compact.json"
    transcript_manifest = _read_json(
        transcript_manifest_path,
        "Transcript Evidence manifest",
    )
    if any(
        (
            transcript.get("enabled") is not True,
            transcript.get("protocol_version") != "compact",
            transcript.get("content_exposed_to_model") is not True,
            transcript_manifest.get("protocol_version") != "compact",
            transcript_manifest.get("label_fields_read") != [],
            transcript_manifest.get("raw_session_identifier_in_model_input")
            is not False,
        )
    ):
        raise ValueError(
            f"unsafe Transcript Evidence identity for session {session_id}"
        )
    return {
        "session_dir": str(session_dir.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "bundle_sha256": fingerprint,
        "reference_id": str(manifest.get("reference_id")),
        "fold_training_summary_sha256": _sha256(training_summary_path),
        "transcript_dir": str(transcript_dir.resolve()),
        "transcript_manifest_sha256": _sha256(transcript_manifest_path),
    }


def prepare_loop_views(request: LoopPreparationRequest) -> dict[str, Any]:
    """Publish immutable label-free inference and outcome-only files."""

    if request.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite loop preparation: {request.output_dir}"
        )
    _, threshold, reference_threshold = _validated_comparison(
        request.comparison_summary_path,
        dataset=request.dataset,
        expected_session_count=request.expected_session_count,
    )
    routes = _read_json(request.reviewer_routes_path, "reviewer routes")
    membership, route_by_fold, assignment_audits = _fold_map(request, routes)
    paired_rows = _read_jsonl(
        request.paired_predictions_path,
        "frozen paired predictions",
    )
    if len(paired_rows) != request.expected_session_count:
        raise ValueError("paired predictions have an unexpected participant count")

    inference_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in sorted(paired_rows, key=lambda item: str(item.get("session_id", ""))):
        session_id = str(row.get("session_id", "")).strip()
        if (
            not session_id
            or session_id in seen
            or row.get("split") != "train"
            or session_id not in membership
        ):
            raise ValueError(f"invalid paired prediction session {session_id!r}")
        seen.add(session_id)
        label = _binary(row.get("label"), f"{session_id}.label")
        initial_probability = _probability(
            row.get("candidate_probability"),
            f"{session_id}.candidate_probability",
        )
        reference_probability = _probability(
            row.get("reference_probability"),
            f"{session_id}.reference_probability",
        )
        if row.get("reference_predicted_label") != int(
            reference_probability >= reference_threshold
        ):
            raise ValueError(
                f"reference threshold/prediction mismatch for {session_id}"
            )
        if row.get("candidate_predicted_label") != int(
            initial_probability >= threshold
        ):
            raise ValueError(
                f"candidate threshold/prediction mismatch for {session_id}"
            )
        fold = membership[session_id]
        route = route_by_fold[fold]
        evidence = _evidence_identity(request, session_id, fold)
        inference = {
            "schema_version": "1.0.0",
            "preparation_protocol_version": PREPARATION_PROTOCOL_VERSION,
            "dataset": request.dataset,
            "session_id": session_id,
            "outer_fold": fold,
            "initial_probability": initial_probability,
            "initial_threshold": threshold,
            "initial_probability_origin": (
                "frozen_three_seed_native_strict_oof_ensemble"
            ),
            "evidence": evidence,
            "reviewer": {
                "reviewer_kind": route["reviewer_kind"],
                "mechanics_passed": route["mechanics_passed"],
                "model_path": route["model_path"],
                "adapter_path": route["adapter_path"],
                "heldout_summary_sha256": route["heldout_summary_sha256"],
            },
            "fit_boundaries": {
                "sample_supervision_visible": False,
                "comparison_baseline_visible": False,
                "offline_score_visible": False,
                "outer_fold_holdout": True,
            },
        }
        forbidden = _forbidden_keys(inference)
        if forbidden:
            raise AssertionError(
                f"label-free inference row contains forbidden keys: {forbidden}"
            )
        inference_rows.append(inference)
        outcome_rows.append(
            {
                "schema_version": "1.0.0",
                "preparation_protocol_version": PREPARATION_PROTOCOL_VERSION,
                "dataset": request.dataset,
                "session_id": session_id,
                "outer_fold": fold,
                "label": label,
                "initial_probability": initial_probability,
                "initial_threshold": threshold,
                "initial_predicted_label": int(
                    initial_probability >= threshold
                ),
                "reference_probability": reference_probability,
                "reference_threshold": reference_threshold,
                "reference_predicted_label": _binary(
                    row.get("reference_predicted_label"),
                    f"{session_id}.reference_predicted_label",
                ),
                "used_in_model_inference": False,
            }
        )
    if set(seen) != set(membership):
        raise ValueError("paired predictions and outer-fold cohort differ")

    inference_text = "".join(
        _canonical_json(row) + "\n" for row in inference_rows
    )
    outcomes_text = "".join(
        _canonical_json(row) + "\n" for row in outcome_rows
    )
    inference_manifest = {
        "schema_version": "1.0.0",
        "preparation_protocol_version": PREPARATION_PROTOCOL_VERSION,
        "dataset": request.dataset,
        "seed": request.seed,
        "fold_count": request.expected_fold_count,
        "session_count": request.expected_session_count,
        "files": {
            "inference_plan.jsonl": {
                "sha256": _sha256_text(inference_text),
                "row_count": len(inference_rows),
                "contains_current_sample_targets": False,
            }
        },
        "sources": {
            "comparison_summary_sha256": _sha256(
                request.comparison_summary_path
            ),
            "reviewer_routes_sha256": _sha256(request.reviewer_routes_path),
            "assignments": assignment_audits,
        },
        "fit_boundaries": {
            "current_sample_targets_present_in_directory": False,
            "comparison_baselines_present_in_directory": False,
            "gpu_inference_accepts_outcome_path": False,
            "dev_accessed": False,
            "test_accessed": False,
        },
    }
    outcome_manifest = {
        "schema_version": "1.0.0",
        "preparation_protocol_version": PREPARATION_PROTOCOL_VERSION,
        "dataset": request.dataset,
        "seed": request.seed,
        "session_count": request.expected_session_count,
        "files": {
            "outcomes.jsonl": {
                "sha256": _sha256_text(outcomes_text),
                "row_count": len(outcome_rows),
                "used_in_model_inference": False,
            }
        },
        "sources": {
            "paired_predictions_sha256": _sha256(
                request.paired_predictions_path
            ),
            "comparison_summary_sha256": _sha256(
                request.comparison_summary_path
            ),
        },
        "fit_boundaries": {
            "opened_only_after_raw_trajectory_freeze": True,
            "dev_accessed": False,
            "test_accessed": False,
            "reference_retrained": False,
            "reference_recompiled": False,
        },
    }
    inference_manifest_text = (
        json.dumps(
            inference_manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    outcome_manifest_text = (
        json.dumps(
            outcome_manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    manifest = {
        "schema_version": "1.0.0",
        "preparation_protocol_version": PREPARATION_PROTOCOL_VERSION,
        "dataset": request.dataset,
        "seed": request.seed,
        "fold_count": request.expected_fold_count,
        "session_count": request.expected_session_count,
        "files": {
            "inference/inference_plan.jsonl": {
                "sha256": _sha256_text(inference_text),
                "row_count": len(inference_rows),
                "contains_current_sample_targets": False,
            },
            "inference/manifest.json": {
                "sha256": _sha256_text(inference_manifest_text),
            },
            "outcomes/outcomes.jsonl": {
                "sha256": _sha256_text(outcomes_text),
                "row_count": len(outcome_rows),
                "visible_to_inference_process": False,
            },
            "outcomes/manifest.json": {
                "sha256": _sha256_text(outcome_manifest_text),
            },
        },
        "sources": {
            "paired_predictions_sha256": _sha256(
                request.paired_predictions_path
            ),
            "comparison_summary_sha256": _sha256(
                request.comparison_summary_path
            ),
            "reviewer_routes_sha256": _sha256(request.reviewer_routes_path),
            "assignments": assignment_audits,
        },
        "fit_boundaries": {
            "trusted_sanitizer_reads_outcomes": True,
            "inference_plan_contains_outcomes": False,
            "inference_and_outcomes_in_distinct_directories": True,
            "inference_manifest_contains_outcome_identity": False,
            "gpu_inference_accepts_outcome_path": False,
            "dev_accessed": False,
            "test_accessed": False,
            "reference_retrained": False,
            "reference_recompiled": False,
        },
        "interpretation_boundary": (
            "The inference plan is label-free and drives OOF trajectory "
            "generation. outcomes.jsonl is physically separate and may be "
            "opened only after raw trajectories are frozen."
        ),
    }
    inference_dir = request.output_dir / "inference"
    outcome_dir = request.output_dir / "outcomes"
    inference_dir.mkdir(parents=True)
    outcome_dir.mkdir()
    (inference_dir / "inference_plan.jsonl").write_text(
        inference_text,
        encoding="utf-8",
    )
    (inference_dir / "manifest.json").write_text(
        inference_manifest_text,
        encoding="utf-8",
    )
    (outcome_dir / "outcomes.jsonl").write_text(
        outcomes_text,
        encoding="utf-8",
    )
    (outcome_dir / "manifest.json").write_text(
        outcome_manifest_text,
        encoding="utf-8",
    )
    (request.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare label-free full-loop OOF inference plans."
    )
    parser.add_argument("--dataset", default="daic_woz")
    parser.add_argument("--paired-predictions", required=True, type=Path)
    parser.add_argument("--comparison-summary", required=True, type=Path)
    parser.add_argument("--reviewer-routes", required=True, type=Path)
    parser.add_argument("--source-run-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-fold-count", type=int, default=5)
    parser.add_argument("--expected-session-count", type=int, default=107)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = prepare_loop_views(
        LoopPreparationRequest(
            dataset=args.dataset,
            paired_predictions_path=args.paired_predictions,
            comparison_summary_path=args.comparison_summary,
            reviewer_routes_path=args.reviewer_routes,
            source_native_run_root=args.source_run_root,
            output_dir=args.output_dir,
            seed=args.seed,
            expected_fold_count=args.expected_fold_count,
            expected_session_count=args.expected_session_count,
        )
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
