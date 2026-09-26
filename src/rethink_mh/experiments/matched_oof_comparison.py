"""Compare candidate native strict-OOF results with a content-frozen reference ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .frozen_artifact_manifest import verify_directory_manifest
from .text_baseline import (
    _metrics,
    _write_json,
    _write_jsonl,
    select_balanced_accuracy_threshold,
)


REFERENCE_EXPERIMENT = "qwen_thinker_av_transcript_label_oof_native"
CANDIDATE_EXPERIMENT = "qwen_thinker_av_transcript_label_oof_native"


@dataclass(frozen=True)
class Aggregate:
    seed: int
    labels: Mapping[str, int]
    probabilities: Mapping[str, float]
    summary: Mapping[str, Any]
    artifact_sha256: str
    manifest_sha256: str


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(payload)
    return rows


def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_summary_metrics(
    summary: Mapping[str, Any],
    labels: Sequence[int],
    probabilities: Sequence[float],
) -> None:
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("aggregate summary has no metrics object")
    threshold = float(metrics.get("threshold", math.nan))
    observed = _metrics(labels, probabilities, threshold)
    for key, value in observed.items():
        recorded = metrics.get(key)
        if isinstance(value, int):
            if recorded != value:
                raise ValueError(f"aggregate metric {key} does not reproduce")
        elif not math.isclose(
            float(recorded),
            float(value),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(f"aggregate metric {key} does not reproduce")


def _load_aggregate(
    directory: Path,
    manifest_path: Path,
    *,
    expected_experiment: str,
    expected_protocol: str,
) -> Aggregate:
    manifest = _read_json(manifest_path)
    verified = verify_directory_manifest(directory, manifest)
    summary = _read_json(directory / "summary.json")
    rows = _read_jsonl(directory / "predictions.jsonl")

    if summary.get("experiment") != expected_experiment:
        raise ValueError(
            f"unexpected aggregate experiment in {directory}: "
            f"{summary.get('experiment')!r}"
        )
    protocol = str(summary.get("evidence_protocol", "native"))
    if protocol != expected_protocol:
        raise ValueError(
            f"unexpected Evidence protocol in {directory}: {protocol!r}"
        )
    if summary.get("dataset") != "daic_woz":
        raise ValueError("matched OOF comparison is locked to daic_woz")
    if int(summary.get("split_seed", -1)) != 42:
        raise ValueError("aggregate did not use the frozen split seed 42")
    if int(summary.get("folds", -1)) != 5:
        raise ValueError("aggregate did not use the frozen five-fold design")
    if summary.get("threshold_source") != (
        "all_training_participants_out_of_fold_probabilities"
    ):
        raise ValueError("aggregate threshold was not selected from training OOF")
    boundaries = summary.get("fit_boundaries")
    if not isinstance(boundaries, Mapping) or any(
        (
            boundaries.get("each_prediction_model_saw_participant") is not False,
            boundaries.get("dev_read_by_fold_runs") is not False,
            boundaries.get("test_read_by_fold_runs") is not False,
            boundaries.get("threshold_uses_only_oof_training_predictions")
            is not True,
            boundaries.get("each_fold_reference_excludes_holdout") is not True,
            boundaries.get("evidence_compiler_label_access") is not False,
        )
    ):
        raise ValueError("aggregate violates the locked fit-only boundaries")
    initialization = summary.get("training_initialization")
    if not isinstance(initialization, Mapping) or any(
        (
            initialization.get("model_name") != "Qwen2.5-Omni-7B",
            initialization.get("adapter_checkpoint_loaded") is not False,
            initialization.get("frozen_reference_model_reused") is not False,
            int(initialization.get("epochs_per_fold", -1)) != 2,
            initialization.get("class_weighting") is not False,
            initialization.get("checkpoint_selection_metric") is not None,
        )
    ):
        raise ValueError("aggregate lacks from-base initialization proof")

    labels: dict[str, int] = {}
    probabilities: dict[str, float] = {}
    for row in rows:
        session_id = str(row.get("session_id", ""))
        if not session_id or session_id in labels:
            raise ValueError("aggregate has empty or duplicate participant IDs")
        label = int(row.get("label", -1))
        probability = float(row.get("probability", math.nan))
        if label not in {0, 1}:
            raise ValueError(f"non-binary label for participant {session_id}")
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"invalid probability for participant {session_id}")
        if row.get("split") != "train":
            raise ValueError("comparison input contains a non-training row")
        if not str(row.get("prediction_origin", "")).startswith("participant_"):
            raise ValueError("comparison input contains a non-OOF prediction")
        labels[session_id] = label
        probabilities[session_id] = probability
    if int(summary.get("participant_count", -1)) != len(labels):
        raise ValueError("aggregate participant count does not reproduce")

    ordered_ids = sorted(labels)
    _validate_summary_metrics(
        summary,
        [labels[session_id] for session_id in ordered_ids],
        [probabilities[session_id] for session_id in ordered_ids],
    )
    return Aggregate(
        seed=int(summary["seed"]),
        labels=labels,
        probabilities=probabilities,
        summary=summary,
        artifact_sha256=str(verified["artifact_sha256"]),
        manifest_sha256=_manifest_sha256(manifest_path),
    )


def _load_arm(
    directories: Sequence[Path],
    manifest_paths: Sequence[Path],
    *,
    expected_experiment: str,
    expected_protocol: str,
    expected_seeds: Sequence[int],
) -> dict[int, Aggregate]:
    if len(directories) != len(manifest_paths):
        raise ValueError("each aggregate directory requires one manifest")
    aggregates: dict[int, Aggregate] = {}
    for directory, manifest_path in zip(directories, manifest_paths):
        aggregate = _load_aggregate(
            directory,
            manifest_path,
            expected_experiment=expected_experiment,
            expected_protocol=expected_protocol,
        )
        if aggregate.seed in aggregates:
            raise ValueError(f"duplicate aggregate seed {aggregate.seed}")
        aggregates[aggregate.seed] = aggregate
    if set(aggregates) != set(expected_seeds):
        raise ValueError(
            f"aggregate seeds differ from the locked seeds: "
            f"observed={sorted(aggregates)}, expected={sorted(expected_seeds)}"
        )
    return aggregates


def _validate_pairing(
    reference: Mapping[int, Aggregate],
    candidate: Mapping[int, Aggregate],
) -> list[str]:
    canonical_labels: Mapping[str, int] | None = None
    for aggregate in (*reference.values(), *candidate.values()):
        if canonical_labels is None:
            canonical_labels = aggregate.labels
        elif aggregate.labels != canonical_labels:
            raise ValueError("aggregate participant IDs or labels are not paired")
    if canonical_labels is None:
        raise ValueError("no aggregate results were supplied")
    ids = sorted(canonical_labels)
    if not ids or set(canonical_labels.values()) != {0, 1}:
        raise ValueError("paired aggregates must contain both label classes")
    return ids


def _ensemble_probabilities(
    aggregates: Mapping[int, Aggregate],
    ids: Sequence[str],
) -> np.ndarray:
    return np.mean(
        np.asarray(
            [
                [aggregates[seed].probabilities[session_id] for session_id in ids]
                for seed in sorted(aggregates)
            ],
            dtype=float,
        ),
        axis=0,
    )


def _binary_log_loss(label: np.ndarray, probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return -(label * np.log(clipped) + (1 - label) * np.log(1 - clipped))


def _paired_stratified_bootstrap(
    labels: np.ndarray,
    reference_probability: np.ndarray,
    candidate_probability: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    if any(group.size == 0 for group in groups):
        raise ValueError("stratified bootstrap requires both label classes")
    brier_delta = (labels - reference_probability) ** 2 - (
        labels - candidate_probability
    ) ** 2
    log_loss_delta = _binary_log_loss(
        labels, reference_probability
    ) - _binary_log_loss(labels, candidate_probability)
    generator = np.random.default_rng(seed)
    estimates = {
        "brier_improvement": np.empty(samples, dtype=float),
        "log_loss_improvement": np.empty(samples, dtype=float),
    }
    for index in range(samples):
        sampled = np.concatenate(
            [
                generator.choice(group, size=group.size, replace=True)
                for group in groups
            ]
        )
        estimates["brier_improvement"][index] = float(
            np.mean(brier_delta[sampled])
        )
        estimates["log_loss_improvement"][index] = float(
            np.mean(log_loss_delta[sampled])
        )

    def interval(values: np.ndarray, estimate: float) -> dict[str, float]:
        return {
            "estimate": float(estimate),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        }

    return {
        "unit": "participant_stratified_by_label",
        "samples": samples,
        "seed": seed,
        "positive_means_candidate_improves": True,
        "brier_improvement": interval(
            estimates["brier_improvement"], float(np.mean(brier_delta))
        ),
        "log_loss_improvement": interval(
            estimates["log_loss_improvement"],
            float(np.mean(log_loss_delta)),
        ),
    }


def _transitions(
    labels: np.ndarray,
    reference_probability: np.ndarray,
    candidate_probability: np.ndarray,
    reference_threshold: float,
    candidate_threshold: float,
) -> dict[str, int | float]:
    frozen_correct = (reference_probability >= reference_threshold) == labels
    native_correct = (candidate_probability >= candidate_threshold) == labels
    correct_count = int(np.sum(frozen_correct))
    wrong_count = int(np.sum(~frozen_correct))
    harmed = int(np.sum(frozen_correct & ~native_correct))
    recovered = int(np.sum(~frozen_correct & native_correct))
    return {
        "reference_correct_count": correct_count,
        "reference_wrong_count": wrong_count,
        "cc_preserved_count": int(np.sum(frozen_correct & native_correct)),
        "cc_harmed_count": harmed,
        "wc_recovered_count": recovered,
        "ww_remaining_count": int(np.sum(~frozen_correct & ~native_correct)),
        "net_corrected_count": recovered - harmed,
        "cc_harm_rate": harmed / correct_count if correct_count else 0.0,
        "wc_recovery_rate": recovered / wrong_count if wrong_count else 0.0,
    }


def _artifact_records(
    aggregates: Mapping[int, Aggregate],
) -> list[dict[str, Any]]:
    return [
        {
            "seed": seed,
            "artifact_sha256": aggregates[seed].artifact_sha256,
            "manifest_sha256": aggregates[seed].manifest_sha256,
        }
        for seed in sorted(aggregates)
    ]


def compare_aggregates(
    reference: Mapping[int, Aggregate],
    candidate: Mapping[int, Aggregate],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ids = _validate_pairing(reference, candidate)
    first = next(iter(reference.values()))
    labels = np.asarray([first.labels[session_id] for session_id in ids], dtype=int)
    frozen_probability = _ensemble_probabilities(reference, ids)
    native_probability = _ensemble_probabilities(candidate, ids)
    frozen_threshold = select_balanced_accuracy_threshold(labels, frozen_probability)
    native_threshold = select_balanced_accuracy_threshold(labels, native_probability)
    frozen_metrics = _metrics(labels, frozen_probability, frozen_threshold)
    native_metrics = _metrics(labels, native_probability, native_threshold)
    bootstrap = _paired_stratified_bootstrap(
        labels,
        frozen_probability,
        native_probability,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    transitions = _transitions(
        labels,
        frozen_probability,
        native_probability,
        frozen_threshold,
        native_threshold,
    )
    point_checks = {
        "brier_strictly_lower": (
            float(native_metrics["brier"]) < float(frozen_metrics["brier"])
        ),
        "log_loss_strictly_lower": (
            float(native_metrics["log_loss"])
            < float(frozen_metrics["log_loss"])
        ),
        "roc_auc_not_lower": (
            float(native_metrics["roc_auc"])
            >= float(frozen_metrics["roc_auc"])
        ),
        "average_precision_not_lower": (
            float(native_metrics["average_precision"])
            >= float(frozen_metrics["average_precision"])
        ),
        "net_corrected_nonnegative": (
            int(transitions["net_corrected_count"]) >= 0
        ),
    }
    developmental_support = all(point_checks.values())
    confirmatory_support = (
        developmental_support
        and float(bootstrap["brier_improvement"]["ci95_low"]) > 0.0
    )
    rows = []
    for index, session_id in enumerate(ids):
        label = int(labels[index])
        frozen_p = float(frozen_probability[index])
        native_p = float(native_probability[index])
        rows.append(
            {
                "session_id": session_id,
                "split": "train",
                "label": label,
                "reference_probability": frozen_p,
                "candidate_probability": native_p,
                "reference_predicted_label": int(
                    frozen_p >= frozen_threshold
                ),
                "candidate_predicted_label": int(
                    native_p >= native_threshold
                ),
                "brier_improvement": float(
                    (label - frozen_p) ** 2 - (label - native_p) ** 2
                ),
                "log_loss_improvement": float(
                    _binary_log_loss(
                        np.asarray([label]), np.asarray([frozen_p])
                    )[0]
                    - _binary_log_loss(
                        np.asarray([label]), np.asarray([native_p])
                    )[0]
                ),
            }
        )
    summary = {
        "schema_version": "1.0.0",
        "experiment": "matched_oof_comparison",
        "status": "developmental_fit_only",
        "dataset": "daic_woz",
        "participant_count": len(ids),
        "class_counts": {
            "0": int(np.sum(labels == 0)),
            "1": int(np.sum(labels == 1)),
        },
        "seeds": sorted(reference),
        "folds": 5,
        "split_seed": 42,
        "probability_rule": "participant_mean_across_seed_oof_aggregates",
        "threshold_rule": (
            "separate_balanced_accuracy_threshold_from_each_arm_training_oof"
        ),
        "reference": {
            "experiment": REFERENCE_EXPERIMENT,
            "metrics": frozen_metrics,
            "artifacts": _artifact_records(reference),
        },
        "candidate": {
            "experiment": CANDIDATE_EXPERIMENT,
            "evidence_protocol": "native",
            "metrics": native_metrics,
            "artifacts": _artifact_records(candidate),
        },
        "delta_candidate_minus_reference": {
            key: float(native_metrics[key]) - float(frozen_metrics[key])
            for key in (
                "brier",
                "log_loss",
                "roc_auc",
                "average_precision",
                "balanced_accuracy",
                "f1",
            )
        },
        "paired_bootstrap": bootstrap,
        "transitions": transitions,
        "preregistered_gate": {
            "point_checks": point_checks,
            "developmental_point_support": developmental_support,
            "confirmatory_brier_ci_above_zero": (
                float(bootstrap["brier_improvement"]["ci95_low"]) > 0.0
            ),
            "confirmatory_support": confirmatory_support,
            "gate_changed_after_results": False,
        },
        "fit_boundaries": {
            "train_oof_only": True,
            "dev_accessed": False,
            "test_accessed": False,
            "reference_retrained": False,
            "reference_recompiled": False,
            "reference_verified_by_content_manifest": True,
            "candidate_trained_from_base_checkpoint": True,
        },
        "interpretation_boundary": (
            "This comparison measures fit-only participant-level strict-OOF "
            "classification. It does not validate a rethink policy, clinical "
            "deployment, or official-test generalization."
        ),
    }
    return rows, summary


def run_comparison(
    *,
    reference_dirs: Sequence[Path],
    reference_manifests: Sequence[Path],
    candidate_dirs: Sequence[Path],
    candidate_manifests: Sequence[Path],
    expected_seeds: Sequence[int],
    output_dir: Path,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite comparison: {output_dir}")
    if len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("expected seeds must be unique")
    reference = _load_arm(
        reference_dirs,
        reference_manifests,
        expected_experiment=REFERENCE_EXPERIMENT,
        expected_protocol="native",
        expected_seeds=expected_seeds,
    )
    candidate = _load_arm(
        candidate_dirs,
        candidate_manifests,
        expected_experiment=CANDIDATE_EXPERIMENT,
        expected_protocol="native",
        expected_seeds=expected_seeds,
    )
    rows, summary = compare_aggregates(
        reference,
        candidate,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    output_dir.mkdir(parents=True)
    _write_jsonl(output_dir / "paired_predictions.jsonl", rows)
    _write_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True, nargs="+", type=Path)
    parser.add_argument(
        "--reference-manifest", required=True, nargs="+", type=Path
    )
    parser.add_argument("--candidate-dir", required=True, nargs="+", type=Path)
    parser.add_argument(
        "--candidate-manifest", required=True, nargs="+", type=Path
    )
    parser.add_argument(
        "--expected-seed", nargs="+", type=int, default=[42, 43, 44]
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--bootstrap-seed", type=int, default=3_725)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_comparison(
        reference_dirs=args.reference_dir,
        reference_manifests=args.reference_manifest,
        candidate_dirs=args.candidate_dir,
        candidate_manifests=args.candidate_manifest,
        expected_seeds=args.expected_seed,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
