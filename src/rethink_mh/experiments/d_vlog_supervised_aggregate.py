"""Validate three supervised D-Vlog runs and freeze their test ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_EXPERIMENT = "qwen_thinker_d_vlog_supervised"
EXPECTED_RUN_KIND = "independent_train_dev_fit"
EXPECTED_TASK_ID = "d_vlog_current_depression"
EXPECTED_SEEDS = (42, 43, 44)
DEFAULT_TRAIN_COUNT = 647
DEFAULT_VALIDATION_COUNT = 102
DEFAULT_TEST_COUNT = 212
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_PREDICTION_KEYS = frozenset(
    {
        "session_id",
        "split",
        "label",
        "probability",
        "logit_margin_1_minus_0",
        "predicted_label",
        "threshold",
        "prediction_origin",
    }
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _ids_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _require_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _require_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _read_json_object(path: Path, *, description: str) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {description} at {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return payload, _sha256_bytes(raw)


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8-sig")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read predictions at {path}: {error}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank prediction line at {path}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"prediction at {path}:{line_number} is not an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"prediction file is empty: {path}")
    return rows, _sha256_bytes(raw)


def _validate_label_policy(policy: object) -> tuple[dict[str, Any], str | None]:
    if not isinstance(policy, Mapping):
        raise ValueError("label_policy must be an object")
    required = {
        "dataset",
        "target_definition",
        "source_file_sha256",
        "test_labels_accessed",
    }
    missing = sorted(required - set(policy))
    if missing:
        raise ValueError(f"label_policy is missing fields: {missing}")
    if policy.get("dataset") != "d_vlog":
        raise ValueError("label_policy dataset must be d_vlog")
    target_definition = policy.get("target_definition")
    if target_definition in (None, "", {}, []):
        raise ValueError("label_policy target_definition must be non-empty")
    _require_sha256(
        policy.get("source_file_sha256"),
        name="label_policy source_file_sha256",
    )
    if policy.get("test_labels_accessed") is not False:
        raise ValueError("label_policy must declare test_labels_accessed=false")
    source_file_path = policy.get("source_file_path")
    if source_file_path is not None and (
        not isinstance(source_file_path, str) or not source_file_path.strip()
    ):
        raise ValueError("label_policy source_file_path must be a non-empty string")
    identity = {
        str(key): value for key, value in policy.items() if key != "source_file_path"
    }
    return identity, source_file_path


def _validate_evidence(evidence: object) -> dict[str, Any]:
    if not isinstance(evidence, Mapping):
        raise ValueError("evidence provenance must be an object")
    required = {
        "root",
        "protocol_version",
        "run_manifest_sha256",
        "reference_id",
        "reference_fit_scope",
        "reference_path",
        "reference_sha256",
    }
    missing = sorted(required - set(evidence))
    if missing:
        raise ValueError(f"evidence provenance is missing fields: {missing}")
    if (
        not isinstance(evidence.get("root"), str)
        or not str(evidence.get("root")).strip()
    ):
        raise ValueError("evidence root must be a non-empty string")
    if evidence.get("protocol_version") != "native":
        raise ValueError("D-Vlog supervised training requires the native Evidence protocol")
    _require_sha256(
        evidence.get("run_manifest_sha256"),
        name="evidence run_manifest_sha256",
    )
    if (
        not isinstance(evidence.get("reference_id"), str)
        or not str(evidence.get("reference_id")).strip()
    ):
        raise ValueError("evidence reference_id must be non-empty")
    if evidence.get("reference_fit_scope") != "official_train_only":
        raise ValueError("evidence reference must be fitted on official train only")
    if (
        not isinstance(evidence.get("reference_path"), str)
        or not str(evidence.get("reference_path")).strip()
    ):
        raise ValueError("evidence reference_path must be a non-empty string")
    _require_sha256(evidence.get("reference_sha256"), name="evidence reference_sha256")
    return dict(evidence)


def _validate_transcript(transcript: object) -> dict[str, Any]:
    if not isinstance(transcript, Mapping):
        raise ValueError("transcript provenance must be an object")
    if transcript.get("enabled") is not False:
        raise ValueError("D-Vlog supervised training must disable transcript input")
    if transcript.get("content_exposed_to_model") is not False:
        raise ValueError("D-Vlog transcript content must not be exposed to the model")
    marker = transcript.get("missing_marker")
    if not isinstance(marker, str) or not marker:
        raise ValueError("transcript missing_marker must be a non-empty string")
    marker_sha256 = _require_sha256(
        transcript.get("marker_sha256"), name="transcript marker_sha256"
    )
    missing_marker_sha256 = transcript.get("missing_marker_sha256")
    if missing_marker_sha256 is not None and missing_marker_sha256 != marker_sha256:
        raise ValueError("transcript marker SHA-256 aliases disagree")
    if marker_sha256 != hashlib.sha256(marker.encode("utf-8")).hexdigest():
        raise ValueError("transcript marker_sha256 does not hash missing_marker")
    return dict(transcript)


def _validate_summary(
    summary: Mapping[str, Any],
    *,
    expected_train_count: int,
    expected_validation_count: int,
    expected_test_count: int,
) -> dict[str, Any]:
    if summary.get("experiment") != EXPECTED_EXPERIMENT:
        raise ValueError("supervised run experiment identity mismatch")
    if summary.get("run_kind") != EXPECTED_RUN_KIND:
        raise ValueError("supervised run_kind must be independent_train_dev_fit")
    if summary.get("dataset") != "d_vlog":
        raise ValueError("supervised run dataset must be d_vlog")
    seed = _require_int(summary.get("seed"), name="summary seed")
    if seed not in EXPECTED_SEEDS:
        raise ValueError(f"unexpected supervised seed {seed}")

    task = summary.get("task")
    if not isinstance(task, Mapping) or task.get("task_id") != EXPECTED_TASK_ID:
        raise ValueError("D-Vlog supervised task identity mismatch")
    label_policy, source_file_path = _validate_label_policy(summary.get("label_policy"))
    evidence = _validate_evidence(summary.get("evidence"))
    transcript = _validate_transcript(summary.get("transcript"))

    counts = summary.get("session_counts")
    expected_counts = {
        "all_train": expected_train_count,
        "fit": expected_train_count,
        "validation": expected_validation_count,
        "test": expected_test_count,
    }
    if not isinstance(counts, Mapping) or any(
        _require_int(counts.get(key), name=f"session_counts.{key}") != value
        for key, value in expected_counts.items()
    ):
        raise ValueError(f"summary session_counts must equal {expected_counts}")
    hashes = {
        key: _require_sha256(summary.get(key), name=key)
        for key in (
            "fit_ids_sha256",
            "validation_ids_sha256",
            "test_ids_sha256",
        )
    }
    if len(set(hashes.values())) != len(hashes):
        raise ValueError("fit/validation/test participant hashes must be distinct")

    training = summary.get("training")
    checkpoint = (
        training.get("checkpoint_selection") if isinstance(training, Mapping) else None
    )
    selected_epoch = (
        checkpoint.get("selected_epoch") if isinstance(checkpoint, Mapping) else None
    )
    valid_selected_epoch = (
        isinstance(selected_epoch, int)
        and not isinstance(selected_epoch, bool)
        and 1 <= selected_epoch <= 4
    )
    if (
        not isinstance(training, Mapping)
        or training.get("class_weighting") is not False
        or training.get("maximum_epochs") != 4
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("metric") != "log_loss"
        or not valid_selected_epoch
    ):
        raise ValueError(
            "training must be unweighted, use four maximum epochs, and select "
            "epoch 1..4 by validation log_loss"
        )
    boundaries = summary.get("fit_boundaries")
    required_boundaries = {
        "test_labels_accessed": False,
        "test_gradient_updates": False,
        "dev_gradient_updates": False,
        "cross_dataset_merging": False,
    }
    if not isinstance(boundaries, Mapping) or any(
        boundaries.get(key) is not value for key, value in required_boundaries.items()
    ):
        raise ValueError("train/valid/test fit boundaries are missing or violated")
    metrics = summary.get("metrics")
    validation_metrics = (
        metrics.get("validation") if isinstance(metrics, Mapping) else None
    )
    if not isinstance(validation_metrics, Mapping):
        raise ValueError("summary must contain metrics.validation")
    _require_number(
        validation_metrics.get("log_loss"), name="metrics.validation.log_loss"
    )
    return {
        "seed": seed,
        "task": dict(task),
        "label_policy": label_policy,
        "source_file_path": source_file_path,
        "evidence": evidence,
        "transcript": transcript,
        "fit_boundaries": dict(boundaries),
        "validation_metrics": dict(validation_metrics),
        **hashes,
    }


def _parse_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    expected_validation_count: int,
    expected_test_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, 1):
        if set(row) != _PREDICTION_KEYS:
            raise ValueError(
                f"seed {seed} prediction row {row_number} must contain exactly "
                f"{sorted(_PREDICTION_KEYS)}"
            )
        session_id = row.get("session_id")
        if (
            not isinstance(session_id, str)
            or not session_id
            or session_id != session_id.strip()
            or session_id in seen
        ):
            raise ValueError(f"seed {seed} has invalid or duplicate session_id")
        seen.add(session_id)
        split = row.get("split")
        if split not in {"valid", "test"}:
            raise ValueError(f"seed {seed} prediction split must be valid or test")
        label = row.get("label")
        if split == "valid":
            if isinstance(label, bool) or label not in {0, 1}:
                raise ValueError(f"seed {seed} valid label must be binary")
        elif label is not None:
            raise ValueError(f"seed {seed} test label must be null")
        probability = _require_number(
            row.get("probability"), name=f"seed {seed} probability for {session_id}"
        )
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"seed {seed} probability must be in [0, 1]")
        margin = _require_number(
            row.get("logit_margin_1_minus_0"),
            name=f"seed {seed} logit margin for {session_id}",
        )
        # Per-seed threshold decisions are intentionally validated only as typed
        # provenance; the aggregate never uses them for its frozen decision rule.
        row_threshold = _require_number(
            row.get("threshold"), name=f"seed {seed} row threshold"
        )
        if not 0.0 <= row_threshold <= 1.0:
            raise ValueError(f"seed {seed} row threshold must be in [0, 1]")
        predicted_label = row.get("predicted_label")
        if isinstance(predicted_label, bool) or predicted_label not in {0, 1}:
            raise ValueError(f"seed {seed} predicted_label must be binary")
        origin = row.get("prediction_origin")
        if not isinstance(origin, str) or not origin:
            raise ValueError(f"seed {seed} prediction_origin must be non-empty")
        parsed = {
            "session_id": session_id,
            "split": split,
            "label": label,
            "probability": probability,
            "logit_margin_1_minus_0": margin,
        }
        (valid if split == "valid" else test).append(parsed)
    if len(valid) != expected_validation_count or len(test) != expected_test_count:
        raise ValueError(
            f"seed {seed} valid/test prediction counts must be "
            f"{expected_validation_count}/{expected_test_count}"
        )
    return valid, test


def _log_loss(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    epsilon = 1e-15
    return -sum(
        label * math.log(min(max(probability, epsilon), 1.0 - epsilon))
        + (1 - label) * math.log(min(max(1.0 - probability, epsilon), 1.0 - epsilon))
        for label, probability in zip(labels, probabilities)
    ) / len(labels)


def _roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("source-valid ROC AUC requires both classes")
    ranked = sorted(zip(scores, labels), key=lambda item: item[0])
    favorable_pairs = 0.0
    negatives_before = 0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        group = ranked[index:end]
        group_positives = sum(label for _, label in group)
        group_negatives = len(group) - group_positives
        favorable_pairs += group_positives * (negatives_before + 0.5 * group_negatives)
        negatives_before += group_negatives
        index = end
    return favorable_pairs / (positive_count * negative_count)


def _average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    positive_count = sum(labels)
    if positive_count == 0:
        raise ValueError("source-valid average precision requires positives")
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    true_positive = 0
    false_positive = 0
    value = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        group = ranked[index:end]
        group_positive = sum(label for _, label in group)
        true_positive += group_positive
        false_positive += len(group) - group_positive
        value += (group_positive / positive_count) * (
            true_positive / (true_positive + false_positive)
        )
        index = end
    return value


def _balanced_accuracy(
    labels: Sequence[int], probabilities: Sequence[float], threshold: float
) -> float:
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("balanced accuracy threshold selection requires both classes")
    true_positive = sum(
        label == 1 and probability >= threshold
        for label, probability in zip(labels, probabilities)
    )
    true_negative = sum(
        label == 0 and probability < threshold
        for label, probability in zip(labels, probabilities)
    )
    return 0.5 * (true_positive / positive_count + true_negative / negative_count)


def select_unique_balanced_accuracy_threshold(
    labels: Sequence[int], probabilities: Sequence[float]
) -> tuple[float, float]:
    """Return one deterministic BA optimum, preferring 0.5 proximity then high t."""

    if len(labels) != len(probabilities) or not labels:
        raise ValueError("threshold selection inputs must be non-empty and aligned")
    unique = sorted(set(float(value) for value in probabilities))
    candidates = set(unique)
    candidates.add(0.5)
    candidates.update((left + right) / 2.0 for left, right in zip(unique, unique[1:]))
    scored = [
        (
            _balanced_accuracy(labels, probabilities, threshold),
            -abs(threshold - 0.5),
            threshold,
        )
        for threshold in sorted(candidates)
    ]
    best = max(scored)
    return float(best[2]), float(best[0])


def _assert_validation_metrics(
    metadata: Mapping[str, Any], valid: Sequence[Mapping[str, Any]], *, seed: int
) -> dict[str, Any]:
    labels = [int(row["label"]) for row in valid]
    probabilities = [float(row["probability"]) for row in valid]
    metrics = metadata["validation_metrics"]
    expected = {
        "count": len(labels),
        "positive_count": sum(labels),
        "negative_count": len(labels) - sum(labels),
    }
    for key, value in expected.items():
        if _require_int(metrics.get(key), name=f"metrics.validation.{key}") != value:
            raise ValueError(f"seed {seed} validation {key} does not match predictions")
    reported_log_loss = _require_number(
        metrics.get("log_loss"), name="metrics.validation.log_loss"
    )
    if not math.isclose(
        reported_log_loss,
        _log_loss(labels, probabilities),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(f"seed {seed} validation log_loss does not match predictions")
    positive_count = sum(labels)
    prevalence = positive_count / len(labels)
    roc_auc = _roc_auc(labels, probabilities)
    average_precision = _average_precision(labels, probabilities)
    brier = statistics.fmean(
        (probability - label) ** 2 for label, probability in zip(labels, probabilities)
    )
    log_loss = _log_loss(labels, probabilities)
    probability_range = max(probabilities) - min(probabilities)
    unique_probability_count = len(set(probabilities))
    constant_brier = prevalence * (1.0 - prevalence)
    constant_log_loss = -(
        prevalence * math.log(prevalence)
        + (1.0 - prevalence) * math.log(1.0 - prevalence)
    )
    checks = {
        "roc_auc_above_random": roc_auc > 0.50,
        "average_precision_above_prior_plus_0_03": average_precision
        > prevalence + 0.03,
        "brier_better_than_constant_prior": brier < constant_brier,
        "log_loss_better_than_constant_prior": log_loss < constant_log_loss,
        "probabilities_not_collapsed": unique_probability_count >= min(10, len(labels))
        and probability_range >= 0.05,
    }
    result = {
        "seed": seed,
        "checks": checks,
        "passed": all(checks.values()),
        "validation": {
            "count": len(labels),
            "positive_count": positive_count,
            "negative_count": len(labels) - positive_count,
            "prevalence": prevalence,
            "roc_auc": roc_auc,
            "average_precision": average_precision,
            "brier": brier,
            "log_loss": log_loss,
        },
        "constant_prior": {
            "average_precision": prevalence,
            "brier": constant_brier,
            "log_loss": constant_log_loss,
        },
        "score_diagnostics": {
            "unique_probability_count": unique_probability_count,
            "probability_min": min(probabilities),
            "probability_max": max(probabilities),
            "probability_range": probability_range,
        },
    }
    if not result["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"seed {seed} failed source-valid gate checks: {failed}")
    return result


def _write_jsonl_text(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(_canonical_json(row) + "\n" for row in rows)


def aggregate_d_vlog_supervised(
    *,
    seed_dirs: Sequence[Path],
    output_dir: Path,
    expected_train_count: int = DEFAULT_TRAIN_COUNT,
    expected_validation_count: int = DEFAULT_VALIDATION_COUNT,
    expected_test_count: int = DEFAULT_TEST_COUNT,
) -> dict[str, Any]:
    """Validate, ensemble, threshold on validation only, and freeze test scores."""

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    if len(seed_dirs) != len(EXPECTED_SEEDS):
        raise ValueError("exactly three seed directories are required")
    if min(expected_train_count, expected_validation_count, expected_test_count) <= 0:
        raise ValueError("expected split counts must be positive")

    runs: dict[int, dict[str, Any]] = {}
    for directory in seed_dirs:
        summary_path = directory / "summary.json"
        predictions_path = directory / "predictions.jsonl"
        summary, summary_sha256 = _read_json_object(
            summary_path, description="supervised summary"
        )
        metadata = _validate_summary(
            summary,
            expected_train_count=expected_train_count,
            expected_validation_count=expected_validation_count,
            expected_test_count=expected_test_count,
        )
        seed = int(metadata["seed"])
        if seed in runs:
            raise ValueError(f"duplicate supervised seed {seed}")
        prediction_rows, predictions_sha256 = _read_jsonl(predictions_path)
        valid, test = _parse_predictions(
            prediction_rows,
            seed=seed,
            expected_validation_count=expected_validation_count,
            expected_test_count=expected_test_count,
        )
        if (
            _ids_sha256([str(row["session_id"]) for row in valid])
            != metadata["validation_ids_sha256"]
        ):
            raise ValueError(f"seed {seed} validation ID hash mismatch")
        if (
            _ids_sha256([str(row["session_id"]) for row in test])
            != metadata["test_ids_sha256"]
        ):
            raise ValueError(f"seed {seed} test ID hash mismatch")
        valid_gate = _assert_validation_metrics(metadata, valid, seed=seed)
        runs[seed] = {
            "directory": directory,
            "summary_path": summary_path,
            "predictions_path": predictions_path,
            "summary_sha256": summary_sha256,
            "predictions_sha256": predictions_sha256,
            "metadata": metadata,
            "valid": valid,
            "test": test,
            "valid_gate": valid_gate,
        }
    if set(runs) != set(EXPECTED_SEEDS):
        raise ValueError("supervised seeds must be exactly 42, 43, and 44")

    first = runs[EXPECTED_SEEDS[0]]
    identity_fields = (
        "task",
        "label_policy",
        "evidence",
        "transcript",
        "fit_boundaries",
        "fit_ids_sha256",
        "validation_ids_sha256",
        "test_ids_sha256",
    )
    for seed in EXPECTED_SEEDS[1:]:
        for field in identity_fields:
            if _canonical_json(runs[seed]["metadata"][field]) != _canonical_json(
                first["metadata"][field]
            ):
                raise ValueError(f"cross-seed {field} identity mismatch")

    valid_ids = [str(row["session_id"]) for row in first["valid"]]
    test_ids = [str(row["session_id"]) for row in first["test"]]
    valid_labels = [int(row["label"]) for row in first["valid"]]
    valid_by_seed: dict[int, dict[str, Mapping[str, Any]]] = {}
    test_by_seed: dict[int, dict[str, Mapping[str, Any]]] = {}
    for seed in EXPECTED_SEEDS:
        valid_by_seed[seed] = {
            str(row["session_id"]): row for row in runs[seed]["valid"]
        }
        test_by_seed[seed] = {str(row["session_id"]): row for row in runs[seed]["test"]}
        if (
            list(valid_by_seed[seed]) != valid_ids
            or list(test_by_seed[seed]) != test_ids
        ):
            raise ValueError(f"seed {seed} valid/test ID order differs across seeds")
        if [
            int(valid_by_seed[seed][item]["label"]) for item in valid_ids
        ] != valid_labels:
            raise ValueError(f"seed {seed} validation labels differ across seeds")

    validation_probabilities = [
        statistics.fmean(
            float(valid_by_seed[seed][session_id]["probability"])
            for seed in EXPECTED_SEEDS
        )
        for session_id in valid_ids
    ]
    threshold, validation_balanced_accuracy = select_unique_balanced_accuracy_threshold(
        valid_labels, validation_probabilities
    )
    validation_fingerprint_rows = [
        {
            "session_id": session_id,
            "label": label,
            "probability": probability,
            "seed_probabilities": [
                {
                    "seed": seed,
                    "probability": float(
                        valid_by_seed[seed][session_id]["probability"]
                    ),
                }
                for seed in EXPECTED_SEEDS
            ],
        }
        for session_id, label, probability in zip(
            valid_ids, valid_labels, validation_probabilities
        )
    ]
    validation_ensemble_sha256 = _sha256_bytes(
        _write_jsonl_text(validation_fingerprint_rows).encode("utf-8")
    )

    frozen_rows: list[dict[str, Any]] = []
    for session_id in test_ids:
        seed_predictions = [
            {
                "seed": seed,
                "probability": float(test_by_seed[seed][session_id]["probability"]),
                "logit_margin_1_minus_0": float(
                    test_by_seed[seed][session_id]["logit_margin_1_minus_0"]
                ),
            }
            for seed in EXPECTED_SEEDS
        ]
        probability = statistics.fmean(
            float(item["probability"]) for item in seed_predictions
        )
        margin = statistics.fmean(
            float(item["logit_margin_1_minus_0"]) for item in seed_predictions
        )
        frozen_rows.append(
            {
                "session_id": session_id,
                "split": "test",
                "label": None,
                "probability": probability,
                "logit_margin_1_minus_0": margin,
                "predicted_label": int(probability >= threshold),
                "threshold": threshold,
                "prediction_origin": "d_vlog_supervised_three_seed_test_mean",
                "seed_predictions": seed_predictions,
            }
        )
    prediction_text = _write_jsonl_text(frozen_rows)
    prediction_sha256 = _sha256_bytes(prediction_text.encode("utf-8"))
    metadata = first["metadata"]
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "experiment": "d_vlog_supervised_three_seed_freeze",
        "source_experiment": EXPECTED_EXPERIMENT,
        "dataset": "d_vlog",
        "task": metadata["task"],
        "label_policy": metadata["label_policy"],
        "evidence": metadata["evidence"],
        "transcript": metadata["transcript"],
        "seeds": list(EXPECTED_SEEDS),
        "session_counts": {
            "all_train": expected_train_count,
            "fit": expected_train_count,
            "validation": expected_validation_count,
            "test": expected_test_count,
        },
        "fit_ids_sha256": metadata["fit_ids_sha256"],
        "validation_ids_sha256": metadata["validation_ids_sha256"],
        "test_ids_sha256": metadata["test_ids_sha256"],
        "threshold": {
            "value": threshold,
            "metric": "balanced_accuracy",
            "balanced_accuracy": validation_balanced_accuracy,
            "source": "three_seed_validation_probability_mean",
            "candidate_rule": "unique_scores_midpoints_and_0.5",
            "tie_break": "nearest_0.5_then_higher_threshold",
            "validation_ensemble_sha256": validation_ensemble_sha256,
            "test_probabilities_or_labels_used": False,
        },
        "source_valid_gate": {
            "gate": "G2_d_vlog_official_valid_three_seed",
            "runs": [runs[seed]["valid_gate"] for seed in EXPECTED_SEEDS],
            "passed": True,
            "test_labels_accessed": False,
        },
        "prediction_file": "blind_predictions.jsonl",
        "prediction_sha256": prediction_sha256,
        "freeze_complete": True,
        "test_labels_present": False,
        "test_labels_accessed": False,
        "known_prior_test_label_access": True,
        "untouched_external_test_claim": False,
        "test_status_note": (
            "D-Vlog official test labels were previously accessed by the transfer "
            "experiment; this supervised freeze is not an untouched external test."
        ),
        "source_artifacts": [
            {
                "seed": seed,
                "summary_path": str(runs[seed]["summary_path"].resolve()),
                "summary_sha256": runs[seed]["summary_sha256"],
                "predictions_path": str(runs[seed]["predictions_path"].resolve()),
                "predictions_sha256": runs[seed]["predictions_sha256"],
                "source_file_path": runs[seed]["metadata"]["source_file_path"],
            }
            for seed in EXPECTED_SEEDS
        ],
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        prediction_path = staging / "blind_predictions.jsonl"
        prediction_path.write_text(prediction_text, encoding="utf-8")
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for path in (prediction_path, manifest_path):
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


# A descriptive alias for callers that use "freeze" terminology.
freeze_supervised_predictions = aggregate_d_vlog_supervised


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-dir", required=True, nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-train-count", type=int, default=DEFAULT_TRAIN_COUNT)
    parser.add_argument(
        "--expected-validation-count", type=int, default=DEFAULT_VALIDATION_COUNT
    )
    parser.add_argument("--expected-test-count", type=int, default=DEFAULT_TEST_COUNT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = aggregate_d_vlog_supervised(
        seed_dirs=args.seed_dir,
        output_dir=args.output_dir,
        expected_train_count=args.expected_train_count,
        expected_validation_count=args.expected_validation_count,
        expected_test_count=args.expected_test_count,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
