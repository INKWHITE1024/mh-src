"""Unlock metrics for a frozen three-seed supervised D-Vlog ensemble."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_EXPECTED_TEST_COUNT = 212
DEFAULT_BOOTSTRAP_SAMPLES = 2_000
DEFAULT_BOOTSTRAP_SEED = 20_260_722
EXPECTED_SEEDS = (42, 43, 44)
LABEL_MAPPING = {"depression": 1, "normal": 0}
_FROZEN_ROW_KEYS = frozenset(
    {
        "session_id",
        "split",
        "label",
        "probability",
        "logit_margin_1_minus_0",
        "predicted_label",
        "threshold",
        "prediction_origin",
        "seed_predictions",
    }
)
_SEED_ROW_KEYS = frozenset({"seed", "probability", "logit_margin_1_minus_0"})


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _ids_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


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
        raise ValueError(f"{description} must be a JSON object")
    return payload, _sha256_bytes(raw)


def _read_frozen_rows(path: Path, raw: bytes) -> list[dict[str, Any]]:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"frozen predictions are not UTF-8: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank frozen prediction line at {path}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid frozen JSON at {path}:{line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"frozen row at {path}:{line_number} is not an object")
        rows.append(row)
    if not rows:
        raise ValueError("frozen prediction file is empty")
    return rows


def _validate_freeze_before_label_access(
    *,
    manifest: Mapping[str, Any],
    predictions_path: Path,
    expected_test_count: int,
) -> tuple[list[dict[str, Any]], float, str]:
    if manifest.get("schema_version") != "1.0.0":
        raise ValueError("unsupported supervised freeze schema")
    if manifest.get("experiment") != "d_vlog_supervised_three_seed_freeze":
        raise ValueError("supervised freeze experiment identity mismatch")
    if manifest.get("source_experiment") != "qwen_thinker_d_vlog_supervised":
        raise ValueError("supervised source experiment identity mismatch")
    if manifest.get("dataset") != "d_vlog":
        raise ValueError("supervised freeze dataset must be d_vlog")
    if manifest.get("seeds") != list(EXPECTED_SEEDS):
        raise ValueError("supervised freeze must contain seeds 42, 43, and 44")
    if manifest.get("prediction_file") != "blind_predictions.jsonl":
        raise ValueError("supervised freeze prediction filename mismatch")
    if manifest.get("freeze_complete") is not True:
        raise ValueError("supervised test predictions are not frozen")
    if manifest.get("test_labels_present") is not False:
        raise ValueError("freeze must declare test_labels_present=false")
    if manifest.get("test_labels_accessed") is not False:
        raise ValueError("freeze must declare test_labels_accessed=false")
    if manifest.get("known_prior_test_label_access") is not True:
        raise ValueError("freeze must record known prior transfer test access")
    if manifest.get("untouched_external_test_claim") is not False:
        raise ValueError("freeze must not claim an untouched external test")
    counts = manifest.get("session_counts")
    if not isinstance(counts, Mapping) or counts.get("test") != expected_test_count:
        raise ValueError("freeze test count does not match the expected cohort")
    threshold_payload = manifest.get("threshold")
    if not isinstance(threshold_payload, Mapping):
        raise ValueError("freeze lacks validation threshold provenance")
    threshold = _require_number(
        threshold_payload.get("value"), name="frozen validation threshold"
    )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("frozen validation threshold must be in [0, 1]")
    if any(
        (
            threshold_payload.get("source") != "three_seed_validation_probability_mean",
            threshold_payload.get("metric") != "balanced_accuracy",
            threshold_payload.get("candidate_rule")
            != "unique_scores_midpoints_and_0.5",
            threshold_payload.get("tie_break") != "nearest_0.5_then_higher_threshold",
            threshold_payload.get("test_probabilities_or_labels_used") is not False,
        )
    ):
        raise ValueError("freeze threshold is not validation-only BA selection")
    source_gate = manifest.get("source_valid_gate")
    gate_runs = source_gate.get("runs") if isinstance(source_gate, Mapping) else None
    expected_checks = {
        "roc_auc_above_random",
        "average_precision_above_prior_plus_0_03",
        "brier_better_than_constant_prior",
        "log_loss_better_than_constant_prior",
        "probabilities_not_collapsed",
    }
    if (
        not isinstance(source_gate, Mapping)
        or source_gate.get("gate") != "G2_d_vlog_official_valid_three_seed"
        or source_gate.get("passed") is not True
        or source_gate.get("test_labels_accessed") is not False
        or not isinstance(gate_runs, list)
        or len(gate_runs) != len(EXPECTED_SEEDS)
    ):
        raise ValueError("freeze lacks the passed three-seed source-valid gate")
    for expected_seed, run in zip(EXPECTED_SEEDS, gate_runs):
        checks = run.get("checks") if isinstance(run, Mapping) else None
        if (
            not isinstance(run, Mapping)
            or run.get("seed") != expected_seed
            or run.get("passed") is not True
            or not isinstance(checks, Mapping)
            or set(checks) != expected_checks
            or not all(value is True for value in checks.values())
        ):
            raise ValueError("freeze source-valid seed checks are incomplete")

    try:
        prediction_bytes = predictions_path.read_bytes()
    except OSError as error:
        raise ValueError(
            f"cannot read frozen predictions at {predictions_path}"
        ) from error
    prediction_sha256 = _sha256_bytes(prediction_bytes)
    if manifest.get("prediction_sha256") != prediction_sha256:
        raise ValueError("frozen prediction SHA-256 mismatch")
    rows = _read_frozen_rows(predictions_path, prediction_bytes)
    if len(rows) != expected_test_count:
        raise ValueError(
            f"frozen test count is {len(rows)}; expected {expected_test_count}"
        )

    seen: set[str] = set()
    parsed_rows: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, 1):
        if set(row) != _FROZEN_ROW_KEYS:
            raise ValueError(
                f"frozen row {row_number} must contain exactly "
                f"{sorted(_FROZEN_ROW_KEYS)}"
            )
        session_id = row.get("session_id")
        if (
            not isinstance(session_id, str)
            or not session_id
            or session_id != session_id.strip()
            or session_id in seen
        ):
            raise ValueError("frozen predictions contain an invalid or duplicate ID")
        seen.add(session_id)
        if row.get("split") != "test":
            raise ValueError(f"frozen prediction {session_id} is not a test row")
        if row.get("label") is not None:
            raise ValueError(f"frozen test label must remain null for {session_id}")
        probability = _require_number(
            row.get("probability"), name=f"frozen probability for {session_id}"
        )
        margin = _require_number(
            row.get("logit_margin_1_minus_0"),
            name=f"frozen logit margin for {session_id}",
        )
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"frozen probability for {session_id} is outside [0, 1]")
        row_threshold = _require_number(
            row.get("threshold"), name=f"frozen threshold for {session_id}"
        )
        if row_threshold != threshold:
            raise ValueError(f"row threshold differs from manifest for {session_id}")
        predicted_label = row.get("predicted_label")
        if (
            isinstance(predicted_label, bool)
            or predicted_label not in {0, 1}
            or predicted_label != int(probability >= threshold)
        ):
            raise ValueError(f"frozen decision is inconsistent for {session_id}")
        if row.get("prediction_origin") != "d_vlog_supervised_three_seed_test_mean":
            raise ValueError(f"frozen prediction origin mismatch for {session_id}")

        seed_rows = row.get("seed_predictions")
        if not isinstance(seed_rows, list) or len(seed_rows) != len(EXPECTED_SEEDS):
            raise ValueError(f"seed predictions are incomplete for {session_id}")
        parsed_seed_rows: list[dict[str, Any]] = []
        for expected_seed, seed_row in zip(EXPECTED_SEEDS, seed_rows):
            if not isinstance(seed_row, Mapping) or set(seed_row) != _SEED_ROW_KEYS:
                raise ValueError(f"invalid seed prediction for {session_id}")
            seed = _require_int(seed_row.get("seed"), name="seed prediction seed")
            if seed != expected_seed:
                raise ValueError(f"seed order mismatch for {session_id}")
            seed_probability = _require_number(
                seed_row.get("probability"),
                name=f"seed {seed} probability for {session_id}",
            )
            seed_margin = _require_number(
                seed_row.get("logit_margin_1_minus_0"),
                name=f"seed {seed} margin for {session_id}",
            )
            if not 0.0 <= seed_probability <= 1.0:
                raise ValueError(f"seed {seed} probability is outside [0, 1]")
            parsed_seed_rows.append(
                {
                    "seed": seed,
                    "probability": seed_probability,
                    "logit_margin_1_minus_0": seed_margin,
                }
            )
        if not math.isclose(
            probability,
            statistics.fmean(item["probability"] for item in parsed_seed_rows),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                f"frozen probability is not the seed mean for {session_id}"
            )
        if not math.isclose(
            margin,
            statistics.fmean(
                item["logit_margin_1_minus_0"] for item in parsed_seed_rows
            ),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"frozen margin is not the seed mean for {session_id}")
        parsed_rows.append(
            {
                **dict(row),
                "probability": probability,
                "logit_margin_1_minus_0": margin,
                "seed_predictions": parsed_seed_rows,
            }
        )
    test_ids_sha256 = _ids_sha256([str(row["session_id"]) for row in parsed_rows])
    if manifest.get("test_ids_sha256") != test_ids_sha256:
        raise ValueError("frozen test ID SHA-256 mismatch")
    return parsed_rows, threshold, prediction_sha256


def _read_official_test_labels(
    path: Path, *, expected_test_count: int
) -> list[dict[str, Any]]:
    """Parse label/duration cells only after the row is identified as test."""

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError("official D-Vlog labels file is empty") from error
        columns = {name.strip().casefold(): index for index, name in enumerate(header)}
        required = {"index", "label", "duration", "fold"}
        missing = sorted(required - set(columns))
        if missing:
            raise ValueError(f"official D-Vlog labels missing columns: {missing}")
        id_index = columns["index"]
        split_index = columns["fold"]
        label_index = columns["label"]
        duration_index = columns["duration"]
        seen: set[str] = set()
        test_rows: list[dict[str, Any]] = []
        for line_number, row in enumerate(reader, 2):
            if max(id_index, split_index) >= len(row):
                raise ValueError(f"short official row at {path}:{line_number}")
            session_id = row[id_index].strip()
            split = row[split_index].strip().casefold()
            if not session_id or session_id in seen:
                raise ValueError(
                    f"invalid or duplicate official ID at line {line_number}"
                )
            seen.add(session_id)
            if split != "test":
                continue
            if max(label_index, duration_index) >= len(row):
                raise ValueError(f"short official test row at line {line_number}")
            raw_label = row[label_index].strip().casefold()
            if raw_label not in LABEL_MAPPING:
                raise ValueError(
                    f"unsupported official test label {raw_label!r} for {session_id}"
                )
            try:
                duration = float(row[duration_index])
            except ValueError as error:
                raise ValueError(
                    f"invalid duration for test ID {session_id}"
                ) from error
            if not math.isfinite(duration) or duration <= 0.0:
                raise ValueError(f"invalid duration for test ID {session_id}")
            test_rows.append(
                {
                    "session_id": session_id,
                    "label": LABEL_MAPPING[raw_label],
                    "duration": duration,
                }
            )
    if len(test_rows) != expected_test_count:
        raise ValueError(
            f"official test count is {len(test_rows)}; expected {expected_test_count}"
        )
    return test_rows


def _official_test_view_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    canonical = "\n".join(
        _canonical_json(
            {
                "session_id": str(row["session_id"]),
                "label": int(row["label"]),
                "duration": float(row["duration"]),
            }
        )
        for row in rows
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ROC AUC requires both classes")
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
        raise ValueError("average precision requires a positive class")
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


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _paired_stratified_bootstrap(
    labels: Sequence[int],
    model_scores: Sequence[float],
    duration_scores: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, dict[str, list[float]]]:
    positive_pairs = [
        (model, duration)
        for label, model, duration in zip(labels, model_scores, duration_scores)
        if label == 1
    ]
    negative_pairs = [
        (model, duration)
        for label, model, duration in zip(labels, model_scores, duration_scores)
        if label == 0
    ]
    if not positive_pairs or not negative_pairs:
        raise ValueError("stratified bootstrap requires both label classes")
    bootstrap_labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)
    generator = random.Random(seed)
    values = {
        "model_auc": [],
        "model_ap": [],
        "duration_auc": [],
        "duration_ap": [],
        "auc_delta": [],
        "ap_delta": [],
    }
    for _ in range(samples):
        sampled = [
            positive_pairs[generator.randrange(len(positive_pairs))]
            for _ in positive_pairs
        ] + [
            negative_pairs[generator.randrange(len(negative_pairs))]
            for _ in negative_pairs
        ]
        model = [item[0] for item in sampled]
        duration = [item[1] for item in sampled]
        model_auc = _roc_auc(bootstrap_labels, model)
        model_ap = _average_precision(bootstrap_labels, model)
        duration_auc = _roc_auc(bootstrap_labels, duration)
        duration_ap = _average_precision(bootstrap_labels, duration)
        values["model_auc"].append(model_auc)
        values["model_ap"].append(model_ap)
        values["duration_auc"].append(duration_auc)
        values["duration_ap"].append(duration_ap)
        values["auc_delta"].append(model_auc - duration_auc)
        values["ap_delta"].append(model_ap - duration_ap)

    def interval(key: str) -> list[float]:
        return [_percentile(values[key], 0.025), _percentile(values[key], 0.975)]

    return {
        "model": {
            "roc_auc": interval("model_auc"),
            "average_precision": interval("model_ap"),
        },
        "duration": {
            "roc_auc": interval("duration_auc"),
            "average_precision": interval("duration_ap"),
        },
        "model_minus_duration": {
            "roc_auc": interval("auc_delta"),
            "average_precision": interval("ap_delta"),
        },
    }


def _secondary_metrics(
    labels: Sequence[int], probabilities: Sequence[float], threshold: float
) -> dict[str, Any]:
    predictions = [int(probability >= threshold) for probability in probabilities]
    true_positive = sum(
        label == 1 and prediction == 1 for label, prediction in zip(labels, predictions)
    )
    true_negative = sum(
        label == 0 and prediction == 0 for label, prediction in zip(labels, predictions)
    )
    false_positive = sum(
        label == 0 and prediction == 1 for label, prediction in zip(labels, predictions)
    )
    false_negative = sum(
        label == 1 and prediction == 0 for label, prediction in zip(labels, predictions)
    )
    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    f1_denominator = 2 * true_positive + false_positive + false_negative
    epsilon = 1e-15
    return {
        "threshold": threshold,
        "threshold_source": "three_seed_validation_probability_mean_frozen_manifest",
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "f1": 0.0 if f1_denominator == 0 else 2 * true_positive / f1_denominator,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "brier_score": sum(
            (probability - label) ** 2
            for label, probability in zip(labels, probabilities)
        )
        / len(labels),
        "log_loss": -sum(
            label * math.log(min(max(probability, epsilon), 1.0 - epsilon))
            + (1 - label)
            * math.log(min(max(1.0 - probability, epsilon), 1.0 - epsilon))
            for label, probability in zip(labels, probabilities)
        )
        / len(labels),
        "confusion_matrix": {
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
        },
    }


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid ledger JSON at {path}:{line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"ledger row at {path}:{line_number} is not an object")
        rows.append(row)
    return rows


def _append_ledger(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def evaluate_d_vlog_supervised(
    *,
    predictions_path: Path,
    freeze_manifest_path: Path,
    labels_path: Path,
    output_dir: Path,
    ledger_path: Path | None = None,
    expected_test_count: int = DEFAULT_EXPECTED_TEST_COUNT,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Verify the freeze, then unlock official test metrics exactly once per output."""

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    if expected_test_count <= 0 or bootstrap_samples <= 0:
        raise ValueError("expected_test_count and bootstrap_samples must be positive")
    manifest, freeze_manifest_sha256 = _read_json_object(
        freeze_manifest_path, description="supervised freeze manifest"
    )
    frozen_rows, threshold, prediction_sha256 = _validate_freeze_before_label_access(
        manifest=manifest,
        predictions_path=predictions_path,
        expected_test_count=expected_test_count,
    )
    if ledger_path is not None and any(
        row.get("event") == "d_vlog_supervised_metric_unlock"
        and row.get("prediction_sha256") == prediction_sha256
        for row in _read_ledger(ledger_path)
    ):
        raise ValueError("this frozen supervised prediction has already been unlocked")

    # This is the first operation in this function that opens the official label file.
    official = _read_official_test_labels(
        labels_path, expected_test_count=expected_test_count
    )
    frozen_by_id = {str(row["session_id"]): row for row in frozen_rows}
    official_ids = {str(row["session_id"]) for row in official}
    if set(frozen_by_id) != official_ids:
        missing = sorted(official_ids - set(frozen_by_id))
        unexpected = sorted(set(frozen_by_id) - official_ids)
        raise ValueError(
            "frozen/official test ID mismatch: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    joined = [
        {**frozen_by_id[str(label_row["session_id"])], **label_row}
        for label_row in official
    ]
    labels = [int(row["label"]) for row in joined]
    probabilities = [float(row["probability"]) for row in joined]
    durations = [float(row["duration"]) for row in joined]
    if not 0 < sum(labels) < len(labels):
        raise ValueError("official D-Vlog test labels must contain both classes")

    bootstrap = _paired_stratified_bootstrap(
        labels,
        probabilities,
        durations,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    model_auc = _roc_auc(labels, probabilities)
    model_ap = _average_precision(labels, probabilities)
    duration_auc = _roc_auc(labels, durations)
    duration_ap = _average_precision(labels, durations)

    seed_runs: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(EXPECTED_SEEDS):
        seed_probabilities = [
            float(row["seed_predictions"][seed_index]["probability"]) for row in joined
        ]
        seed_runs.append(
            {
                "seed": seed,
                "roc_auc": _roc_auc(labels, seed_probabilities),
                "average_precision": _average_precision(labels, seed_probabilities),
            }
        )
    seed_auc = [float(row["roc_auc"]) for row in seed_runs]
    seed_ap = [float(row["average_precision"]) for row in seed_runs]
    metrics: dict[str, Any] = {
        "schema_version": "1.0.0",
        "experiment": "d_vlog_supervised_metric_unlock_v1",
        "dataset": "d_vlog",
        "task": manifest.get("task"),
        "label_mapping": dict(LABEL_MAPPING),
        "cohort": {
            "split": "test",
            "count": len(labels),
            "positive_count": sum(labels),
            "negative_count": len(labels) - sum(labels),
            "test_ids_sha256": manifest.get("test_ids_sha256"),
        },
        "primary": {
            "roc_auc": {"value": model_auc, "ci_95": bootstrap["model"]["roc_auc"]},
            "average_precision": {
                "value": model_ap,
                "ci_95": bootstrap["model"]["average_precision"],
                "random_ranking_baseline": sum(labels) / len(labels),
            },
            "bootstrap": {
                "unit": "video",
                "stratified_by_label": True,
                "samples": bootstrap_samples,
                "seed": bootstrap_seed,
                "interval": "percentile_95",
            },
        },
        "secondary": _secondary_metrics(labels, probabilities, threshold),
        "duration_nuisance": {
            "score": "raw_duration",
            "direction": "longer_duration_scores_more_positive",
            "roc_auc": duration_auc,
            "roc_auc_ci_95": bootstrap["duration"]["roc_auc"],
            "average_precision": duration_ap,
            "average_precision_ci_95": bootstrap["duration"]["average_precision"],
            "paired_model_minus_duration": {
                "roc_auc": {
                    "value": model_auc - duration_auc,
                    "ci_95": bootstrap["model_minus_duration"]["roc_auc"],
                },
                "average_precision": {
                    "value": model_ap - duration_ap,
                    "ci_95": bootstrap["model_minus_duration"]["average_precision"],
                },
                "bootstrap": {
                    "unit": "video",
                    "paired": True,
                    "stratified_by_label": True,
                    "shared_resamples_with_primary": True,
                    "samples": bootstrap_samples,
                    "seed": bootstrap_seed,
                    "interval": "percentile_95",
                },
            },
        },
        "seed_test_metrics": {
            "available_only_after_official_test_unlock": True,
            "runs": seed_runs,
            "roc_auc": {
                "mean": statistics.fmean(seed_auc),
                "sample_sd": statistics.stdev(seed_auc),
            },
            "average_precision": {
                "mean": statistics.fmean(seed_ap),
                "sample_sd": statistics.stdev(seed_ap),
            },
        },
        "test_access_status": {
            "official_test_labels_accessed_during_this_evaluation": True,
            "official_test_previously_accessed_by_transfer_experiment": True,
            "untouched_external_test": False,
            "claim_scope": (
                "supervised post-access D-Vlog evaluation; not an untouched "
                "external-test estimate"
            ),
        },
        "provenance": {
            "prediction_sha256": prediction_sha256,
            "freeze_manifest_sha256": freeze_manifest_sha256,
            "official_test_view_sha256": _official_test_view_sha256(official),
            "freeze_verified_before_official_label_read": True,
            "train_valid_label_values_accessed_by_evaluator": False,
        },
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        metrics_text = (
            json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        metrics_path = staging / "metrics.json"
        metrics_path.write_text(metrics_text, encoding="utf-8")
        with metrics_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if ledger_path is not None:
        _append_ledger(
            ledger_path,
            {
                "schema_version": "1.0.0",
                "event": "d_vlog_supervised_metric_unlock",
                "unlocked_at_utc": datetime.now(timezone.utc).isoformat(),
                "prediction_sha256": prediction_sha256,
                "freeze_manifest_sha256": freeze_manifest_sha256,
                "official_test_view_sha256": metrics["provenance"][
                    "official_test_view_sha256"
                ],
                "metrics_sha256": _sha256_bytes(
                    (output_dir / "metrics.json").read_bytes()
                ),
                "bootstrap_samples": bootstrap_samples,
                "bootstrap_seed": bootstrap_seed,
                "untouched_external_test": False,
            },
        )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--freeze-manifest", required=True, type=Path)
    parser.add_argument("--official-labels", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument(
        "--expected-test-count", type=int, default=DEFAULT_EXPECTED_TEST_COUNT
    )
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = evaluate_d_vlog_supervised(
        predictions_path=args.predictions,
        freeze_manifest_path=args.freeze_manifest,
        labels_path=args.official_labels,
        output_dir=args.output_dir,
        ledger_path=args.ledger,
        expected_test_count=args.expected_test_count,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
