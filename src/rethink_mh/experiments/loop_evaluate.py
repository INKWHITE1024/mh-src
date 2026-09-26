"""Outcome-join evaluation after label-free raw OOF trajectories are frozen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .audit_fit import load_audit_decisions
from .frozen_artifact_manifest import verify_directory_manifest
from .loop_collect import (
    ARM_ORDER,
    COLLECTION_PROTOCOL_VERSION,
)
from .loop_merge import (
    MERGE_PROTOCOL_VERSION,
)
from .loop_prepare import PREPARATION_PROTOCOL_VERSION


EVALUATION_PROTOCOL_VERSION = "loop-evaluation"
DIRECT_REFER_ARM = "same_trigger_direct_refer"
ALL_ARMS = (*ARM_ORDER, DIRECT_REFER_ARM)
_EPSILON = 1e-7


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be one object: {path}")
    return value


def _read_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read {description} {path}: {error}") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise ValueError(f"blank row in {description}: {path}:{line_number}")
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


def _probability(value: object, path: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{path} must be a finite probability")
    return float(value)


def _binary(value: object, path: str) -> int:
    if isinstance(value, bool) or value not in {0, 1}:
        raise ValueError(f"{path} must be binary")
    return int(value)


def _number(value: object, path: str, *, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise ValueError(f"{path} must be finite and at least {minimum}")
    return float(value)


def _binary_log_loss(label: int, probability: float) -> float:
    bounded = min(max(probability, _EPSILON), 1.0 - _EPSILON)
    return -(
        label * math.log(bounded)
        + (1 - label) * math.log(1.0 - bounded)
    )


def _transition(
    *,
    label: int,
    initial_probability: float,
    final_probability: float,
    threshold: float,
    referred: bool,
) -> str:
    initial_correct = int(initial_probability >= threshold) == label
    if referred:
        return "correct_referred" if initial_correct else "wrong_referred"
    final_correct = int(final_probability >= threshold) == label
    if initial_correct and final_correct:
        return "CC_preserved"
    if initial_correct:
        return "CW_harmed"
    if final_correct:
        return "WC_recovered"
    return "WW_unrecovered"


def _roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    positive_rank_sum = sum(
        rank for rank, label in zip(ranks, labels) if label == 1
    )
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _average_precision(
    labels: Sequence[int],
    scores: Sequence[float],
) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    order = sorted(
        range(len(scores)),
        key=lambda index: (-scores[index], index),
    )
    true_positives = 0
    total = 0.0
    for rank, index in enumerate(order, 1):
        if labels[index] == 1:
            true_positives += 1
            total += true_positives / rank
    return total / positives


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = [int(row["label"]) for row in rows]
    probabilities = [float(row["final_probability"]) for row in rows]
    accepted = [row for row in rows if not row["referred"]]
    positive_accepted = [
        row for row in accepted if int(row["label"]) == 1
    ]
    false_negative_count = sum(
        int(float(row["final_probability"]) < float(row["threshold"]))
        for row in positive_accepted
    )
    transitions = Counter(str(row["transition"]) for row in rows)
    return {
        "participant_count": len(rows),
        "brier": sum(
            (probability - label) ** 2
            for label, probability in zip(labels, probabilities)
        )
        / len(rows),
        "log_loss": sum(
            _binary_log_loss(label, probability)
            for label, probability in zip(labels, probabilities)
        )
        / len(rows),
        "roc_auc": _roc_auc(labels, probabilities),
        "average_precision": _average_precision(labels, probabilities),
        "coverage": len(accepted) / len(rows),
        "accepted_count": len(accepted),
        "referred_count": len(rows) - len(accepted),
        "accepted_fnr": (
            false_negative_count / len(positive_accepted)
            if positive_accepted
            else None
        ),
        "accepted_positive_count": len(positive_accepted),
        "transition_counts": dict(sorted(transitions.items())),
        "wc_recovered": transitions["WC_recovered"],
        "cc_harmed": transitions["CW_harmed"],
        "net_correction": (
            transitions["WC_recovered"] - transitions["CW_harmed"]
        ),
        "mean_queries": sum(float(row["queries"]) for row in rows) / len(rows),
        "mean_atomic_records": sum(
            float(row["atomic_records"]) for row in rows
        )
        / len(rows),
        "mean_released_tokens": sum(
            float(row["released_tokens"]) for row in rows
        )
        / len(rows),
        "failure_count": sum(not bool(row["arm_valid"]) for row in rows),
    }


def _accepted_fnr(rows: Sequence[Mapping[str, Any]]) -> float | None:
    positives = [row for row in rows if int(row["label"]) == 1]
    if not positives:
        return None
    return sum(
        float(row["final_probability"]) < float(row["threshold"])
        for row in positives
    ) / len(positives)


def _selective_comparison(
    arm_rows: Sequence[Mapping[str, Any]],
    direct_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    direct_accepted = [row for row in direct_rows if not row["referred"]]
    available = sorted(
        (row for row in arm_rows if not row["referred"]),
        key=lambda row: (
            -float(row["final_confidence"]),
            str(row["session_id"]),
        ),
    )
    matched_count = min(len(direct_accepted), len(available))
    matched_rows = available[:matched_count]
    direct_fnr = _accepted_fnr(direct_accepted)
    maximum_at_direct_fnr: int | None = None
    if direct_fnr is not None:
        maximum_at_direct_fnr = 0
        for count in range(1, len(available) + 1):
            current = _accepted_fnr(available[:count])
            if current is not None and current <= direct_fnr + 1e-12:
                maximum_at_direct_fnr = count
    return {
        "direct_refer_coverage": len(direct_accepted) / len(direct_rows),
        "arm_native_coverage": len(available) / len(arm_rows),
        "matched_coverage": matched_count / len(arm_rows),
        "arm_accepted_fnr_at_direct_coverage": _accepted_fnr(matched_rows),
        "direct_refer_accepted_fnr": direct_fnr,
        "arm_coverage_at_or_below_direct_fnr": (
            maximum_at_direct_fnr / len(arm_rows)
            if maximum_at_direct_fnr is not None
            else None
        ),
    }


def _bootstrap_delta(
    arm_rows: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    labels = np.asarray([row["label"] for row in arm_rows], dtype=float)
    initial = np.asarray(
        [row["initial_probability"] for row in arm_rows],
        dtype=float,
    )
    final = np.asarray(
        [row["final_probability"] for row in arm_rows],
        dtype=float,
    )
    threshold = np.asarray([row["threshold"] for row in arm_rows], dtype=float)
    referred = np.asarray([row["referred"] for row in arm_rows], dtype=bool)
    initial_brier = (initial - labels) ** 2
    final_brier = (final - labels) ** 2
    clipped_initial = np.clip(initial, _EPSILON, 1.0 - _EPSILON)
    clipped_final = np.clip(final, _EPSILON, 1.0 - _EPSILON)
    initial_log = -(
        labels * np.log(clipped_initial)
        + (1.0 - labels) * np.log(1.0 - clipped_initial)
    )
    final_log = -(
        labels * np.log(clipped_final)
        + (1.0 - labels) * np.log(1.0 - clipped_final)
    )
    initial_correct = (initial >= threshold) == labels
    final_correct = (final >= threshold) == labels
    net = (
        (~initial_correct & final_correct & ~referred).astype(float)
        - (initial_correct & ~final_correct & ~referred).astype(float)
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(labels), size=(samples, len(labels)))
    estimates = {
        "brier_improvement": np.mean(
            initial_brier[indices] - final_brier[indices],
            axis=1,
        ),
        "log_loss_improvement": np.mean(
            initial_log[indices] - final_log[indices],
            axis=1,
        ),
        "net_correction_rate": np.mean(net[indices], axis=1),
    }
    return {
        key: {
            "estimate": float(np.mean(values)),
            "ci_95": [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ],
        }
        for key, values in estimates.items()
    }


@dataclass(frozen=True, slots=True)
class UtilityWeights:
    brier: float
    harm: float
    query: float
    atomic: float
    token: float
    refer: float


@dataclass(frozen=True, slots=True)
class LoopEvaluationRequest:
    raw_oof_dir: Path
    raw_artifact_manifest: Path
    prepared_root: Path
    prepared_artifact_manifest: Path
    config_path: Path
    output_dir: Path
    audit_decisions: Path | None = None


def _load_config(
    path: Path,
) -> tuple[dict[str, Any], UtilityWeights, int, int, str]:
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read full-loop config {path}: {error}") from error
    if not isinstance(config, dict):
        raise ValueError("full-loop config must be one mapping")
    collection = config.get("collection")
    utility = config.get("primary_utility")
    reporting = config.get("reporting")
    if (
        config.get("protocol_version") != "full-loop"
        or config.get("status") != "preregistered_before_outcome_join"
        or not isinstance(collection, Mapping)
        or not isinstance(utility, Mapping)
        or not isinstance(reporting, Mapping)
        or collection.get("arms") != list(ALL_ARMS)
        or collection.get("raw_freeze_required_before_outcome_join") is not True
    ):
        raise ValueError("full-loop config is incomplete or not preregistered")
    weights = UtilityWeights(
        brier=_number(
            utility.get("negative_brier_weight"),
            "negative_brier_weight",
        ),
        harm=_number(utility.get("cc_harm_penalty"), "cc_harm_penalty"),
        query=_number(utility.get("query_penalty"), "query_penalty"),
        atomic=_number(
            utility.get("atomic_record_penalty"),
            "atomic_record_penalty",
        ),
        token=_number(
            utility.get("released_token_penalty"),
            "released_token_penalty",
        ),
        refer=_number(
            utility.get("referral_penalty"),
            "referral_penalty",
        ),
    )
    samples = int(reporting.get("paired_bootstrap_samples", 0))
    seed = int(reporting.get("paired_bootstrap_seed", 0))
    primary = str(collection.get("primary_method_arm", ""))
    if samples <= 0 or seed <= 0 or primary not in ARM_ORDER:
        raise ValueError("invalid bootstrap or primary method configuration")
    return config, weights, samples, seed, primary


def _verify_frozen_directory(root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = _read_json(manifest_path, "frozen directory manifest")
    return verify_directory_manifest(root, manifest)


def _raw_arm(
    raw: Mapping[str, Any],
    arm_id: str,
    *,
    initial_probability: float,
    initial_confidence: float,
) -> dict[str, Any]:
    if arm_id == DIRECT_REFER_ARM:
        return {
            "arm_id": arm_id,
            "final_probability": initial_probability,
            "final_confidence": initial_confidence,
            "referred": True,
            "queries": 0,
            "atomic_records": 0,
            "released_tokens": 0,
            "arm_valid": True,
            "failure_reason": None,
            "revision_status": "direct_refer",
        }
    if raw.get("status") == "initial_failed":
        return {
            "arm_id": arm_id,
            "final_probability": initial_probability,
            "final_confidence": initial_confidence,
            "referred": True,
            "queries": 0,
            "atomic_records": 0,
            "released_tokens": 0,
            "arm_valid": False,
            "failure_reason": "initial_assessment_failed",
            "revision_status": "fail_closed_refer",
        }
    indexed = {
        str(arm["arm_id"]): arm for arm in raw.get("arms", [])
    }
    arm = indexed[arm_id]
    if arm.get("status") != "ok":
        failure = arm.get("failure", {})
        return {
            "arm_id": arm_id,
            "final_probability": initial_probability,
            "final_confidence": initial_confidence,
            "referred": True,
            "queries": 0,
            "atomic_records": 0,
            "released_tokens": 0,
            "arm_valid": False,
            "failure_reason": failure.get("error_type", "unknown_failure"),
            "revision_status": "fail_closed_refer",
        }
    workflow = arm["workflow"]
    budget = workflow["budget"]
    revision = workflow["revision_assessment"]
    return {
        "arm_id": arm_id,
        "final_probability": _probability(
            workflow["final_risk_probability"],
            f"{arm_id}.final_risk_probability",
        ),
        "final_confidence": _probability(
            workflow["final_confidence"],
            f"{arm_id}.final_confidence",
        ),
        "referred": workflow["outcome"] == "referred_unresolved",
        "queries": int(budget["queries_used"]),
        "atomic_records": int(budget["atomic_records_released"]),
        "released_tokens": int(budget["released_tokens"]),
        "arm_valid": True,
        "failure_reason": None,
        "revision_status": (
            revision["revision_status"]
            if isinstance(revision, Mapping)
            else None
        ),
    }


def _scored_view(
    *,
    session_id: str,
    label: int,
    initial_probability: float,
    threshold: float,
    arm: Mapping[str, Any],
    initial_confidence: float,
    natural_triggered: bool,
    deployed: bool,
    weights: UtilityWeights,
) -> dict[str, Any]:
    apply_arm = not deployed or natural_triggered
    if apply_arm:
        final_probability = float(arm["final_probability"])
        final_confidence = float(arm["final_confidence"])
        referred = bool(arm["referred"])
        queries = int(arm["queries"])
        atomic = int(arm["atomic_records"])
        tokens = int(arm["released_tokens"])
        valid = bool(arm["arm_valid"])
    else:
        final_probability = initial_probability
        final_confidence = initial_confidence
        referred = False
        queries = 0
        atomic = 0
        tokens = 0
        valid = True
    initial_correct = int(initial_probability >= threshold) == label
    final_correct = int(final_probability >= threshold) == label
    harm = initial_correct and not referred and not final_correct
    brier = (final_probability - label) ** 2
    log_loss = _binary_log_loss(label, final_probability)
    utility = (
        -log_loss
        - weights.brier * brier
        - weights.harm * int(harm)
        - weights.query * queries
        - weights.atomic * atomic
        - weights.token * tokens
        - weights.refer * int(referred)
    )
    return {
        "session_id": session_id,
        "arm_id": arm["arm_id"],
        "view": "natural_trigger_policy" if deployed else "forced_arm",
        "label": label,
        "initial_probability": initial_probability,
        "threshold": threshold,
        "natural_triggered": natural_triggered,
        "arm_applied": apply_arm,
        "final_probability": final_probability,
        "final_confidence": final_confidence,
        "referred": referred,
        "queries": queries,
        "atomic_records": atomic,
        "released_tokens": tokens,
        "arm_valid": valid,
        "failure_reason": arm["failure_reason"],
        "revision_status": arm["revision_status"],
        "transition": _transition(
            label=label,
            initial_probability=initial_probability,
            final_probability=final_probability,
            threshold=threshold,
            referred=referred,
        ),
        "brier": brier,
        "log_loss": log_loss,
        "cc_harm": harm,
        "primary_utility": utility,
    }


def evaluate_frozen_loop(request: LoopEvaluationRequest) -> dict[str, Any]:
    """Join outcomes only after verifying both immutable capability trees."""

    if request.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite loop evaluation: {request.output_dir}"
        )
    raw_frozen = _verify_frozen_directory(
        request.raw_oof_dir,
        request.raw_artifact_manifest,
    )
    prepared_frozen = _verify_frozen_directory(
        request.prepared_root,
        request.prepared_artifact_manifest,
    )
    if not (request.raw_oof_dir / "RAW_OOF_FROZEN").is_file():
        raise ValueError("raw OOF freeze marker is missing")
    raw_manifest_path = request.raw_oof_dir / "manifest.json"
    raw_path = request.raw_oof_dir / "trajectories.raw.jsonl"
    raw_manifest = _read_json(raw_manifest_path, "raw OOF manifest")
    raw_rows = _read_jsonl(raw_path, "raw OOF trajectories")
    collection_protocol_version = raw_manifest.get(
        "collection_protocol_version"
    )
    evaluation_protocol_version = EVALUATION_PROTOCOL_VERSION
    prepared_inference_manifest = request.prepared_root / "inference/manifest.json"
    if any(
        (
            collection_protocol_version != COLLECTION_PROTOCOL_VERSION,
            raw_manifest.get("merge_protocol_version")
            != MERGE_PROTOCOL_VERSION,
            raw_manifest.get("files", {})
            .get("trajectories.raw.jsonl", {})
            .get("sha256")
            != _sha256(raw_path),
            raw_manifest.get("fit_boundaries", {}).get(
                "current_sample_targets_accessed"
            )
            is not False,
            raw_manifest.get("sources", {}).get(
                "inference_manifest_sha256"
            )
            != _sha256(prepared_inference_manifest),
        )
    ):
        raise ValueError("raw OOF artifact failed outcome-join boundary checks")

    outcome_dir = request.prepared_root / "outcomes"
    outcome_manifest_path = outcome_dir / "manifest.json"
    outcome_path = outcome_dir / "outcomes.jsonl"
    outcome_manifest = _read_json(
        outcome_manifest_path,
        "outcome capability manifest",
    )
    outcomes = _read_jsonl(outcome_path, "outcomes")
    if any(
        (
            outcome_manifest.get("preparation_protocol_version")
            != PREPARATION_PROTOCOL_VERSION,
            outcome_manifest.get("files", {})
            .get("outcomes.jsonl", {})
            .get("sha256")
            != _sha256(outcome_path),
            outcome_manifest.get("files", {})
            .get("outcomes.jsonl", {})
            .get("used_in_model_inference")
            is not False,
        )
    ):
        raise ValueError("outcome capability failed verification")

    config, weights, bootstrap_samples, bootstrap_seed, primary_arm = (
        _load_config(request.config_path)
    )
    raw_by_id = {
        str(row.get("session_id", "")): row for row in raw_rows
    }
    outcome_by_id = {
        str(row.get("session_id", "")): row for row in outcomes
    }
    if (
        len(raw_by_id) != len(raw_rows)
        or len(outcome_by_id) != len(outcomes)
        or set(raw_by_id) != set(outcome_by_id)
    ):
        raise ValueError("raw trajectories and outcomes have different cohorts")
    audit_triggered: dict[str, bool] | None = None
    if request.audit_decisions is not None:
        audit_triggered = load_audit_decisions(request.audit_decisions)
        if set(audit_triggered) != set(raw_by_id):
            raise ValueError("audit decisions and raw trajectories have different cohorts")

    forced_by_arm: dict[str, list[dict[str, Any]]] = {
        arm_id: [] for arm_id in ALL_ARMS
    }
    deployed_by_arm: dict[str, list[dict[str, Any]]] = {
        arm_id: [] for arm_id in ALL_ARMS
    }
    decisions: list[dict[str, Any]] = []
    arm_rank = {arm_id: index for index, arm_id in enumerate(ALL_ARMS)}
    for session_id in sorted(raw_by_id):
        raw = raw_by_id[session_id]
        outcome = outcome_by_id[session_id]
        label = _binary(outcome.get("label"), f"{session_id}.label")
        initial_probability = _probability(
            outcome.get("initial_probability"),
            f"{session_id}.initial_probability",
        )
        threshold = _probability(
            outcome.get("initial_threshold"),
            f"{session_id}.initial_threshold",
        )
        initial_failed = raw.get("status") == "initial_failed"
        raw_initial_probability = raw.get("initial", {}).get(
            "initial_probability_anchor"
        )
        if not initial_failed and raw_initial_probability != initial_probability:
            raise ValueError(
                f"cannot outcome-join changed initial: {session_id}"
            )
        initial_confidence = (
            0.0
            if initial_failed
            else _probability(
                raw["initial"]["initial_assessment"]["confidence"],
                f"{session_id}.initial_confidence",
            )
        )
        natural_triggered = (
            True
            if initial_failed
            else audit_triggered[session_id]
            if audit_triggered is not None
            else bool(raw["natural_rethink_decision"]["should_rethink"])
        )
        candidates: list[dict[str, Any]] = []
        for arm_id in ALL_ARMS:
            arm = _raw_arm(
                raw,
                arm_id,
                initial_probability=initial_probability,
                initial_confidence=initial_confidence,
            )
            forced = _scored_view(
                session_id=session_id,
                label=label,
                initial_probability=initial_probability,
                threshold=threshold,
                arm=arm,
                initial_confidence=initial_confidence,
                natural_triggered=natural_triggered,
                deployed=False,
                weights=weights,
            )
            deployed = _scored_view(
                session_id=session_id,
                label=label,
                initial_probability=initial_probability,
                threshold=threshold,
                arm=arm,
                initial_confidence=initial_confidence,
                natural_triggered=natural_triggered,
                deployed=True,
                weights=weights,
            )
            forced_by_arm[arm_id].append(forced)
            deployed_by_arm[arm_id].append(deployed)
            if forced["arm_valid"]:
                candidates.append(forced)
        selected = sorted(
            candidates,
            key=lambda row: (
                -float(row["primary_utility"]),
                bool(row["cc_harm"]),
                bool(row["referred"]),
                int(row["released_tokens"]),
                int(row["atomic_records"]),
                arm_rank[str(row["arm_id"])],
            ),
        )[0]
        decisions.append(
            {
                "schema_version": "1.0.0",
                "evaluation_protocol_version": (
                    evaluation_protocol_version
                ),
                "collection_protocol_version": (
                    collection_protocol_version
                ),
                "dataset": raw["dataset"],
                "session_id": session_id,
                "outer_fold": raw["outer_fold"],
                "label": label,
                "initial_probability": initial_probability,
                "initial_threshold": threshold,
                "initial_correct": (
                    int(initial_probability >= threshold) == label
                ),
                "natural_triggered": natural_triggered,
                "initial_assessment_failed": initial_failed,
                "selected_training_arm": selected["arm_id"],
                "selected_training_utility": selected["primary_utility"],
                "selected_training_transition": selected["transition"],
                "arm_scores": [
                    {
                        key: row[key]
                        for key in (
                            "arm_id",
                            "arm_valid",
                            "referred",
                            "final_probability",
                            "final_confidence",
                            "transition",
                            "queries",
                            "atomic_records",
                            "released_tokens",
                            "brier",
                            "log_loss",
                            "cc_harm",
                            "primary_utility",
                        )
                    }
                    for row in (
                        forced_by_arm[arm_id][-1] for arm_id in ALL_ARMS
                    )
                ],
                "used_in_model_text": False,
                "performance_interpretation": "training_oracle_only",
            }
        )

    initial_rows = [
        {
            **row,
            "final_probability": row["initial_probability"],
            "final_confidence": 1.0,
            "referred": False,
            "queries": 0,
            "atomic_records": 0,
            "released_tokens": 0,
            "arm_valid": True,
            "transition": _transition(
                label=int(row["label"]),
                initial_probability=float(row["initial_probability"]),
                final_probability=float(row["initial_probability"]),
                threshold=float(row["threshold"]),
                referred=False,
            ),
        }
        for row in forced_by_arm[DIRECT_REFER_ARM]
    ]
    reference_rows = [
        {
            "session_id": session_id,
            "label": int(outcome_by_id[session_id]["label"]),
            "initial_probability": float(
                outcome_by_id[session_id]["initial_probability"]
            ),
            "threshold": float(
                outcome_by_id[session_id]["reference_threshold"]
            ),
            "final_probability": float(
                outcome_by_id[session_id]["reference_probability"]
            ),
            "final_confidence": 1.0,
            "referred": False,
            "queries": 0,
            "atomic_records": 0,
            "released_tokens": 0,
            "arm_valid": True,
            "transition": "reference_one_pass",
        }
        for session_id in sorted(outcome_by_id)
    ]
    direct_deployed = deployed_by_arm[DIRECT_REFER_ARM]
    arms_summary: dict[str, Any] = {}
    for arm_id in ALL_ARMS:
        forced = forced_by_arm[arm_id]
        deployed = deployed_by_arm[arm_id]
        arms_summary[arm_id] = {
            "forced_counterfactual": _metrics(forced),
            "natural_trigger_policy": _metrics(deployed),
            "natural_trigger_paired_bootstrap_vs_initial": _bootstrap_delta(
                deployed,
                samples=bootstrap_samples,
                seed=bootstrap_seed + arm_rank[arm_id],
            ),
            "selective_vs_same_trigger_direct_refer": (
                None
                if arm_id == DIRECT_REFER_ARM
                else _selective_comparison(deployed, direct_deployed)
            ),
        }

    decisions_text = "".join(
        _canonical_json(row) + "\n" for row in decisions
    )
    request.output_dir.mkdir(parents=True)
    decisions_path = request.output_dir / "training_decisions.jsonl"
    decisions_path.write_text(decisions_text, encoding="utf-8")
    summary = {
        "schema_version": "1.0.0",
        "evaluation_protocol_version": evaluation_protocol_version,
        "collection_protocol_version": collection_protocol_version,
        "dataset": raw_rows[0]["dataset"],
        "session_count": len(raw_rows),
        "primary_method_arm": primary_arm,
        "fixed_arm_results": arms_summary,
        "references": {
            "frozen_native_initial": _metrics(initial_rows),
            "reference_one_pass": _metrics(reference_rows),
        },
        "primary_method_result": arms_summary[primary_arm],
        "training_oracle": {
            "selected_arm_counts": dict(
                sorted(
                    Counter(
                        row["selected_training_arm"] for row in decisions
                    ).items()
                )
            ),
            "not_an_unbiased_performance_estimate": True,
        },
        "utility": {
            "negative_log_loss_weight": 1.0,
            "negative_brier_weight": weights.brier,
            "cc_harm_penalty": weights.harm,
            "query_penalty": weights.query,
            "atomic_record_penalty": weights.atomic,
            "released_token_penalty": weights.token,
            "referral_penalty": weights.refer,
        },
        "files": {
            "training_decisions.jsonl": {
                "sha256": _sha256(decisions_path),
                "row_count": len(decisions),
            }
        },
        "sources": {
            "raw_artifact_sha256": raw_frozen["artifact_sha256"],
            "prepared_artifact_sha256": prepared_frozen["artifact_sha256"],
            "raw_manifest_sha256": _sha256(raw_manifest_path),
            "outcome_manifest_sha256": _sha256(outcome_manifest_path),
            "config_sha256": _sha256(request.config_path),
            "trigger": (
                "learned_audit"
                if request.audit_decisions is not None
                else "heuristic_rules"
            ),
            "audit_decisions_sha256": (
                _sha256(request.audit_decisions)
                if request.audit_decisions is not None
                else None
            ),
            "utility_config_protocol_version": config[
                "protocol_version"
            ],
        },
        "fit_boundaries": {
            "raw_trajectories_frozen_before_outcome_join": True,
            "outcomes_used_for_fixed_arm_scoring": True,
            "outcomes_used_for_training_selection": True,
            "labels_in_model_messages": False,
            "training_oracle_reported_as_performance": False,
            "utility_inherited_unchanged_from_preregistered_config": True,
            "dev_accessed": False,
            "test_accessed": False,
            "reference_retrained": False,
            "reference_recompiled": False,
        },
        "interpretation_boundary": (
            "Fixed-arm OOF results describe the untrained reviewer routes. "
            "Outcome-selected per-session decisions are training material only; "
            "a loop-trained model requires nested OOF or untouched evaluation."
        ),
    }
    summary_path = request.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (request.output_dir / "OUTCOME_JOIN_COMPLETE").touch()
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen raw loop trajectories."
    )
    parser.add_argument("--raw-oof-dir", required=True, type=Path)
    parser.add_argument(
        "--raw-artifact-manifest",
        required=True,
        type=Path,
    )
    parser.add_argument("--prepared-root", required=True, type=Path)
    parser.add_argument(
        "--prepared-artifact-manifest",
        required=True,
        type=Path,
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--audit-decisions",
        type=Path,
        help="Label-free decisions from `rethink-fit-audit decide`; "
        "without it the stored heuristic trigger is used.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_frozen_loop(
        LoopEvaluationRequest(
            raw_oof_dir=args.raw_oof_dir,
            raw_artifact_manifest=args.raw_artifact_manifest,
            prepared_root=args.prepared_root,
            prepared_artifact_manifest=args.prepared_artifact_manifest,
            config_path=args.config,
            output_dir=args.output_dir,
            audit_decisions=args.audit_decisions,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
