"""Learned reliability audit of first-pass assessments.

The audit estimates the error risk ``rho`` of an initial assessment with an
L2-regularized logistic regression over label-free audit features:

* the self-reported confidence ``c0``;
* the operational reliability ``r_s`` of each source, 0 when unavailable;
* the validity-weighted reliable disagreement ``D_rel``;
* the cross-quality disagreement ``D_unrel``;
* the fraction of unavailable sources;
* the binary entropy ``H(p0)`` of the initial risk probability.

The validity ``v_s`` of a source is its normalized out-of-fold AUROC,
``max(0, 2 * AUROC_s - 1)``, computed from the source's own risk readings in
the initial assessments, and it enters the audit through ``D_rel``.  The
estimator is fit on out-of-fold initial assessments, whose target is whether
the first-pass decision was wrong, optionally together with synthetic
degradations whose target is known.  The risk threshold is fixed on
development data at a target trigger rate, which needs no labels.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from .trigger import (
    RethinkPolicyConfig,
    _read_field,
    _unit_interval_mapping,
    _validate_unit_interval,
    binary_normalized_entropy,
    validity_weighted_disagreement,
)

AUDIT_SCHEMA_VERSION = "1.0.0"
DEFAULT_SOURCES: tuple[str, ...] = ("audio", "visual")


def audit_feature_names(sources: Sequence[str] = DEFAULT_SOURCES) -> tuple[str, ...]:
    """Ordered names of the audit features for the given sources."""

    return (
        "confidence",
        *(f"reliability_{source}" for source in sources),
        "reliable_disagreement",
        "unreliable_disagreement",
        "missing_fraction",
        "entropy",
    )


def build_audit_features(
    *,
    risk_probability: float,
    confidence: float,
    source_reliability: Mapping[str, float],
    reliable_disagreement: float,
    unreliable_disagreement: float,
    sources: Sequence[str] = DEFAULT_SOURCES,
) -> dict[str, float]:
    """Assemble the audit feature vector shared by fitting and inference."""

    features: dict[str, float] = {"confidence": float(confidence)}
    for source in sources:
        features[f"reliability_{source}"] = float(source_reliability.get(source, 0.0))
    available = sum(source in source_reliability for source in sources)
    features["reliable_disagreement"] = float(reliable_disagreement)
    features["unreliable_disagreement"] = float(unreliable_disagreement)
    features["missing_fraction"] = 1.0 - available / len(sources) if sources else 0.0
    features["entropy"] = binary_normalized_entropy(float(risk_probability))
    return features


def assessment_audit_features(
    assessment: object | Mapping[str, Any],
    *,
    source_validity: Mapping[str, float],
    reliable_source_threshold: float,
    sources: Sequence[str] = DEFAULT_SOURCES,
) -> dict[str, float]:
    """Audit features of one initial assessment under fitted source validity."""

    risk_probability = _validate_unit_interval(
        _read_field(assessment, "risk_probability"), "assessment.risk_probability"
    )
    confidence = _validate_unit_interval(
        _read_field(assessment, "confidence"), "assessment.confidence"
    )
    source_risks = _unit_interval_mapping(
        _read_field(assessment, "source_risk_probabilities", {}),
        "source_risk_probabilities",
    )
    source_reliability = _unit_interval_mapping(
        _read_field(assessment, "source_reliability", {}),
        "source_reliability",
    )
    reliable, unreliable, _pairs = validity_weighted_disagreement(
        source_risks,
        source_reliability,
        source_validity,
        reliable_source_threshold=reliable_source_threshold,
    )
    return build_audit_features(
        risk_probability=risk_probability,
        confidence=confidence,
        source_reliability=source_reliability,
        reliable_disagreement=reliable,
        unreliable_disagreement=unreliable,
        sources=sources,
    )


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Rank-based AUROC with average ranks for ties."""

    y = np.asarray(labels, dtype=float)
    s = np.asarray(scores, dtype=float)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC needs both classes")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_scores = s[order]
    index = 0
    while index < len(s):
        end = index
        while end + 1 < len(s) and sorted_scores[end + 1] == sorted_scores[index]:
            end += 1
        ranks[order[index : end + 1]] = (index + end) / 2.0 + 1.0
        index = end + 1
    rank_sum = float(ranks[y == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def fit_source_validity(
    labels: Sequence[int],
    source_probabilities: Mapping[str, Sequence[float | None]],
) -> dict[str, float]:
    """Normalized out-of-fold AUROC ``max(0, 2 * AUROC - 1)`` per source.

    ``source_probabilities[s][i]`` is source ``s``'s risk reading for case
    ``i``, or ``None`` when the source was unavailable; unavailable cases are
    left out of that source's AUROC.
    """

    validity: dict[str, float] = {}
    for source, values in source_probabilities.items():
        if len(values) != len(labels):
            raise ValueError(f"source {source!r} has a different case count")
        kept = [(int(y), float(p)) for y, p in zip(labels, values) if p is not None]
        auc = roc_auc([y for y, _ in kept], [p for _, p in kept])
        validity[source] = min(1.0, max(0.0, 2.0 * auc - 1.0))
    return validity


def fit_logistic_l2(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    l2: float,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, float]:
    """Newton fit of an L2-penalized logistic regression; the intercept is not penalized."""

    if l2 < 0:
        raise ValueError("l2 must be non-negative")
    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError("features must be a 2-D array aligned with targets")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("targets must be binary")
    design = np.hstack([np.ones((len(x), 1)), x])
    penalty = np.full(design.shape[1], float(l2))
    penalty[0] = 0.0
    weights = np.zeros(design.shape[1])
    for _ in range(max_iterations):
        logits = np.clip(design @ weights, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (probabilities - y) + penalty * weights
        curvature = probabilities * (1.0 - probabilities)
        hessian = design.T @ (design * curvature[:, None]) + np.diag(penalty)
        hessian += 1e-9 * np.eye(len(weights))
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if float(np.max(np.abs(step))) < tolerance:
            break
    return weights[1:].copy(), float(weights[0])


def select_risk_threshold(risks: Sequence[float], target_trigger_rate: float) -> float:
    """Smallest threshold that triggers about ``target_trigger_rate`` of the cases.

    The threshold is a quantile of development-set risks and needs no labels.
    """

    rate = _validate_unit_interval(target_trigger_rate, "target_trigger_rate")
    values = sorted((float(value) for value in risks), reverse=True)
    if not values:
        raise ValueError("risk threshold needs at least one development case")
    count = int(round(rate * len(values)))
    if count <= 0:
        return 1.0
    return values[min(count, len(values)) - 1]


@dataclass(frozen=True, slots=True)
class AuditEstimator:
    """Fitted audit ``g_phi`` with its source validity and risk threshold."""

    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    threshold: float
    source_validity: Mapping[str, float]
    sources: tuple[str, ...] = DEFAULT_SOURCES
    l2: float = 1.0
    reliable_source_threshold: float = RethinkPolicyConfig().reliable_source_threshold
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        width = len(self.feature_names)
        if tuple(self.feature_names) != audit_feature_names(self.sources):
            raise ValueError("feature_names do not match the audit sources")
        if not (len(self.mean) == len(self.scale) == len(self.coefficients) == width):
            raise ValueError("audit parameters have inconsistent widths")
        if any(value <= 0 or not math.isfinite(value) for value in self.scale):
            raise ValueError("audit feature scales must be positive")
        _validate_unit_interval(self.threshold, "audit.threshold")
        for source, value in self.source_validity.items():
            _validate_unit_interval(value, f"audit.source_validity[{source!r}]")

    def features(
        self,
        *,
        risk_probability: float,
        confidence: float,
        source_reliability: Mapping[str, float],
        reliable_disagreement: float,
        unreliable_disagreement: float,
    ) -> dict[str, float]:
        return build_audit_features(
            risk_probability=risk_probability,
            confidence=confidence,
            source_reliability=source_reliability,
            reliable_disagreement=reliable_disagreement,
            unreliable_disagreement=unreliable_disagreement,
            sources=self.sources,
        )

    def risk(self, features: Mapping[str, float]) -> float:
        """Estimated probability that the initial decision is wrong."""

        vector = np.array([float(features[name]) for name in self.feature_names])
        standardized = (vector - np.array(self.mean)) / np.array(self.scale)
        logit = float(standardized @ np.array(self.coefficients)) + self.intercept
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))

    def assessment_risk(self, assessment: object | Mapping[str, Any]) -> float:
        return self.risk(
            assessment_audit_features(
                assessment,
                source_validity=self.source_validity,
                reliable_source_threshold=self.reliable_source_threshold,
                sources=self.sources,
            )
        )

    def with_threshold(self, threshold: float) -> "AuditEstimator":
        return replace(self, threshold=float(threshold))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "model": "l2_logistic_regression",
            "feature_names": list(self.feature_names),
            "mean": list(self.mean),
            "scale": list(self.scale),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "threshold": self.threshold,
            "source_validity": dict(sorted(self.source_validity.items())),
            "sources": list(self.sources),
            "l2": self.l2,
            "reliable_source_threshold": self.reliable_source_threshold,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AuditEstimator":
        if data.get("schema_version") != AUDIT_SCHEMA_VERSION:
            raise ValueError("unsupported audit schema version")
        if data.get("model") != "l2_logistic_regression":
            raise ValueError("unsupported audit model")
        return cls(
            feature_names=tuple(data["feature_names"]),
            mean=tuple(float(value) for value in data["mean"]),
            scale=tuple(float(value) for value in data["scale"]),
            coefficients=tuple(float(value) for value in data["coefficients"]),
            intercept=float(data["intercept"]),
            threshold=float(data["threshold"]),
            source_validity={
                str(key): float(value) for key, value in data["source_validity"].items()
            },
            sources=tuple(data["sources"]),
            l2=float(data["l2"]),
            reliable_source_threshold=float(data["reliable_source_threshold"]),
            metadata=dict(data.get("metadata", {})),
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "AuditEstimator":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def sha256(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fit_audit(
    assessments: Sequence[object | Mapping[str, Any]],
    error_targets: Sequence[int],
    *,
    source_validity: Mapping[str, float],
    l2: float = 1.0,
    reliable_source_threshold: float = RethinkPolicyConfig().reliable_source_threshold,
    sources: Sequence[str] = DEFAULT_SOURCES,
    extra_assessments: Iterable[object | Mapping[str, Any]] = (),
    extra_targets: Iterable[int] = (),
    metadata: Mapping[str, Any] | None = None,
) -> AuditEstimator:
    """Fit the audit on out-of-fold assessments and optional degradations.

    ``error_targets[i]`` is 1 when the first-pass decision of case ``i`` was
    wrong.  ``extra_assessments`` are synthetic degradations whose targets are
    known by construction.  The returned estimator has threshold 1, which
    triggers nothing, until :func:`select_risk_threshold` fixes it on
    development data.
    """

    rows = list(assessments) + list(extra_assessments)
    targets = list(error_targets) + list(extra_targets)
    if len(rows) != len(targets):
        raise ValueError("each assessment needs exactly one target")
    names = audit_feature_names(sources)
    matrix = np.array(
        [
            [
                assessment_audit_features(
                    row,
                    source_validity=source_validity,
                    reliable_source_threshold=reliable_source_threshold,
                    sources=sources,
                )[name]
                for name in names
            ]
            for row in rows
        ],
        dtype=float,
    )
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale <= 1e-12] = 1.0
    coefficients, intercept = fit_logistic_l2(
        (matrix - mean) / scale, np.asarray(targets, dtype=float), l2=l2
    )
    return AuditEstimator(
        feature_names=names,
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        coefficients=tuple(float(value) for value in coefficients),
        intercept=intercept,
        threshold=1.0,
        source_validity=dict(source_validity),
        sources=tuple(sources),
        l2=float(l2),
        reliable_source_threshold=float(reliable_source_threshold),
        metadata=dict(metadata or {}),
    )


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "AuditEstimator",
    "DEFAULT_SOURCES",
    "assessment_audit_features",
    "audit_feature_names",
    "build_audit_features",
    "fit_audit",
    "fit_logistic_l2",
    "fit_source_validity",
    "roc_auc",
    "select_risk_threshold",
]
