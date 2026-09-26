"""Deterministic reliability-triggered rethinking policy."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

if TYPE_CHECKING:  # pragma: no cover
    from .audit import AuditEstimator


TriggerReason: TypeAlias = Literal[
    "high_predictive_entropy",
    "low_confidence",
    "reliable_source_disagreement",
    "low_mean_source_reliability",
    "low_min_source_reliability",
    "overlapping_support_and_contradiction",
    "coexisting_support_and_contradiction",
]

_REASON_HIGH_ENTROPY: TriggerReason = "high_predictive_entropy"
_REASON_LOW_CONFIDENCE: TriggerReason = "low_confidence"
_REASON_SOURCE_DISAGREEMENT: TriggerReason = "reliable_source_disagreement"
_REASON_LOW_MEAN_RELIABILITY: TriggerReason = "low_mean_source_reliability"
_REASON_LOW_MIN_RELIABILITY: TriggerReason = "low_min_source_reliability"
_REASON_OVERLAP: TriggerReason = "overlapping_support_and_contradiction"
_REASON_COEXISTENCE: TriggerReason = "coexisting_support_and_contradiction"

_MISSING = object()


@dataclass(frozen=True, slots=True)
class RethinkPolicyConfig:
    """Thresholds and retrieval budget for :class:`RethinkPolicy`.

    All thresholds use the closed interval ``[0, 1]``.  A source contributes to
    disagreement only when its reliability is at least
    ``reliable_source_threshold``; this prevents a broken source from creating a
    spurious cross-modal conflict signal.
    """

    entropy_threshold: float = 0.85
    confidence_threshold: float = 0.60
    reliable_source_threshold: float = 0.60
    source_disagreement_threshold: float = 0.30
    mean_source_reliability_threshold: float = 0.60
    min_source_reliability_threshold: float = 0.40
    trigger_on_evidence_overlap: bool = True
    trigger_on_evidence_coexistence: bool = True
    max_segments: int = 3

    def __post_init__(self) -> None:
        threshold_names = (
            "entropy_threshold",
            "confidence_threshold",
            "reliable_source_threshold",
            "source_disagreement_threshold",
            "mean_source_reliability_threshold",
            "min_source_reliability_threshold",
        )
        for name in threshold_names:
            _validate_unit_interval(getattr(self, name), f"config.{name}")
        if (
            isinstance(self.max_segments, bool)
            or not isinstance(self.max_segments, int)
            or self.max_segments <= 0
        ):
            raise ValueError("config.max_segments must be a positive integer")


@dataclass(frozen=True, slots=True)
class RethinkDecision:
    """Decision and audit metrics emitted by the trigger policy."""

    should_rethink: bool
    trigger_score: float
    reasons: tuple[TriggerReason, ...]
    selected_segment_ids: tuple[str, ...]
    metrics: dict[str, float | int | None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_rethink": self.should_rethink,
            "trigger_score": self.trigger_score,
            "reasons": list(self.reasons),
            "selected_segment_ids": list(self.selected_segment_ids),
            "metrics": dict(self.metrics),
        }


class RethinkPolicy:
    """Evaluate whether targeted evidence retrieval is warranted.

    ``assessment`` may be a mapping or any object exposing the documented
    attributes.  Optional source and segment fields default to empty values, so
    callers can use the policy even when an assessment has only session-level
    probability and confidence.

    With a fitted :class:`~rethink_mh.rethinking.audit.AuditEstimator`, the
    policy follows the paper: the audit estimates the error risk ``rho`` of the
    initial assessment, and the case is re-examined when ``rho`` reaches the
    audit threshold.  The fitted source validity then weights the reliable
    disagreement term.  Without an audit, the threshold rules below act as the
    heuristic baseline.  The rule reasons are reported in both modes so that
    triggered cases can be analysed by reason.
    """

    def __init__(
        self,
        config: RethinkPolicyConfig | None = None,
        *,
        audit: AuditEstimator | None = None,
    ) -> None:
        self.config = config or RethinkPolicyConfig()
        self.audit = audit

    def evaluate(
        self,
        assessment: object | Mapping[str, Any],
        available_segment_ids: Iterable[str],
    ) -> RethinkDecision:
        """Return a deterministic decision without consulting task labels."""

        risk_probability = _unit_interval_field(assessment, "risk_probability")
        confidence = _unit_interval_field(assessment, "confidence")
        source_risks = _unit_interval_mapping(
            _read_field(assessment, "source_risk_probabilities", {}),
            "source_risk_probabilities",
        )
        source_reliability = _unit_interval_mapping(
            _read_field(assessment, "source_reliability", {}),
            "source_reliability",
        )
        source_validity = _unit_interval_mapping(
            _read_field(assessment, "source_validity", {}),
            "source_validity",
        )
        if not source_validity and self.audit is not None:
            source_validity = dict(self.audit.source_validity)

        supporting = _ordered_segment_ids(
            _read_field(assessment, "supporting_segment_ids", ())
        )
        contradictory = _ordered_segment_ids(
            _read_field(assessment, "contradictory_segment_ids", ())
        )
        uncertain = _ordered_segment_ids(
            _read_field(assessment, "uncertain_segment_ids", ())
        )
        requested = _ordered_segment_ids(
            _read_field(assessment, "requested_segment_ids", ())
        )
        available = _ordered_segment_ids(available_segment_ids, sort_unordered=True)

        entropy = _binary_normalized_entropy(risk_probability)
        reliable_risks = [
            source_risks[source]
            for source in sorted(source_risks)
            if source_reliability.get(source, -1.0)
            >= self.config.reliable_source_threshold
        ]
        disagreement = (
            max(reliable_risks) - min(reliable_risks)
            if len(reliable_risks) >= 2
            else 0.0
        )
        (
            reliable_disagreement,
            unreliable_disagreement,
            disagreement_pair_count,
        ) = validity_weighted_disagreement(
            source_risks,
            source_reliability,
            source_validity,
            reliable_source_threshold=self.config.reliable_source_threshold,
        )

        reliability_values = list(source_reliability.values())
        mean_reliability = (
            math.fsum(reliability_values) / len(reliability_values)
            if reliability_values
            else None
        )
        min_reliability = min(reliability_values) if reliability_values else None

        contradictory_set = set(contradictory)
        overlap = tuple(item for item in supporting if item in contradictory_set)
        evidence_union = set(supporting) | contradictory_set
        overlap_ratio = len(overlap) / len(evidence_union) if evidence_union else 0.0
        evidence_coexists = bool(supporting and contradictory)

        reasons: list[TriggerReason] = []
        if entropy >= self.config.entropy_threshold:
            reasons.append(_REASON_HIGH_ENTROPY)
        if confidence <= self.config.confidence_threshold:
            reasons.append(_REASON_LOW_CONFIDENCE)
        if (
            len(reliable_risks) >= 2
            and disagreement >= self.config.source_disagreement_threshold
        ):
            reasons.append(_REASON_SOURCE_DISAGREEMENT)
        if (
            mean_reliability is not None
            and mean_reliability <= self.config.mean_source_reliability_threshold
        ):
            reasons.append(_REASON_LOW_MEAN_RELIABILITY)
        if (
            min_reliability is not None
            and min_reliability <= self.config.min_source_reliability_threshold
        ):
            reasons.append(_REASON_LOW_MIN_RELIABILITY)
        if overlap and self.config.trigger_on_evidence_overlap:
            reasons.append(_REASON_OVERLAP)
        if evidence_coexists and self.config.trigger_on_evidence_coexistence:
            reasons.append(_REASON_COEXISTENCE)

        reliability_mean_deficit = (
            1.0 - mean_reliability if mean_reliability is not None else 0.0
        )
        reliability_min_deficit = (
            1.0 - min_reliability if min_reliability is not None else 0.0
        )
        conflict_signal = 1.0 if evidence_coexists else 0.0
        trigger_score = max(
            entropy,
            1.0 - confidence,
            disagreement,
            reliability_mean_deficit,
            reliability_min_deficit,
            conflict_signal,
        )
        trigger_score = min(1.0, max(0.0, trigger_score))

        should_rethink = bool(reasons)
        audit_metrics: dict[str, float | int | None] = {}
        if self.audit is not None:
            features = self.audit.features(
                risk_probability=risk_probability,
                confidence=confidence,
                source_reliability=source_reliability,
                reliable_disagreement=reliable_disagreement,
                unreliable_disagreement=unreliable_disagreement,
            )
            audit_risk = self.audit.risk(features)
            should_rethink = audit_risk >= self.audit.threshold
            trigger_score = audit_risk
            audit_metrics = {
                "audit_risk": audit_risk,
                "audit_threshold": self.audit.threshold,
                **{f"audit_feature_{name}": value for name, value in features.items()},
            }
        selected = self._select_segments(
            should_rethink=should_rethink,
            available=available,
            requested=requested,
            uncertain=uncertain,
            overlap=overlap,
            supporting=supporting,
            contradictory=contradictory,
        )
        metrics: dict[str, float | int | None] = {
            "risk_probability": risk_probability,
            "binary_normalized_entropy": entropy,
            "confidence": confidence,
            "confidence_deficit": 1.0 - confidence,
            "source_count": len(source_risks),
            "reliable_source_count": len(reliable_risks),
            "reliable_source_risk_disagreement": disagreement,
            "weighted_reliable_disagreement": reliable_disagreement,
            "weighted_unreliable_disagreement": unreliable_disagreement,
            "source_disagreement_pair_count": disagreement_pair_count,
            "source_validity_count": len(source_validity),
            "source_reliability_count": len(reliability_values),
            "mean_source_reliability": mean_reliability,
            "min_source_reliability": min_reliability,
            "supporting_segment_count": len(supporting),
            "contradictory_segment_count": len(contradictory),
            "support_contradiction_overlap_count": len(overlap),
            "support_contradiction_overlap_ratio": overlap_ratio,
            "support_contradiction_coexistence": int(evidence_coexists),
            **audit_metrics,
        }
        return RethinkDecision(
            should_rethink=should_rethink,
            trigger_score=trigger_score,
            reasons=tuple(reasons),
            selected_segment_ids=selected,
            metrics=metrics,
        )

    def _select_segments(
        self,
        *,
        should_rethink: bool,
        available: tuple[str, ...],
        requested: tuple[str, ...],
        uncertain: tuple[str, ...],
        overlap: tuple[str, ...],
        supporting: tuple[str, ...],
        contradictory: tuple[str, ...],
    ) -> tuple[str, ...]:
        if not should_rethink:
            return ()

        available_set = set(available)
        selected: list[str] = []
        seen: set[str] = set()
        priority_groups = (
            requested,
            uncertain,
            overlap,
            supporting,
            contradictory,
        )
        for group in priority_groups:
            for segment_id in group:
                if segment_id not in available_set or segment_id in seen:
                    continue
                selected.append(segment_id)
                seen.add(segment_id)
                if len(selected) >= self.config.max_segments:
                    return tuple(selected)

        if not selected and available:
            # Natural sorting makes the fallback stable for both lists and sets,
            # and orders S2 before S10 when identifiers are not zero-padded.
            selected.extend(
                sorted(available, key=_natural_sort_key)[: self.config.max_segments]
            )
        return tuple(selected)


def validity_weighted_disagreement(
    source_risks: Mapping[str, float],
    source_reliability: Mapping[str, float],
    source_validity: Mapping[str, float],
    *,
    reliable_source_threshold: float,
) -> tuple[float, float, int]:
    """Return ``(D_rel, D_unrel, pair_count)`` over all source pairs.

    ``D_rel`` weights each pairwise risk gap by both reliabilities and both
    validities, and ``D_unrel`` counts gaps where at least one source exceeds
    ``reliable_source_threshold``, weighted by the reliability deficit of the
    weaker source.  A source without a fitted validity has weight 1.
    """

    source_names = sorted(set(source_risks) & set(source_reliability))
    reliable = 0.0
    unreliable = 0.0
    pairs = 0
    for left_index, left_source in enumerate(source_names):
        for right_source in source_names[left_index + 1 :]:
            distance = abs(source_risks[left_source] - source_risks[right_source])
            left_reliability = source_reliability[left_source]
            right_reliability = source_reliability[right_source]
            validity_weight = source_validity.get(
                left_source, 1.0
            ) * source_validity.get(right_source, 1.0)
            reliable += left_reliability * right_reliability * validity_weight * distance
            if max(left_reliability, right_reliability) > reliable_source_threshold:
                unreliable += (1.0 - min(left_reliability, right_reliability)) * distance
            pairs += 1
    return reliable, unreliable, pairs


def binary_normalized_entropy(probability: float) -> float:
    """Binary entropy in bits, which lies in ``[0, 1]``."""

    return _binary_normalized_entropy(probability)


def _read_field(
    assessment: object | Mapping[str, Any],
    name: str,
    default: object = _MISSING,
) -> Any:
    if isinstance(assessment, Mapping):
        value = assessment.get(name, _MISSING)
    else:
        value = getattr(assessment, name, _MISSING)
    if value is _MISSING:
        if default is _MISSING:
            raise ValueError(f"assessment must provide {name!r}")
        return default
    return value


def _unit_interval_field(assessment: object | Mapping[str, Any], name: str) -> float:
    return _validate_unit_interval(_read_field(assessment, name), f"assessment.{name}")


def _unit_interval_mapping(value: Any, name: str) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"assessment.{name} must be a mapping")
    result: dict[str, float] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).strip()
        if not key:
            raise ValueError(f"assessment.{name} contains an empty source identifier")
        if key in result:
            raise ValueError(f"assessment.{name} contains duplicate source {key!r}")
        result[key] = _validate_unit_interval(raw_value, f"assessment.{name}[{key!r}]")
    return result


def _validate_unit_interval(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number in [0, 1]")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number in [0, 1]") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value!r}")
    return number


def _binary_normalized_entropy(probability: float) -> float:
    if probability in (0.0, 1.0):
        return 0.0
    complement = 1.0 - probability
    return -(probability * math.log2(probability) + complement * math.log2(complement))


def _ordered_segment_ids(
    value: Any, *, sort_unordered: bool = False
) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items: Iterable[Any] = (value,)
    elif isinstance(value, (set, frozenset)):
        items = sorted(value, key=lambda item: _natural_sort_key(str(item)))
    elif isinstance(value, Sequence) or isinstance(value, Iterable):
        items = value
    else:
        raise TypeError("segment identifiers must be an iterable of strings")

    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            raise TypeError("segment identifiers must be strings")
        segment_id = item.strip()
        if not segment_id:
            raise ValueError("segment identifiers cannot be empty")
        if segment_id not in seen:
            seen.add(segment_id)
            result.append(segment_id)
    if sort_unordered and not isinstance(value, Sequence):
        result.sort(key=_natural_sort_key)
    return tuple(result)


def _natural_sort_key(value: str) -> tuple[tuple[int, int | str], ...]:
    parts: list[tuple[int, int | str]] = []
    for part in re.split(r"(\d+)", value.casefold()):
        if part.isdigit():
            parts.append((1, int(part)))
        elif part:
            parts.append((0, part))
    return tuple(parts)


__all__ = [
    "RethinkDecision",
    "RethinkPolicy",
    "RethinkPolicyConfig",
    "TriggerReason",
    "binary_normalized_entropy",
    "validity_weighted_disagreement",
]
