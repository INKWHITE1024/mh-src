"""Strict, label-free contracts for the two RETHINK-MH model passes."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal


SourceStatus = Literal["available", "unavailable"]
RevisionStatus = Literal["preserved", "revised", "unresolved"]
RevisionAction = Literal["preserve", "revise", "refer"]
RevisionDirection = Literal["unchanged", "increase", "decrease", "unresolved"]

SEGMENT_ID_PATTERN = re.compile(r"^S[0-9]{3,6}$")
EVIDENCE_ID_PATTERN = re.compile(r"^E[0-9]{3,6}$")
TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

CONTROLLED_CHANGE_SUMMARIES = frozenset(
    {
        "risk_increased_after_review",
        "risk_decreased_after_review",
        "risk_unchanged_after_review",
        "insufficient_reliable_detail",
    }
)

_TASK_KEYS = frozenset(
    {
        "task_id",
        "label_definition",
        "negative_label_name",
        "positive_label_name",
    }
)
_INITIAL_KEYS = frozenset(
    {
        "risk_probability",
        "confidence",
        "audio_source",
        "visual_source",
        "supporting_segment_ids",
        "contradictory_segment_ids",
        "uncertain_segment_ids",
        "requested_segment_ids",
    }
)
_REVISION_KEYS = frozenset(
    {
        "revision_status",
        "revised_risk_probability",
        "revised_confidence",
        "cited_segment_ids",
        "cited_evidence_ids",
        "preserved_evidence_ids",
        "newly_considered_evidence_ids",
        "rejected_evidence_ids",
        "residual_conflict_segment_ids",
        "change_summary",
    }
)
_ACTION_DECISION_KEYS = frozenset({"action", "direction"})


class ContractValidationError(ValueError):
    """Raised when model output does not exactly satisfy its contract."""


def _strict_keys(data: Mapping[str, Any], expected: frozenset[str], path: str) -> None:
    keys = set(data)
    missing = expected - keys
    extra = keys - expected
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if extra:
            details.append(f"unexpected {sorted(extra)}")
        raise ContractValidationError(f"{path} has invalid fields: {', '.join(details)}")


def _probability(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{path} must be a number from 0 to 1")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ContractValidationError(f"{path} must be a finite number from 0 to 1")
    return parsed


def _single_line_text(value: Any, path: str, *, max_length: int) -> str:
    if not isinstance(value, str):
        raise ContractValidationError(f"{path} must be a string")
    if value != value.strip() or not value:
        raise ContractValidationError(f"{path} must be non-empty without outer whitespace")
    if len(value) > max_length:
        raise ContractValidationError(f"{path} cannot exceed {max_length} characters")
    if any(ord(character) < 32 for character in value):
        raise ContractValidationError(f"{path} must be one line without control characters")
    return value


def _identifier_list(
    value: Any,
    path: str,
    pattern: re.Pattern[str],
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ContractValidationError(f"{path} must be an array of evidence identifiers")
    output: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, str) or pattern.fullmatch(item) is None:
            raise ContractValidationError(
                f"{path}[{index}] must match {pattern.pattern!r}"
            )
        if item not in seen:
            output.append(item)
            seen.add(item)
    return tuple(output)


def _limited_identifier_list(
    value: Any,
    path: str,
    pattern: re.Pattern[str],
    *,
    max_items: int,
) -> tuple[str, ...]:
    output = _identifier_list(value, path, pattern)
    if len(output) > max_items:
        raise ContractValidationError(
            f"{path} cannot contain more than {max_items} unique identifiers"
        )
    return output


@dataclass(frozen=True)
class TaskSpec:
    """A reusable task definition that never stores a sample's ground truth."""

    task_id: str
    label_definition: str
    negative_label_name: str
    positive_label_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or TASK_ID_PATTERN.fullmatch(self.task_id) is None:
            raise ContractValidationError(
                "task_id must start with an alphanumeric character and contain only "
                "letters, digits, period, underscore, or hyphen"
            )
        _single_line_text(self.label_definition, "label_definition", max_length=1000)
        _single_line_text(self.negative_label_name, "negative_label_name", max_length=64)
        _single_line_text(self.positive_label_name, "positive_label_name", max_length=64)
        if self.negative_label_name.casefold() == self.positive_label_name.casefold():
            raise ContractValidationError("negative and positive label names must differ")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskSpec":
        if not isinstance(data, Mapping):
            raise ContractValidationError("task must be an object")
        _strict_keys(data, _TASK_KEYS, "task")
        return cls(
            task_id=data["task_id"],
            label_definition=data["label_definition"],
            negative_label_name=data["negative_label_name"],
            positive_label_name=data["positive_label_name"],
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "task_id": self.task_id,
            "label_definition": self.label_definition,
            "negative_label_name": self.negative_label_name,
            "positive_label_name": self.positive_label_name,
        }


@dataclass(frozen=True)
class SourceAssessment:
    """Risk and operational reliability for one measurement source."""

    status: SourceStatus
    risk_probability: float | None = None
    reliability: float | None = None

    def __post_init__(self) -> None:
        if self.status not in {"available", "unavailable"}:
            raise ContractValidationError("source status must be available or unavailable")
        if self.status == "available":
            object.__setattr__(
                self,
                "risk_probability",
                _probability(self.risk_probability, "source.risk_probability"),
            )
            object.__setattr__(
                self,
                "reliability",
                _probability(self.reliability, "source.reliability"),
            )
        elif self.risk_probability is not None or self.reliability is not None:
            raise ContractValidationError(
                "an unavailable source cannot contain risk_probability or reliability"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, path: str = "source") -> "SourceAssessment":
        if not isinstance(data, Mapping):
            raise ContractValidationError(f"{path} must be an object")
        status = data.get("status")
        if status == "available":
            _strict_keys(
                data,
                frozenset({"status", "risk_probability", "reliability"}),
                path,
            )
            return cls(
                status="available",
                risk_probability=_probability(
                    data["risk_probability"], f"{path}.risk_probability"
                ),
                reliability=_probability(data["reliability"], f"{path}.reliability"),
            )
        if status == "unavailable":
            _strict_keys(data, frozenset({"status"}), path)
            return cls(status="unavailable")
        raise ContractValidationError(f"{path}.status must be available or unavailable")

    def to_dict(self) -> dict[str, str | float]:
        if self.status == "unavailable":
            return {"status": "unavailable"}
        assert self.risk_probability is not None
        assert self.reliability is not None
        return {
            "status": "available",
            "risk_probability": self.risk_probability,
            "reliability": self.reliability,
        }


@dataclass(frozen=True)
class InitialAssessment:
    """Strict first-pass model output over segment-level evidence."""

    risk_probability: float
    confidence: float
    audio_source: SourceAssessment
    visual_source: SourceAssessment
    supporting_segment_ids: tuple[str, ...] = ()
    contradictory_segment_ids: tuple[str, ...] = ()
    uncertain_segment_ids: tuple[str, ...] = ()
    requested_segment_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "risk_probability", _probability(self.risk_probability, "risk_probability")
        )
        object.__setattr__(self, "confidence", _probability(self.confidence, "confidence"))
        if not isinstance(self.audio_source, SourceAssessment):
            raise ContractValidationError("audio_source must be a SourceAssessment")
        if not isinstance(self.visual_source, SourceAssessment):
            raise ContractValidationError("visual_source must be a SourceAssessment")
        for field_name in (
            "supporting_segment_ids",
            "contradictory_segment_ids",
            "uncertain_segment_ids",
            "requested_segment_ids",
        ):
            object.__setattr__(
                self,
                field_name,
                _limited_identifier_list(
                    getattr(self, field_name),
                    field_name,
                    SEGMENT_ID_PATTERN,
                    max_items=4,
                ),
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InitialAssessment":
        if not isinstance(data, Mapping):
            raise ContractValidationError("initial assessment must be an object")
        _strict_keys(data, _INITIAL_KEYS, "initial assessment")
        return cls(
            risk_probability=_probability(data["risk_probability"], "risk_probability"),
            confidence=_probability(data["confidence"], "confidence"),
            audio_source=SourceAssessment.from_dict(data["audio_source"], path="audio_source"),
            visual_source=SourceAssessment.from_dict(
                data["visual_source"], path="visual_source"
            ),
            supporting_segment_ids=_limited_identifier_list(
                data["supporting_segment_ids"],
                "supporting_segment_ids",
                SEGMENT_ID_PATTERN,
                max_items=4,
            ),
            contradictory_segment_ids=_limited_identifier_list(
                data["contradictory_segment_ids"],
                "contradictory_segment_ids",
                SEGMENT_ID_PATTERN,
                max_items=4,
            ),
            uncertain_segment_ids=_limited_identifier_list(
                data["uncertain_segment_ids"],
                "uncertain_segment_ids",
                SEGMENT_ID_PATTERN,
                max_items=4,
            ),
            requested_segment_ids=_limited_identifier_list(
                data["requested_segment_ids"],
                "requested_segment_ids",
                SEGMENT_ID_PATTERN,
                max_items=4,
            ),
        )

    @property
    def source_risk_probabilities(self) -> dict[str, float]:
        """Per-source values for trigger duck typing, omitting unavailable sources."""

        output: dict[str, float] = {}
        if self.audio_source.risk_probability is not None:
            output["audio"] = self.audio_source.risk_probability
        if self.visual_source.risk_probability is not None:
            output["visual"] = self.visual_source.risk_probability
        return output

    @property
    def source_reliability(self) -> dict[str, float]:
        """Per-source operational reliability, omitting unavailable sources."""

        output: dict[str, float] = {}
        if self.audio_source.reliability is not None:
            output["audio"] = self.audio_source.reliability
        if self.visual_source.reliability is not None:
            output["visual"] = self.visual_source.reliability
        return output

    @property
    def source_reliabilities(self) -> dict[str, float]:
        """Plural alias for callers that use that naming convention."""

        return self.source_reliability

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_probability": self.risk_probability,
            "confidence": self.confidence,
            "audio_source": self.audio_source.to_dict(),
            "visual_source": self.visual_source.to_dict(),
            "supporting_segment_ids": list(self.supporting_segment_ids),
            "contradictory_segment_ids": list(self.contradictory_segment_ids),
            "uncertain_segment_ids": list(self.uncertain_segment_ids),
            "requested_segment_ids": list(self.requested_segment_ids),
        }


@dataclass(frozen=True)
class RevisionActionDecision:
    """Short, readable action selected before generating a revision payload."""

    action: RevisionAction
    direction: RevisionDirection

    def __post_init__(self) -> None:
        allowed = {
            "preserve": {"unchanged"},
            "revise": {"increase", "decrease"},
            "refer": {"unresolved"},
        }
        if self.action not in allowed:
            raise ContractValidationError(
                "action must be preserve, revise, or refer"
            )
        if self.direction not in allowed[self.action]:
            raise ContractValidationError(
                f"direction {self.direction!r} is invalid for action {self.action!r}"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RevisionActionDecision":
        if not isinstance(data, Mapping):
            raise ContractValidationError("revision action decision must be an object")
        _strict_keys(data, _ACTION_DECISION_KEYS, "revision action decision")
        return cls(action=data["action"], direction=data["direction"])

    def to_dict(self) -> dict[str, str]:
        return {"action": self.action, "direction": self.direction}


def _change_summary(value: Any) -> str:
    text = _single_line_text(value, "change_summary", max_length=160)
    if text in CONTROLLED_CHANGE_SUMMARIES:
        return text
    if len(text) < 8:
        raise ContractValidationError(
            "a custom change_summary must be an 8 to 160 character English clause"
        )
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ,.;:'()/-]*", text) is None:
        raise ContractValidationError(
            "a custom change_summary may contain only plain English text and punctuation"
        )
    return text


@dataclass(frozen=True)
class RevisionAssessment:
    """Strict second-pass output after targeted atomic evidence review."""

    revision_status: RevisionStatus
    revised_risk_probability: float
    revised_confidence: float
    cited_segment_ids: tuple[str, ...]
    cited_evidence_ids: tuple[str, ...]
    preserved_evidence_ids: tuple[str, ...]
    newly_considered_evidence_ids: tuple[str, ...]
    rejected_evidence_ids: tuple[str, ...]
    residual_conflict_segment_ids: tuple[str, ...]
    change_summary: str

    def __post_init__(self) -> None:
        if self.revision_status not in {"preserved", "revised", "unresolved"}:
            raise ContractValidationError(
                "revision_status must be preserved, revised, or unresolved"
            )
        object.__setattr__(
            self,
            "revised_risk_probability",
            _probability(self.revised_risk_probability, "revised_risk_probability"),
        )
        object.__setattr__(
            self,
            "revised_confidence",
            _probability(self.revised_confidence, "revised_confidence"),
        )
        object.__setattr__(
            self,
            "cited_segment_ids",
            _identifier_list(
                self.cited_segment_ids, "cited_segment_ids", SEGMENT_ID_PATTERN
            ),
        )
        object.__setattr__(
            self,
            "cited_evidence_ids",
            _identifier_list(
                self.cited_evidence_ids, "cited_evidence_ids", EVIDENCE_ID_PATTERN
            ),
        )
        for field_name, pattern in (
            ("preserved_evidence_ids", EVIDENCE_ID_PATTERN),
            ("newly_considered_evidence_ids", EVIDENCE_ID_PATTERN),
            ("rejected_evidence_ids", EVIDENCE_ID_PATTERN),
            ("residual_conflict_segment_ids", SEGMENT_ID_PATTERN),
        ):
            object.__setattr__(
                self,
                field_name,
                _identifier_list(getattr(self, field_name), field_name, pattern),
            )
        if not self.cited_segment_ids and not self.cited_evidence_ids:
            raise ContractValidationError("revision must cite at least one SEGMENT or EVIDENCE ID")
        cited = set(self.cited_evidence_ids)
        preserved = set(self.preserved_evidence_ids)
        rejected = set(self.rejected_evidence_ids)
        overlap = preserved & rejected
        if overlap:
            raise ContractValidationError(
                "preserved_evidence_ids and rejected_evidence_ids must be disjoint"
            )
        uncited_classified = (preserved | rejected) - cited
        if uncited_classified:
            raise ContractValidationError(
                "preserved or rejected evidence must also appear in cited_evidence_ids"
            )
        object.__setattr__(self, "change_summary", _change_summary(self.change_summary))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RevisionAssessment":
        if not isinstance(data, Mapping):
            raise ContractValidationError("revision assessment must be an object")
        _strict_keys(data, _REVISION_KEYS, "revision assessment")
        return cls(
            revision_status=data["revision_status"],
            revised_risk_probability=_probability(
                data["revised_risk_probability"], "revised_risk_probability"
            ),
            revised_confidence=_probability(
                data["revised_confidence"], "revised_confidence"
            ),
            cited_segment_ids=_identifier_list(
                data["cited_segment_ids"], "cited_segment_ids", SEGMENT_ID_PATTERN
            ),
            cited_evidence_ids=_identifier_list(
                data["cited_evidence_ids"], "cited_evidence_ids", EVIDENCE_ID_PATTERN
            ),
            preserved_evidence_ids=_identifier_list(
                data["preserved_evidence_ids"],
                "preserved_evidence_ids",
                EVIDENCE_ID_PATTERN,
            ),
            newly_considered_evidence_ids=_identifier_list(
                data["newly_considered_evidence_ids"],
                "newly_considered_evidence_ids",
                EVIDENCE_ID_PATTERN,
            ),
            rejected_evidence_ids=_identifier_list(
                data["rejected_evidence_ids"],
                "rejected_evidence_ids",
                EVIDENCE_ID_PATTERN,
            ),
            residual_conflict_segment_ids=_identifier_list(
                data["residual_conflict_segment_ids"],
                "residual_conflict_segment_ids",
                SEGMENT_ID_PATTERN,
            ),
            change_summary=_change_summary(data["change_summary"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision_status": self.revision_status,
            "revised_risk_probability": self.revised_risk_probability,
            "revised_confidence": self.revised_confidence,
            "cited_segment_ids": list(self.cited_segment_ids),
            "cited_evidence_ids": list(self.cited_evidence_ids),
            "preserved_evidence_ids": list(self.preserved_evidence_ids),
            "newly_considered_evidence_ids": list(
                self.newly_considered_evidence_ids
            ),
            "rejected_evidence_ids": list(self.rejected_evidence_ids),
            "residual_conflict_segment_ids": list(
                self.residual_conflict_segment_ids
            ),
            "change_summary": self.change_summary,
        }
