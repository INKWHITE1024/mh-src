"""Contracts for query-grounded atomic evidence retrieval and selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from rethink_mh.textualization.canonical_slots import CORE_SLOTS

from .contracts import (
    EVIDENCE_ID_PATTERN,
    SEGMENT_ID_PATTERN,
    ContractValidationError,
)


RETRIEVAL_PROTOCOL_VERSION = "query-grounded-atomic"
MAX_QUERY_BUDGET = 4
DEFAULT_MAX_CANDIDATES = 8

RetrievalPurpose = Literal[
    "verify supporting detail",
    "verify contradictory detail",
    "resolve source disagreement",
    "inspect measurement reliability",
    "inspect temporal change",
]
RetrievalPattern = Literal[
    "unusual measurement",
    "change point",
    "quality boundary",
    "representative window",
    "cross-modal co-change",
]

RETRIEVAL_PURPOSES: tuple[RetrievalPurpose, ...] = (
    "verify supporting detail",
    "verify contradictory detail",
    "resolve source disagreement",
    "inspect measurement reliability",
    "inspect temporal change",
)
RETRIEVAL_PATTERNS: tuple[RetrievalPattern, ...] = (
    "unusual measurement",
    "change point",
    "quality boundary",
    "representative window",
    "cross-modal co-change",
)

_SLOT_BY_LABEL = {slot.label: slot for slot in CORE_SLOTS}


def _strict_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    path: str,
) -> None:
    missing = expected - set(value)
    extra = set(value) - expected
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if extra:
            details.append(f"unexpected {sorted(extra)}")
        raise ContractValidationError(
            f"{path} has invalid fields: {', '.join(details)}"
        )


def _plain_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or value != value.strip() or not value:
        raise ContractValidationError(
            f"{path} must be a non-empty string without outer whitespace"
        )
    if any(ord(character) < 32 for character in value):
        raise ContractValidationError(f"{path} must be one line")
    return value


def _bounded_integer(value: Any, path: str, *, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractValidationError(f"{path} must be an integer")
    if not 1 <= value <= upper:
        raise ContractValidationError(
            f"{path} must be between 1 and {upper}, inclusive"
        )
    return value


@dataclass(frozen=True, slots=True)
class EvidenceQuery:
    """A readable, dataset-portable request over the fixed core slots."""

    segment_id: str
    purpose: RetrievalPurpose
    target_slots: tuple[str, ...]
    pattern: RetrievalPattern
    budget: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.segment_id, str)
            or SEGMENT_ID_PATTERN.fullmatch(self.segment_id) is None
        ):
            raise ContractValidationError(
                "evidence_query.segment_id must be a valid S identifier"
            )
        if self.purpose not in RETRIEVAL_PURPOSES:
            raise ContractValidationError(
                f"evidence_query.purpose received {self.purpose!r}; it must be "
                f"one of {list(RETRIEVAL_PURPOSES)}"
            )
        if self.pattern not in RETRIEVAL_PATTERNS:
            raise ContractValidationError(
                f"evidence_query.pattern received {self.pattern!r}; it must be "
                f"one of {list(RETRIEVAL_PATTERNS)}"
            )
        if isinstance(self.target_slots, (str, bytes)) or not isinstance(
            self.target_slots, Sequence
        ):
            raise ContractValidationError(
                "evidence_query.target_slots must be an array"
            )
        slots = tuple(self.target_slots)
        if not 1 <= len(slots) <= 4:
            raise ContractValidationError(
                "evidence_query.target_slots must contain 1 to 4 slots"
            )
        if len(slots) != len(set(slots)):
            raise ContractValidationError(
                "evidence_query.target_slots cannot contain duplicates"
            )
        unknown = [slot for slot in slots if slot not in _SLOT_BY_LABEL]
        if unknown:
            raise ContractValidationError(
                "evidence_query.target_slots contains unknown fixed-slot labels: "
                f"{unknown}"
            )
        object.__setattr__(self, "target_slots", slots)
        object.__setattr__(
            self,
            "budget",
            _bounded_integer(
                self.budget,
                "evidence_query.budget",
                upper=MAX_QUERY_BUDGET,
            ),
        )

        modalities = {_SLOT_BY_LABEL[slot].modality for slot in slots}
        if (
            self.purpose == "resolve source disagreement"
            or self.pattern == "cross-modal co-change"
        ) and modalities != {"audio", "visual"}:
            raise ContractValidationError(
                "cross-modal queries require at least one audio and one visual slot"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceQuery":
        if not isinstance(value, Mapping):
            raise ContractValidationError("evidence_query must be an object")
        _strict_keys(
            value,
            frozenset(
                {
                    "segment_id",
                    "purpose",
                    "target_slots",
                    "pattern",
                    "budget",
                }
            ),
            "evidence_query",
        )
        raw_slots = value["target_slots"]
        if isinstance(raw_slots, (str, bytes)) or not isinstance(
            raw_slots, Sequence
        ):
            raise ContractValidationError(
                "evidence_query.target_slots must be an array"
            )
        return cls(
            segment_id=_plain_string(
                value["segment_id"], "evidence_query.segment_id"
            ),
            purpose=_plain_string(
                value["purpose"], "evidence_query.purpose"
            ),  # type: ignore[arg-type]
            target_slots=tuple(
                _plain_string(item, f"evidence_query.target_slots[{index}]")
                for index, item in enumerate(raw_slots)
            ),
            pattern=_plain_string(
                value["pattern"], "evidence_query.pattern"
            ),  # type: ignore[arg-type]
            budget=value["budget"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "purpose": self.purpose,
            "target_slots": list(self.target_slots),
            "pattern": self.pattern,
            "budget": self.budget,
        }


@dataclass(frozen=True, slots=True)
class AtomicSelection:
    """Atomic identifiers selected from one grounded candidate set."""

    selected_evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.selected_evidence_ids, (str, bytes)) or not isinstance(
            self.selected_evidence_ids, Sequence
        ):
            raise ContractValidationError(
                "atomic_selection.selected_evidence_ids must be an array"
            )
        identifiers = tuple(self.selected_evidence_ids)
        if not 1 <= len(identifiers) <= MAX_QUERY_BUDGET:
            raise ContractValidationError(
                "atomic_selection.selected_evidence_ids must contain 1 to "
                f"{MAX_QUERY_BUDGET} identifiers"
            )
        if len(identifiers) != len(set(identifiers)):
            raise ContractValidationError(
                "atomic_selection.selected_evidence_ids cannot contain duplicates"
            )
        for index, identifier in enumerate(identifiers):
            if (
                not isinstance(identifier, str)
                or EVIDENCE_ID_PATTERN.fullmatch(identifier) is None
            ):
                raise ContractValidationError(
                    "atomic_selection.selected_evidence_ids"
                    f"[{index}] must be a valid E identifier"
                )
        object.__setattr__(self, "selected_evidence_ids", identifiers)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AtomicSelection":
        if not isinstance(value, Mapping):
            raise ContractValidationError("atomic_selection must be an object")
        _strict_keys(
            value,
            frozenset({"selected_evidence_ids"}),
            "atomic_selection",
        )
        identifiers = value["selected_evidence_ids"]
        if isinstance(identifiers, (str, bytes)) or not isinstance(
            identifiers, Sequence
        ):
            raise ContractValidationError(
                "atomic_selection.selected_evidence_ids must be an array"
            )
        return cls(tuple(identifiers))

    def to_dict(self) -> dict[str, list[str]]:
        return {"selected_evidence_ids": list(self.selected_evidence_ids)}


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    """One low-detail atomic card plus auditable deterministic ranking data."""

    rank: int
    evidence_id: str
    segment_id: str
    start_sec: float
    end_sec: float
    cues: tuple[str, ...]
    target_slots_present: tuple[str, ...]
    quality_summary: str
    ranking_score: float
    score_components: tuple[tuple[str, float], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "evidence_id": self.evidence_id,
            "segment_id": self.segment_id,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "cues": list(self.cues),
            "target_slots_present": list(self.target_slots_present),
            "quality_summary": self.quality_summary,
            "ranking_score": round(self.ranking_score, 6),
            "score_components": {
                name: round(value, 6)
                for name, value in self.score_components
            },
        }


@dataclass(frozen=True, slots=True)
class CandidateSet:
    """Candidate cards grounded in one session fingerprint and one query."""

    query: EvidenceQuery
    candidates: tuple[CandidateEvidence, ...]
    text: str
    session_fingerprint: str

    @property
    def candidate_evidence_ids(self) -> tuple[str, ...]:
        return tuple(item.evidence_id for item in self.candidates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval_protocol_version": RETRIEVAL_PROTOCOL_VERSION,
            "query": self.query.to_dict(),
            "candidate_evidence_ids": list(self.candidate_evidence_ids),
            "candidates": [item.to_dict() for item in self.candidates],
            "session_fingerprint": self.session_fingerprint,
        }


__all__ = [
    "AtomicSelection",
    "CandidateEvidence",
    "CandidateSet",
    "EvidenceQuery",
    "RETRIEVAL_PATTERNS",
    "RETRIEVAL_PROTOCOL_VERSION",
    "RETRIEVAL_PURPOSES",
]
