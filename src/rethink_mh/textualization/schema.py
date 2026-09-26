"""Core data schema for raw observations, observations, and evidence units."""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


SCHEMA_VERSION = "1.0.0"
SEMANTIC_LIMITS = (
    "observable_behavior_only",
    "no_emotion_label_inferred",
    "no_clinical_conclusion",
)

RepresentationLevel = Literal[
    "expert_feature",
    "geometric_feature",
    "aggregated_codebook",
    "deep_embedding",
    "latent_prototype",
]
SemanticReliability = Literal["direct", "derived", "weak", "latent"]
AvailabilityStatus = Literal["present", "partial", "unavailable"]
ReferenceCategory = Literal[
    "very_low", "low", "typical", "high", "very_high", "unreferenced"
]


def _finite(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 6)
    return value


@dataclass(frozen=True)
class RawObservation:
    name: str
    value: float
    unit: str
    semantic_reliability: SemanticReliability

    def validate(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.name):
            raise ValueError(f"Invalid observation name: {self.name!r}")
        if not _finite(self.value):
            raise ValueError(f"Observation {self.name} must be finite")
        if self.semantic_reliability not in {"direct", "derived", "weak", "latent"}:
            raise ValueError(f"Invalid semantic reliability: {self.semantic_reliability}")


@dataclass(frozen=True)
class RawEvidenceUnit:
    evidence_id: str
    dataset: str
    session_id: str
    modality: Literal["audio", "visual"]
    source_id: str
    extractor: str
    extractor_version: str
    representation_level: RepresentationLevel
    source_files: tuple[str, ...]
    start_sec: float
    end_sec: float
    availability_status: AvailabilityStatus
    missing_reasons: tuple[str, ...] = ()
    quality: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    observations: tuple[RawObservation, ...] = ()
    semantic_limits: tuple[str, ...] = SEMANTIC_LIMITS

    @property
    def reference_prefix(self) -> str:
        return self.source_id

    def validate(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+", self.evidence_id):
            raise ValueError(f"Invalid evidence_id: {self.evidence_id!r}")
        if self.modality not in {"audio", "visual"}:
            raise ValueError(f"Invalid modality: {self.modality}")
        if self.representation_level not in {
            "expert_feature", "geometric_feature", "aggregated_codebook", "deep_embedding", "latent_prototype"
        }:
            raise ValueError(f"Invalid representation level: {self.representation_level}")
        if self.availability_status not in {"present", "partial", "unavailable"}:
            raise ValueError(f"Invalid availability status: {self.availability_status}")
        if self.start_sec < 0:
            raise ValueError("Evidence start time cannot be negative")
        if self.end_sec <= self.start_sec:
            raise ValueError("Evidence time range must have positive duration")
        if self.availability_status == "unavailable" and self.observations:
            raise ValueError("Unavailable evidence cannot contain observations")
        if self.availability_status == "unavailable" and not self.missing_reasons:
            raise ValueError("Unavailable evidence must state a missing reason")
        if any("/" in name or "\\" in name for name in self.source_files):
            raise ValueError("source_files must contain basenames, not paths")
        for observation in self.observations:
            observation.validate()
        names = [observation.name for observation in self.observations]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate observation names in {self.evidence_id}")


@dataclass(frozen=True)
class Observation:
    name: str
    value: float
    unit: str
    category: ReferenceCategory
    percentile: float | None
    robust_z: float | None
    subject_robust_z: float | None
    semantic_reliability: SemanticReliability
    reference_scope: str

    def validate(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.name):
            raise ValueError(f"Invalid observation name: {self.name!r}")
        if not _finite(self.value):
            raise ValueError(f"Observation {self.name} must be finite")
        if self.category not in {"very_low", "low", "typical", "high", "very_high", "unreferenced"}:
            raise ValueError(f"Invalid reference category: {self.category}")
        if self.semantic_reliability not in {"direct", "derived", "weak", "latent"}:
            raise ValueError(f"Invalid semantic reliability: {self.semantic_reliability}")
        if not self.reference_scope:
            raise ValueError("reference_scope cannot be empty")
        if self.percentile is not None and not 0.0 <= self.percentile <= 100.0:
            raise ValueError(f"Percentile out of bounds: {self.percentile}")
        for name, value in (("robust_z", self.robust_z), ("subject_robust_z", self.subject_robust_z)):
            if value is not None and not _finite(value):
                raise ValueError(f"{name} must be finite or null")


@dataclass(frozen=True)
class EvidenceUnit:
    evidence_id: str
    dataset: str
    session_id: str
    modality: Literal["audio", "visual"]
    source: dict[str, Any]
    time_range: dict[str, float]
    availability: dict[str, Any]
    quality: dict[str, Any]
    context: dict[str, Any]
    observations: tuple[Observation, ...]
    semantic_limits: tuple[str, ...] = SEMANTIC_LIMITS
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema version: {self.schema_version}")
        status = self.availability.get("status")
        if status not in {"present", "partial", "unavailable"}:
            raise ValueError(f"Invalid availability status: {status}")
        if status == "unavailable" and self.observations:
            raise ValueError("Unavailable evidence cannot contain observations")
        if status == "unavailable" and not self.availability.get("missing_reasons"):
            raise ValueError("Unavailable evidence must state a missing reason")
        if float(self.time_range["start_sec"]) < 0:
            raise ValueError("Evidence start time cannot be negative")
        if float(self.time_range["end_sec"]) <= float(self.time_range["start_sec"]):
            raise ValueError("Evidence time range must have positive duration")
        source_files = self.source.get("source_files", [])
        if any("/" in name or "\\" in name for name in source_files):
            raise ValueError("source_files must contain basenames, not paths")
        for observation in self.observations:
            observation.validate()
        names = [observation.name for observation in self.observations]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate observation names in {self.evidence_id}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return _json_safe(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
