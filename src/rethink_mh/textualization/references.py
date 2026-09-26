"""Fit source-specific robust reference statistics and categorize observations against them."""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from .schema import EvidenceUnit, Observation, RawEvidenceUnit


REFERENCE_SCHEMA_VERSION = "1.0.0"
DEFAULT_QUANTILE_PROBABILITIES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


REFERENCE_IRRELEVANT_CONFIG_KEYS = {
    "evidence_protocol_default",
    "max_rendered_observations_per_unit",
    "protocol_v2",
    "protocol_v3",
}


def reference_configuration(config: dict[str, Any]) -> dict[str, Any]:
    """Keep only settings that can change raw observations or reference values."""
    return {
        key: value
        for key, value in config.items()
        if key not in REFERENCE_IRRELEVANT_CONFIG_KEYS
    }


def configuration_digest(config: dict[str, Any]) -> str:
    return _digest(reference_configuration(config))


def session_ids_digest(session_ids: Iterable[str]) -> str:
    """Return the canonical digest used to bind a reference to its fit sessions."""

    return _digest(sorted({str(value) for value in session_ids}))


def _feature_key(source_id: str, observation_name: str) -> str:
    return f"{source_id}::{observation_name}"


def _robust_scale(values: np.ndarray, median: float) -> tuple[float, float]:
    """Return MAD and a finite scale suitable for robust z-scores."""
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-12:
        q25, q75 = np.quantile(values, (0.25, 0.75))
        scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 0.0
    return mad, scale


@dataclass(frozen=True)
class FeatureReference:
    source_id: str
    observation_name: str
    count_seen: int
    count_sampled: int
    median: float
    mad: float
    robust_scale: float
    quantile_probabilities: tuple[float, ...]
    quantile_values: tuple[float, ...]

    def percentile(self, value: float) -> float:
        quantiles = np.asarray(self.quantile_values, dtype=float)
        probabilities = np.asarray(self.quantile_probabilities, dtype=float) * 100.0
        if quantiles.size == 0 or not np.isfinite(value):
            return 50.0
        unique_values, inverse = np.unique(quantiles, return_inverse=True)
        unique_probabilities = np.zeros_like(unique_values)
        for index in range(unique_values.size):
            points = probabilities[inverse == index]
            unique_probabilities[index] = float(np.mean(points))
        if unique_values.size == 1:
            return 50.0
        return float(np.clip(np.interp(value, unique_values, unique_probabilities), 0.0, 100.0))

    def robust_z(self, value: float) -> float | None:
        if self.robust_scale <= 0:
            return 0.0
        return float((value - self.median) / self.robust_scale)


class _Reservoir:
    def __init__(self, limit: int, seed: int):
        self.limit = limit
        self.random = random.Random(seed)
        self.values: list[float] = []
        self.count_seen = 0

    def add(self, value: float) -> None:
        if not math.isfinite(value):
            return
        self.count_seen += 1
        if len(self.values) < self.limit:
            self.values.append(float(value))
            return
        selected = self.random.randrange(self.count_seen)
        if selected < self.limit:
            self.values[selected] = float(value)


@dataclass(frozen=True)
class ReferenceSet:
    dataset: str
    fit_split: str
    session_count: int
    session_ids_sha256: str
    config_sha256: str
    created_at_utc: str
    reservoir_limit_per_feature: int
    features: dict[str, FeatureReference]
    schema_version: str = REFERENCE_SCHEMA_VERSION

    @classmethod
    def fit(
        cls,
        dataset: str,
        fit_split: str,
        session_ids: Iterable[str],
        units: Iterable[RawEvidenceUnit],
        config: dict[str, Any],
        reservoir_limit_per_feature: int = 50_000,
        seed: int = 17,
    ) -> "ReferenceSet":
        if fit_split.strip().lower() not in {"train", "training"}:
            raise ValueError(
                "Reference statistics must be fitted on an explicitly named train/training split"
            )
        if reservoir_limit_per_feature <= 0:
            raise ValueError("reservoir_limit_per_feature must be positive")

        ids = sorted({str(value) for value in session_ids})
        reservoirs: dict[str, _Reservoir] = {}
        for unit in units:
            unit.validate()
            if unit.session_id not in ids:
                raise ValueError(f"Unit {unit.evidence_id} is outside the declared fitting sessions")
            if unit.availability_status == "unavailable":
                continue
            for observation in unit.observations:
                key = _feature_key(unit.source_id, observation.name)
                if key not in reservoirs:
                    feature_seed = int.from_bytes(
                        hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()[:8], "big"
                    )
                    reservoirs[key] = _Reservoir(reservoir_limit_per_feature, feature_seed)
                reservoirs[key].add(float(observation.value))

        features: dict[str, FeatureReference] = {}
        probabilities = np.asarray(DEFAULT_QUANTILE_PROBABILITIES, dtype=float)
        for key, reservoir in sorted(reservoirs.items()):
            values = np.asarray(reservoir.values, dtype=float)
            if values.size == 0:
                continue
            median = float(np.median(values))
            mad, scale = _robust_scale(values, median)
            source_id, observation_name = key.split("::", 1)
            features[key] = FeatureReference(
                source_id=source_id,
                observation_name=observation_name,
                count_seen=reservoir.count_seen,
                count_sampled=int(values.size),
                median=median,
                mad=mad,
                robust_scale=scale,
                quantile_probabilities=tuple(float(value) for value in probabilities),
                quantile_values=tuple(float(value) for value in np.quantile(values, probabilities)),
            )

        if not features:
            raise ValueError("No valid observations were available for fitting references")
        return cls(
            dataset=dataset,
            fit_split="train",
            session_count=len(ids),
            session_ids_sha256=session_ids_digest(ids),
            config_sha256=_digest(config),
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            reservoir_limit_per_feature=reservoir_limit_per_feature,
            features=features,
        )

    @property
    def reference_id(self) -> str:
        payload = {
            "dataset": self.dataset,
            "fit_split": self.fit_split,
            "session_ids_sha256": self.session_ids_sha256,
            "config_sha256": self.config_sha256,
            "features": {key: asdict(value) for key, value in sorted(self.features.items())},
        }
        return _digest(payload)[:16]

    def get(self, source_id: str, observation_name: str) -> FeatureReference | None:
        return self.features.get(_feature_key(source_id, observation_name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "reference_id": self.reference_id,
            "dataset": self.dataset,
            "fit_split": self.fit_split,
            "session_count": self.session_count,
            "session_ids_sha256": self.session_ids_sha256,
            "config_sha256": self.config_sha256,
            "created_at_utc": self.created_at_utc,
            "reservoir_limit_per_feature": self.reservoir_limit_per_feature,
            "features": {key: asdict(value) for key, value in sorted(self.features.items())},
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceSet":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != REFERENCE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported reference schema: {payload.get('schema_version')}")
        features = {
            key: FeatureReference(
                source_id=value["source_id"],
                observation_name=value["observation_name"],
                count_seen=int(value["count_seen"]),
                count_sampled=int(value["count_sampled"]),
                median=float(value["median"]),
                mad=float(value["mad"]),
                robust_scale=float(value["robust_scale"]),
                quantile_probabilities=tuple(float(item) for item in value["quantile_probabilities"]),
                quantile_values=tuple(float(item) for item in value["quantile_values"]),
            )
            for key, value in payload["features"].items()
        }
        output = cls(
            dataset=str(payload["dataset"]),
            fit_split=str(payload["fit_split"]),
            session_count=int(payload["session_count"]),
            session_ids_sha256=str(payload["session_ids_sha256"]),
            config_sha256=str(payload["config_sha256"]),
            created_at_utc=str(payload["created_at_utc"]),
            reservoir_limit_per_feature=int(payload["reservoir_limit_per_feature"]),
            features=features,
            schema_version=str(payload["schema_version"]),
        )
        expected_id = payload.get("reference_id")
        if expected_id and output.reference_id != expected_id:
            raise ValueError("Reference file digest does not match its contents")
        return output


def _category(percentile: float | None) -> str:
    if percentile is None:
        return "unreferenced"
    if percentile < 5.0:
        return "very_low"
    if percentile < 25.0:
        return "low"
    if percentile <= 75.0:
        return "typical"
    if percentile < 95.0:
        return "high"
    return "very_high"


def _subject_statistics(units: Iterable[RawEvidenceUnit]) -> dict[str, tuple[float, float]]:
    grouped: dict[str, list[float]] = {}
    for unit in units:
        if unit.availability_status == "unavailable":
            continue
        for observation in unit.observations:
            grouped.setdefault(_feature_key(unit.source_id, observation.name), []).append(
                float(observation.value)
            )
    output: dict[str, tuple[float, float]] = {}
    for key, values in grouped.items():
        array = np.asarray(values, dtype=float)
        if array.size < 3:
            continue
        median = float(np.median(array))
        _, scale = _robust_scale(array, median)
        if scale > 0:
            output[key] = (median, scale)
    return output


def enrich_session(
    raw_units: Iterable[RawEvidenceUnit],
    references: ReferenceSet,
) -> Iterator[EvidenceUnit]:
    units = list(raw_units)
    if units and any(unit.dataset != references.dataset for unit in units):
        raise ValueError(
            f"Reference dataset {references.dataset!r} does not match the evidence dataset"
        )
    subject_statistics = _subject_statistics(units)
    scope = f"{references.dataset}:train:{references.reference_id}"
    for unit in units:
        observations: list[Observation] = []
        for raw in unit.observations:
            reference = references.get(unit.source_id, raw.name)
            percentile = reference.percentile(raw.value) if reference else None
            robust_z = reference.robust_z(raw.value) if reference else None
            subject = subject_statistics.get(_feature_key(unit.source_id, raw.name))
            subject_z = (
                float((raw.value - subject[0]) / subject[1]) if subject is not None else None
            )
            observations.append(
                Observation(
                    name=raw.name,
                    value=float(raw.value),
                    unit=raw.unit,
                    category=_category(percentile),  # type: ignore[arg-type]
                    percentile=percentile,
                    robust_z=robust_z,
                    subject_robust_z=subject_z,
                    semantic_reliability=raw.semantic_reliability,
                    reference_scope=scope if reference else "unreferenced",
                )
            )

        missing_reasons = list(unit.missing_reasons)
        privacy_scrubbed = "privacy_scrubbed" in missing_reasons or float(
            unit.quality.get("privacy_scrubbed_ratio") or 0.0
        ) > 0.0
        evidence = EvidenceUnit(
            evidence_id=unit.evidence_id,
            dataset=unit.dataset,
            session_id=unit.session_id,
            modality=unit.modality,
            source={
                "source_id": unit.source_id,
                "extractor": unit.extractor,
                "version": unit.extractor_version,
                "representation_level": unit.representation_level,
                "source_files": list(unit.source_files),
            },
            time_range={"start_sec": unit.start_sec, "end_sec": unit.end_sec},
            availability={
                "status": unit.availability_status,
                "missing_reasons": missing_reasons,
                "padding": False,
                "privacy_scrubbed": privacy_scrubbed,
            },
            quality=dict(unit.quality),
            context=dict(unit.context),
            observations=tuple(observations),
            semantic_limits=unit.semantic_limits,
        )
        evidence.validate()
        yield evidence
