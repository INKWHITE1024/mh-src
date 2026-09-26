"""Load compiled evidence stores and materialize targeted evidence for revision passes."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from rethink_mh.textualization.renderer import (
    DISPLAY_NAMES,
    REASON_TEXT,
    assert_semantically_safe,
    format_time,
)
from rethink_mh.textualization.schema import EvidenceUnit, Observation


DEFAULT_MAX_ATOMIC_PER_SEGMENT = 4
DEFAULT_MAX_OBSERVATIONS_PER_MODALITY = 3
MAX_ATOMIC_PER_SEGMENT = 64
MAX_OBSERVATIONS_PER_MODALITY = 32

_ID_PATTERN = {
    "segment": re.compile(r"S\d+\Z"),
    "atomic": re.compile(r"E\d+\Z"),
    "canonical": re.compile(r"[A-Za-z0-9_.:-]+\Z"),
}
_NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_ATOMIC_LINE_PATTERN = re.compile(
    rf"EVIDENCE (?P<atomic_id>E\d+) \| time "
    rf"(?P<start>{_NUMBER_PATTERN}) to (?P<end>{_NUMBER_PATTERN}) seconds(?: \||\Z)"
)
_OPAQUE_SHORTHAND_PATTERNS = (
    re.compile(r"\b(?:spk|au_var|head_var|gaze_var|pitch_var|loud_var)\b", re.I),
    # H1 and H2 are standard acoustic harmonic names, so a bare ``H1`` must not
    # be treated as a high-rank shorthand.  The unambiguous private
    # codes are retained here.
    re.compile(r"\b(?:VL|VH|L|T)\d+\b"),
    re.compile(r"\^[A-Za-z]"),
    re.compile(r"(?:^|\|)(?:A|V)\["),
)

_STABLE_ANCHORS = {
    "audio": (
        "participant_speaking_ratio",
        "transcribed_speech_ratio",
        "voiced_activity_ratio",
        "pitch_variability",
        "loudness_variability",
    ),
    "visual": (
        "facial_action_variability",
        "head_motion_variability",
        "gaze_direction_variability",
        "facial_landmark_shape_change",
    ),
}

_UNIT_NAMES = {
    "ratio": "percent",
    "activation": "activation units",
    "activation_per_frame": "activation units per frame",
    "radian": "radians",
    "radian_per_second": "radians per second",
    "semitone": "semitones",
    "hertz": "hertz",
    "decibel": "decibels",
    "normalized": "normalized units",
}


@dataclass(frozen=True)
class TargetedEvidence:
    """Dynamically rendered evidence and its complete retrieval provenance."""

    text: str
    selected_segment_ids: tuple[str, ...]
    selected_atomic_evidence_ids: tuple[str, ...]
    canonical_evidence_ids: tuple[str, ...]

    @property
    def segment_ids(self) -> tuple[str, ...]:
        """Short alias for callers that already use S identifiers."""

        return self.selected_segment_ids

    @property
    def atomic_evidence_ids(self) -> tuple[str, ...]:
        """Short alias for callers that already use E identifiers."""

        return self.selected_atomic_evidence_ids


@dataclass(frozen=True)
class _IndexRecord:
    segment_id: str
    atomic_id: str
    start_sec: float
    end_sec: float
    canonical_ids: tuple[str, ...]
    modalities: tuple[str, ...]
    source_ids: tuple[str, ...]

    @property
    def midpoint_sec(self) -> float:
        return (self.start_sec + self.end_sec) / 2.0


class EvidenceStore:
    """Validated, session-local store for second-pass evidence retrieval."""

    def __init__(
        self,
        *,
        session_dir: Path,
        records_by_segment: Mapping[str, tuple[_IndexRecord, ...]],
        canonical_by_id: Mapping[str, EvidenceUnit],
    ) -> None:
        self._session_dir = session_dir
        self._records_by_segment = dict(records_by_segment)
        self._canonical_by_id = dict(canonical_by_id)
        self._available_segment_ids = tuple(
            sorted(self._records_by_segment, key=_natural_identifier_key)
        )

    @classmethod
    def from_session_dir(
        cls,
        path: str | Path,
        *,
        protocol_version: Literal["native"] = "native",
    ) -> EvidenceStore:
        """Load a compiled evidence session and require an exact, validated index mapping.

        ``protocol_version`` is fixed to the native file family
        (``evidence_index.native.jsonl`` + ``atomic_evidence.native.txt``).
        Canonical units (``evidence_units.jsonl``) are shared.
        """

        session_dir = Path(path)
        if not session_dir.is_dir():
            raise ValueError(
                f"Evidence session directory does not exist: {session_dir}"
            )
        if protocol_version != "native":
            raise ValueError(f"Unsupported protocol_version: {protocol_version!r}")

        canonical_path = session_dir / "evidence_units.jsonl"
        index_path = (
            session_dir / f"evidence_index.{protocol_version}.jsonl"
        )
        atomic_path = session_dir / f"atomic_evidence.{protocol_version}.txt"
        for required in (canonical_path, index_path, atomic_path):
            if not required.is_file():
                raise ValueError(f"Required evidence file is missing: {required}")

        canonical_by_id = _load_canonical_units(canonical_path)
        index_records = _load_index(index_path, protocol_version=protocol_version)
        atomic_times = _load_atomic_file(atomic_path)
        _cross_validate(canonical_by_id, index_records, atomic_times)

        records_by_segment: dict[str, list[_IndexRecord]] = {}
        for record in index_records:
            records_by_segment.setdefault(record.segment_id, []).append(record)
        ordered_records = {
            segment_id: tuple(
                sorted(
                    records,
                    key=lambda item: (
                        item.midpoint_sec,
                        item.start_sec,
                        _natural_identifier_key(item.atomic_id),
                    ),
                )
            )
            for segment_id, records in records_by_segment.items()
        }
        return cls(
            session_dir=session_dir,
            records_by_segment=ordered_records,
            canonical_by_id=canonical_by_id,
        )

    @property
    def available_segment_ids(self) -> tuple[str, ...]:
        return self._available_segment_ids

    def render_segments(
        self,
        segment_ids: Sequence[str],
        max_atomic_per_segment: int = DEFAULT_MAX_ATOMIC_PER_SEGMENT,
        max_observations_per_modality: int = DEFAULT_MAX_OBSERVATIONS_PER_MODALITY,
        *,
        exclude_atomic_ids: Sequence[str] = (),
        selection_strategy: Literal[
            "time_spanning", "salience_time_spanning"
        ] = "time_spanning",
    ) -> TargetedEvidence:
        """Retrieve novel atomic windows and render canonical observations anew.

        ``exclude_atomic_ids`` is the information boundary for repeated review:
        an atomic window already disclosed to the model cannot be returned again.
        The default selection remains backward compatible; the salience-aware
        strategy mixes unusual, reliable windows with temporal coverage.
        """

        selected_segments = _validate_segment_request(
            segment_ids, self._available_segment_ids
        )
        _validate_bounded_integer(
            "max_atomic_per_segment",
            max_atomic_per_segment,
            upper=MAX_ATOMIC_PER_SEGMENT,
        )
        _validate_bounded_integer(
            "max_observations_per_modality",
            max_observations_per_modality,
            upper=MAX_OBSERVATIONS_PER_MODALITY,
        )
        excluded_atomic = _validate_atomic_exclusions(
            exclude_atomic_ids,
            {
                record.atomic_id
                for records in self._records_by_segment.values()
                for record in records
            },
        )
        if selection_strategy not in {
            "time_spanning",
            "salience_time_spanning",
        }:
            raise ValueError(
                "selection_strategy must be 'time_spanning' or "
                "'salience_time_spanning'"
            )

        lines = [
            "TARGETED EVIDENCE FOR SECOND-PASS REVIEW",
            "These are source measurements relative to training references. Missing "
            "measurements are not measured zeros. Weak or latent observations are signal "
            "patterns only.",
            "Every atomic window below is newly disclosed in this review; previously "
            "seen atomic windows are excluded.",
            "Selected segment identifiers: " + ", ".join(selected_segments) + ".",
        ]
        selected_atomic: list[str] = []
        selected_canonical: list[str] = []

        for segment_id in selected_segments:
            all_records = self._records_by_segment[segment_id]
            unseen_records = tuple(
                record
                for record in all_records
                if record.atomic_id not in excluded_atomic
            )
            records = _select_records(
                unseen_records,
                max_atomic_per_segment,
                strategy=selection_strategy,
                canonical_by_id=self._canonical_by_id,
            )
            lines.extend(
                (
                    "",
                    f"SEGMENT {segment_id}",
                    f"Retrieved {len(records)} of {len(unseen_records)} unseen atomic "
                    f"evidence windows using deterministic "
                    f"{selection_strategy.replace('_', '-')} selection; "
                    f"{len(all_records) - len(unseen_records)} previously disclosed "
                    "windows were excluded.",
                )
            )
            if records:
                lines.append(
                    _render_segment_contrast(
                        segment_id, records, self._canonical_by_id
                    )
                )
            for record in records:
                selected_atomic.append(record.atomic_id)
                lines.append(
                    f"ATOMIC EVIDENCE {record.atomic_id} | Time {format_time(record.start_sec)} "
                    f"to {format_time(record.end_sec)}."
                )
                lines.append(
                    _render_cross_modal_context(record, self._canonical_by_id)
                )
                for canonical_id in record.canonical_ids:
                    selected_canonical.append(canonical_id)
                    unit = self._canonical_by_id[canonical_id]
                    lines.extend(
                        _render_canonical_unit(
                            segment_id,
                            record.atomic_id,
                            unit,
                            max_observations_per_modality,
                        )
                    )

        text = "\n".join(lines).rstrip() + "\n"
        _validate_rendered_text(text)
        return TargetedEvidence(
            text=text,
            selected_segment_ids=selected_segments,
            selected_atomic_evidence_ids=tuple(selected_atomic),
            canonical_evidence_ids=tuple(selected_canonical),
        )


def _load_canonical_units(path: Path) -> dict[str, EvidenceUnit]:
    output: dict[str, EvidenceUnit] = {}
    for line_number, payload in _read_jsonl(path):
        try:
            unit = _parse_evidence_unit(payload)
            unit.validate()
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid canonical evidence at {path}:{line_number}: {error}"
            ) from error
        if not _ID_PATTERN["canonical"].fullmatch(unit.evidence_id):
            raise ValueError(
                f"Invalid canonical evidence ID at {path}:{line_number}: "
                f"{unit.evidence_id!r}"
            )
        if unit.modality not in {"audio", "visual"}:
            raise ValueError(
                f"Invalid canonical modality at {path}:{line_number}: {unit.modality!r}"
            )
        if unit.evidence_id in output:
            raise ValueError(f"Duplicate canonical evidence ID: {unit.evidence_id}")
        output[unit.evidence_id] = unit
    if not output:
        raise ValueError(f"Canonical evidence file is empty: {path}")
    session_keys = {(unit.dataset, unit.session_id) for unit in output.values()}
    if len(session_keys) != 1:
        raise ValueError(
            "Canonical evidence file contains more than one dataset/session"
        )
    return output


def _parse_evidence_unit(payload: Mapping[str, Any]) -> EvidenceUnit:
    observations_payload = payload["observations"]
    if not isinstance(observations_payload, list):
        raise ValueError("observations must be an array")
    observations = tuple(
        Observation(**_require_mapping(value, "observation"))
        for value in observations_payload
    )
    semantic_limits = payload.get("semantic_limits", ())
    if not isinstance(semantic_limits, list):
        raise ValueError("semantic_limits must be an array")
    return EvidenceUnit(
        evidence_id=_require_string(payload.get("evidence_id"), "evidence_id"),
        dataset=_require_string(payload.get("dataset"), "dataset"),
        session_id=_require_string(payload.get("session_id"), "session_id"),
        modality=_require_string(payload.get("modality"), "modality"),  # type: ignore[arg-type]
        source=dict(_require_mapping(payload.get("source"), "source")),
        time_range=dict(_require_mapping(payload.get("time_range"), "time_range")),
        availability=dict(
            _require_mapping(payload.get("availability"), "availability")
        ),
        quality=dict(_require_mapping(payload.get("quality"), "quality")),
        context=dict(_require_mapping(payload.get("context"), "context")),
        observations=observations,
        semantic_limits=tuple(
            _require_string(value, "semantic_limits item") for value in semantic_limits
        ),
        schema_version=_require_string(payload.get("schema_version"), "schema_version"),
    )


def _load_index(
    path: Path,
    *,
    protocol_version: Literal["native"] = "native",
) -> tuple[_IndexRecord, ...]:
    records: list[_IndexRecord] = []
    seen_atomic: set[str] = set()
    seen_canonical: set[str] = set()
    for line_number, payload in _read_jsonl(path):
        try:
            if payload.get("protocol_version") != protocol_version:
                raise ValueError(
                    f"protocol_version must be {protocol_version}"
                )
            segment_id = _require_identifier(payload.get("segment_id"), "segment")
            atomic_id = _require_identifier(payload.get("atomic_evidence_id"), "atomic")
            canonical_ids = _require_string_tuple(
                payload.get("canonical_evidence_ids"), "canonical_evidence_ids"
            )
            modalities = _require_string_tuple(payload.get("modalities"), "modalities")
            source_ids = _require_string_tuple(payload.get("source_ids"), "source_ids")
            if not canonical_ids:
                raise ValueError("canonical_evidence_ids cannot be empty")
            if len(canonical_ids) != len(set(canonical_ids)):
                raise ValueError("canonical_evidence_ids contains a duplicate mapping")
            if len(modalities) != len(canonical_ids) or len(source_ids) != len(
                canonical_ids
            ):
                raise ValueError(
                    "modalities and source_ids must align with canonical_evidence_ids"
                )
            start_sec = _require_finite_number(payload.get("start_sec"), "start_sec")
            end_sec = _require_finite_number(payload.get("end_sec"), "end_sec")
            if start_sec < 0 or end_sec <= start_sec:
                raise ValueError("index time range must have positive duration")
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid {protocol_version} index at {path}:{line_number}: {error}"
            ) from error

        if atomic_id in seen_atomic:
            raise ValueError(f"Duplicate atomic evidence mapping: {atomic_id}")
        duplicates = seen_canonical.intersection(canonical_ids)
        if duplicates:
            raise ValueError(
                "Canonical evidence is mapped more than once: "
                + ", ".join(sorted(duplicates))
            )
        seen_atomic.add(atomic_id)
        seen_canonical.update(canonical_ids)
        records.append(
            _IndexRecord(
                segment_id=segment_id,
                atomic_id=atomic_id,
                start_sec=start_sec,
                end_sec=end_sec,
                canonical_ids=canonical_ids,
                modalities=modalities,
                source_ids=source_ids,
            )
        )
    if not records:
        raise ValueError(f"Evidence index is empty: {path}")
    return tuple(records)


def _load_atomic_file(path: Path) -> dict[str, tuple[float, float]]:
    output: dict[str, tuple[float, float]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        match = _ATOMIC_LINE_PATTERN.match(line)
        if match is None:
            raise ValueError(f"Invalid atomic evidence line at {path}:{line_number}")
        atomic_id = match.group("atomic_id")
        if atomic_id in output:
            raise ValueError(f"Duplicate atomic evidence text record: {atomic_id}")
        start_sec = float(match.group("start"))
        end_sec = float(match.group("end"))
        if not math.isfinite(start_sec) or not math.isfinite(end_sec):
            raise ValueError(f"Non-finite atomic evidence time at {path}:{line_number}")
        output[atomic_id] = (start_sec, end_sec)
    if not output:
        raise ValueError(f"Atomic evidence file is empty: {path}")
    return output


def _cross_validate(
    canonical_by_id: Mapping[str, EvidenceUnit],
    index_records: Sequence[_IndexRecord],
    atomic_times: Mapping[str, tuple[float, float]],
) -> None:
    indexed_atomic = {record.atomic_id for record in index_records}
    atomic_text_ids = set(atomic_times)
    if indexed_atomic != atomic_text_ids:
        missing_text = indexed_atomic - atomic_text_ids
        unknown_text = atomic_text_ids - indexed_atomic
        raise ValueError(
            "Atomic text/index mismatch; missing text records: "
            f"{sorted(missing_text)}; unknown text records: {sorted(unknown_text)}"
        )

    indexed_canonical = {
        canonical_id
        for record in index_records
        for canonical_id in record.canonical_ids
    }
    canonical_ids = set(canonical_by_id)
    if indexed_canonical != canonical_ids:
        missing = indexed_canonical - canonical_ids
        unindexed = canonical_ids - indexed_canonical
        raise ValueError(
            "Canonical/index mismatch; missing canonical IDs: "
            f"{sorted(missing)}; unindexed canonical IDs: {sorted(unindexed)}"
        )

    for record in index_records:
        text_start, text_end = atomic_times[record.atomic_id]
        # The readable atomic file intentionally rounds seconds to two decimals;
        # canonical JSON and the index retain the exact released timestamp.
        if not _same_time(
            record.start_sec, text_start, tolerance=0.005001
        ) or not _same_time(record.end_sec, text_end, tolerance=0.005001):
            raise ValueError(
                f"Atomic text/index time mismatch for {record.atomic_id}: "
                f"index {record.start_sec}-{record.end_sec}, text {text_start}-{text_end}"
            )
        units = [canonical_by_id[canonical_id] for canonical_id in record.canonical_ids]
        indexed_modalities = tuple(unit.modality for unit in units)
        indexed_source_ids = tuple(str(unit.source.get("source_id")) for unit in units)
        if indexed_modalities != record.modalities:
            raise ValueError(f"Modality mapping mismatch for {record.atomic_id}")
        if indexed_source_ids != record.source_ids:
            raise ValueError(f"Source mapping mismatch for {record.atomic_id}")
        for unit in units:
            start_sec = _require_finite_number(
                unit.time_range.get("start_sec"), "canonical start_sec"
            )
            end_sec = _require_finite_number(
                unit.time_range.get("end_sec"), "canonical end_sec"
            )
            if not _same_time(start_sec, record.start_sec) or not _same_time(
                end_sec, record.end_sec
            ):
                raise ValueError(
                    f"Canonical/index time mismatch for {unit.evidence_id} in "
                    f"{record.atomic_id}"
                )


def _render_canonical_unit(
    segment_id: str,
    atomic_id: str,
    unit: EvidenceUnit,
    observation_limit: int,
) -> list[str]:
    modality = unit.modality.capitalize()
    extractor = str(unit.source.get("extractor", "unknown source")).replace("_", " ")
    status = str(unit.availability.get("status", "unavailable")).replace("_", " ")
    quality = str(unit.quality.get("level", "unknown")).replace("_", " ")
    lines = [
        f"Provenance: segment {segment_id}; atomic evidence {atomic_id}; "
        f"{unit.modality} source record.",
        f"{modality} source {extractor} was {status}. Source quality was {quality}.",
    ]
    valid_ratio = unit.quality.get("valid_ratio")
    if _is_finite_number(valid_ratio):
        lines[-1] = (
            lines[-1][:-1]
            + f", with {_format_number(100.0 * float(valid_ratio))} percent valid coverage."
        )
    reasons = unit.availability.get("missing_reasons") or []
    if reasons:
        rendered_reasons = [
            REASON_TEXT.get(str(reason), str(reason).replace("_", " "))
            for reason in reasons
        ]
        lines.append("Availability limitations: " + "; ".join(rendered_reasons) + ".")

    selected = _select_observations(unit.observations, unit.modality, observation_limit)
    if selected:
        lines.append(
            "Observed measurements relative to source-specific training references:"
        )
        lines.extend(f"- {_render_observation(item)}" for item in selected)
        omitted = len(unit.observations) - len(selected)
        if omitted:
            lines.append(
                f"- {omitted} additional canonical observations were not selected for "
                "this retrieval."
            )
    elif status == "unavailable":
        lines.append("No measurement was substituted for this unavailable source.")
    else:
        lines.append("No valid observations were available in this source window.")
    return lines


def _select_observations(
    observations: Sequence[Observation], modality: str, limit: int
) -> tuple[Observation, ...]:
    items = list(observations)
    if len(items) <= limit:
        return tuple(items)

    by_name = {item.name: item for item in items}
    anchor = next(
        (by_name[name] for name in _STABLE_ANCHORS[modality] if name in by_name),
        None,
    )
    selected: list[Observation] = []
    if anchor is not None:
        selected.append(anchor)

    remaining = [item for item in items if item is not anchor]
    reliability_priority = {"direct": 2, "derived": 2, "weak": 1, "latent": 0}
    remaining.sort(
        key=lambda item: (
            reliability_priority[item.semantic_reliability],
            _observation_salience(item),
            item.name,
        ),
        reverse=True,
    )
    selected.extend(remaining[: max(0, limit - len(selected))])
    return tuple(selected)


def _render_observation(observation: Observation) -> str:
    name = DISPLAY_NAMES.get(observation.name, observation.name.replace("_", " "))
    category = observation.category.replace("_", " ")
    if observation.percentile is None or category == "unreferenced":
        reference_phrase = "had no fitted training reference"
    else:
        reference_phrase = (
            f"was {category} at training-reference percentile "
            f"{int(round(observation.percentile))}"
        )

    if observation.unit == "ratio":
        measured = f"{_format_number(100.0 * observation.value)} percent"
    else:
        unit = _UNIT_NAMES.get(observation.unit, observation.unit.replace("_", " "))
        measured = f"{_format_number(observation.value)} {unit}"
    phrase = f"{name.capitalize()} {reference_phrase}; measured value {measured}."

    if observation.semantic_reliability == "direct":
        phrase += " Reliability: direct measurement."
    elif observation.semantic_reliability == "derived":
        phrase += " Reliability: derived measurement."
    else:
        phrase += (
            f" Reliability: {observation.semantic_reliability}; signal pattern only."
        )
    if (
        observation.subject_robust_z is not None
        and abs(observation.subject_robust_z) >= 2.0
    ):
        direction = "above" if observation.subject_robust_z > 0 else "below"
        phrase += f" It was {direction} this session's usual level."
    return phrase


def _uniformly_sample(
    records: Sequence[_IndexRecord], limit: int
) -> tuple[_IndexRecord, ...]:
    if len(records) <= limit:
        return tuple(records)
    if limit == 1:
        return (records[(len(records) - 1) // 2],)
    final_index = len(records) - 1
    selected_indices = [
        int(math.floor(index * final_index / (limit - 1) + 0.5))
        for index in range(limit)
    ]
    return tuple(records[index] for index in selected_indices)


def _select_records(
    records: Sequence[_IndexRecord],
    limit: int,
    *,
    strategy: Literal["time_spanning", "salience_time_spanning"],
    canonical_by_id: Mapping[str, EvidenceUnit],
) -> tuple[_IndexRecord, ...]:
    if strategy == "time_spanning":
        return _uniformly_sample(records, limit)
    if len(records) <= limit:
        return tuple(records)

    salient_slots = max(1, limit // 2)
    ranked = sorted(
        records,
        key=lambda record: (
            -_record_salience(record, canonical_by_id),
            record.midpoint_sec,
            _natural_identifier_key(record.atomic_id),
        ),
    )
    selected = list(ranked[:salient_slots])
    selected_ids = {record.atomic_id for record in selected}
    remaining = [
        record for record in records if record.atomic_id not in selected_ids
    ]
    selected.extend(_uniformly_sample(remaining, limit - len(selected)))
    return tuple(
        sorted(
            selected,
            key=lambda record: (
                record.midpoint_sec,
                record.start_sec,
                _natural_identifier_key(record.atomic_id),
            ),
        )
    )


def _record_salience(
    record: _IndexRecord,
    canonical_by_id: Mapping[str, EvidenceUnit],
) -> float:
    modality_scores: list[float] = []
    present_modalities = 0
    for canonical_id in record.canonical_ids:
        unit = canonical_by_id[canonical_id]
        status = str(unit.availability.get("status", "unavailable"))
        if status != "unavailable":
            present_modalities += 1
        valid_ratio = unit.quality.get("valid_ratio")
        quality_score = (
            max(0.0, min(1.0, float(valid_ratio)))
            if _is_finite_number(valid_ratio)
            else 0.0
        )
        observation_scores: list[float] = []
        for observation in unit.observations:
            percentile_score = (
                abs(float(observation.percentile) - 50.0) / 50.0
                if observation.percentile is not None
                else 0.0
            )
            subject_score = (
                min(abs(float(observation.subject_robust_z)), 4.0) / 4.0
                if observation.subject_robust_z is not None
                else 0.0
            )
            reliability = {
                "direct": 1.0,
                "derived": 0.9,
                "weak": 0.35,
                "latent": 0.2,
            }[observation.semantic_reliability]
            observation_scores.append(
                reliability * (0.7 * percentile_score + 0.3 * subject_score)
            )
        strongest = sorted(observation_scores, reverse=True)[:2]
        modality_scores.append(
            0.8 * (sum(strongest) / len(strongest) if strongest else 0.0)
            + 0.2 * quality_score
        )
    cross_modal_bonus = 0.1 if present_modalities >= 2 else 0.0
    return (
        sum(modality_scores) / len(modality_scores) if modality_scores else 0.0
    ) + cross_modal_bonus


def _render_cross_modal_context(
    record: _IndexRecord,
    canonical_by_id: Mapping[str, EvidenceUnit],
) -> str:
    available: list[str] = []
    deviations: dict[str, int] = {"audio": 0, "visual": 0}
    for canonical_id in record.canonical_ids:
        unit = canonical_by_id[canonical_id]
        if str(unit.availability.get("status", "unavailable")) != "unavailable":
            available.append(unit.modality)
        deviations[unit.modality] += sum(
            1
            for observation in unit.observations
            if observation.subject_robust_z is not None
            and abs(float(observation.subject_robust_z)) >= 2.0
        )
    if set(available) == {"audio", "visual"}:
        availability = "audio and visual measurements were both available"
    elif available:
        availability = f"only {available[0]} measurements were available"
    else:
        availability = "audio and visual measurements were unavailable"
    return (
        "Cross-modal measurement context: "
        f"{availability}; {deviations['audio']} audio and "
        f"{deviations['visual']} visual measurements differed from this session's "
        "usual level by at least two robust deviations. This describes measurement "
        "co-occurrence only."
    )


def _render_segment_contrast(
    segment_id: str,
    records: Sequence[_IndexRecord],
    canonical_by_id: Mapping[str, EvidenceUnit],
) -> str:
    modality_phrases: list[str] = []
    cooccurring_deviations = 0
    for record in records:
        modality_has_deviation = {"audio": False, "visual": False}
        for canonical_id in record.canonical_ids:
            unit = canonical_by_id[canonical_id]
            modality_has_deviation[unit.modality] = any(
                observation.subject_robust_z is not None
                and abs(float(observation.subject_robust_z)) >= 2.0
                for observation in unit.observations
            )
        if all(modality_has_deviation.values()):
            cooccurring_deviations += 1

    for modality in ("audio", "visual"):
        observations_by_name: dict[str, list[tuple[float, Observation]]] = {}
        for record in records:
            for canonical_id in record.canonical_ids:
                unit = canonical_by_id[canonical_id]
                if unit.modality != modality:
                    continue
                for observation in unit.observations:
                    if observation.percentile is not None:
                        observations_by_name.setdefault(
                            observation.name, []
                        ).append((record.midpoint_sec, observation))
        candidates: list[
            tuple[float, int, str, float, float, str]
        ] = []
        for name, values in observations_by_name.items():
            ordered = sorted(values, key=lambda item: item[0])
            if len(ordered) < 2:
                continue
            first = float(ordered[0][1].percentile)
            last = float(ordered[-1][1].percentile)
            reliability = max(
                (item[1].semantic_reliability for item in ordered),
                key=lambda value: {
                    "direct": 3,
                    "derived": 2,
                    "weak": 1,
                    "latent": 0,
                }[value],
            )
            reliability_score = {
                "direct": 3,
                "derived": 2,
                "weak": 1,
                "latent": 0,
            }[reliability]
            anchor_priority = 1 if name in _STABLE_ANCHORS[modality] else 0
            candidates.append(
                (
                    abs(last - first),
                    anchor_priority,
                    name,
                    first,
                    last,
                    reliability,
                )
            )
        if not candidates:
            modality_phrases.append(
                f"{modality} had no repeated referenced measurement to compare"
            )
            continue
        _, _, name, first, last, reliability = max(
            candidates,
            key=lambda item: (
                item[1],
                item[0],
                item[2],
            ),
        )
        display_name = DISPLAY_NAMES.get(name, name.replace("_", " "))
        change = last - first
        if abs(change) < 10.0:
            relation = (
                f"stayed within {int(round(abs(change)))} percentile points"
            )
        else:
            direction = "rose" if change > 0.0 else "fell"
            relation = (
                f"{direction} by {int(round(abs(change)))} percentile points"
            )
        modality_phrases.append(
            f"{modality} {display_name} {relation} from the earlier to the "
            f"later retrieved window ({reliability} measurement)"
        )
    return (
        f"Retrieved-window contrast for {segment_id}: "
        + "; ".join(modality_phrases)
        + f". Large audio and visual within-session deviations co-occurred in "
        f"{cooccurring_deviations} of {len(records)} retrieved windows. This is a "
        "measurement relation only."
    )


def _validate_atomic_exclusions(
    values: Sequence[str],
    available: set[str],
) -> frozenset[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError("exclude_atomic_ids must be a sequence of E identifiers")
    output: list[str] = []
    for value in values:
        if not isinstance(value, str) or _ID_PATTERN["atomic"].fullmatch(value) is None:
            raise ValueError(f"Invalid excluded atomic evidence identifier: {value!r}")
        output.append(value)
    if len(output) != len(set(output)):
        raise ValueError("exclude_atomic_ids contains duplicate identifiers")
    unknown = set(output) - available
    if unknown:
        raise ValueError(
            "Unknown excluded atomic evidence identifiers: "
            + ", ".join(sorted(unknown))
        )
    return frozenset(output)


def _validate_segment_request(
    segment_ids: Sequence[str], available: Sequence[str]
) -> tuple[str, ...]:
    if isinstance(segment_ids, (str, bytes)):
        raise TypeError("segment_ids must be a sequence of segment identifiers")
    requested = tuple(segment_ids)
    if not requested:
        raise ValueError("At least one segment identifier must be requested")
    if any(not isinstance(value, str) for value in requested):
        raise TypeError("Every segment identifier must be a string")
    if len(requested) != len(set(requested)):
        raise ValueError("Duplicate requested segment identifiers are not allowed")
    unknown = set(requested) - set(available)
    if unknown:
        raise ValueError("Unknown segment identifiers: " + ", ".join(sorted(unknown)))
    return requested


def _validate_bounded_integer(name: str, value: int, *, upper: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 1 <= value <= upper:
        raise ValueError(f"{name} must be between 1 and {upper}, inclusive")


def _read_jsonl(path: Path) -> list[tuple[int, Mapping[str, Any]]]:
    output: list[tuple[int, Mapping[str, Any]]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line, object_pairs_hook=_reject_duplicate_json_keys)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(
                f"Invalid JSON at {path}:{line_number}: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise ValueError(f"JSON record at {path}:{line_number} must be an object")
        output.append((line_number, payload))
    return output


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"Duplicate JSON key: {key}")
        output[key] = value
    return output


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return tuple(_require_string(item, f"{name} item") for item in value)


def _require_identifier(value: Any, kind: str) -> str:
    output = _require_string(value, f"{kind}_id")
    if not _ID_PATTERN[kind].fullmatch(output):
        raise ValueError(f"Invalid {kind} identifier: {output!r}")
    return output


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _require_finite_number(value: Any, name: str) -> float:
    if not _is_finite_number(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _same_time(left: float, right: float, *, tolerance: float = 1e-6) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def _natural_identifier_key(value: str) -> tuple[str, int]:
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", value)
    if match is None:
        return value, -1
    return match.group(1), int(match.group(2))


def _observation_salience(observation: Observation) -> float:
    if observation.percentile is None:
        return -1.0
    return abs(float(observation.percentile) - 50.0)


def _format_number(value: float) -> str:
    output = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return output if output not in {"", "-0"} else "0"


def _validate_rendered_text(text: str) -> None:
    assert_semantically_safe(text)
    for pattern in _OPAQUE_SHORTHAND_PATTERNS:
        if pattern.search(text):
            raise ValueError(
                f"Targeted evidence contains opaque private shorthand: {pattern.pattern}"
            )
