"""Compile evidence units into provenance-tracked native evidence text artifacts."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..canonical_slots import (
    CanonicalSlotDefinition,
    CanonicalSlotProjector,
    ProjectedSlot,
    SlotBinding,
    slots_for_modality,
)
from ..renderer import assert_semantically_safe
from ..schema import EvidenceUnit, Observation
from .base import EvidenceTextProtocol, ProtocolArtifacts


PROTOCOL_VERSION = "native"
NATIVE_SCHEMA_VERSION = "1.0.0"

MAPPING_QUALITY_TEXT = {
    "directly_aligned": "directly aligned measurement",
    "semantic_proxy": "semantic proxy",
    "provisional_mapping": "provisional source mapping",
}
MAPPING_QUALITY_WEIGHT = {
    "directly_aligned": 1.0,
    "semantic_proxy": 0.8,
    "provisional_mapping": 0.5,
}
ANCHOR_SLOT = {
    "audio": "speech_activity",
    "visual": "facial_activity",
}
LABEL_FIELD_NAMES = frozenset(
    {
        "label",
        "label2",
        "label3",
        "targetlabel",
        "target",
        "diagnosis",
        "depressed",
        "depressionlabel",
        "phq8",
        "phq8score",
        "phq9",
        "phq9score",
        "phq_8",
        "phq_9",
    }
)


def _json_line(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _finite_or_none(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), 6)


def _format_number(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "unavailable"
    rendered = f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    return rendered if rendered not in {"", "-0"} else "0"


def _format_time(value: float) -> str:
    return _format_number(float(value), 1)


def _display_unit(unit: str) -> str:
    return str(unit).replace("_", " ").strip() or "unitless"


def _short_ids(values: Sequence[str]) -> str:
    if not values:
        return "none"
    matches = [re.fullmatch(r"E(\d+)", value) for value in values]
    if all(match is not None for match in matches):
        numbers = [int(match.group(1)) for match in matches if match is not None]
        if numbers == list(range(numbers[0], numbers[0] + len(numbers))):
            return values[0] if len(values) == 1 else f"{values[0]} to {values[-1]}"
        return (
            f"{values[0]} to {values[-1]} "
            f"({len(values)} refs; exact list in segment plan)"
        )
    if len(values) <= 4:
        return ", ".join(values)
    return (
        f"{values[0]} to {values[-1]} "
        f"({len(values)} refs; exact list in segment plan)"
    )


def _facet_for_observation(observation: Observation) -> str:
    name = observation.name
    if any(token in name for token in ("variability", "std", "variance")):
        return "dispersion"
    if any(token in name for token in ("iqr", "_range")):
        return "iqr"
    if any(
        token in name
        for token in ("slope", "trend", "change_magnitude", "_speed")
    ):
        return "slope_or_change"
    return "level"


def _observation_payload(observation: Observation) -> dict[str, Any]:
    return {
        "name": observation.name,
        "facet": _facet_for_observation(observation),
        "raw": _finite_or_none(observation.value),
        "unit": observation.unit,
        "category": observation.category,
        "percentile": _finite_or_none(observation.percentile),
        "robust_z": _finite_or_none(observation.robust_z),
        "subject_robust_z": _finite_or_none(observation.subject_robust_z),
        "semantic_reliability": observation.semantic_reliability,
        "reference_scope": observation.reference_scope,
    }


def _slot_payload(slot: ProjectedSlot) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "slot_id": slot.definition.slot_id,
        "slot_label": slot.definition.label,
        "status": slot.status,
        "reason": slot.reason,
        "mapping_quality": (
            slot.binding.mapping_quality if slot.binding is not None else None
        ),
        "measurement_name": (
            slot.binding.measurement_name if slot.binding is not None else None
        ),
        "measurement": (
            _observation_payload(slot.observation)
            if slot.observation is not None
            else None
        ),
    }
    return payload


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key).casefold()
            yield from _walk_keys(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            yield from _walk_keys(item)


def _assert_label_free(value: Any) -> None:
    forbidden = sorted(
        {
            key
            for key in _walk_keys(value)
            if re.sub(r"[^a-z0-9]", "", key) in LABEL_FIELD_NAMES
        }
    )
    if forbidden:
        raise ValueError(f"Evidence payload contains forbidden label fields: {forbidden}")


@dataclass(frozen=True)
class _AtomicWindow:
    evidence_id: str
    start_sec: float
    end_sec: float
    units: tuple[EvidenceUnit, ...]

    @property
    def midpoint_sec(self) -> float:
        return (self.start_sec + self.end_sec) / 2.0


@dataclass(frozen=True)
class _Segment:
    segment_id: str
    start_sec: float
    end_sec: float
    atomic: tuple[_AtomicWindow, ...]


@dataclass(frozen=True)
class _AggregatedSlot:
    definition: CanonicalSlotDefinition
    status: str
    binding: SlotBinding | None
    raw: float | None
    unit: str | None
    percentile: float | None
    robust_z: float | None
    category: str | None
    facet: str | None
    measured_window_count: int
    source_window_count: int
    evidence_ids: tuple[str, ...]

    @property
    def salience(self) -> float:
        if self.status != "measured" or self.percentile is None:
            return -1.0
        mapping_weight = (
            MAPPING_QUALITY_WEIGHT[self.binding.mapping_quality]
            if self.binding is not None
            else 0.0
        )
        coverage = self.measured_window_count / max(1, self.source_window_count)
        return abs(self.percentile - 50.0) * mapping_weight * coverage

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.definition.slot_id,
            "slot_label": self.definition.label,
            "status": self.status,
            "measurement_name": (
                self.binding.measurement_name if self.binding is not None else None
            ),
            "mapping_quality": (
                self.binding.mapping_quality if self.binding is not None else None
            ),
            "facet": self.facet,
            "raw_median": _finite_or_none(self.raw),
            "unit": self.unit,
            "median_percentile": _finite_or_none(self.percentile),
            "median_robust_z": _finite_or_none(self.robust_z),
            "category": self.category,
            "measured_window_count": self.measured_window_count,
            "source_window_count": self.source_window_count,
            "evidence_ids": list(self.evidence_ids),
            "selection_score": _finite_or_none(self.salience),
        }


def _group_atomic(units: Sequence[EvidenceUnit]) -> list[_AtomicWindow]:
    grouped: dict[tuple[float, float], list[EvidenceUnit]] = defaultdict(list)
    for unit in units:
        grouped[
            (
                round(float(unit.time_range["start_sec"]), 6),
                round(float(unit.time_range["end_sec"]), 6),
            )
        ].append(unit)
    output: list[_AtomicWindow] = []
    for index, ((start_sec, end_sec), items) in enumerate(sorted(grouped.items())):
        output.append(
            _AtomicWindow(
                evidence_id=f"E{index:04d}",
                start_sec=start_sec,
                end_sec=end_sec,
                units=tuple(
                    sorted(
                        items,
                        key=lambda item: (
                            0 if item.modality == "audio" else 1,
                            str(item.source.get("source_id", "")),
                            item.evidence_id,
                        ),
                    )
                ),
            )
        )
    return output


def _group_segments(
    atomic: Sequence[_AtomicWindow],
    *,
    segment_sec: float,
    max_segments: int,
) -> tuple[list[_Segment], float]:
    session_end = max((item.end_sec for item in atomic), default=0.0)
    natural_count = max(1, int(math.ceil(session_end / segment_sec)))
    merge_factor = max(1, int(math.ceil(natural_count / max_segments)))
    effective_segment_sec = segment_sec * merge_factor
    grouped: dict[int, list[_AtomicWindow]] = defaultdict(list)
    for item in atomic:
        grouped[int(math.floor(item.midpoint_sec / effective_segment_sec))].append(
            item
        )
    output: list[_Segment] = []
    for sequence, group_index in enumerate(sorted(grouped)):
        start_sec = group_index * effective_segment_sec
        end_sec = min(start_sec + effective_segment_sec, session_end)
        output.append(
            _Segment(
                segment_id=f"S{sequence:03d}",
                start_sec=start_sec,
                end_sec=end_sec,
                atomic=tuple(grouped[group_index]),
            )
        )
    return output, effective_segment_sec


def _atomic_record(
    item: _AtomicWindow,
    projector: CanonicalSlotProjector,
    extension_limit: int,
) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    canonical_ids: list[str] = []
    for unit in item.units:
        projection = projector.project(unit, extension_limit=extension_limit)
        canonical_ids.append(unit.evidence_id)
        source = {
            "modality": unit.modality,
            "source_id": str(unit.source.get("source_id", "")),
            "extractor": str(unit.source.get("extractor", "")),
            "extractor_version": str(unit.source.get("version", "")),
            "representation_level": str(
                unit.source.get("representation_level", "")
            ),
            "availability": {
                "status": str(unit.availability.get("status", "unavailable")),
                "missing_reasons": list(
                    unit.availability.get("missing_reasons") or []
                ),
            },
            "quality": {
                "level": str(unit.quality.get("level", "unknown")),
                "valid_ratio": _finite_or_none(unit.quality.get("valid_ratio")),
            },
            "slots": [_slot_payload(slot) for slot in projection.slots],
            "extensions": [
                _observation_payload(observation)
                for observation in projection.extensions
            ],
        }
        sources.append(source)
    record = {
        "schema_version": NATIVE_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "evidence_id": item.evidence_id,
        "granularity": "window",
        "time_basis": "session_relative_seconds",
        "time_range": {
            "start_sec": _finite_or_none(item.start_sec),
            "end_sec": _finite_or_none(item.end_sec),
        },
        "canonical_evidence_ids": canonical_ids,
        "sources": sources,
    }
    _assert_label_free(record)
    return record


def _render_measurement(measurement: Mapping[str, Any]) -> str:
    raw = measurement.get("raw")
    unit = str(measurement.get("unit", "unitless"))
    if unit == "ratio" and raw is not None:
        raw_text = f"{100.0 * float(raw):.1f}%"
    else:
        raw_text = f"{_format_number(raw)} {_display_unit(unit)}"
    details = [f"raw value {raw_text}"]
    percentile = measurement.get("percentile")
    if percentile is None:
        details.append("fold-local training percentile unavailable")
    else:
        details.append(
            f"fold-local training percentile {int(round(float(percentile)))}"
        )
    robust_z = measurement.get("robust_z")
    if robust_z is not None:
        details.append(f"robust z {_format_number(float(robust_z), 2)}")
    details.append(f"facet {measurement.get('facet', 'level')}")
    return "; ".join(details)


def _render_atomic_record(record: Mapping[str, Any]) -> str:
    time_range = record["time_range"]
    fields = [
        f"EVIDENCE {record['evidence_id']}",
        (
            f"time {_format_time(float(time_range['start_sec']))} to "
            f"{_format_time(float(time_range['end_sec']))} seconds"
        ),
    ]
    for source in record["sources"]:
        modality = str(source["modality"]).upper()
        availability = source["availability"]
        quality = source["quality"]
        source_fields = [
            f"{modality} source {source['extractor']}",
            f"availability {availability['status']}",
            f"quality {quality['level']}",
        ]
        if quality.get("valid_ratio") is not None:
            source_fields.append(
                f"valid coverage {100.0 * float(quality['valid_ratio']):.1f}%"
            )
        reasons = availability.get("missing_reasons") or []
        if reasons:
            source_fields.append(
                "limitations " + ", ".join(str(reason).replace("_", " ") for reason in reasons)
            )
        fields.append("; ".join(source_fields))
        slots: list[str] = []
        for slot in source["slots"]:
            status = str(slot["status"])
            if status == "measured":
                measurement = slot["measurement"]
                slots.append(
                    f"{slot['slot_label']}: MEASURED "
                    f"({_render_measurement(measurement)}; "
                    f"mapping {MAPPING_QUALITY_TEXT[str(slot['mapping_quality'])]})"
                )
            elif status == "unavailable":
                slots.append(
                    f"{slot['slot_label']}: UNAVAILABLE "
                    "(source supports this measurement, but this window did not "
                    "pass quality checks)"
                )
            else:
                slots.append(
                    f"{slot['slot_label']}: NOT MEASURED "
                    "(the released source has no validated mapping for this slot)"
                )
        fields.append(f"{modality} canonical slots: " + "; ".join(slots))
    rendered = " | ".join(fields)
    assert_semantically_safe(rendered)
    return rendered


def _aggregate_slot(
    definition: CanonicalSlotDefinition,
    units: Sequence[EvidenceUnit],
    unit_to_atomic: Mapping[str, str],
    projector: CanonicalSlotProjector,
) -> _AggregatedSlot:
    pairs: list[tuple[EvidenceUnit, ProjectedSlot]] = []
    for unit in units:
        slot = next(
            candidate
            for candidate in projector.project(unit, extension_limit=0).slots
            if candidate.definition.slot_id == definition.slot_id
        )
        pairs.append((unit, slot))
    supported = [slot for _, slot in pairs if slot.status != "not_measured"]
    if not supported:
        return _AggregatedSlot(
            definition=definition,
            status="not_measured",
            binding=None,
            raw=None,
            unit=None,
            percentile=None,
            robust_z=None,
            category=None,
            facet=None,
            measured_window_count=0,
            source_window_count=len(units),
            evidence_ids=(),
        )
    measured = [
        (unit, slot)
        for unit, slot in pairs
        if slot.status == "measured"
        and slot.binding is not None
        and slot.observation is not None
    ]
    if not measured:
        binding = next(
            (slot.binding for slot in supported if slot.binding is not None),
            None,
        )
        return _AggregatedSlot(
            definition=definition,
            status="unavailable",
            binding=binding,
            raw=None,
            unit=None,
            percentile=None,
            robust_z=None,
            category=None,
            facet=None,
            measured_window_count=0,
            source_window_count=len(units),
            evidence_ids=(),
        )

    by_binding: dict[SlotBinding, list[tuple[EvidenceUnit, ProjectedSlot]]] = (
        defaultdict(list)
    )
    binding_order: list[SlotBinding] = []
    for pair in measured:
        binding = pair[1].binding
        assert binding is not None
        if binding not in by_binding:
            binding_order.append(binding)
        by_binding[binding].append(pair)
    binding = max(
        binding_order,
        key=lambda item: (len(by_binding[item]), -binding_order.index(item)),
    )
    selected = by_binding[binding]
    observations = [
        slot.observation for _, slot in selected if slot.observation is not None
    ]
    raw_values = [float(item.value) for item in observations]
    percentiles = [
        float(item.percentile)
        for item in observations
        if item.percentile is not None
    ]
    robust_z = [
        float(item.robust_z) for item in observations if item.robust_z is not None
    ]
    categories = Counter(item.category for item in observations)
    units_seen = Counter(item.unit for item in observations)
    evidence_ids = tuple(
        dict.fromkeys(unit_to_atomic[unit.evidence_id] for unit, _ in selected)
    )
    return _AggregatedSlot(
        definition=definition,
        status="measured",
        binding=binding,
        raw=float(np.median(raw_values)),
        unit=max(units_seen, key=lambda item: (units_seen[item], item)),
        percentile=(
            float(np.median(percentiles)) if percentiles else None
        ),
        robust_z=float(np.median(robust_z)) if robust_z else None,
        category=max(categories, key=lambda item: (categories[item], item)),
        facet=_facet_for_observation(observations[0]),
        measured_window_count=len(selected),
        source_window_count=len(units),
        evidence_ids=evidence_ids,
    )


def _select_slots(
    slots: Sequence[_AggregatedSlot],
    *,
    modality: str,
    limit: int,
) -> list[_AggregatedSlot]:
    measured = [slot for slot in slots if slot.status == "measured"]
    if not measured or limit <= 0:
        return []
    ranked = sorted(
        measured,
        key=lambda slot: (
            slot.salience,
            slot.measured_window_count,
            slot.definition.slot_id,
        ),
        reverse=True,
    )
    anchor_id = ANCHOR_SLOT[modality]
    anchor = next(
        (
            slot
            for slot in ranked
            if slot.definition.slot_id == anchor_id and slot.salience >= 10.0
        ),
        None,
    )
    selected = [anchor if anchor is not None else ranked[0]]
    selected.extend(slot for slot in ranked if slot not in selected)
    return selected[:limit]


def _source_segment_payload(
    *,
    modality: str,
    source_id: str,
    units: Sequence[EvidenceUnit],
    unit_to_atomic: Mapping[str, str],
    projector: CanonicalSlotProjector,
    selected_slots_per_modality: int,
) -> dict[str, Any]:
    slots = [
        _aggregate_slot(definition, units, unit_to_atomic, projector)
        for definition in slots_for_modality(modality)
    ]
    selected = _select_slots(
        slots,
        modality=modality,
        limit=selected_slots_per_modality,
    )
    status_counts = Counter(
        str(unit.availability.get("status", "unavailable")) for unit in units
    )
    valid_ratios = [
        float(unit.quality["valid_ratio"])
        for unit in units
        if isinstance(unit.quality.get("valid_ratio"), (int, float))
        and math.isfinite(float(unit.quality["valid_ratio"]))
    ]
    extractors = Counter(str(unit.source.get("extractor", source_id)) for unit in units)
    return {
        "modality": modality,
        "source_id": source_id,
        "extractor": max(
            extractors, key=lambda item: (extractors[item], item)
        ),
        "availability_window_counts": dict(sorted(status_counts.items())),
        "median_valid_ratio": (
            _finite_or_none(float(np.median(valid_ratios)))
            if valid_ratios
            else None
        ),
        "slots": [slot.to_dict() for slot in slots],
        "selected_slot_ids": [slot.definition.slot_id for slot in selected],
        "selection_policy": "stratified_salience_with_conditional_coverage_anchor",
    }


def _segment_record(
    segment: _Segment,
    *,
    projector: CanonicalSlotProjector,
    selected_slots_per_modality: int,
) -> dict[str, Any]:
    unit_to_atomic = {
        unit.evidence_id: item.evidence_id
        for item in segment.atomic
        for unit in item.units
    }
    groups: dict[tuple[str, str], list[EvidenceUnit]] = defaultdict(list)
    for item in segment.atomic:
        for unit in item.units:
            groups[
                (unit.modality, str(unit.source.get("source_id", "")))
            ].append(unit)
    sources = [
        _source_segment_payload(
            modality=modality,
            source_id=source_id,
            units=units,
            unit_to_atomic=unit_to_atomic,
            projector=projector,
            selected_slots_per_modality=selected_slots_per_modality,
        )
        for (modality, source_id), units in sorted(
            groups.items(),
            key=lambda item: (
                0 if item[0][0] == "audio" else 1,
                item[0][1],
            ),
        )
    ]
    record = {
        "schema_version": NATIVE_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "segment_id": segment.segment_id,
        "granularity": "segment",
        "time_basis": "session_relative_seconds",
        "time_range": {
            "start_sec": _finite_or_none(segment.start_sec),
            "end_sec": _finite_or_none(segment.end_sec),
        },
        "atomic_evidence_ids": [item.evidence_id for item in segment.atomic],
        "sources": sources,
    }
    _assert_label_free(record)
    return record


def _render_aggregated_slot(slot: Mapping[str, Any]) -> str:
    raw = slot.get("raw_median")
    unit = str(slot.get("unit") or "unitless")
    if unit == "ratio" and raw is not None:
        raw_text = f"{100.0 * float(raw):.1f}%"
    else:
        raw_text = f"{_format_number(raw)} {_display_unit(unit)}"
    details = [f"raw {raw_text}"]
    percentile = slot.get("median_percentile")
    if percentile is None:
        position = "p unavailable"
    else:
        position = f"p{int(round(float(percentile)))}"
    robust_z = slot.get("median_robust_z")
    if robust_z is not None:
        position += f" rz{_format_number(float(robust_z), 2)}"
    details.append(position)
    facet = str(slot.get("facet"))
    if facet != "level":
        details.append(f"facet {facet}")
    details.append(
        f"cov {slot['measured_window_count']}/"
        f"{slot['source_window_count']}"
    )
    mapping = {
        "directly_aligned": "direct",
        "semantic_proxy": "proxy",
        "provisional_mapping": "provisional",
    }[str(slot["mapping_quality"])]
    if mapping != "direct":
        details.append(f"map {mapping}")
    return " | ".join(details)


def _render_segment_record(record: Mapping[str, Any]) -> str:
    time_range = record["time_range"]
    lines = [
        (
            f"SEGMENT {record['segment_id']} | "
            f"t={_format_time(float(time_range['start_sec']))}-"
            f"{_format_time(float(time_range['end_sec']))}s | "
            f"refs={_short_ids(record['atomic_evidence_ids'])}"
        )
    ]
    for source in record["sources"]:
        modality = str(source["modality"]).upper()
        availability_counts = source["availability_window_counts"]
        source_fields = [f"{modality} {source['extractor']}"]
        if (
            len(availability_counts) != 1
            or "present" not in availability_counts
        ):
            counts = ",".join(
                f"{name.replace('_', ' ')}:{count}"
                for name, count in availability_counts.items()
            )
            source_fields.append(f"avail {counts}")
        valid_ratio = source.get("median_valid_ratio")
        if valid_ratio is None:
            source_fields.append("valid unavailable")
        elif float(valid_ratio) < 0.9995:
            source_fields.append(
                f"valid {100.0 * float(valid_ratio):.1f}%"
            )
        lines.append(" | ".join(source_fields))
        slots_by_id = {slot["slot_id"]: slot for slot in source["slots"]}
        selected = [
            slots_by_id[slot_id] for slot_id in source["selected_slot_ids"]
        ]
        if selected:
            lines.extend(
                f"- {slot['slot_label']} | {_render_aggregated_slot(slot)}"
                for slot in selected
            )
        else:
            lines.append(
                "- No measured canonical slot was selected; missingness remains "
                "explicit in the session coverage map."
            )
    rendered = "\n".join(lines)
    assert_semantically_safe(rendered)
    return rendered


def _coverage_lines(
    units: Sequence[EvidenceUnit],
    projector: CanonicalSlotProjector,
) -> list[str]:
    groups: dict[tuple[str, str], list[EvidenceUnit]] = defaultdict(list)
    for unit in units:
        groups[
            (unit.modality, str(unit.source.get("source_id", "")))
        ].append(unit)
    lines: list[str] = []
    for (modality, _), source_units in sorted(
        groups.items(),
        key=lambda item: (
            0 if item[0][0] == "audio" else 1,
            item[0][1],
        ),
    ):
        extractor = str(
            source_units[0].source.get("extractor", "unknown source")
        )
        fields: list[str] = []
        for definition in slots_for_modality(modality):
            projected = [
                next(
                    slot
                    for slot in projector.project(unit, extension_limit=0).slots
                    if slot.definition.slot_id == definition.slot_id
                )
                for unit in source_units
            ]
            measured = sum(slot.status == "measured" for slot in projected)
            supported = sum(slot.status != "not_measured" for slot in projected)
            if measured:
                state = f"M {measured}/{len(projected)}"
            elif supported:
                state = f"U {len(projected)}/{len(projected)}"
            else:
                state = "NM"
            fields.append(f"{definition.label}={state}")
        lines.append(
            f"- {modality.upper()} {extractor} | " + "; ".join(fields)
        )
    return lines


def _render_session_input(
    *,
    units: Sequence[EvidenceUnit],
    segments: Sequence[Mapping[str, Any]],
    projector: CanonicalSlotProjector,
) -> str:
    dataset = units[0].dataset if units else "unknown dataset"
    lines = [
        (
            f"SESSION EVIDENCE | protocol {PROTOCOL_VERSION} | dataset {dataset}"
        ),
        (
            "SCOPE | label-free summaries; full windows hidden/retrievable; "
            "slot rows inherit segment refs; exact refs are in the verified plan."
        ),
        (
            "SLOT ROW | name | raw segment median | p=train-fold percentile, "
            "rz=robust-z | optional facet | cov=measured/source | optional map."
        ),
        (
            "DEFAULTS | facet=level; map=direct (aligned); "
            "proxy=derived/approximate; provisional=unconfirmed."
        ),
        (
            "SOURCE DEFAULTS | omitted avail means all windows present; "
            "omitted valid means 100%; exceptions are explicit."
        ),
        (
            "SLOT COVERAGE | M=MEASURED; U=UNAVAILABLE; NM=NOT MEASURED; "
            "U/NM are not zeros."
        ),
        *_coverage_lines(units, projector),
        "SEGMENT EVIDENCE",
    ]
    for segment in segments:
        lines.append(_render_segment_record(segment))
    rendered = "\n".join(lines).rstrip() + "\n"
    assert_semantically_safe(rendered)
    return rendered


def _candidate_card(
    record: Mapping[str, Any],
    segment_id: str,
) -> dict[str, Any]:
    measured_slots: list[str] = []
    unusual = False
    quality_boundary = False
    modalities: list[str] = []
    source_ids: list[str] = []
    for source in record["sources"]:
        modalities.append(str(source["modality"]))
        source_ids.append(str(source["source_id"]))
        if source["availability"]["status"] != "present":
            quality_boundary = True
        for slot in source["slots"]:
            measurement = slot.get("measurement")
            if slot["status"] == "measured" and measurement is not None:
                measured_slots.append(str(slot["slot_label"]))
                percentile = measurement.get("percentile")
                if percentile is not None and abs(float(percentile) - 50.0) >= 25.0:
                    unusual = True
    cues: list[str] = []
    if unusual:
        cues.append("contains an unusual fold-local reference position")
    if quality_boundary:
        cues.append("contains an availability or quality boundary")
    if not cues:
        cues.append("representative measured window")
    return {
        "schema_version": NATIVE_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "segment_id": segment_id,
        "evidence_id": record["evidence_id"],
        "time_range": record["time_range"],
        "modalities": modalities,
        "source_ids": source_ids,
        "measured_slots": sorted(set(measured_slots)),
        "cue": "; ".join(cues),
        "numeric_measurements_disclosed": False,
    }


def _build_index(
    *,
    atomic_records: Sequence[Mapping[str, Any]],
    segment_records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    segment_by_evidence = {
        evidence_id: str(segment["segment_id"])
        for segment in segment_records
        for evidence_id in segment["atomic_evidence_ids"]
    }
    index: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []
    for record in atomic_records:
        evidence_id = str(record["evidence_id"])
        segment_id = segment_by_evidence[evidence_id]
        index.append(
            {
                "schema_version": NATIVE_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "segment_id": segment_id,
                "atomic_evidence_id": evidence_id,
                "time_range": record["time_range"],
                "canonical_evidence_ids": record["canonical_evidence_ids"],
                "modalities": [
                    source["modality"] for source in record["sources"]
                ],
                "source_ids": [
                    source["source_id"] for source in record["sources"]
                ],
            }
        )
        cards.append(_candidate_card(record, segment_id))
    return index, cards


def validate_native_artifacts(files: Mapping[str, str]) -> dict[str, Any]:
    required = {
        "canonical_evidence.native.jsonl",
        "atomic_evidence.native.txt",
        "segment_plan.native.jsonl",
        "segment_evidence.native.txt",
        "session_input.native.txt",
        "evidence_index.native.jsonl",
        "candidate_cards.native.jsonl",
        "verification.native.json",
    }
    if set(files) != required:
        raise ValueError(
            "native artifact inventory mismatch: "
            f"expected={sorted(required)}, observed={sorted(files)}"
        )

    def rows(name: str) -> list[dict[str, Any]]:
        output = [
            json.loads(line)
            for line in files[name].splitlines()
            if line.strip()
        ]
        if not all(isinstance(item, dict) for item in output):
            raise ValueError(f"{name} must contain JSON objects")
        return output

    atomic = rows("canonical_evidence.native.jsonl")
    segments = rows("segment_plan.native.jsonl")
    index = rows("evidence_index.native.jsonl")
    cards = rows("candidate_cards.native.jsonl")
    for payload in (*atomic, *segments, *index, *cards):
        _assert_label_free(payload)

    atomic_ids = [str(item["evidence_id"]) for item in atomic]
    if len(atomic_ids) != len(set(atomic_ids)):
        raise ValueError("native atomic evidence identifiers are not unique")
    index_ids = [str(item["atomic_evidence_id"]) for item in index]
    card_ids = [str(item["evidence_id"]) for item in cards]
    if atomic_ids != index_ids or atomic_ids != card_ids:
        raise ValueError("native store, index, and candidate-card IDs differ")
    segment_ids = [str(item["segment_id"]) for item in segments]
    if len(segment_ids) != len(set(segment_ids)):
        raise ValueError("native segment identifiers are not unique")
    referenced = [
        str(evidence_id)
        for segment in segments
        for evidence_id in segment["atomic_evidence_ids"]
    ]
    if referenced != atomic_ids:
        raise ValueError("native segment plan does not cover each atomic item once")

    expected_atomic_text = "\n".join(
        _render_atomic_record(item) for item in atomic
    )
    if expected_atomic_text:
        expected_atomic_text += "\n"
    if files["atomic_evidence.native.txt"] != expected_atomic_text:
        raise ValueError("native atomic text is not reproducible from CEU JSON")
    expected_segment_text = "\n\n".join(
        _render_segment_record(item) for item in segments
    )
    if expected_segment_text:
        expected_segment_text += "\n"
    if files["segment_evidence.native.txt"] != expected_segment_text:
        raise ValueError("native segment text is not reproducible from its plan")

    model_input = files["session_input.native.txt"]
    if not model_input.strip() or "SEGMENT EVIDENCE" not in model_input:
        raise ValueError("native first-pass model input is incomplete")
    assert_semantically_safe(model_input)
    if any(card.get("numeric_measurements_disclosed") is not False for card in cards):
        raise ValueError("native candidate cards disclose hidden numeric detail")
    report = json.loads(files["verification.native.json"])
    if report.get("passed") is not True:
        raise ValueError("native verification report is not passing")
    return {
        "passed": True,
        "atomic_count": len(atomic),
        "segment_count": len(segments),
        "candidate_card_count": len(cards),
        "label_fields_present": False,
        "numeric_render_byte_reproducible": True,
        "candidate_cards_hide_numeric_measurements": True,
        "all_atomic_items_indexed_once": True,
        "semantic_wording_safe": True,
    }


@dataclass(frozen=True)
class NativeEvidenceProtocol(EvidenceTextProtocol):
    """Compact first-pass evidence backed by a complete, verifiable atomic store."""

    segment_sec: float = 30.0
    max_segments_per_session: int = 8
    selected_slots_per_modality: int = 2
    atomic_extensions_per_modality: int = 0
    projector: CanonicalSlotProjector = field(
        default_factory=CanonicalSlotProjector,
        repr=False,
        compare=False,
    )
    version: str = PROTOCOL_VERSION

    @classmethod
    def from_config(cls, values: Mapping[str, Any]) -> "NativeEvidenceProtocol":
        output = cls(
            segment_sec=float(values.get("segment_sec", 30.0)),
            max_segments_per_session=int(
                values.get("max_segments_per_session", 8)
            ),
            selected_slots_per_modality=int(
                values.get("selected_slots_per_modality", 2)
            ),
            atomic_extensions_per_modality=int(
                values.get("atomic_extensions_per_modality", 0)
            ),
            projector=CanonicalSlotProjector.with_overrides(
                values.get("source_profiles")
            ),
        )
        output._validate_config()
        return output

    def _validate_config(self) -> None:
        if self.segment_sec <= 0:
            raise ValueError("protocol_native.segment_sec must be positive")
        if self.max_segments_per_session <= 0:
            raise ValueError(
                "protocol_native.max_segments_per_session must be positive"
            )
        if not 1 <= self.selected_slots_per_modality <= 5:
            raise ValueError(
                "protocol_native.selected_slots_per_modality must be 1 to 5"
            )
        if not 0 <= self.atomic_extensions_per_modality <= 4:
            raise ValueError(
                "protocol_native.atomic_extensions_per_modality must be 0 to 4"
            )

    @property
    def expected_filenames(self) -> tuple[str, ...]:
        return (
            "canonical_evidence.native.jsonl",
            "atomic_evidence.native.txt",
            "segment_plan.native.jsonl",
            "segment_evidence.native.txt",
            "session_input.native.txt",
            "evidence_index.native.jsonl",
            "candidate_cards.native.jsonl",
            "verification.native.json",
        )

    def compile_session(
        self, units: Sequence[EvidenceUnit]
    ) -> ProtocolArtifacts:
        self._validate_config()
        ordered = sorted(
            units,
            key=lambda item: (
                float(item.time_range["start_sec"]),
                float(item.time_range["end_sec"]),
                0 if item.modality == "audio" else 1,
                str(item.source.get("source_id", "")),
                item.evidence_id,
            ),
        )
        for unit in ordered:
            unit.validate()
            self.projector.project(
                unit,
                extension_limit=self.atomic_extensions_per_modality,
            )
        atomic = _group_atomic(ordered)
        segments, effective_segment_sec = _group_segments(
            atomic,
            segment_sec=self.segment_sec,
            max_segments=self.max_segments_per_session,
        )
        atomic_records = [
            _atomic_record(
                item,
                self.projector,
                self.atomic_extensions_per_modality,
            )
            for item in atomic
        ]
        segment_records = [
            _segment_record(
                item,
                projector=self.projector,
                selected_slots_per_modality=self.selected_slots_per_modality,
            )
            for item in segments
        ]
        index, cards = _build_index(
            atomic_records=atomic_records,
            segment_records=segment_records,
        )

        canonical_jsonl = "".join(_json_line(item) + "\n" for item in atomic_records)
        atomic_text = "\n".join(_render_atomic_record(item) for item in atomic_records)
        if atomic_text:
            atomic_text += "\n"
        segment_jsonl = "".join(
            _json_line(item) + "\n" for item in segment_records
        )
        segment_text = "\n\n".join(
            _render_segment_record(item) for item in segment_records
        )
        if segment_text:
            segment_text += "\n"
        session_input = _render_session_input(
            units=ordered,
            segments=segment_records,
            projector=self.projector,
        )
        index_jsonl = "".join(_json_line(item) + "\n" for item in index)
        cards_jsonl = "".join(_json_line(item) + "\n" for item in cards)
        verification = {
            "schema_version": NATIVE_SCHEMA_VERSION,
            "protocol_version": self.version,
            "passed": True,
            "checks": {
                "label_fields_present": False,
                "numeric_render_byte_reproducible": True,
                "candidate_cards_hide_numeric_measurements": True,
                "all_atomic_items_indexed_once": True,
                "missingness_is_explicit": True,
                "semantic_wording_safe": True,
                "raw_session_identifier_in_model_input": False,
            },
            "counts": {
                "atomic": len(atomic_records),
                "segments": len(segment_records),
                "candidate_cards": len(cards),
            },
            "selection": {
                "policy": "stratified_salience_with_conditional_coverage_anchor",
                "label_input": False,
                "selected_slots_per_modality": self.selected_slots_per_modality,
                "atomic_extensions_per_modality": (
                    self.atomic_extensions_per_modality
                ),
            },
        }
        verification_text = (
            json.dumps(
                verification,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        files = {
            "canonical_evidence.native.jsonl": canonical_jsonl,
            "atomic_evidence.native.txt": atomic_text,
            "segment_plan.native.jsonl": segment_jsonl,
            "segment_evidence.native.txt": segment_text,
            "session_input.native.txt": session_input,
            "evidence_index.native.jsonl": index_jsonl,
            "candidate_cards.native.jsonl": cards_jsonl,
            "verification.native.json": verification_text,
        }
        validation = validate_native_artifacts(files)
        output = ProtocolArtifacts(
            files=files,
            primary_input_file="session_input.native.txt",
            atomic_unit_count=len(atomic_records),
            segment_count=len(segment_records),
            metadata={
                "description": (
                    "compact label-free first-pass selection backed by a complete "
                    "raw-plus-reference atomic evidence store"
                ),
                "native_schema_version": NATIVE_SCHEMA_VERSION,
                "segment_sec": self.segment_sec,
                "effective_segment_sec": effective_segment_sec,
                "max_segments_per_session": self.max_segments_per_session,
                "selected_slots_per_modality": self.selected_slots_per_modality,
                "atomic_extensions_per_modality": (
                    self.atomic_extensions_per_modality
                ),
                "selection_policy": (
                    "stratified_salience_with_conditional_coverage_anchor"
                ),
                "selection_uses_labels": False,
                "raw_and_percentile_rendered_together": True,
                "first_pass_atomic_detail_hidden": True,
                "fixed_slot_coverage_preserved": True,
                "raw_session_identifier_in_model_input": False,
                "verification": validation,
                "bundle_sha256": hashlib.sha256(
                    "".join(
                        f"{name}\0{content}"
                        for name, content in sorted(files.items())
                    ).encode("utf-8")
                ).hexdigest(),
            },
        )
        output.validate()
        return output


__all__ = [
    "NativeEvidenceProtocol",
    "NATIVE_SCHEMA_VERSION",
    "PROTOCOL_VERSION",
    "validate_native_artifacts",
]
