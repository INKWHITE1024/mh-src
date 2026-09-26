"""Query-grounded atomic retrieval over native evidence bundles."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from rethink_mh.textualization.canonical_slots import CORE_SLOTS
from rethink_mh.textualization.protocols.native import (
    PROTOCOL_VERSION,
    validate_native_artifacts,
)

from .contracts import ContractValidationError
from .query_retrieval import (
    DEFAULT_MAX_CANDIDATES,
    MAX_QUERY_BUDGET,
    RETRIEVAL_PROTOCOL_VERSION,
    AtomicSelection,
    CandidateEvidence,
    CandidateSet,
    EvidenceQuery,
)
from .retrieval import TargetedEvidence


_NATIVE_FILES = (
    "canonical_evidence.native.jsonl",
    "atomic_evidence.native.txt",
    "segment_plan.native.jsonl",
    "segment_evidence.native.txt",
    "session_input.native.txt",
    "evidence_index.native.jsonl",
    "candidate_cards.native.jsonl",
    "verification.native.json",
)
_ATOMIC_LINE = re.compile(r"^EVIDENCE (?P<evidence_id>E[0-9]{3,6})(?: \||$)")
_IDENTIFIER = re.compile(r"^(?P<prefix>[A-Z]+)(?P<number>[0-9]+)$")
_SLOT_MODALITY = {slot.label: slot.modality for slot in CORE_SLOTS}
_MAPPING_WEIGHT = {
    "directly_aligned": 1.0,
    "semantic_proxy": 0.8,
    "provisional_mapping": 0.5,
}
_SEMANTIC_WEIGHT = {
    "direct": 1.0,
    "derived": 0.8,
    "weak": 0.55,
    "latent": 0.4,
}


def _identifier_key(value: str) -> tuple[str, int]:
    match = _IDENTIFIER.fullmatch(value)
    if match is None:
        return value, -1
    return match.group("prefix"), int(match.group("number"))


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    output: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON at {path}:{line_number}: {error.msg}"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain an object")
        output.append(value)
    return tuple(output)


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{path} must be a finite number")
    return parsed


def _atomic_text_by_id(text: str) -> dict[str, str]:
    output: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        match = _ATOMIC_LINE.match(line)
        if match is None:
            raise ValueError(
                "native atomic evidence must contain exactly one readable "
                f"record per line; invalid line {line_number}"
            )
        evidence_id = match.group("evidence_id")
        if evidence_id in output:
            raise ValueError(f"duplicate native atomic text ID: {evidence_id}")
        output[evidence_id] = line
    return output


def _mean(values: Sequence[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


@dataclass(frozen=True, slots=True)
class _NativeFeature:
    evidence_id: str
    segment_id: str
    start_sec: float
    end_sec: float
    canonical_ids: tuple[str, ...]
    slot_percentiles: Mapping[str, float]
    slot_reliabilities: Mapping[str, float]
    slot_modalities: Mapping[str, str]
    percentile_changes: Mapping[str, float]
    quality_score: float
    quality_boundary_score: float
    availability_changed: bool
    segment_medians: Mapping[str, float]

    @property
    def midpoint_sec(self) -> float:
        return (self.start_sec + self.end_sec) / 2.0


class NativeAtomicRetriever:
    """Native-bundle Adapter at the query-grounded retrieval seam."""

    evidence_protocol_version = PROTOCOL_VERSION

    def __init__(
        self,
        *,
        session_dir: Path,
        records_by_id: Mapping[str, Mapping[str, Any]],
        segment_records: Mapping[str, Mapping[str, Any]],
        atomic_text: Mapping[str, str],
        session_fingerprint: str,
        max_candidates_per_query: int = DEFAULT_MAX_CANDIDATES,
    ) -> None:
        if (
            isinstance(max_candidates_per_query, bool)
            or not isinstance(max_candidates_per_query, int)
            or not MAX_QUERY_BUDGET <= max_candidates_per_query <= 16
        ):
            raise ValueError(
                "max_candidates_per_query must be an integer between 4 and 16"
            )
        self._session_dir = session_dir
        self._records_by_id = dict(records_by_id)
        self._segment_records = dict(segment_records)
        self._atomic_text = dict(atomic_text)
        self._session_fingerprint = session_fingerprint
        self._max_candidates_per_query = max_candidates_per_query
        self._features_by_segment = self._build_features()
        self._available_segment_ids = tuple(
            sorted(self._features_by_segment, key=_identifier_key)
        )
        self._all_atomic_ids = frozenset(self._records_by_id)

    @classmethod
    def from_session_dir(
        cls,
        path: str | Path,
        *,
        max_candidates_per_query: int = DEFAULT_MAX_CANDIDATES,
    ) -> "NativeAtomicRetriever":
        directory = Path(path).expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(
                f"native evidence session directory does not exist: {directory}"
            )
        missing = [
            filename
            for filename in _NATIVE_FILES
            if not (directory / filename).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Not a complete Evidence native session; "
                f"missing: {missing}"
            )
        files = {
            filename: (directory / filename).read_text(encoding="utf-8")
            for filename in _NATIVE_FILES
        }
        validate_native_artifacts(files)

        atomic_records = _read_jsonl(
            directory / "canonical_evidence.native.jsonl"
        )
        segment_rows = _read_jsonl(
            directory / "segment_plan.native.jsonl"
        )
        records_by_id = {
            str(record.get("evidence_id", "")): record
            for record in atomic_records
        }
        segment_records = {
            str(record.get("segment_id", "")): record
            for record in segment_rows
        }
        if len(records_by_id) != len(atomic_records):
            raise ValueError("native atomic records contain duplicate identifiers")
        if len(segment_records) != len(segment_rows):
            raise ValueError("native segment plan contains duplicate identifiers")

        atomic_text = _atomic_text_by_id(
            files["atomic_evidence.native.txt"]
        )
        if set(atomic_text) != set(records_by_id):
            raise ValueError(
                "native readable atomic text and canonical atomic IDs differ"
            )
        digest = hashlib.sha256()
        for filename in _NATIVE_FILES:
            digest.update(filename.encode("utf-8"))
            digest.update(b"\0")
            digest.update(files[filename].encode("utf-8"))
            digest.update(b"\0")
        return cls(
            session_dir=directory,
            records_by_id=records_by_id,
            segment_records=segment_records,
            atomic_text=atomic_text,
            session_fingerprint=digest.hexdigest(),
            max_candidates_per_query=max_candidates_per_query,
        )

    @property
    def available_segment_ids(self) -> tuple[str, ...]:
        return self._available_segment_ids

    @property
    def session_fingerprint(self) -> str:
        return self._session_fingerprint

    @property
    def session_input_text(self) -> str:
        return (
            self._session_dir / "session_input.native.txt"
        ).read_text(encoding="utf-8")

    def render_retrieval_map(
        self,
        segment_ids: Sequence[str] | None = None,
    ) -> str:
        selected = self._validate_segments(segment_ids)
        lines = [
            (
                "ATOMIC RETRIEVAL MAP | evidence protocol "
                f"{PROTOCOL_VERSION} | retrieval protocol "
                f"{RETRIEVAL_PROTOCOL_VERSION}"
            ),
            "- This map exposes query affordances, not atomic identifiers or values.",
            "- Change and cross-modal cues are measurement relations, not task conclusions.",
        ]
        for segment_id in selected:
            features = self._features_by_segment[segment_id]
            available_slots = tuple(
                slot.label
                for slot in CORE_SLOTS
                if any(slot.label in item.slot_percentiles for item in features)
            )
            unusual = sum(
                self._extreme_score(item, available_slots) >= 0.5
                for item in features
            )
            changes = sum(
                self._change_score(item, available_slots) >= 0.3
                for item in features
            )
            quality = sum(
                item.quality_boundary_score >= 0.2
                or item.availability_changed
                for item in features
            )
            cross_modal = sum(
                self._cross_modal_score(item, available_slots) >= 0.3
                for item in features
            )
            lines.extend(
                (
                    "",
                    f"SEGMENT {segment_id}",
                    "- available fixed core slots: "
                    + (", ".join(available_slots) if available_slots else "none"),
                    f"- unusual-measurement candidates: {unusual}",
                    f"- change-point candidates: {changes}",
                    f"- quality-boundary candidates: {quality}",
                    f"- representative-window candidates: {len(features)}",
                    f"- cross-modal co-change candidates: {cross_modal}",
                )
            )
        text = "\n".join(lines).rstrip() + "\n"
        if re.search(r"\bE[0-9]{3,6}\b", text):
            raise AssertionError("native retrieval map disclosed an atomic identifier")
        return text

    def shortlist(
        self,
        query: EvidenceQuery | Mapping[str, Any],
        *,
        exclude_atomic_ids: Sequence[str] = (),
    ) -> CandidateSet:
        parsed = (
            query
            if isinstance(query, EvidenceQuery)
            else EvidenceQuery.from_dict(query)
        )
        if parsed.segment_id not in self._features_by_segment:
            raise ValueError(
                f"Evidence query names an unknown segment: {parsed.segment_id}"
            )
        excluded = self._validate_exclusions(exclude_atomic_ids)
        features = tuple(
            item
            for item in self._features_by_segment[parsed.segment_id]
            if item.evidence_id not in excluded
        )
        if not features:
            raise ValueError(
                f"No unseen atomic evidence remains in {parsed.segment_id}"
            )
        supported = {
            slot for item in features for slot in item.slot_percentiles
        }
        missing_slots = set(parsed.target_slots) - supported
        if missing_slots:
            raise ValueError(
                "Evidence query targets slots with no measured atomic evidence "
                f"in {parsed.segment_id}: {sorted(missing_slots)}"
            )

        ranked: list[tuple[float, _NativeFeature, dict[str, float]]] = []
        for item in features:
            components = self._score_components(item, parsed)
            if components["slot coverage"] <= 0.0:
                continue
            score = (
                0.70 * components["pattern relevance"]
                + 0.20 * components["slot coverage"]
                + 0.10 * components["measurement reliability"]
            )
            ranked.append((score, item, components))
        if not ranked:
            raise ValueError(
                "No atomic evidence contains any requested fixed core slot"
            )
        candidate_limit = min(
            self._max_candidates_per_query,
            len(ranked),
            max(parsed.budget * 2, parsed.budget + 2),
        )
        chosen = self._select_diverse(ranked, candidate_limit)
        candidates: list[CandidateEvidence] = []
        for rank, (score, item, components) in enumerate(chosen, 1):
            slots = tuple(
                slot
                for slot in parsed.target_slots
                if slot in item.slot_percentiles
            )
            candidates.append(
                CandidateEvidence(
                    rank=rank,
                    evidence_id=item.evidence_id,
                    segment_id=item.segment_id,
                    start_sec=item.start_sec,
                    end_sec=item.end_sec,
                    cues=self._candidate_cues(item, parsed),
                    target_slots_present=slots,
                    quality_summary=self._quality_summary(item.quality_score),
                    ranking_score=score,
                    score_components=tuple(
                        (name, components[name])
                        for name in (
                            "pattern relevance",
                            "slot coverage",
                            "measurement reliability",
                        )
                    ),
                )
            )
        candidate_tuple = tuple(candidates)
        return CandidateSet(
            query=parsed,
            candidates=candidate_tuple,
            text=self._render_candidate_cards(parsed, candidate_tuple),
            session_fingerprint=self._session_fingerprint,
        )

    def materialize(
        self,
        candidate_set: CandidateSet,
        selection: AtomicSelection | Mapping[str, Any],
    ) -> TargetedEvidence:
        if not isinstance(candidate_set, CandidateSet):
            raise TypeError("candidate_set must be a CandidateSet")
        if candidate_set.session_fingerprint != self._session_fingerprint:
            raise ValueError(
                "candidate_set belongs to a different evidence session"
            )
        candidates = candidate_set.candidate_evidence_ids
        if len(candidates) != len(set(candidates)):
            raise ValueError("candidate_set contains duplicate atomic identifiers")
        unknown_candidates = set(candidates) - set(self._records_by_id)
        if unknown_candidates:
            raise ValueError(
                "candidate_set contains identifiers outside this evidence session: "
                f"{sorted(unknown_candidates)}"
            )
        for candidate in candidate_set.candidates:
            feature = next(
                (
                    item
                    for item in self._features_by_segment[
                        candidate_set.query.segment_id
                    ]
                    if item.evidence_id == candidate.evidence_id
                ),
                None,
            )
            if (
                feature is None
                or candidate.segment_id != feature.segment_id
                or not math.isclose(
                    candidate.start_sec,
                    feature.start_sec,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    candidate.end_sec,
                    feature.end_sec,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                raise ValueError(
                    "candidate_set metadata does not match native indexed evidence "
                    f"for {candidate.evidence_id}"
                )
        selected = (
            selection
            if isinstance(selection, AtomicSelection)
            else AtomicSelection.from_dict(selection)
        )
        if len(selected.selected_evidence_ids) > candidate_set.query.budget:
            raise ContractValidationError(
                "atomic selection exceeds the query budget"
            )
        not_shown = set(selected.selected_evidence_ids) - set(candidates)
        if not_shown:
            raise ContractValidationError(
                "atomic selection contains identifiers that were not shown in "
                f"the candidate cards: {sorted(not_shown)}"
            )

        records = [
            self._records_by_id[evidence_id]
            for evidence_id in selected.selected_evidence_ids
        ]
        segments = tuple(
            dict.fromkeys(
                self._segment_for_evidence(evidence_id)
                for evidence_id in selected.selected_evidence_ids
            )
        )
        canonical_ids = tuple(
            str(canonical_id)
            for record in records
            for canonical_id in record["canonical_evidence_ids"]
        )
        query = candidate_set.query
        lines = [
            (
                "QUERY-GUIDED ATOMIC EVIDENCE | evidence protocol "
                f"{PROTOCOL_VERSION} | retrieval protocol "
                f"{RETRIEVAL_PROTOCOL_VERSION}"
            ),
            "- Values are released only after a grounded, budgeted selection.",
            "- Missing measurements are not zeros; relations are not task conclusions.",
            f"- review purpose: {query.purpose}",
            "- target fixed core slots: " + ", ".join(query.target_slots),
            f"- requested temporal pattern: {query.pattern}",
            "- selected atomic identifiers: "
            + ", ".join(selected.selected_evidence_ids),
        ]
        for evidence_id, segment_id in zip(
            selected.selected_evidence_ids,
            (self._segment_for_evidence(item) for item in selected.selected_evidence_ids),
        ):
            lines.extend(
                (
                    "",
                    f"SEGMENT {segment_id}",
                    self._atomic_text[evidence_id],
                )
            )
        return TargetedEvidence(
            text="\n".join(lines).rstrip() + "\n",
            selected_segment_ids=segments,
            selected_atomic_evidence_ids=selected.selected_evidence_ids,
            canonical_evidence_ids=canonical_ids,
        )

    def _build_features(self) -> dict[str, tuple[_NativeFeature, ...]]:
        evidence_to_segment = {
            str(evidence_id): segment_id
            for segment_id, segment in self._segment_records.items()
            for evidence_id in segment["atomic_evidence_ids"]
        }
        by_segment: dict[str, list[dict[str, Any]]] = {
            segment_id: [] for segment_id in self._segment_records
        }
        for evidence_id, record in self._records_by_id.items():
            segment_id = evidence_to_segment.get(evidence_id)
            if segment_id is None:
                raise ValueError(
                    f"native atomic evidence is absent from the segment plan: {evidence_id}"
                )
            by_segment[segment_id].append(record)

        output: dict[str, tuple[_NativeFeature, ...]] = {}
        for segment_id, records in by_segment.items():
            ordered = sorted(
                records,
                key=lambda record: (
                    float(record["time_range"]["start_sec"]),
                    _identifier_key(str(record["evidence_id"])),
                ),
            )
            base: list[
                tuple[
                    dict[str, Any],
                    dict[str, float],
                    dict[str, float],
                    dict[str, str],
                    float,
                    tuple[str, ...],
                ]
            ] = []
            for record in ordered:
                slot_values: dict[str, list[float]] = {}
                slot_reliability: dict[str, list[float]] = {}
                slot_modalities: dict[str, str] = {}
                quality_values: list[float] = []
                availability: list[str] = []
                for source in record["sources"]:
                    modality = str(source["modality"])
                    status = str(source["availability"]["status"])
                    availability.append(f"{modality}:{status}")
                    valid_ratio = source["quality"].get("valid_ratio")
                    quality = (
                        min(
                            1.0,
                            max(
                                0.0,
                                _finite_number(
                                    valid_ratio,
                                    f"{record['evidence_id']} valid_ratio",
                                ),
                            ),
                        )
                        if valid_ratio is not None
                        else (1.0 if status == "present" else 0.5)
                    )
                    quality_values.append(quality)
                    for slot in source["slots"]:
                        measurement = slot.get("measurement")
                        if (
                            slot.get("status") != "measured"
                            or not isinstance(measurement, Mapping)
                            or measurement.get("percentile") is None
                        ):
                            continue
                        label = str(slot["slot_label"])
                        percentile = _finite_number(
                            measurement["percentile"],
                            f"{record['evidence_id']} {label} percentile",
                        )
                        slot_values.setdefault(label, []).append(percentile)
                        mapping_weight = _MAPPING_WEIGHT.get(
                            str(slot.get("mapping_quality")),
                            0.35,
                        )
                        semantic_weight = _SEMANTIC_WEIGHT.get(
                            str(measurement.get("semantic_reliability")),
                            0.35,
                        )
                        slot_reliability.setdefault(label, []).append(
                            mapping_weight * semantic_weight * quality
                        )
                        slot_modalities[label] = modality
                base.append(
                    (
                        record,
                        {
                            label: float(median(values))
                            for label, values in slot_values.items()
                        },
                        {
                            label: float(median(values))
                            for label, values in slot_reliability.items()
                        },
                        slot_modalities,
                        _mean(quality_values, 0.0),
                        tuple(availability),
                    )
                )
            segment_medians = {
                slot.label: float(median(values))
                for slot in CORE_SLOTS
                if (
                    values := [
                        slot_values[slot.label]
                        for _, slot_values, _, _, _, _ in base
                        if slot.label in slot_values
                    ]
                )
            }
            features: list[_NativeFeature] = []
            for index, (
                record,
                slot_values,
                slot_reliability,
                slot_modalities,
                quality,
                availability,
            ) in enumerate(base):
                previous = base[index - 1] if index else None
                changes: dict[str, float] = {}
                quality_boundary = 0.0
                availability_changed = False
                if previous is not None:
                    previous_values = previous[1]
                    changes = {
                        label: (value - previous_values[label]) / 100.0
                        for label, value in slot_values.items()
                        if label in previous_values
                    }
                    quality_boundary = abs(quality - previous[4])
                    availability_changed = availability != previous[5]
                    if availability_changed:
                        quality_boundary = max(quality_boundary, 0.65)
                time_range = record["time_range"]
                features.append(
                    _NativeFeature(
                        evidence_id=str(record["evidence_id"]),
                        segment_id=segment_id,
                        start_sec=_finite_number(
                            time_range["start_sec"], "native start_sec"
                        ),
                        end_sec=_finite_number(
                            time_range["end_sec"], "native end_sec"
                        ),
                        canonical_ids=tuple(
                            str(item) for item in record["canonical_evidence_ids"]
                        ),
                        slot_percentiles=slot_values,
                        slot_reliabilities=slot_reliability,
                        slot_modalities=slot_modalities,
                        percentile_changes=changes,
                        quality_score=quality,
                        quality_boundary_score=quality_boundary,
                        availability_changed=availability_changed,
                        segment_medians=segment_medians,
                    )
                )
            output[segment_id] = tuple(features)
        return output

    @staticmethod
    def _extreme_score(
        item: _NativeFeature,
        target_slots: Sequence[str],
    ) -> float:
        return max(
            (
                abs(item.slot_percentiles[slot] - 50.0) / 50.0
                for slot in target_slots
                if slot in item.slot_percentiles
            ),
            default=0.0,
        )

    @staticmethod
    def _change_score(
        item: _NativeFeature,
        target_slots: Sequence[str],
    ) -> float:
        return min(
            1.0,
            max(
                (
                    abs(item.percentile_changes[slot])
                    for slot in target_slots
                    if slot in item.percentile_changes
                ),
                default=0.0,
            ),
        )

    @staticmethod
    def _cross_modal_score(
        item: _NativeFeature,
        target_slots: Sequence[str],
    ) -> float:
        audio = [
            abs(item.percentile_changes[slot])
            for slot in target_slots
            if _SLOT_MODALITY.get(slot) == "audio"
            and slot in item.percentile_changes
        ]
        visual = [
            abs(item.percentile_changes[slot])
            for slot in target_slots
            if _SLOT_MODALITY.get(slot) == "visual"
            and slot in item.percentile_changes
        ]
        return min(1.0, min(max(audio, default=0.0), max(visual, default=0.0)))

    @staticmethod
    def _representative_score(
        item: _NativeFeature,
        target_slots: Sequence[str],
    ) -> float:
        distances = [
            abs(item.slot_percentiles[slot] - item.segment_medians[slot]) / 50.0
            for slot in target_slots
            if slot in item.slot_percentiles and slot in item.segment_medians
        ]
        return (
            max(0.0, 1.0 - _mean(distances))
            if distances
            else 0.0
        )

    def _score_components(
        self,
        item: _NativeFeature,
        query: EvidenceQuery,
    ) -> dict[str, float]:
        present = [
            slot
            for slot in query.target_slots
            if slot in item.slot_percentiles
        ]
        coverage = len(present) / len(query.target_slots)
        reliability = _mean(
            [item.slot_reliabilities.get(slot, 0.0) for slot in present],
            0.0,
        )
        patterns = {
            "unusual measurement": self._extreme_score(
                item, query.target_slots
            ),
            "change point": self._change_score(item, query.target_slots),
            "quality boundary": item.quality_boundary_score,
            "representative window": self._representative_score(
                item, query.target_slots
            ),
            "cross-modal co-change": self._cross_modal_score(
                item, query.target_slots
            ),
        }
        return {
            "pattern relevance": min(1.0, max(0.0, patterns[query.pattern])),
            "slot coverage": coverage,
            "measurement reliability": min(1.0, max(0.0, reliability)),
        }

    @staticmethod
    def _select_diverse(
        ranked: Sequence[
            tuple[float, _NativeFeature, dict[str, float]]
        ],
        limit: int,
    ) -> tuple[
        tuple[float, _NativeFeature, dict[str, float]], ...
    ]:
        remaining = list(ranked)
        selected: list[
            tuple[float, _NativeFeature, dict[str, float]]
        ] = []
        while remaining and len(selected) < limit:
            best = max(
                remaining,
                key=lambda candidate: (
                    candidate[0]
                    - 0.12
                    * NativeAtomicRetriever._temporal_redundancy(
                        candidate[1],
                        [item[1] for item in selected],
                    ),
                    candidate[0],
                    -candidate[1].midpoint_sec,
                    -_identifier_key(candidate[1].evidence_id)[1],
                ),
            )
            selected.append(best)
            remaining.remove(best)
        return tuple(selected)

    @staticmethod
    def _temporal_redundancy(
        candidate: _NativeFeature,
        selected: Sequence[_NativeFeature],
    ) -> float:
        if not selected:
            return 0.0
        duration = max(1.0, candidate.end_sec - candidate.start_sec)
        return max(
            max(
                0.0,
                1.0
                - abs(candidate.midpoint_sec - item.midpoint_sec)
                / (2.0 * duration),
            )
            for item in selected
        )

    def _candidate_cues(
        self,
        item: _NativeFeature,
        query: EvidenceQuery,
    ) -> tuple[str, ...]:
        if query.pattern == "quality boundary":
            first = (
                "measurement availability changes near this window"
                if item.availability_changed
                else "measurement quality changes near this window"
            )
        elif query.pattern == "representative window":
            first = "target slots are representative of this segment"
        elif query.pattern == "cross-modal co-change":
            first = "audio and visual target slots have aligned temporal changes"
        elif query.pattern == "change point":
            first = "target slots change from an earlier window"
        else:
            first = "a target slot is unusual relative to the fold reference"
        cues = [first]
        if item.quality_score < 0.8:
            cues.append("measurement quality is incomplete and requires review")
        return tuple(cues)

    @staticmethod
    def _quality_summary(score: float) -> str:
        if score >= 0.8:
            return "good measured coverage"
        if score >= 0.4:
            return "partial measured coverage"
        return "limited measured coverage"

    @staticmethod
    def _render_candidate_cards(
        query: EvidenceQuery,
        candidates: Sequence[CandidateEvidence],
    ) -> str:
        lines = [
            (
                "ATOMIC EVIDENCE CANDIDATE CARDS | evidence protocol "
                f"{PROTOCOL_VERSION} | retrieval protocol "
                f"{RETRIEVAL_PROTOCOL_VERSION}"
            ),
            "- Cards reveal addresses, cues, and quality; values remain hidden.",
            "- Select only identifiers shown below and stay within the query budget.",
            f"- query segment: {query.segment_id}",
            f"- review purpose: {query.purpose}",
            "- target fixed core slots: " + ", ".join(query.target_slots),
            f"- requested temporal pattern: {query.pattern}",
            f"- maximum selectable atomic records: {query.budget}",
            "",
            "CANDIDATES",
        ]
        for item in sorted(
            candidates,
            key=lambda candidate: (
                candidate.start_sec,
                candidate.end_sec,
                _identifier_key(candidate.evidence_id),
            ),
        ):
            lines.append(
                f"{item.evidence_id} | time {item.start_sec:g} to "
                f"{item.end_sec:g} seconds | retrieval cue: "
                + "; ".join(item.cues)
                + " | requested slots present: "
                + ", ".join(item.target_slots_present)
                + f" | quality: {item.quality_summary}"
            )
        text = "\n".join(lines).rstrip() + "\n"
        forbidden = ("raw value", "percentile ", "robust z", "ranking_score")
        if any(value in text for value in forbidden):
            raise AssertionError(
                "native candidate cards disclosed hidden measurements or scores"
            )
        return text

    def _validate_segments(
        self,
        segment_ids: Sequence[str] | None,
    ) -> tuple[str, ...]:
        if segment_ids is None:
            return self._available_segment_ids
        if isinstance(segment_ids, (str, bytes)):
            raise TypeError("segment_ids must be a sequence")
        values = tuple(segment_ids)
        if not values:
            raise ValueError("At least one segment identifier is required")
        if len(values) != len(set(values)):
            raise ValueError("segment_ids cannot contain duplicates")
        unknown = set(values) - set(self._available_segment_ids)
        if unknown:
            raise ValueError(f"Unknown segment identifiers: {sorted(unknown)}")
        return values

    def _validate_exclusions(
        self,
        values: Sequence[str],
    ) -> frozenset[str]:
        if isinstance(values, (str, bytes)):
            raise TypeError("exclude_atomic_ids must be a sequence")
        identifiers = tuple(values)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("exclude_atomic_ids cannot contain duplicates")
        unknown = set(identifiers) - self._all_atomic_ids
        if unknown:
            raise ValueError(
                f"Unknown excluded atomic identifiers: {sorted(unknown)}"
            )
        return frozenset(identifiers)

    def _segment_for_evidence(self, evidence_id: str) -> str:
        for segment_id, segment in self._segment_records.items():
            if evidence_id in segment["atomic_evidence_ids"]:
                return segment_id
        raise AssertionError(f"unindexed native evidence: {evidence_id}")


__all__ = ["NativeAtomicRetriever"]
