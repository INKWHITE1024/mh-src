"""Label-safe information boundary between initial assessment and review passes."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from rethink_mh.textualization.renderer import (
    DISPLAY_NAMES,
    OBSERVATION_PRIORITY,
    assert_semantically_safe,
    format_time,
)
from rethink_mh.textualization.schema import EvidenceUnit, Observation

from .retrieval import (
    EvidenceStore,
    TargetedEvidence,
    _IndexRecord,
    _natural_identifier_key,
    _record_salience,
)


_SEGMENT_ID = re.compile(r"S\d+\Z")
_ATOMIC_ID = re.compile(r"E\d+\Z")
_TRANSCRIPT_ID = re.compile(r"T\d+\Z")
_NORMALIZED_LABEL_KEYS = frozenset(
    {
        "label",
        "target",
        "groundtruth",
        "goldlabel",
        "samplelabel",
        "targetlabel",
        "truelabel",
    }
)

# Fixed task-level retrieval cues.  Matching never reads the current sample's
# target.  Family names, rather than matched words or utterance text, appear in
# the initial view.
_CUE_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "sleep or rest words",
        ("sleep", "slept", "insomnia", "nightmare", "tired", "rest"),
    ),
    (
        "mood or distress words",
        (
            "depressed",
            "unhappy",
            "upset",
            "cry",
            "crying",
            "irritated",
            "irritable",
        ),
    ),
    (
        "interest or activity words",
        ("enjoy", "interest", "interested", "fun", "hobby", "motivation"),
    ),
    (
        "energy words",
        ("energy", "exhausted", "fatigue", "fatigued", "lazy"),
    ),
    (
        "appetite or weight words",
        ("appetite", "eating", "food", "weight"),
    ),
    (
        "self-evaluation words",
        ("failure", "worthless", "guilty", "guilt", "blame", "proud"),
    ),
    (
        "attention words",
        ("focus", "focused", "concentrate", "concentration", "attention"),
    ),
    (
        "movement or tension words",
        ("restless", "fidget", "fidgeting", "agitated", "slowed"),
    ),
    (
        "safety-related words",
        ("suicide", "suicidal", "kill myself", "hurt myself", "self harm"),
    ),
)


@dataclass(frozen=True, slots=True)
class EvidenceHierarchyConfig:
    """Budgets for the two information views."""

    initial_max_segments: int = 12
    review_max_segments: int = 4
    review_max_atomic_per_segment: int = 2
    review_max_observations_per_modality: int = 2
    review_max_utterances_per_segment: int = 2
    review_max_characters_per_utterance: int = 360

    def __post_init__(self) -> None:
        limits = {
            "initial_max_segments": (self.initial_max_segments, 1, 64),
            "review_max_segments": (self.review_max_segments, 1, 16),
            "review_max_atomic_per_segment": (
                self.review_max_atomic_per_segment,
                1,
                64,
            ),
            "review_max_observations_per_modality": (
                self.review_max_observations_per_modality,
                1,
                32,
            ),
            "review_max_utterances_per_segment": (
                self.review_max_utterances_per_segment,
                1,
                16,
            ),
            "review_max_characters_per_utterance": (
                self.review_max_characters_per_utterance,
                80,
                2_000,
            ),
        }
        for name, (value, lower, upper) in limits.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not lower <= value <= upper:
                raise ValueError(
                    f"{name} must be between {lower} and {upper}, inclusive"
                )


@dataclass(frozen=True, slots=True)
class InitialEvidenceView:
    """First-pass evidence with no atomic measurements or transcript wording."""

    text: str
    visible_segment_ids: tuple[str, ...]
    transcript_index_segment_ids: tuple[str, ...]
    disclosed_atomic_evidence_ids: tuple[str, ...] = ()
    disclosed_transcript_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ReviewQuery:
    """Segments to inspect and evidence identifiers already seen in earlier reviews."""

    priority_segment_ids: tuple[str, ...] = ()
    seen_atomic_evidence_ids: tuple[str, ...] = ()
    seen_transcript_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_identifiers(
            "priority_segment_ids", self.priority_segment_ids, _SEGMENT_ID
        )
        _validate_identifiers(
            "seen_atomic_evidence_ids",
            self.seen_atomic_evidence_ids,
            _ATOMIC_ID,
        )
        _validate_identifiers(
            "seen_transcript_ids", self.seen_transcript_ids, _TRANSCRIPT_ID
        )


@dataclass(frozen=True, slots=True)
class ReviewEvidenceView:
    """New detail released for one review turn."""

    text: str
    selected_segment_ids: tuple[str, ...]
    selected_atomic_evidence_ids: tuple[str, ...]
    selected_transcript_ids: tuple[str, ...]
    canonical_evidence_ids: tuple[str, ...]
    excluded_seen_item_count: int
    newly_revealed_item_count: int
    novelty_fraction: float


@dataclass(frozen=True, slots=True)
class _TranscriptRecord:
    utterance_id: str
    start_sec: float
    end_sec: float
    text: str
    speaker_role: str

    @property
    def midpoint_sec(self) -> float:
        return (self.start_sec + self.end_sec) / 2.0


class HierarchicalEvidenceStore:
    """Deep interface that owns selection, withholding, and readable rendering."""

    def __init__(
        self,
        *,
        evidence_store: EvidenceStore,
        transcript_records: Sequence[_TranscriptRecord],
        config: EvidenceHierarchyConfig,
    ) -> None:
        self._evidence_store = evidence_store
        self._config = config
        self._records_by_segment = evidence_store._records_by_segment
        self._canonical_by_id = evidence_store._canonical_by_id
        self._segment_ranges = _derive_segment_ranges(
            self._records_by_segment
        )
        self._transcript_by_id = {
            record.utterance_id: record for record in transcript_records
        }
        self._transcript_by_segment = self._align_transcript(transcript_records)

    @classmethod
    def from_session_dirs(
        cls,
        evidence_dir: str | Path,
        transcript_dir: str | Path | None = None,
        *,
        config: EvidenceHierarchyConfig | None = None,
        protocol_version: str = "native",
    ) -> "HierarchicalEvidenceStore":
        evidence_store = EvidenceStore.from_session_dir(
            evidence_dir, protocol_version=protocol_version
        )
        transcript_records = (
            _load_transcript_records(Path(transcript_dir))
            if transcript_dir is not None
            else ()
        )
        return cls(
            evidence_store=evidence_store,
            transcript_records=transcript_records,
            config=config or EvidenceHierarchyConfig(),
        )

    @property
    def available_segment_ids(self) -> tuple[str, ...]:
        return self._evidence_store.available_segment_ids

    def initial_view(self) -> InitialEvidenceView:
        """Render coarse aggregates and a wording-free transcript index."""

        segment_ids = self._select_initial_segments(
            self._config.initial_max_segments
        )
        lines = [
            "HIERARCHICAL INITIAL EVIDENCE VIEW | readable protocol h1",
            "INFORMATION BOUNDARY",
            "- This view contains segment-level audio and visual aggregates, not "
            "atomic-window measurements.",
            "- Transcript entries reveal only fixed lexical cue-family names and "
            "counts, not utterance wording.",
            "- Low or high means position in a source-specific training reference, "
            "not a human state or a task label.",
            "- Missing information is not a measured zero. Exact windows and wording "
            "may be requested by S identifier for a later review.",
            "",
            "SEGMENT OVERVIEWS",
        ]
        for segment_id in segment_ids:
            lines.extend(self._render_segment_overview(segment_id))

        lines.extend(("", "TRANSCRIPT RETRIEVAL INDEX"))
        indexed: list[str] = []
        for segment_id in segment_ids:
            records = self._transcript_by_segment.get(segment_id, ())
            if records:
                indexed.append(segment_id)
            family_counts = _cue_family_counts(records)
            cue_text = (
                ", ".join(
                    f"{family} in {count} utterance"
                    f"{'s' if count != 1 else ''}"
                    for family, count in family_counts
                )
                if family_counts
                else "none detected"
            )
            lines.append(
                f"TRANSCRIPT INDEX {segment_id} | {len(records)} retained "
                f"utterances | lexical cue families: {cue_text} | exact wording "
                "withheld until review."
            )
        if not self._transcript_by_id:
            lines.append(
                "Transcript source unavailable; no text was substituted for it."
            )
        lines.append(
            "Cue-family matches are retrieval hints only and are not conclusions "
            "about the participant."
        )

        text = "\n".join(lines).rstrip() + "\n"
        assert_semantically_safe(text)
        if re.search(r"\bE\d+\b|\bT\d+\b", text):
            raise ValueError(
                "initial evidence view must not disclose atomic or transcript IDs"
            )
        return InitialEvidenceView(
            text=text,
            visible_segment_ids=segment_ids,
            transcript_index_segment_ids=tuple(indexed),
        )

    def review(self, query: ReviewQuery) -> ReviewEvidenceView:
        """Release only unseen atomic windows and aligned transcript wording."""

        selected_segments = self._review_segments(query.priority_segment_ids)
        unknown_seen_transcript = (
            set(query.seen_transcript_ids) - set(self._transcript_by_id)
        )
        if unknown_seen_transcript:
            raise ValueError(
                "Unknown previously seen transcript identifiers: "
                + ", ".join(sorted(unknown_seen_transcript))
            )

        targeted = self._evidence_store.render_segments(
            selected_segments,
            max_atomic_per_segment=self._config.review_max_atomic_per_segment,
            max_observations_per_modality=(
                self._config.review_max_observations_per_modality
            ),
            exclude_atomic_ids=query.seen_atomic_evidence_ids,
            selection_strategy="salience_time_spanning",
        )
        transcript_text, transcript_ids = self._render_transcript_review(
            selected_segments,
            excluded=frozenset(query.seen_transcript_ids),
        )
        selected_items = (
            len(targeted.selected_atomic_evidence_ids) + len(transcript_ids)
        )
        novelty_fraction = 1.0 if selected_items else 0.0
        text = "\n".join(
            (
                "HIERARCHICAL REVIEW EVIDENCE | newly disclosed detail only",
                "Previously reviewed E and T identifiers were excluded before "
                "selection. Transcript wording below is observed source content, "
                "not a task label or conclusion.",
                "",
                targeted.text.rstrip(),
                "",
                transcript_text.rstrip(),
            )
        ).rstrip() + "\n"
        if set(targeted.selected_atomic_evidence_ids).intersection(
            query.seen_atomic_evidence_ids
        ):
            raise AssertionError("review repeated an already seen atomic window")
        if set(transcript_ids).intersection(query.seen_transcript_ids):
            raise AssertionError("review repeated an already seen transcript item")
        return ReviewEvidenceView(
            text=text,
            selected_segment_ids=selected_segments,
            selected_atomic_evidence_ids=targeted.selected_atomic_evidence_ids,
            selected_transcript_ids=transcript_ids,
            canonical_evidence_ids=targeted.canonical_evidence_ids,
            excluded_seen_item_count=(
                len(query.seen_atomic_evidence_ids)
                + len(query.seen_transcript_ids)
            ),
            newly_revealed_item_count=selected_items,
            novelty_fraction=novelty_fraction,
        )

    def _align_transcript(
        self, records: Sequence[_TranscriptRecord]
    ) -> dict[str, tuple[_TranscriptRecord, ...]]:
        output: dict[str, list[_TranscriptRecord]] = {
            segment_id: [] for segment_id in self._segment_ranges
        }
        for record in records:
            candidates: list[tuple[float, str]] = []
            for segment_id, (
                start_sec,
                end_sec,
            ) in self._segment_ranges.items():
                overlap = max(
                    0.0,
                    min(record.end_sec, end_sec) - max(record.start_sec, start_sec),
                )
                if overlap > 0.0:
                    candidates.append((overlap, segment_id))
            if not candidates:
                continue
            _, segment_id = max(
                candidates,
                key=lambda item: (
                    item[0],
                    -_natural_identifier_key(item[1])[1],
                ),
            )
            output[segment_id].append(record)
        return {
            segment_id: tuple(
                sorted(
                    aligned,
                    key=lambda item: (
                        item.midpoint_sec,
                        item.start_sec,
                        _natural_identifier_key(item.utterance_id),
                    ),
                )
            )
            for segment_id, aligned in output.items()
        }

    def _select_initial_segments(self, limit: int) -> tuple[str, ...]:
        available = self.available_segment_ids
        if len(available) <= limit:
            return available

        coverage_slots = min(limit, max(2, limit // 3))
        coverage = _uniformly_select_identifiers(available, coverage_slots)
        av_ranked = sorted(
            available,
            key=lambda segment_id: (
                -_segment_salience(
                    self._records_by_segment[segment_id],
                    self._canonical_by_id,
                ),
                _natural_identifier_key(segment_id),
            ),
        )
        transcript_ranked = sorted(
            available,
            key=lambda segment_id: (
                -_transcript_segment_score(
                    self._transcript_by_segment.get(segment_id, ())
                ),
                _natural_identifier_key(segment_id),
            ),
        )

        selected: list[str] = list(coverage)
        if len(selected) == limit:
            return tuple(sorted(selected, key=_natural_identifier_key))
        remaining_budget = limit - len(selected)
        av_slots = (remaining_budget + 1) // 2
        transcript_slots = remaining_budget - av_slots
        for ranked, quota in (
            (av_ranked, av_slots),
            (transcript_ranked, transcript_slots),
        ):
            if quota == 0:
                continue
            added = 0
            for segment_id in ranked:
                if segment_id not in selected:
                    selected.append(segment_id)
                    added += 1
                    if added == quota:
                        break
        if len(selected) == limit:
            return tuple(sorted(selected, key=_natural_identifier_key))
        for segment_id in available:
            if segment_id not in selected:
                selected.append(segment_id)
                if len(selected) == limit:
                    break
        if len(selected) == limit:
            return tuple(sorted(selected, key=_natural_identifier_key))
        raise AssertionError("could not fill initial segment budget")

    def _review_segments(
        self, priority_segment_ids: Sequence[str]
    ) -> tuple[str, ...]:
        available = set(self.available_segment_ids)
        unknown = set(priority_segment_ids) - available
        if unknown:
            raise ValueError(
                "Unknown review segment identifiers: "
                + ", ".join(sorted(unknown))
            )
        selected = list(priority_segment_ids[: self._config.review_max_segments])
        if len(selected) == self._config.review_max_segments:
            return tuple(selected)
        for segment_id in self._default_review_segments():
            if segment_id not in selected:
                selected.append(segment_id)
                if len(selected) == self._config.review_max_segments:
                    break
        return tuple(selected)

    def _default_review_segments(self) -> tuple[str, ...]:
        available = self.available_segment_ids
        transcript_ranked = sorted(
            available,
            key=lambda segment_id: (
                -_transcript_segment_score(
                    self._transcript_by_segment.get(segment_id, ())
                ),
                _natural_identifier_key(segment_id),
            ),
        )
        av_ranked = sorted(
            available,
            key=lambda segment_id: (
                -_segment_salience(
                    self._records_by_segment[segment_id],
                    self._canonical_by_id,
                ),
                _natural_identifier_key(segment_id),
            ),
        )
        # When wording exists, select its most informative indexed segments and
        # retrieve aligned A/V atoms from those same intervals.  This preserves
        # multimodal review while avoiding generic beginning/end coverage that
        # carried little incremental information in the paired OOF ablation.
        transcript_quota = (
            self._config.review_max_segments if self._transcript_by_id else 0
        )
        selected: list[str] = []
        for ranked, quota in (
            (transcript_ranked, transcript_quota),
            (
                av_ranked,
                self._config.review_max_segments - transcript_quota,
            ),
        ):
            if quota == 0:
                continue
            added = 0
            for segment_id in ranked:
                if segment_id not in selected:
                    selected.append(segment_id)
                    added += 1
                    if added == quota:
                        break
        for segment_id in transcript_ranked:
            if (
                len(selected) < self._config.review_max_segments
                and segment_id not in selected
            ):
                selected.append(segment_id)
        return tuple(selected)

    def _render_segment_overview(self, segment_id: str) -> list[str]:
        records = self._records_by_segment[segment_id]
        start_sec, end_sec = self._segment_ranges[segment_id]
        lines = [
            "",
            f"SEGMENT {segment_id} | time {format_time(start_sec)} to "
            f"{format_time(end_sec)} | {len(records)} atomic windows "
            "available for later review.",
        ]
        for modality in ("audio", "visual"):
            units = [
                self._canonical_by_id[canonical_id]
                for record in records
                for canonical_id in record.canonical_ids
                if self._canonical_by_id[canonical_id].modality == modality
            ]
            lines.append(_render_modality_aggregate(modality, units))
        return lines

    def _render_transcript_review(
        self,
        segment_ids: Sequence[str],
        *,
        excluded: frozenset[str],
    ) -> tuple[str, tuple[str, ...]]:
        lines = [
            "TARGETED TRANSCRIPT DETAIL FOR SECOND-PASS REVIEW",
            "Only wording aligned with the selected S identifiers is shown. Every "
            "T identifier below is newly disclosed in this review.",
        ]
        selected_ids: list[str] = []
        for segment_id in segment_ids:
            candidates = tuple(
                record
                for record in self._transcript_by_segment.get(segment_id, ())
                if record.utterance_id not in excluded
            )
            chosen = _select_transcript_records(
                candidates,
                self._config.review_max_utterances_per_segment,
            )
            lines.append("")
            lines.append(
                f"TRANSCRIPT DETAIL {segment_id} | selected {len(chosen)} of "
                f"{len(candidates)} unseen aligned utterances."
            )
            if not chosen:
                lines.append(
                    "No unseen retained transcript wording was available for this "
                    "segment."
                )
                continue
            for record in chosen:
                selected_ids.append(record.utterance_id)
                role = (
                    "Participant"
                    if record.speaker_role == "participant"
                    else "Unassigned speaker"
                )
                wording = _truncate_at_word_boundary(
                    record.text,
                    self._config.review_max_characters_per_utterance,
                )
                lines.append(
                    f"{role} utterance {record.utterance_id} | time "
                    f"{format_time(record.start_sec)} to "
                    f"{format_time(record.end_sec)} | observed wording: "
                    f"{json.dumps(wording, ensure_ascii=False)}"
                )
        if not self._transcript_by_id:
            lines.append(
                "Transcript source unavailable; no wording was substituted for it."
            )
        return "\n".join(lines).rstrip() + "\n", tuple(selected_ids)


def _load_transcript_records(
    transcript_dir: Path,
) -> tuple[_TranscriptRecord, ...]:
    if not transcript_dir.is_dir():
        raise ValueError(
            f"Transcript session directory does not exist: {transcript_dir}"
        )
    manifest_path = transcript_dir / "transcript_manifest.compact.json"
    units_path = transcript_dir / "transcript_units.compact.jsonl"
    for required in (manifest_path, units_path):
        if not required.is_file():
            raise ValueError(f"Required transcript file is missing: {required}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != "compact":
        raise ValueError("transcript protocol_version must be compact")
    if manifest.get("label_fields_read") != []:
        raise ValueError("transcript manifest crossed the label boundary")
    if manifest.get("raw_session_identifier_in_model_input") is not False:
        raise ValueError("transcript manifest does not protect the session identifier")
    speaker_policy = manifest.get("speaker_policy")
    if speaker_policy not in {"participant_only", "all_speakers_unassigned"}:
        raise ValueError(f"unsupported transcript speaker policy: {speaker_policy!r}")
    file_record = manifest.get("files", {}).get("transcript_units.compact.jsonl")
    if not isinstance(file_record, Mapping):
        raise ValueError("transcript manifest lacks the canonical units hash")
    content = units_path.read_text(encoding="utf-8")
    if hashlib.sha256(content.encode("utf-8")).hexdigest() != file_record.get(
        "sha256"
    ):
        raise ValueError("transcript canonical units hash mismatch")

    output: list[_TranscriptRecord] = []
    seen: set[str] = set()
    expected_role = (
        "participant" if speaker_policy == "participant_only" else "unassigned"
    )
    for line_number, line in enumerate(content.splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        _reject_label_fields(payload, f"transcript row {line_number}")
        utterance_id = str(payload.get("utterance_id", ""))
        if _TRANSCRIPT_ID.fullmatch(utterance_id) is None:
            raise ValueError(
                f"invalid transcript identifier at line {line_number}: "
                f"{utterance_id!r}"
            )
        if utterance_id in seen:
            raise ValueError(f"duplicate transcript identifier: {utterance_id}")
        seen.add(utterance_id)
        if payload.get("privacy_scrubbed"):
            continue
        if payload.get("speaker_role") != expected_role:
            continue
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"retained transcript row {line_number} has no readable text"
            )
        start_sec = _finite_number(
            payload.get("start_sec"), f"transcript row {line_number} start_sec"
        )
        end_sec = _finite_number(
            payload.get("end_sec"), f"transcript row {line_number} end_sec"
        )
        if start_sec < 0.0 or end_sec <= start_sec:
            raise ValueError(
                f"invalid transcript time range at line {line_number}"
            )
        output.append(
            _TranscriptRecord(
                utterance_id=utterance_id,
                start_sec=start_sec,
                end_sec=end_sec,
                text=text.strip(),
                speaker_role=expected_role,
            )
        )
    return tuple(
        sorted(
            output,
            key=lambda item: (
                item.midpoint_sec,
                item.start_sec,
                _natural_identifier_key(item.utterance_id),
            ),
        )
    )


def _render_modality_aggregate(
    modality: str,
    units: Sequence[EvidenceUnit],
) -> str:
    label = modality.capitalize()
    if not units:
        return (
            f"{label} aggregate: source unavailable; no value was substituted."
        )
    available = [
        unit
        for unit in units
        if str(unit.availability.get("status", "unavailable")) != "unavailable"
    ]
    valid_ratios = [
        float(unit.quality["valid_ratio"])
        for unit in units
        if _is_finite_number(unit.quality.get("valid_ratio"))
    ]
    coverage = (
        f"median valid coverage {round(100.0 * median(valid_ratios))} percent"
        if valid_ratios
        else "valid coverage not reported"
    )
    selected = _aggregate_observation(units)
    if selected is None:
        pattern = "no referenced observation was available"
    else:
        name, percentile, count, reliability = selected
        display_name = DISPLAY_NAMES.get(name, name.replace("_", " "))
        pattern = (
            f"most salient aggregate was {display_name}, "
            f"{_percentile_category(percentile)} at median training-reference "
            f"percentile {round(percentile)} across {count} windows "
            f"({reliability} measurement)"
        )
    return (
        f"{label} aggregate: {len(available)} of {len(units)} windows available; "
        f"{coverage}; {pattern}. Exact window values and within-session deviations "
        "are withheld until review."
    )


def _aggregate_observation(
    units: Sequence[EvidenceUnit],
) -> tuple[str, float, int, str] | None:
    grouped: dict[str, list[Observation]] = {}
    for unit in units:
        for observation in unit.observations:
            if observation.percentile is not None:
                grouped.setdefault(observation.name, []).append(observation)
    candidates: list[tuple[float, str, float, int, str]] = []
    for name, observations in grouped.items():
        percentiles = [float(item.percentile) for item in observations]
        middle = float(median(percentiles))
        reliability = max(
            (item.semantic_reliability for item in observations),
            key=lambda value: {
                "direct": 3,
                "derived": 2,
                "weak": 1,
                "latent": 0,
            }[value],
        )
        reliability_weight = {
            "direct": 1.0,
            "derived": 0.9,
            "weak": 0.35,
            "latent": 0.2,
        }[reliability]
        prevalence = len(observations) / max(1, len(units))
        anchor_priority = OBSERVATION_PRIORITY.get(name, 0) / 20.0
        score = reliability_weight * (
            0.55 * anchor_priority
            + 0.35 * abs(middle - 50.0) / 50.0
            + 0.10 * prevalence
        )
        candidates.append(
            (score, name, middle, len(observations), reliability)
        )
    if not candidates:
        return None
    _, name, middle, count, reliability = max(
        candidates,
        key=lambda item: (item[0], item[1]),
    )
    return name, middle, count, reliability


def _segment_salience(
    records: Sequence[_IndexRecord],
    canonical_by_id: Mapping[str, EvidenceUnit],
) -> float:
    scores = sorted(
        (_record_salience(record, canonical_by_id) for record in records),
        reverse=True,
    )
    if not scores:
        return 0.0
    strongest = scores[: min(4, len(scores))]
    return 0.7 * (sum(strongest) / len(strongest)) + 0.3 * (
        sum(scores) / len(scores)
    )


def _derive_segment_ranges(
    records_by_segment: Mapping[str, Sequence[_IndexRecord]],
) -> dict[str, tuple[float, float]]:
    ordered = sorted(records_by_segment, key=_natural_identifier_key)
    output: dict[str, tuple[float, float]] = {}
    for index, segment_id in enumerate(ordered):
        records = records_by_segment[segment_id]
        start_sec = (
            min(record.start_sec for record in records)
            if index == 0
            else records[0].midpoint_sec
        )
        end_sec = (
            records_by_segment[ordered[index + 1]][0].midpoint_sec
            if index + 1 < len(ordered)
            else max(record.end_sec for record in records)
        )
        if end_sec <= start_sec:
            raise ValueError(
                f"invalid derived time range for segment {segment_id}"
            )
        output[segment_id] = (start_sec, end_sec)
    return output


def _cue_families(text: str) -> tuple[str, ...]:
    lowered = text.casefold()
    output: list[str] = []
    for family, terms in _CUE_FAMILIES:
        if any(_contains_term(lowered, term) for term in terms):
            output.append(family)
    return tuple(output)


def _cue_families_for_records(
    records: Sequence[_TranscriptRecord],
) -> tuple[str, ...]:
    found: set[str] = set()
    for record in records:
        found.update(_cue_families(record.text))
    return tuple(
        family for family, _ in _CUE_FAMILIES if family in found
    )


def _cue_family_counts(
    records: Sequence[_TranscriptRecord],
) -> tuple[tuple[str, int], ...]:
    counts = {
        family: sum(
            family in _cue_families(record.text) for record in records
        )
        for family, _ in _CUE_FAMILIES
    }
    return tuple(
        (family, counts[family])
        for family, _ in _CUE_FAMILIES
        if counts[family] > 0
    )


def _transcript_segment_score(
    records: Sequence[_TranscriptRecord],
) -> float:
    return float(
        10.0 * sum(len(_cue_families(record.text)) for record in records)
        + 0.001 * min(sum(len(record.text) for record in records), 1_000)
    )


def _select_transcript_records(
    records: Sequence[_TranscriptRecord],
    limit: int,
) -> tuple[_TranscriptRecord, ...]:
    if len(records) <= limit:
        return tuple(records)
    cue_ranked = sorted(
        records,
        key=lambda record: (
            -len(_cue_families(record.text)),
            -len(record.text),
            record.midpoint_sec,
            _natural_identifier_key(record.utterance_id),
        ),
    )
    cue_slots = max(1, (limit + 1) // 2)
    cue_candidates = [
        record for record in cue_ranked if _cue_families(record.text)
    ]
    selected = list((cue_candidates or cue_ranked)[:cue_slots])
    selected_ids = {record.utterance_id for record in selected}
    remaining = [
        record
        for record in records
        if record.utterance_id not in selected_ids
    ]
    selected.extend(
        _uniformly_select_records(remaining, limit - len(selected))
    )
    return tuple(
        sorted(
            selected,
            key=lambda item: (
                item.midpoint_sec,
                _natural_identifier_key(item.utterance_id),
            ),
        )
    )


def _uniformly_select_identifiers(
    values: Sequence[str],
    limit: int,
) -> tuple[str, ...]:
    if len(values) <= limit:
        return tuple(values)
    indices = _uniform_indices(len(values), limit)
    return tuple(values[index] for index in indices)


def _uniformly_select_records(
    values: Sequence[_TranscriptRecord],
    limit: int,
) -> tuple[_TranscriptRecord, ...]:
    if limit <= 0:
        return ()
    if len(values) <= limit:
        return tuple(values)
    indices = _uniform_indices(len(values), limit)
    return tuple(values[index] for index in indices)


def _uniform_indices(length: int, limit: int) -> tuple[int, ...]:
    if limit == 1:
        return ((length - 1) // 2,)
    return tuple(
        int(math.floor(index * (length - 1) / (limit - 1) + 0.5))
        for index in range(limit)
    )


def _contains_term(text: str, term: str) -> bool:
    return re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", text) is not None


def _percentile_category(percentile: float) -> str:
    if percentile < 5.0:
        return "far below the training reference"
    if percentile < 25.0:
        return "below the training reference"
    if percentile <= 75.0:
        return "within the central training-reference range"
    if percentile < 95.0:
        return "above the training reference"
    return "far above the training reference"


def _truncate_at_word_boundary(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    shortened = text[: limit - 1].rsplit(" ", 1)[0].rstrip()
    return (shortened or text[: limit - 1]).rstrip() + "…"


def _validate_identifiers(
    name: str,
    values: Sequence[str],
    pattern: re.Pattern[str],
) -> None:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of identifiers")
    for value in values:
        if not isinstance(value, str) or pattern.fullmatch(value) is None:
            raise ValueError(f"invalid identifier in {name}: {value!r}")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicate identifiers")


def _reject_label_fields(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).casefold())
            if normalized in _NORMALIZED_LABEL_KEYS:
                raise ValueError(f"{path} contains forbidden label field {key!r}")
            _reject_label_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_label_fields(child, f"{path}[{index}]")


def _finite_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )
