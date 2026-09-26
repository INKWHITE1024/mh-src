"""Build leakage-audited Draft/Revision/ORPO records from frozen OOF rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rethink_mh.experiments.qwen_label_baseline import (
    D_VLOG_MISSING_TRANSCRIPT_TEXT,
)
from rethink_mh.rethinking.contracts import (
    InitialAssessment,
    RevisionAssessment,
    SourceAssessment,
    TaskSpec,
)
from rethink_mh.rethinking.prompts import PromptBuilder
from rethink_mh.rethinking.retrieval import EvidenceStore, TargetedEvidence


DatasetKey = Literal["daic_woz", "e_daic", "d_vlog"]
_DATASETS = frozenset({"daic_woz", "e_daic", "d_vlog"})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SEGMENT = re.compile(r"^SEGMENT (S[0-9]{3,6})\b")
_PERCENTILE = re.compile(r"median percentile (\d{1,3})")
_COVERAGE = re.compile(r"median valid coverage (\d{1,3})%")
_SESSION_COVERAGE = {
    modality: re.compile(rf"{modality} median valid coverage (\d{{1,3}})%")
    for modality in ("audio", "visual")
}
_CONSTRAINT_PREFERENCE_KINDS = (
    "constraint_missing_required_field",
    "constraint_out_of_scope_clinical_claim",
    "constraint_unknown_evidence_id",
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_ids(ids: Sequence[str]) -> str:
    return _sha256_bytes("\n".join(ids).encode("utf-8"))


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {description} at {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be one JSON object: {path}")
    return value


def _read_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read {description} at {path}: {error}") from error
    output: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise ValueError(f"blank line in {description}: {path}:{line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid JSON in {description}: {path}:{line_number}"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(
                f"{description} row must be an object: {path}:{line_number}"
            )
        output.append(value)
    if not output:
        raise ValueError(f"{description} is empty: {path}")
    return output


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _probability(value: object, name: str) -> float:
    result = _number(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _binary_label(value: object, name: str) -> int:
    if isinstance(value, bool) or value not in {0, 1}:
        raise ValueError(f"{name} must be binary")
    return int(value)


@dataclass(frozen=True, slots=True)
class FrozenDraft:
    session_id: str
    label: int
    probability: float
    threshold: float

    @property
    def prediction(self) -> int:
        return int(self.probability >= self.threshold)

    @property
    def correct(self) -> bool:
        return self.prediction == self.label


@dataclass(frozen=True, slots=True)
class SourceScores:
    audio: float
    visual: float


@dataclass(frozen=True, slots=True)
class SegmentSignal:
    segment_id: str
    audio_reliability: float | None
    visual_reliability: float | None
    audio_salience: float
    visual_salience: float

    @property
    def minimum_reliability(self) -> float:
        values = [
            value
            for value in (self.audio_reliability, self.visual_reliability)
            if value is not None
        ]
        return min(values) if values else 0.0

    @property
    def disagreement_salience(self) -> float:
        return abs(self.audio_salience - self.visual_salience)


@dataclass(frozen=True, slots=True)
class PackageRequest:
    dataset: DatasetKey
    task_path: Path
    evidence_root: Path
    oof_predictions: Path
    source_predictions: Path
    output_dir: Path
    transcript_root: Path | None = None
    transcript_tier: str = "compact"
    source_model: str = "logistic"
    draft_model: str | None = None
    draft_view: str | None = None
    split: str = "train"
    max_segments: int = 3
    max_atomic_per_segment: int = 4
    max_observations_per_modality: int = 3
    unresolved_error_fraction: float = 0.25
    false_negative_weight: float = 2.0

    def __post_init__(self) -> None:
        if self.dataset not in _DATASETS:
            raise ValueError(f"unsupported dataset: {self.dataset}")
        if self.transcript_tier not in {"full", "compact", "essential", "minimal"}:
            raise ValueError("transcript_tier must be full, compact, essential, or minimal")
        for name in (
            "max_segments",
            "max_atomic_per_segment",
            "max_observations_per_modality",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0.0 <= self.unresolved_error_fraction <= 1.0:
            raise ValueError("unresolved_error_fraction must be in [0, 1]")
        if not math.isfinite(self.false_negative_weight) or self.false_negative_weight < 1.0:
            raise ValueError("false_negative_weight must be finite and at least 1")


def _frozen_drafts(
    path: Path,
    *,
    split: str,
    model: str | None = None,
    view: str | None = None,
) -> list[FrozenDraft]:
    rows = _read_jsonl(path, "frozen OOF predictions")
    drafts: list[FrozenDraft] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, 1):
        row_split = row.get("split")
        if row_split != split:
            continue
        if model is not None and row.get("model") != model:
            continue
        if view is not None and row.get("view") != view:
            continue
        session_id = row.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError(f"OOF row {row_number} has invalid session_id")
        session_id = session_id.strip()
        if session_id in seen:
            raise ValueError(f"duplicate OOF session_id: {session_id}")
        seen.add(session_id)
        origin = str(row.get("prediction_origin", ""))
        if "out_of_fold" not in origin and "oof" not in origin:
            raise ValueError(
                f"draft prediction for {session_id} is not declared out-of-fold"
            )
        drafts.append(
            FrozenDraft(
                session_id=session_id,
                label=_binary_label(row.get("label"), f"OOF label for {session_id}"),
                probability=_probability(
                    row.get("probability"), f"OOF probability for {session_id}"
                ),
                threshold=_probability(
                    row.get("threshold"), f"OOF threshold for {session_id}"
                ),
            )
        )
    if not drafts:
        raise ValueError(f"no OOF rows found for split {split!r}")
    thresholds = {draft.threshold for draft in drafts}
    if len(thresholds) != 1:
        raise ValueError("frozen OOF rows must use one shared threshold")
    if {draft.label for draft in drafts} != {0, 1}:
        raise ValueError("frozen OOF cohort must contain both classes")
    return drafts


def _source_scores(
    path: Path,
    *,
    split: str,
    model: str,
    expected: Mapping[str, FrozenDraft],
) -> dict[str, SourceScores]:
    rows = _read_jsonl(path, "source-view OOF predictions")
    by_key: dict[tuple[str, str], float] = {}
    labels: dict[str, int] = {}
    for row in rows:
        if row.get("split") != split or row.get("model") != model:
            continue
        view = row.get("view")
        if view not in {"audio", "visual"}:
            continue
        origin = str(row.get("prediction_origin", ""))
        if "out_of_fold" not in origin and "oof" not in origin:
            raise ValueError("source-view predictions must be out-of-fold")
        session_id = row.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("source-view prediction has invalid session_id")
        session_id = session_id.strip()
        key = (session_id, str(view))
        if key in by_key:
            raise ValueError(f"duplicate source-view prediction: {key}")
        by_key[key] = _probability(row.get("probability"), f"source score {key}")
        parsed_label = _binary_label(
            row.get("label"), f"source-view label for {session_id}"
        )
        if session_id in labels and labels[session_id] != parsed_label:
            raise ValueError(f"source-view labels disagree within {session_id}")
        labels[session_id] = parsed_label
    result: dict[str, SourceScores] = {}
    for session_id, draft in expected.items():
        if labels.get(session_id) != draft.label:
            raise ValueError(f"source-view label disagrees for {session_id}")
        try:
            result[session_id] = SourceScores(
                audio=by_key[(session_id, "audio")],
                visual=by_key[(session_id, "visual")],
            )
        except KeyError as error:
            raise ValueError(
                f"source-view OOF scores are incomplete for {session_id}"
            ) from error
    extras = {session_id for session_id, _ in by_key} - set(expected)
    if extras:
        raise ValueError(f"source-view predictions contain unexpected IDs: {sorted(extras)}")
    return result


def _task(path: Path) -> TaskSpec:
    return TaskSpec.from_dict(_read_json(path, "task specification"))


def _validated_session_dir(root: Path, session_id: str, dataset: DatasetKey) -> Path:
    directory = root.resolve() / session_id
    required = (
        "session_input.native.txt",
        "atomic_evidence.native.txt",
        "evidence_index.native.jsonl",
        "evidence_units.jsonl",
        "manifest.json",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete native Evidence session {session_id}: {missing}")
    manifest = _read_json(directory / "manifest.json", "Evidence manifest")
    if manifest.get("dataset_key") != dataset:
        raise ValueError(f"Evidence dataset mismatch for {session_id}")
    if manifest.get("label_fields_read") != []:
        raise ValueError(f"Evidence compiler reports label access for {session_id}")
    if manifest.get("transcript_content_exposed") is not False:
        raise ValueError(f"A/V Evidence unexpectedly exposes transcript for {session_id}")
    return directory


def _session_text(request: PackageRequest, directory: Path, session_id: str) -> str:
    av_text = (
        (directory / "session_input.native.txt").read_text(encoding="utf-8").rstrip()
    )
    if request.dataset == "d_vlog":
        if request.transcript_root is not None:
            raise ValueError("D-Vlog must use the uniform missing transcript marker")
        transcript = D_VLOG_MISSING_TRANSCRIPT_TEXT.rstrip()
    else:
        if request.transcript_root is None:
            raise ValueError(f"{request.dataset} loop training requires the compact transcript layer")
        transcript_dir = request.transcript_root.resolve() / session_id
        manifest_path = transcript_dir / "transcript_manifest.compact.json"
        manifest = _read_json(manifest_path, "compact transcript manifest")
        if manifest.get("dataset_key") != request.dataset:
            raise ValueError(f"Transcript dataset mismatch for {session_id}")
        if manifest.get("label_fields_read") not in (None, []):
            raise ValueError(f"Transcript compiler reports label access for {session_id}")
        transcript_path = (
            transcript_dir / f"transcript_input.{request.transcript_tier}.compact.txt"
        )
        transcript = transcript_path.read_text(encoding="utf-8").strip()
    return f"{av_text}\n\n{transcript}\n"


def _segment_signals(text: str) -> list[SegmentSignal]:
    session_reliability: dict[str, float | None] = {}
    for modality, pattern in _SESSION_COVERAGE.items():
        match = pattern.search(text)
        session_reliability[modality] = (
            int(match.group(1)) / 100.0 if match is not None else None
        )
    signals: list[SegmentSignal] = []
    for line in text.splitlines():
        match = _SEGMENT.match(line)
        if match is None:
            continue
        fields = [field.strip() for field in line.split("|")]
        values: dict[str, tuple[float | None, float]] = {}
        for modality in ("audio", "visual"):
            field = next(
                (
                    item
                    for item in fields
                    if item.casefold().startswith(f"{modality}:")
                    or item.startswith(f"{modality.upper()} summary")
                ),
                "",
            )
            coverage_match = _COVERAGE.search(field)
            coverage = (
                int(coverage_match.group(1)) / 100.0
                if coverage_match is not None
                else (
                    0.0
                    if "unavailable" in field.casefold()
                    else session_reliability[modality]
                )
            )
            percentiles = [int(value) for value in _PERCENTILE.findall(field)]
            salience = max((abs(value - 50) / 50.0 for value in percentiles), default=0.0)
            values[modality] = (coverage, min(1.0, salience))
        signals.append(
            SegmentSignal(
                segment_id=match.group(1),
                audio_reliability=values["audio"][0],
                visual_reliability=values["visual"][0],
                audio_salience=values["audio"][1],
                visual_salience=values["visual"][1],
            )
        )
    if not signals:
        raise ValueError("compact session evidence contains no segment records")
    return signals


def _mean_known(values: Sequence[float | None], default: float = 0.0) -> float:
    known = [value for value in values if value is not None]
    return math.fsum(known) / len(known) if known else default


def _source_assessment(
    probability: float, values: Sequence[float | None]
) -> SourceAssessment:
    known = [value for value in values if value is not None]
    if not known or max(known) <= 0.0:
        return SourceAssessment(status="unavailable")
    return SourceAssessment(
        status="available",
        risk_probability=probability,
        reliability=min(1.0, max(0.0, _mean_known(known))),
    )


def _entropy(probability: float) -> float:
    if probability in {0.0, 1.0}:
        return 0.0
    return -(
        probability * math.log2(probability)
        + (1.0 - probability) * math.log2(1.0 - probability)
    )


def _top_segments(
    signals: Sequence[SegmentSignal],
    key: Any,
    *,
    limit: int,
) -> tuple[str, ...]:
    ranked = sorted(signals, key=lambda signal: (key(signal), signal.segment_id), reverse=True)
    return tuple(signal.segment_id for signal in ranked[:limit])


def _construct_initial_target(
    draft: FrozenDraft,
    source: SourceScores,
    signals: Sequence[SegmentSignal],
    *,
    max_segments: int,
) -> InitialAssessment:
    audio_reliability = [signal.audio_reliability for signal in signals]
    visual_reliability = [signal.visual_reliability for signal in signals]
    audio = _source_assessment(source.audio, audio_reliability)
    visual = _source_assessment(source.visual, visual_reliability)
    available_reliabilities = [
        value
        for value in (audio.reliability, visual.reliability)
        if value is not None
    ]
    mean_reliability = _mean_known(available_reliabilities, default=0.0)
    confidence = (1.0 - _entropy(draft.probability)) * (0.5 + 0.5 * mean_reliability)
    confidence = min(0.99, max(0.01, confidence))

    overall_class = draft.prediction
    supporting: list[str] = []
    contradictory: list[str] = []
    for modality, risk in (("audio", source.audio), ("visual", source.visual)):
        agrees = int(risk >= draft.threshold) == overall_class
        key = (
            (lambda signal: signal.audio_salience)
            if modality == "audio"
            else (lambda signal: signal.visual_salience)
        )
        target = supporting if agrees else contradictory
        for segment_id in _top_segments(signals, key, limit=2):
            if segment_id not in target:
                target.append(segment_id)
    low_quality = _top_segments(
        signals,
        lambda signal: 1.0 - signal.minimum_reliability,
        limit=max_segments,
    )
    cross_source = _top_segments(
        signals,
        lambda signal: signal.disagreement_salience,
        limit=max_segments,
    )
    uncertain = tuple(dict.fromkeys((*low_quality, *cross_source)))[:4]
    requested = tuple(dict.fromkeys((*uncertain, *contradictory, *supporting)))[:max_segments]
    return InitialAssessment(
        risk_probability=draft.probability,
        confidence=confidence,
        audio_source=audio,
        visual_source=visual,
        supporting_segment_ids=tuple(supporting[:4]),
        contradictory_segment_ids=tuple(contradictory[:4]),
        uncertain_segment_ids=uncertain,
        requested_segment_ids=requested,
    )


def _selected_segments(
    initial: InitialAssessment,
    signals: Sequence[SegmentSignal],
    *,
    limit: int,
) -> tuple[str, ...]:
    available = {signal.segment_id for signal in signals}
    selected: list[str] = []
    for group in (
        initial.requested_segment_ids,
        initial.uncertain_segment_ids,
        initial.contradictory_segment_ids,
        initial.supporting_segment_ids,
    ):
        for segment_id in group:
            if segment_id in available and segment_id not in selected:
                selected.append(segment_id)
                if len(selected) == limit:
                    return tuple(selected)
    for signal in signals:
        if signal.segment_id not in selected:
            selected.append(signal.segment_id)
            if len(selected) == limit:
                break
    return tuple(selected)


def _quality_score(initial: InitialAssessment) -> float:
    values = [
        value
        for value in (
            initial.audio_source.reliability,
            initial.visual_source.reliability,
        )
        if value is not None
    ]
    return _mean_known(values, default=0.0)


def _correct_side_probability(label: int, threshold: float) -> float:
    if label == 1:
        return min(0.95, max(threshold + 0.10, threshold + (1.0 - threshold) * 0.65))
    return max(0.05, min(threshold - 0.10, threshold * 0.35))


def _opposite_probability(prediction: int, threshold: float) -> float:
    return _correct_side_probability(1 - prediction, threshold)


def _revision(
    *,
    status: str,
    probability: float,
    confidence: float,
    targeted: TargetedEvidence,
    initial_probability: float,
    hallucinate: bool = False,
) -> dict[str, Any]:
    cited_segments = tuple(targeted.selected_segment_ids)
    cited_evidence = tuple(targeted.selected_atomic_evidence_ids[:6])
    if not cited_evidence:
        raise ValueError("targeted retrieval returned no atomic evidence IDs")
    if hallucinate:
        cited_evidence = ("E999999", *cited_evidence[1:])
    if status == "unresolved":
        preserved: tuple[str, ...] = ()
        residual = cited_segments
        summary = "insufficient_reliable_detail"
    else:
        preserved = cited_evidence
        residual = ()
        if math.isclose(probability, initial_probability, abs_tol=1e-12):
            summary = "risk_unchanged_after_review"
        elif probability > initial_probability:
            summary = "risk_increased_after_review"
        else:
            summary = "risk_decreased_after_review"
    payload = {
        "revision_status": status,
        "revised_risk_probability": probability,
        "revised_confidence": confidence,
        "cited_segment_ids": list(cited_segments),
        "cited_evidence_ids": list(cited_evidence),
        "preserved_evidence_ids": list(preserved),
        "newly_considered_evidence_ids": list(cited_evidence),
        "rejected_evidence_ids": [],
        "residual_conflict_segment_ids": list(residual),
        "change_summary": summary,
    }
    # Constraint negatives deliberately contain a syntactically valid but unavailable
    # ID.  All positive targets must pass the strict contract here.
    if not hallucinate:
        RevisionAssessment.from_dict(payload)
    return payload


def _messages_with_assistant(
    prompt: Sequence[Mapping[str, str]], assistant: Mapping[str, Any]
) -> list[dict[str, str]]:
    messages = [dict(message) for message in prompt]
    messages.append(
        {
            "role": "assistant",
            "content": _canonical_json(assistant),
        }
    )
    return messages


def _jsonl_text(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(_canonical_json(row) + "\n" for row in rows)


def select_optimization_preferences(
    rows: Sequence[dict[str, Any]],
    *,
    constraint_session_fraction: float = 0.10,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Keep every grounded hard pair plus a balanced constraint sample."""

    if not 0.0 <= constraint_session_fraction <= 1.0:
        raise ValueError("constraint_session_fraction must be in [0, 1]")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        session_id = row.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("preference rows must contain text session IDs")
        grouped.setdefault(session_id, []).append(row)
    selected: list[dict[str, Any]] = []
    sessions_with_constraint: set[str] = set()
    by_constraint_kind: dict[str, list[dict[str, Any]]] = {
        kind: [] for kind in _CONSTRAINT_PREFERENCE_KINDS
    }
    for session_id in sorted(grouped):
        session_rows = grouped[session_id]
        actions = [
            row
            for row in session_rows
            if not str(row.get("preference_kind", "")).startswith("constraint_")
        ]
        constraints = [
            row
            for row in session_rows
            if str(row.get("preference_kind", "")).startswith("constraint_")
        ]
        if not actions:
            raise ValueError(
                f"session {session_id} must contain a grounded action preference"
            )
        selected.extend(actions)
        for row in constraints:
            kind = str(row.get("preference_kind"))
            if kind not in by_constraint_kind:
                raise ValueError(f"unsupported constraint preference kind: {kind}")
            by_constraint_kind[kind].append(row)
        if not constraints:
            continue
        digest = hashlib.sha256(f"{seed}:{session_id}".encode()).digest()
        draw = int.from_bytes(digest[4:8], "big") / (2**32 - 1)
        if draw < constraint_session_fraction:
            offset = int.from_bytes(digest[:4], "big") % len(constraints)
            selected.append(constraints[offset])
            sessions_with_constraint.add(session_id)

    selected_kinds = {str(row["preference_kind"]) for row in selected}
    for kind in _CONSTRAINT_PREFERENCE_KINDS:
        if kind in selected_kinds:
            continue
        candidate = next(
            (
                row
                for row in by_constraint_kind[kind]
                if str(row["session_id"]) not in sessions_with_constraint
            ),
            None,
        )
        if candidate is None:
            raise ValueError(f"cannot retain required constraint preference kind: {kind}")
        selected.append(candidate)
        sessions_with_constraint.add(str(candidate["session_id"]))
    return selected


def _write_package_file(directory: Path, name: str, text: str) -> dict[str, Any]:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return {
        "file": name,
        "row_count": len(text.splitlines()),
        "sha256": _sha256_bytes(text.encode("utf-8")),
    }


def build_package(request: PackageRequest) -> dict[str, Any]:
    """Build one immutable dataset-specific trajectory package."""

    output_dir = request.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite trajectory package: {output_dir}")
    task = _task(request.task_path)
    drafts = _frozen_drafts(
        request.oof_predictions,
        split=request.split,
        model=request.draft_model,
        view=request.draft_view,
    )
    draft_by_id = {draft.session_id: draft for draft in drafts}
    sources = _source_scores(
        request.source_predictions,
        split=request.split,
        model=request.source_model,
        expected=draft_by_id,
    )

    prepared: list[dict[str, Any]] = []
    for draft in drafts:
        directory = _validated_session_dir(
            request.evidence_root, draft.session_id, request.dataset
        )
        session_text = _session_text(request, directory, draft.session_id)
        signals = _segment_signals(session_text)
        initial = _construct_initial_target(
            draft,
            sources[draft.session_id],
            signals,
            max_segments=request.max_segments,
        )
        selected = _selected_segments(initial, signals, limit=request.max_segments)
        store = EvidenceStore.from_session_dir(directory)
        targeted = store.render_segments(
            selected,
            max_atomic_per_segment=request.max_atomic_per_segment,
            max_observations_per_modality=request.max_observations_per_modality,
        )
        prepared.append(
            {
                "draft": draft,
                "session_text": session_text,
                "initial": initial,
                "selected": selected,
                "targeted": targeted,
                "quality": _quality_score(initial),
            }
        )

    errors = sorted(
        (item for item in prepared if not item["draft"].correct),
        key=lambda item: (item["quality"], item["initial"].confidence, item["draft"].session_id),
    )
    unresolved_count = (
        max(1, int(round(len(errors) * request.unresolved_error_fraction)))
        if errors and request.unresolved_error_fraction > 0.0
        else 0
    )
    unresolved_ids = {
        item["draft"].session_id for item in errors[:unresolved_count]
    }

    draft_rows: list[dict[str, Any]] = []
    loop_rows: list[dict[str, Any]] = []
    preference_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    transition_counts: Counter[str] = Counter()
    for item in prepared:
        draft: FrozenDraft = item["draft"]
        initial: InitialAssessment = item["initial"]
        targeted: TargetedEvidence = item["targeted"]
        draft_record = PromptBuilder.build_supervised_record(
            task,
            item["session_text"],
            draft.label,
            assistant_target=initial,
        )
        draft_record["messages"][-1]["content"] = _canonical_json(initial.to_dict())
        draft_record.update(
            {
                "dataset": request.dataset,
                "session_id": draft.session_id,
                # The source is a frozen OOF rollout, not an independent
                # annotator model.
                "annotator_origin": "frozen_out_of_fold_prediction",
            }
        )
        draft_rows.append(draft_record)

        decision = {
            "should_rethink": True,
            "trigger_score": max(1.0 - initial.confidence, 1.0 - item["quality"]),
            "reasons": ["oof_trajectory_supervision"],
            "selected_segment_ids": list(item["selected"]),
            "metrics": {
                "annotator_confidence": initial.confidence,
                "mean_source_reliability": item["quality"],
            },
        }
        prompt = PromptBuilder.build_revision_messages(
            task,
            initial,
            decision,
            targeted.text,
        )
        if draft.correct:
            transition = "CC_preserve"
            chosen = _revision(
                status="preserved",
                probability=draft.probability,
                confidence=max(initial.confidence, 0.65),
                targeted=targeted,
                initial_probability=draft.probability,
            )
            rejected = _revision(
                status="revised",
                probability=_opposite_probability(draft.prediction, draft.threshold),
                confidence=0.90,
                targeted=targeted,
                initial_probability=draft.probability,
            )
            rejected_kind = "CW_harmful_revision"
        elif draft.session_id in unresolved_ids:
            transition = "WW_refer"
            chosen = _revision(
                status="unresolved",
                probability=0.5,
                confidence=min(initial.confidence, 0.30),
                targeted=targeted,
                initial_probability=draft.probability,
            )
            rejected = _revision(
                status="revised",
                probability=draft.probability,
                confidence=0.95,
                targeted=targeted,
                initial_probability=draft.probability,
            )
            rejected_kind = "WW_forced_confident_prediction"
        else:
            transition = "WC_revise"
            chosen = _revision(
                status="revised",
                probability=_correct_side_probability(draft.label, draft.threshold),
                confidence=max(initial.confidence, 0.75),
                targeted=targeted,
                initial_probability=draft.probability,
            )
            rejected = _revision(
                status="preserved",
                probability=draft.probability,
                confidence=max(initial.confidence, 0.70),
                targeted=targeted,
                initial_probability=draft.probability,
            )
            rejected_kind = "WW_failed_preservation"
        transition_counts[transition] += 1
        sample_weight = (
            request.false_negative_weight
            if draft.label == 1 and draft.prediction == 0
            else 1.0
        )
        chosen_text = _canonical_json(chosen)
        rejected_text = _canonical_json(rejected)
        loop_rows.append(
            {
                "record_type": "revision_sft",
                "dataset": request.dataset,
                "session_id": draft.session_id,
                "task_id": task.task_id,
                "messages": _messages_with_assistant(prompt, chosen),
                "transition_target": transition,
                "sample_weight": sample_weight,
                "supervision": {
                    "label": draft.label,
                    "used_in_model_text": False,
                },
            }
        )
        preference_rows.append(
            {
                "record_type": "revision_preference",
                "dataset": request.dataset,
                "session_id": draft.session_id,
                "task_id": task.task_id,
                "prompt_messages": [dict(message) for message in prompt],
                "chosen": chosen_text,
                "rejected": rejected_text,
                "preference_kind": rejected_kind,
                "transition_target": transition,
                "sample_weight": sample_weight,
                "ground_truth_in_prompt": False,
            }
        )
        missing_field_negative = dict(chosen)
        missing_field_negative.pop("change_summary")
        preference_rows.append(
            {
                "record_type": "revision_preference",
                "dataset": request.dataset,
                "session_id": draft.session_id,
                "task_id": task.task_id,
                "prompt_messages": [dict(message) for message in prompt],
                "chosen": chosen_text,
                "rejected": _canonical_json(missing_field_negative),
                "preference_kind": "constraint_missing_required_field",
                "transition_target": transition,
                "sample_weight": sample_weight,
                "ground_truth_in_prompt": False,
            }
        )
        clinical_negative = dict(chosen)
        clinical_negative["clinical_diagnosis"] = "definitive diagnosis from screening evidence"
        preference_rows.append(
            {
                "record_type": "revision_preference",
                "dataset": request.dataset,
                "session_id": draft.session_id,
                "task_id": task.task_id,
                "prompt_messages": [dict(message) for message in prompt],
                "chosen": chosen_text,
                "rejected": _canonical_json(clinical_negative),
                "preference_kind": "constraint_out_of_scope_clinical_claim",
                "transition_target": transition,
                "sample_weight": sample_weight,
                "ground_truth_in_prompt": False,
            }
        )
        constraint_negative = _revision(
            status=chosen["revision_status"],
            probability=float(chosen["revised_risk_probability"]),
            confidence=float(chosen["revised_confidence"]),
            targeted=targeted,
            initial_probability=draft.probability,
            hallucinate=True,
        )
        preference_rows.append(
            {
                "record_type": "revision_preference",
                "dataset": request.dataset,
                "session_id": draft.session_id,
                "task_id": task.task_id,
                "prompt_messages": [dict(message) for message in prompt],
                "chosen": chosen_text,
                "rejected": _canonical_json(constraint_negative),
                "preference_kind": "constraint_unknown_evidence_id",
                "transition_target": transition,
                "sample_weight": sample_weight,
                "ground_truth_in_prompt": False,
            }
        )
        trajectory_rows.append(
            {
                "schema_version": "1.0.0",
                "record_type": "oof_rethink_trajectory",
                "dataset": request.dataset,
                "session_id": draft.session_id,
                "split": request.split,
                "task_id": task.task_id,
                "initial_assessment": initial.to_dict(),
                "initial_threshold": draft.threshold,
                "initial_predicted_label": draft.prediction,
                "selected_segment_ids": list(item["selected"]),
                "selected_evidence_ids": list(targeted.selected_atomic_evidence_ids),
                "transition_target": transition,
                "chosen_revision": chosen,
                "supervision": {
                    "label": draft.label,
                    "used_in_model_text": False,
                },
            }
        )

    required_transitions = {"CC_preserve", "WC_revise", "WW_refer"}
    missing_transitions = required_transitions - set(transition_counts)
    if missing_transitions:
        raise ValueError(
            "trajectory package lacks required positive transition targets: "
            f"{sorted(missing_transitions)}"
        )
    preference_kinds = Counter(row["preference_kind"] for row in preference_rows)
    if preference_kinds["CW_harmful_revision"] == 0:
        raise ValueError("trajectory package lacks correct-to-wrong harmful negatives")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=str(output_dir.parent))
    )
    try:
        files = {
            "draft_sft": _write_package_file(
                temporary, "draft_sft.jsonl", _jsonl_text(draft_rows)
            ),
            "revision_sft": _write_package_file(
                temporary, "revision_sft.jsonl", _jsonl_text(loop_rows)
            ),
            "preferences_initial": _write_package_file(
                temporary, "preferences_initial.jsonl", _jsonl_text(preference_rows)
            ),
            "trajectories": _write_package_file(
                temporary, "trajectories.jsonl", _jsonl_text(trajectory_rows)
            ),
        }
        manifest = {
            "schema_version": "1.0.0",
            "package": "revision_aware_oof_trajectory_v1",
            "dataset": request.dataset,
            "task": task.to_dict(),
            "session_count": len(drafts),
            "positive_count": sum(draft.label for draft in drafts),
            "negative_count": len(drafts) - sum(draft.label for draft in drafts),
            "initial_error_count": sum(not draft.correct for draft in drafts),
            "initial_false_negative_count": sum(
                draft.label == 1 and draft.prediction == 0 for draft in drafts
            ),
            "transition_counts": dict(sorted(transition_counts.items())),
            "preference_kind_counts": dict(sorted(preference_kinds.items())),
            "fit_boundaries": {
                "initial_predictions_are_out_of_fold": True,
                "evidence_compiler_label_access": False,
                "transcript_compiler_label_access": False,
                "gold_label_in_model_messages": False,
                "dev_or_test_labels_accessed": False,
                "cross_dataset_merging": False,
            },
            "retrieval": {
                "max_segments": request.max_segments,
                "max_atomic_per_segment": request.max_atomic_per_segment,
                "max_observations_per_modality": request.max_observations_per_modality,
            },
            "annotator": {
                "draft_probability": "frozen_out_of_fold_prediction",
                "draft_model_filter": request.draft_model,
                "draft_view_filter": request.draft_view,
                "source_probabilities": f"{request.source_model}_source_view_oof",
                "evidence_selection": "label_free_quality_and_deviation_weak_teacher",
                "revision_action": "oof_state_transition_supervision",
                "unresolved_error_fraction": request.unresolved_error_fraction,
                "false_negative_weight": request.false_negative_weight,
            },
            "inputs": {
                "oof_predictions": {
                    "path": str(request.oof_predictions.resolve()),
                    "sha256": _sha256_file(request.oof_predictions),
                },
                "source_predictions": {
                    "path": str(request.source_predictions.resolve()),
                    "sha256": _sha256_file(request.source_predictions),
                },
                "evidence_root": str(request.evidence_root.resolve()),
                "transcript_root": (
                    str(request.transcript_root.resolve())
                    if request.transcript_root is not None
                    else None
                ),
                "transcript_tier": (
                    request.transcript_tier if request.dataset != "d_vlog" else "unavailable"
                ),
                "session_ids_sha256": _sha256_ids(
                    [draft.session_id for draft in drafts]
                ),
            },
            "files": files,
        }
        manifest_text = json.dumps(
            manifest, ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n"
        (temporary / "manifest.json").write_text(manifest_text, encoding="utf-8")
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build grounded Draft/Loop/ORPO records from frozen OOF predictions."
    )
    parser.add_argument("--dataset", required=True, choices=sorted(_DATASETS))
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--transcript-root", type=Path)
    parser.add_argument(
        "--transcript-tier",
        choices=("full", "compact", "essential", "minimal"),
        default="compact",
    )
    parser.add_argument("--oof-predictions", required=True, type=Path)
    parser.add_argument("--source-predictions", required=True, type=Path)
    parser.add_argument("--source-model", default="logistic")
    parser.add_argument("--draft-model")
    parser.add_argument("--draft-view")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-segments", type=int, default=3)
    parser.add_argument("--max-atomic-per-segment", type=int, default=4)
    parser.add_argument("--max-observations-per-modality", type=int, default=3)
    parser.add_argument("--unresolved-error-fraction", type=float, default=0.25)
    parser.add_argument("--false-negative-weight", type=float, default=2.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_package(
        PackageRequest(
            dataset=args.dataset,
            task_path=args.task,
            evidence_root=args.evidence_root,
            transcript_root=args.transcript_root,
            transcript_tier=args.transcript_tier,
            oof_predictions=args.oof_predictions,
            source_predictions=args.source_predictions,
            source_model=args.source_model,
            draft_model=args.draft_model,
            draft_view=args.draft_view,
            split=args.split,
            output_dir=args.output_dir,
            max_segments=args.max_segments,
            max_atomic_per_segment=args.max_atomic_per_segment,
            max_observations_per_modality=args.max_observations_per_modality,
            unresolved_error_fraction=args.unresolved_error_fraction,
            false_negative_weight=args.false_negative_weight,
        )
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
