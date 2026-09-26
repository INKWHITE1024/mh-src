"""Two-pass evidence-grounded rethinking workflow driven by a trigger policy."""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from .contracts import InitialAssessment, RevisionAssessment, TaskSpec
from .prompts import PromptBuilder
from .retrieval import EvidenceStore, TargetedEvidence
from .trigger import RethinkDecision, RethinkPolicy


WorkflowOutcome = Literal[
    "accepted_initial",
    "accepted_after_preservation",
    "accepted_after_revision",
    "referred_unresolved",
]


class EvidenceGroundingError(ValueError):
    """Raised when a model cites evidence outside the supplied evidence store."""


class ModelCompletionError(ValueError):
    """Raised when a model completion is not exactly one JSON object."""


class CompletionModel(Protocol):
    """Seam between the workflow and local/remote model inference."""

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str | Mapping[str, Any]:
        ...


class RethinkPolicyProtocol(Protocol):
    """Injection seam for the heuristic baseline or a later learned OOF trigger."""

    def evaluate(
        self,
        assessment: InitialAssessment,
        available_segment_ids: Sequence[str],
    ) -> RethinkDecision:
        ...


@dataclass(frozen=True)
class PreparedInitialPass:
    messages: tuple[dict[str, str], ...]
    available_segment_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": [dict(message) for message in self.messages],
            "available_segment_ids": list(self.available_segment_ids),
        }


@dataclass(frozen=True)
class PreparedRevisionPass:
    decision: RethinkDecision
    targeted_evidence: TargetedEvidence
    messages: tuple[dict[str, str], ...]

    def to_dict(self, *, include_evidence_text: bool = True) -> dict[str, Any]:
        output = {
            "decision": _decision_dict(self.decision),
            "selected_segment_ids": list(
                self.targeted_evidence.selected_segment_ids
            ),
            "selected_atomic_evidence_ids": list(
                self.targeted_evidence.selected_atomic_evidence_ids
            ),
            "canonical_evidence_ids": list(
                self.targeted_evidence.canonical_evidence_ids
            ),
            "messages": [dict(message) for message in self.messages],
        }
        if include_evidence_text:
            output["targeted_evidence"] = self.targeted_evidence.text
        return output


@dataclass(frozen=True)
class WorkflowResult:
    task: TaskSpec
    outcome: WorkflowOutcome
    initial_assessment: InitialAssessment
    decision: RethinkDecision
    revision_assessment: RevisionAssessment | None
    final_risk_probability: float
    final_confidence: float
    reviewed_segment_ids: tuple[str, ...]
    reviewed_evidence_ids: tuple[str, ...]
    canonical_evidence_ids: tuple[str, ...]
    model_call_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "task": self.task.to_dict(),
            "outcome": self.outcome,
            "initial_assessment": self.initial_assessment.to_dict(),
            "rethink_decision": _decision_dict(self.decision),
            "revision_assessment": (
                self.revision_assessment.to_dict()
                if self.revision_assessment is not None
                else None
            ),
            "final_risk_probability": self.final_risk_probability,
            "final_confidence": self.final_confidence,
            "reviewed_segment_ids": list(self.reviewed_segment_ids),
            "reviewed_evidence_ids": list(self.reviewed_evidence_ids),
            "canonical_evidence_ids": list(self.canonical_evidence_ids),
            "model_call_count": self.model_call_count,
            "ground_truth_used_by_workflow": False,
        }


class RethinkingWorkflow:
    """Run at most two evidence-grounded model passes behind one interface."""

    def __init__(
        self,
        policy: RethinkPolicyProtocol | None = None,
        prompt_builder: PromptBuilder | None = None,
        *,
        max_atomic_per_segment: int = 4,
        max_observations_per_modality: int = 3,
    ) -> None:
        if max_atomic_per_segment <= 0:
            raise ValueError("max_atomic_per_segment must be positive")
        if max_observations_per_modality <= 0:
            raise ValueError("max_observations_per_modality must be positive")
        self.policy = policy or RethinkPolicy()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.max_atomic_per_segment = max_atomic_per_segment
        self.max_observations_per_modality = max_observations_per_modality

    def prepare_initial(
        self, session_dir: str | Path, task: TaskSpec
    ) -> PreparedInitialPass:
        directory = _v2_session_directory(session_dir)
        store = EvidenceStore.from_session_dir(directory)
        session_text = (directory / "session_input.native.txt").read_text(encoding="utf-8")
        messages = self.prompt_builder.build_initial_messages(task, session_text)
        return PreparedInitialPass(
            messages=tuple(dict(message) for message in messages),
            available_segment_ids=store.available_segment_ids,
        )

    def prepare_revision(
        self,
        session_dir: str | Path,
        task: TaskSpec,
        initial: InitialAssessment | Mapping[str, Any],
    ) -> PreparedRevisionPass | None:
        directory = _v2_session_directory(session_dir)
        store = EvidenceStore.from_session_dir(directory)
        assessment = (
            initial
            if isinstance(initial, InitialAssessment)
            else InitialAssessment.from_dict(initial)
        )
        decision = self.evaluate_initial(directory, assessment)
        if not decision.should_rethink:
            return None
        targeted = store.render_segments(
            decision.selected_segment_ids,
            max_atomic_per_segment=self.max_atomic_per_segment,
            max_observations_per_modality=self.max_observations_per_modality,
        )
        messages = self.prompt_builder.build_revision_messages(
            task,
            assessment,
            _decision_dict(decision),
            targeted.text,
        )
        return PreparedRevisionPass(
            decision=decision,
            targeted_evidence=targeted,
            messages=tuple(dict(message) for message in messages),
        )

    def evaluate_initial(
        self,
        session_dir: str | Path,
        initial: InitialAssessment | Mapping[str, Any],
    ) -> RethinkDecision:
        directory = _v2_session_directory(session_dir)
        store = EvidenceStore.from_session_dir(directory)
        assessment = (
            initial
            if isinstance(initial, InitialAssessment)
            else InitialAssessment.from_dict(initial)
        )
        _validate_initial_grounding(assessment, store.available_segment_ids)
        return self.policy.evaluate(assessment, store.available_segment_ids)

    def finalize(
        self,
        session_dir: str | Path,
        task: TaskSpec,
        initial: InitialAssessment | Mapping[str, Any],
        revision: RevisionAssessment | Mapping[str, Any] | None = None,
    ) -> WorkflowResult:
        directory = _v2_session_directory(session_dir)
        store = EvidenceStore.from_session_dir(directory)
        assessment = (
            initial
            if isinstance(initial, InitialAssessment)
            else InitialAssessment.from_dict(initial)
        )
        _validate_initial_grounding(assessment, store.available_segment_ids)
        decision = self.policy.evaluate(assessment, store.available_segment_ids)
        if not decision.should_rethink:
            if revision is not None:
                raise ValueError("revision must be absent when the policy accepts the initial pass")
            return WorkflowResult(
                task=task,
                outcome="accepted_initial",
                initial_assessment=assessment,
                decision=decision,
                revision_assessment=None,
                final_risk_probability=assessment.risk_probability,
                final_confidence=assessment.confidence,
                reviewed_segment_ids=(),
                reviewed_evidence_ids=(),
                canonical_evidence_ids=(),
                model_call_count=1,
            )

        if revision is None:
            raise ValueError("revision is required when the rethink policy triggers")
        revised = (
            revision
            if isinstance(revision, RevisionAssessment)
            else RevisionAssessment.from_dict(revision)
        )
        targeted = store.render_segments(
            decision.selected_segment_ids,
            max_atomic_per_segment=self.max_atomic_per_segment,
            max_observations_per_modality=self.max_observations_per_modality,
        )
        _validate_revision_grounding(revised, targeted)
        outcome: WorkflowOutcome = {
            "preserved": "accepted_after_preservation",
            "revised": "accepted_after_revision",
            "unresolved": "referred_unresolved",
        }[revised.revision_status]
        return WorkflowResult(
            task=task,
            outcome=outcome,
            initial_assessment=assessment,
            decision=decision,
            revision_assessment=revised,
            final_risk_probability=revised.revised_risk_probability,
            final_confidence=revised.revised_confidence,
            reviewed_segment_ids=targeted.selected_segment_ids,
            reviewed_evidence_ids=targeted.selected_atomic_evidence_ids,
            canonical_evidence_ids=targeted.canonical_evidence_ids,
            model_call_count=2,
        )

    def run(
        self,
        session_dir: str | Path,
        task: TaskSpec,
        model: CompletionModel
        | Callable[[Sequence[Mapping[str, str]]], str | Mapping[str, Any]],
    ) -> WorkflowResult:
        prepared_initial = self.prepare_initial(session_dir, task)
        initial_payload = _parse_completion(
            _invoke_model(model, prepared_initial.messages), "initial model completion"
        )
        initial = InitialAssessment.from_dict(initial_payload)

        decision = self.evaluate_initial(session_dir, initial)
        if not decision.should_rethink:
            return self.finalize(session_dir, task, initial)

        prepared_revision = self.prepare_revision(session_dir, task, initial)
        if prepared_revision is None:  # Defensive: policy inputs are deterministic.
            raise RuntimeError("rethink decision changed between workflow stages")
        revision_payload = _parse_completion(
            _invoke_model(model, prepared_revision.messages),
            "revision model completion",
        )
        revision = RevisionAssessment.from_dict(revision_payload)
        return self.finalize(session_dir, task, initial, revision)


def _v2_session_directory(path: str | Path) -> Path:
    directory = Path(path).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"native session directory does not exist: {directory}")
    required = (
        "session_input.native.txt",
        "atomic_evidence.native.txt",
        "evidence_index.native.jsonl",
        "evidence_units.jsonl",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Not a complete native Evidence session; missing: {missing}"
        )
    return directory


def _validate_initial_grounding(
    assessment: InitialAssessment, available_segment_ids: Sequence[str]
) -> None:
    available = set(available_segment_ids)
    cited = {
        segment_id
        for field in (
            assessment.supporting_segment_ids,
            assessment.contradictory_segment_ids,
            assessment.uncertain_segment_ids,
            assessment.requested_segment_ids,
        )
        for segment_id in field
    }
    unknown = cited - available
    if unknown:
        raise EvidenceGroundingError(
            f"Initial assessment cites unknown segment IDs: {sorted(unknown)}"
        )


def _validate_revision_grounding(
    revision: RevisionAssessment, targeted: TargetedEvidence
) -> None:
    available_segments = set(targeted.selected_segment_ids)
    available_evidence = set(targeted.selected_atomic_evidence_ids)
    segment_ids = set(revision.cited_segment_ids) | set(
        revision.residual_conflict_segment_ids
    )
    evidence_ids = (
        set(revision.cited_evidence_ids)
        | set(revision.preserved_evidence_ids)
        | set(revision.newly_considered_evidence_ids)
        | set(revision.rejected_evidence_ids)
    )
    unknown_segments = segment_ids - available_segments
    unknown_evidence = evidence_ids - available_evidence
    if unknown_segments:
        raise EvidenceGroundingError(
            f"Revision cites segments outside targeted evidence: {sorted(unknown_segments)}"
        )
    if unknown_evidence:
        raise EvidenceGroundingError(
            f"Revision cites atomic evidence outside targeted evidence: "
            f"{sorted(unknown_evidence)}"
        )


def _decision_dict(decision: RethinkDecision) -> dict[str, Any]:
    return decision.to_dict()


def _invoke_model(
    model: CompletionModel
    | Callable[[Sequence[Mapping[str, str]]], str | Mapping[str, Any]],
    messages: Sequence[Mapping[str, str]],
) -> str | Mapping[str, Any]:
    completion = getattr(model, "complete", None)
    if callable(completion):
        return completion(messages)
    if callable(model):
        return model(messages)
    raise TypeError("model must be callable or expose complete(messages)")


_JSON_FENCE = re.compile(r"\A```(?:json)?\s*(\{.*\})\s*```\Z", re.DOTALL)


def _parse_completion(value: Any, path: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str):
        raise ModelCompletionError(f"{path} must be a JSON object or JSON text")
    text = value.strip()
    fenced = _JSON_FENCE.fullmatch(text)
    if fenced:
        text = fenced.group(1)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ModelCompletionError(f"{path} is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, Mapping):
        raise ModelCompletionError(f"{path} must contain exactly one JSON object")
    return payload
