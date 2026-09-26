"""Four-turn evidence-grounded rethinking workflow with explicit query and grounded atomic selection."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import InitialAssessment, RevisionAssessment, TaskSpec
from .native_query_retrieval import NativeAtomicRetriever
from .prompts import PromptBuilder
from .query_retrieval import (
    AtomicSelection,
    CandidateSet,
    EvidenceQuery,
    RETRIEVAL_PROTOCOL_VERSION,
)
from .retrieval_prompts import QueryRetrievalPromptBuilder
from .trigger import RethinkDecision, RethinkPolicy
from .workflow import (
    CompletionModel,
    RethinkPolicyProtocol,
    WorkflowOutcome,
    _invoke_model,
    _parse_completion,
    _validate_initial_grounding,
    _validate_revision_grounding,
)


@dataclass(frozen=True, slots=True)
class QueryGuidedWorkflowResult:
    """Final prediction and complete provenance for one query-guided run."""

    task: TaskSpec
    outcome: WorkflowOutcome
    initial_assessment: InitialAssessment
    decision: RethinkDecision
    evidence_query: EvidenceQuery | None
    candidate_set: CandidateSet | None
    atomic_selection: AtomicSelection | None
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
            "evidence_protocol_version": "native",
            "retrieval_protocol_version": RETRIEVAL_PROTOCOL_VERSION,
            "task": self.task.to_dict(),
            "outcome": self.outcome,
            "initial_assessment": self.initial_assessment.to_dict(),
            "rethink_decision": self.decision.to_dict(),
            "evidence_query": (
                self.evidence_query.to_dict()
                if self.evidence_query is not None
                else None
            ),
            "candidate_set": (
                self.candidate_set.to_dict()
                if self.candidate_set is not None
                else None
            ),
            "atomic_selection": (
                self.atomic_selection.to_dict()
                if self.atomic_selection is not None
                else None
            ),
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


class QueryGuidedRethinkingWorkflow:
    """Run initial, query, grounded selection, and revision over native evidence."""

    def __init__(
        self,
        policy: RethinkPolicyProtocol | None = None,
        initial_prompt_builder: PromptBuilder | None = None,
        retrieval_prompt_builder: QueryRetrievalPromptBuilder | None = None,
        *,
        max_candidates_per_query: int = 8,
    ) -> None:
        self.policy = policy or RethinkPolicy()
        self.initial_prompt_builder = initial_prompt_builder or PromptBuilder()
        self.retrieval_prompt_builder = (
            retrieval_prompt_builder or QueryRetrievalPromptBuilder()
        )
        self.max_candidates_per_query = max_candidates_per_query

    def run(
        self,
        session_dir: str | Path,
        task: TaskSpec,
        model: CompletionModel
        | Callable[
            [Sequence[Mapping[str, str]]],
            str | Mapping[str, Any],
        ],
    ) -> QueryGuidedWorkflowResult:
        retriever = NativeAtomicRetriever.from_session_dir(
            session_dir,
            max_candidates_per_query=self.max_candidates_per_query,
        )
        session_text = retriever.session_input_text

        initial_messages = self.initial_prompt_builder.build_initial_messages(
            task,
            session_text,
        )
        initial_payload = _parse_completion(
            _invoke_model(model, initial_messages),
            "initial model completion",
        )
        initial = InitialAssessment.from_dict(initial_payload)
        _validate_initial_grounding(
            initial,
            retriever.available_segment_ids,
        )
        decision = self.policy.evaluate(
            initial,
            retriever.available_segment_ids,
        )
        if not decision.should_rethink:
            return QueryGuidedWorkflowResult(
                task=task,
                outcome="accepted_initial",
                initial_assessment=initial,
                decision=decision,
                evidence_query=None,
                candidate_set=None,
                atomic_selection=None,
                revision_assessment=None,
                final_risk_probability=initial.risk_probability,
                final_confidence=initial.confidence,
                reviewed_segment_ids=(),
                reviewed_evidence_ids=(),
                canonical_evidence_ids=(),
                model_call_count=1,
            )

        retrieval_map = retriever.render_retrieval_map(
            decision.selected_segment_ids
        )
        query_messages = self.retrieval_prompt_builder.build_query_messages(
            task,
            initial,
            decision,
            retrieval_map,
        )
        query_payload = _parse_completion(
            _invoke_model(model, query_messages),
            "evidence query model completion",
        )
        query = EvidenceQuery.from_dict(query_payload)
        if query.segment_id not in decision.selected_segment_ids:
            raise ValueError(
                "evidence query segment was not selected by the rethink decision"
            )

        candidate_set = retriever.shortlist(query)
        selection_messages = (
            self.retrieval_prompt_builder.build_selection_messages(
                task,
                initial,
                candidate_set,
            )
        )
        selection_payload = _parse_completion(
            _invoke_model(model, selection_messages),
            "atomic selection model completion",
        )
        selection = AtomicSelection.from_dict(selection_payload)
        targeted = retriever.materialize(candidate_set, selection)

        revision_messages = self.initial_prompt_builder.build_revision_messages(
            task,
            initial,
            decision,
            targeted.text,
        )
        revision_payload = _parse_completion(
            _invoke_model(model, revision_messages),
            "revision model completion",
        )
        revision = RevisionAssessment.from_dict(revision_payload)
        _validate_revision_grounding(revision, targeted)
        outcome: WorkflowOutcome = {
            "preserved": "accepted_after_preservation",
            "revised": "accepted_after_revision",
            "unresolved": "referred_unresolved",
        }[revision.revision_status]
        return QueryGuidedWorkflowResult(
            task=task,
            outcome=outcome,
            initial_assessment=initial,
            decision=decision,
            evidence_query=query,
            candidate_set=candidate_set,
            atomic_selection=selection,
            revision_assessment=revision,
            final_risk_probability=revision.revised_risk_probability,
            final_confidence=revision.revised_confidence,
            reviewed_segment_ids=targeted.selected_segment_ids,
            reviewed_evidence_ids=targeted.selected_atomic_evidence_ids,
            canonical_evidence_ids=targeted.canonical_evidence_ids,
            model_call_count=4,
        )


__all__ = [
    "QueryGuidedRethinkingWorkflow",
    "QueryGuidedWorkflowResult",
]
