"""Collect label-free, fold-local full-loop trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rethink_mh.rethinking.contracts import (
    InitialAssessment,
    RevisionAssessment,
    TaskSpec,
)
from rethink_mh.rethinking.evidence_agent import (
    CompletionAdapter,
    EvidenceAgentConfig,
    EvidenceAgentStageModels,
    EvidenceAgentWorkflow,
    TokenizerTokenCounter,
    _complete_with_contract_repair,
)
from rethink_mh.rethinking.native_query_retrieval import (
    NativeAtomicRetriever,
)
from rethink_mh.rethinking.prompts import PromptBuilder
from rethink_mh.rethinking.query_retrieval import (
    AtomicSelection,
    CandidateSet,
    EvidenceQuery,
)
from rethink_mh.rethinking.qwen import (
    QwenAdapterDisabledCompletionModel,
    QwenThinkerCompletionModel,
)
from rethink_mh.rethinking.retrieval import TargetedEvidence
from rethink_mh.rethinking.trigger import RethinkDecision, RethinkPolicy
from rethink_mh.rethinking.workflow import _validate_revision_grounding

from .evidence_literacy import (
    best_label_free_query,
    enumerate_label_free_queries,
)
from .loop_prepare import PREPARATION_PROTOCOL_VERSION


COLLECTION_PROTOCOL_VERSION = "loop-collection"
ARM_ORDER = (
    "targeted_learned",
    "targeted_rank_top",
    "deterministic",
    "random_candidate",
    "full_candidate_context",
    "reflection_no_new_evidence",
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be one JSON object: {path}")
    return value


def _read_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read {description} {path}: {error}") from error
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


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    text = "".join(_canonical_json(row) + "\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _probability(value: Any, path: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{path} must be a finite probability")
    return float(value)


def _safe_error(error: Exception) -> dict[str, str]:
    message = " ".join(str(error).split())[:800]
    return {
        "error_type": error.__class__.__name__,
        "error_message": message,
    }


def _task(path: Path) -> TaskSpec:
    return TaskSpec.from_dict(_read_json(path, "task specification"))


class _NeverRethinkPolicy:
    def evaluate(
        self,
        assessment: InitialAssessment,
        available_segment_ids: Sequence[str],
    ) -> RethinkDecision:
        del assessment, available_segment_ids
        return RethinkDecision(
            should_rethink=False,
            trigger_score=0.0,
            reasons=(),
            selected_segment_ids=(),
            metrics={"collection_initial_only": 1},
        )


@dataclass(frozen=True, slots=True)
class _FixedSegmentsPolicy:
    segment_ids: tuple[str, ...]

    def evaluate(
        self,
        assessment: InitialAssessment,
        available_segment_ids: Sequence[str],
    ) -> RethinkDecision:
        del assessment
        unknown = set(self.segment_ids) - set(available_segment_ids)
        if not self.segment_ids or unknown:
            raise ValueError(
                f"fixed collection segments are empty or unknown: {sorted(unknown)}"
            )
        return RethinkDecision(
            should_rethink=True,
            trigger_score=1.0,
            reasons=("high_predictive_entropy",),
            selected_segment_ids=self.segment_ids,
            metrics={"counterfactual_collection_intervention": 1},
        )


@dataclass(slots=True)
class _StaticCompletion:
    payload: Mapping[str, Any]
    call_count: int = 0

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
    ) -> Mapping[str, Any]:
        del messages
        self.call_count += 1
        return dict(self.payload)


@dataclass(frozen=True, slots=True)
class LoopCollectionRequest:
    prepared_dir: Path
    task_path: Path
    output_dir: Path
    fold_index: int
    device: str = "cuda"
    dtype: str = "bfloat16"
    attention_implementation: str = "sdpa"
    max_new_tokens: int = 640
    max_input_tokens: int = 8_192
    max_contract_retries: int = 2
    random_seed: int = 20_260_726
    max_sessions: int | None = None
    session_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.fold_index, bool)
            or not isinstance(self.fold_index, int)
            or self.fold_index < 0
        ):
            raise ValueError("fold_index must be a non-negative integer")
        for name in ("max_new_tokens", "max_input_tokens", "random_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self.max_contract_retries <= 2:
            raise ValueError("max_contract_retries must be 0, 1, or 2")
        if self.max_sessions is not None and (
            isinstance(self.max_sessions, bool)
            or not isinstance(self.max_sessions, int)
            or self.max_sessions <= 0
        ):
            raise ValueError("max_sessions must be a positive integer")
        if (
            not isinstance(self.session_ids, tuple)
            or any(
                not isinstance(value, str) or not value.strip()
                for value in self.session_ids
            )
            or len(set(self.session_ids)) != len(self.session_ids)
        ):
            raise ValueError(
                "session_ids must be a tuple of unique non-empty strings"
            )

    @property
    def collection_protocol_version(self) -> str:
        return COLLECTION_PROTOCOL_VERSION


CollectionModels = CompletionAdapter | EvidenceAgentStageModels
ModelLoader = Callable[
    [str, str | None, LoopCollectionRequest],
    CollectionModels,
]


def _default_model_loader(
    model_path: str,
    adapter_path: str | None,
    request: LoopCollectionRequest,
) -> CollectionModels:
    reviewer = QwenThinkerCompletionModel.from_pretrained(
        model_path,
        adapter_path=adapter_path,
        device=request.device,
        dtype=request.dtype,
        attention_implementation=request.attention_implementation,
        max_new_tokens=request.max_new_tokens,
        max_input_tokens=request.max_input_tokens,
    )
    if adapter_path is None:
        return EvidenceAgentStageModels.uniform(reviewer)
    base = QwenAdapterDisabledCompletionModel(reviewer)
    return EvidenceAgentStageModels(
        initial=base,
        query=reviewer,
        selection=reviewer,
        revision=base,
    )


def _validate_plan(
    request: LoopCollectionRequest,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = request.prepared_dir / "manifest.json"
    plan_path = request.prepared_dir / "inference_plan.jsonl"
    manifest = _read_json(manifest_path, "full-loop preparation manifest")
    plan = _read_jsonl(plan_path, "label-free inference plan")
    files = manifest.get("files")
    if (
        manifest.get("preparation_protocol_version")
        != PREPARATION_PROTOCOL_VERSION
        or not isinstance(files, Mapping)
        or files.get("inference_plan.jsonl", {}).get("sha256")
        != _sha256(plan_path)
        or files.get("inference_plan.jsonl", {}).get(
            "contains_current_sample_targets"
        )
        is not False
    ):
        raise ValueError("prepared inference plan failed provenance validation")
    selected = [
        row for row in plan if row.get("outer_fold") == request.fold_index
    ]
    if not selected:
        raise ValueError(
            f"inference plan contains no outer fold {request.fold_index}"
        )
    if request.session_ids:
        requested = set(request.session_ids)
        selected = [
            row
            for row in selected
            if str(row.get("session_id", "")) in requested
        ]
        found = {str(row.get("session_id", "")) for row in selected}
        if found != requested:
            raise ValueError(
                "requested sessions are not all present in the selected "
                f"outer fold: {sorted(requested - found)}"
            )
    if request.max_sessions is not None:
        selected = selected[: request.max_sessions]
    seen: set[str] = set()
    route_identities: set[tuple[str, str | None, str]] = set()
    for row in selected:
        session_id = str(row.get("session_id", "")).strip()
        evidence = row.get("evidence")
        reviewer = row.get("reviewer")
        boundaries = row.get("fit_boundaries")
        if any(
            (
                row.get("preparation_protocol_version")
                != PREPARATION_PROTOCOL_VERSION,
                not session_id,
                session_id in seen,
                not isinstance(evidence, Mapping),
                not isinstance(reviewer, Mapping),
                not isinstance(boundaries, Mapping),
                boundaries.get("sample_supervision_visible") is not False,
                boundaries.get("offline_score_visible") is not False,
            )
        ):
            raise ValueError(f"unsafe inference-plan row for session {session_id}")
        seen.add(session_id)
        assert isinstance(evidence, Mapping)
        assert isinstance(reviewer, Mapping)
        session_dir = Path(str(evidence.get("session_dir", "")))
        transcript_dir = Path(str(evidence.get("transcript_dir", "")))
        if (
            _sha256(session_dir / "manifest.json")
            != evidence.get("manifest_sha256")
            or _sha256(transcript_dir / "transcript_manifest.compact.json")
            != evidence.get("transcript_manifest_sha256")
        ):
            raise ValueError(f"Evidence identity changed for session {session_id}")
        model_path = str(reviewer.get("model_path", "")).strip()
        raw_adapter = reviewer.get("adapter_path")
        adapter_path = None if raw_adapter is None else str(raw_adapter).strip()
        if not model_path:
            raise ValueError(f"reviewer model path is empty for {session_id}")
        if reviewer.get("reviewer_kind") == "literacy_adapter" and not adapter_path:
            raise ValueError(f"literacy route omitted adapter for {session_id}")
        route_identities.add(
            (
                model_path,
                adapter_path,
                str(reviewer.get("heldout_summary_sha256", "")),
            )
        )
        _probability(
            row.get("initial_probability"),
            f"{session_id}.initial_probability",
        )
        _probability(
            row.get("initial_threshold"),
            f"{session_id}.initial_threshold",
        )
    if len(route_identities) != 1:
        raise ValueError("one fold contains multiple reviewer routes")
    return selected, manifest


def _stage_models(model: CollectionModels) -> EvidenceAgentStageModels:
    if isinstance(model, EvidenceAgentStageModels):
        return model
    return EvidenceAgentStageModels.uniform(model)


def _counter(models: EvidenceAgentStageModels) -> TokenizerTokenCounter:
    tokenizers = [
        getattr(getattr(models, stage), "tokenizer", None)
        for stage in ("initial", "query", "selection", "revision")
    ]
    available = [tokenizer for tokenizer in tokenizers if tokenizer is not None]
    if not available:
        raise ValueError("collection model must expose its inference tokenizer")
    identities = {
        (
            tokenizer.__class__.__name__,
            str(getattr(tokenizer, "name_or_path", "")),
        )
        for tokenizer in available
    }
    if len(identities) != 1:
        raise ValueError(
            "collection stage models use different tokenizers"
        )
    return TokenizerTokenCounter.from_tokenizer(available[0])


def _initial_result(
    *,
    row: Mapping[str, Any],
    task: TaskSpec,
    models: EvidenceAgentStageModels,
    counter: TokenizerTokenCounter,
    path: Path,
    ledger_path: Path,
    retries: int,
) -> dict[str, Any]:
    if path.is_file():
        value = _read_json(path, "initial collection result")
        if value.get("collection_protocol_version") != COLLECTION_PROTOCOL_VERSION:
            raise ValueError(
                "stored initial assessment uses a different collection "
                "contract policy"
            )
        assessment = InitialAssessment.from_dict(value["initial_assessment"])
        expected = _probability(
            row["initial_probability"],
            "initial_probability",
        )
        if not math.isclose(
            assessment.risk_probability,
            expected,
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise ValueError("stored initial assessment changed its OOF anchor")
        return value
    if ledger_path.exists():
        raise RuntimeError(
            f"partial initial ledger requires audit before resume: {ledger_path}"
        )
    evidence = row["evidence"]
    session_dir = Path(str(evidence["session_dir"]))
    transcript_dir = Path(str(evidence["transcript_dir"]))
    result = EvidenceAgentWorkflow(
        policy=_NeverRethinkPolicy(),
        max_contract_retries=retries,
        initial_grounding_recovery="drop_unknown_ids_after_retries",
    ).run(
        session_dir,
        task,
        models,
        ledger_path=ledger_path,
        transcript_dir=transcript_dir,
        token_counter=counter,
        evidence_protocol="native",
        run_id=uuid.uuid4().hex,
        initial_probability_anchor=float(row["initial_probability"]),
    )
    retriever = NativeAtomicRetriever.from_session_dir(session_dir)
    actual_decision = RethinkPolicy().evaluate(
        result.initial_assessment,
        retriever.available_segment_ids,
    )
    value = {
        "schema_version": "1.0.0",
        "collection_protocol_version": COLLECTION_PROTOCOL_VERSION,
        "dataset": row["dataset"],
        "session_id": row["session_id"],
        "outer_fold": row["outer_fold"],
        "initial_probability_anchor": row["initial_probability"],
        "initial_threshold": row["initial_threshold"],
        "initial_assessment": result.initial_assessment.to_dict(),
        "natural_rethink_decision": actual_decision.to_dict(),
        "first_pass_tokens": result.first_pass_tokens,
        "transcript_tier": result.transcript_tier,
        "model_call_count": result.model_call_count,
        "contract_repair_count": result.contract_repair_count,
        "access_ledger": result.ledger.to_dict(),
        "access_ledger_path": str(ledger_path.resolve()),
        "access_ledger_sha256": _sha256(ledger_path),
        "ground_truth_used": False,
    }
    value["initial_contract_policy"] = (
        "allowlist-audited-drop-nonallowlisted"
    )
    value["initial_grounding_audit"] = (
        dict(result.initial_grounding_audit)
        if result.initial_grounding_audit is not None
        else {
            "policy_version": "initial-grounding-recovery",
            "applied": False,
            "trigger": None,
        }
    )
    _write_json(path, value)
    return value


def _collection_segments(
    retriever: NativeAtomicRetriever,
    session_id: str,
    natural: RethinkDecision,
) -> tuple[str, ...]:
    if natural.selected_segment_ids:
        return natural.selected_segment_ids
    queries = enumerate_label_free_queries(
        retriever,
        session_id,
        maximum=5,
        budget=2,
    )
    selected = tuple(
        dict.fromkeys(query.segment_id for query, _ in queries)
    )[:3]
    if not selected:
        raise ValueError(f"session {session_id} has no collection segment")
    return selected


def _stable_random_selection(
    candidate_set: CandidateSet,
    *,
    session_id: str,
    seed: int,
) -> AtomicSelection:
    identifiers = list(candidate_set.candidate_evidence_ids)
    rng = random.Random(
        int(
            hashlib.sha256(
                f"{seed}:{session_id}:{candidate_set.query.to_dict()}".encode()
            ).hexdigest()[:16],
            16,
        )
    )
    rng.shuffle(identifiers)
    count = min(candidate_set.query.budget, len(identifiers))
    return AtomicSelection(tuple(identifiers[:count]))


def _top_selection(candidate_set: CandidateSet) -> AtomicSelection:
    count = min(
        candidate_set.query.budget,
        len(candidate_set.candidate_evidence_ids),
    )
    return AtomicSelection(candidate_set.candidate_evidence_ids[:count])


def _static_stage_models(
    models: EvidenceAgentStageModels,
    query: EvidenceQuery,
    selection: AtomicSelection,
) -> EvidenceAgentStageModels:
    return EvidenceAgentStageModels(
        initial=models.initial,
        query=_StaticCompletion(query.to_dict()),
        selection=_StaticCompletion(selection.to_dict()),
        revision=models.revision,
    )


def _run_release_arm(
    *,
    row: Mapping[str, Any],
    task: TaskSpec,
    initial: InitialAssessment,
    models: EvidenceAgentStageModels,
    counter: TokenizerTokenCounter,
    segments: tuple[str, ...],
    ledger_path: Path,
    retries: int,
    query: EvidenceQuery | None = None,
    selection: AtomicSelection | None = None,
) -> dict[str, Any]:
    if ledger_path.exists():
        raise RuntimeError(
            f"partial arm ledger requires audit before resume: {ledger_path}"
        )
    arm_models: EvidenceAgentStageModels
    if query is None or selection is None:
        if query is not None or selection is not None:
            raise ValueError("static query and selection must be supplied together")
        arm_models = models
    else:
        arm_models = _static_stage_models(models, query, selection)
    evidence = row["evidence"]
    result = EvidenceAgentWorkflow(
        config=EvidenceAgentConfig(
            max_queries=1,
            max_atomic_records=4,
            max_released_tokens=2_400,
            max_first_pass_tokens=5_000,
            max_candidates_per_query=8,
        ),
        policy=_FixedSegmentsPolicy(segments),
        max_contract_retries=retries,
    ).run_from_initial(
        Path(str(evidence["session_dir"])),
        task,
        initial,
        arm_models,
        ledger_path=ledger_path,
        transcript_dir=Path(str(evidence["transcript_dir"])),
        token_counter=counter,
        evidence_protocol="native",
        run_id=uuid.uuid4().hex,
    )
    return result.to_dict()


def _reflection_arm(
    *,
    task: TaskSpec,
    initial: InitialAssessment,
    model: CompletionAdapter,
    segments: tuple[str, ...],
    retries: int,
) -> dict[str, Any]:
    decision = _FixedSegmentsPolicy(segments).evaluate(initial, segments)
    text = (
        "NO NEW ATOMIC EVIDENCE | reflection control\n"
        "- No hidden measurement was queried or released in this arm.\n"
        "- Reconsider only the already visible InitialAssessment and its "
        "segment references.\n"
        "- visible segment identifiers: "
        + ", ".join(segments)
        + "\n"
    )
    targeted = TargetedEvidence(
        text=text,
        selected_segment_ids=segments,
        selected_atomic_evidence_ids=(),
        canonical_evidence_ids=(),
    )
    messages = PromptBuilder.build_reflection_messages(
        task,
        initial,
        decision,
        text,
    )

    def validate(payload: Mapping[str, Any]) -> RevisionAssessment:
        revision = RevisionAssessment.from_dict(payload)
        _validate_revision_grounding(revision, targeted)
        return revision

    revision, calls, repairs = _complete_with_contract_repair(
        model,
        messages,
        "reflection revision model completion",
        validate,
        max_retries=retries,
        repair_guidance=(
            "Because this arm releases no atomic evidence, delete every E "
            "identifier from all four E-ID arrays; each must be []. "
            "Never substitute another E identifier. Keep only visible S "
            "identifiers."
        ),
    )
    outcome = {
        "preserved": "accepted_after_preservation",
        "revised": "accepted_after_revision",
        "unresolved": "referred_unresolved",
    }[revision.revision_status]
    output = {
        "schema_version": "1.0.0",
        "outcome": outcome,
        "initial_assessment": initial.to_dict(),
        "rethink_decision": decision.to_dict(),
        "revision_assessment": revision.to_dict(),
        "final_risk_probability": revision.revised_risk_probability,
        "final_confidence": revision.revised_confidence,
        "budget": {
            "queries_used": 0,
            "atomic_records_released": 0,
            "released_tokens": 0,
        },
        "access_ledger": None,
        "model_call_count": calls,
        "contract_repair_count": repairs,
        "ground_truth_used_by_workflow": False,
    }
    output["revision_contract_policy"] = (
        "reflection-no-new-atomic-evidence"
    )
    return output


def _arm_record(
    *,
    row: Mapping[str, Any],
    arm_id: str,
    natural_triggered: bool,
    policy_origin: str,
    collection_protocol_version: str = COLLECTION_PROTOCOL_VERSION,
    workflow: Mapping[str, Any] | None = None,
    error: Exception | None = None,
) -> dict[str, Any]:
    if (workflow is None) == (error is None):
        raise ValueError("arm record requires exactly one of workflow or error")
    record: dict[str, Any] = {
        "schema_version": "1.0.0",
        "collection_protocol_version": collection_protocol_version,
        "dataset": row["dataset"],
        "session_id": row["session_id"],
        "outer_fold": row["outer_fold"],
        "arm_id": arm_id,
        "natural_triggered": natural_triggered,
        "counterfactual_collection": not natural_triggered,
        "policy_origin": policy_origin,
        "ground_truth_used": False,
        "status": "ok" if workflow is not None else "failed",
    }
    if workflow is not None:
        record["workflow"] = dict(workflow)
    else:
        assert error is not None
        record["failure"] = _safe_error(error)
    return record


def _load_or_run_arm(
    *,
    path: Path,
    runner: Callable[[], Mapping[str, Any]],
    row: Mapping[str, Any],
    arm_id: str,
    natural_triggered: bool,
    policy_origin: str,
    collection_protocol_version: str = COLLECTION_PROTOCOL_VERSION,
) -> dict[str, Any]:
    if path.is_file():
        value = _read_json(path, f"{arm_id} arm result")
        if (
            value.get("arm_id") != arm_id
            or value.get("session_id") != row["session_id"]
            or value.get("outer_fold") != row["outer_fold"]
            or value.get("collection_protocol_version")
            != collection_protocol_version
        ):
            raise ValueError(f"stored {arm_id} arm identity changed")
        return value
    try:
        workflow = runner()
    except Exception as error:
        value = _arm_record(
            row=row,
            arm_id=arm_id,
            natural_triggered=natural_triggered,
            policy_origin=policy_origin,
            collection_protocol_version=collection_protocol_version,
            error=error,
        )
    else:
        value = _arm_record(
            row=row,
            arm_id=arm_id,
            natural_triggered=natural_triggered,
            policy_origin=policy_origin,
            collection_protocol_version=collection_protocol_version,
            workflow=workflow,
        )
    _write_json(path, value)
    return value


def _collect_session(
    *,
    request: LoopCollectionRequest,
    row: Mapping[str, Any],
    task: TaskSpec,
    models: EvidenceAgentStageModels,
    counter: TokenizerTokenCounter,
) -> dict[str, Any]:
    session_id = str(row["session_id"])
    collection_protocol_version = request.collection_protocol_version
    session_root = request.output_dir / "sessions" / session_id
    result_path = session_root / "session_result.json"
    if result_path.is_file():
        stored = _read_json(result_path, "session collection result")
        if (
            stored.get("collection_protocol_version")
            != collection_protocol_version
        ):
            raise ValueError(
                "stored session uses a different collection contract policy"
            )
        return stored
    session_root.mkdir(parents=True, exist_ok=True)
    initial_path = session_root / "initial.json"
    initial_ledger = session_root / "ledgers" / "initial.jsonl"
    try:
        initial_value = _initial_result(
            row=row,
            task=task,
            models=models,
            counter=counter,
            path=initial_path,
            ledger_path=initial_ledger,
            retries=request.max_contract_retries,
        )
        initial = InitialAssessment.from_dict(
            initial_value["initial_assessment"]
        )
        natural = RethinkDecision(
            **{
                "should_rethink": initial_value["natural_rethink_decision"][
                    "should_rethink"
                ],
                "trigger_score": initial_value["natural_rethink_decision"][
                    "trigger_score"
                ],
                "reasons": tuple(
                    initial_value["natural_rethink_decision"]["reasons"]
                ),
                "selected_segment_ids": tuple(
                    initial_value["natural_rethink_decision"][
                        "selected_segment_ids"
                    ]
                ),
                "metrics": dict(
                    initial_value["natural_rethink_decision"]["metrics"]
                ),
            }
        )
    except Exception as error:
        value = {
            "schema_version": "1.0.0",
            "collection_protocol_version": collection_protocol_version,
            "dataset": row["dataset"],
            "session_id": session_id,
            "outer_fold": row["outer_fold"],
            "status": "initial_failed",
            "failure": _safe_error(error),
            "arms": [],
            "ground_truth_used": False,
        }
        _write_json(result_path, value)
        return value

    retriever = NativeAtomicRetriever.from_session_dir(
        Path(str(row["evidence"]["session_dir"]))
    )
    segments = _collection_segments(retriever, session_id, natural)
    deterministic_query, deterministic_candidates = best_label_free_query(
        retriever,
        session_id,
        budget=2,
    )
    deterministic_selection = _top_selection(deterministic_candidates)
    arm_root = session_root / "arms"
    ledger_root = session_root / "ledgers"
    natural_triggered = natural.should_rethink
    arms: dict[str, dict[str, Any]] = {}

    learned_path = arm_root / "targeted_learned.json"
    learned_ledger = ledger_root / "targeted_learned.jsonl"
    arms["targeted_learned"] = _load_or_run_arm(
        path=learned_path,
        runner=lambda: _run_release_arm(
            row=row,
            task=task,
            initial=initial,
            models=models,
            counter=counter,
            segments=segments,
            ledger_path=learned_ledger,
            retries=request.max_contract_retries,
        ),
        row=row,
        arm_id="targeted_learned",
        natural_triggered=natural_triggered,
        policy_origin="model_query_and_model_selection",
        collection_protocol_version=collection_protocol_version,
    )

    learned_workflow = arms["targeted_learned"].get("workflow")
    if isinstance(learned_workflow, Mapping):
        learned_rounds = learned_workflow.get("rounds")
        if isinstance(learned_rounds, list) and learned_rounds:
            learned_query = EvidenceQuery.from_dict(
                learned_rounds[0]["evidence_query"]
            )
            learned_candidates = retriever.shortlist(learned_query)
            learned_top = _top_selection(learned_candidates)
            rank_ledger = ledger_root / "targeted_rank_top.jsonl"
            arms["targeted_rank_top"] = _load_or_run_arm(
                path=arm_root / "targeted_rank_top.json",
                runner=lambda: _run_release_arm(
                    row=row,
                    task=task,
                    initial=initial,
                    models=models,
                    counter=counter,
                    segments=segments,
                    ledger_path=rank_ledger,
                    retries=request.max_contract_retries,
                    query=learned_query,
                    selection=learned_top,
                ),
                row=row,
                arm_id="targeted_rank_top",
                natural_triggered=natural_triggered,
                policy_origin="model_query_and_retriever_rank_top",
                collection_protocol_version=collection_protocol_version,
            )
    if "targeted_rank_top" not in arms:
        arms["targeted_rank_top"] = _arm_record(
            row=row,
            arm_id="targeted_rank_top",
            natural_triggered=natural_triggered,
            policy_origin="model_query_and_retriever_rank_top",
            collection_protocol_version=collection_protocol_version,
            error=RuntimeError("targeted_learned did not yield a grounded query"),
        )
        _write_json(
            arm_root / "targeted_rank_top.json",
            arms["targeted_rank_top"],
        )

    deterministic_ledger = ledger_root / "deterministic.jsonl"
    arms["deterministic"] = _load_or_run_arm(
        path=arm_root / "deterministic.json",
        runner=lambda: _run_release_arm(
            row=row,
            task=task,
            initial=initial,
            models=models,
            counter=counter,
            segments=(deterministic_query.segment_id,),
            ledger_path=deterministic_ledger,
            retries=request.max_contract_retries,
            query=deterministic_query,
            selection=deterministic_selection,
        ),
        row=row,
        arm_id="deterministic",
        natural_triggered=natural_triggered,
        policy_origin="label_free_best_query_and_rank_top_selection",
        collection_protocol_version=collection_protocol_version,
    )

    random_selection = _stable_random_selection(
        deterministic_candidates,
        session_id=session_id,
        seed=request.random_seed,
    )
    random_ledger = ledger_root / "random_candidate.jsonl"
    arms["random_candidate"] = _load_or_run_arm(
        path=arm_root / "random_candidate.json",
        runner=lambda: _run_release_arm(
            row=row,
            task=task,
            initial=initial,
            models=models,
            counter=counter,
            segments=(deterministic_query.segment_id,),
            ledger_path=random_ledger,
            retries=request.max_contract_retries,
            query=deterministic_query,
            selection=random_selection,
        ),
        row=row,
        arm_id="random_candidate",
        natural_triggered=natural_triggered,
        policy_origin="label_free_best_query_and_stable_random_selection",
        collection_protocol_version=collection_protocol_version,
    )

    full_query = EvidenceQuery(
        segment_id=deterministic_query.segment_id,
        purpose=deterministic_query.purpose,
        target_slots=deterministic_query.target_slots,
        pattern=deterministic_query.pattern,
        budget=4,
    )
    full_candidates = retriever.shortlist(full_query)
    full_selection = _top_selection(full_candidates)
    full_ledger = ledger_root / "full_candidate_context.jsonl"
    arms["full_candidate_context"] = _load_or_run_arm(
        path=arm_root / "full_candidate_context.json",
        runner=lambda: _run_release_arm(
            row=row,
            task=task,
            initial=initial,
            models=models,
            counter=counter,
            segments=(full_query.segment_id,),
            ledger_path=full_ledger,
            retries=request.max_contract_retries,
            query=full_query,
            selection=full_selection,
        ),
        row=row,
        arm_id="full_candidate_context",
        natural_triggered=natural_triggered,
        policy_origin="higher_cost_label_free_information_upper_bound",
        collection_protocol_version=collection_protocol_version,
    )

    arms["reflection_no_new_evidence"] = _load_or_run_arm(
        path=arm_root / "reflection_no_new_evidence.json",
        runner=lambda: _reflection_arm(
            task=task,
            initial=initial,
            model=models.revision,
            segments=segments,
            retries=request.max_contract_retries,
        ),
        row=row,
        arm_id="reflection_no_new_evidence",
        natural_triggered=natural_triggered,
        policy_origin="same_revision_turn_without_atomic_release",
        collection_protocol_version=collection_protocol_version,
    )

    ordered = [arms[arm_id] for arm_id in ARM_ORDER]
    value = {
        "schema_version": "1.0.0",
        "collection_protocol_version": collection_protocol_version,
        "dataset": row["dataset"],
        "session_id": session_id,
        "outer_fold": row["outer_fold"],
        "status": (
            "complete"
            if all(arm["status"] == "ok" for arm in ordered)
            else "complete_with_fail_closed_arms"
        ),
        "initial": initial_value,
        "natural_rethink_decision": natural.to_dict(),
        "collection_segment_ids": list(segments),
        "arms": ordered,
        "same_trigger_direct_refer_generated_offline": True,
        "ground_truth_used": False,
    }
    _write_json(result_path, value)
    return value


def collect_fold(
    request: LoopCollectionRequest,
    *,
    model_loader: ModelLoader | None = None,
) -> dict[str, Any]:
    """Collect one outer fold, resuming only fully materialized session records."""

    plan, preparation_manifest = _validate_plan(request)
    manifest_path = request.output_dir / "manifest.json"
    raw_path = request.output_dir / "trajectories.raw.jsonl"
    if manifest_path.is_file():
        manifest = _read_json(manifest_path, "collection manifest")
        if (
            manifest.get("collection_protocol_version")
            != request.collection_protocol_version
            or manifest.get("files", {})
            .get("trajectories.raw.jsonl", {})
            .get("sha256")
            != _sha256(raw_path)
        ):
            raise ValueError("existing collection manifest failed verification")
        return manifest

    task = _task(request.task_path)
    reviewer = plan[0]["reviewer"]
    model_path = str(reviewer["model_path"])
    adapter_path = reviewer.get("adapter_path")
    loader = model_loader or _default_model_loader
    loaded = loader(
        model_path,
        None if adapter_path is None else str(adapter_path),
        request,
    )
    models = _stage_models(loaded)
    counter = _counter(models)
    results = [
        _collect_session(
            request=request,
            row=row,
            task=task,
            models=models,
            counter=counter,
        )
        for row in plan
    ]
    _write_jsonl(raw_path, results)
    status_counts = Counter(str(row["status"]) for row in results)
    arm_status_counts = Counter(
        str(arm["status"])
        for row in results
        for arm in row.get("arms", [])
    )
    manifest = {
        "schema_version": "1.0.0",
        "collection_protocol_version": request.collection_protocol_version,
        "preparation_protocol_version": PREPARATION_PROTOCOL_VERSION,
        "dataset": plan[0]["dataset"],
        "outer_fold": request.fold_index,
        "session_count": len(results),
        "limited_check": (
            request.max_sessions is not None or bool(request.session_ids)
        ),
        "session_status_counts": dict(sorted(status_counts.items())),
        "arm_status_counts": dict(sorted(arm_status_counts.items())),
        "arm_order": list(ARM_ORDER),
        "reviewer": dict(reviewer),
        "stage_routing": {
            "initial": "base_qwen_with_frozen_probability_anchor",
            "query": (
                "fold_local_literacy_adapter"
                if adapter_path is not None
                else "base_qwen_fallback"
            ),
            "selection": (
                "fold_local_literacy_adapter"
                if adapter_path is not None
                else "base_qwen_fallback"
            ),
            "revision": "base_qwen_pre_loop_training",
        },
        "files": {
            "trajectories.raw.jsonl": {
                "sha256": _sha256(raw_path),
                "row_count": len(results),
            }
        },
        "sources": {
            "preparation_manifest_sha256": _sha256(
                request.prepared_dir / "manifest.json"
            ),
            "inference_plan_sha256": preparation_manifest["files"][
                "inference_plan.jsonl"
            ]["sha256"],
            "task_sha256": _sha256(request.task_path),
        },
        "fit_boundaries": {
            "outcome_path_accepted_by_collector": False,
            "current_sample_targets_accessed": False,
            "frozen_reference_accessed": False,
            "utility_accessed": False,
            "dev_accessed": False,
            "test_accessed": False,
            "raw_trajectories_frozen_before_outcome_join": True,
        },
        "interpretation_boundary": (
            "This is label-free raw trajectory collection. It does not select "
            "an arm, compute WC/CW, or establish full-loop utility."
        ),
    }
    manifest["contract_policy"] = {
        "initial_grounding": (
            "machine_readable_allowlist_then_audited_drop_"
            "nonallowlisted_segment_values"
        ),
        "initial_projection_changes_probability": False,
        "initial_projection_substitutes_identifiers": False,
        "reflection": "dedicated_no_atomic_evidence_contract",
        "prior_artifacts_modified": False,
    }
    if request.session_ids:
        manifest["selected_session_ids"] = list(request.session_ids)
    _write_json(manifest_path, manifest)
    marker = (
        "SMOKE_COMPLETE"
        if request.max_sessions is not None
        else "PIPELINE_COMPLETE"
    )
    (request.output_dir / marker).touch()
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect label-free OOF full-loop trajectories."
    )
    parser.add_argument("--prepared-dir", required=True, type=Path)
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fold-index", required=True, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=640)
    parser.add_argument("--max-input-tokens", type=int, default=8_192)
    parser.add_argument("--max-contract-retries", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=20_260_726)
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument(
        "--session-id",
        action="append",
        default=[],
        help="Limit a mechanics check to a session in the selected outer fold.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = collect_fold(
        LoopCollectionRequest(
            prepared_dir=args.prepared_dir,
            task_path=args.task,
            output_dir=args.output_dir,
            fold_index=args.fold_index,
            device=args.device,
            dtype=args.dtype,
            attention_implementation=args.attention_implementation,
            max_new_tokens=args.max_new_tokens,
            max_input_tokens=args.max_input_tokens,
            max_contract_retries=args.max_contract_retries,
            random_seed=args.random_seed,
            max_sessions=args.max_sessions,
            session_ids=tuple(args.session_id),
        )
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
