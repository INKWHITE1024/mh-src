"""Budgeted, auditable evidence agent over compiled evidence stores with append-only access logging."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias, TypeVar

from rethink_mh.textualization.canonical_slots import CORE_SLOTS

from .contracts import InitialAssessment, RevisionAssessment, TaskSpec
from .native_query_retrieval import NativeAtomicRetriever
from .prompts import PromptBuilder
from .query_retrieval import (
    RETRIEVAL_PROTOCOL_VERSION,
    RETRIEVAL_PATTERNS,
    RETRIEVAL_PURPOSES,
    AtomicSelection,
    CandidateSet,
    EvidenceQuery,
)
from .retrieval import TargetedEvidence
from .retrieval_prompts import QueryRetrievalPromptBuilder
from .trigger import RethinkDecision, RethinkPolicy
from .workflow import (
    CompletionModel,
    EvidenceGroundingError,
    RethinkPolicyProtocol,
    WorkflowOutcome,
    _invoke_model,
    _parse_completion,
    _validate_initial_grounding,
    _validate_revision_grounding,
)


AGENT_PROTOCOL_VERSION = "evidence-agent"
INITIAL_GROUNDING_RECOVERY_POLICY_VERSION = "initial-grounding-recovery"
LEDGER_SCHEMA_VERSION = "1.0.0"
_LEDGER_KEYS = frozenset(
    {
        "schema_version",
        "agent_protocol_version",
        "sequence",
        "recorded_at_utc",
        "run_id",
        "session_fingerprint",
        "event_type",
        "payload",
        "previous_entry_sha256",
        "entry_sha256",
    }
)
_EVENT_TYPES = frozenset(
    {
        "agent_opened",
        "agent_resumed",
        "query_denied",
        "candidates_exposed",
        "release_denied",
        "evidence_released",
        "revision_recorded",
        "agent_aborted",
        "agent_closed",
    }
)
_LABEL_KEYS = frozenset(
    {
        "label",
        "label2",
        "label3",
        "target",
        "targetlabel",
        "diagnosis",
        "depressed",
        "depressionlabel",
        "phq8",
        "phq8score",
        "phq9",
        "phq9score",
        "groundtruth",
        "goldlabel",
        "samplelabel",
        "truelabel",
        "ytrue",
    }
)
_TRANSCRIPT_TIERS = ("full", "compact", "essential", "minimal")

EvidenceProtocolVersion = Literal["native"]
CompletionAdapter: TypeAlias = CompletionModel | Callable[
    [Sequence[Mapping[str, str]]],
    str | Mapping[str, Any],
]


class EvidenceAgentError(RuntimeError):
    """Base failure raised by the Evidence Agent module."""


class EvidenceBudgetExceeded(EvidenceAgentError):
    """A query or release would cross a cumulative agent budget."""


class EvidenceLedgerError(EvidenceAgentError):
    """The append-only access ledger is missing, malformed, or tampered."""


class EvidenceAgentStateError(EvidenceAgentError):
    """The caller attempted an operation in an invalid lifecycle state."""


class EvidenceTokenCounter(Protocol):
    """Adapter at the token-accounting seam."""

    counter_id: str
    exact: bool

    def count(self, text: str) -> int:
        """Return the number of model tokens in ``text``."""


@dataclass(frozen=True, slots=True)
class EvidenceAgentStageModels:
    """Stage-specific adapters behind the Evidence Agent model seam.

    A uniform model remains the default.  The explicit routes are needed by
    strict-OOF experiments where the frozen classifier supplies the initial
    assessment, a literacy adapter supplies query/selection mechanics, and a
    separately aligned adapter supplies the revision.
    """

    initial: CompletionAdapter
    query: CompletionAdapter
    selection: CompletionAdapter
    revision: CompletionAdapter

    def __post_init__(self) -> None:
        for name in ("initial", "query", "selection", "revision"):
            value = getattr(self, name)
            if not callable(value) and not callable(getattr(value, "complete", None)):
                raise TypeError(
                    f"{name} stage model must be callable or expose complete(messages)"
                )

    @classmethod
    def uniform(cls, model: CompletionAdapter) -> "EvidenceAgentStageModels":
        return cls(
            initial=model,
            query=model,
            selection=model,
            revision=model,
        )


@dataclass(frozen=True, slots=True)
class CallableTokenCounter:
    """Token-counter Adapter for tests or an already loaded tokenizer."""

    counter_id: str
    function: Callable[[str], int]
    exact: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.counter_id, str) or not self.counter_id.strip():
            raise ValueError("counter_id must be non-empty text")
        if not callable(self.function):
            raise TypeError("function must be callable")

    def count(self, text: str) -> int:
        value = self.function(text)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("token counter must return a non-negative integer")
        return value


@dataclass(frozen=True, slots=True)
class TokenizerTokenCounter:
    """Exact Adapter for Hugging Face-style tokenizers."""

    tokenizer: Any
    counter_id: str
    exact: bool = True

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> "TokenizerTokenCounter":
        if tokenizer is None or not callable(getattr(tokenizer, "encode", None)):
            raise TypeError("tokenizer must expose encode(text, ...)")
        name = getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)
        digest = hashlib.sha256(str(name).encode("utf-8")).hexdigest()[:12]
        return cls(
            tokenizer=tokenizer,
            counter_id=f"huggingface:{tokenizer.__class__.__name__}:{digest}",
        )

    def count(self, text: str) -> int:
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        return len(encoded)


@dataclass(frozen=True, slots=True)
class EvidenceAgentConfig:
    """Cumulative information budgets for one participant review."""

    max_queries: int = 2
    max_atomic_records: int = 6
    max_released_tokens: int = 2_400
    max_first_pass_tokens: int = 5_000
    max_candidates_per_query: int = 8
    require_exact_token_counter: bool = True

    def __post_init__(self) -> None:
        limits = {
            "max_queries": (self.max_queries, 1, 8),
            "max_atomic_records": (self.max_atomic_records, 1, 32),
            "max_released_tokens": (self.max_released_tokens, 32, 32_768),
            "max_first_pass_tokens": (
                self.max_first_pass_tokens,
                128,
                32_768,
            ),
            "max_candidates_per_query": (
                self.max_candidates_per_query,
                4,
                16,
            ),
        }
        for name, (value, lower, upper) in limits.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not lower <= value <= upper:
                raise ValueError(
                    f"{name} must be between {lower} and {upper}, inclusive"
                )
        if not isinstance(self.require_exact_token_counter, bool):
            raise TypeError("require_exact_token_counter must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_queries": self.max_queries,
            "max_atomic_records": self.max_atomic_records,
            "max_released_tokens": self.max_released_tokens,
            "max_first_pass_tokens": self.max_first_pass_tokens,
            "max_candidates_per_query": self.max_candidates_per_query,
            "require_exact_token_counter": self.require_exact_token_counter,
        }


@dataclass(frozen=True, slots=True)
class EvidenceBudgetSnapshot:
    """Consumed and remaining information budget at one ledger position."""

    queries_used: int
    atomic_records_released: int
    released_tokens: int
    queries_remaining: int
    atomic_records_remaining: int
    released_tokens_remaining: int

    def to_dict(self) -> dict[str, int]:
        return {
            "queries_used": self.queries_used,
            "atomic_records_released": self.atomic_records_released,
            "released_tokens": self.released_tokens,
            "queries_remaining": self.queries_remaining,
            "atomic_records_remaining": self.atomic_records_remaining,
            "released_tokens_remaining": self.released_tokens_remaining,
        }


@dataclass(frozen=True, slots=True)
class EvidenceLedgerVerification:
    """Result of replaying and verifying an access-ledger hash chain."""

    entry_count: int
    run_id: str
    session_fingerprint: str
    final_entry_sha256: str
    sealed: bool
    aborted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_count": self.entry_count,
            "run_id": self.run_id,
            "session_fingerprint": self.session_fingerprint,
            "final_entry_sha256": self.final_entry_sha256,
            "sealed": self.sealed,
            "aborted": self.aborted,
        }


@dataclass(frozen=True, slots=True)
class EvidenceRelease:
    """One grounded disclosure and its cumulative budget transition."""

    candidate_set_sha256: str
    targeted_evidence: TargetedEvidence
    released_tokens: int
    budget_before: EvidenceBudgetSnapshot
    budget_after: EvidenceBudgetSnapshot

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        output: dict[str, Any] = {
            "candidate_set_sha256": self.candidate_set_sha256,
            "selected_segment_ids": list(
                self.targeted_evidence.selected_segment_ids
            ),
            "selected_atomic_evidence_ids": list(
                self.targeted_evidence.selected_atomic_evidence_ids
            ),
            "canonical_evidence_ids": list(
                self.targeted_evidence.canonical_evidence_ids
            ),
            "released_tokens": self.released_tokens,
            "budget_before": self.budget_before.to_dict(),
            "budget_after": self.budget_after.to_dict(),
        }
        if include_text:
            output["text"] = self.targeted_evidence.text
        return output


@dataclass(frozen=True, slots=True)
class EvidenceAgentRound:
    """Audit record for one model-grounded query and release."""

    evidence_query: EvidenceQuery
    candidate_set: CandidateSet
    atomic_selection: AtomicSelection
    release: EvidenceRelease
    query_syntax_normalizations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_query": self.evidence_query.to_dict(),
            "candidate_set": self.candidate_set.to_dict(),
            "atomic_selection": self.atomic_selection.to_dict(),
            "release": self.release.to_dict(include_text=False),
            "query_syntax_normalizations": list(
                self.query_syntax_normalizations
            ),
        }


@dataclass(frozen=True, slots=True)
class EvidenceAgentWorkflowResult:
    """Final state and complete non-sensitive provenance for one agent run."""

    evidence_protocol_version: EvidenceProtocolVersion
    task: TaskSpec
    outcome: WorkflowOutcome
    initial_assessment: InitialAssessment
    decision: RethinkDecision
    rounds: tuple[EvidenceAgentRound, ...]
    revision_assessment: RevisionAssessment | None
    final_risk_probability: float
    final_confidence: float
    first_pass_tokens: int
    transcript_tier: str | None
    budget: EvidenceBudgetSnapshot
    ledger: EvidenceLedgerVerification
    model_call_count: int
    contract_repair_count: int
    initial_grounding_audit: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        output = {
            "schema_version": "1.0.0",
            "agent_protocol_version": AGENT_PROTOCOL_VERSION,
            "evidence_protocol_version": self.evidence_protocol_version,
            "retrieval_protocol_version": RETRIEVAL_PROTOCOL_VERSION,
            "task": self.task.to_dict(),
            "outcome": self.outcome,
            "initial_assessment": self.initial_assessment.to_dict(),
            "rethink_decision": self.decision.to_dict(),
            "rounds": [item.to_dict() for item in self.rounds],
            "revision_assessment": (
                self.revision_assessment.to_dict()
                if self.revision_assessment is not None
                else None
            ),
            "final_risk_probability": self.final_risk_probability,
            "final_confidence": self.final_confidence,
            "first_pass_tokens": self.first_pass_tokens,
            "transcript_tier": self.transcript_tier,
            "budget": self.budget.to_dict(),
            "access_ledger": self.ledger.to_dict(),
            "model_call_count": self.model_call_count,
            "contract_repair_count": self.contract_repair_count,
            "query_syntax_normalization_count": sum(
                len(item.query_syntax_normalizations)
                for item in self.rounds
            ),
            "selector_mode": "model_grounded_query_and_atomic_selection",
            "ground_truth_used_by_workflow": False,
            "learned_selector_weights_validated": False,
        }
        if self.initial_grounding_audit is not None:
            output["initial_grounding_audit"] = dict(
                self.initial_grounding_audit
            )
        return output


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _reject_label_keys(value: Any, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise EvidenceLedgerError(f"{path} keys must be strings")
            if _normalized_key(key) in _LABEL_KEYS:
                raise EvidenceLedgerError(
                    f"{path}.{key} is a forbidden current-sample label field"
                )
            _reject_label_keys(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _reject_label_keys(child, f"{path}[{index}]")


def _json_safe(value: Any, path: str = "payload") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvidenceLedgerError(f"{path} cannot contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(child, f"{path}.{key}")
            for key, child in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _json_safe(child, f"{path}[{index}]")
            for index, child in enumerate(value)
        ]
    raise EvidenceLedgerError(f"{path} contains a non-JSON value")


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _entry_hash(value_without_hash: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        _canonical_json(value_without_hash).encode("utf-8")
    ).hexdigest()


def _load_ledger_entries(text: str, path: Path) -> tuple[dict[str, Any], ...]:
    output: list[dict[str, Any]] = []
    previous: str | None = None
    run_id: str | None = None
    session_fingerprint: str | None = None
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise EvidenceLedgerError(
                f"access ledger contains a blank record at {path}:{line_number}"
            )
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvidenceLedgerError(
                f"invalid access ledger JSON at {path}:{line_number}: {error.msg}"
            ) from error
        if not isinstance(value, dict) or set(value) != _LEDGER_KEYS:
            raise EvidenceLedgerError(
                f"access ledger record {line_number} has invalid fields"
            )
        if value["schema_version"] != LEDGER_SCHEMA_VERSION:
            raise EvidenceLedgerError("access ledger schema version mismatch")
        if value["agent_protocol_version"] != AGENT_PROTOCOL_VERSION:
            raise EvidenceLedgerError("access ledger agent protocol mismatch")
        if value["sequence"] != line_number - 1:
            raise EvidenceLedgerError(
                f"access ledger sequence is broken at record {line_number}"
            )
        if value["event_type"] not in _EVENT_TYPES:
            raise EvidenceLedgerError(
                f"unknown access ledger event: {value['event_type']!r}"
            )
        if value["previous_entry_sha256"] != previous:
            raise EvidenceLedgerError(
                f"access ledger hash link is broken at record {line_number}"
            )
        without_hash = {
            key: child
            for key, child in value.items()
            if key != "entry_sha256"
        }
        expected = _entry_hash(without_hash)
        if value["entry_sha256"] != expected:
            raise EvidenceLedgerError(
                f"access ledger entry hash mismatch at record {line_number}"
            )
        if run_id is None:
            run_id = str(value["run_id"])
            session_fingerprint = str(value["session_fingerprint"])
        elif (
            value["run_id"] != run_id
            or value["session_fingerprint"] != session_fingerprint
        ):
            raise EvidenceLedgerError(
                "access ledger mixes runs or evidence sessions"
            )
        payload = value["payload"]
        if not isinstance(payload, Mapping):
            raise EvidenceLedgerError(
                f"access ledger payload {line_number} must be an object"
            )
        _reject_label_keys(payload, f"ledger[{line_number - 1}].payload")
        previous = expected
        output.append(value)
    if output and output[0]["event_type"] != "agent_opened":
        raise EvidenceLedgerError("access ledger must begin with agent_opened")
    events = [str(item["event_type"]) for item in output]
    if events.count("agent_opened") > 1:
        raise EvidenceLedgerError("access ledger contains multiple open events")
    if sum(item["event_type"] == "agent_closed" for item in output) > 1:
        raise EvidenceLedgerError("access ledger contains multiple close events")
    if any(
        item["event_type"] == "agent_closed"
        for item in output[:-1]
    ):
        raise EvidenceLedgerError("access ledger contains records after closure")
    if events.count("agent_aborted") > 1:
        raise EvidenceLedgerError("access ledger contains multiple abort events")
    if "agent_aborted" in events:
        abort_index = events.index("agent_aborted")
        if events[abort_index + 1 :] not in ([], ["agent_closed"]):
            raise EvidenceLedgerError(
                "only agent_closed may follow an abort event"
            )
    if events.count("revision_recorded") > 1:
        raise EvidenceLedgerError(
            "access ledger contains multiple revision events"
        )
    if "revision_recorded" in events:
        revision_index = events.index("revision_recorded")
        if "evidence_released" not in events[:revision_index]:
            raise EvidenceLedgerError(
                "access ledger records revision before evidence release"
            )
        invalid_after_revision = {
            "candidates_exposed",
            "evidence_released",
        } & set(events[revision_index + 1 :])
        if invalid_after_revision:
            raise EvidenceLedgerError(
                "access ledger performs evidence access after revision"
            )
    if events and events[-1] == "agent_closed":
        outcome = output[-1]["payload"].get("outcome")
        if "agent_aborted" in events:
            if outcome != "aborted":
                raise EvidenceLedgerError(
                    "an aborted access ledger has an invalid close outcome"
                )
        elif outcome == "accepted_initial":
            if {
                "candidates_exposed",
                "evidence_released",
                "revision_recorded",
            } & set(events):
                raise EvidenceLedgerError(
                    "accepted_initial cannot follow evidence access"
                )
        elif outcome in {
            "accepted_after_preservation",
            "accepted_after_revision",
            "referred_unresolved",
        }:
            if "revision_recorded" not in events:
                raise EvidenceLedgerError(
                    "post-review close outcome requires a grounded revision"
                )
        else:
            raise EvidenceLedgerError(
                "access ledger has an invalid close outcome"
            )
    return tuple(output)


def verify_evidence_access_ledger(
    path: str | Path,
) -> EvidenceLedgerVerification:
    ledger_path = Path(path).expanduser().resolve()
    if not ledger_path.is_file():
        raise FileNotFoundError(f"access ledger does not exist: {ledger_path}")
    entries = _load_ledger_entries(
        ledger_path.read_text(encoding="utf-8"),
        ledger_path,
    )
    if not entries:
        raise EvidenceLedgerError("access ledger is empty")
    return EvidenceLedgerVerification(
        entry_count=len(entries),
        run_id=str(entries[0]["run_id"]),
        session_fingerprint=str(entries[0]["session_fingerprint"]),
        final_entry_sha256=str(entries[-1]["entry_sha256"]),
        sealed=entries[-1]["event_type"] == "agent_closed",
        aborted=any(item["event_type"] == "agent_aborted" for item in entries),
    )


class _EvidenceAccessLedger:
    """Append-only JSONL ledger with a per-record SHA-256 chain."""

    def __init__(
        self,
        *,
        path: Path,
        run_id: str,
        session_fingerprint: str,
        clock: Callable[[], datetime],
    ) -> None:
        self.path = path
        self.run_id = run_id
        self.session_fingerprint = session_fingerprint
        self._clock = clock

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        run_id: str,
        session_fingerprint: str,
        resume: bool,
        clock: Callable[[], datetime] | None = None,
    ) -> tuple["_EvidenceAccessLedger", tuple[dict[str, Any], ...]]:
        ledger_path = Path(path).expanduser().resolve()
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        instance = cls(
            path=ledger_path,
            run_id=run_id,
            session_fingerprint=session_fingerprint,
            clock=clock or (lambda: datetime.now(timezone.utc)),
        )
        if ledger_path.exists():
            entries = _load_ledger_entries(
                ledger_path.read_text(encoding="utf-8"),
                ledger_path,
            )
            if not resume:
                raise FileExistsError(
                    f"access ledger already exists: {ledger_path}"
                )
            if not entries:
                raise EvidenceLedgerError(
                    "cannot resume an empty access ledger"
                )
            if (
                entries[0]["run_id"] != run_id
                or entries[0]["session_fingerprint"] != session_fingerprint
            ):
                raise EvidenceLedgerError(
                    "resume run/session does not match the access ledger"
                )
            if entries[-1]["event_type"] == "agent_closed":
                raise EvidenceAgentStateError(
                    "cannot resume a sealed evidence-agent run"
                )
            if any(
                item["event_type"] == "agent_aborted"
                for item in entries
            ):
                raise EvidenceAgentStateError(
                    "cannot resume an aborted evidence-agent run"
                )
            return instance, entries
        if resume:
            raise FileNotFoundError(
                f"cannot resume missing access ledger: {ledger_path}"
            )
        ledger_path.touch(mode=0o600)
        return instance, ()

    def append(
        self,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        if event_type not in _EVENT_TYPES:
            raise ValueError(f"unsupported ledger event: {event_type}")
        safe_payload = _json_safe(payload)
        _reject_label_keys(safe_payload)
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.seek(0)
                entries = _load_ledger_entries(handle.read(), self.path)
                if entries and entries[-1]["event_type"] == "agent_closed":
                    raise EvidenceAgentStateError(
                        "cannot append to a sealed access ledger"
                    )
                if entries and (
                    entries[0]["run_id"] != self.run_id
                    or entries[0]["session_fingerprint"]
                    != self.session_fingerprint
                ):
                    raise EvidenceLedgerError(
                        "access ledger identity changed while appending"
                    )
                previous = (
                    str(entries[-1]["entry_sha256"]) if entries else None
                )
                recorded_at = self._clock()
                if recorded_at.tzinfo is None:
                    raise EvidenceLedgerError(
                        "access-ledger clock must return a timezone-aware datetime"
                    )
                without_hash = {
                    "schema_version": LEDGER_SCHEMA_VERSION,
                    "agent_protocol_version": AGENT_PROTOCOL_VERSION,
                    "sequence": len(entries),
                    "recorded_at_utc": recorded_at.astimezone(
                        timezone.utc
                    ).isoformat(),
                    "run_id": self.run_id,
                    "session_fingerprint": self.session_fingerprint,
                    "event_type": event_type,
                    "payload": safe_payload,
                    "previous_entry_sha256": previous,
                }
                record = {
                    **without_hash,
                    "entry_sha256": _entry_hash(without_hash),
                }
                handle.seek(0, os.SEEK_END)
                handle.write(_canonical_json(record) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                return record
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def entries(self) -> tuple[dict[str, Any], ...]:
        return _load_ledger_entries(
            self.path.read_text(encoding="utf-8"),
            self.path,
        )


class _Catalog(Protocol):
    evidence_protocol_version: EvidenceProtocolVersion
    available_segment_ids: tuple[str, ...]
    session_fingerprint: str
    session_input_text: str

    def render_retrieval_map(
        self,
        segment_ids: Sequence[str] | None = None,
    ) -> str: ...

    def shortlist(
        self,
        query: EvidenceQuery | Mapping[str, Any],
        *,
        exclude_atomic_ids: Sequence[str] = (),
    ) -> CandidateSet: ...

    def materialize(
        self,
        candidate_set: CandidateSet,
        selection: AtomicSelection | Mapping[str, Any],
    ) -> TargetedEvidence: ...


class _NativeCatalogAdapter:
    evidence_protocol_version: EvidenceProtocolVersion = "native"

    def __init__(
        self,
        session_dir: Path,
        *,
        max_candidates_per_query: int,
    ) -> None:
        self._retriever = NativeAtomicRetriever.from_session_dir(
            session_dir,
            max_candidates_per_query=max_candidates_per_query,
        )

    @property
    def available_segment_ids(self) -> tuple[str, ...]:
        return self._retriever.available_segment_ids

    @property
    def session_fingerprint(self) -> str:
        return self._retriever.session_fingerprint

    @property
    def session_input_text(self) -> str:
        return self._retriever.session_input_text

    def render_retrieval_map(
        self,
        segment_ids: Sequence[str] | None = None,
    ) -> str:
        return self._retriever.render_retrieval_map(segment_ids)

    def shortlist(
        self,
        query: EvidenceQuery | Mapping[str, Any],
        *,
        exclude_atomic_ids: Sequence[str] = (),
    ) -> CandidateSet:
        return self._retriever.shortlist(
            query,
            exclude_atomic_ids=exclude_atomic_ids,
        )

    def materialize(
        self,
        candidate_set: CandidateSet,
        selection: AtomicSelection | Mapping[str, Any],
    ) -> TargetedEvidence:
        return self._retriever.materialize(candidate_set, selection)


def _open_catalog(
    session_dir: str | Path,
    *,
    max_candidates_per_query: int,
    evidence_protocol: str,
) -> _Catalog:
    directory = Path(session_dir).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(
            f"evidence session directory does not exist: {directory}"
        )
    if evidence_protocol not in {"auto", "native"}:
        raise ValueError("evidence_protocol must be auto or native")
    if not (directory / "session_input.native.txt").is_file():
        raise FileNotFoundError(
            "Not a complete native evidence session; "
            "missing: session_input.native.txt"
        )
    return _NativeCatalogAdapter(
        directory,
        max_candidates_per_query=max_candidates_per_query,
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_transcript_tiers(
    transcript_dir: str | Path,
) -> tuple[dict[str, str], str]:
    directory = Path(transcript_dir).expanduser().resolve()
    manifest_path = directory / "transcript_manifest.compact.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"transcript manifest does not exist: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid transcript manifest: {manifest_path}: {error.msg}"
        ) from error
    if not isinstance(manifest, Mapping):
        raise ValueError("transcript manifest must contain an object")
    if manifest.get("protocol_version") != "compact":
        raise ValueError("transcript protocol_version must be compact")
    if manifest.get("label_fields_read") != []:
        raise ValueError("transcript manifest crossed the label boundary")
    if manifest.get("raw_session_identifier_in_model_input") is not False:
        raise ValueError("transcript manifest exposes the raw session identifier")
    if manifest.get("speaker_policy") not in {
        "participant_only",
        "all_speakers_unassigned",
    }:
        raise ValueError("transcript manifest has an unsupported speaker policy")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("transcript manifest lacks files")
    output: dict[str, str] = {}
    for tier in _TRANSCRIPT_TIERS:
        filename = f"transcript_input.{tier}.compact.txt"
        path = directory / filename
        record = files.get(filename)
        if not path.is_file() or not isinstance(record, Mapping):
            raise FileNotFoundError(
                f"transcript tier is missing or unmanifested: {filename}"
            )
        text = path.read_text(encoding="utf-8")
        if _sha256_text(text) != record.get("sha256"):
            raise ValueError(f"transcript hash mismatch: {filename}")
        output[tier] = text
    return output, _sha256_text(
        manifest_path.read_text(encoding="utf-8")
    )


def _compose_first_pass(
    measurement_text: str,
    transcript_dir: str | Path | None,
    *,
    token_counter: EvidenceTokenCounter,
    maximum_tokens: int,
) -> tuple[str, int, str | None, str | None]:
    if transcript_dir is None:
        count = token_counter.count(measurement_text)
        if count > maximum_tokens:
            raise EvidenceBudgetExceeded(
                f"measurement first pass has {count} tokens; "
                f"limit is {maximum_tokens}"
            )
        return measurement_text, count, None, None
    tiers, manifest_sha256 = _load_transcript_tiers(transcript_dir)
    observed: list[str] = []
    for tier in _TRANSCRIPT_TIERS:
        combined = (
            "MULTIMODAL MEASUREMENT EVIDENCE\n"
            f"{measurement_text.rstrip()}\n\n"
            f"{tiers[tier].rstrip()}\n"
        )
        count = token_counter.count(combined)
        observed.append(f"{tier}={count}")
        if count <= maximum_tokens:
            return combined, count, tier, manifest_sha256
    raise EvidenceBudgetExceeded(
        "no transcript tier fits the first-pass token budget "
        f"({', '.join(observed)}; limit={maximum_tokens})"
    )


def _candidate_set_sha256(candidate_set: CandidateSet) -> str:
    return hashlib.sha256(
        _canonical_json(candidate_set.to_dict()).encode("utf-8")
    ).hexdigest()


def _acquire_lease(path: Path) -> Any:
    lease_path = path.with_suffix(path.suffix + ".run.lock")
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lease_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise EvidenceAgentStateError(
            f"another process owns the evidence-agent ledger: {path}"
        ) from error
    return handle


class EvidenceAgent:
    """Deep module for controlled evidence access and immutable audit."""

    def __init__(
        self,
        *,
        catalog: _Catalog,
        config: EvidenceAgentConfig,
        token_counter: EvidenceTokenCounter,
        ledger: _EvidenceAccessLedger,
        lease: Any,
        first_pass_text: str,
        first_pass_tokens: int,
        transcript_tier: str | None,
        prior_entries: Sequence[Mapping[str, Any]],
    ) -> None:
        self._catalog = catalog
        self.config = config
        self.token_counter = token_counter
        self._ledger = ledger
        self._lease = lease
        self.first_pass_text = first_pass_text
        self.first_pass_tokens = first_pass_tokens
        self.transcript_tier = transcript_tier
        self._issued: dict[str, CandidateSet] = {}
        self._used_candidate_sets: set[str] = set()
        self._released: list[TargetedEvidence] = []
        self._queries_used = 0
        self._atomic_records_released = 0
        self._released_tokens = 0
        self._seen_atomic_ids: set[str] = set()
        self._revision_recorded = False
        self._closed = False
        self._restore(prior_entries)

    @classmethod
    def open(
        cls,
        session_dir: str | Path,
        *,
        ledger_path: str | Path,
        token_counter: EvidenceTokenCounter,
        config: EvidenceAgentConfig | None = None,
        transcript_dir: str | Path | None = None,
        evidence_protocol: str = "auto",
        run_id: str | None = None,
        resume: bool = False,
        clock: Callable[[], datetime] | None = None,
    ) -> "EvidenceAgent":
        settings = config or EvidenceAgentConfig()
        if (
            settings.require_exact_token_counter
            and not token_counter.exact
        ):
            raise ValueError(
                "formal evidence-agent runs require an exact tokenizer counter"
            )
        catalog = _open_catalog(
            session_dir,
            max_candidates_per_query=settings.max_candidates_per_query,
            evidence_protocol=evidence_protocol,
        )
        first_pass, first_tokens, transcript_tier, transcript_manifest = (
            _compose_first_pass(
                catalog.session_input_text,
                transcript_dir,
                token_counter=token_counter,
                maximum_tokens=settings.max_first_pass_tokens,
            )
        )
        identity = run_id or uuid.uuid4().hex
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{7,127}", identity) is None:
            raise ValueError(
                "run_id must be 8 to 128 safe identifier characters"
            )
        ledger_path_resolved = Path(ledger_path).expanduser().resolve()
        lease = _acquire_lease(ledger_path_resolved)
        try:
            ledger, prior = _EvidenceAccessLedger.open(
                ledger_path_resolved,
                run_id=identity,
                session_fingerprint=catalog.session_fingerprint,
                resume=resume,
                clock=clock,
            )
            if resume:
                opening = prior[0]["payload"]
                expected_opening = {
                    "evidence_protocol_version": (
                        catalog.evidence_protocol_version
                    ),
                    "retrieval_protocol_version": RETRIEVAL_PROTOCOL_VERSION,
                    "configuration": settings.to_dict(),
                    "token_counter_id": token_counter.counter_id,
                    "token_counter_exact": token_counter.exact,
                    "first_pass_tokens": first_tokens,
                    "first_pass_sha256": _sha256_text(first_pass),
                    "transcript_tier": transcript_tier,
                    "transcript_manifest_sha256": transcript_manifest,
                    "raw_participant_identifier_logged": False,
                    "ground_truth_read": False,
                }
                if opening != expected_opening:
                    raise EvidenceLedgerError(
                        "resume configuration, tokenizer, or first-pass input "
                        "does not match the original run"
                    )
                ledger.append(
                    "agent_resumed",
                    {
                        "evidence_protocol_version": (
                            catalog.evidence_protocol_version
                        ),
                        "token_counter_id": token_counter.counter_id,
                    },
                )
                prior = ledger.entries()
            else:
                ledger.append(
                    "agent_opened",
                    {
                        "evidence_protocol_version": (
                            catalog.evidence_protocol_version
                        ),
                        "retrieval_protocol_version": (
                            RETRIEVAL_PROTOCOL_VERSION
                        ),
                        "configuration": settings.to_dict(),
                        "token_counter_id": token_counter.counter_id,
                        "token_counter_exact": token_counter.exact,
                        "first_pass_tokens": first_tokens,
                        "first_pass_sha256": _sha256_text(first_pass),
                        "transcript_tier": transcript_tier,
                        "transcript_manifest_sha256": transcript_manifest,
                        "raw_participant_identifier_logged": False,
                        "ground_truth_read": False,
                    },
                )
                prior = ledger.entries()
            return cls(
                catalog=catalog,
                config=settings,
                token_counter=token_counter,
                ledger=ledger,
                lease=lease,
                first_pass_text=first_pass,
                first_pass_tokens=first_tokens,
                transcript_tier=transcript_tier,
                prior_entries=prior,
            )
        except Exception:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
            lease.close()
            raise

    @property
    def evidence_protocol_version(self) -> EvidenceProtocolVersion:
        return self._catalog.evidence_protocol_version

    @property
    def available_segment_ids(self) -> tuple[str, ...]:
        return self._catalog.available_segment_ids

    @property
    def session_fingerprint(self) -> str:
        return self._catalog.session_fingerprint

    @property
    def budget(self) -> EvidenceBudgetSnapshot:
        return EvidenceBudgetSnapshot(
            queries_used=self._queries_used,
            atomic_records_released=self._atomic_records_released,
            released_tokens=self._released_tokens,
            queries_remaining=max(
                0, self.config.max_queries - self._queries_used
            ),
            atomic_records_remaining=max(
                0,
                self.config.max_atomic_records
                - self._atomic_records_released,
            ),
            released_tokens_remaining=max(
                0,
                self.config.max_released_tokens - self._released_tokens,
            ),
        )

    def render_retrieval_map(
        self,
        segment_ids: Sequence[str] | None = None,
    ) -> str:
        self._ensure_open()
        return self._catalog.render_retrieval_map(segment_ids)

    def issue(
        self,
        query: EvidenceQuery | Mapping[str, Any],
    ) -> CandidateSet:
        """Expose one grounded candidate set and charge one query."""

        self._ensure_open()
        if self._revision_recorded:
            raise EvidenceAgentStateError(
                "cannot issue evidence queries after revision"
            )
        parsed = (
            query
            if isinstance(query, EvidenceQuery)
            else EvidenceQuery.from_dict(query)
        )
        before = self.budget
        if before.queries_remaining <= 0:
            self._deny(
                "query_denied",
                "query_budget_exhausted",
                parsed,
            )
            raise EvidenceBudgetExceeded("evidence query budget is exhausted")
        if parsed.budget > before.atomic_records_remaining:
            self._deny(
                "query_denied",
                "atomic_budget_insufficient",
                parsed,
            )
            raise EvidenceBudgetExceeded(
                "query selection budget exceeds remaining atomic-record budget"
            )
        try:
            candidate_set = self._catalog.shortlist(
                parsed,
                exclude_atomic_ids=tuple(sorted(self._seen_atomic_ids)),
            )
        except Exception as error:
            self._ledger.append(
                "query_denied",
                {
                    "reason_code": "query_not_materializable",
                    "query": parsed.to_dict(),
                    "error_type": error.__class__.__name__,
                    "budget": before.to_dict(),
                },
            )
            raise
        fingerprint = _candidate_set_sha256(candidate_set)
        self._ledger.append(
            "candidates_exposed",
            {
                "query": parsed.to_dict(),
                "candidate_set_sha256": fingerprint,
                "candidate_evidence_ids": list(
                    candidate_set.candidate_evidence_ids
                ),
                "candidate_count": len(candidate_set.candidates),
                "ranking_scores_exposed_to_model": False,
                "numeric_measurements_exposed_to_model": False,
                "budget_before": before.to_dict(),
            },
        )
        self._queries_used += 1
        self._issued[fingerprint] = candidate_set
        return candidate_set

    def release(
        self,
        candidate_set: CandidateSet,
        selection: AtomicSelection | Mapping[str, Any],
    ) -> EvidenceRelease:
        """Release full values after grounding and cumulative budget checks."""

        self._ensure_open()
        if self._revision_recorded:
            raise EvidenceAgentStateError(
                "cannot release evidence after revision"
            )
        if not isinstance(candidate_set, CandidateSet):
            raise TypeError("candidate_set must be a CandidateSet")
        fingerprint = _candidate_set_sha256(candidate_set)
        issued = self._issued.get(fingerprint)
        if issued is None or issued != candidate_set:
            self._ledger.append(
                "release_denied",
                {
                    "reason_code": "candidate_set_not_issued_by_agent",
                    "candidate_set_sha256": fingerprint,
                },
            )
            raise EvidenceAgentStateError(
                "candidate set was not issued by this EvidenceAgent run"
            )
        if fingerprint in self._used_candidate_sets:
            self._ledger.append(
                "release_denied",
                {
                    "reason_code": "candidate_set_already_consumed",
                    "candidate_set_sha256": fingerprint,
                },
            )
            raise EvidenceAgentStateError(
                "candidate set has already been consumed"
            )
        try:
            selected = (
                selection
                if isinstance(selection, AtomicSelection)
                else AtomicSelection.from_dict(selection)
            )
            targeted = self._catalog.materialize(candidate_set, selected)
        except Exception as error:
            self._ledger.append(
                "release_denied",
                {
                    "reason_code": "invalid_or_ungrounded_selection",
                    "candidate_set_sha256": fingerprint,
                    "error_type": error.__class__.__name__,
                },
            )
            raise
        overlap = set(targeted.selected_atomic_evidence_ids) & self._seen_atomic_ids
        if overlap:
            self._ledger.append(
                "release_denied",
                {
                    "reason_code": "atomic_evidence_already_released",
                    "candidate_set_sha256": fingerprint,
                    "duplicate_evidence_ids": sorted(overlap),
                },
            )
            raise EvidenceAgentStateError(
                f"atomic evidence was already released: {sorted(overlap)}"
            )
        released_tokens = self.token_counter.count(targeted.text)
        before = self.budget
        record_count = len(targeted.selected_atomic_evidence_ids)
        if record_count > before.atomic_records_remaining:
            self._ledger.append(
                "release_denied",
                {
                    "reason_code": "atomic_budget_exceeded",
                    "candidate_set_sha256": fingerprint,
                    "requested_record_count": record_count,
                    "budget_before": before.to_dict(),
                },
            )
            raise EvidenceBudgetExceeded(
                "atomic evidence release exceeds the cumulative record budget"
            )
        if released_tokens > before.released_tokens_remaining:
            self._ledger.append(
                "release_denied",
                {
                    "reason_code": "token_budget_exceeded",
                    "candidate_set_sha256": fingerprint,
                    "requested_tokens": released_tokens,
                    "budget_before": before.to_dict(),
                },
            )
            raise EvidenceBudgetExceeded(
                "atomic evidence release exceeds the cumulative token budget"
            )
        self._atomic_records_released += record_count
        self._released_tokens += released_tokens
        self._seen_atomic_ids.update(targeted.selected_atomic_evidence_ids)
        self._used_candidate_sets.add(fingerprint)
        self._released.append(targeted)
        after = self.budget
        self._ledger.append(
            "evidence_released",
            {
                "candidate_set_sha256": fingerprint,
                "selected_segment_ids": list(targeted.selected_segment_ids),
                "selected_atomic_evidence_ids": list(
                    targeted.selected_atomic_evidence_ids
                ),
                "canonical_evidence_count": len(
                    targeted.canonical_evidence_ids
                ),
                "released_tokens": released_tokens,
                "released_text_sha256": _sha256_text(targeted.text),
                "budget_before": before.to_dict(),
                "budget_after": after.to_dict(),
                "ground_truth_read": False,
            },
        )
        return EvidenceRelease(
            candidate_set_sha256=fingerprint,
            targeted_evidence=targeted,
            released_tokens=released_tokens,
            budget_before=before,
            budget_after=after,
        )

    def record_revision(
        self,
        revision: RevisionAssessment | Mapping[str, Any],
    ) -> RevisionAssessment:
        """Ground a final revision against every item released in this run."""

        self._ensure_open()
        if self._revision_recorded:
            raise EvidenceAgentStateError(
                "a revision has already been recorded for this run"
            )
        parsed = (
            revision
            if isinstance(revision, RevisionAssessment)
            else RevisionAssessment.from_dict(revision)
        )
        if not self._released:
            raise EvidenceAgentStateError(
                "revision cannot be recorded before evidence release"
            )
        combined = TargetedEvidence(
            text="\n".join(item.text.rstrip() for item in self._released) + "\n",
            selected_segment_ids=tuple(
                dict.fromkeys(
                    segment_id
                    for item in self._released
                    for segment_id in item.selected_segment_ids
                )
            ),
            selected_atomic_evidence_ids=tuple(
                dict.fromkeys(
                    evidence_id
                    for item in self._released
                    for evidence_id in item.selected_atomic_evidence_ids
                )
            ),
            canonical_evidence_ids=tuple(
                dict.fromkeys(
                    evidence_id
                    for item in self._released
                    for evidence_id in item.canonical_evidence_ids
                )
            ),
        )
        _validate_revision_grounding(parsed, combined)
        self._ledger.append(
            "revision_recorded",
            {
                "revision_status": parsed.revision_status,
                "cited_segment_ids": list(parsed.cited_segment_ids),
                "cited_evidence_ids": list(parsed.cited_evidence_ids),
                "newly_considered_evidence_ids": list(
                    parsed.newly_considered_evidence_ids
                ),
                "change_summary": parsed.change_summary,
                "ground_truth_read": False,
            },
        )
        self._revision_recorded = True
        return parsed

    def seal(self, outcome: WorkflowOutcome) -> EvidenceLedgerVerification:
        self._ensure_open()
        allowed = {
            "accepted_initial",
            "accepted_after_preservation",
            "accepted_after_revision",
            "referred_unresolved",
        }
        if outcome not in allowed:
            raise ValueError(f"unsupported Evidence Agent outcome: {outcome!r}")
        if outcome == "accepted_initial":
            if (
                self._queries_used
                or self._atomic_records_released
                or self._revision_recorded
            ):
                raise EvidenceAgentStateError(
                    "accepted_initial cannot follow evidence access"
                )
        elif not self._revision_recorded:
            raise EvidenceAgentStateError(
                "post-review outcomes require a grounded revision"
            )
        self._ledger.append(
            "agent_closed",
            {
                "outcome": outcome,
                "final_budget": self.budget.to_dict(),
                "released_evidence_ids": sorted(self._seen_atomic_ids),
                "ground_truth_read": False,
            },
        )
        self._closed = True
        self._release_lease()
        return verify_evidence_access_ledger(self._ledger.path)

    def abort(self, error: BaseException) -> None:
        if self._closed:
            return
        try:
            self._ledger.append(
                "agent_aborted",
                {
                    "error_type": error.__class__.__name__,
                    "final_budget": self.budget.to_dict(),
                    "ground_truth_read": False,
                },
            )
            self._ledger.append(
                "agent_closed",
                {
                    "outcome": "aborted",
                    "final_budget": self.budget.to_dict(),
                    "released_evidence_ids": sorted(self._seen_atomic_ids),
                    "ground_truth_read": False,
                },
            )
        finally:
            self._closed = True
            self._release_lease()

    def _restore(self, entries: Sequence[Mapping[str, Any]]) -> None:
        """Replay a verified ledger and reconstruct all security-relevant state."""

        for entry in entries:
            event = entry["event_type"]
            payload = entry["payload"]
            if (
                self._revision_recorded
                and event in {"candidates_exposed", "evidence_released"}
            ):
                raise EvidenceLedgerError(
                    "access ledger performs evidence access after revision"
                )
            if event == "candidates_exposed":
                query = EvidenceQuery.from_dict(payload["query"])
                candidate_set = self._catalog.shortlist(
                    query,
                    exclude_atomic_ids=tuple(sorted(self._seen_atomic_ids)),
                )
                fingerprint = _candidate_set_sha256(candidate_set)
                if (
                    fingerprint != payload["candidate_set_sha256"]
                    or list(candidate_set.candidate_evidence_ids)
                    != payload["candidate_evidence_ids"]
                    or len(candidate_set.candidates) != payload["candidate_count"]
                ):
                    raise EvidenceLedgerError(
                        "resumed candidate set cannot be reproduced from the "
                        "immutable evidence store"
                    )
                self._queries_used += 1
                self._issued[fingerprint] = candidate_set
            elif event == "evidence_released":
                fingerprint = str(payload["candidate_set_sha256"])
                candidate_set = self._issued.get(fingerprint)
                if candidate_set is None:
                    raise EvidenceLedgerError(
                        "released evidence has no reproducible issued candidate set"
                    )
                identifiers = tuple(
                    str(item)
                    for item in payload["selected_atomic_evidence_ids"]
                )
                if set(identifiers) & self._seen_atomic_ids:
                    raise EvidenceLedgerError(
                        "access ledger releases the same atomic evidence twice"
                    )
                targeted = self._catalog.materialize(
                    candidate_set,
                    AtomicSelection(identifiers),
                )
                observed_tokens = self.token_counter.count(targeted.text)
                if (
                    observed_tokens != payload["released_tokens"]
                    or _sha256_text(targeted.text)
                    != payload["released_text_sha256"]
                    or list(targeted.selected_segment_ids)
                    != payload["selected_segment_ids"]
                    or len(targeted.canonical_evidence_ids)
                    != payload["canonical_evidence_count"]
                ):
                    raise EvidenceLedgerError(
                        "resumed evidence release cannot be reproduced from the "
                        "immutable evidence store and tokenizer"
                    )
                self._atomic_records_released += len(identifiers)
                self._released_tokens += observed_tokens
                self._seen_atomic_ids.update(identifiers)
                self._used_candidate_sets.add(fingerprint)
                self._released.append(targeted)
            elif event == "revision_recorded":
                if self._revision_recorded or not self._released:
                    raise EvidenceLedgerError(
                        "access ledger has an invalid revision transition"
                    )
                self._revision_recorded = True
        if self._queries_used > self.config.max_queries:
            raise EvidenceLedgerError(
                "resumed ledger exceeds configured query budget"
            )
        if self._atomic_records_released > self.config.max_atomic_records:
            raise EvidenceLedgerError(
                "resumed ledger exceeds configured atomic-record budget"
            )
        if self._released_tokens > self.config.max_released_tokens:
            raise EvidenceLedgerError(
                "resumed ledger exceeds configured token budget"
            )

    def _deny(
        self,
        event_type: str,
        reason_code: str,
        query: EvidenceQuery,
    ) -> None:
        self._ledger.append(
            event_type,
            {
                "reason_code": reason_code,
                "query": query.to_dict(),
                "budget": self.budget.to_dict(),
            },
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise EvidenceAgentStateError("EvidenceAgent run is closed")

    def _release_lease(self) -> None:
        if self._lease is None:
            return
        fcntl.flock(self._lease.fileno(), fcntl.LOCK_UN)
        self._lease.close()
        self._lease = None


_ContractValue = TypeVar("_ContractValue")


def _completion_text_for_repair(value: Any) -> str:
    if isinstance(value, Mapping):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    text = str(value).strip()
    return text if text else "<empty response>"


def _complete_with_contract_repair(
    model: CompletionModel
    | Callable[
        [Sequence[Mapping[str, str]]],
        str | Mapping[str, Any],
    ],
    messages: Sequence[Mapping[str, str]],
    path: str,
    validate: Callable[[Mapping[str, Any]], _ContractValue],
    *,
    max_retries: int,
    repair_guidance: str | None = None,
) -> tuple[_ContractValue, int, int]:
    """Run one typed model stage with bounded, explicit JSON repair turns."""

    active_messages = [dict(message) for message in messages]
    calls = 0
    repairs = 0
    while True:
        completion = _invoke_model(model, active_messages)
        calls += 1
        try:
            payload = _parse_completion(completion, path)
            return validate(payload), calls, repairs
        except ValueError as error:
            if repairs >= max_retries:
                raise
            repairs += 1
            detail = " ".join(str(error).split())[:800]
            if repair_guidance is None:
                repair_rule = (
                    "When the error lists allowed values, copy one of those "
                    "values verbatim; never replace spaces or hyphens with "
                    "underscores."
                )
            else:
                repair_rule = (
                    "When the error lists allowed values, follow this additional "
                    f"repair rule: {repair_guidance}"
                )
            active_messages.extend(
                (
                    {
                        "role": "assistant",
                        "content": _completion_text_for_repair(completion),
                    },
                    {
                        "role": "user",
                        "content": (
                            "FORMAT REPAIR\n"
                            "Your previous response violated the required JSON "
                            f"contract: {detail}\n"
                            "Re-read the required output contract above and return "
                            "one corrected JSON object only. Do not explain the "
                            "correction, add fields, or use Markdown. "
                            f"{repair_rule}"
                        ),
                    },
                )
            )


def _validated_initial(
    payload: Mapping[str, Any],
    available_segment_ids: Sequence[str],
    *,
    probability_anchor: float | None = None,
) -> InitialAssessment:
    initial = InitialAssessment.from_dict(payload)
    if probability_anchor is not None and not math.isclose(
        initial.risk_probability,
        probability_anchor,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise ValueError(
            "risk_probability must equal the supplied frozen OOF model score "
            f"{probability_anchor!r}; received {initial.risk_probability!r}"
        )
    try:
        _validate_initial_grounding(initial, available_segment_ids)
    except EvidenceGroundingError as error:
        raise EvidenceGroundingError(
            f"{error} Allowed segment IDs are "
            f"{list(available_segment_ids)}."
        ) from error
    return initial


def _initial_messages_with_grounding_allowlist(
    messages: Sequence[Mapping[str, str]],
    available_segment_ids: Sequence[str],
) -> list[dict[str, str]]:
    output = [dict(message) for message in messages]
    allowed = tuple(dict.fromkeys(str(value) for value in available_segment_ids))
    output.append(
        {
            "role": "user",
            "content": (
                "GROUNDING ALLOWLIST\n"
                + json.dumps(
                    {"allowed_segment_ids": list(allowed)},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\nEvery S identifier in every InitialAssessment ID array "
                "must occur in allowed_segment_ids exactly. Never infer the next "
                "numeric ID. If no listed segment supports a field, return [] "
                "for that field."
            ),
        }
    )
    return output


def _project_initial_grounding(
    payload: Mapping[str, Any],
    available_segment_ids: Sequence[str],
    *,
    probability_anchor: float | None,
    raw_completion: Any,
) -> tuple[InitialAssessment, dict[str, Any]]:
    """Drop only non-allowlisted S-ID strings after model repair is exhausted."""

    allowed = tuple(dict.fromkeys(str(value) for value in available_segment_ids))
    allowed_set = set(allowed)
    projected_payload = dict(payload)
    dropped: dict[str, list[str]] = {}
    preserved: dict[str, list[str]] = {}
    for field_name in (
        "supporting_segment_ids",
        "contradictory_segment_ids",
        "uncertain_segment_ids",
        "requested_segment_ids",
    ):
        raw_values = projected_payload.get(field_name)
        if (
            not isinstance(raw_values, list)
            or any(not isinstance(identifier, str) for identifier in raw_values)
        ):
            raise ValueError(
                f"{field_name} must remain an array of identifier strings"
            )
        original = list(raw_values)
        kept = [identifier for identifier in original if identifier in allowed_set]
        removed = [
            identifier for identifier in original if identifier not in allowed_set
        ]
        projected_payload[field_name] = kept
        if kept:
            preserved[field_name] = kept
        if removed:
            dropped[field_name] = removed
    if not dropped:
        raise EvidenceGroundingError(
            "initial grounding recovery found no non-allowlisted segment value "
            "to remove"
        )
    projected = InitialAssessment.from_dict(projected_payload)
    if probability_anchor is not None and not math.isclose(
        projected.risk_probability,
        probability_anchor,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise ValueError(
            "risk_probability must equal the supplied frozen OOF model score "
            f"{probability_anchor!r}; received "
            f"{projected.risk_probability!r}"
        )
    _validate_initial_grounding(projected, allowed)
    raw_text = _completion_text_for_repair(raw_completion)
    projected_text = json.dumps(
        projected.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    allowed_text = json.dumps(
        list(allowed),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return projected, {
        "policy_version": INITIAL_GROUNDING_RECOVERY_POLICY_VERSION,
        "applied": True,
        "trigger": (
            "contract_retries_exhausted_nonallowlisted_segment_values"
        ),
        "raw_completion": raw_text,
        "raw_completion_sha256": hashlib.sha256(
            raw_text.encode("utf-8")
        ).hexdigest(),
        "allowed_segment_ids": list(allowed),
        "allowed_segment_ids_sha256": hashlib.sha256(
            allowed_text.encode("utf-8")
        ).hexdigest(),
        "preserved_segment_ids_by_field": preserved,
        "dropped_nonallowlisted_segment_values_by_field": dropped,
        "projected_assessment_sha256": hashlib.sha256(
            projected_text.encode("utf-8")
        ).hexdigest(),
        "risk_probability_anchor": probability_anchor,
        "risk_probability_anchor_preserved": (
            probability_anchor is None
            or math.isclose(
                projected.risk_probability,
                probability_anchor,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
        ),
        "other_fields_modified": False,
    }


def _complete_initial_with_grounding_recovery(
    model: CompletionModel
    | Callable[
        [Sequence[Mapping[str, str]]],
        str | Mapping[str, Any],
    ],
    messages: Sequence[Mapping[str, str]],
    path: str,
    available_segment_ids: Sequence[str],
    *,
    probability_anchor: float | None,
    max_retries: int,
) -> tuple[InitialAssessment, int, int, dict[str, Any] | None]:
    """Run the initial completion and audibly recover only grounding overflow."""

    completions: list[Any] = []

    def capture(
        active_messages: Sequence[Mapping[str, str]],
    ) -> str | Mapping[str, Any]:
        completion = _invoke_model(model, active_messages)
        completions.append(completion)
        return completion

    allowlisted_messages = _initial_messages_with_grounding_allowlist(
        messages,
        available_segment_ids,
    )
    try:
        initial, calls, repairs = _complete_with_contract_repair(
            capture,
            allowlisted_messages,
            path,
            lambda payload: _validated_initial(
                payload,
                available_segment_ids,
                probability_anchor=probability_anchor,
            ),
            max_retries=max_retries,
            repair_guidance=(
                "Correct the stated violation. For an unknown segment ID, "
                "delete unknown identifiers from that array; never substitute "
                "an unrelated allowed identifier. Use [] when nothing valid "
                "remains. Keep the frozen probability unchanged."
            ),
        )
        return initial, calls, repairs, None
    except ValueError as original_error:
        if not completions:
            raise
        final_completion = completions[-1]
        try:
            final_payload = _parse_completion(final_completion, path)
            projected, audit = _project_initial_grounding(
                final_payload,
                available_segment_ids,
                probability_anchor=probability_anchor,
                raw_completion=final_completion,
            )
        except ValueError:
            raise original_error
        return projected, len(completions), max(0, len(completions) - 1), audit


def _initial_messages_with_probability_anchor(
    messages: Sequence[Mapping[str, str]],
    probability_anchor: float | None,
) -> list[dict[str, str]]:
    output = [dict(message) for message in messages]
    if probability_anchor is None:
        return output
    if (
        isinstance(probability_anchor, bool)
        or not isinstance(probability_anchor, (int, float))
        or not math.isfinite(float(probability_anchor))
        or not 0.0 <= float(probability_anchor) <= 1.0
    ):
        raise ValueError("initial_probability_anchor must be a finite probability")
    output.append(
        {
            "role": "user",
            "content": (
                "FROZEN OOF MODEL SCORE ANCHOR\n"
                "A separately frozen out-of-fold classifier over this same "
                "first-pass input produced the model score "
                f"{float(probability_anchor)!r}. This is a model prediction, "
                "not a target label. Set risk_probability to exactly that "
                "number. Infer confidence, source assessments, and grounded "
                "segment identifiers from the supplied evidence, then return "
                "the required InitialAssessment JSON object only."
            ),
        }
    )
    return output


def _stage_models(
    model: CompletionAdapter | EvidenceAgentStageModels,
) -> EvidenceAgentStageModels:
    if isinstance(model, EvidenceAgentStageModels):
        return model
    return EvidenceAgentStageModels.uniform(model)


def _tokenizer_for_stage_models(models: EvidenceAgentStageModels) -> Any | None:
    """Return one shared tokenizer, rejecting ambiguous automatic accounting."""

    tokenizers = [
        getattr(getattr(models, name), "tokenizer", None)
        for name in ("initial", "query", "selection", "revision")
    ]
    available = [tokenizer for tokenizer in tokenizers if tokenizer is not None]
    if not available:
        return None
    identities = {
        (
            tokenizer.__class__.__name__,
            str(getattr(tokenizer, "name_or_path", "")),
        )
        for tokenizer in available
    }
    if len(identities) != 1:
        raise ValueError(
            "stage models use different tokenizers; provide an explicit token_counter"
        )
    return available[0]


def _validated_query_and_candidates(
    payload: Mapping[str, Any],
    selected_segment_ids: Sequence[str],
    agent: EvidenceAgent,
) -> tuple[EvidenceQuery, CandidateSet, tuple[str, ...]]:
    normalized, syntax_normalizations = _canonicalize_query_syntax(payload)
    try:
        query = EvidenceQuery.from_dict(normalized)
    except ValueError as error:
        raise ValueError(
            f"{error} Allowed fixed core slot labels are "
            f"{[slot.label for slot in CORE_SLOTS]}."
        ) from error
    if query.segment_id not in selected_segment_ids:
        raise ValueError(
            "evidence query segment was not selected by the rethink decision; "
            f"allowed segment IDs are {list(selected_segment_ids)}"
        )
    return query, agent.issue(query), syntax_normalizations


def _canonicalize_query_syntax(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Normalize only bijective punctuation/case aliases of fixed enums."""

    normalized = dict(payload)
    changes: list[str] = []

    def canonical(
        value: Any,
        choices: Sequence[str],
        path: str,
    ) -> Any:
        if not isinstance(value, str) or value in choices:
            return value
        matches = [
            choice
            for choice in choices
            if _normalized_key(choice) == _normalized_key(value)
        ]
        if len(matches) != 1:
            return value
        changes.append(f"{path}: {value!r} -> {matches[0]!r}")
        return matches[0]

    if "purpose" in normalized:
        normalized["purpose"] = canonical(
            normalized["purpose"],
            RETRIEVAL_PURPOSES,
            "purpose",
        )
    if "pattern" in normalized:
        normalized["pattern"] = canonical(
            normalized["pattern"],
            RETRIEVAL_PATTERNS,
            "pattern",
        )
    slots = normalized.get("target_slots")
    if isinstance(slots, Sequence) and not isinstance(slots, (str, bytes)):
        allowed_slots = tuple(slot.label for slot in CORE_SLOTS)
        normalized["target_slots"] = [
            canonical(value, allowed_slots, f"target_slots[{index}]")
            for index, value in enumerate(slots)
        ]
    return normalized, tuple(changes)


def _validated_selection_and_release(
    payload: Mapping[str, Any],
    candidate_set: CandidateSet,
    agent: EvidenceAgent,
) -> tuple[AtomicSelection, EvidenceRelease]:
    selection = AtomicSelection.from_dict(payload)
    return selection, agent.release(candidate_set, selection)


def _validated_revision(
    payload: Mapping[str, Any],
    release: EvidenceRelease,
    agent: EvidenceAgent,
) -> RevisionAssessment:
    try:
        return agent.record_revision(payload)
    except EvidenceGroundingError as error:
        evidence = release.targeted_evidence
        raise EvidenceGroundingError(
            f"{error} Allowed segment IDs are "
            f"{list(evidence.selected_segment_ids)}; allowed atomic evidence "
            f"IDs are {list(evidence.selected_atomic_evidence_ids)}."
        ) from error


class EvidenceAgentWorkflow:
    """Complete initial→audit→query→release→revision model workflow."""

    def __init__(
        self,
        *,
        config: EvidenceAgentConfig | None = None,
        policy: RethinkPolicyProtocol | None = None,
        initial_prompt_builder: PromptBuilder | None = None,
        retrieval_prompt_builder: QueryRetrievalPromptBuilder | None = None,
        max_contract_retries: int = 1,
        initial_grounding_recovery: Literal[
            "disabled",
            "drop_unknown_ids_after_retries",
        ] = "disabled",
    ) -> None:
        if (
            isinstance(max_contract_retries, bool)
            or not isinstance(max_contract_retries, int)
            or not 0 <= max_contract_retries <= 2
        ):
            raise ValueError("max_contract_retries must be 0, 1, or 2")
        if initial_grounding_recovery not in {
            "disabled",
            "drop_unknown_ids_after_retries",
        }:
            raise ValueError(
                "initial_grounding_recovery must be disabled or "
                "drop_unknown_ids_after_retries"
            )
        self.config = config or EvidenceAgentConfig()
        self.policy = policy or RethinkPolicy()
        self.initial_prompt_builder = initial_prompt_builder or PromptBuilder()
        self.retrieval_prompt_builder = (
            retrieval_prompt_builder or QueryRetrievalPromptBuilder()
        )
        self.max_contract_retries = max_contract_retries
        self.initial_grounding_recovery = initial_grounding_recovery

    def run(
        self,
        session_dir: str | Path,
        task: TaskSpec,
        model: CompletionAdapter | EvidenceAgentStageModels,
        *,
        ledger_path: str | Path,
        transcript_dir: str | Path | None = None,
        token_counter: EvidenceTokenCounter | None = None,
        evidence_protocol: str = "auto",
        run_id: str | None = None,
        initial_probability_anchor: float | None = None,
    ) -> EvidenceAgentWorkflowResult:
        """Run every model stage, optionally anchoring the OOF initial score."""

        models = _stage_models(model)
        counter = token_counter
        if counter is None:
            tokenizer = _tokenizer_for_stage_models(models)
            if tokenizer is None:
                raise ValueError(
                    "a token_counter is required when stage models expose no tokenizer"
                )
            counter = TokenizerTokenCounter.from_tokenizer(tokenizer)
        agent = EvidenceAgent.open(
            session_dir,
            ledger_path=ledger_path,
            token_counter=counter,
            config=self.config,
            transcript_dir=transcript_dir,
            evidence_protocol=evidence_protocol,
            run_id=run_id,
        )
        calls = 0
        repairs = 0
        try:
            initial_messages = _initial_messages_with_probability_anchor(
                self.initial_prompt_builder.build_initial_messages(
                    task,
                    agent.first_pass_text,
                ),
                initial_probability_anchor,
            )
            initial_grounding_audit: Mapping[str, Any] | None = None
            if (
                self.initial_grounding_recovery
                == "drop_unknown_ids_after_retries"
            ):
                (
                    initial,
                    stage_calls,
                    stage_repairs,
                    initial_grounding_audit,
                ) = _complete_initial_with_grounding_recovery(
                    models.initial,
                    initial_messages,
                    "initial model completion",
                    agent.available_segment_ids,
                    probability_anchor=initial_probability_anchor,
                    max_retries=self.max_contract_retries,
                )
            else:
                (
                    initial,
                    stage_calls,
                    stage_repairs,
                ) = _complete_with_contract_repair(
                    models.initial,
                    initial_messages,
                    "initial model completion",
                    lambda payload: _validated_initial(
                        payload,
                        agent.available_segment_ids,
                        probability_anchor=initial_probability_anchor,
                    ),
                    max_retries=self.max_contract_retries,
                )
            calls += stage_calls
            repairs += stage_repairs
            return self._continue_run(
                agent=agent,
                task=task,
                initial=initial,
                models=models,
                calls=calls,
                repairs=repairs,
                initial_grounding_audit=initial_grounding_audit,
            )
        except Exception as error:
            agent.abort(error)
            raise

    def run_from_initial(
        self,
        session_dir: str | Path,
        task: TaskSpec,
        initial: InitialAssessment | Mapping[str, Any],
        model: CompletionAdapter | EvidenceAgentStageModels,
        *,
        ledger_path: str | Path,
        transcript_dir: str | Path | None = None,
        token_counter: EvidenceTokenCounter | None = None,
        evidence_protocol: str = "auto",
        run_id: str | None = None,
    ) -> EvidenceAgentWorkflowResult:
        """Run audit/query/release/revision from a frozen OOF initial assessment.

        The supplied assessment is validated against the current Evidence
        bundle.  No initial model call is made, so Stage-0 is an input to the
        loop rather than a gate on whether the loop can be exercised.
        """

        models = _stage_models(model)
        counter = token_counter
        if counter is None:
            tokenizer = _tokenizer_for_stage_models(models)
            if tokenizer is None:
                raise ValueError(
                    "a token_counter is required when stage models expose no tokenizer"
                )
            counter = TokenizerTokenCounter.from_tokenizer(tokenizer)
        agent = EvidenceAgent.open(
            session_dir,
            ledger_path=ledger_path,
            token_counter=counter,
            config=self.config,
            transcript_dir=transcript_dir,
            evidence_protocol=evidence_protocol,
            run_id=run_id,
        )
        try:
            assessment = (
                initial
                if isinstance(initial, InitialAssessment)
                else InitialAssessment.from_dict(initial)
            )
            _validate_initial_grounding(
                assessment,
                agent.available_segment_ids,
            )
            return self._continue_run(
                agent=agent,
                task=task,
                initial=assessment,
                models=models,
                calls=0,
                repairs=0,
            )
        except Exception as error:
            agent.abort(error)
            raise

    def _continue_run(
        self,
        *,
        agent: EvidenceAgent,
        task: TaskSpec,
        initial: InitialAssessment,
        models: EvidenceAgentStageModels,
        calls: int,
        repairs: int,
        initial_grounding_audit: Mapping[str, Any] | None = None,
    ) -> EvidenceAgentWorkflowResult:
        decision = self.policy.evaluate(
            initial,
            agent.available_segment_ids,
        )
        if not decision.should_rethink:
            ledger = agent.seal("accepted_initial")
            return EvidenceAgentWorkflowResult(
                evidence_protocol_version=agent.evidence_protocol_version,
                task=task,
                outcome="accepted_initial",
                initial_assessment=initial,
                decision=decision,
                rounds=(),
                revision_assessment=None,
                final_risk_probability=initial.risk_probability,
                final_confidence=initial.confidence,
                first_pass_tokens=agent.first_pass_tokens,
                transcript_tier=agent.transcript_tier,
                budget=agent.budget,
                ledger=ledger,
                model_call_count=calls,
                contract_repair_count=repairs,
                initial_grounding_audit=initial_grounding_audit,
            )

        retrieval_map = agent.render_retrieval_map(
            decision.selected_segment_ids
        )
        query_messages = (
            self.retrieval_prompt_builder.build_query_messages(
                task,
                initial,
                decision,
                retrieval_map,
            )
        )
        query_result, stage_calls, stage_repairs = (
            _complete_with_contract_repair(
                models.query,
                query_messages,
                "evidence query model completion",
                lambda payload: _validated_query_and_candidates(
                    payload,
                    decision.selected_segment_ids,
                    agent,
                ),
                max_retries=self.max_contract_retries,
            )
        )
        calls += stage_calls
        repairs += stage_repairs
        query, candidate_set, query_syntax_normalizations = query_result

        selection_messages = (
            self.retrieval_prompt_builder.build_selection_messages(
                task,
                initial,
                candidate_set,
            )
        )
        selection_result, stage_calls, stage_repairs = (
            _complete_with_contract_repair(
                models.selection,
                selection_messages,
                "atomic selection model completion",
                lambda payload: _validated_selection_and_release(
                    payload,
                    candidate_set,
                    agent,
                ),
                max_retries=self.max_contract_retries,
            )
        )
        calls += stage_calls
        repairs += stage_repairs
        selection, release = selection_result

        revision_messages = (
            self.initial_prompt_builder.build_revision_messages(
                task,
                initial,
                decision,
                release.targeted_evidence.text,
            )
        )
        revision, stage_calls, stage_repairs = (
            _complete_with_contract_repair(
                models.revision,
                revision_messages,
                "revision model completion",
                lambda payload: _validated_revision(
                    payload,
                    release,
                    agent,
                ),
                max_retries=self.max_contract_retries,
            )
        )
        calls += stage_calls
        repairs += stage_repairs
        outcome: WorkflowOutcome = {
            "preserved": "accepted_after_preservation",
            "revised": "accepted_after_revision",
            "unresolved": "referred_unresolved",
        }[revision.revision_status]
        ledger = agent.seal(outcome)
        return EvidenceAgentWorkflowResult(
            evidence_protocol_version=agent.evidence_protocol_version,
            task=task,
            outcome=outcome,
            initial_assessment=initial,
            decision=decision,
            rounds=(
                EvidenceAgentRound(
                    evidence_query=query,
                    candidate_set=candidate_set,
                    atomic_selection=selection,
                    release=release,
                    query_syntax_normalizations=(
                        query_syntax_normalizations
                    ),
                ),
            ),
            revision_assessment=revision,
            final_risk_probability=revision.revised_risk_probability,
            final_confidence=revision.revised_confidence,
            first_pass_tokens=agent.first_pass_tokens,
            transcript_tier=agent.transcript_tier,
            budget=agent.budget,
            ledger=ledger,
            model_call_count=calls,
            contract_repair_count=repairs,
            initial_grounding_audit=initial_grounding_audit,
        )


__all__ = [
    "AGENT_PROTOCOL_VERSION",
    "CallableTokenCounter",
    "EvidenceAgent",
    "EvidenceAgentConfig",
    "EvidenceAgentError",
    "EvidenceAgentRound",
    "EvidenceAgentStateError",
    "EvidenceAgentStageModels",
    "EvidenceAgentWorkflow",
    "EvidenceAgentWorkflowResult",
    "EvidenceBudgetExceeded",
    "EvidenceBudgetSnapshot",
    "EvidenceLedgerError",
    "EvidenceLedgerVerification",
    "EvidenceRelease",
    "EvidenceTokenCounter",
    "TokenizerTokenCounter",
    "verify_evidence_access_ledger",
]
