"""Strict text-only Draft SFT, Revision-SFT, and revision-aware ORPO training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from array import array
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Literal

from rethink_mh.experiments.qwen_label_baseline import TEXT_LORA_TARGETS
from rethink_mh.rethinking.contracts import (
    InitialAssessment,
    RevisionActionDecision,
    RevisionAssessment,
)
from rethink_mh.rethinking.query_retrieval import AtomicSelection, EvidenceQuery
from rethink_mh.rethinking.qwen import load_pretrained_model_on_device


TrainingStage = Literal[
    "draft_sft",
    "evidence_literacy_sft",
    "action_sft",
    "revision_sft",
    "action_orpo",
    "orpo",
]
TransitionWeighting = Literal["none", "inverse_session_frequency"]
SamplingPolicy = Literal["shuffle", "transition_balanced"]
_STAGES = frozenset(
    {
        "draft_sft",
        "evidence_literacy_sft",
        "action_sft",
        "revision_sft",
        "action_orpo",
        "orpo",
    }
)
_TRANSITION_WEIGHTING_POLICIES = frozenset(
    {"none", "inverse_session_frequency"}
)
_SAMPLING_POLICIES = frozenset({"shuffle", "transition_balanced"})
_TRANSITIONS = frozenset({"CC_preserve", "WC_revise", "WW_refer"})
_ACTION_PREFERENCES = frozenset(
    {
        "CW_harmful_revision",
        "WW_forced_confident_prediction",
        "WW_failed_preservation",
        "action_rejected_preserve_unchanged",
        "action_rejected_revise_increase",
        "action_rejected_revise_decrease",
        "action_rejected_refer_unresolved",
    }
)
_CONSTRAINT_PREFERENCES = frozenset(
    {
        "constraint_unknown_evidence_id",
        "constraint_missing_required_field",
        "constraint_out_of_scope_clinical_claim",
    }
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read training records {path}: {error}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise ValueError(f"blank training record at {path}:{line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"training record must be an object at {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"training records are empty: {path}")
    return rows


def _messages(value: object, path: str, *, assistant: bool) -> list[dict[str, str]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{path} must be a message array")
    parsed: list[dict[str, str]] = []
    for index, message in enumerate(value):
        if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
            raise ValueError(f"{path}[{index}] must contain only role and content")
        role = message["role"]
        content = message["content"]
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"{path}[{index}] has unsupported role {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{path}[{index}].content must be non-empty text")
        parsed.append({"role": str(role), "content": content})
    expected_roles = ["system", "user", "assistant"] if assistant else ["system", "user"]
    if [message["role"] for message in parsed] != expected_roles:
        raise ValueError(f"{path} roles must be exactly {expected_roles}")
    return parsed


def _json_object(text: str, path: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} must be one complete JSON object") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be one JSON object")
    if _canonical_json(value) != text:
        raise ValueError(f"{path} must use canonical compact JSON serialization")
    return value


def _sample_weight(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or not 1.0 <= parsed <= 10.0:
        raise ValueError(f"{path} must be finite and in [1, 10]")
    return parsed


def _grounded(revision: RevisionAssessment, prompt: Sequence[Mapping[str, str]]) -> bool:
    text = "\n".join(message["content"] for message in prompt)
    identifiers = (
        *revision.cited_segment_ids,
        *revision.cited_evidence_ids,
        *revision.preserved_evidence_ids,
        *revision.newly_considered_evidence_ids,
        *revision.rejected_evidence_ids,
        *revision.residual_conflict_segment_ids,
    )
    return all(identifier in text for identifier in identifiers)


@dataclass(frozen=True, slots=True)
class AuditedRecords:
    stage: TrainingStage
    dataset: str
    rows: tuple[dict[str, Any], ...]
    session_count: int
    transition_counts: dict[str, int]
    preference_kind_counts: dict[str, int]

    def summary(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "dataset": self.dataset,
            "record_count": len(self.rows),
            "session_count": self.session_count,
            "transition_counts": self.transition_counts,
            "preference_kind_counts": self.preference_kind_counts,
            "single_dataset": True,
            "ground_truth_in_model_prompt": False,
            "contract_valid": True,
        }


def audit_training_records(stage: TrainingStage, path: Path) -> AuditedRecords:
    """Validate all model-facing records before importing any training runtime."""

    if stage not in _STAGES:
        raise ValueError(f"unsupported training stage: {stage}")
    rows = _read_jsonl(path)
    datasets: set[str] = set()
    sessions: set[str] = set()
    transitions: Counter[str] = Counter()
    preference_kinds: Counter[str] = Counter()
    preference_keys: set[tuple[str, str]] = set()
    expected_type = {
        "draft_sft": "draft_sft",
        "evidence_literacy_sft": "evidence_literacy_sft",
        "action_sft": "action_sft",
        "revision_sft": "revision_sft",
        "action_orpo": "action_preference",
        "orpo": "revision_preference",
    }[stage]

    for index, row in enumerate(rows):
        prefix = f"records[{index}]"
        if row.get("record_type") != expected_type:
            raise ValueError(f"{prefix}.record_type must be {expected_type!r}")
        dataset = row.get("dataset")
        session_id = row.get("session_id")
        if not isinstance(dataset, str) or not dataset:
            raise ValueError(f"{prefix}.dataset must be non-empty text")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError(f"{prefix}.session_id must be non-empty text")
        datasets.add(dataset)
        sessions.add(session_id)
        _sample_weight(row.get("sample_weight", 1.0), f"{prefix}.sample_weight")

        transition = row.get("transition_target")
        if stage not in {"draft_sft", "evidence_literacy_sft"}:
            if transition not in _TRANSITIONS:
                raise ValueError(f"{prefix}.transition_target is invalid")
            transitions[str(transition)] += 1

        if stage in {
            "draft_sft",
            "evidence_literacy_sft",
            "action_sft",
            "revision_sft",
        }:
            messages = _messages(row.get("messages"), f"{prefix}.messages", assistant=True)
            supervision = row.get("supervision")
            if not isinstance(supervision, Mapping):
                raise ValueError(f"{prefix}.supervision must be an object")
            if supervision.get("used_in_model_text") is not False:
                raise ValueError(f"{prefix} does not certify label-free model text")
            assistant_value = _json_object(
                messages[-1]["content"], f"{prefix}.messages[-1].content"
            )
            if stage == "evidence_literacy_sft":
                if supervision.get("sample_targets_accessed") is not False:
                    raise ValueError(
                        f"{prefix} does not certify target-free literacy supervision"
                    )
                if supervision.get("annotator_uses_outcomes") is not False:
                    raise ValueError(
                        f"{prefix} literacy reference target must be outcome-independent"
                    )
                contract_type = row.get("contract_type")
                prompt_text = "\n".join(
                    message["content"] for message in messages[:-1]
                )
                if contract_type == "evidence_query":
                    query = EvidenceQuery.from_dict(assistant_value)
                    grounded_terms = (
                        query.segment_id,
                        *query.target_slots,
                    )
                    if not all(term in prompt_text for term in grounded_terms):
                        raise ValueError(
                            f"{prefix} evidence query is not grounded in its map"
                        )
                elif contract_type == "atomic_selection":
                    selection = AtomicSelection.from_dict(assistant_value)
                    if not all(
                        identifier in prompt_text
                        for identifier in selection.selected_evidence_ids
                    ):
                        raise ValueError(
                            f"{prefix} atomic selection contains an unshown ID"
                        )
                else:
                    raise ValueError(
                        f"{prefix}.contract_type must be evidence_query or "
                        "atomic_selection"
                    )
            else:
                label = supervision.get("label")
                if isinstance(label, bool) or label not in {0, 1}:
                    raise ValueError(f"{prefix}.supervision.label must be binary")
            if stage == "draft_sft":
                InitialAssessment.from_dict(assistant_value)
            elif stage == "action_sft":
                RevisionActionDecision.from_dict(assistant_value)
            elif stage == "revision_sft":
                revision = RevisionAssessment.from_dict(assistant_value)
                if not _grounded(revision, messages[:-1]):
                    raise ValueError(f"{prefix} chosen revision contains an ungrounded ID")
            continue

        prompt = _messages(
            row.get("prompt_messages"), f"{prefix}.prompt_messages", assistant=False
        )
        if row.get("ground_truth_in_prompt") is not False:
            raise ValueError(f"{prefix} does not certify a label-free prompt")
        chosen_text = row.get("chosen")
        rejected_text = row.get("rejected")
        if not isinstance(chosen_text, str) or not isinstance(rejected_text, str):
            raise ValueError(f"{prefix} chosen and rejected must be text")
        if chosen_text == rejected_text:
            raise ValueError(f"{prefix} chosen and rejected completions are identical")
        kind = row.get("preference_kind")
        allowed_kinds = (
            _ACTION_PREFERENCES
            if stage == "action_orpo"
            else _ACTION_PREFERENCES | _CONSTRAINT_PREFERENCES
        )
        if kind not in allowed_kinds:
            raise ValueError(f"{prefix}.preference_kind is unsupported")
        key = (session_id, str(kind))
        if key in preference_keys:
            raise ValueError(f"duplicate preference kind for session {session_id}: {kind}")
        preference_keys.add(key)
        preference_kinds[str(kind)] += 1
        chosen_object = _json_object(chosen_text, f"{prefix}.chosen")
        rejected_object = _json_object(rejected_text, f"{prefix}.rejected")
        if stage == "action_orpo":
            RevisionActionDecision.from_dict(chosen_object)
            RevisionActionDecision.from_dict(rejected_object)
        else:
            chosen = RevisionAssessment.from_dict(chosen_object)
            if not _grounded(chosen, prompt):
                raise ValueError(f"{prefix}.chosen contains an ungrounded ID")
        if stage == "orpo" and kind in _ACTION_PREFERENCES:
            rejected = RevisionAssessment.from_dict(rejected_object)
            if not _grounded(rejected, prompt):
                raise ValueError(f"{prefix}.rejected action contains an ungrounded ID")
        elif stage == "orpo" and kind == "constraint_unknown_evidence_id":
            rejected = RevisionAssessment.from_dict(rejected_object)
            if _grounded(rejected, prompt):
                raise ValueError(f"{prefix} unknown-ID negative is unexpectedly grounded")
        elif stage == "orpo":
            try:
                RevisionAssessment.from_dict(rejected_object)
            except ValueError:
                pass
            else:
                raise ValueError(f"{prefix} schema/clinical constraint negative is valid")

    if len(datasets) != 1:
        raise ValueError(f"records must contain exactly one dataset, found {sorted(datasets)}")
    if stage not in {"draft_sft", "evidence_literacy_sft"}:
        missing = _TRANSITIONS - set(transitions)
        if missing:
            raise ValueError(f"records omit required transition targets: {sorted(missing)}")
    if stage == "orpo":
        missing_constraints = _CONSTRAINT_PREFERENCES - set(preference_kinds)
        if missing_constraints:
            raise ValueError(
                f"ORPO records omit constraint negative types: {sorted(missing_constraints)}"
            )
        if preference_kinds["CW_harmful_revision"] == 0:
            raise ValueError("ORPO records omit correct-to-wrong harmful revisions")
    return AuditedRecords(
        stage=stage,
        dataset=next(iter(datasets)),
        rows=tuple(rows),
        session_count=len(sessions),
        transition_counts=dict(sorted(transitions.items())),
        preference_kind_counts=dict(sorted(preference_kinds.items())),
    )


@dataclass(frozen=True, slots=True)
class EncodedCompletion:
    # Compact 32-bit storage avoids hundreds of MB of boxed Python integers for
    # D-Vlog's four preference pairs per session.
    input_ids: array
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True, slots=True)
class EncodedSFT:
    session_id: str
    transition: str | None
    sequence: EncodedCompletion
    weight: float


@dataclass(frozen=True, slots=True)
class EncodedPreference:
    session_id: str
    transition: str
    kind: str
    chosen: EncodedCompletion
    rejected: EncodedCompletion
    weight: float


def _transition_weighting_plan(
    audited: AuditedRecords,
    *,
    policy: TransitionWeighting,
    max_effective_sample_weight: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Build auditable class weights without changing the frozen records.

    Revision-SFT applies the multiplier to every trajectory. ORPO applies it only
    to the action preference for each session; schema/grounding constraints do
    not represent a transition decision and therefore retain their source
    weights.
    """

    if policy not in _TRANSITION_WEIGHTING_POLICIES:
        raise ValueError(f"unsupported transition weighting policy: {policy}")
    if audited.stage in {"draft_sft", "evidence_literacy_sft"} or policy == "none":
        return (
            {transition: 1.0 for transition in _TRANSITIONS},
            {
                "applied": False,
                "policy": policy,
                "max_effective_sample_weight": max_effective_sample_weight,
            },
        )

    session_transitions: dict[str, str] = {}
    for row in audited.rows:
        session_id = str(row["session_id"])
        transition = str(row["transition_target"])
        previous = session_transitions.setdefault(session_id, transition)
        if previous != transition:
            raise ValueError(
                f"session {session_id} has inconsistent transition targets"
            )
    counts = Counter(session_transitions.values())
    missing = _TRANSITIONS - set(counts)
    if missing:
        raise ValueError(
            f"cannot balance records missing transitions: {sorted(missing)}"
        )
    total = sum(counts.values())
    multipliers = {
        transition: total / (len(_TRANSITIONS) * counts[transition])
        for transition in sorted(_TRANSITIONS)
    }
    effective_mass: Counter[str] = Counter()
    applied_record_counts: Counter[str] = Counter()
    effective_weights: list[float] = []
    for row in audited.rows:
        applies = audited.stage in {"action_sft", "revision_sft"} or (
            audited.stage in {"action_orpo", "orpo"}
            and row.get("preference_kind") in _ACTION_PREFERENCES
        )
        if not applies:
            continue
        transition = str(row["transition_target"])
        source_weight = _sample_weight(row.get("sample_weight", 1.0), "sample_weight")
        effective = min(
            max_effective_sample_weight,
            source_weight * multipliers[transition],
        )
        effective_mass[transition] += effective
        applied_record_counts[transition] += 1
        effective_weights.append(effective)
    return multipliers, {
        "applied": True,
        "policy": policy,
        "scope": (
            "all_sft_rows"
            if audited.stage in {"action_sft", "revision_sft"}
            else "orpo_action_preferences_only"
        ),
        "session_counts": dict(sorted(counts.items())),
        "multipliers": multipliers,
        "applied_record_counts": dict(sorted(applied_record_counts.items())),
        "effective_weight_mass": dict(sorted(effective_mass.items())),
        "effective_weight_min": min(effective_weights),
        "effective_weight_max": max(effective_weights),
        "max_effective_sample_weight": max_effective_sample_weight,
    }


def _ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("chat template unexpectedly returned more than one sequence")
        value = value[0]
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ValueError("chat template must return one integer token sequence")
    return value


def _encode_completion(
    tokenizer: Any,
    prompt: Sequence[Mapping[str, str]],
    completion: str,
    *,
    max_tokens: int,
) -> EncodedCompletion:
    normalized_prompt = [dict(message) for message in prompt]
    full_messages = [
        *normalized_prompt,
        {"role": "assistant", "content": completion},
    ]
    prompt_ids = _ids(
        tokenizer.apply_chat_template(
            normalized_prompt, tokenize=True, add_generation_prompt=True
        )
    )
    full_ids = _ids(
        tokenizer.apply_chat_template(
            full_messages, tokenize=True, add_generation_prompt=False
        )
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("chat template assistant prefix is not a stable token boundary")
    completion_tokens = len(full_ids) - len(prompt_ids)
    if completion_tokens <= 0:
        raise ValueError("assistant completion token sequence is empty")
    if len(full_ids) > max_tokens:
        raise ValueError(
            f"record has {len(full_ids)} tokens, exceeding fixed limit {max_tokens}; "
            "silent truncation is forbidden"
        )
    return EncodedCompletion(array("I", full_ids), len(prompt_ids), completion_tokens)


def _encode_records(
    audited: AuditedRecords,
    tokenizer: Any,
    *,
    max_tokens: int,
    transition_multipliers: Mapping[str, float] | None = None,
    max_effective_sample_weight: float = 10.0,
) -> tuple[EncodedSFT | EncodedPreference, ...]:
    multipliers = transition_multipliers or {
        transition: 1.0 for transition in _TRANSITIONS
    }
    encoded: list[EncodedSFT | EncodedPreference] = []
    for row in audited.rows:
        source_weight = _sample_weight(
            row.get("sample_weight", 1.0), "sample_weight"
        )
        applies = audited.stage in {"action_sft", "revision_sft"} or (
            audited.stage in {"action_orpo", "orpo"}
            and row.get("preference_kind") in _ACTION_PREFERENCES
        )
        multiplier = (
            multipliers[str(row["transition_target"])] if applies else 1.0
        )
        weight = min(max_effective_sample_weight, source_weight * multiplier)
        if audited.stage in {
            "draft_sft",
            "evidence_literacy_sft",
            "action_sft",
            "revision_sft",
        }:
            messages = row["messages"]
            sequence = _encode_completion(
                tokenizer,
                messages[:-1],
                messages[-1]["content"],
                max_tokens=max_tokens,
            )
            encoded.append(
                EncodedSFT(
                    str(row["session_id"]),
                    (
                        str(row["transition_target"])
                        if audited.stage
                        not in {"draft_sft", "evidence_literacy_sft"}
                        else None
                    ),
                    sequence,
                    weight,
                )
            )
        else:
            prompt = row["prompt_messages"]
            encoded.append(
                EncodedPreference(
                    session_id=str(row["session_id"]),
                    transition=str(row["transition_target"]),
                    kind=str(row["preference_kind"]),
                    chosen=_encode_completion(
                        tokenizer, prompt, str(row["chosen"]), max_tokens=max_tokens
                    ),
                    rejected=_encode_completion(
                        tokenizer, prompt, str(row["rejected"]), max_tokens=max_tokens
                    ),
                    weight=weight,
                )
            )
    return tuple(encoded)


@dataclass(frozen=True, slots=True)
class TrainingRequest:
    stage: TrainingStage
    records_path: Path
    model_path: Path
    output_dir: Path
    init_adapter: Path | None = None
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    attention_implementation: str = "sdpa"
    max_tokens: int = 7500
    epochs: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.10
    scheduler_type: str = "linear"
    gradient_accumulation: int = 8
    max_grad_norm: float = 1.0
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    orpo_beta: float = 0.10
    seed: int = 42
    gradient_checkpointing: bool = True
    preference_audit_limit: int = 96
    orpo_forward_mode: str = "concatenated"
    max_preferences_per_session: int = 2
    constraint_session_fraction: float = 0.20
    transition_weighting: TransitionWeighting = "none"
    max_effective_sample_weight: float = 10.0
    experiment_name: str = "rethink_revision_post_training_v1"
    sampling_policy: SamplingPolicy = "shuffle"

    def __post_init__(self) -> None:
        if self.stage not in _STAGES:
            raise ValueError(f"unsupported training stage: {self.stage}")
        for name in (
            "max_tokens",
            "epochs",
            "gradient_accumulation",
            "lora_rank",
            "lora_alpha",
            "preference_audit_limit",
            "max_preferences_per_session",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "max_grad_norm", "orpo_beta"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.orpo_forward_mode not in {"concatenated", "streaming"}:
            raise ValueError("orpo_forward_mode must be concatenated or streaming")
        if not 0.0 <= self.constraint_session_fraction <= 1.0:
            raise ValueError("constraint_session_fraction must be in [0, 1]")
        if self.transition_weighting not in _TRANSITION_WEIGHTING_POLICIES:
            raise ValueError("unsupported transition_weighting policy")
        if (
            not math.isfinite(self.max_effective_sample_weight)
            or self.max_effective_sample_weight < 1.0
        ):
            raise ValueError(
                "max_effective_sample_weight must be finite and at least 1"
            )
        if not self.experiment_name.strip():
            raise ValueError("experiment_name must be non-empty")
        if self.sampling_policy not in _SAMPLING_POLICIES:
            raise ValueError("unsupported sampling_policy")


def _runtime() -> dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as functional
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import (
            AutoTokenizer,
            Qwen2_5OmniThinkerForConditionalGeneration,
            get_scheduler,
            set_seed,
        )
    except (ImportError, RuntimeError) as error:  # pragma: no cover - optional runtime
        raise RuntimeError(
            "post-training requires compatible torch, transformers, and peft"
        ) from error
    return {
        "torch": torch,
        "functional": functional,
        "LoraConfig": LoraConfig,
        "PeftModel": PeftModel,
        "get_peft_model": get_peft_model,
        "AutoTokenizer": AutoTokenizer,
        "QwenModel": Qwen2_5OmniThinkerForConditionalGeneration,
        "get_scheduler": get_scheduler,
        "set_seed": set_seed,
    }


def _length_summary(encoded: Sequence[EncodedSFT | EncodedPreference]) -> dict[str, Any]:
    totals: list[int] = []
    prompts: list[int] = []
    completions: list[int] = []
    length_bias: list[int] = []
    for record in encoded:
        sequences = (
            (record.sequence,)
            if isinstance(record, EncodedSFT)
            else (record.chosen, record.rejected)
        )
        for sequence in sequences:
            totals.append(len(sequence.input_ids))
            prompts.append(sequence.prompt_tokens)
            completions.append(sequence.completion_tokens)
        if isinstance(record, EncodedPreference):
            length_bias.append(record.chosen.completion_tokens - record.rejected.completion_tokens)
    return {
        "total_min": min(totals),
        "total_max": max(totals),
        "total_mean": mean(totals),
        "prompt_min": min(prompts),
        "prompt_max": max(prompts),
        "completion_min": min(completions),
        "completion_max": max(completions),
        "completion_mean": mean(completions),
        "chosen_minus_rejected_completion_mean": mean(length_bias) if length_bias else None,
        "silent_truncation": False,
    }


def _tensor_sequence(sequence: EncodedCompletion, torch: Any, device: Any) -> Any:
    return torch.tensor(sequence.input_ids, dtype=torch.long, device=device).unsqueeze(0)


def _sft_loss(model: Any, sequence: EncodedCompletion, torch: Any, device: Any) -> Any:
    input_ids = _tensor_sequence(sequence, torch, device)
    labels = input_ids.clone()
    labels[:, : sequence.prompt_tokens] = -100
    output = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
        use_cache=False,
    )
    return output.loss


def _average_completion_logp(
    model: Any,
    sequence: EncodedCompletion,
    torch: Any,
    functional: Any,
    device: Any,
) -> Any:
    input_ids = _tensor_sequence(sequence, torch, device)
    output = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=False,
    )
    start = sequence.prompt_tokens - 1
    logits = output.logits[:, start:-1, :].float()
    targets = input_ids[:, sequence.prompt_tokens :]
    if logits.shape[1] != targets.shape[1] or targets.shape[1] != sequence.completion_tokens:
        raise RuntimeError("completion token boundary is inconsistent with shifted logits")
    nll = functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="mean"
    )
    return -nll


def _concatenated_orpo_loss(
    model: Any,
    record: EncodedPreference,
    *,
    beta: float,
    pad_token_id: int,
    torch: Any,
    functional: Any,
    device: Any,
) -> tuple[Any, Any, Any]:
    sequences = (record.chosen, record.rejected)
    maximum = max(len(sequence.input_ids) for sequence in sequences)
    input_ids = torch.full(
        (2, maximum), pad_token_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros((2, maximum), dtype=torch.long, device=device)
    for row_index, sequence in enumerate(sequences):
        length = len(sequence.input_ids)
        input_ids[row_index, :length] = torch.tensor(
            sequence.input_ids, dtype=torch.long, device=device
        )
        attention_mask[row_index, :length] = 1
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logps: list[Any] = []
    for row_index, sequence in enumerate(sequences):
        start = sequence.prompt_tokens - 1
        end = len(sequence.input_ids) - 1
        logits = output.logits[row_index, start:end, :].float()
        targets = input_ids[
            row_index, sequence.prompt_tokens : len(sequence.input_ids)
        ]
        nll = functional.cross_entropy(logits, targets, reduction="mean")
        logps.append(-nll)
    chosen = torch.clamp(logps[0], max=-1e-6)
    rejected = torch.clamp(logps[1], max=-1e-6)
    chosen_odds = chosen - torch.log1p(-torch.exp(chosen))
    rejected_odds = rejected - torch.log1p(-torch.exp(rejected))
    loss = -chosen + beta * functional.softplus(-(chosen_odds - rejected_odds))
    return loss, logps[0], logps[1]


def _log_odds(logp: float) -> float:
    clipped = min(-1e-6, logp)
    return clipped - math.log1p(-math.exp(clipped))


def _softplus(value: float) -> float:
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def _orpo_values(chosen_logp: float, rejected_logp: float, beta: float) -> dict[str, float]:
    chosen = min(-1e-6, chosen_logp)
    rejected = min(-1e-6, rejected_logp)
    log_odds_ratio = _log_odds(chosen) - _log_odds(rejected)
    negative_logsigmoid = _softplus(-log_odds_ratio)
    sigma_negative = (
        math.exp(-log_odds_ratio) / (1.0 + math.exp(-log_odds_ratio))
        if log_odds_ratio >= 0.0
        else 1.0 / (1.0 + math.exp(log_odds_ratio))
    )
    return {
        "loss": -chosen + beta * negative_logsigmoid,
        "chosen_coefficient": -1.0
        - beta * sigma_negative / max(1e-8, 1.0 - math.exp(chosen)),
        "rejected_coefficient": beta
        * sigma_negative
        / max(1e-8, 1.0 - math.exp(rejected)),
        "margin": chosen_logp - rejected_logp,
    }


def _autocast(torch: Any, device: Any, dtype: Any) -> Any:
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=device.type == "cuda" and dtype in {torch.float16, torch.bfloat16},
    )


def _divide_gradients(parameters: Sequence[Any], denominator: float) -> None:
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(denominator)


def _record_transition(record: EncodedSFT | EncodedPreference) -> str | None:
    return record.transition


def _epoch_order(
    records: Sequence[EncodedSFT | EncodedPreference],
    *,
    policy: SamplingPolicy,
    rng: random.Random,
) -> list[int]:
    if policy == "shuffle":
        order = list(range(len(records)))
        rng.shuffle(order)
        return order
    groups: dict[str, list[int]] = {transition: [] for transition in _TRANSITIONS}
    for index, record in enumerate(records):
        transition = _record_transition(record)
        if transition not in groups:
            raise ValueError(
                "transition_balanced sampling requires transition-labelled records"
            )
        groups[transition].append(index)
    missing = sorted(transition for transition, values in groups.items() if not values)
    if missing:
        raise ValueError(
            f"transition_balanced sampling is missing groups: {missing}"
        )
    for values in groups.values():
        rng.shuffle(values)
    maximum = max(len(values) for values in groups.values())
    order: list[int] = []
    for offset in range(maximum):
        round_transitions = list(sorted(groups))
        rng.shuffle(round_transitions)
        for transition in round_transitions:
            values = groups[transition]
            order.append(values[offset % len(values)])
    return order


def _records_per_epoch(
    records: Sequence[EncodedSFT | EncodedPreference], policy: SamplingPolicy
) -> int:
    if policy == "shuffle":
        return len(records)
    counts = Counter(_record_transition(record) for record in records)
    if None in counts or set(counts) != _TRANSITIONS:
        raise ValueError(
            "transition_balanced sampling requires every transition group"
        )
    return len(_TRANSITIONS) * max(counts.values())


def _stratified_preference_sample(
    records: Sequence[EncodedPreference], limit: int, seed: int
) -> list[EncodedPreference]:
    groups: dict[tuple[str, str], list[EncodedPreference]] = defaultdict(list)
    for record in records:
        groups[(record.transition, record.kind)].append(record)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    selected: list[EncodedPreference] = []
    while len(selected) < min(limit, len(records)):
        progressed = False
        for key in sorted(groups):
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop())
                progressed = True
        if not progressed:
            break
    return selected


def _select_orpo_training_rows(
    audited: AuditedRecords,
    *,
    per_session: int,
    constraint_session_fraction: float,
    seed: int,
) -> tuple[AuditedRecords, dict[str, Any]]:
    if audited.stage != "orpo":
        return audited, {"applied": False}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in audited.rows:
        grouped[str(row["session_id"])].append(row)
    hard_count_by_session = {
        session_id: sum(
            row["preference_kind"] in _ACTION_PREFERENCES for row in rows
        )
        for session_id, rows in grouped.items()
    }
    single_pair_layout = all(
        hard_count_by_session[session_id] == 1
        and sum(
            row["preference_kind"] in _CONSTRAINT_PREFERENCES
            for row in rows
        )
        == 3
        for session_id, rows in grouped.items()
    )
    missing_hard = sorted(
        session_id
        for session_id, count in hard_count_by_session.items()
        if count == 0
    )
    if missing_hard:
        raise ValueError(
            f"sessions omit grounded hard preference pairs: {missing_hard}"
        )
    over_budget = {
        session_id: count
        for session_id, count in hard_count_by_session.items()
        if count > per_session
    }
    if over_budget:
        raise ValueError(
            "max_preferences_per_session is smaller than the number of hard "
            f"preference pairs: {over_budget}"
        )
    if max(len(rows) for rows in grouped.values()) <= per_session:
        return audited, {
            "applied": False,
            "policy": "input_already_within_per_session_budget",
            "source_record_count": len(audited.rows),
            "training_record_count": len(audited.rows),
            "maximum_per_session": per_session,
            "constraint_session_fraction": constraint_session_fraction,
            "seed": seed,
            "preference_kind_counts": audited.preference_kind_counts,
            **(
                {"hard_pairs_dropped": 0}
                if not single_pair_layout
                else {}
            ),
        }
    selected: list[dict[str, Any]] = []
    for session_id in sorted(grouped):
        rows = grouped[session_id]
        action = [row for row in rows if row["preference_kind"] in _ACTION_PREFERENCES]
        constraints = [
            row for row in rows if row["preference_kind"] in _CONSTRAINT_PREFERENCES
        ]
        if len(action) != hard_count_by_session[session_id]:
            raise ValueError(f"session {session_id} has unsupported preference kinds")
        selected.extend(action)
        remaining = per_session - len(action)
        digest = hashlib.sha256(f"{seed}:{session_id}".encode()).digest()
        fraction_draw = int.from_bytes(digest[4:8], "big") / (2**32 - 1)
        if (
            remaining > 0
            and constraints
            and fraction_draw < constraint_session_fraction
        ):
            offset = int.from_bytes(digest[:4], "big") % len(constraints)
            rotated = constraints[offset:] + constraints[:offset]
            selected.extend(rotated[:remaining])
    kinds = Counter(str(row["preference_kind"]) for row in selected)
    transitions = Counter(str(row["transition_target"]) for row in selected)
    missing_constraints = _CONSTRAINT_PREFERENCES - set(kinds)
    if missing_constraints:
        for missing_kind in sorted(missing_constraints):
            candidate = next(
                (
                    row
                    for row in audited.rows
                    if row["preference_kind"] == missing_kind
                ),
                None,
            )
            if candidate is None:
                raise ValueError(
                    f"source records omit required constraint kind: {missing_kind}"
                )
            selected.append(candidate)
            kinds[missing_kind] += 1
            transitions[str(candidate["transition_target"])] += 1
    sampled = AuditedRecords(
        stage="orpo",
        dataset=audited.dataset,
        rows=tuple(selected),
        session_count=audited.session_count,
        transition_counts=dict(sorted(transitions.items())),
        preference_kind_counts=dict(sorted(kinds.items())),
    )
    return sampled, {
        "applied": len(selected) != len(audited.rows),
        "policy": (
            "one_action_plus_deterministically_rotated_constraints"
            if single_pair_layout
            else (
                "all_grounded_hard_pairs_plus_deterministically_rotated_constraints"
            )
        ),
        "source_record_count": len(audited.rows),
        "training_record_count": len(selected),
        "maximum_per_session": per_session,
        "constraint_session_fraction": constraint_session_fraction,
        "seed": seed,
        "preference_kind_counts": dict(sorted(kinds.items())),
        **(
            {"hard_pairs_dropped": 0}
            if not single_pair_layout
            else {}
        ),
    }


def _preference_audit(
    model: Any,
    records: Sequence[EncodedPreference],
    *,
    limit: int,
    seed: int,
    torch: Any,
    functional: Any,
    device: Any,
    dtype: Any,
) -> dict[str, Any]:
    sampled = _stratified_preference_sample(records, limit, seed)
    margins: list[float] = []
    by_kind: dict[str, list[float]] = defaultdict(list)
    by_transition: dict[str, list[float]] = defaultdict(list)
    action_by_transition: dict[str, list[float]] = defaultdict(list)
    model.eval()
    with torch.inference_mode():
        for record in sampled:
            with _autocast(torch, device, dtype):
                chosen = float(
                    _average_completion_logp(
                        model, record.chosen, torch, functional, device
                    ).detach().cpu()
                )
                rejected = float(
                    _average_completion_logp(
                        model, record.rejected, torch, functional, device
                    ).detach().cpu()
                )
            margin = chosen - rejected
            margins.append(margin)
            by_kind[record.kind].append(margin)
            by_transition[record.transition].append(margin)
            if record.kind in _ACTION_PREFERENCES:
                action_by_transition[record.transition].append(margin)

    def summarize(values: Sequence[float]) -> dict[str, float | int]:
        return {
            "count": len(values),
            "accuracy": sum(value > 0 for value in values) / len(values),
            "mean_margin": mean(values),
        }

    overall = summarize(margins)
    action_transition_summary = {
        key: summarize(value)
        for key, value in sorted(action_by_transition.items())
    }
    return {
        "sample_count": len(sampled),
        "overall": overall,
        "by_kind": {key: summarize(value) for key, value in sorted(by_kind.items())},
        "by_transition": {
            key: summarize(value) for key, value in sorted(by_transition.items())
        },
        "action_by_transition": action_transition_summary,
        "gate": _preference_audit_gate(overall, action_transition_summary),
    }


def _preference_audit_gate(
    overall: Mapping[str, float | int],
    action_by_transition: Mapping[str, Mapping[str, float | int]],
    *,
    minimum_overall_accuracy: float = 0.60,
    minimum_action_transition_accuracy: float = 0.60,
) -> dict[str, Any]:
    """Fail closed when aggregate preference accuracy hides action collapse."""

    missing = sorted(_TRANSITIONS - set(action_by_transition))
    failing = sorted(
        transition
        for transition, metrics in action_by_transition.items()
        if float(metrics["accuracy"]) < minimum_action_transition_accuracy
    )
    overall_passed = float(overall["accuracy"]) >= minimum_overall_accuracy
    return {
        "minimum_overall_accuracy": minimum_overall_accuracy,
        "minimum_action_transition_accuracy": minimum_action_transition_accuracy,
        "missing_action_transitions": missing,
        "failing_action_transitions": failing,
        "overall_passed": overall_passed,
        "passed": overall_passed and not missing and not failing,
    }


def train_stage(request: TrainingRequest) -> dict[str, Any]:
    """Train one immutable stage and return its auditable run summary."""

    output_dir = request.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite post-training run: {output_dir}")
    audited = audit_training_records(request.stage, request.records_path)
    source_audit = audited.summary()
    audited, preference_sampling = _select_orpo_training_rows(
        audited,
        per_session=request.max_preferences_per_session,
        constraint_session_fraction=request.constraint_session_fraction,
        seed=request.seed,
    )
    transition_multipliers, transition_weighting = _transition_weighting_plan(
        audited,
        policy=request.transition_weighting,
        max_effective_sample_weight=request.max_effective_sample_weight,
    )
    runtime = _runtime()
    torch = runtime["torch"]
    runtime["set_seed"](request.seed)
    random.seed(request.seed)
    device = torch.device(request.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if not hasattr(torch, request.dtype):
        raise ValueError(f"unsupported torch dtype: {request.dtype}")
    torch_dtype = getattr(torch, request.dtype)
    tokenizer = runtime["AutoTokenizer"].from_pretrained(
        str(request.model_path), trust_remote_code=True
    )
    encoded = _encode_records(
        audited,
        tokenizer,
        max_tokens=request.max_tokens,
        transition_multipliers=transition_multipliers,
        max_effective_sample_weight=request.max_effective_sample_weight,
    )

    base_model = load_pretrained_model_on_device(
        runtime["QwenModel"],
        request.model_path,
        device=str(device),
        torch_dtype=torch_dtype,
        attn_implementation=request.attention_implementation,
        low_cpu_mem_usage=True,
    )
    base_model.config.use_cache = False
    if request.gradient_checkpointing:
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        base_model.enable_input_require_grads()
    if request.init_adapter is None:
        lora = runtime["LoraConfig"](
            r=request.lora_rank,
            lora_alpha=request.lora_alpha,
            lora_dropout=request.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TEXT_LORA_TARGETS,
        )
        model = runtime["get_peft_model"](base_model, lora)
        initialization = "new_text_path_lora"
    else:
        adapter = request.init_adapter.expanduser().resolve()
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"initial adapter is incomplete: {adapter}")
        model = runtime["PeftModel"].from_pretrained(
            base_model, str(adapter), is_trainable=True
        )
        initialization = f"continued_adapter:{adapter}"
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("post-training adapter has no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=request.learning_rate,
        weight_decay=request.weight_decay,
    )
    records_per_epoch = _records_per_epoch(encoded, request.sampling_policy)
    updates_per_epoch = math.ceil(records_per_epoch / request.gradient_accumulation)
    total_updates = updates_per_epoch * request.epochs
    scheduler = runtime["get_scheduler"](
        request.scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(round(total_updates * request.warmup_ratio)),
        num_training_steps=total_updates,
    )

    output_dir.mkdir(parents=True)
    started = time.time()
    history: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    rng = random.Random(request.seed)
    for epoch in range(1, request.epochs + 1):
        order = _epoch_order(encoded, policy=request.sampling_policy, rng=rng)
        running_loss = 0.0
        running_weight = 0.0
        margins: list[float] = []
        accumulated_weight = 0.0
        records_since_update = 0
        model.train()
        for position, record_index in enumerate(order, 1):
            record = encoded[record_index]
            if isinstance(record, EncodedSFT):
                with _autocast(torch, device, torch_dtype):
                    loss = _sft_loss(model, record.sequence, torch, device)
                (loss * record.weight).backward()
                loss_value = float(loss.detach().cpu())
            else:
                if request.orpo_forward_mode == "concatenated":
                    pad_token_id = tokenizer.pad_token_id
                    if pad_token_id is None:
                        pad_token_id = tokenizer.eos_token_id
                    if pad_token_id is None:
                        raise ValueError("tokenizer must provide a pad or EOS token")
                    with _autocast(torch, device, torch_dtype):
                        loss, chosen_logp, rejected_logp = _concatenated_orpo_loss(
                            model,
                            record,
                            beta=request.orpo_beta,
                            pad_token_id=int(pad_token_id),
                            torch=torch,
                            functional=runtime["functional"],
                            device=device,
                        )
                    (loss * record.weight).backward()
                    chosen_detached = float(chosen_logp.detach().cpu())
                    rejected_detached = float(rejected_logp.detach().cpu())
                    loss_value = float(loss.detach().cpu())
                    margins.append(chosen_detached - rejected_detached)
                    del loss, chosen_logp, rejected_logp
                else:
                    with torch.inference_mode():
                        with _autocast(torch, device, torch_dtype):
                            chosen_detached = float(
                                _average_completion_logp(
                                    model,
                                    record.chosen,
                                    torch,
                                    runtime["functional"],
                                    device,
                                )
                                .detach()
                                .cpu()
                            )
                            rejected_detached = float(
                                _average_completion_logp(
                                    model,
                                    record.rejected,
                                    torch,
                                    runtime["functional"],
                                    device,
                                )
                                .detach()
                                .cpu()
                            )
                    values = _orpo_values(
                        chosen_detached, rejected_detached, request.orpo_beta
                    )
                    with _autocast(torch, device, torch_dtype):
                        chosen_logp = _average_completion_logp(
                            model,
                            record.chosen,
                            torch,
                            runtime["functional"],
                            device,
                        )
                    (
                        chosen_logp
                        * values["chosen_coefficient"]
                        * record.weight
                    ).backward()
                    del chosen_logp
                    with _autocast(torch, device, torch_dtype):
                        rejected_logp = _average_completion_logp(
                            model,
                            record.rejected,
                            torch,
                            runtime["functional"],
                            device,
                        )
                    (
                        rejected_logp
                        * values["rejected_coefficient"]
                        * record.weight
                    ).backward()
                    del rejected_logp
                    loss_value = values["loss"]
                    margins.append(values["margin"])
            running_loss += loss_value * record.weight
            running_weight += record.weight
            accumulated_weight += record.weight
            records_since_update += 1
            final_record = position == len(order)
            if records_since_update == request.gradient_accumulation or final_record:
                _divide_gradients(parameters, accumulated_weight)
                torch.nn.utils.clip_grad_norm_(parameters, request.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated_weight = 0.0
                records_since_update = 0
        epoch_summary = {
            "epoch": epoch,
            "weighted_train_loss": running_loss / running_weight,
            "preference_accuracy_before_updates": (
                sum(value > 0 for value in margins) / len(margins) if margins else None
            ),
            "mean_preference_margin_before_updates": mean(margins) if margins else None,
            "elapsed_seconds": time.time() - started,
        }
        history.append(epoch_summary)
        print(json.dumps(epoch_summary, ensure_ascii=False, sort_keys=True), flush=True)

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    preference_audit = None
    if request.stage in {"action_orpo", "orpo"}:
        preference_audit = _preference_audit(
            model,
            [record for record in encoded if isinstance(record, EncodedPreference)],
            limit=request.preference_audit_limit,
            seed=request.seed,
            torch=torch,
            functional=runtime["functional"],
            device=device,
            dtype=torch_dtype,
        )
    summary = {
        "experiment": request.experiment_name,
        "stage": request.stage,
        "dataset": audited.dataset,
        "records": {
            "path": str(request.records_path.resolve()),
            "sha256": _sha256_file(request.records_path),
            **audited.summary(),
            "source_audit": source_audit,
            "preference_sampling": preference_sampling,
        },
        "model": {
            "base_model": str(request.model_path.resolve()),
            "initialization": initialization,
            "adapter_dir": str(adapter_dir),
            "text_lora_targets": TEXT_LORA_TARGETS,
        },
        "optimization": {
            "epochs": request.epochs,
            "learning_rate": request.learning_rate,
            "weight_decay": request.weight_decay,
            "warmup_ratio": request.warmup_ratio,
            "scheduler_type": request.scheduler_type,
            "gradient_accumulation": request.gradient_accumulation,
            "max_grad_norm": request.max_grad_norm,
            "orpo_beta": (
                request.orpo_beta
                if request.stage in {"action_orpo", "orpo"}
                else None
            ),
            "orpo_forward_mode": (
                request.orpo_forward_mode
                if request.stage in {"action_orpo", "orpo"}
                else None
            ),
            "seed": request.seed,
            "transition_weighting": transition_weighting,
            "sampling_policy": request.sampling_policy,
            "records_per_epoch_after_sampling": records_per_epoch,
        },
        "token_lengths": {
            **_length_summary(encoded),
            "fixed_limit": request.max_tokens,
        },
        "history": history,
        "preference_audit": preference_audit,
        "fit_boundaries": {
            "single_dataset": True,
            "cross_dataset_merging": False,
            "raw_labels_or_evidence_read_by_trainer": False,
            "ground_truth_in_model_prompt": False,
            "silent_token_truncation": False,
        },
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train text-only Draft/Evidence-Literacy/Revision SFT or "
            "revision-aware ORPO."
        )
    )
    parser.add_argument("--stage", required=True, choices=sorted(_STAGES))
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--init-adapter", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument("--max-tokens", type=int, default=7500)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--scheduler-type", default="linear")
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--orpo-beta", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preference-audit-limit", type=int, default=96)
    parser.add_argument(
        "--orpo-forward-mode",
        choices=("concatenated", "streaming"),
        default="concatenated",
    )
    parser.add_argument("--max-preferences-per-session", type=int, default=2)
    parser.add_argument("--constraint-session-fraction", type=float, default=0.20)
    parser.add_argument(
        "--transition-weighting",
        choices=sorted(_TRANSITION_WEIGHTING_POLICIES),
        default="none",
    )
    parser.add_argument("--max-effective-sample-weight", type=float, default=10.0)
    parser.add_argument(
        "--experiment-name", default="rethink_revision_post_training_v1"
    )
    parser.add_argument(
        "--sampling-policy", choices=sorted(_SAMPLING_POLICIES), default="shuffle"
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
    )
    parser.set_defaults(gradient_checkpointing=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.audit_only:
        summary = audit_training_records(args.stage, args.records).summary()
    else:
        if args.model_path is None or args.output_dir is None:
            raise ValueError("--model-path and --output-dir are required for training")
        summary = train_stage(
            TrainingRequest(
                stage=args.stage,
                records_path=args.records,
                model_path=args.model_path,
                output_dir=args.output_dir,
                init_adapter=args.init_adapter,
                device=args.device,
                dtype=args.dtype,
                attention_implementation=args.attention_implementation,
                max_tokens=args.max_tokens,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                warmup_ratio=args.warmup_ratio,
                scheduler_type=args.scheduler_type,
                gradient_accumulation=args.gradient_accumulation,
                max_grad_norm=args.max_grad_norm,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                orpo_beta=args.orpo_beta,
                seed=args.seed,
                gradient_checkpointing=args.gradient_checkpointing,
                preference_audit_limit=args.preference_audit_limit,
                orpo_forward_mode=args.orpo_forward_mode,
                max_preferences_per_session=args.max_preferences_per_session,
                constraint_session_fraction=args.constraint_session_fraction,
                transition_weighting=args.transition_weighting,
                max_effective_sample_weight=args.max_effective_sample_weight,
                experiment_name=args.experiment_name,
                sampling_policy=args.sampling_policy,
            )
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
