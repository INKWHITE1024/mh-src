"""Build outcome-annotated Revision-SFT/ORPO records from frozen rollout arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rethink_mh.rethinking.contracts import (
    InitialAssessment,
    RevisionAssessment,
    TaskSpec,
)
from rethink_mh.rethinking.native_query_retrieval import (
    NativeAtomicRetriever,
)
from rethink_mh.rethinking.prompts import PromptBuilder
from rethink_mh.rethinking.query_retrieval import (
    AtomicSelection,
    EvidenceQuery,
)
from rethink_mh.rethinking.retrieval import TargetedEvidence

from .frozen_artifact_manifest import verify_directory_manifest
from .rethink_post_training import audit_training_records
from .loop_evaluate import (
    ALL_ARMS,
    DIRECT_REFER_ARM,
    EVALUATION_PROTOCOL_VERSION,
)
from .loop_collect import (
    COLLECTION_PROTOCOL_VERSION,
)
from .loop_merge import (
    MERGE_PROTOCOL_VERSION,
)
from .loop_prepare import PREPARATION_PROTOCOL_VERSION


PACKAGE_PROTOCOL_VERSION = "loop-training-package"
_SOURCE_PROTOCOLS = {
    COLLECTION_PROTOCOL_VERSION: {
        "merge": MERGE_PROTOCOL_VERSION,
        "evaluation": EVALUATION_PROTOCOL_VERSION,
        "package": PACKAGE_PROTOCOL_VERSION,
    },
}
_TRANSITIONS = ("CC_preserve", "WC_revise", "WW_refer")
_CONSTRAINT_KINDS = (
    "constraint_missing_required_field",
    "constraint_out_of_scope_clinical_claim",
    "constraint_unknown_evidence_id",
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


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be one object: {path}")
    return value


def _read_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read {description} {path}: {error}") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise ValueError(f"blank row in {description}: {path}:{line_number}")
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
        rows.append(value)
    if not rows:
        raise ValueError(f"{description} is empty: {path}")
    return rows


def _task(path: Path) -> TaskSpec:
    return TaskSpec.from_dict(_read_json(path, "task specification"))


def _assistant_messages(
    prompt: Sequence[Mapping[str, str]],
    completion: Mapping[str, Any],
) -> list[dict[str, str]]:
    return [
        *(dict(message) for message in prompt),
        {"role": "assistant", "content": _canonical_json(completion)},
    ]


def _reflection_context(
    task: TaskSpec,
    initial: InitialAssessment,
    segments: Sequence[str],
) -> tuple[list[dict[str, str]], TargetedEvidence]:
    selected = tuple(dict.fromkeys(str(item) for item in segments))
    if not selected:
        raise ValueError("reflection context requires a grounded segment")
    decision = {
        "should_rethink": True,
        "trigger_score": 1.0,
        "reasons": ["outcome_supervised_training_context"],
        "selected_segment_ids": list(selected),
        "metrics": {"new_atomic_evidence_released": 0},
    }
    text = (
        "NO NEW ATOMIC EVIDENCE | reflection control\n"
        "- No hidden measurement was queried or released in this arm.\n"
        "- Reconsider only the already visible InitialAssessment and its "
        "segment references.\n"
        "- visible segment identifiers: "
        + ", ".join(selected)
        + "\n"
    )
    targeted = TargetedEvidence(
        text=text,
        selected_segment_ids=selected,
        selected_atomic_evidence_ids=(),
        canonical_evidence_ids=(),
    )
    prompt = PromptBuilder.build_reflection_messages(
        task,
        initial,
        decision,
        targeted.text,
    )
    return prompt, targeted


def _release_context(
    *,
    task: TaskSpec,
    initial: InitialAssessment,
    arm: Mapping[str, Any],
    session_dir: Path,
) -> tuple[list[dict[str, str]], TargetedEvidence]:
    workflow = arm.get("workflow")
    if not isinstance(workflow, Mapping):
        raise ValueError("release context requires a successful workflow")
    rounds = workflow.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != 1:
        raise ValueError("release workflow must contain exactly one round")
    round_record = rounds[0]
    query = EvidenceQuery.from_dict(round_record["evidence_query"])
    selection = AtomicSelection.from_dict(round_record["atomic_selection"])
    retriever = NativeAtomicRetriever.from_session_dir(session_dir)
    candidates = retriever.shortlist(query)
    if candidates.to_dict() != round_record["candidate_set"]:
        raise ValueError("reconstructed candidate set changed after raw freeze")
    targeted = retriever.materialize(candidates, selection)
    release = round_record["release"]
    if (
        list(targeted.selected_segment_ids)
        != release.get("selected_segment_ids")
        or list(targeted.selected_atomic_evidence_ids)
        != release.get("selected_atomic_evidence_ids")
        or list(targeted.canonical_evidence_ids)
        != release.get("canonical_evidence_ids")
    ):
        raise ValueError("reconstructed atomic release changed after raw freeze")
    prompt = PromptBuilder.build_revision_messages(
        task,
        initial,
        workflow["rethink_decision"],
        targeted.text,
    )
    return prompt, targeted


def _arm_index(raw: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    arms = raw.get("arms")
    if not isinstance(arms, list):
        return {}
    return {
        str(arm["arm_id"]): arm
        for arm in arms
        if isinstance(arm, Mapping)
    }


def _score_index(decision: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    scores = decision.get("arm_scores")
    if not isinstance(scores, list):
        raise ValueError("training decision omitted arm scores")
    indexed = {
        str(score.get("arm_id", "")): score
        for score in scores
        if isinstance(score, Mapping)
    }
    if set(indexed) != set(ALL_ARMS):
        raise ValueError("training decision has incomplete arm scores")
    return indexed


def _best_score(
    scores: Mapping[str, Mapping[str, Any]],
    predicate: Any,
) -> Mapping[str, Any] | None:
    eligible = [
        score
        for arm_id, score in scores.items()
        if arm_id != DIRECT_REFER_ARM
        and score.get("arm_valid") is True
        and predicate(score)
    ]
    if not eligible:
        return None
    order = {arm_id: index for index, arm_id in enumerate(ALL_ARMS)}
    return sorted(
        eligible,
        key=lambda score: (
            -float(score["primary_utility"]),
            bool(score["referred"]),
            int(score["released_tokens"]),
            int(score["atomic_records"]),
            order[str(score["arm_id"])],
        ),
    )[0]


def _context_for_arm(
    *,
    arm_id: str,
    raw: Mapping[str, Any],
    plan: Mapping[str, Any],
    task: TaskSpec,
    initial: InitialAssessment,
) -> tuple[list[dict[str, str]], TargetedEvidence, str]:
    arms = _arm_index(raw)
    if arm_id == "reflection_no_new_evidence":
        prompt, targeted = _reflection_context(
            task,
            initial,
            raw["collection_segment_ids"],
        )
        return prompt, targeted, "frozen_reflection_context"
    arm = arms.get(arm_id)
    if arm is not None and arm.get("status") == "ok":
        prompt, targeted = _release_context(
            task=task,
            initial=initial,
            arm=arm,
            session_dir=Path(str(plan["evidence"]["session_dir"])),
        )
        return prompt, targeted, f"frozen_raw_arm:{arm_id}"
    prompt, targeted = _reflection_context(
        task,
        initial,
        raw["collection_segment_ids"],
    )
    return prompt, targeted, "grounded_reflection_fallback"


def _evidence_lists(
    targeted: TargetedEvidence,
) -> tuple[list[str], list[str]]:
    segments = list(targeted.selected_segment_ids)
    evidence = list(targeted.selected_atomic_evidence_ids)
    if not segments:
        raise ValueError("training target has no grounded segment")
    return segments, evidence


def _target_revision(
    *,
    transition: str,
    initial: InitialAssessment,
    targeted: TargetedEvidence,
    recovered_probability: float | None,
    recovered_confidence: float | None,
) -> RevisionAssessment:
    segments, evidence = _evidence_lists(targeted)
    if transition == "CC_preserve":
        payload = {
            "revision_status": "preserved",
            "revised_risk_probability": initial.risk_probability,
            "revised_confidence": max(initial.confidence, 0.65),
            "cited_segment_ids": segments,
            "cited_evidence_ids": evidence,
            "preserved_evidence_ids": evidence,
            "newly_considered_evidence_ids": evidence,
            "rejected_evidence_ids": [],
            "residual_conflict_segment_ids": [],
            "change_summary": "risk_unchanged_after_review",
        }
    elif transition == "WC_revise":
        if recovered_probability is None or recovered_confidence is None:
            raise ValueError("WC target requires a recovered raw probability")
        direction = (
            "risk_increased_after_review"
            if recovered_probability > initial.risk_probability
            else "risk_decreased_after_review"
        )
        payload = {
            "revision_status": "revised",
            "revised_risk_probability": recovered_probability,
            "revised_confidence": recovered_confidence,
            "cited_segment_ids": segments,
            "cited_evidence_ids": evidence,
            "preserved_evidence_ids": evidence,
            "newly_considered_evidence_ids": evidence,
            "rejected_evidence_ids": [],
            "residual_conflict_segment_ids": [],
            "change_summary": direction,
        }
    elif transition == "WW_refer":
        payload = {
            "revision_status": "unresolved",
            "revised_risk_probability": initial.risk_probability,
            "revised_confidence": min(initial.confidence, 0.30),
            "cited_segment_ids": segments,
            "cited_evidence_ids": evidence,
            "preserved_evidence_ids": evidence,
            "newly_considered_evidence_ids": evidence,
            "rejected_evidence_ids": [],
            "residual_conflict_segment_ids": segments,
            "change_summary": "insufficient_reliable_detail",
        }
    else:
        raise ValueError(f"unsupported training transition: {transition}")
    return RevisionAssessment.from_dict(payload)


def _opposite_probability(label: int, threshold: float) -> float:
    if label == 1:
        return max(0.0, threshold - max(0.05, threshold * 0.25))
    return min(1.0, threshold + max(0.05, (1.0 - threshold) * 0.25))


def _action_negative(
    *,
    transition: str,
    label: int,
    threshold: float,
    initial: InitialAssessment,
    targeted: TargetedEvidence,
) -> tuple[RevisionAssessment, str]:
    segments, evidence = _evidence_lists(targeted)
    common = {
        "cited_segment_ids": segments,
        "cited_evidence_ids": evidence,
        "preserved_evidence_ids": evidence,
        "newly_considered_evidence_ids": evidence,
        "rejected_evidence_ids": [],
        "residual_conflict_segment_ids": [],
    }
    if transition == "CC_preserve":
        probability = _opposite_probability(label, threshold)
        payload = {
            **common,
            "revision_status": "revised",
            "revised_risk_probability": probability,
            "revised_confidence": 0.90,
            "change_summary": (
                "risk_increased_after_review"
                if probability > initial.risk_probability
                else "risk_decreased_after_review"
            ),
        }
        kind = "CW_harmful_revision"
    elif transition == "WC_revise":
        payload = {
            **common,
            "revision_status": "preserved",
            "revised_risk_probability": initial.risk_probability,
            "revised_confidence": max(initial.confidence, 0.70),
            "change_summary": "risk_unchanged_after_review",
        }
        kind = "WW_failed_preservation"
    else:
        probability = min(1.0, initial.risk_probability + 0.10)
        if math.isclose(probability, initial.risk_probability):
            probability = max(0.0, initial.risk_probability - 0.10)
        payload = {
            **common,
            "revision_status": "revised",
            "revised_risk_probability": probability,
            "revised_confidence": 0.95,
            "change_summary": (
                "risk_increased_after_review"
                if probability > initial.risk_probability
                else "risk_decreased_after_review"
            ),
        }
        kind = "WW_forced_confident_prediction"
    return RevisionAssessment.from_dict(payload), kind


def _constraint_negative(
    chosen: Mapping[str, Any],
    kind: str,
) -> dict[str, Any]:
    negative = json.loads(json.dumps(chosen))
    if kind == "constraint_missing_required_field":
        negative.pop("change_summary")
    elif kind == "constraint_out_of_scope_clinical_claim":
        negative["clinical_diagnosis"] = (
            "definitive diagnosis from screening evidence"
        )
    elif kind == "constraint_unknown_evidence_id":
        negative["cited_evidence_ids"] = [
            "E999999",
            *negative["cited_evidence_ids"],
        ]
        negative["newly_considered_evidence_ids"] = [
            "E999999",
            *negative["newly_considered_evidence_ids"],
        ]
    else:
        raise ValueError(f"unknown constraint negative: {kind}")
    return negative


@dataclass(frozen=True, slots=True)
class LoopTrainingPackageRequest:
    raw_oof_dir: Path
    raw_artifact_manifest: Path
    prepared_root: Path
    prepared_artifact_manifest: Path
    evaluation_dir: Path
    config_path: Path
    task_path: Path
    output_dir: Path
    heldout_fold: int | None = None

    def __post_init__(self) -> None:
        if self.heldout_fold is not None and (
            isinstance(self.heldout_fold, bool)
            or not isinstance(self.heldout_fold, int)
            or self.heldout_fold < 0
        ):
            raise ValueError("heldout_fold must be a non-negative integer")


def _verified_sources(
    request: LoopTrainingPackageRequest,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    str,
]:
    raw_frozen = verify_directory_manifest(
        request.raw_oof_dir,
        _read_json(request.raw_artifact_manifest, "raw artifact manifest"),
    )
    prepared_frozen = verify_directory_manifest(
        request.prepared_root,
        _read_json(
            request.prepared_artifact_manifest,
            "prepared artifact manifest",
        ),
    )
    raw_manifest = _read_json(
        request.raw_oof_dir / "manifest.json",
        "raw OOF manifest",
    )
    evaluation = _read_json(
        request.evaluation_dir / "summary.json",
        "offline evaluation summary",
    )
    decisions_path = request.evaluation_dir / "training_decisions.jsonl"
    inference_manifest = _read_json(
        request.prepared_root / "inference/manifest.json",
        "prepared inference manifest",
    )
    collection_protocol_version = raw_manifest.get(
        "collection_protocol_version"
    )
    expected = _SOURCE_PROTOCOLS.get(collection_protocol_version)
    if any(
        (
            expected is None,
            raw_manifest.get("merge_protocol_version")
            != (expected or {}).get("merge"),
            inference_manifest.get("preparation_protocol_version")
            != PREPARATION_PROTOCOL_VERSION,
            evaluation.get("evaluation_protocol_version")
            != (expected or {}).get("evaluation"),
            evaluation.get("collection_protocol_version")
            != collection_protocol_version,
            evaluation.get("files", {})
            .get("training_decisions.jsonl", {})
            .get("sha256")
            != _sha256(decisions_path),
            evaluation.get("sources", {}).get("raw_artifact_sha256")
            != raw_frozen["artifact_sha256"],
            evaluation.get("sources", {}).get("prepared_artifact_sha256")
            != prepared_frozen["artifact_sha256"],
            evaluation.get("sources", {}).get("config_sha256")
            != _sha256(request.config_path),
        )
    ):
        raise ValueError("training package sources failed frozen verification")
    raw_rows = _read_jsonl(
        request.raw_oof_dir / "trajectories.raw.jsonl",
        "raw OOF trajectories",
    )
    plan_rows = _read_jsonl(
        request.prepared_root / "inference/inference_plan.jsonl",
        "inference plan",
    )
    decisions = _read_jsonl(decisions_path, "training decisions")
    return (
        raw_rows,
        plan_rows,
        decisions,
        raw_frozen,
        prepared_frozen,
        str(collection_protocol_version),
    )


def build_training_package(
    request: LoopTrainingPackageRequest,
) -> dict[str, Any]:
    """Build audited revision SFT and ORPO records."""

    if request.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite loop training package: {request.output_dir}"
        )
    (
        raw_rows,
        plan_rows,
        decisions,
        raw_frozen,
        prepared_frozen,
        collection_protocol_version,
    ) = _verified_sources(request)
    task = _task(request.task_path)
    config = yaml.safe_load(request.config_path.read_text(encoding="utf-8"))
    if (
        not isinstance(config, Mapping)
        or config.get("protocol_version") != "full-loop"
    ):
        raise ValueError("training package config is invalid")
    false_negative_weight = float(
        config["trajectory_selection"]["false_negative_weight"]
    )
    if not math.isfinite(false_negative_weight) or false_negative_weight < 1.0:
        raise ValueError("false_negative_weight must be finite and at least one")

    raw_by_id = {str(row["session_id"]): row for row in raw_rows}
    plan_by_id = {str(row["session_id"]): row for row in plan_rows}
    decision_by_id = {str(row["session_id"]): row for row in decisions}
    if (
        len(raw_by_id) != len(raw_rows)
        or len(plan_by_id) != len(plan_rows)
        or len(decision_by_id) != len(decisions)
        or set(raw_by_id) != set(plan_by_id)
        or set(raw_by_id) != set(decision_by_id)
    ):
        raise ValueError("training-package source cohorts differ")

    loop_rows: list[dict[str, Any]] = []
    preference_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    transitions: Counter[str] = Counter()
    context_arms: Counter[str] = Counter()
    heldout_session_ids: list[str] = []
    for session_id in sorted(raw_by_id):
        raw = raw_by_id[session_id]
        plan = plan_by_id[session_id]
        decision = decision_by_id[session_id]
        if decision.get("used_in_model_text") is not False:
            raise ValueError(f"unsafe training decision: {session_id}")
        if (
            request.heldout_fold is not None
            and raw.get("outer_fold") == request.heldout_fold
        ):
            heldout_session_ids.append(session_id)
            excluded_rows.append(
                {
                    "session_id": session_id,
                    "outer_fold": request.heldout_fold,
                    "reason": "crossfit_holdout",
                }
            )
            continue
        if decision.get("initial_assessment_failed") is True:
            excluded_rows.append(
                {
                    "session_id": session_id,
                    "reason": "initial_assessment_failed",
                }
            )
            continue
        label = int(decision["label"])
        threshold = float(decision["initial_threshold"])
        initial = InitialAssessment.from_dict(
            raw["initial"]["initial_assessment"]
        )
        scores = _score_index(decision)
        initial_correct = bool(decision["initial_correct"])
        if initial_correct:
            selected = _best_score(
                scores,
                lambda score: (
                    not bool(score["referred"])
                    and score["transition"] == "CC_preserved"
                ),
            )
            transition = "CC_preserve"
            recovered_probability = None
            recovered_confidence = None
        else:
            selected = _best_score(
                scores,
                lambda score: (
                    not bool(score["referred"])
                    and score["transition"] == "WC_recovered"
                ),
            )
            if selected is not None:
                transition = "WC_revise"
                recovered_probability = float(selected["final_probability"])
                recovered_confidence = float(selected["final_confidence"])
                if abs(recovered_probability - initial.risk_probability) < 0.05:
                    recovered_probability = min(
                        1.0,
                        initial.risk_probability + 0.05,
                    ) if label == 1 else max(
                        0.0,
                        initial.risk_probability - 0.05,
                    )
                if int(recovered_probability >= threshold) != label:
                    raise ValueError(
                        f"normalized WC target is not recovered: {session_id}"
                    )
            else:
                transition = "WW_refer"
                selected = _best_score(
                    scores,
                    lambda score: True,
                )
                recovered_probability = None
                recovered_confidence = None
        selected_arm = (
            str(selected["arm_id"])
            if selected is not None
            else "reflection_no_new_evidence"
        )
        prompt, targeted, context_origin = _context_for_arm(
            arm_id=selected_arm,
            raw=raw,
            plan=plan,
            task=task,
            initial=initial,
        )
        chosen = _target_revision(
            transition=transition,
            initial=initial,
            targeted=targeted,
            recovered_probability=recovered_probability,
            recovered_confidence=recovered_confidence,
        )
        rejected, rejected_kind = _action_negative(
            transition=transition,
            label=label,
            threshold=threshold,
            initial=initial,
            targeted=targeted,
        )
        chosen_dict = chosen.to_dict()
        chosen_text = _canonical_json(chosen_dict)
        sample_weight = (
            false_negative_weight
            if label == 1 and initial.risk_probability < threshold
            else 1.0
        )
        loop_rows.append(
            {
                "record_type": "revision_sft",
                "dataset": raw["dataset"],
                "session_id": session_id,
                "outer_fold": raw["outer_fold"],
                "task_id": task.task_id,
                "messages": _assistant_messages(prompt, chosen_dict),
                "transition_target": transition,
                "sample_weight": sample_weight,
                "supervision": {
                    "label": label,
                    "used_in_model_text": False,
                },
            }
        )
        base_preference = {
            "record_type": "revision_preference",
            "dataset": raw["dataset"],
            "session_id": session_id,
            "outer_fold": raw["outer_fold"],
            "task_id": task.task_id,
            "prompt_messages": [dict(message) for message in prompt],
            "chosen": chosen_text,
            "transition_target": transition,
            "sample_weight": sample_weight,
            "ground_truth_in_prompt": False,
        }
        preference_rows.append(
            {
                **base_preference,
                "rejected": _canonical_json(rejected.to_dict()),
                "preference_kind": rejected_kind,
            }
        )
        for kind in _CONSTRAINT_KINDS:
            preference_rows.append(
                {
                    **base_preference,
                    "rejected": _canonical_json(
                        _constraint_negative(chosen_dict, kind)
                    ),
                    "preference_kind": kind,
                }
            )
        trajectory_rows.append(
            {
                "schema_version": "1.0.0",
                "record_type": "native_outcome_supervised_trajectory",
                "dataset": raw["dataset"],
                "session_id": session_id,
                "outer_fold": raw["outer_fold"],
                "initial_assessment": initial.to_dict(),
                "initial_threshold": threshold,
                "selected_segment_ids": list(
                    targeted.selected_segment_ids
                ),
                "selected_evidence_ids": list(
                    targeted.selected_atomic_evidence_ids
                ),
                "context_arm": selected_arm,
                "context_origin": context_origin,
                "offline_primary_selected_arm": decision[
                    "selected_training_arm"
                ],
                "transition_target": transition,
                "chosen_revision": chosen_dict,
                "supervision": {
                    "label": label,
                    "used_in_model_text": False,
                },
            }
        )
        transitions[transition] += 1
        context_arms[selected_arm] += 1

    missing = set(_TRANSITIONS) - set(transitions)
    if missing:
        raise ValueError(
            "outcome-supervised package lacks required transitions; "
            f"training remains blocked: {sorted(missing)}"
        )
    request.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{request.output_dir.name}.",
            dir=str(request.output_dir.parent),
        )
    )
    try:
        collections = {
            "revision_sft.jsonl": loop_rows,
            "preferences_initial.jsonl": preference_rows,
            "trajectories.jsonl": trajectory_rows,
            "excluded.jsonl": excluded_rows,
        }
        files: dict[str, dict[str, Any]] = {}
        for filename, rows in collections.items():
            text = "".join(_canonical_json(row) + "\n" for row in rows)
            path = temporary / filename
            path.write_text(text, encoding="utf-8")
            files[filename] = {
                "sha256": _sha256(path),
                "row_count": len(rows),
            }
        audit_training_records("revision_sft", temporary / "revision_sft.jsonl")
        audit_training_records("orpo", temporary / "preferences_initial.jsonl")
        manifest = {
            "schema_version": "1.0.0",
            "package_protocol_version": PACKAGE_PROTOCOL_VERSION,
            "collection_protocol_version": collection_protocol_version,
            "dataset": raw_rows[0]["dataset"],
            "task": task.to_dict(),
            "source_session_count": len(raw_rows),
            "training_session_count": len(loop_rows),
            "excluded_session_count": len(excluded_rows),
            "crossfit": {
                "enabled": request.heldout_fold is not None,
                "heldout_fold": request.heldout_fold,
                "heldout_session_ids": heldout_session_ids,
                "heldout_session_count": len(heldout_session_ids),
                "heldout_outcomes_used_for_training": False,
                "training_fold_indices": sorted(
                    {
                        int(row["outer_fold"])
                        for row in raw_rows
                        if row["outer_fold"] != request.heldout_fold
                    }
                ),
            },
            "transition_counts": dict(sorted(transitions.items())),
            "context_arm_counts": dict(sorted(context_arms.items())),
            "preference_kind_counts": dict(
                sorted(
                    Counter(
                        row["preference_kind"] for row in preference_rows
                    ).items()
                )
            ),
            "files": files,
            "sources": {
                "raw_artifact_sha256": raw_frozen["artifact_sha256"],
                "prepared_artifact_sha256": prepared_frozen[
                    "artifact_sha256"
                ],
                "evaluation_summary_sha256": _sha256(
                    request.evaluation_dir / "summary.json"
                ),
                "training_decisions_sha256": _sha256(
                    request.evaluation_dir / "training_decisions.jsonl"
                ),
                "config_sha256": _sha256(request.config_path),
                "task_sha256": _sha256(request.task_path),
                "utility_config_protocol_version": config[
                    "protocol_version"
                ],
            },
            "fit_boundaries": {
                "raw_trajectories_frozen_before_outcome_join": True,
                "initial_predictions_are_out_of_fold": True,
                "outcomes_used_for_training_targets": True,
                "gold_label_in_model_messages": False,
                "current_107_are_now_training_material": True,
                "same_107_not_unbiased_for_loop_trained_evaluation": True,
                "crossfit_holdout_outcomes_used_for_training": False,
                "utility_inherited_unchanged_from_preregistered_config": True,
                "dev_accessed": False,
                "test_accessed": False,
            },
            "interpretation_boundary": (
                "This package trains revision behavior. Its participants cannot "
                "also establish unbiased loop-trained performance."
            ),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(request.output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build outcome-supervised loop training records."
    )
    parser.add_argument("--raw-oof-dir", required=True, type=Path)
    parser.add_argument(
        "--raw-artifact-manifest",
        required=True,
        type=Path,
    )
    parser.add_argument("--prepared-root", required=True, type=Path)
    parser.add_argument(
        "--prepared-artifact-manifest",
        required=True,
        type=Path,
    )
    parser.add_argument("--evaluation-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--heldout-fold", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_training_package(
        LoopTrainingPackageRequest(
            raw_oof_dir=args.raw_oof_dir,
            raw_artifact_manifest=args.raw_artifact_manifest,
            prepared_root=args.prepared_root,
            prepared_artifact_manifest=args.prepared_artifact_manifest,
            evaluation_dir=args.evaluation_dir,
            config_path=args.config,
            task_path=args.task,
            output_dir=args.output_dir,
            heldout_fold=args.heldout_fold,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
