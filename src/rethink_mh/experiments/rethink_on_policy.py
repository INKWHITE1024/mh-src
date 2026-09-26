"""Run, filter, and freeze a current-policy rollout refresh for offline ORPO."""

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

from rethink_mh.experiments.rethink_post_training import audit_training_records
from rethink_mh.experiments.rethink_trajectories import select_optimization_preferences
from rethink_mh.rethinking.contracts import RevisionAssessment
from rethink_mh.rethinking.qwen import QwenThinkerCompletionModel


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {description} at {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be one JSON object")
    return value


def _read_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read {description} at {path}: {error}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise ValueError(f"blank line in {description}: {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {description}: {line_number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"{description} row {line_number} must be an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{description} is empty")
    return rows


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(_canonical_json(row) + "\n" for row in rows)


def _optimization_preference_path(source: Path) -> Path:
    """Use the frozen memory-bounded derivative when given a full audit set."""

    resolved = source.expanduser().resolve()
    if resolved.name != "preferences_initial.jsonl" or len(resolved.parents) < 3:
        return resolved
    candidate = (
        resolved.parents[2]
        / "preference_optimization_v1"
        / resolved.parent.name
        / "preferences_optimization.jsonl"
    )
    manifest_path = candidate.parent / "manifest.json"
    if not candidate.is_file() or not manifest_path.is_file():
        return resolved
    manifest = _read_json(manifest_path, "preference optimization manifest")
    source_record = manifest.get("source")
    if not isinstance(source_record, Mapping):
        raise ValueError("preference optimization manifest has no source record")
    if source_record.get("sha256") != _sha256_file(resolved):
        raise ValueError("preference optimization subset does not match the full initial source")
    return candidate


def _grounded(revision: RevisionAssessment, messages: Sequence[Mapping[str, str]]) -> bool:
    prompt = "\n".join(message["content"] for message in messages)
    identifiers = (
        *revision.cited_segment_ids,
        *revision.cited_evidence_ids,
        *revision.preserved_evidence_ids,
        *revision.newly_considered_evidence_ids,
        *revision.rejected_evidence_ids,
        *revision.residual_conflict_segment_ids,
    )
    return all(identifier in prompt for identifier in identifiers)


def _semantic_consistency(
    revision: RevisionAssessment, initial_probability: float
) -> str | None:
    delta = revision.revised_risk_probability - initial_probability
    if revision.revision_status == "preserved":
        if abs(delta) > 0.05:
            return "preserved_probability_moved"
        if revision.change_summary != "risk_unchanged_after_review":
            return "preserved_summary_mismatch"
    elif revision.revision_status == "revised":
        if abs(delta) < 0.05:
            return "revised_probability_unchanged"
        expected = (
            "risk_increased_after_review" if delta > 0 else "risk_decreased_after_review"
        )
        if revision.change_summary != expected:
            return "revised_summary_direction_mismatch"
    else:
        if revision.change_summary != "insufficient_reliable_detail":
            return "unresolved_summary_mismatch"
        if not revision.residual_conflict_segment_ids:
            return "unresolved_without_residual_conflict"
    return None


def _outcome_correct(revision: RevisionAssessment, trajectory: Mapping[str, Any]) -> bool:
    transition = trajectory["transition_target"]
    label = trajectory["supervision"]["label"]
    threshold = float(trajectory["initial_threshold"])
    final_label = int(revision.revised_risk_probability >= threshold)
    if transition == "CC_preserve":
        return revision.revision_status == "preserved" and final_label == label
    if transition == "WC_revise":
        return revision.revision_status == "revised" and final_label == label
    if transition == "WW_refer":
        return revision.revision_status == "unresolved"
    return False


def _constraint_negative(chosen: Mapping[str, Any], kind: str) -> dict[str, Any]:
    negative = json.loads(json.dumps(chosen))
    if kind == "constraint_unknown_evidence_id":
        cited = list(negative["cited_evidence_ids"])
        cited.insert(0, "E999999")
        negative["cited_evidence_ids"] = cited
        newly = list(negative["newly_considered_evidence_ids"])
        newly.insert(0, "E999999")
        negative["newly_considered_evidence_ids"] = newly
    elif kind == "constraint_missing_required_field":
        negative.pop("change_summary")
    elif kind == "constraint_out_of_scope_clinical_claim":
        negative["clinical_diagnosis"] = "definitive diagnosis from screening evidence"
    else:  # pragma: no cover - guarded by audited initial records
        raise ValueError(f"unsupported constraint preference: {kind}")
    return negative


@dataclass(frozen=True, slots=True)
class RefreshRequest:
    preferences_initial: Path
    trajectories: Path
    orpo_summary: Path
    model_path: Path
    adapter_path: Path
    output_dir: Path
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    attention_implementation: str = "sdpa"
    max_input_tokens: int = 7500
    max_new_tokens: int = 420
    minimum_acceptance_ratio: float = 0.15
    minimum_accepted_sessions: int = 12
    max_sessions: int | None = 192

    def __post_init__(self) -> None:
        for name in ("max_input_tokens", "max_new_tokens", "minimum_accepted_sessions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0.0 <= self.minimum_acceptance_ratio <= 1.0:
            raise ValueError("minimum_acceptance_ratio must be in [0, 1]")
        if self.max_sessions is not None and (
            isinstance(self.max_sessions, bool)
            or not isinstance(self.max_sessions, int)
            or self.max_sessions <= 0
        ):
            raise ValueError("max_sessions must be a positive integer or None")


def _stratified_session_ids(
    session_ids: Sequence[str],
    trajectories: Mapping[str, Mapping[str, Any]],
    *,
    limit: int | None,
) -> list[str]:
    if limit is None or len(session_ids) <= limit:
        return sorted(session_ids)
    groups: dict[str, list[str]] = {}
    for session_id in session_ids:
        transition = str(trajectories[session_id]["transition_target"])
        groups.setdefault(transition, []).append(session_id)
    for transition, values in groups.items():
        values.sort(
            key=lambda session_id: hashlib.sha256(
                f"42:{transition}:{session_id}".encode()
            ).hexdigest()
        )
    selected: list[str] = []
    while len(selected) < limit:
        progressed = False
        for transition in sorted(groups):
            if groups[transition] and len(selected) < limit:
                selected.append(groups[transition].pop())
                progressed = True
        if not progressed:
            break
    return selected


def refresh_preferences(request: RefreshRequest) -> dict[str, Any]:
    """Run one immutable current-policy rollout refresh and publish refreshed ORPO pairs."""

    output_dir = request.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite current-policy rollout refresh: {output_dir}"
        )
    optimization_preferences = _optimization_preference_path(request.preferences_initial)
    audited = audit_training_records("orpo", optimization_preferences)
    preference_rows = list(audited.rows)
    trajectory_rows = _read_jsonl(request.trajectories, "OOF trajectories")
    trajectory_by_id: dict[str, dict[str, Any]] = {}
    for row in trajectory_rows:
        session_id = row.get("session_id")
        if not isinstance(session_id, str) or session_id in trajectory_by_id:
            raise ValueError("trajectories must contain unique text session IDs")
        if row.get("dataset") != audited.dataset:
            raise ValueError(f"trajectory dataset mismatch for {session_id}")
        supervision = row.get("supervision")
        if not isinstance(supervision, Mapping) or supervision.get("used_in_model_text") is not False:
            raise ValueError(f"trajectory supervision boundary is invalid for {session_id}")
        trajectory_by_id[session_id] = row
    if set(trajectory_by_id) != {str(row["session_id"]) for row in preference_rows}:
        raise ValueError("preference and trajectory session cohorts differ")

    orpo_summary = _read_json(request.orpo_summary, "initial ORPO summary")
    if orpo_summary.get("stage") != "orpo" or orpo_summary.get("dataset") != audited.dataset:
        raise ValueError("ORPO summary does not match the refresh cohort")
    gate = (orpo_summary.get("preference_audit") or {}).get("gate")
    if not isinstance(gate, Mapping) or gate.get("passed") is not True:
        raise ValueError("initial ORPO preference audit did not pass; refresh is blocked")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in preference_rows:
        grouped.setdefault(str(row["session_id"]), []).append(row)
    for session_id, rows in grouped.items():
        prompts = {_canonical_json(row["prompt_messages"]) for row in rows}
        chosen = {str(row["chosen"]) for row in rows}
        if len(prompts) != 1 or len(chosen) != 1:
            raise ValueError(f"initial preferences disagree within session {session_id}")
    single_pair_layout = all(
        sum(
            not str(row["preference_kind"]).startswith("constraint_")
            for row in rows
        )
        == 1
        and sum(
            str(row["preference_kind"]).startswith("constraint_")
            for row in rows
        )
        == 3
        for rows in grouped.values()
    )
    refresh_session_ids = _stratified_session_ids(
        list(grouped), trajectory_by_id, limit=request.max_sessions
    )

    model = QwenThinkerCompletionModel.from_pretrained(
        request.model_path,
        adapter_path=request.adapter_path,
        device=request.device,
        dtype=request.dtype,
        attention_implementation=request.attention_implementation,
        max_new_tokens=request.max_new_tokens,
        max_input_tokens=request.max_input_tokens,
    )
    generations: list[dict[str, Any]] = []
    accepted_by_id: dict[str, dict[str, Any]] = {}
    rejection_reasons: Counter[str] = Counter()
    acceptance_transitions: Counter[str] = Counter()
    for position, session_id in enumerate(refresh_session_ids, 1):
        rows = grouped[session_id]
        prompt = rows[0]["prompt_messages"]
        raw = model.complete(prompt)
        reason: str | None = None
        parsed: dict[str, Any] | None = None
        try:
            candidate = json.loads(raw)
            if not isinstance(candidate, dict):
                raise ValueError("not an object")
            revision = RevisionAssessment.from_dict(candidate)
            if not _grounded(revision, prompt):
                reason = "ungrounded_identifier"
            else:
                trajectory = trajectory_by_id[session_id]
                initial_probability = float(
                    trajectory["initial_assessment"]["risk_probability"]
                )
                reason = _semantic_consistency(revision, initial_probability)
                if reason is None and not _outcome_correct(revision, trajectory):
                    reason = "incorrect_state_transition_outcome"
                if reason is None:
                    parsed = revision.to_dict()
        except (json.JSONDecodeError, TypeError, ValueError, KeyError):
            reason = "invalid_schema_or_json"
        accepted = parsed is not None
        if accepted:
            accepted_by_id[session_id] = parsed
            acceptance_transitions[str(trajectory_by_id[session_id]["transition_target"])] += 1
        else:
            rejection_reasons[reason or "unknown"] += 1
        generations.append(
            {
                "record_type": "on_policy_revision_generation",
                "dataset": audited.dataset,
                "session_id": session_id,
                "transition_target": trajectory_by_id[session_id]["transition_target"],
                "accepted": accepted,
                "rejection_reason": None if accepted else reason,
                "raw_completion": raw,
                "parsed_revision": parsed,
            }
        )
        print(
            _canonical_json(
                {
                    "generated": position,
                    "total": len(refresh_session_ids),
                    "session_id": session_id,
                    "accepted": accepted,
                    "reason": reason,
                }
            ),
            flush=True,
        )

    required_count = max(
        request.minimum_accepted_sessions,
        math.ceil(len(refresh_session_ids) * request.minimum_acceptance_ratio),
    )
    missing_transitions = sorted(
        transition for transition in ("CC_preserve", "WC_revise", "WW_refer")
        if acceptance_transitions[transition] == 0
    )
    gate_passed = len(accepted_by_id) >= required_count and not missing_transitions

    v2_rows: list[dict[str, Any]] = []
    if gate_passed:
        constraint_kinds = (
            "constraint_missing_required_field",
            "constraint_out_of_scope_clinical_claim",
            "constraint_unknown_evidence_id",
        )
        refreshed_template_by_id: dict[str, dict[str, Any]] = {}
        for session_id in sorted(accepted_by_id):
            action_rows = [
                row
                for row in grouped[session_id]
                if not str(row["preference_kind"]).startswith("constraint_")
            ]
            if not action_rows:
                raise ValueError(
                    f"refresh requires a grounded action pair for {session_id}"
                )
            chosen = accepted_by_id[session_id]
            chosen_text = _canonical_json(chosen)
            refreshed_actions: list[dict[str, Any]] = []
            for source in action_rows:
                updated = dict(source)
                updated["chosen"] = chosen_text
                updated["preference_origin"] = (
                    "accepted_on_policy_orpo_generation"
                    if single_pair_layout
                    else "accepted_current_policy_orpo_rollout"
                )
                if updated["rejected"] != chosen_text:
                    refreshed_actions.append(updated)
            if not refreshed_actions:
                raise ValueError(
                    f"refresh collapsed every hard pair for {session_id}"
                )
            v2_rows.extend(refreshed_actions)
            refreshed_template_by_id[session_id] = refreshed_actions[0]
            source_constraint_kinds = [
                str(row["preference_kind"])
                for row in grouped[session_id]
                if str(row["preference_kind"]).startswith("constraint_")
            ]
            for kind in source_constraint_kinds:
                if kind not in constraint_kinds:
                    raise ValueError(
                        f"unsupported source constraint kind for {session_id}: {kind}"
                    )
                constraint = dict(refreshed_actions[0])
                constraint["preference_kind"] = kind
                constraint["rejected"] = _canonical_json(
                    _constraint_negative(chosen, kind)
                )
                if not single_pair_layout:
                    constraint["preference_origin"] = (
                        "rebuilt_for_accepted_current_policy_rollout"
                    )
                v2_rows.append(constraint)

        # A memory-bounded source subset may place constraint guards on only a
        # few sessions. Retain that sparse ratio, then deterministically top up
        # only globally missing guard types so the published file remains auditable.
        present_constraint_kinds = {
            str(row["preference_kind"])
            for row in v2_rows
            if str(row["preference_kind"]).startswith("constraint_")
        }
        for kind in constraint_kinds:
            if kind in present_constraint_kinds:
                continue
            session_id = next(
                value
                for value in sorted(refreshed_template_by_id)
                if not any(
                    str(row["session_id"]) == value
                    and row["preference_kind"] == kind
                    for row in v2_rows
                )
            )
            template = dict(refreshed_template_by_id[session_id])
            chosen = accepted_by_id[session_id]
            template["preference_kind"] = kind
            template["rejected"] = _canonical_json(
                _constraint_negative(chosen, kind)
            )
            template["preference_origin"] = (
                "global_minimum_contract_guard_top_up"
            )
            v2_rows.append(template)
            present_constraint_kinds.add(kind)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=str(output_dir.parent))
    )
    try:
        generations_text = _jsonl(generations)
        (temporary / "refresh_generations.jsonl").write_text(
            generations_text, encoding="utf-8"
        )
        if gate_passed:
            preferences_text = _jsonl(v2_rows)
            (temporary / "preferences_refreshed.jsonl").write_text(
                preferences_text, encoding="utf-8"
            )
            # Re-audit the exact published file, including refreshed constraints.
            audit_training_records("orpo", temporary / "preferences_refreshed.jsonl")
            optimization_rows = select_optimization_preferences(
                v2_rows, constraint_session_fraction=0.10, seed=42
            )
            optimization_text = _jsonl(optimization_rows)
            (temporary / "preferences_refreshed_optimization.jsonl").write_text(
                optimization_text, encoding="utf-8"
            )
            audit_training_records(
                "orpo", temporary / "preferences_refreshed_optimization.jsonl"
            )
        manifest = {
            "schema_version": "1.0.0",
            "package": "policy_refresh_revision_preferences",
            **(
                {
                    "package_display_name": (
                        "current_policy_rollout_revision_preferences"
                    ),
                    "legacy_package_identifier": True,
                }
                if not single_pair_layout
                else {}
            ),
            "dataset": audited.dataset,
            "source_session_count": len(grouped),
            "generated_session_count": len(refresh_session_ids),
            "refresh_sampling": {
                "max_sessions": request.max_sessions,
                "stratified_by_transition": True,
                "seed": 42,
            },
            "accepted_session_count": len(accepted_by_id),
            "acceptance_ratio": len(accepted_by_id) / len(refresh_session_ids),
            "accepted_transition_counts": dict(sorted(acceptance_transitions.items())),
            "rejection_reason_counts": dict(sorted(rejection_reasons.items())),
            **(
                {
                    "preference_refresh": {
                        "source_layout": "multi_hard_contrastive",
                        "all_noncollapsed_hard_pairs_preserved": gate_passed,
                        "hard_pair_count": sum(
                            not str(row["preference_kind"]).startswith(
                                "constraint_"
                            )
                            for row in v2_rows
                        ),
                        "contract_guard_count": sum(
                            str(row["preference_kind"]).startswith(
                                "constraint_"
                            )
                            for row in v2_rows
                        ),
                    }
                }
                if not single_pair_layout
                else {}
            ),
            "gate": {
                "passed": gate_passed,
                "minimum_accepted_sessions": required_count,
                "missing_transitions": missing_transitions,
                "refreshed_orpo_training_allowed": gate_passed,
            },
            "model": {
                "base_model": str(request.model_path.resolve()),
                "adapter": str(request.adapter_path.resolve()),
                "orpo_initial_summary_sha256": _sha256_file(request.orpo_summary),
            },
            "inputs": {
                "preferences_initial_sha256": _sha256_file(request.preferences_initial),
                "optimization_preferences": str(optimization_preferences),
                "optimization_preferences_sha256": _sha256_file(
                    optimization_preferences
                ),
                "trajectories_sha256": _sha256_file(request.trajectories),
            },
            "files": {
                "refresh_generations.jsonl": {
                    "rows": len(generations),
                    "sha256": hashlib.sha256(generations_text.encode()).hexdigest(),
                },
                **(
                    {
                        "preferences_refreshed.jsonl": {
                            "rows": len(v2_rows),
                            "sha256": hashlib.sha256(preferences_text.encode()).hexdigest(),
                        },
                        "preferences_refreshed_optimization.jsonl": {
                            "rows": len(optimization_rows),
                            "sha256": hashlib.sha256(
                                optimization_text.encode()
                            ).hexdigest(),
                        },
                    }
                    if gate_passed
                    else {}
                ),
            },
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def grpo_gate_report(*, dataset: str, trajectory_count: int) -> dict[str, Any]:
    """Evaluate the preregistered v4 GRPO entry criteria without relaxing them."""

    enough_trajectories = trajectory_count >= 5000
    return {
        "dataset": dataset,
        "decision": "eligible" if enough_trajectories else "blocked",
        "criteria": {
            "trajectory_pool_at_least_5000": {
                "observed": trajectory_count,
                "passed": enough_trajectories,
            },
            "within_group_reward_variance_prompt_ratio_at_least_0_60": {
                "observed": None,
                "passed": False,
                "reason": "not_evaluated_after_trajectory_pool_gate_failed",
            },
            "rule_reward_offline_audit_passed": {
                "observed": None,
                "passed": False,
                "reason": "not_evaluated_after_trajectory_pool_gate_failed",
            },
            "residual_errors_are_reasoning_not_missing_evidence": {
                "observed": None,
                "passed": False,
                "reason": "requires_post_ORPO_residual_error_audit",
            },
        },
        "permitted_training": "ORPO_initial_refresh_ORPO_refreshed",
        "grpo_training_started": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and filter current-policy rollout revisions for the refreshed ORPO pass."
    )
    parser.add_argument("--preferences-initial", required=True, type=Path)
    parser.add_argument("--trajectories", required=True, type=Path)
    parser.add_argument("--orpo-summary", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--adapter-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument("--max-input-tokens", type=int, default=7500)
    parser.add_argument("--max-new-tokens", type=int, default=420)
    parser.add_argument("--minimum-acceptance-ratio", type=float, default=0.15)
    parser.add_argument("--minimum-accepted-sessions", type=int, default=12)
    parser.add_argument("--max-sessions", type=int, default=192)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = refresh_preferences(
        RefreshRequest(
            preferences_initial=args.preferences_initial,
            trajectories=args.trajectories,
            orpo_summary=args.orpo_summary,
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            output_dir=args.output_dir,
            device=args.device,
            dtype=args.dtype,
            attention_implementation=args.attention_implementation,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            minimum_acceptance_ratio=args.minimum_acceptance_ratio,
            minimum_accepted_sessions=args.minimum_accepted_sessions,
            max_sessions=args.max_sessions,
        )
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if manifest["gate"]["passed"] else 5


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
