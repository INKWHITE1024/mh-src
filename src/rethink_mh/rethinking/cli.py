"""Command-line interface for running rethinking workflows and evidence agent sessions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import InitialAssessment, RevisionAssessment, TaskSpec
from .evidence_agent import (
    EvidenceAgentConfig,
    EvidenceAgentWorkflow,
    verify_evidence_access_ledger,
)
from .prompts import PromptBuilder
from .query_retrieval import RETRIEVAL_PROTOCOL_VERSION
from .query_workflow import QueryGuidedRethinkingWorkflow
from .audit import AuditEstimator
from .trigger import RethinkPolicy, RethinkPolicyConfig
from .workflow import RethinkingWorkflow, _decision_dict


def _read_object(path: Path, description: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{description} is not valid JSON: {path}: {exc.msg}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must contain one JSON object: {path}")
    return value


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _write(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _task(path: Path) -> TaskSpec:
    return TaskSpec.from_dict(_read_object(path, "task specification"))


def _policy(args: argparse.Namespace) -> RethinkPolicy:
    """Learned audit when ``--audit-model`` is given, else the heuristic rules."""

    config = RethinkPolicyConfig()
    if args.policy_config is not None:
        values = _read_object(args.policy_config, "rethink policy config")
        config = RethinkPolicyConfig(**dict(values))
    audit = (
        AuditEstimator.load(args.audit_model)
        if getattr(args, "audit_model", None) is not None
        else None
    )
    return RethinkPolicy(config, audit=audit)


def _workflow(args: argparse.Namespace) -> RethinkingWorkflow:
    return RethinkingWorkflow(
        policy=_policy(args),
        max_atomic_per_segment=args.max_atomic_per_segment,
        max_observations_per_modality=args.max_observations_per_modality,
    )


def _workflow_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--policy-config",
        type=Path,
        help="Optional JSON object overriding RethinkPolicyConfig thresholds.",
    )
    parser.add_argument(
        "--audit-model",
        type=Path,
        help="Fitted audit from `rethink-fit-audit fit`; replaces the heuristic trigger.",
    )
    parser.add_argument("--max-atomic-per-segment", type=int, default=4)
    parser.add_argument("--max-observations-per-modality", type=int, default=3)


def _query_guided_workflow(args: argparse.Namespace) -> QueryGuidedRethinkingWorkflow:
    return QueryGuidedRethinkingWorkflow(
        policy=_policy(args),
        max_candidates_per_query=args.max_candidates_per_query,
    )


def _evidence_agent_workflow(args: argparse.Namespace) -> EvidenceAgentWorkflow:
    agent_values = EvidenceAgentConfig().to_dict()
    if args.agent_config is not None:
        agent_values.update(
            _read_object(args.agent_config, "evidence agent config")
        )
    for name in (
        "max_queries",
        "max_atomic_records",
        "max_released_tokens",
        "max_first_pass_tokens",
        "max_candidates_per_query",
    ):
        value = getattr(args, name)
        if value is not None:
            agent_values[name] = value
    return EvidenceAgentWorkflow(
        config=EvidenceAgentConfig(**agent_values),
        policy=_policy(args),
        max_contract_retries=args.max_contract_retries,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rethink-workflow",
        description="Prepare and validate a two-pass RETHINK-MH evidence workflow.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    initial = subparsers.add_parser(
        "prepare-initial", help="Build label-free first-pass Qwen messages."
    )
    initial.add_argument("--session-dir", required=True, type=Path)
    initial.add_argument("--task", required=True, type=Path)
    initial.add_argument("--output", required=True, type=Path)
    _workflow_options(initial)

    revision = subparsers.add_parser(
        "prepare-revision",
        help="Validate first-pass JSON, run the trigger, and retrieve targeted evidence.",
    )
    revision.add_argument("--session-dir", required=True, type=Path)
    revision.add_argument("--task", required=True, type=Path)
    revision.add_argument("--initial-output", required=True, type=Path)
    revision.add_argument("--output-dir", required=True, type=Path)
    _workflow_options(revision)

    finalize = subparsers.add_parser(
        "finalize", help="Validate grounded revision JSON and emit the final workflow state."
    )
    finalize.add_argument("--session-dir", required=True, type=Path)
    finalize.add_argument("--task", required=True, type=Path)
    finalize.add_argument("--initial-output", required=True, type=Path)
    finalize.add_argument("--revision-output", type=Path)
    finalize.add_argument("--output", required=True, type=Path)
    _workflow_options(finalize)

    sft = subparsers.add_parser(
        "build-sft-record",
        help=(
            "Build one draft SFT record; the gold label stays outside all chat text."
        ),
    )
    sft.add_argument("--session-dir", required=True, type=Path)
    sft.add_argument("--task", required=True, type=Path)
    sft.add_argument("--label", required=True)
    sft.add_argument(
        "--assistant-target",
        required=True,
        type=Path,
        help="Strict InitialAssessment JSON used as the schema-matched assistant target.",
    )
    sft.add_argument("--output", required=True, type=Path)

    qwen = subparsers.add_parser(
        "run-qwen",
        help="Run the complete one- or two-pass workflow with the text-only Qwen Thinker.",
    )
    qwen.add_argument("--session-dir", required=True, type=Path)
    qwen.add_argument("--task", required=True, type=Path)
    qwen.add_argument("--model-path", required=True, type=Path)
    qwen.add_argument("--device", default="cuda")
    qwen.add_argument("--dtype", default="bfloat16")
    qwen.add_argument("--attention-implementation", default="sdpa")
    qwen.add_argument("--max-new-tokens", type=int, default=640)
    qwen.add_argument("--max-input-tokens", type=int, default=7500)
    qwen.add_argument("--output", required=True, type=Path)
    _workflow_options(qwen)

    query_qwen = subparsers.add_parser(
        "run-query-guided-qwen",
        help=(
            "Run the four-turn evidence-grounded workflow with explicit query "
            "and grounded atomic selection turns."
        ),
    )
    query_qwen.add_argument("--session-dir", required=True, type=Path)
    query_qwen.add_argument("--task", required=True, type=Path)
    query_qwen.add_argument("--model-path", required=True, type=Path)
    query_qwen.add_argument("--adapter-path", type=Path)
    query_qwen.add_argument("--device", default="cuda")
    query_qwen.add_argument("--dtype", default="bfloat16")
    query_qwen.add_argument("--attention-implementation", default="sdpa")
    query_qwen.add_argument("--max-new-tokens", type=int, default=640)
    query_qwen.add_argument("--max-input-tokens", type=int, default=7500)
    query_qwen.add_argument("--max-candidates-per-query", type=int, default=8)
    query_qwen.add_argument(
        "--policy-config",
        type=Path,
        help="Optional JSON object overriding RethinkPolicyConfig thresholds.",
    )
    query_qwen.add_argument(
        "--audit-model",
        type=Path,
        help="Fitted audit from `rethink-fit-audit fit`; replaces the heuristic trigger.",
    )
    query_qwen.add_argument("--output", required=True, type=Path)

    agent_qwen = subparsers.add_parser(
        "run-evidence-agent-qwen",
        help=(
            "Run the budgeted, hash-audited Evidence Agent workflow over a "
            "native evidence session."
        ),
    )
    agent_qwen.add_argument("--session-dir", required=True, type=Path)
    agent_qwen.add_argument(
        "--transcript-dir",
        type=Path,
        help=(
            "Optional compact transcript directory; the first safe tier "
            "that fits the token budget is selected."
        ),
    )
    agent_qwen.add_argument("--task", required=True, type=Path)
    agent_qwen.add_argument("--model-path", required=True, type=Path)
    agent_qwen.add_argument("--adapter-path", type=Path)
    agent_qwen.add_argument("--device", default="cuda")
    agent_qwen.add_argument("--dtype", default="bfloat16")
    agent_qwen.add_argument("--attention-implementation", default="sdpa")
    agent_qwen.add_argument("--max-new-tokens", type=int, default=640)
    agent_qwen.add_argument("--max-input-tokens", type=int, default=7500)
    agent_qwen.add_argument(
        "--evidence-protocol",
        choices=("auto", "native"),
        default="auto",
    )
    agent_qwen.add_argument(
        "--agent-config",
        type=Path,
        help="Optional JSON object overriding EvidenceAgentConfig.",
    )
    agent_qwen.add_argument(
        "--policy-config",
        type=Path,
        help="Optional JSON object overriding RethinkPolicyConfig thresholds.",
    )
    agent_qwen.add_argument(
        "--audit-model",
        type=Path,
        help="Fitted audit from `rethink-fit-audit fit`; replaces the heuristic trigger.",
    )
    agent_qwen.add_argument("--max-queries", type=int)
    agent_qwen.add_argument("--max-atomic-records", type=int)
    agent_qwen.add_argument("--max-released-tokens", type=int)
    agent_qwen.add_argument("--max-first-pass-tokens", type=int)
    agent_qwen.add_argument("--max-candidates-per-query", type=int)
    agent_qwen.add_argument(
        "--max-contract-retries",
        type=int,
        choices=(0, 1, 2),
        default=1,
        help="Maximum explicit JSON repair turns per model stage.",
    )
    agent_qwen.add_argument(
        "--run-id",
        help=(
            "Optional opaque audit identifier (8-128 safe characters); "
            "must not encode a participant ID."
        ),
    )
    agent_qwen.add_argument(
        "--ledger",
        required=True,
        type=Path,
        help="New append-only JSONL evidence-access ledger.",
    )
    agent_qwen.add_argument("--output", required=True, type=Path)

    verify_ledger = subparsers.add_parser(
        "verify-evidence-ledger",
        help="Verify an Evidence Agent ledger hash chain and terminal state.",
    )
    verify_ledger.add_argument("--ledger", required=True, type=Path)
    verify_ledger.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "verify-evidence-ledger":
        verification = verify_evidence_access_ledger(args.ledger).to_dict()
        if args.output is not None:
            _write_json(args.output, verification)
        print(json.dumps(verification, indent=2, sort_keys=True))
        return 0

    task = _task(args.task)

    if args.command == "prepare-initial":
        prepared = _workflow(args).prepare_initial(args.session_dir, task)
        _write_json(args.output, prepared.to_dict())
        print(
            json.dumps(
                {
                    "stage": "initial_prepared",
                    "available_segment_count": len(prepared.available_segment_ids),
                    "output": str(args.output.resolve()),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "prepare-revision":
        workflow = _workflow(args)
        assessment = InitialAssessment.from_dict(
            _read_object(args.initial_output, "initial model output")
        )
        decision = workflow.evaluate_initial(args.session_dir, assessment)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(args.output_dir / "rethink_decision.json", _decision_dict(decision))
        prepared = workflow.prepare_revision(args.session_dir, task, assessment)
        if prepared is None:
            result = workflow.finalize(args.session_dir, task, assessment)
            _write_json(args.output_dir / "workflow_result.json", result.to_dict())
            print(json.dumps({"stage": "accepted_initial", "should_rethink": False}, indent=2))
            return 0
        _write(
            args.output_dir / "targeted_evidence.txt",
            prepared.targeted_evidence.text,
        )
        _write_json(
            args.output_dir / "revision_messages.json",
            {"messages": [dict(message) for message in prepared.messages]},
        )
        _write_json(
            args.output_dir / "retrieval_manifest.json",
            prepared.to_dict(include_evidence_text=False),
        )
        print(
            json.dumps(
                {
                    "stage": "revision_prepared",
                    "should_rethink": True,
                    "selected_segment_ids": list(decision.selected_segment_ids),
                    "output_dir": str(args.output_dir.resolve()),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "finalize":
        workflow = _workflow(args)
        assessment = InitialAssessment.from_dict(
            _read_object(args.initial_output, "initial model output")
        )
        revision_assessment = (
            RevisionAssessment.from_dict(
                _read_object(args.revision_output, "revision model output")
            )
            if args.revision_output is not None
            else None
        )
        result = workflow.finalize(
            args.session_dir, task, assessment, revision_assessment
        )
        _write_json(args.output, result.to_dict())
        print(
            json.dumps(
                {
                    "outcome": result.outcome,
                    "model_call_count": result.model_call_count,
                    "output": str(args.output.resolve()),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "build-sft-record":
        session_text = (
            args.session_dir.expanduser().resolve() / "session_input.native.txt"
        ).read_text(encoding="utf-8")
        label: str | int = args.label
        if args.label in {"0", "1"}:
            label = int(args.label)
        assistant_target = InitialAssessment.from_dict(
            _read_object(args.assistant_target, "assistant target")
        )
        record = PromptBuilder.build_supervised_record(
            task,
            session_text,
            label,
            assistant_target=assistant_target,
        )
        _write_json(args.output, record)
        print(json.dumps({"stage": "sft_record_built", "output": str(args.output.resolve())}, indent=2))
        return 0

    if args.command == "run-qwen":
        from .qwen import QwenThinkerCompletionModel

        model = QwenThinkerCompletionModel.from_pretrained(
            args.model_path,
            device=args.device,
            dtype=args.dtype,
            attention_implementation=args.attention_implementation,
            max_new_tokens=args.max_new_tokens,
            max_input_tokens=args.max_input_tokens,
        )
        result = _workflow(args).run(args.session_dir, task, model)
        _write_json(args.output, result.to_dict())
        print(
            json.dumps(
                {
                    "outcome": result.outcome,
                    "model_call_count": result.model_call_count,
                    "output": str(args.output.resolve()),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "run-query-guided-qwen":
        from .qwen import QwenThinkerCompletionModel

        model = QwenThinkerCompletionModel.from_pretrained(
            args.model_path,
            adapter_path=args.adapter_path,
            device=args.device,
            dtype=args.dtype,
            attention_implementation=args.attention_implementation,
            max_new_tokens=args.max_new_tokens,
            max_input_tokens=args.max_input_tokens,
        )
        result = _query_guided_workflow(args).run(
            args.session_dir,
            task,
            model,
        )
        _write_json(args.output, result.to_dict())
        print(
            json.dumps(
                {
                    "outcome": result.outcome,
                    "model_call_count": result.model_call_count,
                    "retrieval_protocol_version": RETRIEVAL_PROTOCOL_VERSION,
                    "output": str(args.output.resolve()),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "run-evidence-agent-qwen":
        from .qwen import QwenThinkerCompletionModel

        if args.output.expanduser().resolve() == args.ledger.expanduser().resolve():
            raise ValueError("--output and --ledger must be different files")
        model = QwenThinkerCompletionModel.from_pretrained(
            args.model_path,
            adapter_path=args.adapter_path,
            device=args.device,
            dtype=args.dtype,
            attention_implementation=args.attention_implementation,
            max_new_tokens=args.max_new_tokens,
            max_input_tokens=args.max_input_tokens,
        )
        result = _evidence_agent_workflow(args).run(
            args.session_dir,
            task,
            model,
            ledger_path=args.ledger,
            transcript_dir=args.transcript_dir,
            evidence_protocol=args.evidence_protocol,
            run_id=args.run_id,
        )
        _write_json(args.output, result.to_dict())
        print(
            json.dumps(
                {
                    "outcome": result.outcome,
                    "model_call_count": result.model_call_count,
                    "evidence_protocol_version": (
                        result.evidence_protocol_version
                    ),
                    "ledger_final_sha256": (
                        result.ledger.final_entry_sha256
                    ),
                    "output": str(args.output.resolve()),
                },
                indent=2,
            )
        )
        return 0

    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
