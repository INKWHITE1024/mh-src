"""Fit the learned reliability audit and emit label-free trigger decisions.

``fit`` reads frozen out-of-fold initial assessments together with their
outcome rows, fits the source validity and the L2 logistic-regression audit on
first-pass errors, and fixes the risk threshold on development assessments at a
target trigger rate, which uses no development labels.  ``decide`` applies a
fitted audit to any frozen assessments and writes one decision per session;
evaluation reads these decisions instead of the heuristic trigger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from rethink_mh.rethinking.audit import (
    DEFAULT_SOURCES,
    AuditEstimator,
    fit_audit,
    fit_source_validity,
    select_risk_threshold,
)
from rethink_mh.rethinking.contracts import InitialAssessment
from rethink_mh.rethinking.trigger import RethinkPolicyConfig


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number} is not a JSON object")
            rows.append(value)
    return rows


def _assessments(path: Path) -> dict[str, InitialAssessment | None]:
    """Session id to initial assessment, or ``None`` when the first pass failed."""

    output: dict[str, InitialAssessment | None] = {}
    for row in _read_jsonl(path):
        session_id = str(row["session_id"])
        if session_id in output:
            raise ValueError(f"duplicate session in {path}: {session_id}")
        initial = row.get("initial")
        if row.get("status") == "initial_failed" or not isinstance(initial, Mapping):
            output[session_id] = None
        else:
            output[session_id] = InitialAssessment.from_dict(
                initial["initial_assessment"]
            )
    return output


def fit_from_files(
    *,
    raw_path: Path,
    outcomes_path: Path,
    development_raw_path: Path,
    target_trigger_rate: float,
    l2: float,
    degradations_path: Path | None = None,
    sources: Sequence[str] = DEFAULT_SOURCES,
) -> AuditEstimator:
    assessments = _assessments(raw_path)
    outcomes = {str(row["session_id"]): row for row in _read_jsonl(outcomes_path)}
    if set(assessments) != set(outcomes):
        raise ValueError("raw assessments and outcomes cover different sessions")
    fitted_ids = sorted(
        session_id for session_id, value in assessments.items() if value is not None
    )
    rows = [assessments[session_id] for session_id in fitted_ids]
    labels = [int(outcomes[session_id]["label"]) for session_id in fitted_ids]
    first_pass = [
        int(
            float(outcomes[session_id]["initial_probability"])
            >= float(outcomes[session_id]["initial_threshold"])
        )
        for session_id in fitted_ids
    ]
    errors = [int(decision != label) for decision, label in zip(first_pass, labels)]
    readings = {
        source: [row.source_risk_probabilities.get(source) for row in rows]
        for source in sources
    }
    validity = fit_source_validity(labels, readings)

    extra_rows: list[InitialAssessment] = []
    extra_targets: list[int] = []
    if degradations_path is not None:
        for row in _read_jsonl(degradations_path):
            extra_rows.append(InitialAssessment.from_dict(row["initial_assessment"]))
            extra_targets.append(int(row["target"]))

    config = RethinkPolicyConfig()
    estimator = fit_audit(
        rows,
        errors,
        source_validity=validity,
        l2=l2,
        reliable_source_threshold=config.reliable_source_threshold,
        sources=sources,
        extra_assessments=extra_rows,
        extra_targets=extra_targets,
    )
    development = [
        value for value in _assessments(development_raw_path).values() if value is not None
    ]
    threshold = select_risk_threshold(
        [estimator.assessment_risk(row) for row in development],
        target_trigger_rate,
    )
    metadata = {
        "fit_sessions": len(rows),
        "fit_error_rate": sum(errors) / len(errors),
        "degradation_rows": len(extra_rows),
        "development_sessions": len(development),
        "target_trigger_rate": target_trigger_rate,
        "threshold_rule": "development_risk_quantile_label_free",
        "raw_sha256": _sha256(raw_path),
        "outcomes_sha256": _sha256(outcomes_path),
        "development_raw_sha256": _sha256(development_raw_path),
        "degradations_sha256": (
            _sha256(degradations_path) if degradations_path is not None else None
        ),
    }
    return replace(estimator, threshold=threshold, metadata=metadata)


def decide_from_files(
    *, audit_path: Path, raw_path: Path
) -> list[dict[str, Any]]:
    """One label-free decision per session; a failed first pass is re-examined."""

    estimator = AuditEstimator.load(audit_path)
    model_sha256 = _sha256(audit_path)
    decisions = []
    for session_id, assessment in sorted(_assessments(raw_path).items()):
        if assessment is None:
            decisions.append(
                {
                    "session_id": session_id,
                    "should_rethink": True,
                    "audit_risk": None,
                    "audit_threshold": estimator.threshold,
                    "status": "initial_failed",
                    "audit_model_sha256": model_sha256,
                }
            )
            continue
        risk = estimator.assessment_risk(assessment)
        decisions.append(
            {
                "session_id": session_id,
                "should_rethink": risk >= estimator.threshold,
                "audit_risk": risk,
                "audit_threshold": estimator.threshold,
                "status": "ok",
                "audit_model_sha256": model_sha256,
            }
        )
    return decisions


def load_audit_decisions(path: Path) -> dict[str, bool]:
    """Session id to trigger decision, as written by ``decide``."""

    decisions: dict[str, bool] = {}
    for row in _read_jsonl(path):
        session_id = str(row["session_id"])
        if session_id in decisions:
            raise ValueError(f"duplicate audit decision: {session_id}")
        if not isinstance(row.get("should_rethink"), bool):
            raise ValueError(f"audit decision is not boolean: {session_id}")
        decisions[session_id] = row["should_rethink"]
    return decisions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("fit", help="fit validity, audit, and threshold")
    fit.add_argument("--raw", required=True, type=Path,
                     help="frozen OOF trajectories.raw.jsonl of the training folds")
    fit.add_argument("--outcomes", required=True, type=Path,
                     help="outcomes.jsonl with label, initial_probability, initial_threshold")
    fit.add_argument("--development-raw", required=True, type=Path,
                     help="frozen development trajectories used only for the threshold")
    fit.add_argument("--target-trigger-rate", required=True, type=float)
    fit.add_argument("--l2", type=float, default=1.0)
    fit.add_argument("--degradations", type=Path,
                     help="optional JSONL of synthetic degradations with known targets")
    fit.add_argument("--output", required=True, type=Path)

    decide = commands.add_parser("decide", help="write label-free trigger decisions")
    decide.add_argument("--audit-model", required=True, type=Path)
    decide.add_argument("--raw", required=True, type=Path)
    decide.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "fit":
        estimator = fit_from_files(
            raw_path=args.raw,
            outcomes_path=args.outcomes,
            development_raw_path=args.development_raw,
            target_trigger_rate=args.target_trigger_rate,
            l2=args.l2,
            degradations_path=args.degradations,
        )
        estimator.save(args.output)
        print(json.dumps(estimator.to_dict(), indent=2, sort_keys=True))
        return 0
    decisions = decide_from_files(audit_path=args.audit_model, raw_path=args.raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in decisions),
        encoding="utf-8",
    )
    triggered = sum(row["should_rethink"] for row in decisions)
    print(json.dumps({"sessions": len(decisions), "triggered": triggered}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
