"""Collect the three independent Loop/ORPO runs into JSON and Markdown reports."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


_DATASETS = ("daic_woz", "e_daic", "d_vlog")


def _optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def _last_loss(summary: dict[str, Any] | None) -> float | None:
    if summary is None:
        return None
    history = summary.get("history")
    if not isinstance(history, list) or not history:
        return None
    value = history[-1].get("weighted_train_loss")
    return float(value) if isinstance(value, (int, float)) else None


def _preference_metrics(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    audit = summary.get("preference_audit")
    if not isinstance(audit, dict):
        return None
    return {
        "sample_count": audit.get("sample_count"),
        "accuracy": (audit.get("overall") or {}).get("accuracy"),
        "mean_margin": (audit.get("overall") or {}).get("mean_margin"),
        "gate_passed": (audit.get("gate") or {}).get("passed"),
        "by_transition": audit.get("by_transition"),
        "action_by_transition": audit.get("action_by_transition"),
        "by_kind": audit.get("by_kind"),
    }


def collect_report(
    root: Path,
    *,
    package_root: Path | None = None,
    experiment_name: str = "rethink_revision_post_training",
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    package_root = (
        package_root.expanduser().resolve() if package_root is not None else root
    )
    datasets: dict[str, Any] = {}
    for dataset in _DATASETS:
        package = _optional_json(
            package_root / "trajectories" / dataset / "manifest.json"
        )
        run = root / "training" / dataset
        loop = _optional_json(run / "revision_sft" / "summary.json")
        orpo_initial = _optional_json(run / "orpo_initial" / "summary.json")
        refresh = _optional_json(run / "on_policy_refresh" / "manifest.json")
        orpo_refreshed = _optional_json(run / "orpo_refreshed" / "summary.json")
        grpo = _optional_json(run / "grpo_gate.json")
        datasets[dataset] = {
            "complete": (run / "PIPELINE_COMPLETE").is_file(),
            "trajectory": (
                {
                    "session_count": package["session_count"],
                    "positive_count": package["positive_count"],
                    "initial_error_count": package["initial_error_count"],
                    "initial_false_negative_count": package[
                        "initial_false_negative_count"
                    ],
                    "transition_counts": package["transition_counts"],
                    "preference_kind_counts": package["preference_kind_counts"],
                }
                if package is not None
                else None
            ),
            "revision_sft": {
                "complete": loop is not None,
                "final_weighted_loss": _last_loss(loop),
                "history": loop.get("history") if loop else None,
            },
            "orpo_initial": {
                "complete": orpo_initial is not None,
                "final_weighted_loss": _last_loss(orpo_initial),
                "preference": _preference_metrics(orpo_initial),
            },
            "on_policy_refresh": (
                {
                    "complete": True,
                    "generated_session_count": refresh["generated_session_count"],
                    "accepted_session_count": refresh["accepted_session_count"],
                    "acceptance_ratio": refresh["acceptance_ratio"],
                    "accepted_transition_counts": refresh[
                        "accepted_transition_counts"
                    ],
                    "rejection_reason_counts": refresh["rejection_reason_counts"],
                    "gate": refresh["gate"],
                }
                if refresh is not None
                else {"complete": False}
            ),
            "orpo_refreshed": {
                "complete": orpo_refreshed is not None,
                "final_weighted_loss": _last_loss(orpo_refreshed),
                "preference": _preference_metrics(orpo_refreshed),
            },
            "grpo": grpo,
        }
    return {
        "experiment": experiment_name,
        "training_root": str(root),
        "package_root": str(package_root),
        "all_complete": all(value["complete"] for value in datasets.values()),
        "cross_dataset_merging": False,
        "datasets": datasets,
    }


def _format(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Revision-SFT and revision-aware ORPO results",
        "",
        "The three datasets are always trained independently; preference accuracy below comes from the fixed stratified held-out ranking audit after training completes.",
        "",
        "| Dataset | Status | Annotated trajectories | Initial errors | Revision loss | ORPO initial acc | Refresh filter-pass | ORPO refreshed acc | GRPO |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in _DATASETS:
        value = report["datasets"][dataset]
        trajectory = value["trajectory"] or {}
        initial_pref = value["orpo_initial"]["preference"] or {}
        refreshed_pref = value["orpo_refreshed"]["preference"] or {}
        refresh = value["on_policy_refresh"]
        grpo = value["grpo"] or {}
        accepted = (
            f"{refresh.get('accepted_session_count')}/{refresh.get('generated_session_count')}"
            if refresh.get("complete")
            else "—"
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    dataset,
                    "complete" if value["complete"] else "running/blocked",
                    _format(trajectory.get("session_count")),
                    _format(trajectory.get("initial_error_count")),
                    _format(value["revision_sft"]["final_weighted_loss"]),
                    _format(initial_pref.get("accuracy")),
                    accepted,
                    _format(refreshed_pref.get("accuracy")),
                    _format(grpo.get("decision")),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "GRPO starts only after every per-dataset prerequisite gate passes; `blocked` is the correct fail-closed outcome and does not mean ORPO failed to run.",
            "",
        )
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--package-root", type=Path)
    parser.add_argument(
        "--experiment-name", default="rethink_revision_post_training"
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = collect_report(
        args.root,
        package_root=args.package_root,
        experiment_name=args.experiment_name,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["all_complete"] else 6


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
