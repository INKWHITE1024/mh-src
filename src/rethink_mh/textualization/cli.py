"""Command-line interface for auditing, reference fitting, and compiling evidence text datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_config
from .pipeline import audit_dataset, compile_dataset, fit_references
from .protocols import SUPPORTED_PROTOCOLS
from .transcript import compile_transcript_dataset


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("daic_woz", "e_daic", "d_vlog"),
        help="Dataset-specific compiler to use.",
    )
    parser.add_argument("--root", required=True, type=Path, help="Extracted dataset root.")
    parser.add_argument("--config", type=Path, help="Optional YAML config override.")


def _selection(parser: argparse.ArgumentParser, require_file: bool = False) -> None:
    parser.add_argument(
        "--split-file",
        required=require_file,
        type=Path,
        help="CSV used only for session ID and, when requested, split membership.",
    )
    parser.add_argument(
        "--split-value",
        help="Optional value in a fold/split column (for example train or valid).",
    )
    parser.add_argument("--max-sessions", type=int, help="Deterministic prefix for quick pipeline checks.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rethink-textualize",
        description="Compile released multimodal features into source-aware evidence text.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Check layouts and feature headers without labels.")
    _common(audit)
    _selection(audit)
    audit.add_argument("--output", type=Path, help="Optional JSON audit report.")

    fit = subparsers.add_parser(
        "fit-reference", help="Fit source-specific robust statistics on training IDs only."
    )
    _common(fit)
    _selection(fit, require_file=True)
    fit.add_argument("--fit-split", default="train", choices=("train", "training"))
    fit.add_argument("--seed", type=int, default=17)
    fit.add_argument("--output", required=True, type=Path, help="Reference JSON output.")

    compile_parser = subparsers.add_parser(
        "compile",
        help="Emit canonical evidence JSONL and the selected model-facing text protocol.",
    )
    _common(compile_parser)
    _selection(compile_parser)
    compile_parser.add_argument("--split-name", help="Manifest-only split name.")
    compile_parser.add_argument("--reference", required=True, type=Path)
    compile_parser.add_argument("--output-root", required=True, type=Path)
    compile_parser.add_argument(
        "--protocol-version",
        choices=SUPPORTED_PROTOCOLS,
        help="Model-facing text protocol; defaults to the dataset config value.",
    )
    compile_parser.add_argument("--workers", type=int, default=1)
    compile_parser.add_argument("--overwrite", action="store_true")

    transcript_parser = subparsers.add_parser(
        "compile-transcript",
        help=(
            "Emit a separate readable transcript layer aligned to native "
            "Evidence segments."
        ),
    )
    _common(transcript_parser)
    _selection(transcript_parser)
    transcript_parser.add_argument(
        "--evidence-root",
        required=True,
        type=Path,
        help="Dataset-specific Evidence directory containing session folders.",
    )
    transcript_parser.add_argument("--output-root", required=True, type=Path)
    transcript_parser.add_argument("--overwrite", action="store_true")
    return parser


def _print_progress(message: str) -> None:
    print(message, flush=True)


def _print_result(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.dataset, args.config)

    if args.command == "audit":
        report = audit_dataset(
            root=args.root,
            config=config,
            split_file=args.split_file,
            split_value=args.split_value,
            max_sessions=args.max_sessions,
            output_path=args.output,
            progress=_print_progress,
        )
        _print_result({key: value for key, value in report.items() if key != "sessions"})
        return 0 if report["error_count"] == 0 else 2

    if args.command == "fit-reference":
        references = fit_references(
            root=args.root,
            config=config,
            split_file=args.split_file,
            output_path=args.output,
            fit_split=args.fit_split,
            split_value=args.split_value,
            max_sessions=args.max_sessions,
            seed=args.seed,
            progress=_print_progress,
        )
        _print_result(
            {
                "dataset": references.dataset,
                "fit_split": references.fit_split,
                "session_count": references.session_count,
                "feature_count": len(references.features),
                "reference_id": references.reference_id,
                "output": str(args.output.resolve()),
            }
        )
        return 0

    if args.command == "compile":
        summary = compile_dataset(
            root=args.root,
            config=config,
            reference_path=args.reference,
            output_root=args.output_root,
            split_file=args.split_file,
            split_name=args.split_name,
            split_value=args.split_value,
            max_sessions=args.max_sessions,
            workers=args.workers,
            protocol_version=args.protocol_version,
            overwrite=args.overwrite,
            progress=_print_progress,
        )
        _print_result(summary)
        return 0

    if args.command == "compile-transcript":
        summary = compile_transcript_dataset(
            root=args.root,
            config=config,
            evidence_root=args.evidence_root,
            output_root=args.output_root,
            split_file=args.split_file,
            split_value=args.split_value,
            max_sessions=args.max_sessions,
            overwrite=args.overwrite,
            progress=_print_progress,
        )
        _print_result(summary)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
