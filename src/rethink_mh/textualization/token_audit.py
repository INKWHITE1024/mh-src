"""Audit token counts of compiled evidence text against label-free semantic baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SEMANTIC_PHRASES = (
    "evidence",
    "segment",
    "audio summary",
    "visual summary",
    "training reference",
    "percentile",
    "valid coverage",
    "participant speaking coverage",
    "transcribed speech coverage",
    "voiced frame coverage",
    "pitch variability",
    "facial action variability",
    "head rotation variability",
    "gaze direction variability",
    "partial face tracking",
    "fixed core slots",
    "not measured by this source",
    "semantic proxy",
    "provisional source-column mapping",
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a non-diagnostic research model. Read the supplied measurement evidence, "
    "respect missingness, and cite only supplied SEGMENT or EVIDENCE identifiers."
)


def _distribution(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "median": 0, "p95": 0, "max": 0}
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "min": int(np.min(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": int(np.max(array)),
    }


def audit_evidence_session(
    session_dir: str | Path,
    tokenizer: Any,
    protocol_version: str = "native",
    max_session_tokens: int = 7_500,
    max_atomic_p95_tokens: int | None = None,
    max_segment_p95_tokens: int | None = None,
) -> dict[str, Any]:
    protocol_version = protocol_version.strip().lower()
    if protocol_version != "native":
        raise ValueError("Token audit supports the native protocol")
    if max_atomic_p95_tokens is None:
        max_atomic_p95_tokens = 512
    if max_segment_p95_tokens is None:
        max_segment_p95_tokens = 768
    directory = Path(session_dir).expanduser().resolve()
    paths = {
        "atomic": directory / f"atomic_evidence.{protocol_version}.txt",
        "segment": directory / f"segment_evidence.{protocol_version}.txt",
        "session": directory / f"session_input.{protocol_version}.txt",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {protocol_version} files: {missing}"
        )

    texts = {name: path.read_text(encoding="utf-8") for name, path in paths.items()}

    def encode(text: str) -> list[int]:
        return list(tokenizer.encode(text, add_special_tokens=False))

    line_tokens = {
        "atomic": [len(encode(line)) for line in texts["atomic"].splitlines()],
        "segment": [len(encode(line)) for line in texts["segment"].splitlines()],
    }
    session_ids = encode(texts["session"])
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": texts["session"]},
    ]
    templated_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(templated_ids, "tolist"):
        templated_ids = templated_ids.tolist()
    templated_ids = list(templated_ids)

    unk_id = getattr(tokenizer, "unk_token_id", None)
    unknown_count = (
        sum(token_id == unk_id for token_id in templated_ids)
        if isinstance(unk_id, int) and unk_id >= 0
        else 0
    )
    phrase_report: dict[str, Any] = {}
    for phrase in SEMANTIC_PHRASES:
        token_ids = encode(phrase)
        pieces = tokenizer.convert_ids_to_tokens(token_ids)
        phrase_report[phrase] = {
            "token_count": len(token_ids),
            "token_pieces": pieces,
            "unknown_token_count": (
                sum(token_id == unk_id for token_id in token_ids)
                if isinstance(unk_id, int) and unk_id >= 0
                else 0
            ),
        }

    atomic_distribution = _distribution(line_tokens["atomic"])
    segment_distribution = _distribution(line_tokens["segment"])
    checks = {
        "no_unknown_tokens": unknown_count == 0
        and all(
            item["unknown_token_count"] == 0 for item in phrase_report.values()
        ),
        "session_with_chat_template_within_target": len(templated_ids)
        <= max_session_tokens,
        "atomic_p95_within_target": float(atomic_distribution["p95"])
        <= max_atomic_p95_tokens,
        "segment_p95_within_target": float(segment_distribution["p95"])
        <= max_segment_p95_tokens,
    }
    return {
        "audit_version": "1.0.0",
        "protocol_version": protocol_version,
        "session_directory": str(directory),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", "unknown")),
        "model_max_length": int(getattr(tokenizer, "model_max_length", 0)),
        "thresholds": {
            "max_session_tokens": max_session_tokens,
            "max_atomic_p95_tokens": max_atomic_p95_tokens,
            "max_segment_p95_tokens": max_segment_p95_tokens,
        },
        "tokens": {
            "atomic_records": atomic_distribution,
            "segment_records": segment_distribution,
            "session_input_without_chat_template": len(session_ids),
            "session_input_with_chat_template": len(templated_ids),
            "chat_template_overhead": len(templated_ids) - len(session_ids),
            "unknown_token_count": unknown_count,
        },
        "semantic_phrase_tokenization": phrase_report,
        "checks": checks,
        "passed": all(checks.values()),
    }




def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit an Evidence Text Protocol with the exact training tokenizer."
        )
    )
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--session-dir", required=True, type=Path, nargs="+")
    parser.add_argument(
        "--protocol-version",
        default="native",
        choices=("native",),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-session-tokens", type=int, default=7_500)
    parser.add_argument("--max-atomic-p95-tokens", type=int)
    parser.add_argument("--max-segment-p95-tokens", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - server-only optional dependency
        raise RuntimeError(
            "Token auditing requires transformers in the training environment"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer.expanduser().resolve()),
        trust_remote_code=True,
    )
    reports = [
        audit_evidence_session(
            session_dir,
            tokenizer,
            protocol_version=args.protocol_version,
            max_session_tokens=args.max_session_tokens,
            max_atomic_p95_tokens=args.max_atomic_p95_tokens,
            max_segment_p95_tokens=args.max_segment_p95_tokens,
        )
        for session_dir in args.session_dir
    ]
    output = {
        "protocol_version": args.protocol_version,
        "all_passed": all(report["passed"] for report in reports),
        "reports": reports,
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if output["all_passed"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
