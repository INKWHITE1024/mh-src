"""Held-out contract and grounding gate for Evidence Literacy SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rethink_mh.rethinking.qwen import QwenThinkerCompletionModel
from rethink_mh.rethinking.query_retrieval import AtomicSelection, EvidenceQuery

from .evidence_literacy import PROTOCOL_VERSION, _canonical_json
from .rethink_post_training import audit_training_records


EVALUATION_PROTOCOL_VERSION = "evidence-literacy-heldout-gate"
_CARD_IDENTIFIER = re.compile(r"(?m)^(E[0-9]{3,6}) \|")
_QUERY_BLOCK = re.compile(r"EVIDENCE QUERY\n(?P<query>\{[^\n]+\})")
_PATTERN_COUNT_LABEL = {
    "unusual measurement": "unusual-measurement candidates",
    "change point": "change-point candidates",
    "quality boundary": "quality-boundary candidates",
    "representative window": "representative-window candidates",
    "cross-modal co-change": "cross-modal co-change candidates",
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read literacy manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("literacy manifest must be one JSON object")
    return value


def _parse_object(text: str) -> dict[str, Any]:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("completion must be one JSON object")
    return value


def _query_is_grounded(query: EvidenceQuery, prompt: str) -> bool:
    if query.segment_id not in prompt:
        return False
    if any(slot not in prompt for slot in query.target_slots):
        return False
    label = _PATTERN_COUNT_LABEL[query.pattern]
    match = re.search(rf"(?m)^- {re.escape(label)}: (?P<count>[0-9]+)$", prompt)
    return match is not None and int(match.group("count")) > 0


def _selection_is_grounded(selection: AtomicSelection, prompt: str) -> bool:
    shown = set(_CARD_IDENTIFIER.findall(prompt))
    if not set(selection.selected_evidence_ids) <= shown:
        return False
    match = _QUERY_BLOCK.search(prompt)
    if match is None:
        return False
    query = EvidenceQuery.from_dict(json.loads(match.group("query")))
    return len(selection.selected_evidence_ids) <= query.budget


def _rate(count: int, denominator: int) -> float:
    return count / denominator if denominator else 0.0


def evaluate_literacy(
    *,
    package_dir: Path,
    model_path: Path,
    adapter_path: Path,
    output_dir: Path,
    device: str = "cuda:0",
    limit: int | None = None,
    minimum_query_grounded_rate: float = 0.90,
    minimum_selection_grounded_rate: float = 0.90,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite literacy evaluation: {output_dir}"
        )
    for name, value in (
        ("minimum_query_grounded_rate", minimum_query_grounded_rate),
        ("minimum_selection_grounded_rate", minimum_selection_grounded_rate),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be from 0 to 1")
    records_path = package_dir / "records.jsonl"
    manifest_path = package_dir / "manifest.json"
    audited = audit_training_records("evidence_literacy_sft", records_path)
    manifest = _read_manifest(manifest_path)
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("literacy package has an unexpected protocol")
    if manifest.get("partition") != "holdout":
        raise ValueError("literacy evaluation requires a holdout package")
    boundaries = manifest.get("fit_boundaries")
    if not isinstance(boundaries, Mapping) or any(
        (
            boundaries.get("sample_targets_accessed") is not False,
            boundaries.get("annotator_uses_outcomes") is not False,
            boundaries.get("label_fields_in_model_text") is not False,
        )
    ):
        raise ValueError("literacy holdout package has unsafe fit boundaries")
    rows = list(audited.rows)
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        rows = rows[:limit]

    model = QwenThinkerCompletionModel.from_pretrained(
        model_path,
        adapter_path=adapter_path,
        device=device,
        max_new_tokens=192,
        max_input_tokens=6_000,
    )
    output_rows: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    for index, row in enumerate(rows):
        contract_type = str(row["contract_type"])
        prompt_messages = [dict(message) for message in row["messages"][:-1]]
        prompt_text = "\n".join(message["content"] for message in prompt_messages)
        expected = str(row["messages"][-1]["content"])
        raw = model.complete(prompt_messages)
        contract_valid = False
        grounded = False
        exact = False
        error: str | None = None
        try:
            parsed = _parse_object(raw)
            canonical = _canonical_json(parsed)
            if contract_type == "evidence_query":
                query = EvidenceQuery.from_dict(parsed)
                grounded = _query_is_grounded(query, prompt_text)
            elif contract_type == "atomic_selection":
                selection = AtomicSelection.from_dict(parsed)
                grounded = _selection_is_grounded(selection, prompt_text)
            else:  # pragma: no cover - audited before model load
                raise ValueError(f"unsupported contract type {contract_type}")
            contract_valid = True
            exact = canonical == expected
        except (ValueError, TypeError, json.JSONDecodeError) as caught:
            error = f"{caught.__class__.__name__}: {caught}"

        totals[f"{contract_type}.count"] += 1
        totals[f"{contract_type}.contract_valid"] += int(contract_valid)
        totals[f"{contract_type}.grounded"] += int(contract_valid and grounded)
        totals[f"{contract_type}.exact"] += int(exact)
        opaque_id = hashlib.sha256(
            f"{row['session_id']}:{index}:{contract_type}".encode("utf-8")
        ).hexdigest()[:20]
        output_rows.append(
            {
                "record_id": opaque_id,
                "contract_type": contract_type,
                "contract_valid": contract_valid,
                "grounded": bool(contract_valid and grounded),
                "annotator_exact_match": exact,
                "completion_sha256": hashlib.sha256(
                    raw.encode("utf-8")
                ).hexdigest(),
                "error": error,
            }
        )

    query_count = totals["evidence_query.count"]
    selection_count = totals["atomic_selection.count"]
    query_contract_rate = _rate(
        totals["evidence_query.contract_valid"], query_count
    )
    query_grounded_rate = _rate(totals["evidence_query.grounded"], query_count)
    selection_contract_rate = _rate(
        totals["atomic_selection.contract_valid"], selection_count
    )
    selection_grounded_rate = _rate(
        totals["atomic_selection.grounded"], selection_count
    )
    full_holdout = limit is None or limit >= len(audited.rows)
    gate = {
        "eligible": full_holdout,
        "minimum_query_grounded_rate": minimum_query_grounded_rate,
        "minimum_selection_grounded_rate": minimum_selection_grounded_rate,
        "query_contract_and_grounding_passed": (
            query_contract_rate >= minimum_query_grounded_rate
            and query_grounded_rate >= minimum_query_grounded_rate
        ),
        "selection_contract_and_grounding_passed": (
            selection_contract_rate >= minimum_selection_grounded_rate
            and selection_grounded_rate >= minimum_selection_grounded_rate
        ),
    }
    gate["passed"] = bool(
        gate["eligible"]
        and gate["query_contract_and_grounding_passed"]
        and gate["selection_contract_and_grounding_passed"]
    )
    summary = {
        "schema_version": "1.0.0",
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "training_protocol_version": PROTOCOL_VERSION,
        "dataset": audited.dataset,
        "partition": "holdout",
        "record_count": len(rows),
        "session_count": audited.session_count,
        "model_path": str(model_path.resolve()),
        "adapter_path": str(adapter_path.resolve()),
        "package_manifest_sha256": _sha256_file(manifest_path),
        "records_sha256": _sha256_file(records_path),
        "metrics": {
            "evidence_query": {
                "count": query_count,
                "contract_valid_rate": query_contract_rate,
                "grounded_rate": query_grounded_rate,
                "annotator_exact_match_rate": _rate(
                    totals["evidence_query.exact"], query_count
                ),
            },
            "atomic_selection": {
                "count": selection_count,
                "contract_valid_rate": selection_contract_rate,
                "grounded_rate": selection_grounded_rate,
                "annotator_exact_match_rate": _rate(
                    totals["atomic_selection.exact"], selection_count
                ),
            },
        },
        "gate": gate,
        "fit_boundaries": {
            "sample_targets_accessed": False,
            "annotator_uses_outcomes": False,
            "label_fields_in_model_text": False,
            "heldout_from_literacy_training": True,
            "contract_repairs_allowed": False,
        },
        "interpretation_boundary": (
            "Passing this gate permits a larger literacy experiment only. It "
            "does not validate classification, evidence utility, or revision."
        ),
    }
    output_dir.mkdir(parents=True)
    (output_dir / "predictions.jsonl").write_text(
        "".join(_canonical_json(row) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the held-out Evidence Literacy contract gate."
    )
    parser.add_argument("--package-dir", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--adapter-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--minimum-query-grounded-rate", type=float, default=0.90)
    parser.add_argument(
        "--minimum-selection-grounded-rate",
        type=float,
        default=0.90,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = evaluate_literacy(
        package_dir=args.package_dir,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        device=args.device,
        limit=args.limit,
        minimum_query_grounded_rate=args.minimum_query_grounded_rate,
        minimum_selection_grounded_rate=args.minimum_selection_grounded_rate,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
