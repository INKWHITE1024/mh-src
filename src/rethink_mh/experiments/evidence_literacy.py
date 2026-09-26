"""Build and evaluate label-free Evidence Agent literacy supervision."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rethink_mh.rethinking.contracts import InitialAssessment, TaskSpec
from rethink_mh.rethinking.native_query_retrieval import NativeAtomicRetriever
from rethink_mh.rethinking.query_retrieval import (
    RETRIEVAL_PATTERNS,
    AtomicSelection,
    CandidateSet,
    EvidenceQuery,
)
from rethink_mh.rethinking.retrieval_prompts import QueryRetrievalPromptBuilder
from rethink_mh.textualization.canonical_slots import CORE_SLOTS

from .rethink_post_training import audit_training_records


PROTOCOL_VERSION = "evidence-literacy-sft"
ContractType = Literal["evidence_query", "atomic_selection"]
_AVAILABLE_SLOTS = re.compile(r"^- available fixed core slots: (?P<slots>.+)$")
_PATTERN_PURPOSE = {
    "unusual measurement": "verify contradictory detail",
    "change point": "inspect temporal change",
    "quality boundary": "inspect measurement reliability",
    "representative window": "verify supporting detail",
    "cross-modal co-change": "resolve source disagreement",
}
_PATTERN_ORDER = tuple(RETRIEVAL_PATTERNS)
_SLOT_MODALITY = {slot.label: slot.modality for slot in CORE_SLOTS}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be one JSON object: {path}")
    return value


def _read_task(path: Path) -> TaskSpec:
    return TaskSpec.from_dict(_read_json_object(path, "task specification"))


def _read_assignment(path: Path, partition: str) -> tuple[str, ...]:
    if partition not in {"fit", "holdout"}:
        raise ValueError("partition must be fit or holdout")
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["session_id", "split"]:
                raise ValueError(
                    "assignment columns must be exactly session_id,split"
                )
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise ValueError(f"cannot read assignment {path}: {error}") from error
    identifiers: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, 2):
        session_id = str(row.get("session_id", "")).strip()
        split = str(row.get("split", "")).strip()
        if not session_id or split not in {"fit", "holdout"}:
            raise ValueError(f"invalid assignment row {index}")
        if session_id in seen:
            raise ValueError(f"duplicate assignment session {session_id}")
        seen.add(session_id)
        if split == partition:
            identifiers.append(session_id)
    if not identifiers:
        raise ValueError(f"assignment contains no {partition} sessions")
    return tuple(sorted(identifiers))


def _available_slots(retriever: NativeAtomicRetriever, segment_id: str) -> tuple[str, ...]:
    text = retriever.render_retrieval_map((segment_id,))
    for line in text.splitlines():
        match = _AVAILABLE_SLOTS.fullmatch(line)
        if match is None:
            continue
        value = match.group("slots")
        if value == "none":
            return ()
        parsed = tuple(item.strip() for item in value.split(",") if item.strip())
        unknown = set(parsed) - set(_SLOT_MODALITY)
        if unknown:
            raise ValueError(
                f"retrieval map contains unknown fixed core slots: {sorted(unknown)}"
            )
        return parsed
    raise ValueError(f"retrieval map omitted available slots for {segment_id}")


def _query_slot_sets(
    slots: Sequence[str],
    pattern: str,
) -> tuple[tuple[str, ...], ...]:
    if pattern != "cross-modal co-change":
        return tuple((slot,) for slot in slots)
    audio = [slot for slot in slots if _SLOT_MODALITY[slot] == "audio"]
    visual = [slot for slot in slots if _SLOT_MODALITY[slot] == "visual"]
    return tuple((left, right) for left in audio for right in visual)


def _pattern_relevance(candidate_set: CandidateSet) -> float:
    return max(
        (
            dict(candidate.score_components).get("pattern relevance", 0.0)
            for candidate in candidate_set.candidates
        ),
        default=0.0,
    )


def _candidate_score(candidate_set: CandidateSet) -> tuple[float, float, int]:
    scores = [candidate.ranking_score for candidate in candidate_set.candidates]
    return (
        _pattern_relevance(candidate_set),
        sum(scores[:2]),
        len(candidate_set.candidates),
    )


def _best_query_for(
    retriever: NativeAtomicRetriever,
    segment_id: str,
    slots: Sequence[str],
    pattern: str,
    *,
    budget: int,
) -> tuple[EvidenceQuery, CandidateSet] | None:
    choices: list[
        tuple[
            tuple[float, float, int],
            tuple[str, ...],
            EvidenceQuery,
            CandidateSet,
        ]
    ] = []
    for target_slots in _query_slot_sets(slots, pattern):
        query = EvidenceQuery(
            segment_id=segment_id,
            purpose=_PATTERN_PURPOSE[pattern],  # type: ignore[arg-type]
            target_slots=target_slots,
            pattern=pattern,  # type: ignore[arg-type]
            budget=budget,
        )
        try:
            candidates = retriever.shortlist(query)
        except (ValueError, TypeError):
            continue
        score = _candidate_score(candidates)
        if pattern != "representative window" and score[0] <= 0.0:
            continue
        choices.append((score, target_slots, query, candidates))
    if not choices:
        return None
    choices.sort(
        key=lambda item: (
            -item[0][0],
            -item[0][1],
            -item[0][2],
            item[1],
        )
    )
    return choices[0][2], choices[0][3]


def _curriculum_queries(
    retriever: NativeAtomicRetriever,
    session_id: str,
    *,
    maximum: int,
    budget: int,
) -> tuple[tuple[EvidenceQuery, CandidateSet], ...]:
    best_by_pattern: dict[str, tuple[tuple[float, float, int], EvidenceQuery, CandidateSet]] = {}
    for segment_id in retriever.available_segment_ids:
        slots = _available_slots(retriever, segment_id)
        if not slots:
            continue
        for pattern in _PATTERN_ORDER:
            result = _best_query_for(
                retriever,
                segment_id,
                slots,
                pattern,
                budget=budget,
            )
            if result is None:
                continue
            query, candidates = result
            score = _candidate_score(candidates)
            previous = best_by_pattern.get(pattern)
            candidate_key = (
                score,
                tuple(reversed(query.target_slots)),
                query.segment_id,
            )
            previous_key = (
                (
                    previous[0],
                    tuple(reversed(previous[1].target_slots)),
                    previous[1].segment_id,
                )
                if previous is not None
                else None
            )
            if previous_key is None or candidate_key > previous_key:
                best_by_pattern[pattern] = (score, query, candidates)
    if not best_by_pattern:
        raise ValueError(f"session {session_id} has no feasible literacy query")

    offset = int(hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:8], 16)
    offset %= len(_PATTERN_ORDER)
    rotated = _PATTERN_ORDER[offset:] + _PATTERN_ORDER[:offset]
    selected = [
        (best_by_pattern[pattern][1], best_by_pattern[pattern][2])
        for pattern in rotated
        if pattern in best_by_pattern
    ][:maximum]
    return tuple(selected)


def enumerate_label_free_queries(
    retriever: NativeAtomicRetriever,
    session_id: str,
    *,
    maximum: int = len(_PATTERN_ORDER),
    budget: int = 2,
) -> tuple[tuple[EvidenceQuery, CandidateSet], ...]:
    """Expose the deterministic, outcome-free query catalog to loop collection."""

    return _curriculum_queries(
        retriever,
        session_id,
        maximum=maximum,
        budget=budget,
    )


def best_label_free_query(
    retriever: NativeAtomicRetriever,
    session_id: str,
    *,
    budget: int = 2,
) -> tuple[EvidenceQuery, CandidateSet]:
    """Return the highest-potential deterministic query without sample outcomes."""

    candidates = enumerate_label_free_queries(
        retriever,
        session_id,
        maximum=len(_PATTERN_ORDER),
        budget=budget,
    )
    return sorted(
        candidates,
        key=lambda item: (
            -_candidate_score(item[1])[0],
            -_candidate_score(item[1])[1],
            -_candidate_score(item[1])[2],
            _canonical_json(item[0].to_dict()),
        ),
    )[0]


def _initial_for(segment_id: str) -> InitialAssessment:
    return InitialAssessment.from_dict(
        {
            "risk_probability": 0.5,
            "confidence": 0.5,
            "audio_source": {
                "status": "available",
                "risk_probability": 0.5,
                "reliability": 0.5,
            },
            "visual_source": {
                "status": "available",
                "risk_probability": 0.5,
                "reliability": 0.5,
            },
            "supporting_segment_ids": [],
            "contradictory_segment_ids": [],
            "uncertain_segment_ids": [segment_id],
            "requested_segment_ids": [segment_id],
        }
    )


def _decision_for(segment_id: str) -> dict[str, Any]:
    return {
        "should_rethink": True,
        "trigger_score": 0.5,
        "reasons": ["label-free evidence literacy curriculum"],
        "selected_segment_ids": [segment_id],
        "metrics": {},
    }


def _supervision() -> dict[str, Any]:
    return {
        "policy": "deterministic_native_retriever",
        "sample_targets_accessed": False,
        "annotator_uses_outcomes": False,
        "used_in_model_text": False,
    }


def _record(
    *,
    contract_type: ContractType,
    dataset: str,
    session_id: str,
    partition: str,
    messages: Sequence[Mapping[str, str]],
    completion: Mapping[str, Any],
    bundle_sha256: str,
) -> dict[str, Any]:
    normalized = [dict(message) for message in messages]
    normalized.append(
        {
            "role": "assistant",
            "content": _canonical_json(completion),
        }
    )
    return {
        "record_type": "evidence_literacy_sft",
        "contract_type": contract_type,
        "dataset": dataset,
        "session_id": session_id,
        "partition": partition,
        "messages": normalized,
        "sample_weight": 1.0,
        "supervision": _supervision(),
        "provenance": {
            "protocol_version": PROTOCOL_VERSION,
            "evidence_protocol": "native",
            "bundle_sha256": bundle_sha256,
            "annotator_query_or_selection_sha256": _sha256_bytes(
                _canonical_json(completion).encode("utf-8")
            ),
        },
    }


@dataclass(frozen=True, slots=True)
class LiteracyPackageRequest:
    dataset: str
    task_path: Path
    assignment_path: Path
    evidence_root: Path
    output_dir: Path
    partition: Literal["fit", "holdout"] = "fit"
    maximum_queries_per_session: int = 2
    query_budget: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str) or not self.dataset:
            raise ValueError("dataset must be non-empty text")
        if self.partition not in {"fit", "holdout"}:
            raise ValueError("partition must be fit or holdout")
        if (
            isinstance(self.maximum_queries_per_session, bool)
            or not isinstance(self.maximum_queries_per_session, int)
            or not 1 <= self.maximum_queries_per_session <= len(_PATTERN_ORDER)
        ):
            raise ValueError(
                "maximum_queries_per_session must be between 1 and "
                f"{len(_PATTERN_ORDER)}"
            )
        if (
            isinstance(self.query_budget, bool)
            or not isinstance(self.query_budget, int)
            or not 1 <= self.query_budget <= 4
        ):
            raise ValueError("query_budget must be between 1 and 4")


def build_literacy_package(request: LiteracyPackageRequest) -> dict[str, Any]:
    if request.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite literacy package: {request.output_dir}"
        )
    task = _read_task(request.task_path)
    session_ids = _read_assignment(request.assignment_path, request.partition)
    rows: list[dict[str, Any]] = []
    bundle_fingerprints: list[str] = []
    pattern_counts: Counter[str] = Counter()
    for session_id in session_ids:
        session_dir = request.evidence_root / session_id
        retriever = NativeAtomicRetriever.from_session_dir(session_dir)
        bundle_fingerprints.append(retriever.session_fingerprint)
        curriculum = _curriculum_queries(
            retriever,
            session_id,
            maximum=request.maximum_queries_per_session,
            budget=request.query_budget,
        )
        for query, candidate_set in curriculum:
            initial = _initial_for(query.segment_id)
            query_messages = QueryRetrievalPromptBuilder.build_query_messages(
                task,
                initial,
                _decision_for(query.segment_id),
                retriever.render_retrieval_map((query.segment_id,)),
            )
            rows.append(
                _record(
                    contract_type="evidence_query",
                    dataset=request.dataset,
                    session_id=session_id,
                    partition=request.partition,
                    messages=query_messages,
                    completion=query.to_dict(),
                    bundle_sha256=retriever.session_fingerprint,
                )
            )
            selected_ids = candidate_set.candidate_evidence_ids[
                : min(query.budget, len(candidate_set.candidates))
            ]
            selection = AtomicSelection(selected_ids)
            selection_messages = (
                QueryRetrievalPromptBuilder.build_selection_messages(
                    task,
                    initial,
                    candidate_set,
                )
            )
            rows.append(
                _record(
                    contract_type="atomic_selection",
                    dataset=request.dataset,
                    session_id=session_id,
                    partition=request.partition,
                    messages=selection_messages,
                    completion=selection.to_dict(),
                    bundle_sha256=retriever.session_fingerprint,
                )
            )
            pattern_counts[query.pattern] += 1

    records_text = "".join(_canonical_json(row) + "\n" for row in rows)
    session_hash = _sha256_bytes(
        "\n".join(session_ids).encode("utf-8")
    )
    bundle_hash = _sha256_bytes(
        "\n".join(bundle_fingerprints).encode("utf-8")
    )
    manifest = {
        "schema_version": "1.0.0",
        "protocol_version": PROTOCOL_VERSION,
        "dataset": request.dataset,
        "partition": request.partition,
        "session_count": len(session_ids),
        "record_count": len(rows),
        "record_type_counts": dict(
            sorted(Counter(row["contract_type"] for row in rows).items())
        ),
        "query_pattern_counts": dict(sorted(pattern_counts.items())),
        "query_budget": request.query_budget,
        "maximum_queries_per_session": request.maximum_queries_per_session,
        "task_sha256": _sha256_file(request.task_path),
        "assignment_sha256": _sha256_file(request.assignment_path),
        "session_ids_sha256": session_hash,
        "source_bundle_fingerprints_sha256": bundle_hash,
        "files": {
            "records.jsonl": {
                "sha256": _sha256_bytes(records_text.encode("utf-8")),
                "row_count": len(rows),
            }
        },
        "fit_boundaries": {
            "sample_targets_accessed": False,
            "annotator_uses_outcomes": False,
            "label_fields_in_model_text": False,
            "single_dataset": True,
            "evidence_protocol": "native",
        },
        "interpretation_boundary": (
            "This package supervises JSON/query/grounding mechanics only. "
            "It does not supervise screening correctness or revision utility."
        ),
    }
    request.output_dir.mkdir(parents=True)
    records_path = request.output_dir / "records.jsonl"
    records_path.write_text(records_text, encoding="utf-8")
    audit_training_records("evidence_literacy_sft", records_path)
    (request.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build label-free Evidence Literacy SFT records."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--assignment", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--partition",
        choices=("fit", "holdout"),
        default="fit",
    )
    parser.add_argument("--maximum-queries-per-session", type=int, default=2)
    parser.add_argument("--query-budget", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_literacy_package(
        LiteracyPackageRequest(
            dataset=args.dataset,
            task_path=args.task,
            assignment_path=args.assignment,
            evidence_root=args.evidence_root,
            output_dir=args.output_dir,
            partition=args.partition,
            maximum_queries_per_session=args.maximum_queries_per_session,
            query_budget=args.query_budget,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
