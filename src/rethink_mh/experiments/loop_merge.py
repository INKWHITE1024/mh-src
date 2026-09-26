"""Freeze complete label-free OOF trajectories before any outcome join."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rethink_mh.rethinking.evidence_agent import (
    verify_evidence_access_ledger,
)

from .loop_collect import (
    ARM_ORDER,
    COLLECTION_PROTOCOL_VERSION,
)
from .loop_prepare import PREPARATION_PROTOCOL_VERSION


MERGE_PROTOCOL_VERSION = "loop-raw-oof-freeze"
_SAMPLE_SUPERVISION_KEYS = frozenset(
    {
        "label",
        "target",
        "groundtruth",
        "goldlabel",
        "samplelabel",
        "truelabel",
        "ytrue",
        "frozenv2probability",
        "frozenv2predictedlabel",
        "utility",
    }
)
_RELEASE_ARMS = frozenset(ARM_ORDER) - {"reflection_no_new_evidence"}


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
    output: list[dict[str, Any]] = []
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
        output.append(value)
    if not output:
        raise ValueError(f"{description} is empty: {path}")
    return output


def _normalized_key(value: object) -> str:
    return "".join(
        character
        for character in str(value).casefold()
        if character.isalnum()
    )


def _find_sample_supervision(value: Any, path: str = "root") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _normalized_key(key) in _SAMPLE_SUPERVISION_KEYS:
                found.append(f"{path}.{key}")
            found.extend(
                _find_sample_supervision(child, f"{path}.{key}")
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(
                _find_sample_supervision(child, f"{path}[{index}]")
            )
    return found


def _same_probability_exact(left: object, right: object) -> bool:
    if (
        isinstance(left, bool)
        or isinstance(right, bool)
        or not isinstance(left, (int, float))
        or not isinstance(right, (int, float))
    ):
        return False
    return float(left) == float(right)


@dataclass(frozen=True, slots=True)
class LoopMergeRequest:
    prepared_inference_dir: Path
    fold_dirs: tuple[Path, ...]
    output_dir: Path
    expected_fold_count: int = 5
    expected_session_count: int = 107
    collection_protocol_version: str = COLLECTION_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        for name in ("expected_fold_count", "expected_session_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if len(self.fold_dirs) != self.expected_fold_count:
            raise ValueError(
                "fold_dirs count must equal expected_fold_count"
            )
        if self.collection_protocol_version != COLLECTION_PROTOCOL_VERSION:
            raise ValueError(
                "collection_protocol_version must be "
                f"{COLLECTION_PROTOCOL_VERSION}"
            )


def _verify_ledger(
    path: Path,
    expected: Mapping[str, Any],
) -> None:
    verification = verify_evidence_access_ledger(path)
    if (
        verification.to_dict() != dict(expected)
        or not verification.sealed
        or verification.aborted
    ):
        raise ValueError(f"ledger failed sealed verification: {path}")


def _validate_row(
    row: Mapping[str, Any],
    plan: Mapping[str, Any],
    fold_dir: Path,
    collection_protocol_version: str,
) -> int:
    session_id = str(plan["session_id"])
    if any(
        (
            row.get("collection_protocol_version")
            != collection_protocol_version,
            row.get("session_id") != session_id,
            row.get("outer_fold") != plan["outer_fold"],
            row.get("dataset") != plan["dataset"],
            row.get("ground_truth_used") is not False,
        )
    ):
        raise ValueError(f"raw trajectory identity mismatch for {session_id}")
    forbidden = _find_sample_supervision(row)
    if forbidden:
        raise ValueError(
            f"raw trajectory contains sample supervision for {session_id}: "
            f"{forbidden[:8]}"
        )
    if row.get("status") == "initial_failed":
        if row.get("arms") != []:
            raise ValueError(f"initial failure contains arms for {session_id}")
        return 0
    if row.get("status") not in {
        "complete",
        "complete_with_fail_closed_arms",
    }:
        raise ValueError(f"unknown raw status for {session_id}")
    initial = row.get("initial")
    arms = row.get("arms")
    if not isinstance(initial, Mapping) or not isinstance(arms, list):
        raise ValueError(f"raw trajectory is incomplete for {session_id}")
    probability_matches = _same_probability_exact
    if not probability_matches(
        initial.get("initial_probability_anchor"),
        plan.get("initial_probability"),
    ) or not probability_matches(
        initial.get("initial_assessment", {}).get("risk_probability"),
        plan.get("initial_probability"),
    ):
        raise ValueError(f"initial OOF anchor changed exactly for {session_id}")
    initial_ledger = (
        fold_dir / "sessions" / session_id / "ledgers" / "initial.jsonl"
    )
    _verify_ledger(initial_ledger, initial["access_ledger"])
    if [arm.get("arm_id") for arm in arms] != list(ARM_ORDER):
        raise ValueError(f"arm order or coverage changed for {session_id}")
    ledger_count = 1
    for arm in arms:
        arm_id = str(arm["arm_id"])
        if any(
            (
                arm.get("session_id") != session_id,
                arm.get("outer_fold") != plan["outer_fold"],
                arm.get("ground_truth_used") is not False,
                arm.get("status") not in {"ok", "failed"},
            )
        ):
            raise ValueError(f"invalid arm identity: {session_id}/{arm_id}")
        ledger_path = (
            fold_dir
            / "sessions"
            / session_id
            / "ledgers"
            / f"{arm_id}.jsonl"
        )
        if arm["status"] == "ok":
            workflow = arm.get("workflow")
            if not isinstance(workflow, Mapping):
                raise ValueError(
                    f"successful arm omitted workflow: {session_id}/{arm_id}"
                )
            if arm_id in _RELEASE_ARMS:
                ledger = workflow.get("access_ledger")
                if not isinstance(ledger, Mapping):
                    raise ValueError(
                        f"released arm omitted ledger: {session_id}/{arm_id}"
                    )
                _verify_ledger(ledger_path, ledger)
                ledger_count += 1
            elif (
                workflow.get("access_ledger") is not None
                or ledger_path.exists()
                or workflow.get("budget")
                != {
                    "queries_used": 0,
                    "atomic_records_released": 0,
                    "released_tokens": 0,
                }
            ):
                raise ValueError(
                    f"reflection arm released evidence: {session_id}"
                )
        elif "failure" not in arm:
            raise ValueError(
                f"failed arm omitted reason: {session_id}/{arm_id}"
            )
    return ledger_count


def merge_and_freeze(request: LoopMergeRequest) -> dict[str, Any]:
    """Verify every fold and freeze one label-free raw OOF artifact."""

    if request.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite raw OOF freeze: {request.output_dir}"
        )
    inference_manifest_path = request.prepared_inference_dir / "manifest.json"
    inference_plan_path = (
        request.prepared_inference_dir / "inference_plan.jsonl"
    )
    inference_manifest = _read_json(
        inference_manifest_path,
        "inference manifest",
    )
    plan_rows = _read_jsonl(inference_plan_path, "inference plan")
    if any(
        (
            inference_manifest.get("preparation_protocol_version")
            != PREPARATION_PROTOCOL_VERSION,
            inference_manifest.get("fold_count")
            != request.expected_fold_count,
            inference_manifest.get("session_count")
            != request.expected_session_count,
            inference_manifest.get("files", {})
            .get("inference_plan.jsonl", {})
            .get("sha256")
            != _sha256(inference_plan_path),
        )
    ):
        raise ValueError("inference preparation manifest failed verification")
    plan_by_id = {
        str(row.get("session_id", "")): row for row in plan_rows
    }
    if (
        len(plan_by_id) != request.expected_session_count
        or "" in plan_by_id
    ):
        raise ValueError("inference plan has duplicate or missing sessions")

    fold_indices: set[int] = set()
    collected: list[dict[str, Any]] = []
    source_manifests: list[dict[str, Any]] = []
    seen: set[str] = set()
    ledger_count = 0
    task_hashes: set[str] = set()
    expected_preparation_hash = _sha256(inference_manifest_path)
    for fold_dir in request.fold_dirs:
        manifest_path = fold_dir / "manifest.json"
        raw_path = fold_dir / "trajectories.raw.jsonl"
        manifest = _read_json(manifest_path, "fold collection manifest")
        rows = _read_jsonl(raw_path, "fold raw trajectories")
        fold = manifest.get("outer_fold")
        if isinstance(fold, bool) or not isinstance(fold, int):
            raise ValueError(f"invalid outer fold in {manifest_path}")
        if any(
            (
                manifest.get("collection_protocol_version")
                != request.collection_protocol_version,
                manifest.get("limited_check") is not False,
                manifest.get("sources", {}).get(
                    "preparation_manifest_sha256"
                )
                != expected_preparation_hash,
                manifest.get("files", {})
                .get("trajectories.raw.jsonl", {})
                .get("sha256")
                != _sha256(raw_path),
                manifest.get("session_count") != len(rows),
            )
        ):
            raise ValueError(f"fold collection failed verification: {fold_dir}")
        if fold in fold_indices:
            raise ValueError(f"duplicate fold collection: {fold}")
        fold_indices.add(fold)
        task_hashes.add(str(manifest.get("sources", {}).get("task_sha256")))
        for row in rows:
            session_id = str(row.get("session_id", ""))
            if session_id in seen or session_id not in plan_by_id:
                raise ValueError(
                    f"duplicate or unknown collected session: {session_id}"
                )
            plan = plan_by_id[session_id]
            if plan.get("outer_fold") != fold:
                raise ValueError(
                    f"session collected by wrong fold: {session_id}"
                )
            seen.add(session_id)
            ledger_count += _validate_row(
                row,
                plan,
                fold_dir,
                request.collection_protocol_version,
            )
            collected.append(row)
        source_manifests.append(
            {
                "outer_fold": fold,
                "path": str(manifest_path.resolve()),
                "sha256": _sha256(manifest_path),
                "raw_sha256": _sha256(raw_path),
                "session_count": len(rows),
            }
        )
    if fold_indices != set(range(request.expected_fold_count)):
        raise ValueError("fold collections do not cover contiguous outer folds")
    if seen != set(plan_by_id):
        raise ValueError("fold collections do not cover the inference cohort")
    if len(task_hashes) != 1 or "" in task_hashes:
        raise ValueError("fold collections used different task specifications")

    ordered = sorted(collected, key=lambda row: str(row["session_id"]))
    raw_text = "".join(_canonical_json(row) + "\n" for row in ordered)
    status_counts = Counter(str(row["status"]) for row in ordered)
    arm_status_counts = Counter(
        str(arm["status"])
        for row in ordered
        for arm in row.get("arms", [])
    )
    request.output_dir.mkdir(parents=True)
    raw_path = request.output_dir / "trajectories.raw.jsonl"
    raw_path.write_text(raw_text, encoding="utf-8")
    manifest = {
        "schema_version": "1.0.0",
        "merge_protocol_version": MERGE_PROTOCOL_VERSION,
        "collection_protocol_version": request.collection_protocol_version,
        "preparation_protocol_version": PREPARATION_PROTOCOL_VERSION,
        "dataset": plan_rows[0]["dataset"],
        "fold_count": request.expected_fold_count,
        "session_count": len(ordered),
        "session_status_counts": dict(sorted(status_counts.items())),
        "arm_status_counts": dict(sorted(arm_status_counts.items())),
        "verified_ledger_count": ledger_count,
        "arm_order": list(ARM_ORDER),
        "files": {
            "trajectories.raw.jsonl": {
                "sha256": _sha256(raw_path),
                "row_count": len(ordered),
            }
        },
        "sources": {
            "inference_manifest_path": str(
                inference_manifest_path.resolve()
            ),
            "inference_manifest_sha256": expected_preparation_hash,
            "inference_plan_sha256": _sha256(inference_plan_path),
            "task_sha256": next(iter(task_hashes)),
            "fold_collections": sorted(
                source_manifests,
                key=lambda item: item["outer_fold"],
            ),
        },
        "fit_boundaries": {
            "current_sample_targets_accessed": False,
            "frozen_reference_accessed": False,
            "utility_accessed": False,
            "raw_trajectories_frozen_before_outcome_join": True,
            "dev_accessed": False,
            "test_accessed": False,
        },
        "interpretation_boundary": (
            "This artifact freezes label-free raw OOF trajectories. Failures "
            "remain fail-closed and no outcome-dependent arm selection has run."
        ),
    }
    manifest_path = request.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (request.output_dir / "RAW_OOF_FROZEN").touch()
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify and freeze label-free full-loop OOF trajectories."
    )
    parser.add_argument(
        "--prepared-inference-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--fold-dir",
        required=True,
        action="append",
        type=Path,
        dest="fold_dirs",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-fold-count", type=int, default=5)
    parser.add_argument("--expected-session-count", type=int, default=107)
    parser.add_argument(
        "--collection-protocol-version",
        choices=(COLLECTION_PROTOCOL_VERSION,),
        default=COLLECTION_PROTOCOL_VERSION,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = merge_and_freeze(
        LoopMergeRequest(
            prepared_inference_dir=args.prepared_inference_dir,
            fold_dirs=tuple(args.fold_dirs),
            output_dir=args.output_dir,
            expected_fold_count=args.expected_fold_count,
            expected_session_count=args.expected_session_count,
            collection_protocol_version=args.collection_protocol_version,
        )
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
