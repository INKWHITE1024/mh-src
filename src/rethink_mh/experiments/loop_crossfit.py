"""Cross-fitted evaluation of a trained revision policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rethink_mh.rethinking.contracts import (
    InitialAssessment,
    RevisionAssessment,
    TaskSpec,
)
from rethink_mh.rethinking.evidence_agent import (
    CompletionAdapter,
    _complete_with_contract_repair,
)
from rethink_mh.rethinking.qwen import QwenThinkerCompletionModel
from rethink_mh.rethinking.workflow import _validate_revision_grounding

from .audit_fit import load_audit_decisions
from .frozen_artifact_manifest import verify_directory_manifest
from .loop_collect import COLLECTION_PROTOCOL_VERSION
from .loop_evaluate import (
    DIRECT_REFER_ARM,
    _bootstrap_delta,
    _load_config,
    _metrics,
    _scored_view,
    _selective_comparison,
)
from .loop_merge import MERGE_PROTOCOL_VERSION
from .loop_prepare import PREPARATION_PROTOCOL_VERSION
from .loop_training_package import (
    PACKAGE_PROTOCOL_VERSION,
    _context_for_arm,
)


CROSSFIT_PREDICTION_PROTOCOL_VERSION = "loop-crossfit-prediction"
CROSSFIT_FREEZE_PROTOCOL_VERSION = "loop-crossfit-freeze"
CROSSFIT_EVALUATION_PROTOCOL_VERSION = "loop-crossfit-evaluation"
PRIMARY_ARM = "targeted_learned"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(_canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_error(error: Exception) -> dict[str, str]:
    return {
        "error_type": error.__class__.__name__,
        "error_message": " ".join(str(error).split())[:800],
    }


def _directory_artifact(
    root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    return verify_directory_manifest(
        root,
        _read_json(manifest_path, "frozen directory manifest"),
    )


@dataclass(frozen=True, slots=True)
class CrossfitCollectRequest:
    raw_oof_dir: Path
    raw_artifact_manifest: Path
    prepared_inference_dir: Path
    training_package_dir: Path
    training_package_artifact_manifest: Path
    training_summary_path: Path
    task_path: Path
    output_dir: Path
    fold_index: int
    device: str = "cuda"
    dtype: str = "bfloat16"
    attention_implementation: str = "sdpa"
    max_new_tokens: int = 420
    max_input_tokens: int = 8_192
    max_contract_retries: int = 2
    audit_decisions: Path | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.fold_index, bool)
            or not isinstance(self.fold_index, int)
            or self.fold_index < 0
        ):
            raise ValueError("fold_index must be a non-negative integer")
        for name in ("max_new_tokens", "max_input_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self.max_contract_retries <= 2:
            raise ValueError("max_contract_retries must be 0, 1, or 2")


ModelLoader = Callable[
    [Path, Path, CrossfitCollectRequest],
    CompletionAdapter,
]


def _default_model_loader(
    model_path: Path,
    adapter_path: Path,
    request: CrossfitCollectRequest,
) -> CompletionAdapter:
    return QwenThinkerCompletionModel.from_pretrained(
        model_path,
        adapter_path=adapter_path,
        device=request.device,
        dtype=request.dtype,
        attention_implementation=request.attention_implementation,
        max_new_tokens=request.max_new_tokens,
        max_input_tokens=request.max_input_tokens,
    )


def _validated_revision(
    payload: Mapping[str, Any],
    targeted: Any,
) -> RevisionAssessment:
    revision = RevisionAssessment.from_dict(payload)
    _validate_revision_grounding(revision, targeted)
    return revision


def _verified_collection_inputs(
    request: CrossfitCollectRequest,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    Path,
    Path,
]:
    raw_artifact = _directory_artifact(
        request.raw_oof_dir,
        request.raw_artifact_manifest,
    )
    package_artifact = _directory_artifact(
        request.training_package_dir,
        request.training_package_artifact_manifest,
    )
    raw_manifest = _read_json(
        request.raw_oof_dir / "manifest.json",
        "raw OOF manifest",
    )
    package_manifest = _read_json(
        request.training_package_dir / "manifest.json",
        "training package manifest",
    )
    training_summary = _read_json(
        request.training_summary_path,
        "training summary",
    )
    plan_manifest = _read_json(
        request.prepared_inference_dir / "manifest.json",
        "prepared inference manifest",
    )
    plan_path = request.prepared_inference_dir / "inference_plan.jsonl"
    if any(
        (
            raw_manifest.get("merge_protocol_version")
            != MERGE_PROTOCOL_VERSION,
            raw_manifest.get("collection_protocol_version")
            != COLLECTION_PROTOCOL_VERSION,
            not (request.raw_oof_dir / "RAW_OOF_FROZEN").is_file(),
            package_manifest.get("package_protocol_version")
            != PACKAGE_PROTOCOL_VERSION,
            package_manifest.get("crossfit", {}).get("enabled") is not True,
            package_manifest.get("crossfit", {}).get("heldout_fold")
            != request.fold_index,
            package_manifest.get("crossfit", {}).get(
                "heldout_outcomes_used_for_training"
            )
            is not False,
            plan_manifest.get("preparation_protocol_version")
            != PREPARATION_PROTOCOL_VERSION,
            plan_manifest.get("files", {})
            .get("inference_plan.jsonl", {})
            .get("sha256")
            != _sha256(plan_path),
        )
    ):
        raise ValueError("crossfit collection inputs failed protocol validation")
    raw_rows = _read_jsonl(
        request.raw_oof_dir / "trajectories.raw.jsonl",
        "raw OOF trajectories",
    )
    plan_rows = _read_jsonl(plan_path, "prepared inference plan")
    raw_by_id = {
        str(row["session_id"]): row
        for row in raw_rows
        if row.get("outer_fold") == request.fold_index
    }
    plan_by_id = {
        str(row["session_id"]): row
        for row in plan_rows
        if row.get("outer_fold") == request.fold_index
    }
    heldout_ids = package_manifest["crossfit"]["heldout_session_ids"]
    if (
        not raw_by_id
        or set(raw_by_id) != set(plan_by_id)
        or set(raw_by_id) != set(heldout_ids)
    ):
        raise ValueError("crossfit heldout cohort identity mismatch")
    record_name = {
        "revision_sft": "revision_sft.jsonl",
        "orpo": "preferences_initial.jsonl",
    }.get(training_summary.get("stage"))
    adapter_path = Path(
        str(training_summary.get("model", {}).get("adapter_dir", ""))
    ).expanduser().resolve()
    model_path = Path(
        str(training_summary.get("model", {}).get("base_model", ""))
    ).expanduser().resolve()
    if any(
        (
            record_name is None,
            training_summary.get("records", {}).get("sha256")
            != package_manifest.get("files", {})
            .get(record_name or "", {})
            .get("sha256"),
            Path(
                str(training_summary.get("records", {}).get("path", ""))
            ).expanduser().resolve()
            != (request.training_package_dir / (record_name or "")).resolve(),
            not (adapter_path / "adapter_config.json").is_file(),
            not model_path.exists(),
        )
    ):
        raise ValueError("crossfit training provenance failed verification")
    return (
        [raw_by_id[key] for key in sorted(raw_by_id)],
        plan_by_id,
        raw_artifact,
        package_artifact,
        model_path,
        adapter_path,
    )


def collect_crossfit_fold(
    request: CrossfitCollectRequest,
    *,
    model_loader: ModelLoader | None = None,
) -> dict[str, Any]:
    """Generate held-out revision predictions without reading outcomes."""

    manifest_path = request.output_dir / "manifest.json"
    predictions_path = request.output_dir / "predictions.raw.jsonl"
    if manifest_path.is_file():
        manifest = _read_json(manifest_path, "crossfit fold manifest")
        if (
            manifest.get("prediction_protocol_version")
            != CROSSFIT_PREDICTION_PROTOCOL_VERSION
            or manifest.get("files", {})
            .get("predictions.raw.jsonl", {})
            .get("sha256")
            != _sha256(predictions_path)
        ):
            raise ValueError("existing crossfit fold failed verification")
        return manifest
    (
        raw_rows,
        plan_by_id,
        raw_artifact,
        package_artifact,
        model_path,
        adapter_path,
    ) = _verified_collection_inputs(request)
    task = TaskSpec.from_dict(_read_json(request.task_path, "task specification"))
    loader = model_loader or _default_model_loader
    model = loader(model_path, adapter_path, request)
    audit_triggered = (
        load_audit_decisions(request.audit_decisions)
        if request.audit_decisions is not None
        else None
    )
    predictions: list[dict[str, Any]] = []
    for raw in raw_rows:
        session_id = str(raw["session_id"])
        initial = InitialAssessment.from_dict(
            raw["initial"]["initial_assessment"]
        )
        indexed = {
            str(arm["arm_id"]): arm
            for arm in raw.get("arms", [])
        }
        source_arm = indexed.get(PRIMARY_ARM)
        common = {
            "schema_version": "1.0.0",
            "prediction_protocol_version": (
                CROSSFIT_PREDICTION_PROTOCOL_VERSION
            ),
            "dataset": raw["dataset"],
            "session_id": session_id,
            "outer_fold": request.fold_index,
            "arm_id": PRIMARY_ARM,
            "initial_probability": initial.risk_probability,
            "initial_confidence": initial.confidence,
            "initial_threshold": raw["initial"]["initial_threshold"],
            "natural_triggered": (
                audit_triggered[session_id]
                if audit_triggered is not None
                else bool(raw["natural_rethink_decision"]["should_rethink"])
            ),
            "ground_truth_used": False,
        }
        if not isinstance(source_arm, Mapping) or source_arm.get("status") != "ok":
            failure = (
                source_arm.get("failure", {})
                if isinstance(source_arm, Mapping)
                else {"error_type": "MissingRawArm"}
            )
            predictions.append(
                {
                    **common,
                    "status": "fail_closed",
                    "failure": {
                        "error_type": str(
                            failure.get("error_type", "RawArmFailure")
                        ),
                        "error_message": str(
                            failure.get(
                                "error_message",
                                "frozen targeted arm is unavailable",
                            )
                        ),
                    },
                    "final_probability": initial.risk_probability,
                    "final_confidence": initial.confidence,
                    "referred": True,
                    "revision_status": "fail_closed_refer",
                    "queries": 0,
                    "atomic_records": 0,
                    "released_tokens": 0,
                    "model_call_count": 0,
                    "contract_repair_count": 0,
                }
            )
            continue
        workflow = source_arm["workflow"]
        budget = workflow["budget"]
        try:
            prompt, targeted, context_origin = _context_for_arm(
                arm_id=PRIMARY_ARM,
                raw=raw,
                plan=plan_by_id[session_id],
                task=task,
                initial=initial,
            )
            revision, calls, repairs = _complete_with_contract_repair(
                model,
                prompt,
                "crossfit revision model completion",
                lambda payload: _validated_revision(payload, targeted),
                max_retries=request.max_contract_retries,
            )
            predictions.append(
                {
                    **common,
                    "status": "ok",
                    "context_origin": context_origin,
                    "prompt_sha256": _sha256_text(
                        _canonical_json(prompt)
                    ),
                    "selected_segment_ids": list(
                        targeted.selected_segment_ids
                    ),
                    "selected_evidence_ids": list(
                        targeted.selected_atomic_evidence_ids
                    ),
                    "revision_assessment": revision.to_dict(),
                    "final_probability": revision.revised_risk_probability,
                    "final_confidence": revision.revised_confidence,
                    "referred": revision.revision_status == "unresolved",
                    "revision_status": revision.revision_status,
                    "queries": int(budget["queries_used"]),
                    "atomic_records": int(
                        budget["atomic_records_released"]
                    ),
                    "released_tokens": int(budget["released_tokens"]),
                    "model_call_count": calls,
                    "contract_repair_count": repairs,
                }
            )
        except Exception as error:
            predictions.append(
                {
                    **common,
                    "status": "fail_closed",
                    "failure": _safe_error(error),
                    "final_probability": initial.risk_probability,
                    "final_confidence": initial.confidence,
                    "referred": True,
                    "revision_status": "fail_closed_refer",
                    "queries": int(budget["queries_used"]),
                    "atomic_records": int(
                        budget["atomic_records_released"]
                    ),
                    "released_tokens": int(budget["released_tokens"]),
                    "model_call_count": 0,
                    "contract_repair_count": 0,
                }
            )
    _write_jsonl(predictions_path, predictions)
    status_counts = Counter(str(row["status"]) for row in predictions)
    manifest = {
        "schema_version": "1.0.0",
        "prediction_protocol_version": (
            CROSSFIT_PREDICTION_PROTOCOL_VERSION
        ),
        "dataset": predictions[0]["dataset"],
        "outer_fold": request.fold_index,
        "session_count": len(predictions),
        "status_counts": dict(sorted(status_counts.items())),
        "primary_arm": PRIMARY_ARM,
        "training_stage": _read_json(
            request.training_summary_path,
            "training summary",
        )["stage"],
        "files": {
            "predictions.raw.jsonl": {
                "sha256": _sha256(predictions_path),
                "row_count": len(predictions),
            }
        },
        "sources": {
            "raw_artifact_sha256": raw_artifact["artifact_sha256"],
            "training_package_artifact_sha256": package_artifact[
                "artifact_sha256"
            ],
            "training_summary_sha256": _sha256(
                request.training_summary_path
            ),
            "adapter_config_sha256": _sha256(
                adapter_path / "adapter_config.json"
            ),
            "task_sha256": _sha256(request.task_path),
        },
        "fit_boundaries": {
            "heldout_fold": request.fold_index,
            "heldout_outcomes_used_for_training": False,
            "outcomes_accessed_during_prediction": False,
            "raw_context_frozen_before_training": True,
            "ground_truth_used": False,
            "dev_accessed": False,
            "test_accessed": False,
        },
        "interpretation_boundary": (
            "These are label-free held-out revision predictions. Cross-fold "
            "metrics may be computed only after all folds are frozen."
        ),
    }
    _write_json(manifest_path, manifest)
    (request.output_dir / "PIPELINE_COMPLETE").touch()
    return manifest


@dataclass(frozen=True, slots=True)
class CrossfitMergeRequest:
    fold_dirs: tuple[Path, ...]
    raw_oof_dir: Path
    raw_artifact_manifest: Path
    output_dir: Path
    expected_fold_count: int = 5
    expected_session_count: int = 107

    def __post_init__(self) -> None:
        if len(self.fold_dirs) != self.expected_fold_count:
            raise ValueError("fold_dirs count must equal expected_fold_count")


def merge_crossfit_predictions(
    request: CrossfitMergeRequest,
) -> dict[str, Any]:
    """Freeze complete cross-fitted predictions before outcome access."""

    if request.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite crossfit freeze: {request.output_dir}"
        )
    raw_artifact = _directory_artifact(
        request.raw_oof_dir,
        request.raw_artifact_manifest,
    )
    raw_rows = _read_jsonl(
        request.raw_oof_dir / "trajectories.raw.jsonl",
        "raw OOF trajectories",
    )
    expected_by_fold: dict[int, set[str]] = {}
    initial_by_id: dict[str, float] = {}
    for row in raw_rows:
        fold = int(row["outer_fold"])
        session_id = str(row["session_id"])
        expected_by_fold.setdefault(fold, set()).add(session_id)
        initial_by_id[session_id] = float(
            row["initial"]["initial_probability_anchor"]
        )
    predictions: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    stages: set[str] = set()
    for directory in request.fold_dirs:
        manifest_path = directory / "manifest.json"
        prediction_path = directory / "predictions.raw.jsonl"
        manifest = _read_json(manifest_path, "crossfit fold manifest")
        rows = _read_jsonl(prediction_path, "crossfit fold predictions")
        fold = manifest.get("outer_fold")
        if isinstance(fold, bool) or not isinstance(fold, int):
            raise ValueError(f"invalid crossfit fold: {directory}")
        identifiers = {str(row.get("session_id", "")) for row in rows}
        if any(
            (
                manifest.get("prediction_protocol_version")
                != CROSSFIT_PREDICTION_PROTOCOL_VERSION,
                manifest.get("files", {})
                .get("predictions.raw.jsonl", {})
                .get("sha256")
                != _sha256(prediction_path),
                manifest.get("session_count") != len(rows),
                manifest.get("sources", {}).get("raw_artifact_sha256")
                != raw_artifact["artifact_sha256"],
                identifiers != expected_by_fold.get(fold),
                not (directory / "PIPELINE_COMPLETE").is_file(),
            )
        ):
            raise ValueError(f"crossfit fold failed verification: {directory}")
        stages.add(str(manifest.get("training_stage")))
        for row in rows:
            session_id = str(row["session_id"])
            if (
                session_id in seen
                or row.get("outer_fold") != fold
                or row.get("ground_truth_used") is not False
                or row.get("initial_probability")
                != initial_by_id.get(session_id)
                or row.get("status") not in {"ok", "fail_closed"}
            ):
                raise ValueError(
                    f"invalid crossfit prediction: {session_id}"
                )
            seen.add(session_id)
            predictions.append(row)
        sources.append(
            {
                "outer_fold": fold,
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": _sha256(manifest_path),
                "predictions_sha256": _sha256(prediction_path),
                "session_count": len(rows),
            }
        )
    if (
        set(expected_by_fold) != set(range(request.expected_fold_count))
        or seen != set(initial_by_id)
        or len(predictions) != request.expected_session_count
        or len(stages) != 1
    ):
        raise ValueError("crossfit prediction coverage is incomplete")
    ordered = sorted(predictions, key=lambda row: str(row["session_id"]))
    request.output_dir.mkdir(parents=True)
    prediction_path = request.output_dir / "predictions.raw.jsonl"
    _write_jsonl(prediction_path, ordered)
    status_counts = Counter(str(row["status"]) for row in ordered)
    manifest = {
        "schema_version": "1.0.0",
        "freeze_protocol_version": CROSSFIT_FREEZE_PROTOCOL_VERSION,
        "prediction_protocol_version": (
            CROSSFIT_PREDICTION_PROTOCOL_VERSION
        ),
        "dataset": ordered[0]["dataset"],
        "fold_count": request.expected_fold_count,
        "session_count": len(ordered),
        "status_counts": dict(sorted(status_counts.items())),
        "training_stage": next(iter(stages)),
        "primary_arm": PRIMARY_ARM,
        "files": {
            "predictions.raw.jsonl": {
                "sha256": _sha256(prediction_path),
                "row_count": len(ordered),
            }
        },
        "sources": {
            "raw_artifact_sha256": raw_artifact["artifact_sha256"],
            "fold_predictions": sorted(
                sources,
                key=lambda item: int(item["outer_fold"]),
            ),
        },
        "fit_boundaries": {
            "all_predictions_frozen_before_outcome_join": True,
            "each_fold_trained_without_heldout_outcomes": True,
            "ground_truth_used": False,
            "dev_accessed": False,
            "test_accessed": False,
        },
        "interpretation_boundary": (
            "The frozen predictions are cross-fitted. The protocol was defined "
            "after earlier results, so evaluation remains developmental."
        ),
    }
    _write_json(request.output_dir / "manifest.json", manifest)
    (request.output_dir / "PREDICTIONS_FROZEN").touch()
    return manifest


@dataclass(frozen=True, slots=True)
class CrossfitEvaluationRequest:
    predictions_dir: Path
    predictions_artifact_manifest: Path
    prepared_root: Path
    prepared_artifact_manifest: Path
    config_path: Path
    output_dir: Path


def evaluate_crossfit_predictions(
    request: CrossfitEvaluationRequest,
) -> dict[str, Any]:
    """Join outcomes after cross-fold predictions are immutable."""

    if request.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite crossfit evaluation: {request.output_dir}"
        )
    prediction_artifact = _directory_artifact(
        request.predictions_dir,
        request.predictions_artifact_manifest,
    )
    prepared_artifact = _directory_artifact(
        request.prepared_root,
        request.prepared_artifact_manifest,
    )
    manifest_path = request.predictions_dir / "manifest.json"
    predictions_path = request.predictions_dir / "predictions.raw.jsonl"
    manifest = _read_json(manifest_path, "crossfit freeze manifest")
    predictions = _read_jsonl(
        predictions_path,
        "crossfit frozen predictions",
    )
    if any(
        (
            manifest.get("freeze_protocol_version")
            != CROSSFIT_FREEZE_PROTOCOL_VERSION,
            manifest.get("files", {})
            .get("predictions.raw.jsonl", {})
            .get("sha256")
            != _sha256(predictions_path),
            manifest.get("fit_boundaries", {}).get(
                "all_predictions_frozen_before_outcome_join"
            )
            is not True,
            manifest.get("fit_boundaries", {}).get(
                "each_fold_trained_without_heldout_outcomes"
            )
            is not True,
            not (request.predictions_dir / "PREDICTIONS_FROZEN").is_file(),
        )
    ):
        raise ValueError("crossfit prediction freeze failed verification")
    outcome_dir = request.prepared_root / "outcomes"
    outcome_manifest_path = outcome_dir / "manifest.json"
    outcome_path = outcome_dir / "outcomes.jsonl"
    outcome_manifest = _read_json(
        outcome_manifest_path,
        "outcome capability manifest",
    )
    outcomes = _read_jsonl(outcome_path, "outcomes")
    if any(
        (
            outcome_manifest.get("preparation_protocol_version")
            != PREPARATION_PROTOCOL_VERSION,
            outcome_manifest.get("files", {})
            .get("outcomes.jsonl", {})
            .get("sha256")
            != _sha256(outcome_path),
            outcome_manifest.get("files", {})
            .get("outcomes.jsonl", {})
            .get("used_in_model_inference")
            is not False,
        )
    ):
        raise ValueError("outcome capability failed verification")
    _, weights, samples, seed, _ = _load_config(request.config_path)
    prediction_by_id = {
        str(row["session_id"]): row for row in predictions
    }
    outcome_by_id = {
        str(row["session_id"]): row for row in outcomes
    }
    if (
        len(prediction_by_id) != len(predictions)
        or len(outcome_by_id) != len(outcomes)
        or set(prediction_by_id) != set(outcome_by_id)
    ):
        raise ValueError("crossfit predictions and outcomes differ")
    forced_rows: list[dict[str, Any]] = []
    natural_rows: list[dict[str, Any]] = []
    direct_rows: list[dict[str, Any]] = []
    initial_rows: list[dict[str, Any]] = []
    for session_id in sorted(prediction_by_id):
        prediction = prediction_by_id[session_id]
        outcome = outcome_by_id[session_id]
        label = int(outcome["label"])
        initial_probability = float(outcome["initial_probability"])
        threshold = float(outcome["initial_threshold"])
        if prediction["initial_probability"] != initial_probability:
            raise ValueError(
                f"crossfit initial probability changed: {session_id}"
            )
        initial_confidence = float(prediction["initial_confidence"])
        predicted_arm = {
            "arm_id": PRIMARY_ARM,
            "final_probability": float(
                prediction["final_probability"]
            ),
            "final_confidence": float(prediction["final_confidence"]),
            "referred": bool(prediction["referred"]),
            "queries": int(prediction["queries"]),
            "atomic_records": int(prediction["atomic_records"]),
            "released_tokens": int(prediction["released_tokens"]),
            "arm_valid": prediction["status"] == "ok",
            "failure_reason": (
                None
                if prediction["status"] == "ok"
                else prediction.get("failure", {}).get(
                    "error_type",
                    "crossfit_failure",
                )
            ),
            "revision_status": prediction["revision_status"],
        }
        common = {
            "session_id": session_id,
            "label": label,
            "initial_probability": initial_probability,
            "threshold": threshold,
            "initial_confidence": initial_confidence,
            "natural_triggered": bool(
                prediction["natural_triggered"]
            ),
            "weights": weights,
        }
        forced_rows.append(
            _scored_view(
                arm=predicted_arm,
                deployed=False,
                **common,
            )
        )
        natural_rows.append(
            _scored_view(
                arm=predicted_arm,
                deployed=True,
                **common,
            )
        )
        direct_rows.append(
            _scored_view(
                arm={
                    "arm_id": DIRECT_REFER_ARM,
                    "final_probability": initial_probability,
                    "final_confidence": initial_confidence,
                    "referred": True,
                    "queries": 0,
                    "atomic_records": 0,
                    "released_tokens": 0,
                    "arm_valid": True,
                    "failure_reason": None,
                    "revision_status": "direct_refer",
                },
                deployed=True,
                **common,
            )
        )
        initial_rows.append(
            _scored_view(
                arm={
                    "arm_id": "frozen_native_initial",
                    "final_probability": initial_probability,
                    "final_confidence": initial_confidence,
                    "referred": False,
                    "queries": 0,
                    "atomic_records": 0,
                    "released_tokens": 0,
                    "arm_valid": True,
                    "failure_reason": None,
                    "revision_status": "initial",
                },
                deployed=False,
                **common,
            )
        )
    paired = _bootstrap_delta(
        natural_rows,
        samples=samples,
        seed=seed,
    )
    natural_metrics = _metrics(natural_rows)
    forced_metrics = _metrics(forced_rows)
    point_pass = all(
        (
            paired["brier_improvement"]["estimate"] > 0.0,
            paired["log_loss_improvement"]["estimate"] > 0.0,
            natural_metrics["net_correction"] > 0,
        )
    )
    uncertainty_pass = all(
        (
            paired["brier_improvement"]["ci_95"][0] > 0.0,
            paired["log_loss_improvement"]["ci_95"][0] > 0.0,
            paired["net_correction_rate"]["ci_95"][0] >= 0.0,
        )
    )
    summary = {
        "schema_version": "1.0.0",
        "evaluation_protocol_version": (
            CROSSFIT_EVALUATION_PROTOCOL_VERSION
        ),
        "dataset": predictions[0]["dataset"],
        "session_count": len(predictions),
        "training_stage": manifest["training_stage"],
        "primary_arm": PRIMARY_ARM,
        "frozen_native_initial": _metrics(initial_rows),
        "crossfit_trained_loop": {
            "forced_counterfactual": forced_metrics,
            "natural_trigger_policy": natural_metrics,
            "natural_trigger_paired_bootstrap_vs_initial": paired,
            "selective_vs_same_trigger_direct_refer": (
                _selective_comparison(natural_rows, direct_rows)
            ),
        },
        "benefit_gate": {
            "point_estimate_passed": point_pass,
            "uncertainty_aware_passed": uncertainty_pass,
            "requirements": {
                "brier_improvement": "greater_than_zero",
                "log_loss_improvement": "greater_than_zero",
                "net_correction": "greater_than_zero",
                "uncertainty_lower_bounds": (
                    "proper_scores_above_zero_and_net_correction_nonnegative"
                ),
            },
            "confirmatory_claim_permitted": False,
        },
        "sources": {
            "prediction_artifact_sha256": prediction_artifact[
                "artifact_sha256"
            ],
            "prepared_artifact_sha256": prepared_artifact[
                "artifact_sha256"
            ],
            "prediction_manifest_sha256": _sha256(manifest_path),
            "outcome_manifest_sha256": _sha256(outcome_manifest_path),
            "config_sha256": _sha256(request.config_path),
        },
        "fit_boundaries": {
            "predictions_frozen_before_outcome_join": True,
            "each_participant_outcome_excluded_from_its_model_fit": True,
            "labels_in_model_messages": False,
            "protocol_defined_after_prior_results": True,
            "developmental_not_confirmatory": True,
            "dev_accessed": False,
            "test_accessed": False,
            "reference_retrained": False,
            "reference_recompiled": False,
        },
        "interpretation_boundary": (
            "This is a developmental five-fold cross-fitted estimate. It is "
            "stronger than training replay but remains non-confirmatory because "
            "the protocol follows earlier observations on this population."
        ),
    }
    request.output_dir.mkdir(parents=True)
    _write_json(request.output_dir / "summary.json", summary)
    (request.output_dir / "OUTCOME_JOIN_COMPLETE").touch()
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-fit a trained revision policy."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--raw-oof-dir", required=True, type=Path)
    collect.add_argument(
        "--raw-artifact-manifest",
        required=True,
        type=Path,
    )
    collect.add_argument(
        "--prepared-inference-dir",
        required=True,
        type=Path,
    )
    collect.add_argument(
        "--training-package-dir",
        required=True,
        type=Path,
    )
    collect.add_argument(
        "--training-package-artifact-manifest",
        required=True,
        type=Path,
    )
    collect.add_argument(
        "--training-summary",
        required=True,
        type=Path,
    )
    collect.add_argument("--task", required=True, type=Path)
    collect.add_argument("--output-dir", required=True, type=Path)
    collect.add_argument("--fold-index", required=True, type=int)
    collect.add_argument("--device", default="cuda")
    collect.add_argument("--dtype", default="bfloat16")
    collect.add_argument("--attention-implementation", default="sdpa")
    collect.add_argument("--max-new-tokens", type=int, default=420)
    collect.add_argument("--max-input-tokens", type=int, default=8_192)
    collect.add_argument("--max-contract-retries", type=int, default=2)
    collect.add_argument(
        "--audit-decisions",
        type=Path,
        help="Label-free decisions from `rethink-fit-audit decide`.",
    )

    merge = subparsers.add_parser("merge")
    merge.add_argument("--fold-dir", action="append", required=True, type=Path)
    merge.add_argument("--raw-oof-dir", required=True, type=Path)
    merge.add_argument(
        "--raw-artifact-manifest",
        required=True,
        type=Path,
    )
    merge.add_argument("--output-dir", required=True, type=Path)
    merge.add_argument("--expected-fold-count", type=int, default=5)
    merge.add_argument("--expected-session-count", type=int, default=107)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--predictions-dir", required=True, type=Path)
    evaluate.add_argument(
        "--predictions-artifact-manifest",
        required=True,
        type=Path,
    )
    evaluate.add_argument("--prepared-root", required=True, type=Path)
    evaluate.add_argument(
        "--prepared-artifact-manifest",
        required=True,
        type=Path,
    )
    evaluate.add_argument("--config", required=True, type=Path)
    evaluate.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "collect":
        result = collect_crossfit_fold(
            CrossfitCollectRequest(
                raw_oof_dir=args.raw_oof_dir,
                raw_artifact_manifest=args.raw_artifact_manifest,
                prepared_inference_dir=args.prepared_inference_dir,
                training_package_dir=args.training_package_dir,
                training_package_artifact_manifest=(
                    args.training_package_artifact_manifest
                ),
                training_summary_path=args.training_summary,
                task_path=args.task,
                output_dir=args.output_dir,
                fold_index=args.fold_index,
                device=args.device,
                dtype=args.dtype,
                attention_implementation=args.attention_implementation,
                max_new_tokens=args.max_new_tokens,
                max_input_tokens=args.max_input_tokens,
                max_contract_retries=args.max_contract_retries,
                audit_decisions=args.audit_decisions,
            )
        )
    elif args.command == "merge":
        result = merge_crossfit_predictions(
            CrossfitMergeRequest(
                fold_dirs=tuple(args.fold_dir),
                raw_oof_dir=args.raw_oof_dir,
                raw_artifact_manifest=args.raw_artifact_manifest,
                output_dir=args.output_dir,
                expected_fold_count=args.expected_fold_count,
                expected_session_count=args.expected_session_count,
            )
        )
    else:
        result = evaluate_crossfit_predictions(
            CrossfitEvaluationRequest(
                predictions_dir=args.predictions_dir,
                predictions_artifact_manifest=(
                    args.predictions_artifact_manifest
                ),
                prepared_root=args.prepared_root,
                prepared_artifact_manifest=(
                    args.prepared_artifact_manifest
                ),
                config_path=args.config,
                output_dir=args.output_dir,
            )
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
