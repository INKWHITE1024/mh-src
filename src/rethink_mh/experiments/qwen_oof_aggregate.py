"""Validate and aggregate independently trained Qwen participant-level OOF folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .qwen_label_baseline import participant_stratified_fold
from .text_baseline import (
    SessionRecord,
    _metrics,
    _read_split,
    _sha256_identifiers,
    _write_json,
    _write_jsonl,
    select_balanced_accuracy_threshold,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            output.append(payload)
    return output


def aggregate_oof_payloads(
    expected: Sequence[SessionRecord],
    fold_summaries: Sequence[Mapping[str, Any]],
    fold_predictions: Sequence[Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not fold_summaries or len(fold_summaries) != len(fold_predictions):
        raise ValueError("matching non-empty fold summaries and predictions are required")
    first = fold_summaries[0]
    folds = int(first["folds"])
    dataset = str(first["dataset"])
    seed = int(first["seed"])
    split_seed = int(first.get("split_seed", seed))
    view = str(first["view"])
    experiment = str(first.get("experiment"))
    evidence_protocol = str(first.get("evidence_protocol", "native"))
    supported_experiments = {
        "qwen_thinker_text_label_training_native": (
            "qwen_thinker_text_label_oof_native"
        ),
        "qwen_thinker_av_transcript_label_training_native": (
            "qwen_thinker_av_transcript_label_oof_native"
        ),
    }
    if experiment not in supported_experiments:
        raise ValueError(f"unsupported OOF source experiment {experiment!r}")
    if len(fold_summaries) != folds:
        raise ValueError(f"expected {folds} fold outputs, found {len(fold_summaries)}")
    indices: set[int] = set()
    reference_ids: set[str] = set()
    model_paths: set[str] = set()
    by_id: dict[str, dict[str, Any]] = {}
    for summary, predictions in zip(fold_summaries, fold_predictions):
        identity = (
            summary.get("experiment"),
            summary.get("run_kind"),
            summary.get("dataset"),
            int(summary.get("seed")),
            int(summary.get("split_seed", summary.get("seed"))),
            int(summary.get("folds")),
            summary.get("view"),
            summary.get("evidence_protocol", "native"),
        )
        if identity != (
            experiment,
            "participant_oof_fold",
            dataset,
            seed,
            split_seed,
            folds,
            view,
            evidence_protocol,
        ):
            raise ValueError("fold summaries have incompatible experiment identities")
        expected_score_column = {
            "daic_woz": "PHQ8_Score",
            "e_daic": "PHQ_Score",
        }[dataset]
        label_policy = summary.get("label_policy", {})
        if any(
            (
                label_policy.get("policy") != "phq8_score_ge_10",
                label_policy.get("operator") != ">=",
                label_policy.get("threshold") != 10,
                label_policy.get("target_source") != expected_score_column,
                label_policy.get("released_binary_used_as_target") is not False,
            )
        ):
            raise ValueError(
                f"fold {summary.get('fold_index')} lacks the score-derived "
                "PHQ-8 >= 10 label policy"
            )
        fold_index = int(summary["fold_index"])
        if fold_index in indices or not 0 <= fold_index < folds:
            raise ValueError(f"invalid or duplicate fold index {fold_index}")
        indices.add(fold_index)

        expected_fit, expected_holdout = participant_stratified_fold(
            expected,
            folds=folds,
            fold_index=fold_index,
            seed=split_seed,
        )
        if summary.get("fit_ids_sha256") != _sha256_identifiers(expected_fit):
            raise ValueError(f"fold {fold_index} fit participant hash mismatch")
        if summary.get("validation_ids_sha256") != _sha256_identifiers(
            expected_holdout
        ):
            raise ValueError(f"fold {fold_index} holdout participant hash mismatch")

        provenance = summary.get("oof_reference")
        if not isinstance(provenance, Mapping) or not all(
            (
                provenance.get("strict") is True,
                provenance.get("holdout_excluded_from_reference") is True,
                provenance.get("all_evidence_uses_fold_reference") is True,
                provenance.get("evidence_compiler_label_access") is False,
                (
                    experiment
                    != "qwen_thinker_av_transcript_label_training_native"
                    or provenance.get("transcript_alignment_verified") is True
                ),
            )
        ):
            raise ValueError(
                f"fold {fold_index} lacks strict fold-local reference proof"
            )
        if int(provenance.get("reference_fit_count", -1)) != len(expected_fit):
            raise ValueError(f"fold {fold_index} reference fit count mismatch")
        reference_id = str(provenance.get("reference_id", ""))
        if not reference_id or reference_id in reference_ids:
            raise ValueError("fold-local reference IDs must be non-empty and unique")
        reference_ids.add(reference_id)

        training = summary.get("training", {})
        if training.get("class_weighting") is not False:
            raise ValueError(
                f"fold {fold_index} must use the approved unweighted loss"
            )
        checkpoint = training.get("checkpoint_selection", {})
        if (
            training.get("fixed_epoch_protocol") is not True
            or checkpoint.get("metric") is not None
            or checkpoint.get("oof_holdout_evaluated_each_epoch") is not False
        ):
            raise ValueError(f"fold {fold_index} did not use fixed-epoch OOF training")
        boundaries = summary.get("fit_boundaries", {})
        if any(
            (
                boundaries.get("dev_gradient_updates") is not False,
                boundaries.get("test_labels_accessed") is not False,
                boundaries.get("oof_holdout_gradient_updates") is not False,
                boundaries.get("oof_holdout_threshold_selection") is not False,
                boundaries.get("oof_holdout_in_reference_fit") is not False,
            )
        ):
            raise ValueError(f"fold {fold_index} violates OOF fit boundaries")
        if experiment == "qwen_thinker_av_transcript_label_training_native":
            transcript = summary.get("transcript", {})
            model_path = str(training.get("model_path", ""))
            if (
                transcript.get("enabled") is not True
                or transcript.get("content_exposed_to_model") is not True
                or transcript.get("token_level_text_truncation") is not False
                or summary.get("evidence_protocol") != "native"
                or not model_path
                or Path(model_path).name != "Qwen2.5-Omni-7B"
                or int(training.get("epochs", -1)) != 2
                or int(training.get("maximum_epochs", -1)) != 2
            ):
                raise ValueError(
                    f"fold {fold_index} native provenance is invalid"
                )
            model_paths.add(model_path)

        expected_holdout_ids = {record.session_id for record in expected_holdout}
        prediction_ids = {str(prediction["session_id"]) for prediction in predictions}
        if prediction_ids != expected_holdout_ids:
            raise ValueError(f"fold {fold_index} prediction/holdout coverage mismatch")
        for prediction in predictions:
            session_id = str(prediction["session_id"])
            if session_id in by_id:
                raise ValueError(f"duplicate OOF prediction for participant {session_id}")
            if prediction.get("prediction_origin") != "out_of_fold_fixed_threshold":
                raise ValueError(f"non-OOF prediction supplied for participant {session_id}")
            by_id[session_id] = dict(prediction)
    if indices != set(range(folds)):
        raise ValueError(f"fold indices must be exactly 0 through {folds - 1}")
    if (
        experiment == "qwen_thinker_av_transcript_label_training_native"
        and len(model_paths) != 1
    ):
        raise ValueError("native folds do not share one base-model path")

    expected_by_id = {record.session_id: record for record in expected}
    if set(by_id) != set(expected_by_id):
        missing = sorted(set(expected_by_id) - set(by_id))
        extra = sorted(set(by_id) - set(expected_by_id))
        raise ValueError(f"OOF participant coverage mismatch: missing={missing}, extra={extra}")

    labels: list[int] = []
    probabilities: list[float] = []
    for record in expected:
        if record.label not in {0, 1}:
            raise ValueError("expected training labels must be binary")
        prediction = by_id[record.session_id]
        if int(prediction["label"]) != record.label:
            raise ValueError(f"label mismatch for participant {record.session_id}")
        probability = float(prediction["probability"])
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"invalid probability for participant {record.session_id}")
        labels.append(record.label)
        probabilities.append(probability)
    threshold = select_balanced_accuracy_threshold(labels, probabilities)
    rows = [
        {
            "session_id": record.session_id,
            "split": "train",
            "label": record.label,
            "probability": probability,
            "predicted_label": int(probability >= threshold),
            "threshold": threshold,
            "prediction_origin": f"participant_{folds}fold_oof",
        }
        for record, probability in zip(expected, probabilities)
    ]
    summary = {
        "experiment": supported_experiments[experiment],
        "source_experiment": experiment,
        "dataset": dataset,
        "view": view,
        "evidence_protocol": evidence_protocol,
        "seed": seed,
        "split_seed": split_seed,
        "folds": folds,
        "participant_count": len(expected),
        "threshold_source": "all_training_participants_out_of_fold_probabilities",
        "metrics": _metrics(labels, probabilities, threshold),
        "fold_reference_ids": sorted(reference_ids),
        **(
            {
                "training_initialization": {
                    "model_path": next(iter(model_paths)),
                    "model_name": "Qwen2.5-Omni-7B",
                    "adapter_checkpoint_loaded": False,
                    "frozen_reference_model_reused": False,
                    "epochs_per_fold": 2,
                    "class_weighting": False,
                    "checkpoint_selection_metric": None,
                }
            }
            if (
                experiment == "qwen_thinker_av_transcript_label_training_native"
                and len(model_paths) == 1
            )
            else {}
        ),
        "fit_boundaries": {
            "each_prediction_model_saw_participant": False,
            "dev_read_by_fold_runs": False,
            "test_read_by_fold_runs": False,
            "threshold_uses_only_oof_training_predictions": True,
            "each_fold_reference_excludes_holdout": True,
            "evidence_compiler_label_access": False,
        },
    }
    return rows, summary


def run_aggregate(
    *,
    fold_dirs: Sequence[Path],
    train_labels: Path,
    evidence_root: Path,
    label_column: str,
    label_threshold: float | None,
    id_column: str | None,
    output_dir: Path,
    evidence_protocol: str = "native",
) -> dict[str, Any]:
    expected = _read_split(
        train_labels,
        "train",
        evidence_root,
        label_column,
        id_column,
        require_labels=True,
        label_threshold=label_threshold,
        evidence_protocol=evidence_protocol,
    )
    summaries = [
        json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        for directory in fold_dirs
    ]
    predictions = [
        _read_jsonl(directory / "predictions.jsonl") for directory in fold_dirs
    ]
    rows, summary = aggregate_oof_payloads(expected, summaries, predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "predictions.jsonl", rows)
    summary["fold_directories"] = [str(path.resolve()) for path in fold_dirs]
    summary["label_column"] = label_column
    summary["label_threshold"] = label_threshold
    _write_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate complete, disjoint Qwen participant-level OOF folds."
    )
    parser.add_argument("--fold-dir", required=True, type=Path, nargs="+")
    parser.add_argument("--train-labels", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--label-threshold", required=True, type=float)
    parser.add_argument("--id-column")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--evidence-protocol",
        choices=("native",),
        default="native",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_aggregate(
        fold_dirs=args.fold_dir,
        train_labels=args.train_labels,
        evidence_root=args.evidence_root,
        label_column=args.label_column,
        label_threshold=args.label_threshold,
        id_column=args.id_column,
        output_dir=args.output_dir,
        evidence_protocol=args.evidence_protocol,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
