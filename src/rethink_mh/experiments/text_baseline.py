"""OOF text baselines over frozen native Evidence artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .phq8_labels import build_phq8_label_audit, validate_safe_label_source


SUPPORTED_VIEWS = ("full", "audio", "visual")


@dataclass(frozen=True, slots=True)
class SessionRecord:
    session_id: str
    split: str
    text: str
    label: int | None
    transcript_tiers: tuple[tuple[str, str], ...] = ()
    transcript_manifest_sha256: str | None = None
    transcript_speaker_policy: str | None = None
    transcript_source_kind: str | None = None


def _normalized_column(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _resolve_column(frame: pd.DataFrame, requested: str | None, candidates: Sequence[str]) -> str:
    normalized = {_normalized_column(str(column)): str(column) for column in frame.columns}
    if requested is not None:
        key = _normalized_column(requested)
        if key not in normalized:
            raise ValueError(
                f"Column {requested!r} is absent; available columns are {list(frame.columns)!r}"
            )
        return normalized[key]
    for candidate in candidates:
        key = _normalized_column(candidate)
        if key in normalized:
            return normalized[key]
    raise ValueError(f"Could not find any of columns {list(candidates)!r}")


def evidence_view(session_text: str, view: str) -> str:
    """Return full, audio-only, or visual-only segment evidence.

    Non-segment protocol guidance is retained.  Every segment keeps its identifier,
    time, atomic-window range, and only the requested modality summary.
    """

    if view not in SUPPORTED_VIEWS:
        raise ValueError(f"Unsupported evidence view {view!r}")
    if view == "full":
        return session_text

    selected: list[str] = []
    marker = f"{view.upper()} summary;"
    for line in session_text.splitlines():
        if not line.startswith("SEGMENT S"):
            selected.append(line)
            continue
        fields = [field.strip() for field in line.split("|")]
        shared = fields[:4]
        modality_fields = [field for field in fields[4:] if field.startswith(marker)]
        selected.append(" | ".join((*shared, *modality_fields)))
    return "\n".join(selected).rstrip() + "\n"


def _read_split(
    label_path: Path,
    split: str,
    evidence_root: Path,
    label_column: str,
    id_column: str | None,
    *,
    require_labels: bool,
    load_optional_labels: bool = False,
    transcript_root: Path | None = None,
    label_threshold: float | None = None,
    evidence_protocol: str = "native",
) -> list[SessionRecord]:
    header = pd.read_csv(label_path, nrows=0)
    resolved_id = _resolve_column(
        header,
        id_column,
        ("Participant_ID", "participant_ID", "session_id", "id"),
    )
    if require_labels or load_optional_labels:
        try:
            resolved_label = _resolve_column(header, label_column, ())
        except ValueError:
            if require_labels:
                raise
            resolved_label = None
    else:
        resolved_label = None
    use_columns = [resolved_id]
    if resolved_label is not None and resolved_label != resolved_id:
        use_columns.append(resolved_label)
    frame = pd.read_csv(label_path, usecols=use_columns)

    records: list[SessionRecord] = []
    seen: set[str] = set()
    for row_index, row in frame.iterrows():
        raw_identifier = row[resolved_id]
        if pd.isna(raw_identifier):
            raise ValueError(f"Missing session ID at {label_path}:{row_index + 2}")
        if isinstance(raw_identifier, (int, np.integer)):
            session_id = str(int(raw_identifier))
        elif isinstance(raw_identifier, (float, np.floating)) and float(raw_identifier).is_integer():
            session_id = str(int(raw_identifier))
        else:
            session_id = str(raw_identifier).strip()
        if not session_id or session_id in seen:
            raise ValueError(f"Invalid or duplicate session ID {session_id!r} in {label_path}")
        seen.add(session_id)

        label: int | None = None
        if resolved_label is not None:
            raw_label = row[resolved_label]
            if pd.isna(raw_label):
                if require_labels:
                    raise ValueError(f"Missing label for session {session_id}")
            else:
                numeric = float(raw_label)
                if not np.isfinite(numeric):
                    raise ValueError(f"Label for session {session_id} must be finite")
                if label_threshold is None:
                    parsed = int(numeric)
                    if parsed not in {0, 1} or numeric != parsed:
                        raise ValueError(
                            f"Label for session {session_id} must be binary"
                        )
                    label = parsed
                else:
                    label = int(numeric >= label_threshold)

        session_root = evidence_root / session_id
        manifest_path = session_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing Evidence manifest for session {session_id}: "
                f"{manifest_path}"
            )
        evidence_manifest: dict[str, Any] = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if evidence_manifest.get("protocol_version") != evidence_protocol:
            raise ValueError(
                f"Evidence protocol mismatch for session {session_id}: "
                f"expected {evidence_protocol!r}, got "
                f"{evidence_manifest.get('protocol_version')!r}"
            )
        primary_input = evidence_manifest.get("primary_model_input")
        if (
            not isinstance(primary_input, str)
            or not primary_input
            or "/" in primary_input
            or "\\" in primary_input
        ):
            raise ValueError(
                f"Invalid primary Evidence input for session {session_id}"
            )
        evidence_path = session_root / primary_input
        if not evidence_path.is_file():
            raise FileNotFoundError(f"Missing compiled evidence for session {session_id}: {evidence_path}")
        if evidence_manifest is not None:
            file_records = evidence_manifest.get("files")
            file_record = (
                file_records.get(evidence_path.name)
                if isinstance(file_records, dict)
                else None
            )
            expected_hash = (
                file_record.get("sha256")
                if isinstance(file_record, dict)
                else None
            )
            actual_hash = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
            if expected_hash != actual_hash:
                raise ValueError(
                    f"Primary Evidence input hash mismatch for session {session_id}"
                )
        transcript_tiers: tuple[tuple[str, str], ...] = ()
        transcript_manifest_sha256: str | None = None
        transcript_speaker_policy: str | None = None
        transcript_source_kind: str | None = None
        if transcript_root is not None:
            transcript_dir = transcript_root / session_id
            manifest_path = transcript_dir / "transcript_manifest.compact.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"Missing transcript manifest for session {session_id}: "
                    f"{manifest_path}"
                )
            manifest_text = manifest_path.read_text(encoding="utf-8")
            manifest = json.loads(manifest_text)
            if manifest.get("protocol_version") != "compact":
                raise ValueError(
                    f"Transcript protocol mismatch for session {session_id}"
                )
            if str(manifest.get("session_id")) != session_id:
                raise ValueError(
                    f"Transcript session mismatch for session {session_id}"
                )
            if manifest.get("transcript_content_exposed") is not True:
                raise ValueError(
                    f"Transcript manifest must declare content exposure for {session_id}"
                )
            if manifest.get("raw_session_identifier_in_model_input") is not False:
                raise ValueError(
                    f"Transcript manifest does not protect the raw identifier for {session_id}"
                )
            if manifest.get("label_fields_read") != []:
                raise ValueError(
                    f"Transcript compiler label boundary is invalid for {session_id}"
                )
            speaker_policy = manifest.get("speaker_policy")
            source_kind = manifest.get("source_kind")
            valid_provenance = {
                ("participant_only", "manual_transcript"),
                (
                    "all_speakers_unassigned",
                    "automatic_speech_recognition",
                ),
            }
            if (speaker_policy, source_kind) not in valid_provenance:
                raise ValueError(
                    f"Unsupported transcript provenance for session {session_id}: "
                    f"{speaker_policy!r}, {source_kind!r}"
                )
            files = manifest.get("files")
            if not isinstance(files, dict):
                raise ValueError(
                    f"Transcript manifest files are invalid for session {session_id}"
                )
            loaded_tiers: list[tuple[str, str]] = []
            for tier in ("full", "compact", "essential", "minimal"):
                filename = f"transcript_input.{tier}.compact.txt"
                path = transcript_dir / filename
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Missing transcript tier for session {session_id}: {path}"
                    )
                content = path.read_text(encoding="utf-8")
                file_record = files.get(filename)
                if not isinstance(file_record, dict):
                    raise ValueError(
                        f"Transcript manifest file entry is invalid for "
                        f"session {session_id}: {filename}"
                    )
                expected_hash = file_record.get("sha256")
                actual_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                if expected_hash != actual_hash:
                    raise ValueError(
                        f"Transcript hash mismatch for session {session_id}: {filename}"
                    )
                loaded_tiers.append((tier, content))
            transcript_tiers = tuple(loaded_tiers)
            transcript_manifest_sha256 = hashlib.sha256(
                manifest_text.encode("utf-8")
            ).hexdigest()
            transcript_speaker_policy = str(speaker_policy)
            transcript_source_kind = str(source_kind)

        records.append(
            SessionRecord(
                session_id=session_id,
                split=split,
                text=evidence_path.read_text(encoding="utf-8"),
                label=label,
                transcript_tiers=transcript_tiers,
                transcript_manifest_sha256=transcript_manifest_sha256,
                transcript_speaker_policy=transcript_speaker_policy,
                transcript_source_kind=transcript_source_kind,
            )
        )
    return records


def _sha256_identifiers(records: Iterable[SessionRecord]) -> str:
    payload = "\n".join(record.session_id for record in records).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_balanced_accuracy_threshold(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """Select a frozen threshold from OOF training predictions only."""

    from sklearn.metrics import balanced_accuracy_score

    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or y.size == 0:
        raise ValueError("labels and probabilities must be non-empty one-dimensional arrays")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("threshold selection requires both binary classes")
    if not np.all(np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probabilities must be finite values from 0 to 1")

    unique = np.unique(p)
    candidates = np.unique(
        np.concatenate(
            (
                np.asarray([0.5]),
                unique,
                (unique[:-1] + unique[1:]) / 2.0 if unique.size > 1 else unique,
            )
        )
    )
    scored = [
        (float(balanced_accuracy_score(y, p >= threshold)), -abs(float(threshold) - 0.5), float(threshold))
        for threshold in candidates
    ]
    return max(scored)[2]


def _metrics(
    labels: Sequence[int], probabilities: Sequence[float], threshold: float
) -> dict[str, float | int]:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    predictions = (p >= threshold).astype(int)
    output: dict[str, float | int] = {
        "count": int(y.size),
        "positive_count": int(np.sum(y == 1)),
        "negative_count": int(np.sum(y == 0)),
        "threshold": float(threshold),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }
    if set(np.unique(y)) == {0, 1}:
        output["roc_auc"] = float(roc_auc_score(y, p))
        output["average_precision"] = float(average_precision_score(y, p))
    return output


def _pipeline(seed: int, max_features: int):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return Pipeline(
        (
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=max_features,
                    sublinear_tf=True,
                    dtype=np.float64,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=seed,
                    solver="liblinear",
                ),
            ),
        )
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def run_baseline(
    *,
    evidence_root: Path,
    train_labels: Path,
    dev_labels: Path,
    test_labels: Path | None,
    label_column: str,
    id_column: str | None,
    output_dir: Path,
    views: Sequence[str],
    folds: int,
    seed: int,
    max_features: int,
    evaluate_labeled_test: bool = False,
    label_threshold: float | None = None,
) -> dict[str, Any]:
    try:
        from joblib import dump
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
    except ImportError as exc:  # pragma: no cover - optional experiment dependency
        raise RuntimeError(
            "Text baselines require the project experiments optional dependencies"
        ) from exc

    invalid_views = sorted(set(views) - set(SUPPORTED_VIEWS))
    if invalid_views:
        raise ValueError(f"Unsupported views: {invalid_views}")
    if folds < 2:
        raise ValueError("folds must be at least two")
    if max_features <= 0:
        raise ValueError("max_features must be positive")
    phq8_dataset = validate_safe_label_source(label_column, label_threshold)

    train = _read_split(
        train_labels,
        "train",
        evidence_root,
        label_column,
        id_column,
        require_labels=True,
        label_threshold=label_threshold,
    )
    dev = _read_split(
        dev_labels,
        "dev",
        evidence_root,
        label_column,
        id_column,
        require_labels=True,
        label_threshold=label_threshold,
    )
    test = (
        _read_split(
            test_labels,
            "test",
            evidence_root,
            label_column,
            id_column,
            require_labels=False,
            load_optional_labels=evaluate_labeled_test,
            label_threshold=label_threshold,
        )
        if test_labels is not None
        else []
    )
    label_policy = (
        build_phq8_label_audit(
            [
                train_labels,
                dev_labels,
                *(
                    [test_labels]
                    if test_labels is not None and evaluate_labeled_test
                    else []
                ),
            ],
            dataset=phq8_dataset,
            id_column=id_column,
        )
        if phq8_dataset is not None
        else {
            "policy": (
                "binary_column" if label_threshold is None else "numeric_threshold"
            ),
            "target_source": label_column,
            "threshold": label_threshold,
        }
    )
    train_ids = {record.session_id for record in train}
    dev_ids = {record.session_id for record in dev}
    test_ids = {record.session_id for record in test}
    if train_ids & dev_ids or train_ids & test_ids or dev_ids & test_ids:
        raise ValueError("train/dev/test session IDs must be disjoint")

    y_train = np.asarray([int(record.label) for record in train], dtype=int)
    minimum_class = int(np.min(np.bincount(y_train, minlength=2)))
    if minimum_class < 2:
        raise ValueError("training split must contain at least two examples of each class")
    effective_folds = min(folds, minimum_class)
    cross_validation = StratifiedKFold(
        n_splits=effective_folds,
        shuffle=True,
        random_state=seed,
    )

    all_records = [*train, *dev, *test]
    prediction_rows: list[dict[str, Any]] = []
    metrics_by_view: dict[str, Any] = {}
    probabilities_by_view: dict[str, dict[str, float]] = {}

    for view in views:
        train_text = [evidence_view(record.text, view) for record in train]
        estimator = _pipeline(seed, max_features)
        oof_probability = cross_val_predict(
            estimator,
            train_text,
            y_train,
            cv=cross_validation,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        threshold = select_balanced_accuracy_threshold(y_train, oof_probability)
        estimator.fit(train_text, y_train)

        view_dir = output_dir / view
        view_dir.mkdir(parents=True, exist_ok=True)
        dump(estimator, view_dir / "model.joblib")
        view_probabilities: dict[str, float] = {}
        view_metrics: dict[str, Any] = {
            "train_oof": _metrics(y_train, oof_probability, threshold)
        }

        for record, probability in zip(train, oof_probability):
            parsed_probability = float(probability)
            view_probabilities[record.session_id] = parsed_probability
            prediction_rows.append(
                {
                    "session_id": record.session_id,
                    "split": record.split,
                    "view": view,
                    "label": record.label,
                    "probability": parsed_probability,
                    "predicted_label": int(parsed_probability >= threshold),
                    "threshold": threshold,
                    "prediction_origin": "out_of_fold",
                }
            )

        for split_name, records in (("dev", dev), ("test", test)):
            if not records:
                continue
            probabilities = estimator.predict_proba(
                [evidence_view(record.text, view) for record in records]
            )[:, 1]
            labels: list[int] = []
            labeled_probabilities: list[float] = []
            for record, probability in zip(records, probabilities):
                parsed_probability = float(probability)
                view_probabilities[record.session_id] = parsed_probability
                prediction_rows.append(
                    {
                        "session_id": record.session_id,
                        "split": record.split,
                        "view": view,
                        "label": record.label,
                        "probability": parsed_probability,
                        "predicted_label": int(parsed_probability >= threshold),
                        "threshold": threshold,
                        "prediction_origin": "train_fitted",
                    }
                )
                if record.label is not None:
                    labels.append(record.label)
                    labeled_probabilities.append(parsed_probability)
            if labels:
                view_metrics[split_name] = _metrics(labels, labeled_probabilities, threshold)

        metrics_by_view[view] = view_metrics
        probabilities_by_view[view] = view_probabilities
        _write_json(view_dir / "metrics.json", view_metrics)

    combined_rows: list[dict[str, Any]] = []
    for record in all_records:
        combined_rows.append(
            {
                "session_id": record.session_id,
                "split": record.split,
                "label": record.label,
                "probabilities": {
                    view: probabilities_by_view[view][record.session_id] for view in views
                },
            }
        )
    _write_jsonl(output_dir / "predictions.jsonl", prediction_rows)
    _write_jsonl(output_dir / "combined_predictions.jsonl", combined_rows)

    summary = {
        "experiment": "tfidf_logistic_evidence_v2",
        "label_column": label_column,
        "label_threshold": label_threshold,
        "label_policy": label_policy,
        "views": list(views),
        "seed": seed,
        "requested_folds": folds,
        "effective_folds": effective_folds,
        "max_features": max_features,
        "session_counts": {
            "train": len(train),
            "dev": len(dev),
            "test": len(test),
        },
        "train_ids_sha256": _sha256_identifiers(train),
        "evidence_root": str(evidence_root.resolve()),
        "metrics": metrics_by_view,
        "fit_boundaries": {
            "vectorizer": "train_only",
            "classifier": "train_only",
            "threshold": "train_oof_only",
            "dev_used_for_fit": False,
            "test_used_for_fit": False,
            "test_labels_accessed": evaluate_labeled_test,
        },
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run split-safe OOF text baselines over native Evidence inputs."
    )
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--train-labels", required=True, type=Path)
    parser.add_argument("--dev-labels", required=True, type=Path)
    parser.add_argument("--test-labels", type=Path)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--label-threshold", type=float)
    parser.add_argument("--id-column")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--views",
        nargs="+",
        choices=SUPPORTED_VIEWS,
        default=list(SUPPORTED_VIEWS),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument(
        "--evaluate-labeled-test",
        action="store_true",
        help="Explicitly unlock labeled test metrics after the experiment is frozen.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_baseline(
        evidence_root=args.evidence_root,
        train_labels=args.train_labels,
        dev_labels=args.dev_labels,
        test_labels=args.test_labels,
        label_column=args.label_column,
        id_column=args.id_column,
        output_dir=args.output_dir,
        views=tuple(args.views),
        folds=args.folds,
        seed=args.seed,
        max_features=args.max_features,
        evaluate_labeled_test=args.evaluate_labeled_test,
        label_threshold=args.label_threshold,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
