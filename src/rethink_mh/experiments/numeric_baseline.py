"""Direct numeric baselines over canonical Evidence Unit observations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .phq8_labels import build_phq8_label_audit, validate_safe_label_source

from .text_baseline import (
    SUPPORTED_VIEWS,
    SessionRecord,
    _metrics,
    _read_split,
    _sha256_identifiers,
    _write_json,
    _write_jsonl,
    select_balanced_accuracy_threshold,
)


SUPPORTED_MODELS = ("logistic", "extra_trees")


def _aggregate(prefix: str, values: Sequence[float], times: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    output = {
        f"{prefix}.count": float(array.size),
        f"{prefix}.mean": float(np.mean(array)),
        f"{prefix}.std": float(np.std(array)),
        f"{prefix}.median": float(np.median(array)),
        f"{prefix}.q10": float(np.quantile(array, 0.10)),
        f"{prefix}.q90": float(np.quantile(array, 0.90)),
        f"{prefix}.min": float(np.min(array)),
        f"{prefix}.max": float(np.max(array)),
    }
    time_array = np.asarray(times, dtype=float)
    if array.size >= 2 and float(np.ptp(time_array)) > 0.0:
        centered_time = time_array - float(np.mean(time_array))
        denominator = float(np.dot(centered_time, centered_time))
        output[f"{prefix}.slope_per_second"] = float(
            np.dot(centered_time, array - float(np.mean(array))) / denominator
        )
    return output


def extract_session_features(session_dir: str | Path) -> dict[str, float]:
    """Aggregate raw compiled measurements without using task labels or transcript text."""

    path = Path(session_dir) / "evidence_units.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing canonical evidence file: {path}")

    values: dict[str, list[float]] = defaultdict(list)
    times: dict[str, list[float]] = defaultdict(list)
    status_counts: dict[str, dict[str, int]] = {
        "audio": defaultdict(int),
        "visual": defaultdict(int),
    }
    modality_counts: dict[str, int] = defaultdict(int)
    duration = 0.0
    session_ids: set[str] = set()

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            modality = str(payload.get("modality"))
            if modality not in {"audio", "visual"}:
                raise ValueError(f"Unexpected modality at {path}:{line_number}")
            session_ids.add(str(payload.get("session_id")))
            time_range = payload.get("time_range", {})
            start = float(time_range["start_sec"])
            end = float(time_range["end_sec"])
            midpoint = (start + end) / 2.0
            duration = max(duration, end)
            modality_counts[modality] += 1

            availability = payload.get("availability", {})
            status = str(availability.get("status"))
            if status not in {"present", "partial", "unavailable"}:
                raise ValueError(f"Unexpected availability at {path}:{line_number}")
            status_counts[modality][status] += 1

            for key, raw_value in payload.get("quality", {}).items():
                if isinstance(raw_value, bool):
                    parsed_value = float(raw_value)
                elif isinstance(raw_value, (int, float)) and np.isfinite(float(raw_value)):
                    parsed_value = float(raw_value)
                else:
                    continue
                feature_key = f"{modality}.quality.{key}"
                values[feature_key].append(parsed_value)
                times[feature_key].append(midpoint)

            for observation in payload.get("observations", []):
                raw_value = observation.get("value")
                if not isinstance(raw_value, (int, float)) or not np.isfinite(float(raw_value)):
                    raise ValueError(f"Non-finite observation at {path}:{line_number}")
                feature_key = f"{modality}.observation.{observation['name']}"
                values[feature_key].append(float(raw_value))
                times[feature_key].append(midpoint)

    if len(session_ids) != 1:
        raise ValueError(f"Canonical file must contain exactly one session: {path}")
    features: dict[str, float] = {"session.duration_seconds": duration}
    for modality in ("audio", "visual"):
        total = modality_counts[modality]
        features[f"{modality}.unit_count"] = float(total)
        for status in ("present", "partial", "unavailable"):
            count = status_counts[modality][status]
            features[f"{modality}.availability.{status}_fraction"] = (
                float(count / total) if total else 0.0
            )
    for key in sorted(values):
        features.update(_aggregate(key, values[key], times[key]))
    if not all(np.isfinite(value) for value in features.values()):
        raise ValueError(f"Non-finite aggregate generated for {path}")
    return features


def feature_view(features: Mapping[str, float], view: str) -> dict[str, float]:
    if view not in SUPPORTED_VIEWS:
        raise ValueError(f"Unsupported feature view {view!r}")
    if view == "full":
        return dict(features)
    prefix = f"{view}."
    return {
        key: float(value)
        for key, value in features.items()
        if key.startswith(prefix) or key.startswith("session.")
    }


def _extract_one(payload: tuple[str, str]) -> tuple[str, dict[str, float]]:
    session_id, directory = payload
    return session_id, extract_session_features(directory)


def _extract_all(
    records: Sequence[SessionRecord], evidence_root: Path, workers: int
) -> dict[str, dict[str, float]]:
    payloads = [
        (record.session_id, str(evidence_root / record.session_id)) for record in records
    ]
    if workers == 1:
        return dict(_extract_one(payload) for payload in payloads)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return dict(executor.map(_extract_one, payloads))


def _build_estimator(name: str, seed: int):
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "logistic":
        return Pipeline(
            (
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=0.1,
                        class_weight="balanced",
                        max_iter=2_000,
                        random_state=seed,
                        solver="liblinear",
                    ),
                ),
            )
        )
    if name == "extra_trees":
        return Pipeline(
            (
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "classifier",
                    ExtraTreesClassifier(
                        n_estimators=500,
                        class_weight="balanced",
                        max_features="sqrt",
                        min_samples_leaf=3,
                        n_jobs=1,
                        random_state=seed,
                    ),
                ),
            )
        )
    raise ValueError(f"Unsupported model {name!r}")


def _matrix(
    records: Sequence[SessionRecord],
    features: Mapping[str, Mapping[str, float]],
    view: str,
    columns: Sequence[str],
) -> np.ndarray:
    frame = pd.DataFrame(
        [feature_view(features[record.session_id], view) for record in records]
    )
    return frame.reindex(columns=columns).to_numpy(dtype=float)


def run_numeric_baseline(
    *,
    evidence_root: Path,
    train_labels: Path,
    dev_labels: Path,
    test_labels: Path | None,
    label_column: str,
    id_column: str | None,
    output_dir: Path,
    views: Sequence[str],
    models: Sequence[str],
    folds: int,
    seed: int,
    workers: int,
    evaluate_labeled_test: bool = False,
    label_threshold: float | None = None,
) -> dict[str, Any]:
    try:
        from joblib import dump
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Numeric baselines require the project experiments optional dependencies"
        ) from exc

    invalid_views = sorted(set(views) - set(SUPPORTED_VIEWS))
    invalid_models = sorted(set(models) - set(SUPPORTED_MODELS))
    if invalid_views or invalid_models:
        raise ValueError(f"Unsupported views={invalid_views}, models={invalid_models}")
    if folds < 2 or workers <= 0:
        raise ValueError("folds must be at least two and workers must be positive")
    phq8_dataset = validate_safe_label_source(label_column, label_threshold)

    train = _read_split(
        train_labels, "train", evidence_root, label_column, id_column,
        require_labels=True, label_threshold=label_threshold
    )
    dev = _read_split(
        dev_labels, "dev", evidence_root, label_column, id_column,
        require_labels=True, label_threshold=label_threshold
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
    split_sets = [set(record.session_id for record in split) for split in (train, dev, test)]
    if split_sets[0] & split_sets[1] or split_sets[0] & split_sets[2] or split_sets[1] & split_sets[2]:
        raise ValueError("train/dev/test session IDs must be disjoint")

    all_records = [*train, *dev, *test]
    extracted = _extract_all(all_records, evidence_root, workers)
    _write_jsonl(
        output_dir / "features.jsonl",
        (
            {"session_id": record.session_id, "split": record.split, "features": extracted[record.session_id]}
            for record in all_records
        ),
    )

    y_train = np.asarray([int(record.label) for record in train], dtype=int)
    minimum_class = int(np.min(np.bincount(y_train, minlength=2)))
    effective_folds = min(folds, minimum_class)
    if effective_folds < 2:
        raise ValueError("training split must contain at least two examples of each class")
    cross_validation = StratifiedKFold(
        n_splits=effective_folds, shuffle=True, random_state=seed
    )

    prediction_rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    for view in views:
        train_feature_dicts = [feature_view(extracted[record.session_id], view) for record in train]
        columns = sorted(set().union(*(item.keys() for item in train_feature_dicts)))
        if not columns:
            raise ValueError(f"No training features for view {view}")
        x_train = pd.DataFrame(train_feature_dicts).reindex(columns=columns).to_numpy(dtype=float)
        for model_name in models:
            estimator = _build_estimator(model_name, seed)
            oof_probability = cross_val_predict(
                estimator,
                x_train,
                y_train,
                cv=cross_validation,
                method="predict_proba",
                n_jobs=1,
            )[:, 1]
            threshold = select_balanced_accuracy_threshold(y_train, oof_probability)
            estimator.fit(x_train, y_train)
            model_dir = output_dir / model_name / view
            model_dir.mkdir(parents=True, exist_ok=True)
            dump(estimator, model_dir / "model.joblib")
            _write_json(model_dir / "feature_columns.json", columns)

            key = f"{model_name}.{view}"
            key_metrics: dict[str, Any] = {
                "train_oof": _metrics(y_train, oof_probability, threshold)
            }
            for record, probability in zip(train, oof_probability):
                parsed = float(probability)
                prediction_rows.append(
                    {
                        "session_id": record.session_id,
                        "split": "train",
                        "view": view,
                        "model": model_name,
                        "label": record.label,
                        "probability": parsed,
                        "predicted_label": int(parsed >= threshold),
                        "threshold": threshold,
                        "prediction_origin": "out_of_fold",
                    }
                )

            for split_name, records in (("dev", dev), ("test", test)):
                if not records:
                    continue
                probabilities = estimator.predict_proba(
                    _matrix(records, extracted, view, columns)
                )[:, 1]
                labels: list[int] = []
                labeled_probabilities: list[float] = []
                for record, probability in zip(records, probabilities):
                    parsed = float(probability)
                    prediction_rows.append(
                        {
                            "session_id": record.session_id,
                            "split": split_name,
                            "view": view,
                            "model": model_name,
                            "label": record.label,
                            "probability": parsed,
                            "predicted_label": int(parsed >= threshold),
                            "threshold": threshold,
                            "prediction_origin": "train_fitted",
                        }
                    )
                    if record.label is not None:
                        labels.append(record.label)
                        labeled_probabilities.append(parsed)
                if labels:
                    key_metrics[split_name] = _metrics(labels, labeled_probabilities, threshold)
            metrics[key] = key_metrics
            _write_json(model_dir / "metrics.json", key_metrics)

    _write_jsonl(output_dir / "predictions.jsonl", prediction_rows)
    summary = {
        "experiment": "canonical_numeric_baseline",
        "label_column": label_column,
        "label_threshold": label_threshold,
        "label_policy": label_policy,
        "models": list(models),
        "views": list(views),
        "seed": seed,
        "requested_folds": folds,
        "effective_folds": effective_folds,
        "session_counts": {"train": len(train), "dev": len(dev), "test": len(test)},
        "train_ids_sha256": _sha256_identifiers(train),
        "evidence_root": str(evidence_root.resolve()),
        "feature_source": "raw canonical observation values and operational quality only",
        "metrics": metrics,
        "fit_boundaries": {
            "feature_columns": "train_only",
            "imputer": "train_only",
            "scaler": "train_only_when_used",
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
        description="Run split-safe direct numeric baselines over canonical Evidence Units."
    )
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--train-labels", required=True, type=Path)
    parser.add_argument("--dev-labels", required=True, type=Path)
    parser.add_argument("--test-labels", type=Path)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--label-threshold", type=float)
    parser.add_argument("--id-column")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--views", nargs="+", choices=SUPPORTED_VIEWS, default=list(SUPPORTED_VIEWS))
    parser.add_argument("--models", nargs="+", choices=SUPPORTED_MODELS, default=list(SUPPORTED_MODELS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--evaluate-labeled-test",
        action="store_true",
        help="Explicitly unlock labeled test metrics after the experiment is frozen.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_numeric_baseline(
        evidence_root=args.evidence_root,
        train_labels=args.train_labels,
        dev_labels=args.dev_labels,
        test_labels=args.test_labels,
        label_column=args.label_column,
        id_column=args.id_column,
        output_dir=args.output_dir,
        views=tuple(args.views),
        models=tuple(args.models),
        folds=args.folds,
        seed=args.seed,
        workers=args.workers,
        evaluate_labeled_test=args.evaluate_labeled_test,
        label_threshold=args.label_threshold,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
