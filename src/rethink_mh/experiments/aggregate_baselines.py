"""Aggregate repeated baseline summaries without selecting the best random seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .text_baseline import _write_json


def _numeric_metrics(payload: Mapping[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    output: dict[str, dict[str, dict[str, float]]] = {}
    for model_view, splits in payload["metrics"].items():
        output[model_view] = {}
        for split, metrics in splits.items():
            output[model_view][split] = {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float)) and key not in {"count", "positive_count", "negative_count"}
            }
    return output


def aggregate_summaries(
    summaries: Sequence[Mapping[str, Any]], *, expected_seeds: Sequence[int] | None = None
) -> dict[str, Any]:
    if not summaries:
        raise ValueError("at least one summary is required")
    experiment = summaries[0]["experiment"]
    label_column = summaries[0]["label_column"]
    evidence_root = summaries[0]["evidence_root"]
    session_counts = summaries[0]["session_counts"]
    seeds = [int(summary["seed"]) for summary in summaries]
    if len(set(seeds)) != len(seeds):
        raise ValueError("summary seeds must be unique")
    if expected_seeds is not None and set(seeds) != set(expected_seeds):
        raise ValueError(f"expected seeds {sorted(expected_seeds)}, found {sorted(seeds)}")
    for summary in summaries:
        identity = (
            summary["experiment"],
            summary["label_column"],
            summary["evidence_root"],
            summary["session_counts"],
        )
        if identity != (experiment, label_column, evidence_root, session_counts):
            raise ValueError("summaries do not describe the same experiment and splits")

    parsed = [_numeric_metrics(summary) for summary in summaries]
    model_views = set(parsed[0])
    if any(set(item) != model_views for item in parsed):
        raise ValueError("model/view keys differ across seeds")
    aggregate: dict[str, Any] = {}
    for model_view in sorted(model_views):
        splits = set(parsed[0][model_view])
        if any(set(item[model_view]) != splits for item in parsed):
            raise ValueError(f"split keys differ for {model_view}")
        aggregate[model_view] = {}
        for split in sorted(splits):
            metric_names = set(parsed[0][model_view][split])
            if any(set(item[model_view][split]) != metric_names for item in parsed):
                raise ValueError(f"metric keys differ for {model_view}.{split}")
            aggregate[model_view][split] = {}
            for metric in sorted(metric_names):
                values = np.asarray(
                    [item[model_view][split][metric] for item in parsed], dtype=float
                )
                aggregate[model_view][split][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "values_by_seed": {
                        str(seed): float(value)
                        for seed, value in sorted(zip(seeds, values), key=lambda item: item[0])
                    },
                }
    return {
        "aggregation": "mean_sample_std_across_predeclared_seeds",
        "experiment": experiment,
        "label_column": label_column,
        "evidence_root": evidence_root,
        "session_counts": session_counts,
        "seeds": sorted(seeds),
        "seed_count": len(seeds),
        "best_seed_selection": False,
        "metrics": aggregate,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate repeated baseline summary JSON files.")
    parser.add_argument("--summary", required=True, type=Path, nargs="+")
    parser.add_argument("--expected-seeds", type=int, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in args.summary]
    aggregate = aggregate_summaries(summaries, expected_seeds=args.expected_seeds)
    _write_json(args.output, aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
