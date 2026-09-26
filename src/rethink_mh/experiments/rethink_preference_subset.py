"""Freeze a memory-bounded ORPO optimization subset from a full preference audit set."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rethink_mh.experiments.rethink_post_training import audit_training_records
from rethink_mh.experiments.rethink_trajectories import select_optimization_preferences


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_preference_subset(
    source: Path,
    output_dir: Path,
    *,
    constraint_session_fraction: float = 0.10,
    seed: int = 42,
) -> dict[str, Any]:
    output = output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite preference subset: {output}")
    audited = audit_training_records("orpo", source)
    selected = select_optimization_preferences(
        audited.rows,
        constraint_session_fraction=constraint_session_fraction,
        seed=seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent)))
    records_text = "".join(_canonical_json(row) + "\n" for row in selected)
    records_path = temporary / "preferences_optimization.jsonl"
    records_path.write_text(records_text, encoding="utf-8")
    reduced_audit = audit_training_records("orpo", records_path)
    kinds = Counter(str(row["preference_kind"]) for row in selected)
    manifest = {
        "schema_version": "1.0.0",
        "package": "orpo_memory_bounded_optimization_subset",
        "dataset": audited.dataset,
        "source": {
            "path": str(source.resolve()),
            "sha256": _sha256(source),
            "record_count": len(audited.rows),
        },
        "selection": {
            "all_state_transition_pairs_retained": True,
            "constraint_session_fraction": constraint_session_fraction,
            "seed": seed,
            "record_count": len(selected),
            "session_count": reduced_audit.session_count,
            "preference_kind_counts": dict(sorted(kinds.items())),
        },
        "file": {
            "name": records_path.name,
            "sha256": hashlib.sha256(records_text.encode()).hexdigest(),
        },
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--constraint-session-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_preference_subset(
        args.source,
        args.output_dir,
        constraint_session_fraction=args.constraint_session_fraction,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
