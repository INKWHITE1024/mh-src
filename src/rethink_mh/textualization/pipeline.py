"""Orchestrate dataset audit, reference fitting, and evidence text compilation."""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .adapters import build_adapter
from .config import TextualizationConfig
from .readers import select_session_ids
from .protocols import build_protocol
from .references import (
    ReferenceSet,
    configuration_digest,
    enrich_session,
    reference_configuration,
)


ProgressCallback = Callable[[str], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def audit_dataset(
    root: str | Path,
    config: TextualizationConfig,
    split_file: str | Path | None = None,
    split_value: str | None = None,
    max_sessions: int | None = None,
    output_path: str | Path | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    adapter = build_adapter(root, config)
    session_ids = select_session_ids(
        adapter.list_session_ids(), split_file, split_value, max_sessions
    )
    sessions: list[dict[str, Any]] = []
    for index, session_id in enumerate(session_ids, start=1):
        result = adapter.audit_session(session_id)
        sessions.append(result)
        if progress:
            progress(f"audit {index}/{len(session_ids)} session={session_id} ok={result.get('ok')}")
    output = {
        "schema_version": "1.0.0",
        "created_at_utc": _utc_now(),
        "dataset": config.dataset_name,
        "dataset_key": config.dataset_key,
        "session_count": len(session_ids),
        "ok_count": sum(bool(item.get("ok")) for item in sessions),
        "error_count": sum(not bool(item.get("ok")) for item in sessions),
        "sessions": sessions,
    }
    if output_path is not None:
        _write_atomic(
            Path(output_path),
            json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
    return output


def fit_references(
    root: str | Path,
    config: TextualizationConfig,
    split_file: str | Path,
    output_path: str | Path,
    fit_split: str = "train",
    split_value: str | None = None,
    max_sessions: int | None = None,
    seed: int = 17,
    progress: ProgressCallback | None = None,
) -> ReferenceSet:
    if fit_split.strip().lower() not in {"train", "training"}:
        raise ValueError("fit_split must be named train or training")
    adapter = build_adapter(root, config)
    session_ids = select_session_ids(
        adapter.list_session_ids(), split_file, split_value, max_sessions
    )
    if not session_ids:
        raise ValueError("The selected training split is empty")

    def iter_units() -> Iterable[Any]:
        for index, session_id in enumerate(session_ids, start=1):
            if progress:
                progress(f"fit {index}/{len(session_ids)} session={session_id}")
            yield from adapter.load_session(session_id)

    references = ReferenceSet.fit(
        dataset=config.dataset_name,
        fit_split=fit_split,
        session_ids=session_ids,
        units=iter_units(),
        config=reference_configuration(config.to_dict()),
        reservoir_limit_per_feature=int(
            config.get("reference_reservoir_per_feature", 50_000)
        ),
        seed=seed,
    )
    references.save(output_path)
    return references


def compile_session(
    adapter: Any,
    session_id: str,
    references: ReferenceSet,
    output_root: str | Path,
    protocol_version: str = "native",
    overwrite: bool = False,
) -> dict[str, Any]:
    protocol = build_protocol(protocol_version, adapter.config)
    target = (
        Path(output_root)
        / protocol.version
        / adapter.config.dataset_key
        / str(session_id)
    )
    manifest_path = target / "manifest.json"
    expected_files = tuple(
        target / name
        for name in ("evidence_units.jsonl", *protocol.expected_filenames, "manifest.json")
    )
    if all(path.is_file() for path in expected_files) and not overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("reference_id") != references.reference_id:
            raise FileExistsError(
                f"Session {session_id} was compiled with another reference; pass --overwrite"
            )
        if existing.get("protocol_version") != protocol.version:
            raise FileExistsError(
                f"Session {session_id} has another protocol version; pass --overwrite"
            )
        existing["skipped"] = True
        return existing

    raw_units = adapter.load_session(str(session_id))
    units = sorted(
        enrich_session(raw_units, references),
        key=lambda item: (
            float(item.time_range["start_sec"]),
            0 if item.modality == "audio" else 1,
            item.evidence_id,
        ),
    )
    evidence_ids = [unit.evidence_id for unit in units]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError(f"Duplicate evidence IDs generated for session {session_id}")
    jsonl = "".join(unit.to_json() + "\n" for unit in units)
    protocol_artifacts = protocol.compile_session(units)
    target.mkdir(parents=True, exist_ok=True)
    _write_atomic(target / "evidence_units.jsonl", jsonl)
    for filename, content in protocol_artifacts.files.items():
        _write_atomic(target / filename, content)

    availability = Counter(str(unit.availability["status"]) for unit in units)
    source_counts = Counter(str(unit.source["source_id"]) for unit in units)
    manifest = {
        "schema_version": "1.0.0",
        "created_at_utc": _utc_now(),
        "dataset": adapter.config.dataset_name,
        "dataset_key": adapter.config.dataset_key,
        "session_id": str(session_id),
        "protocol_version": protocol.version,
        "primary_model_input": protocol_artifacts.primary_input_file,
        "protocol_metadata": protocol_artifacts.metadata,
        "reference_id": references.reference_id,
        "unit_count": len(units),
        "atomic_unit_count": protocol_artifacts.atomic_unit_count,
        "segment_count": protocol_artifacts.segment_count,
        "observation_count": sum(len(unit.observations) for unit in units),
        "availability_counts": dict(sorted(availability.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "files": {
            "evidence_units.jsonl": {"sha256": _sha256_text(jsonl)},
            **{
                filename: {"sha256": _sha256_text(content)}
                for filename, content in sorted(protocol_artifacts.files.items())
            },
        },
        "label_fields_read": [],
        "transcript_content_exposed": False,
        "skipped": False,
    }
    manifest_content = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _write_atomic(manifest_path, manifest_content)
    return manifest


def _compile_worker(payload: dict[str, Any]) -> dict[str, Any]:
    config = TextualizationConfig(
        values=payload["config_values"], source_path=payload.get("config_source_path")
    )
    adapter = build_adapter(payload["root"], config)
    references = ReferenceSet.load(payload["reference_path"])
    return compile_session(
        adapter=adapter,
        session_id=payload["session_id"],
        references=references,
        output_root=payload["output_root"],
        protocol_version=str(payload["protocol_version"]),
        overwrite=bool(payload["overwrite"]),
    )


def compile_dataset(
    root: str | Path,
    config: TextualizationConfig,
    reference_path: str | Path,
    output_root: str | Path,
    split_file: str | Path | None = None,
    split_name: str | None = None,
    split_value: str | None = None,
    max_sessions: int | None = None,
    workers: int = 1,
    protocol_version: str | None = None,
    overwrite: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    protocol_version = str(
        protocol_version or config.get("evidence_protocol_default", "native")
    ).lower()
    protocol = build_protocol(protocol_version, config)
    adapter = build_adapter(root, config)
    references = ReferenceSet.load(reference_path)
    if references.dataset != config.dataset_name:
        raise ValueError(
            f"Reference dataset {references.dataset!r} does not match {config.dataset_name!r}"
        )
    digest = configuration_digest(config.to_dict())
    if references.config_sha256 != digest:
        raise ValueError(
            "Textualization config differs from the config used to fit the reference statistics"
        )
    session_ids = select_session_ids(
        adapter.list_session_ids(), split_file, split_value, max_sessions
    )
    if not session_ids:
        raise ValueError("The selected compile split is empty")

    output_root = Path(output_root).expanduser().resolve()
    reference_path = Path(reference_path).expanduser().resolve()
    manifests: list[dict[str, Any]] = []
    if workers == 1:
        for index, session_id in enumerate(session_ids, start=1):
            manifest = compile_session(
                adapter,
                session_id,
                references,
                output_root,
                protocol.version,
                overwrite,
            )
            manifests.append(manifest)
            if progress:
                progress(
                    f"compile {index}/{len(session_ids)} session={session_id} "
                    f"units={manifest['unit_count']} skipped={manifest.get('skipped', False)}"
                )
    else:
        payloads = [
            {
                "root": str(Path(root).expanduser().resolve()),
                "config_values": config.to_dict(),
                "config_source_path": config.source_path,
                "reference_path": str(reference_path),
                "output_root": str(output_root),
                "session_id": session_id,
                "protocol_version": protocol.version,
                "overwrite": overwrite,
            }
            for session_id in session_ids
        ]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_id = {
                executor.submit(_compile_worker, payload): payload["session_id"]
                for payload in payloads
            }
            completed = 0
            for future in as_completed(future_to_id):
                manifest = future.result()
                manifests.append(manifest)
                completed += 1
                if progress:
                    progress(
                        f"compile {completed}/{len(session_ids)} "
                        f"session={future_to_id[future]} units={manifest['unit_count']} "
                        f"skipped={manifest.get('skipped', False)}"
                    )

    order = {session_id: index for index, session_id in enumerate(session_ids)}
    manifests.sort(key=lambda item: order[str(item["session_id"])])
    dataset_root = output_root / protocol.version / config.dataset_key
    sessions_jsonl = "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for item in manifests
    )
    _write_atomic(dataset_root / "sessions_manifest.jsonl", sessions_jsonl)
    summary = {
        "schema_version": "1.0.0",
        "created_at_utc": _utc_now(),
        "dataset": config.dataset_name,
        "dataset_key": config.dataset_key,
        "protocol_version": protocol.version,
        "split_name": split_name,
        "session_count": len(manifests),
        "compiled_count": sum(not item.get("skipped", False) for item in manifests),
        "skipped_count": sum(bool(item.get("skipped", False)) for item in manifests),
        "unit_count": sum(int(item["unit_count"]) for item in manifests),
        "atomic_unit_count": sum(int(item["atomic_unit_count"]) for item in manifests),
        "segment_count": sum(int(item["segment_count"]) for item in manifests),
        "observation_count": sum(int(item["observation_count"]) for item in manifests),
        "reference_id": references.reference_id,
        "reference_fit_split": references.fit_split,
        "label_fields_read": [],
        "transcript_content_exposed": False,
    }
    _write_atomic(
        dataset_root / "run_manifest.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return summary
