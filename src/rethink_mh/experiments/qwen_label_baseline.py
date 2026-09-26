"""Text-only Qwen Thinker screening baseline over frozen native Evidence inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rethink_mh.rethinking.contracts import TaskSpec
from rethink_mh.rethinking.qwen import load_pretrained_model_on_device
from rethink_mh.textualization.references import ReferenceSet, session_ids_digest

from .d_vlog_supervised_prepare import natural_session_key
from .phq8_labels import build_phq8_label_audit, validate_phq8_score_policy
from .text_baseline import (
    SUPPORTED_VIEWS,
    SessionRecord,
    _metrics,
    _read_split,
    _sha256_identifiers,
    _write_json,
    _write_jsonl,
    evidence_view,
    select_balanced_accuracy_threshold,
)


SUPPORTED_DATASETS = ("daic_woz", "e_daic", "d_vlog")
SUPPORTED_EVIDENCE_PROTOCOLS = ("native",)
D_VLOG_MISSING_TRANSCRIPT_TEXT = (
    "TRANSCRIPT EVIDENCE | unavailable\n"
    "Unavailable: no transcript is supplied for this session; treat transcript "
    "evidence as missing information, not silence or zero.\n"
)
_TRANSCRIPT_TIERS = ("full", "compact", "essential", "minimal")
TEXT_LORA_TARGETS = (
    r"^model\.layers\.[0-9]+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(gate_proj|up_proj|down_proj))$"
)

_COMPACT_OBSERVATION_PATTERN = re.compile(
    r"^(?P<name>.+?) (?P<category>very low|very high|low|typical|high) "
    r"\(median percentile (?P<percentile>\d+); (?P<detail>.+)\)$"
)
_COMPACT_PREVALENCE_PATTERN = re.compile(
    r"(?:below the training reference|within the central training-reference range|"
    r"above the training reference) in (?P<percent>\d+)% of "
    r"(?P<count>\d+) valid windows"
)
_COMPACT_TREND_PATTERN = re.compile(
    r"percentile (?P<direction>increased|decreased) across the segment"
)
_COMPACT_ANCHORS = {
    "audio": {
        "participant speaking coverage",
        "transcribed speech coverage (speaker unassigned)",
        "voiced frame coverage",
        "pitch variability",
        "loudness variability",
    },
    "visual": {
        "facial action variability",
        "head rotation variability",
        "gaze direction variability",
        "facial landmark shape change",
    },
}
_OPAQUE_COMPACT_PATTERN = re.compile(
    r"\b(?:spk|au_var|head_var|gaze_var|pitch_var|loud_var|subject_z|sz)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class PromptExample:
    session_id: str
    split: str
    input_ids: tuple[int, ...]
    label: int | None
    view_name: str = "standard"
    evidence_tokens: int = 0
    evidence_density_tier: str = "full"
    transcript_tokens: int = 0
    transcript_density_tier: str = "none"


def with_d_vlog_missing_transcript(
    records: Sequence[SessionRecord],
) -> list[SessionRecord]:
    """Attach one readable, content-free transcript marker to every D-Vlog split."""

    tiers = tuple(
        (tier, D_VLOG_MISSING_TRANSCRIPT_TEXT) for tier in _TRANSCRIPT_TIERS
    )
    output: list[SessionRecord] = []
    for record in records:
        if record.transcript_tiers:
            raise ValueError(
                f"D-Vlog session {record.session_id} unexpectedly has transcript content"
            )
        output.append(
            replace(
                record,
                transcript_tiers=tiers,
                transcript_speaker_policy="unavailable",
                transcript_source_kind="not_supplied",
            )
        )
    return output


def balanced_memorization_subset(
    records: Sequence[SessionRecord], *, per_class: int, seed: int
) -> list[SessionRecord]:
    """Select a deterministic balanced subset for a train-set memorization gate."""

    if per_class <= 0:
        raise ValueError("per_class must be positive")
    rng = np.random.default_rng(seed)
    selected_indices: set[int] = set()
    for label in (0, 1):
        indices = np.asarray(
            [index for index, record in enumerate(records) if record.label == label],
            dtype=int,
        )
        if len(indices) < per_class:
            raise ValueError(
                f"Class {label} has {len(indices)} records, fewer than {per_class}"
            )
        rng.shuffle(indices)
        selected_indices.update(int(index) for index in indices[:per_class])
    return [
        record for index, record in enumerate(records) if index in selected_indices
    ]


def _weighted_cross_entropy(
    functional: Any,
    logits: Any,
    labels: Any,
    class_weights: Any | None,
) -> Any:
    """Average weighted CE without PyTorch cancelling weights at batch size one."""

    batch_size = len(labels)
    if batch_size <= 0:
        raise ValueError("classification batch cannot be empty")
    total = functional.cross_entropy(
        logits,
        labels,
        weight=class_weights,
        reduction="sum",
    )
    return total / batch_size


def _accumulation_window_divisor(
    step: int, total_steps: int, gradient_accumulation: int
) -> int:
    if not 1 <= step <= total_steps:
        raise ValueError("step must be within total_steps")
    if gradient_accumulation <= 0:
        raise ValueError("gradient_accumulation must be positive")
    window_start = ((step - 1) // gradient_accumulation) * gradient_accumulation
    return min(gradient_accumulation, total_steps - window_start)


def load_task(path: str | Path) -> TaskSpec:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return TaskSpec.from_dict(payload)


def validate_d_vlog_supervised_contract(
    label_column: str,
    label_threshold: float | None,
    task: TaskSpec,
) -> None:
    """Reject D-Vlog training that does not use its dataset-specific target."""

    if label_column.strip().casefold() != "target":
        raise ValueError("D-Vlog supervised training requires label_column=target")
    if (
        label_threshold is None
        or not math.isfinite(float(label_threshold))
        or float(label_threshold) != 0.5
    ):
        raise ValueError("D-Vlog supervised training requires label_threshold=0.5")
    if task.task_id != "d_vlog_current_depression":
        raise ValueError(
            "D-Vlog supervised training requires task_id=d_vlog_current_depression"
        )


def _load_d_vlog_preparation_manifest(
    directory: Path,
) -> tuple[Path, dict[str, Any]]:
    manifest_path = directory.resolve() / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"prepared D-Vlog split manifest is missing: {manifest_path}"
        ) from error
    expected_mapping = {"depression": 1, "normal": 0}
    label_policy = manifest.get("label_policy")
    source_label_file = manifest.get("source_label_file")
    source_label_file_sha256 = manifest.get("source_label_file_sha256")
    if (
        manifest.get("schema_version") != "1.0.0"
        or manifest.get("dataset") != "d_vlog"
        or manifest.get("test_label_values_consumed") is not False
        or not isinstance(source_label_file, str)
        or not source_label_file.strip()
        or not isinstance(source_label_file_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_label_file_sha256) is None
        or not isinstance(label_policy, dict)
        or label_policy.get("task_id") != "d_vlog_current_depression"
        or label_policy.get("source_value_mapping") != expected_mapping
        or label_policy.get("target_column") != "target"
        or label_policy.get("test_targets_exported") is not False
    ):
        raise ValueError("D-Vlog preparation manifest has an unsafe label policy")
    return manifest_path, manifest


def build_d_vlog_label_policy(
    paths: Sequence[Path], *, id_column: str | None = None
) -> dict[str, Any]:
    """Audit prepared D-Vlog train/valid targets without opening test truth."""

    if not paths:
        raise ValueError("D-Vlog label policy requires at least one prepared label file")
    if id_column is not None and id_column.strip().casefold() != "session_id":
        raise ValueError("prepared D-Vlog labels require id_column=session_id")
    parent = paths[0].parent.resolve()
    if any(path.parent.resolve() != parent for path in paths):
        raise ValueError("prepared D-Vlog label files must share one manifest directory")
    manifest_path, manifest = _load_d_vlog_preparation_manifest(parent)
    expected_mapping = {"depression": 1, "normal": 0}

    files: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    total_counts: Counter[int] = Counter()
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["session_id", "target", "split"]:
                raise ValueError(
                    "prepared D-Vlog label files require "
                    "session_id,target,split columns"
                )
            ids: list[str] = []
            counts: Counter[int] = Counter()
            observed_split: str | None = None
            for line_number, row in enumerate(reader, 2):
                if None in row:
                    raise ValueError(f"extra cell at {path}:{line_number}")
                session_id = str(row.get("session_id", "")).strip()
                split = str(row.get("split", "")).strip().casefold()
                if split not in {"train", "valid"}:
                    raise ValueError(
                        f"D-Vlog training label file contains split {split!r}"
                    )
                if observed_split is None:
                    observed_split = split
                elif split != observed_split:
                    raise ValueError("one prepared label file cannot mix splits")
                if not session_id or session_id in seen_ids:
                    raise ValueError(
                        f"invalid or overlapping D-Vlog ID at {path}:{line_number}"
                    )
                try:
                    numeric = float(str(row.get("target", "")).strip())
                except ValueError as error:
                    raise ValueError(
                        f"invalid D-Vlog target at {path}:{line_number}"
                    ) from error
                target = int(numeric)
                if numeric != target or target not in {0, 1}:
                    raise ValueError(
                        f"D-Vlog target must be 0 or 1 at {path}:{line_number}"
                    )
                seen_ids.add(session_id)
                ids.append(session_id)
                counts[target] += 1
                total_counts[target] += 1
        if not ids or observed_split is None:
            raise ValueError(f"prepared D-Vlog label file is empty: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        ids_digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
        prepared_file = manifest.get("files", {}).get(observed_split)
        if (
            not isinstance(prepared_file, dict)
            or prepared_file.get("sha256") != digest
            or int(prepared_file.get("count", -1)) != len(ids)
            or prepared_file.get("ids_sha256") != ids_digest
            or prepared_file.get("columns")
            != ["session_id", "target", "split"]
            or prepared_file.get("class_counts")
            != {"0": counts[0], "1": counts[1]}
        ):
            raise ValueError(
                f"prepared D-Vlog {observed_split} labels differ from manifest"
            )
        files.append(
            {
                "path": str(path.resolve()),
                "sha256": digest,
                "split": observed_split,
                "row_count": len(ids),
                "ids_sha256": ids_digest,
                "class_counts": {"0": counts[0], "1": counts[1]},
            }
        )
    return {
        "policy": "d_vlog_current_depression_target",
        "dataset": "d_vlog",
        "task_id": "d_vlog_current_depression",
        "target_column": "target",
        "target_threshold": 0.5,
        "target_definition": "1=depression; 0=normal",
        "source_label_semantics": "released D-Vlog current depression annotation",
        "source_value_mapping": expected_mapping,
        "negative_source_value": "normal",
        "positive_source_value": "depression",
        "test_labels_accessed": False,
        "preparation_manifest_path": str(manifest_path.resolve()),
        "preparation_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "source_label_file": manifest.get("source_label_file"),
        "source_label_file_sha256": manifest.get("source_label_file_sha256"),
        "source_file_sha256": manifest.get("source_label_file_sha256"),
        "class_counts": {"0": total_counts[0], "1": total_counts[1]},
        "files": files,
    }


def audit_d_vlog_test_ids(
    path: Path, *, expected_count: int = 212
) -> dict[str, Any]:
    """Validate the physically label-free D-Vlog test interface."""

    if expected_count <= 0:
        raise ValueError("expected D-Vlog test count must be positive")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["session_id", "split"]:
            raise ValueError(
                "D-Vlog test IDs must contain exactly session_id,split columns"
            )
        ids: list[str] = []
        seen: set[str] = set()
        for line_number, row in enumerate(reader, 2):
            if None in row:
                raise ValueError(f"extra cell at {path}:{line_number}")
            session_id = str(row.get("session_id", "")).strip()
            if not session_id or session_id in seen:
                raise ValueError(
                    f"invalid or duplicate D-Vlog test ID at {path}:{line_number}"
                )
            if row.get("split") != "test":
                raise ValueError(f"non-test row at {path}:{line_number}")
            seen.add(session_id)
            ids.append(session_id)
    if len(ids) != expected_count:
        raise ValueError(
            f"expected {expected_count} D-Vlog test IDs, observed {len(ids)}"
        )
    if ids != sorted(ids, key=natural_session_key):
        raise ValueError("D-Vlog test IDs must use deterministic natural order")
    file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    ids_digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    _, manifest = _load_d_vlog_preparation_manifest(path.parent)
    prepared_file = manifest.get("files", {}).get("test")
    if (
        not isinstance(prepared_file, dict)
        or prepared_file.get("sha256") != file_digest
        or int(prepared_file.get("count", -1)) != len(ids)
        or prepared_file.get("ids_sha256") != ids_digest
        or prepared_file.get("columns") != ["session_id", "split"]
        or prepared_file.get("label_free") is not True
    ):
        raise ValueError("prepared D-Vlog test IDs differ from manifest")
    return {
        "path": str(path.resolve()),
        "count": len(ids),
        "file_sha256": file_digest,
        "ids_sha256": ids_digest,
        "columns": ["session_id", "split"],
        "label_accessed": False,
    }


def build_d_vlog_evidence_metadata(evidence_root: Path) -> dict[str, Any]:
    """Return the fail-closed Evidence provenance required by supervised runs."""

    run_manifest_path = evidence_root / "run_manifest.json"
    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_key") != "d_vlog":
        raise ValueError("D-Vlog Evidence run manifest dataset mismatch")
    if manifest.get("protocol_version") != "native":
        raise ValueError("D-Vlog supervised training requires the native Evidence protocol")
    if manifest.get("reference_fit_split") != "train":
        raise ValueError("D-Vlog Evidence reference must be fit on official train only")
    if manifest.get("label_fields_read") != []:
        raise ValueError("D-Vlog Evidence compiler reports label access")
    if manifest.get("transcript_content_exposed") is not False:
        raise ValueError("D-Vlog Evidence unexpectedly exposes transcript content")
    reference_id = manifest.get("reference_id")
    if not isinstance(reference_id, str) or not reference_id:
        raise ValueError("D-Vlog Evidence run manifest lacks reference_id")

    declared_reference = manifest.get("reference_path")
    if declared_reference is not None:
        if not isinstance(declared_reference, str) or not declared_reference.strip():
            raise ValueError("D-Vlog Evidence reference_path must be a path string")
        reference_path = Path(declared_reference)
        if not reference_path.is_absolute():
            reference_path = evidence_root / reference_path
    else:
        reference_path = evidence_root.resolve().parent / "d_vlog_reference.train.json"
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"D-Vlog frozen train reference is missing: {reference_path}"
        )
    reference = ReferenceSet.load(reference_path)
    if (
        reference.dataset != "D-Vlog"
        or reference.fit_split != "train"
        or reference.reference_id != reference_id
    ):
        raise ValueError(
            "D-Vlog reference.train.json identity differs from run manifest"
        )
    return {
        "root": str(evidence_root.resolve()),
        "protocol_version": "native",
        "run_manifest_sha256": hashlib.sha256(
            run_manifest_path.read_bytes()
        ).hexdigest(),
        "reference_id": reference_id,
        "reference_fit_scope": "official_train_only",
        "reference_path": str(reference_path.resolve()),
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
    }


def build_label_messages(
    task: TaskSpec, session_text: str
) -> list[dict[str, str]]:
    """Build a label-free prompt whose supervised answer is one ASCII digit."""

    if not isinstance(session_text, str) or not session_text.strip():
        raise ValueError("session_text must be non-empty")
    system = (
        "You are a text-only evidence classifier in a non-diagnostic mental-health "
        "screening research pipeline. Use only the supplied measurement and transcript "
        "evidence. Transcript language is observed interview content, not a symptom label "
        "or clinical conclusion; obey its stated speaker-role policy. "
        "Treat unavailable or partial evidence as missing information, not as zero. "
        "Training-reference ranks are measurements, not symptom labels. Reply with "
        "exactly one ASCII digit and no other text: 0 for the negative class or 1 for "
        "the positive class."
    )
    user = (
        "TASK SPECIFICATION\n"
        f"Task identifier: {task.task_id}\n"
        f"Label definition: {task.label_definition}\n"
        f"Negative class (answer 0): {task.negative_label_name}\n"
        f"Positive class (answer 1): {task.positive_label_name}\n\n"
        "SESSION EVIDENCE\n"
        f"{session_text.rstrip()}\n\n"
        "OUTPUT\nReply with 0 or 1 only."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def resolve_label_token_ids(tokenizer: Any) -> tuple[int, int]:
    token_ids: list[int] = []
    for label in ("0", "1"):
        encoded = list(tokenizer.encode(label, add_special_tokens=False))
        if len(encoded) != 1:
            raise ValueError(
                f"Qwen class answer {label!r} must be exactly one token, got {encoded}"
            )
        token_ids.append(int(encoded[0]))
    if token_ids[0] == token_ids[1]:
        raise ValueError("negative and positive answer tokens must differ")
    return token_ids[0], token_ids[1]


def tokenize_records(
    records: Sequence[SessionRecord],
    tokenizer: Any,
    task: TaskSpec,
    *,
    view: str,
    max_input_tokens: int,
    evidence_density: str = "full",
    max_evidence_tokens: int = 2_000,
    max_transcript_tokens: int = 3_000,
    evidence_protocol: str = "native",
) -> list[PromptExample]:
    if view not in SUPPORTED_VIEWS:
        raise ValueError(f"Unsupported evidence view {view!r}")
    if max_input_tokens <= 0:
        raise ValueError("max_input_tokens must be positive")
    if evidence_density != "full":
        raise ValueError("native Evidence is always used at full density")
    if evidence_protocol not in SUPPORTED_EVIDENCE_PROTOCOLS:
        raise ValueError(
            f"Unsupported Evidence protocol {evidence_protocol!r}"
        )
    if max_evidence_tokens <= 0:
        raise ValueError("max_evidence_tokens must be positive")
    if max_transcript_tokens <= 0:
        raise ValueError("max_transcript_tokens must be positive")

    examples: list[PromptExample] = []
    for record in records:
        source_text = evidence_view(record.text, view)
        evidence_candidates = (("full", source_text),)
        transcript_by_tier = dict(record.transcript_tiers)
        if record.transcript_tiers:
            if set(transcript_by_tier) != {
                "full",
                "compact",
                "essential",
                "minimal",
            }:
                raise ValueError(
                    f"Session {record.session_id} has incomplete transcript tiers"
                )
            pairings = (
                ("full", "full"),
                ("full", "compact"),
                ("full", "essential"),
                ("full", "minimal"),
            )
            evidence_by_tier = dict(evidence_candidates)
            candidates = tuple(
                (
                    evidence_tier,
                    transcript_tier,
                    evidence_by_tier[evidence_tier],
                    transcript_by_tier[transcript_tier],
                )
                for evidence_tier, transcript_tier in pairings
            )
        else:
            candidates = tuple(
                (evidence_tier, "none", rendered_text, "")
                for evidence_tier, rendered_text in evidence_candidates
            )

        chosen: tuple[str, str, tuple[int, ...], int, int] | None = None
        observed: list[str] = []
        for (
            density_tier,
            transcript_density_tier,
            measurement_text,
            transcript_text,
        ) in candidates:
            evidence_encoded = tokenizer.encode(
                measurement_text, add_special_tokens=False
            )
            evidence_token_count = len(evidence_encoded)
            transcript_token_count = (
                len(tokenizer.encode(transcript_text, add_special_tokens=False))
                if transcript_text
                else 0
            )
            if transcript_text:
                rendered_text = (
                    "MULTIMODAL MEASUREMENT EVIDENCE\n"
                    f"{measurement_text.rstrip()}\n\n"
                    f"{transcript_text.rstrip()}\n"
                )
            else:
                rendered_text = measurement_text
            messages = build_label_messages(task, rendered_text)
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            if hasattr(encoded, "tolist"):
                encoded = encoded.tolist()
            input_ids = tuple(int(token_id) for token_id in encoded)
            observed.append(
                f"{density_tier}+{transcript_density_tier}:"
                f"evidence={evidence_token_count}:"
                f"transcript={transcript_token_count}:total={len(input_ids)}"
            )
            if not input_ids:
                raise ValueError(
                    f"Tokenizer produced no tokens for session {record.session_id}"
                )
            if (
                evidence_token_count <= max_evidence_tokens
                and transcript_token_count <= max_transcript_tokens
                and len(input_ids) <= max_input_tokens
            ):
                chosen = (
                    density_tier,
                    transcript_density_tier,
                    input_ids,
                    evidence_token_count,
                    transcript_token_count,
                )
                break
        if chosen is None:
            raise ValueError(
                f"Session {record.session_id} exceeds the Evidence/transcript prompt "
                f"budgets without safe truncation ({', '.join(observed)})"
            )
        (
            density_tier,
            transcript_density_tier,
            input_ids,
            evidence_token_count,
            transcript_token_count,
        ) = chosen
        examples.append(
            PromptExample(
                session_id=record.session_id,
                split=record.split,
                input_ids=input_ids,
                label=record.label,
                evidence_tokens=evidence_token_count,
                evidence_density_tier=density_tier,
                transcript_tokens=transcript_token_count,
                transcript_density_tier=transcript_density_tier,
            )
        )
    return examples


def participant_stratified_fold(
    records: Sequence[SessionRecord],
    *,
    folds: int,
    fold_index: int,
    seed: int,
) -> tuple[list[SessionRecord], list[SessionRecord]]:
    """Return a deterministic participant-level fold without optional dependencies."""

    if folds < 2 or not 0 <= fold_index < folds:
        raise ValueError("folds must be at least two and fold_index must be in range")
    if any(record.label not in {0, 1} for record in records):
        raise ValueError("OOF records must all have binary labels")

    rng = np.random.default_rng(seed)
    validation_indices: set[int] = set()
    for label in (0, 1):
        indices = np.asarray(
            [index for index, record in enumerate(records) if record.label == label],
            dtype=int,
        )
        if len(indices) < folds:
            raise ValueError(
                f"Class {label} has {len(indices)} participants, fewer than {folds} folds"
            )
        rng.shuffle(indices)
        validation_indices.update(int(index) for index in indices[fold_index::folds])

    train = [record for index, record in enumerate(records) if index not in validation_indices]
    validation = [
        record for index, record in enumerate(records) if index in validation_indices
    ]
    train_ids = {record.session_id for record in train}
    validation_ids = {record.session_id for record in validation}
    if not train or not validation or train_ids & validation_ids:
        raise AssertionError("Invalid participant-level fold construction")
    return train, validation


def validate_dataset_identity(
    records: Sequence[SessionRecord],
    evidence_root: Path,
    dataset: str,
    transcript_root: Path | None = None,
    evidence_protocol: str = "native",
) -> None:
    """Reject a run whose declared dataset disagrees with compiled manifests."""

    for record in records:
        manifest_path = evidence_root / record.session_id / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing Evidence manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual = manifest.get("dataset_key")
        if actual != dataset:
            raise ValueError(
                f"Dataset isolation failure for session {record.session_id}: "
                f"declared {dataset!r}, Evidence manifest says {actual!r}"
            )
        if manifest.get("protocol_version") != evidence_protocol:
            raise ValueError(
                f"Evidence protocol mismatch for session {record.session_id}: "
                f"expected {evidence_protocol!r}, got "
                f"{manifest.get('protocol_version')!r}"
            )
        if transcript_root is not None:
            transcript_manifest_path = (
                transcript_root / record.session_id / "transcript_manifest.compact.json"
            )
            transcript_manifest = json.loads(
                transcript_manifest_path.read_text(encoding="utf-8")
            )
            transcript_dataset = transcript_manifest.get("dataset_key")
            if transcript_dataset != dataset:
                raise ValueError(
                    f"Dataset isolation failure for session {record.session_id}: "
                    f"declared {dataset!r}, transcript manifest says "
                    f"{transcript_dataset!r}"
                )
            expected_transcript = {
                "daic_woz": ("participant_only", "manual_transcript"),
                "e_daic": (
                    "all_speakers_unassigned",
                    "automatic_speech_recognition",
                ),
            }[dataset]
            actual_transcript = (
                transcript_manifest.get("speaker_policy"),
                transcript_manifest.get("source_kind"),
            )
            if actual_transcript != expected_transcript:
                raise ValueError(
                    f"Transcript provenance mismatch for session "
                    f"{record.session_id}: expected {expected_transcript!r}, "
                    f"got {actual_transcript!r}"
                )


def validate_oof_reference_provenance(
    train_records: Sequence[SessionRecord],
    validation_records: Sequence[SessionRecord],
    *,
    evidence_root: Path,
    reference_path: Path,
    dataset: str,
    transcript_root: Path | None = None,
    evidence_protocol: str = "native",
) -> dict[str, Any]:
    """Prove that an OOF Evidence reference excludes the held-out fold."""

    if not train_records or not validation_records:
        raise ValueError("strict OOF provenance requires non-empty fit and holdout sets")
    fit_ids = {record.session_id for record in train_records}
    holdout_ids = {record.session_id for record in validation_records}
    if fit_ids & holdout_ids:
        raise ValueError("strict OOF fit and holdout participants overlap")
    reference = ReferenceSet.load(reference_path)
    expected_dataset = {
        "daic_woz": "DAIC-WOZ",
        "e_daic": "E-DAIC",
        "d_vlog": "D-Vlog",
    }[dataset]
    if reference.dataset != expected_dataset:
        raise ValueError(
            f"OOF reference dataset mismatch: {reference.dataset!r} != "
            f"{expected_dataset!r}"
        )
    expected_digest = session_ids_digest(fit_ids)
    if (
        reference.session_count != len(fit_ids)
        or reference.session_ids_sha256 != expected_digest
    ):
        raise ValueError(
            "OOF reference fit participant set does not exactly match the "
            "non-holdout fold"
        )

    manifest_rows: list[str] = []
    for record in sorted(
        [*train_records, *validation_records], key=lambda item: item.session_id
    ):
        manifest_path = evidence_root / record.session_id / "manifest.json"
        manifest_text = manifest_path.read_text(encoding="utf-8")
        manifest = json.loads(manifest_text)
        if manifest.get("reference_id") != reference.reference_id:
            raise ValueError(
                f"Session {record.session_id} does not use the fold-local reference"
            )
        if manifest.get("protocol_version") != evidence_protocol:
            raise ValueError(
                f"Session {record.session_id} is not compiled as Evidence "
                f"{evidence_protocol}"
            )
        if manifest.get("label_fields_read") != []:
            raise ValueError(
                f"Evidence compiler label boundary is invalid for {record.session_id}"
            )
        if manifest.get("transcript_content_exposed") is not False:
            raise ValueError(
                f"A/V Evidence unexpectedly embeds transcript for {record.session_id}"
            )
        if transcript_root is not None:
            transcript_manifest_path = (
                transcript_root / record.session_id / "transcript_manifest.compact.json"
            )
            transcript_manifest = json.loads(
                transcript_manifest_path.read_text(encoding="utf-8")
            )
        manifest_rows.append(
            f"{record.session_id}:"
            f"{hashlib.sha256(manifest_text.encode('utf-8')).hexdigest()}"
        )

    run_manifest_path = evidence_root / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("reference_id") != reference.reference_id:
        raise ValueError("OOF Evidence run manifest has another reference")
    if int(run_manifest.get("session_count", -1)) != len(fit_ids | holdout_ids):
        raise ValueError("OOF Evidence run manifest has incomplete participant coverage")
    if run_manifest.get("label_fields_read") != []:
        raise ValueError("OOF Evidence run manifest reports label access")
    return {
        "strict": True,
        "evidence_protocol": evidence_protocol,
        "reference_id": reference.reference_id,
        "reference_path": str(reference_path.resolve()),
        "reference_fit_count": reference.session_count,
        "reference_session_ids_sha256": reference.session_ids_sha256,
        "holdout_excluded_from_reference": True,
        "all_evidence_uses_fold_reference": True,
        "evidence_compiler_label_access": False,
        "transcript_alignment_verified": transcript_root is not None,
        "transcript_alignment_mode": (
            "session_level_parallel_layer"
            if transcript_root is not None
            else "not_enabled"
        ),
        "evidence_manifest_set_sha256": hashlib.sha256(
            "\n".join(manifest_rows).encode("utf-8")
        ).hexdigest(),
    }


def _runtime() -> Mapping[str, Any]:
    try:
        import torch
        import torch.nn.functional as functional
        from peft import LoraConfig, get_peft_model
        from safetensors.torch import save_file as save_safetensors
        from torch.utils.data import DataLoader
        from transformers import (
            AutoTokenizer,
            Qwen2_5OmniThinkerForConditionalGeneration,
            get_scheduler,
        )
    except (ImportError, RuntimeError) as exc:  # pragma: no cover - server runtime
        raise RuntimeError(
            "Qwen label training requires torch, transformers, accelerate, and peft"
        ) from exc
    return {
        "torch": torch,
        "functional": functional,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "save_safetensors": save_safetensors,
        "DataLoader": DataLoader,
        "AutoTokenizer": AutoTokenizer,
        "QwenModel": Qwen2_5OmniThinkerForConditionalGeneration,
        "get_scheduler": get_scheduler,
    }


def _collator(torch: Any, pad_token_id: int):
    def collate(examples: Sequence[PromptExample]) -> dict[str, Any]:
        maximum = max(len(example.input_ids) for example in examples)
        input_rows: list[list[int]] = []
        mask_rows: list[list[int]] = []
        labels: list[int] = []
        for example in examples:
            padding = maximum - len(example.input_ids)
            input_rows.append([*example.input_ids, *([pad_token_id] * padding)])
            mask_rows.append([1] * len(example.input_ids) + [0] * padding)
            labels.append(-1 if example.label is None else int(example.label))
        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "session_ids": [example.session_id for example in examples],
        }

    return collate


def _class_logits(
    peft_model: Any,
    input_ids: Any,
    attention_mask: Any,
    label_token_ids: tuple[int, int],
    torch: Any,
    classification_head: Any | None = None,
) -> Any:
    """Compute only two last-token logits instead of an L x vocabulary tensor."""

    thinker = peft_model.get_base_model()
    inputs_embeds = thinker.get_input_embeddings()(input_ids)
    text_positions = attention_mask.long().cumsum(dim=-1) - 1
    text_positions.masked_fill_(attention_mask == 0, 1)
    position_ids = text_positions.unsqueeze(0).expand(3, -1, -1)
    outputs = thinker.model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        inputs_embeds=inputs_embeds,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    hidden = outputs.last_hidden_state
    final_indices = attention_mask.sum(dim=-1) - 1
    final_hidden = hidden[
        torch.arange(hidden.shape[0], device=hidden.device), final_indices
    ]
    with torch.autocast(device_type=final_hidden.device.type, enabled=False):
        if classification_head is not None:
            return classification_head(final_hidden.float())
        token_indices = torch.as_tensor(
            label_token_ids,
            dtype=torch.long,
            device=thinker.lm_head.weight.device,
        )
        weight = thinker.lm_head.weight.index_select(0, token_indices).float()
        bias = thinker.lm_head.bias
        selected_bias = (
            None
            if bias is None
            else bias.index_select(0, token_indices).float()
        )
        return torch.nn.functional.linear(
            final_hidden.float(), weight, selected_bias
        )


def _build_classification_head(
    peft_model: Any,
    label_token_ids: tuple[int, int],
    torch: Any,
) -> Any:
    """Create an FP32 binary head initialized from the readable 0/1 tokens."""

    thinker = peft_model.get_base_model()
    lm_head = thinker.lm_head
    token_indices = torch.as_tensor(
        label_token_ids,
        dtype=torch.long,
        device=lm_head.weight.device,
    )
    head = torch.nn.Linear(
        int(lm_head.weight.shape[1]),
        2,
        bias=True,
        device=lm_head.weight.device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        head.weight.copy_(lm_head.weight.index_select(0, token_indices).float())
        head.bias.zero_()
    return head


def _solve_prior_bias_delta(
    margins: Sequence[float], target_positive_rate: float
) -> float:
    """Find an intercept shift whose mean sigmoid matches a fit-only prior."""

    if not margins:
        raise ValueError("prior calibration requires at least one fit margin")
    if not 0.0 < target_positive_rate < 1.0:
        raise ValueError("target positive rate must be strictly between zero and one")
    values = np.asarray(margins, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("prior calibration margins must all be finite")
    lower = -80.0
    upper = 80.0
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        probabilities = 1.0 / (
            1.0 + np.exp(-np.clip(values + midpoint, -60.0, 60.0))
        )
        if float(np.mean(probabilities)) < target_positive_rate:
            lower = midpoint
        else:
            upper = midpoint
    return float((lower + upper) / 2.0)


def _evaluate(
    model: Any,
    examples: Sequence[PromptExample],
    *,
    batch_size: int,
    collate: Any,
    label_token_ids: tuple[int, int],
    device: Any,
    runtime: Mapping[str, Any],
    classification_head: Any | None = None,
) -> tuple[list[float], list[int | None], list[float]]:
    torch = runtime["torch"]
    loader = runtime["DataLoader"](
        list(examples), batch_size=batch_size, shuffle=False, collate_fn=collate
    )
    probabilities: list[float] = []
    labels: list[int | None] = []
    margins: list[float] = []
    model.eval()
    if classification_head is not None:
        classification_head.eval()
    with torch.inference_mode():
        for batch in loader:
            class_logits = _class_logits(
                model,
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                label_token_ids,
                torch,
                classification_head,
            )
            positive = torch.softmax(class_logits, dim=-1)[:, 1]
            probabilities.extend(float(value) for value in positive.cpu().tolist())
            margin = class_logits[:, 1] - class_logits[:, 0]
            margins.extend(float(value) for value in margin.float().cpu().tolist())
            labels.extend(
                None if value < 0 else int(value)
                for value in batch["labels"].tolist()
            )
    return probabilities, labels, margins


def _calibrate_binary_head_prior(
    model: Any,
    classification_head: Any,
    examples: Sequence[PromptExample],
    *,
    class_weighting: bool,
    batch_size: int,
    collate: Any,
    label_token_ids: tuple[int, int],
    device: Any,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Calibrate only the binary-head intercept from fit examples.

    Token-row initialization has a large model-specific offset toward answer ``0``.
    Removing that offset prevents LoRA from having to rewrite the representation just
    to learn an intercept.  No dev/holdout example is read here.
    """

    probabilities, raw_labels, margins = _evaluate(
        model,
        examples,
        batch_size=batch_size,
        collate=collate,
        label_token_ids=label_token_ids,
        device=device,
        runtime=runtime,
        classification_head=classification_head,
    )
    if any(label is None for label in raw_labels):
        raise ValueError("fit-only prior calibration requires labels for every example")
    labels = [int(label) for label in raw_labels]
    empirical_positive_rate = float(np.mean(labels))
    target_positive_rate = 0.5 if class_weighting else empirical_positive_rate
    delta = _solve_prior_bias_delta(margins, target_positive_rate)
    torch = runtime["torch"]
    with torch.no_grad():
        classification_head.bias[0].sub_(delta / 2.0)
        classification_head.bias[1].add_(delta / 2.0)
    shifted_probabilities = 1.0 / (
        1.0
        + np.exp(
            -np.clip(np.asarray(margins, dtype=np.float64) + delta, -60.0, 60.0)
        )
    )
    return {
        "enabled": True,
        "source": "fit_examples_only",
        "fit_count": len(examples),
        "empirical_positive_rate": empirical_positive_rate,
        "target_positive_rate": target_positive_rate,
        "margin_shift": delta,
        "mean_probability_before": float(np.mean(probabilities)),
        "mean_probability_after": float(np.mean(shifted_probabilities)),
    }


def _prediction_rows(
    examples: Sequence[PromptExample],
    probabilities: Sequence[float],
    margins: Sequence[float],
    *,
    threshold: float,
    origin: str,
) -> list[dict[str, Any]]:
    return [
        {
            "session_id": example.session_id,
            "split": example.split,
            "view_name": example.view_name,
            "label": example.label,
            "probability": float(probability),
            "logit_margin_1_minus_0": float(margin),
            "predicted_label": int(float(probability) >= threshold),
            "threshold": float(threshold),
            "prediction_origin": origin,
        }
        for example, probability, margin in zip(examples, probabilities, margins)
    ]


def _score_diagnostics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    margins: Sequence[float],
) -> dict[str, Any]:
    if not (len(labels) == len(probabilities) == len(margins)):
        raise ValueError("score diagnostics require aligned labels, probabilities, margins")
    if not labels:
        return {"count": 0, "unique_probability_count": 0}
    output: dict[str, Any] = {
        "count": len(labels),
        "unique_probability_count": len(set(float(value) for value in probabilities)),
        "probability_min": min(float(value) for value in probabilities),
        "probability_max": max(float(value) for value in probabilities),
        "margin_min": min(float(value) for value in margins),
        "margin_max": max(float(value) for value in margins),
    }
    for label in (0, 1):
        selected_probabilities = [
            float(probability)
            for actual, probability in zip(labels, probabilities)
            if actual == label
        ]
        selected_margins = [
            float(margin)
            for actual, margin in zip(labels, margins)
            if actual == label
        ]
        if selected_probabilities:
            output[f"label_{label}_probability_mean"] = float(
                np.mean(selected_probabilities)
            )
            output[f"label_{label}_margin_mean"] = float(np.mean(selected_margins))
    return output


def _memorization_gate(
    metrics: Mapping[str, Any],
    labels: Sequence[int],
    margins: Sequence[float],
) -> dict[str, Any]:
    signed_margins = [
        float(margin) if label == 1 else -float(margin)
        for label, margin in zip(labels, margins)
    ]
    count = len(signed_margins)
    checks = {
        "roc_auc_is_one": float(metrics.get("roc_auc", 0.0)) >= 1.0 - 1e-12,
        "average_precision_is_one": float(
            metrics.get("average_precision", 0.0)
        )
        >= 1.0 - 1e-12,
        "balanced_accuracy_at_0_5_at_least_0_95": float(
            metrics.get("balanced_accuracy", 0.0)
        )
        >= 0.95,
        "log_loss_at_most_0_10": float(metrics.get("log_loss", math.inf)) <= 0.10,
        "all_signed_margins_positive": bool(signed_margins)
        and all(value > 0.0 for value in signed_margins),
        "all_but_one_signed_margin_above_2": count > 0
        and sum(value > 2.0 for value in signed_margins) >= count - 1,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "signed_margin_min": min(signed_margins, default=None),
        "signed_margin_above_2_count": sum(
            value > 2.0 for value in signed_margins
        ),
        "required_count": count,
    }


def _maximum_consecutive_gate_passes(
    history: Sequence[Mapping[str, Any]],
) -> int:
    maximum = 0
    current = 0
    for record in history:
        if record.get("memorization_gate", {}).get("passed"):
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum


def _is_better_checkpoint(
    candidate: Mapping[str, Any],
    incumbent: Mapping[str, Any] | None,
    metric: str,
) -> bool:
    if metric not in {"roc_auc", "average_precision", "log_loss"}:
        raise ValueError(f"Unsupported checkpoint metric {metric!r}")
    if incumbent is None:
        return True
    candidate_value = float(candidate[metric])
    incumbent_value = float(incumbent[metric])
    if metric == "log_loss":
        return candidate_value < incumbent_value - 1e-12
    if candidate_value > incumbent_value + 1e-12:
        return True
    if abs(candidate_value - incumbent_value) <= 1e-12:
        return float(candidate["log_loss"]) < float(incumbent["log_loss"])
    return False


def _named_trainable_parameters(
    model: Any, classification_head: Any | None
) -> list[tuple[str, Any]]:
    output = [
        (f"model::{name}", parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if classification_head is not None:
        output.extend(
            (f"classification_head::{name}", parameter)
            for name, parameter in classification_head.named_parameters()
            if parameter.requires_grad
        )
    return output


def _capture_trainable_state(
    named_parameters: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in named_parameters
    }


def _restore_trainable_state(
    named_parameters: Sequence[tuple[str, Any]], state: Mapping[str, Any]
) -> None:
    expected = {name for name, _ in named_parameters}
    if set(state) != expected:
        raise ValueError("checkpoint trainable-parameter set does not match the model")
    for name, parameter in named_parameters:
        parameter.data.copy_(
            state[name].to(device=parameter.device, dtype=parameter.dtype)
        )


def _token_budget_summary(
    examples: Sequence[PromptExample],
) -> dict[str, Any]:
    evidence_tiers: dict[str, int] = {}
    transcript_tiers: dict[str, int] = {}
    for example in examples:
        evidence_tiers[example.evidence_density_tier] = (
            evidence_tiers.get(example.evidence_density_tier, 0) + 1
        )
        transcript_tiers[example.transcript_density_tier] = (
            transcript_tiers.get(example.transcript_density_tier, 0) + 1
        )
    return {
        "count": len(examples),
        "prompt_min": min((len(example.input_ids) for example in examples), default=0),
        "prompt_max": max((len(example.input_ids) for example in examples), default=0),
        "evidence_min": min(
            (example.evidence_tokens for example in examples), default=0
        ),
        "evidence_max": max(
            (example.evidence_tokens for example in examples), default=0
        ),
        "transcript_min": min(
            (example.transcript_tokens for example in examples), default=0
        ),
        "transcript_max": max(
            (example.transcript_tokens for example in examples), default=0
        ),
        "evidence_density_tiers": dict(sorted(evidence_tiers.items())),
        "density_tiers": dict(sorted(evidence_tiers.items())),
        "transcript_density_tiers": dict(sorted(transcript_tiers.items())),
    }


def _transcript_run_metadata(
    records: Sequence[SessionRecord],
    transcript_root: Path | None,
    *,
    dataset: str | None = None,
) -> dict[str, Any]:
    if dataset == "d_vlog":
        expected_tiers = {
            tier: D_VLOG_MISSING_TRANSCRIPT_TEXT for tier in _TRANSCRIPT_TIERS
        }
        if any(dict(record.transcript_tiers) != expected_tiers for record in records):
            raise ValueError("D-Vlog records do not share the uniform missing marker")
        stored_marker = D_VLOG_MISSING_TRANSCRIPT_TEXT.rstrip("\n")
        marker_sha256 = hashlib.sha256(stored_marker.encode("utf-8")).hexdigest()
        return {
            "enabled": False,
            "content_exposed_to_model": False,
            "protocol_version": None,
            "availability": "uniformly_unavailable",
            "missing_marker": stored_marker,
            "marker_sha256": marker_sha256,
            "missing_marker_sha256": marker_sha256,
            "uniform_across_splits": True,
        }
    if transcript_root is None:
        return {
            "enabled": False,
            "content_exposed_to_model": False,
            "protocol_version": None,
        }
    manifest_rows = []
    for record in sorted(records, key=lambda item: item.session_id):
        if record.transcript_manifest_sha256 is None:
            raise ValueError(
                f"Session {record.session_id} is missing transcript provenance"
            )
        manifest_rows.append(
            f"{record.session_id}:{record.transcript_manifest_sha256}"
        )
    aggregate_hash = hashlib.sha256(
        "\n".join(manifest_rows).encode("utf-8")
    ).hexdigest()
    return {
        "enabled": True,
        "content_exposed_to_model": True,
        "protocol_version": "compact",
        "root": str(transcript_root.resolve()),
        "manifest_set_sha256": aggregate_hash,
        "speaker_policies": sorted(
            {
                str(record.transcript_speaker_policy)
                for record in records
                if record.transcript_speaker_policy is not None
            }
        ),
        "source_kinds": sorted(
            {
                str(record.transcript_source_kind)
                for record in records
                if record.transcript_source_kind is not None
            }
        ),
        "semantic_truncation_only": True,
        "token_level_text_truncation": False,
    }


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset {args.dataset!r}")
    if args.evidence_protocol not in SUPPORTED_EVIDENCE_PROTOCOLS:
        raise ValueError(
            f"Unsupported Evidence protocol {args.evidence_protocol!r}"
        )
    task = load_task(args.task)
    is_d_vlog = args.dataset == "d_vlog"
    if is_d_vlog:
        validate_d_vlog_supervised_contract(
            args.label_column, args.label_threshold, task
        )
        if args.transcript_root is not None:
            raise ValueError("D-Vlog has no transcript_root; use the uniform marker")
        if args.evaluate_labeled_test:
            raise ValueError("D-Vlog supervised training cannot read test truth")
        if args.memorization_per_class is not None and (
            args.dev_labels is not None or args.test_labels is not None
        ):
            raise ValueError(
                "D-Vlog train-only memorization gate must not receive dev/test paths"
            )
    else:
        validate_phq8_score_policy(
            args.dataset, args.label_column, args.label_threshold
        )
    if args.epochs <= 0 or args.batch_size <= 0 or args.gradient_accumulation <= 0:
        raise ValueError("epochs, batch_size, and gradient_accumulation must be positive")
    if args.learning_rate <= 0 or args.head_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    if args.fold_index is not None and args.test_labels is not None:
        raise ValueError("OOF fold runs cannot access a test split")
    if args.fold_index is not None and args.memorization_per_class is not None:
        raise ValueError("memorization gate and OOF mode are mutually exclusive")
    if args.fold_index is not None and args.oof_reference is None:
        raise ValueError("strict OOF fold runs require --oof-reference")
    if args.fold_index is None and args.oof_reference is not None:
        raise ValueError("--oof-reference is only valid for OOF fold runs")
    if args.fold_index is None and args.split_seed is not None:
        raise ValueError("--split-seed is only valid for OOF fold runs")

    split_seed = args.split_seed if args.split_seed is not None else args.seed

    test_ids_metadata = (
        audit_d_vlog_test_ids(args.test_labels)
        if is_d_vlog and args.test_labels is not None
        else None
    )

    runtime = _runtime()
    torch = runtime["torch"]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    all_train = _read_split(
        args.train_labels,
        "train",
        args.evidence_root,
        args.label_column,
        args.id_column,
        require_labels=True,
        transcript_root=args.transcript_root,
        label_threshold=args.label_threshold,
        evidence_protocol=args.evidence_protocol,
    )
    if args.memorization_per_class is not None:
        dev = []
        test = []
    elif args.fold_index is None:
        if args.dev_labels is None:
            raise ValueError("a full train/dev fit requires --dev-labels")
        dev = _read_split(
            args.dev_labels,
            "valid" if is_d_vlog else "dev",
            args.evidence_root,
            args.label_column,
            args.id_column,
            require_labels=True,
            transcript_root=args.transcript_root,
            label_threshold=args.label_threshold,
            evidence_protocol=args.evidence_protocol,
        )
        test = (
            _read_split(
                args.test_labels,
                "test",
                args.evidence_root,
                args.label_column,
                args.id_column,
                require_labels=False,
                load_optional_labels=args.evaluate_labeled_test,
                transcript_root=args.transcript_root,
                label_threshold=args.label_threshold,
                evidence_protocol=args.evidence_protocol,
            )
            if args.test_labels is not None
            else []
        )
    else:
        # OOF generators may not even read dev/test label files. Their only inputs are
        # the training participants outside the held-out fold.
        dev = []
        test = []
    if is_d_vlog:
        all_train = with_d_vlog_missing_transcript(all_train)
        dev = with_d_vlog_missing_transcript(dev)
        test = with_d_vlog_missing_transcript(test)
    split_ids = [set(record.session_id for record in split) for split in (all_train, dev, test)]
    if split_ids[0] & split_ids[1] or split_ids[0] & split_ids[2] or split_ids[1] & split_ids[2]:
        raise ValueError("train/dev/test session IDs must be disjoint")
    all_records = [*all_train, *dev, *test]
    accessed_label_paths = [args.train_labels]
    if args.memorization_per_class is None and args.fold_index is None:
        accessed_label_paths.append(args.dev_labels)
        if args.test_labels is not None and args.evaluate_labeled_test:
            accessed_label_paths.append(args.test_labels)
    accessed_paths = [path for path in accessed_label_paths if path is not None]
    label_policy = (
        build_d_vlog_label_policy(accessed_paths, id_column=args.id_column)
        if is_d_vlog
        else build_phq8_label_audit(
            accessed_paths,
            dataset=args.dataset,
            id_column=args.id_column,
        )
    )
    validate_dataset_identity(
        all_records,
        args.evidence_root,
        args.dataset,
        args.transcript_root,
        args.evidence_protocol,
    )
    transcript_metadata = _transcript_run_metadata(
        all_records, args.transcript_root, dataset=args.dataset
    )
    evidence_metadata = (
        build_d_vlog_evidence_metadata(args.evidence_root) if is_d_vlog else None
    )
    experiment_name = (
        "qwen_thinker_d_vlog_supervised"
        if is_d_vlog
        else (
            "qwen_thinker_av_transcript_label_training_native"
            if args.transcript_root is not None
            else "qwen_thinker_text_label_training_native"
        )
    )

    if args.memorization_per_class is not None:
        train_records = balanced_memorization_subset(
            all_train,
            per_class=args.memorization_per_class,
            seed=args.subset_seed,
        )
        validation_records = train_records
        run_kind = "balanced_train_memorization_gate"
    elif args.fold_index is None:
        train_records = all_train
        validation_records = dev
        run_kind = "independent_train_dev_fit"
    else:
        train_records, validation_records = participant_stratified_fold(
            all_train,
            folds=args.folds,
            fold_index=args.fold_index,
            seed=split_seed,
        )
        run_kind = "participant_oof_fold"
    oof_reference_metadata = (
        validate_oof_reference_provenance(
            train_records,
            validation_records,
            evidence_root=args.evidence_root,
            reference_path=args.oof_reference,
            dataset=args.dataset,
            transcript_root=args.transcript_root,
            evidence_protocol=args.evidence_protocol,
        )
        if args.fold_index is not None and args.oof_reference is not None
        else None
    )

    tokenizer = runtime["AutoTokenizer"].from_pretrained(
        str(args.model_path), trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    label_token_ids = resolve_label_token_ids(tokenizer)

    train_examples = tokenize_records(
        train_records,
        tokenizer,
        task,
        view=args.view,
        max_input_tokens=args.max_input_tokens,
        evidence_density=args.evidence_density,
        max_evidence_tokens=args.max_evidence_tokens,
        max_transcript_tokens=args.max_transcript_tokens,
        evidence_protocol=args.evidence_protocol,
    )
    validation_examples = tokenize_records(
        validation_records,
        tokenizer,
        task,
        view=args.view,
        max_input_tokens=args.max_input_tokens,
        evidence_density=args.evidence_density,
        max_evidence_tokens=args.max_evidence_tokens,
        max_transcript_tokens=args.max_transcript_tokens,
        evidence_protocol=args.evidence_protocol,
    )
    test_examples = tokenize_records(
        test,
        tokenizer,
        task,
        view=args.view,
        max_input_tokens=args.max_input_tokens,
        evidence_density=args.evidence_density,
        max_evidence_tokens=args.max_evidence_tokens,
        max_transcript_tokens=args.max_transcript_tokens,
        evidence_protocol=args.evidence_protocol,
    )
    if is_d_vlog and any(example.label is not None for example in test_examples):
        raise ValueError("D-Vlog test predictions must remain label-free")
    class_counts = np.bincount(
        np.asarray([int(example.label) for example in train_examples]), minlength=2
    )
    if np.any(class_counts == 0):
        raise ValueError("training data must contain both classes")
    collate = _collator(torch, int(tokenizer.pad_token_id))

    if args.validate_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "experiment": experiment_name,
            "validation_only": True,
            "run_kind": run_kind,
            "dataset": args.dataset,
            "label_policy": label_policy,
            "task": task.to_dict(),
            "view": args.view,
            "evidence_protocol": args.evidence_protocol,
            "evidence_density": args.evidence_density,
            "transcript": transcript_metadata,
            "oof_reference": oof_reference_metadata,
            **(
                {
                    "test_ids_sha256": (
                        test_ids_metadata["ids_sha256"]
                        if test_ids_metadata is not None
                        else None
                    ),
                    "test_ids_file_sha256": (
                        test_ids_metadata["file_sha256"]
                        if test_ids_metadata is not None
                        else None
                    ),
                    "evidence": evidence_metadata,
                }
                if is_d_vlog
                else {}
            ),
            "seed": args.seed,
            "split_seed": split_seed if args.fold_index is not None else None,
            "folds": args.folds if args.fold_index is not None else None,
            "fold_index": args.fold_index,
            "session_counts": {
                "all_train": len(all_train),
                "fit": len(train_records),
                "validation": len(validation_records),
                "test": len(test),
            },
            "fit_ids_sha256": _sha256_identifiers(train_records),
            "validation_ids_sha256": _sha256_identifiers(validation_records),
            "label_token_ids": {"0": label_token_ids[0], "1": label_token_ids[1]},
            "token_lengths": {
                "fit_min": min(len(example.input_ids) for example in train_examples),
                "fit_max": max(len(example.input_ids) for example in train_examples),
                "validation_min": min(
                    len(example.input_ids) for example in validation_examples
                ),
                "validation_max": max(
                    len(example.input_ids) for example in validation_examples
                ),
                "test_min": min(
                    (len(example.input_ids) for example in test_examples), default=0
                ),
                "test_max": max(
                    (len(example.input_ids) for example in test_examples), default=0
                ),
                "fixed_limit": args.max_input_tokens,
                "evidence_fixed_limit": args.max_evidence_tokens,
                "transcript_fixed_limit": args.max_transcript_tokens,
                "by_split": {
                    "fit": _token_budget_summary(train_examples),
                    "validation": _token_budget_summary(validation_examples),
                    "test": _token_budget_summary(test_examples),
                },
                "silent_truncation": False,
            },
            "fit_boundaries": {
                "evidence_compiler_label_access": False,
                "dataset_training": "independent_single_dataset",
                "cross_dataset_merging": False,
                "oof_reads_dev_or_test": False,
                "test_labels_accessed": args.evaluate_labeled_test,
                "validation_is_fit_memorization_gate": (
                    args.memorization_per_class is not None
                ),
                "transcript_compiler_label_access": False,
                "oof_holdout_in_reference_fit": False,
            },
        }
        _write_json(args.output_dir / "validation_summary.json", summary)
        return summary

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if not hasattr(torch, args.dtype):
        raise ValueError(f"Unsupported torch dtype {args.dtype!r}")
    torch_dtype = getattr(torch, args.dtype)
    base_model = load_pretrained_model_on_device(
        runtime["QwenModel"],
        args.model_path,
        device=str(device),
        torch_dtype=torch_dtype,
        attn_implementation=args.attention_implementation,
        low_cpu_mem_usage=True,
    )
    base_model.config.use_cache = False
    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        base_model.enable_input_require_grads()
    lora_config = runtime["LoraConfig"](
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TEXT_LORA_TARGETS,
    )
    model = runtime["get_peft_model"](base_model, lora_config)
    model.train()

    model_trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not model_trainable:
        raise RuntimeError("LoRA configuration produced no trainable parameters")
    classification_head = (
        _build_classification_head(model, label_token_ids, torch)
        if args.head_type == "binary"
        else None
    )
    head_prior_calibration: dict[str, Any] = {
        "enabled": False,
        "reason": (
            "disabled_by_argument"
            if classification_head is not None
            else "not_applicable_to_verbalizer_head"
        ),
    }
    if classification_head is not None and args.head_prior_calibration:
        head_prior_calibration = _calibrate_binary_head_prior(
            model,
            classification_head,
            train_examples,
            class_weighting=args.class_weighting,
            batch_size=args.eval_batch_size,
            collate=collate,
            label_token_ids=label_token_ids,
            device=device,
            runtime=runtime,
        )
        model.train()
    if classification_head is not None:
        classification_head.train()
    head_trainable = (
        list(classification_head.parameters())
        if classification_head is not None
        else []
    )
    trainable = [*model_trainable, *head_trainable]
    parameter_groups: list[dict[str, Any]] = [
        {"params": model_trainable, "lr": args.learning_rate}
    ]
    if head_trainable:
        parameter_groups.append(
            {"params": head_trainable, "lr": args.head_learning_rate}
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(
        math.ceil(len(train_examples) / args.batch_size) / args.gradient_accumulation
    )
    total_updates = updates_per_epoch * args.epochs
    warmup_updates = int(round(total_updates * args.warmup_ratio))
    scheduler = runtime["get_scheduler"](
        args.scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_updates,
        num_training_steps=total_updates,
    )
    if args.class_weighting:
        class_weights = torch.tensor(
            [len(train_examples) / (2.0 * count) for count in class_counts],
            dtype=torch.float32,
            device=device,
        )
    else:
        class_weights = None
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = runtime["DataLoader"](
        train_examples,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
    )

    if args.checkpoint_metric == "auto":
        checkpoint_metric: str | None = (
            "log_loss"
            if args.memorization_per_class is not None
            else ("roc_auc" if args.fold_index is None else None)
        )
    elif args.checkpoint_metric == "none":
        checkpoint_metric = None
    else:
        checkpoint_metric = args.checkpoint_metric
    if args.fold_index is not None and checkpoint_metric is not None:
        raise ValueError(
            "OOF holdout cannot select a checkpoint; use --checkpoint-metric none"
        )
    named_trainable = _named_trainable_parameters(model, classification_head)
    best_state: dict[str, Any] | None = None
    best_metrics: dict[str, Any] | None = None
    best_epoch: int | None = None

    history: list[dict[str, Any]] = []
    memorization_pass_streak = 0
    early_stop_epoch: int | None = None
    optimizer.zero_grad(set_to_none=True)
    started = time.time()

    def evaluate_epoch(epoch: int, train_loss: float | None) -> dict[str, Any]:
        probability, raw_labels, margins = _evaluate(
            model,
            validation_examples,
            batch_size=args.eval_batch_size,
            collate=collate,
            label_token_ids=label_token_ids,
            device=device,
            runtime=runtime,
            classification_head=classification_head,
        )
        parsed_labels = [int(value) for value in raw_labels if value is not None]
        epoch_metrics = _metrics(parsed_labels, probability, threshold=0.5)
        record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_at_0_5": epoch_metrics,
            "validation_score_diagnostics": _score_diagnostics(
                parsed_labels,
                probability,
                margins,
            ),
            "elapsed_seconds": time.time() - started,
        }
        if args.memorization_per_class is not None:
            record["memorization_gate"] = _memorization_gate(
                epoch_metrics,
                parsed_labels,
                margins,
            )
        return record

    if args.fold_index is None:
        initial_record = evaluate_epoch(0, None)
        history.append(initial_record)
        print(json.dumps(initial_record, ensure_ascii=False, sort_keys=True), flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        if classification_head is not None:
            classification_head.train()
        running_loss = 0.0
        example_count = 0
        accumulated_examples = 0
        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch_dtype,
                enabled=device.type == "cuda" and torch_dtype in {torch.float16, torch.bfloat16},
            ):
                class_logits = _class_logits(
                    model,
                    input_ids,
                    attention_mask,
                    label_token_ids,
                    torch,
                    classification_head,
                )
                loss = _weighted_cross_entropy(
                    runtime["functional"],
                    class_logits,
                    labels,
                    class_weights,
                )
            (loss * len(labels)).backward()
            running_loss += float(loss.detach().cpu()) * len(labels)
            example_count += len(labels)
            accumulated_examples += len(labels)
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                for parameter in trainable:
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated_examples)
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated_examples = 0

        if args.fold_index is None:
            epoch_record = evaluate_epoch(
                epoch, running_loss / example_count
            )
            candidate_metrics = epoch_record["validation_at_0_5"]
            if checkpoint_metric is not None and _is_better_checkpoint(
                candidate_metrics,
                best_metrics,
                checkpoint_metric,
            ):
                best_metrics = dict(candidate_metrics)
                best_state = _capture_trainable_state(named_trainable)
                best_epoch = epoch
        else:
            epoch_record = {
                "epoch": epoch,
                "train_loss": running_loss / example_count,
                "holdout_evaluated": False,
                "elapsed_seconds": time.time() - started,
            }
        history.append(epoch_record)
        print(json.dumps(history[-1], ensure_ascii=False, sort_keys=True), flush=True)
        if args.memorization_per_class is not None:
            if epoch_record["memorization_gate"]["passed"]:
                memorization_pass_streak += 1
            else:
                memorization_pass_streak = 0
            if memorization_pass_streak >= 3:
                early_stop_epoch = epoch
                break

    if best_state is not None:
        _restore_trainable_state(named_trainable, best_state)
    else:
        best_epoch = args.epochs

    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    if classification_head is not None:
        runtime["save_safetensors"](
            {
                "weight": classification_head.weight.detach().float().cpu().contiguous(),
                "bias": classification_head.bias.detach().float().cpu().contiguous(),
            },
            adapter_dir / "classification_head.safetensors",
        )
        _write_json(
            adapter_dir / "classification_head.json",
            {
                "head_type": "binary",
                "classes": [0, 1],
                "initialization": "qwen_label_token_rows_0_and_1_plus_fit_prior_bias",
                "fit_prior_calibration": head_prior_calibration,
                "dtype": "float32",
                "selected_epoch": best_epoch,
            },
        )

    validation_probability, validation_labels, validation_margins = _evaluate(
        model,
        validation_examples,
        batch_size=args.eval_batch_size,
        collate=collate,
        label_token_ids=label_token_ids,
        device=device,
        runtime=runtime,
        classification_head=classification_head,
    )
    parsed_validation_labels = [int(value) for value in validation_labels if value is not None]
    if args.memorization_per_class is not None:
        threshold = 0.5
        validation_origin = "fit_memorization_fixed_threshold"
    elif args.fold_index is None:
        threshold = select_balanced_accuracy_threshold(
            parsed_validation_labels, validation_probability
        )
        validation_origin = "train_fitted_dev"
    else:
        threshold = 0.5
        validation_origin = "out_of_fold_fixed_threshold"
    predictions = _prediction_rows(
        validation_examples,
        validation_probability,
        validation_margins,
        threshold=threshold,
        origin=validation_origin,
    )
    final_metrics: dict[str, Any] = {
        "validation": _metrics(
            parsed_validation_labels, validation_probability, threshold
        ),
        "validation_score_diagnostics": _score_diagnostics(
            parsed_validation_labels,
            validation_probability,
            validation_margins,
        ),
    }
    memorization_gate: dict[str, Any] | None = None
    if args.memorization_per_class is not None:
        memorization_gate = _memorization_gate(
            _metrics(
                parsed_validation_labels,
                validation_probability,
                threshold=0.5,
            ),
            parsed_validation_labels,
            validation_margins,
        )
        consecutive_pass_epochs = _maximum_consecutive_gate_passes(history)
        memorization_gate["consecutive_pass_epochs"] = consecutive_pass_epochs
        memorization_gate["required_consecutive_pass_epochs"] = 3
        memorization_gate["passed"] = bool(
            memorization_gate["passed"] and consecutive_pass_epochs >= 3
        )

    if test_examples:
        test_probability, test_values, test_margins = _evaluate(
            model,
            test_examples,
            batch_size=args.eval_batch_size,
            collate=collate,
            label_token_ids=label_token_ids,
            device=device,
            runtime=runtime,
            classification_head=classification_head,
        )
        predictions.extend(
            _prediction_rows(
                test_examples,
                test_probability,
                test_margins,
                threshold=threshold,
                origin="train_fitted_dev_threshold",
            )
        )
        labeled_pairs = [
            (int(label), probability)
            for label, probability in zip(test_values, test_probability)
            if label is not None
        ]
        if labeled_pairs:
            labels, probabilities = zip(*labeled_pairs)
            final_metrics["test"] = _metrics(labels, probabilities, threshold)

    _write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    summary = {
        "experiment": experiment_name,
        "run_kind": run_kind,
        "dataset": args.dataset,
        "label_policy": label_policy,
        "task": task.to_dict(),
        "view": args.view,
        "evidence_protocol": args.evidence_protocol,
        "evidence_density": args.evidence_density,
        "transcript": transcript_metadata,
        "oof_reference": oof_reference_metadata,
        **(
            {
                "test_ids_sha256": (
                    test_ids_metadata["ids_sha256"]
                    if test_ids_metadata is not None
                    else None
                ),
                "test_ids_file_sha256": (
                    test_ids_metadata["file_sha256"]
                    if test_ids_metadata is not None
                    else None
                ),
                "evidence": evidence_metadata,
            }
            if is_d_vlog
            else {}
        ),
        "seed": args.seed,
        "split_seed": split_seed if args.fold_index is not None else None,
        "folds": args.folds if args.fold_index is not None else None,
        "fold_index": args.fold_index,
        "session_counts": {
            "all_train": len(all_train),
            "fit": len(train_records),
            "validation": len(validation_records),
            "test": len(test),
        },
        "fit_ids_sha256": _sha256_identifiers(train_records),
        "validation_ids_sha256": _sha256_identifiers(validation_records),
        "label_token_ids": {"0": label_token_ids[0], "1": label_token_ids[1]},
        "token_lengths": {
            "fit_max": max(len(example.input_ids) for example in train_examples),
            "validation_max": max(len(example.input_ids) for example in validation_examples),
            "test_max": max((len(example.input_ids) for example in test_examples), default=0),
            "fixed_limit": args.max_input_tokens,
            "evidence_fixed_limit": args.max_evidence_tokens,
            "transcript_fixed_limit": args.max_transcript_tokens,
            "by_split": {
                "fit": _token_budget_summary(train_examples),
                "validation": _token_budget_summary(validation_examples),
                "test": _token_budget_summary(test_examples),
            },
            "silent_truncation": False,
        },
        "training": {
            "model_path": str(args.model_path.resolve()),
            "epochs": len(history) - (1 if args.fold_index is None else 0),
            "maximum_epochs": args.epochs,
            "early_stopping": {
                "enabled": args.memorization_per_class is not None,
                "reason": "three_consecutive_memorization_gate_passes",
                "stopped_epoch": early_stop_epoch,
            },
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "head_type": args.head_type,
            "head_learning_rate": (
                args.head_learning_rate if classification_head is not None else None
            ),
            "head_prior_calibration": head_prior_calibration,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "scheduler_type": args.scheduler_type,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_targets": TEXT_LORA_TARGETS,
            "class_weighting": args.class_weighting,
            "gradient_checkpointing": args.gradient_checkpointing,
            "dtype": args.dtype,
            "attention_implementation": args.attention_implementation,
            "fixed_epoch_protocol": args.fold_index is not None,
            "checkpoint_selection": {
                "metric": checkpoint_metric,
                "selected_epoch": best_epoch,
                "selected_metrics_at_0_5": best_metrics,
                "epoch_zero_evaluated": args.fold_index is None,
                "epoch_zero_eligible_for_selection": False,
                "oof_holdout_evaluated_each_epoch": False,
            },
        },
        "fit_boundaries": {
            "evidence_compiler_label_access": False,
            "transcript_compiler_label_access": False,
            "dataset_training": "independent_single_dataset",
            "cross_dataset_merging": False,
            "dev_gradient_updates": False,
            "test_gradient_updates": False,
            "test_labels_accessed": args.evaluate_labeled_test,
            "oof_holdout_gradient_updates": False,
            "oof_holdout_threshold_selection": False,
            "oof_holdout_in_reference_fit": False,
            "validation_is_fit_memorization_gate": (
                args.memorization_per_class is not None
            ),
        },
        "history": history,
        "metrics": final_metrics,
        "memorization_gate": memorization_gate,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one dataset-specific Qwen Thinker text label baseline; DAIC-WOZ "
            "and E-DAIC runs must be launched separately."
        )
    )
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument(
        "--transcript-root",
        type=Path,
        help=(
            "Optional dataset-specific compact transcript directory. When supplied, "
            "transcript text is included as a separately budgeted evidence layer."
        ),
    )
    parser.add_argument("--train-labels", required=True, type=Path)
    parser.add_argument("--dev-labels", type=Path)
    parser.add_argument("--test-labels", type=Path)
    parser.add_argument("--label-column", required=True)
    parser.add_argument(
        "--label-threshold",
        required=True,
        type=float,
        help="Inclusive score threshold; PHQ-8 experiments require exactly 10.",
    )
    parser.add_argument("--id-column")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--view", choices=SUPPORTED_VIEWS, default="full")
    parser.add_argument(
        "--evidence-protocol",
        choices=SUPPORTED_EVIDENCE_PROTOCOLS,
        default="native",
        help=(
            "Compiled Evidence protocol expected in every session manifest."
        ),
    )
    parser.add_argument(
        "--evidence-density", choices=("full",), default="full"
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-index", type=int)
    parser.add_argument(
        "--oof-reference",
        type=Path,
        help=(
            "Fold-local ReferenceSet JSON. Required in OOF mode so the held-out "
            "participant set can be proven absent from reference fitting."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-seed",
        type=int,
        help=(
            "OOF partition seed. Defaults to --seed for backward compatibility; "
            "set it explicitly to keep folds fixed across training seeds."
        ),
    )
    parser.add_argument("--subset-seed", type=int, default=17)
    parser.add_argument(
        "--memorization-per-class",
        type=int,
        help=(
            "Run a train-only balanced memorization gate with this many examples "
            "per class; dev/test files are not read."
        ),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--head-learning-rate", type=float, default=5e-4)
    parser.add_argument(
        "--head-type", choices=("verbalizer", "binary"), default="verbalizer"
    )
    parser.add_argument(
        "--no-head-prior-calibration",
        dest="head_prior_calibration",
        action="store_false",
        help=(
            "Disable fit-only binary-head intercept calibration. The calibration "
            "never reads dev, OOF holdout, or test examples."
        ),
    )
    parser.add_argument(
        "--checkpoint-metric",
        choices=("auto", "none", "roc_auc", "average_precision", "log_loss"),
        default="auto",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument(
        "--scheduler-type",
        choices=("linear", "constant_with_warmup"),
        default="linear",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-input-tokens", type=int, default=7_500)
    parser.add_argument("--max-evidence-tokens", type=int, default=2_000)
    parser.add_argument("--max-transcript-tokens", type=int, default=3_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument(
        "--evaluate-labeled-test",
        action="store_true",
        help="Explicitly unlock labeled test metrics after the experiment is frozen.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate splits, exact tokenizer inputs, and budgets without loading Qwen.",
    )
    parser.add_argument(
        "--no-class-weighting", dest="class_weighting", action="store_false"
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
    )
    parser.set_defaults(
        class_weighting=True,
        gradient_checkpointing=True,
        head_prior_calibration=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_training(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    gate = summary.get("memorization_gate")
    if isinstance(gate, Mapping) and not gate.get("passed", False):
        return 6
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
