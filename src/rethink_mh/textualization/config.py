"""Dataset-specific textualization configuration defaults and loading."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


# NOTE: optional protocol sections (protocol_native) stay out of this default
# table; their defaults live in protocols/native.py.
DEFAULT_CONFIGS: dict[str, dict[str, Any]] = {
    "daic_woz": {
        "dataset_key": "daic_woz",
        "dataset_name": "DAIC-WOZ",
        "adapter": "daic_woz",
        "window_sec": 5.0,
        "stride_sec": 2.5,
        "minimum_tail_sec": 2.5,
        "visual_confidence_threshold": 0.80,
        "audio_sample_rate_hz": 100.0,
        "participant_only_audio": True,
        "transcript": {
            "source_kind": "manual_transcript",
            "model_speaker_policy": "participant_only",
            "speaker_role_reliability": "explicit_release_annotation",
        },
        "reference_reservoir_per_feature": 50_000,
        "max_rendered_observations_per_unit": 8,
        "evidence_protocol_default": "native",
        "known_intervals": {
            "373": [{"start": 352.0, "end": 420.0, "reason": "technical_interruption"}],
            "444": [{"start": 286.0, "end": 387.0, "reason": "external_interruption"}],
        },
    },
    "e_daic": {
        "dataset_key": "e_daic",
        "dataset_name": "E-DAIC",
        "adapter": "e_daic",
        "window_sec": 4.0,
        "stride_sec": 1.0,
        "minimum_tail_sec": 2.0,
        "visual_confidence_threshold": 0.80,
        "transcript_aligned_audio": True,
        "transcript": {
            "source_kind": "automatic_speech_recognition",
            "model_speaker_policy": "all_speakers_unassigned",
            "speaker_role_reliability": "unavailable_mixed_speaker_audio",
        },
        "reference_reservoir_per_feature": 50_000,
        "max_rendered_observations_per_unit": 8,
        "evidence_protocol_default": "native",
    },
    "d_vlog": {
        "dataset_key": "d_vlog",
        "dataset_name": "D-Vlog",
        "adapter": "d_vlog",
        "window_sec": 30.0,
        "stride_sec": 15.0,
        "minimum_tail_sec": 10.0,
        "sampling_rate_hz": 1.0,
        "reference_reservoir_per_feature": 50_000,
        "max_rendered_observations_per_unit": 8,
        "evidence_protocol_default": "native",
    },
}

ALIASES = {
    "daic": "daic_woz",
    "daic-woz": "daic_woz",
    "daic_woz": "daic_woz",
    "edaic": "e_daic",
    "e-daic": "e_daic",
    "e_daic": "e_daic",
    "dvlog": "d_vlog",
    "d-vlog": "d_vlog",
    "d_vlog": "d_vlog",
}


def normalize_dataset_key(value: str) -> str:
    try:
        return ALIASES[value.strip().lower()]
    except KeyError as exc:
        choices = ", ".join(sorted(DEFAULT_CONFIGS))
        raise ValueError(f"Unsupported dataset {value!r}; choose one of: {choices}") from exc


@dataclass(frozen=True)
class TextualizationConfig:
    values: dict[str, Any]
    source_path: str | None = None

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    @property
    def dataset_key(self) -> str:
        return str(self.values["dataset_key"])

    @property
    def dataset_name(self) -> str:
        return str(self.values["dataset_name"])

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self.values)


def load_config(dataset: str, path: str | Path | None = None) -> TextualizationConfig:
    key = normalize_dataset_key(dataset)
    values = deepcopy(DEFAULT_CONFIGS[key])
    source_path = None
    if path is not None:
        source_path = str(Path(path).expanduser().resolve())
        with Path(source_path).open("r", encoding="utf-8") as handle:
            overrides = yaml.safe_load(handle) or {}
        if not isinstance(overrides, dict):
            raise ValueError(f"Config must be a mapping: {source_path}")
        override_key = normalize_dataset_key(str(overrides.get("dataset_key", key)))
        if override_key != key:
            raise ValueError(
                f"Config dataset {override_key!r} does not match requested dataset {key!r}"
            )
        values.update(overrides)
        values["dataset_key"] = key
    _validate_config(values)
    return TextualizationConfig(values=values, source_path=source_path)


def _validate_config(values: dict[str, Any]) -> None:
    for name in ("window_sec", "stride_sec", "minimum_tail_sec"):
        value = float(values[name])
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if float(values["minimum_tail_sec"]) > float(values["window_sec"]):
        raise ValueError("minimum_tail_sec cannot exceed window_sec")
    threshold = float(values.get("visual_confidence_threshold", 0.8))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("visual_confidence_threshold must lie in [0, 1]")
    if int(values.get("reference_reservoir_per_feature", 50_000)) <= 0:
        raise ValueError("reference_reservoir_per_feature must be positive")
    if int(values.get("max_rendered_observations_per_unit", 8)) < 0:
        raise ValueError("max_rendered_observations_per_unit cannot be negative")
    if str(values.get("evidence_protocol_default", "native")) != "native":
        raise ValueError("evidence_protocol_default must be native")
    protocol_native = values.get("protocol_native", {})
    if not isinstance(protocol_native, dict):
        raise ValueError("protocol_native must be a mapping")
    for name in ("segment_sec", "max_segments_per_session"):
        if float(protocol_native.get(name, 1)) <= 0:
            raise ValueError(f"protocol_native.{name} must be positive")
    selected = int(
        protocol_native.get("selected_slots_per_modality", 2)
    )
    if not 1 <= selected <= 5:
        raise ValueError(
            "protocol_native.selected_slots_per_modality must be 1 to 5"
        )
    extensions = int(
        protocol_native.get("atomic_extensions_per_modality", 0)
    )
    if not 0 <= extensions <= 4:
        raise ValueError(
            "protocol_native.atomic_extensions_per_modality must be 0 to 4"
        )
    native_source_profiles = protocol_native.get("source_profiles", {})
    if not isinstance(native_source_profiles, dict):
        raise ValueError(
            "protocol_native.source_profiles must be a mapping"
        )
