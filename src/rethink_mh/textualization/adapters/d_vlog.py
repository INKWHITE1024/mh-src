"""Dataset adapter for D-Vlog acoustic and visual feature arrays."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..metrics import D_VLOG_EGEMAPS_COLUMNS, egemaps_observations, landmark_observations
from ..readers import natural_session_key
from ..schema import RawEvidenceUnit
from ..windowing import expected_count, make_windows
from .base import DatasetAdapter


class DVlogAdapter(DatasetAdapter):
    AUDIO_DEFAULTS = {
        "source_id": "dvlog_egemaps_audio",
        "extractor": "openSMILE eGeMAPS LLD",
        "version": "dataset_release",
        "representation_level": "expert_feature",
    }
    VISUAL_DEFAULTS = {
        "source_id": "dvlog_dlib_landmarks",
        "extractor": "dlib 68-point landmarks",
        "version": "dataset_release",
        "representation_level": "geometric_feature",
    }

    def list_session_ids(self) -> list[str]:
        ids = [
            path.name
            for path in self.root.iterdir()
            if path.is_dir()
            and (path / f"{path.name}_acoustic.npy").exists()
            and (path / f"{path.name}_visual.npy").exists()
        ]
        return sorted(ids, key=natural_session_key)

    def _paths(self, session_id: str) -> tuple[Path, Path]:
        directory = self.root / str(session_id)
        return directory / f"{session_id}_acoustic.npy", directory / f"{session_id}_visual.npy"

    def audit_session(self, session_id: str) -> dict[str, Any]:
        audio_path, visual_path = self._paths(session_id)
        result: dict[str, Any] = {"session_id": str(session_id), "errors": [], "warnings": []}
        for name, path, width in (("audio", audio_path, 25), ("visual", visual_path, 136)):
            if not path.exists():
                result["errors"].append(f"missing_{name}_file")
                continue
            try:
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                result[name] = {"shape": list(array.shape), "dtype": str(array.dtype)}
                if array.ndim != 2 or array.shape[1] != width:
                    result["errors"].append(f"invalid_{name}_shape")
                if not np.isfinite(array).all():
                    result["warnings"].append(f"nonfinite_{name}_values")
                if name == "visual" and np.all(array == 0):
                    result["warnings"].append("all_visual_rows_are_zero")
            except Exception as exc:  # pragma: no cover - exercised by real-data audit
                result["errors"].append(f"cannot_read_{name}:{type(exc).__name__}")
        if "audio" in result and "visual" in result:
            if result["audio"]["shape"][0] != result["visual"]["shape"][0]:
                result["warnings"].append("modality_length_mismatch")
        result["ok"] = not result["errors"]
        return result

    def load_session(self, session_id: str) -> list[RawEvidenceUnit]:
        audio_path, visual_path = self._paths(session_id)
        if not audio_path.exists() or not visual_path.exists():
            raise FileNotFoundError(f"Missing D-Vlog feature pair for session {session_id}")
        audio = np.load(audio_path, mmap_mode="r", allow_pickle=False)
        visual = np.load(visual_path, mmap_mode="r", allow_pickle=False)
        if audio.ndim != 2 or audio.shape[1] != 25:
            raise ValueError(f"Expected D-Vlog acoustic shape [T,25], got {audio.shape}")
        if visual.ndim != 2 or visual.shape[1] != 136:
            raise ValueError(f"Expected D-Vlog visual shape [T,136], got {visual.shape}")

        duration = float(max(audio.shape[0], visual.shape[0]))
        windows = make_windows(
            duration,
            float(self.config["window_sec"]),
            float(self.config["stride_sec"]),
            float(self.config["minimum_tail_sec"]),
        )
        units: list[RawEvidenceUnit] = []
        for window in windows:
            units.append(self._audio_unit(str(session_id), window.index, window.start_sec, window.end_sec, audio_path, audio))
            units.append(self._visual_unit(str(session_id), window.index, window.start_sec, window.end_sec, visual_path, visual))
        for unit in units:
            unit.validate()
        return units

    def _audio_unit(
        self,
        session_id: str,
        index: int,
        start_sec: float,
        end_sec: float,
        path: Path,
        audio: np.ndarray,
    ) -> RawEvidenceUnit:
        source = self.source_config("audio", self.AUDIO_DEFAULTS)
        start = max(0, int(np.floor(start_sec)))
        end = min(audio.shape[0], int(np.ceil(end_sec)))
        block = np.asarray(audio[start:end], dtype=float)
        finite_rows = np.all(np.isfinite(block), axis=1) if block.size else np.zeros(0, dtype=bool)
        expected = expected_count(end_sec - start_sec, 1.0)
        valid_ratio = float(np.sum(finite_rows) / expected)
        reasons: list[str] = []
        if end <= start:
            reasons.append("source_ended_before_window")
        elif not np.any(finite_rows):
            reasons.append("invalid_acoustic_values")
        status, missing_reasons = self.availability(valid_ratio, reasons)
        observations = ()
        if status != "unavailable":
            row_indices = np.arange(start, end, dtype=int)[finite_rows]
            observations = tuple(
                egemaps_observations(block[finite_rows], D_VLOG_EGEMAPS_COLUMNS, row_indices)
            )
        return RawEvidenceUnit(
            evidence_id=f"DVLOG.{session_id}.AUDIO.{index:06d}",
            dataset=self.config.dataset_name,
            session_id=session_id,
            modality="audio",
            source_id=str(source["source_id"]),
            extractor=str(source["extractor"]),
            extractor_version=str(source["version"]),
            representation_level=source["representation_level"],
            source_files=(path.name,),
            start_sec=start_sec,
            end_sec=end_sec,
            availability_status=status,
            missing_reasons=missing_reasons,
            quality={
                "level": self.quality_level(valid_ratio, status == "unavailable"),
                "valid_ratio": valid_ratio,
                "sampling_rate_hz": 1.0,
            },
            context={
                "task_context": "in_the_wild_vlog",
                "feature_resolution": "one_second_mean",
                "column_schema": "standard_25_lld_egemaps_order",
                "column_schema_provenance": "reconstructed_from_standard_egemaps_and_release_values",
                "transcript_available": False,
            },
            observations=observations,
            semantic_limits=(
                "observable_behavior_only",
                "no_transcript_content_available",
                "no_clinical_conclusion",
            ),
        )

    def _visual_unit(
        self,
        session_id: str,
        index: int,
        start_sec: float,
        end_sec: float,
        path: Path,
        visual: np.ndarray,
    ) -> RawEvidenceUnit:
        source = self.source_config("visual", self.VISUAL_DEFAULTS)
        start = max(0, int(np.floor(start_sec)))
        end = min(visual.shape[0], int(np.ceil(end_sec)))
        block = np.asarray(visual[start:end], dtype=float)
        finite = np.all(np.isfinite(block), axis=1) if block.size else np.zeros(0, dtype=bool)
        nonzero = np.any(block != 0, axis=1) if block.size else np.zeros(0, dtype=bool)
        valid = finite & nonzero
        expected = expected_count(end_sec - start_sec, 1.0)
        valid_ratio = float(np.sum(valid) / expected)
        zero_ratio = float(np.sum(finite & ~nonzero) / expected)
        reasons: list[str] = []
        if end <= start:
            reasons.append("source_ended_before_window")
        elif not np.any(valid):
            reasons.append("face_detection_failure")
        elif zero_ratio > 0 and valid_ratio < 0.80:
            reasons.append("partial_face_detection_failure")
        status, missing_reasons = self.availability(valid_ratio, reasons)
        observations = ()
        if status != "unavailable":
            row_indices = np.arange(start, end, dtype=int)[valid]
            observations = tuple(landmark_observations(block[valid], row_indices))
        return RawEvidenceUnit(
            evidence_id=f"DVLOG.{session_id}.VISUAL.{index:06d}",
            dataset=self.config.dataset_name,
            session_id=session_id,
            modality="visual",
            source_id=str(source["source_id"]),
            extractor=str(source["extractor"]),
            extractor_version=str(source["version"]),
            representation_level=source["representation_level"],
            source_files=(path.name,),
            start_sec=start_sec,
            end_sec=end_sec,
            availability_status=status,
            missing_reasons=missing_reasons,
            quality={
                "level": self.quality_level(valid_ratio, status == "unavailable"),
                "valid_ratio": valid_ratio,
                "zero_vector_ratio": zero_ratio,
                "sampling_rate_hz": 1.0,
            },
            context={
                "task_context": "in_the_wild_vlog",
                "landmark_layout": "x0_to_x67_then_y0_to_y67",
                "coordinates": "release_normalized",
                "zero_vector_interpretation": "face_detection_failure_within_released_length",
            },
            observations=observations,
            semantic_limits=(
                "observable_behavior_only",
                "no_action_unit_claim",
                "no_microexpression_claim",
                "no_clinical_conclusion",
            ),
        )
