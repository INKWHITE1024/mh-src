"""Dataset adapter for E-DAIC feature files and transcripts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..intervals import Interval, overlap_duration, timestamps_in_intervals
from ..metrics import (
    action_unit_observations,
    egemaps_observations,
    gaze_observations,
    pose_observations,
)
from ..readers import natural_session_key
from ..schema import RawEvidenceUnit, RawObservation
from ..transcript import TranscriptUtterance, normalize_transcript_text
from ..windowing import estimate_sampling_rate, expected_count, make_windows, slice_bounds
from .base import DatasetAdapter


class EDaicAdapter(DatasetAdapter):
    AUDIO_DEFAULTS = {
        "source_id": "edaic_egemaps_audio",
        "extractor": "openSMILE eGeMAPS LLD",
        "version": "2.3.0",
        "representation_level": "expert_feature",
    }
    VISUAL_DEFAULTS = {
        "source_id": "edaic_openface_visual",
        "extractor": "OpenFace",
        "version": "2.1.0",
        "representation_level": "expert_feature",
    }

    @property
    def data_root(self) -> Path:
        nested = self.root / "data"
        return nested if nested.is_dir() else self.root

    def list_session_ids(self) -> list[str]:
        ids = []
        for path in self.data_root.glob("*_P"):
            if not path.is_dir():
                continue
            session_id = path.name[:-2]
            if (path / "features").is_dir():
                ids.append(session_id)
        return sorted(ids, key=natural_session_key)

    def _paths(self, session_id: str) -> dict[str, Path]:
        directory = self.data_root / f"{session_id}_P"
        features = directory / "features"
        return {
            "directory": directory,
            "transcript": directory / f"{session_id}_Transcript.csv",
            "visual": features / f"{session_id}_OpenFace2.1.0_Pose_gaze_AUs.csv",
            "audio": features / f"{session_id}_OpenSMILE2.3.0_egemaps.csv",
        }

    def audit_session(self, session_id: str) -> dict[str, Any]:
        paths = self._paths(session_id)
        result: dict[str, Any] = {"session_id": str(session_id), "errors": [], "warnings": []}
        required = {"visual": paths["visual"], "audio": paths["audio"], "transcript": paths["transcript"]}
        for name, path in required.items():
            if not path.exists():
                result["errors" if name != "transcript" else "warnings"].append(f"missing_{name}_file")
        try:
            if paths["visual"].exists():
                header = [name.strip() for name in pd.read_csv(paths["visual"], nrows=0).columns]
                needed = {"timestamp", "confidence", "success", "pose_Rx", "gaze_angle_x", "AU04_r"}
                missing = sorted(needed - set(header))
                if missing:
                    result["errors"].append(f"visual_columns_missing:{','.join(missing)}")
                result["visual_columns"] = len(header)
            if paths["audio"].exists():
                header = [name.strip() for name in pd.read_csv(paths["audio"], sep=";", nrows=0).columns]
                needed = {"frameTime", "Loudness_sma3", "spectralFlux_sma3", "F0semitoneFrom27.5Hz_sma3nz"}
                missing = sorted(needed - set(header))
                if missing:
                    result["errors"].append(f"audio_columns_missing:{','.join(missing)}")
                result["audio_columns"] = len(header)
        except Exception as exc:  # pragma: no cover - exercised by real-data audit
            result["errors"].append(f"header_read_error:{type(exc).__name__}")
        result["collection_condition"] = self._collection_condition(str(session_id))
        result["ok"] = not result["errors"]
        return result

    def load_session(self, session_id: str) -> list[RawEvidenceUnit]:
        paths = self._paths(session_id)
        if not paths["visual"].exists() or not paths["audio"].exists():
            raise FileNotFoundError(f"Missing E-DAIC expert features for session {session_id}")
        visual = pd.read_csv(paths["visual"], dtype=np.float32)
        visual.columns = [name.strip() for name in visual.columns]
        audio = pd.read_csv(paths["audio"], sep=";", dtype={"name": "string"})
        audio.columns = [name.strip() for name in audio.columns]
        for column in audio.columns:
            if column != "name":
                audio[column] = pd.to_numeric(audio[column], errors="coerce").astype(np.float32)

        visual_times = visual["timestamp"].to_numpy(dtype=float)
        audio_times = audio["frameTime"].to_numpy(dtype=float)
        transcript = self.load_transcript(str(session_id))
        speech_intervals = self._transcript_intervals(transcript)
        transcript_end = max((item.end_sec for item in speech_intervals), default=0.0)
        duration = max(
            float(np.nanmax(visual_times)) if visual_times.size else 0.0,
            float(np.nanmax(audio_times)) if audio_times.size else 0.0,
            transcript_end,
        )
        windows = make_windows(
            duration,
            float(self.config["window_sec"]),
            float(self.config["stride_sec"]),
            float(self.config["minimum_tail_sec"]),
        )
        units: list[RawEvidenceUnit] = []
        for window in windows:
            units.append(
                self._visual_unit(str(session_id), window.index, window.start_sec, window.end_sec, paths["visual"], visual, visual_times)
            )
            units.append(
                self._audio_unit(
                    str(session_id),
                    window.index,
                    window.start_sec,
                    window.end_sec,
                    paths["audio"],
                    audio,
                    audio_times,
                    speech_intervals,
                    bool(speech_intervals),
                )
            )
        for unit in units:
            unit.validate()
        return units

    @staticmethod
    def _collection_condition(session_id: str) -> str:
        try:
            value = int(session_id)
        except ValueError:
            return "unknown"
        if 300 <= value <= 492:
            return "wizard_of_oz"
        if 600 <= value <= 718:
            return "autonomous_ai"
        return "unknown"

    def load_transcript(self, session_id: str) -> list[TranscriptUtterance]:
        path = self._paths(str(session_id))["transcript"]
        if not path.exists():
            return []
        transcript = pd.read_csv(path)
        transcript.columns = [name.strip().lower() for name in transcript.columns]
        start_name = "start_time" if "start_time" in transcript.columns else "start"
        end_name = "end_time" if "end_time" in transcript.columns else "stop_time"
        text_name = "text" if "text" in transcript.columns else "value"
        if (
            start_name not in transcript.columns
            or end_name not in transcript.columns
            or text_name not in transcript.columns
        ):
            return []
        confidence_name = (
            "confidence" if "confidence" in transcript.columns else None
        )
        utterances: list[TranscriptUtterance] = []
        for _, row in transcript.iterrows():
            try:
                start = float(row[start_name])
                end = float(row[end_name])
            except (TypeError, ValueError):
                continue
            raw_text = "" if pd.isna(row[text_name]) else str(row[text_name])
            privacy_scrubbed = "scrubbed_entry" in raw_text.casefold()
            text = None if privacy_scrubbed else normalize_transcript_text(raw_text)
            if not privacy_scrubbed and not text:
                continue
            confidence: float | None = None
            if confidence_name is not None and not pd.isna(row[confidence_name]):
                try:
                    candidate_confidence = float(row[confidence_name])
                    if (
                        np.isfinite(candidate_confidence)
                        and 0.0 <= candidate_confidence <= 1.0
                    ):
                        confidence = candidate_confidence
                except (TypeError, ValueError):
                    confidence = None
            try:
                utterance = TranscriptUtterance(
                    start_sec=start,
                    end_sec=end,
                    text=text,
                    speaker_role="unassigned",
                    source_kind="automatic_speech_recognition",
                    confidence=confidence,
                    privacy_scrubbed=privacy_scrubbed,
                )
                utterance.validate()
            except ValueError:
                continue
            utterances.append(utterance)
        return utterances

    @staticmethod
    def _transcript_intervals(
        utterances: list[TranscriptUtterance],
    ) -> list[Interval]:
        return [
            Interval(
                utterance.start_sec,
                utterance.end_sec,
                "transcribed_speech_speaker_unassigned",
            )
            for utterance in utterances
            if not utterance.privacy_scrubbed
        ]

    def _visual_unit(
        self,
        session_id: str,
        index: int,
        start_sec: float,
        end_sec: float,
        path: Path,
        frame: pd.DataFrame,
        times: np.ndarray,
    ) -> RawEvidenceUnit:
        source = self.source_config("visual", self.VISUAL_DEFAULTS)
        left, right = slice_bounds(times, start_sec, end_sec)
        block = frame.iloc[left:right]
        threshold = float(self.config.get("visual_confidence_threshold", 0.8))
        confidence = block["confidence"].to_numpy(dtype=float) if len(block) else np.empty(0)
        success = block["success"].to_numpy(dtype=float) if len(block) else np.empty(0)
        valid = np.isfinite(confidence) & (confidence >= threshold) & np.isfinite(success) & (success >= 0.5)
        rate = estimate_sampling_rate(times, 30.0)
        expected = expected_count(end_sec - start_sec, rate)
        valid_ratio = float(np.sum(valid) / expected)
        reasons: list[str] = []
        if not len(block):
            reasons.append("source_ended_before_window")
        elif not np.any(valid):
            reasons.append("face_tracking_failure")
        elif valid_ratio < 0.8:
            reasons.append("partial_face_tracking")
        status, missing_reasons = self.availability(valid_ratio, reasons)

        observations: list[RawObservation] = []
        if status != "unavailable":
            selected = block.iloc[np.flatnonzero(valid)]
            row_indices = np.arange(left, right, dtype=int)[valid]
            intensity_columns = [name for name in frame.columns if name.endswith("_r")]
            presence_columns = [name for name in frame.columns if name.endswith("_c")]
            observations.extend(
                action_unit_observations(
                    selected[intensity_columns].to_numpy(dtype=float),
                    intensity_columns,
                    selected[presence_columns].to_numpy(dtype=float) if presence_columns else None,
                    row_indices,
                )
            )
            rotations = selected[["pose_Rx", "pose_Ry", "pose_Rz"]].to_numpy(dtype=float)
            observations.extend(pose_observations(rotations, selected["timestamp"].to_numpy(dtype=float)))
            left_vectors = selected[["gaze_0_x", "gaze_0_y", "gaze_0_z"]].to_numpy(dtype=float)
            right_vectors = selected[["gaze_1_x", "gaze_1_y", "gaze_1_z"]].to_numpy(dtype=float)
            observations.extend(
                gaze_observations(
                    selected["gaze_angle_x"].to_numpy(dtype=float),
                    selected["gaze_angle_y"].to_numpy(dtype=float),
                    left_vectors,
                    right_vectors,
                )
            )
        return RawEvidenceUnit(
            evidence_id=f"EDAIC.{session_id}.VISUAL.{index:06d}",
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
                "tracking_confidence": float(np.nanmedian(confidence[valid])) if np.any(valid) else None,
                "confidence_threshold": threshold,
            },
            context={
                "task_context": "semi_structured_interview",
                "collection_condition": self._collection_condition(session_id),
            },
            observations=tuple(observations),
        )

    def _audio_unit(
        self,
        session_id: str,
        index: int,
        start_sec: float,
        end_sec: float,
        path: Path,
        frame: pd.DataFrame,
        times: np.ndarray,
        speech_intervals: list[Interval],
        transcript_available: bool,
    ) -> RawEvidenceUnit:
        source = self.source_config("audio", self.AUDIO_DEFAULTS)
        left, right = slice_bounds(times, start_sec, end_sec)
        block = frame.iloc[left:right]
        block_times = times[left:right]
        numeric_columns = [name for name in frame.columns if name not in {"name", "frameTime"}]
        matrix = block[numeric_columns].to_numpy(dtype=float) if len(block) else np.empty((0, len(numeric_columns)))
        source_valid = np.all(np.isfinite(matrix), axis=1) if matrix.size else np.zeros(0, dtype=bool)
        transcript_aligned = bool(self.config.get("transcript_aligned_audio", True))
        if transcript_aligned and transcript_available:
            transcript_mask = timestamps_in_intervals(block_times, speech_intervals)
        else:
            transcript_mask = np.ones(block_times.shape, dtype=bool)
        selected_mask = source_valid & transcript_mask
        rate = estimate_sampling_rate(times, 100.0)
        expected = expected_count(end_sec - start_sec, rate)
        source_valid_ratio = float(np.sum(source_valid) / expected)
        transcribed_speech_ratio = (
            overlap_duration(start_sec, end_sec, speech_intervals) / (end_sec - start_sec)
            if transcript_available
            else None
        )
        reasons: list[str] = []
        if not len(block):
            reasons.append("source_ended_before_window")
        elif source_valid_ratio < 0.8:
            reasons.append("partial_acoustic_features")
        if transcript_aligned and not transcript_available:
            reasons.append("transcript_alignment_unavailable")
        status, missing_reasons = self.availability(source_valid_ratio, reasons)
        observations: list[RawObservation] = []
        if status != "unavailable":
            if transcribed_speech_ratio is not None:
                observations.append(
                    RawObservation(
                        "transcribed_speech_ratio",
                        transcribed_speech_ratio,
                        "ratio",
                        "derived",
                    )
                )
            if np.any(selected_mask):
                row_indices = np.arange(left, right, dtype=int)[selected_mask]
                observations.extend(egemaps_observations(matrix[selected_mask], numeric_columns, row_indices))
        voiced = next((item.value for item in observations if item.name == "voiced_activity_ratio"), None)
        return RawEvidenceUnit(
            evidence_id=f"EDAIC.{session_id}.AUDIO.{index:06d}",
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
                "level": self.quality_level(source_valid_ratio, status == "unavailable"),
                "valid_ratio": source_valid_ratio,
                "transcribed_speech_ratio": transcribed_speech_ratio,
                "voiced_ratio_within_speech": voiced,
            },
            context={
                "task_context": "semi_structured_interview",
                "collection_condition": self._collection_condition(session_id),
                "acoustic_scope": (
                    "transcribed_speech_speaker_unassigned"
                    if transcript_aligned and transcript_available
                    else "unsegmented_audio"
                ),
                "transcript_speaker_scope": "mixed_speakers_unassigned",
                "transcript_content_exposed": False,
            },
            observations=tuple(observations),
        )
