"""Dataset adapter for DAIC-WOZ feature files and transcripts."""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..intervals import (
    Interval,
    overlap_duration,
    overlap_reasons,
    timestamps_in_intervals,
)
from ..metrics import (
    COVAREP_COLUMNS,
    action_unit_observations,
    covarep_observations,
    gaze_observations,
    pose_observations,
)
from ..readers import natural_session_key
from ..schema import RawEvidenceUnit, RawObservation
from ..transcript import TranscriptUtterance, normalize_transcript_text
from ..windowing import estimate_sampling_rate, expected_count, make_windows, slice_bounds
from .base import DatasetAdapter


class DaicWozAdapter(DatasetAdapter):
    AUDIO_DEFAULTS = {
        "source_id": "daic_covarep_audio",
        "extractor": "COVAREP",
        "version": "1.3.2",
        "representation_level": "expert_feature",
    }
    VISUAL_DEFAULTS = {
        "source_id": "daic_clnf_visual",
        "extractor": "CLNF/OpenFace",
        "version": "dataset_release",
        "representation_level": "expert_feature",
    }
    REQUIRED_SUFFIXES = (
        "CLNF_AUs.txt",
        "CLNF_gaze.txt",
        "CLNF_pose.txt",
        "COVAREP.csv",
        "TRANSCRIPT.csv",
    )

    def list_session_ids(self) -> list[str]:
        ids: set[str] = set()
        for path in self.root.glob("*_P"):
            if path.is_dir():
                ids.add(path.name[:-2])
        for flat_root in self._flat_roots():
            for path in flat_root.glob("*_COVAREP.csv"):
                ids.add(path.name[: -len("_COVAREP.csv")])
        for path in self.root.glob("*_P.zip"):
            ids.add(path.name[:-6])
        return sorted(ids, key=natural_session_key)

    def _directory(self, session_id: str) -> Path:
        return self.root / f"{session_id}_P"

    def _archive(self, session_id: str) -> Path:
        return self.root / f"{session_id}_P.zip"

    def _flat_roots(self) -> tuple[Path, ...]:
        nested = self.root / "data"
        return (nested, self.root) if nested.is_dir() else (self.root,)

    def _flat_file(self, session_id: str, suffix: str) -> Path | None:
        filename = self._filename(session_id, suffix)
        return next(
            (root / filename for root in self._flat_roots() if (root / filename).is_file()),
            None,
        )

    def _layout(self, session_id: str) -> str:
        if self._directory(session_id).is_dir():
            return "directory"
        if self._flat_file(session_id, "COVAREP.csv") is not None:
            return "flat_directory"
        if self._archive(session_id).is_file():
            return "zip"
        raise FileNotFoundError(
            f"No DAIC-WOZ participant directory, flat extracted files, or zip found for session {session_id}"
        )

    @staticmethod
    def _filename(session_id: str, suffix: str) -> str:
        return f"{session_id}_{suffix}"

    def _member_exists(self, session_id: str, suffix: str) -> bool:
        filename = self._filename(session_id, suffix)
        layout = self._layout(session_id)
        if layout == "directory":
            return (self._directory(session_id) / filename).is_file()
        if layout == "flat_directory":
            return self._flat_file(session_id, suffix) is not None
        with zipfile.ZipFile(self._archive(session_id)) as archive:
            return any(Path(name).name == filename for name in archive.namelist())

    def _archive_member(self, session_id: str, filename: str) -> str:
        with zipfile.ZipFile(self._archive(session_id)) as archive:
            candidates = [name for name in archive.namelist() if Path(name).name == filename]
        if not candidates:
            raise FileNotFoundError(f"{filename} is absent from {self._archive(session_id)}")
        return min(candidates, key=lambda name: (name.count("/"), len(name)))

    def _read_csv(self, session_id: str, suffix: str, **kwargs: Any) -> pd.DataFrame:
        filename = self._filename(session_id, suffix)
        layout = self._layout(session_id)
        if layout == "directory":
            return pd.read_csv(self._directory(session_id) / filename, **kwargs)
        if layout == "flat_directory":
            path = self._flat_file(session_id, suffix)
            if path is None:
                raise FileNotFoundError(filename)
            return pd.read_csv(path, **kwargs)
        with zipfile.ZipFile(self._archive(session_id)) as archive:
            with archive.open(self._archive_member(session_id, filename)) as handle:
                return pd.read_csv(handle, **kwargs)

    def audit_session(self, session_id: str) -> dict[str, Any]:
        result: dict[str, Any] = {
            "session_id": str(session_id),
            "layout": self._layout(str(session_id)),
            "errors": [],
            "warnings": [],
        }
        for suffix in self.REQUIRED_SUFFIXES:
            if not self._member_exists(str(session_id), suffix):
                result["errors"].append(f"missing_{suffix}")
        if not result["errors"]:
            try:
                au_header = [
                    name.strip()
                    for name in self._read_csv(str(session_id), "CLNF_AUs.txt", nrows=0).columns
                ]
                pose_header = [
                    name.strip()
                    for name in self._read_csv(str(session_id), "CLNF_pose.txt", nrows=0).columns
                ]
                gaze_header = [
                    name.strip()
                    for name in self._read_csv(str(session_id), "CLNF_gaze.txt", nrows=0).columns
                ]
                if not {"timestamp", "confidence", "success", "AU04_r"}.issubset(au_header):
                    result["errors"].append("invalid_au_header")
                if not {"timestamp", "confidence", "success", "Rx", "Ry", "Rz"}.issubset(pose_header):
                    result["errors"].append("invalid_pose_header")
                if not {"timestamp", "confidence", "success", "x_h0", "y_h0", "z_h0"}.issubset(gaze_header):
                    result["errors"].append("invalid_gaze_header")
                covarep = self._read_csv(str(session_id), "COVAREP.csv", header=None, nrows=2)
                result["covarep_columns"] = int(covarep.shape[1])
                if covarep.shape[1] != len(COVAREP_COLUMNS):
                    result["errors"].append("invalid_covarep_width")
            except Exception as exc:  # pragma: no cover - exercised by real-data audit
                result["errors"].append(f"header_read_error:{type(exc).__name__}")
        result["ok"] = not result["errors"]
        return result

    def load_session(self, session_id: str) -> list[RawEvidenceUnit]:
        session_id = str(session_id)
        missing = [suffix for suffix in self.REQUIRED_SUFFIXES if not self._member_exists(session_id, suffix)]
        if missing:
            raise FileNotFoundError(f"DAIC-WOZ session {session_id} is missing: {', '.join(missing)}")

        aus = self._read_csv(session_id, "CLNF_AUs.txt", dtype=np.float32, skipinitialspace=True)
        gaze = self._read_csv(session_id, "CLNF_gaze.txt", dtype=np.float32, skipinitialspace=True)
        pose = self._read_csv(session_id, "CLNF_pose.txt", dtype=np.float32, skipinitialspace=True)
        for frame in (aus, gaze, pose):
            frame.columns = [name.strip() for name in frame.columns]
        covarep = self._read_csv(
            session_id,
            "COVAREP.csv",
            header=None,
            names=list(COVAREP_COLUMNS),
            dtype=np.float32,
        )
        transcript = self.load_transcript(session_id)
        participant_intervals, privacy_intervals = self._transcript_intervals(transcript)
        interruption_intervals = self._known_intervals(session_id)

        au_times = aus["timestamp"].to_numpy(dtype=float)
        gaze_times = gaze["timestamp"].to_numpy(dtype=float)
        pose_times = pose["timestamp"].to_numpy(dtype=float)
        audio_rate = float(self.config.get("audio_sample_rate_hz", 100.0))
        audio_times = np.arange(len(covarep), dtype=float) / audio_rate
        transcript_end = max(
            (item.end_sec for item in participant_intervals + privacy_intervals), default=0.0
        )
        duration = max(
            float(np.nanmax(au_times)) if au_times.size else 0.0,
            float(np.nanmax(gaze_times)) if gaze_times.size else 0.0,
            float(np.nanmax(pose_times)) if pose_times.size else 0.0,
            float(audio_times[-1]) if audio_times.size else 0.0,
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
                self._visual_unit(
                    session_id,
                    window.index,
                    window.start_sec,
                    window.end_sec,
                    aus,
                    au_times,
                    gaze,
                    gaze_times,
                    pose,
                    pose_times,
                    interruption_intervals,
                )
            )
            units.append(
                self._audio_unit(
                    session_id,
                    window.index,
                    window.start_sec,
                    window.end_sec,
                    covarep,
                    audio_times,
                    participant_intervals,
                    privacy_intervals,
                    interruption_intervals,
                    transcript_available=bool(transcript),
                )
            )
        for unit in units:
            unit.validate()
        return units

    def load_transcript(self, session_id: str) -> list[TranscriptUtterance]:
        frame = self._read_csv(
            str(session_id), "TRANSCRIPT.csv", sep="\t", dtype="string"
        )
        frame.columns = [name.strip().lower() for name in frame.columns]
        needed = {"start_time", "stop_time", "speaker", "value"}
        if not needed.issubset(frame.columns):
            return []
        output: list[TranscriptUtterance] = []
        for row in frame.itertuples(index=False):
            values = row._asdict()
            try:
                start = float(values["start_time"])
                end = float(values["stop_time"])
            except (TypeError, ValueError):
                continue
            raw_text = "" if pd.isna(values["value"]) else str(values["value"])
            privacy_scrubbed = "scrubbed_entry" in raw_text.casefold()
            text = None if privacy_scrubbed else normalize_transcript_text(raw_text)
            if not privacy_scrubbed and not text:
                continue
            raw_speaker = str(values["speaker"]).strip().casefold()
            if "participant" in raw_speaker:
                speaker_role = "participant"
            elif "ellie" in raw_speaker or "interviewer" in raw_speaker:
                speaker_role = "interviewer"
            else:
                speaker_role = "unassigned"
            try:
                utterance = TranscriptUtterance(
                    start_sec=start,
                    end_sec=end,
                    text=text,
                    speaker_role=speaker_role,
                    source_kind="manual_transcript",
                    privacy_scrubbed=privacy_scrubbed,
                )
                utterance.validate()
            except ValueError:
                continue
            output.append(utterance)
        return output

    @staticmethod
    def _transcript_intervals(
        utterances: list[TranscriptUtterance],
    ) -> tuple[list[Interval], list[Interval]]:
        participant: list[Interval] = []
        privacy: list[Interval] = []
        for utterance in utterances:
            if utterance.speaker_role == "participant" and not utterance.privacy_scrubbed:
                participant.append(
                    Interval(
                        utterance.start_sec,
                        utterance.end_sec,
                        "participant_speech",
                    )
                )
            if utterance.privacy_scrubbed:
                privacy.append(
                    Interval(
                        utterance.start_sec,
                        utterance.end_sec,
                        "privacy_scrubbed",
                    )
                )
        return participant, privacy

    def _known_intervals(self, session_id: str) -> list[Interval]:
        items = self.config.get("known_intervals", {}).get(str(session_id), [])
        return [
            Interval(float(item["start"]), float(item["end"]), str(item["reason"]))
            for item in items
        ]

    def _visual_source_slice(
        self,
        frame: pd.DataFrame,
        times: np.ndarray,
        start_sec: float,
        end_sec: float,
        interruptions: list[Interval],
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, float, float]:
        left, right = slice_bounds(times, start_sec, end_sec)
        block = frame.iloc[left:right]
        block_times = times[left:right]
        threshold = float(self.config.get("visual_confidence_threshold", 0.8))
        confidence = block["confidence"].to_numpy(dtype=float) if len(block) else np.empty(0)
        success = block["success"].to_numpy(dtype=float) if len(block) else np.empty(0)
        interrupted = timestamps_in_intervals(block_times, interruptions)
        valid = (
            np.isfinite(confidence)
            & (confidence >= threshold)
            & np.isfinite(success)
            & (success >= 0.5)
            & ~interrupted
        )
        rate = estimate_sampling_rate(times, 30.0)
        ratio = float(np.sum(valid) / expected_count(end_sec - start_sec, rate))
        return block, valid, np.arange(left, right, dtype=int), ratio, rate

    def _visual_unit(
        self,
        session_id: str,
        index: int,
        start_sec: float,
        end_sec: float,
        aus: pd.DataFrame,
        au_times: np.ndarray,
        gaze: pd.DataFrame,
        gaze_times: np.ndarray,
        pose: pd.DataFrame,
        pose_times: np.ndarray,
        interruptions: list[Interval],
    ) -> RawEvidenceUnit:
        source = self.source_config("visual", self.VISUAL_DEFAULTS)
        au_block, au_valid, au_rows, au_ratio, _ = self._visual_source_slice(
            aus, au_times, start_sec, end_sec, interruptions
        )
        gaze_block, gaze_valid, _, gaze_ratio, _ = self._visual_source_slice(
            gaze, gaze_times, start_sec, end_sec, interruptions
        )
        pose_block, pose_valid, _, pose_ratio, _ = self._visual_source_slice(
            pose, pose_times, start_sec, end_sec, interruptions
        )
        valid_ratio = float(np.mean([au_ratio, gaze_ratio, pose_ratio]))
        reasons = overlap_reasons(start_sec, end_sec, interruptions)
        if not any((len(au_block), len(gaze_block), len(pose_block))):
            reasons.append("source_ended_before_window")
        elif not any((np.any(au_valid), np.any(gaze_valid), np.any(pose_valid))):
            reasons.append("face_tracking_failure")
        elif valid_ratio < 0.8:
            reasons.append("partial_face_tracking")
        status, missing_reasons = self.availability(valid_ratio, reasons)

        observations: list[RawObservation] = []
        confidences: list[np.ndarray] = []
        if status != "unavailable":
            if np.any(au_valid):
                selected = au_block.iloc[np.flatnonzero(au_valid)]
                intensity_columns = [name for name in aus.columns if name.endswith("_r")]
                presence_columns = [name for name in aus.columns if name.endswith("_c")]
                observations.extend(
                    action_unit_observations(
                        selected[intensity_columns].to_numpy(dtype=float),
                        intensity_columns,
                        selected[presence_columns].to_numpy(dtype=float) if presence_columns else None,
                        au_rows[au_valid],
                    )
                )
                confidences.append(selected["confidence"].to_numpy(dtype=float))
            if np.any(pose_valid):
                selected = pose_block.iloc[np.flatnonzero(pose_valid)]
                observations.extend(
                    pose_observations(
                        selected[["Rx", "Ry", "Rz"]].to_numpy(dtype=float),
                        selected["timestamp"].to_numpy(dtype=float),
                    )
                )
                confidences.append(selected["confidence"].to_numpy(dtype=float))
            if np.any(gaze_valid):
                selected = gaze_block.iloc[np.flatnonzero(gaze_valid)]
                left = selected[["x_h0", "y_h0", "z_h0"]].to_numpy(dtype=float)
                right = selected[["x_h1", "y_h1", "z_h1"]].to_numpy(dtype=float)
                mean = (left + right) / 2.0
                angle_x = np.arctan2(mean[:, 0], -mean[:, 2])
                angle_y = np.arctan2(mean[:, 1], -mean[:, 2])
                observations.extend(gaze_observations(angle_x, angle_y, left, right))
                confidences.append(selected["confidence"].to_numpy(dtype=float))
        confidence_value = (
            float(np.nanmedian(np.concatenate(confidences))) if confidences else None
        )
        return RawEvidenceUnit(
            evidence_id=f"DAIC.{session_id}.VISUAL.{index:06d}",
            dataset=self.config.dataset_name,
            session_id=session_id,
            modality="visual",
            source_id=str(source["source_id"]),
            extractor=str(source["extractor"]),
            extractor_version=str(source["version"]),
            representation_level=source["representation_level"],
            source_files=tuple(self._filename(session_id, suffix) for suffix in ("CLNF_AUs.txt", "CLNF_gaze.txt", "CLNF_pose.txt")),
            start_sec=start_sec,
            end_sec=end_sec,
            availability_status=status,
            missing_reasons=missing_reasons,
            quality={
                "level": self.quality_level(valid_ratio, status == "unavailable"),
                "valid_ratio": valid_ratio,
                "au_valid_ratio": au_ratio,
                "gaze_valid_ratio": gaze_ratio,
                "pose_valid_ratio": pose_ratio,
                "tracking_confidence": confidence_value,
                "confidence_threshold": float(self.config.get("visual_confidence_threshold", 0.8)),
            },
            context={"task_context": "semi_structured_interview", "collection_condition": "wizard_of_oz"},
            observations=tuple(observations),
        )

    def _audio_unit(
        self,
        session_id: str,
        index: int,
        start_sec: float,
        end_sec: float,
        frame: pd.DataFrame,
        times: np.ndarray,
        participant_intervals: list[Interval],
        privacy_intervals: list[Interval],
        interruptions: list[Interval],
        transcript_available: bool,
    ) -> RawEvidenceUnit:
        source = self.source_config("audio", self.AUDIO_DEFAULTS)
        left, right = slice_bounds(times, start_sec, end_sec)
        block = frame.iloc[left:right].to_numpy(dtype=float)
        block_times = times[left:right]
        finite = np.all(np.isfinite(block), axis=1) if block.size else np.zeros(0, dtype=bool)
        privacy_mask = timestamps_in_intervals(block_times, privacy_intervals)
        interruption_mask = timestamps_in_intervals(block_times, interruptions)
        source_valid = finite & ~privacy_mask & ~interruption_mask
        participant_only = bool(self.config.get("participant_only_audio", True))
        participant_mask = (
            timestamps_in_intervals(block_times, participant_intervals)
            if participant_only and transcript_available
            else np.ones(block_times.shape, dtype=bool)
        )
        selected = source_valid & participant_mask
        rate = float(self.config.get("audio_sample_rate_hz", 100.0))
        expected = expected_count(end_sec - start_sec, rate)
        valid_ratio = float(np.sum(source_valid) / expected)
        speaking_ratio = (
            overlap_duration(start_sec, end_sec, participant_intervals) / (end_sec - start_sec)
            if transcript_available
            else None
        )
        privacy_ratio = overlap_duration(start_sec, end_sec, privacy_intervals) / (end_sec - start_sec)
        interruption_ratio = overlap_duration(start_sec, end_sec, interruptions) / (end_sec - start_sec)
        reasons = overlap_reasons(start_sec, end_sec, privacy_intervals + interruptions)
        if not len(block):
            reasons.append("source_ended_before_window")
        elif valid_ratio < 0.8:
            reasons.append("partial_acoustic_features")
        if participant_only and not transcript_available:
            reasons.append("speaker_alignment_unavailable")
        status, missing_reasons = self.availability(valid_ratio, reasons)
        observations: list[RawObservation] = []
        if status != "unavailable":
            if speaking_ratio is not None:
                observations.append(
                    RawObservation("participant_speaking_ratio", speaking_ratio, "ratio", "derived")
                )
            if np.any(selected):
                observations.extend(
                    covarep_observations(block[selected], np.arange(left, right, dtype=int)[selected])
                )
        voiced = next((item.value for item in observations if item.name == "voiced_activity_ratio"), None)
        return RawEvidenceUnit(
            evidence_id=f"DAIC.{session_id}.AUDIO.{index:06d}",
            dataset=self.config.dataset_name,
            session_id=session_id,
            modality="audio",
            source_id=str(source["source_id"]),
            extractor=str(source["extractor"]),
            extractor_version=str(source["version"]),
            representation_level=source["representation_level"],
            source_files=(self._filename(session_id, "COVAREP.csv"), self._filename(session_id, "TRANSCRIPT.csv")),
            start_sec=start_sec,
            end_sec=end_sec,
            availability_status=status,
            missing_reasons=missing_reasons,
            quality={
                "level": self.quality_level(valid_ratio, status == "unavailable"),
                "valid_ratio": valid_ratio,
                "participant_speaking_ratio": speaking_ratio,
                "voiced_ratio_within_speech": voiced,
                "privacy_scrubbed_ratio": privacy_ratio,
                "known_interruption_ratio": interruption_ratio,
            },
            context={
                "task_context": "semi_structured_interview",
                "collection_condition": "wizard_of_oz",
                "acoustic_scope": "participant_only" if participant_only and transcript_available else "unsegmented_audio",
                "transcript_content_exposed": False,
            },
            observations=tuple(observations),
        )
