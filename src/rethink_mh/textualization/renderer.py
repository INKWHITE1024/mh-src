"""Render evidence units and observations into model-facing text with semantic safety checks."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

from .schema import EvidenceUnit, Observation


DISPLAY_NAMES = {
    "participant_speaking_ratio": "participant speaking coverage",
    "transcribed_speech_ratio": "transcribed speech coverage (speaker unassigned)",
    "voiced_activity_ratio": "voiced-frame coverage",
    "loudness_level": "loudness level",
    "loudness_variability": "loudness variability",
    "spectral_flux_level": "spectral flux level",
    "spectral_flux_variability": "spectral flux variability",
    "pitch_level": "pitch level",
    "pitch_variability": "pitch variability",
    "jitter_level": "local jitter level",
    "shimmer_level": "local shimmer level",
    "voice_harmonicity": "voice harmonicity",
    "cepstral_change_magnitude": "cepstral change magnitude",
    "naq_level": "normalized amplitude quotient",
    "qoq_level": "quasi-open quotient",
    "h1h2_level": "H1-H2 level",
    "psp_level": "parabolic spectral parameter",
    "mdq_level": "maxima dispersion quotient",
    "peak_slope_level": "peak-slope level",
    "rd_level": "glottal Rd level",
    "creak_activity": "COVAREP creak measurement",
    "spectral_change_magnitude": "spectral change magnitude",
    "facial_action_activity": "facial-action activation",
    "facial_action_variability": "facial-action variability",
    "facial_action_change_magnitude": "facial-action change magnitude",
    "facial_action_presence_ratio": "detected facial-action coverage",
    "au04_intensity": "AU04 intensity",
    "au12_intensity": "AU12 intensity",
    "au15_intensity": "AU15 intensity",
    "au25_intensity": "AU25 intensity",
    "head_pitch_angle": "head pitch angle",
    "head_yaw_angle": "head yaw angle",
    "head_motion_variability": "head-rotation variability",
    "head_rotation_range": "head-rotation range",
    "head_rotation_speed": "head-rotation speed",
    "gaze_horizontal_angle": "horizontal gaze angle",
    "gaze_vertical_angle": "vertical gaze angle",
    "gaze_direction_variability": "gaze-direction variability",
    "binocular_gaze_inconsistency": "binocular gaze inconsistency",
    "facial_landmark_shape_change": "facial-landmark shape change",
    "facial_shape_variability": "facial-shape variability",
    "mouth_region_shape_variability": "mouth-region shape variability",
    "eye_region_shape_variability": "eye-region shape variability",
    "mouth_opening": "mouth opening distance",
}

CATEGORY_TEXT = {
    "very_low": "far below",
    "low": "below",
    "typical": "within the central range of",
    "high": "above",
    "very_high": "far above",
}

REASON_TEXT = {
    "source_ended_before_window": "the released source ended before this window",
    "invalid_acoustic_values": "the acoustic values were invalid",
    "face_detection_failure": "face detection failed",
    "partial_face_detection_failure": "face detection failed for part of the window",
    "face_tracking_failure": "face tracking failed",
    "partial_face_tracking": "face tracking was incomplete",
    "partial_acoustic_features": "the acoustic source was incomplete",
    "speaker_alignment_unavailable": "speaker alignment was unavailable",
    "transcript_alignment_unavailable": "transcript time alignment was unavailable",
    "privacy_scrubbed": "the interval overlaps privacy-scrubbed material",
    "technical_interruption": "the interval overlaps a documented technical interruption",
    "external_interruption": "the interval overlaps a documented external interruption",
    "no_valid_measurements": "no valid measurements were available",
}

FORBIDDEN_INTERPRETIVE_PATTERNS = (
    r"\bdepress(?:ed|ion|ive)?\b",
    r"\banxi(?:ous|ety)\b",
    r"\bsad(?:ness)?\b",
    r"\bhopeless(?:ness)?\b",
    r"\bdull gaze\b",
    r"\bhelpless(?:ness)?\b",
    r"\bsuicid(?:e|al)\b",
)

OBSERVATION_PRIORITY = {
    "participant_speaking_ratio": 20,
    "transcribed_speech_ratio": 20,
    "voiced_activity_ratio": 19,
    "facial_action_variability": 18,
    "head_motion_variability": 17,
    "gaze_direction_variability": 16,
    "pitch_variability": 15,
    "loudness_variability": 14,
    "facial_landmark_shape_change": 13,
    "facial_shape_variability": 12,
}


def format_time(seconds: float) -> str:
    milliseconds = max(0, int(round(float(seconds) * 1000.0)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def _safe_token(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value).strip()).strip("_") or "unknown"


def _number(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "null"
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def _salience(observation: Observation) -> float:
    if observation.percentile is None:
        return -1.0
    return abs(float(observation.percentile) - 50.0)


def select_observations(
    observations: Iterable[Observation], max_observations: int
) -> list[Observation]:
    items = list(observations)
    if max_observations <= 0 or len(items) <= max_observations:
        return items
    reliability = {"direct": 3, "derived": 2, "weak": 1, "latent": 0}
    ranked = sorted(
        enumerate(items),
        key=lambda pair: (
            OBSERVATION_PRIORITY.get(pair[1].name, 0),
            _salience(pair[1]),
            reliability[pair[1].semantic_reliability],
            -pair[0],
        ),
        reverse=True,
    )[:max_observations]
    selected_indices = {index for index, _ in ranked}
    return [item for index, item in enumerate(items) if index in selected_indices]


def assert_semantically_safe(text: str) -> None:
    for pattern in FORBIDDEN_INTERPRETIVE_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            raise ValueError(f"Rendered text contains a forbidden interpretive term: {pattern}")


@dataclass(frozen=True)
class EvidenceRenderer:
    max_observations_per_unit: int = 8

    def render_dsl(self, unit: EvidenceUnit) -> str:
        unit.validate()
        start = format_time(float(unit.time_range["start_sec"]))
        end = format_time(float(unit.time_range["end_sec"]))
        source_id = _safe_token(unit.source.get("source_id", "unknown"))
        header = (
            f"[{_safe_token(unit.dataset)}|{start}-{end}|{unit.modality.upper()}|{source_id}|"
            f"id={unit.evidence_id}]"
        )
        availability = _safe_token(unit.availability.get("status", "unavailable"))
        quality = _safe_token(unit.quality.get("level", "unknown"))
        fields = [f"availability={availability}", f"quality={quality}"]
        valid_ratio = unit.quality.get("valid_ratio")
        if isinstance(valid_ratio, (float, int)) and math.isfinite(float(valid_ratio)):
            fields.append(f"valid_ratio={_number(float(valid_ratio))}")
        if unit.availability.get("privacy_scrubbed"):
            fields.append("privacy_scrubbed=true")
        reasons = unit.availability.get("missing_reasons") or []
        if reasons:
            fields.append("missing=" + ",".join(_safe_token(reason) for reason in reasons))

        selected = select_observations(unit.observations, self.max_observations_per_unit)
        for observation in selected:
            annotations = [f"category={observation.category}"]
            if observation.percentile is not None:
                annotations.append(f"p={int(round(observation.percentile))}")
            if observation.robust_z is not None:
                annotations.append(f"z={_number(observation.robust_z, 2)}")
            if observation.subject_robust_z is not None:
                annotations.append(f"subject_z={_number(observation.subject_robust_z, 2)}")
            annotations.append(f"reliability={observation.semantic_reliability}")
            fields.append(
                f"{observation.name}={_number(observation.value)}:{_safe_token(observation.unit)}"
                f"[{','.join(annotations)}]"
            )
        if len(selected) < len(unit.observations):
            fields.append(f"observations_shown={len(selected)}/{len(unit.observations)}")
        rendered = header + " " + "; ".join(fields) + "."
        assert_semantically_safe(rendered)
        return rendered

    def render_text(self, unit: EvidenceUnit) -> str:
        unit.validate()
        start = format_time(float(unit.time_range["start_sec"]))
        end = format_time(float(unit.time_range["end_sec"]))
        source = str(unit.source.get("extractor", unit.source.get("source_id", "feature")))
        status = str(unit.availability.get("status", "unavailable"))
        opening = (
            f"Evidence {unit.evidence_id}, from {start} to {end}: "
            f"the {unit.modality} measurements from {source} were {status}."
        )
        sentences = [opening]
        reasons = unit.availability.get("missing_reasons") or []
        if reasons:
            reason_text = [REASON_TEXT.get(str(reason), str(reason).replace("_", " ")) for reason in reasons]
            sentences.append("Availability note: " + "; ".join(reason_text) + ".")

        selected = select_observations(unit.observations, self.max_observations_per_unit)
        descriptions: list[str] = []
        for observation in selected:
            name = DISPLAY_NAMES.get(observation.name, observation.name.replace("_", " "))
            if observation.category == "unreferenced":
                phrase = f"{name} had no fitted training reference"
            else:
                category = CATEGORY_TEXT[observation.category]
                phrase = f"{name} was {category} the source-specific training reference"
                if observation.percentile is not None:
                    phrase += f" (percentile {int(round(observation.percentile))})"
            if observation.unit == "ratio":
                phrase += f", with a measured value of {100.0 * observation.value:.1f}%"
            if observation.semantic_reliability in {"weak", "latent"}:
                phrase += f" [{observation.semantic_reliability}-reliability pattern]"
            descriptions.append(phrase)
        if descriptions:
            sentences.append("Observed measurements: " + "; ".join(descriptions) + ".")
        elif status == "unavailable":
            sentences.append("No measurement is substituted for the unavailable source.")

        valid_ratio = unit.quality.get("valid_ratio")
        quality = str(unit.quality.get("level", "unknown")).replace("_", " ")
        if isinstance(valid_ratio, (float, int)) and math.isfinite(float(valid_ratio)):
            sentences.append(
                f"Source quality was {quality}, with {100.0 * float(valid_ratio):.1f}% valid coverage."
            )
        else:
            sentences.append(f"Source quality was {quality}.")
        sentences.append("These statements describe measurements only; no psychological state is inferred.")
        rendered = " ".join(sentences)
        assert_semantically_safe(rendered)
        return rendered
