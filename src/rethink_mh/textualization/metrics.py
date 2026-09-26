"""Convert extractor feature columns into validated observations and derived behavioral metrics."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .schema import RawObservation


COVAREP_COLUMNS = (
    "F0",
    "VUV",
    "NAQ",
    "QOQ",
    "H1H2",
    "PSP",
    "MDQ",
    "peakSlope",
    "Rd",
    "Rd_conf",
    "creak",
    *(f"MCEP_{index}" for index in range(25)),
    *(f"HMPDM_{index}" for index in range(25)),
    *(f"HMPDD_{index}" for index in range(13)),
)

D_VLOG_EGEMAPS_COLUMNS = (
    "Loudness_sma3",
    "alphaRatio_sma3",
    "hammarbergIndex_sma3",
    "slope0-500_sma3",
    "slope500-1500_sma3",
    "spectralFlux_sma3",
    "mfcc1_sma3",
    "mfcc2_sma3",
    "mfcc3_sma3",
    "mfcc4_sma3",
    "F0semitoneFrom27.5Hz_sma3nz",
    "jitterLocal_sma3nz",
    "shimmerLocaldB_sma3nz",
    "HNRdBACF_sma3nz",
    "logRelF0-H1-H2_sma3nz",
    "logRelF0-H1-A3_sma3nz",
    "F1frequency_sma3nz",
    "F1bandwidth_sma3nz",
    "F1amplitudeLogRelF0_sma3nz",
    "F2frequency_sma3nz",
    "F2bandwidth_sma3nz",
    "F2amplitudeLogRelF0_sma3nz",
    "F3frequency_sma3nz",
    "F3bandwidth_sma3nz",
    "F3amplitudeLogRelF0_sma3nz",
)


def _observation(
    name: str,
    value: float,
    unit: str,
    reliability: str,
) -> RawObservation | None:
    value = float(value)
    if not np.isfinite(value):
        return None
    return RawObservation(
        name=name,
        value=value,
        unit=unit,
        semantic_reliability=reliability,  # type: ignore[arg-type]
    )


def _append(
    output: list[RawObservation],
    name: str,
    value: float,
    unit: str,
    reliability: str,
) -> None:
    item = _observation(name, value, unit, reliability)
    if item is not None:
        output.append(item)


def _iqr(values: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    return np.nanpercentile(values, 75, axis=axis) - np.nanpercentile(values, 25, axis=axis)


def _consecutive_change(values: np.ndarray, row_indices: np.ndarray | None = None) -> float:
    if values.ndim != 2 or values.shape[0] < 2:
        return float("nan")
    changes = np.linalg.norm(np.diff(values, axis=0), axis=1) / np.sqrt(values.shape[1])
    if row_indices is not None:
        adjacent = np.diff(np.asarray(row_indices)) == 1
        changes = changes[adjacent]
    changes = changes[np.isfinite(changes)]
    return float(np.median(changes)) if changes.size else float("nan")


def action_unit_observations(
    intensity: np.ndarray,
    intensity_names: Sequence[str],
    presence: np.ndarray | None = None,
    row_indices: np.ndarray | None = None,
) -> list[RawObservation]:
    output: list[RawObservation] = []
    if intensity.ndim != 2 or intensity.shape[0] == 0 or intensity.shape[1] == 0:
        return output
    _append(output, "facial_action_activity", np.nanmean(intensity), "activation", "derived")
    _append(
        output,
        "facial_action_variability",
        np.nanmedian(np.nanstd(intensity, axis=0)),
        "activation",
        "derived",
    )
    _append(
        output,
        "facial_action_change_magnitude",
        _consecutive_change(intensity, row_indices),
        "activation_per_frame",
        "derived",
    )
    if presence is not None and presence.size:
        _append(output, "facial_action_presence_ratio", np.nanmean(presence), "ratio", "derived")
    lookup = {name.strip().lower(): index for index, name in enumerate(intensity_names)}
    for au in ("AU04_r", "AU12_r", "AU15_r", "AU25_r"):
        index = lookup.get(au.lower())
        if index is not None:
            _append(output, f"{au[:4].lower()}_intensity", np.nanmean(intensity[:, index]), "activation", "direct")
    return output


def pose_observations(
    rotations: np.ndarray,
    times: np.ndarray,
) -> list[RawObservation]:
    output: list[RawObservation] = []
    if rotations.ndim != 2 or rotations.shape[0] == 0 or rotations.shape[1] < 3:
        return output
    _append(output, "head_pitch_angle", np.nanmedian(rotations[:, 0]), "radian", "direct")
    _append(output, "head_yaw_angle", np.nanmedian(rotations[:, 1]), "radian", "direct")
    _append(
        output,
        "head_motion_variability",
        np.sqrt(np.nanmean(np.nanvar(rotations[:, :3], axis=0))),
        "radian",
        "derived",
    )
    _append(
        output,
        "head_rotation_range",
        np.nanmedian(np.nanmax(rotations[:, :3], axis=0) - np.nanmin(rotations[:, :3], axis=0)),
        "radian",
        "derived",
    )
    if rotations.shape[0] >= 2:
        dt = np.diff(times)
        delta = np.linalg.norm(np.diff(rotations[:, :3], axis=0), axis=1)
        valid = np.isfinite(dt) & (dt > 1e-6) & np.isfinite(delta)
        if np.any(valid):
            _append(output, "head_rotation_speed", np.median(delta[valid] / dt[valid]), "radian_per_second", "derived")
    return output


def gaze_observations(
    angle_x: np.ndarray,
    angle_y: np.ndarray,
    left_vectors: np.ndarray | None = None,
    right_vectors: np.ndarray | None = None,
) -> list[RawObservation]:
    output: list[RawObservation] = []
    if angle_x.size:
        _append(output, "gaze_horizontal_angle", np.nanmedian(angle_x), "radian", "direct")
    if angle_y.size:
        _append(output, "gaze_vertical_angle", np.nanmedian(angle_y), "radian", "direct")
    if angle_x.size and angle_y.size:
        variability = np.sqrt(np.nanvar(angle_x) + np.nanvar(angle_y))
        _append(output, "gaze_direction_variability", variability, "radian", "derived")
    if left_vectors is not None and right_vectors is not None and left_vectors.size and right_vectors.size:
        left_norm = np.linalg.norm(left_vectors, axis=1)
        right_norm = np.linalg.norm(right_vectors, axis=1)
        valid = (left_norm > 1e-8) & (right_norm > 1e-8)
        if np.any(valid):
            cosine = np.sum(left_vectors[valid] * right_vectors[valid], axis=1) / (
                left_norm[valid] * right_norm[valid]
            )
            angle = np.arccos(np.clip(cosine, -1.0, 1.0))
            _append(output, "binocular_gaze_inconsistency", np.nanmedian(angle), "radian", "derived")
    return output


def egemaps_observations(
    values: np.ndarray,
    columns: Sequence[str],
    row_indices: np.ndarray | None = None,
) -> list[RawObservation]:
    output: list[RawObservation] = []
    if values.ndim != 2 or values.shape[0] == 0:
        return output
    lookup = {name: index for index, name in enumerate(columns)}

    def column(name: str) -> np.ndarray:
        index = lookup.get(name)
        return values[:, index] if index is not None else np.empty(0, dtype=float)

    loudness = column("Loudness_sma3")
    spectral_flux = column("spectralFlux_sma3")
    pitch = column("F0semitoneFrom27.5Hz_sma3nz")
    voiced = np.isfinite(pitch) & (pitch > 0) if pitch.size else np.zeros(values.shape[0], dtype=bool)

    if loudness.size:
        _append(output, "loudness_level", np.nanmedian(loudness), "openSMILE_loudness", "direct")
        _append(output, "loudness_variability", _iqr(loudness), "openSMILE_loudness", "derived")
    if spectral_flux.size:
        _append(output, "spectral_flux_level", np.nanmedian(spectral_flux), "openSMILE_spectral_flux", "direct")
        _append(output, "spectral_flux_variability", _iqr(spectral_flux), "openSMILE_spectral_flux", "derived")
    if pitch.size:
        _append(output, "voiced_activity_ratio", np.mean(voiced), "ratio", "derived")
        if np.any(voiced):
            _append(output, "pitch_level", np.nanmedian(pitch[voiced]), "semitone", "direct")
            _append(output, "pitch_variability", _iqr(pitch[voiced]), "semitone", "derived")

    for source_name, target_name, unit in (
        ("jitterLocal_sma3nz", "jitter_level", "relative"),
        ("shimmerLocaldB_sma3nz", "shimmer_level", "decibel"),
        ("HNRdBACF_sma3nz", "voice_harmonicity", "decibel"),
    ):
        data = column(source_name)
        if data.size and np.any(voiced):
            _append(output, target_name, np.nanmedian(data[voiced]), unit, "direct")

    mfcc_names = [f"mfcc{index}_sma3" for index in range(1, 5)]
    if all(name in lookup for name in mfcc_names):
        mfcc = values[:, [lookup[name] for name in mfcc_names]]
        _append(
            output,
            "cepstral_change_magnitude",
            _consecutive_change(mfcc, row_indices),
            "openSMILE_mfcc_per_step",
            "weak",
        )
    return output


def covarep_observations(
    values: np.ndarray,
    row_indices: np.ndarray | None = None,
) -> list[RawObservation]:
    output: list[RawObservation] = []
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != len(COVAREP_COLUMNS):
        return output
    lookup = {name: index for index, name in enumerate(COVAREP_COLUMNS)}
    f0 = values[:, lookup["F0"]]
    vuv = values[:, lookup["VUV"]]
    voiced = np.isfinite(f0) & np.isfinite(vuv) & (vuv >= 0.5) & (f0 > 0)
    _append(output, "voiced_activity_ratio", np.mean(voiced), "ratio", "direct")
    if np.any(voiced):
        pitch_semitone = 12.0 * np.log2(f0[voiced] / 27.5)
        _append(output, "pitch_level", np.nanmedian(pitch_semitone), "semitone", "direct")
        _append(output, "pitch_variability", _iqr(pitch_semitone), "semitone", "derived")
        for source_name, target_name, unit in (
            ("NAQ", "naq_level", "relative"),
            ("QOQ", "qoq_level", "relative"),
            ("H1H2", "h1h2_level", "decibel"),
            ("PSP", "psp_level", "relative"),
            ("MDQ", "mdq_level", "relative"),
            ("peakSlope", "peak_slope_level", "relative"),
            ("Rd", "rd_level", "relative"),
        ):
            data = values[:, lookup[source_name]]
            _append(output, target_name, np.nanmedian(data[voiced]), unit, "direct")
    creak = values[:, lookup["creak"]]
    _append(output, "creak_activity", np.nanmean(creak), "relative", "direct")
    mcep = values[:, [lookup[f"MCEP_{index}"] for index in range(25)]]
    _append(
        output,
        "spectral_change_magnitude",
        _consecutive_change(mcep, row_indices),
        "mcep_per_frame",
        "weak",
    )
    return output


def landmark_observations(
    normalized_landmarks: np.ndarray,
    row_indices: np.ndarray | None = None,
) -> list[RawObservation]:
    """Compile 68 x/y landmarks stored as x0..x67,y0..y67."""
    output: list[RawObservation] = []
    if normalized_landmarks.ndim != 2 or normalized_landmarks.shape[1] != 136:
        return output
    x = normalized_landmarks[:, :68]
    y = normalized_landmarks[:, 68:]
    points = np.stack((x, y), axis=-1)
    _append(
        output,
        "facial_landmark_shape_change",
        _consecutive_change(normalized_landmarks, row_indices),
        "normalized_distance_per_second",
        "derived",
    )
    _append(
        output,
        "facial_shape_variability",
        np.nanmedian(np.sqrt(np.nanvar(x, axis=0) + np.nanvar(y, axis=0))),
        "normalized_distance",
        "derived",
    )
    mouth = points[:, 48:68, :]
    eyes = points[:, 36:48, :]
    _append(
        output,
        "mouth_region_shape_variability",
        np.nanmedian(np.sqrt(np.nanvar(mouth[:, :, 0], axis=0) + np.nanvar(mouth[:, :, 1], axis=0))),
        "normalized_distance",
        "derived",
    )
    _append(
        output,
        "eye_region_shape_variability",
        np.nanmedian(np.sqrt(np.nanvar(eyes[:, :, 0], axis=0) + np.nanvar(eyes[:, :, 1], axis=0))),
        "normalized_distance",
        "derived",
    )
    if points.shape[0]:
        inner_pairs = ((61, 67), (62, 66), (63, 65))
        mouth_opening = np.mean(
            [np.linalg.norm(points[:, a, :] - points[:, b, :], axis=1) for a, b in inner_pairs],
            axis=0,
        )
        _append(output, "mouth_opening", np.nanmedian(mouth_opening), "normalized_distance", "derived")
    return output


def named_matrix(values: np.ndarray, columns: Sequence[str]) -> Mapping[str, np.ndarray]:
    if values.ndim != 2 or values.shape[1] != len(columns):
        raise ValueError(f"Expected {len(columns)} columns, got {values.shape}")
    return {name: values[:, index] for index, name in enumerate(columns)}

