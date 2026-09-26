"""Canonical slot ontology and projection of source observations onto comparable slots."""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .schema import EvidenceUnit, Observation


ONTOLOGY_VERSION = "1.0.0"

Modality = Literal["audio", "visual"]
SlotStatus = Literal["measured", "unavailable", "not_measured"]
MappingQuality = Literal[
    "directly_aligned",
    "semantic_proxy",
    "provisional_mapping",
]


@dataclass(frozen=True)
class CanonicalSlotDefinition:
    slot_id: str
    label: str
    modality: Modality
    description: str

    def validate(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.slot_id):
            raise ValueError(f"Invalid canonical slot ID: {self.slot_id!r}")
        if self.modality not in {"audio", "visual"}:
            raise ValueError(f"Invalid canonical slot modality: {self.modality!r}")
        if not self.label or not self.description:
            raise ValueError("Canonical slot labels and descriptions cannot be empty")

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return {
            "slot_id": self.slot_id,
            "label": self.label,
            "modality": self.modality,
            "description": self.description,
        }


@dataclass(frozen=True)
class SlotBinding:
    observation_name: str
    measurement_name: str
    mapping_quality: MappingQuality
    note: str = ""

    def validate(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.observation_name):
            raise ValueError(
                f"Invalid source observation name: {self.observation_name!r}"
            )
        if not self.measurement_name:
            raise ValueError("measurement_name cannot be empty")
        if self.mapping_quality not in {
            "directly_aligned",
            "semantic_proxy",
            "provisional_mapping",
        }:
            raise ValueError(f"Invalid mapping quality: {self.mapping_quality!r}")

    def to_dict(self) -> dict[str, str]:
        self.validate()
        output = {
            "observation_name": self.observation_name,
            "measurement_name": self.measurement_name,
            "mapping_quality": self.mapping_quality,
        }
        if self.note:
            output["note"] = self.note
        return output


@dataclass(frozen=True)
class SourceProfile:
    """Declarative adapter from one released extractor to the fixed slot ontology."""

    source_id: str
    modality: Modality
    slot_bindings: tuple[tuple[str, tuple[SlotBinding, ...]], ...]
    note: str

    def validate(
        self, definitions: Mapping[str, CanonicalSlotDefinition]
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+", self.source_id):
            raise ValueError(f"Invalid source profile ID: {self.source_id!r}")
        if self.modality not in {"audio", "visual"}:
            raise ValueError(f"Invalid source profile modality: {self.modality!r}")
        seen: set[str] = set()
        for slot_id, bindings in self.slot_bindings:
            if slot_id in seen:
                raise ValueError(
                    f"Duplicate slot {slot_id!r} in source profile {self.source_id}"
                )
            seen.add(slot_id)
            definition = definitions.get(slot_id)
            if definition is None:
                raise ValueError(
                    f"Unknown slot {slot_id!r} in source profile {self.source_id}"
                )
            if definition.modality != self.modality:
                raise ValueError(
                    f"Slot {slot_id!r} does not belong to {self.modality}"
                )
            if not bindings:
                raise ValueError(
                    f"Supported slot {slot_id!r} must have at least one binding"
                )
            observation_names: set[str] = set()
            for binding in bindings:
                binding.validate()
                if binding.observation_name in observation_names:
                    raise ValueError(
                        f"Duplicate observation {binding.observation_name!r} in "
                        f"{self.source_id}:{slot_id}"
                    )
                observation_names.add(binding.observation_name)
        if not self.note:
            raise ValueError("Source profile note cannot be empty")

    def bindings_for(self, slot_id: str) -> tuple[SlotBinding, ...]:
        return next(
            (bindings for key, bindings in self.slot_bindings if key == slot_id),
            (),
        )

    @property
    def bound_observation_names(self) -> frozenset[str]:
        return frozenset(
            binding.observation_name
            for _, bindings in self.slot_bindings
            for binding in bindings
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "modality": self.modality,
            "note": self.note,
            "slots": {
                slot_id: [binding.to_dict() for binding in bindings]
                for slot_id, bindings in self.slot_bindings
            },
        }


@dataclass(frozen=True)
class ProjectedSlot:
    definition: CanonicalSlotDefinition
    status: SlotStatus
    observation: Observation | None
    binding: SlotBinding | None
    reason: str

    def validate(self) -> None:
        self.definition.validate()
        if self.status not in {"measured", "unavailable", "not_measured"}:
            raise ValueError(f"Invalid slot status: {self.status!r}")
        if self.status == "measured":
            if self.observation is None or self.binding is None:
                raise ValueError("Measured slots require an observation and binding")
            if self.observation.name != self.binding.observation_name:
                raise ValueError("Projected observation does not match its binding")
        elif self.observation is not None:
            raise ValueError("Unmeasured slots cannot contain an observation")
        if not self.reason:
            raise ValueError("Every projected slot must state its status reason")


@dataclass(frozen=True)
class CanonicalProjection:
    source_id: str
    modality: Modality
    slots: tuple[ProjectedSlot, ...]
    extensions: tuple[Observation, ...]

    def validate(
        self, expected_definitions: Sequence[CanonicalSlotDefinition]
    ) -> None:
        expected = [item.slot_id for item in expected_definitions]
        observed = [item.definition.slot_id for item in self.slots]
        if observed != expected:
            raise ValueError(
                f"Canonical slots must have fixed order {expected}, got {observed}"
            )
        for slot in self.slots:
            slot.validate()


AUDIO_CORE_SLOTS = (
    CanonicalSlotDefinition(
        "speech_activity",
        "speech activity",
        "audio",
        "How much speech-like or participant-aligned activity is present.",
    ),
    CanonicalSlotDefinition(
        "pitch_level",
        "pitch level",
        "audio",
        "The central fundamental-frequency level during measured voiced activity.",
    ),
    CanonicalSlotDefinition(
        "pitch_variability",
        "pitch variability",
        "audio",
        "Variation in fundamental frequency during measured voiced activity.",
    ),
    CanonicalSlotDefinition(
        "loudness_level",
        "loudness level",
        "audio",
        "The central loudness level when the source exposes a validated measurement.",
    ),
    CanonicalSlotDefinition(
        "loudness_variability",
        "loudness variability",
        "audio",
        "Variation in loudness when the source exposes a validated measurement.",
    ),
)

VISUAL_CORE_SLOTS = (
    CanonicalSlotDefinition(
        "facial_activity",
        "facial activity",
        "visual",
        "Overall measured facial-action activation.",
    ),
    CanonicalSlotDefinition(
        "facial_movement",
        "facial movement",
        "visual",
        "Variation or frame-to-frame change in measured facial configuration.",
    ),
    CanonicalSlotDefinition(
        "head_movement",
        "head movement",
        "visual",
        "Variation in measured head rotation.",
    ),
    CanonicalSlotDefinition(
        "gaze_movement",
        "gaze movement",
        "visual",
        "Variation in measured gaze direction.",
    ),
    CanonicalSlotDefinition(
        "mouth_activity",
        "mouth activity",
        "visual",
        "Measured mouth opening, action, or configuration change.",
    ),
)

CORE_SLOTS = AUDIO_CORE_SLOTS + VISUAL_CORE_SLOTS
SLOTS_BY_ID = {slot.slot_id: slot for slot in CORE_SLOTS}


def slots_for_modality(modality: str) -> tuple[CanonicalSlotDefinition, ...]:
    if modality == "audio":
        return AUDIO_CORE_SLOTS
    if modality == "visual":
        return VISUAL_CORE_SLOTS
    raise ValueError(f"Unsupported modality: {modality!r}")


def _binding(
    observation_name: str,
    measurement_name: str,
    mapping_quality: MappingQuality,
    note: str = "",
) -> SlotBinding:
    return SlotBinding(
        observation_name=observation_name,
        measurement_name=measurement_name,
        mapping_quality=mapping_quality,
        note=note,
    )


def _profile(
    source_id: str,
    modality: Modality,
    slots: Mapping[str, Sequence[SlotBinding]],
    note: str,
) -> SourceProfile:
    order = {item.slot_id: index for index, item in enumerate(slots_for_modality(modality))}
    profile = SourceProfile(
        source_id=source_id,
        modality=modality,
        slot_bindings=tuple(
            (slot_id, tuple(bindings))
            for slot_id, bindings in sorted(
                slots.items(), key=lambda item: order.get(item[0], len(order))
            )
        ),
        note=note,
    )
    profile.validate(SLOTS_BY_ID)
    return profile


BUILTIN_SOURCE_PROFILES = (
    _profile(
        "daic_covarep_audio",
        "audio",
        {
            "speech_activity": (
                _binding(
                    "participant_speaking_ratio",
                    "participant speaking-time ratio from manual speaker annotation",
                    "directly_aligned",
                ),
                _binding(
                    "voiced_activity_ratio",
                    "acoustic voiced-frame ratio",
                    "semantic_proxy",
                ),
            ),
            "pitch_level": (
                _binding(
                    "pitch_level",
                    "COVAREP fundamental frequency converted to semitones",
                    "directly_aligned",
                ),
            ),
            "pitch_variability": (
                _binding(
                    "pitch_variability",
                    "interquartile range of voiced COVAREP pitch",
                    "directly_aligned",
                ),
            ),
        },
        "Participant-aligned COVAREP interview audio; loudness is not exposed as a "
        "validated core measurement.",
    ),
    _profile(
        "edaic_egemaps_audio",
        "audio",
        {
            "speech_activity": (
                _binding(
                    "transcribed_speech_ratio",
                    "mixed-speaker transcribed speech-time ratio",
                    "semantic_proxy",
                    "E-DAIC ASR intervals do not provide a reliable participant role.",
                ),
                _binding(
                    "participant_speaking_ratio",
                    "legacy transcript-aligned speech-time ratio with unverified speaker scope",
                    "semantic_proxy",
                ),
                _binding(
                    "voiced_activity_ratio",
                    "acoustic voiced-frame ratio",
                    "semantic_proxy",
                ),
            ),
            "pitch_level": (
                _binding(
                    "pitch_level",
                    "eGeMAPS voiced pitch in semitones",
                    "directly_aligned",
                ),
            ),
            "pitch_variability": (
                _binding(
                    "pitch_variability",
                    "interquartile range of voiced eGeMAPS pitch",
                    "directly_aligned",
                ),
            ),
            "loudness_level": (
                _binding(
                    "loudness_level",
                    "eGeMAPS loudness",
                    "directly_aligned",
                ),
            ),
            "loudness_variability": (
                _binding(
                    "loudness_variability",
                    "interquartile range of eGeMAPS loudness",
                    "directly_aligned",
                ),
            ),
        },
        "E-DAIC eGeMAPS audio aligned to ASR intervals with mixed speaker roles.",
    ),
    _profile(
        "dvlog_egemaps_audio",
        "audio",
        {
            "speech_activity": (
                _binding(
                    "voiced_activity_ratio",
                    "acoustic voiced-frame ratio",
                    "provisional_mapping",
                    "The released array lacks official column-name metadata.",
                ),
            ),
            "pitch_level": (
                _binding(
                    "pitch_level",
                    "reconstructed eGeMAPS voiced pitch in semitones",
                    "provisional_mapping",
                    "The released array lacks official column-name metadata.",
                ),
            ),
            "pitch_variability": (
                _binding(
                    "pitch_variability",
                    "reconstructed interquartile range of eGeMAPS pitch",
                    "provisional_mapping",
                    "The released array lacks official column-name metadata.",
                ),
            ),
            "loudness_level": (
                _binding(
                    "loudness_level",
                    "reconstructed eGeMAPS loudness",
                    "provisional_mapping",
                    "The released array lacks official column-name metadata.",
                ),
            ),
            "loudness_variability": (
                _binding(
                    "loudness_variability",
                    "reconstructed interquartile range of eGeMAPS loudness",
                    "provisional_mapping",
                    "The released array lacks official column-name metadata.",
                ),
            ),
        },
        "D-Vlog one-second acoustic arrays; the standard 25-column eGeMAPS order is "
        "reconstructed and remains provisional until verified against official metadata.",
    ),
    _profile(
        "daic_clnf_visual",
        "visual",
        {
            "facial_activity": (
                _binding(
                    "facial_action_activity",
                    "mean OpenFace facial action-unit intensity",
                    "directly_aligned",
                ),
            ),
            "facial_movement": (
                _binding(
                    "facial_action_variability",
                    "median variability across facial action units",
                    "directly_aligned",
                ),
            ),
            "head_movement": (
                _binding(
                    "head_motion_variability",
                    "variability of OpenFace head rotation",
                    "directly_aligned",
                ),
            ),
            "gaze_movement": (
                _binding(
                    "gaze_direction_variability",
                    "variability of OpenFace gaze direction",
                    "directly_aligned",
                ),
            ),
            "mouth_activity": (
                _binding(
                    "au25_intensity",
                    "OpenFace lips-part action-unit intensity",
                    "semantic_proxy",
                ),
            ),
        },
        "DAIC-WOZ CLNF/OpenFace action units, head pose, and gaze measurements.",
    ),
    _profile(
        "edaic_openface_visual",
        "visual",
        {
            "facial_activity": (
                _binding(
                    "facial_action_activity",
                    "mean OpenFace facial action-unit intensity",
                    "directly_aligned",
                ),
            ),
            "facial_movement": (
                _binding(
                    "facial_action_variability",
                    "median variability across facial action units",
                    "directly_aligned",
                ),
            ),
            "head_movement": (
                _binding(
                    "head_motion_variability",
                    "variability of OpenFace head rotation",
                    "directly_aligned",
                ),
            ),
            "gaze_movement": (
                _binding(
                    "gaze_direction_variability",
                    "variability of OpenFace gaze direction",
                    "directly_aligned",
                ),
            ),
            "mouth_activity": (
                _binding(
                    "au25_intensity",
                    "OpenFace lips-part action-unit intensity",
                    "semantic_proxy",
                ),
            ),
        },
        "E-DAIC OpenFace action units, head pose, and gaze measurements.",
    ),
    _profile(
        "dvlog_dlib_landmarks",
        "visual",
        {
            "facial_movement": (
                _binding(
                    "facial_landmark_shape_change",
                    "frame-to-frame change in normalized facial landmarks",
                    "semantic_proxy",
                    "Landmark change is not equivalent to action-unit activation.",
                ),
            ),
            "mouth_activity": (
                _binding(
                    "mouth_region_shape_variability",
                    "variability of normalized mouth-region landmarks",
                    "semantic_proxy",
                ),
                _binding(
                    "mouth_opening",
                    "normalized geometric mouth opening",
                    "semantic_proxy",
                ),
            ),
        },
        "D-Vlog release-normalized dlib landmarks; no validated action-unit, head-pose, "
        "or gaze measurement is claimed.",
    ),
)


def _configured_binding(value: Mapping[str, Any]) -> SlotBinding:
    return _binding(
        observation_name=str(value["observation_name"]),
        measurement_name=str(value["measurement_name"]),
        mapping_quality=str(value["mapping_quality"]),  # type: ignore[arg-type]
        note=str(value.get("note", "")),
    )


def source_profiles_from_config(
    values: Mapping[str, Any] | None,
) -> tuple[SourceProfile, ...]:
    if not values:
        return ()
    output: list[SourceProfile] = []
    for source_id, raw_profile in values.items():
        if not isinstance(raw_profile, Mapping):
            raise ValueError(f"Source profile {source_id!r} must be a mapping")
        modality = str(raw_profile.get("modality", ""))
        if modality not in {"audio", "visual"}:
            raise ValueError(
                f"Source profile {source_id!r} modality must be audio or visual"
            )
        raw_slots = raw_profile.get("slots", {})
        if not isinstance(raw_slots, Mapping):
            raise ValueError(f"Source profile {source_id!r} slots must be a mapping")
        bindings: dict[str, tuple[SlotBinding, ...]] = {}
        for slot_id, raw_bindings in raw_slots.items():
            items = (
                raw_bindings
                if isinstance(raw_bindings, Sequence)
                and not isinstance(raw_bindings, (str, bytes, Mapping))
                else [raw_bindings]
            )
            if not all(isinstance(item, Mapping) for item in items):
                raise ValueError(
                    f"Bindings for {source_id}:{slot_id} must be mappings"
                )
            bindings[str(slot_id)] = tuple(
                _configured_binding(item) for item in items
            )
        output.append(
            _profile(
                source_id=str(source_id),
                modality=modality,  # type: ignore[arg-type]
                slots=bindings,
                note=str(
                    raw_profile.get(
                        "note",
                        "User-supplied source profile; validate before experimental use.",
                    )
                ),
            )
        )
    return tuple(output)


def _extension_salience(observation: Observation) -> float:
    if observation.percentile is None:
        return -1.0
    return abs(float(observation.percentile) - 50.0)


class CanonicalSlotProjector:
    """Deep module that hides source-specific mappings behind one projection interface."""

    def __init__(
        self,
        source_profiles: Iterable[SourceProfile] = BUILTIN_SOURCE_PROFILES,
    ) -> None:
        profiles: dict[str, SourceProfile] = {}
        for profile in source_profiles:
            profile.validate(SLOTS_BY_ID)
            profiles[profile.source_id] = profile
        if not profiles:
            raise ValueError("At least one source profile is required")
        self._profiles = profiles

    @classmethod
    def with_overrides(
        cls,
        values: Mapping[str, Any] | None,
    ) -> "CanonicalSlotProjector":
        profiles = {item.source_id: item for item in BUILTIN_SOURCE_PROFILES}
        for profile in source_profiles_from_config(values):
            profiles[profile.source_id] = profile
        return cls(profiles.values())

    def project(
        self,
        unit: EvidenceUnit,
        extension_limit: int = 2,
    ) -> CanonicalProjection:
        if extension_limit < 0:
            raise ValueError("extension_limit cannot be negative")
        unit.validate()
        source_id = str(unit.source.get("source_id", ""))
        profile = self._profiles.get(source_id)
        if profile is None:
            raise ValueError(
                f"No native source profile is registered for {source_id!r}; add a "
                "protocol_native.source_profiles entry before compiling this source"
            )
        if profile.modality != unit.modality:
            raise ValueError(
                f"Source profile {source_id!r} is {profile.modality}, but unit "
                f"{unit.evidence_id} is {unit.modality}"
            )

        observations = {item.name: item for item in unit.observations}
        projected: list[ProjectedSlot] = []
        for definition in slots_for_modality(unit.modality):
            bindings = profile.bindings_for(definition.slot_id)
            if not bindings:
                projected.append(
                    ProjectedSlot(
                        definition=definition,
                        status="not_measured",
                        observation=None,
                        binding=None,
                        reason="this source has no validated measurement for the slot",
                    )
                )
                continue
            selected_binding = next(
                (
                    binding
                    for binding in bindings
                    if binding.observation_name in observations
                ),
                None,
            )
            if selected_binding is None:
                status = str(unit.availability.get("status", "unavailable"))
                reason = (
                    "the source window is unavailable"
                    if status == "unavailable"
                    else "the source supports this slot, but no valid measurement "
                    "passed the window checks"
                )
                projected.append(
                    ProjectedSlot(
                        definition=definition,
                        status="unavailable",
                        observation=None,
                        binding=bindings[0],
                        reason=reason,
                    )
                )
                continue
            projected.append(
                ProjectedSlot(
                    definition=definition,
                    status="measured",
                    observation=observations[selected_binding.observation_name],
                    binding=selected_binding,
                    reason="a validated source observation is available",
                )
            )

        candidates = [
            item
            for item in unit.observations
            if item.name not in profile.bound_observation_names
            and item.semantic_reliability in {"direct", "derived"}
        ]
        reliability = {"direct": 2, "derived": 1}
        extensions = tuple(
            sorted(
                candidates,
                key=lambda item: (
                    -reliability[item.semantic_reliability],
                    -_extension_salience(item),
                    item.name,
                ),
            )[:extension_limit]
        )
        result = CanonicalProjection(
            source_id=source_id,
            modality=unit.modality,
            slots=tuple(projected),
            extensions=extensions,
        )
        result.validate(slots_for_modality(unit.modality))
        return result

    def describe(self, source_ids: Iterable[str]) -> dict[str, Any]:
        requested = sorted({str(value) for value in source_ids})
        missing = [source_id for source_id in requested if source_id not in self._profiles]
        if missing:
            raise ValueError(f"No native source profiles are registered for: {missing}")
        return {
            "ontology_version": ONTOLOGY_VERSION,
            "core_slots": [slot.to_dict() for slot in CORE_SLOTS],
            "source_profiles": [
                self._profiles[source_id].to_dict() for source_id in requested
            ],
            "status_vocabulary": {
                "measured": "A validated observation is present.",
                "unavailable": (
                    "The source supports the slot, but this interval has no valid "
                    "measurement."
                ),
                "not_measured": (
                    "The source has no validated measurement for the slot; this is "
                    "not a measured zero."
                ),
            },
            "mapping_quality_vocabulary": {
                "directly_aligned": (
                    "The source measurement matches the canonical slot definition."
                ),
                "semantic_proxy": (
                    "The source measurement is related but not measurement-equivalent."
                ),
                "provisional_mapping": (
                    "The semantic or column mapping still requires source verification."
                ),
            },
        }


__all__ = [
    "AUDIO_CORE_SLOTS",
    "BUILTIN_SOURCE_PROFILES",
    "CORE_SLOTS",
    "CanonicalProjection",
    "CanonicalSlotDefinition",
    "CanonicalSlotProjector",
    "MappingQuality",
    "ONTOLOGY_VERSION",
    "ProjectedSlot",
    "SlotBinding",
    "SlotStatus",
    "SourceProfile",
    "VISUAL_CORE_SLOTS",
    "slots_for_modality",
    "source_profiles_from_config",
]
