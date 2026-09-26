"""Dataset-aware compilation of numeric multimodal features into evidence text."""

from .canonical_slots import CanonicalSlotProjector
from .config import TextualizationConfig, load_config
from .references import ReferenceSet, enrich_session
from .renderer import EvidenceRenderer
from .schema import EvidenceUnit, Observation, RawEvidenceUnit, RawObservation

__all__ = [
    "EvidenceRenderer",
    "CanonicalSlotProjector",
    "EvidenceUnit",
    "Observation",
    "RawEvidenceUnit",
    "RawObservation",
    "ReferenceSet",
    "TextualizationConfig",
    "enrich_session",
    "load_config",
]
