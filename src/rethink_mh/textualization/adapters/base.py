"""Abstract base class shared by dataset adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from ..config import TextualizationConfig
from ..schema import RawEvidenceUnit
from ..transcript import TranscriptUtterance


class DatasetAdapter(ABC):
    def __init__(self, root: str | Path, config: TextualizationConfig):
        self.root = Path(root).expanduser().resolve()
        self.config = config
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

    @abstractmethod
    def list_session_ids(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def load_session(self, session_id: str) -> list[RawEvidenceUnit]:
        raise NotImplementedError

    @abstractmethod
    def audit_session(self, session_id: str) -> dict[str, Any]:
        raise NotImplementedError

    def load_transcript(self, session_id: str) -> list[TranscriptUtterance]:
        del session_id
        return []

    def source_config(self, modality: str, defaults: dict[str, Any]) -> dict[str, Any]:
        configured = self.config.get("sources", {}).get(modality, {})
        output = dict(defaults)
        output.update(configured)
        return output

    @staticmethod
    def quality_level(valid_ratio: float, unavailable: bool = False) -> str:
        if unavailable:
            return "unavailable"
        if not np.isfinite(valid_ratio) or valid_ratio < 0.30:
            return "low_quality"
        if valid_ratio < 0.80:
            return "partial"
        return "good"

    @staticmethod
    def availability(valid_ratio: float, missing_reasons: list[str]) -> tuple[str, tuple[str, ...]]:
        reasons = tuple(sorted(set(filter(None, missing_reasons))))
        if valid_ratio <= 0:
            return "unavailable", reasons or ("no_valid_measurements",)
        if valid_ratio < 0.80 or reasons:
            return "partial", reasons
        return "present", ()
