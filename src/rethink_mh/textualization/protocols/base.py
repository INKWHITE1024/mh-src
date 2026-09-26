"""Abstract interface for evidence text protocols and their artifacts."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..schema import EvidenceUnit


@dataclass(frozen=True)
class ProtocolArtifacts:
    """All model-facing files produced through the protocol seam."""

    files: dict[str, str]
    primary_input_file: str
    atomic_unit_count: int
    segment_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.files:
            raise ValueError("A protocol must produce at least one file")
        if self.primary_input_file not in self.files:
            raise ValueError("primary_input_file must name one of the protocol files")
        if self.atomic_unit_count < 0 or self.segment_count < 0:
            raise ValueError("Protocol counts cannot be negative")
        for name, content in self.files.items():
            if "/" in name or "\\" in name:
                raise ValueError("Protocol filenames must be basenames")
            if not isinstance(content, str):
                raise TypeError(f"Protocol file {name} must contain text")


class EvidenceTextProtocol(ABC):
    """Small interface shared by versioned evidence text adapters."""

    version: str

    @property
    @abstractmethod
    def expected_filenames(self) -> tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def compile_session(self, units: Sequence[EvidenceUnit]) -> ProtocolArtifacts:
        raise NotImplementedError

