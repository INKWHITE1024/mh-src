"""Evidence text protocol registry and factory."""
from __future__ import annotations

from typing import Any

from ..config import TextualizationConfig
from .base import EvidenceTextProtocol, ProtocolArtifacts
from .native import NativeEvidenceProtocol


SUPPORTED_PROTOCOLS = ("native",)


def build_protocol(
    version: str,
    config: TextualizationConfig,
) -> EvidenceTextProtocol:
    normalized = version.strip().lower()
    if normalized == "native":
        values: dict[str, Any] = dict(config.get("protocol_native", {}))
        return NativeEvidenceProtocol.from_config(values)
    raise ValueError(
        f"Unsupported evidence text protocol {version!r}; choose one of: "
        f"{', '.join(SUPPORTED_PROTOCOLS)}"
    )


__all__ = [
    "EvidenceTextProtocol",
    "ProtocolArtifacts",
    "SUPPORTED_PROTOCOLS",
    "build_protocol",
]
