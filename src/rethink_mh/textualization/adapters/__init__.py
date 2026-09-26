"""Build the dataset adapter selected by configuration."""
from __future__ import annotations

from pathlib import Path

from ..config import TextualizationConfig
from .base import DatasetAdapter


def build_adapter(root: str | Path, config: TextualizationConfig) -> DatasetAdapter:
    if config.dataset_key == "daic_woz":
        from .daic_woz import DaicWozAdapter

        return DaicWozAdapter(root, config)
    if config.dataset_key == "e_daic":
        from .e_daic import EDaicAdapter

        return EDaicAdapter(root, config)
    if config.dataset_key == "d_vlog":
        from .d_vlog import DVlogAdapter

        return DVlogAdapter(root, config)
    raise ValueError(f"No adapter registered for {config.dataset_key}")


__all__ = ["DatasetAdapter", "build_adapter"]

