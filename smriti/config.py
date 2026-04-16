"""Configuration for Smriti."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


def _default_path() -> str:
    return str(Path.home() / ".smriti")


class SmritiConfig(BaseModel):
    """Global config. Sane defaults, zero required setup."""

    path: str = Field(default_factory=_default_path)
    collection_name: str = "smriti_memories"
    # Search tuning
    semantic_weight: float = 1.0
    bm25_weight: float = 1.0
    default_top_k: int = 10
    # RRF constant (standard = 60)
    rrf_k: int = 60
    # Graph search weight (v0.3)
    graph_weight: float = 1.0
    # Key expansion: append extracted keywords to indexed text (experimental)
    key_expansion: bool = False
