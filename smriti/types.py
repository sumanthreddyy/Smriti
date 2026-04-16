"""Data models for Smriti memory entries and search results."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    """A single memory unit stored in Smriti.

    Fields are designed to support v0.1 (core) through v0.5 (evolution).
    Unused fields default to None and cost nothing until needed.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    content: str
    memory_type: str = "fact"  # fact | belief | opinion | observation (v0.4)
    source: str = "user"
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    # Temporal fields (v0.2) — present from day one so schema never migrates
    valid_from: str | None = None
    valid_to: str | None = None
    superseded_by: str | None = None
    # Non-lossy
    archived: bool = False
    # Flexible metadata
    metadata: dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = None
    # Graph fields (v0.3)
    entities: list[str] = Field(default_factory=list)
    # Evolution fields (v0.5)
    linked_ids: list[str] = Field(default_factory=list)
    version: int = 1
    # Classification fields (v0.4)
    confidence: float = 1.0  # 0.0–1.0
    # Access tracking (v0.5)
    access_count: int = 0
    last_accessed: str | None = None


class SearchResult(BaseModel):
    """A ranked search result combining multiple retrieval signals."""

    entry: MemoryEntry
    score: float  # 0.0-1.0, higher = better
    rank: int
    sources: list[str] = Field(default_factory=list)  # ["semantic", "bm25", "graph"]
