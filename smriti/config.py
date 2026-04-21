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
    # Over-fetch multiplier for hybrid retrieval (v0.7).
    # Each retriever returns top_k * fetch_multiplier candidates before RRF
    # fusion. Prevents the "retriever starvation" failure where a relevant
    # doc is rank 25 in one retriever but rank 1 in another — with a flat
    # top_k cap it would never be fused. Cost: a small overhead on the
    # hottest queries in exchange for materially higher recall.
    fetch_multiplier: int = 5
    # Include the knowledge graph retriever in hybrid search. v0.6 included
    # it by default, but on entity-heavy corpora (LongMemEval, long chat
    # histories) the co-occurrence graph injects topically-unrelated hits
    # with equal RRF weight. Off by default in v0.7; opt in via mode="graph"
    # or by flipping this flag.
    use_graph_in_hybrid: bool = False
    # Cross-encoder reranker: after hybrid fusion, re-score the top
    # (top_k * rerank_fetch_multiplier) with a CE model and keep top_k.
    # Only applied when Memory(reranker=...) is provided.
    rerank_fetch_multiplier: int = 10
