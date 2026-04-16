"""Hybrid search with Reciprocal Rank Fusion (RRF).

Combines semantic (ChromaDB) + keyword (BM25/FTS5) results into a single
ranked list. Inspired by Hindsight's TEMPR retrieval approach.

RRF formula: score(d) = Σ  1 / (k + rank_i(d))
where k = 60 (standard) and rank_i is the rank from retriever i.
"""

from __future__ import annotations

from smriti.types import MemoryEntry, SearchResult


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    list_names: list[str],
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[tuple[str, float, list[str]]]:
    """Fuse multiple ranked lists using RRF.

    Args:
        ranked_lists: List of [(id, score), ...] from each retriever.
        list_names: Name of each retriever (e.g. ["semantic", "bm25"]).
        k: RRF constant (default 60).
        weights: Optional per-source weights (e.g. {"semantic": 1.5, "bm25": 1.0}).

    Returns:
        [(id, rrf_score, [source_names]), ...] sorted by score descending.
    """
    scores: dict[str, float] = {}
    sources: dict[str, list[str]] = {}

    for ranked, name in zip(ranked_lists, list_names):
        w = (weights or {}).get(name, 1.0)
        for rank, (doc_id, _original_score) in enumerate(ranked):
            rrf_score = w / (k + rank + 1)  # rank is 0-indexed
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score
            if doc_id not in sources:
                sources[doc_id] = []
            sources[doc_id].append(name)

    # Sort by RRF score descending
    combined = [
        (doc_id, score, sources[doc_id])
        for doc_id, score in scores.items()
    ]
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined


def build_results(
    fused: list[tuple[str, float, list[str]]],
    entries: dict[str, MemoryEntry],
    top_k: int,
) -> list[SearchResult]:
    """Convert fused RRF output into SearchResult objects."""
    results = []
    for rank, (doc_id, score, srcs) in enumerate(fused[:top_k]):
        if doc_id in entries:
            results.append(
                SearchResult(
                    entry=entries[doc_id],
                    score=score,
                    rank=rank + 1,
                    sources=srcs,
                )
            )
    return results
