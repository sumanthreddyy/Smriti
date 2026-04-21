"""Main Memory class — the public API for Smriti.

Usage:
    from smriti import Memory

    mem = Memory()                          # ~/.smriti/
    mem = Memory(path="/tmp/my_memory")     # custom path

    mem.add("I prefer Postgres over MySQL")
    mem.add("API key rotates every 90 days", memory_type="fact")

    results = mem.search("what database do I like?")
    for r in results:
        print(f"[{r.score:.3f}] {r.entry.content}")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from smriti.config import SmritiConfig
from smriti.graph import KnowledgeGraph
from smriti.search import build_results, reciprocal_rank_fusion
from smriti.store import Store
from smriti.types import MemoryEntry, SearchResult
from smriti.vectors import HashEmbedding, VectorStore


class Memory:
    """The universal memory layer.

    Stores memories in SQLite (structured + BM25) and ChromaDB (semantic).
    Search fuses both via Reciprocal Rank Fusion.
    Never deletes data — only archives.
    """

    def __init__(
        self,
        path: str | None = None,
        config: SmritiConfig | None = None,
        embedding_fn: object | None = None,
        reranker: object | None = None,
    ) -> None:
        self._config = config or SmritiConfig()
        if path:
            self._config = self._config.model_copy(update={"path": path})

        # Auto-tune retrieval defaults for HashEmbedding. The v0.7 defaults
        # (fetch_multiplier=5, use_graph_in_hybrid=False) were measured on
        # MiniLM and regressed HashEmbedding on LongMemEval-S from 0.77 → 0.60
        # R@5 — over-fetch adds noise to bag-of-words signatures, and the
        # co-occurrence graph was doing real work when the embedding has no
        # real semantics. Only override fields the caller did not set.
        if isinstance(embedding_fn, HashEmbedding):
            set_fields = self._config.model_fields_set
            overrides: dict[str, Any] = {}
            if "fetch_multiplier" not in set_fields:
                overrides["fetch_multiplier"] = 1
            if "use_graph_in_hybrid" not in set_fields:
                overrides["use_graph_in_hybrid"] = True
            if overrides:
                self._config = self._config.model_copy(update=overrides)

        base = Path(self._config.path)
        base.mkdir(parents=True, exist_ok=True)

        self._store = Store(base / "smriti.db")
        self._vectors = VectorStore(
            persist_dir=str(base / "chroma"),
            collection_name=self._config.collection_name,
            embedding_fn=embedding_fn,
            key_expansion=self._config.key_expansion,
        )
        self._graph = KnowledgeGraph(base / "smriti.db")
        # Optional callable: (query, [(id, content), ...]) -> [(id, score), ...]
        # Scores are arbitrary (higher = better); we re-sort and truncate.
        self._reranker = reranker

    def add(
        self,
        content: str,
        *,
        memory_type: str = "fact",
        source: str = "user",
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
        valid_from: str | None = None,
        entities: list[str] | None = None,
        confidence: float = 1.0,
    ) -> MemoryEntry:
        """Add a memory. Returns the created entry.

        Args:
            memory_type: Built-in types are "fact", "belief", "opinion",
                "observation". Custom strings are also accepted.
        """
        entry = MemoryEntry(
            content=content,
            memory_type=memory_type,
            source=source,
            metadata=metadata or {},
            session_id=session_id,
            valid_from=valid_from,
            entities=entities or [],
            confidence=confidence,
        )
        self._store.add(entry)
        self._vectors.add(entry)

        # Extract and index entities in knowledge graph (v0.3)
        extracted = self._graph.add_memory(entry)
        if extracted and not entry.entities:
            entry.entities = extracted
            self._store.update(entry.id, entities=extracted)

        return entry

    def add_entry(self, entry: MemoryEntry) -> MemoryEntry:
        """Add a pre-built MemoryEntry directly."""
        self._store.add(entry)
        self._vectors.add(entry)
        return entry

    def get(self, memory_id: str) -> MemoryEntry | None:
        """Get a single memory by ID."""
        entry = self._store.get(memory_id)
        if entry and not entry.archived:
            self._store.record_access(memory_id)
        return entry

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        include_archived: bool = False,
        mode: str = "hybrid",
    ) -> list[SearchResult]:
        """Search memories with hybrid retrieval (semantic + BM25 + RRF).

        Args:
            query: Natural language search query.
            top_k: Max results to return.
            include_archived: Include soft-deleted memories.
            mode: "hybrid" (default), "semantic", "bm25", or "graph".

        Retrieval pipeline (v0.7):
            1. Each retriever returns `top_k * fetch_multiplier` candidates
               (prevents RRF starvation — see SmritiConfig.fetch_multiplier).
            2. Candidates are fused via weighted Reciprocal Rank Fusion.
            3. If a cross-encoder `reranker` is attached, the fused top
               (top_k * rerank_fetch_multiplier) are re-scored by content
               and re-sorted before truncation.
            4. Truncated to `top_k` and returned as SearchResult objects.

        Returns:
            Ranked list of SearchResult objects.
        """
        _VALID_MODES = {"hybrid", "semantic", "bm25", "graph"}
        if mode not in _VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Choose from: {_VALID_MODES}")

        k = top_k or self._config.default_top_k
        # Over-fetch from each retriever so RRF can actually fuse.
        fetch_k = max(k, k * self._config.fetch_multiplier)
        # Include graph in hybrid only when explicitly enabled.
        use_graph = (mode == "graph") or (
            mode == "hybrid" and self._config.use_graph_in_hybrid
        )

        # Fetch from each retriever
        ranked_lists: list[list[tuple[str, float]]] = []
        list_names: list[str] = []

        if mode in ("hybrid", "semantic"):
            semantic = self._vectors.search(
                query, top_k=fetch_k, include_archived=include_archived
            )
            ranked_lists.append(semantic)
            list_names.append("semantic")

        if mode in ("hybrid", "bm25"):
            bm25 = self._store.bm25_search(
                query, top_k=fetch_k, include_archived=include_archived
            )
            ranked_lists.append(bm25)
            list_names.append("bm25")

        if use_graph:
            graph_results = self._graph.search(query, top_k=fetch_k)
            if graph_results:
                ranked_lists.append(graph_results)
                list_names.append("graph")

        if not ranked_lists:
            return []

        # Single mode — skip RRF
        if len(ranked_lists) == 1:
            candidates = [
                (id_, score, [list_names[0]]) for id_, score in ranked_lists[0]
            ]
        else:
            # Hybrid — fuse with weighted RRF.
            weights = {
                "semantic": self._config.semantic_weight,
                "bm25": self._config.bm25_weight,
                "graph": self._config.graph_weight,
            }
            candidates = reciprocal_rank_fusion(
                ranked_lists, list_names, k=self._config.rrf_k, weights=weights
            )

        # Optional cross-encoder rerank on a wider slice before truncation.
        candidates = self._maybe_rerank(query, candidates, k)

        # Fetch full entries and materialize results.
        all_ids = [id_ for id_, _, _ in candidates[:k]]
        entries = {e.id: e for e in self._store.get_many(all_ids)}
        if not include_archived:
            entries = {i: e for i, e in entries.items() if not e.archived}

        return build_results(candidates, entries, k)

    def _maybe_rerank(
        self,
        query: str,
        candidates: list[tuple[str, float, list[str]]],
        top_k: int,
    ) -> list[tuple[str, float, list[str]]]:
        """Apply the attached cross-encoder reranker, if any.

        Only the top `top_k * rerank_fetch_multiplier` candidates are scored
        to bound CE latency (CE is ~1ms/pair on CPU with MiniLM-L6).
        """
        if self._reranker is None or not candidates:
            return candidates

        rerank_pool = top_k * self._config.rerank_fetch_multiplier
        head = candidates[:rerank_pool]
        tail = candidates[rerank_pool:]

        ids = [cid for cid, _, _ in head]
        id_to_entry = {e.id: e for e in self._store.get_many(ids)}
        pairs = [
            (cid, id_to_entry[cid].content)
            for cid, _, _ in head
            if cid in id_to_entry
        ]
        if not pairs:
            return candidates

        try:
            ce_scores = self._reranker(query, pairs)  # [(id, score), ...]
        except Exception:
            # Reranker must never take down a query — fall back to RRF order.
            return candidates

        ce_by_id = {cid: s for cid, s in ce_scores}
        # Merge CE score back into the candidate triple, preserving sources.
        rescored: list[tuple[str, float, list[str]]] = []
        for cid, rrf_score, srcs in head:
            if cid in ce_by_id:
                rescored.append((cid, float(ce_by_id[cid]), srcs + ["rerank"]))
            else:
                rescored.append((cid, rrf_score, srcs))
        rescored.sort(key=lambda x: x[1], reverse=True)
        return rescored + tail

    def update(
        self, memory_id: str, content: str, *, keep_original: bool = True
    ) -> MemoryEntry | None:
        """Update a memory's content.

        If keep_original=True (default), the old entry is archived and a new
        entry is created with superseded_by linking. Nothing is ever lost.
        """
        old = self._store.get(memory_id)
        if not old:
            return None

        if keep_original:
            # Create new version
            new_entry = MemoryEntry(
                content=content,
                memory_type=old.memory_type,
                source=old.source,
                metadata=old.metadata,
                session_id=old.session_id,
                valid_from=old.valid_from,
                entities=old.entities,
                linked_ids=old.linked_ids,
                version=old.version + 1,
            )
            # Archive old, link to new
            self._store.update(
                memory_id, archived=True, superseded_by=new_entry.id
            )
            self._vectors.update_metadata(memory_id, archived=True)
            # Store new
            self._store.add(new_entry)
            self._vectors.add(new_entry)
            return new_entry
        else:
            # In-place update (still non-lossy — SQLite WAL preserves history)
            self._store.update(memory_id, content=content)
            # Re-embed
            updated = self._store.get(memory_id)
            if updated:
                self._vectors.add(updated)
            return updated

    def delete(self, memory_id: str) -> bool:
        """Soft-delete (archive). Data is NEVER removed.

        Use this when you want to hide a memory from search but keep it
        recoverable. The original row stays in SQLite forever.
        """
        success = self._store.archive(memory_id)
        if success:
            self._vectors.update_metadata(memory_id, archived=True)
        return success

    def count(self, include_archived: bool = False) -> int:
        """Count memories."""
        return self._store.count(include_archived=include_archived)

    def list(
        self, *, limit: int = 100, include_archived: bool = False
    ) -> list[MemoryEntry]:
        """List memories, newest first."""
        if include_archived:
            return self._store.list_all(limit=limit)
        return self._store.list_active(limit=limit)

    @property
    def path(self) -> str:
        return self._config.path

    # ── Temporal queries (v0.2) ─────────────────────────────────────────

    def valid_at(self, date_str: str, *, limit: int = 1000) -> list[MemoryEntry]:
        """Get memories valid at a specific date.

        Returns memories where valid_from <= date <= valid_to.
        Memories with no temporal bounds are always included.
        """
        return self._store.valid_at(date_str, limit=limit)

    def history(self, memory_id: str) -> list[MemoryEntry]:
        """Get the full evolution chain of a memory (oldest→newest).

        Follows superseded_by links to trace how a memory evolved.
        """
        return self._store.get_chain(memory_id)

    # ── Knowledge graph (v0.3) ───────────────────────────────────────

    def related(self, memory_id: str) -> list[MemoryEntry]:
        """Find memories that share entities with the given memory."""
        related_ids = self._graph.get_related(memory_id)
        return self._store.get_many(related_ids)

    def entities(self, memory_id: str) -> list[str]:
        """Get entities extracted from a memory."""
        return self._graph.get_entities(memory_id)

    # ── Memory networks (v0.4) ──────────────────────────────────────

    def by_type(self, memory_type: str, *, limit: int = 100) -> list[MemoryEntry]:
        """Get memories filtered by type (fact/belief/opinion/observation)."""
        return self._store.by_type(memory_type, limit=limit)

    # ── Evolution & linking (v0.5) ───────────────────────────────────

    def link(self, id1: str, id2: str) -> bool:
        """Create a bidirectional link between two memories."""
        return self._store.add_link(id1, id2)

    def get_linked(self, memory_id: str) -> list[MemoryEntry]:
        """Get all memories linked to the given memory."""
        entry = self._store.get(memory_id)
        if not entry or not entry.linked_ids:
            return []
        return self._store.get_many(entry.linked_ids)

    def close(self) -> None:
        self._store.close()
        self._graph.close()

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Memory(path='{self._config.path}', count={self.count()})"
