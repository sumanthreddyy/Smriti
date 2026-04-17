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
from smriti.vectors import VectorStore


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
    ) -> None:
        self._config = config or SmritiConfig()
        if path:
            self._config = self._config.model_copy(update={"path": path})

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

    def add(
        self,
        content: str,
        *,
        memory_type: str = "fact",
        source: str = "user",
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
        valid_from: str | None = None,
        valid_to: str | None = None,
        entities: list[str] | None = None,
        confidence: float = 1.0,
    ) -> MemoryEntry:
        """Add a memory. Returns the created entry.

        Args:
            memory_type: Built-in types are "fact", "belief", "opinion",
                "observation". Custom strings are also accepted.
            valid_from: ISO date string — memory is valid starting this date.
            valid_to: ISO date string — memory is valid until this date.
        """
        entry = MemoryEntry(
            content=content,
            memory_type=memory_type,
            source=source,
            metadata=metadata or {},
            session_id=session_id,
            valid_from=valid_from,
            valid_to=valid_to,
            entities=entities or [],
            confidence=confidence,
        )
        return self._ingest(entry)

    def add_entry(self, entry: MemoryEntry) -> MemoryEntry:
        """Add a pre-built MemoryEntry directly."""
        return self._ingest(entry)

    def _ingest(self, entry: MemoryEntry) -> MemoryEntry:
        """Unified write pipeline: store + vectors + graph."""
        self._store.add(entry)
        self._vectors.add(entry)

        # Extract and index entities in knowledge graph
        extracted = self._graph.add_memory(entry)
        if extracted and not entry.entities:
            entry.entities = extracted
            self._store.update(entry.id, entities=extracted)

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
            mode: "hybrid" (default), "semantic", or "bm25".

        Returns:
            Ranked list of SearchResult objects.
        """
        _VALID_MODES = {"hybrid", "semantic", "bm25", "graph"}
        if mode not in _VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Choose from: {_VALID_MODES}")

        k = top_k or self._config.default_top_k

        # Fetch from each retriever
        ranked_lists: list[list[tuple[str, float]]] = []
        list_names: list[str] = []

        if mode in ("hybrid", "semantic"):
            semantic = self._vectors.search(query, top_k=k, include_archived=include_archived)
            ranked_lists.append(semantic)
            list_names.append("semantic")

        if mode in ("hybrid", "bm25"):
            bm25 = self._store.bm25_search(query, top_k=k, include_archived=include_archived)
            ranked_lists.append(bm25)
            list_names.append("bm25")

        if mode in ("hybrid", "graph"):
            graph_results = self._graph.search(query, top_k=k)
            if graph_results:
                ranked_lists.append(graph_results)
                list_names.append("graph")

        if not ranked_lists:
            return []

        # Single mode — skip RRF
        if len(ranked_lists) == 1:
            all_ids = [id_ for id_, _ in ranked_lists[0][:k]]
            entries = {e.id: e for e in self._store.get_many(all_ids)}
            if not include_archived:
                entries = {i: e for i, e in entries.items() if not e.archived}
            return build_results(
                [(id_, score, [list_names[0]]) for id_, score in ranked_lists[0][:k]],
                entries,
                k,
            )

        # Hybrid — fuse with RRF
        weights = {
            "semantic": self._config.semantic_weight,
            "bm25": self._config.bm25_weight,
            "graph": self._config.graph_weight,
        }
        fused = reciprocal_rank_fusion(
            ranked_lists, list_names, k=self._config.rrf_k, weights=weights
        )

        # Fetch full entries from SQLite
        all_ids = [id_ for id_, _, _ in fused[:k]]
        entries = {e.id: e for e in self._store.get_many(all_ids)}
        if not include_archived:
            entries = {i: e for i, e in entries.items() if not e.archived}

        return build_results(fused, entries, k)

    def update(
        self, memory_id: str, content: str
    ) -> MemoryEntry | None:
        """Update a memory's content.

        The old entry is archived and a new entry is created with
        superseded_by linking. Nothing is ever lost.
        """
        old = self._store.get(memory_id)
        if not old:
            return None

        # Create new version — preserve ALL fields from the original
        new_entry = MemoryEntry(
            content=content,
            memory_type=old.memory_type,
            source=old.source,
            metadata=old.metadata,
            session_id=old.session_id,
            valid_from=old.valid_from,
            valid_to=old.valid_to,
            entities=old.entities,
            linked_ids=old.linked_ids,
            version=old.version + 1,
            confidence=old.confidence,
        )
        # Archive old, link to new
        self._store.update(
            memory_id, archived=True, superseded_by=new_entry.id
        )
        self._vectors.update_metadata(memory_id, archived=True)
        # Ingest new through unified pipeline (store + vectors + graph)
        return self._ingest(new_entry)

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
        """Find memories that share entities with the given memory.

        Only returns active (non-archived) memories.
        """
        related_ids = self._graph.get_related(memory_id)
        entries = self._store.get_many(related_ids)
        return [e for e in entries if not e.archived]

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
        """Get all active (non-archived) memories linked to the given memory."""
        entry = self._store.get(memory_id)
        if not entry or not entry.linked_ids:
            return []
        entries = self._store.get_many(entry.linked_ids)
        return [e for e in entries if not e.archived]

    def close(self) -> None:
        self._store.close()
        self._graph.close()

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Memory(path='{self._config.path}', count={self.count()})"
