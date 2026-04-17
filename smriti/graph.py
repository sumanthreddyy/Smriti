"""Knowledge graph for entity-relationship tracking (v0.3).

Stores entity-memory mappings and co-occurrence edges in SQLite.
Uses NetworkX for in-memory graph traversal.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx

from smriti.types import MemoryEntry

_STOP_WORDS = frozenset({
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
    "this", "that", "these", "those", "the", "a", "an", "and", "or", "but",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "can", "shall", "not", "no", "yes", "so", "if", "then", "than", "too",
    "very", "just", "also", "now", "here", "there", "when", "where", "what",
    "which", "who", "how", "why", "all", "each", "every", "both", "few",
    "more", "most", "some", "any", "such", "only", "same", "into", "from",
    "with", "about", "between", "through", "during", "before", "after",
    "for", "nor", "yet", "at", "by", "in", "of", "on", "to", "as",
})

_GRAPH_SCHEMA = """
CREATE TABLE IF NOT EXISTS entity_memories (
    entity TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    PRIMARY KEY (entity, memory_id)
);

CREATE TABLE IF NOT EXISTS graph_edges (
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    relation TEXT NOT NULL DEFAULT 'co_occurs',
    memory_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (source, target, memory_id)
);

CREATE INDEX IF NOT EXISTS idx_em_entity ON entity_memories(entity);
CREATE INDEX IF NOT EXISTS idx_em_memory ON entity_memories(memory_id);
CREATE INDEX IF NOT EXISTS idx_ge_source ON graph_edges(source);
CREATE INDEX IF NOT EXISTS idx_ge_target ON graph_edges(target);
"""


class KnowledgeGraph:
    """Entity-relationship graph backed by SQLite + NetworkX."""

    def __init__(self, db_path: str | Path) -> None:
        self._conn = sqlite3.connect(str(db_path), timeout=10)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_GRAPH_SCHEMA)
        self._g: nx.Graph = nx.Graph()
        self._load()

    def _load(self) -> None:
        """Load persisted graph into NetworkX."""
        for row in self._conn.execute("SELECT entity, memory_id FROM entity_memories"):
            e = row["entity"]
            if e not in self._g:
                self._g.add_node(e, memory_ids=set())
            self._g.nodes[e]["memory_ids"].add(row["memory_id"])

        for row in self._conn.execute("SELECT source, target, relation FROM graph_edges"):
            self._g.add_edge(row["source"], row["target"], relation=row["relation"])

    def add_memory(self, entry: MemoryEntry) -> list[str]:
        """Index a memory's entities in the graph. Returns extracted entities."""
        entities = entry.entities if entry.entities else self.extract_entities(entry.content)
        if not entities:
            return []

        now = datetime.now(timezone.utc).isoformat()
        lower = [e.lower() for e in entities]

        for el in lower:
            self._conn.execute(
                "INSERT OR IGNORE INTO entity_memories (entity, memory_id) VALUES (?, ?)",
                (el, entry.id),
            )
            if el not in self._g:
                self._g.add_node(el, memory_ids=set())
            self._g.nodes[el]["memory_ids"].add(entry.id)

        # Co-occurrence edges between entities in same memory
        for i in range(len(lower)):
            for j in range(i + 1, len(lower)):
                self._conn.execute(
                    "INSERT OR IGNORE INTO graph_edges "
                    "(source, target, relation, memory_id, created_at) "
                    "VALUES (?, ?, 'co_occurs', ?, ?)",
                    (lower[i], lower[j], entry.id, now),
                )
                self._g.add_edge(lower[i], lower[j], relation="co_occurs")

        self._conn.commit()
        return entities

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Find memories via entity matching + 1-hop graph traversal."""
        q_entities = self.extract_entities(query)
        if not q_entities:
            q_entities = self._fuzzy_match(query)
        if not q_entities:
            return []

        scores: dict[str, float] = {}

        for e in q_entities:
            el = e.lower()
            # Direct entity→memory hits
            for mid in self._entity_memories(el):
                scores[mid] = scores.get(mid, 0.0) + 1.0
            # 1-hop neighbours
            if el in self._g:
                for nb in self._g.neighbors(el):
                    for mid in self._g.nodes[nb].get("memory_ids", set()):
                        scores[mid] = scores.get(mid, 0.0) + 0.5

        if not scores:
            return []

        mx = max(scores.values())
        return sorted(
            [(mid, s / mx) for mid, s in scores.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

    def get_related(self, memory_id: str) -> list[str]:
        """Memory IDs sharing entities with the given memory."""
        entities = self._memory_entities(memory_id)
        related: set[str] = set()
        for e in entities:
            for mid in self._entity_memories(e):
                if mid != memory_id:
                    related.add(mid)
        return list(related)

    def get_entities(self, memory_id: str) -> list[str]:
        """All entities associated with a memory."""
        return self._memory_entities(memory_id)

    def get_entity_memories(self, entity: str) -> list[str]:
        """All memory IDs mentioning an entity."""
        return self._entity_memories(entity.lower())

    def entity_count(self) -> int:
        r = self._conn.execute(
            "SELECT COUNT(DISTINCT entity) as c FROM entity_memories"
        ).fetchone()
        return r["c"]

    def edge_count(self) -> int:
        r = self._conn.execute("SELECT COUNT(*) as c FROM graph_edges").fetchone()
        return r["c"]

    # ── helpers ───────────────────────────────────────────────────────────

    def _entity_memories(self, entity: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT memory_id FROM entity_memories WHERE entity = ?", (entity,)
        ).fetchall()
        return [r["memory_id"] for r in rows]

    def _memory_entities(self, memory_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT entity FROM entity_memories WHERE memory_id = ?", (memory_id,)
        ).fetchall()
        return [r["entity"] for r in rows]

    @staticmethod
    def _escape_like(s: str) -> str:
        """Escape LIKE wildcards so user input is treated literally."""
        return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    def _fuzzy_match(self, query: str) -> list[str]:
        """Match query words against known entities via LIKE."""
        words = [
            w for w in query.lower().split()
            if len(w) >= 3 and w not in _STOP_WORDS
        ]
        matched: list[str] = []
        for w in words:
            safe = self._escape_like(w)
            rows = self._conn.execute(
                "SELECT DISTINCT entity FROM entity_memories WHERE entity LIKE ? ESCAPE '\\'",
                (f"%{safe}%",),
            ).fetchall()
            matched.extend(r["entity"] for r in rows)
        return matched

    @staticmethod
    def extract_entities(text: str) -> list[str]:
        """Regex-based entity extraction (no spaCy needed).

        Finds: quoted strings, capitalized proper nouns, acronyms, CamelCase.
        """
        entities: set[str] = set()

        # Quoted strings
        for m in re.finditer(r'"([^"]{2,})"', text):
            entities.add(m.group(1).strip())

        # Capitalized words not at sentence start
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            words = sentence.strip().split()
            for i, w in enumerate(words):
                clean = re.sub(r"[^\w\-]", "", w)
                if len(clean) < 2 or clean.lower() in _STOP_WORDS:
                    continue
                if clean[0].isupper() and i > 0:
                    entities.add(clean)

        # ALL_CAPS acronyms (2+ letters)
        for m in re.finditer(r"\b([A-Z]{2,})\b", text):
            if m.group(1).lower() not in _STOP_WORDS:
                entities.add(m.group(1))

        # CamelCase
        for m in re.finditer(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b", text):
            entities.add(m.group(1))

        return list(entities)

    def close(self) -> None:
        self._conn.close()
