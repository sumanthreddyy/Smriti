"""SQLite backend with FTS5 full-text search (BM25).

Handles structured storage, metadata, temporal fields, and keyword search.
All data is append-only — deletes set archived=True, never remove rows.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from smriti.types import MemoryEntry


def _safe_json(raw: str, default: Any) -> Any:
    """Parse JSON with a fallback for corrupted data."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL DEFAULT 'fact',
    source TEXT NOT NULL DEFAULT 'user',
    created_at TEXT NOT NULL,
    valid_from TEXT,
    valid_to TEXT,
    superseded_by TEXT,
    archived INTEGER NOT NULL DEFAULT 0,
    metadata TEXT NOT NULL DEFAULT '{}',
    session_id TEXT,
    entities TEXT NOT NULL DEFAULT '[]',
    linked_ids TEXT NOT NULL DEFAULT '[]',
    version INTEGER NOT NULL DEFAULT 1,
    confidence REAL NOT NULL DEFAULT 1.0,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    id UNINDEXED,
    content,
    memory_type,
    source,
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(id, content, memory_type, source)
    VALUES (new.id, new.content, new.memory_type, new.source);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    DELETE FROM memories_fts WHERE id = old.id;
    INSERT INTO memories_fts(id, content, memory_type, source)
    VALUES (new.id, new.content, new.memory_type, new.source);
END;

CREATE INDEX IF NOT EXISTS idx_memories_archived ON memories(archived);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_valid ON memories(valid_from, valid_to);
"""


_ALLOWED_COLUMNS = frozenset({
    "content", "memory_type", "source", "valid_from", "valid_to",
    "superseded_by", "archived", "metadata", "session_id",
    "entities", "linked_ids", "version", "confidence",
    "access_count", "last_accessed",
})


class Store:
    """SQLite-backed structured memory store with BM25 search."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Add columns that may not exist in older databases."""
        existing = {
            r[1] for r in self._conn.execute("PRAGMA table_info(memories)").fetchall()
        }
        new_cols = [
            ("confidence", "REAL NOT NULL DEFAULT 1.0"),
            ("access_count", "INTEGER NOT NULL DEFAULT 0"),
            ("last_accessed", "TEXT"),
        ]
        for col, typedef in new_cols:
            if col not in existing:
                self._conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {typedef}")
        self._conn.commit()

    def add(self, entry: MemoryEntry) -> None:
        """Insert a memory entry. Raises on duplicate ID."""
        self._conn.execute(
            """INSERT INTO memories
               (id, content, memory_type, source, created_at,
                valid_from, valid_to, superseded_by, archived,
                metadata, session_id, entities, linked_ids, version,
                confidence, access_count, last_accessed)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.id,
                entry.content,
                entry.memory_type,
                entry.source,
                entry.created_at,
                entry.valid_from,
                entry.valid_to,
                entry.superseded_by,
                int(entry.archived),
                json.dumps(entry.metadata),
                entry.session_id,
                json.dumps(entry.entities),
                json.dumps(entry.linked_ids),
                entry.version,
                entry.confidence,
                entry.access_count,
                entry.last_accessed,
            ),
        )
        self._conn.commit()

    def get(self, memory_id: str) -> MemoryEntry | None:
        """Get a single memory by ID."""
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return self._row_to_entry(row) if row else None

    def get_many(self, ids: Sequence[str]) -> list[MemoryEntry]:
        """Get multiple memories by ID, preserving order."""
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})", tuple(ids)
        ).fetchall()
        by_id = {r["id"]: self._row_to_entry(r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def update(self, memory_id: str, **fields: object) -> bool:
        """Update specific fields on a memory. Returns True if found."""
        if not fields:
            return False
        bad = set(fields) - _ALLOWED_COLUMNS
        if bad:
            raise ValueError(f"Invalid column(s): {bad}")
        # Serialize complex fields
        for key in ("metadata", "entities", "linked_ids"):
            if key in fields and not isinstance(fields[key], str):
                fields[key] = json.dumps(fields[key])
        if "archived" in fields:
            fields["archived"] = int(fields["archived"])
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [memory_id]
        cur = self._conn.execute(
            f"UPDATE memories SET {set_clause} WHERE id = ?", values
        )
        self._conn.commit()
        return cur.rowcount > 0

    def archive(self, memory_id: str) -> bool:
        """Soft-delete: mark as archived. Data stays forever."""
        return self.update(memory_id, archived=True)

    def list_active(self, limit: int = 100) -> list[MemoryEntry]:
        """List non-archived memories, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE archived = 0 ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def list_all(self, limit: int = 100) -> list[MemoryEntry]:
        """List all memories including archived."""
        rows = self._conn.execute(
            "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def bm25_search(
        self, query: str, top_k: int = 10, include_archived: bool = False
    ) -> list[tuple[str, float]]:
        """BM25 keyword search via FTS5. Returns [(id, rank_score), ...]."""
        sql = """
            SELECT f.id, f.rank
            FROM memories_fts f
            JOIN memories m ON m.id = f.id
            WHERE memories_fts MATCH ?
        """
        if not include_archived:
            sql += " AND m.archived = 0"
        sql += " ORDER BY f.rank LIMIT ?"

        try:
            rows = self._conn.execute(sql, (query, top_k)).fetchall()
        except sqlite3.OperationalError:
            # FTS5 match can fail on certain queries (symbols, etc.)
            return []
        # FTS5 rank is negative (more negative = better match)
        return [(row["id"], -row["rank"]) for row in rows]

    def count(self, include_archived: bool = False) -> int:
        """Count memories."""
        sql = "SELECT COUNT(*) as c FROM memories"
        if not include_archived:
            sql += " WHERE archived = 0"
        return self._conn.execute(sql).fetchone()["c"]

    # ── Temporal queries (v0.2) ───────────────────────────────────────────

    def valid_at(
        self, date_str: str, include_archived: bool = False, limit: int = 1000
    ) -> list[MemoryEntry]:
        """Get memories valid at a specific date."""
        sql = """
            SELECT * FROM memories
            WHERE (valid_from IS NULL OR valid_from <= ?)
            AND (valid_to IS NULL OR valid_to >= ?)
        """
        if not include_archived:
            sql += " AND archived = 0"
        sql += " ORDER BY created_at DESC LIMIT ?"
        rows = self._conn.execute(sql, (date_str, date_str, limit)).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_chain(self, memory_id: str) -> list[MemoryEntry]:
        """Follow the supersession chain containing this memory (oldest→newest)."""
        # Walk backwards to find root
        root_id = memory_id
        visited: set[str] = {memory_id}
        while True:
            row = self._conn.execute(
                "SELECT id FROM memories WHERE superseded_by = ?", (root_id,)
            ).fetchone()
            if row and row["id"] not in visited:
                root_id = row["id"]
                visited.add(root_id)
            else:
                break
        # Walk forward from root
        chain: list[MemoryEntry] = []
        current_id: str | None = root_id
        visited.clear()
        while current_id and current_id not in visited:
            visited.add(current_id)
            entry = self.get(current_id)
            if not entry:
                break
            chain.append(entry)
            current_id = entry.superseded_by
        return chain

    # ── Type queries (v0.4) ───────────────────────────────────────────────

    def by_type(
        self, memory_type: str, include_archived: bool = False, limit: int = 100
    ) -> list[MemoryEntry]:
        """Filter memories by type (fact/belief/opinion/observation)."""
        sql = "SELECT * FROM memories WHERE memory_type = ?"
        if not include_archived:
            sql += " AND archived = 0"
        sql += " ORDER BY created_at DESC LIMIT ?"
        rows = self._conn.execute(sql, (memory_type, limit)).fetchall()
        return [self._row_to_entry(r) for r in rows]

    # ── Access tracking (v0.5) ────────────────────────────────────────────

    def record_access(self, memory_id: str) -> None:
        """Increment access count and update last_accessed timestamp."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
            (now, memory_id),
        )
        self._conn.commit()

    # ── Link management (v0.5) ────────────────────────────────────────────

    def add_link(self, id1: str, id2: str) -> bool:
        """Create a bidirectional link between two memories."""
        e1 = self.get(id1)
        e2 = self.get(id2)
        if not e1 or not e2:
            return False
        if id2 not in e1.linked_ids:
            e1.linked_ids.append(id2)
            self.update(id1, linked_ids=e1.linked_ids)
        if id1 not in e2.linked_ids:
            e2.linked_ids.append(id1)
            self.update(id2, linked_ids=e2.linked_ids)
        return True

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            id=row["id"],
            content=row["content"],
            memory_type=row["memory_type"],
            source=row["source"],
            created_at=row["created_at"],
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
            superseded_by=row["superseded_by"],
            archived=bool(row["archived"]),
            metadata=_safe_json(row["metadata"], {}),
            session_id=row["session_id"],
            entities=_safe_json(row["entities"], []),
            linked_ids=_safe_json(row["linked_ids"], []),
            version=row["version"],
            confidence=row["confidence"],
            access_count=row["access_count"],
            last_accessed=row["last_accessed"],
        )
