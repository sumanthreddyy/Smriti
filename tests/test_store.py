"""Tests for SQLite store backend."""

import shutil
import tempfile
from pathlib import Path

import pytest

from smriti.store import Store
from smriti.types import MemoryEntry


@pytest.fixture
def store():
    tmpdir = tempfile.mkdtemp()
    s = Store(Path(tmpdir) / "test.db")
    yield s
    s.close()
    shutil.rmtree(tmpdir, ignore_errors=True)


def _make(content="test content", **kw):
    return MemoryEntry(content=content, **kw)


class TestBasicCRUD:
    def test_add_and_get(self, store):
        e = _make("Postgres is great")
        store.add(e)
        got = store.get(e.id)
        assert got is not None
        assert got.content == "Postgres is great"
        assert got.id == e.id

    def test_get_missing(self, store):
        assert store.get("nonexistent") is None

    def test_count(self, store):
        assert store.count() == 0
        store.add(_make("one"))
        store.add(_make("two"))
        assert store.count() == 2

    def test_list_active(self, store):
        e1 = _make("first")
        e2 = _make("second")
        store.add(e1)
        store.add(e2)
        active = store.list_active()
        assert len(active) == 2

    def test_get_many(self, store):
        entries = [_make(f"entry {i}") for i in range(5)]
        for e in entries:
            store.add(e)
        ids = [entries[0].id, entries[2].id, entries[4].id]
        got = store.get_many(ids)
        assert len(got) == 3
        assert [g.id for g in got] == ids  # preserves order

    def test_get_many_empty(self, store):
        assert store.get_many([]) == []


class TestArchive:
    def test_archive_keeps_data(self, store):
        e = _make("secret info")
        store.add(e)
        store.archive(e.id)
        # Still retrievable by ID
        got = store.get(e.id)
        assert got is not None
        assert got.archived is True
        assert got.content == "secret info"

    def test_archive_hides_from_active(self, store):
        e = _make("will be archived")
        store.add(e)
        store.archive(e.id)
        assert store.count(include_archived=False) == 0
        assert store.count(include_archived=True) == 1

    def test_list_active_excludes_archived(self, store):
        e1 = _make("active")
        e2 = _make("archived")
        store.add(e1)
        store.add(e2)
        store.archive(e2.id)
        active = store.list_active()
        assert len(active) == 1
        assert active[0].id == e1.id


class TestUpdate:
    def test_update_content(self, store):
        e = _make("old content")
        store.add(e)
        store.update(e.id, content="new content")
        got = store.get(e.id)
        assert got.content == "new content"

    def test_update_metadata(self, store):
        e = _make("test", metadata={"key": "old"})
        store.add(e)
        store.update(e.id, metadata={"key": "new"})
        got = store.get(e.id)
        assert got.metadata == {"key": "new"}

    def test_update_nonexistent(self, store):
        assert store.update("nope", content="x") is False


class TestBM25:
    def test_basic_search(self, store):
        store.add(_make("Postgres is a relational database"))
        store.add(_make("Redis is an in-memory store"))
        store.add(_make("Python is a programming language"))

        results = store.bm25_search("database")
        assert len(results) >= 1
        assert results[0][0]  # has an ID

    def test_search_excludes_archived(self, store):
        e = _make("secret database password")
        store.add(e)
        store.archive(e.id)
        results = store.bm25_search("database", include_archived=False)
        ids = [r[0] for r in results]
        assert e.id not in ids

    def test_search_includes_archived(self, store):
        e = _make("secret database password")
        store.add(e)
        store.archive(e.id)
        results = store.bm25_search("database", include_archived=True)
        ids = [r[0] for r in results]
        assert e.id in ids

    def test_empty_results(self, store):
        store.add(_make("hello world"))
        results = store.bm25_search("xyznonexistent")
        assert results == []


class TestMetadata:
    def test_memory_type(self, store):
        e = _make("test", memory_type="opinion")
        store.add(e)
        got = store.get(e.id)
        assert got.memory_type == "opinion"

    def test_session_id(self, store):
        e = _make("test", session_id="session_42")
        store.add(e)
        got = store.get(e.id)
        assert got.session_id == "session_42"

    def test_entities(self, store):
        e = _make("test", entities=["Postgres", "MySQL"])
        store.add(e)
        got = store.get(e.id)
        assert got.entities == ["Postgres", "MySQL"]

    def test_temporal_fields(self, store):
        e = _make("test", valid_from="2026-01-01", valid_to="2026-03-01")
        store.add(e)
        got = store.get(e.id)
        assert got.valid_from == "2026-01-01"
        assert got.valid_to == "2026-03-01"
