"""Invariant tests — properties that must NEVER break across any version.

These are the non-negotiable guarantees of Smriti.
"""

import shutil
import tempfile

import pytest

from smriti import Memory
from smriti.vectors import HashEmbedding


@pytest.fixture
def mem():
    tmpdir = tempfile.mkdtemp()
    m = Memory(path=tmpdir, embedding_fn=HashEmbedding())
    yield m
    m.close()
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestNothingIsEverLost:
    """THE core invariant: data added to Smriti can ALWAYS be retrieved."""

    def test_add_then_get(self, mem):
        e = mem.add("The sky is blue")
        got = mem.get(e.id)
        assert got is not None
        assert got.content == "The sky is blue"

    def test_delete_does_not_destroy(self, mem):
        e = mem.add("Sensitive info")
        mem.delete(e.id)
        # STILL retrievable by direct ID
        got = mem.get(e.id)
        assert got is not None
        assert got.content == "Sensitive info"
        assert got.archived is True

    def test_update_preserves_original(self, mem):
        old = mem.add("Uses MySQL")
        new = mem.update(old.id, "Uses Postgres")
        # Old entry still exists
        old_got = mem.get(old.id)
        assert old_got is not None
        assert old_got.content == "Uses MySQL"
        assert old_got.archived is True
        assert old_got.superseded_by == new.id
        # New entry exists
        new_got = mem.get(new.id)
        assert new_got is not None
        assert new_got.content == "Uses Postgres"

    def test_bulk_add_all_retrievable(self, mem):
        entries = []
        for i in range(50):
            entries.append(mem.add(f"Fact number {i}"))
        for e in entries:
            got = mem.get(e.id)
            assert got is not None
            assert got.content == e.content


class TestSearchFindsWhatWasStored:
    """If you stored it, search MUST find it."""

    def test_exact_content_search(self, mem):
        mem.add("I prefer Postgres over MySQL for large projects")
        results = mem.search("Postgres")
        assert len(results) >= 1
        contents = [r.entry.content for r in results]
        assert any("Postgres" in c for c in contents)

    def test_semantic_search(self, mem):
        mem.add("My favorite database is PostgreSQL")
        results = mem.search("what database do I like?")
        assert len(results) >= 1

    def test_search_after_many_adds(self, mem):
        # Add noise
        for i in range(20):
            mem.add(f"Random fact number {i} about weather")
        # Add the needle
        mem.add("API key rotates every 90 days")
        # Find the needle
        results = mem.search("API key rotation")
        assert len(results) >= 1
        assert any("90 days" in r.entry.content for r in results)


class TestArchiveIntegrity:
    """Archived items hidden from search but never destroyed."""

    def test_archived_hidden_from_default_search(self, mem):
        e = mem.add("old database config: MySQL on port 3306")
        mem.delete(e.id)
        results = mem.search("MySQL")
        ids = [r.entry.id for r in results]
        assert e.id not in ids

    def test_archived_findable_with_flag(self, mem):
        e = mem.add("old database config: MySQL on port 3306")
        mem.delete(e.id)
        results = mem.search("MySQL", include_archived=True)
        ids = [r.entry.id for r in results]
        assert e.id in ids

    def test_count_reflects_archive(self, mem):
        mem.add("one")
        e2 = mem.add("two")
        assert mem.count() == 2
        mem.delete(e2.id)
        assert mem.count() == 1
        assert mem.count(include_archived=True) == 2


class TestSearchModes:
    """Different search modes work correctly."""

    def test_semantic_only(self, mem):
        mem.add("PostgreSQL is a relational database management system")
        results = mem.search("what DB should I use?", mode="semantic")
        assert len(results) >= 1
        assert all("semantic" in r.sources for r in results)

    def test_bm25_only(self, mem):
        mem.add("PostgreSQL is a relational database management system")
        results = mem.search("PostgreSQL", mode="bm25")
        assert len(results) >= 1
        assert all("bm25" in r.sources for r in results)

    def test_hybrid_combines_sources(self, mem):
        mem.add("PostgreSQL is a relational database management system")
        mem.add("Redis is an in-memory key-value store")
        mem.add("Python is a programming language")
        results = mem.search("database", mode="hybrid")
        assert len(results) >= 1


class TestEdgeCases:
    def test_empty_search(self, mem):
        results = mem.search("anything")
        assert results == []

    def test_empty_memory_count(self, mem):
        assert mem.count() == 0

    def test_repr(self, mem):
        r = repr(mem)
        assert "Memory" in r
        assert "count=0" in r

    def test_duplicate_content_separate_entries(self, mem):
        e1 = mem.add("same content")
        e2 = mem.add("same content")
        assert e1.id != e2.id
        assert mem.count() == 2
