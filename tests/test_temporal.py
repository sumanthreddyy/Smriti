"""Tests for temporal memory features (v0.2)."""

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


class TestValidAt:
    def test_no_temporal_bounds_always_valid(self, mem):
        """Memories without valid_from/to are returned for any date."""
        mem.add("timeless fact")
        results = mem.valid_at("2020-01-01")
        assert len(results) >= 1
        assert results[0].content == "timeless fact"

    def test_valid_from_only(self, mem):
        """Memory with valid_from is returned for dates on or after."""
        mem.add("new policy", valid_from="2026-03-01")
        assert len(mem.valid_at("2026-04-01")) >= 1
        assert len(mem.valid_at("2026-02-01")) == 0

    def test_valid_range(self, mem):
        """Memory with both bounds respects the range."""
        mem.add(
            "Q1 budget approved",
            valid_from="2026-01-01",
            valid_to="2026-03-31",
        )

        assert len(mem.valid_at("2026-02-15")) == 1
        assert len(mem.valid_at("2026-04-15")) == 0
        assert len(mem.valid_at("2025-12-01")) == 0

    def test_multiple_temporal_memories(self, mem):
        mem.add("CEO is Kumar", valid_from="2020-01-01")
        mem.add(
            "CEO is Kumlee",
            valid_from="2024-01-01",
            valid_to="2025-12-31",
        )

        at_2023 = mem.valid_at("2023-06-01")
        contents = [e.content for e in at_2023]
        assert "CEO is Kumar" in contents
        assert "CEO is Kumlee" not in contents

        at_2024 = mem.valid_at("2024-06-01")
        contents = [e.content for e in at_2024]
        assert "CEO is Kumar" in contents
        assert "CEO is Kumlee" in contents


class TestHistory:
    def test_single_memory_chain(self, mem):
        """A memory with no updates has a chain of 1."""
        e = mem.add("original fact")
        chain = mem.history(e.id)
        assert len(chain) == 1
        assert chain[0].id == e.id

    def test_update_creates_chain(self, mem):
        """update() creates a supersession chain."""
        old = mem.add("Uses MySQL")
        mem.update(old.id, "Uses Postgres")
        chain = mem.history(old.id)
        assert len(chain) == 2
        assert chain[0].content == "Uses MySQL"
        assert chain[1].content == "Uses Postgres"

    def test_chain_from_new_id(self, mem):
        """Can get full chain starting from the newest entry."""
        old = mem.add("v1")
        mid = mem.update(old.id, "v2")
        new = mem.update(mid.id, "v3")
        chain = mem.history(new.id)
        assert len(chain) == 3
        assert [c.content for c in chain] == ["v1", "v2", "v3"]

    def test_chain_versions_increment(self, mem):
        old = mem.add("original")
        mid = mem.update(old.id, "updated once")
        new = mem.update(mid.id, "updated twice")
        chain = mem.history(new.id)
        assert [c.version for c in chain] == [1, 2, 3]


class TestValidToPublicAPI:
    def test_add_with_valid_to(self, mem):
        """valid_to is exposed directly in add()."""
        e = mem.add("Q1 budget", valid_from="2026-01-01", valid_to="2026-03-31")
        got = mem.get(e.id)
        assert got.valid_to == "2026-03-31"
        assert got.valid_from == "2026-01-01"

    def test_add_entry_runs_graph_indexing(self, mem):
        """add_entry() now goes through the unified pipeline including graph."""
        from smriti.types import MemoryEntry
        entry = MemoryEntry(
            content="Kumar manages the Python team",
            entities=["Kumar", "Python"],
        )
        mem.add_entry(entry)
        entities = mem.entities(entry.id)
        assert len(entities) > 0
