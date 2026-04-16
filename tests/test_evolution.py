"""Tests for memory evolution and linking (v0.5)."""

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


class TestLinking:
    def test_link_two_memories(self, mem):
        e1 = mem.add("Python is great")
        e2 = mem.add("Django is a Python framework")
        success = mem.link(e1.id, e2.id)
        assert success is True

    def test_linked_memories_bidirectional(self, mem):
        e1 = mem.add("fact A")
        e2 = mem.add("fact B")
        mem.link(e1.id, e2.id)

        linked_from_e1 = mem.get_linked(e1.id)
        linked_from_e2 = mem.get_linked(e2.id)

        assert any(lnk.id == e2.id for lnk in linked_from_e1)
        assert any(lnk.id == e1.id for lnk in linked_from_e2)

    def test_link_nonexistent_fails(self, mem):
        e1 = mem.add("real memory")
        success = mem.link(e1.id, "nonexistent")
        assert success is False

    def test_get_linked_empty(self, mem):
        e = mem.add("no links")
        linked = mem.get_linked(e.id)
        assert linked == []

    def test_multiple_links(self, mem):
        e1 = mem.add("central memory")
        e2 = mem.add("related A")
        e3 = mem.add("related B")
        mem.link(e1.id, e2.id)
        mem.link(e1.id, e3.id)

        linked = mem.get_linked(e1.id)
        ids = [lnk.id for lnk in linked]
        assert e2.id in ids
        assert e3.id in ids

    def test_link_idempotent(self, mem):
        e1 = mem.add("A")
        e2 = mem.add("B")
        mem.link(e1.id, e2.id)
        mem.link(e1.id, e2.id)  # duplicate link
        linked = mem.get_linked(e1.id)
        # Should not have duplicate entries
        assert len(linked) == 1


class TestAccessTracking:
    def test_access_count_increments(self, mem):
        e = mem.add("tracked memory")
        # get() triggers access tracking
        mem.get(e.id)
        mem.get(e.id)
        mem.get(e.id)
        # Read directly from store to check (get() increments each time)
        entry = mem._store.get(e.id)
        assert entry.access_count >= 3

    def test_last_accessed_set(self, mem):
        e = mem.add("will be accessed")
        assert e.last_accessed is None
        mem.get(e.id)
        entry = mem._store.get(e.id)
        assert entry.last_accessed is not None

    def test_archived_not_tracked(self, mem):
        e = mem.add("will be archived")
        mem.delete(e.id)
        mem.get(e.id)  # access archived memory
        entry = mem._store.get(e.id)
        # archived memories should not have access tracked
        assert entry.access_count == 0


class TestEvolution:
    def test_update_chain_preserves_all(self, mem):
        """Multi-step evolution preserves every version."""
        v1 = mem.add("version 1")
        v2 = mem.update(v1.id, "version 2")
        v3 = mem.update(v2.id, "version 3")

        # All 3 versions exist
        assert mem.get(v1.id).content == "version 1"
        assert mem.get(v2.id).content == "version 2"
        assert mem.get(v3.id).content == "version 3"

        # Supersession chain
        assert mem.get(v1.id).superseded_by == v2.id
        assert mem.get(v2.id).superseded_by == v3.id
        assert mem.get(v3.id).superseded_by is None

    def test_version_numbers_track(self, mem):
        v1 = mem.add("original")
        v2 = mem.update(v1.id, "updated")
        assert v1.version == 1
        assert v2.version == 2

    def test_update_archives_old(self, mem):
        old = mem.add("old content")
        mem.update(old.id, "new content")
        entry = mem.get(old.id)
        assert entry.archived is True

    def test_linked_memories_survive_update(self, mem):
        e1 = mem.add("A")
        e2 = mem.add("B")
        mem.link(e1.id, e2.id)
        mem.update(e1.id, "A updated")
        # Old e1 is archived but link to e2 still exists
        old = mem.get(e1.id)
        assert e2.id in old.linked_ids
