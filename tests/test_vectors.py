"""Tests for ChromaDB vector store."""

import shutil
import tempfile

import pytest

from smriti.types import MemoryEntry
from smriti.vectors import HashEmbedding, VectorStore


@pytest.fixture
def vstore():
    tmpdir = tempfile.mkdtemp()
    vs = VectorStore(persist_dir=tmpdir, collection_name="test", embedding_fn=HashEmbedding())
    yield vs
    shutil.rmtree(tmpdir, ignore_errors=True)


def _make(content="test", **kw):
    return MemoryEntry(content=content, **kw)


class TestVectorCRUD:
    def test_add_and_search(self, vstore):
        vstore.add(_make("Postgres is a relational database", id="pg"))
        vstore.add(_make("Python is a programming language", id="py"))

        results = vstore.search("database", top_k=5)
        assert len(results) >= 1
        ids = [r[0] for r in results]
        assert "pg" in ids

    def test_count(self, vstore):
        assert vstore.count() == 0
        vstore.add(_make("one", id="1"))
        vstore.add(_make("two", id="2"))
        assert vstore.count() == 2

    def test_similarity_score_range(self, vstore):
        vstore.add(_make("the quick brown fox", id="fox"))
        results = vstore.search("quick fox", top_k=1)
        assert len(results) == 1
        _id, score = results[0]
        assert 0.0 <= score <= 1.0

    def test_upsert_idempotent(self, vstore):
        e = _make("test content", id="same")
        vstore.add(e)
        vstore.add(e)  # upsert, not duplicate
        assert vstore.count() == 1


class TestVectorFiltering:
    def test_excludes_archived_by_default(self, vstore):
        e = _make("secret data", id="secret")
        e.archived = True
        vstore.add(e)
        results = vstore.search("secret", top_k=5, include_archived=False)
        ids = [r[0] for r in results]
        assert "secret" not in ids

    def test_includes_archived_when_asked(self, vstore):
        e = _make("secret data", id="secret")
        e.archived = True
        vstore.add(e)
        results = vstore.search("secret", top_k=5, include_archived=True)
        ids = [r[0] for r in results]
        assert "secret" in ids

    def test_update_metadata(self, vstore):
        e = _make("some data", id="evolve")
        vstore.add(e)
        vstore.update_metadata("evolve", archived=True)
        # Should now be excluded from default search
        results = vstore.search("some data", top_k=5, include_archived=False)
        ids = [r[0] for r in results]
        assert "evolve" not in ids


class TestVectorDelete:
    def test_hard_delete(self, vstore):
        vstore.add(_make("temp data", id="temp"))
        assert vstore.count() == 1
        vstore.delete("temp")
        assert vstore.count() == 0

    def test_delete_nonexistent(self, vstore):
        vstore.delete("nope")  # should not raise
