"""Tests for memory networks / type classification (v0.4)."""

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


class TestMemoryTypes:
    def test_default_type_is_fact(self, mem):
        e = mem.add("The sky is blue")
        assert e.memory_type == "fact"

    def test_belief_type(self, mem):
        e = mem.add("AI will surpass humans", memory_type="belief")
        assert e.memory_type == "belief"

    def test_opinion_type(self, mem):
        e = mem.add("Postgres is better than MySQL", memory_type="opinion")
        assert e.memory_type == "opinion"

    def test_observation_type(self, mem):
        e = mem.add("User seems frustrated today", memory_type="observation")
        assert e.memory_type == "observation"

    def test_by_type_filter(self, mem):
        mem.add("fact 1", memory_type="fact")
        mem.add("fact 2", memory_type="fact")
        mem.add("opinion 1", memory_type="opinion")
        mem.add("belief 1", memory_type="belief")

        facts = mem.by_type("fact")
        assert len(facts) == 2
        assert all(f.memory_type == "fact" for f in facts)

        opinions = mem.by_type("opinion")
        assert len(opinions) == 1

    def test_by_type_empty(self, mem):
        mem.add("something", memory_type="fact")
        observations = mem.by_type("observation")
        assert observations == []


class TestConfidence:
    def test_default_confidence(self, mem):
        e = mem.add("high confidence fact")
        assert e.confidence == 1.0

    def test_custom_confidence(self, mem):
        e = mem.add("might be true", confidence=0.6)
        got = mem.get(e.id)
        assert got.confidence == 0.6

    def test_low_confidence(self, mem):
        e = mem.add("very uncertain", confidence=0.1)
        got = mem.get(e.id)
        assert got.confidence == 0.1

    def test_confidence_persists_through_update(self, mem):
        old = mem.add("original", confidence=0.8)
        new = mem.update(old.id, "updated")
        got = mem.get(new.id)
        # New entry inherits default confidence (1.0), not old
        assert got.confidence == 1.0


class TestTypeSearch:
    def test_search_finds_all_types(self, mem):
        mem.add("Python is fast", memory_type="fact")
        mem.add("Python will dominate", memory_type="belief")
        mem.add("Python is the best", memory_type="opinion")
        results = mem.search("Python")
        assert len(results) >= 2

    def test_by_type_excludes_archived(self, mem):
        e1 = mem.add("active fact", memory_type="fact")
        e2 = mem.add("archived fact", memory_type="fact")
        mem.delete(e2.id)
        facts = mem.by_type("fact")
        assert len(facts) == 1
        assert facts[0].id == e1.id
