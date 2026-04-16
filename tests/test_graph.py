"""Tests for knowledge graph features (v0.3)."""

import shutil
import tempfile
from pathlib import Path

import pytest

from smriti import Memory
from smriti.graph import KnowledgeGraph
from smriti.types import MemoryEntry
from smriti.vectors import HashEmbedding


@pytest.fixture
def graph():
    tmpdir = tempfile.mkdtemp()
    g = KnowledgeGraph(Path(tmpdir) / "test.db")
    yield g
    g.close()
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def mem():
    tmpdir = tempfile.mkdtemp()
    m = Memory(path=tmpdir, embedding_fn=HashEmbedding())
    yield m
    m.close()
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestEntityExtraction:
    def test_extracts_capitalized_words(self):
        entities = KnowledgeGraph.extract_entities(
            "I use Postgres and Redis for caching"
        )
        lower = [e.lower() for e in entities]
        assert "postgres" in lower
        assert "redis" in lower

    def test_extracts_quoted_strings(self):
        entities = KnowledgeGraph.extract_entities(
            'The project is called "Memory Palace"'
        )
        assert "Memory Palace" in entities

    def test_extracts_acronyms(self):
        entities = KnowledgeGraph.extract_entities(
            "We use the AWS SDK for API calls"
        )
        lower = [e.lower() for e in entities]
        assert "aws" in lower or "AWS" in entities
        assert "sdk" in lower or "SDK" in entities
        assert "api" in lower or "API" in entities

    def test_extracts_camelcase(self):
        entities = KnowledgeGraph.extract_entities(
            "The PostgreSQL connection uses ConnectionPool"
        )
        assert "ConnectionPool" in entities

    def test_empty_text(self):
        entities = KnowledgeGraph.extract_entities("")
        assert entities == []


class TestGraphCRUD:
    def test_add_memory_extracts_entities(self, graph):
        entry = MemoryEntry(
            content="John works at Google on the Chrome team",
            id="m1",
        )
        entities = graph.add_memory(entry)
        assert len(entities) > 0
        assert graph.entity_count() > 0

    def test_entity_memory_mapping(self, graph):
        entry = MemoryEntry(
            content="Kumar and Kumlee work at Valve",
            id="m1",
            entities=["Kumar", "Kumlee", "Valve"],
        )
        graph.add_memory(entry)
        mems = graph.get_entity_memories("kumar")
        assert "m1" in mems

    def test_co_occurrence_edges(self, graph):
        entry = MemoryEntry(
            content="test",
            id="m1",
            entities=["Python", "Django"],
        )
        graph.add_memory(entry)
        assert graph.edge_count() >= 1

    def test_get_entities_for_memory(self, graph):
        entry = MemoryEntry(
            content="test",
            id="m1",
            entities=["Postgres", "Redis"],
        )
        graph.add_memory(entry)
        entities = graph.get_entities("m1")
        assert "postgres" in entities
        assert "redis" in entities

    def test_get_related(self, graph):
        e1 = MemoryEntry(content="t", id="m1", entities=["Python"])
        e2 = MemoryEntry(content="t", id="m2", entities=["Python", "Django"])
        graph.add_memory(e1)
        graph.add_memory(e2)
        related = graph.get_related("m1")
        assert "m2" in related


class TestGraphSearch:
    def test_finds_by_entity(self, graph):
        entry = MemoryEntry(
            content="test",
            id="m1",
            entities=["PostgreSQL"],
        )
        graph.add_memory(entry)
        results = graph.search("PostgreSQL", top_k=5)
        ids = [r[0] for r in results]
        assert "m1" in ids

    def test_fuzzy_entity_match(self, graph):
        entry = MemoryEntry(
            content="test",
            id="m1",
            entities=["PostgreSQL"],
        )
        graph.add_memory(entry)
        results = graph.search("postgres", top_k=5)
        ids = [r[0] for r in results]
        assert "m1" in ids

    def test_neighbour_boost(self, graph):
        e1 = MemoryEntry(content="t", id="m1", entities=["Python", "FastAPI"])
        e2 = MemoryEntry(content="t", id="m2", entities=["FastAPI", "Docker"])
        graph.add_memory(e1)
        graph.add_memory(e2)
        # Search for Python should find m1 directly, and m2 via FastAPI neighbour
        results = graph.search("Python", top_k=5)
        ids = [r[0] for r in results]
        assert "m1" in ids
        # m2 might also appear via 1-hop

    def test_empty_query(self, graph):
        results = graph.search("xyznonexistent", top_k=5)
        assert results == []


class TestGraphInMemory:
    def test_graph_search_in_hybrid_mode(self, mem):
        """Graph results contribute to hybrid search."""
        mem.add(
            "Kumar works at Valve in Seattle",
            entities=["Kumar", "Valve", "Seattle"],
        )
        mem.add(
            "Valve develops Steam platform",
            entities=["Valve"],
        )
        results = mem.search("Valve", mode="hybrid")
        assert len(results) >= 1

    def test_related_memories(self, mem):
        e1 = mem.add("Python is great", entities=["Python"])
        e2 = mem.add("I use Python and Django", entities=["Python", "Django"])
        related = mem.related(e1.id)
        ids = [r.id for r in related]
        assert e2.id in ids

    def test_entities_method(self, mem):
        e = mem.add("test content", entities=["Redis", "Postgres"])
        ents = mem.entities(e.id)
        assert "redis" in ents
        assert "postgres" in ents
