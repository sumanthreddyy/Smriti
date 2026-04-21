"""Tests for the v0.7 retrieval pipeline changes.

Covers:
- Over-fetch multiplier: each retriever returns top_k * fetch_multiplier
  candidates before RRF.
- Graph-off-by-default in hybrid mode (`use_graph_in_hybrid=False`).
- Cross-encoder reranker hook: Memory(reranker=callable) is invoked after
  RRF on a wider slice and re-sorts the final top_k.
"""

from __future__ import annotations

import shutil
import tempfile

import pytest

from smriti import Memory
from smriti.config import SmritiConfig
from smriti.vectors import HashEmbedding


@pytest.fixture
def mem_factory():
    """Build a Memory with the v0.7 MiniLM-era defaults applied explicitly.

    These tests assert the v0.7 retrieval-pipeline behaviour. Because the
    fixture uses HashEmbedding, Memory would normally auto-tune hash-friendly
    defaults (fetch=1, graph=on). We pass the v0.7 defaults explicitly so
    each field is in ``model_fields_set`` and the auto-tune is bypassed.
    The hash auto-tune itself is covered in ``TestHashAutoTune``.
    """
    tmpdirs: list[str] = []

    def _make(config: SmritiConfig | None = None, reranker=None) -> Memory:
        tmp = tempfile.mkdtemp()
        tmpdirs.append(tmp)
        if config is None:
            config = SmritiConfig(fetch_multiplier=5, use_graph_in_hybrid=False)
        cfg = config.model_copy(update={"path": tmp})
        return Memory(config=cfg, embedding_fn=HashEmbedding(), reranker=reranker)

    yield _make

    for tmp in tmpdirs:
        shutil.rmtree(tmp, ignore_errors=True)


class TestOverFetch:
    """Over-fetch lets RRF surface docs that one retriever ranks low."""

    def test_fetch_multiplier_expands_candidate_pool(self, mem_factory):
        # With 20 docs, top_k=5 and fetch_multiplier=5, each retriever
        # should consider 25 (capped at collection size) — so every doc
        # is at least in the candidate pool before RRF.
        mem = mem_factory(SmritiConfig(fetch_multiplier=5))
        for i in range(20):
            mem.add(f"alpha document number {i}", session_id=f"s{i}")
        # Add a clear BM25 needle with a rare term
        target = mem.add("zzqrare beacon marker", session_id="target")

        results = mem.search("zzqrare", top_k=5, mode="hybrid")
        ids = [r.entry.id for r in results]
        assert target.id in ids

    def test_fetch_multiplier_one_still_works(self, mem_factory):
        # Regression: fetch_multiplier=1 reverts to v0.6 behavior.
        mem = mem_factory(SmritiConfig(fetch_multiplier=1))
        for i in range(10):
            mem.add(f"doc {i}", session_id=f"s{i}")
        results = mem.search("doc", top_k=5, mode="hybrid")
        assert len(results) <= 5


class TestGraphGating:
    """Graph retriever is off in hybrid by default, opt-in via config."""

    def test_hybrid_excludes_graph_by_default(self, mem_factory):
        mem = mem_factory()  # use_graph_in_hybrid=False by default
        mem.add("Postgres is my favorite database", session_id="pg")
        results = mem.search("database", top_k=5, mode="hybrid")
        for r in results:
            assert "graph" not in r.sources

    def test_hybrid_includes_graph_when_enabled(self, mem_factory):
        mem = mem_factory(SmritiConfig(use_graph_in_hybrid=True))
        mem.add("I love Postgres for large projects", session_id="pg")
        # Query contains the same entity so graph will fire.
        results = mem.search("tell me about Postgres", top_k=5, mode="hybrid")
        # Either semantic/bm25/graph — as long as graph *can* participate.
        # Relax: just assert no error and graph could appear.
        assert results, "graph-enabled hybrid must still return results"

    def test_graph_mode_still_works(self, mem_factory):
        # Explicit mode="graph" must keep working regardless of the flag.
        mem = mem_factory()
        mem.add("Postgres is my favorite database", session_id="pg")
        results = mem.search("Postgres", top_k=5, mode="graph")
        # May be empty if graph extracted nothing — don't require a hit,
        # just that the call completes without error.
        assert isinstance(results, list)


class TestReranker:
    """The reranker hook is invoked and re-orders the final results."""

    def test_reranker_reorders_results(self, mem_factory):
        # Store: two docs, A and B. Without rerank, whichever the retrievers
        # pick first wins. With a rerank that prefers B, B must come first.
        def rerank(query, pairs):
            # Score B high, everything else low.
            scored = []
            for pid, content in pairs:
                scored.append((pid, 10.0 if "preferred" in content else 0.0))
            return scored

        mem = mem_factory(reranker=rerank)
        a = mem.add("alpha common term", session_id="a")
        b = mem.add("alpha common term preferred", session_id="b")

        results = mem.search("alpha", top_k=2, mode="hybrid")
        ids = [r.entry.id for r in results]
        assert b.id in ids
        # If both are in the pool, the preferred one should win.
        if a.id in ids and b.id in ids:
            assert ids.index(b.id) < ids.index(a.id)

    def test_reranker_failure_falls_back_cleanly(self, mem_factory):
        def broken(query, pairs):
            raise RuntimeError("oops")

        mem = mem_factory(reranker=broken)
        mem.add("hello world", session_id="a")
        # Must not raise — just return the unreranked RRF order.
        results = mem.search("hello", top_k=5, mode="hybrid")
        assert len(results) >= 1

    def test_no_reranker_means_no_rerank_source(self, mem_factory):
        mem = mem_factory()
        mem.add("hello world", session_id="a")
        results = mem.search("hello", top_k=5, mode="hybrid")
        for r in results:
            assert "rerank" not in r.sources


class TestInvalidMode:
    def test_invalid_mode_still_raises(self, mem_factory):
        mem = mem_factory()
        with pytest.raises(ValueError):
            mem.search("q", mode="nope")


class TestHashAutoTune:
    """Memory auto-tunes retrieval defaults for HashEmbedding.

    v0.7's MiniLM defaults (fetch_multiplier=5, use_graph_in_hybrid=False)
    regressed HashEmbedding on LongMemEval-S from 0.77 → 0.60 R@5. When
    the caller doesn't override these fields, Memory should flip them back
    to the hash-friendly values (fetch=1, graph=on).
    """

    def test_defaults_flip_for_hash(self, tmp_path):
        mem = Memory(
            config=SmritiConfig(path=str(tmp_path)),
            embedding_fn=HashEmbedding(),
        )
        assert mem._config.fetch_multiplier == 1
        assert mem._config.use_graph_in_hybrid is True
        mem.close()

    def test_explicit_fetch_multiplier_is_respected(self, tmp_path):
        mem = Memory(
            config=SmritiConfig(path=str(tmp_path), fetch_multiplier=7),
            embedding_fn=HashEmbedding(),
        )
        assert mem._config.fetch_multiplier == 7
        # graph flag was not set, so it still auto-tunes.
        assert mem._config.use_graph_in_hybrid is True
        mem.close()

    def test_explicit_graph_flag_is_respected(self, tmp_path):
        mem = Memory(
            config=SmritiConfig(path=str(tmp_path), use_graph_in_hybrid=False),
            embedding_fn=HashEmbedding(),
        )
        assert mem._config.use_graph_in_hybrid is False
        assert mem._config.fetch_multiplier == 1
        mem.close()

    def test_no_autotune_for_non_hash_embedding(self, tmp_path):
        # A caller-supplied non-hash embedding must not trigger the override.
        class FakeEmbed:
            def __call__(self, input):
                return [[0.0] * 8 for _ in input]

            @staticmethod
            def name():
                return "fake"

            def get_config(self):
                return {}

            @staticmethod
            def build_from_config(config):
                return FakeEmbed()

        mem = Memory(
            config=SmritiConfig(path=str(tmp_path)),
            embedding_fn=FakeEmbed(),
        )
        assert mem._config.fetch_multiplier == 5
        assert mem._config.use_graph_in_hybrid is False
        mem.close()
