"""Tests for hybrid search and RRF fusion."""


from smriti.search import build_results, reciprocal_rank_fusion
from smriti.types import MemoryEntry


class TestRRF:
    def test_single_list(self):
        results = reciprocal_rank_fusion(
            [[ ("a", 0.9), ("b", 0.7), ("c", 0.5) ]],
            ["semantic"],
        )
        assert len(results) == 3
        assert results[0][0] == "a"  # highest RRF score
        assert results[0][2] == ["semantic"]

    def test_fusion_boosts_overlap(self):
        """Documents appearing in BOTH lists should rank higher."""
        semantic = [("a", 0.9), ("b", 0.7), ("c", 0.5)]
        bm25 = [("b", 5.0), ("d", 3.0), ("a", 1.0)]

        results = reciprocal_rank_fusion(
            [semantic, bm25], ["semantic", "bm25"]
        )

        ids = [r[0] for r in results]
        # "a" and "b" appear in both lists, should be top 2
        assert set(ids[:2]) == {"a", "b"}
        # "b" is #1 in bm25 and #2 in semantic: should rank very high
        assert ids[0] in ("a", "b")

    def test_fusion_preserves_unique(self):
        """Items from only one list should still appear."""
        semantic = [("a", 0.9)]
        bm25 = [("b", 5.0)]

        results = reciprocal_rank_fusion(
            [semantic, bm25], ["semantic", "bm25"]
        )
        ids = [r[0] for r in results]
        assert "a" in ids
        assert "b" in ids

    def test_sources_tracked(self):
        semantic = [("a", 0.9), ("b", 0.7)]
        bm25 = [("b", 5.0), ("c", 3.0)]

        results = reciprocal_rank_fusion(
            [semantic, bm25], ["semantic", "bm25"]
        )
        by_id = {r[0]: r[2] for r in results}
        assert by_id["a"] == ["semantic"]
        assert set(by_id["b"]) == {"semantic", "bm25"}
        assert by_id["c"] == ["bm25"]

    def test_empty_lists(self):
        results = reciprocal_rank_fusion([], [])
        assert results == []

    def test_rrf_k_affects_scoring(self):
        """Higher k makes ranking more uniform."""
        items = [("a", 0.9), ("b", 0.7)]
        low_k = reciprocal_rank_fusion([items], ["s"], k=1)
        high_k = reciprocal_rank_fusion([items], ["s"], k=1000)

        # With low k, gap between #1 and #2 is larger
        low_gap = low_k[0][1] - low_k[1][1]
        high_gap = high_k[0][1] - high_k[1][1]
        assert low_gap > high_gap


class TestBuildResults:
    def test_builds_search_results(self):
        entries = {
            "a": MemoryEntry(id="a", content="alpha"),
            "b": MemoryEntry(id="b", content="beta"),
        }
        fused = [("a", 0.5, ["semantic"]), ("b", 0.3, ["bm25"])]

        results = build_results(fused, entries, top_k=10)
        assert len(results) == 2
        assert results[0].entry.content == "alpha"
        assert results[0].rank == 1
        assert results[1].rank == 2

    def test_respects_top_k(self):
        entries = {f"e{i}": MemoryEntry(id=f"e{i}", content=f"c{i}") for i in range(10)}
        fused = [(f"e{i}", 1.0 / (i + 1), ["s"]) for i in range(10)]

        results = build_results(fused, entries, top_k=3)
        assert len(results) == 3

    def test_skips_missing_entries(self):
        entries = {"a": MemoryEntry(id="a", content="alpha")}
        fused = [("a", 0.5, ["s"]), ("missing", 0.3, ["s"])]

        results = build_results(fused, entries, top_k=10)
        assert len(results) == 1
