#!/usr/bin/env python3
"""LongMemEval-S benchmark for Smriti.

Downloads the LongMemEval-S dataset (500 questions, ~40 sessions each),
ingests chat sessions into Smriti Memory, runs hybrid search, and
computes Recall@K metrics.

Usage:
    python bench/run_longmemeval.py            # full 470 questions
    python bench/run_longmemeval.py --limit 20 # quick test with 20
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

# Ensure the repo root is on sys.path so `smriti` is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from smriti import Memory, __version__
from smriti.config import SmritiConfig
from smriti.vectors import HashEmbedding

EMBEDDING_LABELS = {
    "hash": "HashEmbedding (deterministic, no model)",
    "default": "ChromaDB default (sentence-transformers)",
    "minilm": "all-MiniLM-L6-v2 (sentence-transformers)",
    "bge": "BGE-large-en-v1.5 (BAAI)",
}

# ── constants ────────────────────────────────────────────────────────────

DATASET_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned"
    "/resolve/main/longmemeval_s_cleaned.json"
)
DATA_DIR = REPO_ROOT / "bench" / "data"
DATASET_PATH = DATA_DIR / "longmemeval_s_cleaned.json"
K_VALUES = [5, 10, 20]


# ── helpers ──────────────────────────────────────────────────────────────

PROGRESS_FILE = REPO_ROOT / "bench" / "progress.log"


def log(msg: str) -> None:
    """Print with immediate flush and also write to a progress file."""
    print(msg, flush=True)
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def download_dataset() -> Path:
    """Download the dataset if it doesn't already exist."""
    if DATASET_PATH.exists():
        log(f"Dataset already present: {DATASET_PATH}")
        return DATASET_PATH

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Downloading LongMemEval-S from HuggingFace …")
    log(f"  URL: {DATASET_URL}")

    urllib.request.urlretrieve(DATASET_URL, str(DATASET_PATH))
    size_mb = DATASET_PATH.stat().st_size / (1024 * 1024)
    log(f"  Saved to {DATASET_PATH} ({size_mb:.1f} MB)")
    return DATASET_PATH


def concat_session(turns: list[dict]) -> str:
    """Concatenate all turns in a session into a single string."""
    parts: list[str] = []
    for turn in turns:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def evaluate_question(
    item: dict,
    embedding_fn: object | None,
    reranker: object | None,
    config: SmritiConfig,
    search_mode: str,
    base_tmp: str,
    idx: int,
) -> dict:
    """Run the benchmark for a single question.

    Returns a dict with question_id, question_type, hit@K booleans, and latencies.
    """
    tmp_dir = os.path.join(base_tmp, f"q{idx}")
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        per_q_config = config.model_copy(update={"path": tmp_dir})
        mem = Memory(
            config=per_q_config,
            embedding_fn=embedding_fn,
            reranker=reranker,
        )

        haystack_session_ids = item["haystack_session_ids"]
        haystack_sessions = item["haystack_sessions"]
        answer_session_ids = set(item["answer_session_ids"])

        # Ingest each session as one memory and track add latency
        add_times: list[float] = []
        for sid, turns in zip(haystack_session_ids, haystack_sessions):
            text = concat_session(turns)
            t0 = time.time()
            mem.add(text, session_id=sid, source="chat")
            add_times.append(time.time() - t0)

        question = item["question"]

        # Search at the largest K, then slice for smaller K values
        max_k = max(K_VALUES)
        t_search = time.time()
        results = mem.search(question, top_k=max_k, mode=search_mode)
        search_latency = time.time() - t_search

        retrieved_sids: list[str | None] = [
            r.entry.session_id for r in results
        ]

        hits: dict[int, bool] = {}
        for k in K_VALUES:
            top_sids = set(retrieved_sids[:k])
            hits[k] = bool(top_sids & answer_session_ids)

        mem.close()
        return {
            "question_id": item["question_id"],
            "question_type": item.get("question_type", "unknown"),
            "hits": hits,
            "search_latency_ms": search_latency * 1000,
            "avg_add_latency_ms": (sum(add_times) / len(add_times) * 1000) if add_times else 0,
            "num_sessions": len(haystack_session_ids),
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── main ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LongMemEval-S benchmark for Smriti")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to N questions (0 = all)")
    parser.add_argument(
        "--embedding",
        choices=["hash", "default", "minilm", "bge"],
        default="hash",
        help=(
            "Embedding: 'hash' (offline), 'default' (ChromaDB's), "
            "'minilm', or 'bge' (BGE-large)"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["hybrid", "semantic", "bm25", "graph"],
        default="hybrid",
        help="Retrieval mode used in mem.search() (default: hybrid)",
    )
    parser.add_argument(
        "--fetch-multiplier",
        type=int,
        default=None,
        help=(
            "Over-fetch multiplier per retriever before RRF. "
            "Default: let Memory pick per-embedding (hash=1, others=5)."
        ),
    )
    parser.add_argument(
        "--graph-in-hybrid",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Include the knowledge graph in hybrid mode. 'auto' lets "
            "Memory decide per-embedding (hash=on, others=off)."
        ),
    )
    parser.add_argument(
        "--rerank",
        choices=[
            "none",
            "ms-marco-minilm",
            "ms-marco-minilm-l12",
            "bge-reranker-base",
        ],
        default="none",
        help="Cross-encoder reranker to apply after RRF (default: none)",
    )
    args = parser.parse_args()

    log(f"Smriti v{__version__} — LongMemEval-S Benchmark")
    log("=" * 64)

    # 1. Download
    dataset_path = download_dataset()
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    log(f"Loaded {len(data)} questions from dataset.\n")

    # 2. Filter out abstention questions
    questions = [q for q in data if not q["question_id"].endswith("_abs")]
    skipped = len(data) - len(questions)
    log(f"Skipping {skipped} abstention questions → {len(questions)} to evaluate.\n")

    if args.limit > 0:
        questions = questions[: args.limit]
        log(f"--limit {args.limit}: evaluating first {len(questions)} questions.\n")

    # 3. Embedding function — build ONCE and reuse across all questions.
    # Previously `--embedding default` passed None per question, forcing
    # ChromaDB to re-instantiate MiniLM for every Memory() built in the loop.
    if args.embedding == "hash":
        embedding_fn: object | None = HashEmbedding()
    elif args.embedding == "minilm":
        from smriti.vectors import load_embedding
        embedding_fn = load_embedding("minilm")
    elif args.embedding == "bge":
        from smriti.vectors import load_embedding
        embedding_fn = load_embedding("bge-large")
    else:  # default
        embedding_fn = None  # Chroma will pick its default
    embed_label = EMBEDDING_LABELS[args.embedding]
    log(f"Embedding: {embed_label}\n")

    # 4. Optional cross-encoder reranker — also load ONCE.
    reranker: object | None = None
    if args.rerank != "none":
        from smriti.vectors import load_reranker
        log(f"Loading reranker: {args.rerank}")
        reranker = load_reranker(args.rerank)

    # 5. Shared config for every per-question Memory. Only pass overrides
    # the caller actually set — that way Memory can auto-tune defaults
    # per embedding (e.g. hash gets fetch=1, graph=on).
    config_overrides: dict[str, object] = {}
    if args.fetch_multiplier is not None:
        config_overrides["fetch_multiplier"] = args.fetch_multiplier
    if args.graph_in_hybrid != "auto":
        config_overrides["use_graph_in_hybrid"] = args.graph_in_hybrid == "on"
    base_config = SmritiConfig(**config_overrides)
    log(
        f"Retrieval: mode={args.mode}  "
        f"fetch_multiplier={base_config.fetch_multiplier}  "
        f"use_graph_in_hybrid={base_config.use_graph_in_hybrid}  "
        f"rerank={args.rerank}\n"
    )

    # 6. Create a single temp base directory for all questions
    base_tmp = tempfile.mkdtemp(prefix="smriti_longmemeval_")
    log(f"Temp directory: {base_tmp}\n")

    # 7. Evaluate
    type_hits: dict[str, dict[int, int]] = defaultdict(lambda: {k: 0 for k in K_VALUES})
    type_counts: dict[str, int] = defaultdict(int)
    search_latencies: list[float] = []
    add_latencies: list[float] = []

    # Clear progress file
    PROGRESS_FILE.write_text("", encoding="utf-8")

    t0 = time.time()
    for i, item in enumerate(questions):
        tq = time.time()
        result = evaluate_question(
            item,
            embedding_fn,
            reranker,
            base_config,
            args.mode,
            base_tmp,
            i,
        )
        dt = time.time() - tq
        qtype = result["question_type"]
        type_counts[qtype] += 1
        for k in K_VALUES:
            if result["hits"][k]:
                type_hits[qtype][k] += 1

        search_latencies.append(result["search_latency_ms"])
        add_latencies.append(result["avg_add_latency_ms"])

        if (i + 1) % 10 == 0 or (i + 1) == len(questions) or i < 3:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(questions) - i - 1) / rate if rate > 0 else 0
            log(f"  [{i + 1:>4}/{len(questions)}]  {elapsed:.0f}s elapsed  "
                f"({rate:.1f} q/s, ETA {eta/60:.1f}m)  last={dt:.1f}s")

    total_time = time.time() - t0

    # Cleanup
    shutil.rmtree(base_tmp, ignore_errors=True)

    # 8. Report
    log("")
    log(f"LongMemEval-S Benchmark Results (Smriti v{__version__}, {embed_label})")
    log(
        f"  mode={args.mode}  fetch_multiplier={base_config.fetch_multiplier}  "
        f"use_graph_in_hybrid={base_config.use_graph_in_hybrid}  rerank={args.rerank}"
    )
    log("=" * 64)
    header = f"{'Question Type':<28}| {'Count':>5} | {'R@5':>5} | {'R@10':>5} | {'R@20':>5}"
    log(header)
    log("-" * 28 + "|" + "-" * 7 + "|" + "-" * 7 + "|" + "-" * 7 + "|" + "-" * 7)

    # Deterministic ordering of question types
    type_order = [
        "single-session-user",
        "single-session-assistant",
        "single-session-preference",
        "temporal-reasoning",
        "knowledge-update",
        "multi-session",
    ]
    # Include any types not in the predefined order
    all_types = type_order + [t for t in sorted(type_counts) if t not in type_order]

    overall_counts = 0
    overall_hits: dict[int, int] = {k: 0 for k in K_VALUES}

    for qtype in all_types:
        if qtype not in type_counts:
            continue
        cnt = type_counts[qtype]
        overall_counts += cnt
        r_strs: list[str] = []
        for k in K_VALUES:
            h = type_hits[qtype][k]
            overall_hits[k] += h
            recall = h / cnt if cnt else 0.0
            r_strs.append(f"{recall:>5.2f}")
        log(f"{qtype:<28}| {cnt:>5} | {' | '.join(r_strs)}")

    log("-" * 28 + "|" + "-" * 7 + "|" + "-" * 7 + "|" + "-" * 7 + "|" + "-" * 7)
    overall_strs = []
    for k in K_VALUES:
        recall = overall_hits[k] / overall_counts if overall_counts else 0.0
        overall_strs.append(f"{recall:>5.2f}")
    log(f"{'OVERALL':<28}| {overall_counts:>5} | {' | '.join(overall_strs)}")
    log("")

    # Latency stats
    def percentile(data: list[float], p: int) -> float:
        if not data:
            return 0.0
        s = sorted(data)
        idx = int(len(s) * p / 100)
        return s[min(idx, len(s) - 1)]

    log("Latency")
    log("-" * 48)
    log(f"  Search  — P50: {percentile(search_latencies, 50):>7.1f}ms  "
        f"P99: {percentile(search_latencies, 99):>7.1f}ms  "
        f"Mean: {sum(search_latencies)/len(search_latencies):>7.1f}ms")
    log(f"  Add     — P50: {percentile(add_latencies, 50):>7.1f}ms  "
        f"P99: {percentile(add_latencies, 99):>7.1f}ms  "
        f"Mean: {sum(add_latencies)/len(add_latencies):>7.1f}ms")
    log("")
    log(f"Total time: {total_time:.1f}s  ({overall_counts / total_time:.1f} questions/sec)")


if __name__ == "__main__":
    main()
