# Smriti स्मृति

[![CI](https://github.com/sumanthreddyy/Smriti/actions/workflows/ci.yml/badge.svg)](https://github.com/sumanthreddyy/Smriti/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/smriti-mem)](https://pypi.org/project/smriti-mem/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/smriti-mem)](https://pypi.org/project/smriti-mem/)

**The universal memory layer. Remember everything. Forget nothing. Understand context.**

Smriti (Sanskrit: स्मृति — "memory, remembrance") is a local-first AI memory system that gives any AI assistant perfect recall. It combines semantic search, keyword search, knowledge graphs, and temporal reasoning into one library.

```python
from smriti import Memory

mem = Memory()
mem.add("I prefer Postgres over MySQL")
results = mem.search("what database do I like?")
# → [SearchResult(content="I prefer Postgres over MySQL", score=0.82)]
```

## Why Smriti?

Most AI memory solutions are cloud-only, lossy (they delete data), or tied to a single framework. Smriti is different:

- **Local-first**: Your data stays on your machine. No API keys, no cloud, no vendor lock-in.
- **Non-lossy**: Nothing is ever deleted. Updates create version chains, deletes just archive.
- **Framework-agnostic**: Use it as a Python library, an MCP server, or both.
- **Actually fast**: HashEmbedding gives you functional search with zero model downloads in ~10ms.

## Table of Contents

- [Features](#features)
- [Benchmark](#benchmark--longmemeval-s-iclr-2025)
- [Install](#install)
- [Quick Start](#quick-start)
- [MCP Server](#mcp-server)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [Credits](#credits--inspiration)
- [License](#license)

## Features

- **Hybrid search**: Semantic (ChromaDB) + keyword (BM25/FTS5) + knowledge graph, fused via weighted RRF
- **Non-lossy**: Data is NEVER deleted. Deletes archive, updates create version chains
- **Local-first**: Everything runs on your machine. No cloud, no API keys required
- **Zero config**: `Memory()` just works. SQLite + ChromaDB, zero setup
- **Bi-temporal**: Memories have valid_from/valid_to, query any point in time
- **Knowledge graph**: Entity extraction + relationship tracking (NetworkX)
- **Memory types**: fact, belief, opinion, observation — each with confidence scores
- **Memory linking**: Bidirectional links, evolution chains, access tracking
- **MCP server**: 11 tools, plug into any AI assistant (VS Code Copilot, Claude Desktop, Cursor)
- **Embedding choice**: Built-in `load_embedding()` — pick your model at install time

## Benchmark — LongMemEval-S (ICLR 2025)

Tested on [LongMemEval](https://github.com/xiaowu0162/LongMemEval) — 470 questions, ~40 sessions each, 115k tokens of chat history per question.

### Recall by Embedding Model

| Embedding | R@5 | R@10 | R@20 | Search P50 | Install |
|-----------|-----|------|------|------------|---------|
| HashEmbedding (no model) | 0.77 | 0.87 | 0.95 | 10ms | `pip install smriti-mem` |
| MiniLM (86MB) | **0.90** | **0.96** | **0.99** | 170ms | `pip install smriti-mem[small]` |
| BGE-large (1.3GB) | 0.88 | 0.91 | 0.97 | — | `pip install smriti-mem[large]` |

### Detailed Breakdown (MiniLM)

```
Question Type               | Count |   R@5 |  R@10 |  R@20
----------------------------|-------|-------|-------|-------
single-session-user         |    64 |  0.78 |  0.92 |  0.98
single-session-assistant    |    56 |  0.79 |  0.89 |  0.98
single-session-preference   |    30 |  0.87 |  0.97 |  0.97
temporal-reasoning          |   127 |  0.91 |  0.98 |  1.00
knowledge-update            |    72 |  1.00 |  1.00 |  1.00
multi-session               |   121 |  0.97 |  0.98 |  1.00
----------------------------|-------|-------|-------|-------
OVERALL                     |   470 |  0.90 |  0.96 |  0.99
```

> **R@20 = 0.99** — Smriti finds the right memory 99% of the time with MiniLM.
> **knowledge-update = 1.00** — Perfect recall on facts that change over time.
> **HashEmbedding R@20 = 0.95** with zero model downloads — great for air-gapped or corporate environments.

## Install

```bash
pip install smriti-mem              # HashEmbedding (offline, no model download)
pip install smriti-mem[small]       # + MiniLM (86MB, recommended)
pip install smriti-mem[large]       # + BGE-large-en-v1.5 (1.3GB)
pip install smriti-mem[stella]      # + Stella V5 (3GB, GPU + xformers required)
```

Or from source:

```bash
git clone https://github.com/sumanthreddyy/Smriti.git
cd Smriti
pip install -e ".[dev]"
```

### Choosing an Embedding Model

```python
from smriti import Memory
from smriti.vectors import load_embedding

# Default — HashEmbedding (offline, zero downloads)
mem = Memory()

# MiniLM — best balance of speed and quality (recommended)
mem = Memory(embedding_fn=load_embedding("minilm"))

# BGE-large — higher quality, larger model
mem = Memory(embedding_fn=load_embedding("bge-large"))

# Stella V5 — top quality (requires GPU + xformers)
mem = Memory(embedding_fn=load_embedding("stella"))
```

## Quick Start

```python
from smriti import Memory

mem = Memory()

# Add memories with types
mem.add("PostgreSQL is our primary database", memory_type="fact")
mem.add("Redis might be overkill for our cache", memory_type="opinion")
mem.add("I think microservices are the way to go", memory_type="belief")

# Search (hybrid: semantic + keyword + graph)
results = mem.search("what database do we use?")
for r in results:
    print(f"[{r.score:.3f}] [{r.entry.memory_type}] {r.entry.content}")
    print(f"  Sources: {r.sources}")

# Temporal queries (v0.2)
mem.add("CEO is Kumar", valid_from="2020-01-01")
mem.add("CEO is Kumlee", valid_from="2024-01-01")
valid_2023 = mem.valid_at("2023-06-01")  # → Kumar only

# Memory evolution
old = mem.add("Uses MySQL")
new = mem.update(old.id, "Uses Postgres")  # old archived, linked
chain = mem.history(old.id)  # → [MySQL entry, Postgres entry]

# Knowledge graph (v0.3)
entry = mem.add("Kumar works at Valve", entities=["Kumar", "Valve"])
related = mem.related(entry.id)  # finds co-occurring entities

# Link memories (v0.5)
e1 = mem.add("Python is great for ML")
e2 = mem.add("TensorFlow is a Python ML framework")
mem.link(e1.id, e2.id)
linked = mem.get_linked(e1.id)  # → [TensorFlow entry]

# Filter by type (v0.4)
facts = mem.by_type("fact")
opinions = mem.by_type("opinion")
```

## MCP Server

Connect Smriti to any AI assistant (VS Code Copilot, Claude Desktop, Cursor):

```bash
python -m smriti.mcp_server
```

### VS Code Configuration

Add to your `settings.json`:

```json
{
    "mcp": {
        "servers": {
            "smriti": {
                "command": "python",
                "args": ["-m", "smriti.mcp_server"]
            }
        }
    }
}
```

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `add_memory` | Store a new memory |
| `search_memory` | Hybrid search (semantic + keyword + graph) |
| `get_memory` | Retrieve by ID |
| `update_memory` | Update (archives original) |
| `delete_memory` | Soft-delete (archive) |
| `list_memories` | List recent memories |
| `memory_stats` | Storage statistics |
| `temporal_search` | Find memories valid at a date |
| `memory_history` | Get evolution chain |
| `related_memories` | Find related via entities/links |
| `link_memories` | Link two memories |

## Architecture

```
┌─────────────────────────────────────────┐
│              Memory (API)               │
├──────────┬──────────┬───────────────────┤
│  Store   │ Vectors  │  KnowledgeGraph   │
│ SQLite   │ ChromaDB │  NetworkX         │
│ + FTS5   │ cosine   │  + SQLite edges   │
│ (BM25)   │ search   │  + entity extract │
├──────────┴──────────┴───────────────────┤
│        Hybrid Search (RRF Fusion)       │
│   semantic + bm25 + graph → ranked      │
├─────────────────────────────────────────┤
│           MCP Server (11 tools)         │
│     stdio / streamable-http transport   │
└─────────────────────────────────────────┘
```

## API Reference

### Memory Class

| Method | Description |
|--------|-------------|
| `add(content, *, memory_type, source, metadata, entities, confidence)` | Store a memory |
| `search(query, *, top_k, mode, include_archived)` | Hybrid search |
| `get(id)` | Get by ID (tracks access) |
| `update(id, content, *, keep_original)` | Update with version chain |
| `delete(id)` | Archive (soft-delete) |
| `valid_at(date_str)` | Temporal query |
| `history(id)` | Supersession chain |
| `related(id)` | Related via shared entities |
| `entities(id)` | Get extracted entities |
| `by_type(type)` | Filter by memory type |
| `link(id1, id2)` | Bidirectional link |
| `get_linked(id)` | Get linked memories |
| `list(*, limit, include_archived)` | List memories |
| `count(include_archived)` | Count memories |

### Search Modes

- `"hybrid"` (default) — Semantic + BM25 + Graph, fused with RRF
- `"semantic"` — ChromaDB cosine similarity only
- `"bm25"` — SQLite FTS5 keyword search only
- `"graph"` — Entity-based graph traversal only

### Tuning Search Weights

Hybrid search fuses results using [Reciprocal Rank Fusion (RRF)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf). Each source (semantic, bm25, graph) has a weight that controls its influence. Default is equal weights (1.0 each).

```python
from smriti import Memory
from smriti.config import SmritiConfig

# Default — equal weights, safe for any embedding
mem = Memory()

# Boost semantic for neural models (MiniLM, BGE, etc.)
mem = Memory(config=SmritiConfig(
    semantic_weight=1.5,
    bm25_weight=0.8,
    graph_weight=0.5,
))
```

> **Higher weight = that source counts more** in the final ranking. No retraining needed — takes effect immediately.

### Memory Types (v0.4)

- `"fact"` — Objective, verifiable information
- `"belief"` — Subjective conviction held as true
- `"opinion"` — Personal preference or judgment
- `"observation"` — Contextual note about a situation

Custom type strings are also accepted — Smriti stores whatever you pass.

## Dependencies

- **Python** ≥ 3.10
- **SQLite** (stdlib — FTS5 for BM25 keyword search)
- **ChromaDB** — Semantic vector search
- **NetworkX** — Knowledge graph traversal
- **Pydantic** — Data models
- **MCP SDK** — AI assistant connectivity (optional: `pip install smriti-mem[mcp]`)

## Limitations

- **Thread safety**: `Memory` holds a single SQLite connection and is not thread-safe. Use one instance per thread, or guard shared instances with a `threading.Lock`. This may matter if you're using Smriti from an async MCP server with multiple threads.
- **ID collisions**: Memory IDs are 12 hex chars (48-bit). Practically unique for single-user local use, but at millions of entries the birthday paradox applies. A future version may extend this.

## Credits & Inspiration

Smriti stands on the shoulders of these projects — each one influenced a specific design choice:

| Concept | Inspiration |
|---------|-------------|
| Bi-temporal facts | [Zep/Graphiti](https://github.com/getzep/graphiti) |
| 4 memory networks | [Hindsight](https://github.com/agiresearch/Hindsight) |
| Procedural memory | [LangMem](https://github.com/langchain-ai/langmem) |
| Memory evolution | [A-Mem](https://github.com/agiresearch/A-mem) (NeurIPS 2025) |
| Knowledge graph | [Mem0](https://github.com/mem0ai/mem0) |
| Hybrid RRF search | Hindsight TEMPR approach |
| Non-lossy archive, offline HashEmbedding | Original design |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, testing, and guidelines.

## License

Apache 2.0 — See [LICENSE](LICENSE) for details.
