"""ChromaDB vector store for semantic search.

Thin wrapper that handles embedding + similarity search.
Supports custom embedding functions or ChromaDB's default model.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.types import EmbeddingFunction

from smriti.types import MemoryEntry

# Stop words for key expansion keyword extraction
_KE_STOP = frozenset({
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
    "this", "that", "the", "a", "an", "and", "or", "but", "is", "are", "was",
    "were", "be", "been", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "not", "so", "if", "than", "too", "very",
    "just", "also", "now", "here", "there", "when", "where", "what", "which",
    "who", "how", "why", "all", "some", "any", "only", "into", "from", "with",
    "about", "for", "at", "by", "in", "of", "on", "to", "as", "can", "no",
    "yes", "then", "more", "most", "each", "both", "few", "such", "same",
    "hi", "hello", "hey", "thanks", "thank", "please", "ok", "okay",
    "user", "assistant", "sure", "right", "well", "like", "get", "got",
    "know", "think", "go", "going", "want", "need", "say", "said",
})


def extract_keywords(text: str, max_keywords: int = 20) -> list[str]:
    """Extract meaningful keywords from text for key expansion.

    Uses simple heuristics: remove stop words, keep unique meaningful words,
    prioritize capitalized words (likely entities) and longer words.
    """
    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_-]*\b', text)
    seen: set[str] = set()
    keywords: list[str] = []

    # First pass: capitalized words (likely entities/names)
    for w in words:
        wl = w.lower()
        if wl not in _KE_STOP and wl not in seen and len(wl) > 2:
            if w[0].isupper():
                seen.add(wl)
                keywords.append(w)

    # Second pass: remaining long words
    for w in words:
        wl = w.lower()
        if wl not in _KE_STOP and wl not in seen and len(wl) > 3:
            seen.add(wl)
            keywords.append(wl)

    return keywords[:max_keywords]


def expand_key(content: str) -> str:
    """Expand content with extracted keywords for better retrieval.

    Prepends extracted keywords to the original content so that
    both keyword-dense and full-context representations are indexed.
    """
    kws = extract_keywords(content)
    if not kws:
        return content
    return " | ".join(kws) + "\n" + content


class HashEmbedding(EmbeddingFunction):
    """Fast deterministic embedding using hashing.

    Works offline, zero downloads. Good for testing and environments
    where model downloads are blocked. Uses character n-gram hashing
    projected to a fixed-size vector with L2 normalization.
    """

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in input]

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        text_lower = text.lower()
        # Character trigram hashing (non-cryptographic use)
        for i in range(len(text_lower) - 2):
            trigram = text_lower[i : i + 3]
            h = int(hashlib.sha256(trigram.encode()).hexdigest(), 16)
            idx = h % self._dim
            val = ((h >> 16) % 1000) / 1000.0 - 0.5
            vec[idx] += val
        # Word-level hashing for broader semantic signal
        for word in text_lower.split():
            h = int(hashlib.sha256(word.encode()).hexdigest(), 16)
            idx = h % self._dim
            val = ((h >> 8) % 1000) / 1000.0 - 0.5
            vec[idx] += val
        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


# ── Supported models for load_embedding() ────────────────────────────────

_MODELS: dict[str, dict[str, Any]] = {
    "minilm": {
        "name": "all-MiniLM-L6-v2",
        "kwargs": {},
    },
    "bge-large": {
        "name": "BAAI/bge-large-en-v1.5",
        "kwargs": {},
    },
    "stella": {
        "name": "dunzhang/stella_en_400M_v5",
        "kwargs": {"trust_remote_code": True, "model_kwargs": {"attn_implementation": "eager"}},
    },
}


class _SentenceTransformerFn(EmbeddingFunction):
    """Wraps a SentenceTransformer model as a ChromaDB EmbeddingFunction."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                f"sentence-transformers is required for '{model_name}'. "
                "Install with: pip install smriti-mem[small]"
            )
        self._model = SentenceTransformer(model_name, **kwargs)

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._model.encode(input, normalize_embeddings=True).tolist()


def load_embedding(model: str) -> EmbeddingFunction:
    """Load a supported embedding model by name.

    Args:
        model: One of "minilm", "bge-large", "stella", or "hash".

    Returns:
        A ChromaDB-compatible EmbeddingFunction.

    Examples:
        >>> from smriti import Memory
        >>> from smriti.vectors import load_embedding
        >>> mem = Memory(embedding_fn=load_embedding("bge-large"))
    """
    if model == "hash":
        return HashEmbedding()

    if model not in _MODELS:
        raise ValueError(
            f"Unknown model '{model}'. Choose from: {', '.join(['hash'] + list(_MODELS))}"
        )

    info = _MODELS[model]
    return _SentenceTransformerFn(info["name"], **info["kwargs"])


# ── Cross-encoder rerankers ──────────────────────────────────────────────

_RERANKERS: dict[str, str] = {
    # 22 MB MiniLM cross-encoder, MIT-licensed, CPU-friendly.
    "ms-marco-minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    # Larger MS-MARCO CE, better quality, still CPU-feasible.
    "ms-marco-minilm-l12": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    # BGE multilingual reranker, larger but strong.
    "bge-reranker-base": "BAAI/bge-reranker-base",
}


def load_reranker(model: str = "ms-marco-minilm") -> Any:
    """Load a cross-encoder reranker as a plain callable.

    The returned callable has the signature:
        (query: str, pairs: list[tuple[str, str]]) -> list[tuple[str, float]]

    Where ``pairs`` is [(id, content), ...] and the output is
    [(id, ce_score), ...] in the same order as the input. Higher score is
    better. This is the contract consumed by ``Memory(reranker=...)``.

    Args:
        model: One of "ms-marco-minilm" (default, 22MB), "ms-marco-minilm-l12",
            or "bge-reranker-base". Custom HF repo IDs are also accepted.

    Examples:
        >>> from smriti import Memory
        >>> from smriti.vectors import load_reranker
        >>> mem = Memory(reranker=load_reranker())
    """
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Cross-encoder rerankers require the 'rerank' extra. "
            "Install with: pip install smriti-mem[rerank]"
        ) from exc

    model_name = _RERANKERS.get(model, model)
    ce = CrossEncoder(model_name)

    def _rerank(query: str, pairs: list[tuple[str, str]]) -> list[tuple[str, float]]:
        if not pairs:
            return []
        sentence_pairs = [[query, content] for _, content in pairs]
        scores = ce.predict(sentence_pairs, convert_to_numpy=True)
        return [(pairs[i][0], float(scores[i])) for i in range(len(pairs))]

    return _rerank


class VectorStore:
    """ChromaDB-backed semantic search."""

    def __init__(
        self,
        persist_dir: str | Path,
        collection_name: str,
        embedding_fn: Any | None = None,
        key_expansion: bool = True,
    ) -> None:
        self._key_expansion = key_expansion
        persist_path = Path(persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_path))
        # Probe caching: avoid a model download + upsert/delete round-trip on
        # every Memory() construction (matters on benchmarks where hundreds
        # of Memory instances are built in a single run).
        probe_marker = persist_path / ".smriti_default_embed_ok"

        # Use provided embedding function, or try default, fall back to hash
        if embedding_fn is None:
            try:
                self._collection = self._client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                if not probe_marker.exists():
                    # First open: verify the default embedder actually works.
                    self._collection.upsert(
                        ids=["__smriti_probe__"],
                        documents=["probe"],
                    )
                    self._collection.delete(ids=["__smriti_probe__"])
                    probe_marker.touch()
                return
            except (ImportError, RuntimeError, OSError):
                # Model download failed — fall back to hash embedding.
                embedding_fn = HashEmbedding()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=embedding_fn,
        )

    def add(self, entry: MemoryEntry) -> None:
        """Embed and store a memory entry."""
        doc = expand_key(entry.content) if self._key_expansion else entry.content
        self._collection.upsert(
            ids=[entry.id],
            documents=[doc],
            metadatas=[
                {
                    "memory_type": entry.memory_type,
                    "source": entry.source,
                    "archived": str(entry.archived),
                    "session_id": entry.session_id or "",
                }
            ],
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        include_archived: bool = False,
    ) -> list[tuple[str, float]]:
        """Semantic search. Returns [(id, similarity_score), ...].

        Score is 0.0-1.0 (cosine similarity converted from distance).
        """
        where_filter = None
        if not include_archived:
            where_filter = {"archived": "False"}

        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
            include=["distances"],
        )

        ids = results["ids"][0] if results["ids"] else []
        distances = results["distances"][0] if results["distances"] else []

        # ChromaDB cosine distance = 1 - similarity → convert back
        return [(id_, 1.0 - dist) for id_, dist in zip(ids, distances)]

    def delete(self, memory_id: str) -> None:
        """Remove from vector index (for hard deletes only)."""
        try:
            self._collection.delete(ids=[memory_id])
        except Exception:
            pass  # Already removed or never existed

    def update_metadata(self, memory_id: str, archived: bool) -> None:
        """Update archived status in vector store metadata."""
        try:
            self._collection.update(
                ids=[memory_id],
                metadatas=[{"archived": str(archived)}],
            )
        except Exception:
            pass

    def count(self) -> int:
        return self._collection.count()
