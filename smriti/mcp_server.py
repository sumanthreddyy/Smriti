"""Smriti MCP Server — connect memory to any AI assistant.

Works with: VS Code Copilot, Claude Desktop, Cursor, any MCP client.

Run:
    python -m smriti.mcp_server                     # stdio (default)
    python -m smriti.mcp_server --transport http     # streamable HTTP

Configure in VS Code (settings.json → mcp):
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
"""

from __future__ import annotations

import json
import sys

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    raise ImportError(
        "MCP support requires the 'mcp' extra. "
        "Install with: pip install smriti-mem[mcp]"
    )

from smriti.memory import Memory

# Initialize
mcp = FastMCP(
    "Smriti",
    instructions=(
        "Smriti is a universal memory layer. Use these tools to store and "
        "retrieve information across conversations. Memories persist forever "
        "and are never deleted — only archived."
    ),
)

# Lazy-init Memory instance (created on first tool call)
_memory: Memory | None = None


def _get_memory() -> Memory:
    global _memory
    if _memory is None:
        _memory = Memory()
    return _memory


# ─── Tools ────────────────────────────────────────────────────────────────

@mcp.tool()
def add_memory(
    content: str,
    memory_type: str = "fact",
    metadata: str = "{}",
) -> str:
    """Store a new memory. Types: fact, belief, opinion, observation.

    Args:
        content: The information to remember.
        memory_type: One of: fact, belief, opinion, observation.
        metadata: Optional JSON string of extra metadata.
    """
    mem = _get_memory()
    try:
        meta = json.loads(metadata) if metadata else {}
    except json.JSONDecodeError:
        meta = {}
    entry = mem.add(content, memory_type=memory_type, metadata=meta)
    return json.dumps({
        "status": "stored",
        "id": entry.id,
        "content": entry.content,
        "type": entry.memory_type,
    })


@mcp.tool()
def search_memory(
    query: str,
    top_k: int = 5,
    mode: str = "hybrid",
) -> str:
    """Search memories using hybrid retrieval (semantic + keyword).

    Args:
        query: Natural language search query.
        top_k: Maximum number of results (default 5).
        mode: Search mode — hybrid, semantic, or bm25.
    """
    mem = _get_memory()
    results = mem.search(query, top_k=top_k, mode=mode)
    if not results:
        return json.dumps({"results": [], "message": "No memories found."})
    return json.dumps({
        "results": [
            {
                "content": r.entry.content,
                "type": r.entry.memory_type,
                "score": round(r.score, 4),
                "sources": r.sources,
                "id": r.entry.id,
                "created_at": r.entry.created_at,
            }
            for r in results
        ]
    })


@mcp.tool()
def get_memory(memory_id: str) -> str:
    """Retrieve a specific memory by its ID.

    Args:
        memory_id: The unique identifier of the memory.
    """
    mem = _get_memory()
    entry = mem.get(memory_id)
    if not entry:
        return json.dumps({"error": "Memory not found", "id": memory_id})
    return json.dumps({
        "id": entry.id,
        "content": entry.content,
        "type": entry.memory_type,
        "archived": entry.archived,
        "created_at": entry.created_at,
        "metadata": entry.metadata,
    })


@mcp.tool()
def update_memory(memory_id: str, new_content: str) -> str:
    """Update a memory. The original is archived (never deleted).

    Args:
        memory_id: ID of the memory to update.
        new_content: The new content to replace it with.
    """
    mem = _get_memory()
    new_entry = mem.update(memory_id, new_content)
    if not new_entry:
        return json.dumps({"error": "Memory not found", "id": memory_id})
    return json.dumps({
        "status": "updated",
        "old_id": memory_id,
        "new_id": new_entry.id,
        "content": new_entry.content,
        "note": "Original memory archived, not deleted.",
    })


@mcp.tool()
def delete_memory(memory_id: str) -> str:
    """Archive a memory (soft-delete). Data is NEVER removed.

    Args:
        memory_id: ID of the memory to archive.
    """
    mem = _get_memory()
    success = mem.delete(memory_id)
    return json.dumps({
        "status": "archived" if success else "not_found",
        "id": memory_id,
        "note": "Memory archived, still recoverable." if success else "Memory ID not found.",
    })


@mcp.tool()
def list_memories(limit: int = 20, include_archived: bool = False) -> str:
    """List recent memories.

    Args:
        limit: Maximum number of memories to return (default 20).
        include_archived: Whether to include archived memories.
    """
    mem = _get_memory()
    entries = mem.list(limit=limit, include_archived=include_archived)
    return json.dumps({
        "count": len(entries),
        "total": mem.count(include_archived=include_archived),
        "memories": [
            {
                "id": e.id,
                "content": e.content[:200],
                "type": e.memory_type,
                "archived": e.archived,
                "created_at": e.created_at,
            }
            for e in entries
        ],
    })


@mcp.tool()
def temporal_search(date: str, query: str = "") -> str:
    """Find memories valid at a specific date.

    Args:
        date: ISO date string (e.g. "2026-01-15").
        query: Optional text filter on results.
    """
    mem = _get_memory()
    entries = mem.valid_at(date)
    if query:
        entries = [e for e in entries if query.lower() in e.content.lower()]
    return json.dumps({
        "date": date,
        "count": len(entries),
        "memories": [
            {
                "id": e.id,
                "content": e.content[:200],
                "type": e.memory_type,
                "valid_from": e.valid_from,
                "valid_to": e.valid_to,
            }
            for e in entries[:20]
        ],
    })


@mcp.tool()
def memory_history(memory_id: str) -> str:
    """Get the evolution chain of a memory (how it changed over time).

    Args:
        memory_id: ID of any memory in the chain.
    """
    mem = _get_memory()
    chain = mem.history(memory_id)
    return json.dumps({
        "chain_length": len(chain),
        "chain": [
            {
                "id": e.id,
                "content": e.content,
                "version": e.version,
                "archived": e.archived,
                "superseded_by": e.superseded_by,
                "created_at": e.created_at,
            }
            for e in chain
        ],
    })


@mcp.tool()
def related_memories(memory_id: str) -> str:
    """Find memories related through shared entities or explicit links.

    Args:
        memory_id: ID of the memory to find relations for.
    """
    mem = _get_memory()
    related = mem.related(memory_id)
    linked = mem.get_linked(memory_id)
    entities = mem.entities(memory_id)
    all_related = {e.id: e for e in related + linked}
    return json.dumps({
        "memory_id": memory_id,
        "entities": entities,
        "related_count": len(all_related),
        "related": [
            {
                "id": e.id,
                "content": e.content[:200],
                "type": e.memory_type,
            }
            for e in all_related.values()
        ],
    })


@mcp.tool()
def link_memories(id1: str, id2: str) -> str:
    """Create a bidirectional link between two memories.

    Args:
        id1: First memory ID.
        id2: Second memory ID.
    """
    mem = _get_memory()
    success = mem.link(id1, id2)
    return json.dumps({
        "status": "linked" if success else "failed",
        "id1": id1,
        "id2": id2,
    })


@mcp.tool()
def memory_stats() -> str:
    """Get memory statistics."""
    mem = _get_memory()
    active = mem.count(include_archived=False)
    total = mem.count(include_archived=True)
    return json.dumps({
        "active_memories": active,
        "archived_memories": total - active,
        "total_memories": total,
        "storage_path": mem.path,
    })


# ─── Resources ────────────────────────────────────────────────────────────

@mcp.resource("smriti://stats")
def stats_resource() -> str:
    """Current memory statistics."""
    return memory_stats()


# ─── Entry point ──────────────────────────────────────────────────────────

def main():
    transport = "stdio"
    if "--transport" in sys.argv:
        idx = sys.argv.index("--transport")
        if idx + 1 < len(sys.argv):
            transport = sys.argv[idx + 1]

    if transport == "http":
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
