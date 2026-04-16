"""Smriti — The universal memory layer.

Remember everything. Forget nothing. Understand context.

Usage:
    from smriti import Memory

    mem = Memory()
    mem.add("I prefer Postgres over MySQL")
    results = mem.search("what database do I like?")
"""

from smriti.memory import Memory
from smriti.types import MemoryEntry, SearchResult

__version__ = "0.6.0"
__all__ = ["Memory", "MemoryEntry", "SearchResult"]
