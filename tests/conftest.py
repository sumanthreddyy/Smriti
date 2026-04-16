"""Pytest configuration for Smriti tests.

Uses HashEmbedding to avoid network downloads in CI/corporate environments.
"""

import pytest

from smriti.vectors import HashEmbedding

# Shared fast embedding function for all tests
_test_embedding = HashEmbedding(dim=384)


@pytest.fixture
def embedding_fn():
    """Provide a test-safe embedding function."""
    return _test_embedding
