"""
Pytest configuration and fixtures for Face Matcher tests
"""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def project_root():
    """Get project root directory"""
    return Path(__file__).parent.parent


@pytest.fixture
def sample_image():
    """Create a dummy image for testing"""
    return np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)


@pytest.fixture
def sample_embedding():
    """Create a dummy embedding for testing"""
    embedding = np.random.randn(128).astype(np.float32)
    return embedding / np.linalg.norm(embedding)  # L2 normalized


@pytest.fixture
def config_override(monkeypatch):
    """Override configuration for testing"""
    from face_matcher.config import config

    # Override with test values
    monkeypatch.setattr(config.database, "db_path", "./data/test_milvus.db")
    monkeypatch.setattr(config.database, "collection_name", "test_faces")

    return config
