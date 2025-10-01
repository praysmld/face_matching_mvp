"""
Tests for vector database module
"""

import pytest
import numpy as np
from face_matcher.core.database import VectorDatabase


class TestVectorDatabase:
    """Tests for VectorDatabase"""

    @pytest.fixture
    def db(self, tmp_path):
        """Create temporary database for testing"""
        db_path = str(tmp_path / "test_milvus.db")
        db = VectorDatabase(db_path=db_path, collection_name="test_faces")
        yield db
        db.close()

    def test_database_initialization(self, db):
        """Test database initializes successfully"""
        assert db is not None

    def test_create_collection(self, db):
        """Test collection creation"""
        db.create_collection(drop_existing=True)
        assert db.collection is not None

    def test_insert_and_search(self, db, sample_embedding):
        """Test inserting and searching embeddings"""
        db.create_collection(drop_existing=True)
        db.create_index()
        db.load_collection()

        # Insert sample data
        names = ["person1", "person2"]
        image_paths = ["/path/to/crop1.jpg", "/path/to/crop2.jpg"]
        original_paths = ["/path/to/orig1.jpg", "/path/to/orig2.jpg"]
        embeddings = np.array([sample_embedding, sample_embedding * 0.9], dtype=np.float32)

        num_inserted = db.insert_batch(names, image_paths, original_paths, embeddings)
        assert num_inserted == 2

        # Search
        results = db.search(sample_embedding, top_k=2)
        assert len(results) > 0
        assert "name" in results[0]
        assert "similarity" in results[0]
