"""
Tests for face recognition module
"""

import pytest
import numpy as np
from pathlib import Path
from face_matcher.core.recognition import FaceEmbeddingExtractor


class TestFaceEmbeddingExtractor:
    """Tests for FaceEmbeddingExtractor"""

    @pytest.fixture
    def extractor(self, project_root):
        """Create extractor instance"""
        model_path = project_root / "models/MobileFaceNet.onnx"
        if not model_path.exists():
            pytest.skip("Model file not found")
        return FaceEmbeddingExtractor(str(model_path))

    def test_extractor_initialization(self, extractor):
        """Test extractor initializes successfully"""
        assert extractor is not None
        assert extractor.input_size == (112, 112)

    def test_extract_embedding_shape(self, extractor, sample_image):
        """Test embedding has correct shape"""
        embedding = extractor.extract_embedding(sample_image)
        assert embedding.shape == (128,)
        assert embedding.dtype == np.float32

    def test_embedding_normalized(self, extractor, sample_image):
        """Test embedding is L2 normalized"""
        embedding = extractor.extract_embedding(sample_image)
        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0, atol=1e-5)
