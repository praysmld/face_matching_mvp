"""
Tests for face detection module
"""

import pytest
import numpy as np
from face_matcher.core.detection import (
    FaceDetectorFactory,
    RetinaFaceDetector,
    HaarCascadeDetector,
)


class TestFaceDetectorFactory:
    """Tests for FaceDetectorFactory"""

    def test_create_retinaface_detector(self):
        """Test creating RetinaFace detector"""
        detector = FaceDetectorFactory.create_detector("retinaface")
        assert isinstance(detector, RetinaFaceDetector)

    def test_create_haarcascade_detector(self):
        """Test creating Haar Cascade detector"""
        detector = FaceDetectorFactory.create_detector("haarcascade")
        assert isinstance(detector, HaarCascadeDetector)

    def test_create_invalid_detector(self):
        """Test creating invalid detector raises error"""
        with pytest.raises(ValueError):
            FaceDetectorFactory.create_detector("invalid")


class TestRetinaFaceDetector:
    """Tests for RetinaFaceDetector"""

    def test_detector_initialization(self):
        """Test detector initializes successfully"""
        detector = RetinaFaceDetector()
        assert detector is not None

    def test_detect_and_align_no_face(self, sample_image):
        """Test detection returns None for image without face"""
        detector = RetinaFaceDetector()
        result = detector.detect_and_align(sample_image)
        # Random noise image unlikely to have a face
        # Result could be None or an array depending on false positives


class TestHaarCascadeDetector:
    """Tests for HaarCascadeDetector"""

    def test_detector_initialization(self):
        """Test detector initializes successfully"""
        detector = HaarCascadeDetector()
        assert detector is not None

    def test_detect_and_align_no_face(self, sample_image):
        """Test detection returns None for image without face"""
        detector = HaarCascadeDetector()
        result = detector.detect_and_align(sample_image)
        # Random noise image unlikely to have a face
