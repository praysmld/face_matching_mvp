"""
Face Detection Module
Provides face detection and alignment using RetinaFace and Haar Cascade
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from abc import ABC, abstractmethod

from uniface import RetinaFace, face_alignment
from uniface.constants import RetinaFaceWeights


class FaceDetector(ABC):
    """Abstract base class for face detectors"""

    @abstractmethod
    def detect_and_align(self, image: np.ndarray, output_size: int = 112) -> Optional[np.ndarray]:
        """
        Detect and align face from image

        Args:
            image: Input image (BGR or RGB format)
            output_size: Size of output aligned face

        Returns:
            Aligned face image or None if no face detected
        """
        pass

    @abstractmethod
    def detect_and_crop(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect face and return both aligned and original crop

        Args:
            image: Input image (BGR or RGB format)

        Returns:
            Tuple of (aligned_face, original_crop) or (None, None) if no face detected
        """
        pass


class RetinaFaceDetector(FaceDetector):
    """Face detector using RetinaFace from uniface library"""

    def __init__(
        self,
        model_name: str = RetinaFaceWeights.MNET_V2,
        conf_thresh: float = 0.5,
        nms_thresh: float = 0.4
    ):
        """
        Initialize RetinaFace detector

        Args:
            model_name: RetinaFace model name from uniface
            conf_thresh: Confidence threshold for detection
            nms_thresh: NMS threshold for detection
        """
        self.detector = RetinaFace(
            model_name=model_name,
            conf_thresh=conf_thresh,
            pre_nms_topk=5000,
            nms_thresh=nms_thresh,
            post_nms_topk=750
        )

    def detect_and_align(self, image: np.ndarray, output_size: int = 112) -> Optional[np.ndarray]:
        """
        Detect and align face using RetinaFace

        Args:
            image: Input image (BGR or RGB format)
            output_size: Size of output aligned face

        Returns:
            Aligned face image or None if no face detected
        """
        try:
            # Detect faces and landmarks
            boxes, landmarks = self.detector.detect(image)

            if len(landmarks) == 0:
                return None

            # Use first detected face
            landmark_array = landmarks[0]

            # Align face using uniface alignment
            aligned_face, _ = face_alignment(image, landmark_array, image_size=output_size)

            return aligned_face

        except Exception as e:
            print(f"Error in RetinaFace detection: {e}")
            return None

    def detect_and_crop(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect face and return both aligned and original crop

        Args:
            image: Input image (BGR or RGB format)

        Returns:
            Tuple of (aligned_face, original_crop) or (None, None) if no face detected
        """
        try:
            # Detect faces and landmarks
            boxes, landmarks = self.detector.detect(image)

            if len(landmarks) == 0 or len(boxes) == 0:
                return None, None

            # Use first detected face
            landmark_array = landmarks[0]
            box = boxes[0]

            # Align face using uniface alignment
            aligned_face, _ = face_alignment(image, landmark_array, image_size=112)

            # Crop original face using bounding box
            x1, y1, x2, y2 = map(int, box[:4])
            # Add some padding
            h, w = image.shape[:2]
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)

            original_crop = image[y1:y2, x1:x2]

            return aligned_face, original_crop

        except Exception as e:
            print(f"Error in RetinaFace detection: {e}")
            return None, None


class HaarCascadeDetector(FaceDetector):
    """Face detector using OpenCV Haar Cascade"""

    def __init__(self, cascade_path: Optional[str] = None):
        """
        Initialize Haar Cascade detector

        Args:
            cascade_path: Path to cascade XML file (uses default if None)
        """
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

        self.detector = cv2.CascadeClassifier(cascade_path)

        if self.detector.empty():
            raise ValueError(f"Failed to load Haar Cascade from {cascade_path}")

    def detect_and_align(self, image: np.ndarray, output_size: int = 112) -> Optional[np.ndarray]:
        """
        Detect and align face using Haar Cascade

        Args:
            image: Input image (BGR or RGB format)
            output_size: Size of output aligned face

        Returns:
            Aligned face image or None if no face detected
        """
        try:
            # Convert to grayscale for detection
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Detect faces
            faces = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            if len(faces) == 0:
                return None

            # Use first detected face
            x, y, w, h = faces[0]

            # Estimate 5-point facial landmarks
            # These are rough estimates based on typical face proportions
            landmarks_5pt = np.array([
                [x + w * 0.3, y + h * 0.35],   # Left eye
                [x + w * 0.7, y + h * 0.35],   # Right eye
                [x + w * 0.5, y + h * 0.55],   # Nose tip
                [x + w * 0.35, y + h * 0.75],  # Left mouth corner
                [x + w * 0.65, y + h * 0.75]   # Right mouth corner
            ], dtype=np.float32)

            # Align face using uniface alignment
            aligned_face, _ = face_alignment(image, landmarks_5pt, image_size=output_size)

            return aligned_face

        except Exception as e:
            print(f"Error in Haar Cascade detection: {e}")
            return None

    def detect_and_crop(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect face and return both aligned and original crop

        Args:
            image: Input image (BGR or RGB format)

        Returns:
            Tuple of (aligned_face, original_crop) or (None, None) if no face detected
        """
        try:
            # Convert to grayscale for detection
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Detect faces
            faces = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            if len(faces) == 0:
                return None, None

            # Use first detected face
            x, y, w, h = faces[0]

            # Estimate 5-point facial landmarks
            landmarks_5pt = np.array([
                [x + w * 0.3, y + h * 0.35],   # Left eye
                [x + w * 0.7, y + h * 0.35],   # Right eye
                [x + w * 0.5, y + h * 0.55],   # Nose tip
                [x + w * 0.35, y + h * 0.75],  # Left mouth corner
                [x + w * 0.65, y + h * 0.75]   # Right mouth corner
            ], dtype=np.float32)

            # Align face using uniface alignment
            aligned_face, _ = face_alignment(image, landmarks_5pt, image_size=112)

            # Crop original face using bounding box with padding
            h_img, w_img = image.shape[:2]
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w_img, x + w + padding)
            y2 = min(h_img, y + h + padding)

            original_crop = image[y1:y2, x1:x2]

            return aligned_face, original_crop

        except Exception as e:
            print(f"Error in Haar Cascade detection: {e}")
            return None, None


class FaceDetectorFactory:
    """Factory class to create face detectors"""

    @staticmethod
    def create_detector(detector_type: str) -> FaceDetector:
        """
        Create face detector instance

        Args:
            detector_type: Type of detector ("retinaface" or "haarcascade")

        Returns:
            FaceDetector instance

        Raises:
            ValueError: If detector type is not supported
        """
        detector_type = detector_type.lower()

        if detector_type == "retinaface":
            return RetinaFaceDetector()
        elif detector_type == "haarcascade":
            return HaarCascadeDetector()
        else:
            raise ValueError(f"Unsupported detector type: {detector_type}")
