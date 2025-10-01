"""
Face Recognition Module
Provides face embedding extraction using MobileFaceNet ONNX model
"""

import onnxruntime as ort
import numpy as np
import cv2
from typing import Optional, Tuple
import os


class FaceEmbeddingExtractor:
    """Extract face embeddings using MobileFaceNet ONNX model"""

    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize face embedding extractor

        Args:
            model_path: Path to MobileFaceNet ONNX model
            device: Device to run inference ('cpu' or 'cuda')
        """
        self.model_path = model_path
        self.device = device

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Initialize ONNX Runtime session
        self._init_model()

    def _init_model(self):
        """Initialize ONNX Runtime session"""
        # Set up execution providers
        providers = ['CPUExecutionProvider']
        if self.device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        # Create session
        self.session = ort.InferenceSession(self.model_path, providers=providers)

        # Get input/output details
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

        # Determine input format and size
        shape = [dim if isinstance(dim, int) else 1 for dim in self.input_shape]

        if len(shape) == 4:
            if shape[1] == 3 or shape[1] == 1:  # CHW format
                self.input_format = 'CHW'
                self.input_size = (shape[2], shape[3])  # (height, width)
            elif shape[3] == 3 or shape[3] == 1:  # HWC format
                self.input_format = 'HWC'
                self.input_size = (shape[1], shape[2])  # (height, width)
            else:
                # Infer from shape
                if shape[1] > shape[3]:
                    self.input_format = 'HWC'
                    self.input_size = (shape[1], shape[2])
                else:
                    self.input_format = 'CHW'
                    self.input_size = (shape[2], shape[3])
        else:
            raise ValueError(f"Unsupported input shape: {self.input_shape}")

        print(f"Model loaded: {os.path.basename(self.model_path)}")
        print(f"  Input shape: {self.input_shape}")
        print(f"  Input format: {self.input_format}")
        print(f"  Input size: {self.input_size}")
        print(f"  Providers: {self.session.get_providers()}")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model inference

        Args:
            image: Input image (BGR or RGB format)

        Returns:
            Preprocessed image array ready for inference
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume input is BGR from OpenCV
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # Resize to model input size
        image_resized = cv2.resize(image_rgb, self.input_size)

        # Normalize to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0

        # Apply model-specific normalization ([-1, 1] for MobileFaceNet)
        image_normalized = (image_normalized - 0.5) / 0.5

        # Handle input format (CHW or HWC)
        if self.input_format == 'CHW':
            # Transpose from HWC to CHW
            image_normalized = np.transpose(image_normalized, (2, 0, 1))

        # Add batch dimension
        image_batch = np.expand_dims(image_normalized, axis=0)

        return image_batch

    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Extract face embedding from aligned face image

        Args:
            image: Aligned face image (BGR or RGB format)

        Returns:
            Face embedding vector (L2 normalized)
        """
        # Preprocess image
        input_data = self.preprocess_image(image)

        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_data})

        # Get embedding (remove batch dimension)
        embedding = outputs[0][0]

        # L2 normalize
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def extract_embedding_from_path(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract face embedding from image file path

        Args:
            image_path: Path to aligned face image

        Returns:
            Face embedding vector or None if image cannot be loaded
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None

        # Extract embedding
        return self.extract_embedding(image)

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings

        Args:
            embedding1: First face embedding
            embedding2: Second face embedding

        Returns:
            Cosine similarity score (0-1, higher means more similar)
        """
        # For L2-normalized vectors, cosine similarity = dot product
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)

    def compare_faces(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        threshold: float = 0.5
    ) -> Tuple[float, bool]:
        """
        Compare two face images

        Args:
            image1: First aligned face image
            image2: Second aligned face image
            threshold: Similarity threshold for matching

        Returns:
            Tuple of (similarity_score, is_same_person)
        """
        # Extract embeddings
        embedding1 = self.extract_embedding(image1)
        embedding2 = self.extract_embedding(image2)

        # Compute similarity
        similarity = self.compute_similarity(embedding1, embedding2)

        # Determine if same person
        is_same_person = similarity > threshold

        return similarity, is_same_person

    def get_embedding_dim(self) -> int:
        """
        Get embedding dimension

        Returns:
            Embedding dimension
        """
        # Get output shape from model
        output_shape = self.session.get_outputs()[0].shape
        # Typically the last dimension is the embedding dimension
        return output_shape[-1] if isinstance(output_shape[-1], int) else 128
