"""
Configuration Module
Centralized configuration for Face Matching MVP
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# Project root directory (go up 3 levels: config.py -> face_matcher -> src -> root)
PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class ModelConfig:
    """Model configuration"""
    model_path: str = str(PROJECT_ROOT / "models/MobileFaceNet.onnx")
    device: str = "cpu"  # "cpu" or "cuda"
    embedding_dim: int = 128


@dataclass
class DetectorConfig:
    """Face detector configuration"""
    default_detector: str = "retinaface"  # "retinaface" or "haarcascade"
    retinaface_conf_thresh: float = 0.5
    retinaface_nms_thresh: float = 0.4
    face_output_size: int = 112  # Aligned face output size


@dataclass
class DatabaseConfig:
    """Database configuration"""
    db_path: str = str(PROJECT_ROOT / "data" / "milvus_face_data.db")
    collection_name: str = "face_embeddings"
    embedding_dim: int = 128
    index_type: str = "IVF_FLAT"
    nlist: int = 128  # Number of clusters for IVF index
    nprobe: int = 10  # Number of clusters to search


@dataclass
class DataConfig:
    """Data pipeline configuration"""
    csv_path: str = str(PROJECT_ROOT / "facescrub_metadata.csv")
    output_dir: str = str(PROJECT_ROOT / "data")
    download_dir: str = str(PROJECT_ROOT / "data" / "downloads")
    aligned_dir: str = str(PROJECT_ROOT / "data" / "aligned_faces")
    batch_size: int = 100
    max_images: Optional[int] = None
    skip_existing: bool = True
    download_timeout: int = 10


@dataclass
class AppConfig:
    """Application configuration"""
    host: str = "0.0.0.0"
    port: int = 7860
    share: bool = False
    similarity_threshold: float = 0.6  # Threshold for duplicate detection
    top_k_results: int = 6  # Number of similar faces to return


@dataclass
class Config:
    """Main configuration object"""
    model: ModelConfig = field(default_factory=ModelConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    data: DataConfig = field(default_factory=DataConfig)
    app: AppConfig = field(default_factory=AppConfig)

    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        config = cls()
        dirs = [
            Path(config.data.output_dir),
            Path(config.data.download_dir),
            Path(config.data.aligned_dir),
            Path(config.database.db_path).parent,
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = Config()


def validate_config():
    """Validate configuration"""
    errors = []

    # Check model file exists
    if not os.path.exists(config.model.model_path):
        errors.append(f"Model file not found: {config.model.model_path}")

    # Check CSV file exists
    if not os.path.exists(config.data.csv_path):
        errors.append(f"CSV file not found: {config.data.csv_path}")

    # Validate device
    if config.model.device not in ["cpu", "cuda"]:
        errors.append(f"Invalid device: {config.model.device}")

    # Validate detector
    if config.detector.default_detector not in ["retinaface", "haarcascade"]:
        errors.append(f"Invalid detector: {config.detector.default_detector}")

    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

    return True


if __name__ == "__main__":
    # Test configuration
    print("Configuration:")
    print(f"  Model: {config.model.model_path}")
    print(f"  Database: {config.database.db_path}")
    print(f"  CSV: {config.data.csv_path}")
    print(f"  Detector: {config.detector.default_detector}")
    print(f"  Threshold: {config.app.similarity_threshold}")

    # Create directories
    Config.create_directories()
    print("\nDirectories created successfully")

    # Validate
    try:
        validate_config()
        print("\nConfiguration valid ✓")
    except ValueError as e:
        print(f"\nConfiguration errors:\n{e}")
