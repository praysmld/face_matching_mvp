"""
Data Preparation Pipeline
Downloads images, detects faces, extracts embeddings, and populates Milvus database
"""

import os
import csv
import requests
import cv2
import numpy as np
import argparse
import logging
from typing import List, Tuple
from tqdm import tqdm
import time

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from face_matcher.config import config, Config
from face_matcher.core.detection import FaceDetectorFactory
from face_matcher.core.recognition import FaceEmbeddingExtractor
from face_matcher.core.database import VectorDatabase

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_image(url: str, save_path: str, timeout: int = 10) -> bool:
    """
    Download image from URL

    Args:
        url: Image URL
        save_path: Path to save the image
        timeout: Request timeout in seconds

    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return True
    except Exception as e:
        logger.debug(f"Failed to download {url}: {e}")
        return False


def process_dataset(
    csv_path: str,
    output_dir: str,
    detector_type: str = "retinaface",
    max_images: int = None,
    skip_existing: bool = True
) -> Tuple[List[str], List[str], List[str], List[np.ndarray]]:
    """
    Process dataset: download images and detect faces

    Args:
        csv_path: Path to metadata CSV file
        output_dir: Directory to save aligned face crops
        detector_type: "retinaface" or "haarcascade"
        max_images: Maximum number of images to process
        skip_existing: Skip if aligned face already exists

    Returns:
        Tuple of (names, image_paths, original_paths, aligned_faces)
    """
    # Create output directories
    download_dir = os.path.join(output_dir, "downloads")
    aligned_dir = os.path.join(output_dir, "aligned_faces")
    cropped_dir = os.path.join(output_dir, "cropped_faces")  # Original face crops
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(aligned_dir, exist_ok=True)
    os.makedirs(cropped_dir, exist_ok=True)

    # Initialize face detector
    logger.info(f"Initializing {detector_type} detector...")
    detector = FaceDetectorFactory.create_detector(detector_type)

    # Read CSV
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if max_images:
        rows = rows[:max_images]

    logger.info(f"Processing {len(rows)} images with {detector_type} detector...")

    names = []
    image_paths = []
    original_paths = []
    aligned_faces = []

    for row in tqdm(rows, desc="Processing images"):
        name = row['name']
        image_id = row['image_id']
        url = row['url']

        # Create filenames
        download_filename = f"{name}_{image_id}.jpg".replace(" ", "_")
        download_path = os.path.join(download_dir, download_filename)

        aligned_filename = f"{name}_{image_id}_aligned.jpg".replace(" ", "_")
        aligned_path = os.path.join(aligned_dir, aligned_filename)

        cropped_filename = f"{name}_{image_id}_crop.jpg".replace(" ", "_")
        cropped_path = os.path.join(cropped_dir, cropped_filename)

        # Skip if both aligned and cropped faces already exist
        if skip_existing and os.path.exists(aligned_path) and os.path.exists(cropped_path) and os.path.exists(download_path):
            aligned_face = cv2.imread(aligned_path)
            if aligned_face is not None:
                names.append(name)
                image_paths.append(cropped_path)  # Store path to cropped face
                original_paths.append(download_path)  # Store path to original download
                aligned_faces.append(aligned_face)
            continue

        # Download image if not exists
        if not os.path.exists(download_path):
            if not download_image(url, download_path):
                continue
            time.sleep(0.1)  # Rate limiting

        # Load image
        image = cv2.imread(download_path)
        if image is None:
            continue

        # Detect and get both aligned and original crop
        aligned_face, original_crop = detector.detect_and_crop(image)
        if aligned_face is None or original_crop is None:
            continue

        # Save both versions
        cv2.imwrite(aligned_path, aligned_face)  # For embedding extraction
        cv2.imwrite(cropped_path, original_crop)  # For display in gallery

        # Store results
        names.append(name)
        image_paths.append(cropped_path)  # Store path to original crop for gallery
        original_paths.append(download_path)  # Store path to original download
        aligned_faces.append(aligned_face)

    logger.info(f"Successfully processed {len(names)} faces")

    return names, image_paths, original_paths, aligned_faces


def extract_embeddings(
    aligned_faces: List[np.ndarray],
    model_path: str
) -> np.ndarray:
    """
    Extract face embeddings using MobileFaceNet

    Args:
        aligned_faces: List of aligned face images
        model_path: Path to ONNX model

    Returns:
        Numpy array of embeddings (N x embedding_dim)
    """
    logger.info("Extracting embeddings...")

    # Initialize embedding extractor
    extractor = FaceEmbeddingExtractor(model_path)

    embeddings = []
    for aligned_face in tqdm(aligned_faces, desc="Extracting embeddings"):
        embedding = extractor.extract_embedding(aligned_face)
        embeddings.append(embedding)

    embeddings_array = np.array(embeddings, dtype=np.float32)
    logger.info(f"Extracted {len(embeddings_array)} embeddings with shape {embeddings_array.shape}")

    return embeddings_array


def populate_database(
    names: List[str],
    image_paths: List[str],
    original_paths: List[str],
    embeddings: np.ndarray,
    db_path: str,
    collection_name: str,
    batch_size: int = 100,
    reset_database: bool = False
):
    """
    Populate Milvus database with face embeddings

    Args:
        names: List of person names
        image_paths: List of cropped face image paths
        original_paths: List of original downloaded image paths
        embeddings: Numpy array of embeddings
        db_path: Path to Milvus database
        collection_name: Name of collection
        batch_size: Batch size for insertion
        reset_database: If True, drop existing collection and create new one
    """
    logger.info("Populating Milvus database...")

    # Initialize vector database
    db = VectorDatabase(db_path=db_path, collection_name=collection_name)

    # Create collection (drop existing if reset_database=True)
    db.create_collection(drop_existing=reset_database)

    # Create index if doesn't exist
    db.create_index()

    # Load collection
    db.load_collection()

    # Insert embeddings in batches
    total_inserted = 0
    for i in range(0, len(names), batch_size):
        batch_names = names[i:i+batch_size]
        batch_paths = image_paths[i:i+batch_size]
        batch_original_paths = original_paths[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]

        num_inserted = db.insert_batch(batch_names, batch_paths, batch_original_paths, batch_embeddings)
        total_inserted += num_inserted

    # Get stats
    stats = db.get_stats()
    logger.info(f"Database populated successfully!")
    logger.info(f"Total entities: {stats['num_entities']}")

    db.close()


def main():
    parser = argparse.ArgumentParser(
        description='Prepare Face Matching Database',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--csv',
        type=str,
        default=config.data.csv_path,
        help='Path to metadata CSV file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=config.data.output_dir,
        help='Output directory for processed images'
    )
    parser.add_argument(
        '--detector',
        type=str,
        choices=['retinaface', 'haarcascade'],
        default=config.detector.default_detector,
        help='Face detector to use'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=config.model.model_path,
        help='Path to ONNX model'
    )
    parser.add_argument(
        '--db_path',
        type=str,
        default=config.database.db_path,
        help='Path to Milvus database'
    )
    parser.add_argument(
        '--max_images',
        type=int,
        default=config.data.max_images,
        help='Maximum number of images to process'
    )
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        default=config.data.skip_existing,
        help='Skip processing if aligned face already exists'
    )
    parser.add_argument(
        '--reset_database',
        action='store_true',
        default=False,
        help='Reset database by dropping existing collection and creating new one'
    )

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("Face Matching MVP - Database Preparation")
    logger.info("="*60)

    # Create necessary directories
    Config.create_directories()

    # Process dataset
    names, image_paths, original_paths, aligned_faces = process_dataset(
        csv_path=args.csv,
        output_dir=args.output_dir,
        detector_type=args.detector,
        max_images=args.max_images,
        skip_existing=args.skip_existing
    )

    if len(names) == 0:
        logger.error("No faces detected. Exiting.")
        return

    # Extract embeddings
    embeddings = extract_embeddings(aligned_faces, model_path=args.model)

    # Populate database
    populate_database(
        names,
        image_paths,
        original_paths,
        embeddings,
        db_path=args.db_path,
        collection_name=config.database.collection_name,
        batch_size=config.data.batch_size,
        reset_database=args.reset_database
    )

    logger.info("="*60)
    logger.info("Database preparation completed!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
