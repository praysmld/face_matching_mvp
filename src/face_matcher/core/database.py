"""
Vector Database Module
Manages face embeddings storage and retrieval using Milvus Lite
"""

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class VectorDatabase:
    """Vector database manager for face embeddings using Milvus Lite"""

    def __init__(
        self,
        db_path: str = "./data/milvus_face_data.db",
        collection_name: str = "face_embeddings",
        embedding_dim: int = 128
    ):
        """
        Initialize vector database manager

        Args:
            db_path: Path to local Milvus Lite database file
            collection_name: Name of the collection
            embedding_dim: Dimension of face embeddings
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.collection = None

        # Connect to database
        self._connect()

    def _connect(self):
        """Establish connection to Milvus Lite"""
        try:
            connections.connect(
                alias="default",
                uri=self.db_path
            )
            logger.info(f"Connected to Milvus Lite: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus Lite: {e}")
            raise

    def create_collection(self, drop_existing: bool = False):
        """
        Create collection with schema for face embeddings

        Args:
            drop_existing: If True, drop existing collection before creating
        """
        # Check if collection exists
        if utility.has_collection(self.collection_name):
            if drop_existing:
                logger.info(f"Dropping existing collection: {self.collection_name}")
                utility.drop_collection(self.collection_name)
            else:
                logger.info(f"Loading existing collection: {self.collection_name}")
                self.collection = Collection(self.collection_name)
                return

        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="original_path", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        ]

        schema = CollectionSchema(
            fields=fields,
            description="Face embeddings for duplicate account detection"
        )

        # Create collection
        self.collection = Collection(
            name=self.collection_name,
            schema=schema,
            using='default'
        )

        logger.info(f"Created collection: {self.collection_name}")

    def create_index(self, index_type: str = "IVF_FLAT", nlist: int = 128):
        """
        Create index on embedding field for efficient similarity search

        Args:
            index_type: Type of index (IVF_FLAT, HNSW, etc.)
            nlist: Number of cluster units for IVF index
        """
        if self.collection is None:
            raise ValueError("Collection not initialized")

        # Check if index already exists on embedding field
        try:
            # Check specifically for index on the embedding field
            indexes = self.collection.indexes
            for index in indexes:
                if index.field_name == "embedding":
                    logger.info("Index already exists on embedding field")
                    return
        except Exception:
            # No index exists, continue to create one
            pass

        # Define index parameters
        index_params = {
            "index_type": index_type,
            "metric_type": "L2",
            "params": {"nlist": nlist}
        }

        # Create index
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )

        logger.info(f"Created {index_type} index with nlist={nlist}")

    def load_collection(self):
        """Load collection into memory for search operations"""
        if self.collection is None:
            raise ValueError("Collection not initialized")

        self.collection.load()
        logger.info(f"Loaded collection: {self.collection_name}")

    def insert_batch(
        self,
        names: List[str],
        image_paths: List[str],
        original_paths: List[str],
        embeddings: np.ndarray
    ) -> int:
        """
        Insert batch of face embeddings

        Args:
            names: List of person names
            image_paths: List of cropped face image file paths
            original_paths: List of original downloaded image paths
            embeddings: Numpy array of embeddings (N x embedding_dim)

        Returns:
            Number of entities inserted
        """
        if self.collection is None:
            raise ValueError("Collection not initialized")

        if len(names) != len(image_paths) != len(original_paths) != len(embeddings):
            raise ValueError("Length mismatch: names, image_paths, original_paths, and embeddings must have same length")

        # Prepare data
        data = [
            names,
            image_paths,
            original_paths,
            embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        ]

        # Insert data
        insert_result = self.collection.insert(data)
        self.collection.flush()

        num_inserted = len(names)
        logger.info(f"Inserted {num_inserted} embeddings")

        return num_inserted

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 6,
        nprobe: int = 10
    ) -> List[Dict]:
        """
        Search for similar faces

        Args:
            query_embedding: Query face embedding
            top_k: Number of results to return
            nprobe: Number of clusters to search (for IVF index)

        Returns:
            List of search results with metadata
        """
        if self.collection is None:
            raise ValueError("Collection not initialized")

        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Search parameters
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": nprobe}
        }

        # Perform search
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["name", "image_path", "original_path"]
        )

        # Parse results
        parsed_results = []
        for hits in results:
            for hit in hits:
                distance = hit.distance
                # Convert L2 distance to similarity score
                # For normalized vectors: similarity ≈ 1 - (distance^2 / 4)
                similarity = max(0.0, 1.0 - (distance ** 2) / 4.0)

                parsed_results.append({
                    "id": hit.id,
                    "name": hit.entity.get("name"),
                    "image_path": hit.entity.get("image_path"),
                    "original_path": hit.entity.get("original_path"),
                    "distance": float(distance),
                    "similarity": float(similarity)
                })

        return parsed_results

    def get_stats(self) -> Dict:
        """
        Get collection statistics

        Returns:
            Dictionary with collection stats
        """
        if self.collection is None:
            raise ValueError("Collection not initialized")

        return {
            "name": self.collection.name,
            "num_entities": self.collection.num_entities,
            "description": self.collection.description
        }

    def delete_collection(self):
        """Delete the collection"""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            self.collection = None

    def close(self):
        """Close database connection"""
        connections.disconnect("default")
        logger.info("Disconnected from Milvus Lite")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
