"""
Face Matcher - Duplicate Account Detection System

A face matching MVP for duplicate account detection using face embeddings
and vector similarity search with Milvus Lite.
"""

__version__ = "0.1.0"
__author__ = "Face Matcher Team"

from .config import config, Config

__all__ = ["config", "Config", "__version__"]
