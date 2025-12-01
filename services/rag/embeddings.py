"""
BGE-M3 Embedding Service for Bulgarian legal documents.
Runs on CPU to preserve GPU for vLLM.
"""

from sentence_transformers import SentenceTransformer
from typing import List
import logging

logger = logging.getLogger(__name__)

# Global model instance (loaded once)
_model: SentenceTransformer = None


def get_model() -> SentenceTransformer:
    """Get or initialize the embedding model."""
    global _model
    if _model is None:
        logger.info("Loading BGE-M3 embedding model...")
        _model = SentenceTransformer('BAAI/bge-m3')
        # Force CPU usage
        _model = _model.to('cpu')
        logger.info("BGE-M3 model loaded successfully")
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed

    Returns:
        List of embedding vectors (1024 dimensions for BGE-M3)
    """
    model = get_model()

    # BGE-M3 works best with instruction prefix for queries
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 10
    )

    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """
    Generate embedding for a search query.
    Uses instruction prefix for better retrieval.

    Args:
        query: Search query string

    Returns:
        Embedding vector (1024 dimensions)
    """
    model = get_model()

    # Add instruction prefix for queries (improves retrieval quality)
    prefixed_query = f"Represent this sentence for searching relevant passages: {query}"

    embedding = model.encode(
        prefixed_query,
        normalize_embeddings=True
    )

    return embedding.tolist()
