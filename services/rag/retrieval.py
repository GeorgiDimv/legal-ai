"""
Qdrant Vector Database Client for ATE Knowledge Base.
Handles storage and retrieval of document embeddings.
"""

import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from embeddings import embed_texts, embed_query
from ingest import DocumentChunk

logger = logging.getLogger(__name__)

# Collection configuration
COLLECTION_NAME = "ate_knowledge"
VECTOR_SIZE = 1024  # BGE-M3 embedding dimension


class QdrantRetriever:
    """Client for Qdrant vector database operations."""

    def __init__(self, host: str = "qdrant", port: int = 6333):
        """Initialize Qdrant client."""
        self.client = QdrantClient(host=host, port=port)
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)

        if not exists:
            logger.info(f"Creating collection: {COLLECTION_NAME}")
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )

            # Create payload indexes for filtering
            self.client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="document",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            self.client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="chunk_type",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            self.client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="article",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            logger.info("Collection created with indexes")
        else:
            logger.info(f"Collection {COLLECTION_NAME} already exists")

    def index_chunks(self, chunks: List[DocumentChunk], batch_size: int = 32) -> int:
        """
        Index document chunks into Qdrant.

        Args:
            chunks: List of DocumentChunk objects
            batch_size: Number of chunks to embed at once

        Returns:
            Number of chunks indexed
        """
        if not chunks:
            return 0

        logger.info(f"Indexing {len(chunks)} chunks...")

        # Get current max ID
        collection_info = self.client.get_collection(COLLECTION_NAME)
        start_id = collection_info.points_count

        # Process in batches
        total_indexed = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.text for c in batch]

            # Generate embeddings
            embeddings = embed_texts(texts)

            # Create points
            points = []
            for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                point_id = start_id + i + j
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": chunk.text,
                        "document": chunk.document,
                        "section": chunk.section,
                        "article": chunk.article,
                        "page": chunk.page,
                        "chunk_type": chunk.chunk_type,
                        "chunk_index": chunk.chunk_index
                    }
                ))

            # Upsert to Qdrant
            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )

            total_indexed += len(points)
            logger.info(f"Indexed batch {i // batch_size + 1}: {total_indexed}/{len(chunks)}")

        logger.info(f"Successfully indexed {total_indexed} chunks")
        return total_indexed

    def search(
        self,
        query: str,
        limit: int = 5,
        document_filter: Optional[str] = None,
        chunk_type_filter: Optional[List[str]] = None,
        score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks.

        Args:
            query: Search query text
            limit: Maximum results to return
            document_filter: Filter by document type ("naredba_24" or "uchebnik_ate")
            chunk_type_filter: Filter by chunk types
            score_threshold: Minimum similarity score

        Returns:
            List of matching chunks with scores
        """
        # Generate query embedding
        query_vector = embed_query(query)

        # Build filter conditions
        filter_conditions = []

        if document_filter:
            filter_conditions.append(
                models.FieldCondition(
                    key="document",
                    match=models.MatchValue(value=document_filter)
                )
            )

        if chunk_type_filter:
            filter_conditions.append(
                models.FieldCondition(
                    key="chunk_type",
                    match=models.MatchAny(any=chunk_type_filter)
                )
            )

        # Create filter if conditions exist
        search_filter = None
        if filter_conditions:
            search_filter = models.Filter(must=filter_conditions)

        # Execute search
        results = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=limit,
            score_threshold=score_threshold
        )

        # Format results
        formatted = []
        for r in results:
            formatted.append({
                "text": r.payload.get("text", ""),
                "document": r.payload.get("document", ""),
                "section": r.payload.get("section", ""),
                "article": r.payload.get("article"),
                "page": r.payload.get("page", 0),
                "chunk_type": r.payload.get("chunk_type", ""),
                "score": r.score
            })

        return formatted

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed collection."""
        try:
            info = self.client.get_collection(COLLECTION_NAME)
            return {
                "collection": COLLECTION_NAME,
                "points_count": info.points_count,
                "vectors_count": info.points_count,  # Each point has one vector
                "status": info.status.value
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}

    def delete_collection(self) -> bool:
        """Delete the entire collection (use with caution)."""
        try:
            self.client.delete_collection(COLLECTION_NAME)
            logger.info(f"Deleted collection: {COLLECTION_NAME}")
            self._ensure_collection()  # Recreate empty collection
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False
