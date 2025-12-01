"""
RAG Service for Автотехническа Експертиза (ATE).
Provides knowledge retrieval from Naredba 24 and ATE textbook.
"""

import os
import logging
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ingest import process_pdf
from retrieval import QdrantRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ATE RAG Service",
    description="Knowledge retrieval for Автотехническа Експертиза",
    version="1.0.0"
)

# Initialize retriever (connects to Qdrant)
retriever: QdrantRetriever = None

KNOWLEDGE_BASE_DIR = Path("/app/knowledge_base")


# ============== Request/Response Models ==============

class SearchRequest(BaseModel):
    query: str
    limit: int = 5
    document_filter: Optional[str] = None  # "naredba_24" or "uchebnik_ate"
    chunk_type_filter: Optional[List[str]] = None  # ["regulation", "methodology", "formula"]
    score_threshold: float = 0.5


class SearchResult(BaseModel):
    text: str
    document: str
    section: str
    article: Optional[str]
    page: int
    chunk_type: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    count: int


class IngestResponse(BaseModel):
    filename: str
    chunks_indexed: int
    document_type: str
    message: str


class CollectionStats(BaseModel):
    collection: str
    points_count: int
    vectors_count: int
    status: str


# ============== Startup/Shutdown ==============

@app.on_event("startup")
async def startup():
    """Initialize services on startup."""
    global retriever

    # Get Qdrant host from environment
    qdrant_host = os.getenv("QDRANT_HOST", "qdrant")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))

    logger.info(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}")
    retriever = QdrantRetriever(host=qdrant_host, port=qdrant_port)

    # Ensure knowledge base directory exists
    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("RAG Service started successfully")


# ============== Endpoints ==============

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    stats = retriever.get_collection_stats() if retriever else {}
    return {
        "status": "healthy",
        "service": "rag",
        "collection_stats": stats
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Ingest a PDF document into the knowledge base.

    Processes the PDF, chunks it, generates embeddings, and stores in Qdrant.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save uploaded file
    file_path = KNOWLEDGE_BASE_DIR / file.filename
    content = await file.read()

    with open(file_path, 'wb') as f:
        f.write(content)

    logger.info(f"Saved uploaded file: {file_path}")

    try:
        # Process PDF into chunks
        chunks = process_pdf(str(file_path))

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the PDF"
            )

        # Index chunks
        indexed_count = retriever.index_chunks(chunks)

        # Determine document type from chunks
        doc_type = chunks[0].document if chunks else "unknown"

        return IngestResponse(
            filename=file.filename,
            chunks_indexed=indexed_count,
            document_type=doc_type,
            message=f"Successfully indexed {indexed_count} chunks from {file.filename}"
        )

    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search_knowledge(request: SearchRequest):
    """
    Search the knowledge base for relevant content.

    Returns chunks from Naredba 24 and ATE textbook that match the query.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    results = retriever.search(
        query=request.query,
        limit=request.limit,
        document_filter=request.document_filter,
        chunk_type_filter=request.chunk_type_filter,
        score_threshold=request.score_threshold
    )

    return SearchResponse(
        query=request.query,
        results=[SearchResult(**r) for r in results],
        count=len(results)
    )


@app.get("/collections", response_model=CollectionStats)
async def get_collections():
    """Get statistics about the indexed knowledge base."""
    stats = retriever.get_collection_stats()

    if "error" in stats:
        raise HTTPException(status_code=500, detail=stats["error"])

    return CollectionStats(**stats)


@app.delete("/collections")
async def delete_collections():
    """
    Delete all indexed documents (use with caution).
    Recreates an empty collection.
    """
    success = retriever.delete_collection()

    if success:
        return {"message": "Collection deleted and recreated"}
    else:
        raise HTTPException(status_code=500, detail="Failed to delete collection")


@app.post("/ingest-local")
async def ingest_local_pdf(filename: str):
    """
    Ingest a PDF that's already in the knowledge_base directory.
    Useful for processing pre-uploaded files.
    """
    file_path = KNOWLEDGE_BASE_DIR / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {filename}. Place it in /app/knowledge_base/"
        )

    if not filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        chunks = process_pdf(str(file_path))

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the PDF"
            )

        indexed_count = retriever.index_chunks(chunks)
        doc_type = chunks[0].document if chunks else "unknown"

        return IngestResponse(
            filename=filename,
            chunks_indexed=indexed_count,
            document_type=doc_type,
            message=f"Successfully indexed {indexed_count} chunks"
        )

    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Convenience Endpoint for Gateway ==============

@app.post("/context")
async def get_ate_context(request: SearchRequest):
    """
    Get formatted ATE context for LLM prompts.

    This is the main endpoint used by the Gateway service to retrieve
    relevant knowledge for enhancing LLM responses.
    """
    results = retriever.search(
        query=request.query,
        limit=request.limit,
        document_filter=request.document_filter,
        chunk_type_filter=request.chunk_type_filter,
        score_threshold=request.score_threshold
    )

    if not results:
        return {
            "context": "",
            "sources": [],
            "count": 0
        }

    # Format context for LLM
    context_parts = []
    sources = []

    for i, r in enumerate(results, 1):
        # Build reference string
        ref_parts = [r["document"]]
        if r.get("article"):
            ref_parts.append(r["article"])
        ref_parts.append(f"стр. {r['page']}")
        reference = ", ".join(ref_parts)

        # Add formatted chunk
        context_parts.append(f"[{i}] {r['text']}\n   Източник: {reference}")

        sources.append({
            "reference": reference,
            "document": r["document"],
            "page": r["page"],
            "article": r.get("article"),
            "score": r["score"]
        })

    formatted_context = "\n\n".join(context_parts)

    return {
        "context": formatted_context,
        "sources": sources,
        "count": len(results)
    }
