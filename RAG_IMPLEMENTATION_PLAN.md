# RAG System for Автотехническа Експертиза (ATE)

## Goal

Build a RAG system that enables the Legal AI to generate **expert-level ATE reports** (Автотехническа Експертиза) by learning from:
1. **Naredba 24** - Bulgarian regulation governing ATE methodology
2. **Uchebnik ATE II** - Official textbook for certified ATE experts

The system should produce reports like a certified Bulgarian automotive technical expert would - citing proper articles, using correct methodology, and following regulatory standards.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    New RAG Components                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  PDF Ingest  │───▶│  Embedding   │───▶│   Qdrant     │  │
│  │  (PyMuPDF)   │    │  (BGE-M3)    │    │  Vector DB   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                    │          │
│         ▼                   ▼                    ▼          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              RAG Service (FastAPI)                   │   │
│  │  - /ingest: Process expert PDFs                      │   │
│  │  - /search: Retrieve relevant chunks                 │   │
│  │  - /report: Generate full ATE report                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
└───────────────────────────┼─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Existing Services                           │
│  Gateway (:80) ──▶ vLLM (:8000) ──▶ Physics (:8004)        │
│      │                                                       │
│      ▼                                                       │
│  OCR (:8001)  Nominatim (:8002)  Car Value (:8003)         │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Recommendations

### 1. Vector Database: **Qdrant**

**Why Qdrant over pgvector:**
- Better filtering (by document type, article number, section)
- Native support for hybrid search (dense + sparse vectors)
- Optimized for retrieval workloads
- Easy Docker deployment
- Free and open source

### 2. Embedding Model: **BGE-M3**

**Why BGE-M3:**
- Multilingual (excellent Bulgarian/Cyrillic support)
- Produces both dense and sparse embeddings (hybrid search)
- 1024 dimensions, good balance of quality/speed
- Runs on CPU (preserves GPU for vLLM)
- Apache 2.0 license

### 3. PDF Processing: **PyMuPDF + Hierarchical Chunking**

**Strategy for legal/technical documents:**
```
Level 1: Document summaries (what is Naredba 24 about?)
Level 2: Section/Chapter summaries
Level 3: Article-level chunks (Чл. 5, Чл. 12, etc.)
Level 4: Specific formulas, tables, definitions
```

**Metadata per chunk:**
- `document`: "naredba_24" | "uchebnik_ate"
- `section`: Chapter/section name
- `article`: Article number (if applicable)
- `page`: Page number for citations
- `chunk_type`: "regulation" | "methodology" | "formula" | "definition" | "example"

---

## Implementation Plan

### Phase 1: RAG Service Setup

**New service: `services/rag/`**

```
services/rag/
├── Dockerfile
├── requirements.txt
├── main.py              # FastAPI endpoints
├── ingest.py            # PDF processing & chunking
├── embeddings.py        # BGE-M3 embedding logic
├── retrieval.py         # Qdrant search logic
└── knowledge_base/      # Store processed PDFs
    ├── naredba_24/
    └── uchebnik_ate/
```

**Docker additions to docker-compose.yml:**
```yaml
rag:
  build: ./services/rag
  ports:
    - "8005:8005"
  volumes:
    - ./knowledge_base:/app/knowledge_base
  depends_on:
    - qdrant

qdrant:
  image: qdrant/qdrant:latest
  ports:
    - "6333:6333"
  volumes:
    - qdrant_data:/qdrant/storage
```

### Phase 2: PDF Ingestion Pipeline

**Steps:**
1. Extract text from PDFs using PyMuPDF (preserves structure better than OCR for digital PDFs)
2. Detect document structure (articles, sections, chapters)
3. Smart chunking based on document type:
   - Naredba 24: Chunk by article (Чл. X)
   - Textbook: Chunk by section/topic
4. Generate embeddings with BGE-M3
5. Store in Qdrant with rich metadata

**Chunk size:** 512 tokens with 50 token overlap

### Phase 3: Gateway Integration

**Modify `services/gateway/main.py`:**

1. Add RAG client to fetch relevant context
2. Enhance extraction prompt with retrieved ATE methodology
3. New endpoint: `POST /generate-ate-report`

**Context injection strategy:**
```python
# Before LLM call, retrieve relevant ATE knowledge
relevant_chunks = await rag_client.search(
    query=ocr_text[:1000],  # Use beginning of document
    filters={"chunk_type": ["methodology", "regulation"]},
    limit=5
)

# Inject into system prompt
system_prompt = f"""You are a certified Bulgarian Automotive Technical Expert (ATE).
Use the following regulatory and methodological references:

{format_chunks(relevant_chunks)}

Follow Naredba 24 methodology when analyzing accidents.
Cite specific articles when making determinations.
"""
```

### Phase 4: ATE Report Generation

**New endpoint: `POST /generate-ate-report`**

**Input:** Processed claim data (from /process or /process-text)

**Output:** Full ATE expert report containing:
1. **Заглавна част** (Header) - Case info, expert credentials
2. **Въведение** (Introduction) - Assignment, questions to answer
3. **Изследвана документация** (Examined documents)
4. **Фактическа обстановка** (Factual circumstances)
5. **Техническо изследване** (Technical analysis)
   - Vehicle damage description
   - Crash reconstruction (using Physics service)
   - Speed calculations with formulas from textbook
6. **Изводи** (Conclusions) - Citing Naredba 24 articles
7. **Отговори на въпросите** (Answers to questions)

**Report uses RAG to:**
- Cite correct Naredba 24 articles for each determination
- Apply proper ATE methodology from textbook
- Include relevant formulas with proper notation
- Reference regulatory requirements

---

## Files to Create/Modify

### New Files:
- `services/rag/Dockerfile`
- `services/rag/requirements.txt`
- `services/rag/main.py`
- `services/rag/ingest.py`
- `services/rag/embeddings.py`
- `services/rag/retrieval.py`
- `services/rag/report_generator.py`

### Modified Files:
- `docker-compose.yml` - Add rag and qdrant services
- `services/gateway/main.py` - Add RAG integration, new endpoint
- `services/gateway/extractors.py` - Enhance prompts with RAG context

---

## Service Ports (Updated)

| Service    | Port | Description                          |
|------------|------|--------------------------------------|
| Gateway    | 80   | Main API entry point                 |
| vLLM       | 8000 | LLM API                              |
| OCR        | 8001 | Text extraction                      |
| Nominatim  | 8002 | Geocoding                            |
| Car Value  | 8003 | Vehicle pricing                      |
| Physics    | 8004 | Crash reconstruction                 |
| **RAG**    | 8005 | Knowledge retrieval (NEW)            |
| **Qdrant** | 6333 | Vector database (NEW)                |

---

## RAG Service Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest` | POST | Process and index expert PDFs |
| `/search` | POST | Retrieve relevant knowledge chunks |
| `/collections` | GET | List indexed document collections |
| `/health` | GET | Service health check |

## Gateway New Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate-ate-report` | POST | Generate full ATE expert report |
| `/process` | POST | (Enhanced) Now includes RAG context |

---

## Dependencies

**services/rag/requirements.txt:**
```
fastapi==0.109.0
uvicorn[standard]==0.27.0
pymupdf==1.24.0
sentence-transformers==2.2.2
FlagEmbedding==1.2.10
qdrant-client==1.7.0
httpx==0.26.0
pydantic==2.5.3
```

---

## GPU Allocation (6 GPUs Available)

**Recommended: vLLM on 6 GPUs + Embeddings on CPU**

```
GPU 0-5: vLLM (Qwen3-32B-AWQ)
  - Tensor parallelism across 6 GPUs
  - Increase context from 8k → 16k tokens
  - Better throughput for report generation

CPU: BGE-M3 Embeddings
  - PDF ingestion is one-time (not latency-sensitive)
  - Keeps all GPU VRAM for LLM
```

**docker-compose.yml change for vLLM:**
```yaml
llm:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 6  # Changed from 4
            capabilities: [gpu]
  command: >
    --model Qwen/Qwen3-32B-AWQ
    --tensor-parallel-size 6  # Changed from 4
    --max-model-len 16384     # Increased from 8192
```

---

## How RAG Handles 600+ Page PDFs

**The 8k/16k context is NOT a problem because:**

1. **Chunking:** PDFs split into ~512 token chunks
   - 600 pages ≈ 180,000 words ≈ 240,000 tokens
   - Results in ~470 chunks per PDF
   - Total: ~1000 chunks for both documents

2. **Retrieval:** Only fetch relevant chunks per query
   - Query: "What is the formula for calculating braking distance?"
   - Retrieve: Top 5-10 most relevant chunks (~2500-5000 tokens)
   - Remaining context: 11k-13k tokens for case document + response

3. **Indexed once, queried many times**
   - Initial processing: ~30-60 minutes (one-time)
   - Query time: <100ms to retrieve relevant chunks

**Context budget per request:**
```
Total context: 16,384 tokens (with 6 GPUs)
├── System prompt:        ~500 tokens
├── RAG context:        ~3,000 tokens (5-6 chunks)
├── Case document:      ~8,000 tokens
└── Response buffer:    ~4,800 tokens
```

---

## Estimated Resource Usage

- **Qdrant:** ~500MB RAM, ~100MB storage for ~1000 vectors
- **BGE-M3:** ~2GB RAM (runs on CPU)
- **PDF Processing:** ~30-60 min one-time for 600+ pages
- **GPU:** All 6 GPUs for vLLM (increased context)
