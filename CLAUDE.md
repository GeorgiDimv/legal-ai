# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Legal AI is a document processing pipeline for Bulgarian automotive insurance claims. It uses OCR to extract text from claim documents, then LLM-based extraction to parse structured data, enriched with geocoding, car market values, and physics-based crash reconstruction.

## Architecture

```
                    ┌─────────────────┐
                    │   API Gateway   │ :80
                    │   (FastAPI)     │
                    └────────┬────────┘
                             │
   ┌───────────┬─────────────┼─────────────┬───────────┬───────────┐
   │           │             │             │           │           │
   ▼           ▼             ▼             ▼           ▼           ▼
┌───────┐ ┌───────┐   ┌───────────┐   ┌───────┐ ┌─────────┐ ┌───────┐
│  OCR  │ │Physics│   │   vLLM    │   │  Car  │ │Nominatim│ │  RAG  │
│:8001  │ │:8004  │   │   :8000   │   │ Value │ │  :8002  │ │ :8005 │
└───────┘ └───────┘   └───────────┘   │ :8003 │ └─────────┘ └───┬───┘
                             │        └───────┘                 │
                    ┌────────┴────────┐                    ┌────┴────┐
                    │                 │                    │ Qdrant  │
                    ▼                 ▼                    │  :6333  │
              ┌──────────┐      ┌──────────┐               └─────────┘
              │  Redis   │      │PostgreSQL│
              │  :6379   │      │  :5432   │
              └──────────┘      └──────────┘
```

**Key Services:**
- **Gateway** (`services/gateway/`): FastAPI orchestrator with `/process`, `/process-text`, and `/generate-ate-report` endpoints
- **OCR** (`services/ocr/`): PaddleOCR with Cyrillic language support for Bulgarian documents
- **LLM**: vLLM serving Qwen3-32B-AWQ across 4 GPUs with tensor parallelism (16k context)
- **Physics** (`services/physics/`): Crash reconstruction using Momentum 360 and Impact Theory formulas
- **Car Value** (`services/car_value/`): On-demand price scraper with VIN decoding (cars.bg + mobile.bg + NHTSA API)
- **Nominatim**: Geocoding service with Bulgaria OSM data (returns lat/lon coordinates)
- **RAG** (`services/rag/`): Knowledge retrieval from ATE expert materials (Naredba 24, textbook)
- **Qdrant**: Vector database for RAG embeddings
- **PostgreSQL**: Stores processed claims
- **Redis**: Caching layer for car values (24h TTL) and VIN decodes (permanent)

## Commands

### Connecting to the Rig

```bash
# SSH to GPU server
ssh ubuntu@192.168.1.32

# Project location on rig
cd /home/ubuntu/legal-ai
```

### Running the Pipeline

```bash
# Initial setup (requires 4 GPUs)
./init.sh

# Start all services
docker compose up -d

# View logs
docker compose logs -f
docker compose logs -f llm      # LLM model loading
docker compose logs -f nominatim # OSM data import

# Check container status
docker compose ps

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Testing

```bash
# Health checks only
python test_pipeline.py --health-only

# Run all service tests (OCR, geocoding, car value, LLM)
python test_pipeline.py --all

# Process a document
python test_pipeline.py sample.pdf
```

### Building Individual Services

```bash
docker compose build ocr
docker compose build gateway
docker compose build car_value
docker compose build physics
```

## Service Ports

| Service    | Port | Description                          |
|------------|------|--------------------------------------|
| Gateway    | 80   | Main API entry point                 |
| LLM (vLLM) | 8000 | OpenAI-compatible LLM API            |
| OCR        | 8001 | PaddleOCR text extraction            |
| Nominatim  | 8002 | Geocoding (Bulgaria only)            |
| Car Value  | 8003 | Vehicle market value lookup          |
| Physics    | 8004 | Crash physics calculations           |
| RAG        | 8005 | ATE knowledge retrieval              |
| Qdrant     | 6333 | Vector database                      |
| PostgreSQL | 5432 | Database                             |
| Redis      | 6379 | Cache                                |

## Key Files

- `services/gateway/main.py`: Main orchestration logic, `/process`, `/process-text`, and `/generate-ate-report` endpoints
- `services/gateway/extractors.py`: LLM prompt and JSON extraction logic (includes VIN extraction)
- `services/gateway/schemas.py`: Pydantic models for API responses
- `services/ocr/main.py`: PaddleOCR service with PDF/image support
- `services/car_value/main.py`: On-demand scraper + VIN decoder (NHTSA API) with Redis caching
- `services/physics/main.py`: Crash physics calculations (Momentum 360, Impact Theory)
- `services/rag/main.py`: RAG service for ATE knowledge retrieval
- `services/rag/ingest.py`: PDF processing and chunking for knowledge base
- `services/rag/embeddings.py`: BGE-M3 embedding generation
- `services/rag/retrieval.py`: Qdrant vector search
- `database/init.sql`: PostgreSQL schema and seed data
- `knowledge_base/`: Directory for ATE expert PDFs (Naredba 24, textbook)

## API Endpoints

### Gateway Endpoints

| Endpoint | Method | Input | Description |
|----------|--------|-------|-------------|
| `/process` | POST | PDF/image (multipart form) | Full pipeline: OCR → LLM → enrichment |
| `/process-text` | POST | JSON `{"text": "..."}` | Skip OCR, direct text → LLM → enrichment |
| `/generate-ate-report` | POST | JSON (processed_result or raw_text) | Generate professional ATE expert report |
| `/health` | GET | - | Health check with service status |

## Data Flow

### `/process` (with OCR)
1. Document (PDF/image) uploaded to `/process`
2. OCR service extracts Bulgarian/English text
3. LLM extracts structured JSON (claim number, vehicles, parties, fault, settlement)
4. Geocoding enriches location data with coordinates
5. Car value service adds current market prices from scraped data
6. Physics service analyzes collision dynamics (if data available)
7. Result stored in PostgreSQL and returned

### `/process-text` (skip OCR)
1. Raw text submitted to `/process-text`
2. LLM extracts structured JSON (same as above)
3. Enrichment with geocoding, car values, physics (same as above)
4. Result stored in PostgreSQL and returned

Use `/process-text` for testing without PDF or processing pre-OCR'd documents.

## Environment Variables

Key variables configured in `.env` (copy from `.env.example`):
- `HF_TOKEN`: HuggingFace token for model download
- `POSTGRES_USER`, `POSTGRES_PASSWORD`: Database credentials
- `NOMINATIM_PASSWORD`: Nominatim admin password

## Physics Service (v2.0.0)

The physics service provides crash reconstruction based on Bulgarian methodology from the "l.xlsx" formulas.

**Full documentation:** `docs/physics-formulas.md`

### 15 Formulas for ATE Reports

| # | Formula | Description |
|---|---------|-------------|
| 1 | `F = μ × m × g` | Friction force |
| 2 | `u = √(2×μ×g×σ)` | Post-impact velocity from skid |
| 3 | `Vx = V×cos(α)` | Velocity X-component |
| 4 | `V = √(Vx² + Vy²)` | Resultant velocity |
| 5-6 | Momentum X, Y | Conservation of momentum (vector) |
| 7 | Matrix method | Impact Theory solution |
| 8 | `ΔV = √(V² + u² - 2Vu×cos(β-α))` | Delta-V |
| 9 | `Sоз = V×(tr+tsp+0.5tn) + V²/(2j)` | Dangerous zone |
| 10 | `Tу` formula | Stopping time |
| 11 | `Vбезоп` | Safe speed (quadratic) |
| 12 | `k = (u₂-u₁)/(V₁-V₂)` | Restitution coefficient |
| 13 | `E = ½×m×V²` | Kinetic energy |
| 14 | `ΔE = E₁ - E₂` | Dissipated energy |
| 15 | `W = μ×m×g×s` | Work of braking forces |

### Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /velocity-from-skid` | Calculate pre-braking speed from skid mark length |
| `POST /momentum-360` | Full 360° momentum analysis with angular vectors |
| `POST /dangerous-zone` | Calculate stopping distance, time, safe speed |
| `POST /validate-claimed-speed` | Validate driver's claimed speed against physical evidence |
| `GET /formulas` | Returns all physics formulas and friction coefficients |

### Future Formulas (not yet implemented)

| Formula | Use Case | Needed Data |
|---------|----------|-------------|
| Pedestrian throw | Pedestrian accidents | `throw_distance_m` from protocol |
| EES deformation | When crush depth available | `crush_depth_cm` from inspection |

See `docs/physics-formulas.md` for implementation details.

## Car Value Service (v4.0.0)

On-demand price scraper with VIN decoding, parts pricing, and **full Naredba 24 compliance**:

### Data Flow
1. **Check Redis cache** (24h TTL for prices/parts, permanent for VIN)
2. **Parallel scrape** both cars.bg AND mobile.bg simultaneously
3. **Merge prices** weighted by sample size for higher confidence
4. **Return combined result** with source breakdown (e.g., "cars.bg(15) + mobile.bg(8)")
5. **Return vehicle info only** if both sources fail (no static fallback)

### Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /vin/{vin}` | Decode VIN via NHTSA API (make, model, year, country) |
| `GET /value-by-vin/{vin}` | Decode VIN + lookup market value |
| `GET /value/{make}/{model}/{year}` | Get car market value (on-demand scrape) |
| `POST /parts/search` | Search for parts prices with full Naredba 24 compliance |
| `GET /naredba24/coefficient/{make}/{year}` | Get depreciation coefficient (чл. 12) |
| `GET /naredba24/labor/{part_name}` | Get labor hours from structured table (Глава III) |
| `GET /naredba24/paint` | Calculate paint costs (Глава IV) |

### VIN Decoding
- **Primary**: NHTSA vPIC API (free, no API key, works for EU vehicles)
- **Fallback**: Local WMI decode (manufacturer + country from first 3 chars)
- **Cache**: Permanent Redis cache (VINs don't change)

### Parts Pricing
LLM extracts `damaged_parts` list from claim documents, then gateway calls `/parts/search` to get real-time prices:
- **bazar.bg**: Bulgarian classifieds (new and used parts)
- **alochasti.bg**: Dedicated auto parts store
- **autoprofi.bg**: Sofia-based OEM and aftermarket parts

Parts search includes automatic translation (English → Bulgarian) and labor cost estimation.

### Supported Makes
BMW, Mercedes-Benz, Audi, Volkswagen, Toyota, Opel, Ford, Renault, Peugeot, Skoda, Honda, Mazda, Nissan, Volvo, Hyundai, Kia

### Damage Estimation
1. LLM extracts `estimated_damage` from document (if mentioned)
2. LLM extracts `damaged_parts` list (e.g., ["front bumper", "headlight left"])
3. Gateway searches Bulgarian websites for real-time parts prices
4. Returns both LLM estimate and web-scraped parts breakdown

### Naredba 24 Compliance (v4.0.0)

Full implementation of Bulgarian ATE regulation for accurate repair cost estimation:

#### Depreciation Coefficients (чл. 12)
- **Eastern vehicles** (Lada, Dacia, Hyundai, Kia, etc.): 0.20-0.50
- **Western vehicles** (BMW, Mercedes, VW, etc.): 0.40-1.00
- Age brackets: 0-3, 4-7, 8-14, 15+ years
- Special adjustments for Peugeot, Opel, Citroen, Ford

#### Vehicle Classes (чл. 13)
| Class | Length | Examples |
|-------|--------|----------|
| A | <4.00m | Smart, Fiat 500, Mini |
| B | 4.00-4.60m | Golf, Focus, Corolla |
| C | >4.60m | BMW 5, Audi A6, Passat |
| D | SUV/Van | X5, Land Cruiser, Transit |

#### Labor Norms (Глава III)
- **40+ parts** with structured hours per vehicle class
- Examples: Front bumper (0.8-1.4h), Front fender (2.8-4.0h), Door (1.9-3.0h)
- Automatic translation: English part names → Bulgarian

#### Paint Costs (Глава IV)
- Labor hours per panel per vehicle class
- Materials: Standard (80-150 BGN), Metallic (120-220 BGN), Special (180-350 BGN)
- Color matching: 1.0 hour fixed
- Oven drying: 1.5-4.5 hours by panel count

#### Total Repair Cost Formula
```
total_repair_cost = parts_after_depreciation + labor_cost + paint_cost

Where:
- parts_after_depreciation = scraped_price × depreciation_coefficient
- labor_cost = labor_hours × hourly_rate_by_work_type
- paint_cost = paint_labor + paint_materials + color_matching + oven_drying
```

See `docs/naredba24-implementation-status.md` for full details and potential gaps.

## RAG Service (Автотехническа Експертиза Knowledge Base)

The RAG service enables expert-level ATE report generation by providing relevant knowledge from:
- **Naredba 24**: Bulgarian regulation governing ATE methodology
- **Uchebnik ATE II**: Official textbook for certified automotive technical experts

### How RAG Works

#### Architecture Overview
```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INGESTION (One-Time)                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PDF File ──► PyMuPDF ──► Text + Pages ──► Chunker ──► DocumentChunks  │
│                                               │                         │
│                                               ▼                         │
│                                          BGE-M3 Model                   │
│                                               │                         │
│                                               ▼                         │
│                                      1024-dim Vectors                   │
│                                               │                         │
│                                               ▼                         │
│                                     Qdrant (ate_knowledge)              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         RETRIEVAL (Per Query)                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Query ──► BGE-M3 ──► Query Vector ──► Qdrant Search ──► Top K Chunks  │
│  "методика за                              │                            │
│   скоростта"                               │ cosine similarity          │
│                                            ▼                            │
│                                   Chunks + Scores + Metadata            │
│                                            │                            │
│                                            ▼                            │
│                                   Formatted Context for LLM             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Step 1: PDF Ingestion (`ingest.py`)
```python
# What happens when you call /ingest-local:

1. PDF → Text Extraction (PyMuPDF)
   - Extracts text page-by-page
   - Preserves page numbers for citations

2. Document Type Detection
   - Filename patterns: "naredba" → naredba_24, "uchebnik" → uchebnik_ate
   - Content patterns: "чл." (articles) → regulation

3. Smart Chunking
   - Target: 512 words per chunk
   - Overlap: 50 words (preserves context at boundaries)
   - Breaks at sentence boundaries (. ! ?)
   - Skips chunks < 50 chars

4. Metadata Extraction per Chunk:
   - document: "naredba_24" or "uchebnik_ate"
   - section: "Глава II" or "Раздел 3"
   - article: "Чл. 5" (if present)
   - page: source page number
   - chunk_type: "regulation", "formula", "methodology", "definition", "example"
```

#### Step 2: Embedding Generation (`embeddings.py`)
```python
# BGE-M3 Model (BAAI/bge-m3)
- Multilingual: Bulgarian, English, Russian, etc.
- Output: 1024-dimensional vectors
- Normalized embeddings (unit length for cosine similarity)

# Query vs Document Embedding
- Documents: Embedded as-is
- Queries: Prefixed with "Represent this sentence for searching relevant passages: "
  (Improves retrieval accuracy)
```

#### Step 3: Vector Storage (`retrieval.py`)
```python
# Qdrant Collection: "ate_knowledge"
- Vector size: 1024
- Distance metric: Cosine similarity
- Indexed fields: document, chunk_type, article (for filtering)

# Each stored point contains:
{
    "id": 123,
    "vector": [0.023, -0.045, ...],  # 1024 floats
    "payload": {
        "text": "Съгласно методиката за определяне...",
        "document": "naredba_24",
        "section": "Глава III",
        "article": "Чл. 12",
        "page": 45,
        "chunk_type": "methodology"
    }
}
```

#### Step 4: Search & Retrieval
```python
# When /context is called:
1. Query text → BGE-M3 → query_vector
2. Qdrant.query_points(query_vector, limit=8, score_threshold=0.5)
3. Returns chunks sorted by cosine similarity

# Example search result:
{
    "text": "При определяне скоростта на движение се използва формулата V = √(2·μ·g·S)...",
    "document": "uchebnik_ate",
    "section": "Глава V",
    "page": 127,
    "chunk_type": "formula",
    "score": 0.87  # 87% similar to query
}
```

### Why Chunks Are Real Data (Not Magic)

The 570 vectors in your collection are **actual text chunks** from the PDFs:

```bash
# Verify chunks are real text:
curl -X POST http://localhost:8005/search \
  -H "Content-Type: application/json" \
  -d '{"query": "скорост", "limit": 1}'

# Returns actual text from Naredba 24 or Uchebnik ATE II
# with page numbers you can verify in the original PDF
```

Each chunk is stored with its **source page number**, so you can cross-reference with the original document.

### Setup

```bash
# 1. Copy expert PDFs to knowledge_base directory
cp "naredba 24.pdf" "Uchebnik full ATE II.pdf" knowledge_base/

# 2. Start RAG and Qdrant services
docker compose up -d qdrant rag

# 3. Ingest the PDFs (one-time, ~30-60 min for 600+ pages)
curl -X POST "http://localhost:8005/ingest-local?filename=naredba%2024.pdf"
curl -X POST "http://localhost:8005/ingest-local?filename=Uchebnik%20full%20ATE%20II.pdf"

# 4. Check indexing status
curl http://localhost:8005/collections
# Expected: {"points_count": 570, "vectors_count": 570, "status": "green"}
```

### Testing RAG

```bash
# 1. Verify embeddings worked
curl http://localhost:8005/collections
# Should show points_count > 0

# 2. Test search retrieval
curl -X POST http://localhost:8005/search \
  -H "Content-Type: application/json" \
  -d '{"query": "методика за определяне на скоростта при ПТП", "limit": 3}'
# Should return relevant chunks with scores > 0.5

# 3. Test context formatting (what gateway uses)
curl -X POST http://localhost:8005/context \
  -H "Content-Type: application/json" \
  -d '{"query": "автотехническа експертиза изисквания", "limit": 5}'
# Returns formatted context with source citations
```

### RAG Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest` | POST | Upload and index a PDF |
| `/ingest-local` | POST | Index a PDF from knowledge_base/ |
| `/search` | POST | Search for relevant knowledge chunks |
| `/context` | POST | Get formatted context for LLM prompts |
| `/collections` | GET | Get indexing statistics |
| `/health` | GET | Service health check |

### ATE Report Generation

Generate professional Bulgarian ATE reports:

```bash
# Using processed result from /process
curl -X POST http://localhost/generate-ate-report \
  -H "Content-Type: application/json" \
  -d '{
    "processed_result": {...},
    "expert_questions": [
      "Какъв е механизмът на произшествието?",
      "Каква е скоростта на МПС преди удара?"
    ]
  }'

# Using raw text
curl -X POST http://localhost/generate-ate-report \
  -H "Content-Type: application/json" \
  -d '{"raw_text": "..."}'
```

The report follows Bulgarian ATE standards with sections:
1. Заглавна част (Header)
2. Въведение (Introduction)
3. Изследвана документация (Examined documents)
4. Фактическа обстановка (Factual circumstances)
5. Техническо изследване (Technical analysis)
6. Изводи (Conclusions)
7. Отговори на въпросите (Answers)

#### Internal Process Flow (2 LLM Calls)

The `/generate-ate-report` endpoint uses 2 sequential LLM calls:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Raw Text   │────▶│  LLM Call 1 │────▶│ RAG Search  │────▶│  LLM Call 2 │
│  (input)    │     │  (extract)  │     │  (context)  │     │  (generate) │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                          │                   │                    │
                          ▼                   ▼                    ▼
                    Structured JSON     8 Expert Chunks      ATE Report
                    (vehicles, etc.)    (methodology)        (Bulgarian)
```

**Step 1: Data Extraction** (~2 min @ 4.5 tokens/s)
- Input: Raw claim text
- Output: Structured JSON with vehicles, parties, location, damages
- Purpose: Understand what happened in the claim

**Step 2: RAG Context Retrieval** (~1 sec)
- Query: Summary of extracted data
- Output: 5-8 relevant chunks from Naredba 24 / Uchebnik ATE
- Purpose: Get expert methodology and legal requirements

**Step 3: Report Generation** (~4-5 min @ 4.5 tokens/s)
- Input: Extracted JSON + RAG context + expert questions
- Output: Professional Bulgarian ATE report
- Purpose: Generate report following expert standards

**Performance Notes:**
- Total time: ~5-6 minutes per report
- Concurrency: Limited to 1 report at a time (GPU constraint)
- Queue: Max 2 waiting requests (503 if exceeded)

### Key Files

| File | Purpose |
|------|---------|
| `services/rag/main.py` | FastAPI service with /search, /context, /ingest endpoints |
| `services/rag/ingest.py` | PDF text extraction, chunking, metadata detection |
| `services/rag/embeddings.py` | BGE-M3 model loading and embedding generation |
| `services/rag/retrieval.py` | Qdrant client, vector storage, similarity search |

### Technical Details

- **Embedding Model**: BGE-M3 (`BAAI/bge-m3`) - multilingual, 1024 dimensions
- **Vector Database**: Qdrant (cosine similarity, payload indexing)
- **Chunk Size**: 512 words with 50 word overlap
- **Chunk Types**: regulation, methodology, formula, definition, example
- **Retrieval**: Top 5-8 most relevant chunks per query (score > 0.5)
- **Context Budget**: ~3000 tokens for RAG context in 16k window
- **Device**: CPU by default (set USE_GPU=true for GPU acceleration)

## Notes

- LLM requires 4 GPUs with tensor parallelism; first startup downloads ~20GB model
- LLM context window is 16k tokens; generation speed ~4.5 tokens/s
- vLLM runs with `--enforce-eager --disable-custom-all-reduce` flags for GPU stability (RTX 3060 Ti)
- ATE reports use optimized token budget: ~7k input, 4k max output (no duplicate sections field)
- RAG embedding model runs on CPU to preserve GPU for LLM
- Nominatim takes 30-60 minutes on first run to import Bulgaria OSM data
- All monetary values are in Bulgarian Lev (BGN)
- OCR runs in CPU mode to reserve GPUs for LLM
- Scrapers run with 2-second delays to avoid rate limiting
- mobile.bg uses special URL slugs: `volkswagen` → `vw`, `mercedes-benz` → `mercedes-benz`
- VIN extraction from documents looks for "VIN:", "Рама №", "Шаси №" patterns
- GPU persistence mode enabled (`nvidia-smi -pm 1`) to prevent Xid 79 crashes
- Naredba 24 labor norms now use structured tables (faster than RAG lookup)
- Token estimation for Cyrillic: ~2.5 chars/token with Qwen model

## Documentation

- `docs/naredba24-implementation-status.md`: Full Naredba 24 compliance status and gaps
- `docs/physics-formulas.md`: All 15 physics formulas for ATE reports + future implementation roadmap
