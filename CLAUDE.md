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
- **LLM**: vLLM serving Qwen3-32B-AWQ across 6 GPUs with tensor parallelism (16k context)
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

The physics service provides crash reconstruction based on Bulgarian methodology from the "l.xlsx" formulas:

### Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /velocity-from-skid` | Calculate pre-braking speed from skid mark length |
| `POST /momentum-360` | Full 360° momentum analysis with angular vectors |
| `POST /impact-energy` | Calculate impact energy, delta-V, and damage severity |
| `POST /validate-claimed-speed` | Validate driver's claimed speed against physical evidence |
| `GET /formulas` | Returns all physics formulas and friction coefficients |

### Momentum 360 Analysis

Uses angular momentum conservation for two-vehicle collisions:
- **Input**: Vehicle masses, post-impact speeds, pre/post impact angles (α, β)
- **Output**: Pre-impact velocities (V1, V2), delta-V, impact energy
- **Angles**: 0° = East, 90° = North, 180° = West, 270° = South

Key formulas:
```
V1x = (m1*V1'*cos(β1) + m2*V2'*cos(β2)) / m1*cos(α1)
V1y = (m1*V1'*sin(β1) + m2*V2'*sin(β2)) / m1*sin(α1)
V1 = sqrt(V1x² + V1y²)
```

## Car Value Service (v3.3.0)

On-demand price scraper with VIN decoding and parts pricing (no database required):

### Data Flow
1. **Check Redis cache** (24h TTL for prices/parts, permanent for VIN)
2. **Live scrape cars.bg** (primary source for vehicle values)
3. **Live scrape mobile.bg** (backup source)
4. **Return vehicle info only** if scraping fails (no static fallback)

### Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /vin/{vin}` | Decode VIN via NHTSA API (make, model, year, country) |
| `GET /value-by-vin/{vin}` | Decode VIN + lookup market value |
| `GET /value/{make}/{model}/{year}` | Get car market value (on-demand scrape) |
| `POST /parts/search` | Search for parts prices across Bulgarian websites |

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

## RAG Service (Автотехническа Експертиза Knowledge Base)

The RAG service enables expert-level ATE report generation by providing relevant knowledge from:
- **Naredba 24**: Bulgarian regulation governing ATE methodology
- **Uchebnik ATE II**: Official textbook for certified automotive technical experts

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

### Technical Details

- **Embedding Model**: BGE-M3 (multilingual, runs on CPU)
- **Vector Database**: Qdrant (cosine similarity)
- **Chunk Size**: 512 tokens with 50 token overlap
- **Retrieval**: Top 5-8 most relevant chunks per query
- **Context Budget**: ~3000 tokens for RAG context in 16k window

## Notes

- LLM requires 6 GPUs with tensor parallelism; first startup downloads ~20GB model
- LLM context window increased to 16k tokens (from 8k) with 6 GPUs
- RAG embedding model runs on CPU to preserve GPU for LLM
- Nominatim takes 30-60 minutes on first run to import Bulgaria OSM data
- All monetary values are in Bulgarian Lev (BGN)
- OCR runs in CPU mode to reserve GPUs for LLM
- Scrapers run with 2-second delays to avoid rate limiting
- mobile.bg uses special URL slugs: `volkswagen` → `vw`, `mercedes-benz` → `mercedes-benz`
- VIN extraction from documents looks for "VIN:", "Рама №", "Шаси №" patterns
- GPU persistence mode enabled (`nvidia-smi -pm 1`) to prevent Xid 79 crashes
