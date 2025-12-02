# Legal AI - Technical Documentation

## Version 2.0.0 | Document Processing Pipeline for Bulgarian Automotive Insurance Claims

---

# Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Service Details](#3-service-details)
4. [Data Flow Diagrams](#4-data-flow-diagrams)
5. [Database Schema](#5-database-schema)
6. [LLM Extraction](#6-llm-extraction)
7. [Physics Calculations](#7-physics-calculations)
8. [Car Value Service](#8-car-value-service)
9. [Deployment](#9-deployment)
10. [API Reference](#10-api-reference)

---

# 1. Executive Summary

## 1.1 System Purpose

Legal AI is a document processing pipeline designed specifically for Bulgarian automotive insurance claims. The system automates the extraction, analysis, and enrichment of claim data from scanned documents (PDFs and images).

## 1.2 What the System Does

1. **Document Ingestion**: Accepts PDF and image files containing Bulgarian insurance claim documents
2. **Text Extraction**: Uses Tesseract OCR with Bulgarian/Russian/English language support to extract text
3. **Structured Data Extraction**: Employs a large language model (Qwen3-32B-AWQ) to parse unstructured text into structured JSON
4. **VIN Decoding**: Extracts and decodes Vehicle Identification Numbers (VINs) from documents using NHTSA API
5. **Location Enrichment**: Geocodes accident locations using Nominatim with Bulgaria OSM data
6. **Vehicle Valuation**: Provides current market values via on-demand scraping of Bulgarian car listing sites (cars.bg, mobile.bg)
7. **Physics Analysis**: Performs crash reconstruction using Bulgarian methodology (Momentum 360, Impact Theory)
8. **Settlement Recommendation**: Calculates recommended settlement amounts based on extracted data

## 1.3 Target Users

| User Type | Use Case |
|-----------|----------|
| Insurance Claims Adjusters | Automate initial claim processing and data extraction |
| Legal Professionals | Analyze crash physics and fault determination |
| Insurance Companies | Batch process claims for settlement calculations |
| Data Analysts | Extract structured data for reporting and analytics |
| Developers | Integrate with existing claims management systems |

## 1.4 Key Features

- Support for Bulgarian Cyrillic text (OCR optimized for Bulgarian documents)
- VIN extraction and decoding from document text (supports Bulgarian terms: "Rama No", "Shasi No")
- All monetary values in Bulgarian Lev (BGN)
- Physics calculations based on Bulgarian crash reconstruction methodology
- Real-time car market values from Bulgarian listing sites (on-demand scraping)
- Geocoding restricted to Bulgaria for accurate location data

---

# 2. System Architecture

## 2.1 High-Level Architecture Diagram

```
                                    +------------------------------------------+
                                    |              CLIENT LAYER                |
                                    |   (Web Browser, API Client, Mobile App)  |
                                    +--------------------+---------------------+
                                                         |
                                                         | HTTP/HTTPS
                                                         v
+----------------------------------------------------------------------------------------+
|                                        PORT 80                                          |
|  +---------------------------------------------------------------------------------+   |
|  |                           API GATEWAY (FastAPI)                                 |   |
|  |                                                                                 |   |
|  |   Endpoints:                                                                    |   |
|  |   - POST /process       (PDF/Image upload -> Full pipeline)                     |   |
|  |   - POST /process-text  (Raw text -> Skip OCR)                                  |   |
|  |   - GET  /health        (Service health status)                                 |   |
|  +---------------------------------------------------------------------------------+   |
+----------------------------------------------------------------------------------------+
                                             |
           +-----------+-----------+---------+---------+-----------+-----------+
           |           |           |                   |           |           |
           v           v           v                   v           v           v
    +----------+ +----------+ +-----------+     +----------+ +-----------+ +----------+
    |   OCR    | | Physics  | |    LLM    |     |Car Value | | Nominatim | |  WebUI   |
    | Service  | | Service  | |   (vLLM)  |     | Service  | | Geocoding | | (Debug)  |
    +----------+ +----------+ +-----------+     +----------+ +-----------+ +----------+
    | Port 8001| | Port 8004| | Port 8000 |     | Port 8003| | Port 8002 | | Port 3000|
    +----------+ +----------+ +-----------+     +----------+ +-----------+ +----------+
    | Tesseract| | Momentum | | Qwen3-32B |     | On-demand| | Bulgaria  | | Open     |
    | bul+rus  | | 360 &    | | AWQ       |     | scraping | | OSM Data  | | WebUI    |
    | +eng     | | Impact   | | 4 GPUs    |     | + VIN    | |           | |          |
    +----------+ | Theory   | | Tensor    |     | decode   | +-----------+ +----------+
                 +----------+ | Parallel  |     +----+-----+
                              +-----------+          |
                                    |                |
                                    v                v
                              +----------+    +----------+
                              |  Redis   |    |PostgreSQL|
                              |  :6379   |    |  :5432   |
                              +----------+    +----------+
                              | Cache    |    | Claims   |
                              | 24h TTL  |    | Parts    |
                              | prices   |    |          |
                              | VIN perm |    |          |
                              +----------+    +----------+
```

## 2.2 Component Descriptions

### 2.2.1 API Gateway (FastAPI)
- **Purpose**: Main orchestrator and entry point for all document processing
- **Technology**: Python FastAPI with async/await
- **Port**: 80
- **Responsibilities**:
  - Accept document uploads (multipart form data)
  - Coordinate calls to downstream services
  - Aggregate results into unified response
  - Store processed claims in PostgreSQL

### 2.2.2 OCR Service (Tesseract)
- **Purpose**: Extract text from scanned documents
- **Technology**: Tesseract OCR via pytesseract
- **Port**: 8001
- **Languages**: Bulgarian (bul) + Russian (rus) + English (eng)
- **Supported Formats**: PDF, PNG, JPG, JPEG, TIFF
- **Mode**: CPU-only (GPUs reserved for LLM)

### 2.2.3 LLM Service (vLLM)
- **Purpose**: Extract structured data from unstructured text
- **Technology**: vLLM serving Qwen/Qwen3-32B-AWQ
- **Port**: 8000
- **GPU Requirement**: 4 GPUs with tensor parallelism
- **API Compatibility**: OpenAI-compatible endpoints

### 2.2.4 Physics Service
- **Purpose**: Crash reconstruction and speed validation
- **Technology**: Python FastAPI with NumPy
- **Port**: 8004
- **Methods**: Momentum 360, Impact Theory, Dangerous Zone

### 2.2.5 Car Value Service (v4.0.0)
- **Purpose**: Provide current market values for vehicles, decode VINs, and calculate Naredba 24 compliant repair costs
- **Technology**: Python FastAPI with web scraping (httpx, BeautifulSoup) and NHTSA API
- **Port**: 8003
- **Data Sources**: On-demand scraping (cars.bg, mobile.bg), NHTSA VIN API
- **Caching**: Redis (24h TTL for prices, permanent for VIN decodes)
- **Naredba 24**: Full compliance with depreciation (чл. 12), labor norms (Глава III), paint costs (Глава IV)
- **Details**: See `docs/naredba24-implementation-status.md` for full implementation status

### 2.2.6 Nominatim (Geocoding)
- **Purpose**: Convert addresses to coordinates
- **Technology**: Nominatim with Bulgaria OSM data
- **Port**: 8002
- **Coverage**: Bulgaria only
- **Data**: OpenStreetMap Bulgaria extract

### 2.2.7 PostgreSQL
- **Purpose**: Persistent storage for claims and parts data
- **Port**: 5432
- **Database**: legal_ai

### 2.2.8 Redis
- **Purpose**: Caching layer for car values and VIN decodes
- **Port**: 6379
- **TTL**: 24 hours for price data, permanent for VIN decodes

### 2.2.9 Open WebUI (Debug)
- **Purpose**: ChatGPT-like interface for testing the LLM
- **Port**: 3000
- **Use**: Development and debugging only

## 2.3 Technology Stack

| Layer | Technology |
|-------|------------|
| API Framework | FastAPI (Python 3.11+) |
| LLM Serving | vLLM with Qwen3-32B-AWQ |
| OCR Engine | Tesseract 5.x |
| Database | PostgreSQL 15 |
| Cache | Redis 7 |
| Geocoding | Nominatim 4.4 |
| Containerization | Docker Compose |
| Web Scraping | httpx, BeautifulSoup4 |
| VIN Decoding | NHTSA vPIC API |
| Async DB | asyncpg |
| HTTP Client | httpx (async) |

---

# 3. Service Details

## 3.1 Gateway Service

### 3.1.1 Purpose
The Gateway service is the main entry point for all document processing. It orchestrates the entire pipeline from document upload to final enriched result.

### 3.1.2 Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/process` | POST | Process PDF/image through full pipeline |
| `/process-text` | POST | Process raw text (skip OCR) |
| `/health` | GET | Health check with service status |
| `/` | GET | Service info and supported formats |

### 3.1.3 Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `LLM_URL` | `http://localhost:8000` | vLLM service URL |
| `OCR_URL` | `http://localhost:8001` | OCR service URL |
| `NOMINATIM_URL` | `http://localhost:8002` | Geocoding service URL |
| `CAR_VALUE_URL` | `http://localhost:8003` | Car value service URL |
| `PHYSICS_URL` | `http://localhost:8004` | Physics service URL |
| `DATABASE_URL` | - | PostgreSQL connection string |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection string |
| `REQUEST_TIMEOUT` | `300` | Request timeout in seconds |

### 3.1.4 Dependencies
- LLM Service (required for extraction)
- OCR Service (required for `/process` endpoint)
- Nominatim (optional - degrades gracefully)
- Car Value Service (optional - degrades gracefully)
- Physics Service (optional - degrades gracefully)
- PostgreSQL (optional - results not stored if unavailable)
- Redis (optional - no caching if unavailable)

### 3.1.5 Example Request/Response

**Request (POST /process):**
```bash
curl -X POST "http://localhost:80/process" \
  -F "file=@claim_document.pdf"
```

**Request (POST /process-text):**
```bash
curl -X POST "http://localhost:80/process-text" \
  -H "Content-Type: application/json" \
  -d '{"text": "Claim #12345\nAccident Date: 2024-01-15..."}'
```

**Response:**
```json
{
  "claim_number": "12345",
  "accident_date": "2024-01-15",
  "accident_time": "14:30",
  "accident_location": {
    "address": "ul. Vitosha 15",
    "city": "Sofia",
    "latitude": 42.6977,
    "longitude": 23.3219
  },
  "vehicles": [
    {
      "vin": "WVWZZZ3CZWE123456",
      "registration": "CB 1234 AB",
      "make": "Volkswagen",
      "model": "Golf",
      "year": 2018,
      "owner_name": "Ivan Petrov",
      "insurance_company": "Bulstrad",
      "current_market_value_bgn": 25000,
      "estimated_damage_bgn": 5000
    }
  ],
  "parties": [
    {
      "name": "Ivan Petrov",
      "role": "driver",
      "vehicle_index": 0
    }
  ],
  "fault_determination": {
    "primary_fault_party": "Ivan Petrov",
    "fault_percentage": 100,
    "reasoning": "Failed to yield at intersection",
    "traffic_violations": ["Failure to yield right of way"]
  },
  "settlement_recommendation": {
    "amount_bgn": 5500,
    "components": {
      "vehicle_damage": 5000,
      "medical_expenses": 500
    }
  },
  "physics_analysis": {
    "vehicle_a_pre_impact_kmh": 45.5,
    "delta_v_a_kmh": 25.3,
    "physics_method": "momentum_360"
  },
  "confidence_score": 0.85,
  "processing_time_seconds": 12.5,
  "errors": [],
  "warnings": []
}
```

---

## 3.2 OCR Service

### 3.2.1 Purpose
Extract text from scanned documents using Tesseract OCR with Bulgarian Cyrillic support.

### 3.2.2 Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ocr` | POST | Extract text from uploaded document |
| `/health` | GET | Health check with Tesseract version |
| `/` | GET | Service info |

### 3.2.3 Supported Formats
- PDF (multi-page support)
- PNG
- JPG/JPEG
- TIFF/TIF

### 3.2.4 Language Support
The OCR uses a combination of language packs for optimal Bulgarian document processing:
- `bul` - Bulgarian
- `rus` - Russian (for additional Cyrillic coverage)
- `eng` - English (for mixed documents)

### 3.2.5 Example Request/Response

**Request:**
```bash
curl -X POST "http://localhost:8001/ocr" \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "text": "ZASTRAKHOVATELNA POLITSA\nNomer: 12345\n...",
  "pages": 2,
  "page_texts": [
    "ZASTRAKHOVATELNA POLITSA...",
    "Page 2 content..."
  ],
  "filename": "document.pdf"
}
```

---

## 3.3 LLM Service (vLLM)

### 3.3.1 Purpose
Serve the Qwen3-32B-AWQ model for structured data extraction from insurance claim text.

### 3.3.2 Model Configuration

| Parameter | Value |
|-----------|-------|
| Model | `Qwen/Qwen3-32B-AWQ` |
| Quantization | AWQ |
| Tensor Parallel Size | 4 |
| GPU Memory Utilization | 85% |
| Max Model Length | 4096 |
| Max Sequences | 8 |

### 3.3.3 Endpoints (OpenAI-Compatible)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (used by Gateway) |
| `/v1/models` | GET | List available models |
| `/health` | GET | Service health check |

### 3.3.4 GPU Requirements
- Minimum: 4 NVIDIA GPUs with 16GB+ VRAM each
- Recommended: 4x A100 or 4x RTX 4090
- The model uses tensor parallelism across all 4 GPUs
- GPU persistence mode should be enabled (`nvidia-smi -pm 1`) to prevent Xid 79 crashes

### 3.3.5 Startup Time
- First startup: ~5 minutes (model loading)
- First run: ~20GB model download from HuggingFace

---

## 3.4 Physics Service

### 3.4.1 Purpose
Perform crash physics calculations using Bulgarian methodology for:
- Speed validation from physical evidence
- Pre-impact velocity reconstruction
- Delta-V (change in velocity) calculation
- Dangerous zone and safe speed analysis

### 3.4.2 Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/momentum-360` | POST | Full 360-degree momentum analysis |
| `/impact-theory` | POST | Matrix-based impact theory analysis |
| `/velocity-from-skid` | POST | Calculate speed from skid marks |
| `/validate-claimed-speed` | POST | Validate driver's claimed speed |
| `/dangerous-zone` | POST | Calculate stopping distance |
| `/impact-energy` | POST | Calculate impact energy, delta-V, and damage severity |
| `/formulas` | GET | Return all physics formulas used |
| `/health` | GET | Health check |

### 3.4.3 Friction Coefficients

| Surface Type | Coefficient |
|--------------|-------------|
| Dry Asphalt | 0.7 |
| Wet Asphalt | 0.5 |
| Dry Concrete | 0.75 |
| Wet Concrete | 0.55 |
| Gravel | 0.4 |
| Snow | 0.2 |
| Ice | 0.1 |

---

## 3.5 Car Value Service (v3.0.0)

### 3.5.1 Purpose
Provide current market values for vehicles in Bulgaria and decode VINs through:
- VIN decoding via NHTSA API with local WMI fallback
- On-demand live web scraping (cars.bg, mobile.bg)
- Redis caching for performance

### 3.5.2 Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/vin/{vin}` | GET | Decode VIN via NHTSA API (make, model, year, country, engine info) |
| `/value-by-vin/{vin}` | GET | Decode VIN + lookup market value |
| `/value/{make}/{model}/{year}` | GET | Get vehicle market value (on-demand scrape) |
| `/parts/{make}/{category}` | GET | Get parts prices |
| `/parts/estimate-damage` | GET | Estimate repair cost |
| `/health` | GET | Health check |

### 3.5.3 Data Source Priority (v3.0.0)

```
+---------------------+
|  1. Redis Cache     |  <-- Fastest (24h TTL for prices, permanent for VIN)
+---------------------+
         |
         v (cache miss)
+---------------------+
|  2. Live Scrape     |  <-- cars.bg (primary)
|     cars.bg         |
+---------------------+
         |
         v (no results)
+---------------------+
|  3. Live Scrape     |  <-- mobile.bg (backup)
|     mobile.bg       |
+---------------------+
         |
         v (no results)
+---------------------+
|  4. Return vehicle  |  <-- No price, manual valuation required
|     info only       |
+---------------------+
```

**Note**: Unlike v2.0.0, there is no database storage of scraped listings, no static fallback prices, and no pre-computed aggregates. All pricing is on-demand.

### 3.5.4 VIN Decoding

The VIN decoding feature was added in v3.0.0:

**Primary Source**: NHTSA vPIC API
- Free, no API key required
- Works for EU vehicles (decodes make, model, year, engine, body class)
- Returns detailed vehicle specifications

**Fallback**: Local WMI Decode
- Uses first 3 characters of VIN (World Manufacturer Identifier)
- Returns manufacturer and country of origin
- Model details not available in fallback mode

**Caching**: VIN decodes are cached permanently in Redis (VINs don't change)

**Validation**: VINs are validated for:
- Exactly 17 characters
- No I, O, Q characters (prohibited in VINs)
- Alphanumeric characters only
- Check digit validation (position 9)

### 3.5.5 Supported Makes
BMW, Mercedes-Benz, Audi, Volkswagen, Toyota, Opel, Ford, Renault, Peugeot, Skoda, Honda, Mazda, Nissan, Volvo, Hyundai, Kia

### 3.5.6 Removed Features (from v2.0.0)
The following features were removed in the v3.0.0 rewrite:
- Database storage of scraped listings (`car_listings` table)
- Pre-computed price aggregates (`car_prices_aggregated` table)
- Static fallback prices (`vehicle_prices` table)
- Admin endpoints (`/admin/scrape`, `/admin/aggregate`)
- Cron job for scheduled scraping (`scripts/price_updater.sh`)
- Scraper job logging (`scraper_runs` table)

---

## 3.6 Nominatim (Geocoding)

### 3.6.1 Purpose
Convert Bulgarian addresses to geographic coordinates (latitude/longitude).

### 3.6.2 Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/search` | GET | Search for address |
| `/reverse` | GET | Reverse geocode coordinates |
| `/status` | GET | Service status |

### 3.6.3 Data Source
- OpenStreetMap Bulgaria extract
- URL: `https://download.geofabrik.de/europe/bulgaria-latest.osm.pbf`
- Updates: Daily replication from Geofabrik

### 3.6.4 First-Run Import Time
- Initial import: 30-60 minutes
- This imports the entire Bulgaria OSM dataset

---

# 4. Data Flow Diagrams

## 4.1 /process Endpoint (Full Pipeline with OCR)

```
+--------+
| Client |
+---+----+
    |
    | 1. POST /process
    |    (multipart form: PDF/image)
    v
+---+--------+
|  Gateway   |
|  Service   |
+---+--------+
    |
    | 2. POST /ocr
    |    (forward file)
    v
+---+--------+     +------------------+
|    OCR     |---->| Return extracted |
|  Service   |     | text + pages     |
+------------+     +------------------+
    |
    | 3. Extracted text
    v
+---+--------+
|  Gateway   |
|  Service   |
+---+--------+
    |
    | 4. POST /v1/chat/completions
    |    (prompt + extracted text)
    |    (includes VIN extraction)
    v
+---+--------+     +------------------+
|    LLM     |---->| Return structured|
|  (vLLM)    |     | JSON with VIN    |
+------------+     +------------------+
    |
    | 5. Structured claim data (includes VIN)
    v
+---+--------+
|  Gateway   |
|  Service   |
+---+--------+
    |
    +-------------+-------------+-------------+
    |             |             |             |
    | 6a. GET     | 6b. GET     | 6c. POST    |
    | /search     | /value/...  | /momentum-  |
    |             | or /value-  | 360         |
    |             | by-vin/...  |             |
    v             v             v             |
+----------+ +----------+ +----------+       |
| Nominatim| |Car Value |  | Physics  |       |
| Geocoding| | Service  |  | Service  |       |
+----+-----+ +----+-----+  +----+-----+       |
     |            |             |             |
     |            +--+          |             |
     |               |          |             |
     | lat/lon    +--+--+       | velocity    |
     |            |Redis|       | analysis    |
     |            |Cache|       |             |
     |            +--+--+       |             |
     |               |          |             |
     v            v  v          v             |
+---+------------+--+---------+-------------+
|              Gateway Service               |
|         (aggregate all results)            |
+---+----------------------------------------+
    |
    | 7. INSERT INTO processed_claims
    v
+---+--------+
| PostgreSQL |
+------------+
    |
    | 8. Return ProcessingResult
    v
+---+----+
| Client |
+--------+
```

## 4.2 /process-text Endpoint (Skip OCR)

```
+--------+
| Client |
+---+----+
    |
    | 1. POST /process-text
    |    {"text": "raw document text..."}
    v
+---+--------+
|  Gateway   |
|  Service   |
+---+--------+
    |
    | 2. POST /v1/chat/completions
    |    (prompt + raw text)
    v
+---+--------+     +------------------+
|    LLM     |---->| Return structured|
|  (vLLM)    |     | JSON extraction  |
+------------+     +------------------+
    |
    | 3. Structured claim data
    v
+---+--------+
|  Gateway   |
|  Service   |
+---+--------+
    |
    +-------------+-------------+-------------+
    |             |             |             |
    v             v             v             |
+----------+ +----------+ +----------+       |
| Nominatim| |Car Value | | Physics  |       |
+----+-----+ +----+-----+ +----+-----+       |
     |            |            |             |
     v            v            v             |
+---+------------+------------+-------------+
|              Gateway Service              |
|         (aggregate all results)           |
+---+---------------------------------------+
    |
    | 4. INSERT INTO processed_claims
    v
+---+--------+
| PostgreSQL |
+------------+
    |
    | 5. Return ProcessingResult
    v
+---+----+
| Client |
+--------+
```

## 4.3 Car Value Lookup Flow (v3.0.0)

```
+-------------------+
| GET /value/{make} |
|     /{model}/{yr} |
+--------+----------+
         |
         v
+--------+----------+
| Check Redis Cache |
| Key: car:v3:{m}:  |
|      {model}:{yr} |
+--------+----------+
         |
    +----+----+
    |         |
  HIT       MISS
    |         |
    v         v
+-------+ +--------+----------+
| Return| | Scrape cars.bg    |
| cached| | (live request)    |
| result| +--------+----------+
+-------+          |
              +----+----+
              |         |
           FOUND    NOT FOUND
              |         |
              v         v
        +-------+ +--------+----------+
        | Cache | | Scrape mobile.bg  |
        | &     | | (backup)          |
        | Return| +--------+----------+
        +-------+          |
                      +----+----+
                      |         |
                   FOUND    NOT FOUND
                      |         |
                      v         v
                +-------+ +--------+----------+
                | Cache | | Return vehicle    |
                | &     | | info only         |
                | Return| | (no price)        |
                +-------+ +--------+----------+
```

## 4.4 VIN Lookup Flow (v3.0.0)

```
+-------------------+
| GET /vin/{vin}    |
+--------+----------+
         |
         v
+--------+----------+
| Validate VIN      |
| (17 chars, no IOQ)|
+--------+----------+
         |
         v
+--------+----------+
| Check Redis Cache |
| Key: vin:{vin}    |
+--------+----------+
         |
    +----+----+
    |         |
  HIT       MISS
    |         |
    v         v
+-------+ +--------+----------+
| Return| | Query NHTSA API   |
| cached| | (vPIC decode)     |
| result| +--------+----------+
+-------+          |
              +----+----+
              |         |
           SUCCESS   FAILED
              |         |
              v         v
        +-------+ +--------+----------+
        | Cache | | Local WMI decode  |
        | perm  | | (first 3 chars)   |
        | &     | +--------+----------+
        | Return|          |
        +-------+          v
                    +--------+----------+
                    | Cache perm &      |
                    | Return (limited)  |
                    +-------------------+
```

---

# 5. Database Schema

## 5.1 Entity-Relationship Diagram

```
+------------------+       +----------------------+
|  vehicle_specs   |       |   processed_claims   |
+------------------+       +----------------------+
| id (PK)          |       | id (PK)              |
| make             |       | claim_number         |
| model            |       | filename             |
| year_from        |       | processing_time_ms   |
| year_to          |       | confidence_score     |
| weight_kg        |       | fault_percentage     |
| length_mm        |       | settlement_amount_bgn|
| width_mm         |       | result_json (JSONB)  |
| height_mm        |       | raw_ocr_text         |
| engine_cc        |       | created_at           |
| horsepower       |       +----------------------+
| body_type        |
| created_at       |
+------------------+

+------------------+
|    car_parts     |
+------------------+
| id (PK)          |
| make             |
| model            |
| year_from        |
| year_to          |
| part_category    |
| part_name        |
| part_name_bg     |
| oem_price_bgn    |
| aftermarket_     |
|   price_bgn      |
| labor_hours      |
| labor_rate_bgn   |
| source           |
| updated_at       |
+------------------+
```

**Note**: In v3.0.0, the following tables were removed:
- `car_listings` - Raw scraped listings (moved to on-demand scraping + Redis)
- `car_prices_aggregated` - Pre-computed averages (now computed on-demand)
- `car_price_history` - Historical price trends (no longer tracked)
- `scraper_runs` - Scraper job log (no scheduled scraping)
- `vehicle_prices` - Static fallback prices (no longer used)

## 5.2 Table Descriptions

### 5.2.1 processed_claims
Stores all processed insurance claims with their results.

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL | Primary key |
| `claim_number` | VARCHAR(100) | Extracted claim reference number |
| `filename` | VARCHAR(255) | Original uploaded filename |
| `processing_time_ms` | INT | Total processing time in milliseconds |
| `confidence_score` | FLOAT | LLM extraction confidence (0-1) |
| `fault_percentage` | INT | Determined fault percentage (0-100) |
| `settlement_amount_bgn` | FLOAT | Recommended settlement in BGN |
| `result_json` | JSONB | Complete ProcessingResult as JSON |
| `raw_ocr_text` | TEXT | Raw OCR-extracted text |
| `created_at` | TIMESTAMP | Record creation timestamp |

### 5.2.2 vehicle_specs
Vehicle specifications for reference data.

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL | Primary key |
| `make` | VARCHAR(100) | Manufacturer |
| `model` | VARCHAR(100) | Model name |
| `year_from` | INT | Production start year |
| `year_to` | INT | Production end year |
| `weight_kg` | INT | Curb weight in kg |
| `length_mm` | INT | Length in mm |
| `width_mm` | INT | Width in mm |
| `height_mm` | INT | Height in mm |
| `engine_cc` | INT | Engine displacement |
| `horsepower` | INT | Engine power |
| `body_type` | VARCHAR(50) | Body style (sedan, hatchback, etc.) |

### 5.2.3 car_parts
Parts pricing for damage estimation.

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL | Primary key |
| `make` | VARCHAR(100) | Manufacturer (NULL = universal) |
| `model` | VARCHAR(100) | Model (NULL = all models) |
| `year_from` | INT | Applicable from year |
| `year_to` | INT | Applicable to year |
| `part_category` | VARCHAR(100) | Category (body, engine, etc.) |
| `part_name` | VARCHAR(200) | Part name in English |
| `part_name_bg` | VARCHAR(200) | Part name in Bulgarian |
| `oem_price_bgn` | FLOAT | OEM part price |
| `aftermarket_price_bgn` | FLOAT | Aftermarket price |
| `labor_hours` | FLOAT | Installation labor hours |
| `labor_rate_bgn` | FLOAT | Hourly labor rate (default 50) |

---

# 6. LLM Extraction

## 6.1 Prompt Structure

The LLM extraction uses a structured prompt that instructs the model to extract specific fields from Bulgarian insurance claim documents.

### 6.1.1 System Message
```
You are a legal document analysis expert specializing in Bulgarian
automotive insurance claims. Extract information accurately and
return valid JSON only.
```

### 6.1.2 User Prompt Template

```
Analyze this Bulgarian automotive insurance claim document and extract
structured information.

Document Text:
{document_text}

Extract the following information and return as valid JSON:

{
  "claim_number": "string or null - the claim/case reference number",
  "accident_date": "YYYY-MM-DD format or null",
  "accident_time": "HH:MM format or null",
  "accident_location": {
    "address": "street address in Bulgarian or English",
    "city": "city name"
  },
  "vehicles": [
    {
      "vin": "17-character Vehicle Identification Number (VIN) or null",
      "registration": "license plate number",
      "make": "manufacturer (e.g., Volkswagen, BMW)",
      "model": "model name (e.g., Golf, 3 Series)",
      "year": integer year of manufacture,
      "mass_kg": integer vehicle mass in kg,
      "owner_name": "owner's full name",
      "insurance_company": "insurance company name",
      "policy_number": "insurance policy number",
      "damage_description": "description of damage in English",
      "estimated_damage": float amount in BGN,
      "skid_distance_m": float length of skid marks in meters or null,
      "post_impact_travel_m": float distance after impact in meters or null,
      "claimed_speed_kmh": float driver's claimed speed in km/h or null,
      "pre_impact_angle_deg": float angle before impact (0-360) or null,
      "post_impact_angle_deg": float angle after impact (0-360) or null
    }
  ],
  "parties": [
    {
      "name": "person's full name",
      "role": "driver|passenger|pedestrian|witness",
      "vehicle_index": integer index into vehicles array or null,
      "injuries": "injury description or null",
      "statement_summary": "brief summary of their statement"
    }
  ],
  "accident_description": "detailed description in English",
  "fault_determination": {
    "primary_fault_party": "name of person primarily at fault",
    "fault_percentage": integer 0-100,
    "reasoning": "explanation of fault determination",
    "traffic_violations": ["list of traffic rules violated"]
  },
  "police_report": {
    "report_number": "police report number or null",
    "officer_name": "officer name or null",
    "findings": "police findings summary or null"
  },
  "settlement_recommendation": {
    "amount_bgn": float total recommended settlement in BGN,
    "components": {
      "vehicle_damage": float,
      "medical_expenses": float,
      "lost_income": float,
      "pain_and_suffering": float
    },
    "reasoning": "explanation of settlement calculation"
  },
  "collision_details": {
    "collision_type": "head_on|rear_end|side_impact|angle or null",
    "road_surface": "dry_asphalt|wet_asphalt|gravel|snow|ice or null",
    "road_grade_percent": float road incline percentage or 0,
    "impact_angle_deg": float angle of impact impulse (0-360) or null,
    "restitution_coefficient": float (0.0-1.0, default 0.4) or null
  },
  "risk_factors": ["list of factors that could affect the claim"],
  "confidence_score": float 0.0-1.0 indicating extraction confidence
}
```

### 6.1.3 Important Instructions (included in prompt)
1. All monetary amounts must be in Bulgarian Lev (BGN)
2. Dates must be in ISO format (YYYY-MM-DD)
3. Translate Bulgarian text to English in the response
4. If information is missing or unclear, use null
5. For fault_percentage, 100 means fully at fault, 0 means not at fault
6. Be conservative with confidence_score - lower if document is unclear
7. Extract actual values from the document, do not make up information
8. Return ONLY the JSON object, no additional text
9. For physics angles: 0 deg = East, 90 deg = North, 180 deg = West, 270 deg = South
10. VIN (Vehicle Identification Number) is exactly 17 characters (letters A-H,J-N,P,R-Z and digits 0-9, excludes I,O,Q)
11. VIN may appear as "VIN:", "Rama No", "Shasi No", or similar in Bulgarian documents

## 6.2 VIN Extraction (New in v2.0.0)

The LLM extraction prompt now includes specific instructions for extracting VINs:

### 6.2.1 VIN Field in Vehicle Schema
```json
{
  "vehicles": [
    {
      "vin": "17-character Vehicle Identification Number (VIN) or null",
      ...
    }
  ]
}
```

### 6.2.2 Bulgarian VIN Terms
The prompt instructs the LLM to look for VINs under various Bulgarian labels:
- `VIN:` - Standard English label
- `Rama No` (Cyrillic) - Bulgarian for "Frame Number"
- `Shasi No` (Cyrillic) - Bulgarian for "Chassis Number"

### 6.2.3 VIN Validation
Extracted VINs are validated using a regex pattern:
```python
vin_pattern = re.compile(r'^[A-HJ-NPR-Z0-9]{17}$')
```

Invalid VINs are set to `null` and a warning is added to the response.

## 6.3 LLM Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `temperature` | 0.1 | Low temperature for factual extraction |
| `max_tokens` | 4096 | Sufficient for complete JSON response |
| `response_format` | `{"type": "json_object"}` | Force JSON output |

## 6.4 Document Truncation

Documents longer than 15,000 characters are truncated to preserve beginning and end:

```python
max_doc_length = 15000
if len(document_text) > max_doc_length:
    half = max_doc_length // 2
    document_text = (
        document_text[:half] +
        "\n\n[... middle section truncated ...]\n\n" +
        document_text[-half:]
    )
```

## 6.5 JSON Parsing and Validation

The system includes robust JSON parsing that handles:
- Markdown code blocks around JSON
- Trailing text after JSON
- Single quotes instead of double quotes
- Trailing commas
- Unquoted keys

### 6.5.1 Validation Rules

| Field | Validation |
|-------|------------|
| `accident_date` | Must match YYYY-MM-DD pattern |
| `accident_time` | Must match HH:MM pattern |
| `vehicle.year` | Must be between 1900-2030 |
| `vehicle.vin` | Must match 17-char VIN pattern (no I,O,Q) |
| `fault_percentage` | Must be 0-100 |
| `confidence_score` | Must be 0.0-1.0 |
| `settlement_amount` | Must be non-negative |

## 6.6 Example Input/Output

### Input (OCR-extracted text):
```
KONSTATIVEN PROTOKOL No 123456
Data: 15.01.2024 g.
Chas: 14:30

MYASTO NA PROIZSHESTVIYETO:
gr. Sofiya, bul. Vitosha No 15

UCHASTNITSI:
1. Ivan Petrov Ivanov - vodach na MPS s reg. No SV 1234 AV
   Marka: Folksvagen Golf, 2018 g.
   VIN: WVWZZZ3CZWE123456
   Zastrahovatel: Bulstrad, politsa No 12345678

2. Mariya Georgieva Dimitrova - vodach na MPS s reg. No SA 5678 VS
   Marka: BMW 320, 2020 g.
   Rama No: WBA8E9C50JK123456

OPISANIE: Pri presichane na krastovishtyeto vodachat na MPS 1
ne dava predimstvo na MPS 2...
```

### Output (LLM extraction):
```json
{
  "claim_number": "123456",
  "accident_date": "2024-01-15",
  "accident_time": "14:30",
  "accident_location": {
    "address": "bul. Vitosha 15",
    "city": "Sofia"
  },
  "vehicles": [
    {
      "vin": "WVWZZZ3CZWE123456",
      "registration": "CB 1234 AB",
      "make": "Volkswagen",
      "model": "Golf",
      "year": 2018,
      "mass_kg": 1300,
      "owner_name": "Ivan Petrov Ivanov",
      "insurance_company": "Bulstrad",
      "policy_number": "12345678"
    },
    {
      "vin": "WBA8E9C50JK123456",
      "registration": "CA 5678 BC",
      "make": "BMW",
      "model": "320",
      "year": 2020,
      "mass_kg": 1500
    }
  ],
  "parties": [
    {
      "name": "Ivan Petrov Ivanov",
      "role": "driver",
      "vehicle_index": 0
    },
    {
      "name": "Maria Georgieva Dimitrova",
      "role": "driver",
      "vehicle_index": 1
    }
  ],
  "accident_description": "At the intersection, the driver of vehicle 1 failed to yield right of way to vehicle 2",
  "fault_determination": {
    "primary_fault_party": "Ivan Petrov Ivanov",
    "fault_percentage": 100,
    "reasoning": "Failed to yield at intersection",
    "traffic_violations": ["Failure to yield right of way"]
  },
  "confidence_score": 0.85
}
```

---

# 7. Physics Calculations

## 7.1 Overview

The Physics Service implements Bulgarian crash reconstruction methodology based on formulas from the "l.xlsx" reference document. It provides two main analysis methods:

1. **Momentum 360** - Vector-based momentum conservation with angular components
2. **Impact Theory** - Matrix equation system solution

## 7.2 Core Formulas

### 7.2.1 Post-Impact Velocity from Travel Distance

Formula [5]:
```
u = sqrt(2 * mu * g * sigma + Vy^2)
```

Where:
- `u` = post-impact velocity (m/s)
- `mu` = friction coefficient
- `g` = gravity (9.81 m/s^2)
- `sigma` = post-impact travel distance (m)
- `Vy` = final velocity (usually 0)

### 7.2.2 Braking Deceleration

```
j = mu * g * cos(grade) +/- g * sin(grade)
```

Simplified for small grades:
```
j = mu * g +/- grade_factor * g
```

### 7.2.3 Momentum 360 (Vector Analysis)

Pre-impact velocities using angular momentum conservation:

```
V1 = ((sin(beta1 - alpha2) * m1 * u1) + (sin(beta2 - alpha2) * m2 * u2))
     / (sin(alpha1 - alpha2) * m1)

V2 = ((sin(beta1 - alpha1) * m1 * u1) + (sin(beta2 - alpha1) * m2 * u2))
     / (sin(alpha2 - alpha1) * m2)
```

Where:
- `V1, V2` = pre-impact velocities
- `u1, u2` = post-impact velocities
- `m1, m2` = vehicle masses
- `alpha1, alpha2` = pre-impact angles
- `beta1, beta2` = post-impact angles

### 7.2.4 Impact Theory (Matrix Method)

Matrix equation system:
```
[a11  a12]   [V1]   [b1]
[a21  a22] * [V2] = [b2]
```

Coefficients:
```
a11 = cos(alpha1) * m1
a12 = cos(alpha2) * m2
a21 = -cos(alpha1 - alpha_s)
a22 = cos(alpha2 - alpha_s)
b1 = (cos(beta1) * m1 * u1) + (cos(beta2) * m2 * u2)
b2 = ((cos(beta2 - alpha_s) * u2) - (cos(beta1 - alpha_s) * u1)) / k
```

Where:
- `alpha_s` = impact impulse directrix angle
- `k` = coefficient of restitution

### 7.2.5 Delta-V (Change in Velocity)

```
Delta_V = sqrt(V^2 + u^2 - 2*V*u*cos(beta - alpha))
```

This is a key indicator for injury severity assessment.

### 7.2.6 Dangerous Zone

Formula [11] - Stopping distance from reaction point:
```
S_oz = V * (t_r + t_sp + 0.5*t_n) + V^2 / (2*j)
```

Where:
- `t_r` = driver reaction time (default 1.0s)
- `t_sp` = brake system activation time (default 0.2s)
- `t_n` = brake force rise time (default 0.0s)

Formula [16] - Safe speed:
```
Solve: V*t + V^2/(2*j) = O_u for V
```

## 7.3 Angle Convention

```
         90 deg (North)
              |
              |
180 deg ------+------ 0 deg (East)
(West)        |
              |
         270 deg (South)
```

- 0 deg = East (vehicle heading right)
- 90 deg = North (vehicle heading up)
- 180 deg = West (vehicle heading left)
- 270 deg = South (vehicle heading down)

## 7.4 Example Calculations

### 7.4.1 Velocity from Skid Marks

**Input:**
- Skid distance: 20 meters
- Friction coefficient: 0.7 (dry asphalt)
- Grade: 0%

**Calculation:**
```
j = 0.7 * 9.81 = 6.867 m/s^2
u = sqrt(2 * 6.867 * 20) = sqrt(274.68) = 16.57 m/s = 59.7 km/h
```

**API Call:**
```bash
curl -X POST "http://localhost:8004/velocity-from-skid" \
  -H "Content-Type: application/json" \
  -d '{"skid_distance_m": 20, "friction_coefficient": 0.7}'
```

**Response:**
```json
{
  "velocity_ms": 16.57,
  "velocity_kmh": 59.7,
  "formula": "u = sqrt(2 * mu * g * sigma + Vy^2)",
  "inputs": {
    "skid_distance_m": 20,
    "friction_coefficient": 0.7,
    "braking_deceleration_ms2": 6.87
  }
}
```

### 7.4.2 Momentum 360 Analysis

**Input:**
- Vehicle A: 1400 kg, post-impact travel 8m, alpha=0 deg, beta=45 deg
- Vehicle B: 1600 kg, post-impact travel 6m, alpha=180 deg, beta=135 deg
- Friction: 0.7

**API Call:**
```bash
curl -X POST "http://localhost:8004/momentum-360" \
  -H "Content-Type: application/json" \
  -d '{
    "vehicle_a": {
      "mass_kg": 1400,
      "post_impact_travel_m": 8,
      "alpha_deg": 0,
      "beta_deg": 45,
      "final_velocity_ms": 0
    },
    "vehicle_b": {
      "mass_kg": 1600,
      "post_impact_travel_m": 6,
      "alpha_deg": 180,
      "beta_deg": 135,
      "final_velocity_ms": 0
    },
    "friction_coefficient": 0.7
  }'
```

**Response:**
```json
{
  "vehicle_a_post_impact_ms": 10.49,
  "vehicle_a_post_impact_kmh": 37.8,
  "vehicle_b_post_impact_ms": 9.09,
  "vehicle_b_post_impact_kmh": 32.7,
  "vehicle_a_impact_velocity_ms": 15.2,
  "vehicle_a_impact_velocity_kmh": 54.7,
  "vehicle_b_impact_velocity_ms": 12.8,
  "vehicle_b_impact_velocity_kmh": 46.1,
  "delta_v_a_ms": 8.5,
  "delta_v_a_kmh": 30.6,
  "delta_v_b_ms": 7.2,
  "delta_v_b_kmh": 25.9,
  "method": "momentum_360"
}
```

### 7.4.3 Speed Validation

**Input:**
- Claimed speed: 50 km/h
- Skid marks: 25 meters
- Friction: 0.7

**API Call:**
```bash
curl -X POST "http://localhost:8004/validate-claimed-speed" \
  -H "Content-Type: application/json" \
  -d '{
    "claimed_speed_kmh": 50,
    "skid_distance_m": 25,
    "friction_coefficient": 0.7
  }'
```

**Response:**
```json
{
  "claimed_speed_valid": false,
  "calculated_speed_kmh": 66.8,
  "claimed_speed_kmh": 50,
  "speed_difference_kmh": 16.8,
  "confidence": 0.9,
  "physics_method": "skid_marks",
  "explanation": "Based on 25m skid marks with mu=0.7. Physics shows ~67 km/h, 17 km/h HIGHER than claimed"
}
```

---

# 8. Car Value Service

## 8.1 Overview (v3.0.0)

The Car Value Service was completely rewritten in v3.0.0. It now provides:
- **VIN decoding** via NHTSA API with local WMI fallback
- **On-demand scraping** from cars.bg and mobile.bg
- **Redis caching** (24h TTL for prices, permanent for VIN)
- **Parts pricing** from database for damage estimation

**Removed in v3.0.0:**
- Database storage of scraped listings
- Pre-computed price aggregates
- Static fallback prices
- Admin endpoints for scraping/aggregation
- Cron job for scheduled scraping

## 8.2 VIN Decoding

### 8.2.1 Endpoint: GET /vin/{vin}

Decode a VIN to get vehicle information.

**Request:**
```bash
curl "http://localhost:8003/vin/WVWZZZ3CZWE123456"
```

**Response (NHTSA decode):**
```json
{
  "vin": "WVWZZZ3CZWE123456",
  "valid": true,
  "source": "nhtsa",
  "make": "Volkswagen",
  "model": "Golf",
  "year": 2018,
  "series": null,
  "body_class": "Hatchback",
  "drive_type": "Front Wheel Drive",
  "engine_cylinders": "4",
  "engine_displacement_l": "2.0",
  "engine_hp": "147",
  "fuel_type": "Gasoline",
  "plant_city": "Wolfsburg",
  "plant_country": "Germany",
  "manufacturer": "Volkswagen AG",
  "vehicle_type": "PASSENGER CAR",
  "check_digit_valid": true
}
```

**Response (local WMI fallback):**
```json
{
  "vin": "WVWZZZ3CZWE123456",
  "valid": true,
  "source": "local",
  "make": "Volkswagen",
  "model": null,
  "year": 2018,
  "country": "Germany",
  "wmi": "WVW",
  "vds": "ZZZ3CZ",
  "vis": "WE123456",
  "check_digit_valid": true,
  "note": "Limited decode - model/engine details not available"
}
```

### 8.2.2 VIN Validation

VINs are validated for:
- Exactly 17 characters
- No I, O, Q characters (prohibited in VINs)
- Alphanumeric characters only (A-H, J-N, P-R, Z, 0-9)
- Check digit validation (position 9)

**Invalid VIN response:**
```json
{
  "detail": "Invalid VIN format. Must be 17 alphanumeric characters (no I, O, Q)"
}
```

### 8.2.3 WMI Mapping (Local Fallback)

The service includes a mapping of World Manufacturer Identifiers:

| WMI | Manufacturer | Country |
|-----|--------------|---------|
| WBA, WBS, WBY | BMW | Germany |
| WDB, WDC, WDD | Mercedes-Benz | Germany |
| WAU, WUA | Audi | Germany |
| WVW, WV1, WV2 | Volkswagen | Germany |
| VF1 | Renault | France |
| VF3 | Peugeot | France |
| TMB | Skoda | Czech Republic |
| JTD | Toyota | Japan |
| KMH | Hyundai | South Korea |
| KNA, KNM | Kia | South Korea |
| YV1, YV4 | Volvo | Sweden |
| W0L | Opel | Germany |

## 8.3 Scraping Strategy

### 8.3.1 cars.bg Scraping

**URL Pattern:**
```
https://www.cars.bg/carslist.php?mession=search&make={make}&model={model}&yearFrom={year}&yearTo={year}&currencyId=1
```

**Price Extraction:**
cars.bg stores prices in HTML comments:
```html
<!--32,999 BGN-->
<!--15,500 EUR-->
```

**Regex Patterns:**
```python
bgn_pattern = r'<!--\s*([\d,\.]+)\s*BGN\s*-->'
eur_pattern = r'<!--\s*([\d,\.]+)\s*EUR\s*-->'
```

### 8.3.2 mobile.bg Scraping

**URL Pattern:**
```
https://www.mobile.bg/obiavi/avtomobili-dzhipove/{make_slug}/{model_slug}?yearFrom={year}&yearTo={year}
```

**Special URL Slugs:**
| Make | Slug |
|------|------|
| Volkswagen | vw |
| Mercedes-Benz | mercedes-benz |
| Alfa Romeo | alfa-romeo |
| Land Rover | land-rover |

**Encoding:**
mobile.bg uses Windows-1251 encoding for Bulgarian text.

**Price Patterns:**
```python
bgn_pattern = r'(\d{1,3}(?:\s?\d{3})*)\s*(?:lv|BGN)'
eur_pattern = r'(\d{1,3}(?:\s?\d{3})*)\s*EUR'
```

### 8.3.3 EUR to BGN Conversion
```python
price_bgn = price_eur * 1.96  # Fixed rate (BGN is pegged to EUR)
```

## 8.4 Outlier Filtering

Prices are filtered using the Interquartile Range (IQR) method:

```python
if len(prices) > 5:
    q1_idx = len(prices) // 4
    q3_idx = (3 * len(prices)) // 4
    q1, q3 = prices[q1_idx], prices[q3_idx]
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    prices = [p for p in prices if lower_bound <= p <= upper_bound]
```

## 8.5 Caching Strategy

### 8.5.1 Redis Cache
- **Price Cache Key:** `car:v3:{make}:{model}:{year}`
- **Price TTL:** 24 hours (86400 seconds)
- **VIN Cache Key:** `vin:{vin}`
- **VIN TTL:** Permanent (VINs don't change)
- **Serialization:** JSON

### 8.5.2 Cache Flow
```
Request -> Check Redis -> Hit? Return cached
                       -> Miss? Scrape -> Cache & Return (or return info only)
```

## 8.6 No Price Available

When no live data is available from scraping (unlike v2.0.0, there is no static fallback):

```json
{
  "make": "SomeMake",
  "model": "SomeModel",
  "year": 2020,
  "average_price_bgn": null,
  "source": "none",
  "currency": "BGN",
  "note": "No market data available - manual valuation required"
}
```

## 8.7 Parts Valuation

### 8.7.1 Part Categories
- **body**: Bumpers, doors, fenders, hood, trunk, mirrors, glass
- **electrical**: Headlights, taillights, radiator, AC condenser
- **suspension**: Shock absorbers, control arms
- **engine**: Alternator, starter, water pump
- **interior**: Seats, steering wheel, dashboard

### 8.7.2 Pricing Structure
Each part has:
- **OEM price**: Original manufacturer part price
- **Aftermarket price**: Generic/third-party part price
- **Labor hours**: Typical installation time
- **Labor rate**: 50 BGN/hour default

### 8.7.3 Damage Cost Calculation
```
Total Cost = SUM(Part Price + (Labor Hours * Labor Rate))
```

### 8.7.4 Example: Damage Estimate

**Request:**
```bash
curl "http://localhost:8003/parts/estimate-damage?make=BMW&parts=Front%20Bumper,Headlight%20Left&use_oem=true"
```

**Response:**
```json
{
  "make": "BMW",
  "pricing_type": "OEM",
  "parts_cost_bgn": 3500,
  "labor_cost_bgn": 175,
  "total_cost_bgn": 3675,
  "breakdown": [
    {
      "part": "Front Bumper",
      "part_cost_bgn": 1500,
      "labor_cost_bgn": 125
    },
    {
      "part": "Headlight Left",
      "part_cost_bgn": 2000,
      "labor_cost_bgn": 50
    }
  ],
  "currency": "BGN"
}
```

---

# 9. Deployment

## 9.1 System Requirements

### 9.1.1 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 8 cores | 16+ cores |
| RAM | 32 GB | 64 GB |
| GPU | 4x 16GB VRAM | 4x 24GB+ VRAM |
| Storage | 100 GB SSD | 500 GB NVMe SSD |
| Network | 100 Mbps | 1 Gbps |

### 9.1.2 GPU Requirements
- **Quantity:** 4 NVIDIA GPUs
- **VRAM:** 16GB+ per GPU
- **Supported:** A100, A10, RTX 4090, RTX 3090
- **Driver:** NVIDIA Driver 525+ with CUDA 12+
- **Persistence Mode:** Enable with `nvidia-smi -pm 1` to prevent Xid 79 crashes

## 9.2 Docker Compose Setup

### 9.2.1 Pre-requisites
- Docker 24+
- Docker Compose 2.20+
- NVIDIA Container Toolkit (for GPU support)

### 9.2.2 Initial Setup

```bash
# Clone repository
git clone <repository-url>
cd legal-ai

# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env

# Run initialization script
./init.sh

# Start all services
docker compose up -d
```

### 9.2.3 Service Startup Order

```
1. PostgreSQL (database)
2. Redis (cache)
3. LLM (vLLM) - takes 5 min to load model
4. OCR Service
5. Nominatim - takes 30-60 min on first run
6. Car Value Service
7. Physics Service
8. Gateway
9. WebUI (optional)
```

## 9.3 Environment Variables

### 9.3.1 Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `HF_TOKEN` | HuggingFace token for model download | `hf_xxxxx` |
| `POSTGRES_USER` | PostgreSQL username | `legal_ai` |
| `POSTGRES_PASSWORD` | PostgreSQL password | `changeme` |

### 9.3.2 Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NOMINATIM_PASSWORD` | `nominatim` | Nominatim admin password |
| `REQUEST_TIMEOUT` | `300` | Gateway request timeout (seconds) |

### 9.3.3 Sample .env File

```bash
# HuggingFace token (required for model download)
HF_TOKEN=hf_your_token_here

# Database credentials
POSTGRES_USER=legal_ai
POSTGRES_PASSWORD=your_secure_password

# Nominatim (optional)
NOMINATIM_PASSWORD=nominatim_admin_pass
```

## 9.4 Monitoring Commands

### 9.4.1 Service Status
```bash
# Check all container status
docker compose ps

# View all logs
docker compose logs -f

# View specific service logs
docker compose logs -f llm
docker compose logs -f gateway
docker compose logs -f nominatim
```

### 9.4.2 GPU Monitoring
```bash
# Real-time GPU usage
watch -n 1 nvidia-smi

# GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Enable persistence mode (recommended)
sudo nvidia-smi -pm 1
```

### 9.4.3 Health Checks
```bash
# Run health check script
python test_pipeline.py --health-only

# Manual health checks
curl http://localhost:80/health      # Gateway
curl http://localhost:8000/health    # LLM
curl http://localhost:8001/health    # OCR
curl http://localhost:8003/health    # Car Value
curl http://localhost:8004/health    # Physics
```

## 9.5 Building Services

```bash
# Build all services
docker compose build

# Build specific service
docker compose build gateway
docker compose build ocr
docker compose build car_value
docker compose build physics

# Force rebuild without cache
docker compose build --no-cache gateway
```

## 9.6 First-Run Considerations

### 9.6.1 LLM Model Download
- First startup downloads ~20GB model from HuggingFace
- Requires valid HF_TOKEN
- Takes 5-10 minutes on fast connection

### 9.6.2 Nominatim OSM Import
- First startup imports Bulgaria OSM data
- Takes 30-60 minutes
- Creates ~10GB of data in volume
- Progress visible in logs: `docker compose logs -f nominatim`

### 9.6.3 Database Initialization
- Schema created automatically from `database/init.sql`
- Seed data loaded on first run
- Migrations in `database/migrations/` directory:
  - `002_price_aggregator.sql` - Parts tables for damage estimation
  - `003_drop_unused_price_tables.sql` - Removes old price tables (v3.0.0 cleanup)

---

# 10. API Reference

## 10.1 Gateway API

### 10.1.1 POST /process

Process a document through the full pipeline (OCR + LLM + enrichment).

**Request:**
```http
POST /process HTTP/1.1
Host: localhost:80
Content-Type: multipart/form-data

file: <binary PDF or image>
```

**cURL Example:**
```bash
curl -X POST "http://localhost:80/process" \
  -F "file=@claim_document.pdf"
```

**Response Schema:**
```json
{
  "claim_number": "string | null",
  "accident_date": "YYYY-MM-DD | null",
  "accident_time": "HH:MM | null",
  "accident_location": {
    "address": "string | null",
    "city": "string | null",
    "latitude": "number | null",
    "longitude": "number | null"
  },
  "vehicles": [{
    "vin": "string (17 chars) | null",
    "registration": "string | null",
    "make": "string | null",
    "model": "string | null",
    "year": "integer | null",
    "mass_kg": "integer | null",
    "owner_name": "string | null",
    "insurance_company": "string | null",
    "policy_number": "string | null",
    "damage_description": "string | null",
    "estimated_damage_bgn": "number | null",
    "current_market_value_bgn": "number | null",
    "market_value_source": "string | null",
    "skid_distance_m": "number | null",
    "post_impact_travel_m": "number | null",
    "claimed_speed_kmh": "number | null",
    "pre_impact_angle_deg": "number | null",
    "post_impact_angle_deg": "number | null"
  }],
  "parties": [{
    "name": "string | null",
    "role": "driver | passenger | pedestrian | witness",
    "vehicle_index": "integer | null",
    "injuries": "string | null",
    "statement_summary": "string | null"
  }],
  "accident_description": "string | null",
  "fault_determination": {
    "primary_fault_party": "string | null",
    "fault_percentage": "integer 0-100 | null",
    "reasoning": "string | null",
    "traffic_violations": ["string"]
  },
  "police_report": {
    "report_number": "string | null",
    "officer_name": "string | null",
    "findings": "string | null"
  },
  "settlement_recommendation": {
    "amount_bgn": "number | null",
    "components": {
      "vehicle_damage": "number | null",
      "medical_expenses": "number | null",
      "lost_income": "number | null",
      "pain_and_suffering": "number | null"
    },
    "reasoning": "string | null"
  },
  "physics_analysis": {
    "claimed_speed_valid": "boolean | null",
    "calculated_speed_kmh": "number | null",
    "speed_validation_method": "string | null",
    "speed_validation_explanation": "string | null",
    "vehicle_a_post_impact_kmh": "number | null",
    "vehicle_b_post_impact_kmh": "number | null",
    "vehicle_a_pre_impact_kmh": "number | null",
    "vehicle_b_pre_impact_kmh": "number | null",
    "delta_v_a_kmh": "number | null",
    "delta_v_b_kmh": "number | null",
    "physics_method": "momentum_360 | impact_theory | null",
    "physics_confidence": "number | null",
    "physics_notes": ["string"]
  },
  "risk_factors": ["string"],
  "confidence_score": "number 0.0-1.0",
  "processing_time_seconds": "number",
  "raw_ocr_text": "string | null",
  "errors": ["string"],
  "warnings": ["string"]
}
```

**Response Codes:**
| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Invalid file type or no filename |
| 500 | Processing error |

---

### 10.1.2 POST /process-text

Process raw text directly (skip OCR).

**Request:**
```http
POST /process-text HTTP/1.1
Host: localhost:80
Content-Type: application/json

{
  "text": "string (required)",
  "filename": "string (optional, default: direct_text_input)"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:80/process-text" \
  -H "Content-Type: application/json" \
  -d '{"text": "Claim document text here..."}'
```

**Response:**
Same schema as POST /process

**Response Codes:**
| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Text too short or empty |
| 500 | Processing error |

---

### 10.1.3 GET /health

Health check endpoint.

**Request:**
```http
GET /health HTTP/1.1
Host: localhost:80
```

**Response:**
```json
{
  "status": "healthy",
  "service": "gateway",
  "llm_url": "http://llm:8000",
  "database": true
}
```

---

## 10.2 OCR API

### 10.2.1 POST /ocr

Extract text from document.

**Request:**
```http
POST /ocr HTTP/1.1
Host: localhost:8001
Content-Type: multipart/form-data

file: <binary PDF or image>
```

**Response:**
```json
{
  "text": "Combined text from all pages...",
  "pages": 2,
  "page_texts": [
    "Page 1 text...",
    "Page 2 text..."
  ],
  "filename": "document.pdf"
}
```

**Response Codes:**
| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Unsupported file type |
| 500 | OCR processing error |

---

## 10.3 Physics API

### 10.3.1 POST /momentum-360

Full momentum 360 analysis.

**Request:**
```json
{
  "vehicle_a": {
    "mass_kg": 1400,
    "post_impact_travel_m": 8,
    "alpha_deg": 0,
    "beta_deg": 45,
    "final_velocity_ms": 0
  },
  "vehicle_b": {
    "mass_kg": 1600,
    "post_impact_travel_m": 6,
    "alpha_deg": 180,
    "beta_deg": 135,
    "final_velocity_ms": 0
  },
  "friction_coefficient": 0.7,
  "grade_percent": 0,
  "restitution_coefficient": 0.4,
  "alpha_s_deg": 0
}
```

**Response:**
```json
{
  "vehicle_a_post_impact_ms": 10.49,
  "vehicle_a_post_impact_kmh": 37.8,
  "vehicle_b_post_impact_ms": 9.09,
  "vehicle_b_post_impact_kmh": 32.7,
  "vehicle_a_impact_velocity_ms": 15.2,
  "vehicle_a_impact_velocity_kmh": 54.7,
  "vehicle_b_impact_velocity_ms": 12.8,
  "vehicle_b_impact_velocity_kmh": 46.1,
  "delta_v_a_ms": 8.5,
  "delta_v_a_kmh": 30.6,
  "delta_v_b_ms": 7.2,
  "delta_v_b_kmh": 25.9,
  "velocity_plan": {...},
  "method": "momentum_360",
  "notes": ["u1 = 10.49 m/s from sigma1 = 8m"]
}
```

---

### 10.3.2 POST /velocity-from-skid

Calculate velocity from skid marks.

**Request (Query Parameters):**
```
skid_distance_m: float (required)
friction_coefficient: float (default: 0.7)
grade_percent: float (default: 0)
final_velocity_ms: float (default: 0)
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8004/velocity-from-skid?skid_distance_m=20&friction_coefficient=0.7"
```

**Response:**
```json
{
  "velocity_ms": 16.57,
  "velocity_kmh": 59.7,
  "formula": "u = sqrt(2 * mu * g * sigma + Vy^2)",
  "inputs": {
    "skid_distance_m": 20,
    "friction_coefficient": 0.7,
    "grade_percent": 0,
    "final_velocity_ms": 0,
    "braking_deceleration_ms2": 6.87
  }
}
```

---

### 10.3.3 POST /validate-claimed-speed

Validate driver's claimed speed.

**Request (Query Parameters):**
```
claimed_speed_kmh: float (required)
skid_distance_m: float (optional)
post_impact_travel_m: float (optional)
friction_coefficient: float (default: 0.7)
```

**Response:**
```json
{
  "claimed_speed_valid": false,
  "calculated_speed_kmh": 66.8,
  "claimed_speed_kmh": 50,
  "speed_difference_kmh": 16.8,
  "confidence": 0.9,
  "physics_method": "skid_marks",
  "explanation": "Based on 25m skid marks with mu=0.7. Physics shows ~67 km/h, 17 km/h HIGHER than claimed"
}
```

---

### 10.3.4 POST /impact-energy

Calculate impact energy, delta-V, and damage severity.

**Response:**
```json
{
  "total_impact_energy_j": 125000,
  "energy_dissipated_j": 75000,
  "delta_v_a_kmh": 30.6,
  "delta_v_b_kmh": 25.9,
  "estimated_damage_severity": "moderate"
}
```

---

### 10.3.5 GET /formulas

Return all physics formulas used.

**Response:**
```json
{
  "service": "Crash Physics Service v2.0",
  "based_on": "Bulgarian crash reconstruction (l.xlsx)",
  "methods": {
    "momentum_360": {
      "description": "Vector-based momentum analysis with angular components",
      "formulas": {...}
    },
    "impact_theory": {...},
    "dangerous_zone": {...}
  },
  "friction_coefficients": {
    "dry_asphalt": 0.7,
    "wet_asphalt": 0.5,
    ...
  },
  "reaction_time_defaults": {...}
}
```

---

## 10.4 Car Value API (v3.0.0)

### 10.4.1 GET /vin/{vin}

Decode VIN to get vehicle information.

**Request:**
```http
GET /vin/WVWZZZ3CZWE123456 HTTP/1.1
Host: localhost:8003
```

**Response:**
```json
{
  "vin": "WVWZZZ3CZWE123456",
  "valid": true,
  "source": "nhtsa",
  "make": "Volkswagen",
  "model": "Golf",
  "year": 2018,
  "body_class": "Hatchback",
  "drive_type": "Front Wheel Drive",
  "engine_cylinders": "4",
  "engine_displacement_l": "2.0",
  "fuel_type": "Gasoline",
  "plant_country": "Germany",
  "check_digit_valid": true,
  "from_cache": false
}
```

**Error Response (invalid VIN):**
```json
{
  "detail": "Invalid VIN format. Must be 17 alphanumeric characters (no I, O, Q)"
}
```

---

### 10.4.2 GET /value-by-vin/{vin}

Get car value by VIN (decode + lookup).

**Request:**
```http
GET /value-by-vin/WVWZZZ3CZWE123456 HTTP/1.1
Host: localhost:8003
```

**Response:**
```json
{
  "vin": "WVWZZZ3CZWE123456",
  "vehicle": {
    "vin": "WVWZZZ3CZWE123456",
    "valid": true,
    "source": "nhtsa",
    "make": "Volkswagen",
    "model": "Golf",
    "year": 2018
  },
  "value": {
    "make": "Volkswagen",
    "model": "Golf",
    "year": 2018,
    "average_price_bgn": 28500.00,
    "min_price_bgn": 22000,
    "max_price_bgn": 35000,
    "sample_size": 15,
    "source": "cars.bg",
    "currency": "BGN"
  }
}
```

---

### 10.4.3 GET /value/{make}/{model}/{year}

Get vehicle market value.

**Request:**
```http
GET /value/Volkswagen/Golf/2018 HTTP/1.1
Host: localhost:8003
```

**Response:**
```json
{
  "make": "Volkswagen",
  "model": "Golf",
  "year": 2018,
  "average_price_bgn": 28500.00,
  "min_price_bgn": 22000,
  "max_price_bgn": 35000,
  "median_price_bgn": 27500.00,
  "sample_size": 15,
  "source": "cars.bg",
  "currency": "BGN",
  "confidence": 1.0,
  "from_cache": false
}
```

**Response (no data available):**
```json
{
  "make": "SomeMake",
  "model": "SomeModel",
  "year": 2020,
  "average_price_bgn": null,
  "source": "none",
  "currency": "BGN",
  "note": "No market data available - manual valuation required"
}
```

---

### 10.4.4 GET /parts/{make}/{category}

Get parts prices.

**Request:**
```http
GET /parts/BMW/body?model=3%20Series HTTP/1.1
Host: localhost:8003
```

**Response:**
```json
{
  "make": "BMW",
  "model": "3 Series",
  "category": "body",
  "parts": [
    {
      "part_name": "Front Bumper",
      "part_name_bg": "Predna bronya",
      "oem_price_bgn": 1500,
      "aftermarket_price_bgn": 600,
      "labor_hours": 2.5,
      "labor_cost_bgn": 125,
      "total_oem_bgn": 1625,
      "total_aftermarket_bgn": 725
    }
  ],
  "currency": "BGN"
}
```

---

### 10.4.5 GET /parts/estimate-damage

Estimate total repair cost.

**Request:**
```http
GET /parts/estimate-damage?make=BMW&parts=Front%20Bumper,Headlight%20Left&use_oem=true HTTP/1.1
Host: localhost:8003
```

**Response:**
```json
{
  "make": "BMW",
  "pricing_type": "OEM",
  "parts_cost_bgn": 3500,
  "labor_cost_bgn": 175,
  "total_cost_bgn": 3675,
  "breakdown": [
    {"part": "Front Bumper", "part_cost_bgn": 1500, "labor_cost_bgn": 125},
    {"part": "Headlight Left", "part_cost_bgn": 2000, "labor_cost_bgn": 50}
  ],
  "currency": "BGN"
}
```

---

## 10.5 Nominatim API

### 10.5.1 GET /search

Search for address.

**Request:**
```http
GET /search?q=bul.%20Vitosha%2015,%20Sofia,%20Bulgaria&format=json&limit=1&countrycodes=bg HTTP/1.1
Host: localhost:8002
```

**Response:**
```json
[
  {
    "lat": "42.6977",
    "lon": "23.3219",
    "display_name": "15, bul. Vitosha, Sofia, Bulgaria",
    "type": "building"
  }
]
```

---

# Appendix A: Error Codes and Troubleshooting

## A.1 Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `OCR failed: 503` | OCR service unavailable | Check `docker compose ps ocr` |
| `LLM extraction failed` | vLLM not ready | Wait for model loading, check `docker compose logs llm` |
| `Geocoding failed` | Nominatim import incomplete | Wait for OSM import to finish |
| `Car value lookup failed` | Scraping blocked/timeout | Check rate limits, try later |
| `Invalid VIN format` | VIN doesn't match 17-char pattern | Verify VIN has no I, O, Q characters |
| `Physics analysis failed` | Invalid input data | Verify vehicle masses and angles |
| `Xid 79 GPU crash` | GPU persistence mode disabled | Run `nvidia-smi -pm 1` |

## A.2 Performance Optimization

| Issue | Solution |
|-------|----------|
| Slow OCR | Increase OCR service replicas |
| LLM timeout | Reduce max_tokens or document length |
| Slow geocoding | Pre-warm Nominatim cache |
| Car value slow | Prime Redis cache with common models |
| GPU crashes | Enable persistence mode: `nvidia-smi -pm 1` |

---

# Appendix B: Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-01 | Initial release |
| 2.0.0 | 2024-06 | Added Momentum 360, Impact Theory, multi-source car values |
| 2.0.0 | 2024-11 | Car Value Service v3.0.0: VIN decoding, on-demand scraping, removed DB storage. Gateway: VIN extraction in LLM prompt. Database: Removed unused price tables. LLM: 4 GPUs (was 6). |
| 3.0.0 | 2025-12 | Car Value Service v4.0.0: Full Naredba 24 compliance (depreciation, labor norms, paint costs). ATE Report optimization: removed duplicate sections, ~4.5 tokens/s generation speed. See `docs/naredba24-implementation-status.md`. |

---

# Appendix C: Migration Notes (v2.0.0)

## C.1 Car Value Service Changes (v2.0.0 to v3.0.0)

### Removed Features
- Database storage of scraped listings
- Pre-computed price aggregates
- Static fallback prices
- Admin endpoints (`/admin/scrape`, `/admin/aggregate`)
- Cron job (`scripts/price_updater.sh`)

### Added Features
- VIN decoding via NHTSA API (`/vin/{vin}`)
- Combined VIN + value lookup (`/value-by-vin/{vin}`)
- Permanent VIN caching in Redis

### Database Migration
Run migration `003_drop_unused_price_tables.sql` to remove:
- `car_listings`
- `car_prices_aggregated`
- `car_price_history`
- `scraper_runs`
- `vehicle_prices`

## C.2 Gateway Changes

- VIN field added to vehicle extraction
- LLM prompt updated with VIN instructions
- VIN validation regex added to extractors
- Schemas updated with VIN field

## C.3 Infrastructure Changes

- LLM now uses 4 GPUs (was 6)
- GPU persistence mode recommended (`nvidia-smi -pm 1`)

---

# Appendix D: License and Credits

- **OCR**: Tesseract (Apache 2.0)
- **LLM**: Qwen3-32B (Apache 2.0)
- **Geocoding**: Nominatim (GPL v2)
- **VIN Decoding**: NHTSA vPIC API (Public Domain)
- **Physics Formulas**: Bulgarian crash reconstruction methodology (l.xlsx)

---

*Document generated: 2024-11*
*Legal AI Technical Documentation v2.0.0*
