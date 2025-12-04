"""
API Gateway - Main entry point for document processing pipeline

Single /process endpoint that orchestrates:
1. OCR text extraction
2. LLM-based information extraction
3. Geocoding enrichment
4. Car value lookup
5. Physics-based crash analysis
6. Result storage
"""

import os
import re
import json
import time
import logging
import asyncio
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
from openai import AsyncOpenAI
import asyncpg
import redis

from schemas import (
    ProcessingResult,
    LocationData,
    VehicleData,
    PartyData,
    FaultDetermination,
    PoliceReportData,
    SettlementRecommendation,
    SettlementComponents,
    PhysicsAnalysis,
    PartsEstimate,
    PartPricing
)
from extractors import extract_with_llm, validate_extraction, enrich_extraction_with_fallback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration from environment
LLM_URL = os.environ.get("LLM_URL", "http://localhost:8000")
OCR_URL = os.environ.get("OCR_URL", "http://localhost:8001")
NOMINATIM_URL = os.environ.get("NOMINATIM_URL", "http://localhost:8002")
CAR_VALUE_URL = os.environ.get("CAR_VALUE_URL", "http://localhost:8003")
PHYSICS_URL = os.environ.get("PHYSICS_URL", "http://localhost:8004")
RAG_URL = os.environ.get("RAG_URL", "http://localhost:8005")
DATABASE_URL = os.environ.get("DATABASE_URL")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "300"))

# LLM model name (vLLM uses the model name as-is)
LLM_MODEL = "Qwen/Qwen3-32B-AWQ"

app = FastAPI(
    title="Legal AI Document Processing Gateway",
    description="Process Bulgarian automotive insurance claim documents",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clients
llm_client: Optional[AsyncOpenAI] = None
db_pool: Optional[asyncpg.Pool] = None
redis_client: Optional[redis.Redis] = None

# Concurrency control - only 1 ATE report at a time to prevent GPU overload
ate_report_semaphore = asyncio.Semaphore(1)
ate_report_queue_size = 0  # Track waiting requests


@app.on_event("startup")
async def startup():
    """Initialize clients on startup."""
    global llm_client, db_pool, redis_client

    # Initialize OpenAI client pointing to vLLM
    # Set 45 min timeout for long report generation (~5000 tokens at 3 tokens/s = 28 min)
    llm_client = AsyncOpenAI(
        base_url=f"{LLM_URL}/v1",
        api_key="not-needed",  # vLLM doesn't require API key
        timeout=2700.0  # 45 minutes in seconds
    )
    logger.info(f"LLM client configured for {LLM_URL}")

    # Connect to PostgreSQL
    if DATABASE_URL:
        try:
            db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
            logger.info("Connected to PostgreSQL")
        except Exception as e:
            logger.warning(f"PostgreSQL connection failed: {e}")

    # Connect to Redis
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("Connected to Redis")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        redis_client = None


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    if db_pool:
        await db_pool.close()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "gateway",
        "llm_url": LLM_URL,
        "database": db_pool is not None
    }


@app.post("/process", response_model=ProcessingResult)
async def process_document(file: UploadFile = File(...)):
    """
    Process an insurance claim document.

    1. Extract text using OCR
    2. Extract structured data using LLM
    3. Enrich with geocoding, car values, parts pricing, physics

    Supports: PDF, PNG, JPG, JPEG, TIFF

    Returns complete ProcessingResult with all extracted and enriched data.
    """
    start_time = time.time()

    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = file.filename.lower().split(".")[-1]
    if ext not in ["pdf", "png", "jpg", "jpeg", "tiff", "tif"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}"
        )

    logger.info(f"Processing file: {file.filename}")

    # Step 1: OCR
    try:
        ocr_result = await call_ocr_service(file)
        raw_text = ocr_result.get("text", "")
        logger.info(f"OCR extracted {len(raw_text)} characters from {ocr_result.get('pages', 1)} pages")
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return ProcessingResult(
            errors=[f"OCR failed: {str(e)}"],
            processing_time_seconds=time.time() - start_time,
            raw_ocr_text=""
        )

    if not raw_text:
        return ProcessingResult(
            errors=["No text could be extracted from document"],
            processing_time_seconds=time.time() - start_time,
            raw_ocr_text=""
        )

    # Step 2: LLM Extraction
    try:
        extracted = await extract_with_llm(llm_client, LLM_MODEL, raw_text)
        extracted, validation_warnings = validate_extraction(extracted)
        # Fallback extraction for VINs and parts if LLM missed them
        extracted, fallback_info = enrich_extraction_with_fallback(extracted, raw_text)
        for info in fallback_info:
            logger.info(info)
        logger.info(f"LLM extraction complete, confidence: {extracted.get('confidence_score', 0)}")
    except Exception as e:
        logger.error(f"LLM extraction failed: {e}")
        return ProcessingResult(
            errors=[f"LLM extraction failed: {str(e)}"],
            processing_time_seconds=time.time() - start_time,
            raw_ocr_text=raw_text
        )

    # Step 3: Enrich and build result (shared logic)
    result = await process_extraction_result(extracted, raw_text, file.filename)

    logger.info(f"Processing complete in {result.processing_time_seconds}s")
    return result


from pydantic import BaseModel

class TextInput(BaseModel):
    """Input model for text processing endpoint."""
    text: str
    filename: Optional[str] = "direct_text_input"


@app.post("/process-text", response_model=ProcessingResult)
async def process_text(input_data: TextInput):
    """
    Process raw text directly (skip OCR).

    Useful for:
    - Testing LLM extraction without OCR
    - Processing already-extracted text
    - Debugging the pipeline

    Input: {"text": "your document text here"}
    """
    start_time = time.time()
    raw_text = input_data.text

    if not raw_text or len(raw_text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Text too short or empty")

    logger.info(f"Processing text input: {len(raw_text)} characters")

    # Step 1: LLM Extraction (skip OCR)
    try:
        extracted = await extract_with_llm(llm_client, LLM_MODEL, raw_text)
        extracted, validation_warnings = validate_extraction(extracted)
        # Fallback extraction for VINs and parts if LLM missed them
        extracted, fallback_info = enrich_extraction_with_fallback(extracted, raw_text)
        for info in fallback_info:
            logger.info(info)
        logger.info(f"LLM extraction complete, confidence: {extracted.get('confidence_score', 0)}")
    except Exception as e:
        logger.error(f"LLM extraction failed: {e}")
        return ProcessingResult(
            errors=[f"LLM extraction failed: {str(e)}"],
            processing_time_seconds=time.time() - start_time,
            raw_ocr_text=raw_text
        )

    # Step 2: Enrich and build result (shared logic)
    result = await process_extraction_result(extracted, raw_text, input_data.filename)

    logger.info(f"Text processing complete in {result.processing_time_seconds}s")
    return result


async def call_ocr_service(file: UploadFile) -> dict:
    """Call the OCR service to extract text."""
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        # Reset file position
        await file.seek(0)
        content = await file.read()

        files = {"file": (file.filename, content, file.content_type or "application/octet-stream")}
        response = await client.post(f"{OCR_URL}/ocr", files=files)

        if response.status_code != 200:
            raise Exception(f"OCR service error: {response.status_code} - {response.text}")

        return response.json()


def clean_bulgarian_address(address: str) -> str:
    """Clean Bulgarian address for better geocoding results."""
    import re
    if not address:
        return ""

    # Remove quotes around street names
    address = re.sub(r'["\'\„\"]', '', address)

    # Remove intersection info (кръстовище с = intersection with)
    address = re.sub(r',?\s*кръстовище\s+с\s+.*$', '', address, flags=re.IGNORECASE)

    # Replace № with space
    address = address.replace('№', ' ')

    # Clean up multiple spaces
    address = re.sub(r'\s+', ' ', address).strip()

    # Expand common abbreviations
    address = re.sub(r'\bбул\.\s*', 'булевард ', address, flags=re.IGNORECASE)
    address = re.sub(r'\bул\.\s*', 'улица ', address, flags=re.IGNORECASE)
    address = re.sub(r'\bгр\.\s*', '', address, flags=re.IGNORECASE)  # Remove city prefix

    return address


async def geocode_location(location: dict) -> dict:
    """
    Geocode an address using Nominatim structured search.

    Uses structured parameters (street=, city=) for more accurate results
    than free-form queries. Falls back to street-only if full address fails.
    """
    import re as re_module

    # Extract and clean address components
    street = ""
    city = ""

    if location.get("address"):
        street = clean_bulgarian_address(location["address"])
    if location.get("city"):
        city = clean_bulgarian_address(location["city"])

    if not street and not city:
        return {}

    async with httpx.AsyncClient(timeout=10) as client:
        # Strategy 1: Structured search with street and city
        if street and city:
            params = {
                "street": street,
                "city": city,
                "country": "Bulgaria",
                "format": "json",
                "limit": 5,  # Get multiple results to filter
                "addressdetails": 1
            }
            logger.info(f"Geocoding structured: street='{street}', city='{city}'")

            response = await client.get(f"{NOMINATIM_URL}/search", params=params)

            if response.status_code == 200:
                results = response.json()
                # Filter to prefer road/street results over shops/POIs
                for r in results:
                    if r.get("class") in ("highway", "place", "boundary"):
                        logger.info(f"Geocoding found street: {r.get('display_name', '')[:60]}")
                        return {
                            "lat": float(r["lat"]),
                            "lon": float(r["lon"])
                        }
                # If no road found, use first result
                if results:
                    logger.info(f"Geocoding using first result: {results[0].get('display_name', '')[:60]}")
                    return {
                        "lat": float(results[0]["lat"]),
                        "lon": float(results[0]["lon"])
                    }

        # Strategy 2: Try with street only (without house numbers) in city
        if street:
            street_no_numbers = re_module.sub(r'\s*\d+\s*', ' ', street).strip()
            street_no_numbers = re_module.sub(r'\s+', ' ', street_no_numbers)  # Clean double spaces

            if street_no_numbers:
                params = {
                    "street": street_no_numbers,
                    "city": city if city else "София",  # Default to Sofia
                    "country": "Bulgaria",
                    "format": "json",
                    "limit": 5,
                    "addressdetails": 1
                }
                logger.info(f"Geocoding fallback: street='{street_no_numbers}', city='{params['city']}'")

                response = await client.get(f"{NOMINATIM_URL}/search", params=params)

                if response.status_code == 200:
                    results = response.json()
                    # Prefer highway/road results
                    for r in results:
                        if r.get("class") in ("highway", "place"):
                            logger.info(f"Geocoding found road: {r.get('display_name', '')[:60]}")
                            return {
                                "lat": float(r["lat"]),
                                "lon": float(r["lon"])
                            }
                    if results:
                        return {
                            "lat": float(results[0]["lat"]),
                            "lon": float(results[0]["lon"])
                        }

        # Strategy 3: Free-form search as last resort
        query = f"{street}, {city}, Bulgaria" if city else f"{street}, Bulgaria"
        logger.info(f"Geocoding free-form fallback: {query}")

        response = await client.get(
            f"{NOMINATIM_URL}/search",
            params={
                "q": query,
                "format": "json",
                "limit": 1
                # Note: removed countrycodes=bg as it was too restrictive for Bulgarian addresses
            }
        )

        if response.status_code == 200:
            results = response.json()
            if results:
                return {
                    "lat": float(results[0]["lat"]),
                    "lon": float(results[0]["lon"])
                }

    return {}


async def get_car_value(make: str, model: str, year: int) -> dict:
    """Get car market value from the car value service."""
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(
            f"{CAR_VALUE_URL}/value/{make}/{model}/{year}"
        )

        if response.status_code == 200:
            return response.json()

    return {"error": "Value lookup failed"}


async def search_parts_prices(make: str, model: str, year: int, parts: list) -> Optional[dict]:
    """
    Search for parts prices using web scraping.

    Args:
        make: Vehicle make (e.g., "BMW")
        model: Vehicle model (e.g., "320i")
        year: Vehicle year
        parts: List of damaged part names

    Returns:
        Parts pricing data from car_value service
    """
    if not parts:
        return None

    try:
        async with httpx.AsyncClient(timeout=60) as client:  # Longer timeout for web scraping
            response = await client.post(
                f"{CAR_VALUE_URL}/parts/search",
                json={
                    "make": make,
                    "model": model,
                    "year": year,
                    "parts": parts,
                    "include_labor": False  # Labor norms from gateway dual-query RAG instead
                }
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Parts search returned {response.status_code}: {response.text}")

    except Exception as e:
        logger.warning(f"Parts search failed: {e}")

    return None


async def validate_speed_with_physics(
    claimed_speed_kmh: float,
    skid_distance_m: float = None,
    post_impact_travel_m: float = None,
    friction_coefficient: float = 0.7
) -> Optional[dict]:
    """
    Validate claimed speed using crash physics calculations.

    Uses skid marks or post-impact travel distance to calculate
    actual speed and compare with claimed speed.
    """
    if skid_distance_m is None and post_impact_travel_m is None:
        return None

    params = {
        "claimed_speed_kmh": claimed_speed_kmh,
        "friction_coefficient": friction_coefficient
    }
    if skid_distance_m is not None:
        params["skid_distance_m"] = skid_distance_m
    if post_impact_travel_m is not None:
        params["post_impact_travel_m"] = post_impact_travel_m

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                f"{PHYSICS_URL}/validate-claimed-speed",
                params=params
            )

            if response.status_code == 200:
                return response.json()
    except Exception as e:
        logger.warning(f"Physics validation failed: {e}")

    return None


async def calculate_momentum_360(
    vehicle_a: dict,
    vehicle_b: dict,
    friction_coefficient: float = 0.7,
    grade_percent: float = 0,
    restitution_coefficient: float = 0.4,
    alpha_s_deg: float = 0
) -> Optional[dict]:
    """
    Calculate collision physics using Momentum 360 method.

    Uses the Bulgarian crash reconstruction methodology with angular analysis.
    Returns pre-impact velocities, post-impact velocities, and delta-V.
    """
    payload = {
        "vehicle_a": vehicle_a,
        "vehicle_b": vehicle_b,
        "friction_coefficient": friction_coefficient,
        "grade_percent": grade_percent,
        "restitution_coefficient": restitution_coefficient,
        "alpha_s_deg": alpha_s_deg
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                f"{PHYSICS_URL}/momentum-360",
                json=payload
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Momentum 360 returned {response.status_code}: {response.text}")
    except Exception as e:
        logger.warning(f"Momentum 360 calculation failed: {e}")

    return None


async def calculate_collision_physics(
    vehicle_a: dict,
    vehicle_b: dict,
    friction: float = 0.7,
    restitution: float = 0.4,
    collision_type: str = "head_on"
) -> dict:
    """
    Legacy collision physics using simple momentum analysis.
    Kept for backward compatibility.
    """
    payload = {
        "vehicle_a": vehicle_a,
        "vehicle_b": vehicle_b,
        "friction_coefficient": friction,
        "restitution_coefficient": restitution,
        "collision_type": collision_type
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                f"{PHYSICS_URL}/momentum-analysis",
                json=payload
            )

            if response.status_code == 200:
                return response.json()
    except Exception as e:
        logger.warning(f"Physics calculation failed: {e}")

    return None


async def store_result(filename: str, result: ProcessingResult):
    """Store processing result in database."""
    if not db_pool:
        return

    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO processed_claims (
                claim_number, filename, processing_time_ms,
                confidence_score, fault_percentage, settlement_amount_bgn,
                result_json, raw_ocr_text
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
            result.claim_number,
            filename,
            int(result.processing_time_seconds * 1000),
            result.confidence_score,
            result.fault_determination.fault_percentage if result.fault_determination else None,
            result.settlement_recommendation.amount_bgn if result.settlement_recommendation else None,
            json.dumps(result.model_dump(), default=str),
            result.raw_ocr_text
        )


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "Legal AI Document Processing Gateway",
        "version": "1.0.0",
        "endpoints": {
            "/process": "POST - Process insurance claim document",
            "/process-text": "POST - Process raw text (skip OCR)",
            "/generate-ate-report": "POST - Generate expert ATE report",
            "/health": "GET - Health check"
        },
        "supported_formats": ["PDF", "PNG", "JPG", "JPEG", "TIFF"]
    }


# ============== RAG Integration ==============

async def get_ate_context(query: str, limit: int = 5, document_filter: str = None) -> dict:
    """
    Retrieve relevant ATE knowledge from RAG service.

    Args:
        query: Query text (usually OCR text or case description)
        limit: Maximum number of chunks to retrieve
        document_filter: Optional filter for document source ("naredba_24" or "uchebnik_ate")

    Returns:
        Dict with 'context' (formatted text) and 'sources' (references)
    """
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            request_data = {
                "query": query[:2000],  # Limit query length
                "limit": limit,
                "score_threshold": 0.4
            }
            if document_filter:
                request_data["document_filter"] = document_filter

            response = await client.post(
                f"{RAG_URL}/context",
                json=request_data
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"RAG service returned {response.status_code}")

    except Exception as e:
        logger.warning(f"RAG context retrieval failed: {e}")

    return {"context": "", "sources": [], "count": 0}


async def get_dual_ate_context(
    query: str,
    damaged_parts: list[str] = None,
    naredba_limit: int = 3,
    uchebnik_limit: int = 3,
    court_limit: int = 2
) -> tuple[str, list[str]]:
    """
    Retrieve ATE knowledge from THREE sources with separate queries.

    Sources:
    - naredba_24: Legal requirements and regulations
    - uchebnik_ate: Technical methodology and formulas
    - court_expertise: Real court case examples (professional phrasing)

    Returns tuple of:
    - Formatted context with clearly labeled sections
    - List of source citations (document, section, page)
    """
    # Query for legal requirements from Naredba 24
    naredba_query = "наредба 24 изисквания автотехническа експертиза структура доклад чл."
    naredba_context = await get_ate_context(naredba_query, limit=naredba_limit, document_filter="naredba_24")

    # Query for technical methodology from Uchebnik
    uchebnik_query = f"методика скорост формула удар ПТП {query[:500]}"
    uchebnik_context = await get_ate_context(uchebnik_query, limit=uchebnik_limit, document_filter="uchebnik_ate")

    # Query for real court expertise examples (professional phrasing and structure)
    court_query = f"експертиза механизъм скорост изводи {query[:300]}"
    court_context = await get_ate_context(court_query, limit=court_limit, document_filter="court_expertise")

    # Combine with clear labels
    combined_parts = []
    all_sources = []

    if naredba_context.get("context"):
        combined_parts.append(f"═══ НАРЕДБА 24 (Правни изисквания) ═══\n{naredba_context['context']}")
        all_sources.extend(naredba_context.get("sources", []))
        logger.info(f"Retrieved {naredba_context.get('count', 0)} Naredba 24 legal chunks")

    if uchebnik_context.get("context"):
        combined_parts.append(f"═══ УЧЕБНИК АТЕ (Техническа методология) ═══\n{uchebnik_context['context']}")
        all_sources.extend(uchebnik_context.get("sources", []))
        logger.info(f"Retrieved {uchebnik_context.get('count', 0)} Uchebnik chunks")

    if court_context.get("context"):
        combined_parts.append(f"═══ СЪДЕБНА ЕКСПЕРТИЗА (Примери от реални дела) ═══\n{court_context['context']}")
        all_sources.extend(court_context.get("sources", []))
        logger.info(f"Retrieved {court_context.get('count', 0)} Court expertise chunks")

    context = "\n\n".join(combined_parts) if combined_parts else ""
    return context, all_sources


# ============== Shared Processing Logic ==============

async def process_extraction_result(
    extracted: dict,
    raw_text: str,
    filename: str = "unknown"
) -> ProcessingResult:
    """
    Shared processing logic for both /process and /process-text endpoints.

    Takes LLM extraction output and enriches it with:
    - Geocoding (lat/lon)
    - Car market values
    - Parts pricing
    - Physics analysis

    Returns a complete ProcessingResult ready for storage/response.
    """
    start_time = time.time()
    errors = []
    warnings = []

    # Step 1: Geocoding enrichment
    location_data = None
    if extracted.get("accident_location"):
        loc = extracted["accident_location"]
        if loc.get("address") or loc.get("city"):
            try:
                geocoded = await geocode_location(loc)
                location_data = LocationData(
                    address=loc.get("address"),
                    city=loc.get("city"),
                    latitude=geocoded.get("lat"),
                    longitude=geocoded.get("lon")
                )
                if geocoded.get("lat"):
                    logger.info(f"Geocoded: {geocoded['lat']}, {geocoded['lon']}")
            except Exception as e:
                logger.warning(f"Geocoding failed: {e}")
                location_data = LocationData(
                    address=loc.get("address"),
                    city=loc.get("city")
                )

    # Step 2: Enrich vehicles with market values and parts pricing
    vehicles = []
    raw_vehicles = extracted.get("vehicles", [])
    collision_details = extracted.get("collision_details", {})

    for v in raw_vehicles:
        damaged_parts = v.get("damaged_parts", [])

        vehicle = VehicleData(
            vin=v.get("vin"),
            registration=v.get("registration"),
            make=v.get("make"),
            model=v.get("model"),
            year=v.get("year"),
            mass_kg=v.get("mass_kg"),
            owner_name=v.get("owner_name"),
            insurance_company=v.get("insurance_company"),
            policy_number=v.get("policy_number"),
            damage_description=v.get("damage_description"),
            damaged_parts=damaged_parts,
            estimated_damage_bgn=v.get("estimated_damage"),
            skid_distance_m=v.get("skid_distance_m"),
            post_impact_travel_m=v.get("post_impact_travel_m"),
            claimed_speed_kmh=v.get("claimed_speed_kmh"),
            pre_impact_angle_deg=v.get("pre_impact_angle_deg"),
            post_impact_angle_deg=v.get("post_impact_angle_deg")
        )

        # Lookup market value
        if vehicle.make and vehicle.model and vehicle.year:
            try:
                value_data = await get_car_value(vehicle.make, vehicle.model, vehicle.year)
                if value_data and not value_data.get("error"):
                    vehicle.current_market_value_bgn = value_data.get("average_price_bgn")
                    vehicle.market_value_source = value_data.get("source")
                    logger.info(f"Car value: {vehicle.make} {vehicle.model} = {vehicle.current_market_value_bgn} BGN")
            except Exception as e:
                logger.warning(f"Car value lookup failed: {e}")

            # Search for parts prices if we have damaged parts
            if damaged_parts:
                try:
                    parts_data = await search_parts_prices(
                        vehicle.make, vehicle.model, vehicle.year, damaged_parts
                    )
                    if parts_data and parts_data.get("summary"):
                        summary = parts_data["summary"]
                        parts_list = []
                        for p in parts_data.get("parts", []):
                            parts_list.append(PartPricing(
                                part_name=p.get("part_name"),
                                part_name_bg=p.get("part_name_bg"),
                                best_price_bgn=p.get("best_price_bgn"),
                                best_source=p.get("best_source"),
                                price_range_min_bgn=p.get("price_range", {}).get("min_bgn") if p.get("price_range") else None,
                                price_range_max_bgn=p.get("price_range", {}).get("max_bgn") if p.get("price_range") else None,
                                labor_cost_bgn=p.get("labor_cost_bgn"),
                                total_cost_bgn=(p.get("best_price_bgn") or 0) + (p.get("labor_cost_bgn") or 0) if p.get("best_price_bgn") else None
                            ))
                        vehicle.parts_estimate = PartsEstimate(
                            parts=parts_list,
                            total_parts_cost_bgn=summary.get("total_parts_cost_bgn"),
                            total_labor_cost_bgn=summary.get("total_labor_cost_bgn"),
                            total_repair_cost_bgn=summary.get("total_repair_cost_bgn"),
                            parts_found=summary.get("parts_found", 0),
                            parts_not_found=summary.get("parts_not_found", 0),
                            source="web_search"
                        )
                        logger.info(f"Parts estimate: {summary.get('total_repair_cost_bgn')} BGN")
                except Exception as e:
                    logger.warning(f"Parts search failed: {e}")

        vehicles.append(vehicle)

    # Step 3: Process parties
    parties = [
        PartyData(
            name=p.get("name"),
            role=p.get("role"),
            vehicle_index=p.get("vehicle_index"),
            injuries=p.get("injuries"),
            statement_summary=p.get("statement_summary")
        )
        for p in extracted.get("parties", [])
    ]

    # Step 4: Fault determination
    fault_data = None
    if extracted.get("fault_determination"):
        fd = extracted["fault_determination"]
        fault_data = FaultDetermination(
            primary_fault_party=fd.get("primary_fault_party"),
            fault_percentage=fd.get("fault_percentage"),
            reasoning=fd.get("reasoning"),
            traffic_violations=fd.get("traffic_violations", [])
        )

    # Step 5: Police report
    police_data = None
    if extracted.get("police_report"):
        pr = extracted["police_report"]
        police_data = PoliceReportData(
            report_number=pr.get("report_number"),
            officer_name=pr.get("officer_name"),
            findings=pr.get("findings")
        )

    # Step 6: Settlement recommendation
    settlement_data = None
    if extracted.get("settlement_recommendation"):
        sr = extracted["settlement_recommendation"]
        components = None
        if sr.get("components"):
            components = SettlementComponents(
                vehicle_damage=sr["components"].get("vehicle_damage"),
                medical_expenses=sr["components"].get("medical_expenses"),
                lost_income=sr["components"].get("lost_income"),
                pain_and_suffering=sr["components"].get("pain_and_suffering")
            )
        settlement_data = SettlementRecommendation(
            amount_bgn=sr.get("amount_bgn"),
            components=components,
            reasoning=sr.get("reasoning")
        )

    # Step 7: Physics analysis
    physics_data = None

    if len(raw_vehicles) >= 2:
        # Two-vehicle collision - run full Momentum 360 physics analysis
        try:
            # Get friction coefficient based on road surface
            road_surface = collision_details.get("road_surface", "dry_asphalt")
            friction_map = {
                "dry_asphalt": 0.7, "wet_asphalt": 0.5, "dry_concrete": 0.75,
                "wet_concrete": 0.55, "gravel": 0.4, "snow": 0.2, "ice": 0.1
            }
            friction = friction_map.get(road_surface, 0.7)

            v_a = raw_vehicles[0]
            v_b = raw_vehicles[1]

            # Determine default angles based on collision type if not provided
            # Angles: α (alpha) = pre-impact direction, β (beta) = post-impact direction
            # 0° = East, 90° = North, 180° = West, 270° = South
            collision_type = collision_details.get("collision_type", "head_on")
            default_angles = {
                "head_on": {"a_alpha": 0, "a_beta": 180, "b_alpha": 180, "b_beta": 0},
                "rear_end": {"a_alpha": 0, "a_beta": 0, "b_alpha": 0, "b_beta": 0},
                # side_impact: lane-change sideswipe - both traveling same direction, small deflection
                "side_impact": {"a_alpha": 0, "a_beta": 45, "b_alpha": 0, "b_beta": 315},
                "angle": {"a_alpha": 0, "a_beta": 45, "b_alpha": 135, "b_beta": 180},
                # left_turn: Vehicle A going straight East (0°), Vehicle B turning left across A's path
                # At impact moment, B is perpendicular to A (coming from South = 270°)
                # After impact: A continues mostly straight (0°), B deflects (315°)
                "left_turn": {"a_alpha": 0, "a_beta": 0, "b_alpha": 270, "b_beta": 315},
                # right_turn: Vehicle A going straight, Vehicle B turning right into A's path
                "right_turn": {"a_alpha": 0, "a_beta": 10, "b_alpha": 270, "b_beta": 315},
                # perpendicular: T-bone at intersection, A going east, B going north
                "perpendicular": {"a_alpha": 0, "a_beta": 45, "b_alpha": 270, "b_beta": 315}
            }
            defaults = default_angles.get(collision_type, default_angles["head_on"])

            # Always use default angles based on collision type for consistency
            # LLM angle extraction is unreliable when document doesn't have explicit direction data
            # This gives consistent physics results across runs
            va_alpha = defaults["a_alpha"]
            va_beta = defaults["a_beta"]
            vb_alpha = defaults["b_alpha"]
            vb_beta = defaults["b_beta"]
            logger.info(f"Physics params: collision_type={collision_type}, A_angles=({va_alpha},{va_beta}), B_angles=({vb_alpha},{vb_beta})")

            physics_result = await calculate_momentum_360(
                vehicle_a={
                    "mass_kg": v_a.get("mass_kg") or 1400,
                    "post_impact_travel_m": v_a.get("post_impact_travel_m") or 5.0,
                    "alpha_deg": va_alpha,
                    "beta_deg": va_beta,
                    "final_velocity_ms": 0
                },
                vehicle_b={
                    "mass_kg": v_b.get("mass_kg") or 1400,
                    "post_impact_travel_m": v_b.get("post_impact_travel_m") or 5.0,
                    "alpha_deg": vb_alpha,
                    "beta_deg": vb_beta,
                    "final_velocity_ms": 0
                },
                friction_coefficient=friction,
                grade_percent=collision_details.get("road_grade_percent") or 0,
                restitution_coefficient=collision_details.get("restitution_coefficient") or 0.4,
                alpha_s_deg=collision_details.get("impact_angle_deg") or 0
            )

            if physics_result:
                logger.info(f"Raw physics result: pre_a={physics_result.get('vehicle_a_impact_velocity_kmh')}, post_a={physics_result.get('vehicle_a_post_impact_kmh')}")
                physics_data = PhysicsAnalysis(
                    vehicle_a_post_impact_kmh=physics_result.get("vehicle_a_post_impact_kmh"),
                    vehicle_b_post_impact_kmh=physics_result.get("vehicle_b_post_impact_kmh"),
                    vehicle_a_pre_impact_kmh=physics_result.get("vehicle_a_impact_velocity_kmh"),
                    vehicle_b_pre_impact_kmh=physics_result.get("vehicle_b_impact_velocity_kmh"),
                    delta_v_a_kmh=physics_result.get("delta_v_a_kmh"),
                    delta_v_b_kmh=physics_result.get("delta_v_b_kmh"),
                    physics_method="momentum_360",
                    physics_confidence=0.9,
                    physics_notes=physics_result.get("notes", [])
                )
                logger.info(f"Momentum 360: V1={physics_data.vehicle_a_pre_impact_kmh} km/h, V2={physics_data.vehicle_b_pre_impact_kmh} km/h")
        except Exception as e:
            logger.warning(f"Physics analysis failed: {e}")
            warnings.append(f"Physics analysis failed: {str(e)}")

    elif len(raw_vehicles) == 1 and raw_vehicles[0].get("claimed_speed_kmh"):
        # Single vehicle - validate claimed speed if we have physical evidence
        v = raw_vehicles[0]
        if v.get("skid_distance_m") or v.get("post_impact_travel_m"):
            try:
                road_surface = collision_details.get("road_surface", "dry_asphalt")
                friction_map = {
                    "dry_asphalt": 0.7, "wet_asphalt": 0.5, "gravel": 0.4, "snow": 0.2, "ice": 0.1
                }
                friction = friction_map.get(road_surface, 0.7)

                validation = await validate_speed_with_physics(
                    claimed_speed_kmh=v["claimed_speed_kmh"],
                    skid_distance_m=v.get("skid_distance_m"),
                    post_impact_travel_m=v.get("post_impact_travel_m"),
                    friction_coefficient=friction
                )

                if validation:
                    physics_data = PhysicsAnalysis(
                        claimed_speed_valid=validation.get("claimed_speed_valid"),
                        calculated_speed_kmh=validation.get("calculated_speed_kmh"),
                        speed_validation_method=validation.get("physics_method"),
                        speed_validation_explanation=validation.get("explanation"),
                        physics_confidence=validation.get("confidence")
                    )
                    logger.info(f"Speed validation: claimed={v['claimed_speed_kmh']} km/h, valid={validation.get('claimed_speed_valid')}")
            except Exception as e:
                logger.warning(f"Speed validation failed: {e}")

    processing_time = time.time() - start_time

    # Build final result
    result = ProcessingResult(
        claim_number=extracted.get("claim_number"),
        accident_date=extracted.get("accident_date"),
        accident_time=extracted.get("accident_time"),
        accident_location=location_data,
        vehicles=vehicles,
        parties=parties,
        accident_description=extracted.get("accident_description"),
        fault_determination=fault_data,
        police_report=police_data,
        settlement_recommendation=settlement_data,
        physics_analysis=physics_data,
        risk_factors=extracted.get("risk_factors", []),
        confidence_score=extracted.get("confidence_score", 0.0),
        processing_time_seconds=round(processing_time, 2),
        raw_ocr_text=raw_text,
        errors=errors,
        warnings=warnings
    )

    # Store result in database
    if db_pool:
        try:
            await store_result(filename, result)
        except Exception as e:
            logger.error(f"Failed to store result: {e}")

    return result


# ============== ATE Report Enrichment Helpers ==============

async def enrich_with_geocoding(case_data: dict) -> dict:
    """Add lat/lon coordinates to accident_location in case_data dict."""
    if not case_data.get("accident_location"):
        return case_data

    loc = case_data["accident_location"]
    if loc.get("address") or loc.get("city"):
        try:
            geocoded = await geocode_location(loc)
            if geocoded.get("lat") and geocoded.get("lon"):
                loc["latitude"] = geocoded["lat"]
                loc["longitude"] = geocoded["lon"]
                logger.info(f"Geocoded location: {geocoded['lat']}, {geocoded['lon']}")
        except Exception as e:
            logger.warning(f"Geocoding failed: {e}")

    return case_data


async def enrich_with_car_values(case_data: dict) -> dict:
    """Add market values and parts pricing to vehicles in case_data dict."""
    vehicles = case_data.get("vehicles", [])

    for v in vehicles:
        if v.get("make") and v.get("model") and v.get("year"):
            try:
                market_value = await get_car_value(v["make"], v["model"], v["year"])
                v["current_market_value_bgn"] = market_value.get("average_price_bgn")
                v["market_value_source"] = market_value.get("source")
                logger.info(f"Car value: {v['make']} {v['model']} = {v.get('current_market_value_bgn')} BGN")
            except Exception as e:
                logger.warning(f"Car value lookup failed: {e}")

            # Search for parts prices if we have damaged parts
            damaged_parts = v.get("damaged_parts", [])
            if damaged_parts:
                try:
                    parts_data = await search_parts_prices(
                        v["make"], v["model"], v["year"], damaged_parts
                    )
                    if parts_data and parts_data.get("summary"):
                        v["parts_estimate"] = parts_data["summary"]
                        v["parts_details"] = parts_data.get("parts", [])
                        logger.info(f"Parts estimate: {parts_data['summary'].get('total_repair_estimate_bgn')} BGN")
                except Exception as e:
                    logger.warning(f"Parts pricing failed: {e}")

    return case_data


async def enrich_with_physics(case_data: dict) -> dict:
    """Add physics analysis to case_data dict if sufficient data available."""
    vehicles = case_data.get("vehicles", [])
    collision_details = case_data.get("collision_details") or {}

    # Need at least 2 vehicles for Momentum 360 analysis
    if len(vehicles) >= 2:
        v1, v2 = vehicles[0], vehicles[1]
        # Check if we have the required data (mass is required, others have defaults)
        if v1.get("mass_kg") and v2.get("mass_kg"):
            try:
                # Determine angles based on collision type
                # Angles: α (alpha) = pre-impact direction, β (beta) = post-impact direction
                # 0° = East, 90° = North, 180° = West, 270° = South
                collision_type = collision_details.get("collision_type", "head_on")
                default_angles = {
                    "head_on": {"a_alpha": 0, "a_beta": 180, "b_alpha": 180, "b_beta": 0},
                    "rear_end": {"a_alpha": 0, "a_beta": 0, "b_alpha": 0, "b_beta": 0},
                    "side_impact": {"a_alpha": 0, "a_beta": 45, "b_alpha": 0, "b_beta": 315},
                    "angle": {"a_alpha": 0, "a_beta": 45, "b_alpha": 135, "b_beta": 180},
                    # left_turn: Vehicle A going straight East (0°), Vehicle B turning left across A's path
                    # At impact moment, B is perpendicular to A (coming from South = 270°)
                    "left_turn": {"a_alpha": 0, "a_beta": 0, "b_alpha": 270, "b_beta": 315},
                    "right_turn": {"a_alpha": 0, "a_beta": 10, "b_alpha": 270, "b_beta": 315},
                    "perpendicular": {"a_alpha": 0, "a_beta": 45, "b_alpha": 270, "b_beta": 315}
                }
                angles = default_angles.get(collision_type, default_angles["head_on"])
                logger.info(f"Physics: collision_type={collision_type}, A_angles=({angles['a_alpha']},{angles['a_beta']}), B_angles=({angles['b_alpha']},{angles['b_beta']})")

                # Build vehicle dicts matching VehiclePhysics schema
                vehicle_a = {
                    "mass_kg": v1["mass_kg"],
                    "post_impact_travel_m": v1.get("post_impact_travel_m") or 5.0,
                    "alpha_deg": angles["a_alpha"],
                    "beta_deg": angles["a_beta"],
                    "final_velocity_ms": 0
                }
                vehicle_b = {
                    "mass_kg": v2["mass_kg"],
                    "post_impact_travel_m": v2.get("post_impact_travel_m") or 5.0,
                    "alpha_deg": angles["b_alpha"],
                    "beta_deg": angles["b_beta"],
                    "final_velocity_ms": 0
                }
                # Get physics parameters from collision_details with safe defaults
                friction = 0.7
                road_surface = collision_details.get("road_surface")
                if road_surface:
                    friction_map = {
                        "dry_asphalt": 0.7, "wet_asphalt": 0.5,
                        "gravel": 0.4, "snow": 0.2, "ice": 0.1
                    }
                    friction = friction_map.get(road_surface, 0.7)

                physics_result = await calculate_momentum_360(
                    vehicle_a=vehicle_a,
                    vehicle_b=vehicle_b,
                    friction_coefficient=friction,
                    grade_percent=collision_details.get("road_grade_percent") or 0,
                    restitution_coefficient=collision_details.get("restitution_coefficient") or 0.4,
                    alpha_s_deg=collision_details.get("impact_angle_deg") or 0
                )
                if physics_result:
                    case_data["physics_analysis"] = physics_result
                    logger.info(f"Physics: V1={physics_result.get('vehicle_a_impact_velocity_kmh')} km/h")
            except Exception as e:
                logger.warning(f"Physics analysis failed: {e}")
    # Single vehicle - try speed validation from skid marks
    elif len(vehicles) == 1:
        v = vehicles[0]
        if v.get("skid_distance_m") and v.get("claimed_speed_kmh"):
            try:
                # Convert road surface to friction coefficient
                road_surface = case_data.get("collision_details", {}).get("road_surface", "dry_asphalt")
                friction_map = {
                    "dry_asphalt": 0.7,
                    "wet_asphalt": 0.5,
                    "gravel": 0.4,
                    "snow": 0.2,
                    "ice": 0.1
                }
                friction = friction_map.get(road_surface, 0.7)

                validation = await validate_speed_with_physics(
                    claimed_speed_kmh=v["claimed_speed_kmh"],
                    skid_distance_m=v["skid_distance_m"],
                    friction_coefficient=friction
                )
                if validation:
                    case_data["physics_analysis"] = validation
                    logger.info(f"Speed validation: {validation.get('speed_valid')}")
            except Exception as e:
                logger.warning(f"Speed validation failed: {e}")

    return case_data


# ============== Two-Stage ATE Report Generation ==============
# Stage 1: Sections 1-4 + Formulas 5.1-5.8 (physics analysis)
# Stage 2: Formulas 5.9-5.15 + Sections 6-9 (conclusions & answers)
# Each stage uses ~4k input tokens, ~10k output tokens (1/4 : 3/4 ratio)


def build_stage1_prompt(
    report_data_json: str,
    ate_knowledge: str,
    physics_values: dict,
    hourly_rate: float
) -> str:
    """Build compact prompt for Stage 1: Sections 1-4 + Formulas 5.1-5.8."""
    speed_a = physics_values.get("speed_a", "неизвестна")
    speed_b = physics_values.get("speed_b", "неизвестна")
    delta_v = physics_values.get("delta_v", "неизвестна")
    method = physics_values.get("method", "Momentum 360")

    return f"""/no_think
Генерирайте ЕТАП 1 от Автотехническа Експертиза (АТЕ).

ФОРМАТ: Прост текст с Unicode (√ × ² ³). БЕЗ LaTeX/$$!

╔══════════════════════════════════════════════════════════════════════════════╗
║ КРИТИЧНО ВАЖНО: ИЗПОЛЗВАЙТЕ ТОЧНО ТЕЗИ СТОЙНОСТИ - НЕ ГИ ПРЕИЗЧИСЛЯВАЙТЕ!  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ • МПС А преди удара: V₁ = {speed_a} km/h (от Momentum 360)                  ║
║ • МПС Б преди удара: V₂ = {speed_b} km/h (от Momentum 360)                  ║
║ • Delta-V: ΔV = {delta_v} km/h                                              ║
║ • Метод: {method}                                                            ║
║ • Коефициент триене: μ = 0.7 (сух асфалт)                                   ║
║ • g = 9.81 m/s²                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ ЗАБРАНЕНО Е да използвате други стойности за V₁ и V₂!                       ║
║ Тези стойности са изчислени от физическия двигател с правилните ъгли.      ║
╚══════════════════════════════════════════════════════════════════════════════╝

══ ДАННИ ══
{report_data_json}

══ РЕФЕРЕНЦИИ ══
{ate_knowledge}

══ СТРУКТУРА ЕТАП 1 (секции 1-4 + формули 5.1-5.8) ══

1. ЗАГЛАВНА ЧАСТ (10 реда)
   - "АВТОТЕХНИЧЕСКА ЕКСПЕРТИЗА"
   - Номер дело, дата, възложител

2. ВЪВЕДЕНИЕ (25 реда)
   - Основание за назначаване, задачи на експерта
   - Нормативна база: Наредба 24, ЗДвП
   - Методи: Momentum 360, запазване на импулса

3. ИЗСЛЕДВАНА ДОКУМЕНТАЦИЯ (20 реда)
   - Списък документи: протокол, декларации, снимки

4. ФАКТИЧЕСКА ОБСТАНОВКА (60 реда)
   - Дата, час, място на ПТП
   - Описание на пътя: широчина, настилка, знаци
   - Метеоусловия: осветление, видимост
   - За ВСЯКО МПС: марка, модел, год., рег.№, VIN, маса, направление
   - Хронология на събитията

5. ТЕХНИЧЕСКО ИЗСЛЕДВАНЕ - ЧАСТ 1 (100 реда)

   5.1 КОЕФИЦИЕНТ НА ТРИЕНЕ (μ):
   - Стойност μ = 0.7 (сух асфалт), обосновка
   - Формула: F = μ × m × g

   5.2 СКОРОСТ СЛЕД УДАРА (от следи):
   - u = √(2×μ×g×σ) - покажете изчисление стъпка по стъпка
   - Резултат за всяко МПС в m/s и km/h

   5.3 ВЕКТОРНО РАЗЛАГАНЕ:
   - Vx = V×cos(α), Vy = V×sin(α)
   - V = √(Vx² + Vy²)

   5.4 MOMENTUM 360 АНАЛИЗ:
   - X: m₁V₁cos(α₁) + m₂V₂cos(α₂) = m₁u₁cos(β₁) + m₂u₂cos(β₂)
   - Y: m₁V₁sin(α₁) + m₂V₂sin(α₂) = m₁u₁sin(β₁) + m₂u₂sin(β₂)
   - Входни данни и изчислени V₁, V₂

   5.5 МАТРИЧЕН МЕТОД (Impact Theory):
   - [a₁₁ a₁₂; a₂₁ a₂₂] × [V₁; V₂] = [b₁; b₂]
   - Коефициенти и решение

   5.6 DELTA-V АНАЛИЗ:
   - ΔV = √(V² + u² - 2Vu×cos(β-α))
   - ΔV за МПС А и Б, оценка тежест удар

   5.7 ОПАСНА ЗОНА И СПИРАЧЕН ПЪТ:
   - Sоз = V×(tr + tsp + 0.5tn) + V²/(2j)
   - tr=1.0s, tsp=0.2s, tn=0.3s, j=μg
   - Разстояние реакция + спирачен път

   5.8 ВРЕМЕ ЗА СПИРАНЕ:
   - Tу = tr + tsp + 0.5tn + V/j
   - Изчисление с реални стойности

Пишете ПОДРОБНО с числа и единици. Това е ЕТАП 1 - продължава в ЕТАП 2.
"""


def build_stage2_prompt(
    report_data_json: str,
    ate_knowledge: str,
    physics_values: dict,
    expert_questions: list[str],
    hourly_rate: float,
    stage1_summary: str
) -> str:
    """Build compact prompt for Stage 2: Formulas 5.9-5.15 + Sections 6-9."""
    speed_a = physics_values.get("speed_a", "неизвестна")
    speed_b = physics_values.get("speed_b", "неизвестна")
    delta_v = physics_values.get("delta_v", "неизвестна")

    questions_text = chr(10).join(f'{i+1}. {q}' for i, q in enumerate(expert_questions))

    return f"""/no_think
Генерирайте ЕТАП 2 от Автотехническа Експертиза (АТЕ).

ФОРМАТ: Прост текст с Unicode (√ × ² ³). БЕЗ LaTeX/$$!

══ РЕЗЮМЕ ОТ ЕТАП 1 ══
{stage1_summary}

╔══════════════════════════════════════════════════════════════════════════════╗
║ КРИТИЧНО: ИЗПОЛЗВАЙТЕ ТОЧНО ТЕЗИ СТОЙНОСТИ (от Momentum 360)               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ • V₁ = {speed_a} km/h, V₂ = {speed_b} km/h, ΔV = {delta_v} km/h             ║
║ • μ = 0.7, g = 9.81 m/s²                                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

══ КОЕФИЦИЕНТИ НА ОВЕХТЯВАНЕ (по Наредба 24) ══
Германски автомобили (BMW, VW, Audi, Mercedes, Opel, Porsche):
• 1-2 год: 15-20%    • 3-4 год: 25-32%    • 5-6 год: 38-44%
• 7-8 год: 33-38%    • 9-10 год: 28-33%   • >10 год: 20-25%
Японски/Корейски (Toyota, Honda, Hyundai, Kia):
• 1-2 год: 12-18%    • 3-4 год: 22-28%    • 5-6 год: 30-36%
Френски/Италиански (Peugeot, Renault, Fiat):
• 1-2 год: 18-25%    • 3-4 год: 30-38%    • 5-6 год: 42-50%

══ ТРУД И БОЯДИСВАНЕ (минимални стойности) ══
• Труд: минимум 2-4 часа × {hourly_rate} лв/час за основен ремонт
• Боядисване: минимум 80-150 лв за една детайл

══ ДАННИ ЗА ЩЕТИ ══
{report_data_json}

══ РЕФЕРЕНЦИИ ══
{ate_knowledge}

══ ВЪПРОСИ ЗА ОТГОВОР ══
{questions_text}

══ СТРУКТУРА ЕТАП 2 (формули 5.9-5.15 + секции 6-9) ══

5. ТЕХНИЧЕСКО ИЗСЛЕДВАНЕ - ЧАСТ 2 (продължение)

   5.9 БЕЗОПАСНА СКОРОСТ (Vбезоп):
   - Квадратно уравнение за V при което водачът би спрял
   - Сравнение с действителната скорост

   5.10 ВЪЗМОЖНОСТ ЗА ПРЕДОТВРАТЯВАНЕ:
   - Sоз vs разстояние до обекта
   - Ако Sоз > разстояние → НЕ е имал възможност
   - Времеви анализ

   5.11 АНАЛИЗ НА ЧУВСТВИТЕЛНОСТ:
   - При μ = 0.65: V = ?
   - При μ = 0.70: V = ?
   - При μ = 0.75: V = ?
   - Грешка: ±5%

   5.12 КОЕФИЦИЕНТ НА ВЪЗСТАНОВЯВАНЕ (k):
   - k = (u₂ - u₁)/(V₁ - V₂)
   - Таблица: 12-16 km/h→k=0.24, 23-25→0.16, 40-41→0.13, 55-57→0.12
   - Използвана стойност

   5.13 ЕНЕРГИЕН БАЛАНС:
   - E₁ = ½mV² (преди) за всяко МПС
   - E₂ = ½mu² (след) за всяко МПС
   - Обща енергия преди и след

   5.14 ДИСИПИРАНА ЕНЕРГИЯ:
   - ΔE = E₁ - E₂
   - Процент загуба: (ΔE/E₁)×100%
   - Енергия за деформация

   5.15 РАБОТА НА СПИРАЧНИ СИЛИ:
   - W = μ×m×g×s
   - Изчисление за всяко МПС
   - Проверка: W ≈ ½mu²

6. ОЦЕНКА НА ЩЕТИТЕ (60 реда) - по Наредба 24
   За ВСЯКО МПС задължително:
   - Списък повредени части с РЕАЛИСТИЧНИ цени (капак: 400-800 лв, бампер: 200-500 лв, фар: 150-400 лв, врата: 500-1200 лв, огледало: 80-200 лв)
   - Коефициент овехтяване: използвайте таблицата по-горе според марка и възраст!
     Пример: BMW 2019 (5 год) → 38-44% овехтяване
     Пример: VW 2017 (7 год) → 33-38% овехтяване
   - Труд: МИНИМУМ 3-6 часа × {hourly_rate} лв/час (не 0!)
   - Боядисване: МИНИМУМ 80-150 лв на детайл (не 0!)
   - ОБЩА СТОЙНОСТ РЕМОНТ (трябва да е реалистична: 1000-5000 лв за умерени щети)

7. ИЗВОДИ (35 реда)
   - Механизъм на ПТП
   - Технически причини
   - Възможност за предотвратяване
   - Виновно поведение

8. ОТГОВОРИ НА ВЪПРОСИТЕ (60 реда)
   - Всеки въпрос отделно
   - Подробен отговор с числа
   - Позоваване на изчисленията

9. ЗАКЛЮЧИТЕЛНА ЧАСТ (5 реда)
   - Дата, подпис: "Изготвил: инж. [Име]"

Пишете ПОДРОБНО. Използвайте данните от ЕТАП 1 за консистентност.
"""


class ATEReportRequest(BaseModel):
    """Request model for ATE report generation."""
    # Can provide either processed result or raw text
    processed_result: Optional[dict] = None
    raw_text: Optional[str] = None
    # Additional context
    expert_questions: Optional[list[str]] = None  # Questions for the expert to answer
    include_physics: bool = True
    include_methodology: bool = True
    # Labor cost calculation
    hourly_rate_bgn: float = 60.0  # Default: official dealer rate (60 лв/час)


class ATEReportResponse(BaseModel):
    """Response model for ATE report."""
    report_text: str
    report_sections: dict
    sources_cited: list[dict]
    processing_time_seconds: float


@app.post("/generate-ate-report", response_model=ATEReportResponse)
async def generate_ate_report(request: ATEReportRequest):
    """
    Generate a professional Автотехническа Експертиза (ATE) report.

    Only 1 report generates at a time to prevent GPU overload.
    Additional requests will wait in queue (max 2 waiting).
    """
    global ate_report_queue_size

    # Check queue size - reject if too many waiting
    if ate_report_queue_size >= 2:
        raise HTTPException(
            status_code=503,
            detail="ATE report queue full. Please try again later."
        )

    ate_report_queue_size += 1
    logger.info(f"ATE report queued. Queue size: {ate_report_queue_size}")

    try:
        async with ate_report_semaphore:
            logger.info("ATE report generation started")
            start_time = time.time()

            # Get case data
            if request.processed_result:
                case_data = request.processed_result
                raw_text = case_data.get("raw_ocr_text", "")
            elif request.raw_text:
                raw_text = request.raw_text
                try:
                    extracted = await extract_with_llm(llm_client, LLM_MODEL, raw_text)
                    case_data = extracted
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to extract data: {e}")
            else:
                raise HTTPException(status_code=400, detail="Provide either processed_result or raw_text")

            # Enrich with external services (geocoding, car values, physics)
            logger.info("Enriching case data with external services...")
            case_data = await enrich_with_geocoding(case_data)
            case_data = await enrich_with_car_values(case_data)
            case_data = await enrich_with_physics(case_data)

            # Extract damaged parts from all vehicles for labor norms query
            damaged_parts = []
            for vehicle in case_data.get("vehicles", []):
                damaged_parts.extend(vehicle.get("damaged_parts", []))
            logger.info(f"Damaged parts for labor query: {damaged_parts}")

            # ===== TWO-STAGE GENERATION =====
            # Stage 1: Sections 1-4 + Formulas 5.1-5.8 (~4k input, ~10k output)
            # Stage 2: Formulas 5.9-5.15 + Sections 6-9 (~4k input, ~10k output)
            # Total output: ~20k tokens = 6,500+ Bulgarian words

            # Get MINIMAL RAG context (1+1+1 = 3 chunks instead of 3+3+2 = 8)
            logger.info("Retrieving minimal RAG context for two-stage generation...")
            ate_knowledge, sources = await get_dual_ate_context(
                raw_text[:1000],
                damaged_parts=damaged_parts,
                naredba_limit=1,
                uchebnik_limit=1,
                court_limit=1
            )

            # Build the expert questions section
            expert_questions = request.expert_questions or [
                "Какъв е механизмът на произшествието?",
                "Каква е скоростта на движение на МПС преди удара?",
                "Кой от участниците е имал възможност да предотврати произшествието?",
                "Какви са техническите причини за произшествието?"
            ]

            # Prepare MINIMAL case data for report (only essential fields)
            report_data = {
                "claim_number": case_data.get("claim_number"),
                "accident_date": case_data.get("accident_date"),
                "accident_time": case_data.get("accident_time"),
                "accident_location": case_data.get("accident_location"),
                "accident_description": case_data.get("accident_description"),
                "fault_determination": case_data.get("fault_determination"),
            }

            # Extract only essential vehicle data
            if case_data.get("vehicles"):
                report_data["vehicles"] = []
                for v in case_data["vehicles"]:
                    vehicle_summary = {
                        "make": v.get("make"),
                        "model": v.get("model"),
                        "year": v.get("year"),
                        "registration": v.get("registration"),
                        "vin": v.get("vin"),
                        "mass_kg": v.get("mass_kg"),
                        "damage_description": v.get("damage_description"),
                        "damaged_parts": v.get("damaged_parts", []),
                        "current_market_value_bgn": v.get("current_market_value_bgn"),
                    }
                    if v.get("parts_estimate"):
                        pe = v["parts_estimate"]
                        vehicle_summary["repair_estimate"] = {
                            "total_parts_bgn": pe.get("total_parts_cost_bgn"),
                            "total_labor_bgn": pe.get("total_labor_cost_bgn"),
                            "total_repair_bgn": pe.get("total_repair_cost_bgn"),
                        }
                    report_data["vehicles"].append(vehicle_summary)

            if case_data.get("physics_analysis"):
                report_data["physics_analysis"] = case_data["physics_analysis"]

            report_data_json = json.dumps(report_data, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Report data size: {len(report_data_json)} chars")

            # Extract physics values
            physics = report_data.get("physics_analysis", {})
            physics_values = {
                "speed_a": physics.get("vehicle_a_pre_impact_kmh") or physics.get("vehicle_a_impact_velocity_kmh") or "неизвестна",
                "speed_b": physics.get("vehicle_b_pre_impact_kmh") or physics.get("vehicle_b_impact_velocity_kmh") or "неизвестна",
                "delta_v": physics.get("delta_v_a_kmh") or "неизвестна",
                "method": physics.get("physics_method") or physics.get("method") or "Momentum 360"
            }

            # ===== STAGE 1: Sections 1-4 + Formulas 5.1-5.8 =====
            logger.info("STAGE 1: Generating sections 1-4 + formulas 5.1-5.8...")
            stage1_prompt = build_stage1_prompt(
                report_data_json=report_data_json,
                ate_knowledge=ate_knowledge,
                physics_values=physics_values,
                hourly_rate=request.hourly_rate_bgn
            )

            # Target: ~4k input, ~10k output (1/4 : 3/4 ratio)
            stage1_chars = len(stage1_prompt) + 200
            stage1_input_tokens = int(stage1_chars / 1.5)
            logger.info(f"Stage 1: ~{stage1_input_tokens} input tokens, 10000 max output")

            stage1_response = await llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "Вие сте сертифициран автотехнически експерт."},
                    {"role": "user", "content": stage1_prompt}
                ],
                temperature=0.3,
                max_tokens=10000
            )

            stage1_text = stage1_response.choices[0].message.content or ""
            stage1_text = re.sub(r'<think>[\s\S]*?</think>\s*', '', stage1_text).strip()
            logger.info(f"Stage 1 complete: {len(stage1_text)} chars")

            # Extract summary from Stage 1 for context in Stage 2
            # Take last 500 chars as summary (end of technical analysis)
            stage1_summary = stage1_text[-1500:] if len(stage1_text) > 1500 else stage1_text

            # ===== STAGE 2: Formulas 5.9-5.15 + Sections 6-9 =====
            logger.info("STAGE 2: Generating formulas 5.9-5.15 + sections 6-9...")
            stage2_prompt = build_stage2_prompt(
                report_data_json=report_data_json,
                ate_knowledge=ate_knowledge,
                physics_values=physics_values,
                expert_questions=expert_questions,
                hourly_rate=request.hourly_rate_bgn,
                stage1_summary=stage1_summary
            )

            stage2_chars = len(stage2_prompt) + 200
            stage2_input_tokens = int(stage2_chars / 1.5)
            logger.info(f"Stage 2: ~{stage2_input_tokens} input tokens, 10000 max output")

            stage2_response = await llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "Вие сте сертифициран автотехнически експерт."},
                    {"role": "user", "content": stage2_prompt}
                ],
                temperature=0.3,
                max_tokens=10000
            )

            stage2_text = stage2_response.choices[0].message.content or ""
            stage2_text = re.sub(r'<think>[\s\S]*?</think>\s*', '', stage2_text).strip()
            logger.info(f"Stage 2 complete: {len(stage2_text)} chars")

            # ===== COMBINE STAGES =====
            report_text = stage1_text + "\n\n" + stage2_text
            logger.info(f"Combined report: {len(report_text)} chars (~{len(report_text.split())} words)")

            processing_time = time.time() - start_time

            return ATEReportResponse(
                report_text=report_text,
                report_sections={},  # No longer used - saves output tokens
                sources_cited=sources if sources else ["Наредба № 24/2019 г.", "Учебник АТЕ II"],
                processing_time_seconds=round(processing_time, 2)
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ATE report generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")
    finally:
        ate_report_queue_size -= 1
        logger.info(f"ATE report done. Queue size: {ate_report_queue_size}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
