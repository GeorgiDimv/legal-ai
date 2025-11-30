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
import json
import time
import logging
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
    PhysicsAnalysis
)
from extractors import extract_with_llm, validate_extraction

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


@app.on_event("startup")
async def startup():
    """Initialize clients on startup."""
    global llm_client, db_pool, redis_client

    # Initialize OpenAI client pointing to vLLM
    llm_client = AsyncOpenAI(
        base_url=f"{LLM_URL}/v1",
        api_key="not-needed"  # vLLM doesn't require API key
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
    3. Enrich with geocoding
    4. Lookup car market values
    5. Store result and return

    Supports: PDF, PNG, JPG, JPEG, TIFF

    Returns complete ProcessingResult with all extracted and enriched data.
    """
    start_time = time.time()
    errors = []
    warnings = []

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
        errors.append(f"OCR failed: {str(e)}")
        raw_text = ""

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
        warnings.extend(validation_warnings)
        logger.info(f"LLM extraction complete, confidence: {extracted.get('confidence_score', 0)}")
    except Exception as e:
        logger.error(f"LLM extraction failed: {e}")
        errors.append(f"LLM extraction failed: {str(e)}")
        extracted = {}

    # Step 3: Geocoding enrichment
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
            except Exception as e:
                logger.warning(f"Geocoding failed: {e}")
                location_data = LocationData(
                    address=loc.get("address"),
                    city=loc.get("city")
                )

    # Step 4: Car value enrichment
    vehicles = []
    for v in extracted.get("vehicles", []):
        vehicle = VehicleData(
            registration=v.get("registration"),
            make=v.get("make"),
            model=v.get("model"),
            year=v.get("year"),
            mass_kg=v.get("mass_kg"),
            owner_name=v.get("owner_name"),
            insurance_company=v.get("insurance_company"),
            policy_number=v.get("policy_number"),
            damage_description=v.get("damage_description"),
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
            except Exception as e:
                logger.warning(f"Car value lookup failed: {e}")

        vehicles.append(vehicle)

    # Build parties list
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

    # Build fault determination
    fault_data = None
    if extracted.get("fault_determination"):
        fd = extracted["fault_determination"]
        fault_data = FaultDetermination(
            primary_fault_party=fd.get("primary_fault_party"),
            fault_percentage=fd.get("fault_percentage"),
            reasoning=fd.get("reasoning"),
            traffic_violations=fd.get("traffic_violations", [])
        )

    # Build police report
    police_data = None
    if extracted.get("police_report"):
        pr = extracted["police_report"]
        police_data = PoliceReportData(
            report_number=pr.get("report_number"),
            officer_name=pr.get("officer_name"),
            findings=pr.get("findings")
        )

    # Build settlement recommendation
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

    # Step 5: Physics analysis
    physics_data = None
    raw_vehicles = extracted.get("vehicles", [])
    collision_details = extracted.get("collision_details", {})

    if len(raw_vehicles) >= 2:
        # Two-vehicle collision - run full Momentum 360 physics analysis
        try:
            # Get friction coefficient based on road surface
            road_surface = collision_details.get("road_surface", "dry_asphalt")
            friction_map = {
                "dry_asphalt": 0.7,
                "wet_asphalt": 0.5,
                "dry_concrete": 0.75,
                "wet_concrete": 0.55,
                "gravel": 0.4,
                "snow": 0.2,
                "ice": 0.1
            }
            friction = friction_map.get(road_surface, 0.7)

            # Prepare vehicle physics data for Momentum 360
            v_a = raw_vehicles[0]
            v_b = raw_vehicles[1]

            # Determine default angles based on collision type if not provided
            collision_type = collision_details.get("collision_type", "head_on")
            default_angles = {
                "head_on": {"a_alpha": 0, "a_beta": 180, "b_alpha": 180, "b_beta": 0},
                "rear_end": {"a_alpha": 0, "a_beta": 0, "b_alpha": 0, "b_beta": 0},
                "side_impact": {"a_alpha": 0, "a_beta": 90, "b_alpha": 90, "b_beta": 0},
                "angle": {"a_alpha": 0, "a_beta": 45, "b_alpha": 135, "b_beta": 180}
            }
            defaults = default_angles.get(collision_type, default_angles["head_on"])

            physics_result = await calculate_momentum_360(
                vehicle_a={
                    "mass_kg": v_a.get("mass_kg", 1400),
                    "post_impact_travel_m": v_a.get("post_impact_travel_m", 5),
                    "alpha_deg": v_a.get("pre_impact_angle_deg") or defaults["a_alpha"],
                    "beta_deg": v_a.get("post_impact_angle_deg") or defaults["a_beta"],
                    "final_velocity_ms": 0
                },
                vehicle_b={
                    "mass_kg": v_b.get("mass_kg", 1400),
                    "post_impact_travel_m": v_b.get("post_impact_travel_m", 5),
                    "alpha_deg": v_b.get("pre_impact_angle_deg") or defaults["b_alpha"],
                    "beta_deg": v_b.get("post_impact_angle_deg") or defaults["b_beta"],
                    "final_velocity_ms": 0
                },
                friction_coefficient=friction,
                grade_percent=collision_details.get("road_grade_percent", 0),
                restitution_coefficient=collision_details.get("restitution_coefficient", 0.4),
                alpha_s_deg=collision_details.get("impact_angle_deg", 0)
            )

            if physics_result:
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
                logger.info(f"Momentum 360 analysis: V1={physics_data.vehicle_a_pre_impact_kmh} km/h, V2={physics_data.vehicle_b_pre_impact_kmh} km/h")
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

    # Build result
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

    # Step 6: Store result
    if db_pool:
        try:
            await store_result(file.filename, result)
        except Exception as e:
            logger.error(f"Failed to store result: {e}")

    logger.info(f"Processing complete in {processing_time:.2f}s")
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
    errors = []
    warnings = []

    raw_text = input_data.text

    if not raw_text or len(raw_text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Text too short or empty")

    logger.info(f"Processing text input: {len(raw_text)} characters")

    # Step 1: LLM Extraction (skip OCR)
    try:
        extracted = await extract_with_llm(llm_client, LLM_MODEL, raw_text)
        extracted, validation_warnings = validate_extraction(extracted)
        warnings.extend(validation_warnings)
        logger.info(f"LLM extraction complete, confidence: {extracted.get('confidence_score', 0)}")
    except Exception as e:
        logger.error(f"LLM extraction failed: {e}")
        errors.append(f"LLM extraction failed: {str(e)}")
        extracted = {}

    # Step 2: Geocoding enrichment
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
                logger.info(f"Geocoded: {geocoded.get('lat')}, {geocoded.get('lon')}")
            except Exception as e:
                logger.warning(f"Geocoding failed: {e}")
                location_data = LocationData(
                    address=loc.get("address"),
                    city=loc.get("city")
                )

    # Step 3: Enrich vehicles with market values
    vehicles = []
    for v in extracted.get("vehicles", []):
        vehicle = VehicleData(
            registration=v.get("registration"),
            make=v.get("make"),
            model=v.get("model"),
            year=v.get("year"),
            mass_kg=v.get("mass_kg"),
            owner_name=v.get("owner_name"),
            insurance_company=v.get("insurance_company"),
            policy_number=v.get("policy_number"),
            damage_description=v.get("damage_description"),
            estimated_damage_bgn=v.get("estimated_damage"),
            skid_distance_m=v.get("skid_distance_m"),
            post_impact_travel_m=v.get("post_impact_travel_m"),
            claimed_speed_kmh=v.get("claimed_speed_kmh"),
            pre_impact_angle_deg=v.get("pre_impact_angle_deg"),
            post_impact_angle_deg=v.get("post_impact_angle_deg")
        )

        if v.get("make") and v.get("model") and v.get("year"):
            try:
                market_value = await get_car_value(v["make"], v["model"], v["year"])
                vehicle.current_market_value_bgn = market_value.get("average_price_bgn")
                vehicle.market_value_source = market_value.get("source")
            except Exception as e:
                logger.warning(f"Car value lookup failed: {e}")

        vehicles.append(vehicle)

    # Step 4: Process parties
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

    # Step 5: Fault determination
    fault_data = None
    if extracted.get("fault_determination"):
        fd = extracted["fault_determination"]
        fault_data = FaultDetermination(
            primary_fault_party=fd.get("primary_fault_party"),
            fault_percentage=fd.get("fault_percentage"),
            reasoning=fd.get("reasoning"),
            traffic_violations=fd.get("traffic_violations", [])
        )

    # Step 6: Police report
    police_data = None
    if extracted.get("police_report"):
        pr = extracted["police_report"]
        police_data = PoliceReportData(
            report_number=pr.get("report_number"),
            officer_name=pr.get("officer_name"),
            findings=pr.get("findings")
        )

    # Step 7: Settlement recommendation
    settlement_data = None
    if extracted.get("settlement_recommendation"):
        sr = extracted["settlement_recommendation"]
        components = None
        if sr.get("components"):
            components = SettlementComponents(**sr["components"])
        settlement_data = SettlementRecommendation(
            amount_bgn=sr.get("amount_bgn"),
            components=components,
            reasoning=sr.get("reasoning")
        )

    # Step 8: Physics analysis
    physics_data = None
    collision_details = extracted.get("collision_details", {})

    if len(vehicles) >= 2 and vehicles[0].mass_kg and vehicles[1].mass_kg:
        try:
            physics_result = await calculate_momentum_360(
                vehicles[0], vehicles[1], collision_details
            )
            if physics_result:
                physics_data = PhysicsAnalysis(**physics_result)
        except Exception as e:
            logger.warning(f"Physics analysis failed: {e}")
    elif len(vehicles) == 1 and vehicles[0].skid_distance_m:
        try:
            physics_result = await validate_speed_with_physics(
                vehicles[0], collision_details.get("road_surface", "dry_asphalt")
            )
            if physics_result:
                physics_data = PhysicsAnalysis(**physics_result)
        except Exception as e:
            logger.warning(f"Speed validation failed: {e}")

    processing_time = time.time() - start_time

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

    # Store result
    if db_pool:
        try:
            await store_result(input_data.filename, result)
        except Exception as e:
            logger.error(f"Failed to store result: {e}")

    logger.info(f"Text processing complete in {processing_time:.2f}s")
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


async def geocode_location(location: dict) -> dict:
    """Geocode an address using Nominatim."""
    query_parts = []
    if location.get("address"):
        query_parts.append(location["address"])
    if location.get("city"):
        query_parts.append(location["city"])

    if not query_parts:
        return {}

    query = ", ".join(query_parts) + ", Bulgaria"

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(
            f"{NOMINATIM_URL}/search",
            params={
                "q": query,
                "format": "json",
                "limit": 1,
                "countrycodes": "bg"
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


async def validate_speed_with_physics(
    claimed_speed_kmh: float,
    skid_distance_m: float = None,
    post_impact_travel_m: float = None,
    friction_coefficient: float = 0.7
) -> dict:
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
) -> dict:
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
            "/health": "GET - Health check"
        },
        "supported_formats": ["PDF", "PNG", "JPG", "JPEG", "TIFF"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
