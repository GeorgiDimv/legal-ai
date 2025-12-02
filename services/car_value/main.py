"""
Car Value Service - On-Demand Price Lookup + VIN Decoding + Parts Search
Provides current market values for Bulgarian car insurance claims

Features:
1. VIN decoding via NHTSA API (free, works for EU vehicles)
2. On-demand scraping from cars.bg, mobile.bg
3. Redis caching (24h for prices, forever for VIN)
4. Parts pricing via web search (autodoc.bg, autopower.bg, alo.bg)
5. Naredba 24 compliant estimation (depreciation, labor norms, paint)
"""

import os
import re
import json
import logging
import asyncio
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from urllib.parse import quote_plus

from fastapi import FastAPI, HTTPException
import httpx
import redis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Car Value Service",
    description="Bulgarian car market value + VIN decoder + parts pricing + Naredba 24 compliance",
    version="4.0.0"
)

# =============================================================================
# NAREDBA 24 COMPLIANCE - Vehicle Origin & Depreciation
# =============================================================================

# Vehicle origin classification (Naredba 24 чл. 12)
EASTERN_MAKES = {
    # Former Soviet bloc + South Korea (per Naredba 24)
    "lada", "vaz", "moskvitch", "gaz", "uaz", "zaz", "izh",  # Russia/USSR
    "dacia",  # Romania (old models before Super Nova)
    "skoda",  # Czech (old models before Favorit)
    "trabant", "wartburg",  # East Germany
    "polonez", "fso",  # Poland
    "oltcit", "aro",  # Romania
    "zastava", "yugo",  # Yugoslavia
    "volga",  # Russia
    "hyundai", "kia", "daewoo", "ssangyong", "genesis",  # South Korea
}

# Western makes that are actually new Skoda/Dacia (after specific models)
WESTERN_SKODA_MODELS = {"octavia", "fabia", "superb", "kodiaq", "karoq", "scala", "kamiq", "enyaq"}
WESTERN_DACIA_MODELS = {"logan", "sandero", "duster", "jogger", "spring"}  # After Super Nova

# Special coefficient adjustments (Naredba 24 notes)
SPECIAL_MAKE_ADJUSTMENTS = {
    "peugeot": {"max_years": 7, "coefficient": 0.70},  # До 7 години = 0.70
    "opel": {"all_years": True, "adjustment": -0.10},  # Намалява се с 10%
    "citroen": {"max_years": 10, "adjustment": -0.10},  # До 10 години = -10%
    "ford": {"all_years": True, "adjustment": -0.10},  # Намалява се с 10%
}

# Depreciation coefficients by vehicle age (Naredba 24 чл. 12)
# Eastern European vehicles
EASTERN_COEFFICIENTS = {
    3: 0.50,   # до 3 г. вкл.
    7: 0.35,   # от 4 до 7 г. вкл.
    14: 0.30,  # от 8 до 14 г. вкл.
    99: 0.20,  # над 15 г.
}

# Western vehicles (including new Skoda after Favorit, new Dacia after Super Nova)
WESTERN_COEFFICIENTS = {
    3: 1.00,   # до 3 г. вкл.
    7: 0.70,   # от 4 до 7 г. вкл.
    14: 0.50,  # от 8 до 14 г. вкл.
    99: 0.40,  # над 15 г.
}


def get_vehicle_origin(make: str, model: Optional[str] = None) -> str:
    """
    Determine vehicle origin for Naredba 24 coefficient selection.

    Returns: "eastern" or "western"
    """
    make_lower = make.lower().strip()
    model_lower = (model or "").lower().strip()

    # Check if it's a "new" Skoda (after Favorit = western)
    if make_lower == "skoda":
        if model_lower in WESTERN_SKODA_MODELS or not model_lower:
            # Assume new Skoda if no model or if model is known western
            return "western"
        return "eastern"

    # Check if it's a "new" Dacia (after Super Nova = western)
    if make_lower == "dacia":
        if model_lower in WESTERN_DACIA_MODELS or not model_lower:
            return "western"
        return "eastern"

    # Check against eastern makes list
    if make_lower in EASTERN_MAKES:
        return "eastern"

    return "western"


def get_depreciation_coefficient(make: str, model: Optional[str], year: int) -> Tuple[float, str]:
    """
    Calculate parts depreciation coefficient per Naredba 24 чл. 12.

    Returns: (coefficient, explanation)
    """
    current_year = datetime.now().year
    vehicle_age = current_year - year

    origin = get_vehicle_origin(make, model)
    coefficients = EASTERN_COEFFICIENTS if origin == "eastern" else WESTERN_COEFFICIENTS

    # Find applicable coefficient by age bracket
    base_coefficient = 0.40  # Default for very old vehicles
    age_bracket = "над 15 г."

    for max_age, coef in sorted(coefficients.items()):
        if vehicle_age <= max_age:
            base_coefficient = coef
            if max_age == 3:
                age_bracket = "до 3 г."
            elif max_age == 7:
                age_bracket = "4-7 г."
            elif max_age == 14:
                age_bracket = "8-14 г."
            else:
                age_bracket = "над 15 г."
            break

    # Apply special make adjustments
    make_lower = make.lower().strip()
    adjustment_note = ""

    if make_lower in SPECIAL_MAKE_ADJUSTMENTS:
        spec = SPECIAL_MAKE_ADJUSTMENTS[make_lower]

        if spec.get("coefficient") and vehicle_age <= spec.get("max_years", 0):
            # Peugeot special case: fixed coefficient for young vehicles
            base_coefficient = spec["coefficient"]
            adjustment_note = f" (Peugeot ≤{spec['max_years']}г. = {spec['coefficient']})"
        elif spec.get("adjustment"):
            if spec.get("all_years") or vehicle_age <= spec.get("max_years", 99):
                base_coefficient = max(0.20, base_coefficient + spec["adjustment"])
                adjustment_note = f" ({make} {spec['adjustment']:+.0%})"

    explanation = f"Naredba 24 чл.12: {origin} производство, {age_bracket}, коеф. {base_coefficient}{adjustment_note}"

    return base_coefficient, explanation


# =============================================================================
# NAREDBA 24 - Vehicle Classes by Length (чл. 13)
# =============================================================================

# Vehicle class by габаритна дължина (length in meters)
VEHICLE_CLASSES = {
    "A": {"max_length": 4.00, "name": "клас А", "description": "L < 4.00m"},
    "B": {"max_length": 4.60, "name": "клас B", "description": "4.00m ≤ L < 4.60m"},
    "C": {"max_length": 99.0, "name": "клас C", "description": "L ≥ 4.60m"},
    "D": {"max_length": 99.0, "name": "клас D", "description": "Джипове дълга база, фургони, пикапи"},
}

# Approximate vehicle lengths by make/model (for when we don't have exact data)
VEHICLE_LENGTH_ESTIMATES = {
    # Class A (compact/small) - L < 4.00m
    "smart": 2.7, "fiat 500": 3.6, "toyota aygo": 3.5, "peugeot 108": 3.5,
    "vw up": 3.6, "hyundai i10": 3.7, "kia picanto": 3.6, "suzuki swift": 3.9,
    "mini": 3.8, "fiat panda": 3.7, "opel corsa": 4.0,

    # Class B (medium) - 4.00m ≤ L < 4.60m
    "vw golf": 4.3, "ford focus": 4.4, "opel astra": 4.4, "peugeot 308": 4.4,
    "toyota corolla": 4.4, "honda civic": 4.5, "mazda 3": 4.5, "skoda octavia": 4.7,
    "bmw 1": 4.3, "audi a3": 4.3, "mercedes a": 4.4,

    # Class C (large) - L ≥ 4.60m
    "bmw 3": 4.7, "bmw 5": 5.0, "bmw 7": 5.3, "audi a4": 4.8, "audi a6": 4.9,
    "audi a8": 5.2, "mercedes c": 4.7, "mercedes e": 4.9, "mercedes s": 5.2,
    "vw passat": 4.8, "skoda superb": 4.9, "toyota camry": 4.9,

    # Class D (SUV/Jeep long base, vans, pickups)
    "land rover": 4.8, "jeep": 4.5, "toyota land cruiser": 4.9, "nissan patrol": 5.1,
    "vw touareg": 4.9, "bmw x5": 4.9, "audi q7": 5.1, "mercedes gle": 4.9,
    "ford ranger": 5.4, "toyota hilux": 5.3, "vw amarok": 5.2,
    "vw transporter": 4.9, "mercedes vito": 5.1, "ford transit": 5.5,
}

# Body types that are always Class D
CLASS_D_BODY_TYPES = {"suv", "jeep", "pickup", "van", "фургон", "пикап", "джип"}


def get_vehicle_class(make: str, model: Optional[str] = None,
                      body_type: Optional[str] = None,
                      length_m: Optional[float] = None) -> Tuple[str, str]:
    """
    Determine vehicle class per Naredba 24 чл. 13.

    Returns: (class_letter, explanation)
    """
    # If exact length provided
    if length_m:
        if length_m < 4.00:
            return "A", f"клас А (L={length_m}m < 4.00m)"
        elif length_m < 4.60:
            return "B", f"клас B (4.00m ≤ L={length_m}m < 4.60m)"
        else:
            return "C", f"клас C (L={length_m}m ≥ 4.60m)"

    # Check body type for Class D
    if body_type:
        body_lower = body_type.lower()
        for d_type in CLASS_D_BODY_TYPES:
            if d_type in body_lower:
                return "D", f"клас D ({body_type})"

    # Estimate from make/model
    make_lower = make.lower().strip()
    model_lower = (model or "").lower().strip()
    search_key = f"{make_lower} {model_lower}".strip()

    # Check model-specific estimates first
    for key, length in VEHICLE_LENGTH_ESTIMATES.items():
        if key in search_key or search_key in key:
            if length < 4.00:
                return "A", f"клас А (~{length}m)"
            elif length < 4.60:
                return "B", f"клас B (~{length}m)"
            else:
                return "C", f"клас C (~{length}m)"

    # Check make-only estimates
    for key, length in VEHICLE_LENGTH_ESTIMATES.items():
        if make_lower in key:
            if length < 4.00:
                return "A", f"клас А (~{length}m, {make})"
            elif length < 4.60:
                return "B", f"клас B (~{length}m, {make})"
            else:
                return "C", f"клас C (~{length}m, {make})"

    # Default to Class B (most common)
    return "B", "клас B (по подразбиране)"


# =============================================================================
# NAREDBA 24 - Structured Labor Norms (Глава трета)
# =============================================================================

# Labor norms in hours by vehicle class [A, B, C, D]
# From Naredba 24 Раздел I "Подмяна на нови детайли"
LABOR_NORMS_REPLACEMENT = {
    # Body panels
    "предна броня": [0.8, 1.0, 1.2, 1.4],
    "задна броня": [0.8, 1.0, 1.2, 1.4],
    "преден капак": [0.6, 0.7, 1.0, 1.2],
    "заден капак": [0.6, 0.7, 1.0, 1.2],
    "преден калник": [2.8, 3.0, 3.5, 4.0],  # на болтове
    "заден калник": [5.0, 6.0, 7.0, 8.0],
    "врата": [1.9, 2.0, 2.5, 3.0],  # base, add complexity
    "предна врата": [1.9, 2.0, 2.5, 3.0],
    "задна врата": [1.9, 2.0, 2.5, 3.0],
    "праг": [7.5, 8.5, 9.5, 10.0],
    "таван": [12.0, 17.0, 19.0, 21.0],
    "под": [10.0, 12.0, 14.0, 16.0],
    "предна колона": [5.0, 5.0, 6.0, 6.0],
    "средна колона": [5.0, 5.0, 6.0, 6.0],
    "задна колона": [6.0, 6.0, 7.0, 7.0],
    "преден рог": [5.0, 8.0, 9.0, 10.0],
    "заден рог": [5.0, 8.0, 9.0, 10.0],

    # Glass
    "предно стъкло": [1.5, 2.0, 2.5, 3.0],
    "задно стъкло": [1.0, 1.5, 2.0, 2.5],
    "странично стъкло": [0.5, 0.6, 0.8, 1.0],

    # Lights
    "фар": [0.5, 0.6, 0.8, 1.0],
    "стоп": [0.3, 0.4, 0.5, 0.6],
    "мигач": [0.2, 0.3, 0.3, 0.4],
    "халоген": [0.3, 0.4, 0.5, 0.6],

    # Mirrors
    "огледало": [0.4, 0.5, 0.6, 0.8],
    "странично огледало": [0.4, 0.5, 0.6, 0.8],

    # Grille/trim
    "решетка": [0.3, 0.4, 0.5, 0.6],
    "предна решетка": [0.3, 0.4, 0.5, 0.6],

    # Mechanical
    "радиатор": [1.5, 2.0, 2.5, 3.0],
    "кондензатор": [1.0, 1.5, 2.0, 2.5],
    "интеркулер": [1.0, 1.5, 2.0, 2.5],
    "амортисьор": [0.8, 1.0, 1.2, 1.5],
    "носач": [1.5, 2.0, 2.5, 3.0],
    "спирачен диск": [0.5, 0.6, 0.8, 1.0],
    "накладки": [0.4, 0.5, 0.6, 0.8],

    # Interior
    "табло": [4.0, 5.0, 6.0, 8.0],
    "волан": [0.5, 0.6, 0.8, 1.0],
    "седалка": [1.0, 1.2, 1.5, 2.0],
    "еърбег": [0.5, 0.6, 0.8, 1.0],
}

# Paint labor norms (Naredba 24 Глава четвърта)
# Hours per panel by vehicle class [A, B, C, D]
PAINT_LABOR_NORMS = {
    "предна броня": [2.5, 3.0, 3.5, 4.0],
    "задна броня": [2.5, 3.0, 3.5, 4.0],
    "преден капак": [3.0, 3.5, 4.0, 4.5],
    "заден капак": [2.5, 3.0, 3.5, 4.0],
    "преден калник": [2.5, 3.0, 3.5, 4.0],
    "заден калник": [3.0, 3.5, 4.0, 4.5],
    "врата": [3.0, 3.5, 4.0, 4.5],
    "предна врата": [3.0, 3.5, 4.0, 4.5],
    "задна врата": [3.0, 3.5, 4.0, 4.5],
    "праг": [2.0, 2.5, 3.0, 3.5],
    "таван": [5.0, 6.0, 7.0, 8.0],
    "default": [2.5, 3.0, 3.5, 4.0],  # Default for unlisted parts
}

# Paint materials cost per panel (BGN) by vehicle class [A, B, C, D]
PAINT_MATERIALS_COST = {
    "standard": [80, 100, 120, 150],  # Solid colors
    "metallic": [120, 150, 180, 220],  # Metallic/pearl
    "special": [180, 220, 280, 350],  # Special effects (candy, chameleon)
}

# Color matching (тониране) - fixed cost
COLOR_MATCHING_HOURS = 1.0

# Oven drying costs by panel count
OVEN_DRYING_HOURS = {
    3: 1.5,   # 1-3 panels
    6: 3.0,   # 4-6 panels
    99: 4.5,  # >6 panels
}


def get_labor_hours_from_table(part_name: str, vehicle_class: str) -> Optional[Tuple[float, str]]:
    """
    Get labor hours from Naredba 24 structured table.

    Returns: (hours, source) or None if not found
    """
    part_lower = part_name.lower().strip()
    class_index = {"A": 0, "B": 1, "C": 2, "D": 3}.get(vehicle_class, 1)

    # Direct match
    if part_lower in LABOR_NORMS_REPLACEMENT:
        hours = LABOR_NORMS_REPLACEMENT[part_lower][class_index]
        return hours, f"Naredba 24 таблица ({vehicle_class})"

    # Partial match
    for norm_part, hours_list in LABOR_NORMS_REPLACEMENT.items():
        if norm_part in part_lower or part_lower in norm_part:
            hours = hours_list[class_index]
            return hours, f"Naredba 24 таблица ({norm_part}, {vehicle_class})"

    # Check for compound parts (e.g., "ляв преден калник" -> "преден калник")
    for norm_part, hours_list in LABOR_NORMS_REPLACEMENT.items():
        norm_words = set(norm_part.split())
        part_words = set(part_lower.split())
        if norm_words & part_words:  # Any word match
            hours = hours_list[class_index]
            return hours, f"Naredba 24 таблица (~{norm_part}, {vehicle_class})"

    return None


def get_paint_hours_from_table(part_name: str, vehicle_class: str) -> Tuple[float, str]:
    """
    Get paint labor hours from Naredba 24 table.

    Returns: (hours, source)
    """
    part_lower = part_name.lower().strip()
    class_index = {"A": 0, "B": 1, "C": 2, "D": 3}.get(vehicle_class, 1)

    # Direct match
    if part_lower in PAINT_LABOR_NORMS:
        hours = PAINT_LABOR_NORMS[part_lower][class_index]
        return hours, f"Naredba 24 бояджийски ({vehicle_class})"

    # Partial match
    for norm_part, hours_list in PAINT_LABOR_NORMS.items():
        if norm_part in part_lower or part_lower in norm_part:
            hours = hours_list[class_index]
            return hours, f"Naredba 24 бояджийски ({norm_part}, {vehicle_class})"

    # Default
    hours = PAINT_LABOR_NORMS["default"][class_index]
    return hours, f"Naredba 24 бояджийски (default, {vehicle_class})"


def calculate_paint_cost(parts: List[str], vehicle_class: str,
                         paint_type: str = "metallic",
                         hourly_rate: float = 60.0) -> Dict:
    """
    Calculate complete paint cost per Naredba 24.

    Returns dict with labor hours, materials, oven drying, color matching
    """
    class_index = {"A": 0, "B": 1, "C": 2, "D": 3}.get(vehicle_class, 1)

    total_labor_hours = 0
    total_materials = 0
    panel_details = []

    for part in parts:
        hours, source = get_paint_hours_from_table(part, vehicle_class)
        materials = PAINT_MATERIALS_COST.get(paint_type, PAINT_MATERIALS_COST["metallic"])[class_index]

        total_labor_hours += hours
        total_materials += materials
        panel_details.append({
            "part": part,
            "labor_hours": hours,
            "materials_bgn": materials,
        })

    # Add color matching if any panels need paint
    color_matching = COLOR_MATCHING_HOURS if parts else 0

    # Add oven drying based on panel count
    oven_hours = 0
    for max_panels, hours in sorted(OVEN_DRYING_HOURS.items()):
        if len(parts) <= max_panels:
            oven_hours = hours
            break

    total_labor_hours += color_matching + oven_hours
    labor_cost = total_labor_hours * hourly_rate

    return {
        "panels": panel_details,
        "panel_count": len(parts),
        "paint_type": paint_type,
        "color_matching_hours": color_matching,
        "oven_drying_hours": oven_hours,
        "total_labor_hours": round(total_labor_hours, 1),
        "labor_cost_bgn": round(labor_cost, 2),
        "materials_cost_bgn": round(total_materials, 2),
        "total_paint_cost_bgn": round(labor_cost + total_materials, 2),
        "source": "Naredba 24 Глава четвърта",
    }

# Redis client for caching
redis_client: Optional[redis.Redis] = None

# Cache TTL
PRICE_CACHE_TTL = 86400  # 24 hours for prices
PARTS_CACHE_TTL = 86400  # 24 hours for parts
VIN_CACHE_TTL = 0  # Forever for VIN (they don't change)

# Current year
CURRENT_YEAR = datetime.now().year

# User agent for scraping
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# NHTSA API endpoint
NHTSA_API = "https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues"

# RAG service for Naredba 24 labor norms
RAG_URL = os.environ.get("RAG_URL", "http://rag:8005")
LABOR_NORM_CACHE_TTL = 604800  # 7 days for labor norms (they rarely change)

# WMI (World Manufacturer Identifier) mapping for local fallback
WMI_MAP = {
    "WBA": ("BMW", "Germany"), "WBS": ("BMW M", "Germany"), "WBY": ("BMW i", "Germany"),
    "WDB": ("Mercedes-Benz", "Germany"), "WDC": ("Mercedes-Benz", "Germany"), "WDD": ("Mercedes-Benz", "Germany"),
    "WAU": ("Audi", "Germany"), "WUA": ("Audi", "Germany"),
    "WVW": ("Volkswagen", "Germany"), "WV1": ("Volkswagen Commercial", "Germany"), "WV2": ("Volkswagen", "Germany"),
    "VF1": ("Renault", "France"), "VF3": ("Peugeot", "France"), "VF7": ("Citroen", "France"),
    "TMB": ("Skoda", "Czech Republic"),
    "SAL": ("Land Rover", "UK"), "SAJ": ("Jaguar", "UK"),
    "ZFA": ("Fiat", "Italy"), "ZAR": ("Alfa Romeo", "Italy"),
    "JTD": ("Toyota", "Japan"), "JHM": ("Honda", "Japan"), "JN1": ("Nissan", "Japan"), "JMZ": ("Mazda", "Japan"),
    "KMH": ("Hyundai", "South Korea"), "KNA": ("Kia", "South Korea"), "KNM": ("Kia", "South Korea"),
    "YV1": ("Volvo", "Sweden"), "YV4": ("Volvo", "Sweden"),
    "WF0": ("Ford", "Germany"), "W0L": ("Opel", "Germany"),
    "TRU": ("Audi", "Hungary"), "VSS": ("SEAT", "Spain"),
}

# Year code mapping (position 10 of VIN)
YEAR_CODES = {
    'A': 2010, 'B': 2011, 'C': 2012, 'D': 2013, 'E': 2014,
    'F': 2015, 'G': 2016, 'H': 2017, 'J': 2018, 'K': 2019,
    'L': 2020, 'M': 2021, 'N': 2022, 'P': 2023, 'R': 2024,
    'S': 2025, 'T': 2026, 'V': 2027, 'W': 2028, 'X': 2029,
    'Y': 2030, '1': 2031, '2': 2032, '3': 2033, '4': 2034,
    '5': 2035, '6': 2036, '7': 2037, '8': 2038, '9': 2039,
}


@app.on_event("startup")
async def startup():
    """Initialize connections on startup."""
    global redis_client

    # Connect to Redis
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    try:
        redis_client = redis.from_url(redis_url, decode_responses=True)
        redis_client.ping()
        logger.info("Connected to Redis")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        redis_client = None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    redis_ok = False
    try:
        if redis_client:
            redis_ok = redis_client.ping()
    except:
        pass

    return {
        "status": "healthy",
        "service": "car_value",
        "version": "3.3.0",
        "redis": redis_ok
    }


# =============================================================================
# VIN Decoding
# =============================================================================

@app.get("/vin/{vin}")
async def decode_vin(vin: str):
    """
    Decode a VIN (Vehicle Identification Number).

    Uses NHTSA vPIC API (free, works for EU vehicles).
    Falls back to local WMI decode if API fails.

    Args:
        vin: 17-character VIN

    Returns:
        Decoded vehicle info (make, model, year, engine, etc.)
    """
    # Normalize VIN
    vin = vin.upper().strip()

    # Validate VIN format
    if not is_valid_vin(vin):
        raise HTTPException(status_code=400, detail="Invalid VIN format. Must be 17 alphanumeric characters (no I, O, Q)")

    # Check cache first
    cache_key = f"vin:{vin}"
    if redis_client:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                logger.info(f"VIN cache hit: {vin}")
                result = json.loads(cached)
                result["from_cache"] = True
                return result
        except Exception as e:
            logger.warning(f"Redis read error: {e}")

    # Try NHTSA API
    result = await decode_vin_nhtsa(vin)

    if result and not result.get("error"):
        # Cache forever (VINs don't change)
        if redis_client:
            try:
                redis_client.set(cache_key, json.dumps(result))
            except Exception as e:
                logger.warning(f"Redis cache write error: {e}")
        return result

    # Fallback to local WMI decode
    result = decode_vin_local(vin)

    if redis_client and not result.get("error"):
        try:
            redis_client.set(cache_key, json.dumps(result))
        except:
            pass

    return result


def is_valid_vin(vin: str) -> bool:
    """Validate VIN format."""
    if len(vin) != 17:
        return False
    # VIN cannot contain I, O, Q
    if re.search(r'[IOQ]', vin):
        return False
    # Must be alphanumeric
    if not re.match(r'^[A-HJ-NPR-Z0-9]{17}$', vin):
        return False
    return True


def validate_check_digit(vin: str) -> bool:
    """Validate VIN check digit (position 9)."""
    transliteration = {
        'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8,
        'J': 1, 'K': 2, 'L': 3, 'M': 4, 'N': 5, 'P': 7, 'R': 9,
        'S': 2, 'T': 3, 'U': 4, 'V': 5, 'W': 6, 'X': 7, 'Y': 8, 'Z': 9,
    }
    weights = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]

    total = 0
    for i, char in enumerate(vin):
        if char.isdigit():
            value = int(char)
        else:
            value = transliteration.get(char, 0)
        total += value * weights[i]

    remainder = total % 11
    check_digit = 'X' if remainder == 10 else str(remainder)

    return vin[8] == check_digit


async def decode_vin_nhtsa(vin: str) -> dict:
    """Decode VIN using NHTSA vPIC API."""
    try:
        url = f"{NHTSA_API}/{vin}?format=json"

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url)

            if resp.status_code != 200:
                logger.warning(f"NHTSA API returned {resp.status_code}")
                return {"error": f"NHTSA API error: {resp.status_code}"}

            data = resp.json()
            results = data.get("Results", [{}])[0]

            # Extract relevant fields
            make = results.get("Make", "").strip()
            model = results.get("Model", "").strip()
            year = results.get("ModelYear", "").strip()

            if not make:
                return {"error": "NHTSA could not decode this VIN"}

            return {
                "vin": vin,
                "valid": True,
                "source": "nhtsa",
                "make": make,
                "model": model,
                "year": int(year) if year.isdigit() else None,
                "series": results.get("Series", "").strip() or None,
                "body_class": results.get("BodyClass", "").strip() or None,
                "drive_type": results.get("DriveType", "").strip() or None,
                "engine_cylinders": results.get("EngineCylinders", "").strip() or None,
                "engine_displacement_l": results.get("DisplacementL", "").strip() or None,
                "engine_hp": results.get("EngineHP", "").strip() or None,
                "fuel_type": results.get("FuelTypePrimary", "").strip() or None,
                "plant_city": results.get("PlantCity", "").strip() or None,
                "plant_country": results.get("PlantCountry", "").strip() or None,
                "manufacturer": results.get("Manufacturer", "").strip() or None,
                "vehicle_type": results.get("VehicleType", "").strip() or None,
                "check_digit_valid": validate_check_digit(vin),
            }

    except httpx.TimeoutException:
        logger.warning("NHTSA API timeout")
        return {"error": "NHTSA API timeout"}
    except Exception as e:
        logger.error(f"NHTSA decode error: {e}")
        return {"error": str(e)}


def decode_vin_local(vin: str) -> dict:
    """Local VIN decode using WMI mapping (fallback)."""
    wmi = vin[:3]
    year_code = vin[9]

    make_info = WMI_MAP.get(wmi)
    year = YEAR_CODES.get(year_code)

    if not make_info:
        # Try first 2 characters
        wmi2 = vin[:2]
        for key, value in WMI_MAP.items():
            if key.startswith(wmi2):
                make_info = value
                break

    if not make_info:
        return {
            "vin": vin,
            "valid": False,
            "source": "local",
            "error": "Unknown manufacturer (WMI not in database)"
        }

    make, country = make_info

    return {
        "vin": vin,
        "valid": True,
        "source": "local",
        "make": make,
        "model": None,  # Can't determine model from WMI alone
        "year": year,
        "country": country,
        "wmi": wmi,
        "vds": vin[3:9],
        "vis": vin[9:17],
        "check_digit_valid": validate_check_digit(vin),
        "note": "Limited decode - model/engine details not available"
    }


# =============================================================================
# Car Value Lookup
# =============================================================================

@app.get("/value/{make}/{model}/{year}")
async def get_car_value(make: str, model: str, year: int):
    """
    Get estimated market value for a vehicle.

    Strategy:
    1. Check Redis cache
    2. Scrape BOTH cars.bg and mobile.bg in parallel with exact year
    3. If no results, fallback to ±2 year range
    4. Merge prices for better confidence
    5. Return combined result
    """
    # Normalize inputs
    make = make.strip()
    model = model.strip()

    if year < 1990 or year > CURRENT_YEAR + 1:
        raise HTTPException(status_code=400, detail=f"Invalid year: {year}")

    cache_key = f"car:v5:{make.lower()}:{model.lower()}:{year}"

    # 1. Check cache
    if redis_client:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                logger.info(f"Cache hit for {cache_key}")
                result = json.loads(cached)
                result["from_cache"] = True
                return result
        except Exception as e:
            logger.warning(f"Redis read error: {e}")

    # 2. First try exact year
    cars_bg_task = scrape_cars_bg(make, model, year, year_range=0)
    mobile_bg_task = scrape_mobile_bg(make, model, year, year_range=0)

    results = await asyncio.gather(cars_bg_task, mobile_bg_task, return_exceptions=True)
    cars_bg_result = results[0] if not isinstance(results[0], Exception) else {"error": str(results[0])}
    mobile_bg_result = results[1] if not isinstance(results[1], Exception) else {"error": str(results[1])}
    year_range_used = 0

    # 3. If no results from exact year, try ±2 year range
    cars_has_results = not cars_bg_result.get("error") and cars_bg_result.get("sample_size", 0) > 0
    mobile_has_results = not mobile_bg_result.get("error") and mobile_bg_result.get("sample_size", 0) > 0

    if not cars_has_results and not mobile_has_results:
        logger.info(f"No results for exact year {year}, trying ±2 year range")
        cars_bg_task = scrape_cars_bg(make, model, year, year_range=2)
        mobile_bg_task = scrape_mobile_bg(make, model, year, year_range=2)

        results = await asyncio.gather(cars_bg_task, mobile_bg_task, return_exceptions=True)
        cars_bg_result = results[0] if not isinstance(results[0], Exception) else {"error": str(results[0])}
        mobile_bg_result = results[1] if not isinstance(results[1], Exception) else {"error": str(results[1])}
        year_range_used = 2

    # 3. Merge prices from both sources
    all_prices = []
    sources_used = []

    if not cars_bg_result.get("error") and cars_bg_result.get("sample_size", 0) > 0:
        # Extract individual prices from cars.bg
        min_p = cars_bg_result.get("min_price_bgn", 0)
        max_p = cars_bg_result.get("max_price_bgn", 0)
        avg_p = cars_bg_result.get("average_price_bgn", 0)
        sample = cars_bg_result.get("sample_size", 0)
        # Approximate the price distribution
        if sample > 0 and avg_p > 0:
            all_prices.extend([avg_p] * sample)  # Weight by sample size
            sources_used.append(f"cars.bg({sample})")

    if not mobile_bg_result.get("error") and mobile_bg_result.get("sample_size", 0) > 0:
        avg_p = mobile_bg_result.get("average_price_bgn", 0)
        sample = mobile_bg_result.get("sample_size", 0)
        if sample > 0 and avg_p > 0:
            all_prices.extend([avg_p] * sample)
            sources_used.append(f"mobile.bg({sample})")

    if not all_prices:
        # No data from either source
        return {
            "make": make,
            "model": model,
            "year": year,
            "average_price_bgn": None,
            "source": "none",
            "currency": "BGN",
            "note": "No market data available - manual valuation required"
        }

    # Calculate combined statistics
    total_samples = len(all_prices)
    avg_price = round(sum(all_prices) / total_samples, 2)

    # Get min/max from both sources
    all_mins = [r.get("min_price_bgn") for r in [cars_bg_result, mobile_bg_result] if r.get("min_price_bgn")]
    all_maxs = [r.get("max_price_bgn") for r in [cars_bg_result, mobile_bg_result] if r.get("max_price_bgn")]

    result = {
        "make": make,
        "model": model,
        "year": year,
        "year_range": f"{year - year_range_used}-{year + year_range_used}" if year_range_used > 0 else str(year),
        "average_price_bgn": avg_price,
        "min_price_bgn": min(all_mins) if all_mins else None,
        "max_price_bgn": max(all_maxs) if all_maxs else None,
        "sample_size": total_samples,
        "source": " + ".join(sources_used),
        "currency": "BGN",
        "confidence": min(1.0, total_samples / 10),
        "sources_detail": {
            "cars_bg": cars_bg_result if not cars_bg_result.get("error") else None,
            "mobile_bg": mobile_bg_result if not mobile_bg_result.get("error") else None
        }
    }

    # Add note if year range fallback was used
    if year_range_used > 0:
        result["note"] = f"No exact {year} listings found, using {year - year_range_used}-{year + year_range_used} range"

    cache_result(cache_key, result, PRICE_CACHE_TTL)
    return result


@app.get("/value-by-vin/{vin}")
async def get_car_value_by_vin(vin: str):
    """
    Get car value by VIN.

    1. Decode VIN to get make/model/year
    2. Look up market value
    """
    # Decode VIN first
    vin_data = await decode_vin(vin)

    if vin_data.get("error") or not vin_data.get("make"):
        return {
            "vin": vin,
            "error": "Could not decode VIN",
            "vin_data": vin_data
        }

    make = vin_data.get("make")
    model = vin_data.get("model") or ""
    year = vin_data.get("year")

    if not year:
        return {
            "vin": vin,
            "error": "Could not determine year from VIN",
            "vin_data": vin_data
        }

    # Get value
    value_data = await get_car_value(make, model, year)

    return {
        "vin": vin,
        "vehicle": vin_data,
        "value": value_data
    }


async def scrape_cars_bg(make: str, model: str, year: int, year_range: int = 0) -> dict:
    """
    Scrape cars.bg for current market prices.

    Args:
        make: Car manufacturer
        model: Car model
        year: Target year
        year_range: Search year ± this value (default 0 = exact year only)
    """
    try:
        search_url = "https://www.cars.bg/carslist.php"

        year_from = year - year_range
        year_to = year + year_range

        params = {
            "mession": "search",
            "make": make,
            "model": model,
            "yearFrom": str(year_from),
            "yearTo": str(year_to),
            "currencyId": "1",
        }

        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "bg-BG,bg;q=0.9,en;q=0.8",
            "Referer": "https://www.cars.bg/"
        }

        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            resp = await client.get(search_url, params=params, headers=headers)

            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}"}

            html = resp.text
            prices = []

            # cars.bg stores prices in HTML comments
            for match in re.findall(r'<!--\s*([\d,\.]+)\s*(BGN|EUR)\s*-->', html):
                try:
                    price_str, currency = match
                    cleaned = price_str.replace(',', '').replace('.', '')
                    price = int(cleaned)

                    if currency == "EUR":
                        price = int(price * 1.96)

                    if 1000 < price < 500000:
                        prices.append(price)
                except:
                    continue

            if not prices:
                return {"error": "No listings found"}

            # Filter outliers
            prices = filter_outliers(prices)

            if not prices:
                return {"error": "No valid prices after filtering"}

            return {
                "make": make,
                "model": model,
                "year": year,
                "average_price_bgn": round(sum(prices) / len(prices), 2),
                "min_price_bgn": min(prices),
                "max_price_bgn": max(prices),
                "median_price_bgn": prices[len(prices) // 2],
                "sample_size": len(prices),
                "source": "cars.bg",
                "currency": "BGN",
                "confidence": min(1.0, len(prices) / 10)
            }

    except httpx.TimeoutException:
        return {"error": "Timeout"}
    except Exception as e:
        logger.error(f"cars.bg error: {e}")
        return {"error": str(e)}


async def scrape_mobile_bg(make: str, model: str, year: int, year_range: int = 0) -> dict:
    """
    Scrape mobile.bg for current market prices.

    Args:
        make: Car manufacturer
        model: Car model
        year: Target year
        year_range: Search year ± this value (default 0 = exact year only)
    """
    try:
        SLUGS = {
            "volkswagen": "vw",
            "mercedes-benz": "mercedes-benz",
            "alfa romeo": "alfa-romeo",
            "land rover": "land-rover",
        }

        make_slug = SLUGS.get(make.lower(), make.lower().replace(" ", "-"))
        model_slug = model.lower().replace(" ", "-") if model else ""

        url = f"https://www.mobile.bg/obiavi/avtomobili-dzhipove/{make_slug}"
        if model_slug:
            url += f"/{model_slug}"

        year_from = year - year_range
        year_to = year + year_range
        params = {"yearFrom": str(year_from), "yearTo": str(year_to)}
        headers = {"User-Agent": USER_AGENT, "Accept-Language": "bg-BG,bg;q=0.9"}

        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            resp = await client.get(url, params=params, headers=headers)

            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}"}

            try:
                html = resp.content.decode('windows-1251')
            except:
                html = resp.text

            prices = []

            # Split HTML into listing blocks to validate year for each
            # Mobile.bg shows year as "Месец YYYY г." format
            # Skip TOP/promoted listings which ignore year filters

            # Pattern to find listing blocks with price and year
            # Look for price followed by year in the same context
            listing_pattern = re.compile(
                r'(\d{1,3}(?:[\s\xa0]?\d{3})*)\s*(?:лв|BGN|€|EUR).*?'  # Price
                r'(?:Януари|Февруари|Март|Април|Май|Юни|Юли|Август|Септември|Октомври|Ноември|Декември|\d{2}/)\s*(\d{4})\s*г\.',  # Year
                re.DOTALL | re.IGNORECASE
            )

            for match in listing_pattern.findall(html):
                try:
                    price_str, listing_year_str = match
                    listing_year = int(listing_year_str)

                    # Skip if year doesn't match our target range
                    if listing_year < year_from or listing_year > year_to:
                        logger.debug(f"Skipping mobile.bg listing: year {listing_year} not in {year_from}-{year_to}")
                        continue

                    price = int(price_str.replace(' ', '').replace('\xa0', '').replace('.', '').replace(',', ''))

                    # Convert EUR to BGN if needed (check if original had € or EUR)
                    # For now assume BGN since we're matching лв|BGN first

                    if 1000 < price < 500000:
                        prices.append(price)
                except Exception as e:
                    logger.debug(f"mobile.bg parse error: {e}")
                    continue

            # Fallback: if pattern matching found nothing, try simpler extraction
            # but only for non-TOP listings (skip first few results which are usually TOP)
            if not prices:
                # Try EUR prices with year validation
                eur_pattern = re.compile(
                    r'(\d{1,3}(?:[\s\xa0]?\d{3})*)\s*(?:€|EUR).*?'
                    r'(?:\d{2}/|\w+\s+)(\d{4})\s*г\.',
                    re.DOTALL
                )
                for match in eur_pattern.findall(html):
                    try:
                        price_str, listing_year_str = match
                        listing_year = int(listing_year_str)
                        if listing_year < year_from or listing_year > year_to:
                            continue
                        price = int(price_str.replace(' ', '').replace('\xa0', ''))
                        price_bgn = int(price * 1.96)
                        if 1000 < price_bgn < 500000:
                            prices.append(price_bgn)
                    except:
                        continue

            if not prices:
                return {"error": "No listings found matching year filter"}

            prices = filter_outliers(prices)

            if not prices:
                return {"error": "No valid prices"}

            return {
                "make": make,
                "model": model,
                "year": year,
                "average_price_bgn": round(sum(prices) / len(prices), 2),
                "min_price_bgn": min(prices),
                "max_price_bgn": max(prices),
                "median_price_bgn": prices[len(prices) // 2],
                "sample_size": len(prices),
                "source": "mobile.bg",
                "currency": "BGN",
                "confidence": min(1.0, len(prices) / 10)
            }

    except httpx.TimeoutException:
        return {"error": "Timeout"}
    except Exception as e:
        logger.error(f"mobile.bg error: {e}")
        return {"error": str(e)}


def filter_outliers(prices: List[int]) -> List[int]:
    """Filter outliers using IQR method."""
    if len(prices) <= 5:
        return sorted(prices)

    prices = sorted(prices)
    q1_idx = len(prices) // 4
    q3_idx = (3 * len(prices)) // 4
    q1, q3 = prices[q1_idx], prices[q3_idx]
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return [p for p in prices if lower <= p <= upper]


def cache_result(key: str, result: dict, ttl: int):
    """Cache result in Redis."""
    if redis_client and not result.get("error"):
        try:
            if ttl > 0:
                redis_client.setex(key, ttl, json.dumps(result))
            else:
                redis_client.set(key, json.dumps(result))
        except Exception as e:
            logger.warning(f"Redis cache error: {e}")


# =============================================================================
# Parts Search - Real-time web scraping for parts prices
# =============================================================================

# Part name translations (English -> Bulgarian)
PART_TRANSLATIONS = {
    # Body parts
    "front bumper": "предна броня",
    "rear bumper": "задна броня",
    "hood": "преден капак",
    "bonnet": "преден капак",
    "trunk": "заден капак",
    "trunk lid": "заден капак",
    "front fender": "преден калник",
    "rear fender": "заден калник",
    "front door": "предна врата",
    "rear door": "задна врата",
    "side mirror": "странично огледало",
    "wing mirror": "странично огледало",
    "windshield": "предно стъкло",
    "windscreen": "предно стъкло",
    "rear window": "задно стъкло",
    "door handle": "дръжка за врата",
    "grille": "решетка",
    "front grille": "предна решетка",
    # Lights
    "headlight": "фар",
    "headlamp": "фар",
    "taillight": "стоп",
    "tail light": "стоп",
    "rear light": "заден стоп",
    "fog light": "халоген",
    "turn signal": "мигач",
    "indicator": "мигач",
    # Mechanical
    "radiator": "радиатор",
    "condenser": "климатичен кондензатор",
    "ac condenser": "климатичен кондензатор",
    "alternator": "алтернатор",
    "starter": "стартер",
    "water pump": "водна помпа",
    "shock absorber": "амортисьор",
    "strut": "амортисьор",
    "control arm": "носач",
    "brake disc": "спирачен диск",
    "brake pad": "накладки",
    "brake caliper": "спирачен апарат",
    # Interior
    "steering wheel": "волан",
    "dashboard": "табло",
    "seat": "седалка",
    "airbag": "еърбег",
}


def translate_part_name(part_name: str) -> str:
    """Translate English part name to Bulgarian."""
    part_lower = part_name.lower().strip()

    # Direct match
    if part_lower in PART_TRANSLATIONS:
        return PART_TRANSLATIONS[part_lower]

    # Partial match
    for eng, bg in PART_TRANSLATIONS.items():
        if eng in part_lower or part_lower in eng:
            return bg

    # Return original if no translation
    return part_name


from pydantic import BaseModel

# Work type categories with default hourly rates (BGN)
# These can be overridden via API parameter
WORK_TYPE_RATES = {
    "тенекеджийски": 50,    # Bodywork (изчукване, заваряване)
    "бояджийски": 60,       # Painting
    "механичен": 45,        # Mechanical repairs
    "електрически": 55,     # Electrical work
    "стъкла": 60,           # Glass replacement
    "default": 50,          # Default rate
}

# Part to work type mapping
PART_WORK_TYPES = {
    # Bodywork parts (тенекеджийски)
    "броня": "тенекеджийски", "bumper": "тенекеджийски",
    "калник": "тенекеджийски", "fender": "тенекеджийски",
    "врата": "тенекеджийски", "door": "тенекеджийски",
    "капак": "тенекеджийски", "hood": "тенекеджийски",
    "праг": "тенекеджийски", "sill": "тенекеджийски",
    "таван": "тенекеджийски", "roof": "тенекеджийски",
    "колона": "тенекеджийски", "pillar": "тенекеджийски",
    "под": "тенекеджийски", "floor": "тенекеджийски",
    "панел": "тенекеджийски", "panel": "тенекеджийски",

    # Glass parts (стъкла)
    "стъкло": "стъкла", "glass": "стъкла",
    "windshield": "стъкла", "челно": "стъкла",

    # Electrical parts (електрически)
    "фар": "електрически", "headlight": "електрически",
    "стоп": "електрически", "taillight": "електрически",
    "огледало": "електрически", "mirror": "електрически",  # Often has electronics

    # Mechanical parts (механичен)
    "радиатор": "механичен", "radiator": "механичен",
    "амортисьор": "механичен", "shock": "механичен",
}


class PartsSearchRequest(BaseModel):
    """Request body for parts search with Naredba 24 compliance."""
    make: str
    model: str
    year: int
    parts: List[str]
    include_labor: bool = True
    include_paint: bool = True  # Include paint cost calculation
    include_depreciation: bool = True  # Apply depreciation coefficient to parts
    hourly_rate_bgn: Optional[float] = None  # Override default hourly rate
    paint_hourly_rate_bgn: Optional[float] = None  # Override paint hourly rate
    work_type_rates: Optional[Dict[str, float]] = None  # Override rates by work type
    paint_type: str = "metallic"  # standard, metallic, special
    vehicle_class: Optional[str] = None  # Override auto-detected class (A, B, C, D)
    body_type: Optional[str] = None  # SUV, sedan, etc. for class detection


async def get_labor_hours_from_rag(part_name: str) -> Optional[float]:
    """
    Query RAG service to get official labor hours from Naredba 24.

    Args:
        part_name: Part name in Bulgarian or English

    Returns:
        Labor hours from Naredba 24, or None if not found
    """
    # Check cache first
    cache_key = f"labor_norm:{part_name.lower()}"
    if redis_client:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                logger.info(f"Labor norm cache hit: {part_name}")
                return float(cached)
        except Exception as e:
            logger.warning(f"Redis read error: {e}")

    try:
        import re
        async with httpx.AsyncClient(timeout=10) as client:
            # Search for labor norm in Naredba 24
            # Try multiple query formats to find the right chunk

            # Extract base part name (e.g., "предна врата" -> "врата")
            base_parts = part_name.lower().split()
            base_name = base_parts[-1] if base_parts else part_name  # Last word is usually the part

            queries = [
                f"{part_name} технологични часове група автомобили",
                f"{part_name} смяна подмяна часове",
                f"{base_name} смяна часове 0,6 0,8 1,0",  # Use base name
                f"Подмяна детайли {part_name}",
                f"{base_name} възстановяване сложност часове",  # Restoration section
                f"{part_name}—смяна",  # Exact format from Naredba
                f"{base_name} 1 2 3 часове група",  # Table format
                # Broader queries for finding parts in tables
                f"преден {base_name} болтове часове",  # "преден калник на болтове"
                f"заден {base_name} болтове часове",
                f"{base_name} на купе часове",  # "таван на купе"
                f"подмяна нови детайли леки автомобили {base_name}",  # Full table query
                f"раздел подмяна {base_name}",
                # Catch-all: retrieve full parts table (this works for all parts)
                "раздел подмяна нови детайли леки автомобили технологични часове",
            ]

            for query in queries:
                response = await client.post(
                    f"{RAG_URL}/search",
                    json={"query": query, "limit": 5}
                )

                if response.status_code != 200:
                    continue

                data = response.json()
                results = data.get("results", [])

                if not results:
                    continue

                # Parse the text to extract hours
                # Naredba 24 format: "20. Праг 7,5 8,5 9,5 10" (number. part hours1 hours2 hours3 hours4)
                part_lower = part_name.lower()

                for result in results:
                    text = result.get("text", "")
                    text_lower = text.lower()

                    # Try multiple patterns to find the part and its hours
                    # Also try with base_name for compound parts like "предна врата"
                    search_terms = [part_lower, base_name]

                    for search_term in search_terms:
                        patterns = [
                            # Numbered format: "20. Праг 7,5 8,5 9,5 10"
                            rf'\d+\.\s*{re.escape(search_term)}[^0-9]*?([\d,.\s]+?)(?:\d+\.|[А-Яа-я]{{3,}}|$)',
                            # With dash: "Врата—смяна 0,4 0,6"
                            rf'{re.escape(search_term)}[—\-][^\d]{{0,20}}([\d,.\s]+?)(?:\d+\.|[А-Яа-я]{{3,}}|$)',
                            # Simple format: "Праг 7,5 8,5"
                            rf'{re.escape(search_term)}\s+([\d,.\s]+?)(?:\d+\.|[А-Яа-я]{{3,}}|$)',
                            # With complexity: "119. Врата 1 1,9 2,0"
                            rf'\d+\.\s*{re.escape(search_term)}\s+\d\s+([\d,.\s]+)',
                            # Compound name: "5. Преден калник на болтове 2,8 3 3,5 4"
                            rf'\d+\.\s*[а-яА-Я]+\s+{re.escape(search_term)}[^0-9]*?([\d,.\s]+?)(?:\d+\.|[А-Яа-я]{{3,}}|$)',
                            # With "на" suffix: "18. Таван на купе 12 17 19 21"
                            rf'\d+\.\s*{re.escape(search_term)}\s+на\s+[а-яА-Я]+\s+([\d,.\s]+?)(?:\d+\.|[А-Яа-я]{{3,}}|$)',
                            # Any word before part: "Преден калник" or "Задна броня"
                            rf'\d+\.\s*(?:[а-яА-Я]+\s+)?{re.escape(search_term)}(?:\s+(?:на\s+)?[а-яА-Я]+)?\s+([\d,.\s]+?)(?:\d+\.|[А-Яа-я]{{3,}}|$)',
                        ]

                        for pattern in patterns:
                            match = re.search(pattern, text.lower(), re.IGNORECASE)
                            if match:
                                numbers_str = match.group(1)
                                # Extract all numbers from the matched string
                                numbers = re.findall(r'(\d+[,.]?\d*)', numbers_str)
                                if numbers:
                                    # Take average of vehicle classes (usually 4 values)
                                    valid_hours = []
                                    for num_str in numbers[:4]:  # Max 4 vehicle classes
                                        try:
                                            h = float(num_str.replace(",", "."))
                                            if 0.1 <= h <= 50:  # Sanity check
                                                valid_hours.append(h)
                                        except:
                                            continue

                                    if valid_hours:
                                        # Use average across vehicle classes
                                        avg_hours = round(sum(valid_hours) / len(valid_hours), 1)
                                        # Cache the result
                                        if redis_client:
                                            try:
                                                redis_client.setex(cache_key, LABOR_NORM_CACHE_TTL, str(avg_hours))
                                            except Exception as e:
                                                logger.warning(f"Redis write error: {e}")
                                        logger.info(f"Labor norm from RAG: {part_name} = {avg_hours}h (from {valid_hours})")
                                        return avg_hours

    except Exception as e:
        logger.warning(f"RAG labor norm lookup failed for {part_name}: {e}")

    return None


def get_work_type_for_part(part_name: str) -> str:
    """Determine work type category for a part."""
    part_lower = part_name.lower()
    for keyword, work_type in PART_WORK_TYPES.items():
        if keyword in part_lower:
            return work_type
    return "default"


def get_hourly_rate(part_name: str, hourly_rate_override: Optional[float],
                    work_type_rates_override: Optional[Dict[str, float]]) -> float:
    """
    Get hourly rate for a part based on work type.

    Priority:
    1. Global hourly_rate_override (if provided)
    2. Work type specific rate from override dict
    3. Default work type rate
    """
    if hourly_rate_override:
        return hourly_rate_override

    work_type = get_work_type_for_part(part_name)

    if work_type_rates_override and work_type in work_type_rates_override:
        return work_type_rates_override[work_type]

    return WORK_TYPE_RATES.get(work_type, WORK_TYPE_RATES["default"])


@app.post("/parts/search")
async def search_parts_prices(request: PartsSearchRequest):
    """
    Search for parts prices with full Naredba 24 compliance.

    Features:
    - Web scraping for real-time prices (bazar.bg, alochasti.bg, autoprofi.bg)
    - Depreciation coefficients by vehicle age and origin (чл. 12)
    - Vehicle class detection for labor norms (чл. 13)
    - Structured labor norms from Naredba 24 tables (Глава трета)
    - Paint cost calculation with materials (Глава четвърта)

    Args:
        request: PartsSearchRequest with:
        - make, model, year, parts list
        - include_depreciation: Apply age/origin coefficient to parts
        - include_labor: Calculate labor from Naredba 24 tables
        - include_paint: Calculate paint costs
        - paint_type: standard, metallic, special
        - vehicle_class: Override auto-detected A/B/C/D
        - hourly_rate_bgn: Override labor rate

    Returns:
        Complete repair estimate with Naredba 24 compliance
    """
    make = request.make
    model = request.model
    year = request.year
    parts = request.parts
    include_labor = request.include_labor
    include_paint = request.include_paint
    include_depreciation = request.include_depreciation
    hourly_rate_override = request.hourly_rate_bgn
    paint_hourly_rate = request.paint_hourly_rate_bgn or 60.0
    work_type_rates_override = request.work_type_rates
    paint_type = request.paint_type
    body_type = request.body_type

    # Determine vehicle class (Naredba 24 чл. 13)
    if request.vehicle_class:
        vehicle_class = request.vehicle_class.upper()
        vehicle_class_explanation = f"клас {vehicle_class} (ръчно зададен)"
    else:
        vehicle_class, vehicle_class_explanation = get_vehicle_class(make, model, body_type)

    # Calculate depreciation coefficient (Naredba 24 чл. 12)
    depreciation_coefficient, depreciation_explanation = get_depreciation_coefficient(make, model, year)

    # Get vehicle origin
    vehicle_origin = get_vehicle_origin(make, model)

    results = []
    total_parts_cost_raw = 0  # Before depreciation
    total_parts_cost = 0      # After depreciation
    total_labor_cost = 0
    paintable_parts = []

    for part_name in parts:
        # Check cache first (include depreciation in cache key)
        cache_key = f"part:v2:{make.lower()}:{model.lower()}:{year}:{part_name.lower()}:{vehicle_class}"

        if redis_client:
            try:
                cached = redis_client.get(cache_key)
                if cached:
                    logger.info(f"Parts cache hit: {cache_key}")
                    part_result = json.loads(cached)
                    part_result["from_cache"] = True
                    results.append(part_result)
                    if part_result.get("price_after_depreciation_bgn"):
                        total_parts_cost += part_result["price_after_depreciation_bgn"]
                    elif part_result.get("best_price_bgn"):
                        total_parts_cost += part_result["best_price_bgn"]
                    if part_result.get("labor_cost_bgn"):
                        total_labor_cost += part_result["labor_cost_bgn"]
                    # Check if paintable
                    if is_paintable_part(part_name):
                        paintable_parts.append(part_name)
                    continue
            except Exception as e:
                logger.warning(f"Redis read error: {e}")

        # Translate to Bulgarian for better search results
        part_bg = translate_part_name(part_name)

        # Search multiple sources in parallel
        search_results = await search_part_prices(make, model, year, part_name, part_bg)

        # Calculate labor cost from Naredba 24 structured tables (faster than RAG)
        labor_cost = 0
        labor_hours = None
        labor_source = None
        hourly_rate = None
        work_type = None

        if include_labor:
            # First try structured table (faster, more reliable)
            table_result = get_labor_hours_from_table(part_bg, vehicle_class)

            if table_result:
                labor_hours, labor_source = table_result
            else:
                # Fallback to RAG for parts not in table
                labor_hours = await get_labor_hours_from_rag(part_bg)
                if labor_hours is None:
                    labor_hours = await get_labor_hours_from_rag(part_name)
                if labor_hours:
                    labor_source = "naredba_24_rag"
                else:
                    logger.warning(f"Labor hours not found for: {part_name} ({part_bg})")

            # Get hourly rate based on work type (only if hours found)
            if labor_hours is not None:
                work_type = get_work_type_for_part(part_bg or part_name)
                hourly_rate = get_hourly_rate(part_name, hourly_rate_override, work_type_rates_override)
                labor_cost = round(labor_hours * hourly_rate, 2)

        # Find best price from search results
        all_prices = []
        for source, data in search_results.items():
            if data.get("prices"):
                all_prices.extend([(p, source) for p in data["prices"]])

        best_price_raw = None
        best_price_depreciated = None
        best_source = None
        price_range = None

        if all_prices:
            all_prices.sort(key=lambda x: x[0])
            best_price_raw = all_prices[0][0]
            best_source = all_prices[0][1]
            price_range = {
                "min_bgn": all_prices[0][0],
                "max_bgn": all_prices[-1][0],
                "avg_bgn": round(sum(p[0] for p in all_prices) / len(all_prices), 2)
            }

            # Apply depreciation coefficient if enabled
            if include_depreciation:
                best_price_depreciated = round(best_price_raw * depreciation_coefficient, 2)
                total_parts_cost_raw += best_price_raw
                total_parts_cost += best_price_depreciated
            else:
                total_parts_cost_raw += best_price_raw
                total_parts_cost += best_price_raw

        if labor_cost:
            total_labor_cost += labor_cost

        # Check if this part needs painting
        if is_paintable_part(part_name):
            paintable_parts.append(part_bg or part_name)

        part_result = {
            "part_name": part_name,
            "part_name_bg": part_bg,
            "search_results": search_results,
            "best_price_bgn": best_price_raw,
            "best_source": best_source,
            "price_range": price_range,
            # Depreciation
            "depreciation_coefficient": depreciation_coefficient if include_depreciation else None,
            "price_after_depreciation_bgn": best_price_depreciated if include_depreciation else None,
            # Labor
            "labor_hours": labor_hours if include_labor else None,
            "labor_hours_source": labor_source if include_labor else None,
            "hourly_rate_bgn": hourly_rate if include_labor else None,
            "work_type": work_type if include_labor else None,
            "labor_cost_bgn": labor_cost if include_labor else None,
            "vehicle_class": vehicle_class if include_labor else None,
        }

        # Cache result
        if redis_client and part_result.get("best_price_bgn"):
            try:
                redis_client.setex(cache_key, PARTS_CACHE_TTL, json.dumps(part_result))
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")

        results.append(part_result)

    # Calculate paint costs if enabled
    paint_estimate = None
    total_paint_cost = 0

    if include_paint and paintable_parts:
        paint_estimate = calculate_paint_cost(
            paintable_parts,
            vehicle_class,
            paint_type,
            paint_hourly_rate
        )
        total_paint_cost = paint_estimate["total_paint_cost_bgn"]

    # Build response with full Naredba 24 compliance data
    response = {
        "make": make,
        "model": model,
        "year": year,
        "parts": results,
        # Naredba 24 compliance data
        "naredba_24": {
            "vehicle_origin": vehicle_origin,
            "vehicle_class": vehicle_class,
            "vehicle_class_explanation": vehicle_class_explanation,
            "depreciation_coefficient": depreciation_coefficient if include_depreciation else None,
            "depreciation_explanation": depreciation_explanation if include_depreciation else None,
        },
        # Paint estimate (if enabled)
        "paint_estimate": paint_estimate,
        # Summary
        "summary": {
            "total_parts_cost_raw_bgn": round(total_parts_cost_raw, 2),
            "total_parts_cost_bgn": round(total_parts_cost, 2),  # After depreciation
            "depreciation_savings_bgn": round(total_parts_cost_raw - total_parts_cost, 2) if include_depreciation else 0,
            "total_labor_cost_bgn": round(total_labor_cost, 2),
            "total_paint_cost_bgn": round(total_paint_cost, 2) if include_paint else 0,
            "total_repair_cost_bgn": round(total_parts_cost + total_labor_cost + total_paint_cost, 2),
            "parts_found": sum(1 for r in results if r.get("best_price_bgn")),
            "parts_not_found": sum(1 for r in results if not r.get("best_price_bgn")),
        },
        "currency": "BGN",
        "source": "Naredba 24 (чл. 12, 13, Глава III-IV)",
        "note": "Prices include Naredba 24 depreciation and labor norms. Paint calculated separately."
    }

    return response


def is_paintable_part(part_name: str) -> bool:
    """Check if a part typically needs painting after replacement."""
    paintable_keywords = {
        "броня", "bumper", "калник", "fender", "капак", "hood", "bonnet",
        "врата", "door", "праг", "sill", "таван", "roof", "панел", "panel",
        "колона", "pillar", "рог", "под", "floor"
    }
    part_lower = part_name.lower()
    return any(kw in part_lower for kw in paintable_keywords)


async def search_part_prices(make: str, model: str, year: int, part_en: str, part_bg: str) -> Dict:
    """Search multiple Bulgarian auto parts sites for prices."""
    results = {}

    # Run searches in parallel on proper auto parts websites
    tasks = [
        search_bazar_bg(make, model, year, part_bg),
        search_alochasti_bg(make, model, year, part_bg),
        search_autoprofi_bg(make, model, year, part_bg),
    ]

    search_results = await asyncio.gather(*tasks, return_exceptions=True)

    sources = ["bazar.bg", "alochasti.bg", "autoprofi.bg"]
    for source, result in zip(sources, search_results):
        if isinstance(result, Exception):
            logger.error(f"Search error for {source}: {result}")
            results[source] = {"error": str(result), "prices": []}
        else:
            results[source] = result

    return results


async def search_bazar_bg(make: str, model: str, year: int, part_bg: str) -> Dict:
    """
    Search bazar.bg for auto parts (new and used).

    Bazar.bg is a Bulgarian classifieds site with auto parts section.
    Price format: "200 лв" with optional EUR equivalent.
    """
    try:
        # Bazar.bg search URL - search in auto parts category
        query = f"{part_bg} {make}"
        search_url = f"https://bazar.bg/obiavi?q={quote_plus(query)}"

        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "bg-BG,bg;q=0.9,en;q=0.8",
        }

        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(search_url, headers=headers)

            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}", "prices": []}

            html = resp.text
            prices = []

            # Bazar.bg price format: "200 лв" or "1 200 лв"
            # Also shows EUR: "102.26 €"
            for match in re.findall(r'(\d{1,3}(?:[\s,]?\d{3})*)\s*лв', html):
                try:
                    price = int(match.replace(' ', '').replace(',', ''))
                    if 20 < price < 50000:  # Filter reasonable part prices
                        prices.append(price)
                except:
                    continue

            # Deduplicate and sort
            prices = sorted(list(set(prices)))

            return {
                "prices": prices[:15],  # Top 15 prices
                "count": len(prices),
                "url": search_url,
                "type": "classifieds"
            }

    except Exception as e:
        logger.error(f"bazar.bg error: {e}")
        return {"error": str(e), "prices": []}


async def search_alochasti_bg(make: str, model: str, year: int, part_bg: str) -> Dict:
    """
    Search alochasti.bg for auto parts.

    Alochasti.bg is a dedicated Bulgarian auto parts store.
    Price format: "180.00 лв." with EUR equivalent "€ 92.03"
    """
    try:
        # Alochasti.bg search URL
        query = f"{part_bg} {make}"
        search_url = f"https://alochasti.bg/search?search={quote_plus(query)}"

        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "bg-BG,bg;q=0.9,en;q=0.8",
        }

        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(search_url, headers=headers)

            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}", "prices": []}

            html = resp.text
            prices = []

            # Alochasti.bg price format: "180.00 лв." or "1 234.56 лв."
            for match in re.findall(r'(\d{1,3}(?:[\s,]?\d{3})*(?:\.\d{2})?)\s*лв\.?', html):
                try:
                    # Remove spaces and convert decimal
                    clean = match.replace(' ', '').replace(',', '')
                    price = int(float(clean))
                    if 20 < price < 50000:
                        prices.append(price)
                except:
                    continue

            # Also check EUR prices: "€ 92.03"
            for match in re.findall(r'€\s*(\d{1,3}(?:[\s,]?\d{3})*(?:\.\d{2})?)', html):
                try:
                    clean = match.replace(' ', '').replace(',', '')
                    price_eur = float(clean)
                    price_bgn = int(price_eur * 1.96)
                    if 20 < price_bgn < 50000:
                        prices.append(price_bgn)
                except:
                    continue

            prices = sorted(list(set(prices)))

            return {
                "prices": prices[:15],
                "count": len(prices),
                "url": search_url,
                "type": "store"
            }

    except Exception as e:
        logger.error(f"alochasti.bg error: {e}")
        return {"error": str(e), "prices": []}


async def search_autoprofi_bg(make: str, model: str, year: int, part_bg: str) -> Dict:
    """
    Search autoprofi.bg for auto parts.

    AutoProfi.bg is a Sofia-based auto parts store with OEM and aftermarket parts.
    """
    try:
        # AutoProfi.bg search URL - uses /products/search path
        query = f"{part_bg} {make}"
        search_url = f"https://autoprofi.bg/products/search?search={quote_plus(query)}"

        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "bg-BG,bg;q=0.9,en;q=0.8",
        }

        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(search_url, headers=headers)

            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}", "prices": []}

            html = resp.text
            prices = []

            # AutoProfi price format: various BGN formats
            for match in re.findall(r'(\d{1,3}(?:[\s,.]?\d{3})*(?:[.,]\d{2})?)\s*(?:лв|BGN|лева)', html, re.IGNORECASE):
                try:
                    # Normalize: remove spaces, handle both . and , as decimal
                    clean = match.replace(' ', '')
                    # If has comma as thousands separator
                    if ',' in clean and '.' in clean:
                        clean = clean.replace(',', '')
                    elif ',' in clean:
                        clean = clean.replace(',', '.')
                    price = int(float(clean))
                    if 20 < price < 50000:
                        prices.append(price)
                except:
                    continue

            prices = sorted(list(set(prices)))

            return {
                "prices": prices[:15],
                "count": len(prices),
                "url": search_url,
                "type": "store"
            }

    except Exception as e:
        logger.error(f"autoprofi.bg error: {e}")
        return {"error": str(e), "prices": []}


@app.get("/naredba24/coefficient/{make}/{year}")
async def get_naredba24_coefficient(make: str, year: int, model: Optional[str] = None):
    """
    Get Naredba 24 depreciation coefficient for a vehicle.

    Returns coefficient based on:
    - Vehicle origin (Eastern vs Western per чл. 12)
    - Vehicle age (0-3, 4-7, 8-14, 15+ years)
    - Special adjustments (Peugeot, Opel, Citroen, Ford)
    """
    coefficient, explanation = get_depreciation_coefficient(make, model, year)
    origin = get_vehicle_origin(make, model)
    vehicle_class, class_explanation = get_vehicle_class(make, model)

    current_year = datetime.now().year
    vehicle_age = current_year - year

    return {
        "make": make,
        "model": model,
        "year": year,
        "vehicle_age": vehicle_age,
        "vehicle_origin": origin,
        "vehicle_class": vehicle_class,
        "vehicle_class_explanation": class_explanation,
        "depreciation_coefficient": coefficient,
        "explanation": explanation,
        "source": "Naredba 24 чл. 12",
        "example": {
            "new_part_price_bgn": 1000,
            "depreciated_price_bgn": round(1000 * coefficient, 2)
        }
    }


@app.get("/naredba24/labor/{part_name}")
async def get_naredba24_labor(part_name: str, vehicle_class: str = "B"):
    """
    Get Naredba 24 labor norm for a part.

    Returns labor hours from structured tables (Глава трета).
    """
    part_bg = translate_part_name(part_name)

    # Try structured table first
    table_result = get_labor_hours_from_table(part_bg, vehicle_class.upper())

    if table_result:
        hours, source = table_result
        return {
            "part_name": part_name,
            "part_name_bg": part_bg,
            "vehicle_class": vehicle_class.upper(),
            "labor_hours": hours,
            "source": source,
            "all_classes": {
                "A": LABOR_NORMS_REPLACEMENT.get(part_bg, [None])[0],
                "B": LABOR_NORMS_REPLACEMENT.get(part_bg, [None, None])[1] if len(LABOR_NORMS_REPLACEMENT.get(part_bg, [])) > 1 else None,
                "C": LABOR_NORMS_REPLACEMENT.get(part_bg, [None, None, None])[2] if len(LABOR_NORMS_REPLACEMENT.get(part_bg, [])) > 2 else None,
                "D": LABOR_NORMS_REPLACEMENT.get(part_bg, [None, None, None, None])[3] if len(LABOR_NORMS_REPLACEMENT.get(part_bg, [])) > 3 else None,
            }
        }

    return {
        "part_name": part_name,
        "part_name_bg": part_bg,
        "vehicle_class": vehicle_class.upper(),
        "labor_hours": None,
        "source": None,
        "error": "Part not found in Naredba 24 tables",
        "available_parts": list(LABOR_NORMS_REPLACEMENT.keys())[:20]  # Show first 20 available
    }


@app.get("/naredba24/paint")
async def get_naredba24_paint(parts: str, vehicle_class: str = "B",
                               paint_type: str = "metallic",
                               hourly_rate: float = 60.0):
    """
    Calculate paint costs per Naredba 24 (Глава четвърта).

    Args:
        parts: Comma-separated list of parts (e.g., "предна броня,преден калник")
        vehicle_class: A, B, C, or D
        paint_type: standard, metallic, or special
        hourly_rate: Hourly rate for paint work (default 60 BGN)
    """
    parts_list = [p.strip() for p in parts.split(",") if p.strip()]

    if not parts_list:
        raise HTTPException(status_code=400, detail="No parts provided")

    result = calculate_paint_cost(parts_list, vehicle_class.upper(), paint_type, hourly_rate)
    return result


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "Car Value Service",
        "version": "4.0.0",
        "description": "Bulgarian car market value + VIN decoder + parts pricing + Naredba 24 compliance",
        "endpoints": {
            "/vin/{vin}": "Decode VIN to get vehicle info",
            "/value/{make}/{model}/{year}": "Get car market value",
            "/value-by-vin/{vin}": "Get car value by VIN",
            "/parts/search": "Search for parts prices with Naredba 24 compliance (POST)",
            "/naredba24/coefficient/{make}/{year}": "Get depreciation coefficient (чл. 12)",
            "/naredba24/labor/{part_name}": "Get labor hours from table (Глава III)",
            "/naredba24/paint": "Calculate paint costs (Глава IV)",
        },
        "naredba_24_features": {
            "depreciation": "Coefficients by vehicle age and origin (Eastern/Western)",
            "vehicle_classes": "A (<4m), B (4-4.6m), C (>4.6m), D (SUV/Van)",
            "labor_norms": "Structured hours per part per class",
            "paint_calculation": "Labor + materials + color matching + oven drying",
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
