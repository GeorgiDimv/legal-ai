"""
Car Value Service - On-Demand Price Lookup + VIN Decoding + Parts Search
Provides current market values for Bulgarian car insurance claims

Features:
1. VIN decoding via NHTSA API (free, works for EU vehicles)
2. On-demand scraping from cars.bg, mobile.bg
3. Redis caching (24h for prices, forever for VIN)
4. Parts pricing via web search (autodoc.bg, autopower.bg, alo.bg)
"""

import os
import re
import json
import logging
import asyncio
from typing import Optional, List, Dict
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
    description="Bulgarian car market value + VIN decoder + parts pricing",
    version="3.2.0"
)

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
        "version": "3.2.0",
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
    2. Try live scraping (cars.bg)
    3. Try mobile.bg as backup
    4. Return vehicle info only (no price) if scraping fails
    """
    # Normalize inputs
    make = make.strip()
    model = model.strip()

    if year < 1990 or year > CURRENT_YEAR + 1:
        raise HTTPException(status_code=400, detail=f"Invalid year: {year}")

    cache_key = f"car:v3:{make.lower()}:{model.lower()}:{year}"

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

    # 2. Try cars.bg
    result = await scrape_cars_bg(make, model, year)

    if not result.get("error") and result.get("sample_size", 0) > 0:
        cache_result(cache_key, result, PRICE_CACHE_TTL)
        return result

    # 3. Try mobile.bg as backup
    result = await scrape_mobile_bg(make, model, year)

    if not result.get("error") and result.get("sample_size", 0) > 0:
        cache_result(cache_key, result, PRICE_CACHE_TTL)
        return result

    # 4. Return vehicle info only (no price available)
    return {
        "make": make,
        "model": model,
        "year": year,
        "average_price_bgn": None,
        "source": "none",
        "currency": "BGN",
        "note": "No market data available - manual valuation required"
    }


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


async def scrape_cars_bg(make: str, model: str, year: int) -> dict:
    """Scrape cars.bg for current market prices."""
    try:
        search_url = "https://www.cars.bg/carslist.php"

        params = {
            "mession": "search",
            "make": make,
            "model": model,
            "yearFrom": str(year),
            "yearTo": str(year),
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


async def scrape_mobile_bg(make: str, model: str, year: int) -> dict:
    """Scrape mobile.bg for current market prices."""
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

        params = {"yearFrom": str(year), "yearTo": str(year)}
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

            # BGN prices
            for match in re.findall(r'(\d{1,3}(?:\s?\d{3})*)\s*(?:лв|BGN)', html):
                try:
                    price = int(match.replace(' ', '').replace('\xa0', ''))
                    if 1000 < price < 500000:
                        prices.append(price)
                except:
                    continue

            # EUR prices
            for match in re.findall(r'(\d{1,3}(?:\s?\d{3})*)\s*EUR', html):
                try:
                    price = int(match.replace(' ', '').replace('\xa0', ''))
                    price_bgn = int(price * 1.96)
                    if 1000 < price_bgn < 500000:
                        prices.append(price_bgn)
                except:
                    continue

            if not prices:
                return {"error": "No listings found"}

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

class PartsSearchRequest(BaseModel):
    """Request body for parts search."""
    make: str
    model: str
    year: int
    parts: List[str]
    include_labor: bool = True


@app.post("/parts/search")
async def search_parts_prices(request: PartsSearchRequest):
    """
    Search for parts prices across Bulgarian auto parts websites.

    Uses web scraping to find real-time prices from:
    - autopower.bg (OEM and aftermarket)
    - alo.bg (used parts marketplace)
    - mobile.bg parts section

    Args:
        request: PartsSearchRequest with make, model, year, parts list

    Returns:
        Parts breakdown with prices and sources
    """
    make = request.make
    model = request.model
    year = request.year
    parts = request.parts
    include_labor = request.include_labor

    results = []
    total_parts_cost = 0
    total_labor_cost = 0

    # Labor rates (BGN per hour) - varies by part complexity
    LABOR_RATES = {
        "bumper": {"hours": 2.5, "rate": 50},
        "броня": {"hours": 2.5, "rate": 50},
        "fender": {"hours": 2.0, "rate": 50},
        "калник": {"hours": 2.0, "rate": 50},
        "door": {"hours": 2.5, "rate": 50},
        "врата": {"hours": 2.5, "rate": 50},
        "hood": {"hours": 1.5, "rate": 50},
        "капак": {"hours": 1.5, "rate": 50},
        "headlight": {"hours": 1.0, "rate": 50},
        "фар": {"hours": 1.0, "rate": 50},
        "taillight": {"hours": 0.5, "rate": 50},
        "стоп": {"hours": 0.5, "rate": 50},
        "mirror": {"hours": 0.5, "rate": 50},
        "огледало": {"hours": 0.5, "rate": 50},
        "windshield": {"hours": 2.0, "rate": 60},
        "стъкло": {"hours": 2.0, "rate": 60},
        "radiator": {"hours": 2.5, "rate": 50},
        "радиатор": {"hours": 2.5, "rate": 50},
        "shock": {"hours": 1.5, "rate": 50},
        "амортисьор": {"hours": 1.5, "rate": 50},
    }

    for part_name in parts:
        # Check cache first
        cache_key = f"part:{make.lower()}:{model.lower()}:{year}:{part_name.lower()}"

        if redis_client:
            try:
                cached = redis_client.get(cache_key)
                if cached:
                    logger.info(f"Parts cache hit: {cache_key}")
                    part_result = json.loads(cached)
                    part_result["from_cache"] = True
                    results.append(part_result)
                    if part_result.get("best_price_bgn"):
                        total_parts_cost += part_result["best_price_bgn"]
                    if part_result.get("labor_cost_bgn"):
                        total_labor_cost += part_result["labor_cost_bgn"]
                    continue
            except Exception as e:
                logger.warning(f"Redis read error: {e}")

        # Translate to Bulgarian for better search results
        part_bg = translate_part_name(part_name)

        # Search multiple sources in parallel
        search_results = await search_part_prices(make, model, year, part_name, part_bg)

        # Calculate labor cost
        labor_cost = 0
        if include_labor:
            for keyword, labor_info in LABOR_RATES.items():
                if keyword in part_name.lower() or keyword in part_bg.lower():
                    labor_cost = labor_info["hours"] * labor_info["rate"]
                    break
            if labor_cost == 0:
                labor_cost = 1.5 * 50  # Default: 1.5 hours at 50 BGN

        part_result = {
            "part_name": part_name,
            "part_name_bg": part_bg,
            "search_results": search_results,
            "best_price_bgn": None,
            "best_source": None,
            "price_range": None,
            "labor_cost_bgn": labor_cost if include_labor else None,
        }

        # Find best price
        all_prices = []
        for source, data in search_results.items():
            if data.get("prices"):
                all_prices.extend([(p, source) for p in data["prices"]])

        if all_prices:
            all_prices.sort(key=lambda x: x[0])
            part_result["best_price_bgn"] = all_prices[0][0]
            part_result["best_source"] = all_prices[0][1]
            part_result["price_range"] = {
                "min_bgn": all_prices[0][0],
                "max_bgn": all_prices[-1][0],
                "avg_bgn": round(sum(p[0] for p in all_prices) / len(all_prices), 2)
            }
            total_parts_cost += part_result["best_price_bgn"]

        if labor_cost:
            total_labor_cost += labor_cost

        # Cache result
        if redis_client and part_result.get("best_price_bgn"):
            try:
                redis_client.setex(cache_key, PARTS_CACHE_TTL, json.dumps(part_result))
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")

        results.append(part_result)

    return {
        "make": make,
        "model": model,
        "year": year,
        "parts": results,
        "summary": {
            "total_parts_cost_bgn": round(total_parts_cost, 2),
            "total_labor_cost_bgn": round(total_labor_cost, 2),
            "total_repair_cost_bgn": round(total_parts_cost + total_labor_cost, 2),
            "parts_found": sum(1 for r in results if r.get("best_price_bgn")),
            "parts_not_found": sum(1 for r in results if not r.get("best_price_bgn")),
        },
        "currency": "BGN",
        "note": "Prices are estimates from online sources. Actual prices may vary."
    }


async def search_part_prices(make: str, model: str, year: int, part_en: str, part_bg: str) -> Dict:
    """Search multiple sources for part prices."""
    results = {}

    # Run searches in parallel
    tasks = [
        search_autopower(make, model, year, part_bg),
        search_alo_bg(make, model, year, part_bg),
        search_mobile_bg_parts(make, model, year, part_bg),
    ]

    search_results = await asyncio.gather(*tasks, return_exceptions=True)

    sources = ["autopower.bg", "alo.bg", "mobile.bg"]
    for source, result in zip(sources, search_results):
        if isinstance(result, Exception):
            logger.error(f"Search error for {source}: {result}")
            results[source] = {"error": str(result), "prices": []}
        else:
            results[source] = result

    return results


async def search_autopower(make: str, model: str, year: int, part_bg: str) -> Dict:
    """Search autopower.bg for parts."""
    try:
        # Build search query
        query = f"{part_bg} {make} {model}"
        search_url = f"https://www.autopower.bg/search?q={quote_plus(query)}"

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

            # Extract prices (format: "1 234 лв" or "1234 лв" or "€123")
            # autopower.bg uses various formats
            for match in re.findall(r'(\d{1,3}(?:[\s\xa0]?\d{3})*)\s*(?:лв|BGN)', html):
                try:
                    price = int(match.replace(' ', '').replace('\xa0', ''))
                    if 10 < price < 50000:  # Parts typically 10-50000 BGN
                        prices.append(price)
                except:
                    continue

            # Also check for EUR prices
            for match in re.findall(r'€\s*(\d{1,3}(?:[\s\xa0]?\d{3})*)', html):
                try:
                    price_eur = int(match.replace(' ', '').replace('\xa0', ''))
                    price_bgn = int(price_eur * 1.96)
                    if 10 < price_bgn < 50000:
                        prices.append(price_bgn)
                except:
                    continue

            # Deduplicate and sort
            prices = sorted(list(set(prices)))

            return {
                "prices": prices[:10],  # Top 10 prices
                "count": len(prices),
                "url": search_url
            }

    except Exception as e:
        logger.error(f"autopower.bg error: {e}")
        return {"error": str(e), "prices": []}


async def search_alo_bg(make: str, model: str, year: int, part_bg: str) -> Dict:
    """Search alo.bg for used parts."""
    try:
        # alo.bg search URL
        query = f"{part_bg} {make}"
        search_url = f"https://www.alo.bg/obiavi/?q={quote_plus(query)}"

        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "bg-BG,bg;q=0.9",
        }

        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(search_url, headers=headers)

            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}", "prices": []}

            html = resp.text
            prices = []

            # alo.bg price format: "цена: 123 лв" or just "123 лв"
            for match in re.findall(r'(\d{1,3}(?:[\s\xa0]?\d{3})*)\s*(?:лв|лева|BGN)', html, re.IGNORECASE):
                try:
                    price = int(match.replace(' ', '').replace('\xa0', ''))
                    if 10 < price < 30000:  # Used parts typically cheaper
                        prices.append(price)
                except:
                    continue

            prices = sorted(list(set(prices)))

            return {
                "prices": prices[:10],
                "count": len(prices),
                "url": search_url,
                "type": "used"  # Mark as used parts
            }

    except Exception as e:
        logger.error(f"alo.bg error: {e}")
        return {"error": str(e), "prices": []}


async def search_mobile_bg_parts(make: str, model: str, year: int, part_bg: str) -> Dict:
    """Search mobile.bg parts section."""
    try:
        # mobile.bg auto parts section
        query = f"{part_bg} {make}"
        search_url = f"https://www.mobile.bg/obiavi/avto-chasti-aksesori/?q={quote_plus(query)}"

        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "bg-BG,bg;q=0.9",
        }

        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(search_url, headers=headers)

            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}", "prices": []}

            try:
                html = resp.content.decode('windows-1251')
            except:
                html = resp.text

            prices = []

            # Extract prices
            for match in re.findall(r'(\d{1,3}(?:[\s\xa0]?\d{3})*)\s*(?:лв|BGN)', html):
                try:
                    price = int(match.replace(' ', '').replace('\xa0', ''))
                    if 10 < price < 50000:
                        prices.append(price)
                except:
                    continue

            prices = sorted(list(set(prices)))

            return {
                "prices": prices[:10],
                "count": len(prices),
                "url": search_url
            }

    except Exception as e:
        logger.error(f"mobile.bg parts error: {e}")
        return {"error": str(e), "prices": []}


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "Car Value Service",
        "version": "3.2.0",
        "description": "Bulgarian car market value + VIN decoder + parts pricing",
        "endpoints": {
            "/vin/{vin}": "Decode VIN to get vehicle info",
            "/value/{make}/{model}/{year}": "Get car market value",
            "/value-by-vin/{vin}": "Get car value by VIN",
            "/parts/search": "Search for parts prices (POST with make, model, year, parts[])"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
