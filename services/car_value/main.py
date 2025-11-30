"""
Car Value Service - On-Demand Price Lookup + VIN Decoding
Provides current market values for Bulgarian car insurance claims

Features:
1. VIN decoding via NHTSA API (free, works for EU vehicles)
2. On-demand scraping from cars.bg, mobile.bg
3. Redis caching (24h for prices, forever for VIN)
4. Parts pricing from database
"""

import os
import re
import json
import logging
import hashlib
from typing import Optional, List, Dict
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import httpx
import redis
import asyncpg

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Car Value Service",
    description="Bulgarian car market value + VIN decoder",
    version="3.0.0"
)

# Redis client for caching
redis_client: Optional[redis.Redis] = None

# PostgreSQL connection pool (only for parts)
db_pool: Optional[asyncpg.Pool] = None

# Cache TTL
PRICE_CACHE_TTL = 86400  # 24 hours for prices
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
    global redis_client, db_pool

    # Connect to Redis
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    try:
        redis_client = redis.from_url(redis_url, decode_responses=True)
        redis_client.ping()
        logger.info("Connected to Redis")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        redis_client = None

    # Connect to PostgreSQL (for parts only)
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        try:
            db_pool = await asyncpg.create_pool(database_url, min_size=1, max_size=5)
            logger.info("Connected to PostgreSQL")
        except Exception as e:
            logger.warning(f"PostgreSQL connection failed: {e}")
            db_pool = None


@app.on_event("shutdown")
async def shutdown():
    """Cleanup connections on shutdown."""
    if db_pool:
        await db_pool.close()


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
        "version": "3.0.0",
        "redis": redis_ok,
        "database": db_pool is not None
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
# Parts Endpoints (kept from v2)
# =============================================================================

@app.get("/parts/{make}/{part_category}")
async def get_parts_prices(make: str, part_category: str, model: Optional[str] = None):
    """Get parts prices for a vehicle."""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        async with db_pool.acquire() as conn:
            if model:
                rows = await conn.fetch("""
                    SELECT part_name, part_name_bg, oem_price_bgn, aftermarket_price_bgn,
                           labor_hours, labor_rate_bgn
                    FROM car_parts
                    WHERE (LOWER(make) = LOWER($1) OR make IS NULL)
                      AND (LOWER(model) = LOWER($2) OR model IS NULL)
                      AND LOWER(part_category) = LOWER($3)
                    ORDER BY make DESC NULLS LAST, part_name
                """, make, model, part_category)
            else:
                rows = await conn.fetch("""
                    SELECT part_name, part_name_bg, oem_price_bgn, aftermarket_price_bgn,
                           labor_hours, labor_rate_bgn
                    FROM car_parts
                    WHERE (LOWER(make) = LOWER($1) OR make IS NULL)
                      AND LOWER(part_category) = LOWER($2)
                    ORDER BY make DESC NULLS LAST, part_name
                """, make, part_category)

            if not rows:
                rows = await conn.fetch("""
                    SELECT part_name, part_name_bg, oem_price_bgn, aftermarket_price_bgn,
                           labor_hours, labor_rate_bgn
                    FROM car_parts
                    WHERE make IS NULL AND LOWER(part_category) = LOWER($1)
                """, part_category)

            parts = []
            for row in rows:
                labor_cost = (row["labor_hours"] or 0) * (row["labor_rate_bgn"] or 50)
                parts.append({
                    "part_name": row["part_name"],
                    "part_name_bg": row["part_name_bg"],
                    "oem_price_bgn": row["oem_price_bgn"],
                    "aftermarket_price_bgn": row["aftermarket_price_bgn"],
                    "labor_hours": row["labor_hours"],
                    "labor_cost_bgn": round(labor_cost, 2),
                    "total_oem_bgn": round((row["oem_price_bgn"] or 0) + labor_cost, 2),
                    "total_aftermarket_bgn": round((row["aftermarket_price_bgn"] or 0) + labor_cost, 2)
                })

            return {
                "make": make,
                "model": model,
                "category": part_category,
                "parts": parts,
                "currency": "BGN"
            }

    except Exception as e:
        logger.error(f"Parts lookup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/parts/estimate-damage")
async def estimate_damage_cost(
    make: str,
    parts: str,
    use_oem: bool = False
):
    """Estimate total damage cost for multiple parts."""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")

    part_list = [p.strip() for p in parts.split(",")]

    try:
        async with db_pool.acquire() as conn:
            total_parts = 0
            total_labor = 0
            breakdown = []

            for part_name in part_list:
                row = await conn.fetchrow("""
                    SELECT part_name, oem_price_bgn, aftermarket_price_bgn,
                           labor_hours, labor_rate_bgn
                    FROM car_parts
                    WHERE (LOWER(make) = LOWER($1) OR make IS NULL)
                      AND LOWER(part_name) = LOWER($2)
                    ORDER BY make DESC NULLS LAST
                    LIMIT 1
                """, make, part_name)

                if row:
                    price = row["oem_price_bgn"] if use_oem else row["aftermarket_price_bgn"]
                    labor = (row["labor_hours"] or 0) * (row["labor_rate_bgn"] or 50)

                    if price:
                        total_parts += price
                        total_labor += labor
                        breakdown.append({
                            "part": row["part_name"],
                            "part_cost_bgn": price,
                            "labor_cost_bgn": round(labor, 2)
                        })

            return {
                "make": make,
                "pricing_type": "OEM" if use_oem else "Aftermarket",
                "parts_cost_bgn": round(total_parts, 2),
                "labor_cost_bgn": round(total_labor, 2),
                "total_cost_bgn": round(total_parts + total_labor, 2),
                "breakdown": breakdown,
                "currency": "BGN"
            }

    except Exception as e:
        logger.error(f"Damage estimate error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "Car Value Service",
        "version": "3.0.0",
        "description": "Bulgarian car market value + VIN decoder",
        "endpoints": {
            "/vin/{vin}": "Decode VIN to get vehicle info",
            "/value/{make}/{model}/{year}": "Get car market value",
            "/value-by-vin/{vin}": "Get car value by VIN",
            "/parts/{make}/{category}": "Get parts prices",
            "/parts/estimate-damage": "Estimate damage repair cost"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
