"""
LLM-based extraction logic for insurance claim documents
"""

import json
import re
import logging
from typing import Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Extraction prompt for Bulgarian insurance claims
EXTRACTION_PROMPT = """Analyze this Bulgarian automotive insurance claim document and extract structured information.

Document Text:
{document_text}

Extract the following information and return as valid JSON:

{{
  "claim_number": "string or null - the claim/case reference number",
  "accident_date": "YYYY-MM-DD format or null",
  "accident_time": "HH:MM format or null",
  "accident_location": {{
    "address": "street address in Bulgarian or English",
    "city": "city name"
  }},
  "vehicles": [
    {{
      "vin": "17-character Vehicle Identification Number (VIN) or null",
      "registration": "license plate number",
      "make": "manufacturer (e.g., Volkswagen, BMW)",
      "model": "model name (e.g., Golf, 3 Series)",
      "year": integer year of manufacture,
      "mass_kg": integer vehicle mass in kg (estimate 1200-1800 for cars if not specified),
      "owner_name": "owner's full name",
      "insurance_company": "insurance company name",
      "policy_number": "insurance policy number",
      "damage_description": "description of damage in English",
      "damaged_parts": ["REQUIRED: extract each damaged part as separate item e.g. front bumper, headlight left, hood, door, fender"],
      "estimated_damage": float amount in BGN,
      "skid_distance_m": float length of skid marks in meters or null,
      "post_impact_travel_m": float distance traveled after impact (σ) in meters or null,
      "claimed_speed_kmh": float driver's claimed speed in km/h or null,
      "pre_impact_angle_deg": float angle of travel before impact (α) 0-360 degrees or null,
      "post_impact_angle_deg": float angle of travel after impact (β) 0-360 degrees or null
    }}
  ],
  "parties": [
    {{
      "name": "person's full name",
      "role": "driver|passenger|pedestrian|witness",
      "vehicle_index": integer index into vehicles array or null,
      "injuries": "injury description or null",
      "statement_summary": "brief summary of their statement"
    }}
  ],
  "accident_description": "detailed description of what happened in English",
  "fault_determination": {{
    "primary_fault_party": "name of person primarily at fault",
    "fault_percentage": integer 0-100,
    "reasoning": "explanation of fault determination",
    "traffic_violations": ["list of traffic rules violated"]
  }},
  "police_report": {{
    "report_number": "police report number or null",
    "officer_name": "officer name or null",
    "findings": "police findings summary or null"
  }},
  "settlement_recommendation": {{
    "amount_bgn": float total recommended settlement in BGN,
    "components": {{
      "vehicle_damage": float,
      "medical_expenses": float,
      "lost_income": float,
      "pain_and_suffering": float
    }},
    "reasoning": "explanation of settlement calculation"
  }},
  "collision_details": {{
    "collision_type": "head_on|rear_end|side_impact|angle or null",
    "road_surface": "dry_asphalt|wet_asphalt|gravel|snow|ice or null",
    "road_grade_percent": float road incline percentage or 0,
    "impact_angle_deg": float angle of impact impulse (αs) 0-360 degrees or null,
    "restitution_coefficient": float coefficient of restitution 0.0-1.0 (default 0.4 for vehicles) or null
  }},
  "risk_factors": ["list of factors that could affect the claim"],
  "confidence_score": float 0.0-1.0 indicating extraction confidence
}}

Important instructions:
1. All monetary amounts must be in Bulgarian Lev (BGN)
2. Dates must be in ISO format (YYYY-MM-DD)
3. Translate Bulgarian text to English in the response
4. If information is missing or unclear, use null
5. For fault_percentage, 100 means fully at fault, 0 means not at fault
6. Be conservative with confidence_score - lower if document is unclear
7. Extract actual values from the document, do not make up information
8. Return ONLY the JSON object, no additional text
9. For physics angles (α, β, αs): 0° = East/right, 90° = North/up, 180° = West/left, 270° = South/down
10. pre_impact_angle_deg (α) = direction vehicle was traveling BEFORE collision
11. post_impact_angle_deg (β) = direction vehicle traveled AFTER collision
12. If angles are described as "heading north" use 90°, "heading east" use 0°, etc.
13. For head-on collisions: vehicle 1 α≈0°, vehicle 2 α≈180° typically
14. IMPORTANT: Always extract VIN if present - it is exactly 17 characters (e.g. WVWZZZ1KZAW123456)
15. VIN may appear as "VIN:", "Рама №", "Шаси №" in documents - ALWAYS include it if found
16. IMPORTANT: Always extract damaged_parts as a list - parse damage description into individual parts
17. CRITICAL: Each vehicle MUST have a UNIQUE VIN - NEVER assign the same VIN to multiple vehicles!
18. CRITICAL: Each vehicle has its own registration number - extract EACH vehicle's registration separately
19. Look for vehicle data in separate sections/blocks - match VIN, registration, make, model, year to the correct vehicle
20. If a vehicle is missing data, use null - do NOT copy data from another vehicle
21. CRITICAL FOR DAMAGES: In Bulgarian protocols, vehicle A (МПС А) and vehicle B (МПС Б) have SEPARATE damage sections
    - Look for "ВИДИМИ ЩЕТИ" or "ПОВРЕДИ" under EACH vehicle's section
    - Vehicle A's damages are listed ONLY under "ПРЕВОЗНО СРЕДСТВО А" section
    - Vehicle B's damages are listed ONLY under "ПРЕВОЗНО СРЕДСТВО Б" section
    - DO NOT mix or swap damages between vehicles!
    - If vehicle A hit vehicle B, A's damage is typically FRONT, B's damage is typically REAR/SIDE
22. In lane-change accidents: the vehicle changing lanes (at fault) hits with its FRONT, victim is hit on SIDE/REAR

JSON Response:"""


async def extract_with_llm(
    client: AsyncOpenAI,
    model: str,
    document_text: str,
    max_tokens: int = 4000
) -> dict:
    """
    Extract structured data from document text using LLM.

    Args:
        client: OpenAI-compatible async client
        model: Model name to use
        document_text: OCR-extracted text from document
        max_tokens: Maximum tokens in response

    Returns:
        Extracted data as dictionary
    """
    # Truncate document if too long (preserve beginning and end)
    max_doc_length = 15000
    if len(document_text) > max_doc_length:
        half = max_doc_length // 2
        document_text = (
            document_text[:half] +
            "\n\n[... middle section truncated ...]\n\n" +
            document_text[-half:]
        )

    prompt = EXTRACTION_PROMPT.format(document_text=document_text)

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a legal document analysis expert specializing in Bulgarian automotive insurance claims. Extract information accurately and return valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,  # Low temperature for factual extraction
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        if not content:
            logger.warning("LLM returned empty response")
            return get_empty_result()

        # Parse JSON response
        result = parse_llm_response(content)
        return result

    except Exception as e:
        logger.error(f"LLM extraction error: {str(e)}")
        return get_empty_result(error=str(e))


def parse_llm_response(content: str) -> dict:
    """
    Parse LLM response, handling various formats.

    Handles:
    - Clean JSON
    - JSON in markdown code blocks
    - JSON with trailing text
    """
    content = content.strip()

    # Remove markdown code blocks if present
    if content.startswith("```"):
        # Find the actual JSON content
        lines = content.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```"):
                in_block = not in_block
                continue
            if in_block or not line.startswith("```"):
                json_lines.append(line)
        content = "\n".join(json_lines)

    # Try to find JSON object boundaries
    start_idx = content.find("{")
    if start_idx == -1:
        logger.warning("No JSON object found in response")
        return get_empty_result()

    # Find matching closing brace
    depth = 0
    end_idx = start_idx
    for i, char in enumerate(content[start_idx:], start=start_idx):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end_idx = i + 1
                break

    json_str = content[start_idx:end_idx]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e}")
        # Try to fix common issues
        json_str = fix_json_issues(json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.error("Could not parse LLM response as JSON")
            return get_empty_result()


def fix_json_issues(json_str: str) -> str:
    """Attempt to fix common JSON issues from LLM output."""
    # Replace single quotes with double quotes
    json_str = re.sub(r"(?<!\\)'", '"', json_str)

    # Fix trailing commas
    json_str = re.sub(r",\s*}", "}", json_str)
    json_str = re.sub(r",\s*]", "]", json_str)

    # Fix unquoted keys
    json_str = re.sub(r'(\w+)(?=\s*:)', r'"\1"', json_str)

    return json_str


def get_empty_result(error: Optional[str] = None) -> dict:
    """Return empty extraction result with optional error."""
    result = {
        "claim_number": None,
        "accident_date": None,
        "accident_time": None,
        "accident_location": None,
        "vehicles": [],
        "parties": [],
        "accident_description": None,
        "fault_determination": None,
        "police_report": None,
        "settlement_recommendation": None,
        "risk_factors": [],
        "confidence_score": 0.0
    }
    if error:
        result["error"] = error
    return result


def validate_extraction(data: dict) -> tuple[dict, list]:
    """
    Validate and clean extracted data.

    Returns:
        Tuple of (cleaned_data, warnings)
    """
    warnings = []

    # Validate dates
    if data.get("accident_date"):
        if not re.match(r"\d{4}-\d{2}-\d{2}", data["accident_date"]):
            warnings.append(f"Invalid date format: {data['accident_date']}")
            data["accident_date"] = None

    # Validate time
    if data.get("accident_time"):
        if not re.match(r"\d{2}:\d{2}", data["accident_time"]):
            warnings.append(f"Invalid time format: {data['accident_time']}")
            data["accident_time"] = None

    # Validate vehicles
    valid_vehicles = []
    # VIN regex: 17 chars, excludes I, O, Q
    vin_pattern = re.compile(r'^[A-HJ-NPR-Z0-9]{17}$')
    for v in data.get("vehicles", []):
        if isinstance(v, dict):
            # Ensure year is reasonable
            if v.get("year") and (v["year"] < 1900 or v["year"] > 2030):
                warnings.append(f"Invalid vehicle year: {v['year']}")
                v["year"] = None
            # Validate VIN format
            if v.get("vin"):
                vin = v["vin"].upper().replace(" ", "").replace("-", "")
                if vin_pattern.match(vin):
                    v["vin"] = vin
                else:
                    warnings.append(f"Invalid VIN format: {v['vin']}")
                    v["vin"] = None
            valid_vehicles.append(v)
    data["vehicles"] = valid_vehicles

    # Validate fault percentage
    fault = data.get("fault_determination", {})
    if isinstance(fault, dict) and fault.get("fault_percentage") is not None:
        pct = fault["fault_percentage"]
        if not isinstance(pct, (int, float)) or pct < 0 or pct > 100:
            warnings.append(f"Invalid fault percentage: {pct}")
            fault["fault_percentage"] = None

    # Validate confidence score
    if data.get("confidence_score") is not None:
        conf = data["confidence_score"]
        if not isinstance(conf, (int, float)):
            data["confidence_score"] = 0.5
        elif conf < 0:
            data["confidence_score"] = 0.0
        elif conf > 1:
            data["confidence_score"] = 1.0

    # Validate settlement amounts
    settlement = data.get("settlement_recommendation", {})
    if isinstance(settlement, dict):
        if settlement.get("amount_bgn") and settlement["amount_bgn"] < 0:
            warnings.append(f"Negative settlement amount: {settlement['amount_bgn']}")
            settlement["amount_bgn"] = None

    return data, warnings


def extract_vins_from_text(text: str) -> list[str]:
    """
    Extract VINs from raw text using regex.
    VINs are exactly 17 characters, excluding I, O, Q.
    Returns VINs in order of appearance in the document.
    """
    if not text:
        return []

    # Look for VIN patterns - various label formats in Bulgarian and English
    vin_pattern = re.compile(
        r'(?:VIN|Рама\s*№?|Шаси\s*№?|vin|Vehicle\s+Identification)\s*[:\s]*([A-HJ-NPR-Z0-9]{17})',
        re.IGNORECASE
    )
    matches = vin_pattern.findall(text)

    # Also look for standalone 17-char sequences that look like VINs
    standalone_pattern = re.compile(r'\b([A-HJ-NPR-Z0-9]{17})\b')
    all_matches = standalone_pattern.findall(text.upper())

    # Combine and dedupe while preserving order
    vins = list(dict.fromkeys([m.upper() for m in matches] + all_matches))

    # Filter out obvious non-VINs (all digits, all letters)
    valid_vins = []
    for vin in vins:
        has_digit = any(c.isdigit() for c in vin)
        has_letter = any(c.isalpha() for c in vin)
        if has_digit and has_letter:
            valid_vins.append(vin)

    return valid_vins


def extract_registrations_from_text(text: str) -> list[str]:
    """
    Extract Bulgarian vehicle registration numbers from text.
    Format: 2 letters + space + 4 digits + space + 2 letters (e.g., СА 4521 КМ)
    """
    if not text:
        return []

    # Bulgarian registration pattern: 2 Cyrillic letters, 4 digits, 2 Cyrillic letters
    # Also handles Latin letters (common in OCR)
    reg_pattern = re.compile(
        r'[АВЕКМНОРСТУХ]{1,2}\s*\d{4}\s*[АВЕКМНОРСТУХ]{2}|'  # Cyrillic
        r'[ABEKMHOPCTYX]{1,2}\s*\d{4}\s*[ABEKMHOPCTYX]{2}',  # Latin transliteration
        re.IGNORECASE
    )
    matches = reg_pattern.findall(text)

    # Normalize: add spaces between parts
    normalized = []
    for m in matches:
        # Remove extra spaces and normalize
        clean = re.sub(r'\s+', '', m.upper())
        if len(clean) >= 7:  # At least 1 letter + 4 digits + 2 letters
            # Insert spaces: XX 1234 XX
            letters1 = clean[:2] if clean[1].isalpha() else clean[:1]
            rest = clean[len(letters1):]
            digits = rest[:4]
            letters2 = rest[4:6]
            formatted = f"{letters1} {digits} {letters2}"
            if formatted not in normalized:
                normalized.append(formatted)

    return normalized


def extract_year_from_text(text: str) -> list[int]:
    """
    Extract vehicle manufacture years from text.
    Specifically looks for "Година:" (Year:) labels to find vehicle years.
    Filters out document dates, policy expiry dates, and future years.
    """
    if not text:
        return []

    # Current year - vehicles can't be newer than current year
    import datetime
    current_year = datetime.datetime.now().year

    # First priority: Look for years specifically labeled as "Година:" (vehicle year)
    vehicle_year_pattern = re.compile(
        r'(?:Година|година)[:\s]*(\d{4})',
        re.IGNORECASE
    )
    labeled_years = vehicle_year_pattern.findall(text)

    years = []
    for year_str in labeled_years:
        year = int(year_str)
        # Vehicle years should be: not in the future, not more than 35 years old
        if 1990 <= year <= current_year and year not in years:
            years.append(year)

    # If no labeled years found, try standalone years but filter aggressively
    if not years:
        standalone_pattern = re.compile(r'\b(19[9][0-9]|20[0-2][0-5])\b')
        all_matches = standalone_pattern.findall(text)
        for year_str in all_matches:
            year = int(year_str)
            # Only include if it looks like a reasonable vehicle year
            if 1990 <= year <= current_year and year not in years:
                years.append(year)

    return years


def extract_damaged_parts_from_text(text: str) -> list[str]:
    """
    Extract damaged parts from Bulgarian damage descriptions using keyword matching.
    """
    if not text:
        return []

    # Bulgarian -> English part mappings
    part_keywords = {
        # Bumpers
        'предна броня': 'front bumper',
        'задна броня': 'rear bumper',
        'броня': 'bumper',
        # Lights
        'преден фар': 'front headlight',
        'заден фар': 'rear taillight',
        'фар': 'headlight',
        'ляв фар': 'left headlight',
        'десен фар': 'right headlight',
        'мигач': 'turn signal',
        # Body panels
        'калник': 'fender',
        'преден калник': 'front fender',
        'заден калник': 'rear fender',
        'капак': 'hood',
        'врата': 'door',
        'предна врата': 'front door',
        'задна врата': 'rear door',
        'лява врата': 'left door',
        'дясна врата': 'right door',
        'задна дясна врата': 'rear right door',
        'задна лява врата': 'rear left door',
        'предна дясна врата': 'front right door',
        'предна лява врата': 'front left door',
        'багажник': 'trunk',
        'таван': 'roof',
        'праг': 'rocker panel',
        # Glass
        'предно стъкло': 'windshield',
        'задно стъкло': 'rear window',
        'странично стъкло': 'side window',
        'огледало': 'mirror',
        # Damage descriptions (action words)
        'деформиран': 'deformed',
        'счупен': 'broken',
        'вдлъбнатина': 'dent',
        'драскотина': 'scratch',
    }

    text_lower = text.lower()
    found_parts = []

    # Look for Bulgarian part keywords
    for bg_term, en_term in part_keywords.items():
        if bg_term in text_lower:
            # Avoid duplicates, prefer specific terms over generic
            if en_term not in found_parts:
                found_parts.append(en_term)

    # Filter out action words (keep only parts)
    action_words = {'deformed', 'broken', 'dent', 'scratch'}
    found_parts = [p for p in found_parts if p not in action_words]

    return found_parts


def enrich_extraction_with_fallback(data: dict, raw_text: str) -> tuple[dict, list]:
    """
    Enrich LLM extraction with regex-based fallbacks for commonly missed fields.
    Runs after validate_extraction to fill in gaps.
    Also fixes duplicate VINs, registrations, and years.

    Returns:
        Tuple of (enriched_data, info_messages)
    """
    info = []

    # Extract all VINs, registrations, and years from raw text
    text_vins = extract_vins_from_text(raw_text)
    text_regs = extract_registrations_from_text(raw_text)
    text_years = extract_year_from_text(raw_text)

    logger.info(f"Fallback extraction found: VINs={text_vins}, regs={text_regs}, years={text_years}")

    vehicles = data.get("vehicles", [])

    # Step 1: Detect and fix duplicate VINs
    if len(vehicles) >= 2 and len(text_vins) >= 2:
        vins_in_vehicles = [v.get("vin") for v in vehicles]
        # Check if all vehicles have the same VIN (duplicate bug)
        unique_vins = set(v for v in vins_in_vehicles if v)
        if len(unique_vins) == 1 and len(vehicles) > 1:
            # All vehicles have same VIN - this is wrong! Reassign from text_vins
            info.append(f"Detected duplicate VIN {unique_vins.pop()} - reassigning from text")
            for i, v in enumerate(vehicles):
                if i < len(text_vins):
                    old_vin = v.get("vin")
                    v["vin"] = text_vins[i]
                    info.append(f"Vehicle {i}: VIN {old_vin} -> {text_vins[i]}")

    # Step 2: Fill in missing VINs
    vin_index = 0
    for v in vehicles:
        if not v.get("vin") and vin_index < len(text_vins):
            v["vin"] = text_vins[vin_index]
            info.append(f"Fallback: extracted VIN {text_vins[vin_index]} for vehicle")
            vin_index += 1

    # Step 3: Detect and fix duplicate/missing registrations
    if len(vehicles) >= 2 and len(text_regs) >= 2:
        regs_in_vehicles = [v.get("registration") for v in vehicles]
        unique_regs = set(r for r in regs_in_vehicles if r)
        # Fix if duplicate or if some are missing
        if len(unique_regs) < len(vehicles):
            for i, v in enumerate(vehicles):
                if i < len(text_regs):
                    if not v.get("registration") or v.get("registration") == regs_in_vehicles[0]:
                        old_reg = v.get("registration")
                        v["registration"] = text_regs[i]
                        if old_reg != text_regs[i]:
                            info.append(f"Vehicle {i}: registration {old_reg} -> {text_regs[i]}")

    # Step 4: Fill in missing registrations
    for i, v in enumerate(vehicles):
        if not v.get("registration") and i < len(text_regs):
            v["registration"] = text_regs[i]
            info.append(f"Fallback: extracted registration {text_regs[i]} for vehicle {i}")

    # Step 5: Fill in missing years
    for i, v in enumerate(vehicles):
        if not v.get("year") and i < len(text_years):
            v["year"] = text_years[i]
            info.append(f"Fallback: extracted year {text_years[i]} for vehicle {i}")

    # Step 6: Extract damaged parts if missing
    for v in vehicles:
        if not v.get("damaged_parts") or len(v.get("damaged_parts", [])) == 0:
            # Try to extract from vehicle's damage_description first
            damage_desc = v.get("damage_description", "")
            parts = extract_damaged_parts_from_text(damage_desc)

            # If nothing from description, try the raw text
            if not parts:
                parts = extract_damaged_parts_from_text(raw_text)

            if parts:
                v["damaged_parts"] = parts
                info.append(f"Fallback: extracted parts {parts} from text")

    return data, info
