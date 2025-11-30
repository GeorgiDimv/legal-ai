#!/usr/bin/env python3
"""
Test script for Legal AI Document Processing Pipeline

Usage:
    python test_pipeline.py                    # Run health checks only
    python test_pipeline.py sample.pdf         # Process a sample document
    python test_pipeline.py --all              # Run all tests
"""

import sys
import time
import json
import argparse
from pathlib import Path

import httpx

# Service URLs
GATEWAY_URL = "http://localhost:80"
LLM_URL = "http://localhost:8000"
OCR_URL = "http://localhost:8001"
NOMINATIM_URL = "http://localhost:8002"
CAR_VALUE_URL = "http://localhost:8003"


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_result(name: str, success: bool, message: str = ""):
    """Print a test result."""
    status = "[OK]" if success else "[FAIL]"
    color = "\033[92m" if success else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{status}{reset} {name}")
    if message:
        print(f"     {message}")


def test_health_checks() -> bool:
    """Test health endpoints for all services."""
    print_header("Health Checks")

    services = [
        ("Gateway", f"{GATEWAY_URL}/health"),
        ("LLM (vLLM)", f"{LLM_URL}/health"),
        ("OCR (PaddleOCR)", f"{OCR_URL}/health"),
        ("Nominatim", f"{NOMINATIM_URL}/status.php"),
        ("Car Value", f"{CAR_VALUE_URL}/health"),
    ]

    all_healthy = True

    for name, url in services:
        try:
            r = httpx.get(url, timeout=10)
            if r.status_code == 200:
                print_result(name, True, f"Status: {r.status_code}")
            else:
                print_result(name, False, f"Status: {r.status_code}")
                all_healthy = False
        except httpx.ConnectError:
            print_result(name, False, "Connection refused - service not running")
            all_healthy = False
        except httpx.TimeoutException:
            print_result(name, False, "Request timed out")
            all_healthy = False
        except Exception as e:
            print_result(name, False, str(e))
            all_healthy = False

    return all_healthy


def test_ocr_service() -> bool:
    """Test OCR service with a simple image."""
    print_header("OCR Service Test")

    # Create a simple test image with text (requires PIL)
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io

        # Create test image
        img = Image.new('RGB', (400, 100), color='white')
        d = ImageDraw.Draw(img)
        d.text((10, 40), "Test OCR 12345", fill='black')

        # Save to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        # Send to OCR
        files = {"file": ("test.png", img_bytes, "image/png")}
        r = httpx.post(f"{OCR_URL}/ocr", files=files, timeout=30)

        if r.status_code == 200:
            result = r.json()
            text = result.get("text", "")
            if "Test" in text or "12345" in text:
                print_result("OCR extraction", True, f"Extracted: '{text[:50]}...'")
                return True
            else:
                print_result("OCR extraction", False, f"Expected text not found: '{text[:50]}'")
                return False
        else:
            print_result("OCR extraction", False, f"Status: {r.status_code}")
            return False

    except ImportError:
        print_result("OCR extraction", False, "PIL not installed - skipping image test")
        return True
    except Exception as e:
        print_result("OCR extraction", False, str(e))
        return False


def test_car_value_service() -> bool:
    """Test car value lookup."""
    print_header("Car Value Service Test")

    test_cases = [
        ("Volkswagen", "Golf", 2018),
        ("BMW", "3 Series", 2020),
        ("Toyota", "Corolla", 2019),
    ]

    all_passed = True

    for make, model, year in test_cases:
        try:
            r = httpx.get(f"{CAR_VALUE_URL}/value/{make}/{model}/{year}", timeout=15)

            if r.status_code == 200:
                result = r.json()
                if "error" not in result:
                    price = result.get("average_price_bgn", 0)
                    source = result.get("source", "unknown")
                    print_result(
                        f"{make} {model} {year}",
                        True,
                        f"{price:,.0f} BGN (source: {source})"
                    )
                else:
                    print_result(f"{make} {model} {year}", False, result.get("error"))
                    all_passed = False
            else:
                print_result(f"{make} {model} {year}", False, f"Status: {r.status_code}")
                all_passed = False

        except Exception as e:
            print_result(f"{make} {model} {year}", False, str(e))
            all_passed = False

    return all_passed


def test_nominatim_service() -> bool:
    """Test geocoding service."""
    print_header("Nominatim Geocoding Test")

    test_queries = [
        "Sofia, Bulgaria",
        "Plovdiv, Bulgaria",
        "бул. Витоша, София",  # Bulgarian text
    ]

    all_passed = True

    for query in test_queries:
        try:
            r = httpx.get(
                f"{NOMINATIM_URL}/search",
                params={"q": query, "format": "json", "limit": 1},
                timeout=10
            )

            if r.status_code == 200:
                results = r.json()
                if results:
                    lat = results[0].get("lat")
                    lon = results[0].get("lon")
                    print_result(query, True, f"({lat}, {lon})")
                else:
                    print_result(query, False, "No results")
                    all_passed = False
            else:
                print_result(query, False, f"Status: {r.status_code}")
                all_passed = False

        except Exception as e:
            print_result(query, False, str(e))
            all_passed = False

    return all_passed


def test_llm_service() -> bool:
    """Test LLM inference."""
    print_header("LLM Service Test")

    try:
        # Test with a simple prompt
        r = httpx.post(
            f"{LLM_URL}/v1/chat/completions",
            json={
                "model": "Qwen/Qwen3-32B-AWQ",
                "messages": [{"role": "user", "content": "Say 'Hello' in Bulgarian."}],
                "max_tokens": 50,
                "temperature": 0.1
            },
            timeout=60
        )

        if r.status_code == 200:
            result = r.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            print_result("LLM inference", True, f"Response: '{content[:100]}'")
            return True
        else:
            print_result("LLM inference", False, f"Status: {r.status_code} - {r.text[:200]}")
            return False

    except httpx.TimeoutException:
        print_result("LLM inference", False, "Timeout - model may still be loading")
        return False
    except Exception as e:
        print_result("LLM inference", False, str(e))
        return False


def test_process_endpoint(pdf_path: str) -> bool:
    """Test the main /process endpoint with a real document."""
    print_header(f"Processing Document: {pdf_path}")

    if not Path(pdf_path).exists():
        print_result("File check", False, f"File not found: {pdf_path}")
        return False

    try:
        with open(pdf_path, "rb") as f:
            files = {"file": (Path(pdf_path).name, f, "application/pdf")}

            print("Sending to /process endpoint...")
            start = time.time()

            r = httpx.post(
                f"{GATEWAY_URL}/process",
                files=files,
                timeout=300  # 5 minutes for large documents
            )

            elapsed = time.time() - start

        if r.status_code == 200:
            result = r.json()

            print_result("Processing", True, f"Completed in {elapsed:.1f}s")
            print("\n--- Extraction Results ---\n")

            print(f"Claim Number: {result.get('claim_number', 'N/A')}")
            print(f"Accident Date: {result.get('accident_date', 'N/A')}")
            print(f"Accident Time: {result.get('accident_time', 'N/A')}")

            if result.get("accident_location"):
                loc = result["accident_location"]
                print(f"Location: {loc.get('address', 'N/A')}, {loc.get('city', 'N/A')}")
                if loc.get("latitude"):
                    print(f"  Coordinates: ({loc['latitude']}, {loc['longitude']})")

            print(f"\nVehicles: {len(result.get('vehicles', []))}")
            for i, v in enumerate(result.get("vehicles", [])):
                print(f"  {i+1}. {v.get('make', '?')} {v.get('model', '?')} ({v.get('year', '?')})")
                print(f"     Reg: {v.get('registration', 'N/A')}")
                print(f"     Damage: {v.get('estimated_damage_bgn', 0):,.0f} BGN")
                if v.get("current_market_value_bgn"):
                    print(f"     Market Value: {v['current_market_value_bgn']:,.0f} BGN ({v.get('market_value_source', 'N/A')})")

            print(f"\nParties: {len(result.get('parties', []))}")
            for p in result.get("parties", []):
                print(f"  - {p.get('name', 'N/A')} ({p.get('role', 'N/A')})")

            if result.get("fault_determination"):
                fd = result["fault_determination"]
                print(f"\nFault: {fd.get('primary_fault_party', 'N/A')} - {fd.get('fault_percentage', 0)}%")
                print(f"  Reason: {fd.get('reasoning', 'N/A')[:100]}...")

            if result.get("settlement_recommendation"):
                sr = result["settlement_recommendation"]
                print(f"\nSettlement: {sr.get('amount_bgn', 0):,.0f} BGN")
                if sr.get("components"):
                    comp = sr["components"]
                    print(f"  Vehicle damage: {comp.get('vehicle_damage', 0):,.0f} BGN")
                    print(f"  Medical: {comp.get('medical_expenses', 0):,.0f} BGN")

            print(f"\nConfidence: {result.get('confidence_score', 0)*100:.1f}%")
            print(f"Processing Time: {result.get('processing_time_seconds', 0):.2f}s")

            if result.get("errors"):
                print(f"\nErrors: {result['errors']}")
            if result.get("warnings"):
                print(f"Warnings: {result['warnings']}")

            return True

        else:
            print_result("Processing", False, f"Status: {r.status_code} - {r.text[:500]}")
            return False

    except Exception as e:
        print_result("Processing", False, str(e))
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test Legal AI Document Processing Pipeline"
    )
    parser.add_argument(
        "document",
        nargs="?",
        help="Path to PDF document to process"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests including service-specific tests"
    )
    parser.add_argument(
        "--health-only",
        action="store_true",
        help="Only run health checks"
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("  Legal AI Document Processing Pipeline - Test Suite")
    print("="*60)

    # Always run health checks first
    healthy = test_health_checks()

    if args.health_only:
        sys.exit(0 if healthy else 1)

    if not healthy:
        print("\n[WARNING] Some services are not healthy. Tests may fail.\n")

    if args.all:
        # Run all service tests
        test_ocr_service()
        test_car_value_service()
        test_nominatim_service()
        test_llm_service()

    if args.document:
        test_process_endpoint(args.document)
    elif not args.all:
        print("\n[INFO] To process a document, run:")
        print("  python test_pipeline.py <path_to_pdf>")
        print("\n[INFO] To run all tests:")
        print("  python test_pipeline.py --all")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
