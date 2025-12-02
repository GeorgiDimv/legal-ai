# Naredba 24 Implementation Status

**Date:** 2025-12-02
**Service:** car_value v4.0.0
**Purpose:** Track compliance with Bulgarian ATE regulation (Naredba 24)

---

## Implemented Features

| Feature | Status | Code Location | Description |
|---------|--------|---------------|-------------|
| **Depreciation (чл. 12)** | Complete | `EASTERN_COEFFICIENTS`, `WESTERN_COEFFICIENTS` | Coefficients 0.20-1.00 by vehicle age |
| **Vehicle Origin** | Complete | `EASTERN_MAKES`, `get_vehicle_origin()` | Eastern vs Western classification |
| **Special Adjustments** | Complete | `SPECIAL_MAKE_ADJUSTMENTS` | Peugeot, Opel, Citroen, Ford special rules |
| **Vehicle Classes (чл. 13)** | Complete | `get_vehicle_class()` | A (<4m), B (4-4.6m), C (>4.6m), D (SUV/Van) |
| **Replacement Labor (Глава III)** | Complete | `LABOR_NORMS_REPLACEMENT` | 40+ parts with hours per vehicle class |
| **Paint Labor (Глава IV)** | Complete | `PAINT_LABOR_NORMS` | Hours per panel per vehicle class |
| **Paint Materials** | Complete | `PAINT_MATERIALS_COST` | Standard, metallic, special effect costs |
| **Color Matching** | Complete | `COLOR_MATCHING_HOURS` | Fixed 1.0 hour |
| **Oven Drying** | Complete | `OVEN_DRYING_HOURS` | By panel count (1-3, 4-6, 7+) |
| **Work Type Rates** | Complete | `WORK_TYPE_RATES` | тенекеджийски: 50, бояджийски: 60 BGN/h |

---

## Total Repair Cost Formula

```
total_repair_cost = parts_after_depreciation + labor_cost + paint_cost

Where:
- parts_after_depreciation = scraped_price × depreciation_coefficient
- labor_cost = labor_hours × hourly_rate_by_work_type
- paint_cost = paint_labor + paint_materials + color_matching + oven_drying
```

---

## Potential Gaps (Need PO Decision)

### 1. Restoration/Repair Norms (възстановяване)

**Current state:** Only replacement norms implemented
**What's missing:** Naredba 24 also has restoration hours for *repairing* parts instead of replacing them

**Example:**
- Replacement of front fender: 3.0 hours (Class B)
- Restoration of front fender: 1.5-2.0 hours (estimated)

**Impact:** If a part can be repaired instead of replaced, we currently don't calculate repair labor
**Effort:** Medium - need to extract restoration tables from Naredba 24

---

### 2. Disassembly/Assembly Extras

**Current state:** Only base replacement time included
**What's missing:** Complex parts require additional disassembly time

**Example:**
- Dashboard replacement: 5.0 hours (base)
- Additional: remove steering wheel, airbags, wiring = +2-3 hours

**Impact:** Some complex repairs may underestimate labor
**Effort:** Medium - need to identify which parts require extra teardown

---

### 3. Consumables/Additional Materials

**Current state:** Only paint materials calculated
**What's missing:** Other consumables not included

**Examples:**
- Clips and fasteners
- Seals and gaskets
- Adhesives and sealants
- Welding materials

**Impact:** 5-10% of repair cost may be missing
**Effort:** Low - could add fixed percentage or itemized list

---

### 4. Regional Labor Rates

**Current state:** Fixed default rates (50-60 BGN/h)
**What's missing:** Rates vary by region and shop type

**Examples:**
- Sofia authorized dealer: 80-100 BGN/h
- Provincial independent shop: 40-50 BGN/h

**Impact:** Labor cost accuracy varies by location
**Effort:** Low - make rates configurable per request (already supported via API)

---

## Current Confidence Level

| Component | Accuracy | Source |
|-----------|----------|--------|
| Parts price | High | Real-time web scraping |
| Depreciation | Exact | Naredba 24 чл. 12 formula |
| Labor hours | Exact | Naredba 24 Глава III table |
| Paint cost | Exact | Naredba 24 Глава IV formula |

**Overall Naredba 24 Compliance: ~90%**

Main gap: Restoration (repair) norms vs replacement norms

---

## Recommendations for PO

1. **Priority 1:** Decide if restoration norms are needed (depends on claim types)
2. **Priority 2:** Consider adding consumables as fixed percentage (e.g., 5%)
3. **Priority 3:** Regional labor rates can be passed via API already - document this
4. **Priority 4:** Disassembly extras for complex parts (nice-to-have)

---

## API Endpoints

All Naredba 24 features accessible via:

```bash
# Get depreciation coefficient
GET /naredba24/coefficient/{make}/{year}?model=optional

# Get labor hours for part
GET /naredba24/labor/{part_name}?vehicle_class=B

# Calculate paint costs
GET /naredba24/paint?parts=броня,калник&vehicle_class=B&paint_type=metallic

# Full parts search with all Naredba 24 calculations
POST /parts/search
{
  "make": "BMW",
  "model": "320d",
  "year": 2018,
  "parts": ["front bumper", "headlight"],
  "include_depreciation": true,
  "include_labor": true,
  "include_paint": true
}
```
