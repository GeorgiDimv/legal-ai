# Physics Formulas for ATE Reports

## Current Implementation (15 Formulas)

All formulas are implemented in `/services/physics/main.py` and used by the ATE report generator.

### Section 5: Technical Analysis Formulas

| # | Section | Formula | Description |
|---|---------|---------|-------------|
| 1 | 5.1 Friction Coefficient | `F = μ × m × g` | Friction force calculation |
| 2 | 5.2 Post-Impact Velocity | `u = √(2×μ×g×σ)` | Speed from skid marks |
| 3 | 5.3 Vector Decomposition | `Vx = V×cos(α)`, `Vy = V×sin(α)` | Velocity components |
| 4 | 5.3 Resultant Velocity | `V = √(Vx² + Vy²)` | Combined velocity |
| 5 | 5.4 Momentum X-component | `m₁V₁cos(α₁) + m₂V₂cos(α₂) = m₁u₁cos(β₁) + m₂u₂cos(β₂)` | Conservation of momentum (X) |
| 6 | 5.4 Momentum Y-component | `m₁V₁sin(α₁) + m₂V₂sin(α₂) = m₁u₁sin(β₁) + m₂u₂sin(β₂)` | Conservation of momentum (Y) |
| 7 | 5.5 Impact Theory Matrix | `[a₁₁ a₁₂; a₂₁ a₂₂] × [V₁; V₂] = [b₁; b₂]` | Matrix solution for velocities |
| 8 | 5.6 Delta-V | `ΔV = √(V² + u² - 2×V×u×cos(β - α))` | Change in velocity at impact |
| 9 | 5.7 Dangerous Zone | `Sоз = V×(tr + tsp + 0.5×tn) + V²/(2×j)` | Stopping distance from reaction |
| 10 | 5.8 Stopping Time | `Tу = tr + tsp + 0.5×tn + (V - Vy)/j` | Total time to stop |
| 11 | 5.9 Safe Speed | `Vбезоп` (quadratic solution) | Speed to avoid collision |
| 12 | 5.12 Restitution Coefficient | `k = (u₂ - u₁) / (V₁ - V₂)` | Elasticity of collision |
| 13 | 5.13 Kinetic Energy | `E = ½ × m × V²` | Energy before/after impact |
| 14 | 5.14 Dissipated Energy | `ΔE = E₁ - E₂` | Energy absorbed in deformation |
| 15 | 5.15 Work of Braking | `W = μ × m × g × s` | Work-energy theorem |

### Restitution Coefficient Table (from Uchebnik ATE)

| Impact Speed (km/h) | k value |
|---------------------|---------|
| 12-16 | 0.24 |
| 23-25 | 0.16 |
| 40-41 | 0.13 |
| 46-49 | 0.14 |
| 55-57 | 0.12 |

### Friction Coefficients

| Surface | μ (dry) | μ (wet) |
|---------|---------|---------|
| Asphalt | 0.7 | 0.5 |
| Concrete | 0.75 | 0.55 |
| Gravel | 0.4 | 0.35 |
| Snow | 0.2 | 0.2 |
| Ice | 0.1 | 0.05 |

---

## Future Implementation (2 Formulas)

These formulas require additional data extraction from documents.

### 1. Pedestrian Throw Distance

**Use case:** Pedestrian accidents only (~5-10% of cases)

**Formula:**
```
V = (S / 0.0453)^(1/1.47)  (empirical, pedestrian on hood)

Or projectile motion:
S = V² × sin(2θ) / (2×g) + h
```

**Required data:**
| Parameter | Source | Extraction Pattern |
|-----------|--------|-------------------|
| throw_distance_m | Police protocol | "отхвърлен на X м", "хвърлен на разстояние X" |
| pedestrian_mass_kg | Optional (default 70) | "тегло X кг" |
| impact_height_m | Vehicle type | bumper height lookup |

**Implementation steps:**
1. Add to `extractors.py`:
   ```python
   "throw_distance_m": "Разстояние на отхвърляне на пешеходец в метри"
   ```
2. Add endpoint to `physics/main.py`:
   ```python
   @app.post("/pedestrian-throw")
   async def pedestrian_throw(throw_distance_m: float, mass_kg: float = 70):
       V = (throw_distance_m / 0.0453) ** (1/1.47)
       return {"velocity_kmh": V * 3.6}
   ```
3. Update gateway to call when `throw_distance_m` is present
4. Add formula to report prompt section 5.16

---

### 2. EES (Energy Equivalent Speed) from Deformation

**Use case:** When crush/deformation measurements are available (~10-20% of cases)

**Formula:**
```
EES = √(2 × E_deform / m)

Where E_deform = (A × C + B × C²/2) × L
- A, B = vehicle stiffness coefficients
- C = crush depth (m)
- L = crush width (m)
```

**Stiffness coefficients (CRASH3 database):**
| Vehicle Class | A (kN/m) | B (kN/m²) |
|---------------|----------|-----------|
| Small (Fiat 500, Smart) | 40-60 | 400-600 |
| Medium (Golf, Focus) | 60-80 | 600-800 |
| Large (BMW 5, Audi A6) | 80-100 | 800-1000 |
| SUV/Truck (X5, Land Cruiser) | 100-150 | 1000-1500 |

**Required data:**
| Parameter | Source | Extraction Pattern |
|-----------|--------|-------------------|
| crush_depth_cm | Inspection report | "деформация X см", "смачкване X см" |
| crush_width_m | Vehicle width or measured | "ширина на деформация X м" |
| vehicle_class | From make/model | Lookup table |

**Implementation steps:**
1. Add to `extractors.py`:
   ```python
   "crush_depth_cm": "Дълбочина на деформация в см",
   "crush_width_m": "Ширина на деформирана зона в метри"
   ```
2. Add stiffness lookup to `car_value/main.py` or separate config
3. Add endpoint to `physics/main.py`:
   ```python
   @app.post("/ees-deformation")
   async def ees_deformation(
       crush_depth_m: float,
       crush_width_m: float,
       vehicle_mass_kg: float,
       stiffness_A: float = 70,
       stiffness_B: float = 700
   ):
       E = (stiffness_A * crush_depth_m + stiffness_B * crush_depth_m**2 / 2) * crush_width_m * 1000
       EES = math.sqrt(2 * E / vehicle_mass_kg)
       return {"ees_ms": EES, "ees_kmh": EES * 3.6, "energy_joules": E}
   ```
4. Update gateway to call when `crush_depth_cm` is present
5. Add formula to report prompt section 5.17

---

## Physics Service Endpoints

### Current Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/momentum-360` | POST | Full vector momentum analysis |
| `/impact-theory` | POST | Matrix method solution |
| `/dangerous-zone` | POST | Stopping distance calculation |
| `/velocity-from-skid` | POST | Speed from skid marks |
| `/validate-claimed-speed` | POST | Check if claimed speed matches evidence |
| `/formulas` | GET | List all formulas |

### Future Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/pedestrian-throw` | POST | Speed from pedestrian throw distance |
| `/ees-deformation` | POST | EES from crush depth |

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     DOCUMENT INPUT                               │
│  "BMW удари пешеходец, отхвърлен на 12м, деформация 25 см"      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LLM EXTRACTION                               │
│  {                                                               │
│    "vehicles": [...],                                            │
│    "throw_distance_m": 12,      ← NEW (for pedestrian)          │
│    "crush_depth_cm": 25,        ← NEW (for EES)                 │
│    "skid_distance_m": 8                                          │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PHYSICS SERVICE                              │
│                                                                  │
│  if throw_distance_m:                                            │
│      → /pedestrian-throw → V = 48 km/h                          │
│                                                                  │
│  if crush_depth_cm:                                              │
│      → /ees-deformation → EES = 42 km/h                         │
│                                                                  │
│  if skid_distance_m:                                             │
│      → /velocity-from-skid → u = 35 km/h                        │
│                                                                  │
│  always:                                                         │
│      → /momentum-360 → V₁, V₂, ΔV                               │
│      → /dangerous-zone → Sоз, Tу, Vбезоп                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ATE REPORT                                   │
│  Section 5: All 15-17 formulas with calculations                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Priority

| Priority | Formula | Effort | Benefit |
|----------|---------|--------|---------|
| ✅ Done | 15 core formulas | - | Works now |
| Medium | Pedestrian throw | 2-3 hours | 5-10% of cases |
| Low | EES deformation | 3-4 hours | 10-20% of cases |

Implement pedestrian/EES when you have documents that contain this data.
