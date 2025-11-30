"""
Pydantic schemas for API Gateway responses
"""

from typing import Optional, List
from datetime import date, time
from pydantic import BaseModel, Field


class LocationData(BaseModel):
    """Accident location with optional geocoding."""
    address: Optional[str] = None
    city: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class PartPricing(BaseModel):
    """Pricing data for a damaged part."""
    part_name: str
    part_name_bg: Optional[str] = None
    best_price_bgn: Optional[float] = None
    best_source: Optional[str] = None
    price_range_min_bgn: Optional[float] = None
    price_range_max_bgn: Optional[float] = None
    labor_cost_bgn: Optional[float] = None
    total_cost_bgn: Optional[float] = None


class PartsEstimate(BaseModel):
    """Parts pricing estimate from web search."""
    parts: List[PartPricing] = Field(default_factory=list)
    total_parts_cost_bgn: Optional[float] = None
    total_labor_cost_bgn: Optional[float] = None
    total_repair_cost_bgn: Optional[float] = None
    parts_found: int = 0
    parts_not_found: int = 0
    source: str = "web_search"


class VehicleData(BaseModel):
    """Vehicle information from claim document."""
    vin: Optional[str] = Field(None, description="17-character Vehicle Identification Number")
    registration: Optional[str] = None
    make: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    mass_kg: Optional[int] = None
    owner_name: Optional[str] = None
    insurance_company: Optional[str] = None
    policy_number: Optional[str] = None
    damage_description: Optional[str] = None
    damaged_parts: List[str] = Field(default_factory=list, description="List of damaged parts extracted from description")
    estimated_damage_bgn: Optional[float] = Field(None, description="LLM estimate from document")
    parts_estimate: Optional[PartsEstimate] = Field(None, description="Web-scraped parts pricing")
    current_market_value_bgn: Optional[float] = None
    market_value_source: Optional[str] = None
    # Physics data
    skid_distance_m: Optional[float] = None
    post_impact_travel_m: Optional[float] = None
    claimed_speed_kmh: Optional[float] = None
    pre_impact_angle_deg: Optional[float] = Field(None, description="α - direction before impact (0-360°)")
    post_impact_angle_deg: Optional[float] = Field(None, description="β - direction after impact (0-360°)")


class PartyData(BaseModel):
    """Party involved in the accident."""
    name: Optional[str] = None
    role: Optional[str] = Field(None, description="driver, passenger, pedestrian, witness")
    vehicle_index: Optional[int] = None
    injuries: Optional[str] = None
    statement_summary: Optional[str] = None


class FaultDetermination(BaseModel):
    """Fault analysis from the claim."""
    primary_fault_party: Optional[str] = None
    fault_percentage: Optional[int] = Field(None, ge=0, le=100)
    reasoning: Optional[str] = None
    traffic_violations: List[str] = Field(default_factory=list)


class PoliceReportData(BaseModel):
    """Police report information if available."""
    report_number: Optional[str] = None
    officer_name: Optional[str] = None
    findings: Optional[str] = None


class SettlementComponents(BaseModel):
    """Breakdown of settlement recommendation."""
    vehicle_damage: Optional[float] = None
    medical_expenses: Optional[float] = None
    lost_income: Optional[float] = None
    pain_and_suffering: Optional[float] = None


class SettlementRecommendation(BaseModel):
    """Settlement recommendation from LLM analysis."""
    amount_bgn: Optional[float] = None
    components: Optional[SettlementComponents] = None
    reasoning: Optional[str] = None


class PhysicsAnalysis(BaseModel):
    """Physics-based collision analysis from crash reconstruction."""
    # Speed validation (single vehicle)
    claimed_speed_valid: Optional[bool] = None
    calculated_speed_kmh: Optional[float] = None
    speed_validation_method: Optional[str] = None
    speed_validation_explanation: Optional[str] = None

    # Collision analysis (two vehicles)
    vehicle_a_post_impact_kmh: Optional[float] = None
    vehicle_b_post_impact_kmh: Optional[float] = None
    vehicle_a_pre_impact_kmh: Optional[float] = None
    vehicle_b_pre_impact_kmh: Optional[float] = None

    # Delta-V (change in velocity - key injury indicator)
    delta_v_a_kmh: Optional[float] = None
    delta_v_b_kmh: Optional[float] = None

    # Energy analysis
    total_impact_energy_j: Optional[float] = None
    energy_dissipated_j: Optional[float] = None
    estimated_damage_severity: Optional[str] = None

    # Method used
    physics_method: Optional[str] = Field(None, description="momentum_360 or impact_theory")

    # Analysis metadata
    physics_confidence: Optional[float] = None
    physics_notes: List[str] = Field(default_factory=list)


class ProcessingResult(BaseModel):
    """Complete processing result for an insurance claim document."""

    # Core claim data
    claim_number: Optional[str] = None
    accident_date: Optional[str] = None  # ISO format YYYY-MM-DD
    accident_time: Optional[str] = None  # HH:MM format

    # Location (enriched with geocoding)
    accident_location: Optional[LocationData] = None

    # Vehicles (enriched with market values)
    vehicles: List[VehicleData] = Field(default_factory=list)

    # Parties involved
    parties: List[PartyData] = Field(default_factory=list)

    # Analysis
    accident_description: Optional[str] = None
    fault_determination: Optional[FaultDetermination] = None
    police_report: Optional[PoliceReportData] = None
    settlement_recommendation: Optional[SettlementRecommendation] = None
    physics_analysis: Optional[PhysicsAnalysis] = None

    # Metadata
    risk_factors: List[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    processing_time_seconds: float = 0.0
    raw_ocr_text: Optional[str] = None

    # Error handling
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class LLMExtractionResult(BaseModel):
    """Raw extraction result from LLM before enrichment."""
    claim_number: Optional[str] = None
    accident_date: Optional[str] = None
    accident_time: Optional[str] = None
    accident_location: Optional[dict] = None
    vehicles: List[dict] = Field(default_factory=list)
    parties: List[dict] = Field(default_factory=list)
    accident_description: Optional[str] = None
    fault_determination: Optional[dict] = None
    police_report: Optional[dict] = None
    settlement_recommendation: Optional[dict] = None
    risk_factors: List[str] = Field(default_factory=list)
    confidence_score: float = 0.5
