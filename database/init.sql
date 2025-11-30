-- Legal AI Document Processing Pipeline - Database Schema
-- PostgreSQL 15

-- =============================================================================
-- Vehicle Specifications Table
-- =============================================================================
CREATE TABLE vehicle_specs (
    id SERIAL PRIMARY KEY,
    make VARCHAR(100) NOT NULL,
    model VARCHAR(100) NOT NULL,
    year_from INT,
    year_to INT,
    weight_kg INT,
    length_mm INT,
    width_mm INT,
    height_mm INT,
    engine_cc INT,
    horsepower INT,
    body_type VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_vehicle_specs_make_model ON vehicle_specs(make, model);

-- =============================================================================
-- Vehicle Prices Table (fallback for Mobile.bg when scraping fails)
-- =============================================================================
CREATE TABLE vehicle_prices (
    id SERIAL PRIMARY KEY,
    make VARCHAR(100) NOT NULL,
    model VARCHAR(100) NOT NULL,
    base_price_bgn FLOAT NOT NULL,            -- Price when new (in Bulgarian Lev)
    depreciation_per_year FLOAT DEFAULT 0.08, -- 8% per year depreciation
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_vehicle_prices_make_model ON vehicle_prices(LOWER(make), LOWER(model));

-- =============================================================================
-- Processed Claims Log
-- =============================================================================
CREATE TABLE processed_claims (
    id SERIAL PRIMARY KEY,
    claim_number VARCHAR(100),
    filename VARCHAR(255),
    processing_time_ms INT,
    confidence_score FLOAT,
    fault_percentage INT,
    settlement_amount_bgn FLOAT,
    result_json JSONB,
    raw_ocr_text TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_processed_claims_created ON processed_claims(created_at DESC);
CREATE INDEX idx_processed_claims_claim_number ON processed_claims(claim_number);

-- =============================================================================
-- Seed Data: Common Bulgarian Vehicle Specifications
-- =============================================================================
INSERT INTO vehicle_specs (make, model, year_from, year_to, weight_kg, body_type) VALUES
('Volkswagen', 'Golf', 2012, 2024, 1300, 'hatchback'),
('Volkswagen', 'Passat', 2010, 2024, 1500, 'sedan'),
('Volkswagen', 'Polo', 2010, 2024, 1100, 'hatchback'),
('Volkswagen', 'Tiguan', 2016, 2024, 1600, 'suv'),
('BMW', '3 Series', 2010, 2024, 1500, 'sedan'),
('BMW', '5 Series', 2010, 2024, 1700, 'sedan'),
('BMW', 'X3', 2010, 2024, 1800, 'suv'),
('BMW', 'X5', 2010, 2024, 2200, 'suv'),
('Mercedes-Benz', 'C-Class', 2010, 2024, 1550, 'sedan'),
('Mercedes-Benz', 'E-Class', 2010, 2024, 1750, 'sedan'),
('Mercedes-Benz', 'GLC', 2015, 2024, 1850, 'suv'),
('Opel', 'Astra', 2010, 2024, 1300, 'hatchback'),
('Opel', 'Corsa', 2010, 2024, 1100, 'hatchback'),
('Opel', 'Insignia', 2010, 2024, 1500, 'sedan'),
('Renault', 'Megane', 2010, 2024, 1350, 'hatchback'),
('Renault', 'Clio', 2010, 2024, 1100, 'hatchback'),
('Toyota', 'Corolla', 2010, 2024, 1350, 'sedan'),
('Toyota', 'Yaris', 2010, 2024, 1050, 'hatchback'),
('Toyota', 'RAV4', 2010, 2024, 1700, 'suv'),
('Ford', 'Focus', 2010, 2024, 1350, 'hatchback'),
('Ford', 'Fiesta', 2010, 2024, 1100, 'hatchback'),
('Audi', 'A4', 2010, 2024, 1500, 'sedan'),
('Audi', 'A6', 2010, 2024, 1700, 'sedan'),
('Audi', 'Q5', 2010, 2024, 1900, 'suv'),
('Skoda', 'Octavia', 2010, 2024, 1400, 'sedan'),
('Skoda', 'Superb', 2010, 2024, 1600, 'sedan'),
('Skoda', 'Fabia', 2010, 2024, 1100, 'hatchback'),
('Peugeot', '308', 2010, 2024, 1300, 'hatchback'),
('Peugeot', '208', 2012, 2024, 1100, 'hatchback'),
('Hyundai', 'i30', 2010, 2024, 1350, 'hatchback'),
('Hyundai', 'Tucson', 2015, 2024, 1600, 'suv'),
('Kia', 'Ceed', 2010, 2024, 1350, 'hatchback'),
('Kia', 'Sportage', 2010, 2024, 1600, 'suv'),
('Dacia', 'Duster', 2010, 2024, 1300, 'suv'),
('Dacia', 'Sandero', 2010, 2024, 1050, 'hatchback'),
('Dacia', 'Logan', 2010, 2024, 1100, 'sedan'),
('Fiat', 'Punto', 2010, 2020, 1100, 'hatchback'),
('Fiat', '500', 2010, 2024, 940, 'hatchback'),
('Seat', 'Leon', 2010, 2024, 1350, 'hatchback'),
('Seat', 'Ibiza', 2010, 2024, 1100, 'hatchback');

-- =============================================================================
-- Seed Data: Vehicle Prices (fallback estimates in BGN when new)
-- =============================================================================
INSERT INTO vehicle_prices (make, model, base_price_bgn, depreciation_per_year) VALUES
-- Volkswagen
('Volkswagen', 'Golf', 45000, 0.08),
('Volkswagen', 'Passat', 55000, 0.08),
('Volkswagen', 'Polo', 32000, 0.09),
('Volkswagen', 'Tiguan', 65000, 0.08),
-- BMW
('BMW', '3 Series', 75000, 0.10),
('BMW', '5 Series', 95000, 0.10),
('BMW', 'X3', 90000, 0.10),
('BMW', 'X5', 130000, 0.10),
-- Mercedes-Benz
('Mercedes-Benz', 'C-Class', 80000, 0.10),
('Mercedes-Benz', 'E-Class', 100000, 0.10),
('Mercedes-Benz', 'GLC', 95000, 0.10),
-- Opel
('Opel', 'Astra', 35000, 0.09),
('Opel', 'Corsa', 28000, 0.09),
('Opel', 'Insignia', 45000, 0.09),
-- Renault
('Renault', 'Megane', 32000, 0.09),
('Renault', 'Clio', 25000, 0.09),
-- Toyota
('Toyota', 'Corolla', 42000, 0.07),
('Toyota', 'Yaris', 28000, 0.08),
('Toyota', 'RAV4', 60000, 0.07),
-- Ford
('Ford', 'Focus', 38000, 0.09),
('Ford', 'Fiesta', 28000, 0.09),
-- Audi
('Audi', 'A4', 70000, 0.09),
('Audi', 'A6', 90000, 0.09),
('Audi', 'Q5', 85000, 0.09),
-- Skoda
('Skoda', 'Octavia', 40000, 0.08),
('Skoda', 'Superb', 55000, 0.08),
('Skoda', 'Fabia', 28000, 0.08),
-- Peugeot
('Peugeot', '308', 35000, 0.10),
('Peugeot', '208', 28000, 0.10),
-- Hyundai
('Hyundai', 'i30', 32000, 0.09),
('Hyundai', 'Tucson', 50000, 0.09),
-- Kia
('Kia', 'Ceed', 30000, 0.09),
('Kia', 'Sportage', 48000, 0.09),
-- Dacia
('Dacia', 'Duster', 28000, 0.08),
('Dacia', 'Sandero', 18000, 0.08),
('Dacia', 'Logan', 20000, 0.08),
-- Fiat
('Fiat', 'Punto', 22000, 0.10),
('Fiat', '500', 25000, 0.09),
-- Seat
('Seat', 'Leon', 38000, 0.09),
('Seat', 'Ibiza', 28000, 0.09);

-- =============================================================================
-- Grant permissions
-- =============================================================================
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO legal_ai;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO legal_ai;
