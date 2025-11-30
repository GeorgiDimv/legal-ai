-- Migration 002: Price Aggregator Schema
-- Adds tables for scraped prices, price history, and parts

-- =============================================================================
-- Scraped Car Listings (raw data from scrapers)
-- =============================================================================
CREATE TABLE IF NOT EXISTS car_listings (
    id SERIAL PRIMARY KEY,
    source VARCHAR(50) NOT NULL,          -- 'mobile.bg', 'cars.bg', 'autoscout24.bg'
    external_id VARCHAR(100),              -- ID from source site
    make VARCHAR(100) NOT NULL,
    model VARCHAR(100) NOT NULL,
    variant VARCHAR(100),                  -- e.g., '320i', '2.0 TDI'
    year INT NOT NULL,
    price_bgn FLOAT NOT NULL,
    price_original FLOAT,                  -- Original price if in EUR
    currency VARCHAR(10) DEFAULT 'BGN',
    mileage_km INT,
    fuel_type VARCHAR(50),                 -- petrol, diesel, electric, hybrid
    transmission VARCHAR(50),              -- manual, automatic
    engine_cc INT,
    horsepower INT,
    color VARCHAR(50),
    location VARCHAR(100),                 -- City in Bulgaria
    listing_url TEXT,
    scraped_at TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_car_listings_make_model_year ON car_listings(LOWER(make), LOWER(model), year);
CREATE INDEX idx_car_listings_source ON car_listings(source);
CREATE INDEX idx_car_listings_scraped ON car_listings(scraped_at DESC);

-- =============================================================================
-- Aggregated Car Prices (computed from listings)
-- =============================================================================
CREATE TABLE IF NOT EXISTS car_prices_aggregated (
    id SERIAL PRIMARY KEY,
    make VARCHAR(100) NOT NULL,
    model VARCHAR(100) NOT NULL,
    variant VARCHAR(100),                  -- NULL means all variants
    year INT NOT NULL,
    avg_price_bgn FLOAT NOT NULL,
    min_price_bgn FLOAT,
    max_price_bgn FLOAT,
    median_price_bgn FLOAT,
    sample_count INT DEFAULT 0,
    sources TEXT[],                        -- Array of sources used
    confidence FLOAT DEFAULT 0.5,          -- 0-1 based on sample size
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(make, model, variant, year)
);

CREATE INDEX idx_car_prices_agg_lookup ON car_prices_aggregated(LOWER(make), LOWER(model), year);

-- =============================================================================
-- Price History (for tracking market trends)
-- =============================================================================
CREATE TABLE IF NOT EXISTS car_price_history (
    id SERIAL PRIMARY KEY,
    make VARCHAR(100) NOT NULL,
    model VARCHAR(100) NOT NULL,
    year INT NOT NULL,
    avg_price_bgn FLOAT NOT NULL,
    sample_count INT,
    recorded_at DATE DEFAULT CURRENT_DATE,
    UNIQUE(make, model, year, recorded_at)
);

CREATE INDEX idx_price_history_lookup ON car_price_history(make, model, year, recorded_at DESC);

-- =============================================================================
-- Car Parts Prices
-- =============================================================================
CREATE TABLE IF NOT EXISTS car_parts (
    id SERIAL PRIMARY KEY,
    make VARCHAR(100) NOT NULL,
    model VARCHAR(100),                    -- NULL for universal parts
    year_from INT,
    year_to INT,
    part_category VARCHAR(100) NOT NULL,  -- 'body', 'engine', 'suspension', 'electrical', 'interior'
    part_name VARCHAR(200) NOT NULL,
    part_name_bg VARCHAR(200),             -- Bulgarian name
    oem_price_bgn FLOAT,                   -- Original manufacturer price
    aftermarket_price_bgn FLOAT,           -- Aftermarket/generic price
    labor_hours FLOAT,                     -- Typical installation hours
    labor_rate_bgn FLOAT DEFAULT 50,       -- Hourly labor rate
    source VARCHAR(100),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_car_parts_lookup ON car_parts(LOWER(make), LOWER(model), part_category);

-- =============================================================================
-- Seed common car parts with typical prices
-- =============================================================================
INSERT INTO car_parts (make, model, part_category, part_name, part_name_bg, oem_price_bgn, aftermarket_price_bgn, labor_hours) VALUES
-- Universal parts (all makes)
(NULL, NULL, 'body', 'Front Bumper', 'Предна броня', 800, 350, 2.0),
(NULL, NULL, 'body', 'Rear Bumper', 'Задна броня', 700, 300, 1.5),
(NULL, NULL, 'body', 'Hood', 'Преден капак', 900, 400, 1.0),
(NULL, NULL, 'body', 'Front Fender Left', 'Ляв преден калник', 500, 200, 1.5),
(NULL, NULL, 'body', 'Front Fender Right', 'Десен преден калник', 500, 200, 1.5),
(NULL, NULL, 'body', 'Front Door Left', 'Лява предна врата', 1200, 500, 2.0),
(NULL, NULL, 'body', 'Front Door Right', 'Дясна предна врата', 1200, 500, 2.0),
(NULL, NULL, 'body', 'Rear Door Left', 'Лява задна врата', 1000, 450, 2.0),
(NULL, NULL, 'body', 'Rear Door Right', 'Дясна задна врата', 1000, 450, 2.0),
(NULL, NULL, 'body', 'Trunk Lid', 'Заден капак', 800, 350, 1.5),
(NULL, NULL, 'body', 'Side Mirror Left', 'Ляво странично огледало', 400, 150, 0.5),
(NULL, NULL, 'body', 'Side Mirror Right', 'Дясно странично огледало', 400, 150, 0.5),
(NULL, NULL, 'body', 'Windshield', 'Предно стъкло', 600, 300, 2.0),
(NULL, NULL, 'body', 'Rear Window', 'Задно стъкло', 400, 200, 1.5),
(NULL, NULL, 'electrical', 'Headlight Left', 'Ляв фар', 800, 300, 0.5),
(NULL, NULL, 'electrical', 'Headlight Right', 'Десен фар', 800, 300, 0.5),
(NULL, NULL, 'electrical', 'Taillight Left', 'Ляв стоп', 300, 120, 0.3),
(NULL, NULL, 'electrical', 'Taillight Right', 'Десен стоп', 300, 120, 0.3),
(NULL, NULL, 'electrical', 'Radiator', 'Радиатор', 500, 200, 2.0),
(NULL, NULL, 'electrical', 'AC Condenser', 'Климатичен кондензатор', 450, 180, 2.0),
(NULL, NULL, 'suspension', 'Front Shock Absorber', 'Преден амортисьор', 250, 100, 1.0),
(NULL, NULL, 'suspension', 'Rear Shock Absorber', 'Заден амортисьор', 200, 80, 1.0),
(NULL, NULL, 'suspension', 'Control Arm', 'Носач', 300, 120, 1.5),
(NULL, NULL, 'engine', 'Alternator', 'Алтернатор', 500, 200, 1.5),
(NULL, NULL, 'engine', 'Starter Motor', 'Стартер', 400, 150, 1.5),
(NULL, NULL, 'engine', 'Water Pump', 'Водна помпа', 200, 80, 2.0),
(NULL, NULL, 'interior', 'Front Seat', 'Предна седалка', 1000, 400, 1.0),
(NULL, NULL, 'interior', 'Steering Wheel', 'Волан', 600, 250, 1.0),
(NULL, NULL, 'interior', 'Dashboard', 'Табло', 1500, 600, 4.0),
-- BMW specific (premium pricing)
('BMW', NULL, 'body', 'Front Bumper', 'Предна броня', 1500, 600, 2.5),
('BMW', NULL, 'body', 'Hood', 'Преден капак', 1800, 700, 1.0),
('BMW', NULL, 'electrical', 'Headlight Left', 'Ляв фар', 2000, 800, 1.0),
('BMW', NULL, 'electrical', 'Headlight Right', 'Десен фар', 2000, 800, 1.0),
-- Mercedes specific (premium pricing)
('Mercedes-Benz', NULL, 'body', 'Front Bumper', 'Предна броня', 1600, 650, 2.5),
('Mercedes-Benz', NULL, 'body', 'Hood', 'Преден капак', 1900, 750, 1.0),
('Mercedes-Benz', NULL, 'electrical', 'Headlight Left', 'Ляв фар', 2200, 900, 1.0),
-- Volkswagen specific
('Volkswagen', NULL, 'body', 'Front Bumper', 'Предна броня', 900, 400, 2.0),
('Volkswagen', NULL, 'electrical', 'Headlight Left', 'Ляв фар', 1000, 400, 0.5);

-- =============================================================================
-- Scraper Job Log (track scraping runs)
-- =============================================================================
CREATE TABLE IF NOT EXISTS scraper_runs (
    id SERIAL PRIMARY KEY,
    source VARCHAR(50) NOT NULL,
    started_at TIMESTAMP DEFAULT NOW(),
    finished_at TIMESTAMP,
    listings_found INT DEFAULT 0,
    listings_new INT DEFAULT 0,
    listings_updated INT DEFAULT 0,
    status VARCHAR(20) DEFAULT 'running',  -- 'running', 'completed', 'failed'
    error_message TEXT
);

-- =============================================================================
-- Add more vehicle models to vehicle_prices
-- =============================================================================
INSERT INTO vehicle_prices (make, model, base_price_bgn, depreciation_per_year) VALUES
-- BMW variants
('BMW', '316i', 55000, 0.10),
('BMW', '318i', 60000, 0.10),
('BMW', '320i', 70000, 0.10),
('BMW', '320d', 72000, 0.10),
('BMW', '325i', 78000, 0.10),
('BMW', '330i', 85000, 0.10),
('BMW', '330d', 88000, 0.10),
('BMW', '520i', 85000, 0.10),
('BMW', '520d', 88000, 0.10),
('BMW', '525d', 95000, 0.10),
('BMW', '530d', 105000, 0.10),
('BMW', 'X1', 70000, 0.10),
('BMW', 'X6', 150000, 0.10),
-- Mercedes variants
('Mercedes-Benz', 'A-Class', 55000, 0.10),
('Mercedes-Benz', 'B-Class', 50000, 0.10),
('Mercedes-Benz', 'C180', 70000, 0.10),
('Mercedes-Benz', 'C200', 75000, 0.10),
('Mercedes-Benz', 'C220', 80000, 0.10),
('Mercedes-Benz', 'E200', 90000, 0.10),
('Mercedes-Benz', 'E220', 95000, 0.10),
('Mercedes-Benz', 'E350', 120000, 0.10),
('Mercedes-Benz', 'GLA', 75000, 0.10),
('Mercedes-Benz', 'GLB', 80000, 0.10),
('Mercedes-Benz', 'GLE', 130000, 0.10),
-- Audi variants
('Audi', 'A3', 55000, 0.09),
('Audi', 'A4 Avant', 75000, 0.09),
('Audi', 'A5', 80000, 0.09),
('Audi', 'A6 Avant', 95000, 0.09),
('Audi', 'A7', 110000, 0.09),
('Audi', 'A8', 150000, 0.09),
('Audi', 'Q3', 65000, 0.09),
('Audi', 'Q7', 120000, 0.09),
('Audi', 'Q8', 140000, 0.09),
-- Volkswagen variants
('Volkswagen', 'Golf GTI', 60000, 0.09),
('Volkswagen', 'Golf R', 75000, 0.09),
('Volkswagen', 'Passat Variant', 58000, 0.08),
('Volkswagen', 'Arteon', 70000, 0.09),
('Volkswagen', 'T-Roc', 55000, 0.08),
('Volkswagen', 'T-Cross', 45000, 0.08),
('Volkswagen', 'Touareg', 120000, 0.09),
('Volkswagen', 'ID.3', 65000, 0.07),
('Volkswagen', 'ID.4', 80000, 0.07),
-- Toyota variants
('Toyota', 'Camry', 65000, 0.07),
('Toyota', 'Avensis', 45000, 0.08),
('Toyota', 'Auris', 35000, 0.08),
('Toyota', 'C-HR', 55000, 0.07),
('Toyota', 'Land Cruiser', 140000, 0.06),
('Toyota', 'Prius', 50000, 0.08),
-- Honda
('Honda', 'Civic', 40000, 0.08),
('Honda', 'Accord', 55000, 0.08),
('Honda', 'CR-V', 60000, 0.07),
('Honda', 'HR-V', 45000, 0.08),
('Honda', 'Jazz', 30000, 0.08),
-- Mazda
('Mazda', '3', 42000, 0.08),
('Mazda', '6', 55000, 0.08),
('Mazda', 'CX-3', 45000, 0.08),
('Mazda', 'CX-5', 60000, 0.07),
('Mazda', 'CX-30', 52000, 0.08),
('Mazda', 'MX-5', 55000, 0.07),
-- Nissan
('Nissan', 'Qashqai', 50000, 0.09),
('Nissan', 'Juke', 38000, 0.09),
('Nissan', 'X-Trail', 58000, 0.09),
('Nissan', 'Micra', 25000, 0.09),
('Nissan', 'Leaf', 55000, 0.10),
-- Volvo
('Volvo', 'S60', 65000, 0.09),
('Volvo', 'S90', 95000, 0.09),
('Volvo', 'V60', 70000, 0.09),
('Volvo', 'V90', 100000, 0.09),
('Volvo', 'XC40', 75000, 0.08),
('Volvo', 'XC60', 95000, 0.08),
('Volvo', 'XC90', 130000, 0.08),
-- Citroen
('Citroen', 'C3', 28000, 0.10),
('Citroen', 'C4', 35000, 0.10),
('Citroen', 'C5 Aircross', 50000, 0.10),
-- Mitsubishi
('Mitsubishi', 'Outlander', 55000, 0.09),
('Mitsubishi', 'ASX', 42000, 0.09),
('Mitsubishi', 'Eclipse Cross', 48000, 0.09),
-- Subaru
('Subaru', 'Forester', 60000, 0.08),
('Subaru', 'Outback', 65000, 0.08),
('Subaru', 'XV', 52000, 0.08),
('Subaru', 'Impreza', 45000, 0.08)
ON CONFLICT DO NOTHING;
