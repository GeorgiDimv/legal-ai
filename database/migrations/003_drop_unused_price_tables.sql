-- Migration 003: Drop Unused Price Tables
-- Part of car_value service simplification (v3.0.0)
-- Removes DB storage tables in favor of on-demand scraping with Redis cache

-- Drop indices first
DROP INDEX IF EXISTS idx_car_listings_make_model_year;
DROP INDEX IF EXISTS idx_car_listings_source;
DROP INDEX IF EXISTS idx_car_listings_scraped;
DROP INDEX IF EXISTS idx_car_prices_agg_lookup;
DROP INDEX IF EXISTS idx_price_history_lookup;

-- Drop unused tables
DROP TABLE IF EXISTS car_listings;           -- Raw scraped listings (moved to on-demand + Redis)
DROP TABLE IF EXISTS car_prices_aggregated;  -- Pre-computed averages (now computed on-demand)
DROP TABLE IF EXISTS car_price_history;      -- Historical trends (no longer tracked)
DROP TABLE IF EXISTS scraper_runs;           -- Scraper job log (no more scheduled scraping)
DROP TABLE IF EXISTS vehicle_prices;         -- Static fallback prices (no longer used)

-- Keep car_parts table for damage estimation
-- (defined in 002_price_aggregator.sql)
