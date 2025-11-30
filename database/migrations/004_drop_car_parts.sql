-- Migration 004: Drop car_parts table
-- Parts estimation removed - using LLM's estimated_damage from documents instead

DROP INDEX IF EXISTS idx_car_parts_lookup;
DROP TABLE IF EXISTS car_parts;
