-- =================================================================================
-- TimescaleDB SQL Schema: Smart City Air Intelligence Data Persistence
-- =================================================================================

-- Enable TimescaleDB Extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create core hyper-table schema for time-series fused data
CREATE TABLE IF NOT EXISTS grid_air_quality (
    timestamp TIMESTAMPTZ NOT NULL,
    grid_id VARCHAR(50) NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    
    -- IoT Metrics
    pm25 DOUBLE PRECISION,
    pm10 DOUBLE PRECISION,
    no2 DOUBLE PRECISION,
    
    -- Spatial & Traffic Features
    traffic_index DOUBLE PRECISION,
    thermal_flag INTEGER,
    
    -- Meteorological Parameters
    wind_speed DOUBLE PRECISION,
    wind_direction DOUBLE PRECISION,
    pblh DOUBLE PRECISION,
    
    -- Engineered Lags & Rolling Features
    pm25_lag_1h DOUBLE PRECISION,
    pm25_lag_3h DOUBLE PRECISION,
    pm25_lag_24h DOUBLE PRECISION,
    rolling_mean_6h DOUBLE PRECISION,
    
    -- Static/Categorical Features
    land_use_type VARCHAR(50)
);

-- Convert standard table to TimescaleDB hypertable partitioned on 'timestamp'
SELECT create_hypertable('grid_air_quality', 'timestamp', if_not_exists => TRUE);

-- =================================================================================
-- Optimized Indexing
-- =================================================================================

-- Composite index for fast lookup of a specific grid over time
CREATE INDEX IF NOT EXISTS ix_grid_time ON grid_air_quality (grid_id, timestamp DESC);

-- Spatial index for bounding-box/geometry based queries
CREATE INDEX IF NOT EXISTS ix_spatial ON grid_air_quality (latitude, longitude);

-- Note: We can add PostGIS geometry columns if further spatial analysis is needed at the DB layer, 
-- e.g. ADD COLUMN geom geometry(Point, 4326)
