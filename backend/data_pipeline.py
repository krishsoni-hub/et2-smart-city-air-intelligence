import asyncio
import aiohttp
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import ee
import logging
import datetime
from datetime import timedelta
import warnings

warnings.filterwarnings("ignore")

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# ==============================================================================
# 1. CAAQMS IoT DATA INGESTION ENGINE
# ==============================================================================


async def fetch_caaqms_data(session, url, retries=6):
    """
    Fetch CAAQMS data with exponential backoff (2s to 64s).
    Handles network drops and API rate limits (HTTP 429).
    """
    delay = 2
    for attempt in range(retries):
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    logging.warning(f"Rate limit hit. Retrying in {delay}s...")
                else:
                    logging.error(f"Failed with status {response.status}")
        except aiohttp.ClientError as e:
            logging.error(f"Request failed: {e}")

        await asyncio.sleep(delay)
        delay *= 2
    return None


def clean_and_impute_caaqms(df):
    """
    Clean and impute IoT CAAQMS data.
    - Flags outliers (e.g. PM2.5 > 999 or < 0) as NaN.
    - Linearly interpolates gaps < 3h.
    - Applies a rolling 7-day time-of-day mean window for gaps >= 3h.
    """
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()

    # Outlier Mitigation
    for col in ["pm25", "pm10", "no2", "so2", "co", "o3"]:
        if col in df.columns:
            # Mask out physical sensor malfunctions
            mask = (df[col] < 0) | (df[col] > 999)
            df.loc[mask, col] = np.nan

    # Temporal interpolation for gaps < 3h (Assume hourly data for simplicity -> limit=3)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="time", limit=3)

    # Resolve longer gaps using rolling 7-day time-of-day mean
    df["time_of_day"] = df.index.time

    for col in ["pm25", "pm10", "no2", "so2", "co", "o3"]:
        if col in df.columns:
            # Fill remaining NaNs using rolling 7-day time-of-day mean
            rolling_tod_mean = df.groupby("time_of_day")[col].transform(
                lambda x: x.rolling(window=7, min_periods=1).mean()
            )
            df[col] = df[col].fillna(rolling_tod_mean)
            # Fallback to global mean if still NaN
            df[col] = df[col].fillna(df[col].mean())

    df = df.drop(columns=["time_of_day"]).reset_index()
    return df


# ==============================================================================
# 2. SATELLITE IMAGERY & REMOTE SENSING PROCESSING ENGINE
# ==============================================================================


def init_earth_engine():
    try:
        ee.Initialize()
        return True
    except Exception:
        logging.warning("Earth Engine not initialized automatically, using mock logic.")
        return False


def get_satellite_data(start_date, end_date):
    """
    Extract daily raster layers for Sentinel-5P (NO2, CO) and Sentinel-2 (Level-2A MSI).
    Uses ee API if authenticated, else falls back to localized mock data.
    """
    if not init_earth_engine():
        return mock_satellite_data(start_date, end_date)

    # Delhi NCR Hardcoded Bounding Box
    delhi_bbox = ee.Geometry.Rectangle([76.83, 28.40, 77.34, 28.88])

    # Sentinel-5P (NO2 column number density)
    s5p_no2 = (
        ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_NO2")
        .filterBounds(delhi_bbox)
        .filterDate(start_date, end_date)
        .select("NO2_column_number_density")
        .mean()
    )

    # Sentinel-2 QA Mask / Cloud Mask logic based on QA60
    def mask_s2_clouds(image):
        qa = image.select("QA60")
        cloudBitMask = 1 << 10
        cirrusBitMask = 1 << 11
        mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
        return image.updateMask(mask).divide(10000)

    # Sentinel-2 Tracking thermal anomalies (proxied via filtering over bounding box)
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(delhi_bbox)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .map(mask_s2_clouds)
        .median()
    )

    # In a fully integrated pipeline we would download or sample this rasters over the grid.
    # We fallback to generating a structured mock to ensure end-to-end execution loop.
    return mock_satellite_data(start_date, end_date)


def mock_satellite_data(start_date, end_date):
    """Mock localized raster extraction over Delhi grid."""
    np.random.seed(42)
    latitudes = np.linspace(28.40, 28.88, 50)
    longitudes = np.linspace(76.83, 77.34, 50)

    data = []
    for lat in latitudes:
        for lon in longitudes:
            data.append(
                {
                    "latitude": lat,
                    "longitude": lon,
                    "sat_no2": np.random.uniform(0.00001, 0.00005),
                    "thermal_flag": np.random.choice([0, 1], p=[0.95, 0.05]),
                }
            )
    return pd.DataFrame(data)


# ==============================================================================
# 3. TRAFFIC CONGESTION & METEOROLOGICAL AGGREGATOR
# ==============================================================================


def fetch_traffic_and_meteo(timestamp):
    """
    Simulates automated ingestion module (15-minute cron job).
    Pulls traffic mobility matrices and meteorological parameters.
    """
    np.random.seed(int(timestamp.timestamp()))

    # Traffic Vectors (Mocking segment vectors)
    traffic_data = {
        "segment_id": [f"seg_{i}" for i in range(100)],
        "current_speed": np.random.uniform(10, 60, 100),
        "free_flow_speed": np.random.uniform(40, 60, 100),
    }
    traffic_df = pd.DataFrame(traffic_data)
    traffic_df["congestion_index"] = (
        traffic_df["free_flow_speed"] / traffic_df["current_speed"]
    )

    # Meteorological Parameters
    meteo_data = {
        "temperature": np.random.uniform(15, 45),
        "wind_speed": np.random.uniform(0, 10),  # U
        "wind_direction": np.random.uniform(0, 360),  # \theta
        "relative_humidity": np.random.uniform(20, 90),
        "pblh": np.random.uniform(500, 2000),  # Planetary Boundary Layer Height
    }

    return traffic_df, meteo_data


# ==============================================================================
# 4. SPATIAL-TEMPORAL GRID FUSION ENGINE
# ==============================================================================


def generate_master_grid(bbox, grid_size_km=1):
    """Generate 1x1 km master grid over bounding box using geopandas/shapely."""
    lon_min, lat_min, lon_max, lat_max = bbox

    # 1 degree lat ~ 111 km
    lat_step = grid_size_km / 111.0
    mean_lat = (lat_min + lat_max) / 2
    # 1 degree lon ~ 111 * cos(lat) km
    lon_step = grid_size_km / (111.0 * np.cos(np.radians(mean_lat)))

    grid_cells = []
    grid_ids = []
    r_id = 0

    lat = lat_min
    while lat < lat_max:
        lon = lon_min
        while lon < lon_max:
            cell = Polygon(
                [
                    (lon, lat),
                    (lon + lon_step, lat),
                    (lon + lon_step, lat + lat_step),
                    (lon, lat + lat_step),
                ]
            )
            grid_cells.append(cell)
            grid_ids.append(f"G_{r_id:05d}")
            r_id += 1
            lon += lon_step
        lat += lat_step

    grid_gdf = gpd.GeoDataFrame(
        {"grid_id": grid_ids}, geometry=grid_cells, crs="EPSG:4326"
    )
    grid_gdf["centroid"] = grid_gdf.geometry.centroid
    grid_gdf["latitude"] = grid_gdf.centroid.y
    grid_gdf["longitude"] = grid_gdf.centroid.x
    return grid_gdf


def spatial_temporal_fusion(caaqms_df, sat_df, traffic_df, meteo_data, master_grid):
    """Perform spatial joins (gpd.sjoin) to fuse IoT, remote sensing, and vectors onto master grid."""

    # 1. Project CAAQMS onto Grid
    if not caaqms_df.empty:
        caaqms_gdf = gpd.GeoDataFrame(
            caaqms_df,
            geometry=gpd.points_from_xy(caaqms_df.longitude, caaqms_df.latitude),
            crs="EPSG:4326",
        )
        grid_caaqms = gpd.sjoin_nearest(master_grid, caaqms_gdf, how="left")
    else:
        grid_caaqms = master_grid.copy()
        for col in ["pm25", "pm10", "no2"]:
            grid_caaqms[col] = np.nan

    # 2. Project Satellite Raster Pixels onto Grid
    if not sat_df.empty:
        sat_gdf = gpd.GeoDataFrame(
            sat_df,
            geometry=gpd.points_from_xy(sat_df.longitude, sat_df.latitude),
            crs="EPSG:4326",
        )
        grid_sat = gpd.sjoin_nearest(
            master_grid[["grid_id", "geometry"]], sat_gdf, how="left"
        )
        grid_sat = grid_sat.groupby("grid_id").mean(numeric_only=True).reset_index()
    else:
        grid_sat = pd.DataFrame(
            {"grid_id": master_grid["grid_id"], "sat_no2": np.nan, "thermal_flag": 0}
        )

    # Merge datasets on grid_id
    fused_df = pd.merge(
        master_grid[["grid_id", "latitude", "longitude"]],
        grid_caaqms[["grid_id", "pm25", "pm10", "no2"]],
        on="grid_id",
        how="left",
    )
    fused_df = pd.merge(
        fused_df,
        grid_sat[["grid_id", "sat_no2", "thermal_flag"]],
        on="grid_id",
        how="left",
    )

    # 3. Add Traffic (Aggregate traffic vectors onto grid)
    # Using uniform representation due to lack of distinct road geometries in mock,
    # but practically we would sjoin segment polylines to the 1x1 polygon grid.
    fused_df["traffic_index"] = traffic_df["congestion_index"].mean()

    # 4. Add Meteorological
    fused_df["wind_speed"] = meteo_data["wind_speed"]
    fused_df["wind_direction"] = meteo_data["wind_direction"]
    fused_df["pblh"] = meteo_data["pblh"]

    # Land Use Spatial Attribute
    fused_df["land_use_type"] = np.random.choice(
        ["residential", "commercial", "industrial", "forest"], size=len(fused_df)
    )

    # Deduplicate generated by spatial join 1:N matches
    fused_df = fused_df.groupby("grid_id").first().reset_index()

    return fused_df


def feature_engineering(history_df, current_fused_df):
    """
    Compute lag features (1h, 3h, 6h, 24h) and rolling stats for PM2.5, traffic density, wind.
    Returns fully formatted model-ready dataframe.
    """
    df = pd.concat([history_df, current_fused_df])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by=["grid_id", "timestamp"])

    # Lags (Assuming 1 interval = 15 min here: 4 = 1hr, 12 = 3hr, 96 = 24hr)
    df["pm25_lag_1h"] = df.groupby("grid_id")["pm25"].shift(4)
    df["pm25_lag_3h"] = df.groupby("grid_id")["pm25"].shift(12)
    df["pm25_lag_24h"] = df.groupby("grid_id")["pm25"].shift(96)

    # Rolling stats
    df["rolling_mean_6h"] = df.groupby("grid_id")["pm25"].transform(
        lambda x: x.rolling(window=24, min_periods=1).mean()
    )
    df["rolling_std_6h"] = df.groupby("grid_id")["pm25"].transform(
        lambda x: x.rolling(window=24, min_periods=1).std()
    )

    # Extract only the current timestep we want to predict on
    latest_timestamp = current_fused_df["timestamp"].max()
    final_df = df[df["timestamp"] == latest_timestamp].copy()

    # Backfill NaN lags for newly initialized pipelines
    final_df.bfill(inplace=True)
    final_df.ffill(inplace=True)

    return final_df


# ==============================================================================
# 5. DATA PERSISTENCE & OUTPUT SPECIFICATION
# ==============================================================================


async def run_pipeline():
    """Main execution loop returning X_train dataframe."""
    logging.info("Initializing Spatial-Temporal Grid Pipeline...")

    # Grid specification: Delhi NCR approx box
    delhi_bbox = (76.83, 28.40, 77.34, 28.88)
    master_grid = generate_master_grid(delhi_bbox, grid_size_km=1)

    # 1. Ingest CAAQMS Data
    async with aiohttp.ClientSession() as session:
        api_url = "https://api.openaq.org/v2/measurements"  # Mock endpoint
        # Simulated payload return for standard IoT schema
        np.random.seed(42)
        current_time = datetime.datetime.now()

        caaqms_raw = pd.DataFrame(
            {
                "timestamp": [current_time] * 20,
                "station_id": [f"CPCB_{i}" for i in range(20)],
                "latitude": np.random.uniform(28.40, 28.88, 20),
                "longitude": np.random.uniform(76.83, 77.34, 20),
                "pm25": np.random.uniform(10, 1100, 20),  # Included outlier (>999)
                "pm10": np.random.uniform(30, 400, 20),
                "no2": np.random.uniform(10, 80, 20),
            }
        )
        caaqms_cleaned = clean_and_impute_caaqms(caaqms_raw)

    # 2. Ingest Satellite
    end_date = current_time.strftime("%Y-%m-%d")
    start_date = (current_time - timedelta(days=1)).strftime("%Y-%m-%d")
    sat_data = get_satellite_data(start_date, end_date)

    # 3. Traffic & Meteo
    traffic_data, meteo_data = fetch_traffic_and_meteo(current_time)

    # 4. Fusion Engine
    logging.info("Executing Spatial Joins...")
    current_fused = spatial_temporal_fusion(
        caaqms_cleaned, sat_data, traffic_data, meteo_data, master_grid
    )
    current_fused["timestamp"] = current_time

    # Generating 24h mock historical data to compute proper lags in feature engineering
    logging.info("Computing Feature Engineering Lags...")
    history_dfs = []
    for i in range(1, 100):
        h = current_fused.copy()
        h["timestamp"] = current_time - timedelta(minutes=15 * i)
        history_dfs.append(h)
    history_df = pd.concat(history_dfs)

    X_train = feature_engineering(history_df, current_fused)

    # Format and Enforce Schema Specification
    expected_columns = [
        "timestamp",
        "grid_id",
        "latitude",
        "longitude",
        "pm25",
        "pm10",
        "no2",
        "traffic_index",
        "thermal_flag",
        "wind_speed",
        "wind_direction",
        "pblh",
        "pm25_lag_1h",
        "pm25_lag_3h",
        "pm25_lag_24h",
        "rolling_mean_6h",
        "land_use_type",
    ]

    # Missing columns will be added as 0 (for robustness)
    for col in expected_columns:
        if col not in X_train.columns:
            X_train[col] = 0.0

    X_train = X_train[expected_columns]
    X_train = X_train.fillna(0)  # Final safety check ensuring non-null

    logging.info(f"Pipeline Complete. X_train generated with shape: {X_train.shape}")
    return X_train


if __name__ == "__main__":
    # Event loop execution
    X_train_final = asyncio.run(run_pipeline())
    print("\n[Output Spec: X_train Sample]")
    print(X_train_final.head())
