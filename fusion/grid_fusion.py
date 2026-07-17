import logging
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
from typing import Dict, Any, List
import numpy as np

logger = logging.getLogger("Spatial-Temporal-Fusion")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class GridFusionEngine:
    """
    Fuses multi-modal data streams (IoT CAAQMS, Satellite Rasters, Traffic, Weather)
    into a standardized 1km x 1km vector grid for the AI Engine.
    """
    def __init__(self, bbox: tuple, resolution: float = 0.01):
        self.bbox = bbox # (min_lon, min_lat, max_lon, max_lat)
        self.resolution = resolution
        self.grid_gdf = self._generate_vector_grid()
        logger.info(f"Initialized Fusion Engine with {len(self.grid_gdf)} grid cells.")

    def _generate_vector_grid(self) -> gpd.GeoDataFrame:
        """Generates the foundational 1km x 1km polygon grid."""
        min_lon, min_lat, max_lon, max_lat = self.bbox
        polygons = []
        grid_ids = []
        
        lon = min_lon
        i = 0
        while lon < max_lon:
            lat = min_lat
            while lat < max_lat:
                polygons.append(Polygon([
                    (lon, lat),
                    (lon + self.resolution, lat),
                    (lon + self.resolution, lat + self.resolution),
                    (lon, lat + self.resolution)
                ]))
                grid_ids.append(f"GRID-{i:05d}")
                lat += self.resolution
                i += 1
            lon += self.resolution

        gdf = gpd.GeoDataFrame({'grid_id': grid_ids}, geometry=polygons, crs="EPSG:4326")
        return gdf

    def fuse_multimodal_data(self, 
                             iot_data: List[Dict[str, Any]], 
                             traffic_data: List[Dict[str, Any]], 
                             weather_data: Dict[str, Any], 
                             satellite_features: Dict[str, Any]) -> pd.DataFrame:
        """
        Takes disparate data streams, performs spatial joins to map them to grid cells,
        and temporal alignment to create a unified feature vector per grid cell.
        """
        logger.info("Executing Spatial-Temporal Fusion pipeline...")
        
        # 1. Map IoT Data to Grid
        if iot_data:
            iot_df = pd.DataFrame(iot_data)
            iot_df['geometry'] = iot_df['location'].apply(lambda loc: Point(loc['lon'], loc['lat']))
            iot_gdf = gpd.GeoDataFrame(iot_df, geometry='geometry', crs="EPSG:4326")
            # Spatial Join: Assign each IoT sensor to a grid cell
            joined_iot = gpd.sjoin(iot_gdf, self.grid_gdf, how="inner", predicate="within")
            # Aggregate by grid_id (mean of measurements if multiple sensors in one cell)
            # In production, measurement extraction requires flattening the dict
            measurements = pd.json_normalize(joined_iot['measurements'])
            measurements['grid_id'] = joined_iot['grid_id'].values
            grid_iot_agg = measurements.groupby('grid_id').mean().reset_index()
        else:
            grid_iot_agg = pd.DataFrame(columns=['grid_id', 'pm25', 'pm10', 'no2', 'so2', 'co', 'o3'])

        # 2. Map Traffic Data to Grid
        if traffic_data:
            trf_df = pd.DataFrame(traffic_data)
            trf_df['geometry'] = trf_df['location'].apply(lambda loc: Point(loc['lon'], loc['lat']))
            trf_gdf = gpd.GeoDataFrame(trf_df, geometry='geometry', crs="EPSG:4326")
            joined_trf = gpd.sjoin(trf_gdf, self.grid_gdf, how="inner", predicate="within")
            grid_trf_agg = joined_trf.groupby('grid_id').agg({
                'current_speed_kmh': 'mean',
                'vehicle_count_est': 'sum'
            }).reset_index()
        else:
            grid_trf_agg = pd.DataFrame(columns=['grid_id', 'current_speed_kmh', 'vehicle_count_est'])

        # 3. Merge everything into the base grid
        master_df = pd.DataFrame(self.grid_gdf.drop(columns='geometry'))
        master_df = master_df.merge(grid_iot_agg, on='grid_id', how='left')
        master_df = master_df.merge(grid_trf_agg, on='grid_id', how='left')
        
        # Fill missing values for grid cells without sensors/traffic with regional interpolation (mocked via forward fill/mean here)
        master_df.fillna(master_df.mean(numeric_only=True), inplace=True)
        # If still NaN, fill with 0
        master_df.fillna(0, inplace=True)

        # 4. Append Macro Weather & Satellite features (Broadcasted to all cells)
        master_df['temp_c'] = weather_data.get('temperature_celsius', 25.0)
        master_df['wind_spd'] = weather_data.get('wind_speed_ms', 2.0)
        master_df['sat_mean_density'] = satellite_features.get('mean_column_density', 0.0)

        logger.info(f"Fusion complete. Produced {len(master_df)} unified feature vectors.")
        return master_df

if __name__ == "__main__":
    # Test the fusion engine
    fusion = GridFusionEngine(bbox=(77.0, 28.4, 77.4, 28.8))
    # Provide empty lists for test, it should gracefully handle missing data via broadcast/fill
    fused_state = fusion.fuse_multimodal_data([], [], {}, {})
    logger.info(f"Sample Fused State:\n{fused_state.head()}")
