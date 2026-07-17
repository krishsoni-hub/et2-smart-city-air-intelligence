import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger("Persistence-Baseline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class PersistenceBaseline:
    """
    Implements a strict persistence baseline: prediction(t) = observation(t-1)
    All AI models (LightGBM/GNN) MUST outperform this baseline to be deployed.
    """
    def __init__(self, target_column: str = "pm25"):
        self.target = target_column

    def evaluate(self, df: pd.DataFrame, time_col: str = "timestamp", entity_col: str = "grid_id") -> Dict[str, float]:
        """
        Evaluates the persistence model across the spatio-temporal dataframe.
        Assumes data is sorted chronologically per grid_id.
        """
        logger.info(f"Evaluating Persistence Baseline for target: {self.target}")
        
        # Ensure sorting by time and space
        df_sorted = df.sort_values(by=[entity_col, time_col]).copy()
        
        # Shift to get the persistence prediction (t-1)
        df_sorted['predicted_target'] = df_sorted.groupby(entity_col)[self.target].shift(1)
        
        # Drop the first timestep per grid which has no t-1 prediction
        df_eval = df_sorted.dropna(subset=['predicted_target', self.target])
        
        y_true = df_eval[self.target].values
        y_pred = df_eval['predicted_target'].values
        
        metrics = {
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred))
        }
        
        logger.info(f"Baseline Metrics - RMSE: {metrics['RMSE']:.2f}, MAE: {metrics['MAE']:.2f}, R2: {metrics['R2']:.2f}")
        return metrics

if __name__ == "__main__":
    # Test with synthetic data
    dates = pd.date_range("2026-07-17", periods=24, freq="H")
    synthetic_data = []
    for i in range(2):  # 2 grid cells
        base_pm25 = 100.0
        for dt in dates:
            synthetic_data.append({
                "timestamp": dt,
                "grid_id": f"GRID-{i:05d}",
                "pm25": base_pm25 + np.random.normal(0, 10)
            })
    df_synthetic = pd.DataFrame(synthetic_data)
    
    baseline = PersistenceBaseline()
    baseline.evaluate(df_synthetic)
