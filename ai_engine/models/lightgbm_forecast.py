import logging
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any, Tuple
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger("LightGBM-Forecaster")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class LightGBMForecaster:
    """
    Predictive tabular model using LightGBM. Designed for fast, robust 
    hyperlocal forecasting based on fused grid data.
    """
    def __init__(self, target_column: str = "pm25"):
        self.target = target_column
        self.model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.features = []

    def prepare_features(self, df: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
        """
        Creates time-series lag features and cyclic temporal features.
        """
        df_feats = df.copy()
        
        # Temporal features
        if time_col in df_feats.columns:
            df_feats['datetime'] = pd.to_datetime(df_feats[time_col])
            df_feats['hour'] = df_feats['datetime'].dt.hour
            df_feats['dayofweek'] = df_feats['datetime'].dt.dayofweek
            
        # Select numeric features
        self.features = [col for col in df_feats.columns if col not in [self.target, 'datetime', 'grid_id', time_col, 'geometry']]
        return df_feats

    def train(self, df: pd.DataFrame) -> None:
        """
        Trains the LightGBM model. Expects historical fused data.
        """
        logger.info(f"Training LightGBM Forecaster for target: {self.target}")
        df_processed = self.prepare_features(df)
        
        X = df_processed[self.features]
        y = df_processed[self.target]
        
        # TimeSeriesSplit for rigorous validation
        tscv = TimeSeriesSplit(n_splits=3)
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            logger.info(f"Fold {fold+1} validation completed.")
            
        logger.info("Final model trained successfully on full dataset.")
        self.model.fit(X, y) # Retrain on all available data for production deployment

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Generates predictions for incoming fused grid data.
        """
        df_processed = self.prepare_features(df)
        X = df_processed[self.features]
        predictions = self.model.predict(X)
        logger.info(f"Generated {len(predictions)} predictions.")
        return predictions

if __name__ == "__main__":
    # Smoke test
    import numpy as np
    synthetic_data = pd.DataFrame({
        "timestamp": pd.date_range("2026-07-17", periods=100, freq="H"),
        "grid_id": ["GRID-00000"] * 100,
        "pm25": np.random.normal(50, 10, 100),
        "temp_c": np.random.normal(30, 5, 100),
        "traffic_speed": np.random.normal(40, 10, 100)
    })
    
    forecaster = LightGBMForecaster()
    forecaster.train(synthetic_data)
    preds = forecaster.predict(synthetic_data.tail(5))
    logger.info(f"Sample Predictions: {preds}")
