import logging
import shap
import pandas as pd
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger("SHAP-Causal-Attribution")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class SHAPCausalEngine:
    """
    Computes exact geospatial pollution source attributions using SHAP values.
    Moves the platform from reactive forecasting to proactive root-cause analysis.
    """
    def __init__(self, trained_model: Any, feature_names: List[str]):
        """
        trained_model: e.g., the LGBMRegressor instance.
        """
        self.model = trained_model
        self.feature_names = feature_names
        # TreeExplainer is highly optimized for LightGBM
        self.explainer = shap.TreeExplainer(self.model)
        logger.info("Initialized SHAP TreeExplainer for the predictive model.")

    def explain_grid_prediction(self, grid_features: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Returns feature attributions for a set of grid cells.
        Each attribution shows how much a specific feature (e.g., 'traffic_speed')
        pushed the predicted PM2.5 above the baseline expected value.
        """
        logger.info(f"Computing SHAP values for {len(grid_features)} grid cells.")
        
        # Ensure only the trained features are passed to SHAP
        X = grid_features[self.feature_names]
        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(X)
        expected_value = self.explainer.expected_value
        
        # Handle scalar expected value if returned as array
        if isinstance(expected_value, (np.ndarray, list)):
            expected_value = float(expected_value[0])
            
        attributions = []
        for i, row in X.iterrows():
            sv = shap_values[i]
            
            # Sort features by absolute impact magnitude
            feature_impacts = [
                {"feature": feat, "value": float(val), "shap_impact": float(impact)}
                for feat, val, impact in zip(self.feature_names, row, sv)
            ]
            feature_impacts.sort(key=lambda x: abs(x["shap_impact"]), reverse=True)
            
            grid_id = grid_features.loc[i, "grid_id"] if "grid_id" in grid_features.columns else f"IDX-{i}"
            prediction = expected_value + float(np.sum(sv))
            
            attributions.append({
                "grid_id": grid_id,
                "baseline_expected_value": expected_value,
                "predicted_value": prediction,
                "top_causal_factors": feature_impacts[:3] # Top 3 root causes
            })
            
        logger.info(f"Successfully generated causal attributions.")
        return attributions

if __name__ == "__main__":
    # Smoke test with dummy model and data
    from sklearn.tree import DecisionTreeRegressor
    
    # Dummy data
    X_train = pd.DataFrame(np.random.rand(100, 3), columns=['traffic', 'temp', 'humidity'])
    y_train = X_train['traffic'] * 50 + X_train['temp'] * -10 + 20
    
    # Train dummy model
    dummy_model = DecisionTreeRegressor(max_depth=3).fit(X_train, y_train)
    
    # Initialize Engine
    engine = SHAPCausalEngine(dummy_model, feature_names=['traffic', 'temp', 'humidity'])
    
    # Test Explanation
    test_grid = X_train.head(2).copy()
    test_grid['grid_id'] = ['GRID-A', 'GRID-B']
    
    explanations = engine.explain_grid_prediction(test_grid)
    import json
    logger.info(f"Sample Explanation: {json.dumps(explanations[0], indent=2)}")
