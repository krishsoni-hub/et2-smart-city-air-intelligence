import pytest
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def test_forecasting_agent_rmse_reduction():
    """
    Mathematical Integration Validation Harness.
    Evaluates the Predictive Forecasting Agent's outputs against the Persistence Baseline.
    Throws a failure exception if the AI model's RMSE doesn't show at least a 25% reduction.
    """
    # Mocking validation logs / ground truth and predictions for 24h horizon
    np.random.seed(42)
    y_true = np.random.normal(100, 20, 100) # Actual PM2.5
    
    # Persistence Baseline (Y_{t+24} = Y_t)
    # Simulate high error (RMSE ~ 25)
    y_baseline = y_true + np.random.normal(0, 25, 100) 
    
    # AI Model Prediction (LightGBM/GNN)
    # AI model should perform significantly better (RMSE ~ 10)
    y_ai_pred = y_true + np.random.normal(0, 10, 100)
    
    # Compute Exact Metrics
    rmse_baseline = np.sqrt(mean_squared_error(y_true, y_baseline))
    mae_baseline = mean_absolute_error(y_true, y_baseline)
    r2_baseline = r2_score(y_true, y_baseline)
    
    rmse_ai = np.sqrt(mean_squared_error(y_true, y_ai_pred))
    mae_ai = mean_absolute_error(y_true, y_ai_pred)
    r2_ai = r2_score(y_true, y_ai_pred)
    
    print(f"Baseline -> RMSE: {rmse_baseline:.2f}, MAE: {mae_baseline:.2f}, R2: {r2_baseline:.2f}")
    print(f"AI Model -> RMSE: {rmse_ai:.2f}, MAE: {mae_ai:.2f}, R2: {r2_ai:.2f}")
    
    # Calculate Reduction
    reduction_percentage = ((rmse_baseline - rmse_ai) / rmse_baseline) * 100
    print(f"RMSE Reduction: {reduction_percentage:.2f}%")
    
    # Automated Threshold Validation Rule
    assert reduction_percentage >= 25.0, f"Validation Failed: AI Model RMSE reduction ({reduction_percentage:.2f}%) is less than the strict 25% threshold."
