import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger("API-Gateway")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="ET AI 2.0 Smart City Air Intelligence API",
    description="Backend gateway for real-time pollution forecasting, SHAP causal analysis, and inspector routing.",
    version="1.0.0"
)

# Enable CORS for the React/Leaflet dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Schemas
class GridForecastResponse(BaseModel):
    grid_id: str
    timestamp: str
    current_pm25: float
    forecasted_pm25_1hr: float
    hotspot_severity: str

class AdvisoryResponse(BaseModel):
    grid_id: str
    advisories: Dict[str, str]

@app.get("/")
def health_check():
    return {"status": "operational", "system": "ET AI 2.0 Backend"}

@app.get("/api/v1/forecast/grid", response_model=List[GridForecastResponse])
def get_grid_forecasts():
    """
    Returns the latest fused PM2.5 forecasts across the city grid.
    In production, this queries the Redis cache which is hydrated by the AI Engine.
    """
    # Mock response for dashboard rendering
    mock_data = [
        GridForecastResponse(
            grid_id=f"GRID-{i:03d}",
            timestamp=datetime.utcnow().isoformat() + "Z",
            current_pm25=float(50 + (i * 2)),
            forecasted_pm25_1hr=float(55 + (i * 2.5)),
            hotspot_severity="CRITICAL" if i > 40 else "MODERATE"
        ) for i in range(50)
    ]
    return mock_data

@app.get("/api/v1/advisory/{grid_id}", response_model=AdvisoryResponse)
def get_grid_advisory(grid_id: str):
    """
    Returns multilingual health advisories for a specific grid.
    """
    return AdvisoryResponse(
        grid_id=grid_id,
        advisories={
            "English": "ALERT: High pollution driven by heavy traffic. Limit outdoor time.",
            "Hindi": "चेतावनी: भारी यातायात के कारण उच्च प्रदूषण। बाहर जाने से बचें।"
        }
    )

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI Uvicorn Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
