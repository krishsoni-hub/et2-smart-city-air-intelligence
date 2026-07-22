from datetime import datetime, timezone
import os
import logging
from typing import Dict, List

from google import genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger("API-Gateway")
logging.basicConfig(level=logging.INFO)

# ==========================================
# 🔑 PASTE YOUR GOOGLE AI STUDIO KEY HERE
# ==========================================
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini Client
try:
    if GEMINI_API_KEY:
        ai_client = genai.Client(api_key=GEMINI_API_KEY)
    else:
        ai_client = None
except Exception as e:
    logger.warning(f"Gemini client initialization failed: {e}")
    ai_client = None

app = FastAPI(
    title="ET AI 2.0 Smart City Air Intelligence API",
    description=(
        "Backend gateway for real-time pollution forecasting, SHAP causal"
        " analysis, and inspector routing."
    ),
    version="1.0.0",
)

# Enable CORS for the React/Leaflet dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten for production
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
    """Returns the latest fused PM2.5 forecasts across the city grid."""
    mock_data = [
        GridForecastResponse(
            grid_id=f"GRID-{i:03d}",
            timestamp=datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            current_pm25=float(50 + (i * 2)),
            forecasted_pm25_1hr=float(55 + (i * 2.5)),
            hotspot_severity="CRITICAL" if i > 40 else "MODERATE",
        )
        for i in range(50)
    ]
    return mock_data


@app.get("/api/v1/advisory/{grid_id}", response_model=AdvisoryResponse)
def get_grid_advisory(grid_id: str):
    """Returns dynamic AI-generated health advisories using Gemini for a specific grid."""
    if ai_client and GEMINI_API_KEY:
        try:
            prompt = (
                f"Generate a short public health warning for grid area {grid_id} experiencing high air pollution (PM2.5 spike). "
                "Provide the warning in both English and Hindi in 1 concise sentence each. "
                "Format response as:\nEnglish: <message>\nHindi: <message>"
            )

            response = ai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )

            # Return AI response
            return AdvisoryResponse(
                grid_id=grid_id,
                advisories={
                    "AI_Generated_Notice": response.text.strip(),
                    "English": (
                        f"ALERT for {grid_id}: High pollution detected. Limit"
                        " outdoor exposure."
                    ),
                    "Hindi": (
                        f"चेतावनी ({grid_id}): उच्च वायु प्रदूषण। बाहर निकलने से"
                        " बचें।"
                    ),
                },
            )
        except Exception as e:
            logger.error(f"Error generating AI advisory: {e}")

    # Fallback response if API key is not set or call fails
    return AdvisoryResponse(
        grid_id=grid_id,
        advisories={
            "English": (
                "ALERT: High pollution driven by heavy traffic. Limit outdoor"
                " time."
            ),
            "Hindi": (
                "चेतावनी: भारी यातायात के कारण उच्च प्रदूषण। बाहर जाने से"
                " बचें।"
            ),
        },
    )


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting FastAPI Uvicorn Server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)