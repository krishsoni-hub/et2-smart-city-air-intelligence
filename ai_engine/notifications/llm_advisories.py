import logging
import os
from typing import Dict, Any, List
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage

logger = logging.getLogger("LLM-Citizen-Advisory")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class CitizenAdvisoryEngine:
    """
    GenAI-powered notification engine. Converts raw forecasting and causality 
    data into contextualized, multilingual health advisories for local citizens.
    """
    def __init__(self, target_languages: List[str] = ["English", "Hindi"]):
        self.languages = target_languages
        # In production, OPENAI_API_KEY must be set in environment
        # Utilizing a mock/fallback if key is missing to prevent crash during setup
        self.api_key_present = "OPENAI_API_KEY" in os.environ
        if self.api_key_present:
            self.llm = ChatOpenAI(temperature=0.3, model="gpt-4")
        else:
            logger.warning("OPENAI_API_KEY not found. Using Mock LLM generation for testing.")
            self.llm = None

    def _generate_mock_advisory(self, pm25_level: float, causes: str, lang: str) -> str:
        """Fallback mock generator for CI/CD or missing API keys."""
        if lang.lower() == "hindi":
            return f"चेतावनी: वायु गुणवत्ता खराब है (PM2.5: {pm25_level:.1f}). कारण: {causes}. कृपया घर के अंदर रहें।"
        return f"ALERT: Poor air quality detected (PM2.5: {pm25_level:.1f}). Primary cause: {causes}. Please limit outdoor activities."

    def generate_advisories(self, grid_id: str, pm25_level: float, causal_factors: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Generates localized advisories based on the forecasted severity and SHAP causes.
        """
        logger.info(f"Generating GenAI advisories for {grid_id} (PM2.5: {pm25_level:.1f})")
        
        # Format the causal factors for the prompt
        cause_string = ", ".join([f"{c['feature']} (Impact: {c['shap_impact']:.1f})" for c in causal_factors])
        
        advisories = {}
        for lang in self.languages:
            if not self.api_key_present:
                advisories[lang] = self._generate_mock_advisory(pm25_level, cause_string, lang)
                continue

            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="You are a public health official for a Smart City. Your job is to issue brief, actionable, and empathetic air quality alerts based on telemetry data."),
                HumanMessage(content=f"Write a 2-sentence SMS alert in {lang}. Current PM2.5 is {pm25_level:.1f}. Main causes: {cause_string}. Tell citizens what to do.")
            ])
            
            try:
                response = self.llm(prompt.format_messages())
                advisories[lang] = response.content
            except Exception as e:
                logger.error(f"LLM Generation failed for {lang}: {e}")
                advisories[lang] = "Automated Advisory System Error."
                
        logger.info(f"Generated advisories in {len(self.languages)} languages.")
        return advisories

if __name__ == "__main__":
    engine = CitizenAdvisoryEngine()
    mock_causes = [
        {"feature": "Heavy Traffic", "shap_impact": 45.2},
        {"feature": "Low Wind Speed", "shap_impact": 15.1}
    ]
    alerts = engine.generate_advisories("GRID-XYZ", 155.0, mock_causes)
    import json
    logger.info(f"Generated Alerts:\n{json.dumps(alerts, indent=2, ensure_ascii=False)}")
