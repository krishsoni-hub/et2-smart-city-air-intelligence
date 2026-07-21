import pytest
import json

class MockSpatialTemporalFusionEngine:
    def process_telemetry(self, sensor_value, satellite_flag):
        # Align anomaly to target 1km grid cell
        if sensor_value > 300 and satellite_flag == 'ANOMALY':
            return {"grid_id": "G_DEL_01", "fused_pm25": 315.5}
        return {"grid_id": "G_DEL_01", "fused_pm25": 50.0}

class MockCausalAttributionEngine:
    def attribute(self, fused_data, wind_vector):
        if fused_data['fused_pm25'] > 300 and wind_vector == 'HIGH_WIND':
            return {"primary_source": "Industrial", "confidence": 0.95, "weights": {"Industrial": 0.95, "Traffic": 0.05}}
        return {"primary_source": "Traffic", "confidence": 0.60, "weights": {"Traffic": 0.60, "Industrial": 0.40}}

class MockEnforcementPrioritizationAgent:
    def rank_hazards(self, fused_data, attribution):
        if fused_data['fused_pm25'] > 300:
            return {"priority_rank": 1, "target_grid": fused_data['grid_id'], "dispatch_action": "Deploy Inspector Teams"}
        return {"priority_rank": 99, "target_grid": fused_data['grid_id'], "dispatch_action": "None"}

class MockGenAILangchainEngine:
    def generate_advisory(self, priority_event):
        if priority_event['priority_rank'] == 1:
            return json.dumps({
                "english": "Severe air quality hazard detected. Please stay indoors.",
                "hindi": "गंभीर वायु गुणवत्ता खतरा। कृपया घर के अंदर रहें।"
            })
        return "{}"

def test_pipeline_end_to_end_cycle():
    """
    Pipeline End-to-End Integration Test Suite.
    Simulates a full cycle from ingestion anomaly to LangChain advisory.
    """
    # 1. Inject anomalous real-time IoT PM2.5 spike (>300) + satellite flag
    raw_sensor_pm25 = 320.0
    satellite_flag = 'ANOMALY'
    wind_vector = 'HIGH_WIND'
    
    # Initialize mock engines
    fusion_engine = MockSpatialTemporalFusionEngine()
    attribution_engine = MockCausalAttributionEngine()
    enforcement_agent = MockEnforcementPrioritizationAgent()
    genai_engine = MockGenAILangchainEngine()
    
    # 2. Verify Spatial-Temporal Fusion Engine
    fused_data = fusion_engine.process_telemetry(raw_sensor_pm25, satellite_flag)
    assert fused_data['grid_id'] == "G_DEL_01"
    assert fused_data['fused_pm25'] > 300.0, "Fusion engine failed to capture the anomalous spike."
    
    # 3. Assert Causal Attribution Engine shifts weights
    attribution = attribution_engine.attribute(fused_data, wind_vector)
    assert attribution['primary_source'] == "Industrial"
    assert attribution['weights']['Industrial'] > 0.80, "Attribution failed to shift weight towards Industrial under high wind/spike."
    
    # 4. Verify Enforcement Prioritization Agent
    dispatch_card = enforcement_agent.rank_hazards(fused_data, attribution)
    assert dispatch_card['priority_rank'] == 1, "High hazard event not assigned top priority."
    assert dispatch_card['dispatch_action'] == "Deploy Inspector Teams"
    
    # 5. Assert LangChain GenAI engine triggers translated advisory
    advisory_json = genai_engine.generate_advisory(dispatch_card)
    advisory = json.loads(advisory_json)
    assert "english" in advisory
    assert "hindi" in advisory
    assert advisory["hindi"] == "गंभीर वायु गुणवत्ता खतरा। कृपया घर के अंदर रहें।"
