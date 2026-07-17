import logging
import random
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger("Traffic-Mobility-Ingestor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class TrafficMobilityIngestor:
    """
    Simulates polling of live traffic APIs (e.g., TomTom, Google Maps, Waze).
    Traffic congestion is a primary causal factor for NO2 and PM2.5 emissions.
    """
    def __init__(self, num_intersections: int = 20):
        self.num_intersections = num_intersections
        self.intersections = [
            {"id": f"INT-{i:03d}", "lat": 28.61 + random.uniform(-0.05, 0.05), "lon": 77.20 + random.uniform(-0.05, 0.05)}
            for i in range(self.num_intersections)
        ]

    def poll_live_traffic(self) -> List[Dict[str, Any]]:
        """
        Returns traffic speed metrics and congestion indices mapped to spatial nodes.
        """
        logger.info(f"Polling live traffic data for {self.num_intersections} key intersections...")
        timestamp = datetime.utcnow().isoformat() + "Z"
        payloads = []
        
        # Simulate rush hour behavior based on current UTC hour
        # (Assuming local timezone adjustments in a real system)
        current_hour = datetime.utcnow().hour
        is_rush_hour = (4 <= current_hour <= 6) or (11 <= current_hour <= 14) # Approx morning/evening in India

        for node in self.intersections:
            base_speed = 40.0 # km/h
            if is_rush_hour:
                current_speed = max(5.0, base_speed - random.uniform(15.0, 30.0))
                congestion_level = "SEVERE" if current_speed < 15.0 else "HEAVY"
            else:
                current_speed = base_speed - random.uniform(0.0, 15.0)
                congestion_level = "MODERATE" if current_speed < 30.0 else "FREE_FLOW"

            payloads.append({
                "intersection_id": node["id"],
                "timestamp": timestamp,
                "location": {"lat": node["lat"], "lon": node["lon"]},
                "current_speed_kmh": round(current_speed, 1),
                "congestion_level": congestion_level,
                "vehicle_count_est": int(random.uniform(10, 100) if not is_rush_hour else random.uniform(80, 250))
            })
            
        logger.info(f"Generated {len(payloads)} traffic mobility records. Sample: {payloads[0]}")
        return payloads

if __name__ == "__main__":
    ingestor = TrafficMobilityIngestor()
    ingestor.poll_live_traffic()
