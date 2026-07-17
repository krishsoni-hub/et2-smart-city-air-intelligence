import json
import logging
import time
import random
from typing import Dict, Any, List
from datetime import datetime
from kafka import KafkaProducer

# Configure logging for production tracing
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)
logger = logging.getLogger("CAAQMS-Ingestion")

class IoTDataIngestor:
    """Base interface for IoT Data Ingestion."""
    def fetch_data(self) -> List[Dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement fetch_data")

class MockCAAQMSProvider(IoTDataIngestor):
    """
    Simulates production CAAQMS IoT gateways pushing multi-pollutant data.
    In a live system, this interfaces with actual hardware APIs or MQTT brokers.
    """
    def __init__(self, grid_cells: int = 100):
        self.grid_cells = grid_cells
        # Simulated sensor locations across the 1km grid
        self.sensors = [
            {"sensor_id": f"CAQMS-{i:03d}", "lat": 28.61 + random.uniform(-0.1, 0.1), "lon": 77.20 + random.uniform(-0.1, 0.1)}
            for i in range(self.grid_cells)
        ]

    def fetch_data(self) -> List[Dict[str, Any]]:
        """Generates a batch of synthetic IoT sensor readings."""
        batch = []
        timestamp = datetime.utcnow().isoformat() + "Z"
        for sensor in self.sensors:
            payload = {
                "sensor_id": sensor["sensor_id"],
                "timestamp": timestamp,
                "location": {"lat": sensor["lat"], "lon": sensor["lon"]},
                "measurements": {
                    "pm25": round(random.uniform(15.0, 300.0), 2),
                    "pm10": round(random.uniform(30.0, 500.0), 2),
                    "no2": round(random.uniform(10.0, 150.0), 2),
                    "so2": round(random.uniform(5.0, 80.0), 2),
                    "co": round(random.uniform(0.5, 5.0), 2),
                    "o3": round(random.uniform(10.0, 100.0), 2)
                },
                "status": "ACTIVE"
            }
            batch.append(payload)
        return batch

class KafkaIoTProducer:
    """Production Kafka producer for real-time streaming to the Fusion Engine."""
    def __init__(self, bootstrap_servers: str, topic: str):
        self.topic = topic
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                retries=5
            )
            logger.info(f"Connected to Kafka broker at {bootstrap_servers}")
        except Exception as e:
            logger.warning(f"Kafka connection failed (Expected if broker is offline): {e}. Falling back to stdout.")
            self.producer = None

    def publish(self, data_batch: List[Dict[str, Any]]) -> None:
        if self.producer:
            for record in data_batch:
                self.producer.send(self.topic, record)
            self.producer.flush()
            logger.info(f"Published {len(data_batch)} records to Kafka topic '{self.topic}'")
        else:
            logger.info(f"[Mock Kafka Publish] 1st record sample: {data_batch[0] if data_batch else 'Empty'}")

def main():
    logger.info("Starting CAAQMS IoT Ingestion Service...")
    provider = MockCAAQMSProvider(grid_cells=50) # 50 simulated sensors
    producer = KafkaIoTProducer(bootstrap_servers="localhost:9092", topic="caaqms_telemetry")
    
    try:
        while True:
            # Poll at 10 second intervals for high-frequency updates
            data = provider.fetch_data()
            producer.publish(data)
            time.sleep(10)
    except KeyboardInterrupt:
        logger.info("Graceful shutdown initiated.")

if __name__ == "__main__":
    main()
