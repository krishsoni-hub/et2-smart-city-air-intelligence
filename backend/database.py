from sqlalchemy import Column, String, Float, DateTime, MetaData, Integer, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import text
import os
import logging

logger = logging.getLogger("TimescaleDB-Schema")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

Base = declarative_base()

class GridTelemetry(Base):
    """
    Core TimescaleDB Hypertable schema for spatio-temporal telemetry.
    """
    __tablename__ = 'grid_telemetry'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    time = Column(DateTime(timezone=True), nullable=False, index=True)
    grid_id = Column(String(50), nullable=False, index=True)
    
    # Target and Features
    pm25 = Column(Float)
    pm10 = Column(Float)
    no2 = Column(Float)
    temperature_c = Column(Float)
    traffic_speed_kmh = Column(Float)
    
    # Forecasted values
    predicted_pm25 = Column(Float)
    
    # Spatial geom - natively supported via PostGIS if enabled
    # geom = Column(Geometry('POLYGON', srid=4326))

def init_db(db_url: str):
    """
    Initializes the database and converts standard tables to TimescaleDB hypertables.
    """
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    
    # Convert to hypertable for timeseries optimization
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT create_hypertable('grid_telemetry', 'time', if_not_exists => TRUE);"))
            conn.commit()
            logger.info("TimescaleDB hypertable 'grid_telemetry' created successfully.")
    except Exception as e:
        logger.warning(f"Failed to create hypertable (Is TimescaleDB extension installed?): {e}")

    return sessionmaker(bind=engine)

if __name__ == "__main__":
    # Test DB Init using SQLite fallback for local dev if Postgres unavailable
    # Note: SQLite does not support TimescaleDB extensions
    test_engine_url = os.getenv("DATABASE_URL", "sqlite:///./test_timescale.db")
    SessionLocal = init_db(test_engine_url)
    logger.info("Database schema initialized.")
