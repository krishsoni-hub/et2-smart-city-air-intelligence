from celery import Celery
import os
import logging

logging.basicConfig(level=logging.INFO)

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

celery_app = Celery('core_tasks', broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)

@celery_app.task(name="execute_ai_pipeline")
def execute_ai_pipeline(grid_id: str):
    """
    Mock Celery task representing the asynchronous offloading of the AI prediction script.
    """
    logging.info(f"Executing AI pipeline for Grid ID: {grid_id}")
    return {"grid_id": grid_id, "status": "processed", "predicted_pm25": 145.0}
