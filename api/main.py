from fastapi import FastAPI

from data.silver.clean import clean_alert
from data.silver.models import Alert

app = FastAPI(
    title="Crisis Intel Agent API",
    description="API for geospatial crisis analysis",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    """
    Check if the API is running.

    Returns:
        A dictionary with the API status.
    """
    return {"status": "ok", "version": "0.1.0"}


@app.post("/ingest")
def ingest(alert: dict) -> Alert:
    """
    Receive a raw alert, clean and validate it.

    Args:
        alert: Raw alert dictionary.

    Returns:
        A validated and cleaned Alert object.
    """
    return clean_alert(alert)