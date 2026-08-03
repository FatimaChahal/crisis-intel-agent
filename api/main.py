import os
from agents.wildfire_agent_v2 import run_agent_v2
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv

from data.silver.clean import clean_alert
from data.silver.models import Alert
from pydantic import BaseModel

load_dotenv()

app = FastAPI(
    title="Crisis Intel Agent API",
    description="API for geospatial crisis analysis",
    version="0.1.0",
)

# API Key security
API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key")


def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Verify the API key from request header.

    Args:
        api_key: The API key from the request header.

    Returns:
        The validated API key.

    Raises:
        HTTPException: If the API key is invalid.
    """
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key


@app.get("/health")
def health() -> dict:
    """
    Check if the API is running.

    Returns:
        A dictionary with the API status.
    """
    return {"status": "ok", "version": "0.1.0"}


@app.post("/ingest")
def ingest(alert: dict, api_key: str = Security(verify_api_key)) -> Alert:
    """
    Receive a raw alert, clean and validate it.

    Args:
        alert: Raw alert dictionary.
        api_key: Validated API key.

    Returns:
        A validated and cleaned Alert object.
    """
    return clean_alert(alert)


class AnalyzeRequest(BaseModel):
    """Request model for wildfire analysis."""

    question: str
    history: list = []


class AnalyzeResponse(BaseModel):
    """Response model for wildfire analysis."""

    answer: str
    question: str
    status: str


@app.post("/analyze")
def analyze(
    request: AnalyzeRequest, api_key: str = Security(verify_api_key)
) -> AnalyzeResponse:
    """
    Analyze a wildfire-related question using the 4-agent LangGraph pipeline.

    Args:
        request: Question and optional chat history.
        api_key: Validated API key.

    Returns:
        Structured analysis response with answer and status.
    """
    answer = run_agent_v2(request.question, request.history)
    return AnalyzeResponse(answer=answer, question=request.question, status="success")
