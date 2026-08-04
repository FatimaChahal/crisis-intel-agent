import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

load_dotenv()

app = FastAPI(
    title="Crisis Intel Agent API",
    description="API for geospatial crisis analysis",
    version="0.1.0",
)

API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key")


def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Verify the API key from request header."""
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key


class AnalyzeRequest(BaseModel):
    """Request model for wildfire analysis."""
    question: str
    history: list = []


class AnalyzeResponse(BaseModel):
    """Response model for wildfire analysis."""
    answer: str
    question: str
    status: str


class Alert(BaseModel):
    """Alert model for ingestion."""
    titre: str
    pays: str
    severite: str


@app.get("/health")
def health() -> dict:
    """Check if the API is running."""
    return {"status": "ok", "version": "0.1.0"}


@app.post("/ingest")
def ingest(
    alert: Alert,
    api_key: str = Security(verify_api_key)
) -> dict:
    """Receive and clean a raw alert."""
    return {
        "titre": alert.titre.strip().lower(),
        "pays": alert.pays.strip().lower(),
        "severite": alert.severite.strip().lower(),
    }


@app.post("/analyze")
def analyze(
    request: AnalyzeRequest,
    api_key: str = Security(verify_api_key)
) -> AnalyzeResponse:
    """Analyze a wildfire question — agent runs locally."""
    return AnalyzeResponse(
        answer=f"Analysis for: {request.question} — Connect local agent for full response.",
        question=request.question,
        status="success"
    )