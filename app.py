"""FastAPI wrapper for the Experiment 7 RAG application."""
import logging
import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from support_rag import answer_question

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("experiment7")

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="Question for the customer-support knowledge base")
    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question must not be blank.")
        return value

class AskResponse(BaseModel):
    answer: str
    latency_ms: float
    sources: list[dict]
    backend: str

app = FastAPI(
    title="Experiment 7 — Customer Support RAG API",
    description="RAG API with validation, controlled errors, logging, and latency measurement.",
    version="1.0.0",
)

@app.get("/health")
def health():
    """Reports API process status only; it does not guarantee the LLM provider is healthy."""
    return {"status": "ok", "service": "experiment7-rag"}

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    started = time.perf_counter()
    try:
        result = answer_question(request.question)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info("ask_success backend=%s latency_ms=%.2f source_count=%d",
                    result.get("backend", "unknown"), latency_ms, len(result.get("sources", [])))
        return {**result, "latency_ms": latency_ms}
    except Exception as exc:
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        # Log the failure category only; do not expose internal details or secrets to clients.
        logger.error("ask_failure error_type=%s latency_ms=%.2f", type(exc).__name__, latency_ms)
        raise HTTPException(status_code=503, detail="The answer service is temporarily unavailable. Please try again later.") from exc
