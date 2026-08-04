"""FastAPI entrypoint for EvidenceAI.

POST /verify runs the retrieval and stance pipeline, which is loaded once at
startup. Verdicts are not persisted, so the lookup and report endpoints are
declared but not yet implemented.
"""

import os
import sys
from contextlib import asynccontextmanager

import psutil
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models import ClaimRequest, VerdictResponse
from app.pipeline.service import PipelineService, get_service, init_service

_process = psutil.Process(os.getpid())


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading retrieval index and stance classifier...", file=sys.stderr)
    try:
        init_service()
    except FileNotFoundError as exc:
        print(f"\nSTARTUP FAILED: {exc}\n", file=sys.stderr)
        raise
    print("Pipeline ready.", file=sys.stderr)
    yield


app = FastAPI(
    title="EvidenceAI",
    description="Explainable research claim verification for mental health literature.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    """Report process and model state, not just liveness."""
    rss_mb = round(_process.memory_info().rss / (1024 * 1024), 1)

    try:
        service: PipelineService = get_service()
        models = {
            "embedding_model": settings.EMBEDDING_MODEL,
            "stance_model_mode": settings.STANCE_MODEL_MODE,
            "stance_model_source": service.stance_classifier.source,
            "stance_model_device": service.stance_classifier.device,
            "passages_indexed": len(service.retrieval.passages),
        }
        pipeline_loaded = True
    except RuntimeError:
        models = None
        pipeline_loaded = False

    return {
        "status": "ok" if pipeline_loaded else "degraded",
        "pipeline_loaded": pipeline_loaded,
        "rss_mb": rss_mb,
        "models": models,
    }


@app.post("/verify", response_model=VerdictResponse)
def verify_claim(request: ClaimRequest) -> VerdictResponse:
    """Retrieve evidence for a claim, classify it, and return a verdict."""
    return get_service().verify(request.claim)


@app.get("/verify/{verification_id}", response_model=VerdictResponse)
def get_verdict(verification_id: str) -> VerdictResponse:
    """Fetch a previously computed verdict by id.

    Not implemented: verdicts are computed per request and never stored.
    """
    raise HTTPException(
        status_code=501,
        detail="Verdict lookup is not implemented — verdicts are not persisted. "
        "Submit the claim to POST /verify instead.",
    )


@app.get("/verify/{verification_id}/report")
def get_report(verification_id: str) -> None:
    """Export a verdict as a .docx report.

    Not implemented: depends on verdict persistence above.
    """
    raise HTTPException(
        status_code=501,
        detail="Report export is not implemented — it depends on verdict persistence.",
    )
