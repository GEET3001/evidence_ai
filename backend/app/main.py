"""FastAPI entrypoint for EvidenceAI.

POST /verify runs the retrieval and stance pipeline, which is loaded once at
startup, and records the result. Verdicts are persisted to SQLite so they can
be fetched back by id and exported as a report on a later request.
"""

import os
import sys
from contextlib import asynccontextmanager
from urllib.parse import quote

import psutil
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models import ClaimRequest, VerdictResponse
from app.pipeline.service import PipelineService, get_service, init_service
from app.reporting.corpus_profile import CorpusProfile
from app.reporting.docx_report import build_report, report_filename
from app.storage import StoredPayloadError, get_store

DOCX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)

_process = psutil.Process(os.getpid())


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading retrieval index and stance classifier...", file=sys.stderr)
    try:
        init_service()
    except FileNotFoundError as exc:
        print(f"\nSTARTUP FAILED: {exc}\n", file=sys.stderr)
        raise
    # Opening the store here means a bad path or an unwritable directory fails
    # at boot rather than on the first verification that tries to save.
    get_store()
    print("Pipeline ready.", file=sys.stderr)
    yield


app = FastAPI(
    title="EvidenceAI",
    description="Explainable research claim verification for mental health literature.",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Without this the browser hides Content-Disposition from the fetch that
    # downloads a report, and the client cannot recover the filename.
    expose_headers=["Content-Disposition"],
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
    response = get_service().verify(request.claim)
    get_store().save(response)
    return response


def _load_verification(verification_id: str):
    """Fetch a stored verdict or raise the right HTTP error."""
    try:
        stored = get_store().get(verification_id)
    except StoredPayloadError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if stored is None:
        raise HTTPException(
            status_code=404,
            detail=f"No verification found with id '{verification_id}'. Verdicts "
            f"are stored when they are computed, so an unknown id means the "
            f"claim was never verified on this instance, or the store was reset.",
        )
    return stored


@app.get("/verify/{verification_id}", response_model=VerdictResponse)
def get_verdict(verification_id: str) -> VerdictResponse:
    """Fetch a previously computed verdict by id."""
    return _load_verification(verification_id).response


@app.get("/verify/{verification_id}/report")
def get_report(verification_id: str) -> Response:
    """Export a stored verdict as a .docx analysis report."""
    stored = _load_verification(verification_id)
    service = get_service()

    document = build_report(
        response=stored.response,
        corpus=CorpusProfile.from_index(service.retrieval),
        verified_at=stored.created_at,
        stance_model_source=service.stance_classifier.source,
    )

    filename = report_filename(stored.response)
    return Response(
        content=document,
        media_type=DOCX_MEDIA_TYPE,
        headers={
            # Both forms: the ASCII fallback for older clients, the RFC 5987
            # form so a claim with non-ASCII characters still names the file.
            "Content-Disposition": (
                f'attachment; filename="{filename}"; '
                f"filename*=UTF-8''{quote(filename)}"
            )
        },
    )
