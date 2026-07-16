"""FastAPI app for detector inference and trust checks."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from server import audit_coalesce, cache
from server.cache import AnalysisRecord
from server.inference import (
    ALLOWED_UPLOAD_TYPES,
    CANONICAL_TILES,
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_MB,
    InferencePipeline,
    canonical_tile_bytes,
    png_data_url,
    validate_upload,
)
from server.schemas import (
    AnalyzeResponse,
    AskRequest,
    AskResponse,
    AuditRequest,
    AuditResponse,
    ModelId,
    ScoreResponse,
)
from src.trust.auditor import (
    audit_tile,
    claude_executor,
    follow_up_attention,
    plain_reason_line,
)

_INDEX_PATH = Path(__file__).resolve().parent / "static" / "index.html"
_LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.inference = InferencePipeline.load()
    _LOGGER.info(
        "SecondLook server ready with models: %s",
        ", ".join(sorted(app.state.inference.models)),
    )
    yield


app = FastAPI(title="SecondLook", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(?:localhost|127\.0\.0\.1)(?::\d+)?$",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_INDEX_PATH)


@app.get("/health")
def health(request: Request) -> dict[str, object]:
    pipeline = request.app.state.inference
    return {
        "status": "ok",
        "models": sorted(pipeline.models),
        "device": str(pipeline.device),
    }


@app.get("/api/tile/{tile_id}", include_in_schema=False)
def tile_original(tile_id: str) -> FileResponse:
    """Serve a canonical tile original (no inference)."""
    path = CANONICAL_TILES.get(tile_id)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="unknown tile_id")
    suffix = path.suffix.lower()
    media = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }.get(suffix, "application/octet-stream")
    return FileResponse(path, media_type=media)


async def _read_image_input(
    tile_id: str | None,
    file: UploadFile | None,
) -> bytes:
    """Load image bytes from exactly one of tile_id or file."""
    normalized_tile_id = (tile_id or "").strip()
    if bool(normalized_tile_id) == (file is not None):
        raise HTTPException(status_code=422, detail="provide exactly one tile_id or file")

    if file is None:
        try:
            return canonical_tile_bytes(normalized_tile_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except OSError as exc:
            _LOGGER.exception("canonical tile read failed")
            raise HTTPException(status_code=500, detail="canonical tile unavailable") from exc

    if file.content_type not in ALLOWED_UPLOAD_TYPES:
        raise HTTPException(status_code=415, detail="unsupported image type")
    try:
        image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"image exceeds the {MAX_UPLOAD_MB} MB limit",
        )
    if not image_bytes:
        raise HTTPException(status_code=400, detail="image is empty")
    try:
        validate_upload(image_bytes, file.content_type)
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
        SyntaxError,
        ValueError,
    ) as exc:
        raise HTTPException(status_code=400, detail="invalid image") from exc
    return image_bytes


@app.post("/api/score", response_model=ScoreResponse)
async def score(
    request: Request,
    model: Annotated[ModelId, Form()],
    tile_id: Annotated[str | None, Form()] = None,
    file: Annotated[UploadFile | None, File()] = None,
) -> ScoreResponse:
    image_bytes = await _read_image_input(tile_id, file)
    try:
        value = await run_in_threadpool(
            request.app.state.inference.score,
            image_bytes,
            model,
        )
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
        SyntaxError,
        ValueError,
    ) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        _LOGGER.exception("score failed")
        raise HTTPException(status_code=500, detail="score failed") from exc
    return ScoreResponse(score=float(value))


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: Request,
    model: Annotated[ModelId, Form()],
    tile_id: Annotated[str | None, Form()] = None,
    file: Annotated[UploadFile | None, File()] = None,
) -> AnalyzeResponse:
    image_bytes = await _read_image_input(tile_id, file)

    try:
        result = await run_in_threadpool(
            request.app.state.inference.analyze,
            image_bytes,
            model,
        )
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
        SyntaxError,
        ValueError,
    ) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        _LOGGER.exception("analysis failed")
        raise HTTPException(status_code=500, detail="analysis failed") from exc

    cache.put(
        result.evidence_hash,
        AnalysisRecord(
            overlay_png=result.overlay_png,
            score=result.score,
            tier=result.tier,
            metrics=result.metrics.model_dump(),
            model=model,
        ),
    )
    return AnalyzeResponse(
        score=result.score,
        tier=result.tier,
        metrics=result.metrics,
        overlay_png=png_data_url(result.overlay_png),
        original_png=png_data_url(result.original_png),
        evidence_hash=result.evidence_hash,
    )


@app.post("/api/audit", response_model=AuditResponse)
async def audit(body: AuditRequest) -> AuditResponse:
    record = cache.get(body.evidence_hash)
    if record is None:
        raise HTTPException(status_code=404, detail="evidence not found")
    # call_tier = live slider band; record.tier = server cutoffs at analyze time
    call_tier = body.call_tier or record.tier
    if call_tier != "POSITIVE":
        raise HTTPException(status_code=409, detail="audit requires a POSITIVE call tier")
    # Slider may lift UNCERTAIN into POSITIVE; never force audit on a true NEGATIVE.
    if record.tier == "NEGATIVE":
        raise HTTPException(status_code=409, detail="audit requires a POSITIVE call tier")
    if record.audit is not None and record.audit.status != "DEFER":
        return record.audit

    coalesce_key = f"{body.evidence_hash}:{call_tier}"
    # Snapshot accepted at request time - pool queue may outlive LRU residency.
    accepted = record

    def _run() -> AuditResponse:
        latest = cache.get(body.evidence_hash)
        if latest is not None and latest.audit is not None and latest.audit.status != "DEFER":
            return latest.audit
        src = latest if latest is not None else accepted
        result = audit_tile(
            score=src.score,
            verdict=call_tier,
            metrics=src.metrics,
            image_png=src.overlay_png,
            model_id=src.model,
        )
        if result.error:
            _LOGGER.warning("attention audit deferred: %s", result.error)
        response = AuditResponse(
            status=result.status,
            reason_lines=[plain_reason_line(line) for line in result.reason_lines],
            numbers_cited=result.numbers_cited,
            defer_reason=result.defer_reason,
        )
        # Store all statuses (incl. DEFER) so /api/ask works; DEFER still re-runs above.
        cache.set_audit(body.evidence_hash, response, call_tier=call_tier)
        return response

    async def _run_async() -> AuditResponse:
        return await asyncio.get_running_loop().run_in_executor(claude_executor(), _run)

    try:
        return await audit_coalesce.run_once_async(coalesce_key, _run_async, timeout=300.0)
    except TimeoutError:
        _LOGGER.warning("attention audit coalesce wait timed out for %s", body.evidence_hash)
        # Work may have finished at the boundary and already cached a final audit.
        latest = cache.get(body.evidence_hash)
        if latest is not None and latest.audit is not None and latest.audit.status != "DEFER":
            return latest.audit
        response = AuditResponse(
            status="DEFER",
            reason_lines=[
                "Could not finish the trust check.",
                "Please review this tile yourself.",
            ],
            numbers_cited=[],
            defer_reason="AUDIT_UNAVAILABLE",
        )
        cache.set_audit(body.evidence_hash, response, call_tier=call_tier)
        return response


@app.post("/api/ask", response_model=AskResponse)
async def ask(body: AskRequest) -> AskResponse:
    record = cache.get(body.evidence_hash)
    if record is None:
        raise HTTPException(status_code=404, detail="evidence not found")
    if record.audit is None:
        raise HTTPException(status_code=409, detail="attention audit not found")

    # Prefer the tier used when the audit ran (e.g. slider-lifted POSITIVE).
    ask_tier = record.audit_call_tier or record.tier

    def _run() -> tuple:
        return follow_up_attention(
            score=record.score,
            tier=ask_tier,
            metrics=record.metrics,
            audit_status=record.audit.status,
            reason_lines=record.audit.reason_lines,
            image_png=record.overlay_png,
            question=body.question,
        )

    try:
        answer, error = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(claude_executor(), _run),
            timeout=300.0,
        )
    except TimeoutError:
        return AskResponse(answer=None, error="api_error")
    return AskResponse(answer=answer, error=error)
