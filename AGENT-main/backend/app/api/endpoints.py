
import os
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    status,
)
from starlette.concurrency import run_in_threadpool

from app.models.schemas import ChatRequest, ChatResponse
from app.services.llm_service import ask_the_principal, MODEL_NAME
from app.services import whisper_service, redis_cache
from app.services.rag.pipeline import (
    search_bookshelf,
    ingest_all_pdfs_from_folder,
)
from app.database import chromadb_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

MAX_AUDIO_BYTES = 25 * 1024 * 1024
PDF_FOLDER = os.getenv("PDF_FOLDER", "pdfs")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_health_check(check_function):
    """Prevent one unhealthy dependency from breaking the health response."""
    try:
        return check_function()
    except Exception:
        logger.exception("Health check failed")
        return "unavailable"


async def process_core_chat_pipeline(
    question: str,
    request_id: str,
    start_time: float,
) -> ChatResponse:
    """
    Shared text and voice processing pipeline:
    model-specific cache -> RAG search -> LLM -> cache successful answers.
    """
    normalized_question = (question or "").strip()

    if not normalized_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please enter a question.",
        )

    cache_key = None

    # 1. Generate a model-specific cache key.
    # This helps prevent answers from an older model being reused.
    try:
        cache_key = await run_in_threadpool(
            redis_cache.generate_cache_key,
            f"{MODEL_NAME}:{normalized_question}",
        )
    except Exception:
        logger.warning(
            "[%s] Cache-key generation failed; continuing without cache.",
            request_id,
            exc_info=True,
        )

    # 2. Read cache.
    if cache_key:
        try:
            cached_response = await run_in_threadpool(
                redis_cache.get,
                cache_key,
            )

            if (
                isinstance(cached_response, dict)
                and cached_response.get("answer")
                and cached_response.get("success", True)
            ):
                elapsed = round(time.time() - start_time, 3)

                logger.info(
                    "[%s] Cache hit | latency=%.3fs",
                    request_id,
                    elapsed,
                )

                return ChatResponse(
                    success=True,
                    question=normalized_question,
                    answer=cached_response["answer"],
                    detected_language=cached_response.get(
                        "detected_language", "Unknown"
                    ),
                    source="Redis Cache",
                    response_time=elapsed,
                    timestamp=_timestamp(),
                )

        except Exception:
            logger.warning(
                "[%s] Cache read failed; continuing to RAG.",
                request_id,
                exc_info=True,
            )

    # 3. Search the existing RAG pipeline.
    context = ""
    source_identity = "General AI Knowledge"

    try:
        rag_result = await run_in_threadpool(
            search_bookshelf,
            normalized_question,
        )

        if rag_result and rag_result.get("found"):
            context = rag_result.get("context", "") or ""
            source_identity = rag_result.get(
                "source", "PDF Knowledge Base"
            )

        logger.info(
            "[%s] RAG search completed | context_found=%s",
            request_id,
            bool(context),
        )

    except Exception:
        logger.exception(
            "[%s] RAG search failed; attempting the LLM without context.",
            request_id,
        )
        source_identity = "General AI Knowledge"

    # 4. Generate the answer.
    try:
        ai_output = await run_in_threadpool(
            ask_the_principal,
            normalized_question,
            context,
        )

        if not isinstance(ai_output, dict):
            raise ValueError("LLM returned an invalid response structure.")

        final_answer = str(ai_output.get("answer") or "").strip()
        detected_language = ai_output.get(
            "detected_language", "Unknown"
        )
        llm_success = ai_output.get("success", True) is True

        if not llm_success or not final_answer:
            logger.warning(
                "[%s] LLM did not return a successful answer.",
                request_id,
            )

            return ChatResponse(
                success=False,
                question=normalized_question,
                answer=(
                    final_answer
                    or "Sorry, I couldn't prepare your answer. "
                       "Please try again shortly."
                ),
                detected_language=detected_language,
                source="Groq",
                response_time=round(time.time() - start_time, 3),
                timestamp=_timestamp(),
            )

    except Exception:
        logger.exception(
            "[%s] LLM generation failed.",
            request_id,
        )

        return ChatResponse(
            success=False,
            question=normalized_question,
            answer=(
                "The AI service is temporarily unavailable. "
                "Please try again shortly."
            ),
            detected_language="Unknown",
            source="Groq",
            response_time=round(time.time() - start_time, 3),
            timestamp=_timestamp(),
        )

    # 5. Cache successful answers only.
    if cache_key and final_answer:
        try:
            cache_payload: Dict[str, Any] = {
                "answer": final_answer,
                "detected_language": detected_language,
                "success": True,
                "model": MODEL_NAME,
            }

            await run_in_threadpool(
                redis_cache.set,
                cache_key,
                cache_payload,
            )

        except Exception:
            logger.warning(
                "[%s] Cache write failed; answer remains available.",
                request_id,
                exc_info=True,
            )

    elapsed = round(time.time() - start_time, 3)

    logger.info(
        "[%s] Chat completed | model=%s | source=%s | latency=%.3fs",
        request_id,
        MODEL_NAME,
        source_identity,
        elapsed,
    )

    return ChatResponse(
        success=True,
        question=normalized_question,
        answer=final_answer,
        detected_language=detected_language,
        source=source_identity,
        response_time=elapsed,
        timestamp=_timestamp(),
    )


# ============================================================
# TEXT CHAT
# ============================================================
@router.post("/chat", response_model=ChatResponse)
async def text_chat_endpoint(payload: ChatRequest):
    start_time = time.time()
    request_id = str(uuid.uuid4())

    question = (payload.question or "").strip()

    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    return await process_core_chat_pipeline(
        question,
        request_id,
        start_time,
    )


# ============================================================
# VOICE CHAT
# ============================================================
@router.post("/voice-chat", response_model=ChatResponse)
async def voice_chat_endpoint(
    audio_file: UploadFile = File(...),
):
    start_time = time.time()
    request_id = str(uuid.uuid4())

    if not audio_file or not audio_file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please upload a valid audio file.",
        )

    try:
        # Read at most the allowed size plus one byte to detect oversize files.
        audio_bytes = await audio_file.read(MAX_AUDIO_BYTES + 1)

        if not audio_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded audio file is empty.",
            )

        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Audio file exceeds the 25 MB limit.",
            )

        # Keep the original filename for transcription format detection.
        from io import BytesIO

        audio_stream = BytesIO(audio_bytes)

        result = await run_in_threadpool(
            whisper_service.transcribe_audio,
            audio_stream,
            audio_file.filename,
        )

        if not result or not result.get("success"):
            logger.warning(
                "[%s] Audio transcription was unsuccessful.",
                request_id,
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Audio transcription failed. Please try again.",
            )

        question = str(result.get("transcription") or "").strip()

        if not question:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No speech was detected. Please try speaking clearly.",
            )

        response = await process_core_chat_pipeline(
            question,
            request_id,
            start_time,
        )

        response.response_time = round(time.time() - start_time, 3)
        return response

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "[%s] Voice processing failed.",
            request_id,
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Voice processing is temporarily unavailable.",
        )

    finally:
        await audio_file.close()


# ============================================================
# HEALTH CHECK
# ============================================================
@router.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    redis_status = _safe_health_check(redis_cache.health_check)
    chroma_status = _safe_health_check(
        chromadb_client.check_database
    )
    whisper_status = _safe_health_check(
        whisper_service.health_check
    )

    return {
        "api": "healthy",
        "redis": redis_status,
        "chromadb": chroma_status,
        "whisper": whisper_status,
        "llm": {
            "provider": "Groq",
            "model": MODEL_NAME,
            "status": "configured",
        },
        "timestamp": _timestamp(),
    }


# ============================================================
# PDF REINDEX
# ============================================================
@router.post("/reindex", status_code=status.HTTP_200_OK)
def trigger_reindexing():
    start_time = time.time()

    try:
        indexing_summary = ingest_all_pdfs_from_folder(PDF_FOLDER)

        elapsed = round(time.time() - start_time, 3)

        logger.info(
            "PDF reindex completed | latency=%.3fs",
            elapsed,
        )

        return {
            "success": True,
            "indexed_documents": indexing_summary,
            "execution_time": f"{elapsed}s",
            "message": "PDF knowledge-base indexing completed.",
        }

    except Exception:
        logger.exception("PDF reindex failed")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PDF indexing failed. Check the server logs.",
        )
