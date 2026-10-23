"""Groq Whisper transcription service for FastAPI voice chat.

The public functions retain the existing API: connect, validate_audio,
transcribe_audio, and health_check.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional

import groq
from groq import Groq

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024
ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a", "mp4", "mpeg", "mpga", "webm", "ogg", "flac"}
ALLOWED_MIME_TYPES = {
    "audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave",
    "audio/mpeg", "audio/mp3", "audio/mp4", "audio/x-m4a",
    "audio/webm", "video/webm", "audio/ogg", "application/ogg",
    "audio/flac", "audio/x-flac", "video/mp4",
    "application/octet-stream",  # Generic upload header; does not prove file format.
}
LANGUAGE_MAP = {
    "en": "English", "hi": "Hindi", "bn": "Bengali", "mr": "Marathi",
    "ta": "Tamil", "te": "Telugu", "gu": "Gujarati", "kn": "Kannada",
    "ml": "Malayalam", "pa": "Punjabi", "ur": "Urdu",
}
_LANGUAGE_ALIASES = {name.casefold(): name for name in LANGUAGE_MAP.values()}

_lock = threading.Lock()
_client: Optional[Groq] = None
_client_key: Optional[str] = None


def _api_key() -> str:
    return (
        os.getenv("GROQ_WHISPER_API_KEY")
        or os.getenv("GROQ_API_KEY")
        or ""
    ).strip()


def _timeout_seconds() -> float:
    try:
        value = float(os.getenv("GROQ_WHISPER_TIMEOUT", "60"))
        return min(max(value, 5.0), 180.0)
    except (TypeError, ValueError):
        return 60.0


def connect() -> None:
    """Initialize the Groq client safely; no network call is made here."""
    global _client, _client_key
    key = _api_key()

    if not key:
        with _lock:
            _client = None
            _client_key = None
        logger.warning("Whisper credentials are not configured.")
        return

    if _client is not None and _client_key == key:
        return

    with _lock:
        if _client is not None and _client_key == key:
            return
        try:
            new_client = Groq(
                api_key=key,
                timeout=_timeout_seconds(),
                max_retries=1,
            )
        except Exception:
            _client = None
            _client_key = None
            logger.exception("Unable to initialize Groq Whisper client.")
            return

        _client = new_client
        _client_key = key
        logger.info("Groq Whisper client initialized.")


def _get_client() -> Optional[Groq]:
    key = _api_key()
    if not key:
        return None
    if _client is None or _client_key != key:
        connect()
    return _client if _client_key == key else None


def validate_audio(
    filename: str,
    file_size_bytes: int,
    mime_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate upload metadata; this does not authenticate file contents."""
    if not isinstance(filename, str) or not filename.strip():
        return {"valid": False, "message": "Audio filename is required."}

    extension = Path(filename).suffix.lower().lstrip(".")
    if extension not in ALLOWED_EXTENSIONS:
        return {"valid": False, "message": "Unsupported audio format."}

    if (
        not isinstance(file_size_bytes, int)
        or isinstance(file_size_bytes, bool)
        or file_size_bytes <= 0
    ):
        return {"valid": False, "message": "Audio file is empty or unreadable."}

    if file_size_bytes > MAX_FILE_SIZE_BYTES:
        return {"valid": False, "message": "Audio exceeds the 25 MB limit."}

    if mime_type:
        normalized_mime = mime_type.split(";", 1)[0].strip().lower()
        if normalized_mime not in ALLOWED_MIME_TYPES:
            return {"valid": False, "message": "Unsupported audio MIME type."}

    return {"valid": True, "message": "Audio metadata validated."}


def _result(
    request_id: str,
    timestamp: str,
    started: float,
    *,
    success: bool = False,
    transcription: str = "",
    detected_language: Optional[str] = None,
    message: str = "",
    error_type: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "request_id": request_id,
        "success": success,
        "transcription": transcription,
        "detected_language": detected_language,
        "provider": "Groq",
        "model": MODEL_NAME,
        "processing_time": round(time.monotonic() - started, 3),
        "timestamp": timestamp,
        "message": message,
        "error_type": error_type,
    }


def _language_name(raw: Any) -> Optional[str]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    normalized = raw.strip().casefold()
    code = normalized.split("-", 1)[0]
    return (
        LANGUAGE_MAP.get(code)
        or _LANGUAGE_ALIASES.get(normalized)
        or raw.strip()
    )


def transcribe_audio(
    file_object: BinaryIO,
    filename: str,
    mime_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Transcribe a readable, seekable audio stream."""
    started = time.monotonic()
    request_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    if file_object is None or not all(
        hasattr(file_object, attr) for attr in ("seek", "tell", "read")
    ):
        return _result(
            request_id,
            timestamp,
            started,
            message="Audio must be a readable, seekable binary stream.",
            error_type="invalid_audio",
        )

    try:
        file_object.seek(0, os.SEEK_END)
        file_size = file_object.tell()
        file_object.seek(0)
    except (OSError, TypeError, ValueError):
        return _result(
            request_id,
            timestamp,
            started,
            message="Unable to read audio stream.",
            error_type="invalid_audio",
        )

    validation = validate_audio(filename, file_size, mime_type)
    if not validation["valid"]:
        return _result(
            request_id,
            timestamp,
            started,
            message=validation["message"],
            error_type="invalid_audio",
        )

    client = _get_client()
    if client is None:
        return _result(
            request_id,
            timestamp,
            started,
            message="Voice transcription is not configured.",
            error_type="not_configured",
        )

    try:
        file_object.seek(0)
        response = client.audio.transcriptions.create(
            model=MODEL_NAME,
            file=(Path(filename).name, file_object),
            response_format="verbose_json",
            temperature=0,
        )

        if isinstance(response, dict):
            raw_text = response.get("text", "")
            raw_language = response.get("language")
        else:
            raw_text = getattr(response, "text", "")
            raw_language = getattr(response, "language", None)

        transcription = raw_text.strip() if isinstance(raw_text, str) else ""
        if not transcription:
            return _result(
                request_id,
                timestamp,
                started,
                message="No speech detected.",
                error_type="no_speech",
            )

        language = _language_name(raw_language)
        logger.info(
            "Whisper request=%s success=true duration=%.2fs",
            request_id,
            time.monotonic() - started,
        )
        return _result(
            request_id,
            timestamp,
            started,
            success=True,
            transcription=transcription,
            detected_language=language,
            message="Transcription completed successfully.",
        )

    except groq.AuthenticationError:
        kind, message = (
            "authentication_error",
            "Voice transcription authentication failed.",
        )
    except groq.RateLimitError:
        kind, message = (
            "rate_limit",
            "Voice transcription is busy. Please retry shortly.",
        )
    except groq.APITimeoutError:
        kind, message = (
            "timeout",
            "Voice transcription timed out. Please retry.",
        )
    except groq.APIConnectionError:
        kind, message = (
            "connection_error",
            "Voice transcription service is unreachable.",
        )
    except groq.BadRequestError:
        kind, message = (
            "invalid_audio",
            "The audio could not be processed. Try another recording.",
        )
    except groq.APIStatusError as exc:
        kind, message = (
            "provider_error",
            "Voice transcription service returned an error.",
        )
        logger.warning(
            "Whisper request=%s provider_status=%s",
            request_id,
            exc.status_code,
        )
    except Exception:
        kind, message = (
            "unexpected_error",
            "Unable to transcribe audio right now.",
        )
        logger.exception("Unexpected Whisper failure request=%s", request_id)

    logger.warning(
        "Whisper request=%s success=false error_type=%s",
        request_id,
        kind,
    )
    return _result(
        request_id,
        timestamp,
        started,
        message=message,
        error_type=kind,
    )


def health_check() -> Dict[str, Any]:
    """Report local configuration without testing provider connectivity."""
    started = time.monotonic()
    key = _api_key()
    configured = bool(key)
    initialized = _client is not None and _client_key == key

    if initialized:
        message = "Client initialized; provider connectivity not tested."
    elif configured:
        message = "Client not initialized."
    else:
        message = "Whisper API key is missing."

    return {
        "status": "healthy" if configured and initialized else "unhealthy",
        "provider": "Groq",
        "model": MODEL_NAME,
        "connection_status": initialized,
        "latency_ms": round((time.monotonic() - started) * 1000, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": message,
    }
