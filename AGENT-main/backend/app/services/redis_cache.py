
import os
import time
import json
import hashlib
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import redis

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

DEFAULT_TTL_SECONDS = max(
    60,
    int(os.getenv("REDIS_CACHE_TTL", "86400")),
)

REDIS_SOCKET_TIMEOUT = float(
    os.getenv("REDIS_SOCKET_TIMEOUT", "2.0")
)

REDIS_CONNECT_TIMEOUT = float(
    os.getenv("REDIS_CONNECT_TIMEOUT", "2.0")
)

CACHE_NAMESPACE = os.getenv(
    "REDIS_CACHE_NAMESPACE",
    "appna_bank:cache:v2",
)

DEFAULT_MODEL = os.getenv(
    "GROQ_MODEL",
    "qwen/qwen3.8-27b",
)

_init_lock = threading.Lock()
_redis_pool: Optional[redis.ConnectionPool] = None
_redis_client: Optional[redis.Redis] = None


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_log_error(operation: str) -> None:
    """Log a Redis failure without exposing credentials or connection URLs."""
    logger.warning("Redis %s operation failed.", operation, exc_info=True)


# ============================================================
# CONNECTION INITIALIZATION
# ============================================================
def connect() -> None:
    """
    Initialize and verify the Redis connection pool.

    Safe to call more than once. If initialization fails, Redis caching
    remains unavailable without intentionally crashing the application.
    """
    global _redis_pool, _redis_client

    if _redis_client is not None:
        try:
            _redis_client.ping()
            return
        except redis.RedisError:
            logger.warning("Existing Redis connection failed; reconnecting.")
            _close_connection()

    with _init_lock:
        if _redis_client is not None:
            try:
                _redis_client.ping()
                return
            except redis.RedisError:
                _close_connection()

        pool = None
        client = None

        try:
            pool = redis.ConnectionPool.from_url(
                REDIS_URL,
                socket_timeout=REDIS_SOCKET_TIMEOUT,
                socket_connect_timeout=REDIS_CONNECT_TIMEOUT,
                decode_responses=True,
                health_check_interval=30,
                max_connections=int(
                    os.getenv("REDIS_MAX_CONNECTIONS", "50")
                ),
            )

            client = redis.Redis(connection_pool=pool)
            client.ping()

            _redis_pool = pool
            _redis_client = client

            logger.info("Redis connection initialized successfully.")

        except Exception:
            logger.warning(
                "Redis is unavailable. Chat will continue without caching.",
                exc_info=True,
            )

            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

            if pool is not None:
                try:
                    pool.disconnect()
                except Exception:
                    pass

            _redis_pool = None
            _redis_client = None


def _close_connection() -> None:
    """Close the current client and pool safely."""
    global _redis_pool, _redis_client

    old_client = _redis_client
    old_pool = _redis_pool

    _redis_client = None
    _redis_pool = None

    if old_client is not None:
        try:
            old_client.close()
        except Exception:
            logger.debug("Redis client cleanup failed.", exc_info=True)

    if old_pool is not None:
        try:
            old_pool.disconnect()
        except Exception:
            logger.debug("Redis pool cleanup failed.", exc_info=True)


def close() -> None:
    """Call during FastAPI shutdown to release Redis resources."""
    with _init_lock:
        _close_connection()
        logger.info("Redis connection resources released.")


def _get_client() -> Optional[redis.Redis]:
    """Return the initialized client without forcing a network connection."""
    return _redis_client


# ============================================================
# CACHE KEY GENERATION
# ============================================================
def generate_cache_key(question: str) -> Optional[str]:
    """
    Generate a deterministic SHA-256 cache key.

    The caller can include the model name in the question string to
    isolate cached responses by model, for example:
    'qwen/qwen3.8-27b:What is a stock market?'
    """
    if not question or not question.strip():
        return None

    try:
        normalized = " ".join(question.casefold().split())

        digest = hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()

        return f"{CACHE_NAMESPACE}:{digest}"

    except Exception:
        logger.exception("Cache key generation failed.")
        return None


# ============================================================
# READ CACHE
# ============================================================
def get(key: str) -> Optional[Dict[str, Any]]:
    """Return a valid cached response or None on a cache miss/failure."""
    if not key:
        return None

    client = _get_client()
    if client is None:
        return None

    try:
        cached_data = client.get(key)

        if not cached_data:
            return None

        payload = json.loads(cached_data)

        if not isinstance(payload, dict):
            logger.warning("Ignoring malformed cached response.")
            delete(key)
            return None

        answer = payload.get("answer")

        if not isinstance(answer, str) or not answer.strip():
            logger.warning("Ignoring cached response with an invalid answer.")
            delete(key)
            return None

        # Never return cached failures as successful responses.
        if payload.get("success", True) is not True:
            delete(key)
            return None

        return payload

    except (json.JSONDecodeError, TypeError, ValueError):
        logger.warning("Ignoring corrupted cached JSON.")
        delete(key)
        return None

    except redis.RedisError:
        _safe_log_error("GET")
        return None

    except Exception:
        _safe_log_error("GET")
        return None


# ============================================================
# WRITE CACHE
# ============================================================
def set(
    key: str,
    value: Dict[str, Any],
    ttl: int = DEFAULT_TTL_SECONDS,
) -> bool:
    """
    Cache a successful, non-empty answer with an expiration time.

    Returns False if Redis is unavailable or the payload is invalid.
    """
    if not key or not isinstance(value, dict):
        return False

    answer = value.get("answer")

    if not isinstance(answer, str) or not answer.strip():
        return False

    if value.get("success", True) is not True:
        return False

    client = _get_client()
    if client is None:
        return False

    try:
        ttl_seconds = max(1, int(ttl))

        payload = {
            "answer": answer.strip(),
            "detected_language": value.get(
                "detected_language", "English"
            ),
            "model": value.get("model", DEFAULT_MODEL),
            "source": value.get("source", "Groq"),
            "success": True,
            "cached_at": _timestamp(),
        }

        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        client.set(key, serialized, ex=ttl_seconds)
        return True

    except (TypeError, ValueError):
        logger.warning("Cache payload could not be serialized.")
        return False

    except redis.RedisError:
        _safe_log_error("SET")
        return False

    except Exception:
        _safe_log_error("SET")
        return False


# ============================================================
# DELETE CACHE ENTRY
# ============================================================
def delete(key: str) -> bool:
    if not key:
        return False

    client = _get_client()
    if client is None:
        return False

    try:
        return client.delete(key) > 0

    except Exception:
        _safe_log_error("DELETE")
        return False


# ============================================================
# CLEAR APPLICATION CACHE
# ============================================================
def clear() -> bool:
    """
    Delete keys belonging to this application's cache namespace only.

    Uses incremental scanning rather than Redis FLUSHDB, which could
    delete data belonging to other applications.
    """
    client = _get_client()
    if client is None:
        return False

    pattern = f"{CACHE_NAMESPACE}:*"

    try:
        batch = []

        for key in client.scan_iter(
            match=pattern,
            count=200,
        ):
            batch.append(key)

            if len(batch) >= 200:
                client.delete(*batch)
                batch.clear()

        if batch:
            client.delete(*batch)

        logger.info("Application cache cleanup completed.")
        return True

    except Exception:
        _safe_log_error("CLEAR")
        return False


# ============================================================
# HEALTH CHECK
# ============================================================
def health_check() -> Dict[str, Any]:
    """Return Redis connectivity diagnostics without revealing credentials."""
    start_time = time.time()
    client = _get_client()

    connected = False
    message = "Redis is not initialized."
    estimated_keys = None

    if client is not None:
        try:
            client.ping()
            connected = True
            message = "Redis connection is available."

            # This is a bounded sample, not the total key count.
            estimated_keys = 0
            for _ in client.scan_iter(
                match=f"{CACHE_NAMESPACE}:*",
                count=100,
            ):
                estimated_keys += 1
                if estimated_keys >= 1000:
                    break

        except Exception:
            message = "Redis connectivity check failed."
            logger.warning(
                "Redis health check failed.",
                exc_info=True,
            )

    return {
        "status": "healthy" if connected else "unhealthy",
        "redis_connected": connected,
        "cached_keys_sampled": estimated_keys,
        "latency_ms": round((time.time() - start_time) * 1000, 2),
        "timestamp": _timestamp(),
        "message": message,
    }
