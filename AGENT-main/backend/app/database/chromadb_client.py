import os
import time
import logging
import threading
import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, List
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction

# =====================================================================
# Centralized Production Constants
# =====================================================================
CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "chroma_db_storage")
EMBEDDING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

COLLECTION_METADATA_BLUEPRINT: Dict[str, Any] = {
    "hnsw:space": "cosine",  # Enforcing strict cosine similarity
    "project": "Appna Bank AI",
    "system_version": "1.0.0",
    "description": "Multilingual financial knowledge base vector storage partitions."
}

# Improved Logging configuration to match rules (no basicConfig() inside this file)
logger = logging.getLogger(__name__)

# =====================================================================
# Custom Zero-RAM Serverless Embedding Function with Auto-Retry & Timeout
# =====================================================================
class ZeroRamHuggingFaceEmbedding(EmbeddingFunction):
    """Custom embedding function that routes text formatting tasks remotely
    to the current Hugging Face Inference API. Consumes 0MB local server memory.
    """
    def __init__(self, api_key: str, model_name: str):
        if not api_key:
            raise RuntimeError(
                "Critical Configuration Error: 'HUGGINGFACE_API_KEY' environment variable is missing. "
                "Remote embedding computations cannot proceed without a valid token."
            )
        self.api_key = api_key
        # Updated to use the modern, officially supported Feature Extraction pipeline path
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"

    def __call__(self, input: Documents) -> Embeddings:
        # Normalize incoming inputs into standard list strings
        texts = [input] if isinstance(input, str) else list(input)
        batch_size = len(texts)
        
        payload = json.dumps({"inputs": texts, "options": {"wait_for_model": True}}).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Parameters meeting requirement criteria precisely
        max_retries = 3
        backoff_delays = [1.0, 2.0, 4.0]  # Exponential backoff array sequence
        timeout_seconds = 30.0            # Calibrated exactly to 30s timeout parameters

        logger.info(f"embedding request start: Dispatching remote vectors to Hugging Face API for batch size: {batch_size}")

        for attempt in range(1, max_retries + 1):
            try:
                req = urllib.request.Request(self.api_url, data=payload, headers=headers, method="POST")
                
                with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
                    status_code = response.getcode()
                    logger.info(f"Hugging Face Remote API Status Code: {status_code} on attempt {attempt}")
                    
                    result = json.loads(response.read().decode("utf-8"))
                    
                    # Check for explicit API error response mapping structures
                    if isinstance(result, dict) and "error" in result:
                        error_msg = result['error']
                        logger.error(f"API error message on attempt {attempt}: {error_msg}")
                        raise RuntimeError(f"Hugging Face Remote API Exception: {error_msg}")
                    
                    if not isinstance(result, list):
                        raise ValueError("Received an invalid response structure payload from Hugging Face endpoint.")
                    
                    logger.info(f"Embedding request successfully generated for batch size: {batch_size}")
                    return result

            except urllib.error.HTTPError as http_err:
                status_code = http_err.code
                error_body = ""
                try:
                    error_body = http_err.read().decode("utf-8")
                except Exception:
                    pass
                
                # Logs both the HTTP status code and response body when Hugging Face returns an error
                logger.error(f"HTTP Error {status_code} on attempt {attempt}: {http_err.reason} | Response Body: {error_body}")
                
                # Check for explicit authorization or validation limits (Do not retry if invalid API key or bad schema)
                if status_code in [401, 403, 400, 422]:
                    raise RuntimeError(f"Permanent API Exception (HTTP {status_code}): {http_err.reason} - {error_body}")
                
                # Transient errors or Model Loading (503) updates retry paths
                if attempt == max_retries:
                    raise RuntimeError(f"Failed to generate embeddings after {max_retries} attempts. HTTP Error: {status_code}")
                
            except (urllib.error.URLError, TimeoutError) as network_err:
                logger.error(f"Network error/timeout on attempt {attempt}: {str(network_err)}")
                if attempt == max_retries:
                    raise RuntimeError(f"Failed to generate embeddings due to network failure or timeout after {max_retries} attempts.")
                    
            except Exception as e:
                # Catch-all unexpected anomalies (e.g., parsing failures) - immediate fail to keep ingestion pipeline safe
                logger.error(f"Non-transient or parsing error exception on attempt {attempt}: {str(e)}")
                raise RuntimeError(f"Unexpected processing exception inside embedding wrapper: {str(e)}")

            # Wait using explicit exponential backoff scale (1s, 2s, 4s)
            current_delay = backoff_delays[attempt - 1]
            logger.warning(f"Transient error encountered. Waiting {current_delay} seconds before retry attempt {attempt + 1}/{max_retries}...")
            time.sleep(current_delay)

        raise RuntimeError("Final failure: Embedding function evaluation route exited retry loop unexpectedly.")

# =====================================================================
# System Lifecycle Singletons & Cache Matrices
# =====================================================================
_init_lock = threading.Lock()
_initialized: bool = False

_chroma_client: Optional[ClientAPI] = None
_embedding_function: Optional[ZeroRamHuggingFaceEmbedding] = None
_collection_cache: Dict[str, Collection] = {}


def initialize_database() -> None:
    """Thread-safe instantiation sequence using a double-check locking layout.
    
    CRITICAL: Does not run automatically on import. Must be explicitly invoked
    from within the FastAPI application lifespan workflow inside main.py.
    """
    global _chroma_client, _embedding_function, _initialized
    
    if _initialized:
        return

    with _init_lock:
        if _initialized:
            return
            
        try:
            logger.info(f"Initializing Production ChromaDB Persistent Architecture at: {CHROMA_DB_PATH}")
            os.makedirs(CHROMA_DB_PATH, exist_ok=True)
            
            _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            
            # Fetch authorization token securely
            hf_token = os.getenv("HUGGINGFACE_API_KEY", "")
            _embedding_function = ZeroRamHuggingFaceEmbedding(api_key=hf_token, model_name=EMBEDDING_MODEL_NAME)
            
            _initialized = True
            logger.info("ChromaDB singletons and remote embedding pipelines mounted successfully.")
            
        except Exception as e:
            logger.critical(f"Fail-Fast Core Lockout: Database Vector Matrix failed initialization: {str(e)}")
            raise RuntimeError(f"Database Core Matrix Initialization Failure: {str(e)}")


def get_collection(name: str) -> Collection:
    """Creates or fetches a cached collection instance partition.
    Enforces safe execution under concurrent threaded requests.
    """
    if not _initialized:
        raise RuntimeError("Core Database is uninitialized. Ensure initialize_database() is fired on lifespan start.")
        
    if name in _collection_cache:
        return _collection_cache[name]
        
    with _init_lock:
        if name in _collection_cache:
            return _collection_cache[name]
            
        try:
            logger.info(f"Cache lookup miss for collection partition [{name}]. Initializing structural sync.")
            
            custom_metadata = COLLECTION_METADATA_BLUEPRINT.copy()
            custom_metadata["collection_identifier_name"] = name
            
            collection_instance = _chroma_client.get_or_create_collection(
                name=name,
                embedding_function=_embedding_function,
                metadata=custom_metadata
            )
            
            _collection_cache[name] = collection_instance
            logger.info(f"Collection partition instance [{name}] successfully written back to memory cache.")
            
            return collection_instance
            
        except Exception as e:
            logger.error(f"Failed mounting or creating collection reference identifier [{name}]: {str(e)}")
            raise ValueError(f"Collection initialization error exception failure on context name '{name}': {str(e)}")


def check_database() -> Dict[str, Any]:
    """Verifies the health, network availability, and connection telemetry statistics of vector components."""
    start_time = time.time()
    is_healthy = False
    diagnostic_message = "All database modules responding optimally."
    cached_partitions_count = 0
    
    try:
        if not _initialized:
            raise RuntimeError("Database component layer is currently uninitialized.")
            
        if _chroma_client is None or _embedding_function is None:
            raise RuntimeError("Database configuration runtime matrices are broken or uninstantiated.")
            
        # Execute diagnostic heartbeat pulse check
        _chroma_client.heartbeat()
        cached_partitions_count = len(_collection_cache)
        is_healthy = True
        
    except Exception as e:
        diagnostic_message = f"Health diagnostic check failed: {str(e)}"
        logger.error(f"Health status assessment failure exception tracking log: {diagnostic_message}")
        
    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "database_reachable": is_healthy,
        "database_path": CHROMA_DB_PATH,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "cached_collections_count": cached_partitions_count,
        "latency_ms": elapsed_ms,
        "message": diagnostic_message
    }
