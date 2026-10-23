import time
import logging
import io
from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File, status
from pydantic import BaseModel, Field

# Strict Orchestration Imports - Reusing your production single-responsibility services
from app.services import whisper_service, redis_cache, llm_service, rag_pipeline 
from app.services.api_key_manager import SupabaseKeyManager

# Configure production logging format
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [Voice Router] - %(message)s")

router = APIRouter(prefix="/api/v1", tags=["Voice Orchestration"])

# Instantiating the Key Manager to safely handle manual fallback rotations if needed
key_manager = SupabaseKeyManager()

# =====================================================================
# RESPONSE SCHEMA DEFINITION
# =====================================================================
class VoiceResponse(BaseModel):
    success: bool = Field(..., description="Indicates if the entire pipeline execution completed successfully.")
    question: str = Field(..., description="The textual transcription processed from the user speech.")
    answer: str = Field(..., description="The final generated response from the LLM or Redis cache.")
    detected_language: str = Field(..., description="The language detected during processing.")
    source: str = Field(..., description="The source of the generation payload ('Cache' or provider name like 'Groq').")
    response_time: float = Field(..., description="Total pipeline execution latency metrics tracked in seconds.")
    timestamp: str = Field(..., description="ISO timestamp representing when the response transaction completed.")


# =====================================================================
# CORE ORCHESTRATION ENDPOINT
# =====================================================================
@router.post("/voice", response_model=VoiceResponse, status_code=status.HTTP_200_OK)
async def process_voice_query(audio_file: UploadFile = File(...)) -> VoiceResponse:
    """Asynchronously orchestrates the entire Appna Bank AI voice pipeline.
    
    Flow execution:
    Receive Audio -> Validate -> Whisper (STT) -> Redis Cache Key Gen -> Cache Lookup -> 
    (If Hit: Return) -> (If Miss: RAG Context Retrieval -> LLM Generation -> Cache Save -> Return)
    """
    start_time = time.time()
    
    # 1. Pre-flight Validation of Upload Components
    if not audio_file or not audio_file.filename:
        logging.error("Voice route rejected: Missing or completely unallocated multi-part form file object.")
        return _build_error_response("No audio file payload found in submission request headers.", start_time)

    filename = audio_file.filename
    content_type = audio_file.content_type

    # Read binary bytes asynchronously from incoming stream data chunk maps
    try:
        audio_bytes = await audio_file.read()
        file_size = len(audio_bytes)
    except Exception as read_err:
        logging.error(f"Failed to read raw incoming audio file stream blocks: {str(read_err)}")
        return _build_error_response("Failed reading incoming multipart audio stream.", start_time)

    # 2. Invoke Whisper Audio Validation Layer (MIME and extension checks)
    validation = whisper_service.validate_audio(filename, file_size, content_type)
    if not validation["valid"]:
        logging.warning(f"Audio file failed initial system compliance checks: {validation['message']}")
        return _build_error_response(validation["message"], start_time)

    # 3. Step 1: Speech-to-Text via Whisper Service
    audio_stream = io.BytesIO(audio_bytes)
    
    try:
        whisper_result = whisper_service.transcribe_audio(audio_stream, filename, content_type)
    except Exception as e:
        logging.error(f"Cascading crash intercepted at whisper_service processing boundary: {str(e)}")
        return _build_error_response("Speech processing pipeline experienced an infrastructure failure.", start_time)

    if not whisper_result.get("success"):
        error_msg = whisper_result.get("message", "Speech-to-text conversion failed.")
        logging.warning(f"Whisper processing execution aborted payload compilation: {error_msg}")
        return _build_error_response(error_msg, start_time)

    question_text = whisper_result.get("transcription", "").strip()
    request_id = whisper_result.get("request_id", "unknown-trace")

    if not question_text:
        logging.warning(f"[Req: {request_id}] Whisper returned empty textual string content parameters.")
        return _build_error_response("No speech detected.", start_time, question=question_text)

    # 4. Step 2: Deterministic Redis Cache Key Generation & Lookup Block
    cache_key = None
    try:
        cache_key = redis_cache.generate_cache_key(question_text)
        if cache_key:
            cached_payload = redis_cache.get(cache_key)
            if cached_payload and cached_payload.get("success"):
                elapsed_time = round(time.time() - start_time, 3)
                logging.info(f"[Req: {request_id}] Production Cache HIT. Bypassing RAG and LLM loops entirely.")
                
                return VoiceResponse(
                    success=True,
                    question=question_text,
                    answer=cached_payload.get("answer", ""),
                    detected_language=cached_payload.get("detected_language", "English"),
                    source="Cache",
                    response_time=elapsed_time,
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
    except Exception as cache_err:
        logging.error(f"[Req: {request_id}] Graceful suppression: Redis cache extraction error caught: {str(cache_err)}")

    # 5. Step 3: Knowledge Base Context Retrieval via RAG Pipeline
    logging.info(f"[Req: {request_id}] Production Cache MISS. Initiating RAG vector retrieval pipeline.")
    try:
        retrieved_context = rag_pipeline.search_bookshelf(question_text)
    except Exception as rag_err:
        logging.error(f"[Req: {request_id}] RAG subsystem threw an execution error exception: {str(rag_err)}")
        return _build_error_response("Knowledge base retrieval failed.", start_time, question=question_text)

    # 6. Step 4: Text Generation via single-responsibility LLM Service
    try:
        llm_response = llm_service.ask_the_principal(question_text, retrieved_context)
        
        # FIXED: Catch key exhaustion hidden inside unsuccessful dictionary payload responses
        if not llm_response.get("success") and "Authentication" not in llm_response.get("answer", ""):
            logging.warning(f"[Req: {request_id}] LLM Generation failed on active key. Triggering automatic fallback rotation...")
            
            # Drop the current key state out of circulation completely in Supabase
            key_manager.handle_quota_exhausted()
            
            # Immediately invoke a seamless second-pass execution with the next active key queue item
            logging.info(f"[Req: {request_id}] Retrying generation with fresh cyclic queue worker...")
            llm_response = llm_service.ask_the_principal(question_text, retrieved_context)

    except Exception as llm_err:
        logging.error(f"[Req: {request_id}] LLM answer generation subsystem threw a fatal error: {str(llm_err)}")
        return _build_error_response("Answer generation service experienced a platform timeout error.", start_time, question=question_text)

    elapsed_time = round(time.time() - start_time, 3)
    final_timestamp = datetime.now(timezone.utc).isoformat()

    # 7. Step 5: Enforce Cache Policies and Commit Successful Transactions
    # FIXED: Prevents caching error text payloads into your production Redis instances
    if llm_response.get("success") and str(llm_response.get("answer")).strip():
        try:
            if cache_key:
                redis_cache.set(cache_key, llm_response)
        except Exception as cache_set_err:
            logging.error(f"[Req: {request_id}] Safe bypass: Failed committing new success key records to Redis: {str(cache_set_err)}")

    logging.info(
        f"[Req: {request_id}] Total Processing Complete -> Format: {filename.split('.')[-1]} | "
        f"Total Latency: {elapsed_time}s | Pipeline Success: {llm_response.get('success', False)}"
    )

    return VoiceResponse(
        success=llm_response.get("success", False),
        question=question_text,
        answer=llm_response.get("answer", "Unable to formulate response guidance strategy."),
        detected_language=llm_response.get("detected_language", "English"),
        source=llm_response.get("source", "Groq"),
        response_time=elapsed_time,
        timestamp=final_timestamp
    )


# =====================================================================
# PRIVATE AUXILIARY ROUTER HELPERS
# =====================================================================
def _build_error_response(message: str, start_time: float, question: str = "") -> VoiceResponse:
    """Helper formatting block to assemble clean, non-crashing structured error models."""
    return VoiceResponse(
        success=False,
        question=question,
        answer=message,
        detected_language="English",
        source="System-Safety-Guardrail",
        response_time=round(time.time() - start_time, 3),
        timestamp=datetime.now(timezone.utc).isoformat()
    )
