import os
import time
import uuid
import hashlib
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from pypdf import PdfReader
# Configuration and retrieval initialization client wrapper from your database module
from app.database.chromadb_client import get_collection

# Configure production module-scoped logger (No basicConfig overrides)
logger = logging.getLogger(__name__)

# =====================================================================
# TUNED CONFIGURATION & CONSTANTS
# =====================================================================
COLLECTION_ROUTING = {
    "default": "banking_knowledge",
    "English": "stock_market_english",
    "Hindi": "stock_market_hindi",
    "Bengali": "stock_market_bengali",
    "schemes": "government_schemes",
    "loans": "loan_information"
}

# Optimized text window parameter boundaries
DEFAULT_CHUNK_SIZE = 300       # Tuned down to 250-350 range for high semantic precision
DEFAULT_CHUNK_OVERLAP = 50     # Kept within 40-60 optimal boundary limits
SIMILARITY_THRESHOLD = 0.65    # Calibrated distance threshold metric
MAX_CONTEXT_CHUNKS = 4
MAX_CONTEXT_CHAR_LIMIT = 8000

# =====================================================================
# INTENT ROUTING DICTIONARY MAP
# =====================================================================
INTENT_KEYWORDS = {
    "loans": ["loan", "emi", "interest", "borrow", "mortgage", "kcc", "rin", "byaj"],
    "schemes": ["scheme", "yojana", "pm-kisan", "sukanya", "suraksha", "bima", "pension", "government", "sarkari"]
}

# High-Performance Chroma Collection Reuse Cache Workspace
_LOCAL_COLLECTION_INSTANCE_CACHE: Dict[str, Any] = {}

def _get_cached_or_fresh_collection(collection_name: str) -> Any:
    """Reuses cached collection references to drastically minimize I/O overhead."""
    if collection_name in _LOCAL_COLLECTION_INSTANCE_CACHE:
        return _LOCAL_COLLECTION_INSTANCE_CACHE[collection_name]
    
    collection_obj = get_collection(collection_name)
    _LOCAL_COLLECTION_INSTANCE_CACHE[collection_name] = collection_obj
    return collection_obj


# =====================================================================
# CHUNKING, TEXT PROCESSING & RICH METADATA INGESTION
# =====================================================================
def split_text_into_chunks(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    """Splits text content into granular overlapping chunk windows while normalizing spaces."""
    if not text or not text.strip():
        return []
    
    cleaned_text = " ".join(text.split())
    words = cleaned_text.split()
    
    if not words:
        return []
        
    chunks = []
    step = max(chunk_size - overlap, 1)
    
    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
            
    return chunks


def generate_content_hash(text: str) -> str:
    """Generates deterministic hashes to eliminate block collision duplication issues."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def extract_pdf_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """Extracts raw text map strings paired with corresponding source pages."""
    pages_data = []
    try:
        reader = PdfReader(pdf_path)
        for idx, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text and text.strip():
                    pages_data.append({
                        "page_number": idx + 1,
                        "text": text
                    })
            except Exception as page_err:
                logger.error(f"Error parsing page {idx + 1} within file {pdf_path}: {str(page_err)}")
    except Exception as e:
        logger.error(f"Failed handling parsing pipeline sequence on file {pdf_path}: {str(e)}")
    return pages_data


def ingest_all_pdfs_from_folder(folder_path: str = "pdfs", language: str = "English", category: str = "banking", topic: str = "general", version: str = "1.0.0") -> str:
    """Ingests filesystem PDF binaries directly inside configured target partitions with rich metadata.
    
    Optimized: Implements high-performance batch duplicate validation and robust, isolated 
    exponential backoff retry blocks around data creation matrices.
    """
    folder = Path(folder_path)
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
        return "Created directory structure workspace. Seed target data objects."
        
    pdf_files = list(folder.glob("*.pdf"))
    if not pdf_files:
        return "No processable raw file data types detected inside file directory workspace bounds."

    collection_name = COLLECTION_ROUTING.get(language, COLLECTION_ROUTING["default"])
    collection = _get_cached_or_fresh_collection(collection_name)
    
    total_chunks = 0
    total_files = 0

    for pdf_file in pdf_files:
        try:
            logger.info(f"Beginning ingestion tracking execution for PDF: {pdf_file.name}")
            file_size_kb = round(os.path.getsize(pdf_file) / 1024, 2)
            pages_extracted = extract_pdf_pages(str(pdf_file))
            
            if not pages_extracted:
                logger.warning(f"Document '{pdf_file.name}' returned an empty layout payload string. Skipping.")
                continue
            
            # --- PHASE 1: PRE-COMPILE ALL CHUNKS AND TARGET IDS FOR BATCH DISCOVERY ---
            staged_chunks = []
            all_staged_ids = []
            
            for page_obj in pages_extracted:
                p_num = page_obj["page_number"]
                chunks = split_text_into_chunks(page_obj["text"])
                
                for idx, chunk in enumerate(chunks):
                    content_hash = generate_content_hash(chunk)
                    chunk_id = f"doc_{pdf_file.stem}_p{p_num}_{idx}_{content_hash[:12]}"
                    
                    staged_chunks.append({
                        "id": chunk_id,
                        "text": chunk,
                        "page_number": p_num,
                        "chunk_number": idx,
                        "hash": content_hash
                    })
                    all_staged_ids.append(chunk_id)
            
            if not all_staged_ids:
                continue

            # --- PHASE 2: BATCH DUPLICATE DETECTION (SINGLE ROUND-TRIP OVERHEAD) ---
            existing_ids = set()
            try:
                # Retrieve existing matching IDs inside this batch window
                existing_records = collection.get(ids=all_staged_ids)
                if existing_records and existing_records.get("ids"):
                    existing_ids = set(existing_records["ids"])
            except Exception as batch_check_err:
                logger.warning(f"Batch duplication validation lookup failed, defaulting to safe append path: {str(batch_check_err)}")

            # --- PHASE 3: PROCESS, INDEX AND RETRY NEW CHUNKS EXCLUSIVELY ---
            for chunk_data in staged_chunks:
                c_id = chunk_data["id"]
                
                # Deduplicate immediately against our generated batch index set
                if c_id in existing_ids:
                    continue
                
                # Wrap creation matrices with exponential backoff retry guidelines
                max_retries = 3
                success = False
                backoff_delays = [1.0, 2.0, 4.0]
                
                for attempt in range(1, max_retries + 1):
                    try:
                        collection.add(
                            ids=[c_id],
                            documents=[chunk_data["text"]],
                            metadatas=[{
                                "source": pdf_file.name,
                                "source_type": "pdf",
                                "language": language,
                                "category": category,
                                "topic": topic,
                                "document_version": version,
                                "page_number": chunk_data["page_number"],
                                "chunk_number": chunk_data["chunk_number"],
                                "hash": chunk_data["hash"],
                                "file_size_kb": file_size_kb,
                                "timestamp": str(time.time())
                            }]
                        )
                        success = True
                        if attempt > 1:
                            logger.info(f"Success after retry: Chunk [{c_id}] written successfully on attempt {attempt}.")
                        break
                    except Exception as add_err:
                        logger.warning(
                            f"Retry attempt {attempt} failed writing chunk {chunk_data['chunk_number']} "
                            f"on page {chunk_data['page_number']} for PDF '{pdf_file.name}'. Error: {str(add_err)}"
                        )
                        if attempt < max_retries:
                            time.sleep(backoff_delays[attempt - 1])
                        else:
                            # Final failure tracking log mapping capture
                            logger.error(
                                f"Final failure reason: All {max_retries} indexing operations exhausted for chunk "
                                f"#{chunk_data['chunk_number']} on page {chunk_data['page_number']} of PDF '{pdf_file.name}'. "
                                f"Exception Details: {str(add_err)}", exc_info=True
                            )

                if success:
                    total_chunks += 1
            
            total_files += 1
            logger.info(f"Successfully finished processing document file data matrix: {pdf_file.name}")

        except Exception as file_err:
            logger.error(f"Ingestion failure on file '{pdf_file.name}': {str(file_err)}", exc_info=True)
            logger.warning("Isolating broken document chain. Moving to next available asset inside target directory pool...")
            continue

    return f"Ingested {total_files} files into {total_chunks} blocks inside target partition context [{collection_name}]."


# =====================================================================
# ROUTING AND INTENT EXTRACTION LAYER
# =====================================================================
def _determine_intent_collections(query: str, detected_language: str) -> List[str]:
    """Evaluates incoming user intent parameters to route queries efficiently,
    minimizing redundant cross-collection searching.
    """
    query_lower = query.lower()
    collections_to_search = []
    
    if any(kw in query_lower for kw in INTENT_KEYWORDS["loans"]):
        collections_to_search.append(COLLECTION_ROUTING["loans"])
    elif any(kw in query_lower for kw in INTENT_KEYWORDS["schemes"]):
        collections_to_search.append(COLLECTION_ROUTING["schemes"])
    
    lang_collection = COLLECTION_ROUTING.get(detected_language, COLLECTION_ROUTING["default"])
    collections_to_search.append(lang_collection)
    
    if COLLECTION_ROUTING["default"] not in collections_to_search:
        collections_to_search.append(COLLECTION_ROUTING["default"])
        
    return list(set(collections_to_search))


# =====================================================================
# METRIC METADATA EXTRACTION & ADJACENT CHUNK MERGING
# =====================================================================
def _query_single_collection(collection_name: str, query: str, n_results: int) -> List[Dict[str, Any]]:
    """Fetches precision vector segments using robust exception guards."""
    try:
        collection = _get_cached_or_fresh_collection(collection_name)
        results = collection.query(query_texts=[query], n_results=n_results)
        
        extracted_chunks = []
        if not results or not results.get("documents"):
            return []
            
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0] if results.get("metadatas") else [{} for _ in docs]
        distances = results.get("distances", [[]])[0] if results.get("distances") else [0.0 for _ in docs]
        
        for i in range(len(docs)):
            raw_distance = distances[i]
            similarity_score = round(float(1.0 - raw_distance), 3)
            
            extracted_chunks.append({
                "text": docs[i],
                "metadata": metas[i] if i < len(metas) else {},
                "score": similarity_score,
                "collection": collection_name
            })
        return extracted_chunks
    except Exception as e:
        # Returns empty list context tracking safely instead of crashing downstream flows
        logger.error(f"ChromaDB extraction anomaly caught inside partition partition layer [{collection_name}]: {str(e)}")
        return []


def _preserve_neighboring_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Checks matching criteria signatures page and sequence identifiers
    to combine contiguous content nodes before LLM passing pipelines.
    """
    if len(chunks) <= 1:
        return chunks
        
    merged_chunks = []
    skip_indices = set()
    
    for i in range(len(chunks)):
        if i in skip_indices:
            continue
            
        current_chunk = chunks[i]
        curr_meta = current_chunk.get("metadata", {})
        
        if curr_meta and "source" in curr_meta and "page_number" in curr_meta and "chunk_number" in curr_meta:
            for j in range(i + 1, len(chunks)):
                if j in skip_indices:
                    continue
                    
                next_chunk = chunks[j]
                next_meta = next_chunk.get("metadata", {})
                
                if (curr_meta["source"] == next_meta.get("source") and 
                    curr_meta["page_number"] == next_meta.get("page_number") and 
                    abs(curr_meta["chunk_number"] - next_meta.get("chunk_number")) == 1):
                    
                    if curr_meta["chunk_number"] < next_meta["chunk_number"]:
                        current_chunk["text"] = f"{current_chunk['text']}\n\n{next_chunk['text']}"
                    else:
                        current_chunk["text"] = f"{next_chunk['text']}\n\n{current_chunk['text']}"
                        
                    current_chunk["score"] = max(current_chunk["score"], next_chunk["score"])
                    skip_indices.add(j)
                    
        merged_chunks.append(current_chunk)
    return merged_chunks


def _execute_tavily_web_search(query: str) -> str:
    """Executes search updates using Tavily API layer when local vector hits miss thresholds limits."""
    # Fix Tavily API Key Validation
    tavily_api_key = os.getenv("TAVILY_API_KEY", "")
    if not tavily_api_key or not tavily_api_key.strip():
        return ""
    try:
        from tavily import TavilyClient
        tavily = TavilyClient(api_key=tavily_api_key)
        response = tavily.search(query=query, search_depth="basic", max_results=2)
        
        web_context_blocks = []
        for result in response.get("results", []):
            web_context_blocks.append(f"[Web Source: {result.get('url')}] {result.get('content')}")
        return "\n\n".join(web_context_blocks)
    except Exception as e:
        logger.error(f"Fallback internet search routing encountered an exception: {str(e)}")
        return ""


# =====================================================================
# RAG PIPELINE EXPORT INTERFACE ENTRY POINT
# =====================================================================
def search_bookshelf(query: str, detected_language: str = "English", n_results: int = MAX_CONTEXT_CHUNKS) -> Dict[str, Any]:
    """Independent RAG Processing Pipeline Segment Interface Node Entry.
    
    Consolidated singleton routing station executing safe bounded threads
    and processing single-execution global web lookups directly.
    """
    start_time = time.time()
    
    if not query or not query.strip():
        return {"found": False, "context": "", "source": "No Query Provided Target Parameters", "similarity_score": 0.0}

    target_collections = _determine_intent_collections(query, detected_language)
    all_retrieved_chunks: List[Dict[str, Any]] = []
    
    bounded_max_workers = max(1, len(target_collections))
    
    with ThreadPoolExecutor(max_workers=bounded_max_workers) as executor:
        futures = {executor.submit(_query_single_collection, col, query, n_results): col for col in target_collections}
        for future in as_completed(futures):
            all_retrieved_chunks.extend(future.result())

    # De-duplicate matching paragraph texts overlapping matrix segments 
    seen_contents = set()
    unique_chunks = []
    for chunk in all_retrieved_chunks:
        norm_text = "".join(chunk["text"].lower().split())
        if norm_text not in seen_contents:
            seen_contents.add(norm_text)
            unique_chunks.append(chunk)

    unique_chunks.sort(key=lambda x: x["score"], reverse=True)

    # Filter items directly inside configured Threshold verification parameters
    filtered_chunks = [c for c in unique_chunks if c["score"] >= SIMILARITY_THRESHOLD]
    
    processed_local_chunks = _preserve_neighboring_chunks(filtered_chunks)
    
    top_score = unique_chunks[0]["score"] if unique_chunks else 0.0
    collection_utilized = unique_chunks[0]["collection"] if unique_chunks else "None"
    
    local_context_str = "\n\n".join([c["text"].strip() for c in processed_local_chunks[:MAX_CONTEXT_CHUNKS]])
    local_context_str = local_context_str[:MAX_CONTEXT_CHAR_LIMIT]

    # Singleton execution point for Tavily fallback. Eliminates backend duplication loop trends.
    if not filtered_chunks or top_score < SIMILARITY_THRESHOLD:
        logger.info(f"RAG confidence ({top_score}) below threshold ({SIMILARITY_THRESHOLD}). Activating singleton Tavily fallback mapping loop.")
        web_context = _execute_tavily_web_search(query)
        
        if web_context:
            if local_context_str.strip():
                context_string = f"--- LOCAL KNOWLEDGE DATA (CONFIDENCE MARGINAL) ---\n{local_context_str}\n\n--- COMPLEMENTARY GLOBAL WEB INTELLIGENCE ---\n{web_context}"
                final_source = "Hybrid Fusion (ChromaDB + Tavily Fallback Web Search)"
            else:
                context_string = web_context
                final_source = "Tavily Web Intelligence Network Search Engine"
            top_score = max(top_score, 0.85)  
        else:
            context_string = local_context_str
            final_source = f"Knowledge Base Lower Limit Bounds Match ({collection_utilized})"
    else:
        context_string = local_context_str
        final_source = f"ChromaDB Enterprise Clusters: {collection_utilized}"

    elapsed_time = round(time.time() - start_time, 3)
    
    logger.info(
        f"RAG Search Complete -> Routed Collections: {target_collections} | Selected Source Node: {final_source} | "
        f"Confidence Score: {top_score} | Processing Latency Delay: {elapsed_time}s"
    )

    return {
        "found": len(context_string.strip()) > 0,
        "context": context_string[:MAX_CONTEXT_CHAR_LIMIT],
        "source": final_source,
        "collection_name": collection_utilized,
        "similarity_score": top_score,
        "retrieved_chunk_count": len(processed_local_chunks),
        "search_time": elapsed_time
    }
