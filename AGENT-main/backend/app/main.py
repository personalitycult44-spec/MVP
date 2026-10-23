import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Core Architectural Routing and Database Imports
from app.api.endpoints import router
from app.services.rag.pipeline import ingest_all_pdfs_from_folder
from app.services import redis_cache
from app.database import chromadb_client

# 7. Configure isolated module-scoped logger (Without basicConfig overrides)
logger = logging.getLogger(__name__)

# =====================================================================
# 4. LIFESPAN WORKSPACE ENGINE (PROPER MODERN FASTAPI INITIALIZATION)
# =====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 8. Start exact visual log sequence
    logger.info("---------------------------------------------------")
    logger.info("Starting Appna Finance AI Backend...")
    
    # --- STEP 1: INITIALIZE REDIS ---
    logger.info("Initializing Redis...")
    try:
        # 4 & 9. Safe execution connection attempt
        redis_cache.connect()
        logger.info("Redis Connected")
    except Exception as e:
        logger.error(f"Redis initialization failed: {str(e)}", exc_info=True)
        logger.warning("Continuing deployment without active Redis cache tier.")

    # --- STEP 2: INITIALIZE CHROMADB ---
    logger.info("Initializing ChromaDB...")
    try:
        # 4 & 10. Isolated database initialization execution
        chromadb_client.initialize_database()
        logger.info("ChromaDB Ready")
    except Exception as e:
        logger.error(f"ChromaDB database layer setup failed: {str(e)}", exc_info=True)

    # --- STEP 3: LOAD KNOWLEDGE BASE COLLECTION ---
    logger.info("Checking Knowledge Base...")
    try:
        collection = chromadb_client.get_collection("appna_bank_knowledge")
        
        # 4. Evaluate document payload caching matrix bounds
        if collection is not None and collection.count() == 0:
            logger.info("Knowledge Base Empty → Auto Index Started")
            try:
                # 11. Run safe local indexing automation script 
                status = ingest_all_pdfs_from_folder("pdfs")
                logger.info(f"Auto-index sequence finished: {status}")
            except Exception as index_err:
                logger.error(f"PDF indexing automation pipeline failed: {str(index_err)}", exc_info=True)
        else:
            count = collection.count() if collection is not None else 0
            logger.info(f"Existing knowledge base detected. Loaded {count} cached data points.")
            
    except Exception as e:
        logger.error(f"Error validating knowledge base metrics status: {str(e)}", exc_info=True)

    logger.info("Startup Complete")
    logger.info("---------------------------------------------------")
    
    yield  # Hand over operational runtime flow control to the ASGI application layer
    
    # Optional: Run cleanup logic here if needed when the container stops

# 1. Initialize core FastAPI framework using the standard lifespan routing agent
app = FastAPI(
    title="Appna Bank AI - Backend MVP", 
    version="1.0",
    lifespan=lifespan
)

# 2. Keep the existing communication access layers (CORS Middleware Configuration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Maintain existing functional API routes map binding instance
app.include_router(router)

# =====================================================================
# 12. ROOT SYSTEM APP CHECK ENDPOINT
# =====================================================================
@app.get("/")
def root():
    # 12. Return the exact standardized structural analytics payload frame
    return {
        "message": "Appna Finance AI Backend is running.",
        "version": "1.0",
        "status": "healthy"
    }
