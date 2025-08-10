import base64
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
import os
import tempfile
import shutil
import logging
import time
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Retrieval API",
    description="Document retrieval system using RAG - Retrieval Only (No Inference)",
    version="1.1.5"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LOCAL_CACHE_DIR = Path("/tmp/app_model_cache")
DEFAULT_CACHE_AGE_HOURS = 24 

def is_cache_fresh(cache_dir: Path, max_age_hours: int) -> bool:
    """Checks if the local cache directory exists and is within the age threshold."""
    if not cache_dir.exists() or not any(cache_dir.iterdir()):
        logging.info("📁 No local cache found.")
        return False

    try:
        # Check cache age using the main directory's modification time
        cache_mtime = cache_dir.stat().st_mtime
        cache_age_seconds = time.time() - cache_mtime
        age_hours = cache_age_seconds / 3600

        if age_hours < max_age_hours:
            logging.info(f"✅ Cache is fresh ({age_hours:.1f}h old, threshold: {max_age_hours}h).")
            return True
        else:
            logging.info(f"⏰ Cache is stale ({age_hours:.1f}h old, threshold: {max_age_hours}h).")
            return False
    except Exception as e:
        logging.error(f"Error checking cache age: {e}")
        return False

def download_cache_from_gcs(bucket_name: str, local_dir: Path) -> bool:
    """Downloads the entire cache from a GCS bucket, clearing the old cache first."""
    from google.cloud import storage

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # Clear existing cache to ensure a clean slate
        if local_dir.exists():
            logging.info(f"🧹 Clearing stale cache at {local_dir}...")
            shutil.rmtree(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"📥 Downloading cache from GCS bucket: gs://{bucket_name}...")
        blobs = list(bucket.list_blobs())

        if not blobs:
            logging.warning("📭 GCS cache bucket is empty.")
            return False

        for i, blob in enumerate(blobs):
            local_path = local_dir / blob.name
            local_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local_path))
            if (i + 1) % 100 == 0:
                 logging.info(f"📦 Downloaded {i + 1}/{len(blobs)} files...")


        local_dir.touch()
        logging.info(f"✅ Cache download complete. {len(blobs)} files synced.")
        return True

    except Exception as e:
        logging.error(f"💥 Failed to download cache from GCS: {e}")
        return False

def initialize_cache_and_offline_env(cache_age_hours: int = DEFAULT_CACHE_AGE_HOURS):
    """
    Initializes the application environment for offline model usage.

    This function performs the following steps:
    1. Sets up Google Cloud credentials.
    2. Checks if the local model cache is fresh based on a TTL (Time-To-Live).
    3. If the cache is missing or stale, it downloads the latest version from GCS.
    4. Sets environment variables to force all supported libraries (Hugging Face, Docling)
       to use the local cache in a strict offline mode.
    """
    logging.info("🚀 Initializing model cache and offline environment...")

    # 1. Setup GCS credentials if provided as a base64 JSON
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"):
        try:
            key_data = base64.b64decode(os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"))
            cred_path = "/tmp/gcs-service-account.json"
            with open(cred_path, "wb") as f:
                f.write(key_data)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path
            logging.info("✅ Service account credentials loaded from ENV var.")
        except Exception as e:
            logging.error(f"Failed to decode GCS credentials: {e}")

    # 2. Sync cache from GCS if it's stale or missing
    bucket_name = os.getenv("MODEL_CACHE_BUCKET")
    if not bucket_name:
        logging.warning("⚠️ MODEL_CACHE_BUCKET not set. Skipping GCS sync. Models must be manually cached.")
    else:
        if not is_cache_fresh(LOCAL_CACHE_DIR, cache_age_hours):
            download_cache_from_gcs(bucket_name, LOCAL_CACHE_DIR)

    # 3. Point all libraries to the correct subdirectories in the local cache
    hf_cache_path = LOCAL_CACHE_DIR / "huggingface"
    docling_cache_path = LOCAL_CACHE_DIR / "docling"
    
    # Create subdirectories if they don't exist
    hf_cache_path.mkdir(parents=True, exist_ok=True)
    docling_cache_path.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_cache_path)
    os.environ["HF_DATASETS_CACHE"] = str(hf_cache_path / "datasets")
    os.environ["HF_HUB_OFFLINE"] = "1"

    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(hf_cache_path)

    os.environ["DOCLING_ARTIFACTS_PATH"] = str(docling_cache_path)

    logging.info(f"HF_HOME set to: {os.getenv('HF_HOME')}")
    logging.info(f"DOCLING_ARTIFACTS_PATH set to: {os.getenv('DOCLING_ARTIFACTS_PATH')}")
    logging.info("✅ Environment is now configured for OFFLINE model loading.")


def prewarm_models():
    """
    Pre-loads models at startup to avoid runtime delays.
    This will fail if models are not in the cache due to the offline setting.
    """
    try:
        from transformers import AutoTokenizer
        from sentence_transformers import SentenceTransformer

        logging.info("🔥 Pre-warming models (offline)...")

        # This call will look ONLY in the local cache.
        # It will fail if "bert-base-uncased" is not present.
        tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            local_files_only=True
        )
        logging.info("   ✅ Tokenizer 'bert-base-uncased' ready.")

        model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True,
            local_files_only=True
        )
        logging.info("   ✅ Embedding model 'nomic-embed-text-v1.5' ready.")

        logging.info("✅ All models pre-warmed successfully from local cache.")

    except Exception as e:
        # This error is expected if the models aren't in the GCS bucket/local cache.
        logging.error(f"💥 Pre-warming failed. Ensure required models are in the cache bucket. Error: {e}")
        tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased", trust_remote_code=True
        )
        logging.info("   ✅ Tokenizer 'bert-base-uncased' ready.")

        model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True
        )
        logging.info("   ✅ Embedding model 'nomic-embed-text-v1.5' ready.")


# Initialize model cache on startup
initialize_cache_and_offline_env()
prewarm_models()

# Global variables to store retriever instances
retrievers: Dict[str, Any] = {} # Dict[str, LocalRAGRetriever]

# Domain-specific threshold configurations
DOMAIN_CONFIGS = {
    "medical": {"semantic_threshold": 0.99, "cross_encoder_threshold": 0.98},
    "financial": {"semantic_threshold": 0.99, "cross_encoder_threshold": 0.98}, 
    "legal": {"semantic_threshold": 0.99, "cross_encoder_threshold": 0.98},
    "general": {"semantic_threshold": 0.95, "cross_encoder_threshold": 0.95}
}

class CacheConfig(BaseModel):
    cache_dir: str = Field(default="cache", description="Cache directory path")
    max_cache_size: int = Field(default=1000, description="Maximum cache size")
    semantic_threshold: float = Field(default=0.99, ge=0.0, le=1.0, description="Semantic similarity threshold")
    cross_encoder_threshold: float = Field(default=0.98, ge=0.0, le=1.0, description="Cross-encoder threshold")
    ttl_hours: int = Field(default=24, ge=1, description="Time to live in hours")
    domain_type: Literal["medical", "financial", "legal", "general"] = Field(default="medical", description="Domain type")
    enable_semantic_cache: bool = Field(default=True, description="Enable semantic caching")
    performance_mode: Literal["speed", "balanced", "safety"] = Field(default="safety", description="Performance mode")
    max_memory_mb: int = Field(default=1024, ge=128, description="Maximum memory in MB")

class RAGConfigModel(BaseModel):
    embed_model: str = Field(default="nomic-ai/nomic-embed-text-v1.5", description="Embedding model")
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", description="Reranker model")
    chunk_size: int = Field(default=200, ge=50, le=2000, description="Chunk size")
    chunk_overlap: int = Field(default=10, ge=0, le=100, description="Chunk overlap")
    full_dim: int = Field(default=768, description="Full dimension size")
    retrieve_k: int = Field(default=10, ge=1, le=100, description="Number of documents to retrieve")
    rerank_n: int = Field(default=5, ge=1, le=50, description="Number of documents to rerank")
    enable_caching: bool = Field(default=True, description="Enable caching")
    cache_performance_mode: Literal["speed", "balanced", "safety"] = Field(default="balanced", description="Cache performance mode")

class RetrieverConfig(BaseModel):
    name: str = Field(description="Unique name for this retriever instance")
    sources: List[str] = Field(description="List of document sources (URLs or file paths)")
    rag_config: RAGConfigModel = Field(description="RAG configuration")
    cache_config: CacheConfig = Field(description="Cache configuration")

class QueryRequest(BaseModel):
    retriever_name: str = Field(description="Name of the retriever instance to use")
    query: str = Field(description="Query text")
    matryoshka_dim: int = Field(default=768, description="Matryoshka dimension")
    retrieve_k: Optional[int] = Field(default=None, description="Override retrieve_k for this query")
    rerank_n: Optional[int] = Field(default=None, description="Override rerank_n for this query")

class RetrievalResult(BaseModel):
    text: str
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    results: List[RetrievalResult]
    query: str
    retriever_name: str
    total_results: int

@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "message": "RAG Retrieval API - Document Retrieval Only (No Inference)",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": {
            "create_retriever": "/retriever/create",
            "list_retrievers": "/retriever/list",
            "query": "/query",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_retrievers": len(retrievers),
        "retriever_names": list(retrievers.keys()),
    }

@app.post("/retriever/create")
async def create_retriever(config: RetrieverConfig):
    """Create a new RAG retriever instance"""
    try:
        if config.name in retrievers:
            raise HTTPException(status_code=400, detail=f"Retriever '{config.name}' already exists")
        
        from robustRAG import LocalRAGRetriever, RAGConfig
        from layeredCache import HighConfidenceCacheManager

        domain_config = DOMAIN_CONFIGS.get(config.cache_config.domain_type, DOMAIN_CONFIGS["general"])
        semantic_threshold = config.cache_config.semantic_threshold if config.cache_config.semantic_threshold != 0.99 else domain_config["semantic_threshold"]
        cross_encoder_threshold = config.cache_config.cross_encoder_threshold if config.cache_config.cross_encoder_threshold != 0.98 else domain_config["cross_encoder_threshold"]
        
        # Create cache manager
        cache_manager = HighConfidenceCacheManager(
            cache_dir=config.cache_config.cache_dir,
            max_cache_size=config.cache_config.max_cache_size,
            semantic_threshold=semantic_threshold,
            cross_encoder_threshold=cross_encoder_threshold,
            ttl_hours=config.cache_config.ttl_hours,
            domain_type=config.cache_config.domain_type,
            enable_semantic_cache=config.cache_config.enable_semantic_cache,
            performance_mode=config.cache_config.performance_mode,
            max_memory_mb=config.cache_config.max_memory_mb
        )
        
        # Create RAG config
        rag_config = RAGConfig(
            embed_model=config.rag_config.embed_model,
            reranker_model=config.rag_config.reranker_model,
            chunk_size=config.rag_config.chunk_size,
            chunk_overlap=config.rag_config.chunk_overlap,
            full_dim=config.rag_config.full_dim,
            retrieve_k=config.rag_config.retrieve_k,
            rerank_n=config.rag_config.rerank_n,
            enable_caching=config.rag_config.enable_caching,
            cache_performance_mode=config.rag_config.cache_performance_mode
        )
        
        # Create retriever
        logger.info(f"Creating retriever '{config.name}' - models will be cached for future use")
        retriever = LocalRAGRetriever(rag_config, cache_manager)
        
        # Build index
        logger.info(f"Building index for retriever '{config.name}' with sources: {config.sources}")
        logger.info("Using Docling for PDF parsing - this is highly accurate but time-intensive for first-time processing")
        if isinstance(config.sources, list):
            retriever.build_index(sources=config.sources)
        else:
            retriever.build_index(sources=[config.sources])
                
        # Store retriever
        retrievers[config.name] = retriever
        
        return {
            "message": f"Retriever '{config.name}' created successfully",
            "name": config.name,
            "sources_count": len(config.sources)
        }
        
    except Exception as e:
        logger.error(f"Error creating retriever '{config.name}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create retriever: {str(e)}")

@app.post("/retriever/upload")
async def upload_and_create_retriever(
    name: str = Form(...),
    files: List[UploadFile] = File(...),
    rag_config: str = Form(default='{}'),
    cache_config: str = Form(default='{}')
):
    """Upload files and create a retriever instance"""
    try:
        import json
        
        # Parse configs
        rag_conf = RAGConfigModel(**json.loads(rag_config if rag_config != '{}' else '{}'))
        cache_conf = CacheConfig(**json.loads(cache_config if cache_config != '{}' else '{}'))
        
        if name in retrievers:
            raise HTTPException(status_code=400, detail=f"Retriever '{name}' already exists")
        
        # Save uploaded files
        temp_dir = tempfile.mkdtemp()
        file_paths = []
        
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            file_paths.append(file_path)
        
        # Create retriever config
        config = RetrieverConfig(
            name=name,
            sources=file_paths,
            rag_config=rag_conf,
            cache_config=cache_conf
        )
        
        # Create retriever (reuse the existing logic)
        return await create_retriever(config)
        
    except Exception as e:
        logger.error(f"Error uploading and creating retriever '{name}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload and create retriever: {str(e)}")

@app.get("/retriever/list")
async def list_retrievers() -> Dict[str, Any]:
    """List all available retriever instances"""
    return {
        "retrievers": list(retrievers.keys()),
        "total": len(retrievers)
    }

@app.delete("/retriever/{name}")
async def delete_retriever(name: str):
    """Delete a retriever instance"""
    if name not in retrievers:
        raise HTTPException(status_code=404, detail=f"Retriever '{name}' not found")
    
    del retrievers[name]
    return {"message": f"Retriever '{name}' deleted successfully"}

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents using the specified retriever - RETRIEVAL ONLY"""
    try:
        if request.retriever_name not in retrievers:
            raise HTTPException(
                status_code=404, 
                detail=f"Retriever '{request.retriever_name}' not found. Available retrievers: {list(retrievers.keys())}"
            )
        
        retriever = retrievers[request.retriever_name]
        
        # Prepare query parameters
        query_params = {
            "query": request.query,
            "matryoshka_dim": request.matryoshka_dim
        }
        
        if request.retrieve_k is not None:
            query_params["retrieve_k"] = request.retrieve_k
        if request.rerank_n is not None:
            query_params["rerank_n"] = request.rerank_n
        
        # Perform retrieval
        logger.info(f"Querying retriever '{request.retriever_name}' with query: '{request.query}'")
        results = retriever.retrieve(**query_params)
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_result = RetrievalResult(
                text=result.get("text", ""),
                score=result.get("score"),
                metadata=result.get("metadata", {})
            )
            formatted_results.append(formatted_result)
        
        return QueryResponse(
            results=formatted_results,
            query=request.query,
            retriever_name=request.retriever_name,
            total_results=len(formatted_results)
        )
        
    except Exception as e:
        logger.error(f"Error querying retriever '{request.retriever_name}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/retriever/{name}/info")
async def get_retriever_info(name: str):
    """Get information about a specific retriever"""
    if name not in retrievers:
        raise HTTPException(status_code=404, detail=f"Retriever '{name}' not found")
    
    return {
        "name": name,
        "status": "active",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
