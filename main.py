from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
import os
import tempfile
import shutil
import logging
import time
import datetime
from google.cloud import storage
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

def setup_model_cache(cache_age_hours: int = 24) -> str:
    """
    Simple timestamp-based cache management:
    1. Check if local cache exists and is fresh
    2. If not, download entire cache from GCS
    3. Never upload - only download
    """
    try:
        # Setup credentials
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"):
            import base64
            key_data = base64.b64decode(os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"))
            with open("/tmp/service-account.json", "wb") as f:
                f.write(key_data)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/service-account.json"
            logger.info("✅ Service account credentials loaded")
        
        # Set HuggingFace cache directory
        hf_cache_dir = "/tmp/huggingface_cache"
        os.makedirs(hf_cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = hf_cache_dir
        os.environ["HF_DATASETS_CACHE"] = hf_cache_dir
        
        # Set HuggingFace token if provided
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            logger.info("✅ HuggingFace token configured")
        
        # Check cache freshness and sync if needed
        bucket_name = os.getenv("MODEL_CACHE_BUCKET")
        if bucket_name:
            cache_fresh = is_cache_fresh(hf_cache_dir, cache_age_hours)
            
            if cache_fresh:
                logger.info(f"💾 Using fresh local cache (< {cache_age_hours}h old)")
            else:
                logger.info(f"📥 Local cache missing/stale, downloading from GCS bucket: {bucket_name}")
                download_entire_cache(bucket_name, hf_cache_dir)
        else:
            logger.info("⚠️ No cache bucket configured, will download models fresh")
        
        return hf_cache_dir
        
    except Exception as e:
        logger.error(f"Error setting up model cache: {str(e)}")
        # Fallback to local directory
        hf_cache_dir = "/tmp/huggingface_cache"
        os.makedirs(hf_cache_dir, exist_ok=True)
        return hf_cache_dir

def is_cache_fresh(cache_dir: str, max_age_hours: int) -> bool:
    """
    Check if local cache exists and is within age threshold
    """
    cache_path = Path(cache_dir)
    
    # Check if cache directory exists and has content
    if not cache_path.exists() or not any(cache_path.iterdir()):
        logger.info("📁 No local cache found")
        return False
    
    # Check cache age using directory modification time
    try:
        cache_mtime = cache_path.stat().st_mtime
        cache_age = time.time() - cache_mtime
        age_hours = cache_age / 3600
        
        if age_hours < max_age_hours:
            logger.info(f"✅ Cache is fresh ({age_hours:.1f}h old, threshold: {max_age_hours}h)")
            return True
        else:
            logger.info(f"⏰ Cache is stale ({age_hours:.1f}h old, threshold: {max_age_hours}h)")
            return False
            
    except Exception as e:
        logger.error(f"Error checking cache age: {e}")
        return False

def download_entire_cache(bucket_name: str, local_cache_dir: str) -> bool:
    """
    Download entire cache from GCS bucket, preserving directory structure
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Clear existing cache to avoid conflicts
        cache_path = Path(local_cache_dir)
        if cache_path.exists():
            logger.info("🧹 Clearing old cache...")
            shutil.rmtree(cache_path)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Get all blobs and download
        logger.info("📥 Downloading cache from GCS...")
        blobs = list(bucket.list_blobs())
        
        if not blobs:
            logger.warning("📭 Cache bucket is empty")
            return False
        
        downloaded_count = 0
        total_size = 0
        
        for blob in blobs:
            try:
                # Create local path preserving bucket structure
                local_path = cache_path / blob.name
                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Download blob
                blob.download_to_filename(str(local_path))
                downloaded_count += 1
                total_size += blob.size
                
                # Log progress periodically
                if downloaded_count % 100 == 0:
                    logger.info(f"📦 Downloaded {downloaded_count}/{len(blobs)} files...")
                    
            except Exception as e:
                logger.error(f"Failed to download {blob.name}: {e}")
                continue
        
        # Update cache directory timestamp to now
        cache_path.touch()
        
        logger.info(f"✅ Cache download complete: {downloaded_count} files, {total_size / 1024 / 1024:.1f} MB")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download cache: {e}")
        return False

def get_cache_info(cache_dir: str) -> dict:
    """Get information about current cache state"""
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        return {
            "exists": False,
            "age_hours": None,
            "size_mb": 0,
            "file_count": 0
        }
    
    try:
        # Calculate cache age
        cache_mtime = cache_path.stat().st_mtime
        age_hours = (time.time() - cache_mtime) / 3600
        
        # Calculate cache size and file count
        total_size = 0
        file_count = 0
        
        for root, dirs, files in os.walk(cache_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    total_size += os.path.getsize(file_path)
                    file_count += 1
                except:
                    continue
        
        return {
            "exists": True,
            "age_hours": age_hours,
            "size_mb": total_size / 1024 / 1024,
            "file_count": file_count,
            "last_updated": datetime.fromtimestamp(cache_mtime).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting cache info: {e}")
        return {
            "exists": True,
            "age_hours": None,
            "size_mb": 0,
            "file_count": 0,
            "error": str(e)
        }


def prewarm_models() -> None:
    """Pre-download models at startup to avoid runtime delays"""
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("🔥 Pre-warming embedding model...")
        model = SentenceTransformer("nomic-AI/nomic-embed-text-v1.5", trust_remote_code=True)
        logger.info("✅ Embedding model ready")
    except Exception as e:
        logger.warning(f"Pre-warming failed: {e}")


# Initialize model cache on startup
model_cache_dir: str = setup_model_cache()
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
