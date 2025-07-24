import os
import numpy as np
from sentence_transformers import  CrossEncoder
import xxhash
import logging
import time
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Literal, Optional, Tuple
from dataclasses import dataclass, asdict, field
import faiss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('cached_rag_retriever.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class KnowledgeBaseVersion:
    """Track knowledge base versions for cache invalidation"""
    version_id: str
    created_at: datetime
    document_hashes: Dict[str, str]  # filename -> hash
    total_chunks: int
    embedding_model: str
    chunk_params: Dict[str, Any]

@dataclass
class CacheEntry:
    """Base class for cache entries"""
    query_normalized: str
    query_original: str
    query_hash: str
    timestamp: datetime
    kb_version: str  # Knowledge base version when cached
    access_count: int = 0
    last_accessed: datetime = None

@dataclass
class ExactMatchCache(CacheEntry):
    """Layer 1: Exact match cache for complete LLM responses"""
    llm_response: str = ""
    confidence: float = 1.0
    
@dataclass
class EmbeddingCache(CacheEntry):
    """Layer 2: High-confidence semantic cache"""
    query_embedding: np.ndarray = field(default_factory=lambda: np.array([]))
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    matryoshka_dim: int = 0
    retrieval_params: Dict[str, Any] = field(default_factory=dict)
    semantic_confidence: float = 0.0  # CrossEncoder confidence score

class HighConfidenceCacheManager:
    """Cache manager with strict confidence requirements"""
    
    def __init__(self, cache_dir: str = "cache", max_cache_size: int = 1000, 
                 semantic_threshold: float = 0.98, cross_encoder_threshold: float = 0.95,
                 ttl_hours: int = 24, domain_type: Literal["medical", "financial", "legal", "general"] = "medical",
                 enable_semantic_cache : bool = True, 
                 performance_mode : Literal["speed", "balanced", "safety"] = "safety",
                 max_memory_mb: int = 1024):
        
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size
        self.semantic_threshold = semantic_threshold  # Very high for safety
        self.cross_encoder_threshold = cross_encoder_threshold  # CrossEncoder confidence
        self.ttl = timedelta(hours=ttl_hours)
        self.domain_type = domain_type
        self.enable_semantic_cache = enable_semantic_cache 
        self.max_memory_mb = max_memory_mb

        if performance_mode not in ("speed", "balanced", "safety"):
            self.performance_mode = "safety"
        else:
            self.performance_mode = performance_mode

        if performance_mode == "speed":
            self.enable_semantic_cache = False  # Disable semantic cache for speed
            logger.info("Speed mode: Semantic cache disabled")
        elif performance_mode == "balanced":
            self.semantic_threshold = max(0.95, semantic_threshold - 0.02)
            self.cross_encoder_threshold = max(0.85, cross_encoder_threshold - 0.05)
            logger.info("Balanced mode: Relaxed thresholds")

        # Domain-specific configurations
        self.domain_configs = {
            "medical": {"semantic_threshold": 0.99, "cross_encoder_threshold": 0.98},
            "financial": {"semantic_threshold": 0.99, "cross_encoder_threshold": 0.98}, 
            "legal": {"semantic_threshold": 0.99, "cross_encoder_threshold": 0.98},
            "general": {"semantic_threshold": 0.95, "cross_encoder_threshold": 0.95}
        }
        
        # Apply domain-specific thresholds
        if domain_type in self.domain_configs:
            config: Dict[str, float] = self.domain_configs[domain_type]
            self.semantic_threshold: float = config["semantic_threshold"]
            self.cross_encoder_threshold: float = config["cross_encoder_threshold"]
        
        # Initialize CrossEncoder for semantic validation
        self.semantic_validator = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Cache storage
        self.exact_cache: Dict[str, ExactMatchCache] = {}
        self.embedding_cache: List[EmbeddingCache] = []
        self.cached_embeddings_index: Optional[faiss.Index] = None # FAISS index for cached embeddings
        self.cached_embedding_to_entry_map: Dict[int, EmbeddingCache] = {} # Map FAISS ID -> EmbeddingCache entry
        self._next_faiss_id: int = 0 # Counter to assign unique IDs for FAISS
        self._estimated_embedding_cache_memory_mb: float = 0.0 # Tracks estimated memory for embedding cache
        
        # Knowledge base version tracking
        self.current_kb_version: Optional[KnowledgeBaseVersion] = None
        self.kb_version_file: str = os.path.join(cache_dir, "kb_version.json")
        
        # Cache files
        self.exact_cache_file: str = os.path.join(cache_dir, "exact_cache.json")
        self.embedding_cache_file: str = os.path.join(cache_dir, "embedding_cache.npz")
        
        os.makedirs(cache_dir, exist_ok=True)
        self._load_kb_version()
        self._load_cache()
        
        logger.info(f"HighConfidenceCacheManager initialized for {domain_type} domain:")
        logger.info(f"  - Semantic similarity threshold: {self.semantic_threshold}")
        logger.info(f"  - CrossEncoder confidence threshold: {self.cross_encoder_threshold}")
        logger.info(f"  - Cache TTL: {ttl_hours} hours")
    
    def _generate_document_hash(self, file_path: str) -> str:
        """Generate hash of document content for change detection"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return xxhash.xxh64(content).hexdigest()
        except Exception as e:
            logger.warning(f"Could not hash file {file_path}: {e}")
            return ""
    
    def _generate_kb_version(self, document_paths: List[str], total_chunks: int,
                           embedding_model: str, chunk_params: Dict[str, Any]) -> KnowledgeBaseVersion:
        """Generate a new knowledge base version"""
        document_hashes: Dict[str, str] = {}
        for path in document_paths:
            if os.path.exists(path):
                document_hashes[os.path.basename(path)] = self._generate_document_hash(path)
        
        # Create version ID from all hashes and parameters
        version_data = json.dumps({
            "docs": document_hashes,
            "chunks": total_chunks,
            "model": embedding_model,
            "params": chunk_params
        }, sort_keys=True)

        version_id = xxhash.xxh64(version_data.encode()).hexdigest()[:16]

        return KnowledgeBaseVersion(
            version_id=version_id,
            created_at=datetime.now(),
            document_hashes=document_hashes,
            total_chunks=total_chunks,
            embedding_model=embedding_model,
            chunk_params=chunk_params
        )
    def _rebuild_cached_embeddings_index(self) -> None:
        """Rebuilds the FAISS index and mapping from the current embedding_cache list."""
        logger.debug("Rebuilding cached embeddings FAISS index...")
        self.cached_embedding_to_entry_map.clear()
        if not self.embedding_cache:
            self.cached_embeddings_index = None
            self._next_faiss_id = 0
            logger.debug("No cached embeddings to index.")
            return

        dim = None
        valid_embeddings_for_indexing = []
        temp_id_map = {} # Temporary map for this rebuild process: temp_id -> entry
        temp_id_counter = 0

        for entry in self.embedding_cache:
             # Consider only valid entries for indexing
            if (hasattr(entry, 'query_embedding') and
                isinstance(entry.query_embedding, np.ndarray) and
                entry.query_embedding.size > 0):
                if dim is None:
                    dim = entry.query_embedding.shape[0]
                    logger.debug(print(f"Detected embedding dimension: {dim}"))
                 # Only index if the embedding dimension matches the first one found
                embedding = entry.query_embedding
                if embedding.ndim == 2 and embedding.shape[0] == 1:
                    embedding = embedding.flatten()

                if entry.query_embedding.shape[0] == dim:
                    logger.debug(f"Adding entry with embedding dim {entry.query_embedding.shape[0]} (expected {dim})")
                    valid_embeddings_for_indexing.append(entry.query_embedding)
                    temp_id_map[temp_id_counter] = entry
                    temp_id_counter += 1
                else:
                    logger.warning(f"Skipping entry with mismatched embedding dim {entry.query_embedding.shape[0]} (expected {dim})")


        if not valid_embeddings_for_indexing:
            self.cached_embeddings_index = None
            self._next_faiss_id = 0
            logger.debug("No valid embeddings found to index.")
            return

        # Stack embeddings
        try:
            embeddings_matrix = np.vstack(valid_embeddings_for_indexing).astype('float32')
            logger.debug(f"Stacked {len(valid_embeddings_for_indexing)} embeddings into matrix of shape {embeddings_matrix.shape}")
        except ValueError as e:
            logger.error(f"Error stacking embeddings for FAISS index: {e}")
            self.cached_embeddings_index = None
            self._next_faiss_id = 0
            return

        # Create FAISS index
        self.cached_embeddings_index = faiss.IndexFlatIP(dim) # Use Inner Product for normalized vectors
        self.cached_embeddings_index.add(embeddings_matrix)

        # Populate the mapping using the temp_id_map
        # FAISS assigns IDs sequentially starting from 0 for the vectors added
        for temp_id, entry in temp_id_map.items():
            faiss_id = temp_id # FAISS ID corresponds to the order they were added
            self.cached_embedding_to_entry_map[faiss_id] = entry

        # Set _next_faiss_id to avoid ID conflicts if new entries are added later
        self._next_faiss_id = len(valid_embeddings_for_indexing)
        logger.debug(f"Valid embeddings = {valid_embeddings_for_indexing}")

        logger.info(f"Rebuilt cached embeddings FAISS index with {len(valid_embeddings_for_indexing)} entries (dimension: {dim}).")

    
    def update_kb_version(self, document_paths: List[str], total_chunks: int,
                         embedding_model: str, chunk_params: Dict[str, Any]) -> bool:
        """Update knowledge base version and invalidate cache if changed"""
        new_version = self._generate_kb_version(document_paths, total_chunks, 
                                              embedding_model, chunk_params)
        
        version_changed = (self.current_kb_version is None or 
                          self.current_kb_version.version_id != new_version.version_id)
        
        if version_changed:
            logger.info(f"Knowledge base version changed: {new_version.version_id}")
            if self.current_kb_version:
                logger.info(f"Previous version: {self.current_kb_version.version_id}")
                
                # Log what changed
                old_hashes: Dict[str, str] = self.current_kb_version.document_hashes
                new_hashes: Dict[str, str] = new_version.document_hashes
                
                for doc, new_hash in new_hashes.items():
                    old_hash = old_hashes.get(doc, "")
                    if old_hash != new_hash:
                        logger.info(f"  - Document changed: {doc}")
                
                if old_hashes.keys() != new_hashes.keys():
                    logger.info(f"  - Document set changed")
                
                if self.current_kb_version.total_chunks != new_version.total_chunks:
                    logger.info(f"  - Chunk count: {self.current_kb_version.total_chunks} -> {new_version.total_chunks}")
            
            # Clear all caches
            self._invalidate_all_caches()
            
            # Update current version
            self.current_kb_version = new_version
            self._save_kb_version()
            
            return True
        
        logger.info(f"Knowledge base unchanged: {new_version.version_id}")
        return False
    
    def _invalidate_all_caches(self) -> None:
        """Clear all cached data due to knowledge base changes"""
        cache_count = len(self.exact_cache) + len(self.embedding_cache)
        self.exact_cache.clear()
        self.embedding_cache.clear()
        logger.info(f"✓ Invalidated {cache_count} cache entries due to KB version change")
    
    def _load_kb_version(self) -> None:
        """Load knowledge base version from disk"""
        try:
            if os.path.exists(self.kb_version_file):
                with open(self.kb_version_file, 'r') as f:
                    data = json.load(f)
                    data['created_at'] = datetime.fromisoformat(data['created_at'])
                    self.current_kb_version = KnowledgeBaseVersion(**data)
                    logger.info(f"Loaded KB version: {self.current_kb_version.version_id}")
        except Exception as e:
            logger.warning(f"Failed to load KB version: {e}")
            self.current_kb_version = None

    def _save_kb_version(self) -> None:
        """Save knowledge base version to disk"""
        try:
            if self.current_kb_version:
                data: Dict[str, Any] = asdict(self.current_kb_version)
                data['created_at'] = self.current_kb_version.created_at.isoformat()
                with open(self.kb_version_file, 'w') as f:
                    json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save KB version: {e}")
    
    @staticmethod
    def normalize_query(query: str) -> str:
        """Normalize query for caching with medical/financial precision"""
        # More conservative normalization
        normalized = query.lower().strip()
        # Only remove clearly redundant punctuation
        normalized = re.sub(r'\s+', ' ', normalized)  # Normalize whitespace only
        return normalized
    
    def _generate_cache_key(self, query: str) -> str:
        """Generate a hash key for the normalized query"""
        normalized = self.normalize_query(query)
        return xxhash.xxh64(normalized.encode()).hexdigest()

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if a cache entry has expired"""
        return datetime.now() - entry.timestamp > self.ttl
    
    def _is_kb_version_valid(self, entry: CacheEntry) -> bool:
        """Check if cache entry is from current knowledge base version"""
        if not self.current_kb_version:
            return False
        return entry.kb_version == self.current_kb_version.version_id
    
    def get_exact_match(self, query: str) -> Optional[str]:
        """Layer 1: Check for exact query match with KB version validation"""
        cache_key: str = self._generate_cache_key(query)
        
        if cache_key in self.exact_cache:
            entry: ExactMatchCache = self.exact_cache[cache_key]
            if (not self._is_expired(entry) and 
                self._is_kb_version_valid(entry)):
                entry.access_count += 1
                entry.last_accessed = datetime.now()
                logger.info(f"✓ CACHE HIT (Layer 1): Exact match found")
                logger.info(f"  - Query: '{entry.query_normalized}'")
                logger.info(f"  - KB Version: {entry.kb_version}")
                logger.info(f"  - Access count: {entry.access_count}")
                return entry.llm_response
            else:
                # Remove invalid entry
                del self.exact_cache[cache_key]
        
        logger.info("✗ CACHE MISS (Layer 1): No valid exact match")
        return None
    
    def get_similar_chunks(self, query: str, query_embedding: np.ndarray, 
                          matryoshka_dim: int) -> Optional[List[Dict[str, Any]]]:
        """Layer 2: Ultra-conservative semantic similarity check"""
        
        if not self.enable_semantic_cache or not self.cached_embeddings_index:
            logger.debug("✗ CACHE MISS (Layer 2): Semantic cache disabled or index empty.")
            return None

        if self.cached_embeddings_index.ntotal == 0:
             logger.info("✗ CACHE MISS (Layer 2): FAISS index is empty.")
             return None

        logger.info(f"Checking Layer 2 cache using FAISS index (entries: {self.cached_embeddings_index.ntotal})")
        logger.info(f"Domain: {self.domain_type} (semantic_threshold: {self.semantic_threshold})")

        try:
            normalized_query = query_embedding / np.linalg.norm(query_embedding)
            query_embedding_f32 = normalized_query.astype('float32').reshape(1, -1)
            # Using Inner Product (IP), higher scores are better.
            k = min(10, self.cached_embeddings_index.ntotal) # Limit search
            scores, faiss_ids = self.cached_embeddings_index.search(query_embedding_f32, k)

            candidates: List[Tuple[EmbeddingCache, float]] = [] # (entry, score)
            for i in range(len(faiss_ids[0])):
                faiss_id = faiss_ids[0][i]
                score = scores[0][i]
                # FAISS returns -1 for IDs if k > ntotal, or if no vectors found
                if faiss_id != -1 and faiss_id in self.cached_embedding_to_entry_map:
                    entry = self.cached_embedding_to_entry_map[faiss_id]

                    if (self._is_expired(entry) or
                        not self._is_kb_version_valid(entry) or
                        entry.matryoshka_dim != matryoshka_dim):
                         continue # Skip invalid/expired entries

                    # FAISS IP scores can theoretically be > 1 or < -1 due to floating point precision,
                    # but for normalized vectors, they should be in [-1, 1]. Clamp for safety.
                    #cosine_sim_like = max(-1.0, min(1.0, score))
                    cosine_sim_like = score  
                    logger.debug(f"FAISS ID {faiss_id}: score={cosine_sim_like:.4f}, entry={entry.query_normalized[:50]}...")

                    if cosine_sim_like >= self.semantic_threshold:
                        candidates.append((entry, cosine_sim_like))
                        logger.debug(f"  Candidate (FAISS ID {faiss_id}): score={cosine_sim_like:.4f}")

            if not candidates:
                logger.info(f"✗ CACHE MISS (Layer 2): No candidates above cosine threshold {self.semantic_threshold} via FAISS")
                return None

            logger.info(f"Found {len(candidates)} candidates via FAISS search above threshold.")
        except Exception as e:
            logger.error(f"Error during FAISS search in semantic cache: {e}")
            return None
        
        # Second pass: CrossEncoder validation for highest confidence (Layer 2.5)
        best_entry: Optional[EmbeddingCache] = None
        best_cross_score: float = 0.0

        validation_start = time.time()
        for entry, cosine_sim in candidates:
            # Use CrossEncoder to validate semantic similarity
            try:
                cross_score = self.semantic_validator.predict([(query, entry.query_original)])[0]
                cross_score = np.tanh(cross_score) 
                logger.debug(f"CrossEncoder score for '{entry.query_original[:50]}...': {cross_score:.4f}")
            except Exception as e:
                logger.warning(f"CrossEncoder prediction failed for candidate '{entry.query_original[:50]}...': {e}")
                continue
            
            logger.info(f"  Candidate: cosine={cosine_sim:.4f}, cross_encoder={cross_score:.4f}")
            logger.info(f"    Cached: '{entry.query_normalized}'")
            logger.info(f"    Current: '{self.normalize_query(query)}'")
            
            if cross_score > best_cross_score and cross_score >= self.cross_encoder_threshold:
                best_cross_score = cross_score
                best_entry = entry
        
        validation_time = time.time() - validation_start
        
        if best_entry and best_cross_score >= self.cross_encoder_threshold:
            best_entry.access_count += 1
            best_entry.last_accessed = datetime.now()
            
            logger.info(f"✓ CACHE HIT (Layer 2): High-confidence semantic match")
            logger.info(f"  - CrossEncoder score: {best_cross_score:.4f}")
            logger.info(f"  - Validation time: {validation_time:.3f}s")
            logger.info(f"  - KB Version: {best_entry.kb_version}")
            logger.info(f"  - Returning {len(best_entry.retrieved_chunks)} cached chunks")
            
            return best_entry.retrieved_chunks
        
        logger.info(f"✗ CACHE MISS (Layer 2): Best CrossEncoder score {best_cross_score:.4f} < threshold {self.cross_encoder_threshold}")
        logger.info(f"  - Validation time: {validation_time:.3f}s")
        return None
    
    def cache_exact_match(self, query: str, llm_response: str) -> None:
        """Cache a complete LLM response for exact query match"""
        if not self.current_kb_version:
            logger.warning("Cannot cache: No knowledge base version set")
            return
            
        cache_key: str = self._generate_cache_key(query)
        
        entry = ExactMatchCache(
            query_normalized=self.normalize_query(query),
            query_original=query,
            query_hash=cache_key,
            timestamp=datetime.now(),
            kb_version=self.current_kb_version.version_id,
            llm_response=llm_response,
            access_count=0
        )
        
        self.exact_cache[cache_key] = entry
        logger.info(f"✓ CACHED (Layer 1): Exact match response")
        logger.info(f"  - Query: '{query}'")
        logger.info(f"  - KB Version: {self.current_kb_version.version_id}")
        logger.info(f"  - Response length: {len(llm_response)} chars")
    
    
    
    def cache_embedding_result(self, query: str, query_embedding: np.ndarray,
                             retrieved_chunks: List[Dict[str, Any]], 
                             matryoshka_dim: int, retrieval_params: Dict[str, Any]):
        """Cache retrieved chunks with high-confidence validation"""
        if not self.current_kb_version:
            logger.warning("Cannot cache: No knowledge base version set")
            return
        
        # Validate that this is a high-quality result worth caching
        if len(retrieved_chunks) < 3:  # Don't cache poor retrievals
            logger.info("Skipping cache: Insufficient retrieved chunks")
            return
        
        # Calculate semantic confidence using CrossEncoder on top results
        top_chunk_text: str | Literal[""] = retrieved_chunks[0]["text"] if retrieved_chunks else ""
        raw_semantic_confidence: float = self.semantic_validator.predict([(query, top_chunk_text)])[0] if top_chunk_text else 0.0
        semantic_confidence: float = np.tanh(raw_semantic_confidence)
        
        # Only cache high-confidence results
        if semantic_confidence < self.cross_encoder_threshold:
            logger.info(f"Skipping cache: Low semantic confidence {semantic_confidence:.4f}")
            return
    
        # Normalize query for caching
        normalized_query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = normalized_query_embedding.astype('float32')

        cache_key: str = self._generate_cache_key(query)
        entry = EmbeddingCache(
            query_normalized=self.normalize_query(query),
            query_original=query,
            query_hash=cache_key,
            timestamp=datetime.now(),
            kb_version=self.current_kb_version.version_id,
            query_embedding=query_embedding,
            retrieved_chunks=retrieved_chunks.copy(),
            matryoshka_dim=matryoshka_dim,
            retrieval_params=retrieval_params.copy(),
            semantic_confidence=semantic_confidence,
            access_count=0
        )
        
        self.embedding_cache.append(entry)

        if self.enable_semantic_cache:
            try:
                if self.cached_embeddings_index is None:
                    # Initialize index if it's the first entry
                    dim = query_embedding.shape[0]
                    self.cached_embeddings_index = faiss.IndexFlatIP(dim)
                    self._next_faiss_id = 0

                # Add the embedding to the FAISS index
                query_embedding_f32 = query_embedding.reshape(1, -1)
                faiss_id = self._next_faiss_id
                self.cached_embeddings_index.add(query_embedding_f32)
                # Add to mapping
                self.cached_embedding_to_entry_map[faiss_id] = entry
                self._next_faiss_id += 1
                logger.debug(f"Added query embedding to FAISS cache index (ID: {faiss_id}).")
                logger.info(f"✓ CACHED (Layer 2): High-confidence embedding result")
                logger.info(f"  - Query: '{query}'")
                logger.info(f"  - Semantic confidence: {semantic_confidence:.4f}")
                logger.info(f"  - KB Version: {self.current_kb_version.version_id}")
                logger.info(f"  - Chunks: {len(retrieved_chunks)}")

            except Exception as e:
                logger.error(f"Failed to add embedding to FAISS cache index: {type(e).__name__}: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Cleanup if cache is too large
        estimated_new_entry_mb = self._estimate_entry_memory_mb(entry)
        self._estimated_embedding_cache_memory_mb += estimated_new_entry_mb
        self._manage_cache_memory()
        if len(self.embedding_cache) > self.max_cache_size:
            self.embedding_cache.sort(key=lambda x: x.last_accessed or x.timestamp)
            self.embedding_cache = self.embedding_cache[-self.max_cache_size:]


    def _estimate_entry_memory_mb(self, entry: EmbeddingCache) -> float:
        """Estimate memory usage of a single EmbeddingCache entry in MB."""
        # Estimate size of query_embedding (assuming float32)
        embedding_size_mb = (entry.query_embedding.size * 4) / (1024 * 1024)

        # Estimate size of retrieved_chunks
        chunks_size_bytes = 0
        # Iterate through the list of retrieved chunk dictionaries
        for chunk_dict in entry.retrieved_chunks:
            # Iterate through key-value pairs within each chunk dictionary
            for k, v in chunk_dict.items():
                # Roughly estimate size based on key and value string representations
                chunks_size_bytes += len(str(k).encode('utf-8')) # Size of key string
                if isinstance(v, str):
                    chunks_size_bytes += len(v.encode('utf-8')) # Size of string value
                elif isinstance(v, (int, float)):
                    chunks_size_bytes += 8 # Approximate size for numbers
                else:
                    # For other types (lists, dicts), estimate based on string representation
                    chunks_size_bytes += len(str(v).encode('utf-8'))

        chunks_size_mb = chunks_size_bytes / (1024 * 1024)

        # Add overhead for the dataclass structure, lists, dicts (rough estimate)
        overhead_mb = 0.001 # 1KB overhead per entry is a rough guess

        total_mb = embedding_size_mb + chunks_size_mb + overhead_mb
        # logger.debug(f"Estimated memory for entry: {total_mb:.4f} MB (Embedding: {embedding_size_mb:.4f} MB, Chunks: {chunks_size_mb:.4f} MB)")
        return total_mb

    def _manage_cache_memory(self) -> None:
        """Memory-aware cache management based on estimated total entry size."""
        if not self.embedding_cache:
            self._estimated_embedding_cache_memory_mb = 0.0
            return

        # Recalculate total estimated memory (could be optimized to be incremental)
        self._estimated_embedding_cache_memory_mb = sum(
            self._estimate_entry_memory_mb(entry) for entry in self.embedding_cache
        )

        # logger.debug(f"Total estimated embedding cache memory before management: {self._estimated_embedding_cache_mb:.2f} MB (Limit: {self.max_memory_mb} MB)")

        if self._estimated_embedding_cache_memory_mb <= self.max_memory_mb:
            return # Memory is within limit

        logger.info(f"Memory limit exceeded ({self._estimated_embedding_cache_memory_mb:.2f} MB > {self.max_memory_mb} MB). Initiating cache management...")

        # Sort entries by access count (ascending) and then by last accessed/timestamp (LRU)
        # Entries with lower access count and older access time are candidates for removal first.
        self.embedding_cache.sort(key=lambda x: (x.access_count, x.last_accessed or x.timestamp))

        removed_count = 0
        memory_released_mb = 0.0

        # Iterate and remove entries until memory is under the limit
        # Iterate in reverse order because we are removing from the list we are iterating over
        # Start from the least recently/frequently used (beginning of sorted list)
        for i in range(len(self.embedding_cache) - 1, -1, -1):
            entry = self.embedding_cache[i]
            entry_memory_mb = self._estimate_entry_memory_mb(entry)
            
            # logger.debug(f"Considering removal of entry '{entry.query_normalized[:30]}...' (Est. size: {entry_memory_mb:.4f} MB)")

            # Remove the entry
            removed_entry = self.embedding_cache.pop(i)
            removed_count += 1
            memory_released_mb += entry_memory_mb
            self._estimated_embedding_cache_memory_mb -= entry_memory_mb

            # logger.debug(f"Removed entry. Total memory now: {self._estimated_embedding_cache_memory_mb:.2f} MB")

            # Check if memory is now under the limit
            if self._estimated_embedding_cache_memory_mb <= self.max_memory_mb:
                break

        if removed_count > 0:
            logger.info(f"Memory management: Removed {removed_count} entries, estimated memory released: {memory_released_mb:.2f} MB. Current estimated memory: {self._estimated_embedding_cache_memory_mb:.2f} MB")
        else:
            logger.warning(f"Memory management: Could not reduce memory below limit ({self._estimated_embedding_cache_memory_mb:.2f} MB > {self.max_memory_mb} MB) by removing entries.")

    
    def _cleanup_expired(self) -> None:
        """Remove expired and invalid entries"""
        # Clean exact cache
        expired_keys = []
        for key, entry in self.exact_cache.items():
            if self._is_expired(entry) or not self._is_kb_version_valid(entry):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.exact_cache[key]
            
        # Clean embedding cache
        valid_entries = []
        for entry in self.embedding_cache:
            if not self._is_expired(entry) and self._is_kb_version_valid(entry):
                valid_entries.append(entry)
        
        removed_count = len(self.embedding_cache) - len(valid_entries)
        self.embedding_cache = valid_entries
        
        if expired_keys or removed_count:
            logger.info(f"Cleaned {len(expired_keys)} exact + {removed_count} embedding cache entries")

        if self.enable_semantic_cache and self.cached_embeddings_index:
             valid_entries_after_cleanup = set(self.embedding_cache) # Set of object references

             # Identify FAISS IDs to remove (those whose entries are no longer valid)
             ids_to_remove = []
             keys_to_remove_from_map = [] # Keys to remove from cached_embedding_to_entry_map
             for faiss_id, entry in self.cached_embedding_to_entry_map.items():
                 if entry not in valid_entries_after_cleanup:
                     ids_to_remove.append(faiss_id)
                     keys_to_remove_from_map.append(faiss_id)

             if ids_to_remove:
                 logger.info(f"Rebuilding FAISS index after removing {len(ids_to_remove)} expired/invalid entries.")
                 self._rebuild_cached_embeddings_index()

             elif keys_to_remove_from_map:
                 # If somehow IDs were removed from FAISS but map wasn't cleared (shouldn't happen with rebuild)
                 for key in keys_to_remove_from_map:
                     self.cached_embedding_to_entry_map.pop(key, None)
    
    def _load_cache(self) -> None:
        """Load cache from disk with KB version validation"""
        try:
            # Load exact cache
            if os.path.exists(self.exact_cache_file):
                with open(self.exact_cache_file, 'r') as f:
                    data = json.load(f)
                    for key, entry_dict in data.items():
                        entry_dict['timestamp'] = datetime.fromisoformat(entry_dict['timestamp'])
                        if entry_dict.get('last_accessed'):
                            entry_dict['last_accessed'] = datetime.fromisoformat(entry_dict['last_accessed'])
                        
                        entry = ExactMatchCache(**entry_dict)
                        # Only load if KB version is valid
                        if self._is_kb_version_valid(entry):
                            self.exact_cache[key] = entry
                
                logger.info(f"Loaded {len(self.exact_cache)} valid exact cache entries")
            
            # Load embedding cache
            if os.path.exists(self.embedding_cache_file):
                data = np.load(self.embedding_cache_file, allow_pickle=True)
                all_entries = data['embedding_cache'].tolist()
                
                # Filter by KB version
                valid_entries = [e for e in all_entries if self._is_kb_version_valid(e)]
                self.embedding_cache = valid_entries
                if self.enable_semantic_cache:
                    self._rebuild_cached_embeddings_index()

                self._estimated_embedding_cache_memory_mb = sum(
                    self._estimate_entry_memory_mb(entry) for entry in self.embedding_cache
                )
                
                logger.info(f"Loaded {len(self.embedding_cache)} valid embedding cache entries")
                logger.info(f"Recalculated estimated embedding cache memory after loading: {self._estimated_embedding_cache_memory_mb:.2f} MB")
                
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            self.exact_cache = {}
            self.embedding_cache = []
    
    def save_cache(self) -> None:
        """Save cache to disk"""
        try:
            # Clean expired entries before saving
            self._cleanup_expired()
            self._manage_cache_memory()
            
            # Save exact cache
            serializable_exact = {}
            for key, entry in self.exact_cache.items():
                entry_dict = asdict(entry)
                entry_dict['timestamp'] = entry.timestamp.isoformat()
                if entry.last_accessed:
                    entry_dict['last_accessed'] = entry.last_accessed.isoformat()
                serializable_exact[key] = entry_dict
            
            with open(self.exact_cache_file, 'w') as f:
                json.dump(serializable_exact, f, indent=2)
            
            # Save embedding cache
            np.savez_compressed(self.embedding_cache_file, 
                              embedding_cache=np.array(self.embedding_cache, dtype=object))
            
            logger.info(f"Cache saved: {len(self.exact_cache)} exact + {len(self.embedding_cache)} embedding entries")
            
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        self._cleanup_expired()
        embedding_memory_mb = self._estimated_embedding_cache_memory_mb
        return {
            'domain_type': self.domain_type,
            'performance_mode': self.performance_mode,
            'semantic_cache_enabled': self.enable_semantic_cache,
            'estimated_memory_mb': round(embedding_memory_mb, 2),
            'max_memory_mb': self.max_memory_mb,
            'exact_cache_size': len(self.exact_cache),
            'embedding_cache_size': len(self.embedding_cache),
            'total_exact_accesses': sum(entry.access_count for entry in self.exact_cache.values()),
            'total_embedding_accesses': sum(entry.access_count for entry in self.embedding_cache),
            'semantic_threshold': self.semantic_threshold,
            'cross_encoder_threshold': self.cross_encoder_threshold,
            'current_kb_version': self.current_kb_version.version_id if self.current_kb_version else None,
            'kb_created_at': self.current_kb_version.created_at.isoformat() if self.current_kb_version else None,
            'ttl_hours': self.ttl.total_seconds() / 3600
        }

