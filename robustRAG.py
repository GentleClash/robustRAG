import json
import os
import requests
import numpy as np
import faiss
import torch.nn.functional as F
from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Literal, Optional
from urllib.parse import urlparse
from typing import List, Dict, Any
from layeredCache import HighConfidenceCacheManager
import xxhash
from tqdm import tqdm
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('rag_retriever.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    embed_model: str = "nomic-ai/nomic-embed-text-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chunk_size: int = 200
    chunk_overlap: int = 0
    full_dim: int = 768
    retrieve_k: int = 10
    rerank_n: int = 5
    enable_caching: bool = True
    cache_performance_mode: Literal["speed", "balanced", "safety"] = "balanced"

class LocalRAGRetriever:
    def __init__(self, config: RAGConfig = None,
                 cache_manager: Optional[HighConfidenceCacheManager] = None):

        self.config = config or RAGConfig()
        cache_manager: HighConfidenceCacheManager = cache_manager

        logger.info("=" * 60)
        logger.info("INITIALIZING LocalRAGRetriever")
        logger.info("=" * 60)
        
        start_time = time.time()

        logger.info(f"Configuration:")
        logger.info(f"  - Embedding model: {self.config.embed_model}")
        logger.info(f"  - Reranker model: {self.config.reranker_model}")
        logger.info(f"  - Full dimension: {self.config.full_dim}")
        logger.info(f"  - Chunk size: {self.config.chunk_size} tokens")
        logger.info(f"  - Chunk overlap: {self.config.chunk_overlap} tokens")
        logger.info(f"  - Retrieve top-k: {self.config.retrieve_k}")
        logger.info(f"  - Rerank top-n: {self.config.rerank_n}")
        logger.info(f"  - Caching enabled: {self.config.enable_caching}")
        
        logger.info("Loading embedding model...")
        embed_start = time.time()
        self.embed_model = SentenceTransformer(self.config.embed_model, trust_remote_code=True)
        self._tokenizer: Optional[AutoTokenizer] = None 
        embed_time = time.time() - embed_start
        logger.info(f"✓ Embedding model loaded in {embed_time:.2f}s")
        logger.info(f"  - Model max sequence length: {self.embed_model.max_seq_length}")
        logger.info(f"  - Model device: {self.embed_model.device}")
        
        logger.info("Loading reranker model...")
        rerank_start = time.time()
        self.reranker_model = CrossEncoder(self.config.reranker_model)
        rerank_time = time.time() - rerank_start
        logger.info(f"✓ Reranker model loaded in {rerank_time:.2f}s")

        self.full_dim: int = self.config.full_dim
        self.index = None #Stores FAISS L2 index
        self.chunks: List[str] = []
        self.doc_embeddings: np.ndarray = None
        self.chunk_metadata: List[Dict[str, Any]] = []
        self.cache_manager = cache_manager
        self.enable_caching = self.config.enable_caching

        if self.enable_caching and self.cache_manager:
            logger.info("Cache manager enabled for retrieval acceleration")

        total_time = time.time() - start_time
        logger.info(f"✓ LocalRAGRetriever initialization complete in {total_time:.2f}s")
        logger.info("=" * 60)
    
    @staticmethod
    def is_url(text: str) -> bool:
        try:
            result = urlparse(text)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False
        
    @property
    def tokenizer(self) -> AutoTokenizer :
        """Lazy-load and cache the tokenizer."""
        if self._tokenizer is None:
            try:
                self._tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", trust_remote_code=True)
            except Exception as e:
                logger.error(f"Error loading tokenizer: {e}")
                raise Exception("Please ensure you have the 'transformers' library installed: pip install transformers")
        return self._tokenizer

    def chunk_text(self, markdown_content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Splits the given text into chunks using RecursiveCharacterTextSplitter,
        counting lengths based on the Nomic-compatible tokenizer (bert-base-uncased).
        """
        if not markdown_content:
            return [] # Return an empty list for consistency if content is empty

        logger.info(f"Chunking configuration:")
        logger.info(f"  - Chunk size: {chunk_size} tokens")
        logger.info(f"  - Chunk overlap: {chunk_overlap} tokens")
        logger.info(f"  - Total content length: {len(markdown_content):,} characters")
        
        # Create text splitter with cached tokenizer
        logger.info("Initializing text splitter...")
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            is_separator_regex=False,
        )
        
        # Split with progress bar
        logger.info("Splitting text into chunks...")
        chunk_start = time.time()
        
        # For very large texts, show progress during splitting
        with tqdm(desc="Chunking text", unit="chars", total=len(markdown_content)) as pbar:
            chunks = text_splitter.split_text(markdown_content)
            pbar.update(len(markdown_content))
        
        chunk_time = time.time() - chunk_start
        
        if chunks:
            logger.info(f"✓ Text chunked successfully in {chunk_time:.3f}s")
            logger.info(f"  - Total chunks created: {len(chunks)}")
            logger.info(f"  - Processing rate: {len(markdown_content)/chunk_time:,.0f} chars/second")
            
            sample_size = len(chunks)
            sample_chunks = chunks[:sample_size]
            
            logger.info("Calculating chunk statistics (sampling)...")
            with tqdm(desc="Analyzing chunks", total=sample_size) as pbar:
                sample_token_counts = []
                for chunk in sample_chunks:
                    token_count = self.count_tokens(chunk)
                    sample_token_counts.append(token_count)
                    pbar.update(1)
            
            if sample_token_counts:
                avg_tokens = sum(sample_token_counts) / len(sample_token_counts)
                min_tokens = min(sample_token_counts)
                max_tokens = max(sample_token_counts)
                
                logger.info(f"  - Sample chunk statistics ({sample_size} chunks):")
                logger.info(f"    • Average tokens: {avg_tokens:.1f}")
                logger.info(f"    • Token range: {min_tokens} - {max_tokens}")
                logger.info(f"    • Estimated total tokens: {avg_tokens * len(chunks):,.0f}")
        else:
            logger.warning("No chunks were created")
        
        return chunks


    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in a text string using a specified tokenizer.

        Args:
            text (str): The input text.
            model_name (str, optional): The name of the model to use for tokenization.


        Returns:
            int: The number of tokens in the text.
        """
        
        if not text:
            return 0
        encoding = self.tokenizer.encode(text, add_special_tokens=False)

        return len(encoding)
    
    @staticmethod
    def get_file_hash(file_path: str) -> str:
        """
        Computes a hash for the file content using xxHash for fast and efficient hashing.
        
        Args:
            file_path (str): Path to the file to hash.
        
        Returns:
            str: Hexadecimal hash of the file content.
        """
        hasher = xxhash.xxh64()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()


    def _ingest_and_chunk(self, source_url: str,  
                          local_filename: str = "source_document.pdf", 
                          chunk_size: int = 200, 
                          chunk_overlap: int = 0) -> List[str]:
        logger.info("STAGE 1: DOCUMENT INGESTION")
        logger.info("-" * 40)
        
        # Setup document directory
        doc_dir = "documents"
        os.makedirs(doc_dir, exist_ok=True)
        local_path = os.path.join(doc_dir, local_filename)

        logger.info(f"Document directory: {os.path.abspath(doc_dir)}")
        #logger.info(f"Target file path: {os.path.abspath(local_path)}")

        # Download document if needed
        if self.is_url(source_url):
            logger.info(f"Document not found locally, downloading from: {source_url}")
            download_start = time.time()
            
            try:
                response = requests.get(source_url, stream=True)
                response.raise_for_status()
                
                file_size = int(response.headers.get('content-length', 0))
                logger.info(f"Document size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
                
                downloaded_bytes = 0
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded_bytes += len(chunk)
                        
                download_time = time.time() - download_start
                logger.info(f"✓ Document downloaded successfully in {download_time:.2f}s")
                logger.info(f"  - Downloaded: {downloaded_bytes:,} bytes")
                logger.info(f"  - Average speed: {downloaded_bytes/1024/1024/download_time:.2f} MB/s")
                
            except requests.exceptions.RequestException as e:
                logger.error(f"✗ Error downloading document: {e}")
                return []
        else:
            # Check if source url is a local file
            if not os.path.exists(source_url):
                logger.error(f"✗ Local file not found: {source_url}")
                return []

            file_size = os.path.getsize(source_url)
            logger.info(f"Using existing document ({file_size:,} bytes)")
            local_path = source_url  # Use the local file directly
        
        # Parse document with Docling
        logger.info("Parsing document with Docling...")
        parse_start = time.time()
        
        try:
            os.makedirs("cache", exist_ok=True)

            file_hash: str = self.get_file_hash(local_path)
            
            parsed_cache_file = os.path.join("cache", f"{file_hash}_parsed.json")
            chunks_cache_file = os.path.join("cache", f"{file_hash}_chunks_{chunk_size}_{chunk_overlap}.json")

            # Check parsed content cache first
            if os.path.exists(parsed_cache_file):
                logger.info(f"✓ Parsed content cache hit: {parsed_cache_file}")
                with open(parsed_cache_file, "r") as f:
                    cached_parsed = json.load(f)
                markdown_content = cached_parsed.get("content", "")
            else:
                # Parse document and cache result
                converter = DocumentConverter()
                result = converter.convert(local_path)
                markdown_content = result.document.export_to_markdown()
                
                # Cache parsed content
                with open(parsed_cache_file, "w") as f:
                    json.dump({"content": markdown_content, "timestamp": datetime.now().isoformat()}, f)
                logger.info(f"✓ Parsed content cached at: {parsed_cache_file}")

            # Check chunks cache
            if os.path.exists(chunks_cache_file):
                logger.info(f"✓ Chunks cache hit: {chunks_cache_file}")
                with open(chunks_cache_file, "r") as f:
                    cached_chunks = json.load(f)
                return cached_chunks.get("chunks", [])
        
            parse_end = time.time() - parse_start
            logger.info(f"✓ Document parsed successfully in {parse_end:.2f}s") 
        except Exception as e:
            logger.error(f"✗ Error parsing document: {e}")
            return []

        # Chunking
        logger.info("\nSTAGE 2: TEXT CHUNKING")
        logger.info("-" * 40)
        
        if not markdown_content:
            logger.warning("No content to chunk")
            return []
            
        logger.info(f"Chunking configuration:")
        logger.info(f"  - Chunk size: {chunk_size} tokens")
        logger.info(f"  - Chunk overlap: {chunk_overlap} tokens")
        logger.info(f"  - Total content length: {len(markdown_content):,} characters")
        
        chunk_start = time.time()
        chunks = self.chunk_text(markdown_content, chunk_size, chunk_overlap)
        chunk_time = time.time() - chunk_start
        
        if chunks:
            avg_chunk_size = sum(self.count_tokens(chunk) for chunk in chunks) / len(chunks)
            min_chunk_size = min(self.count_tokens(chunk) for chunk in chunks)
            max_chunk_size = max(self.count_tokens(chunk) for chunk in chunks)

            logger.info(f"✓ Text chunked successfully in {chunk_time:.3f}s")
            logger.info(f"  - Total chunks created: {len(chunks)}")
            logger.info(f"  - Average chunk size: {avg_chunk_size:.1f} characters")
            logger.info(f"  - Size range: {min_chunk_size} - {max_chunk_size} characters")
            logger.info(f"  - Processing rate: {len(markdown_content)/chunk_time:,.0f} chars/second")
        else:
            logger.warning("No chunks were created")
        
        # Cache the chunks
        with open(chunks_cache_file, "w") as f:
            json.dump({
                "chunks": chunks,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "timestamp": datetime.now().isoformat()
            }, f)
        logger.info(f"✓ Chunks cached at: {chunks_cache_file}")
        
        return chunks

    def _embed_documents(self, chunks: List[str]) -> np.ndarray:
        logger.info("\nSTAGE 3: DOCUMENT EMBEDDING")
        logger.info("-" * 40)

        # Check embeddings cache
        if hasattr(self, '_document_hash') and self._document_hash:
            embeddings_cache_file = os.path.join("cache", f"{self._document_hash}_embeddings_{self.full_dim}.npz")
            
            if os.path.exists(embeddings_cache_file):
                logger.info(f"✓ Embeddings cache hit: {embeddings_cache_file}")
                cached_data = np.load(embeddings_cache_file)
                return cached_data['embeddings']
        
        if not chunks:
            logger.warning("No chunks to embed")
            return np.array([])
            
        logger.info(f"Embedding configuration:")
        logger.info(f"  - Number of chunks: {len(chunks)}")
        logger.info(f"  - Target dimension: {self.full_dim}")
        logger.info(f"  - Model device: {self.embed_model.device}")
        
        embed_start = time.time()
        
        # Add prefixes and log sample
        prefixed_chunks = [f"search_document: {chunk}" for chunk in chunks]
        logger.info(f"Sample prefixed chunk: {prefixed_chunks[0][:100]}...")
        
        # Batch embedding with progress tracking
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(prefixed_chunks), batch_size):
            batch_start = time.time()
            batch = prefixed_chunks[i:i+batch_size]
            batch_embeddings = self.embed_model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            all_embeddings.append(batch_embeddings)
            
            batch_time = time.time() - batch_start
            progress = (i + len(batch)) / len(prefixed_chunks) * 100
            logger.info(f"  Batch {i//batch_size + 1}: {len(batch)} chunks embedded in {batch_time:.2f}s [{progress:.1f}%]")
        
        embeddings = np.vstack(all_embeddings)
        embed_time = time.time() - embed_start
        
        logger.info(f"✓ Document embedding complete in {embed_time:.2f}s")
        logger.info(f"  - Final embedding shape: {embeddings.shape}")
        logger.info(f"  - Processing rate: {len(chunks)/embed_time:.1f} chunks/second")
        logger.info(f"  - Memory usage: {embeddings.nbytes/1024/1024:.2f} MB")

        # Cache the embeddings
        if hasattr(self, '_document_hash') and self._document_hash:
            embeddings_cache_file = os.path.join("cache", f"{self._document_hash}_embeddings_{self.full_dim}.npz")
            np.savez_compressed(embeddings_cache_file, 
                            embeddings=embeddings,
                            timestamp=datetime.now().isoformat())
            logger.info(f"✓ Embeddings cached at: {embeddings_cache_file}")
        
        return embeddings

    def build_index(self, sources: Optional[List[str]], 
                    local_filenames: Optional[List[str]], 
                    chunk_size: int = 200, 
                    chunk_overlap: int = 0) -> None:
        """
            Build a RAG index from documents with flexible input handling.

        Args:
            sources (List[str], optional): List of URLs or local file paths to process.
                If None and local_filenames provided, will look for files in documents/ dir.
                If None and local_filenames is None, will process all files in documents/ dir.
            
            local_filenames (List[str], optional): List of desired local filenames for downloaded/cached files.
                If shorter than sources, missing filenames will be auto-generated.
                If None, filenames will be extracted from URLs or source paths.
            
            chunk_size (int, default=200): Size of text chunks for splitting documents.
            
            chunk_overlap (int, default=0): Number of overlapping characters between chunks.
        
        Examples:
            **Process all documents in documents/ directory**
            build_index()

            **Process specific URLs with auto-generated filenames**
            build_index(sources=['http://example.com/doc.pdf', 'http://example.com/doc2.pdf'])

            **Process URLs with custom filenames**
            build_index(sources=['http://example.com/doc.pdf'], local_filenames=['my_doc.pdf'])

            **Process local files only**
            build_index(local_filenames=['doc1.pdf', 'doc2.pdf'])
        """

        logger.info("=" * 60)
        logger.info("BUILDING RAG INDEX")
        logger.info("=" * 60)
        
        # Handle input normalization and edge cases
        if isinstance(sources, str):
            sources = [sources]  # Backward compatibility

        # Neither sources nor local_filenames provided then look for all documents in "documents/" directory
        if sources is None and local_filenames is None:
            doc_dir = "documents"
            if os.path.exists(doc_dir):
                local_filenames = [f for f in os.listdir(doc_dir) if f.lower().endswith(('.pdf', '.txt', '.md', '.docx'))]
                if local_filenames:
                    sources = [os.path.join(doc_dir, f) for f in local_filenames]
                    logger.info(f"No sources provided, found {len(sources)} documents in {doc_dir}/")
                else:
                    logger.error("No sources provided and no documents found in documents/ directory")
                    return
            else:
                logger.error("No sources provided and documents/ directory doesn't exist")
                return
            
        # Only local_filenames provided, look for files in documents/ directory
        elif sources is None and local_filenames is not None:
            if isinstance(local_filenames, str):
                local_filenames = [local_filenames]
            
            doc_dir = "documents"
            sources = []
            for filename in local_filenames:
                # Check if it's already a full path
                if os.path.exists(filename):
                    sources.append(filename)
                # Otherwise look in documents directory
                elif os.path.exists(os.path.join(doc_dir, filename)):
                    sources.append(os.path.join(doc_dir, filename))
                else:
                    logger.error(f"Local file not found: {filename}")
                    return
            logger.info(f"Using {len(sources)} local files")
        
        # If sources provided, ensure they are valid URLs or local paths
        else:
            if local_filenames is None:
                local_filenames = []
                for i, source in enumerate(sources):
                    if self.is_url(source):
                        # Extract filename from URL or use default
                        try:
                            filename = source.split('/')[-1]
                            if not filename or '.' not in filename:
                                filename = f"document_{i}.pdf"
                        except:
                            filename = f"document_{i}.pdf"
                    else:
                        # Use the original filename for local files
                        filename = os.path.basename(source)
                    local_filenames.append(filename)
                logger.info(f"Auto-generated {len(local_filenames)} filenames for sources")
            
            elif len(local_filenames) < len(sources):
                # Extend local_filenames to match sources length
                for i in range(len(local_filenames), len(sources)):
                    source = sources[i]
                    if self.is_url(source):
                        try:
                            filename = source.split('/')[-1]
                            if not filename or '.' not in filename:
                                filename = f"document_{i}.pdf"
                        except:
                            filename = f"document_{i}.pdf"
                    else:
                        filename = os.path.basename(source)
                    local_filenames.append(filename)
                logger.info(f"Extended filenames list to match {len(sources)} sources")
            
            elif len(local_filenames) > len(sources):
                logger.warning(f"More filenames ({len(local_filenames)}) than sources ({len(sources)}), truncating filenames")
                local_filenames = local_filenames[:len(sources)]
        
        # Validate final state
        if len(sources) != len(local_filenames):
            raise ValueError(f"Mismatch: {len(sources)} sources vs {len(local_filenames)} filenames")
        
        logger.info(f"Processing {len(sources)} documents:")
        for i, (source, filename) in enumerate(zip(sources, local_filenames)):
            source_type = "URL" if self.is_url(source) else "Local"
            logger.info(f"  {i+1}. {source_type}: {source} -> {filename}")
        
        total_start = time.time()
        
        # Ingest and chunk
        all_chunks: List[str] = []
        all_doc_metadata: List[Dict[str, Any]] = []
        document_hashes: List[str] = []
        
        for i, (source, filename) in enumerate(zip(sources, local_filenames)):
            logger.info(f"\nProcessing document {i+1}/{len(sources)}: {source}")
            doc_chunks: List[str] = self._ingest_and_chunk(source, filename, chunk_size, chunk_overlap)
            
            # Add document metadata to each chunk
            for chunk_idx, chunk in enumerate(doc_chunks):
                all_chunks.append(chunk)
                all_doc_metadata.append({
                    'doc_id': i,
                    'doc_source': source,
                    'doc_filename': filename,
                    'chunk_id': chunk_idx,
                    'global_chunk_id': len(all_chunks) - 1
                })
        
            # Store document hash for caching
            if self.is_url(source):
                local_path = os.path.join("documents", filename)
                if os.path.exists(local_path):
                    file_hash: str | None = self.get_file_hash(local_path)
                    document_hashes.append(file_hash)
                else:
                    logger.warning(f"Downloaded file not found for hash calculation: {local_path}")
                    document_hashes.append("missing_file")
            else:
                file_hash: str = self.get_file_hash(os.path.join("documents", filename))
                document_hashes.append(file_hash)
            
        composite_hash_input = {
            "document_hashes" : sorted(document_hashes),  
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'total_chunks': len(all_chunks)
        }
        composite_hash_str = json.dumps(composite_hash_input, sort_keys=True)
        self._document_hash = xxhash.xxh64(composite_hash_str.encode()).hexdigest()
        
        self.chunks = all_chunks
        self.chunk_metadata = all_doc_metadata
        logger.info(f"Combined total: {len(self.chunks)} chunks from {len(sources)} documents")

        # Update cache manager with actual chunk count
        if self.cache_manager:
            self.cache_manager.update_kb_version(
                document_paths=[os.path.join("documents", fn) for fn in local_filenames],
                total_chunks=len(all_chunks),
                embedding_model=self.embed_model._modules['0'].auto_model.name_or_path if hasattr(self.embed_model, '_modules') else "unknown",
                chunk_params={'chunk_size': chunk_size, 'chunk_overlap': chunk_overlap}
            )
        
        if not self.chunks:
            logger.error("✗ No chunks were created. Aborting index build.")
            return

        # Embed documents
        self.doc_embeddings = self._embed_documents(self.chunks)
        
        # Build FAISS index
        logger.info("\nSTAGE 4: BUILDING FAISS INDEX")
        logger.info("-" * 40)
        
        if self.doc_embeddings.size == 0:
            logger.error("✗ No embeddings generated. Aborting index build.")
            return
            
        index_start = time.time()
        d = self.doc_embeddings.shape[1]
        
        logger.info(f"FAISS index configuration:")
        logger.info(f"  - Vector count: {self.doc_embeddings.shape[0]:,}")
        logger.info(f"  - Vector dimension: {d}")
        logger.info(f"  - Index type: IndexFlatL2 (exact search)")
        logger.info(f"  - Memory requirement: ~{self.doc_embeddings.nbytes/1024/1024:.2f} MB")
        
        self.index = faiss.IndexFlatL2(d)
        self.index.add(self.doc_embeddings.astype('float32'))
        
        index_time = time.time() - index_start
        total_time = time.time() - total_start
        
        logger.info(f"✓ FAISS index built successfully in {index_time:.3f}s")
        logger.info(f"  - Index size: {self.index.ntotal:,} vectors")
        logger.info(f"  - Index is trained: {self.index.is_trained}")
        
        logger.info("=" * 60)
        logger.info(f"INDEX BUILD COMPLETE - Total time: {total_time:.2f}s")
        logger.info("=" * 60)

    def retrieve(self, query: str, matryoshka_dim: int, retrieve_k: int = 10, rerank_n: int = 5) -> List[Dict[str, Any]]:
        
        if self.enable_caching and self.cache_manager:
            exact_result = self.cache_manager.get_exact_match(query)
            if exact_result:
                logger.info("Returning cached exact match response")
                return json.loads(exact_result) if exact_result.startswith('[') else []
            
        logger.info("=" * 60)
        logger.info("EXECUTING RETRIEVAL PIPELINE")
        logger.info("=" * 60)
        
        if self.index is None:
            logger.error("✗ Index not built. Please call build_index() first.")
            return []

        retrieval_start = time.time()
        
        logger.info(f"Retrieval configuration:")
        logger.info(f"  - Query: '{query}'")
        logger.info(f"  - Matryoshka dimension: {matryoshka_dim}")
        logger.info(f"  - Retrieve top-k: {retrieve_k}")
        logger.info(f"  - Rerank top-n: {rerank_n}")

        # Query embedding and truncation
        logger.info("\nSTAGE 5-6: QUERY EMBEDDING & SEARCH")
        logger.info("-" * 40)
        
        query_start = time.time()
        prefixed_query = f"search_query: {query}"
        logger.info(f"Prefixed query: '{prefixed_query[:100]}...'")

        full_embedding = self.embed_model.encode(prefixed_query, convert_to_tensor=True)
        normalized_embedding = F.layer_norm(full_embedding, normalized_shape=(full_embedding.shape[0],))
        normalized_embedding_np = normalized_embedding.cpu().numpy()
        query_embedding = normalized_embedding_np

        if self.enable_caching and self.cache_manager:

            # Check semantic cache
            cached_chunks = self.cache_manager.get_similar_chunks(query, query_embedding, matryoshka_dim)
            if cached_chunks:
                logger.info("Returning cached semantic match results")
                return cached_chunks[:rerank_n]
    
        
        logger.info(f"Full query embedding shape: {full_embedding.shape}")
        
        truncated_embedding = normalized_embedding_np[:matryoshka_dim]
        logger.info(f"Truncated embedding shape: {truncated_embedding.shape}")

        # Pad to match index dimension
        padded_query = np.zeros(self.full_dim, dtype='float32')
        padded_query[:len(truncated_embedding)] = truncated_embedding
        query_vector = np.expand_dims(padded_query, axis=0)
        
        query_time = time.time() - query_start
        logger.info(f"✓ Query processed in {query_time:.3f}s")
        logger.info(f"  - Final query vector shape: {query_vector.shape}")
        
        # FAISS search
        search_start = time.time()
        distances, indices = self.index.search(query_vector, retrieve_k)
        search_time = time.time() - search_start
        
        logger.info(f"✓ FAISS search completed in {search_time:.3f}s")
        logger.info(f"  - Retrieved {len(indices[0])} results")
        logger.info(f"  - Distance range: {distances[0].min():.4f} - {distances[0].max():.4f}")
        
        retrieved_chunks: List[str] = []
        chunk_metadata: List[Dict[str, Any]] = []
        for i in indices[0]:
            retrieved_chunks.append(self.chunks[i])
            if hasattr(self, 'chunk_metadata'):
                chunk_metadata.append(self.chunk_metadata[i])
            else:
                chunk_metadata.append({'doc_id': 0, 'doc_source': 'unknown'})

        # Reranking
        logger.info("\nSTAGE 7: CROSS-ENCODER RERANKING")
        logger.info("-" * 40)
        
        rerank_start = time.time()
        logger.info(f"Creating {len(retrieved_chunks)} query-document pairs for reranking")
        
        pairs = [(query, chunk) for chunk in retrieved_chunks]
        scores = self.reranker_model.predict(pairs)
        
        rerank_time = time.time() - rerank_start
        logger.info(f"✓ Reranking completed in {rerank_time:.3f}s")
        logger.info(f"  - Score range: {scores.min():.4f} - {scores.max():.4f}")
        logger.info(f"  - Processing rate: {len(pairs)/rerank_time:.1f} pairs/second")
        
        # Compile and sort results
        reranked_results: List[Dict[str, Any]] = []
        for i, (score, chunk, metadata) in enumerate(zip(scores, retrieved_chunks, chunk_metadata)):
            reranked_results.append({
                "rank": i + 1,
                "score": float(score),
                "text": chunk,
                "source_document": metadata.get('doc_source', 'unknown'),
                "document_id": metadata.get('doc_id', 0),
                "chunk_id": metadata.get('chunk_id', 0),
                "faiss_distance": float(distances[0][i])
            })
            
        reranked_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Update ranks after sorting
        for i, result in enumerate(reranked_results):
            result["final_rank"] = i + 1
        
        final_results = reranked_results[:rerank_n]

        if self.enable_caching and self.cache_manager:
            # Cache exact match (full response)
            cache_response = json.dumps(final_results, indent=2)
            self.cache_manager.cache_exact_match(query, cache_response)
            
            # Cache embedding result for semantic similarity
            retrieval_params = {
                'matryoshka_dim': matryoshka_dim,
                'retrieve_k': retrieve_k,
                'rerank_n': rerank_n,
                'embed_model': type(self.embed_model).__name__,
                'reranker_model': type(self.reranker_model).__name__
            }

            self.cache_manager.cache_embedding_result(
                query, query_embedding, final_results, matryoshka_dim, retrieval_params
            )

        total_retrieval_time = time.time() - retrieval_start
        
        logger.info(f"\n✓ RETRIEVAL PIPELINE COMPLETE in {total_retrieval_time:.3f}s")
        logger.info(f"  - Query processing: {query_time:.3f}s")
        logger.info(f"  - FAISS search: {search_time:.3f}s") 
        logger.info(f"  - Reranking: {rerank_time:.3f}s")
        logger.info(f"  - Returned top {len(final_results)} results")
        
        # Log top results summary
        logger.info(f"\nTOP RESULTS SUMMARY:")
        for result in final_results:
            logger.info(f"  Rank {result['final_rank']}: Score={result['score']:.4f}, "
                       f"Text='{result['text'][:60]}...'")
        
        return final_results

### ==========Demonstration and Testing Functions========
### Demo and interactive functionality for testing the retriever

def setup_retriever(enable_cache: bool = True, cache_performance_mode: str = "balanced", cache_semantic: bool = True) -> LocalRAGRetriever:
    """Initializes the RAG retriever and cache manager."""
    logger.info("=" * 80)
    logger.info("INITIALIZING RAG RETRIEVER")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    # Configuration
    EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # SOURCE_URL = "https://arxiv.org/pdf/2305.10601.pdf"  # Llama 2 Paper
    #SOURCE_URL = "documents/llama_2_paper.pdf"  # Local file for demo
    #LOCAL_FILENAME = "llama_2_paper.pdf"
    LOCAL_FILENAME = None
    SOURCE_URL = ["https://hackrx.in/policies/ICIHLIP22012V012223.pdf", \
                  "https://hackrx.in/policies/EDLHLGA23009V012223.pdf", 
                  ]

    cache_manager = None
    if enable_cache:
        cache_manager = HighConfidenceCacheManager(
            domain_type="general",
            performance_mode=cache_performance_mode, # Use parameter
            enable_semantic_cache=cache_semantic # Use parameter
        )

    logger.info(f"Demo configuration:")
    logger.info(f"  - Embedding model: {EMBED_MODEL}")
    logger.info(f"  - Reranker model: {RERANKER_MODEL}")
    logger.info(f"  - Source document(s): {SOURCE_URL}")
    logger.info(f"  - Local filename(s): {LOCAL_FILENAME}")
    logger.info(f"  - Caching enabled: {enable_cache}")
    if enable_cache:
        logger.info(f"    - Cache Performance Mode: {cache_performance_mode}")
        logger.info(f"    - Semantic Cache Enabled: {cache_semantic}")

    # Initialize retriever
    retriever = LocalRAGRetriever(config=RAGConfig(),
                                 cache_manager=cache_manager)
    return retriever, SOURCE_URL, [LOCAL_FILENAME] 

def build_knowledge_base(retriever: LocalRAGRetriever, sources: List[str], filenames: List[str]) -> bool:
    """Builds the RAG index from sources."""
    logger.info("=" * 80)
    logger.info("BUILDING KNOWLEDGE BASE")
    logger.info("=" * 80)
    retriever.build_index(sources=sources, local_filenames=filenames)
    success = retriever.index is not None
    if success:
        logger.info("✓ Knowledge base built successfully.")
    else:
        logger.error("✗ Failed to build knowledge base.")
    return success

def run_single_demo_query(retriever: LocalRAGRetriever, matryoshka_dim: int = 768) -> None:
    """Runs a predefined demo query."""
    if not retriever.index:
        logger.error("Cannot run demo query: Index not built.")
        return

    user_query = "What is task setup in game of 24?"
    logger.info(f"Executing demo query: '{user_query}'")
    overall_query_start = time.time()

    final_results = retriever.retrieve(
        query=user_query,
        matryoshka_dim=matryoshka_dim,
        retrieve_k=10,
        rerank_n=5
    )

    overall_query_time = time.time() - overall_query_start
    logger.info(f"Demo query executed in {overall_query_time:.2f} seconds")

    if final_results:
        logger.info("=" * 80)
        logger.info("DEMO QUERY RESULTS")
        logger.info("=" * 80)
        for i, result in enumerate(final_results):
            logger.info(f"RANK {result['final_rank']} (Score: {result['score']:.4f})")
            logger.info(f"FAISS distance: {result['faiss_distance']:.4f}")
            logger.info(f"Text length: {len(result['text'])} characters")
            # Truncate display text for log readability
            display_text = result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
            logger.info(f"Content: {display_text}")
            logger.info("-" * 80)
    else:
        logger.info("No results found for the demo query.")

def run_interactive_mode(retriever: LocalRAGRetriever, matryoshka_dim: int = 768):
    """Runs the interactive query loop."""
    logger.info("=" * 80)
    logger.info("ENTERING INTERACTIVE MODE")
    logger.info("Type 'exit' to quit.")
    logger.info("=" * 80)

    while True:
        try:
            user_query = input("\nEnter a query: ").strip()
            if user_query.lower() == 'exit':
                print("Exiting interactive mode.")
                break
            if not user_query:
                print("Empty query entered. Please provide a valid query.")
                continue

            print(f"\nExecuting user query: '{user_query}'")
            overall_query_start = time.time()
            final_results = retriever.retrieve(
                query=user_query,
                matryoshka_dim=matryoshka_dim,
                retrieve_k=10,
                rerank_n=5
            )
            overall_query_time = time.time() - overall_query_start
            print(f"\nQuery executed in {overall_query_time:.2f} seconds")

            if final_results:
                print("\n" + "=" * 80)
                print("FINAL RESULTS")
                print("=" * 80)
                for i, result in enumerate(final_results):
                    print(f"\nRANK {result['final_rank']} (Score: {result['score']:.4f})")
                    print(f"Source document: {result['source_document']}, Chunk ID: {result['chunk_id']}, Document ID: {result['document_id']}")
                    print(f"FAISS distance: {result['faiss_distance']:.4f}")
                    print(f"Text length: {len(result['text'])} characters")
                    # Truncate display text for console readability
                    display_text = result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
                    print(f"Content: {display_text}")
                    print("-" * 80)
            else:
                print("No results found for the query.")
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting interactive mode.")
            break
        except Exception as e:
            logger.error(f"An error occurred during query processing: {e}")
            print(f"An error occurred: {e}. Please try again.")

def report_cache_statistics(retriever: LocalRAGRetriever):
    """Reports cache statistics and saves the cache."""
    if hasattr(retriever, 'cache_manager') and retriever.cache_manager:
        logger.info("=" * 80)
        logger.info("CACHE STATISTICS & SAVING")
        logger.info("=" * 80)
        try:
            cache_stats = retriever.cache_manager.get_cache_stats()
            logger.info("CACHE STATISTICS:")
            logger.info(f"  - Performance mode: {cache_stats['performance_mode']}")
            logger.info(f"  - Semantic cache enabled: {cache_stats['semantic_cache_enabled']}")
            logger.info(f"  - Exact cache entries: {cache_stats['exact_cache_size']}")
            logger.info(f"  - Embedding cache entries: {cache_stats['embedding_cache_size']}")
            logger.info(f"  - Estimated memory usage: {cache_stats['estimated_memory_mb']} MB")
            logger.info(f"  - Total Exact Accesses: {cache_stats['total_exact_accesses']}")
            logger.info(f"  - Total Embedding Accesses: {cache_stats['total_embedding_accesses']}")
            logger.info(f"  - Current KB Version: {cache_stats['current_kb_version']}")

            # Save cache for next run
            retriever.cache_manager.save_cache()
            logger.info("✓ Cache saved for future runs")
        except Exception as e:
            logger.error(f"Error reporting/saving cache statistics: {e}")
    else:
        logger.info("Cache manager not enabled or not found.")

if __name__ == "__main__":
    # 1. Setup
    retriever, sources, filenames = setup_retriever(
        enable_cache=True,
        cache_performance_mode="balanced",
        cache_semantic=True
    )

    # 2. Build Knowledge Base
    if build_knowledge_base(retriever, sources=sources, filenames=None):
        # 3. (Optional) Run a single demo query
      #  run_single_demo_query(retriever, matryoshka_dim=768)

        # 4. Run Interactive Mode
        run_interactive_mode(retriever, matryoshka_dim=768)

    # 5. Report and Save Cache Stats
    report_cache_statistics(retriever)

    logger.info("=" * 80)
    logger.info("DEMONSTRATION COMPLETE")
    logger.info("=" * 80)

