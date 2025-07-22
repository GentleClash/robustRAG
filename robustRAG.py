# Full runnable script with comprehensive descriptive logging for traceability

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
from typing import List, Dict, Any
from urllib.parse import urlparse

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

class LocalRAGRetriever:
    def __init__(self, embed_model_name: str, reranker_model_name: str, full_dim: int = 768):
        logger.info("=" * 60)
        logger.info("INITIALIZING LocalRAGRetriever")
        logger.info("=" * 60)
        
        start_time = time.time()
        logger.info(f"Configuration:")
        logger.info(f"  - Embedding model: {embed_model_name}")
        logger.info(f"  - Reranker model: {reranker_model_name}")
        logger.info(f"  - Full dimension: {full_dim}")
        
        logger.info("Loading embedding model...")
        embed_start = time.time()
        self.embed_model = SentenceTransformer(embed_model_name, trust_remote_code=True)
        embed_time = time.time() - embed_start
        logger.info(f"✓ Embedding model loaded in {embed_time:.2f}s")
        logger.info(f"  - Model max sequence length: {self.embed_model.max_seq_length}")
        logger.info(f"  - Model device: {self.embed_model.device}")
        
        logger.info("Loading reranker model...")
        rerank_start = time.time()
        self.reranker_model = CrossEncoder(reranker_model_name)
        rerank_time = time.time() - rerank_start
        logger.info(f"✓ Reranker model loaded in {rerank_time:.2f}s")
        
        self.full_dim = full_dim
        self.index = None
        self.chunks = []
        self.doc_embeddings = None
        
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
    
    @staticmethod
    def chunk_text(markdown_content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Splits the given text into chunks using RecursiveCharacterTextSplitter,
        counting lengths based on the Nomic-compatible tokenizer (bert-base-uncased).
        """
        if not markdown_content:
            return [] # Return an empty list for consistency if content is empty

        try:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            raise Exception("Please ensure you have the 'transformers' library installed: pip install transformers")

        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            is_separator_regex=False,
        )

        chunks = text_splitter.split_text(markdown_content)

        return chunks

    @staticmethod
    def count_tokens(text, model_name="bert-base-uncased") -> int:
        """
        Counts the number of tokens in a text string using a specified tokenizer.

        Args:
            text (str): The input text.
            model_name (str, optional): The name of the model to use for tokenization.


        Returns:
            int: The number of tokens in the text.
        """
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encoding = tokenizer.encode(text)
        return len(encoding)

    def _ingest_and_chunk(self, source_url: str,  local_filename: str = "source_document.pdf", chunk_size: int = 200, chunk_overlap: int = 0) -> List[str]:
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
            converter = DocumentConverter()
            result = converter.convert(local_path)
            markdown_content = result.document.export_to_markdown()
            
            parse_time = time.time() - parse_start
            content_length = len(markdown_content) if markdown_content else 0
            logger.info(f"✓ Document parsed successfully in {parse_time:.2f}s")
            logger.info(f"  - Extracted content length: {content_length:,} characters")
            logger.info(f"  - Content preview: {markdown_content[:200]}..." if markdown_content else "  - No content extracted")
            
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
        
        return chunks

    def _embed_documents(self, chunks: List[str]) -> np.ndarray:
        logger.info("\nSTAGE 3: DOCUMENT EMBEDDING")
        logger.info("-" * 40)
        
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
        
        return embeddings

    def build_index(self, source_url: str, local_filename: str, chunk_size: int = 200, chunk_overlap: int = 0):
        logger.info("\n" + "=" * 60)
        logger.info("BUILDING RAG INDEX")
        logger.info("=" * 60)
        
        total_start = time.time()
        
        # Ingest and chunk
        self.chunks = self._ingest_and_chunk(source_url, local_filename, chunk_size, chunk_overlap)
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
        
        logger.info("\n" + "=" * 60)
        logger.info(f"INDEX BUILD COMPLETE - Total time: {total_time:.2f}s")
        logger.info("=" * 60)

    def retrieve(self, query: str, matryoshka_dim: int, retrieve_k: int = 10, rerank_n: int = 5) -> List[Dict[str, Any]]:
        logger.info("\n" + "=" * 60)
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
        
        # Embed and process query
        full_embedding = self.embed_model.encode(prefixed_query, convert_to_tensor=True)
        logger.info(f"Full query embedding shape: {full_embedding.shape}")
        
        normalized_embedding = F.layer_norm(full_embedding, normalized_shape=(full_embedding.shape[0],))
        truncated_embedding = normalized_embedding[:matryoshka_dim].cpu().numpy()
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
        
        retrieved_chunks = [self.chunks[i] for i in indices[0]]

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
        reranked_results = []
        for i, (score, chunk) in enumerate(zip(scores, retrieved_chunks)):
            reranked_results.append({
                "rank": i + 1,
                "score": float(score),
                "text": chunk,
                "original_faiss_rank": int(indices[0][i]),
                "faiss_distance": float(distances[0][i])
            })
            
        reranked_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Update ranks after sorting
        for i, result in enumerate(reranked_results):
            result["final_rank"] = i + 1
        
        final_results = reranked_results[:rerank_n]
        
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

### End-to-End Demonstration with Enhanced Logging

if __name__ == "__main__":
    logger.info("\n" + "=" * 80)
    logger.info("STARTING RAG RETRIEVAL DEMONSTRATION")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    # Configuration
    EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    SOURCE_URL = "https://arxiv.org/pdf/2305.10601.pdf"  # Llama 2 Paper
    #SOURCE_URL = "documents/llama_2_paper.pdf"  # Local file for demo
    LOCAL_FILENAME = "llama_2_paper.pdf"

    logger.info(f"Demo configuration:")
    logger.info(f"  - Embedding model: {EMBED_MODEL}")
    logger.info(f"  - Reranker model: {RERANKER_MODEL}")
    logger.info(f"  - Source document: {SOURCE_URL}")
    logger.info(f"  - Local filename: {LOCAL_FILENAME}")

    # Initialize retriever
    retriever = LocalRAGRetriever(embed_model_name=EMBED_MODEL, reranker_model_name=RERANKER_MODEL)
    
    # Build index
    retriever.build_index(source_url=SOURCE_URL, local_filename=LOCAL_FILENAME)

    # Execute query
    if retriever.index:
        user_query = "What is task setup in game of 24?"
        matryoshka_dimension = 768
        
        logger.info(f"\nExecuting user query: '{user_query}'")
        final_results = retriever.retrieve(
            query=user_query, 
            matryoshka_dim=matryoshka_dimension,
            retrieve_k=10,
            rerank_n=5
        )
        
        # Display final results
        logger.info("\n" + "=" * 80)
        logger.info("FINAL RESULTS")
        logger.info("=" * 80)
        
        for i, result in enumerate(final_results):
            logger.info(f"\nRANK {result['final_rank']} (Score: {result['score']:.4f})")
            logger.info(f"Original FAISS rank: {result['original_faiss_rank']}")
            logger.info(f"FAISS distance: {result['faiss_distance']:.4f}")
            logger.info(f"Text length: {len(result['text'])} characters")
            logger.info(f"Content:\n{result['text']}")
            logger.info("-" * 80)
    else:
        logger.error("Failed to build index - cannot execute query")
    
    logger.info("\n" + "=" * 80)
    logger.info("DEMONSTRATION COMPLETE")
    logger.info("=" * 80)
