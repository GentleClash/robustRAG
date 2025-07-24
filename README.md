# LocalRAG: Offline RAG System with Intelligent Caching

A high-performance, completely offline Retrieval-Augmented Generation (RAG) system featuring a 2.5-layer caching architecture for accelerated document retrieval and enhanced accuracy in critical domains.

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
   - [Core RAG Capabilities](#core-rag-capabilities)
   - [2.5-Layer Intelligent Caching System](#25-layer-intelligent-caching-system)
   - [Performance Optimization](#performance-optimization)
   - [Domain-Specific Safety](#domain-specific-safety)
3. [Architecture](#architecture)
   - [LocalRAGRetriever](#localragretriever-robustraggpy)
   - [HighConfidenceCacheManager](#highconfidencecachemanager-layeredcachepy)
4. [Configuration Options](#configuration-options)
   - [Embedding Configuration](#embedding-configuration)
   - [Caching Configuration](#caching-configuration)
   - [Retrieval Parameters](#retrieval-parameters)
5. [Performance Characteristics](#performance-characteristics)
   - [Speed Improvements](#speed-improvements)
   - [Accuracy Guarantees](#accuracy-guarantees)
   - [Resource Efficiency](#resource-efficiency)
6. [Getting Started](#getting-started)
   - [Installation](#installation)
   - [Basic Usage](#basic-usage)
7. [Technical Implementation](#technical-implementation)
   - [Caching Strategy](#caching-strategy)
   - [Knowledge Base Versioning](#knowledge-base-versioning)
   - [Memory Management](#memory-management)
8. [Monitoring and Statistics](#monitoring-and-statistics)
9. [Security and Privacy](#security-and-privacy)

---

## Overview

LocalRAG is designed for applications requiring reliable, fast, and secure document retrieval without external API dependencies. The system combines advanced embedding techniques with intelligent caching to deliver production-ready performance for medical, financial, legal, and general domains.

[↑ Back to top](#table-of-contents)

## Key Features

### Core RAG Capabilities
- **Complete Offline Operation**: No external API calls required
- **Multi-Document Processing**: Handles PDFs, Word documents, text files, and markdown
- **Advanced Text Chunking**: Token-aware chunking with configurable overlap
- **Matryoshka Embeddings**: Flexible embedding dimensions for performance tuning
- **Cross-Encoder Reranking**: High-precision result ranking using cross-attention models
- **FAISS Integration**: Efficient vector similarity search with exact L2 indexing

### 2.5-Layer Intelligent Caching System

#### Layer 1: Exact Match Cache
- Instant retrieval for identical queries
- Perfect accuracy for repeated questions
- KB version validation ensures cache consistency

#### Layer 2: Semantic Similarity Cache
- FAISS-indexed embedding cache for similar queries
- Configurable similarity thresholds
- Normalized vector comparison for reliable matching

#### Layer 2.5: Cross-Encoder Validation
- Dual-validation using both embedding similarity and cross-encoder confidence
- Domain-specific thresholds (medical: 99%, financial: 99%, legal: 99%, general: 95%)
- Prevents false positives through multi-model verification

### Performance Optimization
- **Memory Management**: Intelligent cache size management with LRU eviction
- **Performance Modes**: Speed, balanced, and safety configurations
- **Document Hashing**: Content-aware cache invalidation
- **Batch Processing**: Efficient embedding generation with progress tracking

### Domain-Specific Safety
- **Conservative Thresholds**: Ultra-high confidence requirements for critical domains
- **Version Tracking**: Automatic cache invalidation on document changes
- **Robust Validation**: Multiple validation layers prevent incorrect retrievals

[↑ Back to top](#table-of-contents)

## Architecture

The system consists of two main components:

### LocalRAGRetriever (`robustRAG.py`)
Handles document ingestion, embedding generation, and retrieval pipeline:
- Document parsing with Docling
- Token-aware text chunking
- Nomic embedding model integration
- FAISS index construction and search
- Cross-encoder reranking

### HighConfidenceCacheManager (`layeredCache.py`)
Manages the intelligent caching system:
- Multi-layer cache architecture
- Knowledge base version tracking
- Memory-aware cache management
- Domain-specific configuration
- Performance monitoring and statistics

[↑ Back to top](#table-of-contents)

## Configuration Options

### Embedding Configuration
- **Model**: nomic-ai/nomic-embed-text-v1.5 (768 dimensions)
- **Chunking**: Configurable token-based splitting
- **Matryoshka Dimensions**: Support for reduced dimensionality retrieval

### Caching Configuration
- **Performance Modes**: Speed (layer 2 cache disabled for small knowledge bases), Balanced (relaxed thresholds), Safety (maximum validation)
- **Domain Types**: Medical, Financial, Legal, General
- **Memory Limits**: Configurable cache size with automatic management
- **TTL Settings**: Time-based cache expiration

### Retrieval Parameters
- **Retrieve K**: Number of candidates from vector search
- **Rerank N**: Final number of results after cross-encoder reranking
- **Similarity Thresholds**: Domain-specific confidence requirements

[↑ Back to top](#table-of-contents)

## Performance Characteristics

### Speed Improvements
- Layer 1 cache: Instant retrieval (sub-millisecond)
- Layer 2 cache: Faster than full pipeline
- Smart indexing: O(log n) similarity search

### Accuracy Guarantees
- Exact match: 100% accuracy for repeated queries
- Semantic cache: Domain-tuned confidence thresholds
- Cross-validation: Dual-model verification prevents false positives

### Resource Efficiency
- Memory management: Automatic cleanup and optimization
- Disk caching: Persistent storage for embeddings and results
- Batch processing: Efficient GPU utilization

[↑ Back to top](#table-of-contents)

## Getting Started

### Installation

To set up the environment using Poetry:

1. Clone the repository:
    ```bash
    git clone https://github.com/GentleClash/robustRAG.git
    cd robustRAG
    ```

2. Install dependencies:
    ```bash
    poetry install
    ```

3. Activate the virtual environment:
    ```bash
    poetry shell
    ```

Now you're ready to use LocalRAG!

### Basic Usage

```python
from robustRAG import LocalRAGRetriever, RAGConfig
from layeredCache import HighConfidenceCacheManager

# Initialize cache manager
cache_manager = HighConfidenceCacheManager(
    domain_type="medical",
    performance_mode="safety"
)

# Configure retriever
config = RAGConfig(
    chunk_size=200,
    retrieve_k=10,
    rerank_n=5
)

# Create retriever
retriever = LocalRAGRetriever(config, cache_manager)

# Build index from documents
retriever.build_index(sources=["path/to/document.pdf"])

# Query the system
results = retriever.retrieve(
    query="What are the treatment options?",
    matryoshka_dim=768
)
```

[↑ Back to top](#table-of-contents)

## Technical Implementation

### Caching Strategy
The 2.5-layer cache architecture provides multiple levels of optimization:

1. **Exact matching** for perfect query repetition
2. **Semantic similarity** using normalized embedding comparison
3. **Cross-encoder validation** for confidence verification

### Knowledge Base Versioning
Automatic detection of document changes through content hashing ensures cache consistency and prevents stale results.

### Memory Management
Intelligent cache sizing with LRU eviction maintains performance while respecting memory constraints.

[↑ Back to top](#table-of-contents)

## Monitoring and Statistics

The system provides comprehensive metrics:
- Cache hit rates by layer
- Memory usage tracking
- Query processing times
- Knowledge base version history
- Confidence score distributions

[↑ Back to top](#table-of-contents)

## Security and Privacy

- **No External Dependencies**: Complete offline operation
- **Data Isolation**: All processing occurs locally
- **Version Control**: Audit trails for document changes
- **Configurable Thresholds**: Domain-appropriate safety levels

[↑ Back to top](#table-of-contents)