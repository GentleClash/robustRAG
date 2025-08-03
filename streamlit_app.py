import streamlit as st
import requests
import json
from typing import Dict, Any
import pandas as pd
import os
from google.auth.transport.requests import Request

# Configure Streamlit page
st.set_page_config(
    page_title="RAG Retrieval System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
#API_BASE_URL = st.sidebar.text_input("API Base URL", value=os.getenv("API_BASE_URL"))
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
# Domain-specific threshold configurations
DOMAIN_CONFIGS = {
    "medical": {"semantic_threshold": 0.99, "cross_encoder_threshold": 0.98},
    "financial": {"semantic_threshold": 0.99, "cross_encoder_threshold": 0.98}, 
    "legal": {"semantic_threshold": 0.99, "cross_encoder_threshold": 0.98},
    "general": {"semantic_threshold": 0.95, "cross_encoder_threshold": 0.95}
}


@st.cache_data(ttl=3000)
def get_auth_headers():
    """Gets a Google-signed ID token and returns it in an auth header."""
   # credentials, project = google.auth.default()
   # auth_req = Request()
   # id_token = google.oauth2.id_token.fetch_id_token(auth_req, API_BASE_URL)
   # return {"Authorization": f"Bearer {id_token}"}
    return {} # No auth required for now

def check_api_health():
    """Check if the API is accessible"""
    try:
        # Add a spinner to indicate loading
        with st.spinner("Cold starting the backend..."):
            response = requests.get(f"{API_BASE_URL}/health", timeout=60, headers=get_auth_headers()) # Backend takes time to start
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except Exception as e:
        return False, str(e)

def get_retrievers():
    """Get list of available retrievers"""
    try:
        response = requests.get(f"{API_BASE_URL}/retriever/list", headers=get_auth_headers())
        if response.status_code == 200:
            return response.json().get("retrievers", [])
        return []
    except Exception:
        return []

def create_retriever(config: Dict[str, Any]):
    """Create a new retriever"""
    try:
        response = requests.post(f"{API_BASE_URL}/retriever/create", json=config, headers=get_auth_headers())
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, str(e)

def query_retriever(query_request: Dict[str, Any]):
    """Query a retriever"""
    try:
        response = requests.post(f"{API_BASE_URL}/query", json=query_request, headers=get_auth_headers())
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, str(e)

def upload_files_and_create_retriever(name: str, files, rag_config: Dict, cache_config: Dict):
    """Upload files and create retriever"""
    try:
        files_data = []
        for file in files:
            files_data.append(("files", (file.name, file, file.type)))
        
        data = {
            "name": name,
            "rag_config": json.dumps(rag_config),
            "cache_config": json.dumps(cache_config)
        }

        response = requests.post(f"{API_BASE_URL}/retriever/upload", files=files_data, data=data, headers=get_auth_headers())
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, str(e)

# Main App
st.title("🔍 RAG Retrieval System")
st.markdown("**Document Retrieval Only - No Inference/Generation**")

# Check API connectivity
is_healthy, health_info = check_api_health()
if is_healthy:
    st.success(f"✅ API Connected - {health_info.get('active_retrievers', 0)} active retrievers")
else:
    st.error(f"❌ API Connection Failed: {health_info}")
    st.stop()

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

# Tab selection
tab = st.sidebar.radio("Select Operation", ["Query Documents", "Create Retriever", "Manage Retrievers"])

if tab == "Query Documents":
    st.header("📄 Query Documents")
    
    # Get available retrievers
    retrievers = get_retrievers()
    
    if not retrievers:
        st.warning("No retrievers available. Please create a retriever first.")
        st.stop()
    
    # Retriever selection
    selected_retriever = st.selectbox("Select Retriever", retrievers)
    
    # Query input
    query = st.text_area("Enter your query:", height=100, 
                        placeholder="e.g., Theft of a mobile phone on MG road by an unknown person...")
    
    # Query parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        matryoshka_dim = st.number_input("Matryoshka Dimension", value=768, min_value=64, max_value=2048)
    with col2:
        retrieve_k = st.number_input("Retrieve K", value=10, min_value=1, max_value=100)
    with col3:
        rerank_n = st.number_input("Rerank N", value=5, min_value=1, max_value=50)
    
    if st.button("🔍 Retrieve Documents", type="primary"):
        if query.strip():
            with st.spinner("Retrieving documents..."):
                query_request = {
                    "retriever_name": selected_retriever,
                    "query": query,
                    "matryoshka_dim": matryoshka_dim,
                    "retrieve_k": retrieve_k,
                    "rerank_n": rerank_n
                }
                
                success, result = query_retriever(query_request)
                
                if success:
                    st.success(f"✅ Found {result['total_results']} relevant documents")
                    
                    # Display results
                    for i, doc in enumerate(result['results'], 1):
                        with st.expander(f"Document {i} (Score: {doc.get('score', 'N/A')})"):
                            st.text_area(f"Content {i}", value=doc['text'], height=200, disabled=True)
                            if doc.get('metadata'):
                                st.json(doc['metadata'])
                else:
                    st.error(f"❌ Query failed: {result}")
        else:
            st.warning("Please enter a query")

elif tab == "Create Retriever":
    st.header("🔧 Create New Retriever")
    
    # Retriever basic info
    retriever_name = st.text_input("Retriever Name", placeholder="e.g., legal_documents_retriever")
    
    # Source input method
    source_method = st.radio("Source Input Method", ["URLs", "File Upload"])
    
    if source_method == "URLs":
        sources_text = st.text_area("Document Sources (one URL per line)", 
                                   placeholder="https://example.com/document1.pdf\nhttps://example.com/document2.pdf")
        sources = [url.strip() for url in sources_text.split('\n') if url.strip()]
    else:
        uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True, 
                                         type=['pdf', 'txt', 'docx'])
        sources = uploaded_files if uploaded_files else []
    
    # Configuration sections
    st.subheader("RAG Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        embed_model = st.text_input("Embedding Model", value="nomic-ai/nomic-embed-text-v1.5")
        chunk_size = st.number_input("Chunk Size", value=200, min_value=50, max_value=2000)
        full_dim = st.number_input("Full Dimension", value=768)
        retrieve_k = st.number_input("Retrieve K", value=10, min_value=1, max_value=100)
        enable_caching = st.checkbox("Enable Caching", value=True)
    
    with col2:
        reranker_model = st.text_input("Reranker Model", value="cross-encoder/ms-marco-MiniLM-L-6-v2")
        chunk_overlap = st.number_input("Chunk Overlap", value=10, min_value=0, max_value=100)
        rerank_n = st.number_input("Rerank N", value=5, min_value=1, max_value=50)
        cache_performance_mode = st.selectbox("Cache Performance Mode", ["speed", "balanced", "safety"], index=1)
    
    st.subheader("Cache Configuration")
    col3, col4 = st.columns(2)
    
    with col3:
        cache_dir = st.text_input("Cache Directory", value="cache")
        max_cache_size = st.number_input("Max Cache Size", value=1000, min_value=100)
        ttl_hours = st.number_input("TTL Hours", value=24, min_value=1)
        enable_semantic_cache = st.checkbox("Enable Semantic Cache", value=True)

    with col4:
        domain_type = st.selectbox("Domain Type", ["medical", "financial", "legal", "general"], index=2)
        use_domain_defaults = st.checkbox("Use Domain-Specific Thresholds", value=True)
        
        if use_domain_defaults:
            domain_config = DOMAIN_CONFIGS[domain_type]
            semantic_threshold = domain_config["semantic_threshold"]
            cross_encoder_threshold = domain_config["cross_encoder_threshold"]
            st.info(f"**{domain_type.title()} Domain Defaults:**")
            st.write(f"• Semantic: {semantic_threshold}")
            st.write(f"• Cross-encoder: {cross_encoder_threshold}")
        else:
            semantic_threshold = st.slider("Semantic Threshold", 0.0, 1.0, 0.95)
            cross_encoder_threshold = st.slider("Cross Encoder Threshold", 0.0, 1.0, 0.95)
        
        performance_mode = st.selectbox("Performance Mode", ["speed", "balanced", "safety"], index=2)
        max_memory_mb = st.number_input("Max Memory (MB)", value=1024, min_value=128)
    
    if st.button("🚀 Create Retriever", type="primary"):
        if retriever_name and sources:
            with st.spinner("Creating retriever and building index..."):
                st.info("🔄 Using Docling for PDF parsing - this provides highly accurate results but may take several minutes for large documents on first processing ~5 minutes")
                rag_config = {
                    "embed_model": embed_model,
                    "reranker_model": reranker_model,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "full_dim": full_dim,
                    "retrieve_k": retrieve_k,
                    "rerank_n": rerank_n,
                    "enable_caching": enable_caching,
                    "cache_performance_mode": cache_performance_mode
                }
                
                cache_config = {
                    "cache_dir": cache_dir,
                    "max_cache_size": max_cache_size,
                    "semantic_threshold": semantic_threshold,
                    "cross_encoder_threshold": cross_encoder_threshold,
                    "ttl_hours": ttl_hours,
                    "domain_type": domain_type,
                    "enable_semantic_cache": enable_semantic_cache,
                    "performance_mode": performance_mode,
                    "max_memory_mb": max_memory_mb
                }
                
                if source_method == "File Upload":
                    success, result = upload_files_and_create_retriever(
                        retriever_name, sources, rag_config, cache_config
                    )
                else:
                    config = {
                        "name": retriever_name,
                        "sources": sources,
                        "rag_config": rag_config,
                        "cache_config": cache_config
                    }
                    success, result = create_retriever(config)
                
                if success:
                    st.success("✅ Retriever created successfully!")
                    st.info("📚 Index built and ready for querying. Subsequent queries will be much faster.")
                    st.json(result)
                    st.rerun()
                else:
                    st.error(f"❌ Failed to create retriever: {result}")
        else:
            st.warning("Please provide a retriever name and at least one source")

elif tab == "Manage Retrievers":
    st.header("📊 Manage Retrievers")
    
    retrievers = get_retrievers()
    
    if retrievers:
        st.subheader("Active Retrievers")
        
        # Display retrievers in a table format
        retriever_data = []
        for retriever in retrievers:
            retriever_data.append({
                "Name": retriever,
                "Status": "Active",
                "Actions": "Delete"
            })
        
        df = pd.DataFrame(retriever_data)
        st.dataframe(df, use_container_width=True)
        
        # Delete retriever
        st.subheader("Delete Retriever")
        selected_to_delete = st.selectbox("Select retriever to delete", retrievers)
        
        if st.button("🗑️ Delete Retriever", type="secondary"):
            if selected_to_delete:
                try:
                    response = requests.delete(f"{API_BASE_URL}/retriever/{selected_to_delete}", headers=get_auth_headers())
                    if response.status_code == 200:
                        st.success(f"✅ Retriever '{selected_to_delete}' deleted successfully")
                        st.rerun()
                    else:
                        st.error("❌ Failed to delete retriever")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    else:
        st.info("No retrievers available")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.markdown("This is a **retrieval-only** system. It finds and returns relevant document chunks but does not generate or infer answers.")
st.sidebar.markdown("**Features:**")
st.sidebar.markdown("- Document indexing")
st.sidebar.markdown("- Semantic search")
st.sidebar.markdown("- Configurable caching")
st.sidebar.markdown("- Multiple domain types")
st.sidebar.markdown("- Performance optimization")
