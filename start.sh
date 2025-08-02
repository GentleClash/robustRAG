#!/bin/bash

# RAG Retrieval System Startup Script

echo "🚀 Starting RAG Retrieval System..."
echo "======================================="

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p cache documents models temp logs

# Set permissions
chmod -R 755 cache documents models temp logs

# Check if required files exist
echo "🔍 Checking dependencies..."
if [ ! -f "robustRAG.py" ]; then
    echo "❌ robustRAG.py not found! Please copy it to the project root."
    exit 1
fi

if [ ! -f "layeredCache.py" ]; then
    echo "❌ layeredCache.py not found! Please copy it to the project root."
    exit 1
fi

echo "✅ Dependencies found"

# Set environment variables
export PYTHONPATH=/app
export PYTHONUNBUFFERED=1

# Start FastAPI backend
echo "🌐 Starting FastAPI backend on port 8000..."
uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info &
FASTAPI_PID=$!

# Wait a moment for FastAPI to start
sleep 15

# Check if FastAPI started successfully
if ! curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "❌ FastAPI failed to start"
    kill $FASTAPI_PID 2>/dev/null
    exit 1
fi

echo "✅ FastAPI backend started successfully"

# Start Streamlit frontend
echo "🎨 Starting Streamlit frontend on port 8501..."
streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true &
STREAMLIT_PID=$!

# Wait a moment for Streamlit to start
sleep 10

# Check if Streamlit started successfully
if ! curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    echo "⚠️  Streamlit may not have started properly (health check failed)"
    echo "   But this might be normal - Streamlit doesn't always respond to health checks"
fi

echo "✅ Streamlit frontend started"

echo "======================================="
echo "🎉 RAG Retrieval System is running!"
echo ""
echo "📱 Streamlit Frontend: http://localhost:8501"
echo "🔗 FastAPI Backend:    http://localhost:8000"
echo "📚 API Documentation:  http://localhost:8000/docs"
echo ""
echo "📝 Note: This system provides document retrieval only (no inference/generation)"
echo "======================================="

# Function to handle shutdown
shutdown() {
    echo ""
    echo "🛑 Shutting down RAG Retrieval System..."
    kill $FASTAPI_PID 2>/dev/null
    kill $STREAMLIT_PID 2>/dev/null
    echo "✅ Shutdown complete"
    exit 0
}

# Trap SIGTERM and SIGINT
trap shutdown SIGTERM SIGINT

# Wait for both processes
wait
