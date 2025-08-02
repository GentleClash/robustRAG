FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create necessary directories
RUN mkdir -p cache documents models temp

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000 8501

RUN echo '#!/bin/bash\n\
echo "Starting RAG Retrieval System..."\n\
echo "Starting FastAPI backend on port 8000..."\n\
uvicorn main:app --host 0.0.0.0 --port 8000 &\n\
echo "Starting Streamlit frontend on port 8501..."\n\
streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0 &\n\
wait\n\
' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]
