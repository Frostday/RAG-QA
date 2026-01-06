# Multi-stage Dockerfile for RAG-QA Application
# This creates optimized images for both backend (FastAPI) and frontend (Streamlit)

FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Copy .env file if it exists (optional)
COPY .env* ./

# Create necessary directories
RUN mkdir -p data/uploads data/vector_stores

# ==============================================================================
# Backend Stage (FastAPI)
# ==============================================================================
FROM base as backend

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run FastAPI with uvicorn
# --timeout-keep-alive: Increase timeout for long-running requests (default is 5s)
# --timeout-graceful-shutdown: Allow time for requests to complete during shutdown
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "600", "--timeout-graceful-shutdown", "30"]

# ==============================================================================
# Frontend Stage (Streamlit)
# ==============================================================================
FROM base as frontend

# Expose Streamlit port
EXPOSE 8501

# Health check for Streamlit
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

