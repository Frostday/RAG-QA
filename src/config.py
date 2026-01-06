"""
Configuration settings for the Question-Answering Bot.

This file contains all configurable parameters for the application,
including limits, timeouts, and model settings.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file FIRST
load_dotenv()

# ============================================================================
# Application Settings
# ============================================================================
APP_NAME = "Question-Answering Bot API"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "AI-powered QA bot that answers questions based on document content"

# ============================================================================
# Directory Paths
# ============================================================================
BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_stores"

# ============================================================================
# File Upload Limits & Constraints
# ============================================================================
MAX_FILE_SIZE_MB = 50  # Maximum document file size in MB
MAX_QUESTIONS = 100    # Maximum number of questions per request
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# ============================================================================
# Processing Settings
# ============================================================================
REQUEST_TIMEOUT_SECONDS = 600  # 10 minutes timeout for API requests (used by frontend)

# PERFORMANCE: Optimized retrieval configuration
# k=5 provides balanced context (not too little, not too much)
# - Too low (k=1-2): May miss important context
# - Too high (k=10+): Slower retrieval, more noise, higher costs
# - Optimal (k=5): Balance of accuracy, speed, and cost
RETRIEVAL_K = 5  # Number of chunks to retrieve for context in QA

# OPTIMIZATION: Similarity threshold for filtering irrelevant results
# Only include chunks with similarity score above this threshold
# - Lower (0.3): Very lenient, includes marginal matches
# - Moderate (0.4-0.5): Balanced, filters clearly irrelevant content
# - Higher (0.6-0.7): Strict, only highly relevant content
# Note: FAISS uses L2 distance, lower scores = more similar
# For cosine similarity (normalized): 0.4 provides balanced filtering
SIMILARITY_THRESHOLD = 0.4

# ============================================================================
# OpenAI Settings
# ============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate API key is loaded
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found in environment variables!\n"
        "Please create a .env file in the project root with:\n"
        "OPENAI_API_KEY=your_openai_api_key_here\n"
        f"Looking for .env file at: {BASE_DIR / '.env'}"
    )

# PERFORMANCE: Optimized model selection for speed and cost
LLM_MODEL = "gpt-4o-mini"  # Fast, cost-effective (~80% cheaper than GPT-4)
LLM_TEMPERATURE = 0  # Deterministic responses, faster generation
EMBEDDING_MODEL = "text-embedding-3-small"  # Fast inference, 1536 dimensions

# ============================================================================
# Document Processing Settings
# ============================================================================
# PERFORMANCE: Optimized chunking strategy
# - 1000 chars ≈ 200-250 tokens (fits context window efficiently)
# - 20% overlap preserves context across chunk boundaries
# - Smaller, focused chunks improve relevance scoring
JSON_CHUNK_SIZE = 1000      # Maximum size for text chunks (characters)
JSON_CHUNK_OVERLAP = 200    # Overlap between chunks (characters)

# PDF processing
PDF_MERGE_PEERS = True      # Merge peer elements in PDF chunking

# ============================================================================
# Supported File Formats
# ============================================================================
SUPPORTED_DOCUMENT_FORMATS = [".pdf", ".json"]
SUPPORTED_QUESTIONS_FORMAT = [".json"]

# ============================================================================
# API Configuration
# ============================================================================
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_URL = os.getenv("API_URL", f"http://{API_HOST}:{API_PORT}")

# FastAPI settings
FASTAPI_TITLE = APP_NAME
FASTAPI_DESCRIPTION = APP_DESCRIPTION
FASTAPI_VERSION = APP_VERSION

# ============================================================================
# Error Messages
# ============================================================================
ERROR_MESSAGES = {
    "file_too_large": f"Document file is too large ({{size_mb:.2f}} MB). Maximum allowed size is {MAX_FILE_SIZE_MB} MB.",
    "too_many_questions": f"Too many questions ({{count}}). Maximum allowed is {MAX_QUESTIONS} questions per request.",
    "empty_file": "Document file is empty",
    "invalid_document_type": f"Document must be a PDF or JSON file. Received: {{ext}}",
    "invalid_questions_type": f"Questions file must be a JSON file. Received: {{ext}}",
    "timeout": "Processing timed out. The document may be too large or complex. Please try with a smaller document or fewer questions.",
    "out_of_memory": "Out of memory while processing document. The document may be too large. Please try with a smaller document.",
    "openai_error": "OpenAI API error. Please check your API key configuration and try again.",
    "corrupted_pdf": "PDF file appears to be corrupted or invalid. Please try with a different PDF file.",
    "network_error": "Network connection error. Please check your internet connection and try again.",
}

