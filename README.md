# RAG-based Question-Answering Bot

An AI-powered Question-Answering bot that leverages Large Language Models (LLMs) to answer questions based on document content. Built with FastAPI, LangChain, Docling, and FAISS.

## Features

- **Multi-format Support**: Process both PDF and JSON documents
- **RAG-based QA**: Uses Retrieval-Augmented Generation (RAG) for accurate, context-aware answers
- **Vector Search**: Uses LangChain's FAISS vector store (saved to disk) for efficient semantic search
- **LangChain Integration**: Built on LangChain for robust LLM orchestration
- **Web Interface**: User-friendly Streamlit frontend for easy interaction
- **RESTful API**: Clean FastAPI endpoints for programmatic access
- **Production Ready**: Includes tests, error handling, and comprehensive documentation

## Technology Stack

- **Python 3.x**: Core programming language
- **FastAPI**: Modern, fast web framework for building APIs
- **Streamlit**: Interactive web interface for easy user interaction
- **LangChain**: Framework for building LLM applications
- **OpenAI GPT-4o-mini**: Language model for generating answers
- **Docling**: Document parsing and chunking (for PDFs)
- **FAISS**: Vector database from LangChain for semantic search (saved to disk)
- **Pydantic**: Data validation

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Frostday/RAG-QA.git
cd RAG-QA
```

2. Create and activate a conda environment (recommended):
```bash
conda create -n rag_qa python=3.12
conda activate rag_qa
pip install -r requirements.txt
```

Alternatively, use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**If `pip install -r requirements.txt` doesn't work**, try these alternatives:

**Option 1: Use pinned versions file**
```bash
pip install -r req.txt
```

**Option 2: Use conda with environment.yml**
```bash
conda env create -f environment.yml
conda activate rag_qa
```

**Option 3: Install packages manually**
```bash
pip install fastapi uvicorn[standard] python-multipart langchain langchain-openai langchain-core langchain-community langchain-text-splitters openai faiss-cpu docling python-dotenv pydantic pandas pytest pytest-asyncio httpx requests streamlit
```

3. Set up environment variables:
Create a `.env` file in the project root directory (same level as `src/` and `README.md`):
```env
OPENAI_API_KEY=your_openai_api_key_here
```

**Important:** 
- Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
- The `.env` file is automatically loaded by `src/config.py` at startup
- Never commit your `.env` file to version control (it's already in `.gitignore`)

## Quick Start

### Using Streamlit (Easiest)

1. Start the API server:
```bash
cd RAG-QA
uvicorn src.app:app --reload
```

2. In a new terminal, start Streamlit:
```bash
cd RAG-QA
streamlit run src/streamlit_app.py
```

3. Open `http://localhost:8501` in your browser and upload your files!

### Using the API Directly

1. Start the API server:
```bash
cd RAG-QA
uvicorn src.app:app --reload
```

2. Use the interactive docs at `http://localhost:8000/docs` or make API calls programmatically.

## Docker Deployment (Recommended for Production)

Get the RAG-QA application running in under 5 minutes using Docker.

### Prerequisites

- [Docker Desktop](https://docs.docker.com/get-docker/) installed
- Docker Compose (usually included with Docker Desktop)
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Quick Start with Docker

**1. Clone and Setup**

```bash
# Clone the repository
git clone https://github.com/Frostday/RAG-QA.git
cd RAG-QA

# Create .env file with your API key
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

**2. Start the Application**

```bash
docker-compose up -d
```

This will:
- Build both backend and frontend images
- Start the FastAPI backend on port 8000
- Start the Streamlit frontend on port 8501
- Set up health checks and auto-restart

**3. Access the Application**

- **Web Interface**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **Metrics**: http://localhost:8000/metrics

### Common Docker Commands

**View Logs**

```bash
# All services
docker-compose logs -f

# Backend only
docker-compose logs -f backend

# Frontend only
docker-compose logs -f frontend
```

**Check Status**

```bash
docker-compose ps
```

**Stop the Application**

```bash
docker-compose down
```

**Restart Services**

```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart backend
```

**Rebuild After Changes**

```bash
docker-compose build
docker-compose up -d
```

### Docker Services

The `docker-compose.yml` file defines two services:

1. **backend** (FastAPI API server)
   - Port: 8000
   - Health checks every 30s
   - Simple logging with timestamps
   - Metrics at `/metrics`

2. **frontend** (Streamlit web interface)
   - Port: 8501
   - Depends on backend being healthy
   - Automatically connects to backend

### Docker Features

- ✅ **Multi-stage builds**: Optimized image sizes
- ✅ **Health checks**: Automatic service health monitoring
- ✅ **Auto-restart**: Services restart on failure
- ✅ **Volume persistence**: Data persists across restarts
- ✅ **Environment variables**: Easy configuration via `.env`
- ✅ **Isolated networking**: Services communicate securely

### Development with Docker

For local development with hot-reload:

```bash
# Start with logs visible
docker-compose up

# Code changes in ./src will be reflected automatically
# No need to rebuild for code changes
```

**Rebuild after dependency changes:**
```bash
docker-compose build
docker-compose up
```

### Production Deployment

For production, edit `docker-compose.yml` and remove volume mounts:

```yaml
# Comment out these lines in both services:
# volumes:
#   - ./src:/app/src
```

Then build and deploy:
```bash
docker-compose build
docker-compose up -d
```

### Troubleshooting

**Container Won't Start**

```bash
# Check logs
docker-compose logs backend

# Common issue: Invalid API key
# Solution: Check your .env file
cat .env
```

**Port Already in Use**

```bash
# Check what's using the port
lsof -i :8000  # Backend
lsof -i :8501  # Frontend

# Kill the process or change ports in docker-compose.yml
```

**Out of Memory**

```bash
# Increase Docker memory limit in Docker Desktop settings
# Preferences > Resources > Memory
```

**Can't Connect to Backend**

```bash
# Check backend health
curl http://localhost:8000/

# Check if backend is running
docker-compose ps backend

# Restart backend
docker-compose restart backend
```

### Docker Monitoring

**View Logs and Metrics**

```bash
# View logs
docker-compose logs -f backend

# View metrics
curl http://localhost:8000/metrics | jq

# Monitor continuously
watch -n 5 'curl -s http://localhost:8000/metrics | jq'
```

**Check Health**

```bash
# Backend health
curl http://localhost:8000/

# Frontend health
curl http://localhost:8501/_stcore/health
```

### Docker Cleanup

**Remove Containers and Networks**

```bash
docker-compose down
```

**Remove Volumes (Data)**

```bash
docker-compose down -v
```

**Remove Images**

```bash
docker-compose down --rmi all
```

**Complete Cleanup**

```bash
# Remove everything
docker-compose down -v --rmi all

# Remove any orphaned data
rm -rf data/uploads/* data/vector_stores/*
```

### Docker Support

For issues:
1. Check logs: `docker-compose logs -f`
2. Check metrics: `curl http://localhost:8000/metrics`
3. Verify API key: `cat .env`
4. Restart services: `docker-compose restart`

Still having issues? Open an issue on GitHub with:
- Output of `docker-compose logs`
- Output of `docker-compose ps`
- Your Docker version: `docker --version`

## Usage

There are two ways to use the Question-Answering Bot:

1. **Streamlit Web Interface** (Recommended for beginners)
2. **REST API** (For programmatic access and integration)

---

## Option 1: Streamlit Web Interface

### Starting the Streamlit App

1. **Start the FastAPI server** (required backend):
```bash
cd RAG-QA
uvicorn src.app:app --reload
```

2. **In a new terminal, start the Streamlit app**:
```bash
cd RAG-QA
streamlit run src/streamlit_app.py
```

3. **Open your browser** to `http://localhost:8501`

### Using the Streamlit Interface

1. **Upload Document**: Click "Choose document file" and select a PDF or JSON file
2. **Upload Questions**: Click "Choose questions file" and select a JSON file with questions
3. **Process**: Click the "🚀 Process Documents" button
4. **View Results**: Answers will be displayed below, and you can download them as JSON

**Questions File Format:**
- List format: `["question1", "question2", ...]`
- Object format: `{"questions": ["question1", "question2", ...]}`

**Features:**
- ✅ **Limits displayed upfront**: File size (50 MB), questions (100 max), timeout (5 min)
- ✅ **Real-time validation**: File size and question count checked before submission
- ✅ **Error handling tips**: Expandable section with common issues and solutions
- ✅ **File validation and preview**: Shows file size, question count, and preview
- ✅ **Real-time processing status**: Loading spinner with progress indication
- ✅ **Download answers as JSON**: Export results for later use
- ✅ **Friendly error messages**: Clear explanations when something goes wrong
- ✅ **Configurable API URL**: Easy to point to different backend instances

---

## Option 2: REST API

### Starting the API Server

Run the FastAPI application:
```bash
cd RAG-QA
uvicorn src.app:app --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, you can access:
- **Interactive API docs**: `http://localhost:8000/docs`
- **Alternative docs**: `http://localhost:8000/redoc`

### API Endpoints

#### Root / Health Check
```http
GET /
```
Returns API information, status, and configuration limits.

**Response:**
```json
{
  "name": "Question-Answering Bot API",
  "version": "1.0.0",
  "status": "operational",
  "limits": {
    "max_file_size_mb": 50,
    "max_questions_per_request": 100
  },
  "supported_formats": {
    "documents": ["PDF", "JSON"],
    "questions": ["JSON"]
  }
}
```

#### Process Documents (Main Endpoint)
```http
POST /process-documents
```
Upload both document and questions file, get answers in one request.

**Request:**
- `document`: File (PDF or JSON) - **Required**
- `questions_file`: JSON file containing list of questions - **Required**

**Questions File Format:**

Option 1 - List format:
```json
[
  "Question 1?",
  "Question 2?",
  "Question 3?"
]
```

Option 2 - Object format:
```json
{
  "questions": [
    "Question 1?",
    "Question 2?"
  ]
}
```

**Response:**
```json
{
  "Question 1?": "Answer 1",
  "Question 2?": "Answer 2"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/process-documents" \
  -F "document=@path/to/document.pdf" \
  -F "questions_file=@path/to/questions.json"
```

**Python Example:**
```python
import requests

files = {
    "document": ("document.pdf", open("document.pdf", "rb"), "application/pdf"),
    "questions_file": ("questions.json", open("questions.json", "rb"), "application/json")
}

response = requests.post("http://localhost:8000/process-documents", files=files)
answers = response.json()
print(answers)
```

**JavaScript/Node.js Example:**
```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

const form = new FormData();
form.append('document', fs.createReadStream('document.pdf'));
form.append('questions_file', fs.createReadStream('questions.json'));

axios.post('http://localhost:8000/process-documents', form, {
  headers: form.getHeaders()
})
.then(response => {
  console.log(response.data);
})
.catch(error => {
  console.error(error);
});
```

## Project Structure

```
RAG-QA/
├── src/                   # Source code
│   ├── __init__.py
│   ├── config.py         # Centralized configuration settings
│   ├── app.py            # FastAPI application and endpoints
│   ├── streamlit_app.py  # Streamlit web interface
│   ├── document_indexer.py  # Document indexing service (PDF/JSON)
│   └── qa_service.py     # QA service using LangChain
├── tests/                 # Test files (48 tests total)
│   ├── __init__.py
│   ├── test_api.py       # Integration tests (19 tests)
│   ├── test_document_indexer.py  # Document processing unit tests (16 tests)
│   └── test_qa_service.py        # QA service unit tests (13 tests)
├── data/                  # Data directory
│   ├── uploads/          # Temporary upload storage
│   └── vector_stores/    # FAISS vector store storage
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── .env                  # Environment variables (create this)
```

### Architecture Highlights

The application follows **clean architecture** principles with clear separation of concerns:

- **Configuration Layer:** Centralized settings in `config.py`
- **API Layer:** FastAPI endpoints in `app.py` for request handling and orchestration
- **Business Logic Layer:** Separate services for document indexing and QA
- **Presentation Layer:** Streamlit web interface and FastAPI docs
- **Infrastructure Layer:** OpenAI, FAISS, and file system integrations


## How It Works

### 1. Document Upload and Processing

When a document is uploaded, it's processed based on its type:

#### PDF Documents
- **Parsing**: Uses Docling's `DocumentConverter` to parse PDF into a structured document representation
- **Chunking**: Uses Docling's `HybridChunker` with `merge_peers=True` for intelligent, structure-aware chunking
  - Preserves document layout, tables, headings, and formatting
  - Chunks are created based on semantic and structural boundaries (not arbitrary text splits)
  - Each chunk maintains metadata including:
    - Page numbers where content appears
    - Section headings/titles
    - Chunk index for ordering
- **Advantages**: Better context preservation, especially for documents with tables, multi-column layouts, and complex structures

#### JSON Documents
- **Parsing**: JSON is parsed and converted to a readable text representation
- **Chunking**: Uses structure-aware chunking to preserve JSON semantics:
  - **For JSON arrays**: Each list item becomes a potential chunk (if small enough)
  - **For JSON objects**: Entire object is kept as a single chunk if small, or split if large
  - **Large chunks**: If a chunk exceeds 1000 characters, it's further split using LangChain's `RecursiveCharacterTextSplitter` with:
    - `chunk_size=1000` characters
    - `chunk_overlap=200` characters (to preserve context across chunk boundaries)
  - Each chunk includes metadata:
    - File type identifier
    - Chunk index
    - List index (for array items)
- **Advantages**: Preserves JSON structure and relationships while ensuring chunks are appropriately sized for embedding

### 2. Embedding and Vector Storage

- **Embedding Model**: Both PDF and JSON chunks are embedded using OpenAI's `text-embedding-3-small` model
  - This model converts text chunks into high-dimensional vectors (embeddings)
  - Embeddings capture semantic meaning, allowing similar content to be found even with different wording
- **Vector Store**: Embeddings are stored in a FAISS (Facebook AI Similarity Search) vector store
  - FAISS enables fast similarity search across all document chunks
  - Vector stores are saved to disk and can be loaded for subsequent queries
  - Each vector store is uniquely identified by a session ID

### 3. Question Answering (RAG Process)

When questions are submitted:

1. **Semantic Search**: 
   - The question is embedded using the same `text-embedding-3-small` model
   - FAISS performs similarity search to find the most relevant document chunks
   - By default, retrieves the top `k=5` most relevant chunks (configurable)

2. **Context Assembly**:
   - Retrieved chunks are combined into a context string
   - This context is passed to the LLM along with the question

3. **Answer Generation**:
   - GPT-4o-mini generates the answer based on the retrieved context
   - The LLM can infer information from context while being transparent about what's directly stated vs. inferred
   - If context is incomplete, the model indicates this in the response

This RAG (Retrieval-Augmented Generation) approach ensures answers are grounded in the actual document content while leveraging the LLM's reasoning capabilities.

## Testing

The application has comprehensive test coverage with **48 tests** across unit and integration testing.

### Quick Start

```bash
cd RAG-QA

# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_api.py -v              # Integration tests (19 tests)
pytest tests/test_document_indexer.py -v # Document processing unit tests (16 tests)
pytest tests/test_qa_service.py -v       # QA service unit tests (13 tests)

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

### Test Coverage

| Test File | Tests | Type | Coverage |
|-----------|-------|------|----------|
| `test_api.py` | 19 | Integration | API endpoints with mocked LLM |
| `test_document_indexer.py` | 16 | Unit | Document processing & chunking (JSON + PDF) |
| `test_qa_service.py` | 13 | Unit | Question answering & retrieval |
| **Total** | **48** | **Mixed** | **Comprehensive** |

**Key Features:**
- ✅ All external dependencies mocked (OpenAI, FAISS)
- ✅ No real API calls required
- ✅ Fast execution (<10 seconds)
- ✅ Edge cases and error scenarios covered
- ✅ Async operations tested
- ✅ Integration tests with complete workflow

---

## Configuration

### Centralized Configuration File

All application settings are now centralized in `src/config.py` for easy maintenance and consistency across all components.

**How it works:**
1. `src/config.py` loads environment variables from `.env` file using `load_dotenv()`
2. All other modules (`app.py`, `streamlit_app.py`, `document_indexer.py`, `qa_service.py`) import settings from `config.py`
3. This ensures consistent configuration across all components

### Environment Variables

**Required Setup:**

Create a `.env` file in the project root directory (`RAG-QA/.env`):
```env
OPENAI_API_KEY=your_openai_api_key_here
```

**Where to get your OpenAI API key:**
- Sign up or log in at [OpenAI Platform](https://platform.openai.com/)
- Navigate to [API Keys](https://platform.openai.com/api-keys)
- Click "Create new secret key"
- Copy the key and paste it in your `.env` file

**Security Notes:**
- The `.env` file is automatically ignored by Git (listed in `.gitignore`)
- Never share or commit your API key
- The API key is loaded once at startup by `config.py`

**Verify Configuration:**

The application will automatically validate the API key on startup. If the API key is missing, you'll see a clear error message with instructions.

You can also run the configuration tests:
```bash
pytest tests/test_config.py -v
```

If you encounter issues, check:
1. `.env` file exists in the project root (not in `src/`)
2. `.env` file contains: `OPENAI_API_KEY=your_actual_key_here`
3. No spaces around the `=` sign
4. No quotes around the API key value

### Adjustable Parameters in `src/config.py`

**Limits & Constraints:**
- `MAX_FILE_SIZE_MB`: Maximum document file size (default: 50 MB)
- `MAX_QUESTIONS`: Maximum questions per request (default: 100)
- `REQUEST_TIMEOUT_SECONDS`: Client timeout for API requests (default: 300 seconds / 5 minutes)

**OpenAI Settings:**
- `LLM_MODEL`: Language model name (default: "gpt-4o-mini")
- `LLM_TEMPERATURE`: LLM temperature for response generation (default: 0)
- `EMBEDDING_MODEL`: Embedding model for vectorization (default: "text-embedding-3-small")

**Document Processing:**
- `RETRIEVAL_K`: Number of chunks to retrieve for context (default: 5)
- `SIMILARITY_THRESHOLD`: Minimum similarity score for retrieved chunks (default: 0.4)
  - Filters out irrelevant content to avoid unnecessary LLM calls
  - Lower values (0.3) are more lenient and include marginal matches
  - Moderate values (0.4-0.5) provide balanced filtering
  - Higher values (0.6-0.7) are stricter and only include highly relevant content
- `JSON_CHUNK_SIZE`: Maximum size for text chunks (default: 1000 characters)
- `JSON_CHUNK_OVERLAP`: Overlap between chunks (default: 200 characters)
- `PDF_MERGE_PEERS`: Merge peer elements in PDF chunking (default: True)

**Supported Formats:**
- `SUPPORTED_DOCUMENT_FORMATS`: List of accepted document types (default: [".pdf", ".json"])
- `SUPPORTED_QUESTIONS_FORMAT`: List of accepted question file types (default: [".json"])

**API Configuration:**
- `API_HOST`: API server host (default: "localhost")
- `API_PORT`: API server port (default: 8000)
- `API_URL`: Full API URL (default: "http://localhost:8000")

**Error Messages:**
- `ERROR_MESSAGES`: Dictionary of all error message templates for consistent messaging

All services (app.py, streamlit_app.py, document_indexer.py, qa_service.py) import their settings from this centralized config file, ensuring consistency across the application.

## Error Handling

The API includes comprehensive error handling with friendly, actionable error messages:

### Validation Errors (400 Bad Request)
- Invalid file types (must be PDF or JSON for documents, JSON for questions)
- Invalid JSON format or structure
- Empty files or empty questions list
- Files exceeding size limits (50 MB maximum)
- Too many questions (100 maximum per request)
- Missing required fields

### Resource Errors
- 413 Payload Too Large: File exceeds 50 MB limit
- 504 Gateway Timeout: Processing took too long (document too complex)
- 507 Insufficient Storage: Out of memory (document too large)

### Service Errors
- 500 Internal Server Error: OpenAI API errors, corrupted PDFs, or other processing errors
- 503 Service Unavailable: Network connection issues

All errors include descriptive messages to help diagnose and resolve issues.

## Observability

The application includes simple logging and metrics tracking for monitoring and debugging:

### Logging

Simple console logging with timestamps for key events:

```
2026-01-05 11:44:02 - INFO - app - Application starting - Question-Answering Bot API v1.0.0
2026-01-05 11:44:02 - INFO - app - Limits: max_file_size=50MB, max_questions=100
2026-01-05 11:45:12 - INFO - app - Processing request: doc=report.pdf, size=2.5MB, questions=10, session=abc-123
2026-01-05 11:45:14 - INFO - app - Document indexed: 45 chunks in 2.341s
2026-01-05 11:45:18 - INFO - app - Request completed: 10 questions answered in 3.2s (total: 5.541s)
```

**Log Levels:**
- **INFO**: Normal operations (startup, requests, completions)
- **WARNING**: Non-critical issues (cleanup failures)
- **ERROR**: Processing errors, API failures, exceptions

**Viewing Logs:**
```bash
# Local development
uvicorn src.app:app --reload

# Docker
docker-compose logs -f backend
```

### Metrics Endpoint

Access real-time metrics at `/metrics`:

```bash
curl http://localhost:8000/metrics
```

**Example Response:**
```json
{
  "requests_total": 42,
  "requests_success": 40,
  "requests_failed": 2,
  "documents_processed": 42,
  "documents_processed_pdf": 30,
  "documents_processed_json": 12,
  "questions_answered": 315,
  "total_tokens_used": 0,
  "total_latency_seconds": 245.678,
  "total_chunks_created": 1250,
  "avg_latency_seconds": 5.849,
  "success_rate": 0.952,
  "timestamp": "2026-01-05T12:34:56.789Z"
}
```

**Tracked Metrics:**
- Request counts (total, success, failed)
- Success rate
- Average and total latency
- Documents processed by type
- Questions answered
- Token usage (if available from OpenAI)
- Chunk counts

### Monitoring

**View logs and metrics:**
```bash
# View logs in real-time
docker-compose logs -f backend

# Monitor metrics
watch -n 5 'curl -s http://localhost:8000/metrics | jq'

# Check success rate
curl -s http://localhost:8000/metrics | jq '.success_rate'

# Check average latency
curl -s http://localhost:8000/metrics | jq '.avg_latency_seconds'
```

**Integration with monitoring systems:**
- View logs in Docker logs or redirect to log files
- Scrape `/metrics` with **Prometheus** or **Datadog**
- Create dashboards for visualization
- Set up alerts on error rates or latency thresholds

## Limitations & Safeguards

### Built-in Limits
- **Maximum file size**: 50 MB per document (configurable)
- **Maximum questions**: 100 questions per request (configurable)
- **Request timeout**: 300 seconds (5 minutes) for Streamlit client
- **Empty files**: Rejected with clear error message

### Performance Considerations
- Processing large PDFs may take time (especially 30+ MB files)
- Complex PDFs with many images or tables require more processing time
- Concurrent question answering improves throughput for multiple questions

### Requirements
- Requires OpenAI API key and internet connection
- FAISS vector stores are stored locally on disk
- Sufficient disk space for temporary files and vector stores
