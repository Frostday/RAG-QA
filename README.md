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
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

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
- ✅ File validation and preview
- ✅ Real-time processing status
- ✅ Download answers as JSON
- ✅ Configurable API URL

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
│   ├── app.py            # FastAPI application and endpoints
│   ├── streamlit_app.py  # Streamlit web interface
│   ├── document_indexer.py  # Document indexing service (PDF/JSON)
│   └── qa_service.py     # QA service using LangChain
├── tests/                 # Test files
│   ├── __init__.py
│   └── test_api.py       # API tests
├── data/                  # Data directory
│   ├── uploads/          # Temporary upload storage
│   └── vector_stores/    # FAISS vector store storage
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── .env                  # Environment variables (create this)
```

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

Run the test suite:
```bash
cd RAG-QA
pytest tests/test_api.py -v
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Adjustable Parameters

**In `src/qa_service.py`:**
- `k`: Number of chunks to retrieve for context (default: 5)
- `model`: LLM model name (default: "gpt-4o-mini")
- `temperature`: LLM temperature (default: 0)

**In `src/document_indexer.py`:**
- `chunk_size`: Maximum size for text chunks when splitting large JSON items (default: 1000 characters)
- `chunk_overlap`: Overlap between chunks to preserve context (default: 200 characters)
- `embedding_model`: Embedding model for vectorization (default: "text-embedding-3-small")

## Error Handling

The API includes comprehensive error handling:
- Invalid file types return 400 Bad Request
- Invalid JSON or missing required fields return 400 Bad Request
- Server errors return 500 Internal Server Error with details

## Limitations

- Maximum file size depends on your server configuration
- Processing large PDFs may take time
- Requires OpenAI API key and internet connection
- FAISS vector stores are stored locally on disk
