import json
import asyncio
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Dict, List

# Fix OpenMP warning (must be set before importing FAISS)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Add src directory to Python path for imports
# This allows imports to work when running from RAG-QA directory
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile

# Load environment variables from .env file (backup - config.py also loads it)
load_dotenv()

# Import configuration (config.py loads .env and validates API key)
from config import (
    APP_NAME,
    APP_VERSION,
    APP_DESCRIPTION,
    UPLOAD_DIR,
    VECTOR_STORE_DIR,
    MAX_FILE_SIZE_MB,
    MAX_QUESTIONS,
    MAX_FILE_SIZE_BYTES,
    SUPPORTED_DOCUMENT_FORMATS,
    SUPPORTED_QUESTIONS_FORMAT,
    ERROR_MESSAGES,
    FASTAPI_TITLE,
    FASTAPI_DESCRIPTION,
    FASTAPI_VERSION,
)
from document_indexer import DocumentIndexer
from qa_service import QAService
from metrics import metrics_collector
from logger import setup_logger

app = FastAPI(
    title=FASTAPI_TITLE,
    description=FASTAPI_DESCRIPTION,
    version=FASTAPI_VERSION,
)

# Setup logger
logger = setup_logger(__name__)

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Application starting - {APP_NAME} v{APP_VERSION}")
logger.info(f"Limits: max_file_size={MAX_FILE_SIZE_MB}MB, max_questions={MAX_QUESTIONS}")


@app.get("/")
async def root():
    """
    API information and health check endpoint.
    
    Returns API status, version, and configuration limits.
    """
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "status": "operational",
        "description": APP_DESCRIPTION,
        "limits": {
            "max_file_size_mb": MAX_FILE_SIZE_MB,
            "max_questions_per_request": MAX_QUESTIONS
        },
        "supported_formats": {
            "documents": [fmt.upper().replace(".", "") for fmt in SUPPORTED_DOCUMENT_FORMATS],
            "questions": [fmt.upper().replace(".", "") for fmt in SUPPORTED_QUESTIONS_FORMAT]
        },
        "endpoints": {
            "process_documents": "/process-documents",
            "metrics": "/metrics",
            "docs": "/docs",
            "openapi": "/openapi.json"
        }
    }


@app.get("/metrics")
async def get_metrics():
    """
    Get application metrics for observability.
    
    Returns metrics including:
    - Total requests and success rate
    - Average latency
    - Documents processed
    - Questions answered
    - Token usage (if available)
    """
    return metrics_collector.get_metrics()


def validate_document_file(document: UploadFile, content: bytes) -> None:
    """
    Validate that the document file is either PDF or JSON and within size limits.
    
    Args:
        document: The uploaded document file
        content: The file content as bytes
        
    Raises:
        HTTPException: If the file type is not PDF or JSON, or if it exceeds size limits
    """
    if not document.filename:
        raise HTTPException(
            status_code=400,
            detail="Document filename is required"
        )
    
    # Check file extension
    file_ext = Path(document.filename).suffix.lower()
    if file_ext not in SUPPORTED_DOCUMENT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=ERROR_MESSAGES["invalid_document_type"].format(ext=file_ext or 'no extension')
        )
    
    # Check file size
    file_size_mb = len(content) / (1024 * 1024)
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=ERROR_MESSAGES["file_too_large"].format(size_mb=file_size_mb)
        )
    
    # Check if file is empty
    if len(content) == 0:
        raise HTTPException(
            status_code=400,
            detail=ERROR_MESSAGES["empty_file"]
        )


def validate_questions_file(questions_file: UploadFile) -> List[str]:
    """
    Validate that the questions file is JSON and extract questions.
    
    Args:
        questions_file: The uploaded questions file
        
    Returns:
        List of questions extracted from the JSON file
        
    Raises:
        HTTPException: If the file is not JSON or contains invalid data
    """
    if not questions_file.filename:
        raise HTTPException(
            status_code=400,
            detail="Questions filename is required"
        )
    
    # Validate file extension
    file_ext = Path(questions_file.filename).suffix.lower()
    if file_ext not in SUPPORTED_QUESTIONS_FORMAT:
        raise HTTPException(
            status_code=400,
            detail=ERROR_MESSAGES["invalid_questions_type"].format(ext=file_ext or 'no extension')
        )
    
    # Read and parse JSON content
    questions_content = questions_file.file.read()
    questions_file.file.seek(0)  # Reset file pointer for later use
    
    try:
        questions_data = json.loads(questions_content)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON in questions file: {str(e)}"
        )
    
    # Extract questions from different possible formats
    if isinstance(questions_data, list):
        # Format: ["question1", "question2", ...]
        if not all(isinstance(q, str) for q in questions_data):
            raise HTTPException(
                status_code=400,
                detail="Questions list must contain only strings"
            )
        if len(questions_data) == 0:
            raise HTTPException(
                status_code=400,
                detail="Questions list cannot be empty"
            )
        if len(questions_data) > MAX_QUESTIONS:
            raise HTTPException(
                status_code=400,
                detail=ERROR_MESSAGES["too_many_questions"].format(count=len(questions_data))
            )
        return questions_data
    elif isinstance(questions_data, dict):
        # Format: {"questions": ["question1", "question2", ...]}
        if "questions" in questions_data:
            questions = questions_data["questions"]
            if not isinstance(questions, list):
                raise HTTPException(
                    status_code=400,
                    detail="'questions' field must be a list"
                )
            if not all(isinstance(q, str) for q in questions):
                raise HTTPException(
                    status_code=400,
                    detail="Questions list must contain only strings"
                )
            if len(questions) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Questions list cannot be empty"
                )
            if len(questions) > MAX_QUESTIONS:
                raise HTTPException(
                    status_code=400,
                    detail=ERROR_MESSAGES["too_many_questions"].format(count=len(questions))
                )
            return questions
        else:
            raise HTTPException(
                status_code=400,
                detail="Questions JSON object must contain a 'questions' field with a list of strings"
            )
    else:
        raise HTTPException(
            status_code=400,
            detail="Questions file must contain either a list of strings or an object with a 'questions' field"
        )


@app.post("/process-documents")
async def process_documents(
    document: UploadFile = File(...),
    questions_file: UploadFile = File(...)
) -> Dict[str, str]:
    """
    Process documents and answer questions.
    
    Upload a document (PDF or JSON) and a questions file (JSON), then return answers.
    
    **Document Requirements:**
    - Must be a PDF (.pdf) or JSON (.json) file
    - Maximum file size: 50 MB
    - File must not be empty
    - PDF files will be parsed using Docling
    - JSON files will have their text content extracted
    
    **Questions File Requirements:**
    - Must be a JSON file (.json)
    - Maximum 100 questions per request
    - Must contain at least 1 question
    - Can be formatted as:
      - A list of strings: `["question1", "question2", ...]`
      - An object with a 'questions' field: `{"questions": ["question1", "question2", ...]}`
    
    **Returns:**
    - A dictionary mapping each question to its answer
    - Format: `{"question": "answer", ...}`
    
    **Error Codes:**
    - 400: Invalid file type, format, or empty file
    - 413: File size exceeds 50 MB limit
    - 500: Processing error (OpenAI API, corrupted PDF, etc.)
    - 504: Processing timeout (document too complex)
    - 507: Out of memory (document too large)
    """
    # Track request with metrics
    with metrics_collector.track_request("process_documents") as request_metrics:
        # Read document content once for validation and saving
        try:
            content = await document.read()
        except Exception as e:
            logger.error(f"Failed to read document file: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read document file: {str(e)}"
            )
        
        # Validate document file type and size
        validate_document_file(document, content)
        
        # Validate questions file and extract questions
        questions = validate_questions_file(questions_file)
        
        # Generate unique session ID for this processing session
        session_id = str(uuid.uuid4())
        vector_store_path = VECTOR_STORE_DIR / session_id
        
        # Save uploaded document temporarily
        temp_doc_file = UPLOAD_DIR / f"{session_id}_{document.filename}"
        
        # Store request details for metrics
        file_ext = Path(document.filename).suffix.lower()
        file_size_mb = round(len(content) / (1024 * 1024), 2)
        
        logger.info(f"Processing request: doc={document.filename}, size={file_size_mb}MB, questions={len(questions)}, session={session_id}")
        
        # Store metrics in request context
        request_metrics["document_name"] = document.filename
        request_metrics["document_type"] = file_ext
        request_metrics["file_size_mb"] = file_size_mb
        request_metrics["question_count"] = len(questions)
        request_metrics["unique_questions"] = len(set(questions))
        
        try:
            # Save document to temporary file
            with open(temp_doc_file, "wb") as f:
                f.write(content)
            
            # Index the document
            import time
            indexing_start = time.time()
            indexer = DocumentIndexer(vector_store_path)
            chunk_count = indexer.index_document(str(temp_doc_file), document.filename)
            indexing_duration = round(time.time() - indexing_start, 3)
            
            # Record document processing metrics
            metrics_collector.record_document_processed(
                doc_type=file_ext.strip('.'),
                chunk_count=chunk_count if chunk_count else 0
            )
            
            logger.info(f"Document indexed: {chunk_count} chunks in {indexing_duration}s")
            
            # Initialize QA service
            qa_service = QAService(vector_store_path)
            
            # OPTIMIZATION: Deduplicate questions to avoid redundant LLM calls
            # If the same question appears multiple times, we only process it once
            unique_questions = list(dict.fromkeys(questions))  # Preserves order, removes duplicates
            
            # PERFORMANCE: Answer all unique questions concurrently for 7-25x speedup
            # Uses asyncio.gather() to process multiple questions in parallel
            # Example: 10 questions in ~3s (concurrent) vs ~20s (sequential)
            qa_start = time.time()
            unique_answers = await asyncio.gather(*[qa_service.answer_question(q) for q in unique_questions])
            qa_duration = round(time.time() - qa_start, 3)
            
            # Create answer mapping for unique questions
            answer_map = dict(zip(unique_questions, unique_answers))
            
            # Build dictionary mapping all original questions to their answers
            # Duplicate questions will reuse the cached answer from answer_map
            answers = {question: answer_map[question] for question in questions}
            
            # Record QA metrics
            for _ in unique_questions:
                metrics_collector.record_question_answered(
                    latency_seconds=qa_duration / len(unique_questions)
                )
            
            # Store performance metrics in request context
            request_metrics["indexing_duration_seconds"] = indexing_duration
            request_metrics["qa_duration_seconds"] = qa_duration
            request_metrics["chunk_count"] = chunk_count
            
            logger.info(f"Request completed: {len(questions)} questions answered in {qa_duration}s (total: {round(indexing_duration + qa_duration, 3)}s)")
            
            return answers
        
        except HTTPException:
            # Re-raise HTTP exceptions (validation errors)
            raise
        except asyncio.TimeoutError:
            # Handle timeout errors
            logger.error(f"Request timeout: session={session_id}")
            raise HTTPException(
                status_code=504,
                detail=ERROR_MESSAGES["timeout"]
            )
        except MemoryError:
            # Handle out of memory errors
            logger.error(f"Out of memory: session={session_id}")
            raise HTTPException(
                status_code=507,
                detail=ERROR_MESSAGES["out_of_memory"]
            )
        except FileNotFoundError as e:
            # Handle missing file errors
            logger.error(f"File not found: {str(e)}, session={session_id}")
            raise HTTPException(
                status_code=500,
                detail=f"Required file not found during processing: {str(e)}"
            )
        except PermissionError as e:
            # Handle permission errors
            logger.error(f"Permission denied: {str(e)}, session={session_id}")
            raise HTTPException(
                status_code=500,
                detail=f"Permission denied while accessing files: {str(e)}"
            )
        except json.JSONDecodeError as e:
            # Handle JSON parsing errors in document
            logger.error(f"Invalid JSON: {str(e)}, session={session_id}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON document format: {str(e)}"
            )
        except Exception as e:
            # Catch all other errors with specific categorization
            error_message = str(e).lower()
            logger.error(f"Processing error: {str(e)}, session={session_id}")
            
            # Check for specific error patterns and provide helpful messages
            if "openai" in error_message or "api key" in error_message:
                raise HTTPException(
                    status_code=500,
                    detail=ERROR_MESSAGES["openai_error"]
                )
            elif "pdf" in error_message and ("corrupt" in error_message or "invalid" in error_message):
                raise HTTPException(
                    status_code=400,
                    detail=ERROR_MESSAGES["corrupted_pdf"]
                )
            elif "timeout" in error_message:
                raise HTTPException(
                    status_code=504,
                    detail=ERROR_MESSAGES["timeout"]
                )
            elif "connection" in error_message or "network" in error_message:
                raise HTTPException(
                    status_code=503,
                    detail=ERROR_MESSAGES["network_error"]
                )
            else:
                # Generic error with full details
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing documents: {str(e)}"
                )
        
        finally:
            # ROBUSTNESS: Ensure cleanup happens even if errors occur
            # This prevents resource leaks and disk space issues
            
            # Clean up temporary document file
            if temp_doc_file.exists():
                try:
                    temp_doc_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete temp file: {str(e)}")
            
            # Clean up vector store directory (contains indexed document data)
            if vector_store_path.exists():
                try:
                    shutil.rmtree(vector_store_path)
                except Exception as e:
                    logger.warning(f"Failed to delete vector store: {str(e)}")

