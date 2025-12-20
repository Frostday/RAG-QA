import json
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

# Load environment variables from .env file
load_dotenv()

from document_indexer import DocumentIndexer
from qa_service import QAService

app = FastAPI(
    title="Question-Answering Bot API",
    description="AI-powered QA bot that answers questions based on document content",
    version="1.0.0",
)

# BASE_DIR points to RAG-QA root directory (parent of src)
BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_stores"

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)


def validate_document_file(document: UploadFile) -> None:
    """
    Validate that the document file is either PDF or JSON.
    
    Args:
        document: The uploaded document file
        
    Raises:
        HTTPException: If the file type is not PDF or JSON
    """
    if not document.filename:
        raise HTTPException(
            status_code=400,
            detail="Document filename is required"
        )
    
    file_ext = Path(document.filename).suffix.lower()
    if file_ext not in [".pdf", ".json"]:
        raise HTTPException(
            status_code=400,
            detail=f"Document must be a PDF or JSON file. Received: {file_ext or 'no extension'}"
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
    if file_ext != ".json":
        raise HTTPException(
            status_code=400,
            detail=f"Questions file must be a JSON file. Received: {file_ext or 'no extension'}"
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
):
    """
    Process documents and answer questions.
    
    Upload a document (PDF or JSON) and a questions file (JSON), then return answers.
    
    **Document Requirements:**
    - Must be a PDF (.pdf) or JSON (.json) file
    - PDF files will be parsed using Docling
    - JSON files will have their text content extracted
    
    **Questions File Requirements:**
    - Must be a JSON file (.json)
    - Can be formatted as:
      - A list of strings: `["question1", "question2", ...]`
      - An object with a 'questions' field: `{"questions": ["question1", "question2", ...]}`
    
    **Returns:**
    - A dictionary mapping each question to its answer
    - Format: `{"question": "answer", ...}`
    """
    # Validate document file type
    validate_document_file(document)
    
    # Validate questions file and extract questions
    questions = validate_questions_file(questions_file)
    
    # Generate unique session ID for this processing session
    session_id = str(uuid.uuid4())
    vector_store_path = VECTOR_STORE_DIR / session_id
    
    # Save uploaded document temporarily
    temp_doc_file = UPLOAD_DIR / f"{session_id}_{document.filename}"
    
    try:
        # Save document to temporary file
        with open(temp_doc_file, "wb") as f:
            content = await document.read()
            f.write(content)
        
        # Index the document
        indexer = DocumentIndexer(vector_store_path)
        indexer.index_document(str(temp_doc_file), document.filename)
        
        # Initialize QA service
        qa_service = QAService(vector_store_path)
        
        # Answer each question
        answers = {}
        for question in questions:
            answer = qa_service.answer_question(question)
            answers[question] = answer
        
        return answers
        
    except HTTPException:
        # Re-raise HTTP exceptions (validation errors)
        raise
    except Exception as e:
        # Clean up on error
        if temp_doc_file.exists():
            temp_doc_file.unlink()
        if vector_store_path.exists():
            # Remove FAISS index files
            for file in vector_store_path.parent.glob(f"{vector_store_path.name}*"):
                if file.is_file():
                    file.unlink()
                elif file.is_dir():
                    shutil.rmtree(file)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing documents: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_doc_file.exists():
            temp_doc_file.unlink()

