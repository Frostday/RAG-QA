"""Tests for the QA Bot API - process-documents endpoint with validation."""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from app import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_openai(monkeypatch):
    """Mock OpenAI API calls to avoid making real API requests during tests."""
    # Set a dummy API key for tests
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-for-testing")
    
    # Mock ChatOpenAI for QA service
    with patch('qa_service.ChatOpenAI') as mock_chat:
        # Create a mock response object
        mock_response = MagicMock()
        mock_response.content = "This is a test answer based on the context provided."
        
        # Configure the mock LLM
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm_instance
        
        # Mock OpenAIEmbeddings for vector store
        with patch('document_indexer.OpenAIEmbeddings') as mock_embeddings_class, \
             patch('qa_service.OpenAIEmbeddings') as mock_embeddings_class_qa:
            
            # Create mock embedding instance
            mock_embeddings_instance = MagicMock()
            
            # Mock the embed_documents and embed_query methods
            def mock_embed(texts):
                # Return fake 1536-dimensional embeddings (text-embedding-3-small dimension)
                if isinstance(texts, str):
                    return [0.1] * 1536
                return [[0.1] * 1536 for _ in texts]
            
            mock_embeddings_instance.embed_documents = MagicMock(side_effect=mock_embed)
            mock_embeddings_instance.embed_query = MagicMock(side_effect=mock_embed)
            mock_embeddings_class.return_value = mock_embeddings_instance
            mock_embeddings_class_qa.return_value = mock_embeddings_instance
            
            # Mock FAISS vector store
            with patch('document_indexer.FAISS') as mock_faiss_class, \
                 patch('qa_service.FAISS') as mock_faiss_class_qa:
                
                # Create mock vector store instance
                mock_vector_store = MagicMock()
                
                # Mock the retriever
                mock_retriever = MagicMock()
                
                def mock_get_relevant_documents(query):
                    # Return mock documents
                    from langchain_core.documents import Document
                    return [
                        Document(
                            page_content="Test document content about data storage.",
                            metadata={"filename": "test.json", "chunk_index": 0}
                        )
                    ]
                
                mock_retriever.get_relevant_documents = mock_get_relevant_documents
                mock_vector_store.as_retriever.return_value = mock_retriever
                
                # Mock save_local to create the directory so QAService can find it
                def mock_save_local(path):
                    from pathlib import Path
                    path_obj = Path(path)
                    # Create the directory
                    path_obj.mkdir(parents=True, exist_ok=True)
                    # Create FAISS index files that FAISS expects
                    (path_obj / "index.faiss").touch()
                    (path_obj / "index.pkl").touch()
                
                mock_vector_store.save_local = mock_save_local
                mock_faiss_class.from_documents.return_value = mock_vector_store
                mock_faiss_class_qa.load_local.return_value = mock_vector_store
                
                yield mock_chat


@pytest.fixture
def sample_questions():
    """Sample questions for testing."""
    return [
        "What is the main topic?",
        "Where is the data stored?",
    ]


@pytest.fixture
def sample_json_document():
    """Create a sample JSON document for testing."""
    return {
        "title": "Test Document",
        "content": "This is a test document with some information about data storage.",
        "data_location": "Data is stored in the cloud.",
        "description": "A sample document for testing purposes."
    }


def test_process_documents_valid_json_document_and_json_questions(
    sample_json_document, sample_questions, tmp_path
):
    """Test processing with valid JSON document and JSON questions file."""
    # Create temporary files
    json_file = tmp_path / "test.json"
    questions_file = tmp_path / "questions.json"
    
    with open(json_file, "w") as f:
        json.dump(sample_json_document, f)
    
    with open(questions_file, "w") as f:
        json.dump(sample_questions, f)
    
    # Upload both files
    with open(json_file, "rb") as doc_file, open(questions_file, "rb") as q_file:
        response = client.post(
            "/process-documents",
            files={
                "document": ("test.json", doc_file, "application/json"),
                "questions_file": ("questions.json", q_file, "application/json")
            }
        )
    
    if response.status_code != 200:
        # Print error details for debugging
        print(f"Error response: {response.status_code}")
        print(f"Error detail: {response.json()}")
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert len(data) == len(sample_questions)
    
    # Verify each question has an answer
    for question in sample_questions:
        assert question in data
        assert isinstance(data[question], str)
        assert len(data[question]) > 0


def test_process_documents_questions_as_object(sample_json_document, sample_questions, tmp_path):
    """Test processing with questions file formatted as object with 'questions' field."""
    json_file = tmp_path / "test.json"
    questions_file = tmp_path / "questions.json"
    
    with open(json_file, "w") as f:
        json.dump(sample_json_document, f)
    
    # Format questions as object
    with open(questions_file, "w") as f:
        json.dump({"questions": sample_questions}, f)
    
    with open(json_file, "rb") as doc_file, open(questions_file, "rb") as q_file:
        response = client.post(
            "/process-documents",
            files={
                "document": ("test.json", doc_file, "application/json"),
                "questions_file": ("questions.json", q_file, "application/json")
            }
        )
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == len(sample_questions)


def test_process_documents_invalid_document_type_txt(sample_questions, tmp_path):
    """Test that non-PDF/non-JSON documents are rejected."""
    # Create a text file (invalid)
    txt_file = tmp_path / "test.txt"
    questions_file = tmp_path / "questions.json"
    
    with open(txt_file, "w") as f:
        f.write("Some text content")
    
    with open(questions_file, "w") as f:
        json.dump(sample_questions, f)
    
    with open(txt_file, "rb") as doc_file, open(questions_file, "rb") as q_file:
        response = client.post(
            "/process-documents",
            files={
                "document": ("test.txt", doc_file, "text/plain"),
                "questions_file": ("questions.json", q_file, "application/json")
            }
        )
    
    assert response.status_code == 400
    assert "PDF or JSON" in response.json()["detail"]


def test_process_documents_invalid_document_type_docx(sample_questions, tmp_path):
    """Test that DOCX documents are rejected."""
    questions_file = tmp_path / "questions.json"
    
    with open(questions_file, "w") as f:
        json.dump(sample_questions, f)
    
    # Create a fake docx file
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
        f.write(b"fake docx content")
        docx_path = f.name
    
    try:
        with open(docx_path, "rb") as doc_file, open(questions_file, "rb") as q_file:
            response = client.post(
                "/process-documents",
                files={
                    "document": ("test.docx", doc_file, "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
                    "questions_file": ("questions.json", q_file, "application/json")
                }
            )
        
        assert response.status_code == 400
        assert "PDF or JSON" in response.json()["detail"]
    finally:
        os.unlink(docx_path)


def test_process_documents_document_no_extension(sample_questions, tmp_path):
    """Test that documents without extension are rejected."""
    questions_file = tmp_path / "questions.json"
    
    with open(questions_file, "w") as f:
        json.dump(sample_questions, f)
    
    # Create a file without extension
    no_ext_file = tmp_path / "test"
    with open(no_ext_file, "w") as f:
        f.write("some content")
    
    with open(no_ext_file, "rb") as doc_file, open(questions_file, "rb") as q_file:
        response = client.post(
            "/process-documents",
            files={
                "document": ("test", doc_file, "text/plain"),
                "questions_file": ("questions.json", q_file, "application/json")
            }
        )
    
    assert response.status_code == 400
    assert "PDF or JSON" in response.json()["detail"]


def test_process_documents_invalid_questions_file_type_txt(sample_json_document, tmp_path):
    """Test that non-JSON questions files are rejected."""
    json_file = tmp_path / "test.json"
    questions_file = tmp_path / "questions.txt"
    
    with open(json_file, "w") as f:
        json.dump(sample_json_document, f)
    
    with open(questions_file, "w") as f:
        f.write("not json content")
    
    with open(json_file, "rb") as doc_file, open(questions_file, "rb") as q_file:
        response = client.post(
            "/process-documents",
            files={
                "document": ("test.json", doc_file, "application/json"),
                "questions_file": ("questions.txt", q_file, "text/plain")
            }
        )
    
    assert response.status_code == 400
    assert "JSON file" in response.json()["detail"]


def test_process_documents_invalid_questions_file_type_pdf(sample_json_document, tmp_path):
    """Test that PDF questions files are rejected."""
    json_file = tmp_path / "test.json"
    questions_file = tmp_path / "questions.pdf"
    
    with open(json_file, "w") as f:
        json.dump(sample_json_document, f)
    
    # Create a fake PDF
    with open(questions_file, "wb") as f:
        f.write(b"%PDF-1.4 fake pdf content")
    
    with open(json_file, "rb") as doc_file, open(questions_file, "rb") as q_file:
        response = client.post(
            "/process-documents",
            files={
                "document": ("test.json", doc_file, "application/json"),
                "questions_file": ("questions.pdf", q_file, "application/pdf")
            }
        )
    
    assert response.status_code == 400
    assert "JSON file" in response.json()["detail"]


def test_process_documents_questions_file_invalid_json(sample_json_document, tmp_path):
    """Test that invalid JSON in questions file is rejected."""
    json_file = tmp_path / "test.json"
    questions_file = tmp_path / "questions.json"
    
    with open(json_file, "w") as f:
        json.dump(sample_json_document, f)
    
    # Write invalid JSON
    with open(questions_file, "w") as f:
        f.write("{ invalid json }")
    
    with open(json_file, "rb") as doc_file, open(questions_file, "rb") as q_file:
        response = client.post(
            "/process-documents",
            files={
                "document": ("test.json", doc_file, "application/json"),
                "questions_file": ("questions.json", q_file, "application/json")
            }
        )
    
    assert response.status_code == 400
    assert "Invalid JSON" in response.json()["detail"]


def test_process_documents_questions_file_empty_list(sample_json_document, tmp_path):
    """Test that empty questions list is rejected."""
    json_file = tmp_path / "test.json"
    questions_file = tmp_path / "questions.json"
    
    with open(json_file, "w") as f:
        json.dump(sample_json_document, f)
    
    with open(questions_file, "w") as f:
        json.dump([], f)
    
    with open(json_file, "rb") as doc_file, open(questions_file, "rb") as q_file:
        response = client.post(
            "/process-documents",
            files={
                "document": ("test.json", doc_file, "application/json"),
                "questions_file": ("questions.json", q_file, "application/json")
            }
        )
    
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]


def test_process_documents_questions_file_not_list_or_object(sample_json_document, tmp_path):
    """Test that questions file with invalid structure is rejected."""
    json_file = tmp_path / "test.json"
    questions_file = tmp_path / "questions.json"
    
    with open(json_file, "w") as f:
        json.dump(sample_json_document, f)
    
    # Questions as a string (invalid)
    with open(questions_file, "w") as f:
        json.dump("just a string", f)
    
    with open(json_file, "rb") as doc_file, open(questions_file, "rb") as q_file:
        response = client.post(
            "/process-documents",
            files={
                "document": ("test.json", doc_file, "application/json"),
                "questions_file": ("questions.json", q_file, "application/json")
            }
        )
    
    assert response.status_code == 400
    assert "list of strings" in response.json()["detail"] or "object with a 'questions' field" in response.json()["detail"]


def test_process_documents_questions_file_object_without_questions_field(sample_json_document, tmp_path):
    """Test that questions object without 'questions' field is rejected."""
    json_file = tmp_path / "test.json"
    questions_file = tmp_path / "questions.json"
    
    with open(json_file, "w") as f:
        json.dump(sample_json_document, f)
    
    # Object without 'questions' field
    with open(questions_file, "w") as f:
        json.dump({"other_field": "value"}, f)
    
    with open(json_file, "rb") as doc_file, open(questions_file, "rb") as q_file:
        response = client.post(
            "/process-documents",
            files={
                "document": ("test.json", doc_file, "application/json"),
                "questions_file": ("questions.json", q_file, "application/json")
            }
        )
    
    assert response.status_code == 400
    assert "questions" in response.json()["detail"].lower()


def test_process_documents_questions_file_non_string_in_list(sample_json_document, tmp_path):
    """Test that questions list with non-string items is rejected."""
    json_file = tmp_path / "test.json"
    questions_file = tmp_path / "questions.json"
    
    with open(json_file, "w") as f:
        json.dump(sample_json_document, f)
    
    # List with non-string items
    with open(questions_file, "w") as f:
        json.dump(["question1", 123, "question2"], f)
    
    with open(json_file, "rb") as doc_file, open(questions_file, "rb") as q_file:
        response = client.post(
            "/process-documents",
            files={
                "document": ("test.json", doc_file, "application/json"),
                "questions_file": ("questions.json", q_file, "application/json")
            }
        )
    
    assert response.status_code == 400
    assert "only strings" in response.json()["detail"]


def test_process_documents_questions_file_object_questions_not_list(sample_json_document, tmp_path):
    """Test that questions field in object that is not a list is rejected."""
    json_file = tmp_path / "test.json"
    questions_file = tmp_path / "questions.json"
    
    with open(json_file, "w") as f:
        json.dump(sample_json_document, f)
    
    # Object with 'questions' as string instead of list
    with open(questions_file, "w") as f:
        json.dump({"questions": "not a list"}, f)
    
    with open(json_file, "rb") as doc_file, open(questions_file, "rb") as q_file:
        response = client.post(
            "/process-documents",
            files={
                "document": ("test.json", doc_file, "application/json"),
                "questions_file": ("questions.json", q_file, "application/json")
            }
        )
    
    assert response.status_code == 400
    assert "must be a list" in response.json()["detail"]


def test_process_documents_missing_document_file():
    """Test that missing document file raises error."""
    questions_file_content = json.dumps(["question1", "question2"]).encode()
    
    response = client.post(
        "/process-documents",
        files={
            "questions_file": ("questions.json", questions_file_content, "application/json")
        }
    )
    
    assert response.status_code == 422  # FastAPI validation error


def test_process_documents_missing_questions_file(sample_json_document, tmp_path):
    """Test that missing questions file raises error."""
    json_file = tmp_path / "test.json"
    
    with open(json_file, "w") as f:
        json.dump(sample_json_document, f)
    
    with open(json_file, "rb") as doc_file:
        response = client.post(
            "/process-documents",
            files={
                "document": ("test.json", doc_file, "application/json")
            }
        )
    
    assert response.status_code == 422  # FastAPI validation error
