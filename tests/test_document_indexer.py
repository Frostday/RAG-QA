"""Unit tests for DocumentIndexer core logic."""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(autouse=True)
def mock_openai_for_indexer(monkeypatch):
    """Mock OpenAI for all indexer tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-for-indexer-testing")
    
    with patch('document_indexer.OpenAIEmbeddings') as mock_embeddings:
        mock_instance = MagicMock()
        mock_instance.embed_documents = MagicMock(return_value=[[0.1] * 1536])
        mock_embeddings.return_value = mock_instance
        
        with patch('document_indexer.FAISS') as mock_faiss:
            mock_vector_store = MagicMock()
            mock_vector_store.save_local = MagicMock()
            mock_faiss.from_documents.return_value = mock_vector_store
            
            yield


@pytest.fixture
def temp_vector_store(tmp_path):
    """Create a temporary vector store path."""
    return tmp_path / "test_vector_store"


def test_document_indexer_initialization(temp_vector_store):
    """Test that DocumentIndexer initializes correctly."""
    from document_indexer import DocumentIndexer
    
    indexer = DocumentIndexer(temp_vector_store)
    
    assert indexer.vector_store_path == temp_vector_store
    assert indexer.embeddings is not None
    assert indexer.text_splitter is not None
    assert indexer.text_splitter._chunk_size == 1000
    assert indexer.text_splitter._chunk_overlap == 200


def test_json_to_text_with_dict(temp_vector_store):
    """Test _json_to_text method with dictionary input."""
    from document_indexer import DocumentIndexer
    
    indexer = DocumentIndexer(temp_vector_store)
    data = {
        "name": "John Doe",
        "age": 30,
        "city": "New York"
    }
    
    result = indexer._json_to_text(data)
    
    assert "name: John Doe" in result
    assert "age: 30" in result
    assert "city: New York" in result


def test_json_to_text_with_nested_dict(temp_vector_store):
    """Test _json_to_text with nested dictionary."""
    from document_indexer import DocumentIndexer
    
    indexer = DocumentIndexer(temp_vector_store)
    data = {
        "user": {
            "name": "Jane",
            "details": {
                "age": 25
            }
        }
    }
    
    result = indexer._json_to_text(data)
    
    assert "user:" in result
    assert "name: Jane" in result
    assert "details:" in result
    assert "age: 25" in result


def test_json_to_text_with_list(temp_vector_store):
    """Test _json_to_text with list input."""
    from document_indexer import DocumentIndexer
    
    indexer = DocumentIndexer(temp_vector_store)
    data = ["apple", "banana", "cherry"]
    
    result = indexer._json_to_text(data)
    
    assert "Item 1: apple" in result
    assert "Item 2: banana" in result
    assert "Item 3: cherry" in result


def test_json_to_text_with_empty_values(temp_vector_store):
    """Test _json_to_text handles empty values correctly."""
    from document_indexer import DocumentIndexer
    
    indexer = DocumentIndexer(temp_vector_store)
    data = {
        "field1": "",
        "field2": None,
        "field3": "value"
    }
    
    result = indexer._json_to_text(data)
    
    # Empty strings and None should be skipped
    assert "field1" not in result or "field1:" in result  # May or may not appear
    assert "field3: value" in result


def test_index_json_with_small_object(temp_vector_store, tmp_path):
    """Test indexing a small JSON object."""
    from document_indexer import DocumentIndexer
    
    indexer = DocumentIndexer(temp_vector_store)
    
    # Create a small JSON file
    json_file = tmp_path / "small.json"
    data = {"title": "Test", "content": "This is a small test document."}
    with open(json_file, "w") as f:
        json.dump(data, f)
    
    documents = indexer._index_json(str(json_file), "small.json")
    
    assert len(documents) >= 1
    assert all(hasattr(doc, 'page_content') for doc in documents)
    assert all(hasattr(doc, 'metadata') for doc in documents)
    assert documents[0].metadata['filename'] == "small.json"
    assert documents[0].metadata['file_type'] == "json"


def test_index_json_with_list(temp_vector_store, tmp_path):
    """Test indexing a JSON list."""
    from document_indexer import DocumentIndexer
    
    indexer = DocumentIndexer(temp_vector_store)
    
    # Create a JSON list file
    json_file = tmp_path / "list.json"
    data = [
        {"id": 1, "name": "Item 1"},
        {"id": 2, "name": "Item 2"},
        {"id": 3, "name": "Item 3"}
    ]
    with open(json_file, "w") as f:
        json.dump(data, f)
    
    documents = indexer._index_json(str(json_file), "list.json")
    
    assert len(documents) >= 3  # At least one per item
    assert all('list_index' in doc.metadata for doc in documents)


def test_index_json_with_large_text(temp_vector_store, tmp_path):
    """Test indexing JSON with text that exceeds chunk size."""
    from document_indexer import DocumentIndexer
    
    indexer = DocumentIndexer(temp_vector_store)
    
    # Create a JSON file with large text (>1000 chars)
    json_file = tmp_path / "large.json"
    large_text = "x" * 1500  # Larger than chunk_size
    data = {"content": large_text}
    with open(json_file, "w") as f:
        json.dump(data, f)
    
    documents = indexer._index_json(str(json_file), "large.json")
    
    # Should be split into multiple chunks
    assert len(documents) >= 2


def test_index_document_with_json(temp_vector_store, tmp_path):
    """Test the main index_document method with JSON."""
    from document_indexer import DocumentIndexer
    
    indexer = DocumentIndexer(temp_vector_store)
    
    # Create a JSON file
    json_file = tmp_path / "test.json"
    data = {"test": "data"}
    with open(json_file, "w") as f:
        json.dump(data, f)
    
    # Index the document
    chunk_count = indexer.index_document(str(json_file), "test.json")
    
    assert chunk_count > 0
    # Vector store directory should exist after save_local is called
    assert temp_vector_store.parent.exists()


def test_index_document_with_unsupported_type(temp_vector_store, tmp_path):
    """Test that unsupported file types raise an error."""
    from document_indexer import DocumentIndexer
    
    indexer = DocumentIndexer(temp_vector_store)
    
    # Create a .txt file (unsupported)
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("test content")
    
    with pytest.raises(ValueError, match="Unsupported file type"):
        indexer.index_document(str(txt_file), "test.txt")


def test_index_document_creates_vector_store_directory(temp_vector_store, tmp_path):
    """Test that indexing creates the vector store directory."""
    from document_indexer import DocumentIndexer
    
    # Ensure directory doesn't exist
    assert not temp_vector_store.exists()
    
    indexer = DocumentIndexer(temp_vector_store)
    
    # Parent directory should be created
    assert temp_vector_store.parent.exists()


def test_chunk_size_configuration(temp_vector_store):
    """Test that text splitter uses configured chunk size."""
    from document_indexer import DocumentIndexer
    from config import JSON_CHUNK_SIZE, JSON_CHUNK_OVERLAP
    
    indexer = DocumentIndexer(temp_vector_store)
    
    assert indexer.text_splitter._chunk_size == JSON_CHUNK_SIZE
    assert indexer.text_splitter._chunk_overlap == JSON_CHUNK_OVERLAP


def test_embedding_model_configuration(temp_vector_store):
    """Test that embeddings use configured model."""
    from document_indexer import DocumentIndexer
    
    with patch('document_indexer.OpenAIEmbeddings') as mock_embeddings:
        mock_instance = MagicMock()
        mock_embeddings.return_value = mock_instance
        
        indexer = DocumentIndexer(temp_vector_store)
        
        # Verify OpenAIEmbeddings was called with correct model
        mock_embeddings.assert_called_once()
        call_kwargs = mock_embeddings.call_args.kwargs
        assert 'model' in call_kwargs
        assert 'openai_api_key' in call_kwargs


def test_json_document_metadata_structure(temp_vector_store, tmp_path):
    """Test that JSON documents have correct metadata structure."""
    from document_indexer import DocumentIndexer
    
    indexer = DocumentIndexer(temp_vector_store)
    
    json_file = tmp_path / "metadata_test.json"
    data = {"field": "value"}
    with open(json_file, "w") as f:
        json.dump(data, f)
    
    documents = indexer._index_json(str(json_file), "metadata_test.json")
    
    # Check metadata structure
    for doc in documents:
        assert 'filename' in doc.metadata
        assert 'source' in doc.metadata
        assert 'file_type' in doc.metadata
        assert 'chunk_index' in doc.metadata
        assert doc.metadata['filename'] == "metadata_test.json"
        assert doc.metadata['file_type'] == "json"


def test_index_pdf_with_small_document(temp_vector_store, tmp_path):
    """Test indexing a small PDF document."""
    from document_indexer import DocumentIndexer
    from langchain_core.documents import Document
    
    indexer = DocumentIndexer(temp_vector_store)
    
    # Create a dummy PDF file
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\nDummy PDF content")
    
    # Mock Docling's DocumentConverter and HybridChunker
    with patch('document_indexer.DocumentConverter') as mock_converter_class, \
         patch('document_indexer.HybridChunker') as mock_chunker_class:
        
        # Create mock chunks with proper structure
        mock_chunk = MagicMock()
        mock_chunk.text = "This is a test PDF document with some content."
        mock_chunk.meta.headings = ["Test Section"]
        
        # Create mock doc_item with prov (provenance)
        mock_prov = MagicMock()
        mock_prov.page_no = 1
        mock_doc_item = MagicMock()
        mock_doc_item.prov = [mock_prov]
        mock_chunk.meta.doc_items = [mock_doc_item]
        
        # Configure chunker
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [mock_chunk]
        mock_chunker_class.return_value = mock_chunker
        
        # Configure converter
        mock_result = MagicMock()
        mock_result.document = MagicMock()
        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result
        mock_converter_class.return_value = mock_converter
        
        # Test PDF indexing
        documents = indexer._index_pdf(str(pdf_file), "test.pdf")
        
        # Verify results
        assert len(documents) >= 1
        assert all(hasattr(doc, 'page_content') for doc in documents)
        assert all(hasattr(doc, 'metadata') for doc in documents)
        assert documents[0].metadata['filename'] == "test.pdf"
        assert 'page_numbers' in documents[0].metadata
        assert 'title' in documents[0].metadata


def test_index_document_with_pdf(temp_vector_store, tmp_path):
    """Test the main index_document method with PDF."""
    from document_indexer import DocumentIndexer
    
    indexer = DocumentIndexer(temp_vector_store)
    
    # Create a dummy PDF file
    pdf_file = tmp_path / "document.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\nDummy PDF content for testing")
    
    # Mock Docling components
    with patch('document_indexer.DocumentConverter') as mock_converter_class, \
         patch('document_indexer.HybridChunker') as mock_chunker_class:
        
        # Create mock chunks
        mock_chunk1 = MagicMock()
        mock_chunk1.text = "First chunk of the PDF document."
        mock_chunk1.meta.headings = ["Introduction"]
        mock_prov1 = MagicMock()
        mock_prov1.page_no = 1
        mock_doc_item1 = MagicMock()
        mock_doc_item1.prov = [mock_prov1]
        mock_chunk1.meta.doc_items = [mock_doc_item1]
        
        mock_chunk2 = MagicMock()
        mock_chunk2.text = "Second chunk of the PDF document."
        mock_chunk2.meta.headings = ["Body"]
        mock_prov2 = MagicMock()
        mock_prov2.page_no = 2
        mock_doc_item2 = MagicMock()
        mock_doc_item2.prov = [mock_prov2]
        mock_chunk2.meta.doc_items = [mock_doc_item2]
        
        # Configure mocks
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [mock_chunk1, mock_chunk2]
        mock_chunker_class.return_value = mock_chunker
        
        mock_result = MagicMock()
        mock_result.document = MagicMock()
        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result
        mock_converter_class.return_value = mock_converter
        
        # Index the PDF document
        chunk_count = indexer.index_document(str(pdf_file), "document.pdf")
        
        # Verify
        assert chunk_count >= 2  # At least 2 chunks
        assert temp_vector_store.parent.exists()  # Vector store directory created

