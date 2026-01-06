"""Unit tests for QAService core logic."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(autouse=True)
def mock_openai_for_qa(monkeypatch):
    """Mock OpenAI for all QA service tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-for-qa-testing")
    
    with patch('qa_service.OpenAIEmbeddings') as mock_embeddings, \
         patch('qa_service.ChatOpenAI') as mock_chat, \
         patch('qa_service.FAISS') as mock_faiss:
        
        # Mock embeddings
        mock_embed_instance = MagicMock()
        mock_embeddings.return_value = mock_embed_instance
        
        # Mock LLM
        mock_response = MagicMock()
        mock_response.content = "Test answer"
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm_instance
        
        # Mock FAISS with similarity_search_with_score
        from langchain_core.documents import Document
        
        mock_vector_store = MagicMock()
        # Return documents with scores (lower score = more similar for L2 distance)
        # Score of 0.3 means high similarity (passes default threshold of 0.5)
        mock_vector_store.similarity_search_with_score.return_value = [
            (Document(page_content="Test context", metadata={}), 0.3)
        ]
        mock_faiss.load_local.return_value = mock_vector_store
        
        yield


@pytest.fixture
def temp_vector_store(tmp_path):
    """Create a temporary vector store directory."""
    vector_store = tmp_path / "vector_store"
    vector_store.mkdir()
    (vector_store / "index.faiss").touch()
    (vector_store / "index.pkl").touch()
    return vector_store


def test_qa_service_initialization(temp_vector_store):
    """Test that QAService initializes correctly."""
    from qa_service import QAService
    
    service = QAService(temp_vector_store)
    
    assert service.vector_store_path == temp_vector_store
    assert service.k == 5  # Default from config
    assert service.similarity_threshold == 0.4  # Default from config
    assert service.vector_store is not None
    assert service.llm is not None


def test_qa_service_with_custom_k(temp_vector_store):
    """Test QAService with custom k parameter."""
    from qa_service import QAService
    
    service = QAService(temp_vector_store, k=10)
    
    assert service.k == 10


def test_qa_service_with_nonexistent_vector_store(tmp_path):
    """Test that QAService raises error for nonexistent vector store."""
    from qa_service import QAService
    
    nonexistent_path = tmp_path / "nonexistent"
    
    with pytest.raises(RuntimeError, match="Vector store not found"):
        QAService(nonexistent_path)


@pytest.mark.asyncio
async def test_answer_question_returns_string(temp_vector_store):
    """Test that answer_question returns a string."""
    from qa_service import QAService
    
    service = QAService(temp_vector_store)
    answer = await service.answer_question("What is the test?")
    
    assert isinstance(answer, str)
    assert len(answer) > 0


@pytest.mark.asyncio
async def test_answer_question_with_empty_question(temp_vector_store):
    """Test answer_question with empty question string."""
    from qa_service import QAService
    
    service = QAService(temp_vector_store)
    answer = await service.answer_question("")
    
    # Should still return a string (even if it's a default response)
    assert isinstance(answer, str)


@pytest.mark.asyncio
async def test_answer_question_uses_similarity_search(temp_vector_store):
    """Test that answer_question uses similarity search with scores."""
    from qa_service import QAService
    from langchain_core.documents import Document
    
    with patch('qa_service.FAISS') as mock_faiss:
        mock_vector_store = MagicMock()
        mock_vector_store.similarity_search_with_score = MagicMock(return_value=[
            (Document(page_content="Specific test content", metadata={}), 0.3)
        ])
        mock_faiss.load_local.return_value = mock_vector_store
        
        service = QAService(temp_vector_store)
        await service.answer_question("Test question")
        
        # Verify similarity_search_with_score was called
        mock_vector_store.similarity_search_with_score.assert_called_once()


@pytest.mark.asyncio
async def test_answer_question_with_no_documents(temp_vector_store):
    """Test answer_question when no documents are retrieved or all filtered out."""
    from qa_service import QAService
    from langchain_core.documents import Document
    
    with patch('qa_service.FAISS') as mock_faiss:
        mock_vector_store = MagicMock()
        # Return documents with very low similarity (high distance score)
        # Score > 1.0 means very dissimilar, should be filtered out
        mock_vector_store.similarity_search_with_score = MagicMock(return_value=[
            (Document(page_content="Irrelevant content", metadata={}), 1.5)
        ])
        mock_faiss.load_local.return_value = mock_vector_store
        
        service = QAService(temp_vector_store)
        answer = await service.answer_question("Test question")
        
        assert answer == "Not found."


@pytest.mark.asyncio
async def test_answer_question_combines_context(temp_vector_store):
    """Test that multiple retrieved documents are combined."""
    from qa_service import QAService
    from langchain_core.documents import Document
    
    with patch('qa_service.FAISS') as mock_faiss, \
         patch('qa_service.ChatOpenAI') as mock_chat:
        
        # Multiple documents with good similarity scores
        mock_vector_store = MagicMock()
        mock_vector_store.similarity_search_with_score = MagicMock(return_value=[
            (Document(page_content="First context", metadata={}), 0.2),
            (Document(page_content="Second context", metadata={}), 0.3),
        ])
        mock_faiss.load_local.return_value = mock_vector_store
        
        # Mock LLM to capture the prompt
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Combined answer"
        mock_llm_instance.invoke = MagicMock(return_value=mock_response)
        mock_chat.return_value = mock_llm_instance
        
        service = QAService(temp_vector_store)
        await service.answer_question("Test question")
        
        # Verify LLM was called
        mock_llm_instance.invoke.assert_called_once()
        # The prompt should contain both contexts
        call_args = mock_llm_instance.invoke.call_args[0][0]
        assert "First context" in call_args
        assert "Second context" in call_args


@pytest.mark.asyncio
async def test_answer_question_prompt_template(temp_vector_store):
    """Test that the prompt template is correctly formatted."""
    from qa_service import QAService
    from langchain_core.documents import Document
    
    with patch('qa_service.ChatOpenAI') as mock_chat, \
         patch('qa_service.FAISS') as mock_faiss:
        
        # Mock vector store with good similarity
        mock_vector_store = MagicMock()
        mock_vector_store.similarity_search_with_score = MagicMock(return_value=[
            (Document(page_content="Test context", metadata={}), 0.2)
        ])
        mock_faiss.load_local.return_value = mock_vector_store
        
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Test answer"
        mock_llm_instance.invoke = MagicMock(return_value=mock_response)
        mock_chat.return_value = mock_llm_instance
        
        service = QAService(temp_vector_store)
        await service.answer_question("What is this?")
        
        # Check that prompt was formatted correctly
        call_args = mock_llm_instance.invoke.call_args[0][0]
        assert "Context:" in call_args
        assert "Question: What is this?" in call_args
        assert "Answer:" in call_args


def test_qa_service_uses_config_values(temp_vector_store):
    """Test that QAService uses configuration values."""
    from qa_service import QAService
    from config import LLM_MODEL, LLM_TEMPERATURE, EMBEDDING_MODEL
    
    with patch('qa_service.ChatOpenAI') as mock_chat, \
         patch('qa_service.OpenAIEmbeddings') as mock_embeddings:
        
        service = QAService(temp_vector_store)
        
        # Verify ChatOpenAI was called with config values
        mock_chat.assert_called_once()
        chat_kwargs = mock_chat.call_args.kwargs
        assert chat_kwargs['model'] == LLM_MODEL
        assert chat_kwargs['temperature'] == LLM_TEMPERATURE
        
        # Verify OpenAIEmbeddings was called with config values
        mock_embeddings.assert_called_once()
        embed_kwargs = mock_embeddings.call_args.kwargs
        assert embed_kwargs['model'] == EMBEDDING_MODEL


def test_qa_service_similarity_threshold_configuration(temp_vector_store):
    """Test that similarity threshold is configurable."""
    from qa_service import QAService
    
    service = QAService(temp_vector_store, k=7, similarity_threshold=0.7)
    
    assert service.k == 7
    assert service.similarity_threshold == 0.7


@pytest.mark.asyncio
async def test_answer_question_strips_whitespace(temp_vector_store):
    """Test that answer is stripped of leading/trailing whitespace."""
    from qa_service import QAService
    from langchain_core.documents import Document
    
    with patch('qa_service.ChatOpenAI') as mock_chat, \
         patch('qa_service.FAISS') as mock_faiss:
        
        # Mock vector store with good similarity
        mock_vector_store = MagicMock()
        mock_vector_store.similarity_search_with_score = MagicMock(return_value=[
            (Document(page_content="Test context", metadata={}), 0.2)
        ])
        mock_faiss.load_local.return_value = mock_vector_store
        
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "  Answer with spaces  \n"
        mock_llm_instance.invoke = MagicMock(return_value=mock_response)
        mock_chat.return_value = mock_llm_instance
        
        service = QAService(temp_vector_store)
        answer = await service.answer_question("Test")
        
        assert answer == "Answer with spaces"
        assert not answer.startswith(" ")
        assert not answer.endswith(" ")


@pytest.mark.asyncio
async def test_concurrent_questions(temp_vector_store):
    """Test that multiple questions can be answered concurrently."""
    import asyncio
    from qa_service import QAService
    
    service = QAService(temp_vector_store)
    
    questions = ["Question 1", "Question 2", "Question 3"]
    answers = await asyncio.gather(*[
        service.answer_question(q) for q in questions
    ])
    
    assert len(answers) == 3
    assert all(isinstance(a, str) for a in answers)
    assert all(len(a) > 0 for a in answers)

