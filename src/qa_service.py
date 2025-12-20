"""Question-Answering service using LangChain and FAISS vector store."""
import os
from pathlib import Path
from typing import List

# Fix OpenMP warning (must be set before importing FAISS)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()


class QAService:
    """Service for answering questions using RAG with LangChain."""
    
    def __init__(self, vector_store_path: Path, k: int = 5):
        """
        Initialize the QA service.
        
        Args:
            vector_store_path: Path where the FAISS vector store is stored
            k: Number of chunks to retrieve for context
        """
        self.vector_store_path = Path(vector_store_path)
        self.k = k
        
        # Check if vector store directory exists (FAISS saves to a directory)
        if not self.vector_store_path.exists():
            raise RuntimeError(f"Vector store not found: {self.vector_store_path}")
        
        # Load FAISS vector store
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.vector_store = FAISS.load_local(
            str(self.vector_store_path),
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Create retriever from vector store
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": self.k})
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create prompt template
        self.prompt_template = PromptTemplate(
            template="""Use the following pieces of context to answer the question. 
You may infer information from the provided context, but be transparent about what is directly stated versus what you are inferring.
If you need to make inferences or the information appears incomplete, mention this in your answer (e.g., "Based on the context, it appears that..." or "The document suggests that..., though this may be incomplete").
Only if the context provides absolutely no relevant information to answer the question should you indicate that the required context is not available.

Context:
{context}

Question: {question}

Answer: """,
            input_variables=["context", "question"]
        )
    
    def answer_question(self, question: str) -> str:
        """
        Answer a question based on the indexed documents.
        
        Args:
            question: The question to answer
            
        Returns:
            The answer to the question
        """
        # Retrieve relevant documents
        # Use invoke() for newer LangChain versions (LangChain 0.1+)
        docs = self.retriever.invoke(question)
        
        if not docs:
            return "Required context not available in the document."
        
        # Combine context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Format prompt
        prompt = self.prompt_template.format(context=context, question=question)
        
        # Get answer from LLM
        response = self.llm.invoke(prompt)
        
        # Extract content from response
        if hasattr(response, "content"):
            return response.content.strip()
        elif isinstance(response, str):
            return response.strip()
        else:
            return str(response).strip()
