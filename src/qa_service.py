"""Question-Answering service using LangChain and FAISS vector store."""
import asyncio
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

# Load environment variables from .env file (backup - config.py also loads it)
load_dotenv()

# Import configuration (config.py loads .env and validates API key)
from config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    RETRIEVAL_K,
    SIMILARITY_THRESHOLD,
)


class QAService:
    """Service for answering questions using RAG with LangChain."""
    
    def __init__(self, vector_store_path: Path, k: int = None, similarity_threshold: float = None):
        """
        Initialize the QA service.
        
        Args:
            vector_store_path: Path where the FAISS vector store is stored
            k: Number of chunks to retrieve for context (defaults to config value)
            similarity_threshold: Minimum similarity score for retrieved chunks (defaults to config value)
        """
        self.vector_store_path = Path(vector_store_path)
        self.k = k if k is not None else RETRIEVAL_K
        self.similarity_threshold = similarity_threshold if similarity_threshold is not None else SIMILARITY_THRESHOLD
        
        # Check if vector store directory exists (FAISS saves to a directory)
        if not self.vector_store_path.exists():
            raise RuntimeError(f"Vector store not found: {self.vector_store_path}")
        
        # Load FAISS vector store
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
        self.vector_store = FAISS.load_local(
            str(self.vector_store_path),
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Note: We don't create a retriever here anymore
        # Instead, we'll use similarity_search_with_score for threshold filtering
        
        # PERFORMANCE: Initialize LLM with optimized settings
        # - gpt-4o-mini: Fast inference, cost-effective (~80% cheaper than GPT-4)
        # - temperature=0: Deterministic responses, faster generation
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=OPENAI_API_KEY
        )
        
        # Create prompt template with citation instructions
        self.prompt_template = PromptTemplate(
            template="""Use the following pieces of context to answer the question. 
Each piece of context includes source information in brackets (e.g., [Source: filename | Pages: 1-2]).

IMPORTANT INSTRUCTIONS:
1. Ground your answer in the provided context - only use information from the given sources
2. When possible, cite your sources by mentioning the document name or page numbers (e.g., "According to the document..." or "As stated on page 5...")
3. Be transparent about what is directly stated versus what you are inferring (e.g., "Based on the context, it appears that..." or "The document suggests that...")
4. If information is incomplete or requires inference, mention this in your answer
5. Only if the context provides absolutely no relevant information should you indicate that the required context is not found

Context:
{context}

Question: {question}

Answer: """,
            input_variables=["context", "question"]
        )
    
    async def answer_question(self, question: str) -> str:
        """
        Answer a question based on the indexed documents.
        
        Args:
            question: The question to answer
            
        Returns:
            The answer to the question
        """
        # PERFORMANCE: Retrieve relevant documents with similarity scores (run in thread to avoid blocking)
        # asyncio.to_thread() offloads blocking I/O to thread pool
        # This allows multiple questions to be processed concurrently
        docs_with_scores = await asyncio.to_thread(
            self.vector_store.similarity_search_with_score,
            question,
            k=self.k
        )
        
        # OPTIMIZATION: Filter by similarity threshold to avoid irrelevant context
        # FAISS uses L2 distance: lower scores = more similar
        # We filter to keep only documents with similarity above threshold
        # Note: For FAISS with L2 distance, we want scores BELOW a threshold (inverse)
        # But for cosine similarity (after normalization), higher is better
        # OpenAI embeddings are normalized, so we use distance as inverse similarity
        filtered_docs = [
            doc for doc, score in docs_with_scores 
            if score <= (2 - 2 * self.similarity_threshold)  # Convert cosine similarity threshold to L2 distance
        ]
        
        # OPTIMIZATION: Early exit if no relevant context found
        # Avoids unnecessary LLM API call, saving time and cost
        if not filtered_docs:
            return "Not found."
        
        docs = filtered_docs
        
        # GROUNDING: Combine context with source metadata for citations
        # Include filename, page numbers, and section titles when available
        context_parts = []
        for idx, doc in enumerate(docs, 1):
            metadata_info = []
            
            # Add filename if available
            if 'filename' in doc.metadata:
                metadata_info.append(f"Source: {doc.metadata['filename']}")
            
            # Add page numbers for PDFs
            if 'page_numbers' in doc.metadata and doc.metadata['page_numbers']:
                pages = doc.metadata['page_numbers']
                if isinstance(pages, list):
                    metadata_info.append(f"Pages: {', '.join(map(str, pages))}")
                else:
                    metadata_info.append(f"Page: {pages}")
            
            # Add section title/heading for PDFs
            if 'title' in doc.metadata and doc.metadata['title']:
                metadata_info.append(f"Section: {doc.metadata['title']}")
            
            # Create metadata header
            meta_str = " | ".join(metadata_info) if metadata_info else f"Chunk {idx}"
            
            # Format: [Source metadata]\nContent
            context_parts.append(f"[{meta_str}]\n{doc.page_content}")
        
        # Join all context parts with clear separators
        context = "\n\n---\n\n".join(context_parts)
        
        # Format prompt
        prompt = self.prompt_template.format(context=context, question=question)
        
        # PERFORMANCE: Get answer from LLM (run in thread to avoid blocking)
        # Thread offloading enables concurrent API calls to OpenAI
        response = await asyncio.to_thread(self.llm.invoke, prompt)
        
        # Extract content from response
        if hasattr(response, "content"):
            return response.content.strip()
        elif isinstance(response, str):
            return response.strip()
        else:
            return str(response).strip()
