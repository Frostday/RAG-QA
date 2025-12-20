"""
Document indexing service that supports both PDF and JSON documents.

For PDFs: Uses Docling for structured parsing and chunking (layout, tables, headings).
For JSON: Uses LangChain's RecursiveCharacterTextSplitter for chunking.
Storage: Uses LangChain's FAISS (in-memory) vector store.
"""
import json
import os
import pickle
from pathlib import Path
from typing import List

# Fix OpenMP warning (must be set before importing FAISS)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dotenv import load_dotenv
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()


class DocumentIndexer:
    """Indexes documents (PDF or JSON) into LangChain FAISS vector store."""
    
    def __init__(self, vector_store_path: Path):
        """
        Initialize the document indexer.
        
        Args:
            vector_store_path: Path where the vector store will be saved/loaded
        """
        self.vector_store_path = Path(vector_store_path)
        self.vector_store_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize text splitter for JSON
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def _index_pdf(self, file_path: str, filename: str) -> List[Document]:
        """Index a PDF file using docling, convert to LangChain Documents."""
        converter = DocumentConverter()
        chunker = HybridChunker(merge_peers=True)
        
        # Convert PDF to structured document
        result = converter.convert(file_path)
        dl_doc = result.document
        
        # Chunk the document using docling
        chunk_iter = chunker.chunk(dl_doc=dl_doc)
        chunks = list(chunk_iter)
        
        # Convert docling chunks to LangChain Documents
        documents = []
        for idx, chunk in enumerate(chunks):
            page_numbers = [
                page_no
                for page_no in sorted(
                    set(
                        prov.page_no
                        for item in chunk.meta.doc_items
                        for prov in item.prov
                    )
                )
            ] or None
            
            title = chunk.meta.headings[0] if chunk.meta.headings else None
            
            # Create LangChain Document
            doc = Document(
                page_content=chunk.text,
                metadata={
                    "filename": filename,
                    "page_numbers": page_numbers,
                    "title": title,
                    "chunk_index": idx,
                    "source": filename,
                }
            )
            documents.append(doc)
        
        return documents
    
    def _index_json(self, file_path: str, filename: str) -> List[Document]:
        """
        Index a JSON file using structure-aware chunking.
        
        For JSON, we chunk at the object/array level to preserve structure:
        - If JSON is a list: each item becomes a chunk
        - If JSON is an object: each top-level key-value pair or the whole object becomes a chunk
        - Large chunks are further split using RecursiveCharacterTextSplitter
        
        This approach preserves JSON structure better than pure text splitting.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        documents = []
        
        if isinstance(data, list):
            # For lists: each item becomes a potential chunk
            for idx, item in enumerate(data):
                # Convert item to text representation
                item_text = self._json_to_text(item)
                if not item_text.strip():
                    item_text = json.dumps(item, ensure_ascii=False, indent=2)
                
                # If item is large, split it further
                if len(item_text) > 1000:
                    # Use text splitter for large items
                    item_doc = Document(
                        page_content=item_text,
                        metadata={
                            "filename": filename,
                            "source": filename,
                            "file_type": "json",
                            "list_index": idx
                        }
                    )
                    split_docs = self.text_splitter.split_documents([item_doc])
                    for split_idx, doc in enumerate(split_docs):
                        doc.metadata["chunk_index"] = len(documents) + split_idx
                        doc.metadata["list_index"] = idx
                    documents.extend(split_docs)
                else:
                    # Small enough to be a single chunk
                    doc = Document(
                        page_content=item_text,
                        metadata={
                            "filename": filename,
                            "source": filename,
                            "file_type": "json",
                            "chunk_index": len(documents),
                            "list_index": idx
                        }
                    )
                    documents.append(doc)
        
        elif isinstance(data, dict):
            # For objects: try to chunk by top-level keys, but keep related data together
            # Convert entire object to text first
            text_content = self._json_to_text(data)
            if not text_content.strip():
                text_content = json.dumps(data, ensure_ascii=False, indent=2)
            
            # If object is small, keep as single chunk
            if len(text_content) <= 1000:
                doc = Document(
                    page_content=text_content,
                    metadata={
                        "filename": filename,
                        "source": filename,
                        "file_type": "json",
                        "chunk_index": 0
                    }
                )
                documents.append(doc)
            else:
                # Large object: use text splitter but try to preserve structure
                initial_doc = Document(
                    page_content=text_content,
                    metadata={
                        "filename": filename,
                        "source": filename,
                        "file_type": "json"
                    }
                )
                split_docs = self.text_splitter.split_documents([initial_doc])
                for idx, doc in enumerate(split_docs):
                    doc.metadata["chunk_index"] = idx
                documents.extend(split_docs)
        
        else:
            # Primitive value: convert to text
            text_content = str(data)
            doc = Document(
                page_content=text_content,
                metadata={
                    "filename": filename,
                    "source": filename,
                    "file_type": "json",
                    "chunk_index": 0
                }
            )
            documents.append(doc)
        
        return documents
    
    def _json_to_text(self, data, indent=0) -> str:
        """
        Convert JSON data to a readable text representation.
        
        Args:
            data: JSON data (dict, list, or primitive)
            indent: Current indentation level
            
        Returns:
            Text representation of the JSON
        """
        text_parts = []
        indent_str = "  " * indent
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    text_parts.append(f"{indent_str}{key}:")
                    text_parts.append(self._json_to_text(value, indent + 1))
                elif isinstance(value, str) and len(value) > 0:
                    text_parts.append(f"{indent_str}{key}: {value}")
                elif value is not None:
                    text_parts.append(f"{indent_str}{key}: {value}")
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    text_parts.append(f"{indent_str}Item {idx + 1}:")
                    text_parts.append(self._json_to_text(item, indent + 1))
                elif isinstance(item, str) and len(item) > 0:
                    text_parts.append(f"{indent_str}Item {idx + 1}: {item}")
                elif item is not None:
                    text_parts.append(f"{indent_str}Item {idx + 1}: {item}")
        else:
            return f"{indent_str}{data}"
        
        return "\n".join(text_parts)
    
    def index_document(self, file_path: str, filename: str):
        """
        Index a document (PDF or JSON) into FAISS vector store.
        
        Args:
            file_path: Path to the document file
            filename: Original filename
        """
        file_ext = Path(file_path).suffix.lower()
        
        # Extract chunks based on file type
        if file_ext == ".pdf":
            documents = self._index_pdf(file_path, filename)
        elif file_ext == ".json":
            documents = self._index_json(file_path, filename)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        if not documents:
            raise RuntimeError("No chunks extracted from document")
        
        # Create FAISS vector store from documents
        vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # Save vector store to disk (FAISS can be saved/loaded)
        vector_store.save_local(str(self.vector_store_path))
        
        return len(documents)
