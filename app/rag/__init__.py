"""
RAG (Retrieval-Augmented Generation) module for document processing and vector search.

This module provides functionality for:
- Loading and processing PDF documents
- Embedding text using BGE models
- Storing embeddings in Qdrant vector database
- Searching for similar documents
"""

# Import main functions from each module
from .embeddings import (
    load_embedding_model,
    batch_embed_documents,
    batch_embed_with_metadata,
    embed_single_query
)

from .vectorstore import (
    create_vectorstore,
    upsert_points,
    upsert_embeddings_from_documents,
    search_similar_documents,
    get_collection_info
)

from .ingest import (
    clean_text,
    load_pdf_docs,
    clean_and_chunk_docs
)

# Version info
__version__ = "1.0.0"
__author__ = "VJ"

# Export all main functions
__all__ = [
    # Embedding functions
    "load_embedding_model",
    "batch_embed_documents", 
    "batch_embed_with_metadata",
    "embed_single_query",
    
    # Vector store functions
    "create_vectorstore",
    "upsert_points",
    "upsert_embeddings_from_documents", 
    "search_similar_documents",
    "get_collection_info",
    
    # Document processing functions
    "clean_text",
    "load_pdf_docs",
    "clean_and_chunk_docs"
]
