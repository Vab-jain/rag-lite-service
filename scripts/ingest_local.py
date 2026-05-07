"""
Basic end-to-end ingestion pipeline for RAG system.
Loads PDFs, processes them, embeds, and stores in Qdrant.
"""

import sys
import os

# Add the app directory to the path so we can import from rag modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from rag.ingest import load_pdf_docs, clean_and_chunk_docs
from rag.vectorstore import create_vectorstore, upsert_embeddings_from_documents, get_collection_info
from rag.embeddings import batch_embed_with_metadata
from rag.retrievers import BM25Retriever

def run_ingestion_pipeline():
    """
    Basic end-to-end ingestion pipeline:
    1. Load PDF documents
    2. Clean and chunk them
    3. Create vector store
    4. Embed and upsert to Qdrant
    5. Show final stats
    """
    print("=== Starting RAG Ingestion Pipeline ===\n")
    
    # Step 1: Load PDF documents
    print("Step 1: Loading PDF documents...")
    try:
        docs = load_pdf_docs()
        print(f"Loaded {len(docs)} pages from PDF")
    except Exception as e:
        print(f"Error loading PDFs: {e}")
        return False
    
    if not docs:
        print("No documents loaded. Exiting.")
        return False
    
    # Step 2: Clean and chunk documents
    print("\nStep 2: Cleaning and chunking documents...")
    try:
        chunks = clean_and_chunk_docs(docs)
        print(f"Created {len(chunks)} chunks")
        avg_size = sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
        print(f"Average chunk size: {avg_size:.1f} characters")
    except Exception as e:
        print(f"Error processing documents: {e}")
        return False
    
    # Step 3: Create vector store collection
    print("\nStep 3: Creating vector store...")
    try:
        create_vectorstore()
        print("Vector store ready")
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return False
    
    # Step 4: Embed and upsert documents
    print(f"\nStep 4: Embedding and storing {len(chunks)} chunks...")
    try:
        success = upsert_embeddings_from_documents(chunks, batch_size=16)
        if success:
            print("Successfully stored all embeddings")
        else:
            print("Failed to store embeddings")
            return False
    except Exception as e:
        print(f"Error embedding/storing documents: {e}")
        return False
        
    print("\nStep 4b: Creating and saving BM25 index...")
    try:
        bm25_retriever = BM25Retriever()
        
        # Prepare format for BM25 (text and metadata)
        documents_for_bm25 = [{"text": doc.page_content, "metadata": doc.metadata} for doc in chunks]
        bm25_retriever.fit(documents_for_bm25)
        
        # Save index
        os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'data'), exist_ok=True)
        bm25_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'bm25_index.pkl')
        bm25_retriever.save(bm25_path)
        print(f"Successfully saved BM25 index to {bm25_path}")
    except Exception as e:
        print(f"Error creating BM25 index: {e}")
    
    # Step 5: Show final stats
    print("\nStep 5: Final statistics...")
    try:
        info = get_collection_info()
        if info:
            print(f"Collection: {info['collection_name']}")
            print(f"Total documents stored: {info['points_count']}")
            print(f"Vector dimension: {info['vector_size']}")
            print(f"Status: {info['status']}")
        else:
            print("Could not retrieve collection info")
    except Exception as e:
        print(f"Error getting collection info: {e}")
    
    print("\n=== Ingestion Pipeline Complete ===")
    return True

def test_search():
    """
    Test the search functionality with sample queries.
    """
    from rag.vectorstore import search_similar_documents
    
    print("\n=== Testing Search Functionality ===")
    
    test_queries = [
        "What is reinforcement learning?",
        "How does Q-learning work?",
        "What are the main components of an MDP?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        try:
            results = search_similar_documents(query, limit=2, score_threshold=0.5)
            if results:
                print(f"Found {len(results)} results:")
                for i, result in enumerate(results):
                    print(f"  {i+1}. Score: {result['score']:.3f}")
                    print(f"     Text: {result['text'][:100]}...")
            else:
                print("  No results found")
        except Exception as e:
            print(f"  Search error: {e}")

if __name__ == "__main__":
    # Run the ingestion pipeline
    success = run_ingestion_pipeline()
    
    # If successful, test search functionality
    if success:
        test_search()
    else:
        print("Ingestion failed. Cannot test search.")
        sys.exit(1)
