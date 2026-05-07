import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from rag.retrievers import VectorRetriever, BM25Retriever, EnsembleRetriever
from rag.reranker import DocumentReranker

def main():
    print("=== RAG Query System ===")
    query = input("Enter your question: ")
    
    print("\nRetrieving documents...")
    # Initialize retrievers
    vector_retriever = VectorRetriever()
    bm25_retriever = BM25Retriever()
    
    # Try to load BM25 index if exists, else fallback to vector only
    bm25_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'bm25_index.pkl')
    bm25_retriever.load(bm25_path)
    
    ensemble = EnsembleRetriever(bm25_retriever, vector_retriever)
    
    # Retrieve top 20 candidates
    candidates = ensemble.retrieve(query, top_k=20)
    
    if not candidates:
        print("No candidates found.")
        return
        
    print(f"Retrieved {len(candidates)} candidates. Reranking...")
    reranker = DocumentReranker()
    top_docs = reranker.rerank(query, candidates, top_k=5)
    
    print("\n=== Top Results ===")
    for i, doc in enumerate(top_docs):
        print(f"\n[{i+1}] Score: {doc.get('rerank_score', 0):.4f} | Page: {doc.get('metadata', {}).get('page', 'N/A')}")
        print(f"Text: {doc['text'][:200]}...")

if __name__ == "__main__":
    main()
