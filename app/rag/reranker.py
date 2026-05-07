from sentence_transformers import CrossEncoder
from typing import List, Dict

# Using the lightweight CPU-friendly model for local testing as requested
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class DocumentReranker:
    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        # We load this lazily or once on startup
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Re-ranks a list of documents based on cross-encoder similarity with the query.
        documents: List of dicts, must contain 'text'
        """
        if not documents:
            return []

        # CrossEncoder expects pairs of (query, document)
        pairs = [[query, doc["text"]] for doc in documents]
        
        # Predict scores
        scores = self.model.predict(pairs)
        
        # Add scores to documents
        for i, doc in enumerate(documents):
            doc["rerank_score"] = float(scores[i])
            
        # Sort by rerank score descending
        reranked_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        
        return reranked_docs[:top_k]
