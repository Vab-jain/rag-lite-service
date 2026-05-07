from rank_bm25 import BM25Okapi
import numpy as np
import re
from typing import List, Dict

class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.documents = []

    def fit(self, documents: List[Dict]):
        """
        documents: List of dicts, e.g., [{"text": "...", "metadata": {...}}, ...]
        """
        self.documents = documents
        tokenized_corpus = [self._tokenize(doc["text"]) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        # Simple tokenization by word, lowercased
        return re.findall(r'\w+', text.lower())

    def save(self, filepath: str):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({'documents': self.documents, 'bm25': self.bm25}, f)

    def load(self, filepath: str):
        import pickle
        import os
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.bm25 = data['bm25']

    def retrieve(self, query: str, top_k: int = 20) -> List[Dict]:
        if not self.bm25:
            return []
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_n_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_n_indices:
            if scores[idx] > 0:
                doc = self.documents[idx].copy()
                doc['bm25_score'] = float(scores[idx])
                results.append(doc)
        return results

class VectorRetriever:
    def __init__(self, collection_name="rag-lite"):
        self.collection_name = collection_name
        
    def retrieve(self, query: str, top_k: int = 20) -> List[Dict]:
        from rag.vectorstore import search_similar_documents
        # Returns List[dict] with "text", "metadata", "score"
        results = search_similar_documents(query, collection_name=self.collection_name, limit=top_k)
        for r in results:
            r['vector_score'] = r.pop('score', 0)
        return results

class EnsembleRetriever:
    def __init__(self, bm25_retriever: BM25Retriever, vector_retriever: VectorRetriever):
        self.bm25 = bm25_retriever
        self.vector = vector_retriever

    def retrieve(self, query: str, top_k: int = 20, rrf_k: int = 60) -> List[Dict]:
        bm25_results = self.bm25.retrieve(query, top_k=top_k)
        vector_results = self.vector.retrieve(query, top_k=top_k)

        # Reciprocal Rank Fusion (RRF)
        # RRF_score = 1 / (rrf_k + rank)
        
        doc_map = {}
        
        # Process BM25 results
        for rank, doc in enumerate(bm25_results):
            text = doc["text"]
            if text not in doc_map:
                doc_map[text] = {"text": text, "metadata": doc["metadata"], "rrf_score": 0.0}
            doc_map[text]["rrf_score"] += 1.0 / (rrf_k + rank + 1)
            doc_map[text]["bm25_score"] = doc["bm25_score"]

        # Process Vector results
        for rank, doc in enumerate(vector_results):
            text = doc["text"]
            if text not in doc_map:
                doc_map[text] = {"text": text, "metadata": doc["metadata"], "rrf_score": 0.0}
            doc_map[text]["rrf_score"] += 1.0 / (rrf_k + rank + 1)
            doc_map[text]["vector_score"] = doc["vector_score"]

        # Sort by RRF score
        fused_results = list(doc_map.values())
        fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        
        return fused_results[:top_k]
