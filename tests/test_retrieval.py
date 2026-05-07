import pytest
from unittest.mock import patch
from app.rag.retrievers import BM25Retriever, EnsembleRetriever

@pytest.fixture
def sample_docs():
    return [
        {"text": "Reinforcement learning is learning what to do.", "metadata": {"page": 1}},
        {"text": "Q-learning is an off-policy TD control algorithm.", "metadata": {"page": 2}},
        {"text": "Markov Decision Processes formally describe an environment.", "metadata": {"page": 3}}
    ]

def test_bm25_retriever(sample_docs):
    retriever = BM25Retriever()
    retriever.fit(sample_docs)
    
    # "TD control" should match the second doc
    results = retriever.retrieve("TD control algorithm", top_k=1)
    
    assert len(results) == 1
    assert "Q-learning" in results[0]["text"]
    assert results[0]["bm25_score"] > 0

@patch('app.rag.retrievers.VectorRetriever')
def test_ensemble_retriever(MockVectorRetriever, sample_docs):
    # Mock vector retriever to return doc 3 for "MDP"
    mock_vr_instance = MockVectorRetriever.return_value
    mock_vr_instance.retrieve.return_value = [
        {"text": "Markov Decision Processes formally describe an environment.", "metadata": {"page": 3}, "vector_score": 0.9}
    ]
    
    bm25 = BM25Retriever()
    bm25.fit(sample_docs)
    
    ensemble = EnsembleRetriever(bm25, mock_vr_instance)
    results = ensemble.retrieve("MDP formally describe environment", top_k=2)
    
    assert len(results) > 0
    assert "Markov Decision Processes" in results[0]["text"]
    assert "rrf_score" in results[0]
