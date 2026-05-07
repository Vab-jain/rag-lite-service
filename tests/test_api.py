import pytest
from fastapi.testclient import TestClient
from app.api.server import app

client = TestClient(app)

def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_query_validation_error():
    # Test query with profanity
    response = client.post("/query", json={"query": "What the fuck?"})
    assert response.status_code == 400
    assert "prohibited language" in response.json()["detail"]

    # Test empty query or invalid schema
    response = client.post("/query", json={})
    assert response.status_code == 422 # FastAPI validation error

# A full query test would require mocking the qa_chain dependency.
# In FastAPI, we can use app.dependency_overrides.
from app.api.server import get_qa_chain
from unittest.mock import MagicMock

def test_query_success():
    mock_chain = MagicMock()
    mock_chain.answer_question.return_value = {
        "answer": "Mocked RL answer.",
        "citations": ["1"],
        "sources": [{"metadata": {"page": "1"}}],
        "attribution_passed": True
    }
    
    app.dependency_overrides[get_qa_chain] = lambda: mock_chain
    
    response = client.post("/query", json={"query": "What is RL?"})
    assert response.status_code == 200
    
    data = response.json()
    assert data["answer"] == "Mocked RL answer."
    assert data["attribution_passed"] is True
    
    # Clean up overrides
    app.dependency_overrides.clear()
