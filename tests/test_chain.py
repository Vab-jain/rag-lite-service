import pytest
from unittest.mock import patch, MagicMock
from app.guards.guards import sanitize_input, check_attribution, RAGResponse

def test_sanitize_input():
    # Valid input
    assert sanitize_input("What is Q-learning?") == "What is Q-learning?"
    
    # Too long
    long_query = "a" * 600
    with pytest.raises(ValueError, match="too long"):
        sanitize_input(long_query)
        
    # Profanity
    with pytest.raises(ValueError, match="prohibited language"):
        sanitize_input("What the fuck is this?")

def test_check_attribution():
    response = RAGResponse(
        answer="Q-learning is an off-policy TD control algorithm.",
        citations=["15", "42"]
    )
    
    docs = [
        {"text": "...", "metadata": {"page": "15"}},
        {"text": "...", "metadata": {"page": "42"}}
    ]
    
    assert check_attribution(response, docs) is True
    
    # Hallucinated citation
    response_bad = RAGResponse(
        answer="...",
        citations=["99"]
    )
    assert check_attribution(response_bad, docs) is False

@patch('app.rag.qa_chain.ChatGroq')
@patch('app.rag.qa_chain.EnsembleRetriever')
@patch('app.rag.qa_chain.DocumentReranker')
def test_qa_chain(MockReranker, MockRetriever, MockChatGroq):
    from app.rag.qa_chain import QAChain
    
    # Mock LLM structured output
    mock_llm_instance = MockChatGroq.return_value
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = RAGResponse(
        answer="A simulated answer about RL.",
        citations=["25"]
    )
    
    # Mock chain composition behavior
    mock_prompt = MagicMock()
    mock_prompt.__or__.return_value = mock_structured
    
    # Since we use self.prompt | self.structured_llm, we patch the chain instance inside QAChain
    with patch('app.rag.qa_chain.ChatPromptTemplate.from_messages', return_value=mock_prompt):
        chain = QAChain()
        
        # Override the composed chain locally to test
        chain.prompt = mock_prompt
        chain.structured_llm = mock_structured
        
        # Mock retrieval
        mock_retriever_instance = MockRetriever.return_value
        mock_retriever_instance.retrieve.return_value = [
            {"text": "doc 1", "metadata": {"page": "25"}}
        ]
        
        # Mock reranker
        mock_reranker_instance = MockReranker.return_value
        mock_reranker_instance.rerank.return_value = [
            {"text": "doc 1", "metadata": {"page": "25"}}
        ]
        
        result = chain.answer_question("Tell me about RL.")
        
        assert result["answer"] == "A simulated answer about RL."
        assert result["attribution_passed"] is True
