from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import settings
from app.guards.guards import RAGResponse, sanitize_input, check_attribution
from app.rag.retrievers import VectorRetriever, BM25Retriever, EnsembleRetriever
from app.rag.reranker import DocumentReranker
import os

# Define the Prompt for the Rugby/RL Agent
SYSTEM_PROMPT = """You are an expert Reinforcement Learning teaching assistant, specifically grounded in the book by Sutton & Barto.
Your goal is to help the user understand RL concepts based ONLY on the provided context.
If the context does not contain the answer, say "I cannot answer this based on the provided RL book context."
Do NOT use external knowledge. 
Always cite the source page numbers for your claims.

Context Documents:
{context}
"""

class QAChain:
    def __init__(self):
        # We assume GROQ_API_KEY is in env or settings
        api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        
        # We'll use a fast model
        self.llm = ChatGroq(
            api_key=api_key,
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_retries=2
        )
        
        # Use structured output to force adherence to our schema
        self.structured_llm = self.llm.with_structured_output(RAGResponse)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}")
        ])
        
        # Initialize Retrievers
        self.vector_retriever = VectorRetriever()
        self.bm25_retriever = BM25Retriever()
        
        # Load BM25 index if available
        bm25_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'bm25_index.pkl')
        self.bm25_retriever.load(bm25_path)
        
        self.ensemble = EnsembleRetriever(self.bm25_retriever, self.vector_retriever)
        self.reranker = DocumentReranker()

    def format_docs(self, docs: List[Dict]) -> str:
        formatted = []
        for d in docs:
            page = d.get('metadata', {}).get('page', 'unknown')
            text = d.get('text', '')
            formatted.append(f"--- Page: {page} ---\n{text}")
        return "\n\n".join(formatted)

    def answer_question(self, query: str) -> Dict[str, Any]:
        """
        End-to-end pipeline: Input Guard -> Retrieve -> Rerank -> Prompt -> LLM -> Output Guard
        """
        # 1. Input Guard
        clean_query = sanitize_input(query)
        
        # 2. Retrieve
        retrieved_docs = self.ensemble.retrieve(clean_query, top_k=15)
        
        # 3. Rerank
        top_docs = self.reranker.rerank(clean_query, retrieved_docs, top_k=5)
        
        # Format for prompt
        context_str = self.format_docs(top_docs)
        
        # 4. Prompt & LLM Call
        chain = self.prompt | self.structured_llm
        
        # It's possible to mock this entirely during tests
        try:
            response: RAGResponse = chain.invoke({
                "context": context_str,
                "question": clean_query
            })
        except Exception as e:
            # Fallback if Groq API fails or schema fails
            return {
                "answer": f"Error generating answer: {e}",
                "citations": [],
                "sources": top_docs,
                "attribution_passed": False
            }
            
        # 5. Output Guard (Attribution)
        is_attributed = check_attribution(response, top_docs)
        
        return {
            "answer": response.answer,
            "citations": response.citations,
            "sources": top_docs,
            "attribution_passed": is_attributed
        }
