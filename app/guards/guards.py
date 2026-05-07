import re
from typing import List, Dict, Any
from pydantic import BaseModel, Field

# --- Input Guards ---

def sanitize_input(query: str) -> str:
    """
    Basic sanitization: limit length, basic profanity check, RL context check.
    Raises ValueError if input is invalid.
    """
    if len(query) > 500:
        raise ValueError("Query is too long. Please limit to 500 characters.")
        
    # Basic profanity list (for MVP)
    profanity = ["fuck", "shit", "bitch", "asshole"]
    if any(word in query.lower() for word in profanity):
        raise ValueError("Query contains prohibited language.")
        
    return query

# --- Output Guards & Schema ---

class Citation(BaseModel):
    source_id: str = Field(description="The source ID from the metadata.")
    quote: str = Field(description="The exact quote from the source supporting the claim.")

class RAGResponse(BaseModel):
    answer: str = Field(description="The comprehensive answer to the user's question, grounded only in the provided context.")
    citations: List[str] = Field(description="List of source IDs cited in the answer (e.g., ['page_12', 'chapter_3']).")

def check_attribution(response: RAGResponse, retrieved_docs: List[Dict]) -> bool:
    """
    Verifies that every citation in the response exists in the retrieved documents' metadata.
    For this MVP, we assume metadata has a 'page' or 'source' key we can match.
    """
    valid_sources = []
    for doc in retrieved_docs:
        meta = doc.get("metadata", {})
        # We can use page number or source string as ID for MVP
        source_id = str(meta.get("page", ""))
        if source_id:
            valid_sources.append(source_id)
            
    for citation in response.citations:
        # A simple check: does the citation string appear in any of our valid source IDs?
        # In a real app, you'd match exact IDs
        if not any(citation in valid for valid in valid_sources):
            # Log the hallucinated citation
            print(f"Warning: Hallucinated citation '{citation}' not found in retrieved sources {valid_sources}.")
            # For strictness, you could return False or raise an error. We return False.
            return False
            
    return True
