from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.rag.qa_chain import QAChain
from app.core.logger import logger
import time

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="RL RAG Agent MVP")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (lazy init)
qa_chain = None

def get_qa_chain():
    global qa_chain
    if qa_chain is None:
        qa_chain = QAChain()
    return qa_chain

class QueryRequest(BaseModel):
    query: str

@app.get("/healthz")
def health_check():
    return {"status": "ok", "service": "RL RAG Agent"}

@app.post("/query")
@limiter.limit("30/minute")
def query_rag(request: Request, body: QueryRequest, chain: QAChain = Depends(get_qa_chain)):
    logger.info(f"Received query: {body.query}")
    start_time = time.time()
    
    try:
        result = chain.answer_question(body.query)
        latency = time.time() - start_time
        logger.info(f"Query processed in {latency:.2f}s")
        
        return {
            "answer": result["answer"],
            "citations": result["citations"],
            "attribution_passed": result["attribution_passed"],
            "latency_seconds": round(latency, 2)
        }
    except ValueError as ve:
        logger.warning(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while processing query.")
