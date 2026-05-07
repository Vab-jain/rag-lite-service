# RL RAG Agent MVP

A minimal-yet-production-grade RAG Agent designed to help users learn **Reinforcement Learning** concepts, grounded in the canonical textbook by Richard S. Sutton and Andrew G. Barto.

## Features
- **Hybrid Retrieval**: Combines sparse (BM25) and dense (Qdrant Vector) search with Reciprocal Rank Fusion for high-recall document fetching.
- **Reranking**: Uses a CrossEncoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to re-score and select the top 5 most relevant context snippets.
- **Strict Guardrails**: Validates input to ensure it pertains to Reinforcement Learning and avoids profanity. Enforces citations in the LLM output ensuring grounding to the provided context.
- **FastAPI Service**: Provides robust endpoints with `slowapi` rate-limiting.
- **Dockerized**: Easy to deploy with a multi-stage Dockerfile and Docker Compose.

## Tech Stack
- **LLM**: Groq API (Llama 3.1) via LangChain.
- **Embeddings**: `BAAI/bge-large-en-v1.5` via HuggingFace.
- **Retrievers**: `rank_bm25` (Sparse) and `Qdrant` (Dense).
- **Serving**: FastAPI + Uvicorn.
- **Validation**: Pydantic.

## Getting Started Locally

### 1. Prerequisites
- Python 3.10+
- Docker & Docker Compose (Optional but recommended)
- A Groq API Key

### 2. Setup
Clone the repo and set up your virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set up your `.env` file (see `.env.example`):
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Ingest Documents (Run Locally)
First, spin up a local Qdrant instance if you are using Docker:
```bash
docker-compose up -d qdrant
```

Then, run the ingestion script to process the RL book, embed the chunks, and save the BM25 index:
```bash
python scripts/ingest_local.py
```

### 4. Query from CLI
You can test the RAG engine directly from your terminal:
```bash
python scripts/query.py
```

### 5. Run the API Server
Start the FastAPI application:
```bash
uvicorn app.api.server:app --host 0.0.0.0 --port 8000 --reload
```
Access the interactive docs at `http://localhost:8000/docs`.

### 6. Testing
Run the local unit tests (which mock the LLM and Qdrant database):
```bash
pytest tests/ -v
```
