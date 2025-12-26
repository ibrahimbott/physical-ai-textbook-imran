# Implementation Plan: RAG System Rewrite

**Feature**: RAG System Rewrite
**Status**: Implemented

## Technical Context
The previous system used `google-generativeai` and had stability issues. We moved to `openai` SDK for better stability and tooling support.

### Architecture
- **LLM**: Google Gemini (via OpenAI SDK Base URL)
- **Embeddings**: Cohere `embed-multilingual-v3.0`
- **Vector DB**: Qdrant Cloud
- **Interface**: CLI (`argparse`) + FastAPI foundation

## Implementation Steps
1.  **Environment Setup**: Configured `.env` with `GOOGLE_API_KEY`, `COHERE_API_KEY`, `QDRANT_URL`.
2.  **Ingestion Logic**:
    -   Glob all `.md`/`.mdx` files.
    -   Split using `RecursiveCharacterTextSplitter`.
    -   Embed batches using Cohere.
    -   Upsert to Qdrant with UUID5 deterministic IDs.
3.  **Retrieval Logic**:
    -   Embed query with Cohere.
    -   Query Qdrant (Hybrid/Dense search).
    -   Format context string.
4.  **Generation Logic**:
    -   System Prompt: "Answer ONLY from context."
    -   OpenAI Chat Completion call.
