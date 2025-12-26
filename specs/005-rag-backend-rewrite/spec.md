# Feature Specification: RAG System Rewrite
**Feature Branch**: `005-rag-backend-rewrite`
**Created**: 2025-12-26
**Status**: Implemented
**Input**: User request: "Fix RAG system, use Gemini via OpenAI SDK + Cohere Embeddings + Qdrant"

## Goal
Rewrite the legacy RAG system to be more robust, cost-effective, and easier to use.

## Success Criteria
- [x] Integrate Google Gemini Models via OpenAI Python SDK (Base URL compatibility).
- [x] Use Cohere `embed-multilingual-v3.0` for high-quality embeddings.
- [x] Use Qdrant (Cloud) for vector storage.
- [x] Implement CLI tools (`--ingest`, `--ask`) for easy management.
- [x] Handle model fallbacks (e.g., `gemini-1.5-flash` -> `gemini-flash-latest`).

## User Stories
### Story 1: Reliable Data Ingestion
As a developer, I want to ingest all textbook content into a vector database so the AI knows the course material.
**Acceptance:** Running `python rag_system.py --ingest` parses docs, embeds them with Cohere, and upserts to Qdrant.

### Story 2: Accurate Question Answering
As a student, I want to ask questions about "Physical AI" and get answers based ONLY on the textbook.
**Acceptance:** The system retrieves relevant chunks and the AI answers using that context, citing sources.

### Story 3: Robustness
As a user, I want the system to work even if a specific Gemini model version is deprecated.
**Acceptance:** The system automatically falls back to available models.
