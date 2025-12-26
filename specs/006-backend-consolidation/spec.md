# Feature Specification: Backend Consolidation
**Feature Branch**: `006-backend-consolidation`
**Created**: 2025-12-26
**Status**: Implemented
**Input**: User request: "Merge chatbot folder logic into backend, delete chatbot folder, standalone backend."

## Goal
Simplify the project structure by consolidating all backend logic into a single `backend` directory and removing the redundant `chatbot` demo folder.

## Success Criteria
- [x] `chatbot` folder is deleted from root.
- [x] Agentic logic (Function Calling) from `chatbot/api.py` is merged into `backend/rag_system.py`.
- [x] `backend/rag_system.py` serves both API (FastAPI) and CLI.
- [x] Static UI files are served from `backend/static`.
- [x] `start_all.bat` launches the consolidated backend successfully.

## User Stories
### Story 1: Unified Architecture
As a developer, I want a single source of truth for the RAG backend so I don't have to maintain two duplicate codes.
**Acceptance:** `rag_system.py` contains the advanced agent logic, and `chatbot/` is gone.

### Story 2: Agentic RAG
As a user, I want the chatbot to use "smart" agentic behaviors (function calling) to decide when to search.
**Acceptance:** `rag_system.py` uses `openai.chat.completions` with `tools=[retrieve_knowledge]`.
