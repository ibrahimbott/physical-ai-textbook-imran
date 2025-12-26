# Implementation Plan: Deployment Readiness

**Feature**: Deployment Readiness
**Status**: Implemented

## Technical Context
Vercel requires specific configuration to bridge a compiled frontend (Docusaurus) and a Python backend.

### Changes
1.  **Vercel Config**:
    -   Created `vercel.json`.
    -   Builds: `frontend-book` (Static), `backend/rag_system.py` (Python).
    -   Rewrites: `/api/*` -> `backend/rag_system.py`.
2.  **Frontend Config**:
    -   Updated `frontend-book/src/components/Chatbot/index.js` to use relative path `/api/chat` in production.
3.  **Git Hygiene**:
    -   Created `.gitignore` with Python/Node templates.
    -   Removed `node_modules` from git index.
