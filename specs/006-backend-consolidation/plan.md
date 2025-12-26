# Implementation Plan: Backend Consolidation

**Feature**: Backend Consolidation
**Status**: Implemented

## Technical Context
The project had a "split brain" problem with `backend/` (old logic) and `chatbot/` (new logic). We merged them.

### Changes
1.  **Migration**:
    -   Copied `retrieve_knowledge` tool definition from `chatbot/api.py`.
    -   Copied `run_agent` loop (tool calling) to `rag_system.py`.
    -   Copied `static` assets (HTML/CSS) to `backend/static`.
2.  **Cleanup**:
    -   Deleted `e:\Physical-TextBook\chatbot`.
    -   Deleted unused `test_*.py` scripts in `backend/`.
    -   Added `httpx` and `uvicorn` to `requirements.txt`.
3.  **Verification**:
    -   Validated `start_all.bat` launches the merged server.
    -   Validated `/health` and `/chat` endpoints.
