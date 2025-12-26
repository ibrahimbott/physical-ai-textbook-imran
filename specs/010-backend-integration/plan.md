# Plan: Backend Integration & Unification

## 1. Code Unification
-   [x] Merge `chatbot/` functionality into `backend/`.
-   [x] Ensure `rag_system.py` defines `app = FastAPI()`.

## 2. Script Updates
-   [x] Verify `start_all.bat` points to the correct module.
-   [x] Test `python rag_system.py --ingest` CLI args.

## 3. Static Files
-   [x] Verify `backend/static` exists and contains the frontend build.
