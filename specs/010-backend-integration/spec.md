# Spec 010: Backend Integration & Unification

## 1. Background
User reported error: "Attribute 'app' not found in module 'rag_system'". The backend was fragmented between `chatbot/` and `backend/`.

## 2. Goal
Unify the backend into a single, cohesive module that works with the existing `start_all.bat` script.

## 3. Requirements
*   **FR-001**: `rag_system.py` MUST import `FastAPI` and define `app = FastAPI(...)`.
*   **FR-002**: `rag_system.py` MUST include the `if __name__ == "__main__":` block to support both CLI args and Uvicorn execution.
*   **FR-003**: All references to `test_setup.py` or helper scripts must be compatible with the `backend/` directory structure.

## 4. Success Criteria
*   `start_all.bat` successfully launches the backend.
*   `python rag_system.py --ingest` triggers ingestion.
