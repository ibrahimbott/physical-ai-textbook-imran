@echo off
echo Starting Backend Server...
cd backend
python -m pip install uvicorn fastapi
python -m uvicorn rag_system:app --reload --host 0.0.0.0 --port 8000
pause
