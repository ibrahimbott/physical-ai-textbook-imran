@echo off
echo Starting RAG Backend...
start "RAG Backend" cmd /k "cd backend && python -m uvicorn rag_system:app --host 127.0.0.1 --port 8000 --reload"

echo Waiting for Backend to initialize...
timeout /t 5

echo Starting Docusaurus Frontend...
start "Docusaurus Frontend" cmd /k "cd frontend-book && npm start"

echo Both servers are launching. Please check the new windows.
pause
