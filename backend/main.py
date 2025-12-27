from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Allow CORS for frontend accessibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for now, or specify your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "Backend is running!"}

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    return {"response": f"Echo: {request.message} (Backend is working)"}

# Root endpoint for basic verification
@app.get("/")
def read_root():
    return {"message": "Welcome to Physical AI Backend"}
