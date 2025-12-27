import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from .rag import search_textbook

load_dotenv()

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
API_KEY = os.getenv("AI_API_KEY") # User to set this in Vercel
if API_KEY:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash') # Lightweight model
else:
    model = None

class ChatRequest(BaseModel):
    message: str

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "backend": "FastAPI",
        "ai_status": "Ready" if model else "Missing API Key"
    }

@app.post("/api/chat")
@app.post("/chat") # Handle cases where Vercel strips /api prefix
async def chat_endpoint(request: ChatRequest):
    if not model:
        # Fallback if key missing
        return {"response": "Error: AI_API_KEY is missing in Vercel Environment Variables."}

    try:
        # 1. Search Textbook
        context_chunks = search_textbook(request.message, top_k=3)
        context_text = "\n\n".join([c["content"] for c in context_chunks])
        
        # 2. Construct Prompt
        prompt = f"""
        You are an AI Tutor for the "Physical AI & Humanoid Robotics" textbook.
        Answer the student's question based ONLY on the following textbook context.
        If the answer is not in the context, say "I don't have that information in the notes."
        
        CONTEXT:
        {context_text}
        
        STUDENT QUESTION:
        {request.message}
        """
        
        # 3. Generate Answer
        response = model.generate_content(prompt)
        return {"response": response.text}
        
    except Exception as e:
        return {"response": f"AI Error: {str(e)}"}

@app.get("/")
def read_root():
    return {"message": "Physical AI Textbook Backend Active"}
