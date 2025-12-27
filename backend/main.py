import os
import httpx
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

# Configuration
# Check for AI_API_KEY (Standard) or GOOGLE_API_KEY (User Preference)
API_KEY = os.getenv("AI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"

class ChatRequest(BaseModel):
    message: str

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "backend": "FastAPI (Lightweight)",
        "key_present": bool(API_KEY)
    }

@app.post("/api/chat")
@app.post("/chat")
@app.post("/") # Catch-all for stripped paths
async def chat_endpoint(request: ChatRequest):
    if not API_KEY:
        print("LOG: API Key is MISSING.")
        return {"response": "Error: API Key missing. Please set 'AI_API_KEY' or 'GOOGLE_API_KEY' in Vercel."}
    
    # Safe logging
    masked_key = f"{API_KEY[:4]}...{API_KEY[-4:]}" if len(API_KEY) > 8 else "INVALID"
    print(f"LOG: API Key found: {masked_key}. Proceeding to Gemini.")

    try:
        # 1. Search Textbook
        context_chunks = search_textbook(request.message, top_k=3)
        context_text = "\n\n".join([c["content"] for c in context_chunks])
        
        # 2. Construct Prompt
        system_instruction = "You are an AI Tutor for the 'Physical AI & Humanoid Robotics' textbook. Answer based ONLY on the context. If context is empty, say you don't know."
        full_prompt = f"""
        Context:
        {context_text}
        
        Question:
        {request.message}
        """

        # 3. Call Gemini API via HTTPX (With Fallback)
        payload = {
            "contents": [{
                "role": "user",
                "parts": [{"text": full_prompt}]
            }],
            "systemInstruction": {
                "role": "user", 
                "parts": [{"text": system_instruction}]
            }
        }
        
        # Models to try (Mix of v1beta and v1)
        # Structure: (model_name, api_version)
        model_candidates = [
            ("gemini-1.5-flash", "v1beta"),
            ("gemini-1.5-flash-latest", "v1beta"),
            ("gemini-pro", "v1beta"),
            ("gemini-pro", "v1"), # Stable v1 endpoint
            ("gemini-1.0-pro", "v1beta")
        ]

        async with httpx.AsyncClient() as client:
            for model, version in model_candidates:
                url = f"https://generativelanguage.googleapis.com/{version}/models/{model}:generateContent?key={API_KEY}"
                try:
                    # v1 endpoint uses 'contents' just like v1beta usually, but let's be safe
                    response = await client.post(url, json=payload, timeout=30.0)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if "candidates" in data and data["candidates"]:
                             answer = data["candidates"][0]["content"]["parts"][0]["text"]
                             return {"response": answer}
                        else:
                             print(f"LOG: Model {model} returned 200 but no candidates.")
                             continue
                    
                    if response.status_code == 404:
                        print(f"LOG: Model {model} ({version}) not found (404). Trying next...")
                        continue 
                        
                    # If other error (e.g. 400, 403), print and try next just in case
                    print(f"LOG: Error with {model} ({version}): {response.status_code} - {response.text}")
                    continue

                except Exception as e:
                    print(f"LOG: Exception trying {model}: {e}")
                    continue

            return {"response": "Error: Could not connect to any Gemini model (Flash, Pro, 1.0). Please verify your API Key has access to Gemini API."}

    except Exception as e:
        print(f"Server Error: {e}")
        return {"response": "Sorry, I encountered an internal error."}

@app.get("/")
def read_root():
    return {"message": "Physical AI Textbook Backend (Lightweight Only)"}
