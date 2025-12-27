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
        
        # STRATEGY: "Try Everything" with Clean Payload.
        # 1. 2.0 Flash Exp (Returned 429 before -> It exists!)
        # 2. 1.5 Flash 8b (New, often less busy)
        # 3. 1.5 Flash (Standard)
        # 4. Pro (Legacy)
        
        # 1. Merge System Prompt (Safe for all versions)
        final_prompt = f"""{system_instruction}
        
        Context:
        {context_text}
        
        Question:
        {request.message}
        """

        payload = {
            "contents": [{
                "role": "user",
                "parts": [{"text": final_prompt}]
            }]
        }
        
        # List of (Model, Version) to try
        candidates = [
            ("gemini-2.0-flash-exp", "v1beta"), # 429 before = FOUND
            ("gemini-1.5-flash-8b", "v1beta"),  # New lightweight
            ("gemini-1.5-flash", "v1beta"),     # Standard
            ("gemini-pro", "v1beta")            # Legacy
        ]

        async with httpx.AsyncClient() as client:
             import asyncio
             logs = []
             
             for model, version in candidates:
                url = f"https://generativelanguage.googleapis.com/{version}/models/{model}:generateContent?key={API_KEY}"
                
                # Retry loop only for this model
                for attempt in range(2): 
                    try:
                        response = await client.post(url, json=payload, timeout=30.0)
                        
                        if response.status_code == 200:
                            data = response.json()
                            if "candidates" in data and data["candidates"]:
                                    answer = data["candidates"][0]["content"]["parts"][0]["text"]
                                    return {"response": answer}
                        
                        # If 429, wait and retry (only for this model)
                        if response.status_code == 429:
                            if attempt == 0: # Only wait once per model
                                await asyncio.sleep(1.5)
                                continue
                            else:
                                logs.append(f"{model}: 429 (Busy)")
                        
                        elif response.status_code == 404:
                            logs.append(f"{model}: 404 (Not Found)")
                            break # Don't retry 404
                        else:
                             logs.append(f"{model}: {response.status_code}")
                             break 
                             
                    except Exception as e:
                        logs.append(f"{model}: Error")
                        break
             
             return {"response": f"Error: All models failed. Debug: {', '.join(logs)}. Key: {API_KEY[:4]}..."}

    except Exception as e:
        print(f"Server Error: {e}")
        return {"response": "Sorry, I encountered an internal error."}

    except Exception as e:
        print(f"Server Error: {e}")
        return {"response": "Sorry, I encountered an internal error."}

@app.get("/")
def read_root():
    return {"message": "Physical AI Textbook Backend (Lightweight Only)"}
