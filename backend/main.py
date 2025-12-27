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
        
        # STRATEGY: Compatibility Mode.
        # Status 400 (Bad Request) means the API didn't like 'systemInstruction' on the v1 endpoint.
        # FIX: We merge the system prompt INTO the user message. This works on ALL versions (v1, v1beta).
        
        # 1. Merge System Prompt into User Message
        final_prompt = f"""{system_instruction}
        
        Context:
        {context_text}
        
        Question:
        {request.message}
        """

        # 2. Simplified Payload (No 'systemInstruction' field)
        payload = {
            "contents": [{
                "role": "user",
                "parts": [{"text": final_prompt}]
            }]
        }
        
        async with httpx.AsyncClient() as client:
             # Try 1.5 Flash on v1beta (Standard)
             # We went back to v1beta because 400 on v1 proves v1 was reached but rejected the payload.
             # With the simplified payload, v1beta is the best bet for Flash.
             url_flash = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"
             
             try:
                response = await client.post(url_flash, json=payload, timeout=30.0)
                if response.status_code == 200:
                    data = response.json()
                    if "candidates" in data and data["candidates"]:
                            answer = data["candidates"][0]["content"]["parts"][0]["text"]
                            return {"response": answer}
                
                # Fallback: Gemini Pro on v1 (The ultimate fallback)
                print(f"LOG: Flash failed ({response.status_code}). Switching to Gemini Pro (v1)...")
                url_pro = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={API_KEY}"
                response_pro = await client.post(url_pro, json=payload, timeout=30.0)
                
                if response_pro.status_code == 200:
                    data = response_pro.json()
                    if "candidates" in data and data["candidates"]:
                        answer = data["candidates"][0]["content"]["parts"][0]["text"]
                        return {"response": answer}

                return {"response": f"Error: Flash and Pro failed. Flash Status: {response.status_code}. Pro Status: {response_pro.status_code}. Key: {API_KEY[:4]}..."}
             except Exception as e:
                return {"response": f"Connection Error: {e}"}

    except Exception as e:
        print(f"Server Error: {e}")
        return {"response": "Sorry, I encountered an internal error."}

    except Exception as e:
        print(f"Server Error: {e}")
        return {"response": "Sorry, I encountered an internal error."}

@app.get("/")
def read_root():
    return {"message": "Physical AI Textbook Backend (Lightweight Only)"}
