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
        
        # STRATEGY: "Dynamic Discovery & Robust Fallback".
        # 1. Ask Google: "What models are permitted for this Key?"
        # 2. Try the best available ones.
        # 3. If discovery fails, fallback to hardcoded list including 1.5 Flash 8b (Lite).
        
        async with httpx.AsyncClient() as client:
            model_to_use = None
            available_models = []
            
            try:
                # 1. Fetch Available Models
                print("LOG: Discovering available models...")
                discovery_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"
                resp = await client.get(discovery_url, timeout=10.0)
                
                if resp.status_code == 200:
                    data = resp.json()
                    if "models" in data:
                        # Filter for models that support 'generateContent'
                        available_models = [
                            m["name"].replace("models/", "") 
                            for m in data["models"] 
                            if "generateContent" in m.get("supportedGenerationMethods", [])
                        ]
                        print(f"LOG: Discovery Success! Available Models: {available_models}")
                        
                        # Priority Sort: 2.0 -> 1.5 -> Flash
                        available_models.sort(key=lambda x: (
                            "2.0" not in x, # 2.0 first
                            "1.5" not in x, # 1.5 second
                            "flash" not in x # Flash preferred
                        ))
                else:
                    print(f"LOG: Discovery failed ({resp.status_code}). Using fallback list.")
            except Exception as e:
                print(f"LOG: Discovery Exception: {e}")

            # Fallback list if discovery fails or returns empty
            # crucial: Added gemini-1.5-flash-8b as it is often unrestricted
            if not available_models:
                available_models = [
                    "gemini-2.0-flash-exp", 
                    "gemini-1.5-flash", 
                    "gemini-1.5-flash-8b", 
                    "gemini-pro"
                ]

            # 2. Try models in order
            import asyncio
            
            # Safe Payload
            final_prompt = f"""{system_instruction}
            Context:
            {context_text}
            Question:
            {request.message}
            """
            
            payload = {
                "contents": [{"role": "user", "parts": [{"text": final_prompt}]}]
            }

            logs = []
            for model in available_models:
                print(f"LOG: Trying model: {model}")
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}"
                
                for attempt in range(2): # 2 attempts per model
                    try:
                        response = await client.post(url, json=payload, timeout=30.0)
                        
                        if response.status_code == 200:
                            data = response.json()
                            if "candidates" in data and data["candidates"]:
                                    answer = data["candidates"][0]["content"]["parts"][0]["text"]
                                    return {"response": answer}
                        
                        if response.status_code == 429:
                            if attempt == 0:
                                print(f"LOG: {model} 429 (Busy). Waiting 2s...")
                                await asyncio.sleep(2)
                                continue
                            else:
                                logs.append(f"{model}: 429")
                        elif response.status_code == 404:
                            logs.append(f"{model}: 404")
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

@app.get("/")
def read_root():
    return {"message": "Physical AI Textbook Backend (Lightweight Only)"}
