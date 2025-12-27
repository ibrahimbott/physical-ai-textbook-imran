import os
import httpx
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

app = FastAPI()

# Allow CORS for Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
API_KEY = os.getenv("AI_API_KEY") or os.getenv("GOOGLE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "physical_ai_textbook")

# Initialize Qdrant Client
qdrant_client = None
if QDRANT_URL and QDRANT_API_KEY:
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        print("LOG: Qdrant Client Initialized Successfully.")
    except Exception as e:
        print(f"LOG: Qdrant Init Failed: {e}")

class ChatRequest(BaseModel):
    message: str

async def get_embedding(text: str):
    """Generates NATIVE 1024-dimension embedding using Gemini 004"""
    if not API_KEY:
        return None
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={API_KEY}"
    
    # CRITICAL FIX: Explicitly request 1024 dimensionality from Google API
    payload = {
        "model": "models/text-embedding-004",
        "content": {"parts": [{"text": text}]},
        "taskType": "RETRIEVAL_QUERY",
        "outputDimensionality": 1024 
    }
    
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(url, json=payload, timeout=15.0)
            if resp.status_code == 200:
                data = resp.json()
                vector = data["embedding"]["values"]
                print(f"LOG: Successfully generated NATIVE {len(vector)} dim embedding.")
                return vector
            else:
                print(f"LOG: Embedding API Error {resp.status_code}: {resp.text}")
                return None
        except Exception as e:
            print(f"LOG: Embedding Exception: {e}")
            return None

def search_qdrant(query_embedding, top_k=3):
    """Searches Qdrant with the 1024-dim vector"""
    if not qdrant_client or not query_embedding:
        return []
    
    try:
        # Verify dimension match
        if len(query_embedding) != 1024:
            print(f"LOG WARNING: Vector dimension is {len(query_embedding)}, but Qdrant expects 1024.")

        # Using query_points for modern Qdrant Client
        response = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=top_k
        )
        
        extracted_chunks = []
        for point in response.points:
            payload = point.payload
            if payload:
                # Try common content keys
                content = payload.get("page_content") or payload.get("text") or payload.get("content")
                if content:
                    extracted_chunks.append(content)
        
        return extracted_chunks

    except Exception as e:
        print(f"LOG: Qdrant Search Error: {e}")
        return []

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not API_KEY:
        return {"response": "System Error: API Key missing in Vercel."}
    
    user_msg = request.message.strip()

    # 1. Search Database
    context_text = ""
    query_vector = await get_embedding(user_msg)
    
    if query_vector:
        chunks = search_qdrant(query_vector)
        if chunks:
            context_text = "\n\n".join(chunks)
            print(f"LOG: Context found ({len(chunks)} chunks).")
        else:
            print("LOG: No matching textbook data found.")

    # 2. Strict Instructions
    system_prompt = (
        "You are a professional AI Tutor for the 'Physical AI & Humanoid Robotics' textbook. "
        "Rules:\n"
        "1. If the user greets you (Hi, Hello, Salam), reply with a very short polite greeting and ask about the textbook.\n"
        "2. For technical questions, use ONLY the provided Context from the textbook. Do not use outside knowledge.\n"
        "3. If the answer is not in the context, say: 'I'm sorry, this specific topic is not covered in our textbook.'\n"
        "4. If the user asks to 'explain' or 'detail', then provide a long and clear explanation. Otherwise, keep it concise."
    )
    
    final_input = f"{system_prompt}\n\nContext:\n{context_text}\n\nUser Question: {user_msg}"

    # 3. Model Discovery & Fallback
    models_to_try = ["gemini-robotics-er-1.5-preview", "gemini-1.5-flash", "gemini-2.0-flash-exp"]
    
    async with httpx.AsyncClient() as client:
        for model in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}"
            payload = {"contents": [{"parts": [{"text": final_input}]}]}
            
            try:
                resp = await client.post(url, json=payload, timeout=30.0)
                if resp.status_code == 200:
                    data = resp.json()
                    return {"response": data["candidates"][0]["content"]["parts"][0]["text"]}
                elif resp.status_code == 429:
                    print(f"LOG: {model} busy (429), trying next...")
                    continue
            except Exception as e:
                print(f"LOG: Error with {model}: {e}")
                continue

    return {"response": "I'm having trouble connecting to my brain right now. Please try again in a moment."}

@app.get("/")
def home():
    return {"status": "Physical AI Backend Active", "database": COLLECTION_NAME}