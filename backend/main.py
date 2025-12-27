import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from qdrant_client import QdrantClient

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
API_KEY = os.getenv("AI_API_KEY") or os.getenv("GOOGLE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "physical_ai_textbook") # Default if not set

# Initialize Qdrant Client (Lazy loading to allow startup even if keys missing initially)
qdrant_client = None
if QDRANT_URL and QDRANT_API_KEY:
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        print("LOG: Qdrant Client Initialized.")
    except Exception as e:
        print(f"LOG: Qdrant Init Failed: {e}")

class ChatRequest(BaseModel):
    message: str

async def get_embedding(text: str):
    """Generates embedding using Gemini Text Embedding 004 via HTTPX"""
    if not API_KEY:
        return None
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={API_KEY}"
    payload = {
        "model": "models/text-embedding-004",
        "content": {"parts": [{"text": text}]},
        "outputDimensionality": 1024  # Request 1024 (API may ignore if >768)
    }
    
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(url, json=payload, timeout=10.0)
            if resp.status_code == 200:
                data = resp.json()
                if "embedding" in data:
                    vector = data["embedding"]["values"]
                    
                    # FORCE COMPATIBILITY: Pad to 1024 if API returns 768 (Common mismatch)
                    target_dim = 1024
                    current_dim = len(vector)
                    
                    if current_dim < target_dim:
                        print(f"LOG: Padding vector from {current_dim} to {target_dim} for Qdrant.")
                        vector += [0.0] * (target_dim - current_dim)
                    elif current_dim > target_dim:
                        print(f"LOG: Truncating vector from {current_dim} to {target_dim}.")
                        vector = vector[:target_dim]
                        
                    return vector
            
            print(f"LOG: Embedding Failed {resp.status_code}: {resp.text}")
            return None
        except Exception as e:
            print(f"LOG: Embedding Error: {e}")
            return None

def search_qdrant(query_embedding, top_k=3):
    """Searches Qdrant for similar chunks using correct method"""
    if not qdrant_client or not query_embedding:
        return []
    
    try:
        # 1. Verify Collection Exists
        collections = qdrant_client.get_collections()
        exists = any(c.name == COLLECTION_NAME for c in collections.collections)
        if not exists:
            print(f"LOG: Collection '{COLLECTION_NAME}' not found in Qdrant.")
            return []

        # 2. Execute Search using query_points (Modern Method)
        try:
            results = qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_embedding,
                limit=top_k
            ).points
        except AttributeError:
             print("LOG: 'query_points' method missing. Trying .search()...")
             # Fallback for older clients
             try:
                results = qdrant_client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_embedding,
                    limit=top_k
                )
             except AttributeError:
                return ["Search Error: Client version mismatch. Neither 'query_points' nor 'search' found."]

        # Extract textual content from payload
        extracted_chunks = []
        for res in results:
            if hasattr(res, 'payload') and res.payload:
                content = res.payload.get("page_content", "") or res.payload.get("text", "")
                if content:
                    extracted_chunks.append(content)
        
        return extracted_chunks

    except Exception as e:
        print(f"LOG: Search Error: {e}")
        return [f"Search Error: {str(e)}"]

@app.get("/api/health")
def health_check():
    q_status = False
    try:
        if qdrant_client:
            q_status = True
    except:
        pass
        
    return {
        "status": "ok",
        "backend": "FastAPI + Qdrant Fixed",
        "qdrant_connected": q_status,
        "key_present": bool(API_KEY)
    }

@app.post("/api/chat")
@app.post("/chat")
@app.post("/") 
async def chat_endpoint(request: ChatRequest):
    if not API_KEY:
        return {"response": "Error: API Key missing. Please set 'AI_API_KEY' in Vercel."}
    
    # 1. Generate Embedding & Search Qdrant
    context_text = ""
    query_embedding = await get_embedding(request.message)
    
    if query_embedding:
        chunks = search_qdrant(query_embedding)
        # Check if chunks contain error message
        if chunks and chunks[0].startswith("Search Error:"):
             return {"response": chunks[0]}
             
        if chunks:
            context_text = "\n\n".join(chunks)
            print(f"LOG: Retrieved {len(chunks)} chunks from Qdrant.")
        else:
            print("LOG: No chunks found in Qdrant (or search failed).")
    else:
        print("LOG: Embedding generation failed. Context will be empty.")

    # 1.5. Log Collection Name (Debug)
    print(f"LOG: Querying Collection: {COLLECTION_NAME}")

    # 2. Smart Context System Prompt (Relaxed for Greetings)
    system_instruction = (
        "You are an AI Tutor for the 'Physical AI & Humanoid Robotics' textbook. "
        "1. **Greetings**: If the user says 'Hi', 'Hello', or 'Thanks', reply politely and ask how you can help with the textbook. "
        "2. **Strict Context**: For technical questions, answer ONLY based on the provided Context below. "
        "3. **Missing Info**: If the answer is NOT in the context, say 'I couldn't find that specific topic in the textbook. Could you rephrase or ask about [Topic]?' "
        "4. **Tone**: Be helpful, professional, and encouraging."
    )
    
    final_prompt = f"""{system_instruction}
    
    Context from Textbook:
    {context_text}
    
    User Question:
    {request.message}
    """

    # 3. Dynamic Model Discovery (Vercel Fix)
    async with httpx.AsyncClient() as client:
        available_models = []
        try:
            # Ask Google what's allowed
            discovery_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"
            resp = await client.get(discovery_url, timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                if "models" in data:
                    available_models = [
                        m["name"].replace("models/", "") 
                        for m in data["models"] 
                        if "generateContent" in m.get("supportedGenerationMethods", [])
                    ]
                    # Sort: Preferred User Model -> 2.0 -> 1.5 -> Flash
                    # Note: "gemini-robotics-er-1.5-preview" is hypothetical but we prioritize it if found
                    available_models.sort(key=lambda x: (
                        "robotics" not in x, # Highest priority
                        "2.0" not in x, 
                        "1.5" not in x, 
                        "flash" not in x
                    ))
        except Exception as e:
            print(f"LOG: Discovery Error: {e}")

        # Fallback List
        # Added user's specific working model to the list
        if not available_models:
             available_models = [
                 "gemini-robotics-er-1.5-preview", # User said this works 200 OK
                 "gemini-2.0-flash-exp", 
                 "gemini-1.5-flash", 
                 "gemini-1.5-flash-8b", 
                 "gemini-pro"
             ]
        else:
            # Force inject user's preferred model at top if discovery didn't find it (but it might work)
             if "gemini-robotics-er-1.5-preview" not in available_models:
                  available_models.insert(0, "gemini-robotics-er-1.5-preview")

        # Try models in order
        payload = {
            "contents": [{"role": "user", "parts": [{"text": final_prompt}]}]
        }
        
        logs = []
        for model in available_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}"
            
            # Simple retry logic
            for attempt in range(2): 
                try:
                    response = await client.post(url, json=payload, timeout=30.0)
                    if response.status_code == 200:
                        data = response.json()
                        if "candidates" in data and data["candidates"]:
                             return {"response": data["candidates"][0]["content"]["parts"][0]["text"]}
                    
                    if response.status_code == 429:
                        import asyncio
                        await asyncio.sleep(1.5)
                        continue # Retry same model
                    
                    if response.status_code == 404:
                         break # Skip to next model
                    
                except Exception as e:
                    logs.append(f"{model}: Error {e}")
                    break
        
        return {"response": "Error: critical system failure. Unable to connect to any AI models or Database. Please check Vercel Logs."}

@app.get("/")
def read_root():
    return {"message": "Physical AI Textbook Backend (Qdrant Enabled)"}
