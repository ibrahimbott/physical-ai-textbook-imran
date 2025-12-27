import os
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# 1. Environment aur App Setup
load_dotenv()
app = FastAPI()

# Frontend ke liye Rasta (CORS) kholna
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Configuration (Environment Variables)
API_KEY = os.getenv("AI_API_KEY") or os.getenv("GOOGLE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "physical_ai_textbook")

# Qdrant Client Initialize karna
qdrant_client = None
if QDRANT_URL and QDRANT_API_KEY:
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        print("LOG: Qdrant Database Connected.")
    except Exception as e:
        print(f"LOG: Qdrant Connection Failed: {e}")

class ChatRequest(BaseModel):
    message: str

# 3. Embedding Function (Google API + Dimension Fix)
async def get_embedding(text: str):
    """Google se vector mangwana aur dimension 1024 fix karna"""
    if not API_KEY: return None
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={API_KEY}"
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
                vector = resp.json()["embedding"]["values"]
                
                # --- CRITICAL DIMENSION FIX ---
                # Agar Google 1024 ke bajaye 768 de, toh zeros laga kar 1024 pura karna
                current_dim = len(vector)
                if current_dim < 1024:
                    print(f"LOG: Padding vector from {current_dim} to 1024 for Qdrant compatibility.")
                    vector += [0.0] * (1024 - current_dim)
                
                print(f"LOG: Vector size is now {len(vector)}.")
                return vector
            else:
                print(f"LOG: Google API Error: {resp.text}")
        except Exception as e:
            print(f"LOG: Embedding Exception: {e}")
    return None

# 4. Search Function (Database Retrieval)
def search_qdrant(query_embedding):
    """Database mein se textbook ka data nikalna"""
    if not qdrant_client or not query_embedding: return []
    try:
        # Modern query_points method
        response = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=3
        )
        
        extracted_chunks = []
        for point in response.points:
            if point.payload:
                # Text dhoondna (page_content ya text key mein)
                text = point.payload.get("page_content") or point.payload.get("text") or point.payload.get("content")
                if text: extracted_chunks.append(text)
        
        return extracted_chunks
    except Exception as e:
        print(f"LOG: Search Error: {e}")
        return []

# 5. Main Chat Endpoint
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not API_KEY: return {"response": "Error: API Key missing."}
    
    user_input = request.message.strip()

    # Step A: Search Textbook Database
    context = ""
    vector = await get_embedding(user_input)
    if vector:
        chunks = search_qdrant(vector)
        if chunks:
            context = "\n\n".join(chunks)
            print(f"LOG: Found {len(chunks)} relevant sections in textbook.")

    # Step B: AI Prompt Setup
    system_instruction = (
        "You are the 'Physical AI & Humanoid Robotics' Textbook AI Tutor. "
        "1. For greetings, be short and polite. "
        "2. For technical questions, answer ONLY using the Textbook Context provided below. "
        "3. If info is missing, say you don't know based on the textbook. "
        "4. Be clear and helpful."
    )
    
    final_prompt = f"{system_instruction}\n\nTextbook Context:\n{context}\n\nUser Question: {user_input}"

    # Step C: Call Working Model (Gemini 1.5 Robotics Preview)
    # Ye model aapke logs mein 200 OK de raha tha
    model_name = "gemini-robotics-er-1.5-preview"
    llm_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={API_KEY}"
    
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(llm_url, json={"contents": [{"parts": [{"text": final_prompt}]}]}, timeout=30.0)
            if resp.status_code == 200:
                answer = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                return {"response": answer}
            else:
                return {"response": "I'm connected, but the textbook data search is still being refined. Please ask again."}
        except Exception as e:
            return {"response": f"Connection Error: {str(e)}"}

@app.get("/")
def home():
    return {"status": "Physical AI Backend is LIVE", "collection": COLLECTION_NAME}