import os
import glob
import httpx
import uuid
import time
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

# Load Environment Variables
load_dotenv()

# Configuration
API_KEY = os.getenv("AI_API_KEY") or os.getenv("GOOGLE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "physical_ai_textbook")
DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend-book", "docs")

print(f"LOG: Config Loaded. Collection: {COLLECTION_NAME}")
print(f"LOG: Docs Dir: {DOCS_DIR}")

if not API_KEY or not QDRANT_URL or not QDRANT_API_KEY:
    print("ERROR: Missing Environment Variables. Check .env file.")
    exit(1)

# Initialize Qdrant
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def recreate_collection():
    """Deletes and recreates the collection with 768 dimensions"""
    print(f"LOG: Recreating Collection '{COLLECTION_NAME}' with size 768...")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    print("LOG: Collection Reset Complete.")

def get_embedding(text):
    """Get 768-dim embedding from Gemini"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={API_KEY}"
    payload = {
        "model": "models/text-embedding-004",
        "content": {"parts": [{"text": text[:9000]}]} # Limit text size just in case
    }
    
    # Simple retry loop
    for attempt in range(3):
        try:
            resp = httpx.post(url, json=payload, timeout=20.0)
            if resp.status_code == 200:
                return resp.json()["embedding"]["values"] # Should be 768
            elif resp.status_code == 429:
                time.sleep(2)
                continue
            else:
                print(f"ERROR: Embedding Failed {resp.status_code}: {resp.text}")
                return None
        except Exception as e:
            print(f"ERROR: Connection failed: {e}")
            time.sleep(1)
    return None

def process_files():
    """Reads MD files, chunks them, and uploads to Qdrant"""
    recreate_collection()
    
    # Find all .md and .mdx files
    files = glob.glob(os.path.join(DOCS_DIR, "**", "*.md"), recursive=True)
    files += glob.glob(os.path.join(DOCS_DIR, "**", "*.mdx"), recursive=True)
    
    print(f"LOG: Found {len(files)} files.")
    
    points = []
    
    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            
        if not content.strip():
            continue
            
        # simple chunking (by double newline)
        chunks = content.split("\n\n")
        
        filename = os.path.basename(filepath)
        print(f"Processing {filename} ({len(chunks)} chunks)...")
        
        for i, chunk in enumerate(chunks):
            if len(chunk) < 50: # Skip tiny chunks
                continue
                
            vector = get_embedding(chunk)
            if vector:
                # Validate Logic: Ensure 768
                if len(vector) != 768:
                    print(f"WARNING: Vector size mismatch! Got {len(vector)}.")
                
                point_id = str(uuid.uuid4())
                points.append({
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "page_content": chunk,
                        "source": filename,
                        "chunk_index": i
                    }
                })
                
                # Batch Upload (every 20 chunks)
                if len(points) >= 20:
                    try:
                        client.upsert(
                            collection_name=COLLECTION_NAME,
                            points=points
                        )
                        print(f"Uploaded 20 points for {filename}")
                        points = []
                        time.sleep(0.5) # Rate limit kindness
                    except Exception as e:
                        print(f"Upload Failed: {e}")

    # Upload remaining
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"Uploaded final {len(points)} points.")

if __name__ == "__main__":
    process_files()
