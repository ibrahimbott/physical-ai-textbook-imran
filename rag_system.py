"""
FastAPI Backend with Agentic RAG for Physical AI Textbook
Merged with CLI Ingestion tools.
"""

import json
import logging
import os
import glob
import sys
import argparse
import time
import uuid
import traceback
import asyncio
from typing import List, Optional, Tuple, Dict

import cohere
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment
load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "physical_ai_textbook")
DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend-book", "docs"))
TOP_K = int(os.getenv("TOP_K_RESULTS", "5"))
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-flash-latest")

_available_models_cache: List[str] = []
ACTIVE_MODEL = MODEL_NAME
MODEL_NOTICE: Optional[str] = None

# Initialize clients
try:
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    cohere_client = cohere.Client(COHERE_API_KEY)
    openai_client = AsyncOpenAI(
        api_key=GOOGLE_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
except Exception as e:
    logging.error(f"Failed to initialize clients: {e}")

async def refresh_model_selection(force: bool = False) -> Tuple[str, Optional[str]]:
    global _available_models_cache, ACTIVE_MODEL, MODEL_NOTICE

    if _available_models_cache and not force:
        return ACTIVE_MODEL, MODEL_NOTICE

    try:
        models_response = await openai_client.models.list()
        _available_models_cache = [model.id for model in models_response.data]
    except Exception as error:
        logging.error("Failed to list Gemini models: %s", error, exc_info=True)
        MODEL_NOTICE = (
            "⚠️ Unable to verify available Gemini models automatically."
            " Using configured model and relying on manual configuration."
        )
        ACTIVE_MODEL = MODEL_NAME
        return ACTIVE_MODEL, MODEL_NOTICE

    if MODEL_NAME in _available_models_cache:
        ACTIVE_MODEL = MODEL_NAME
        MODEL_NOTICE = None
    else:
        preferred_flash_models = [
            "gemini-2.0-flash",
            "gemini-flash-latest",
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest",
        ]
        fallback = next((m for m in preferred_flash_models if m in _available_models_cache), None)
        if not fallback:
            # Try to find any model with "flash" in it
            fallback = next((m for m in _available_models_cache if "flash" in m), None)
        if not fallback and _available_models_cache:
            fallback = _available_models_cache[0]

        if fallback:
            logging.warning(
                "Requested Gemini model '%s' not available. Falling back to '%s'.",
                MODEL_NAME,
                fallback,
            )
            ACTIVE_MODEL = fallback
            MODEL_NOTICE = (
                f"Requested Gemini model '{MODEL_NAME}' is unavailable; using '{fallback}' instead."
            )
        else:
            ACTIVE_MODEL = MODEL_NAME
            MODEL_NOTICE = (
                "No Gemini models are accessible with the current API key."
            )

    return ACTIVE_MODEL, MODEL_NOTICE

# Ensure models are loaded on startup
# We can't await this at module level, so we just let the first request handle it or create a task
# asyncio.create_task(refresh_model_selection()) # Requires loop

# --- FastAPI app ---
app = FastAPI(title="Physical AI Textbook Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files if they exist
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/api")
async def read_root():
    if os.path.exists(os.path.join(static_dir, "index.html")):
        return FileResponse(os.path.join(static_dir, "index.html"))
    return {"message": "Physical AI Agent Backend Running"}

# --- Models ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    response: str
    sources: List[dict]

class IngestResponse(BaseModel):
    status: str
    message: str

# --- RAG Logic (from chatbot/api.py) ---

def retrieve_knowledge(query: str, top_k: int = TOP_K) -> dict:
    """
    Retrieves relevant context from the Physical AI textbook using semantic search.
    """
    try:
        # Embed the query using Cohere
        response = cohere_client.embed(
            texts=[query],
            model='embed-multilingual-v3.0',
            input_type='search_query'
        )
        
        query_vector = response.embeddings[0]
        
        # Search Qdrant logic (Robust)
        search_response = None
        
        # Try query_points first (newer client)
        try:
             search_response = qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=top_k,
                with_payload=True
             )
        except Exception:
             # Fallback to search
            try:
                search_response = qdrant_client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_vector,
                    limit=top_k,
                    with_payload=True
                )
            except Exception as e:
                 logging.error(f"All Qdrant search methods failed: {e}")
                 raise

        # Normalize results
        search_results = getattr(search_response, "points", None)
        if search_results is None:
             search_results = getattr(search_response, "result", search_response)
        if not isinstance(search_results, list):
             search_results = list(search_results)
        
        # Format results
        contexts = []
        sources = []

        for result in search_results:
            payload = getattr(result, "payload", None)
            if payload is None and isinstance(result, dict):
                payload = result.get("payload") or {}
            payload = payload or {}
            
            text_chunk = (payload.get('text') or '').strip()
            if not text_chunk:
                continue

            contexts.append(text_chunk)
            score = getattr(result, "score", None)
            if score is None and isinstance(result, dict):
                score = result.get("score")
                
            sources.append({
                'title': payload.get('source', 'Unknown section'),
                'module': payload.get('source', 'Unknown module'), 
                'file_path': payload.get('source', ''),
                'score': score
            })
        
        combined_context = "\\n\\n---\\n\\n".join(contexts)
        
        return {
            'context': combined_context,
            'sources': sources,
            'success': True
        }
    
    except Exception as e:
        logging.error("Error in retrieve_knowledge: %s", e, exc_info=True)
        return {
            'context': '',
            'sources': [],
            'success': False,
            'error': str(e)
        }

RETRIEVE_KNOWLEDGE_FUNCTION = {
    "name": "retrieve_knowledge",
    "description": "Retrieves relevant information from the Physical AI & Humanoid Robotics textbook based on the user's question. Use this tool EVERY time before answering to ensure accuracy.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The user's question or search query"
            },
            "top_k": {
                "type": "integer",
                "description": "Number of relevant text chunks to retrieve (default: 5)",
                "default": 5
            }
        },
        "required": ["query"]
    }
}

SYSTEM_PROMPT = """You are the AI Tutor for the **Physical AI & Humanoid Robotics** course by Muhammed Ibrahim.

**CRITICAL RULES:**
1. You MUST use the `retrieve_knowledge` function BEFORE answering ANY question about the course content.
2. You MUST answer questions using ONLY the information retrieved from the textbook.
3. If the retrieved context does not contain the answer, state: "I don't have that information in the textbook. Please check the relevant module or ask a more specific question."
4. NEVER hallucinate or use external knowledge beyond the textbook.
5. Be concise but thorough. Use bullet points and code examples when relevant.
"""

async def run_agent(user_message: str, conversation_history: List[dict]) -> dict:
    """
    Runs the OpenAI agent with function calling capabilities (ASYNC)
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation_history[-5:])
    messages.append({"role": "user", "content": user_message})
    
    model_in_use, model_notice = await refresh_model_selection()

    try:
        response = await openai_client.chat.completions.create(
            model=model_in_use,
            messages=messages,
            tools=[{"type": "function", "function": RETRIEVE_KNOWLEDGE_FUNCTION}],
            tool_choice="auto",
            temperature=0.3,
            max_tokens=1500
        )
    except Exception as e:
        logging.error("Gemini initial completion failed: %s", e)
        return {'response': f"Thinking failed: {e}", 'sources': []}
    
    assistant_message = response.choices[0].message
    
    if assistant_message.tool_calls:
        tool_call = assistant_message.tool_calls[0]
        function_name = tool_call.function.name
        
        try:
             function_args = json.loads(tool_call.function.arguments or '{}')
        except:
             function_args = {}
             
        if function_name == "retrieve_knowledge":
            # Run the synchronous retrieval in a thread to keep loop unblocked
            function_result = await asyncio.to_thread(
                retrieve_knowledge,
                query=function_args.get('query', user_message),
                top_k=function_args.get('top_k', TOP_K)
            )
            
            messages.append(assistant_message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": function_result.get('context', "No context found.")
            })
            
            try:
                final_response = await openai_client.chat.completions.create(
                    model=model_in_use,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=1200
                )
                final_text = final_response.choices[0].message.content
                if model_notice:
                    final_text += f"\\n\\nℹ️ {model_notice}"
                return {'response': final_text, 'sources': function_result['sources']}
            except Exception as e:
                 return {'response': f"Follow-up failed: {e}", 'sources': []}

    return {'response': assistant_message.content or "No response.", 'sources': []}

# --- Ingestion Logic (from original rag_system.py) ---

def perform_ingestion() -> str:
    print(f"Starting ingestion from: {DOCS_DIR}")
    
    files = glob.glob(os.path.join(DOCS_DIR, "**/*.md"), recursive=True) +             glob.glob(os.path.join(DOCS_DIR, "**/*.mdx"), recursive=True)
            
    if not files:
        return "No markdown files found."

    documents = []
    metadatas = []

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():
                    documents.append(content)
                    rel_path = os.path.relpath(file_path, DOCS_DIR)
                    metadatas.append({"source": rel_path})
        except Exception as e:
            print(f"Skipping file {file_path}: {e}")

    if not documents:
        return "No content to ingest."

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    all_chunks = []
    all_metas = []
    
    print("Splitting documents...")
    for doc, meta in zip(documents, metadatas):
        chunks = text_splitter.split_text(doc)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_meta = meta.copy()
            chunk_meta["text"] = chunk
            chunk_meta["chunk_id"] = i
            all_metas.append(chunk_meta)
            
    try:
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if COLLECTION_NAME not in collection_names:
             qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=1024,
                    distance=models.Distance.COSINE
                )
            )
    except Exception as e:
        return f"Error managing Qdrant collection: {e}"

    batch_size = 90
    print("Embedding and upserting...")
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i : i + batch_size]
        batch_metas = all_metas[i : i + batch_size]
        
        try:
            response = cohere_client.embed(
                texts=batch_chunks,
                model="embed-multilingual-v3.0",
                input_type="search_document"
            )
            embeddings = response.embeddings
            
            points = []
            for j, (embedding, meta) in enumerate(zip(embeddings, batch_metas)):
                unique_str = f"{meta['source']}_{meta['chunk_id']}"
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))
                
                points.append(models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=meta
                ))
            
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            time.sleep(0.5)
            
        except Exception as e:
            return f"Error processing batch starting at index {i}: {e}"
    
    return f"Ingestion complete. Processed {len(all_chunks)} chunks from {len(files)} files."

# --- API Endpoints ---

@app.post("/api/ingest", response_model=IngestResponse)
async def api_ingest():
    if not qdrant_client:
         raise HTTPException(status_code=500, detail="Database client not initialized")
    
    try:
        result = await asyncio.to_thread(perform_ingestion)
        return IngestResponse(status="success", message=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(request: ChatRequest):
    try:
        history = [msg.model_dump() for msg in request.conversation_history]
        result = await run_agent(request.message, history)
        return ChatResponse(response=result['response'], sources=result['sources'])
    except Exception as e:
        logging.error("Chat endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

# --- CLI Entry Point ---

def main():
    parser = argparse.ArgumentParser(description="RAG System for Physical AI Textbook")
    parser.add_argument("--ingest", action="store_true", help="Ingest documents into Qdrant")
    parser.add_argument("--ask", type=str, help="Ask a question to the RAG system")
    
    args = parser.parse_args()
    
    if args.ingest:
        print(perform_ingestion())
    elif args.ask:
        print(f"Question: {args.ask}")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(run_agent(args.ask, []))
        print(f"Answer: {result['response']}")
    else:
        print("Starting API Server on port 8000...")
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
