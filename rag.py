import os
import glob
import re

# Define path to docs
# Vercel places the whole repo in the task root, but we need to find where docs are relative to backend
# Ideally: ../frontend-book/docs
# We will use relative paths assuming the structure:
# root/
#   backend/
#   frontend-book/
#     docs/

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend-book", "docs")

def load_textbook_content():
    """
    Scans all .md and .mdx files in the docs directory.
    Returns a list of dicts: {'path': str, 'content': str}
    """
    chunks = []
    
    # Simple recursive glob
    search_path = os.path.join(DOCS_DIR, "**", "*.md*")
    files = glob.glob(search_path, recursive=True)
    
    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read()
                # Basic cleaning: remove frontmatter (between --- and ---)
                text = re.sub(r'^---\n.*?\n---\n', '', text, flags=re.DOTALL)
                chunks.append({
                    "path": os.path.basename(fpath),
                    "content": text
                })
        except Exception as e:
            print(f"Skipping {fpath}: {e}")
            
    return chunks

# Cache content in memory (serverless warm start will benefit)
TEXTBOOK_CHUNKS = load_textbook_content()

def search_textbook(query: str, top_k: int = 3):
    """
    Simple keyword-based scoring.
    """
    query_terms = query.lower().split()
    results = []
    
    for chunk in TEXTBOOK_CHUNKS:
        score = 0
        content_lower = chunk["content"].lower()
        
        # Scoring: +1 per term occurrence
        for term in query_terms:
            score += content_lower.count(term)
        
        if score > 0:
            results.append({
                "path": chunk["path"],
                "content": chunk["content"][:2000], # Trucate for context window
                "score": score
            })
            
    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
