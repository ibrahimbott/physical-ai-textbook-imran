import os
import sys

# Mock Env Vars to prevent startup crash
os.environ["QDRANT_URL"] = "http://mock"
os.environ["QDRANT_API_KEY"] = "mock"
os.environ["COHERE_API_KEY"] = "mock"
os.environ["GOOGLE_API_KEY"] = "mock"

print("Attempting to import api.index...")
try:
    from api.index import app
    print("SUCCESS: Imported app from api.index")
except Exception as e:
    print(f"FAILURE: {e}")
    import traceback
    traceback.print_exc()
