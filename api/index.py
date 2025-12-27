import sys
import os

# Create a robust path to the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import the FastAPI instance
from backend.main import app

# This 'app' object is what Vercel needs to see
