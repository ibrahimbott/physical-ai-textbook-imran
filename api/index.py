import sys
import os

# Add the parent directory/backend to sys.path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.rag_system import app

# Vercel needs the object named 'app' available at the module level

