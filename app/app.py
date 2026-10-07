import os
import sys
from pathlib import Path

# Pathlib resolution for project root and app directory
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import and run the main streamlit application
import app.streamlit_app as main_app

if __name__ == "__main__":
    pass
