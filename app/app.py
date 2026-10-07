import sys
from pathlib import Path

# Pathlib resolution for project root and app directory
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

# Ensure PROJECT_ROOT is at the front of sys.path so 'app' package is found
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# Append APP_DIR to the end of sys.path for direct module lookups
if str(APP_DIR) in sys.path:
    sys.path.remove(str(APP_DIR))
sys.path.append(str(APP_DIR))

import app.streamlit_app as main_app
