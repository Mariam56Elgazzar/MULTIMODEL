"""Preferred Streamlit entrypoint for local development."""
from pathlib import Path
import sys
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from multimodel_rag.app.streamlit_app import *  # noqa: F401,F403
