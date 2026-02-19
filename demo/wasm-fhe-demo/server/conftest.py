import sys
from pathlib import Path

import pytest  # noqa: F401

# Ensure `from app import app` works when pytest runs from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest_plugins = ["anyio"]
