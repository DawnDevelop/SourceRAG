import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "indexer"))
sys.path.insert(0, str(ROOT / "api"))

# Both modules read these at import time (and fail hard if missing).
os.environ.setdefault("LLM_BASE_URL", "http://localhost:9999/v1")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_MODEL", "test-model")
# Port 1 is never a real Postgres -- any test that accidentally reaches the
# database fails fast with a refused connection instead of hanging. Database
# access itself is lazy in both modules, so importing needs no server.
os.environ.setdefault("DATABASE_URL", "postgresql://x:x@127.0.0.1:1/none?connect_timeout=1")

import chunk_and_embed  # noqa: E402,F401
import main  # noqa: E402,F401
