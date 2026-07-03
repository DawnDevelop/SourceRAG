import os
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "indexer"))
sys.path.insert(0, str(ROOT / "api"))

# Both modules read these at import time (and fail hard if missing).
os.environ.setdefault("LMSTUDIO_BASE_URL", "http://localhost:9999/v1")
os.environ.setdefault("EMBEDDING_MODEL", "test-model")

# chromadb.HttpClient connects to the server on construction, and both
# modules construct one at import time -- stub it out so the modules can be
# imported without a running Chroma. The stubbed client lands in
# module.chroma; tests that need it configure the mock per-test.
with mock.patch("chromadb.HttpClient"):
    import chunk_and_embed  # noqa: F401
    import main  # noqa: F401
