import hashlib
import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import psycopg
from openai import OpenAI, RateLimitError
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

import pull_requests
from repos_config import load_config, valid_repos, wiki_repos, should_index_wikis, should_index_pull_requests

CONFIG_PATH = os.environ.get("REPOS_CONFIG", "/config/repos.yaml")
DATA_DIR = os.environ.get("REPO_DATA_DIR", "/data/repos")
# Lives at the DATA_DIR root, not inside any repo, so it's never picked up by
# iter_files(). One JSON file tracks every repo's last-indexed commit --
# simple enough to inspect/edit by hand, and it rides along in the same
# repo_data volume that already persists across container restarts.
STATE_PATH = os.environ.get("INDEX_STATE_PATH", os.path.join(DATA_DIR, ".index_state.json"))
OUTPUT_CHUNK_SIZE = int(os.environ.get("OUTPUT_CHUNK_SIZE", "1500"))
OUTPUT_CHUNK_OVERLAP = int(os.environ.get("OUTPUT_CHUNK_OVERLAP", "200"))
EMBED_BATCH = int(os.environ.get("EMBED_BATCH_SIZE", "32"))

# INDEX_CONCURRENCY: repos processed (scan + chunk + embed, per repo, in one
# worker) in parallel. Deliberately one pass per repo, not a scan-everything
# -then-embed-everything split -- that gated all embedding behind scanning
# every repo first, which added a long delay before any real (embedding)
# progress was visible.
# EMBED_CONCURRENCY: concurrent embedding HTTP calls to LM Studio -- kept low
# and separate from INDEX_CONCURRENCY since it's bounded by what the local
# model server can actually parallelize, not by host CPU cores.
INDEX_CONCURRENCY = int(os.environ.get("INDEX_CONCURRENCY", "4"))
EMBED_CONCURRENCY = int(os.environ.get("EMBED_CONCURRENCY", "2"))
EMBED_MAX_RETRIES = int(os.environ.get("EMBED_MAX_RETRIES", "6"))

LM_BASE_URL = os.environ["LLM_BASE_URL"]
LM_API_KEY = os.environ["LLM_API_KEY"]
EMBED_MODEL = os.environ["EMBEDDING_MODEL"]
# Single Postgres store: pgvector carries the semantic leg, a 'simple'-config
# tsvector GIN index carries the lexical leg of the API's hybrid search.
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://sourcerag:sourcerag@postgres:5432/sourcerag")

# Extension -> langchain Language, for language-aware splitting. Anything not
# listed here falls back to a generic recursive splitter.
EXT_LANGUAGE = {
    ".py": Language.PYTHON,
    ".cs": Language.CSHARP,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".java": Language.JAVA,
    ".go": Language.GO,
    ".rb": Language.RUBY,
    ".php": Language.PHP,
    ".rs": Language.RUST,
    ".kt": Language.KOTLIN,
    ".scala": Language.SCALA,
    ".swift": Language.SWIFT,
    ".ps1": Language.POWERSHELL,
    ".psm1": Language.POWERSHELL,
    ".md": Language.MARKDOWN,
    ".cpp": Language.CPP,
    ".c": Language.C,
    ".lua": Language.LUA,
}

# langchain's Language enum has no HCL/YAML entry, so these get a hand-tuned
# separator list instead: split on blank lines (top-level block boundaries in
# well-formatted HCL/YAML) before falling back to brace/indent/newline splits.
HCL_SEPARATORS = ["\n\n", "\n}\n", "\n}", "\n  ", "\n", " ", ""]
YAML_SEPARATORS = ["\n---\n", "\n\n", "\n- ", "\n  ", "\n", " ", ""]

SKIP_DIRS = {
    ".git", "node_modules", "bin", "obj", "dist", "build",
    "venv", ".venv", "target", "vendor", "packages", ".idea", ".vs",
    # Data-dump/fixture directories -- not source code, just noise for
    # "does this code exist" search. Test *code* dirs (tests/, __tests__/)
    # are deliberately NOT here -- that's real logic, often useful to find.
    "testdata", "test-data", "fixtures", "__snapshots__",
    # Translation/i18n resources -- high-volume natural-language JSON that
    # semantically "sticks" to unrelated queries (observed: locale files
    # dominating code searches).
    "locales", "i18n", "l10n",
}
SKIP_EXT = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip",
    ".dll", ".exe", ".so", ".pdb", ".woff", ".woff2", ".ttf", ".svg",
    # Tabular/columnar data formats -- data, not code.
    ".csv", ".tsv", ".parquet",
    # Source maps -- generated, huge, never what a code search wants.
    ".map",
}
# Generated dependency manifests: large, churn constantly, and match tons of
# package-name queries without being code anyone wrote.
SKIP_FILENAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "cargo.lock", "composer.lock", "packages.lock.json", "pipfile.lock",
    "gemfile.lock", "bun.lock",
}
# Pre-minified assets carry a marker suffix; unmarked bundles are caught by
# the is_minified() line-length heuristic instead.
MINIFIED_SUFFIXES = (".min.js", ".min.css")
MAX_FILE_BYTES = 1_000_000

# pgvector's HNSW index supports at most 2000 dimensions with vector_cosine_ops.
PGVECTOR_HNSW_MAX_DIM = 2000

client = OpenAI(base_url=LM_BASE_URL, api_key=LM_API_KEY)
embed_semaphore = threading.Semaphore(EMBED_CONCURRENCY)
print_lock = threading.Lock()
state_lock = threading.Lock()
# One shared connection guarded by a lock: DB writes are tiny next to the
# embedding calls that dominate each batch, so serializing them costs nothing
# and avoids per-thread connection management. RLock because get_conn() is
# also called from inside locked sections.
db_lock = threading.RLock()
_db_conn = None


def log(msg):
    with print_lock:
        print(msg, flush=True)


def config_fingerprint():
    # Everything that changes what a chunk looks like or which files are even
    # considered. If any of this changes between runs, previously-indexed
    # chunks aren't comparable to new ones -- stored commit hashes get
    # ignored and every repo gets a full reindex (see main()).
    payload = {
        "OUTPUT_CHUNK_SIZE": OUTPUT_CHUNK_SIZE,
        "OUTPUT_CHUNK_OVERLAP": OUTPUT_CHUNK_OVERLAP,
        "embed_model": EMBED_MODEL,
        "skip_ext": sorted(SKIP_EXT),
        "skip_dirs": sorted(SKIP_DIRS),
        "skip_filenames": sorted(SKIP_FILENAMES),
        "max_file_bytes": MAX_FILE_BYTES,
        # Covers changes to the embedding input itself (path context header),
        # BOM handling, the minified/import-chunk skip heuristics, and the
        # storage backend -- anything that alters chunk content or layout
        # without touching the knobs above. Bumped to 4 for the content_hash
        # and content_type columns (dedup report + wiki/PR indexing); to 5 for
        # the symbols column (definition lookup) which every chunk must carry.
        "pipeline_version": 5,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def load_state():
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        state = {"config_fingerprint": None, "repos": {}}
    state.setdefault("pull_requests", {})
    return state


def _write_state_to_disk(state):
    # Write-then-rename so a process kill mid-write (this run has been killed
    # by Docker Desktop crashes more than once) can't leave a half-written,
    # unparseable state file behind.
    tmp_path = STATE_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, STATE_PATH)


def update_repo_state(state, project, name, commit):
    indexed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with state_lock:
        state["repos"][f"{project}/{name}"] = {"commit": commit, "indexed_at": indexed_at}
        _write_state_to_disk(state)
    record_repo_indexed(project, name, commit, indexed_at)


def record_repo_indexed(project, name, commit, indexed_at):
    # Mirror the just-written freshness into Postgres for the API's list_repos.
    # Kept outside state_lock (it's a DB write, guarded by db_lock instead) and
    # separate from the JSON write so a DB hiccup here can't corrupt the state
    # file that incremental indexing depends on.
    #
    # Best-effort: the repo's chunks are already committed by the time we reach
    # here, so a failure must not fail the repo. Crucially it rolls back on
    # error -- the shared _db_conn is serialized but not reset by db_lock, so an
    # unrolled-back failure would leave the transaction aborted and poison every
    # subsequent write from the other concurrent index_repo workers.
    conn = get_conn()
    with db_lock:
        try:
            conn.execute(
                """
                INSERT INTO repo_index_state (project, repo, commit_hash, indexed_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (project, repo) DO UPDATE SET
                    commit_hash = EXCLUDED.commit_hash, indexed_at = EXCLUDED.indexed_at
                """,
                (project, name, commit, indexed_at),
            )
            conn.commit()
        except psycopg.Error as e:
            conn.rollback()
            log(f"[freshness] {project}/{name}: could not record index state, skipping: {e}")


def filename_is_skipped(fname):
    lower = fname.lower()
    return lower in SKIP_FILENAMES or lower.endswith(MINIFIED_SUFFIXES)


def path_is_skipped(rel_path):
    parts = Path(rel_path).parts
    if any(part in SKIP_DIRS for part in parts[:-1]):
        return True
    if filename_is_skipped(parts[-1]):
        return True
    return Path(rel_path).suffix.lower() in SKIP_EXT


def is_minified(text):
    """Catch minified/bundled files that don't carry a .min.* suffix: real
    source code averages well under 300 chars per line, minified bundles are
    a few enormous lines. Short files are exempt -- a one-line config file
    isn't a bundle."""
    if len(text) < 2000:
        return False
    return len(text) / max(text.count("\n") + 1, 1) > 300


# Import-style lines across the indexed languages: C# using directives (not
# `using var ...` statements), Python/JS/TS/Java/Go imports, C includes,
# CommonJS requires, Rust use/extern.
IMPORT_LINE_RE = re.compile(
    r"^\s*("
    r"using\s+[\w.]+\s*;"
    r"|import\s"
    r"|from\s+\S+\s+import\s"
    r"|#include\s"
    r"|require\s*\("
    r"|use\s+[\w:]+\s*;"
    r"|extern\s+crate\s"
    r")"
)


def is_import_dominated(chunk):
    """Chunks that are (almost) nothing but import statements embed as pure
    noise -- they rank highly for any query mentioning a namespace-ish word
    while containing no actual logic. The real code follows in later chunks
    of the same file, so skipping these loses nothing."""
    lines = [line for line in chunk.splitlines() if line.strip()]
    if len(lines) < 3:
        return False
    import_lines = sum(1 for line in lines if IMPORT_LINE_RE.match(line))
    return import_lines / len(lines) > 0.8


# Symbol (definition) extraction: per-extension regexes capturing the NAME
# being *defined* -- types, functions, methods -- so the API's find_definition
# tool can answer "where is X defined" precisely, instead of matching every
# chunk that merely mentions X. Deliberately regex, not a real parser: cheap
# enough to run on every chunk, good enough to locate a definition. Each
# pattern's group(1) is the defined name. False positives (the odd call
# mistaken for a definition) only cost a little precision; a real AST/
# tree-sitter chunker would be exact but is a far larger change.
#
# Possessive quantifiers (*+/++) on the "typed" function pattern below are
# deliberate: without them the nested repetition backtracks catastrophically
# on long non-matching lines. Python's re has supported them since 3.11.
_TYPED_FN = re.compile(
    r"^[ \t]*+"
    r"(?!(?:return|if|for|while|switch|catch|foreach|else|do|await|yield|throw|new|lock|using)\b)"
    r"(?:[A-Za-z_][\w<>\[\],.:?]*+[ \t*&]++)+([A-Za-z_]\w*+)[ \t]*+\(",
    re.MULTILINE,
)
_TSJS = [
    re.compile(r"\b(?:class|interface|enum|type)\s+([A-Za-z_$][\w$]*)"),
    re.compile(r"\bfunction\s+([A-Za-z_$][\w$]*)"),
    re.compile(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*"
               r"(?:async\s+)?(?:function\b|\([^\n)]*\)\s*(?::[^\n={]+)?=>|[A-Za-z_$][\w$]*\s*=>)"),
]
SYMBOL_PATTERNS = {
    ".py": [re.compile(r"^[ \t]*(?:async[ \t]+)?def[ \t]+(\w+)", re.MULTILINE),
            re.compile(r"^[ \t]*class[ \t]+(\w+)", re.MULTILINE)],
    ".cs": [re.compile(r"\b(?:class|interface|struct|enum|record)\s+(\w+)"), _TYPED_FN],
    ".ts": _TSJS, ".tsx": _TSJS, ".js": _TSJS, ".jsx": _TSJS,
    ".java": [re.compile(r"\b(?:class|interface|enum|record)\s+(\w+)"), _TYPED_FN],
    ".go": [re.compile(r"\bfunc\s+(?:\([^)\n]*\)\s*)?(\w+)"), re.compile(r"\btype\s+(\w+)")],
    ".rb": [re.compile(r"^[ \t]*def[ \t]+([\w?!=]+)", re.MULTILINE),
            re.compile(r"^[ \t]*(?:class|module)[ \t]+(\w+)", re.MULTILINE)],
    ".php": [re.compile(r"\bfunction\s+(\w+)"), re.compile(r"\b(?:class|interface|trait)\s+(\w+)")],
    ".rs": [re.compile(r"\bfn\s+(\w+)"), re.compile(r"\b(?:struct|enum|trait|union)\s+(\w+)"),
            re.compile(r"\btype\s+(\w+)")],
    ".kt": [re.compile(r"\bfun\s+(\w+)"), re.compile(r"\b(?:class|interface|object)\s+(\w+)")],
    ".scala": [re.compile(r"\bdef\s+(\w+)"), re.compile(r"\b(?:class|trait|object)\s+(\w+)")],
    ".swift": [re.compile(r"\bfunc\s+(\w+)"),
               re.compile(r"\b(?:class|struct|enum|protocol|extension)\s+(\w+)")],
    ".c": [re.compile(r"\b(?:struct|enum|union)\s+(\w+)"), _TYPED_FN],
    ".cpp": [re.compile(r"\b(?:struct|class|enum|union)\s+(\w+)"), _TYPED_FN],
    ".ps1": [re.compile(r"\bfunction\s+([\w-]+)", re.IGNORECASE)],
    ".psm1": [re.compile(r"\bfunction\s+([\w-]+)", re.IGNORECASE)],
    ".lua": [re.compile(r"\b(?:local[ \t]+)?function\s+([\w.:]+)")],
}


def extract_symbols(text, ext):
    """Names defined in `text`, per the language of `ext`. Empty for
    extensions with no pattern set (data/config/markdown), which is fine --
    those just contribute nothing to definition lookup."""
    patterns = SYMBOL_PATTERNS.get(ext.lower())
    if not patterns:
        return set()
    names = set()
    for pat in patterns:
        names.update(m.group(1) for m in pat.finditer(text))
    return names


def get_head_commit(repo_root):
    result = subprocess.run(
        ["git", "-C", repo_root, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def parse_name_status(output):
    """Parse `git diff --name-status` output into [(status, rel_path)],
    status in "A"/"M"/"D". Renames (git detects via -M) are resolved to a
    delete of the old path plus an add of the new path, so callers don't
    need special rename logic. Typechanges (T, e.g. symlink -> regular
    file) are treated as modifications so the new content gets indexed.
    """
    changes = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0]
        if status.startswith("R"):
            old_path, new_path = parts[1], parts[2]
            changes.append(("D", old_path))
            changes.append(("A", new_path))
        else:
            changes.append(("M" if status[0] == "T" else status[0], parts[1]))
    return changes


def git_diff_status(repo_root, old_commit, new_commit):
    result = subprocess.run(
        ["git", "-C", repo_root, "diff", "--name-status", "-M", old_commit, new_commit],
        capture_output=True, text=True, check=True,
    )
    return parse_name_status(result.stdout)


def content_hash(text):
    # Whitespace/case-insensitive so reformatted copies still collapse to the
    # same hash. Mirrors content_key() in api/main.py (kept as a separate
    # function since api/ and indexer/ are independent services/images) --
    # that one does the same collapse at query time for a single result set,
    # this one persists the hash so /duplicates can GROUP BY it across the
    # whole index.
    return hashlib.sha1("".join(text.split()).lower().encode("utf-8")).hexdigest()


def chunk_id(project, repo, rel_path, index):
    # Deterministic (not a shared counter) so parallel workers never collide,
    # and so incremental reindexing can address/replace a file's exact chunks
    # by recomputing the same IDs instead of tracking them separately.
    key = f"{project}/{repo}/{rel_path}#{index}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def get_splitter(ext):
    # chunk_size/chunk_overlap are langchain's parameter names -- only our
    # constants carry the OUTPUT_ prefix.
    if ext in (".tf", ".tfvars"):
        return RecursiveCharacterTextSplitter(
            separators=HCL_SEPARATORS, chunk_size=OUTPUT_CHUNK_SIZE, chunk_overlap=OUTPUT_CHUNK_OVERLAP
        )
    if ext in (".yaml", ".yml"):
        return RecursiveCharacterTextSplitter(
            separators=YAML_SEPARATORS, chunk_size=OUTPUT_CHUNK_SIZE, chunk_overlap=OUTPUT_CHUNK_OVERLAP
        )
    lang = EXT_LANGUAGE.get(ext)
    if lang:
        return RecursiveCharacterTextSplitter.from_language(
            language=lang, chunk_size=OUTPUT_CHUNK_SIZE, chunk_overlap=OUTPUT_CHUNK_OVERLAP
        )
    return RecursiveCharacterTextSplitter(chunk_size=OUTPUT_CHUNK_SIZE, chunk_overlap=OUTPUT_CHUNK_OVERLAP)


def iter_files(repo_root):
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext in SKIP_EXT or filename_is_skipped(fname):
                continue
            full = os.path.join(dirpath, fname)
            try:
                if os.path.getsize(full) > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield full, ext


def get_last_commit_map(repo_root, live_paths):
    """rel_path -> (commit_hash, commit_date) for each file's most recent commit.

    Streams `git log` and stops as soon as every path in live_paths has been
    resolved, instead of walking the full history -- a large win on repos
    with long history relative to their current file count. Uses a NUL byte
    to delimit commit headers from filenames (unlike a string prefix, NUL
    can never appear in a real path, so there's no ambiguity).

    Deliberately excludes author/committer identity (personal data) -- only
    the commit hash and (committer) timestamp, neither of which is tied to
    a person.
    """
    live_paths = {p.replace(os.sep, "/") for p in live_paths}
    mapping = {}
    if not live_paths:
        return mapping

    proc = subprocess.Popen(
        ["git", "-C", repo_root, "log", "--name-only", "--pretty=format:%x00%H%x09%cI"],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        current_hash = current_date = None
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line.startswith("\x00"):
                current_hash, current_date = line[1:].split("\t")
                continue
            if not line or line in mapping or line not in live_paths:
                continue
            mapping[line] = (current_hash, current_date)
            if len(mapping) == len(live_paths):
                break
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()

    return mapping


def line_offsets(text):
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def offset_to_line(offsets, pos):
    lo, hi = 0, len(offsets) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if offsets[mid] <= pos:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1


class Progress:
    """Thread-safe repo-completion counter. Deliberately tracks repos done,
    not chunks -- a chunk-based percentage needs every repo scanned up front
    to know the total, which reintroduces the scan-everything-first delay
    this design avoids."""

    def __init__(self, total_repos):
        self.total_repos = total_repos
        self.repos_done = 0
        self.lock = threading.Lock()

    def add(self):
        with self.lock:
            self.repos_done += 1
            pct = (self.repos_done / self.total_repos * 100) if self.total_repos else 100.0
            return self.repos_done, pct


def get_conn():
    global _db_conn
    with db_lock:
        if _db_conn is None:
            _db_conn = psycopg.connect(DATABASE_URL)
        return _db_conn


def embedding_dimension():
    # Probed from the model rather than configured: the dimension is a
    # property of EMBEDDING_MODEL, and a config knob for it could silently
    # disagree with reality.
    return len(embed_batch(["dimension probe"])[0])


SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    id text PRIMARY KEY,
    project text NOT NULL,
    repo text NOT NULL,
    path text NOT NULL,
    language text NOT NULL,
    content_type text NOT NULL DEFAULT 'code',
    start_line integer,
    end_line integer,
    file_size_bytes bigint,
    chunk_tokens_estimate integer,
    commit_hash text,
    commit_date text,
    content_hash text,
    symbols text,
    text text NOT NULL,
    embedding vector({dim}) NOT NULL
);
-- Per-repo index freshness, projected into Postgres so the API (which has no
-- access to the indexer-only state file) can report how current each repo is.
-- The .index_state.json file remains the source of truth for incremental
-- decisions; this table is a read-only mirror for the API's list_repos.
CREATE TABLE IF NOT EXISTS repo_index_state (
    project text NOT NULL,
    repo text NOT NULL,
    commit_hash text,
    indexed_at text,
    PRIMARY KEY (project, repo)
);
CREATE INDEX IF NOT EXISTS idx_chunks_repo_path ON chunks (project, repo, path);
CREATE INDEX IF NOT EXISTS idx_chunks_content_type ON chunks (content_type);
-- Supports the /duplicates report's GROUP BY content_hash.
CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks (content_hash);
-- Lexical leg of hybrid search. 'simple' config: no stemming/stopwords, so
-- code identifiers (CamelCase names, etc.) match exactly as written.
CREATE INDEX IF NOT EXISTS idx_chunks_fts ON chunks USING gin (to_tsvector('simple', text));
-- Definition lookup (find_definition): same 'simple' config over the names a
-- chunk defines. coalesce so the NULL symbols of pattern-less languages index
-- cleanly. Must match the query expression in api/main.py exactly to be used.
CREATE INDEX IF NOT EXISTS idx_chunks_symbols ON chunks USING gin (to_tsvector('simple', coalesce(symbols, '')));
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);
"""


def ensure_schema():
    dim = embedding_dimension()
    if dim > PGVECTOR_HNSW_MAX_DIM:
        raise RuntimeError(
            f"Embedding model '{EMBED_MODEL}' produces {dim}-dim vectors, above pgvector's "
            f"{PGVECTOR_HNSW_MAX_DIM}-dim HNSW limit -- use a smaller model, or switch the "
            f"index/column to halfvec."
        )
    conn = get_conn()
    with db_lock:
        conn.execute(SCHEMA_SQL.format(dim=dim))
        conn.commit()


def reset_index():
    # Used when the config fingerprint changed (or on first run) -- old
    # chunks aren't comparable to what a new config would produce, so they
    # can't be kept around. The freshness mirror goes too, so it never lists
    # a repo whose chunks were just dropped.
    conn = get_conn()
    with db_lock:
        conn.execute("DROP TABLE IF EXISTS chunks")
        conn.execute("DROP TABLE IF EXISTS repo_index_state")
        conn.commit()
    ensure_schema()


def delete_path_chunks(project, name, rel_path):
    conn = get_conn()
    with db_lock:
        conn.execute(
            "DELETE FROM chunks WHERE project = %s AND repo = %s AND path = %s",
            (project, name, rel_path),
        )
        conn.commit()


def delete_repo_chunks(project, name):
    conn = get_conn()
    with db_lock:
        conn.execute("DELETE FROM chunks WHERE project = %s AND repo = %s", (project, name))
        conn.commit()


def vector_literal(vec):
    return "[" + ",".join(map(str, vec)) + "]"


def add_chunks(ids, payloads, texts, vectors):
    rows = [
        (
            cid, p["project"], p["repo"], p["path"], p["language"], p["content_type"],
            p["start_line"], p["end_line"], p["file_size_bytes"],
            p["chunk_tokens_estimate"], p["commit_hash"], p["commit_date"], p["content_hash"],
            p["symbols"], text, vector_literal(vec),
        )
        for cid, p, text, vec in zip(ids, payloads, texts, vectors)
    ]
    conn = get_conn()
    with db_lock:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks (id, project, repo, path, language, content_type, start_line, end_line,
                                    file_size_bytes, chunk_tokens_estimate, commit_hash, commit_date,
                                    content_hash, symbols, text, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (id) DO UPDATE SET
                    project = EXCLUDED.project, repo = EXCLUDED.repo, path = EXCLUDED.path,
                    language = EXCLUDED.language, content_type = EXCLUDED.content_type,
                    start_line = EXCLUDED.start_line, end_line = EXCLUDED.end_line,
                    file_size_bytes = EXCLUDED.file_size_bytes,
                    chunk_tokens_estimate = EXCLUDED.chunk_tokens_estimate,
                    commit_hash = EXCLUDED.commit_hash, commit_date = EXCLUDED.commit_date,
                    content_hash = EXCLUDED.content_hash, symbols = EXCLUDED.symbols,
                    text = EXCLUDED.text, embedding = EXCLUDED.embedding
                """,
                rows,
            )
        conn.commit()


def embed_batch(texts):
    # Qwen3-Embedding documents get no instruct prefix -- the instruction +
    # "Query:" prefix (see api/main.py) is query-side only. (The path context
    # header prepended by iter_chunk_rows is document content, not a prefix.)
    # NOTE: this GGUF logs "last token is not SEP" (see LM Studio warning) --
    # manually appending "<|endoftext|>" does NOT fix it and instead triggers
    # a second "double EOS" warning, so the server is applying its own EOS
    # logic independent of what we send. Not fixable from the client side;
    # would need the GGUF's tokenizer.ggml.add_eos_token metadata patched.
    # Bounded separately from INDEX_CONCURRENCY: this hits LM Studio's actual
    # model server, which can only usefully parallelize so much regardless of
    # how many host CPU cores are free.
    # The semaphore is deliberately held across rate-limit sleeps: while the
    # provider is throttling us there's no point letting another thread rush
    # in and burn the same exhausted quota window.
    with embed_semaphore:
        return _embed_with_retry(texts)


def _embed_with_retry(texts):
    for attempt in range(EMBED_MAX_RETRIES):
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
            return [d.embedding for d in resp.data]
        except RateLimitError as e:
            _rate_limit_backoff(e, attempt)


def _rate_limit_backoff(exc, attempt):
    """Sleep out a 429 before the next attempt, or re-raise once attempts are
    exhausted. Prefers the server's Retry-After header (Azure says exactly
    when the quota window refills); falls back to exponential backoff capped
    at the typical one-minute window."""
    if attempt >= EMBED_MAX_RETRIES - 1:
        raise exc
    retry_after = exc.response.headers.get("retry-after", "") if exc.response is not None else ""
    delay = int(retry_after) if retry_after.isdigit() else min(60, 2 ** (attempt + 1))
    log(f"[embed] rate limited (429), waiting {delay}s (attempt {attempt + 1}/{EMBED_MAX_RETRIES})")
    time.sleep(delay)


def chunk_payloads(project, name, rel_path, text, ext, file_size_bytes,
                   content_type="code", commit_hash="", commit_date=""):
    """Split `text` into indexable chunks, yielding (chunk_id, payload, text,
    embed_text) for each. Shared by file-based chunking (iter_chunk_rows,
    below) and text with no file on disk (PR titles+descriptions, see
    embed_pull_requests) -- both are just "some text at a repo-relative path"
    once split.

    embed_text is the chunk with a path context header prepended -- that
    variant is what gets embedded (so the vector carries repo/path context),
    while the raw chunk is what gets stored and returned by search.

    commit_hash/commit_date default to blank placeholders: for files,
    embed_files() backfills them from its background commit-map lookup at
    flush time (so that lookup doesn't block the start of embedding); PR
    chunks pass the PR's creation date directly since there's no git commit
    to look up.
    """
    embed_header = f"// {project}/{name}/{rel_path.replace(os.sep, '/')}\n"

    offsets = line_offsets(text)
    cursor = 0
    for i, chunk in enumerate(get_splitter(ext).split_text(text)):
        start = text.find(chunk, cursor)
        if start == -1:
            start = text.find(chunk)
        start_line = offset_to_line(offsets, max(start, 0))
        end_line = offset_to_line(offsets, max(start, 0) + len(chunk))
        cursor = max(start, 0) + 1

        if is_import_dominated(chunk):
            continue

        payload = {
            "project": project,
            "repo": name,
            "path": rel_path,
            "language": ext.lstrip("."),
            "content_type": content_type,
            "start_line": start_line,
            "end_line": end_line,
            "file_size_bytes": file_size_bytes,
            "chunk_tokens_estimate": len(chunk.split()),
            "commit_hash": commit_hash,
            "commit_date": commit_date,
            "content_hash": content_hash(chunk),
            "symbols": " ".join(sorted(extract_symbols(chunk, ext))),
        }
        yield chunk_id(project, name, rel_path, i), payload, chunk, embed_header + chunk


def iter_chunk_rows(project, name, repo_root, file_specs, content_type="code"):
    """Yield (chunk_id, payload, text, embed_text) for every indexable chunk
    across `file_specs`. See chunk_payloads() for the actual splitting."""
    for full_path, ext in file_specs:
        rel_path = os.path.relpath(full_path, repo_root)
        try:
            # utf-8-sig: strips a leading BOM. Plain utf-8 left BOM-only files
            # looking non-empty (U+FEFF isn't whitespace to str.strip()),
            # which put literal one-character "chunks" into the index.
            text = Path(full_path).read_text(encoding="utf-8-sig")
        except (UnicodeDecodeError, OSError):
            continue
        # Postgres text columns reject embedded NUL bytes outright; some source
        # files (encoding artifacts, embedded binary-ish content) legitimately
        # contain them.
        text = text.replace("\x00", "")
        if not text.strip():
            continue
        if is_minified(text):
            continue

        file_size_bytes = os.path.getsize(full_path)
        yield from chunk_payloads(project, name, rel_path, text, ext, file_size_bytes, content_type)


def embed_files(project, name, repo_root, file_specs, content_type="code"):
    """Chunk + embed + upsert a given list of (full_path, ext) pairs.

    Shared by full and incremental indexing -- the only difference between
    them is which files get passed in here (every file in the repo, vs. just
    the ones a git diff says changed).
    """
    if not file_specs:
        return 0

    rel_paths = [os.path.relpath(full_path, repo_root) for full_path, _ in file_specs]
    # Resolved in the background instead of upfront: this is a git-log subprocess
    # that can take seconds on repos with deep history, and commit_hash/commit_date
    # are only needed at DB-write time, not before embedding can start. Running it
    # inline here was serializing every repo's first embedding call behind git log.
    commit_map = {}
    commit_map_error = []

    def _resolve_commit_map():
        try:
            commit_map.update(get_last_commit_map(repo_root, rel_paths))
        except Exception as e:
            commit_map_error.append(e)

    commit_map_thread = threading.Thread(target=_resolve_commit_map, daemon=True)
    commit_map_thread.start()

    batch_ids, batch_texts, batch_embed_texts, batch_payloads = [], [], [], []
    chunk_count = 0

    def flush():
        nonlocal batch_ids, batch_texts, batch_embed_texts, batch_payloads
        if not batch_texts:
            return
        vectors = embed_batch(batch_embed_texts)
        commit_map_thread.join()
        if commit_map_error:
            raise commit_map_error[0]
        for payload in batch_payloads:
            commit_hash, commit_date = commit_map.get(payload["path"].replace(os.sep, "/"), ("", ""))
            payload["commit_hash"] = commit_hash
            payload["commit_date"] = commit_date
        add_chunks(batch_ids, batch_payloads, batch_texts, vectors)
        batch_ids, batch_texts, batch_embed_texts, batch_payloads = [], [], [], []

    for cid, payload, text, embed_text in iter_chunk_rows(project, name, repo_root, file_specs, content_type):
        batch_ids.append(cid)
        batch_payloads.append(payload)
        batch_texts.append(text)
        batch_embed_texts.append(embed_text)
        chunk_count += 1
        if len(batch_texts) >= EMBED_BATCH:
            flush()

    flush()
    return chunk_count


def full_index_repo(project, name, repo_root, content_type="code"):
    # A full pass over a repo that still has rows in the table (fallback
    # after an unusable stored commit, or a lost state file) would otherwise
    # keep stale rows for deleted files searchable. Drop the repo's chunks
    # first; on a fresh repo this is a no-op.
    delete_repo_chunks(project, name)
    return embed_files(project, name, repo_root, list(iter_files(repo_root)), content_type)


def incremental_index_repo(project, name, repo_root, old_commit, new_commit, content_type="code"):
    """Diff old_commit..new_commit and only touch what actually changed.

    Every changed/deleted path's existing chunks are deleted outright (not
    just overwritten by ID) so a file that now produces fewer chunks than
    before doesn't leave orphaned stale chunks at the old, no-longer-used
    indices. Added/modified files are then re-chunked and re-embedded.
    """
    changes = git_diff_status(repo_root, old_commit, new_commit)

    to_embed = []
    for status, rel_path in changes:
        delete_path_chunks(project, name, rel_path)
        if status in ("A", "M") and not path_is_skipped(rel_path):
            full_path = os.path.join(repo_root, rel_path)
            if not os.path.isfile(full_path):
                continue
            try:
                if os.path.getsize(full_path) > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            to_embed.append((full_path, Path(rel_path).suffix.lower()))

    chunk_count = embed_files(project, name, repo_root, to_embed, content_type)
    return len(changes), chunk_count


def index_repo(project, name, repo_root, progress, state, content_type="code"):
    key = f"{project}/{name}"
    head = get_head_commit(repo_root)
    prior = state["repos"].get(key)

    if prior and prior["commit"] == head:
        repos_done, pct = progress.add()
        log(f"[index] ({repos_done}/{progress.total_repos}, {pct:.1f}%) {project}/{name} unchanged, skipped")
        return

    if prior:
        try:
            changed, chunk_count = incremental_index_repo(
                project, name, repo_root, prior["commit"], head, content_type
            )
            repos_done, pct = progress.add()
            log(
                f"[index] ({repos_done}/{progress.total_repos}, {pct:.1f}%) {project}/{name} "
                f"incremental: {changed} files changed, {chunk_count} chunks re-embedded"
            )
        except subprocess.CalledProcessError:
            # e.g. the stored commit no longer exists locally (history was
            # rewritten/force-pushed since we last indexed) -- a diff against
            # it is meaningless, so fall back to a clean full reindex.
            chunk_count = full_index_repo(project, name, repo_root, content_type)
            repos_done, pct = progress.add()
            log(
                f"[index] ({repos_done}/{progress.total_repos}, {pct:.1f}%) {project}/{name} "
                f"full reindex (stored commit unusable, {chunk_count} chunks)"
            )
    else:
        chunk_count = full_index_repo(project, name, repo_root, content_type)
        repos_done, pct = progress.add()
        log(f"[index] ({repos_done}/{progress.total_repos}, {pct:.1f}%) {project}/{name} full index ({chunk_count} chunks)")

    update_repo_state(state, project, name, head)


def update_pr_state(state, project, name, last_pr_id):
    with state_lock:
        state.setdefault("pull_requests", {})[f"{project}/{name}"] = {
            "last_pr_id": last_pr_id,
            "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _write_state_to_disk(state)


def embed_pull_requests(project, name, prs):
    """Chunk + embed + upsert a batch of PRs: [{"id", "title", "description",
    "created_date", "work_item_ids"}, ...]. Each PR becomes one synthetic
    "file" under _pull_requests/ -- there's no file on disk, but
    chunk_payloads() only needs text and a repo-relative path, so a PR's
    title+description slots into the exact same splitting/storage path as a
    real file.

    work_item_ids are appended as bare numbers, not fetched work item content
    -- deliberately: the work item's own title/description/comments are
    where personal data (reporter/customer names) would actually live, and
    this repo indexes no personal data.
    """
    if not prs:
        return 0

    batch_ids, batch_texts, batch_embed_texts, batch_payloads = [], [], [], []
    chunk_count = 0

    def flush():
        nonlocal batch_ids, batch_texts, batch_embed_texts, batch_payloads
        if not batch_texts:
            return
        vectors = embed_batch(batch_embed_texts)
        add_chunks(batch_ids, batch_payloads, batch_texts, vectors)
        batch_ids, batch_texts, batch_embed_texts, batch_payloads = [], [], [], []

    for pr in prs:
        rel_path = f"_pull_requests/{pr['id']}.md"
        text = f"# {pr['title']}\n\n{pr['description']}"
        if pr.get("work_item_ids"):
            text += "\n\nLinked work items: " + ", ".join(f"#{i}" for i in pr["work_item_ids"])
        for cid, payload, chunk, embed_text in chunk_payloads(
            project, name, rel_path, text, ".md", len(text.encode("utf-8")),
            content_type="pr", commit_date=pr["created_date"],
        ):
            batch_ids.append(cid)
            batch_payloads.append(payload)
            batch_texts.append(chunk)
            batch_embed_texts.append(embed_text)
            chunk_count += 1
            if len(batch_texts) >= EMBED_BATCH:
                flush()

    flush()
    return chunk_count


def index_pull_requests(session, org_url, project, name, state):
    """Fetch and embed any PRs newer than what's already indexed for this
    repo. Descriptions edited after their PR was first indexed are not
    re-fetched -- the cursor only tracks the highest PR id seen, not content
    changes, to avoid re-embedding a repo's entire PR history every run.
    """
    key = f"{project}/{name}"
    since_id = state.get("pull_requests", {}).get(key, {}).get("last_pr_id", 0)
    prs = pull_requests.fetch_new_pull_requests(session, org_url, project, name, since_id)
    if not prs:
        log(f"[pr] {project}/{name} up to date")
        return
    chunk_count = embed_pull_requests(project, name, prs)
    update_pr_state(state, project, name, max(pr["id"] for pr in prs))
    log(f"[pr] {project}/{name}: {len(prs)} new PRs, {chunk_count} chunks")


def main():
    config = load_config(CONFIG_PATH)
    code_repos = valid_repos(config)
    all_repos = code_repos + (wiki_repos(config) if should_index_wikis(config) else [])

    synced = []
    for repo in all_repos:
        project, name = repo["project"], repo["name"]
        repo_root = os.path.join(DATA_DIR, project, name)
        if not os.path.isdir(repo_root):
            log(f"[skip] {project}/{name} not synced yet")
            continue
        content_type = "wiki" if name.endswith(".wiki") else "code"
        synced.append((project, name, repo_root, content_type))

    state = load_state()
    fingerprint = config_fingerprint()
    if state.get("config_fingerprint") == fingerprint:
        log("[index] config unchanged since last run -- incremental mode")
        ensure_schema()
    else:
        log("[index] config changed (or first run) -- full reindex, dropping old state")
        reset_index()
        state = {"config_fingerprint": fingerprint, "repos": {}, "pull_requests": {}}
        _write_state_to_disk(state)  # persist immediately: a crash mid-run must not leave a stale fingerprint

    progress = Progress(len(synced))

    failures = []
    with ThreadPoolExecutor(max_workers=INDEX_CONCURRENCY) as pool:
        futures = {
            pool.submit(index_repo, project, name, repo_root, progress, state, content_type): (project, name)
            for project, name, repo_root, content_type in synced
        }
        for future in as_completed(futures):
            project, name = futures[future]
            try:
                future.result()
            except Exception as e:
                failures.append(f"{project}/{name}")
                log(f"[fail] {project}/{name}: {e}")

    log(f"[index] complete: {len(synced) - len(failures)}/{len(synced)} ok, {len(failures)} failed")
    if failures:
        log("[index] failed repos: " + ", ".join(failures))

    if should_index_pull_requests(config):
        pat = os.environ["AZURE_DEVOPS_PAT"]
        org_url = config["azure_devops"]["organization"]
        session = pull_requests.make_session(pat)

        pr_failures = []
        with ThreadPoolExecutor(max_workers=INDEX_CONCURRENCY) as pool:
            futures = {
                pool.submit(index_pull_requests, session, org_url, r["project"], r["name"], state): (
                    r["project"], r["name"]
                )
                for r in code_repos
            }
            for future in as_completed(futures):
                project, name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    pr_failures.append(f"{project}/{name}")
                    log(f"[pr-fail] {project}/{name}: {e}")

        log(f"[pr] complete: {len(code_repos) - len(pr_failures)}/{len(code_repos)} ok, {len(pr_failures)} failed")


if __name__ == "__main__":
    main()
