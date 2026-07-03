import hashlib
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
import chromadb
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

CONFIG_PATH = os.environ.get("REPOS_CONFIG", "/config/repos.yaml")
DATA_DIR = os.environ.get("REPO_DATA_DIR", "/data/repos")
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "code_chunks")
# Lives at the DATA_DIR root, not inside any repo, so it's never picked up by
# iter_files(). One JSON file tracks every repo's last-indexed commit --
# simple enough to inspect/edit by hand, and it rides along in the same
# repo_data volume that already persists across container restarts.
STATE_PATH = os.environ.get("INDEX_STATE_PATH", os.path.join(DATA_DIR, ".index_state.json"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))
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

LM_BASE_URL = os.environ["LMSTUDIO_BASE_URL"]
EMBED_MODEL = os.environ["EMBEDDING_MODEL"]
CHROMA_HOST = os.environ.get("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))

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
}
SKIP_EXT = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip",
    ".dll", ".exe", ".so", ".pdb", ".woff", ".woff2", ".ttf", ".svg",
    # Tabular/columnar data formats -- data, not code.
    ".csv", ".tsv", ".parquet",
}
MAX_FILE_BYTES = 1_000_000

client = OpenAI(base_url=LM_BASE_URL, api_key="lm-studio")
chroma = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
embed_semaphore = threading.Semaphore(EMBED_CONCURRENCY)
print_lock = threading.Lock()
state_lock = threading.Lock()


def log(msg):
    with print_lock:
        print(msg, flush=True)


def config_fingerprint():
    # Everything that changes what a chunk looks like or which files are even
    # considered. If any of this changes between runs, previously-indexed
    # chunks aren't comparable to new ones -- stored commit hashes get
    # ignored and every repo gets a full reindex (see main()).
    payload = {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embed_model": EMBED_MODEL,
        "skip_ext": sorted(SKIP_EXT),
        "skip_dirs": sorted(SKIP_DIRS),
        "max_file_bytes": MAX_FILE_BYTES,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def load_state():
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"config_fingerprint": None, "repos": {}}


def _write_state_to_disk(state):
    # Write-then-rename so a process kill mid-write (this run has been killed
    # by Docker Desktop crashes more than once) can't leave a half-written,
    # unparseable state file behind.
    tmp_path = STATE_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, STATE_PATH)


def update_repo_state(state, project, name, commit):
    with state_lock:
        state["repos"][f"{project}/{name}"] = {
            "commit": commit,
            "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _write_state_to_disk(state)


def path_is_skipped(rel_path):
    parts = Path(rel_path).parts
    if any(part in SKIP_DIRS for part in parts[:-1]):
        return True
    return Path(rel_path).suffix.lower() in SKIP_EXT


def get_head_commit(repo_root):
    result = subprocess.run(
        ["git", "-C", repo_root, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def git_diff_status(repo_root, old_commit, new_commit):
    """[(status, rel_path)] changed between two commits, status in "A"/"M"/"D".
    Renames (git detects via -M) are resolved to a delete of the old path
    plus an add of the new path, so callers don't need special rename logic.
    """
    result = subprocess.run(
        ["git", "-C", repo_root, "diff", "--name-status", "-M", old_commit, new_commit],
        capture_output=True, text=True, check=True,
    )
    changes = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0]
        if status.startswith("R"):
            old_path, new_path = parts[1], parts[2]
            changes.append(("D", old_path))
            changes.append(("A", new_path))
        else:
            changes.append((status[0], parts[1]))
    return changes


def chunk_id(project, repo, rel_path, index):
    # Deterministic (not a shared counter) so parallel workers never collide,
    # and so a future incremental-reindex can address/replace a file's exact
    # chunks by recomputing the same IDs instead of tracking them separately.
    key = f"{project}/{repo}/{rel_path}#{index}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def get_splitter(ext):
    if ext in (".tf", ".tfvars"):
        return RecursiveCharacterTextSplitter(
            separators=HCL_SEPARATORS, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
    if ext in (".yaml", ".yml"):
        return RecursiveCharacterTextSplitter(
            separators=YAML_SEPARATORS, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
    lang = EXT_LANGUAGE.get(ext)
    if lang:
        return RecursiveCharacterTextSplitter.from_language(
            language=lang, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
    return RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def iter_files(repo_root):
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext in SKIP_EXT:
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


def recreate_collection():
    # Used when the config fingerprint changed (or on first run) -- old
    # chunks aren't comparable to what a new config would produce, so they
    # can't be kept around.
    try:
        chroma.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return chroma.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def get_or_create_collection():
    try:
        return chroma.get_collection(COLLECTION_NAME)
    except Exception:
        return chroma.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def delete_path_chunks(collection, project, name, rel_path):
    collection.delete(where={"$and": [
        {"project": {"$eq": project}},
        {"repo": {"$eq": name}},
        {"path": {"$eq": rel_path}},
    ]})


def embed_batch(texts):
    # Qwen3-Embedding expects documents with no special prefix -- the
    # instruction+query prefix (see api/main.py) is query-side only.
    # NOTE: this GGUF logs "last token is not SEP" (see LM Studio warning) --
    # manually appending "<|endoftext|>" does NOT fix it and instead triggers
    # a second "double EOS" warning, so the server is applying its own EOS
    # logic independent of what we send. Not fixable from the client side;
    # would need the GGUF's tokenizer.ggml.add_eos_token metadata patched.
    # Bounded separately from INDEX_CONCURRENCY: this hits LM Studio's actual
    # model server, which can only usefully parallelize so much regardless of
    # how many host CPU cores are free.
    with embed_semaphore:
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [d.embedding for d in resp.data]


def embed_files(collection, project, name, repo_root, file_specs):
    """Chunk + embed + add a given list of (full_path, ext) pairs.

    Shared by full and incremental indexing -- the only difference between
    them is which files get passed in here (every file in the repo, vs. just
    the ones a git diff says changed).
    """
    if not file_specs:
        return 0

    rel_paths = [os.path.relpath(full_path, repo_root) for full_path, _ in file_specs]
    commit_map = get_last_commit_map(repo_root, rel_paths)

    batch_ids, batch_texts, batch_payloads = [], [], []
    chunk_count = 0

    def flush():
        nonlocal batch_ids, batch_texts, batch_payloads
        if not batch_texts:
            return
        vectors = embed_batch(batch_texts)
        collection.add(ids=batch_ids, embeddings=vectors, documents=batch_texts, metadatas=batch_payloads)
        batch_ids, batch_texts, batch_payloads = [], [], []

    for full_path, ext in file_specs:
        rel_path = os.path.relpath(full_path, repo_root)
        try:
            text = Path(full_path).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if not text.strip():
            continue

        file_size_bytes = os.path.getsize(full_path)
        commit_hash, commit_date = commit_map.get(rel_path.replace(os.sep, "/"), (None, None))

        offsets = line_offsets(text)
        cursor = 0
        for i, chunk in enumerate(get_splitter(ext).split_text(text)):
            start = text.find(chunk, cursor)
            if start == -1:
                start = text.find(chunk)
            start_line = offset_to_line(offsets, max(start, 0))
            end_line = offset_to_line(offsets, max(start, 0) + len(chunk))
            cursor = max(start, 0) + 1

            batch_ids.append(chunk_id(project, name, rel_path, i))
            batch_payloads.append({
                "project": project,
                "repo": name,
                "path": rel_path,
                "language": ext.lstrip("."),
                "start_line": start_line,
                "end_line": end_line,
                "file_size_bytes": file_size_bytes,
                "chunk_tokens_estimate": len(chunk.split()),
                "commit_hash": commit_hash or "",
                "commit_date": commit_date or "",
            })
            batch_texts.append(chunk)
            chunk_count += 1
            if len(batch_texts) >= EMBED_BATCH:
                flush()

    flush()
    return chunk_count


def full_index_repo(collection, project, name, repo_root):
    return embed_files(collection, project, name, repo_root, list(iter_files(repo_root)))


def incremental_index_repo(collection, project, name, repo_root, old_commit, new_commit):
    """Diff old_commit..new_commit and only touch what actually changed.

    Every changed/deleted path's existing chunks are deleted outright (not
    just overwritten by ID) so a file that now produces fewer chunks than
    before doesn't leave orphaned stale chunks at the old, no-longer-used
    indices. Added/modified files are then re-chunked and re-embedded.
    """
    changes = git_diff_status(repo_root, old_commit, new_commit)

    to_embed = []
    for status, rel_path in changes:
        delete_path_chunks(collection, project, name, rel_path)
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

    chunk_count = embed_files(collection, project, name, repo_root, to_embed)
    return len(changes), chunk_count


def index_repo(collection, project, name, repo_root, progress, state):
    key = f"{project}/{name}"
    head = get_head_commit(repo_root)
    prior = state["repos"].get(key)

    if prior and prior["commit"] == head:
        repos_done, pct = progress.add()
        log(f"[index] ({repos_done}/{progress.total_repos}, {pct:.1f}%) {project}/{name} unchanged, skipped")
        return

    if prior:
        try:
            changed, chunk_count = incremental_index_repo(collection, project, name, repo_root, prior["commit"], head)
            repos_done, pct = progress.add()
            log(
                f"[index] ({repos_done}/{progress.total_repos}, {pct:.1f}%) {project}/{name} "
                f"incremental: {changed} files changed, {chunk_count} chunks re-embedded"
            )
        except subprocess.CalledProcessError:
            # e.g. the stored commit no longer exists locally (history was
            # rewritten/force-pushed since we last indexed) -- a diff against
            # it is meaningless, so fall back to a clean full reindex.
            chunk_count = full_index_repo(collection, project, name, repo_root)
            repos_done, pct = progress.add()
            log(
                f"[index] ({repos_done}/{progress.total_repos}, {pct:.1f}%) {project}/{name} "
                f"full reindex (stored commit unusable, {chunk_count} chunks)"
            )
    else:
        chunk_count = full_index_repo(collection, project, name, repo_root)
        repos_done, pct = progress.add()
        log(f"[index] ({repos_done}/{progress.total_repos}, {pct:.1f}%) {project}/{name} full index ({chunk_count} chunks)")

    update_repo_state(state, project, name, head)


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    synced = []
    for i, repo in enumerate(config["repos"]):
        if "project" not in repo or "name" not in repo:
            log(f"[config-error] repos.yaml entry #{i} is missing 'project' or 'name', skipping: {repo}")
            continue
        project, name = repo["project"], repo["name"]
        repo_root = os.path.join(DATA_DIR, project, name)
        if not os.path.isdir(repo_root):
            log(f"[skip] {project}/{name} not synced yet")
            continue
        synced.append((project, name, repo_root))

    state = load_state()
    fingerprint = config_fingerprint()
    if state.get("config_fingerprint") == fingerprint:
        log("[index] config unchanged since last run -- incremental mode")
        collection = get_or_create_collection()
    else:
        log("[index] config changed (or first run) -- full reindex, dropping old state")
        collection = recreate_collection()
        state = {"config_fingerprint": fingerprint, "repos": {}}
        _write_state_to_disk(state)  # persist immediately: a crash mid-run must not leave a stale fingerprint

    progress = Progress(len(synced))

    failures = []
    with ThreadPoolExecutor(max_workers=INDEX_CONCURRENCY) as pool:
        futures = {
            pool.submit(index_repo, collection, project, name, repo_root, progress, state): (project, name)
            for project, name, repo_root in synced
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


if __name__ == "__main__":
    main()
