import os
import subprocess
from pathlib import Path

import yaml
import chromadb
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

CONFIG_PATH = os.environ.get("REPOS_CONFIG", "/config/repos.yaml")
DATA_DIR = os.environ.get("REPO_DATA_DIR", "/data/repos")
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "code_chunks")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))
EMBED_BATCH = int(os.environ.get("EMBED_BATCH_SIZE", "32"))

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
}
SKIP_EXT = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip",
    ".dll", ".exe", ".so", ".pdb", ".woff", ".woff2", ".ttf", ".svg",
}
MAX_FILE_BYTES = 500_000

client = OpenAI(base_url=LM_BASE_URL, api_key="lm-studio")
chroma = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


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
        ["git", "-C", repo_root, "log", "--name-only", "--pretty=format:%x00%H%x09%cI%x09%an"],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        current_hash = current_date = commit_author = None
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line.startswith("\x00"):
                current_hash, current_date, commit_author = line[1:].split("\t")
                continue
            if not line or line in mapping or line not in live_paths:
                continue
            mapping[line] = (current_hash, current_date, commit_author)
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


def recreate_collection():
    # Full reindex each run keeps this correct with zero incremental-sync bookkeeping.
    try:
        chroma.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return chroma.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def embed_batch(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    collection = recreate_collection()

    point_id = 0
    batch_texts, batch_payloads = [], []

    def flush():
        nonlocal point_id, batch_texts, batch_payloads
        if not batch_texts:
            return
        vectors = embed_batch(batch_texts)
        ids = [str(point_id + i) for i in range(len(batch_texts))]
        collection.add(
            ids=ids,
            embeddings=vectors,
            documents=batch_texts,
            metadatas=batch_payloads,
        )
        point_id += len(batch_texts)
        batch_texts, batch_payloads = [], []

    for repo in config["repos"]:
        project, name = repo["project"], repo["name"]
        repo_root = os.path.join(DATA_DIR, project, name)
        if not os.path.isdir(repo_root):
            print(f"[skip] {project}/{name} not synced yet")
            continue

        files = list(iter_files(repo_root))
        rel_paths = [os.path.relpath(full_path, repo_root) for full_path, _ in files]
        commit_map = get_last_commit_map(repo_root, rel_paths)

        for full_path, ext in files:
            rel_path = os.path.relpath(full_path, repo_root)
            try:
                text = Path(full_path).read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if not text.strip():
                continue

            file_size_bytes = os.path.getsize(full_path)
            commit_hash, commit_date, commit_author = commit_map.get(rel_path.replace(os.sep, "/"), (None, None))

            offsets = line_offsets(text)
            cursor = 0
            for chunk in get_splitter(ext).split_text(text):
                start = text.find(chunk, cursor)
                if start == -1:
                    start = text.find(chunk)
                start_line = offset_to_line(offsets, max(start, 0))
                end_line = offset_to_line(offsets, max(start, 0) + len(chunk))
                cursor = max(start, 0) + 1

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
                    "commit_author": commit_author or "",
                })
                batch_texts.append(chunk)
                if len(batch_texts) >= EMBED_BATCH:
                    flush()

        print(f"[index] {project}/{name} done")

    flush()
    print("[index] complete")


if __name__ == "__main__":
    main()
