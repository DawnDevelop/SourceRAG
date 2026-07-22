import hashlib
import os
import re
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Optional
from urllib.parse import urlparse

import psycopg
from fastapi import Depends, FastAPI, HTTPException
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from openai import OpenAI

from auth import AUTH_ENABLED, OAUTH_ISSUER, OAUTH_SCOPES, PUBLIC_BASE_URL, require_auth, verifier

LLM_BASE_URL = os.environ["LLM_BASE_URL"]
LLM_API_KEY = os.environ["LLM_API_KEY"]
EMBED_MODEL = os.environ["EMBEDDING_MODEL"]
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://sourcerag:sourcerag@postgres:5432/sourcerag")
# Hard cap on results per query -- each hit carries a full chunk of source
# text, so an unbounded top_k lets one request pull arbitrarily much data.
MAX_TOP_K = 50
# Both retrieval legs over-fetch beyond top_k so that near-duplicate collapse
# and post-filters still leave top_k distinct results to return.
OVERFETCH_CAP = 200
# Standard reciprocal-rank-fusion constant: big enough that a mediocre rank
# in one leg can't drown out a top rank in the other.
RRF_K = 60

HIT_COLUMNS = (
    "id, project, repo, path, content_type, start_line, end_line, file_size_bytes, "
    "chunk_tokens_estimate, commit_hash, commit_date, text"
)
DEFAULT_DUPLICATE_MIN_CHARS = 200
MAX_DUPLICATE_CLUSTERS = 200

client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


class IndexNotReady(Exception):
    """The chunks table isn't reachable or doesn't exist yet
    (typically: the indexer hasn't completed a first run)."""


def get_conn():
    try:
        return psycopg.connect(DATABASE_URL)
    except psycopg.Error as e:
        raise IndexNotReady(f"Database is not reachable -- is the postgres service up? ({e})") from e


def rows_to_hits(rows) -> list[dict]:
    return [
        {
            "id": r[0], "project": r[1], "repo": r[2], "path": r[3], "content_type": r[4],
            "start_line": r[5], "end_line": r[6], "file_size_bytes": r[7],
            "chunk_tokens_estimate": r[8], "commit_hash": r[9], "commit_date": r[10],
            "text": r[11],
        }
        for r in rows
    ]


def filter_clauses(repo: Optional[str], language: Optional[str], path_contains: Optional[str],
                   content_type: Optional[str] = None):
    sql, params = "", []
    if repo:
        sql += " AND repo = %s"
        params.append(repo)
    if language:
        sql += " AND language = %s"
        params.append(language)
    if content_type:
        sql += " AND content_type = %s"
        params.append(content_type)
    if path_contains:
        # chr(92) is backslash -- tolerate either path separator in the index.
        # %/_ are escaped so they match literally, not as LIKE wildcards.
        escaped = (path_contains.replace("\\", "/")
                   .replace("%", r"\%").replace("_", r"\_"))
        sql += " AND replace(path, chr(92), '/') ILIKE %s"
        params.append(f"%{escaped}%")
    return sql, params


def vector_search(conn, vec_literal: str, n: int, repo: Optional[str], language: Optional[str],
                  path_contains: Optional[str], content_type: Optional[str] = None) -> list[dict]:
    filters, params = filter_clauses(repo, language, path_contains, content_type)
    sql = f"""
        SELECT {HIT_COLUMNS}, 1 - (embedding <=> %s::vector) AS score
        FROM chunks
        WHERE TRUE{filters}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    try:
        rows = conn.execute(sql, [vec_literal, *params, vec_literal, n]).fetchall()
    except psycopg.errors.UndefinedTable as e:
        raise IndexNotReady(
            f"The chunks table does not exist -- has the indexer completed a first run? ({e})"
        ) from e
    hits = rows_to_hits(rows)
    for hit, row in zip(hits, rows):
        hit["score"] = row[12]
    return hits


def tokenize_query(q: str) -> Optional[str]:
    """Turn a free-text query into a websearch_to_tsquery input: identifier-ish
    tokens, deduplicated case-insensitively, OR-joined. websearch_to_tsquery
    is total (never raises on odd input), so no further escaping is needed."""
    terms, seen = [], set()
    for t in re.findall(r"[A-Za-z0-9_]{2,}", q):
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            terms.append(t)
    terms = terms[:16]
    if not terms:
        return None
    return " OR ".join(terms)


def lexical_search(conn, q: str, n: int, repo: Optional[str], language: Optional[str],
                   path_contains: Optional[str], content_type: Optional[str] = None) -> list[dict]:
    """Full-text leg of hybrid search, ranked best-first. 'simple' config:
    no stemming/stopwords, so code identifiers match exactly as written."""
    tsquery = tokenize_query(q)
    if not tsquery:
        return []

    filters, params = filter_clauses(repo, language, path_contains, content_type)
    sql = f"""
        SELECT {HIT_COLUMNS}
        FROM chunks, websearch_to_tsquery('simple', %s) query
        WHERE to_tsvector('simple', text) @@ query{filters}
        ORDER BY ts_rank(to_tsvector('simple', text), query) DESC
        LIMIT %s
    """
    try:
        rows = conn.execute(sql, [tsquery, *params, n]).fetchall()
    except psycopg.errors.UndefinedTable as e:
        raise IndexNotReady(
            f"The chunks table does not exist -- has the indexer completed a first run? ({e})"
        ) from e
    return rows_to_hits(rows)


def content_key(text: str) -> str:
    # Whitespace-insensitive so reformatted copies still collapse.
    return hashlib.sha1("".join(text.split()).lower().encode("utf-8")).hexdigest()


def fuse_hits(vector_hits: list[dict], lexical_hits: list[dict], top_k: int,
              min_score: float = 0.0, compact: bool = False, max_chars: int = 0) -> list[dict]:
    """Reciprocal-rank fusion of both retrieval legs, then near-duplicate
    collapse. Pure function -- both inputs are best-first hit lists keyed by
    "id"; vector hits additionally carry "score" (cosine similarity)."""
    candidates: dict[str, dict] = {}
    for rank, hit in enumerate(vector_hits):
        candidates[hit["id"]] = {**hit, "matched_by": {"semantic"}, "rrf": 1 / (RRF_K + rank)}
    for rank, hit in enumerate(lexical_hits):
        if hit["id"] in candidates:
            candidates[hit["id"]]["matched_by"].add("lexical")
            candidates[hit["id"]]["rrf"] += 1 / (RRF_K + rank)
        else:
            candidates[hit["id"]] = {**hit, "score": None, "matched_by": {"lexical"}, "rrf": 1 / (RRF_K + rank)}

    ranked = sorted(candidates.values(), key=lambda h: (-h["rrf"], h["id"]))

    hits: list[dict] = []
    seen_content: dict[str, dict] = {}
    for hit in ranked:
        # min_score applies to semantic similarity; exact keyword matches
        # stay in regardless (their relevance isn't cosine-measured).
        if "lexical" not in hit["matched_by"] and hit["score"] is not None and hit["score"] < min_score:
            continue

        key = content_key(hit["text"])
        if key in seen_content:
            # Near-identical copy (the org copy-pastes files across services):
            # fold it into the kept hit instead of burning a result slot.
            keeper = seen_content[key]
            if len(keeper.setdefault("duplicates", [])) < 5:
                keeper["duplicates"].append({
                    "project": hit["project"], "repo": hit["repo"],
                    "path": hit["path"], "start_line": hit["start_line"],
                })
            continue

        if len(hits) < top_k:
            seen_content[key] = hit
            hits.append(hit)
        # Past top_k, keep scanning only to attach duplicates to kept hits.

    for hit in hits:
        hit.pop("id")
        hit.pop("rrf")
        matched = hit.pop("matched_by")
        hit["matched_by"] = "both" if len(matched) == 2 else next(iter(matched))
        text = hit.pop("text")
        if compact:
            hit["snippet"] = text[:200]
        elif max_chars > 0 and len(text) > max_chars:
            hit["text"] = text[:max_chars]
            hit["truncated"] = True
        else:
            hit["text"] = text
    return hits


def search_chunks(q: str, top_k: int = 8, repo: Optional[str] = None, language: Optional[str] = None,
                  path_contains: Optional[str] = None, min_score: float = 0.0,
                  compact: bool = False, max_chars: int = 0, content_type: Optional[str] = None) -> list[dict]:
    top_k = max(1, min(top_k, MAX_TOP_K))
    fetch_n = min(max(top_k * 4, 40), OVERFETCH_CAP)

    # Qwen3-Embedding: queries need an instruction + "Query:" prefix for best
    # retrieval quality (documents are embedded with only a path header).
    instruct = "Instruct: Given a code search query, retrieve relevant source code snippets\nQuery:"
    vector = client.embeddings.create(model=EMBED_MODEL, input=[f"{instruct}{q}"]).data[0].embedding
    vec_literal = "[" + ",".join(map(str, vector)) + "]"

    with get_conn() as conn:
        vector_hits = vector_search(conn, vec_literal, fetch_n, repo, language, path_contains, content_type)
        lexical_hits = lexical_search(conn, q, fetch_n, repo, language, path_contains, content_type)

    return fuse_hits(vector_hits, lexical_hits, top_k, min_score, compact, max_chars)


def find_duplicate_clusters(min_repos: int = 2, min_chars: int = DEFAULT_DUPLICATE_MIN_CHARS,
                            limit: int = 50, content_type: str = "code") -> list[dict]:
    """Clusters of near-identical chunks (see content_hash in the indexer),
    ranked by how many distinct repos each spans -- the signal for "should
    this be a shared package", not raw occurrence count (a chunk copy-pasted
    5x within one repo is normal reuse-by-copy; duplicated across 5 repos is
    a package-extraction candidate).
    """
    limit = max(1, min(limit, MAX_DUPLICATE_CLUSTERS))
    min_repos = max(1, min_repos)
    sql = """
        SELECT content_hash,
               count(*) AS occurrence_count,
               count(DISTINCT (project, repo)) AS repo_count,
               (array_agg(text ORDER BY project, repo, path))[1] AS snippet,
               jsonb_agg(
                   jsonb_build_object('project', project, 'repo', repo, 'path', path,
                                      'start_line', start_line, 'end_line', end_line)
                   ORDER BY project, repo, path
               ) AS locations
        FROM chunks
        WHERE content_hash IS NOT NULL AND length(text) >= %s AND content_type = %s
        GROUP BY content_hash
        HAVING count(DISTINCT (project, repo)) >= %s
        ORDER BY repo_count DESC, occurrence_count DESC
        LIMIT %s
    """
    with get_conn() as conn:
        try:
            rows = conn.execute(sql, [min_chars, content_type, min_repos, limit]).fetchall()
        except (psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn) as e:
            raise IndexNotReady(
                f"The chunks table isn't ready for duplicate reporting -- has the indexer completed "
                f"a full reindex since this feature was added? ({e})"
            ) from e
    return [
        {"occurrence_count": r[1], "repo_count": r[2], "snippet": r[3][:200], "locations": r[4]}
        for r in rows
    ]


def _find_gaps(present, lo, hi) -> list[dict]:
    """Line ranges in [lo, hi] with no stored content -- dropped import chunks,
    or a file whose top/middle wasn't indexed. `present` is the set (or dict
    keyed by) of line numbers that DO have content."""
    gaps, run_start = [], None
    for ln in range(lo, hi + 1):
        if ln not in present:
            run_start = run_start if run_start is not None else ln
            continue
        if run_start is not None:
            gaps.append({"start_line": run_start, "end_line": ln - 1})
            run_start = None
    if run_start is not None:
        gaps.append({"start_line": run_start, "end_line": hi})
    return gaps


def stitch_chunks(chunks, from_line: int = 0, to_line: int = 0):
    """Reassemble a file from its stored chunks. Pure function; `chunks` is a
    list of {"start_line", "end_line", "text"}. Returns (text, gaps).

    Reconstruction is approximate by design -- chunks overlap (deduped here by
    line number: the first chunk to supply a line wins) and import-dominated
    chunks were dropped at index time (surfaced as gaps). It is context, not a
    byte-exact copy of the file in git. Optional from_line/to_line clip the
    output to a window (e.g. the neighbourhood of a search hit)."""
    line_map = {}
    for c in sorted(chunks, key=lambda c: c["start_line"]):
        for i, line in enumerate(c["text"].splitlines()):
            line_map.setdefault(c["start_line"] + i, line)
    if not line_map:
        return "", []
    lo = from_line if from_line > 0 else 1
    hi = min(to_line, max(line_map)) if to_line > 0 else max(line_map)
    present = {ln: line_map[ln] for ln in line_map if lo <= ln <= hi}
    if not present:
        return "", []
    text = "\n".join(present[ln] for ln in sorted(present))
    return text, _find_gaps(present, lo, hi)


def file_context(project: str, repo: str, path: str, from_line: int = 0,
                 to_line: int = 0, max_chars: int = 0) -> dict:
    """Reconstruct one indexed file's text from its stored chunks. See
    stitch_chunks for the fidelity caveats. Named differently from the
    get_file_context MCP tool that wraps it, so the tool doesn't shadow it."""
    sql = """
        SELECT start_line, end_line, text FROM chunks
        WHERE project = %s AND repo = %s AND path = %s
        ORDER BY start_line
    """
    with get_conn() as conn:
        try:
            rows = conn.execute(sql, [project, repo, path]).fetchall()
        except psycopg.errors.UndefinedTable as e:
            raise IndexNotReady(
                f"The chunks table does not exist -- has the indexer completed a first run? ({e})"
            ) from e
    base = {"project": project, "repo": repo, "path": path}
    if not rows:
        return {**base, "found": False, "text": "",
                "note": "No indexed chunks for this exact project/repo/path -- check list_repos "
                        "for indexed repo names, or the file may have been skipped at index time."}
    chunks = [{"start_line": r[0], "end_line": r[1], "text": r[2]} for r in rows]
    text, gaps = stitch_chunks(chunks, from_line, to_line)
    truncated = max_chars > 0 and len(text) > max_chars
    return {**base, "found": True,
            "text": text[:max_chars] if truncated else text,
            "truncated": truncated, "gaps": gaps,
            "indexed_start_line": min(c["start_line"] for c in chunks),
            "indexed_end_line": max(c["end_line"] for c in chunks)}


def _repo_freshness(conn) -> dict:
    """(project, repo) -> {last_indexed_commit, last_indexed_at} from the
    indexer's freshness mirror. Absent table (index predates the feature and
    hasn't been rebuilt) is not fatal -- freshness is then simply unknown."""
    try:
        rows = conn.execute(
            "SELECT project, repo, commit_hash, indexed_at FROM repo_index_state"
        ).fetchall()
    except psycopg.errors.UndefinedTable:
        conn.rollback()  # clear the aborted transaction so nothing after us fails
        return {}
    return {(r[0], r[1]): {"last_indexed_commit": r[2], "last_indexed_at": r[3]} for r in rows}


def list_repositories() -> dict:
    """What's actually in the index: repos (with chunk/file counts and index
    freshness), languages, and the total chunk count -- so an agent can learn
    valid `repo`/`language` filter values instead of guessing, and judge how
    current an answer is."""
    with get_conn() as conn:
        try:
            repo_rows = conn.execute("""
                SELECT project, repo, content_type, count(*), count(DISTINCT path)
                FROM chunks GROUP BY project, repo, content_type
                ORDER BY project, repo, content_type
            """).fetchall()
            lang_rows = conn.execute("""
                SELECT language, count(*) FROM chunks GROUP BY language ORDER BY count(*) DESC
            """).fetchall()
        except psycopg.errors.UndefinedTable as e:
            raise IndexNotReady(
                f"The chunks table does not exist -- has the indexer completed a first run? ({e})"
            ) from e
        freshness = _repo_freshness(conn)  # last: an aborted lookup mustn't poison the reads above
    unknown = {"last_indexed_commit": None, "last_indexed_at": None}
    repos = [
        {"project": r[0], "repo": r[1], "content_type": r[2], "chunks": r[3], "files": r[4],
         **freshness.get((r[0], r[1]), unknown)}
        for r in repo_rows
    ]
    return {
        "repositories": repos,
        "languages": [{"language": r[0], "chunks": r[1]} for r in lang_rows],
        "total_chunks": sum(r[3] for r in repo_rows),
    }


def find_definitions(name: str, repo: Optional[str] = None, language: Optional[str] = None,
                     path_contains: Optional[str] = None, top_k: int = 8,
                     max_chars: int = 0) -> list[dict]:
    """Chunks that *define* `name` (class/type/function/method), via the
    symbols column the indexer populates -- distinct from search_code, which
    matches any mention. Returns the same hit shape plus a `symbols` field."""
    top_k = max(1, min(top_k, MAX_TOP_K))
    token = " ".join(re.findall(r"[A-Za-z0-9_]+", name))
    if not token:
        return []
    filters, params = filter_clauses(repo, language, path_contains)
    sql = f"""
        SELECT {HIT_COLUMNS}, symbols,
               ts_rank(to_tsvector('simple', coalesce(symbols, '')), query) AS rank
        FROM chunks, websearch_to_tsquery('simple', %s) query
        WHERE to_tsvector('simple', coalesce(symbols, '')) @@ query{filters}
        ORDER BY rank DESC
        LIMIT %s
    """
    with get_conn() as conn:
        try:
            rows = conn.execute(sql, [token, *params, top_k]).fetchall()
        except (psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn) as e:
            raise IndexNotReady(
                f"The chunks table isn't ready for definition lookup -- has the indexer completed "
                f"a full reindex since this feature was added? ({e})"
            ) from e
    hits = rows_to_hits(rows)
    for hit, row in zip(hits, rows):
        hit["symbols"] = row[12]
        text = hit.pop("text")
        truncated = max_chars > 0 and len(text) > max_chars
        hit["text"] = text[:max_chars] if truncated else text
        if truncated:
            hit["truncated"] = True
    return hits


# MCP tools -- share search_chunks/find_duplicate_clusters with the REST
# routes below rather than calling them over HTTP, so there's exactly one
# implementation of each.
# The SDK rejects a token_verifier without AuthSettings (and vice versa), so
# in open mode both are omitted entirely rather than passed as no-ops.
mcp_auth_args = {}
if AUTH_ENABLED:
    mcp_auth_args = {
        "token_verifier": verifier,
        "auth": AuthSettings(
            # The SDK insists on an issuer even in static-key-only mode; pointing
            # it at ourselves is inert (no human OAuth client should connect then).
            issuer_url=OAUTH_ISSUER or PUBLIC_BASE_URL,
            resource_server_url=f"{PUBLIC_BASE_URL}/mcp",
            required_scopes=OAUTH_SCOPES or None,
        ),
    }


_public_host = urlparse(PUBLIC_BASE_URL).netloc
if not _public_host:
    raise RuntimeError(
        f"PUBLIC_BASE_URL must include a scheme (https://...), got {PUBLIC_BASE_URL!r}"
    )
mcp_transport_security = TransportSecuritySettings(
    allowed_hosts=[_public_host, "127.0.0.1:*", "localhost:*", "[::1]:*"],
    allowed_origins=[PUBLIC_BASE_URL, "http://127.0.0.1:*", "http://localhost:*"],
)

print(f"[mcp] transport security allowed hosts: {mcp_transport_security.allowed_hosts}", flush=True)
mcp = FastMCP("sourcerag", streamable_http_path="/",
              transport_security=mcp_transport_security, **mcp_auth_args)


@mcp.tool()
def search_code(query: str, top_k: int = 8, repo: str = "", language: str = "",
                path_contains: str = "", min_score: float = 0.0,
                compact: bool = False, max_chars: int = 2500, content_type: str = "") -> list[dict]:
    """Hybrid (semantic + keyword) search over indexed source code across all
    of the org's Azure DevOps repos.

    USE FOR: any question about whether code, a function, a class, a service,
    or a pattern already exists somewhere in the org -- e.g. "does X exist",
    "have we already built Y", "is there an existing implementation of Z",
    "where do we do X elsewhere", "find examples of X in our repos". Also use
    before writing new code, to check for something reusable first. Exact
    identifiers in the query (class/function names) are keyword-matched, so
    include them verbatim when you know them.

    Call this proactively whenever a question is about the org's own codebase
    rather than public/general knowledge, even if the user doesn't mention
    "search" or name this tool.

    DO NOT USE FOR: code already open in the current session/repo (use local
    file search instead), or general programming questions unrelated to this
    org's code.

    Near-identical copies of the same chunk (copy-pasted across services) are
    collapsed into one hit with a "duplicates" list of the other locations.

    Args:
        query: Natural language or code description of what you're looking for.
        top_k: Max number of results to return (default 8, capped at 50).
        repo: Optional exact repo name to filter results to.
        language: Optional file extension to filter results to (e.g. "cs", "py", "tf").
        path_contains: Optional substring the file path must contain (e.g. "tests/").
        min_score: Drop semantic-only hits scoring below this (0..1). Keyword
            matches are always kept.
        compact: Return only metadata plus a 200-char snippet per hit --
            use for broad surveys, then re-query with a filter for full text.
        max_chars: Truncate each hit's text to this many chars (sets
            "truncated": true). Default 2500 keeps responses within MCP token
            limits; pass 0 for untruncated text.
        content_type: Optional filter to "code", "wiki", or "pr" (pull
            request titles+descriptions) -- only populated if the org has
            enabled wiki/PR indexing. Omit to search across all of them.
    """
    # IndexNotReady is deliberately not caught here -- FastMCP reports the
    # raised exception's message as the tool error, which is exactly the
    # explanation the MCP client should see.
    return search_chunks(query, top_k, repo or None, language or None,
                         path_contains or None, min_score, compact, max_chars,
                         content_type or None)


@mcp.tool()
def find_duplicates(min_repos: int = 2, min_chars: int = DEFAULT_DUPLICATE_MIN_CHARS,
                    limit: int = 50, content_type: str = "code") -> list[dict]:
    """Org-wide report of near-identical code duplicated across repos, ranked
    by how many distinct repos each cluster spans.

    USE FOR: "what code should we de-dup / move into a shared package",
    "is this pattern already duplicated elsewhere", or before proposing a new
    shared library -- checking whether the functionality is already
    copy-pasted across enough repos to justify extracting it. This is a
    whole-index report, not a per-query check: it's independent of any
    specific piece of code you're looking at.

    DO NOT USE FOR: checking whether one specific snippet already exists
    elsewhere (use search_code for that) -- this only surfaces exact/
    near-exact duplicate clusters already in the index.

    Args:
        min_repos: Only include clusters spanning at least this many distinct
            repos (default 2). Duplication within a single repo is normal
            reuse-by-copy, not a package-extraction signal -- raise this to
            focus on the strongest cross-repo candidates.
        min_chars: Skip chunks shorter than this to filter out trivial/
            boilerplate matches (default 200).
        limit: Max clusters to return (default 50, capped at 200).
        content_type: Restrict to one content type (default "code"; pass
            "wiki" to check for duplicated doc pages instead).
    """
    return find_duplicate_clusters(min_repos, min_chars, limit, content_type)


@mcp.tool()
def get_file_context(project: str, repo: str, path: str, from_line: int = 0,
                     to_line: int = 0, max_chars: int = 6000) -> dict:
    """Reassemble an indexed file's text from its stored chunks, so you can see
    the code *around* a search hit -- the enclosing function, the rest of a
    class -- without re-querying and hoping the neighbouring chunk surfaces.

    USE FOR: expanding context after search_code/find_definition returns a hit
    whose chunk is cut off ("show me the whole method", "what's above/below
    this"). Pass the hit's exact project/repo/path; narrow to a window with
    from_line/to_line (e.g. a bit either side of the hit's start_line).

    DO NOT USE FOR: files in the session's own working repo (read them
    directly), or discovering which files exist (use search_code / list_repos).

    Reconstruction is approximate: chunks overlap (deduped by line) and pure-
    import chunks were dropped at index time, so `gaps` lists line ranges with
    no stored content. It is context, not a byte-exact copy from git.

    Args:
        project: Exact project name of the file (as returned by a search hit).
        repo: Exact repo name of the file.
        path: Exact file path of the file.
        from_line: Optional first line to return (1-based); 0 = from the start.
        to_line: Optional last line to return; 0 = to the end.
        max_chars: Truncate the returned text to this many chars (sets
            "truncated": true). Default 6000; pass 0 for untruncated.
    """
    return file_context(project, repo, path, from_line, to_line, max_chars)


@mcp.tool()
def list_repos() -> dict:
    """Inventory of what is actually indexed: every repo with its chunk/file
    counts and index freshness (last-indexed commit + timestamp), the languages
    present, and the total chunk count.

    USE FOR: learning valid `repo` and `language` filter values before calling
    search_code (so you can scope a search instead of guessing names), checking
    whether a given repo is indexed at all, and judging how current the index
    is (a repo last indexed weeks ago may be stale). Call this first when you're
    unsure what the org's index covers.

    Freshness may be null for repos indexed before freshness tracking existed
    and not rebuilt since.
    """
    return list_repositories()


@mcp.tool()
def find_definition(name: str, repo: str = "", language: str = "",
                    path_contains: str = "", top_k: int = 8, max_chars: int = 2500) -> list[dict]:
    """Find where a symbol is DEFINED -- the class/interface/struct/enum or the
    function/method declaration named `name` -- rather than everywhere it is
    merely mentioned.

    USE FOR: "where is X defined/declared/implemented", jumping to the source
    of a class or function by its exact name. This is the precise counterpart
    to search_code: search_code ranks any chunk that references the term;
    find_definition returns only chunks whose *declarations* include it.

    DO NOT USE FOR: conceptual/natural-language queries ("retry logic for http
    calls") -- use search_code. Definition detection is regex-based per
    language, so it covers the common declaration forms but isn't a full
    parser; if a definition isn't found this way, fall back to search_code.

    Args:
        name: The symbol name to locate the definition of (exact identifier).
        repo: Optional exact repo name to filter to.
        language: Optional file extension to filter to (e.g. "cs", "py", "ts").
        path_contains: Optional substring the file path must contain.
        top_k: Max results to return (default 8, capped at 50).
        max_chars: Truncate each hit's text to this many chars (sets
            "truncated": true). Default 2500; pass 0 for untruncated.

    Each hit carries a `symbols` field listing every name that chunk defines.
    """
    return find_definitions(name, repo or None, language or None,
                            path_contains or None, top_k, max_chars)


mcp_app = mcp.streamable_http_app()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Mounted sub-apps' lifespans aren't invoked automatically by FastAPI/Starlette --
    # the MCP session manager's own lifespan has to be entered explicitly here, or
    # the /mcp endpoint accepts connections but never actually initializes.
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(mcp_app.router.lifespan_context(mcp_app))
        yield


app = FastAPI(title="SourceRAG", lifespan=lifespan)
app.mount("/mcp", mcp_app)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/.well-known/oauth-protected-resource/mcp")
@app.get("/.well-known/oauth-protected-resource")
def oauth_protected_resource():
    # Claude Code follows the 401's resource_metadata URL, which RFC 9728 puts
    # at the ROOT (/.well-known/... inserted between host and path). The copy
    # the SDK registers lives inside the /mcp mount and is unreachable there.
    meta = {"resource": f"{PUBLIC_BASE_URL}/mcp",
            "authorization_servers": [OAUTH_ISSUER or PUBLIC_BASE_URL],
            "bearer_methods_supported": ["header"]}
    if OAUTH_SCOPES:
        meta["scopes_supported"] = OAUTH_SCOPES
    return meta


@app.get("/search", dependencies=[Depends(require_auth)])
def search(q: str, top_k: int = 8, repo: Optional[str] = None, language: Optional[str] = None,
           path_contains: Optional[str] = None, min_score: float = 0.0,
           compact: bool = False, max_chars: int = 0, content_type: Optional[str] = None):
    try:
        return search_chunks(q, top_k, repo, language, path_contains, min_score, compact, max_chars, content_type)
    except IndexNotReady as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/duplicates", dependencies=[Depends(require_auth)])
def duplicates(min_repos: int = 2, min_chars: int = DEFAULT_DUPLICATE_MIN_CHARS,
               limit: int = 50, content_type: str = "code"):
    try:
        return find_duplicate_clusters(min_repos, min_chars, limit, content_type)
    except IndexNotReady as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/file", dependencies=[Depends(require_auth)])
def file(project: str, repo: str, path: str, from_line: int = 0,
         to_line: int = 0, max_chars: int = 0):
    try:
        return file_context(project, repo, path, from_line, to_line, max_chars)
    except IndexNotReady as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/repos", dependencies=[Depends(require_auth)])
def repos():
    try:
        return list_repositories()
    except IndexNotReady as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/definitions", dependencies=[Depends(require_auth)])
def definitions(name: str, repo: Optional[str] = None, language: Optional[str] = None,
                path_contains: Optional[str] = None, top_k: int = 8, max_chars: int = 0):
    try:
        return find_definitions(name, repo, language, path_contains, top_k, max_chars)
    except IndexNotReady as e:
        raise HTTPException(status_code=503, detail=str(e))
