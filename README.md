# SourceRAG

Hybrid (semantic + keyword) code search over your organization's Azure DevOps repositories, running entirely locally. Repos are cloned and kept in sync, chunked language-aware, embedded with a local model via [LM Studio](https://lmstudio.ai/), and stored in Postgres ([pgvector](https://github.com/pgvector/pgvector) for vectors, native full-text search for keywords). Search is exposed as a REST endpoint and as an [MCP](https://modelcontextprotocol.io/) tool, so coding agents can check whether functionality already exists somewhere in the org before reimplementing it.

No code or queries ever leave your machine.

## Architecture

```
Azure DevOps ──git──> indexer ──embeddings via──> LM Studio (host)
                         │
                         └──chunks + vectors──> Postgres ──> api (REST /search + MCP /mcp)
```

| Service    | What it does                                                                                                                                                         | Port (loopback only) |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- |
| `indexer`  | Clones/pulls the configured repos, chunks + embeds, writes to Postgres. Runs once, then exits (see [Keeping the index fresh](#keeping-the-index-fresh)).             | —                    |
| `postgres` | Single store for the whole index: pgvector HNSW for the semantic leg, a `'simple'`-config tsvector GIN index for the keyword leg. Persisted in the `pg_data` volume. | `127.0.0.1:5433`     |
| `api`      | FastAPI app serving `/search`, `/health`, and the MCP server at `/mcp`.                                                                                              | `127.0.0.1:4141`     |

Both published ports are deliberately bound to loopback: the index contains internal source code and the credentials are local-only defaults.

At query time both retrieval legs run against the same table and are merged with reciprocal-rank fusion, so exact identifiers (`TestAuthenticationHandler`) hit via keywords while conceptual queries ("retry policy for http calls") hit via embeddings. Near-identical chunks (files copy-pasted across services) are collapsed into one result with a `duplicates` list at query time, or surfaced org-wide via [`/duplicates`](#finding-duplicate-code).

## Prerequisites

- Docker Desktop
- LM Studio with an embedding model loaded (default config expects `text-embedding-qwen3-embedding-0.6b`) and the local server running (Developer tab → Start Server). Containers reach it via `host.docker.internal:1234`.
- An Azure DevOps Personal Access Token with **Code (Read)** scope.

## Setup

1. Copy `.env.example` to `.env` and set `AZURE_DEVOPS_PAT`. Never commit `.env`.
2. Copy `config/repos.yaml.example` to `config/repos.yaml` and list your repos. This file is gitignored — real project/repo names reveal internal structure and must not be committed.
3. Start everything:

   ```sh
   docker compose up -d --build
   ```

The indexer clones all repos and builds the index on first run (this can take a while for a large repo list — progress is logged per repo, `docker compose logs -f indexer`). The API is usable as soon as the first repos are indexed; before the first run has created the table, `/search` returns `503` with an explanation.

## Searching

REST:

```sh
curl "http://localhost:4141/search?q=retry+policy+for+http+calls&top_k=5&language=cs"
```

Parameters:

| Parameter       | Meaning                                                                                 |
| --------------- | --------------------------------------------------------------------------------------- |
| `q`             | Required. Natural language and/or exact identifiers — both retrieval legs run on it.    |
| `top_k`         | Results to return (default 8, capped at 50).                                            |
| `repo`          | Exact repo name filter.                                                                 |
| `language`      | File extension without dot, e.g. `cs`, `py`, `tf`.                                      |
| `path_contains` | Substring the file path must contain, e.g. `tests/`.                                    |
| `min_score`     | Drop semantic-only hits with cosine similarity below this. Keyword matches always stay. |
| `compact`       | Metadata + 200-char snippet per hit instead of full chunk text.                         |
| `max_chars`     | Truncate each hit's text (`"truncated": true` marks affected hits). `0` = full text.    |
| `content_type`  | Filter to `code`, `wiki`, or `pr` (only meaningful if wiki/PR indexing is enabled — see [Configuration](#configuration)). Omit to search across all indexed content. |

Each hit reports `matched_by` (`semantic`, `lexical`, or `both`) and, where copies were collapsed, a `duplicates` list of the other locations.

MCP: the server lives at `http://localhost:4141/mcp/` (streamable HTTP) and exposes a single `search_code` tool with the same parameters (there `max_chars` defaults to 2500 to stay within MCP token limits). `.mcp.json` in this repo registers it for Claude Code; other MCP clients can use the same URL.

## Finding duplicate code

`/duplicates` surfaces clusters of near-identical chunks across the whole index (not just within one query's results), ranked by how many **distinct repos** each cluster spans rather than raw occurrence count — a chunk copy-pasted 5x within one repo is normal reuse-by-copy, the same chunk duplicated across 5 repos is a shared-package candidate.

```sh
curl "http://localhost:4141/duplicates?min_repos=2&limit=20"
```

| Parameter    | Meaning                                                                                     |
| ------------ | --------------------------------------------------------------------------------------------|
| `min_repos`  | Only include clusters spanning at least this many distinct repos (default 2).               |
| `min_chars`  | Skip chunks shorter than this — filters out trivial/boilerplate matches (default 200).       |
| `limit`      | Max clusters to return (default 50, capped at 200).                                          |
| `content_type` | Restrict to one content type (default `code` — dedup only makes sense for source by default; pass `wiki` to check for duplicated doc pages instead). |

Each cluster reports `occurrence_count`, `repo_count`, a `snippet` of the duplicated text, and a `locations` list (`project`/`repo`/`path`/`start_line`/`end_line`) for every occurrence.

Also available as the MCP tool `find_duplicates` (same parameters) — so a coding agent can check for existing de-dup candidates before proposing a new shared package, not just a developer running it manually.

## Keeping the index fresh

The indexer is a one-shot job by design: it syncs, indexes incrementally, and exits. Schedule it externally, e.g. Windows Task Scheduler / cron:

```sh
docker compose run --rm indexer
```

Re-runs are cheap: unchanged repos are skipped by commit hash, changed repos are diffed against the last indexed commit and only changed files are re-embedded. A full reindex happens automatically when chunking-relevant config changes (chunk size/overlap, embedding model, skip lists — tracked via a fingerprint in the state file), or per repo when its stored commit no longer exists (force-push).

To force a full rebuild from scratch: `docker compose down -v` (drops both the clones and the index), then `docker compose up -d`.

## What gets indexed

Binary/asset extensions, dependency lockfiles, minified bundles (`*.min.js` and a line-length heuristic for unmarked ones), source maps, and data/locale directories (`node_modules`, `fixtures`, `locales`, …) are skipped — see `SKIP_DIRS` / `SKIP_EXT` / `SKIP_FILENAMES` in `indexer/chunk_and_embed.py`. Chunks that are almost entirely import statements are dropped. Each chunk is embedded with a `// project/repo/path` header prepended so the vector carries its location context, while the stored text stays raw.

Every chunk carries a `content_type`: `code` by default, or `wiki`/`pr` when the optional indexing below is enabled — see [Configuration](#configuration) to turn these on.

- **Wikis** (`index_wikis`): Azure DevOps project wikis are themselves a git repo (`<project>.wiki`), so this reuses the normal clone/chunk/embed pipeline unchanged — wiki pages get indexed exactly like code files, just tagged `content_type: wiki`. Projects without a wiki log a harmless clone failure on sync, not an error.
- **Pull requests** (`index_pull_requests`): each PR's title + description is fetched via the Azure DevOps REST API (PRs aren't git objects) and indexed as one synthetic chunk under `_pull_requests/<id>.md`, tagged `content_type: pr`. This is incremental by PR id only — a description edited *after* its PR was first indexed will not be re-fetched on later runs, to avoid re-embedding a repo's entire PR history every time. Any linked work items are appended as bare numbers (e.g. `Linked work items: #7, #42`) — only the IDs, never the work item's own title/description/comments, since that's where personal data (reporter/customer names) would actually live.

## Configuration

All knobs live in `.env` (see `.env.example` for documentation of each):

| Variable                                                        | Meaning                                                                                                  |
| --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `AZURE_DEVOPS_PAT`                                              | PAT with Code (Read) scope, used via `GIT_ASKPASS` so it never appears in process args.                  |
| `LLM_BASE_URL`, `EMBEDDING_MODEL`                               | Where and what to embed with.                                                                            |
| `OUTPUT_CHUNK_SIZE`, `OUTPUT_CHUNK_OVERLAP`, `EMBED_BATCH_SIZE` | Chunking/embedding parameters. Changing chunking parameters triggers a full reindex.                     |
| `CLONE_CONCURRENCY`, `INDEX_CONCURRENCY`, `EMBED_CONCURRENCY`   | Parallelism — see `.env.example` for how they differ.                                                    |
| `DATABASE_URL`                                                  | Set in `docker-compose.yml` for the in-compose Postgres; override only to point at a different database. |

Repos are configured as a mapping of ADO project → repos in `config/repos.yaml`, with shorthands for the common cases (a blank value means "one repo named after the project"; see `config/repos.yaml.example` for all supported shapes, including per-repo `branch` overrides).

Two more knobs live under `azure_devops` in `config/repos.yaml`, both opt-in (default `false` — existing setups are unaffected unless you turn them on):

| Key                    | Meaning                                                                                          |
| ---------------------- | -------------------------------------------------------------------------------------------------|
| `index_wikis`          | Also clone and index each project's wiki. See [What gets indexed](#what-gets-indexed).            |
| `index_pull_requests`  | Also fetch and index each repo's PR titles+descriptions via the REST API. Same PAT, same scope.   |

Enabling either changes what a full reindex picks up (bumps the pipeline fingerprint, see [Keeping the index fresh](#keeping-the-index-fresh)) and mixes `wiki`/`pr` content into `/search` results unless you filter with `content_type`.

## Development

```sh
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt
.venv/Scripts/python -m pytest tests
```

The tests cover the pure logic (diff parsing, chunk IDs, line mapping, skip rules, content heuristics, rank fusion, dedup, PR-fetch pagination against a fake HTTP session); the indexing pipeline itself, and any live Azure DevOps wiki/PR behavior, is exercised against a real Postgres + LM Studio + ADO org via `config/repos.smoketest.yaml`-style smoke runs.
