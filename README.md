# SourceRAG

Semantic code search over your organization's Azure DevOps repositories, running entirely locally. Repos are cloned and kept in sync, chunked language-aware, embedded with a local model via [LM Studio](https://lmstudio.ai/), and stored in [Chroma](https://www.trychroma.com/). Search is exposed as a REST endpoint and as an [MCP](https://modelcontextprotocol.io/) tool, so coding agents can check whether functionality already exists somewhere in the org before reimplementing it.

No code or queries ever leave your machine.

## Architecture

```
Azure DevOps ──git──> indexer ──embeddings via──> LM Studio (host)
                         │
                         └──chunks + vectors──> Chroma ──> api (REST /search + MCP /mcp)
```

| Service   | What it does                                                                | Port (loopback only)  |
| --------- | --------------------------------------------------------------------------- | --------------------- |
| `indexer` | Clones/pulls the configured repos, chunks + embeds, writes to Chroma. Runs once, then exits (see [Keeping the index fresh](#keeping-the-index-fresh)). | — |
| `chroma`  | Vector store, persisted in the `chroma_data` volume.                         | `127.0.0.1:8001`      |
| `api`     | FastAPI app serving `/search`, `/health`, and the MCP server at `/mcp`.      | `127.0.0.1:8080`      |

Both published ports are deliberately bound to loopback: the index contains internal source code and neither Chroma nor the API has authentication.

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

The indexer clones all repos and builds the index on first run (this can take a while for a large repo list — progress is logged per repo, `docker compose logs -f indexer`). The API is usable as soon as the first repos are indexed; before the first run has created the collection, `/search` returns `503` with an explanation.

## Searching

REST:

```sh
curl "http://localhost:8080/search?q=retry+policy+for+http+calls&top_k=5&language=cs"
```

Parameters: `q` (required), `top_k` (default 8, capped at 50), `repo` (exact repo name), `language` (file extension without dot, e.g. `cs`, `py`, `tf`).

MCP: the server lives at `http://localhost:8080/mcp/` (streamable HTTP) and exposes a single `search_code` tool. `.mcp.json` in this repo registers it for Claude Code; other MCP clients can use the same URL.

## Keeping the index fresh

The indexer is a one-shot job by design: it syncs, indexes incrementally, and exits. Schedule it externally, e.g. Windows Task Scheduler / cron:

```sh
docker compose run --rm indexer
```

Re-runs are cheap: unchanged repos are skipped by commit hash, changed repos are diffed against the last indexed commit and only changed files are re-embedded. A full reindex happens automatically when chunking-relevant config changes (chunk size/overlap, embedding model, skip lists — tracked via a fingerprint in the state file), or per repo when its stored commit no longer exists (force-push).

To force a full rebuild from scratch: `docker compose down -v` (drops both the clones and the index), then `docker compose up -d`.

## Configuration

All knobs live in `.env` (see `.env.example` for documentation of each):

| Variable | Meaning |
| --- | --- |
| `AZURE_DEVOPS_PAT` | PAT with Code (Read) scope, used via `GIT_ASKPASS` so it never appears in process args. |
| `LMSTUDIO_BASE_URL`, `EMBEDDING_MODEL` | Where and what to embed with. |
| `CHUNK_SIZE`, `CHUNK_OVERLAP`, `EMBED_BATCH_SIZE` | Chunking/embedding parameters. Changing chunking parameters triggers a full reindex. |
| `CLONE_CONCURRENCY`, `INDEX_CONCURRENCY`, `EMBED_CONCURRENCY` | Parallelism — see `.env.example` for how they differ. |

Per-repo options live in `config/repos.yaml` (`project`, `name`, optional `branch`).

## Development

```sh
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt
.venv/Scripts/python -m pytest tests
```

The tests cover the pure logic (diff parsing, chunk IDs, line mapping, skip rules, filter construction); the indexing pipeline itself is exercised against a real Chroma + LM Studio via `config/repos.smoketest.yaml`-style smoke runs.
