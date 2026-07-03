import os
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Optional

import chromadb
from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

LM_BASE_URL = os.environ["LMSTUDIO_BASE_URL"]
EMBED_MODEL = os.environ["EMBEDDING_MODEL"]
CHROMA_HOST = os.environ.get("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "code_chunks")

client = OpenAI(base_url=LM_BASE_URL, api_key="lm-studio")
chroma = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


def get_collection():
    return chroma.get_collection(COLLECTION_NAME)


def search_chunks(q: str, top_k: int = 8, repo: Optional[str] = None, language: Optional[str] = None) -> list[dict]:
    # Qwen3-Embedding: documents are indexed with no prefix (chunk_and_embed.py),
    # but queries need an instruction + "Query:" prefix for best retrieval quality.
    instruct = "Instruct: Given a code search query, retrieve relevant source code snippets\nQuery:"
    vector = client.embeddings.create(model=EMBED_MODEL, input=[f"{instruct}{q}"]).data[0].embedding

    conditions = []
    if repo:
        conditions.append({"repo": {"$eq": repo}})
    if language:
        conditions.append({"language": {"$eq": language}})
    where = None
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    result = get_collection().query(query_embeddings=[vector], n_results=top_k, where=where)

    hits = []
    for doc, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        hits.append({
            "score": 1 - distance,
            "project": meta["project"],
            "repo": meta["repo"],
            "path": meta["path"],
            "start_line": meta["start_line"],
            "end_line": meta["end_line"],
            "file_size_bytes": meta.get("file_size_bytes"),
            "chunk_tokens_estimate": meta.get("chunk_tokens_estimate"),
            "commit_hash": meta.get("commit_hash"),
            "commit_date": meta.get("commit_date"),
            "text": doc,
        })
    return hits


# MCP tool -- shares search_chunks with the /search REST route below rather
# than calling it over HTTP, so there's exactly one implementation of the
# actual search logic.
mcp = FastMCP("sourcerag", streamable_http_path="/")


@mcp.tool()
def search_code(query: str, top_k: int = 8, repo: str = "", language: str = "") -> list[dict]:
    """Semantically search indexed code across all configured Azure DevOps repos.

    Use this before implementing something new to check whether equivalent
    functionality already exists somewhere in the org's repos.

    Args:
        query: Natural language or code description of what you're looking for.
        top_k: Max number of results to return (default 8).
        repo: Optional exact repo name to filter results to.
        language: Optional file extension to filter results to (e.g. "cs", "py", "tf").
    """
    return search_chunks(query, top_k, repo or None, language or None)


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


@app.get("/search")
def search(q: str, top_k: int = 8, repo: Optional[str] = None, language: Optional[str] = None):
    return search_chunks(q, top_k, repo, language)
