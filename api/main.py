import os
from typing import Optional

import chromadb
from fastapi import FastAPI
from openai import OpenAI

LM_BASE_URL = os.environ["LMSTUDIO_BASE_URL"]
EMBED_MODEL = os.environ["EMBEDDING_MODEL"]
CHROMA_HOST = os.environ.get("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "code_chunks")

app = FastAPI(title="SourceRAG")
client = OpenAI(base_url=LM_BASE_URL, api_key="lm-studio")
chroma = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


def get_collection():
    return chroma.get_collection(COLLECTION_NAME)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/search")
def search(q: str, top_k: int = 8, repo: Optional[str] = None, language: Optional[str] = None):
    vector = client.embeddings.create(model=EMBED_MODEL, input=[q]).data[0].embedding

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

    result = get_collection().query(
        query_embeddings=[vector],
        n_results=top_k,
        where=where,
    )

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
