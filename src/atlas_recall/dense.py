"""
Optional dense (semantic) retrieval: Chroma + a local Ollama embedding
model (bge-m3 by default). Nothing in this module is imported at package
load time -- `recall.py`/`retrieval.py` reach into it lazily and catch
ImportError/ConnectionError, so a machine with no `chromadb` installed and
no Ollama running never sees this file executed at all.

Opt in with `recall index --dense` (requires `pip install atlas-recall[dense]`
and a running `ollama serve` with the embed model pulled).
"""
from __future__ import annotations

import os
import socket
from typing import Dict, List

from .config import Config, resolve_chroma_dir, resolve_config_path
from .corpus import load_corpus


def ollama_reachable(host: str = "127.0.0.1", port: int = 11434, timeout: float = 0.2) -> bool:
    """Fast reachability probe -- a plain TCP connect, not a full request.
    Lets callers short-circuit instead of paying a slow connection-refused
    cost on every query when Ollama simply isn't running."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _embedding_function(cfg: Config):
    from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
    return OllamaEmbeddingFunction(
        url="http://127.0.0.1:11434/api/embeddings",
        model_name=cfg.embed_model,
    )


def _collection(cfg: Config, delete_existing: bool = False, config_path: str = None):
    import chromadb
    chroma_dir = resolve_chroma_dir(cfg, config_path or resolve_config_path())
    os.makedirs(chroma_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_dir)
    ef = _embedding_function(cfg)
    if delete_existing:
        try:
            client.delete_collection(cfg.collection)
        except Exception:
            pass
    return client.get_or_create_collection(
        cfg.collection,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def index_dense(cfg: Config, config_path: str = None) -> int:
    """(Re)index cfg.notes_dir into the configured Chroma collection.
    Returns the number of documents indexed. Raises if chromadb isn't
    installed or Ollama isn't reachable -- callers surface that plainly,
    this function does not degrade silently (indexing is an explicit,
    opt-in action, unlike query-time retrieval)."""
    docs = load_corpus(cfg.notes_dir)
    col = _collection(cfg, delete_existing=True, config_path=config_path)
    if not docs:
        return 0
    ids = [d["key"] for d in docs]
    texts = [f"{d['name']}\n{d['description']}\n\n{d['body']}" for d in docs]
    metas = [
        {"name": d["name"], "description": d["description"], "path": d["path"]}
        for d in docs
    ]
    col.upsert(ids=ids, documents=texts, metadatas=metas)
    return len(ids)


def query_dense(query: str, cfg: Config, topn: int = 15, config_path: str = None) -> List[Dict]:
    """Return ranked hits from the dense collection: [{name, path, distance}].
    Raises on any failure (chromadb absent, Ollama down, empty/missing
    collection) -- callers (retrieval.py) catch this and fuse with BM25
    only, which is the documented degrade path."""
    col = _collection(cfg, config_path=config_path)
    if col.count() == 0:
        raise RuntimeError(f"empty dense index (collection={cfg.collection!r}) -- run `recall index --dense` first")
    res = col.query(query_texts=[query], n_results=topn)
    return [
        {"name": m["name"], "path": m.get("path", ""), "distance": dist}
        for m, dist in zip(res["metadatas"][0], res["distances"][0])
    ]
