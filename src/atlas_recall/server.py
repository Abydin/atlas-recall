"""
MCP server surface for atlas-recall: the same retrieval + knowledge-graph
engine cli.py drives, exposed as MCP tools over stdio so any MCP-capable
client (Claude Desktop, Cursor, Windsurf, Cline, Zed, Codex CLI) can call
it natively -- no shelling out to `recall`, no copy-pasted hook JSON.

Local stdio transport only. No SSE, no HTTP server, no host/port binding
-- exposing a user's private notes over a network endpoint is a
deployment decision for the person running this, not something this
package does on its own. See the README for exactly which clients that
covers (all of them take a local stdio server) and which it does not
(ChatGPT's connectors need a remote HTTP endpoint; that's out of scope
here).

Every tool returns structured data (dicts/lists of dicts), never
pre-formatted prose -- the calling model should act on fields, not
re-parse English. Tools never raise past this file: a missing notes_dir,
a bad FTS5 query, or a not-yet-built index all come back as
`{"error": ...}` rather than an MCP-level exception, because a raised
exception is a worse experience for the calling model than a field it can
check.

Requires the `mcp` extra (`pip install atlas-recall[mcp]`) and Python
3.10+, both of which are the SDK's requirements, not this package's --
the rest of atlas-recall stays on the 3.9 floor. See main() below for how
that's enforced without making import errors atlas-recall's problem.
"""
from __future__ import annotations

from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from . import knowledge
from .cli import _db_path
from .config import load_config
from .retrieval import top_pointers

mcp = FastMCP("atlas-recall")


def _cfg_or_error():
    cfg = load_config()
    if not cfg.notes_dir:
        return None, {"error": "no notes_dir configured -- run `recall init <dir>` first"}
    return cfg, None


def _conn_or_error():
    cfg, err = _cfg_or_error()
    if err:
        return None, err
    try:
        return knowledge.connect(_db_path(cfg), notes_dir=cfg.notes_dir), None
    except knowledge.CorpusMismatch as e:
        return None, {"error": str(e)}


@mcp.tool()
def recall_query(text: str, top_k: Optional[int] = None) -> dict[str, Any]:
    """Retrieve the notes most relevant to `text` -- the same hybrid
    BM25+dense, RRF-fused, recency-decayed ranking the Claude Code hook
    injects automatically. Returns structured pointer records (id, path,
    title, body, score, why) rather than a formatted block; `top_k`
    overrides the configured default count for this call only."""
    cfg, err = _cfg_or_error()
    if err:
        return err
    if top_k is not None:
        cfg.top_k = top_k
    pointers = top_pointers(text, cfg)
    return {"pointers": pointers}


@mcp.tool()
def recall_search(query: str, doc_type: Optional[str] = None, n: int = 8) -> dict[str, Any]:
    """Full-text (BM25/FTS5) search over the indexed corpus -- the human
    `recall find` path. `query` uses SQLite FTS5 match syntax. `doc_type`
    optionally restricts to one frontmatter `type`."""
    conn, err = _conn_or_error()
    if err:
        return err
    try:
        rows = knowledge.find(conn, query, doc_type=doc_type, n=n)
    except Exception as e:  # noqa: BLE001
        return {"error": f"search query error: {e}"}
    finally:
        conn.close()
    return {"results": rows}


@mcp.tool()
def recall_list_notes(doc_type: Optional[str] = None, limit: int = 100) -> dict[str, Any]:
    """List indexed notes, newest-modified first, optionally filtered by
    frontmatter `type`. Capped at `limit` (default 100)."""
    conn, err = _conn_or_error()
    if err:
        return err
    try:
        rows = knowledge.list_docs(conn, doc_type=doc_type, limit=limit)
    finally:
        conn.close()
    return {"results": rows}


@mcp.tool()
def recall_node(doc_id: str, include_edges: bool = True) -> dict[str, Any]:
    """Get one note by its id (the frontmatter `name`, or filename stem)
    -- full body, plus its 1-hop wikilink/frontmatter edges unless
    `include_edges` is False."""
    conn, err = _conn_or_error()
    if err:
        return err
    try:
        n = knowledge.node(conn, doc_id)
        if n is None:
            return {"error": f"not found: {doc_id}"}
        conn.commit()
        if not include_edges:
            n = {k: v for k, v in n.items() if k != "edges"}
        return n
    finally:
        conn.close()


@mcp.tool()
def recall_trace(doc_id: str, depth: int = 2, kind: Optional[str] = None) -> dict[str, Any]:
    """Follow the wikilink/frontmatter edge graph outward from `doc_id`
    up to `depth` hops (max 3), optionally restricted to one edge `kind`.
    Flags edges whose target no longer exists as divergent."""
    conn, err = _conn_or_error()
    if err:
        return err
    try:
        root = conn.execute("SELECT id FROM docs WHERE id = ?", (doc_id,)).fetchone()
        if not root:
            return {"error": f"not found: {doc_id}"}
        rows = knowledge.trace(conn, doc_id, depth=depth, kind=kind)
    finally:
        conn.close()
    return {"root": doc_id, "edges": rows}


@mcp.tool()
def recall_map(topic: str, n: int = 5) -> dict[str, Any]:
    """The shape of a topic: FTS5 search hits for `topic` plus each hit's
    1-hop wikilink neighbors, so a caller sees what the matching notes
    connect to, not just that they matched."""
    conn, err = _conn_or_error()
    if err:
        return err
    try:
        result = knowledge.map_topic(conn, topic, n=n)
    finally:
        conn.close()
    if not result["hits"]:
        return {"topic": topic, "hits": [], "error": f"no notes matched {topic!r}"}
    return result


@mcp.tool()
def recall_verify(limit: int = 50) -> dict[str, Any]:
    """Check up to `limit` wikilink/frontmatter edges for divergence
    (target note deleted or renamed) and report which ones are broken."""
    cfg, err = _cfg_or_error()
    if err:
        return err
    db_path = _db_path(cfg)
    try:
        lock = knowledge.acquire_writer_lock(db_path)
    except knowledge.WriterLockBusy:
        return {"error": "another indexer is running, try again shortly"}
    try:
        conn = knowledge.connect(db_path, notes_dir=cfg.notes_dir)
        result = knowledge.verify(conn, limit=limit)
        conn.close()
    except knowledge.CorpusMismatch as e:
        return {"error": str(e)}
    finally:
        knowledge.release_writer_lock(lock)
    return result


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
