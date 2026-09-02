"""
Knowledge graph over the same notes corpus retrieval.py reads: a SQLite
FTS5 full-text index plus a wikilink/frontmatter edge graph, so a human can
search and trace links by hand (`recall find`, `recall node`, `recall
trace`, `recall map`, `recall verify`) the exact corpus the automatic
`recall query` path retrieves from. One directory, one config, one index --
retrieval and search are two interfaces on the same engine, not two
products bolted together.

This package has no concept of "project" -- it indexes ONE directory you
point it at, nothing more. What it does: frontmatter parsing, wikilink
edge derivation, FTS5 search, recursive trace, and divergence tracking for
stale links.

Known limitation (see this package's README, "Known limitation" section):
`index` (incremental) only re-derives edges for docs it re-parses this
pass. An edge whose TARGET doc was deleted or renamed, but whose SOURCE
doc was untouched this pass, is never re-derived, so it never gets a
chance to self-heal -- it sits flagged `divergence=1` forever, and the
only thing that clears it is a full `rebuild`.
"""
from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

WIKILINK_RE = re.compile(r"\[\[([A-Za-z0-9_.\-]+)\]\]")
HEADING_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")

# Wikilink slugs that are placeholder/syntax-example prose (e.g. "see
# [[name]] for how to link"), never real targets. Deliberately conservative:
# only exact, unambiguous placeholder tokens, so a real doc happening to be
# named e.g. "reference-slug" is never dropped by accident.
WIKILINK_PLACEHOLDER_DENYLIST = {"name", "link", "links", "their-name", "the-name", "slug"}

DDL = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS docs (
  id          TEXT PRIMARY KEY,
  type        TEXT NOT NULL,
  title       TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  path        TEXT NOT NULL,
  mtime       INTEGER NOT NULL,
  hash        TEXT NOT NULL,
  body        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_docs_type ON docs(type);

CREATE TABLE IF NOT EXISTS edges (
  src           TEXT NOT NULL,
  dst           TEXT NOT NULL,
  kind          TEXT NOT NULL,
  last_verified INTEGER NOT NULL,
  divergence    INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (src, dst, kind)
);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
CREATE INDEX IF NOT EXISTS idx_edges_verify ON edges(divergence DESC, last_verified ASC);

CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
  title, description, body,
  content = 'docs', content_rowid = 'rowid',
  tokenize = 'porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON docs BEGIN
  INSERT INTO docs_fts(rowid, title, description, body)
  VALUES (new.rowid, new.title, new.description, new.body);
END;
CREATE TRIGGER IF NOT EXISTS docs_ad AFTER DELETE ON docs BEGIN
  INSERT INTO docs_fts(docs_fts, rowid, title, description, body)
  VALUES('delete', old.rowid, old.title, old.description, old.body);
END;
CREATE TRIGGER IF NOT EXISTS docs_au AFTER UPDATE ON docs BEGIN
  INSERT INTO docs_fts(docs_fts, rowid, title, description, body)
  VALUES('delete', old.rowid, old.title, old.description, old.body);
  INSERT INTO docs_fts(rowid, title, description, body)
  VALUES (new.rowid, new.title, new.description, new.body);
END;
"""


def extract_wikilinks(body: str):
    stripped = FENCED_CODE_RE.sub(" ", body)
    stripped = INLINE_CODE_RE.sub(" ", stripped)
    return {
        slug for slug in WIKILINK_RE.findall(stripped)
        if slug.lower() not in WIKILINK_PLACEHOLDER_DENYLIST
    }


def _retry_eagain(fn, retries=6, base_delay=0.05):
    """A process reading a cloud-synced-drive file (iCloud, Dropbox, etc.)
    can get a transient EAGAIN mid-read. Retry a small bounded number of
    times with backoff; anything else (e.g. ENOENT) raises immediately."""
    last_err = None
    for attempt in range(retries):
        try:
            return fn()
        except OSError as e:
            if e.errno == errno.EAGAIN:
                last_err = e
                time.sleep(base_delay * (2 ** attempt))
                continue
            raise
    raise OSError(errno.EAGAIN, "persistent EAGAIN") from last_err


def _read_text(path: Path) -> str:
    return _retry_eagain(lambda: path.read_text(encoding="utf-8", errors="replace"))


def _unquote(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
        return v[1:-1].replace('\\"', '"')
    if len(v) >= 2 and v[0] == "'" and v[-1] == "'":
        return v[1:-1]
    return v


def parse_frontmatter(text: str):
    """Minimal frontmatter parser -- stdlib only. Handles a `---` fenced
    block of flat `key: value` lines plus one level of 2-space-indented
    nesting. Returns (meta_dict, body_text)."""
    if not text.startswith("---"):
        return {}, text
    lines = text.split("\n")
    if lines[0].strip() != "---":
        return {}, text
    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return {}, text
    fm_lines = lines[1:end_idx]
    body = "\n".join(lines[end_idx + 1:])
    meta: Dict = {}
    nested_key = None
    for line in fm_lines:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            continue
        if indent == 0:
            key, _, val = stripped.partition(":")
            key, val = key.strip(), val.strip()
            if val == "":
                nested_key = key
                meta[key] = {}
            else:
                nested_key = None
                meta[key] = _unquote(val)
        elif nested_key is not None:
            nk, _, nv = stripped.partition(":")
            if not isinstance(meta.get(nested_key), dict):
                meta[nested_key] = {}
            meta[nested_key][nk.strip()] = _unquote(nv.strip())
    return meta, body


def infer_type(meta: Dict, filename_stem: str) -> str:
    m = meta.get("metadata")
    if isinstance(m, dict) and m.get("type"):
        return m["type"]
    if isinstance(meta.get("type"), str) and meta["type"]:
        return meta["type"]
    for prefix, t in (("feedback-", "feedback"), ("reference-", "reference"),
                      ("project-", "project"), ("idea-", "idea")):
        if filename_stem.startswith(prefix):
            return t
    return "doc"


def infer_title(meta: Dict, body: str, filename_stem: str) -> str:
    if isinstance(meta.get("name"), str) and meta["name"]:
        return meta["name"]
    m = HEADING_RE.search(body)
    if m:
        return m.group(1).strip()
    return filename_stem


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _enumerate_files(notes_dir: Path) -> List[Path]:
    if not notes_dir.is_dir():
        return []
    out = []
    for f in sorted(notes_dir.rglob("*.md")):
        if "_archive" in f.parts or ".snapshots" in f.parts:
            continue
        out.append(f)
    return out


def parse_doc(path: Path) -> Dict:
    raw = _read_text(path)
    meta, body = parse_frontmatter(raw)
    stem = path.stem
    doc_id = meta.get("name") if isinstance(meta.get("name"), str) and meta.get("name") else stem
    doc_type = infer_type(meta, stem) if meta else "doc"
    title = infer_title(meta, body, stem)
    description = meta.get("description", "") if isinstance(meta.get("description"), str) else ""
    # The normal incremental path must distinguish edits made within one
    # second; second-resolution mtimes silently skipped those changes.
    mtime = path.stat().st_mtime_ns
    content_hash = sha256_of(raw)
    return {
        "id": doc_id, "type": doc_type, "title": title, "description": description,
        "path": str(path), "mtime": mtime, "hash": content_hash, "body": body,
        "meta": meta,
    }


# --------------------------------------------------------------------------
# Writer lock -- guards index/rebuild/verify (the paths that write the DB).
# Reads (find/node/trace/map) never take it.
# --------------------------------------------------------------------------
class WriterLockBusy(Exception):
    pass


def _lock_path(db_path: Path) -> Path:
    return Path(str(db_path) + ".lock")


def acquire_writer_lock(db_path: Path, timeout: float = 60.0, poll: float = 0.5):
    lock_path = _lock_path(db_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(str(lock_path), "w")
    deadline = time.time() + timeout
    while True:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fh
        except OSError:
            if time.time() >= deadline:
                fh.close()
                raise WriterLockBusy()
            time.sleep(poll)


def release_writer_lock(fh) -> None:
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


# --------------------------------------------------------------------------
# DB
# --------------------------------------------------------------------------
def connect(db_path: Path, fresh: bool = False) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if fresh and db_path.exists():
        db_path.unlink()
        for ext in ("-wal", "-shm"):
            p = Path(str(db_path) + ext)
            if p.exists():
                p.unlink()
    conn = sqlite3.connect(str(db_path))
    conn.executescript(DDL)
    conn.commit()
    return conn


def all_doc_ids(conn) -> set:
    return {r[0] for r in conn.execute("SELECT id FROM docs")}


def upsert_doc(conn, d: Dict) -> None:
    existing = conn.execute("SELECT id FROM docs WHERE id = ?", (d["id"],)).fetchone()
    if existing:
        conn.execute(
            "UPDATE docs SET type=?, title=?, description=?, path=?, mtime=?, hash=?, body=? WHERE id=?",
            (d["type"], d["title"], d["description"], d["path"], d["mtime"], d["hash"], d["body"], d["id"]),
        )
    else:
        conn.execute(
            "INSERT INTO docs(id, type, title, description, path, mtime, hash, body) VALUES(?,?,?,?,?,?,?,?)",
            (d["id"], d["type"], d["title"], d["description"], d["path"], d["mtime"], d["hash"], d["body"]),
        )


def derive_edges_for_doc(d: Dict, ids: set) -> List[Tuple[str, str, str]]:
    edges = []
    for slug in extract_wikilinks(d["body"]):
        edges.append((d["id"], slug, "link"))
    meta = d.get("meta") or {}
    if isinstance(meta.get("derived_from"), str) and meta["derived_from"]:
        edges.append((d["id"], meta["derived_from"], "derived-from"))
    if isinstance(meta.get("supersedes"), str) and meta["supersedes"]:
        edges.append((d["id"], meta["supersedes"], "supersedes"))
    return edges


def verify_target(dst: str, known_ids: set) -> bool:
    return dst in known_ids


def upsert_edge(conn, src, dst, kind, known_ids, now) -> int:
    ok = verify_target(dst, known_ids)
    row = conn.execute(
        "SELECT last_verified FROM edges WHERE src=? AND dst=? AND kind=?", (src, dst, kind)
    ).fetchone()
    last_verified = now if ok else (row[0] if row else 0)
    divergence = 0 if ok else 1
    if row:
        conn.execute(
            "UPDATE edges SET last_verified=?, divergence=? WHERE src=? AND dst=? AND kind=?",
            (last_verified, divergence, src, dst, kind),
        )
    else:
        conn.execute(
            "INSERT INTO edges(src, dst, kind, last_verified, divergence) VALUES(?,?,?,?,?)",
            (src, dst, kind, last_verified, divergence),
        )
    return divergence


def run_index_pass(conn, notes_dir: Path, full: bool = True) -> Dict:
    """Shared body of `recall index`/`recall index --full`/`recall rebuild`.
    full=True treats every discovered doc as changed."""
    discovered = _enumerate_files(notes_dir)
    existing_paths = {r[0]: r for r in conn.execute("SELECT path, id, mtime, hash FROM docs")}
    discovered_paths = {str(f) for f in discovered}

    removed = 0
    for path_str, (_, doc_id, _, _) in list(existing_paths.items()):
        if path_str not in discovered_paths:
            conn.execute("DELETE FROM docs WHERE id = ?", (doc_id,))
            conn.execute("DELETE FROM edges WHERE src = ?", (doc_id,))
            removed += 1

    changed_docs = []
    unchanged = 0
    for path in discovered:
        path_str = str(path)
        prior = existing_paths.get(path_str)
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            continue
        if (not full) and prior is not None and prior[2] == mtime:
            unchanged += 1
            continue
        d = parse_doc(path)
        # A frontmatter name is the graph ID. If it changed for an existing
        # path, discard the former ID instead of leaving a stale, searchable
        # duplicate row behind.
        if prior is not None and prior[1] != d["id"]:
            conn.execute("DELETE FROM docs WHERE id = ?", (prior[1],))
            conn.execute("DELETE FROM edges WHERE src = ?", (prior[1],))

        # Two files claiming the same graph ID cannot be represented without
        # silently replacing one another. Fail the index pass loudly instead.
        owner = conn.execute("SELECT path FROM docs WHERE id = ?", (d["id"],)).fetchone()
        if owner is not None and owner[0] != path_str:
            raise ValueError(
                f"duplicate note id {d['id']!r}: {owner[0]!r} and {path_str!r}"
            )
        if (not full) and prior is not None and prior[3] == d["hash"]:
            conn.execute("UPDATE docs SET mtime=? WHERE id=?", (d["mtime"], d["id"]))
            unchanged += 1
            continue
        upsert_doc(conn, d)
        changed_docs.append(d)
    conn.commit()

    known_ids = all_doc_ids(conn)
    now = int(time.time())
    verified = 0
    divergent = 0
    for d in changed_docs:
        # NOTE: this only re-derives edges for docs reparsed THIS pass. An
        # edge whose target was removed/renamed, sourced from a doc that
        # was NOT touched this pass, never gets a chance to self-heal here
        # -- see the module docstring's "known carried-over bug" note.
        conn.execute("DELETE FROM edges WHERE src = ?", (d["id"],))
        for src, dst, kind in derive_edges_for_doc(d, known_ids):
            div = upsert_edge(conn, src, dst, kind, known_ids, now)
            verified += 1
            divergent += div
    conn.commit()

    return {
        "discovered": len(discovered), "removed": removed, "changed": len(changed_docs),
        "unchanged": unchanged, "edges_verified": verified, "edges_divergent": divergent,
    }


# --------------------------------------------------------------------------
# Query-side commands (no writer lock -- read the live DB only)
# --------------------------------------------------------------------------
def find(conn, query: str, doc_type: Optional[str] = None, n: int = 8) -> List[Dict]:
    sql = ("SELECT d.id, d.type, d.title, d.description, d.path "
           "FROM docs_fts JOIN docs d ON d.rowid = docs_fts.rowid "
           "WHERE docs_fts MATCH ?")
    params: List = [query]
    if doc_type:
        sql += " AND d.type = ?"
        params.append(doc_type)
    sql += " ORDER BY bm25(docs_fts, 8.0, 4.0, 1.0), d.mtime DESC LIMIT ?"
    params.append(n)
    rows = conn.execute(sql, params).fetchall()
    return [{"id": r[0], "type": r[1], "title": r[2], "description": r[3], "path": r[4]} for r in rows]


def node(conn, doc_id: str) -> Optional[Dict]:
    row = conn.execute(
        "SELECT id, type, title, description, path, mtime, body FROM docs WHERE id = ?", (doc_id,)
    ).fetchone()
    if not row:
        return None
    known_ids = all_doc_ids(conn)
    now = int(time.time())
    edge_rows = conn.execute("SELECT dst, kind FROM edges WHERE src = ?", (doc_id,)).fetchall()
    edges = []
    for dst, kind in edge_rows:
        div = upsert_edge(conn, doc_id, dst, kind, known_ids, now)
        conn.commit()
        t = conn.execute("SELECT title FROM docs WHERE id=?", (dst,)).fetchone()
        edges.append({"dst": dst, "kind": kind, "target_title": t[0] if t else dst, "divergent": bool(div)})
    return {
        "id": row[0], "type": row[1], "title": row[2], "description": row[3],
        "path": row[4], "mtime": row[5], "body": row[6], "edges": edges,
    }


def trace(conn, doc_id: str, depth: int = 2, kind: Optional[str] = None) -> List[Dict]:
    depth = min(depth, 3)
    base_kind = "AND edges.kind = ?" if kind else ""
    recurse_kind = "AND e.kind = ?" if kind else ""
    sql = f"""
      WITH RECURSIVE tree(src, dst, kind, depth) AS (
        SELECT src, dst, kind, 1 FROM edges WHERE src = ? {base_kind}
        UNION ALL
        SELECT e.src, e.dst, e.kind, tree.depth + 1
        FROM edges e JOIN tree ON e.src = tree.dst
        WHERE tree.depth < ? {recurse_kind}
      )
      SELECT src, dst, kind, depth FROM tree LIMIT 100
    """
    params = [doc_id, kind, depth, kind] if kind else [doc_id, depth]
    rows = conn.execute(sql, params).fetchall()
    known_ids = all_doc_ids(conn)
    titles = {r[0]: r[1] for r in conn.execute("SELECT id, title FROM docs")}
    out = []
    for src, dst, k, d in rows:
        div = 0 if verify_target(dst, known_ids) else 1
        out.append({
            "src": src, "dst": dst, "kind": k, "depth": d,
            "target_title": titles.get(dst, dst), "divergent": bool(div),
        })
    return out


def verify(conn, limit: int = 50) -> Dict:
    known_ids = all_doc_ids(conn)
    now = int(time.time())
    rows = conn.execute(
        "SELECT src, dst, kind FROM edges ORDER BY divergence DESC, last_verified ASC LIMIT ?",
        (limit,),
    ).fetchall()
    divergent = 0
    divergent_edges = []
    for src, dst, kind in rows:
        div = upsert_edge(conn, src, dst, kind, known_ids, now)
        divergent += div
        if div:
            divergent_edges.append({"src": src, "dst": dst, "kind": kind})
    conn.commit()
    return {"checked": len(rows), "divergent": divergent, "divergent_edges": divergent_edges}


def map_topic(conn, topic: str, n: int = 5) -> Dict:
    """The shape of a topic: FTS5 hits plus each hit's 1-hop wikilink
    neighbors, so a human sees not just "which docs match" but "what they
    connect to". A single-directory corpus has no project concept to
    materialize a card for, so this synthesizes the view at read time
    instead of point-reading a precomputed one."""
    hits = find(conn, topic, n=n)
    known_ids = all_doc_ids(conn)
    now = int(time.time())
    for h in hits:
        edge_rows = conn.execute("SELECT dst, kind FROM edges WHERE src = ?", (h["id"],)).fetchall()
        neighbors = []
        for dst, kind in edge_rows:
            div = upsert_edge(conn, h["id"], dst, kind, known_ids, now)
            t = conn.execute("SELECT title FROM docs WHERE id=?", (dst,)).fetchone()
            neighbors.append({"dst": dst, "kind": kind, "target_title": t[0] if t else dst, "divergent": bool(div)})
        h["neighbors"] = neighbors
    conn.commit()
    return {"topic": topic, "hits": hits}
