"""
The `recall` console script. One binary, one config, one index over a
directory of markdown notes: BM25 retrieval (always on) and dense
retrieval (opt-in) feed the automatic `query`/`hook` path; the same corpus
is also indexed into a SQLite FTS5 + wikilink graph for the human `find`/
`node`/`trace`/`map`/`verify` path. Two interfaces, one engine.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

from . import __version__
from .config import (
    Config,
    load_config,
    save_config,
    DEFAULT_CONFIG_PATH,
    resolve_config_path,
    default_corpus_dir,
)

HOOK_BLOCK = """{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "recall hook"
          }
        ]
      }
    ]
  }
}"""


def _legacy_meta_notes_dir(legacy_path: Path):
    """Read notes_dir out of a legacy db's meta table without applying our
    DDL or touching the file -- we may not own it. Returns the stamped
    notes_dir, or None if the db predates corpus tracking (no meta table
    or no row), or None on any read error (treated the same as "can't
    verify")."""
    try:
        conn = sqlite3.connect(f"file:{legacy_path}?mode=ro", uri=True)
        try:
            has_meta = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'meta'"
            ).fetchone()
            if not has_meta:
                return None
            row = conn.execute("SELECT value FROM meta WHERE key = 'notes_dir'").fetchone()
            return row[0] if row else None
        finally:
            conn.close()
    except sqlite3.Error:
        return None


def _migrate_legacy_db(legacy_path: Path, new_path: Path, notes_dir: str) -> bool:
    """One-time migration off the old shared index path (every notes_dir,
    every config, one db at chroma_dir's parent -- the bug this module
    fixes). Claims the legacy db by moving it to `new_path` ONLY when it can
    be verified to belong to `notes_dir`; otherwise leaves it exactly where
    it was -- it may belong to a different corpus still reading it under
    old code, or it may predate corpus tracking entirely and be
    unverifiable. Never deletes anything. Returns True if it migrated."""
    from . import knowledge

    target = knowledge._normalize_notes_dir(notes_dir)
    stamped = _legacy_meta_notes_dir(legacy_path)

    if stamped is None:
        print(
            f"index: found an existing index at {legacy_path} with no "
            f"recorded notes_dir, so it can't be verified against "
            f"{notes_dir!r}. Leaving it in place -- run `recall index` to "
            f"build a fresh index for this notes_dir. If {legacy_path} was "
            f"in fact built from this notes_dir, delete it afterward to "
            f"reclaim the disk space once the new index looks right; "
            f"otherwise it's safe to leave alone.",
            file=sys.stderr,
        )
        return False

    if knowledge._normalize_notes_dir(stamped) != target:
        # Belongs to a different corpus. Not an error, not our problem --
        # just don't touch it.
        return False

    new_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        src = Path(str(legacy_path) + suffix)
        if src.exists():
            src.rename(str(new_path) + suffix)
    lock_src = Path(str(legacy_path) + ".lock")
    if lock_src.exists():
        try:
            lock_src.unlink()
        except OSError:
            pass
    print(f"index: migrated existing index from {legacy_path} to {new_path}", file=sys.stderr)
    return True


def _db_path(cfg: Config, config_path: str = None) -> Path:
    """Per-corpus index db path -- see config.default_corpus_dir(). Falls
    back to migrating an index found at the old shared path (chroma_dir's
    parent, the same for every notes_dir/config) the first time this corpus
    is asked for, so upgrading users don't silently lose an index."""
    config_path = config_path or resolve_config_path()
    new_path = Path(default_corpus_dir(config_path, cfg.notes_dir)) / "recall.db"
    if new_path.exists():
        return new_path
    legacy_path = Path(os.path.expanduser(cfg.chroma_dir)).parent / "recall.db"
    if legacy_path.exists() and legacy_path != new_path:
        _migrate_legacy_db(legacy_path, new_path, cfg.notes_dir)
    return new_path


def cmd_init(args) -> int:
    notes_dir = os.path.expanduser(args.dir)
    os.makedirs(notes_dir, exist_ok=True)
    cfg = load_config()
    cfg.notes_dir = notes_dir
    path = save_config(cfg)
    print(f"Wrote config to {path}")
    print(f"notes_dir = {notes_dir}\n")
    print("Next: recall install --client <claude-code|claude-desktop|cursor|windsurf|cline|codex>")
    print("      wires this into your AI client's config directly (merges, backs up, idempotent;")
    print("      pass --dry-run to preview first).\n")
    print("Then: recall index    (builds the search index)")
    print("      recall query \"...\"   (see what would be injected)\n")
    print("Prefer to do it by hand? Here's the raw Claude Code hook block")
    print("(paste into settings.json, merging the \"hooks\" key if one exists):\n")
    print(HOOK_BLOCK)
    return 0


def cmd_install(args) -> int:
    from .install import install

    result = install(args.client, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))
    if not result.get("success"):
        return 1
    if args.dry_run:
        print("(dry run -- nothing written)", file=sys.stderr)
    elif not result.get("changed"):
        print(f"{args.client}: already installed, nothing to do", file=sys.stderr)
    return 0


def cmd_uninstall(args) -> int:
    from .install import uninstall

    result = uninstall(args.client, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))
    if not result.get("success"):
        return 1
    if args.dry_run:
        print("(dry run -- nothing written)", file=sys.stderr)
    elif not result.get("changed"):
        print(f"{args.client}: not installed, nothing to do", file=sys.stderr)
    return 0


def cmd_index(args) -> int:
    cfg = load_config()
    if not cfg.notes_dir:
        print("No notes_dir configured -- run `recall init <dir>` first.", file=sys.stderr)
        return 1
    from . import knowledge

    db_path = _db_path(cfg)
    try:
        lock = knowledge.acquire_writer_lock(db_path)
    except knowledge.WriterLockBusy:
        print("index: another indexer is running, skipping", file=sys.stderr)
        return 0
    try:
        if args.rebuild:
            conn = knowledge.connect(db_path, fresh=True, notes_dir=cfg.notes_dir)
        else:
            conn = knowledge.connect(db_path, fresh=False, notes_dir=cfg.notes_dir)
        stats = knowledge.run_index_pass(conn, Path(cfg.notes_dir), full=args.rebuild or args.full)
        conn.close()
    finally:
        knowledge.release_writer_lock(lock)
    print(
        "index: discovered=%(discovered)d changed=%(changed)d unchanged=%(unchanged)d "
        "removed=%(removed)d edges_verified=%(edges_verified)d "
        "edges_divergent=%(edges_divergent)d" % stats
    )

    if args.dense:
        from . import dense as dense_mod
        try:
            n = dense_mod.index_dense(cfg)
        except ImportError:
            print(
                "index --dense: chromadb not installed -- run `pip install atlas-recall[dense]`",
                file=sys.stderr,
            )
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"index --dense: failed ({type(e).__name__}: {e}) -- is `ollama serve` running "
                  f"with {cfg.embed_model} pulled?", file=sys.stderr)
            return 1
        # Only persist dense_enabled=True once the build actually succeeded --
        # flipping it on speculatively would make every later query/distill
        # call probe a Chroma collection that was never built.
        cfg.dense_enabled = True
        save_config(cfg)
        print(f"index --dense: indexed {n} notes into Chroma collection {cfg.collection!r}")
    return 0


def cmd_query(args) -> int:
    cfg = load_config()
    if not cfg.notes_dir:
        print("No notes_dir configured -- run `recall init <dir>` first.", file=sys.stderr)
        return 1
    from .retrieval import format_pointer_block, top_pointers

    pointers = top_pointers(" ".join(args.query), cfg)
    if args.json:
        print(json.dumps(pointers, indent=2, default=str))
    else:
        print(format_pointer_block(pointers) or "(nothing cleared the admission floor)")
    return 0


def cmd_hook(args) -> int:
    from .hook import run_hook
    return run_hook()


def cmd_find(args) -> int:
    cfg = load_config()
    if not cfg.notes_dir:
        print("No notes_dir configured -- run `recall init <dir>` first.", file=sys.stderr)
        return 1
    from . import knowledge

    conn = knowledge.connect(_db_path(cfg), notes_dir=cfg.notes_dir)
    try:
        rows = knowledge.find(conn, " ".join(args.query), doc_type=args.type, n=args.n)
    except Exception as e:  # noqa: BLE001
        print(f"find: query error: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(rows, indent=2))
    elif not rows:
        print("(no results)")
    else:
        for r in rows:
            print(f"{r['id']} | {r['type']} | {r['title']} | {r['description']} | {r['path']}")
    return 0 if rows else 1


def cmd_node(args) -> int:
    cfg = load_config()
    from . import knowledge

    conn = knowledge.connect(_db_path(cfg), notes_dir=cfg.notes_dir)
    n = knowledge.node(conn, args.id)
    conn.commit()
    if not n:
        print(f"node: not found: {args.id}", file=sys.stderr)
        return 1
    print(f"id: {n['id']}")
    print(f"type: {n['type']}")
    print(f"title: {n['title']}")
    if n["description"]:
        print(f"description: {n['description']}")
    print(f"path: {n['path']}\n")
    print(n["body"])
    if not args.no_edges:
        print("\n-- 1-hop edges --")
        for e in n["edges"]:
            line = f"  {e['kind']} -> {e['target_title']} [{e['dst']}]"
            if e["divergent"]:
                line += " [DIVERGENT: target missing]"
            print(line)
    return 0


def cmd_trace(args) -> int:
    cfg = load_config()
    from . import knowledge

    conn = knowledge.connect(_db_path(cfg), notes_dir=cfg.notes_dir)
    root = conn.execute("SELECT id FROM docs WHERE id = ?", (args.id,)).fetchone()
    if not root:
        print(f"trace: not found: {args.id}", file=sys.stderr)
        return 1
    rows = knowledge.trace(conn, args.id, depth=args.depth, kind=args.kind)
    print(args.id)
    for r in rows:
        indent = "  " * r["depth"]
        line = f"{indent}{r['src']} -[{r['kind']}]-> {r['target_title']}"
        if r["divergent"]:
            line += " [DIVERGENT: target missing]"
        print(line)
    return 0


def cmd_map(args) -> int:
    cfg = load_config()
    from . import knowledge

    conn = knowledge.connect(_db_path(cfg), notes_dir=cfg.notes_dir)
    result = knowledge.map_topic(conn, " ".join(args.topic), n=args.n)
    if not result["hits"]:
        print(f"map: no notes matched {result['topic']!r}", file=sys.stderr)
        return 1
    print(f"# {result['topic']}\n")
    for h in result["hits"]:
        print(f"## {h['title']}  [{h['id']}]")
        if h["description"]:
            print(h["description"])
        for nb in h["neighbors"]:
            line = f"  -{nb['kind']}-> {nb['target_title']}"
            if nb["divergent"]:
                line += " [DIVERGENT]"
            print(line)
        print()
    return 0


def cmd_verify(args) -> int:
    cfg = load_config()
    from . import knowledge

    db_path = _db_path(cfg)
    try:
        lock = knowledge.acquire_writer_lock(db_path)
    except knowledge.WriterLockBusy:
        print("verify: another indexer is running, skipping", file=sys.stderr)
        return 0
    try:
        conn = knowledge.connect(db_path, notes_dir=cfg.notes_dir)
        result = knowledge.verify(conn, limit=args.limit)
    finally:
        knowledge.release_writer_lock(lock)
    print(f"verify: checked={result['checked']} divergent={result['divergent']}")
    for e in result["divergent_edges"]:
        print(f"  DIVERGENT: {e['src']} -[{e['kind']}]-> {e['dst']}")
    return 0


def cmd_distill(args) -> int:
    cfg = load_config()
    if not cfg.notes_dir:
        print("No notes_dir configured -- run `recall init <dir>` first.", file=sys.stderr)
        return 1
    from .distill import distill

    raw = sys.stdin.read()
    candidates = json.loads(raw) if raw.strip() else []
    ops = distill(candidates, cfg, verbose=True)
    print(json.dumps(ops, indent=2))
    return 0


def cmd_distill_apply(args) -> int:
    cfg = load_config()
    from .apply import apply_ops

    raw = open(args.ops_file, encoding="utf-8").read() if args.ops_file else sys.stdin.read()
    ops = json.loads(raw)
    summary = apply_ops(ops, cfg)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_warm(args) -> int:
    from .warmer import warm
    return warm()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="recall",
        description=(
            "Deterministic markdown memory and knowledge graph for AI agents -- "
            "BM25+wikilink retrieval over your notes, an MCP server, and a Claude "
            "Code hook, all over one index."
        ),
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("init", help="configure a notes directory")
    sp.add_argument("dir", nargs="?", default=".")
    sp.set_defaults(func=cmd_init)

    from .install import CLIENTS

    sp = sub.add_parser("install", help="wire recall into an AI client's config (merges, backs up)")
    sp.add_argument("--client", required=True, choices=sorted(CLIENTS))
    sp.add_argument("--dry-run", action="store_true", help="print what would change; write nothing")
    sp.set_defaults(func=cmd_install)

    sp = sub.add_parser("uninstall", help="remove recall's entry from an AI client's config")
    sp.add_argument("--client", required=True, choices=sorted(CLIENTS))
    sp.add_argument("--dry-run", action="store_true", help="print what would change; write nothing")
    sp.set_defaults(func=cmd_uninstall)

    sp = sub.add_parser("index", help="(re)index the configured directory")
    sp.add_argument("--dense", action="store_true", help="also build the optional Chroma/Ollama dense index")
    sp.add_argument("--full", action="store_true", help="reparse every discovered doc")
    sp.add_argument("--rebuild", action="store_true", help="drop and rebuild the whole index from scratch")
    sp.set_defaults(func=cmd_index)

    sp = sub.add_parser("query", help="what would be injected for a prompt (the automatic path)")
    sp.add_argument("query", nargs="+")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_query)

    sp = sub.add_parser("hook", help="UserPromptSubmit hook entry point (reads stdin JSON)")
    sp.set_defaults(func=cmd_hook)

    sp = sub.add_parser("find", help="full-text search (the human path)")
    sp.add_argument("query", nargs="+")
    sp.add_argument("--type")
    sp.add_argument("-n", type=int, default=8)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_find)

    sp = sub.add_parser("node", help="a note plus its 1-hop edges")
    sp.add_argument("id")
    sp.add_argument("--no-edges", action="store_true")
    sp.set_defaults(func=cmd_node)

    sp = sub.add_parser("trace", help="follow links from a note")
    sp.add_argument("id")
    sp.add_argument("--depth", type=int, default=2)
    sp.add_argument("--kind")
    sp.set_defaults(func=cmd_trace)

    sp = sub.add_parser("map", help="the shape of a topic: search + linked neighbors")
    sp.add_argument("topic", nargs="+")
    sp.add_argument("-n", type=int, default=5)
    sp.set_defaults(func=cmd_map)

    sp = sub.add_parser("verify", help="report broken wikilinks")
    sp.add_argument("--limit", type=int, default=50)
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("distill", help="propose ADD/UPDATE ops from candidates on stdin (writes nothing)")
    sp.set_defaults(func=cmd_distill)

    sp = sub.add_parser("distill-apply", help="apply proposed ops, asking before every write")
    sp.add_argument("ops_file", nargs="?", help="JSON file of ops; defaults to stdin")
    sp.set_defaults(func=cmd_distill_apply)

    sp = sub.add_parser("warm", help="(macOS) force-materialize an evicted iCloud notes directory")
    sp.set_defaults(func=cmd_warm)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as e:  # noqa: BLE001
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
