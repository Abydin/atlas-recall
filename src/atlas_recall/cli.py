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
import sys
from pathlib import Path

from . import __version__
from .config import Config, load_config, save_config, DEFAULT_CONFIG_PATH

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


def _db_path(cfg: Config) -> Path:
    return Path(os.path.expanduser(cfg.chroma_dir)).parent / "recall.db"


def cmd_init(args) -> int:
    notes_dir = os.path.expanduser(args.dir)
    os.makedirs(notes_dir, exist_ok=True)
    cfg = load_config()
    cfg.notes_dir = notes_dir
    path = save_config(cfg)
    print(f"Wrote config to {path}")
    print(f"notes_dir = {notes_dir}\n")
    print("Paste this block into your Claude Code settings.json (merge the")
    print("\"hooks\" key if one already exists there):\n")
    print(HOOK_BLOCK)
    print("\nThen: recall index    (builds the search index)")
    print("      recall query \"...\"   (see what would be injected)")
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
            conn = knowledge.connect(db_path, fresh=True)
        else:
            conn = knowledge.connect(db_path, fresh=False)
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

    conn = knowledge.connect(_db_path(cfg))
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

    conn = knowledge.connect(_db_path(cfg))
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

    conn = knowledge.connect(_db_path(cfg))
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

    conn = knowledge.connect(_db_path(cfg))
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
        conn = knowledge.connect(db_path)
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
    p = argparse.ArgumentParser(prog="recall", description="Deterministic markdown memory for Claude Code.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("init", help="configure a notes directory and print the hook block")
    sp.add_argument("dir", nargs="?", default=".")
    sp.set_defaults(func=cmd_init)

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
