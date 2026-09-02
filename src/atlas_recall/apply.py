"""
The human-approval half of propose-only curation.

`distill.py` never writes a file. This module is the ONLY place in the
package that does, and it never writes without an explicit yes for that
specific op -- there is no batch "approve all" here on purpose. Read each
op, show the diff, ask.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List

from .config import Config


def _write_add(cfg: Config, op: Dict) -> str:
    path = os.path.join(os.path.expanduser(cfg.notes_dir), f"{op['name']}.md")
    content = (
        f"---\nname: {op['name']}\ndescription: {op['description']}\n"
        f"priority: normal\n---\n\n{op['body']}\n"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


def _write_update(cfg: Config, op: Dict) -> str:
    target_path = op["matched_memory"]["path"]
    content = (
        f"---\nname: {op['name']}\ndescription: {op['description']}\n"
        f"priority: normal\n---\n\n{op['body']}\n"
    )
    with open(target_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return target_path


def _describe(op: Dict) -> str:
    lines = [f"{op['op']} {op['name']!r} (type={op['type']})"]
    if op.get("description"):
        lines.append(f"  description: {op['description']}")
    if op["op"] == "UPDATE":
        lines.append(f"  target: {op['matched_memory']['path']}")
    elif op.get("matched_memory"):
        lines.append(f"  advisory near-duplicate: {op['matched_memory']['name']!r} "
                      f"({op['matched_memory']['path']})")
    body_preview = op["body"].strip().splitlines()[0][:100] if op["body"].strip() else ""
    lines.append(f"  body: {body_preview}{'...' if len(op['body'].strip()) > 100 else ''}")
    return "\n".join(lines)


def apply_ops(ops: List[Dict], cfg: Config, auto_confirm=None) -> Dict:
    """Walk each proposed op, print it, ask for approval, write only on
    yes. `auto_confirm` is a callable(op) -> bool for non-interactive use
    (tests, scripting); interactive stdin y/n otherwise. Returns a summary
    dict. NEVER applies anything the caller didn't explicitly approve."""
    applied, skipped = [], []
    for op in ops:
        print(_describe(op))
        if auto_confirm is not None:
            ok = auto_confirm(op)
        else:
            ans = input("Apply this op? [y/N] ").strip().lower()
            ok = ans == "y"
        if not ok:
            print("  -> skipped\n")
            skipped.append(op["name"])
            continue
        if op["op"] == "ADD":
            path = _write_add(cfg, op)
        else:
            path = _write_update(cfg, op)
        print(f"  -> wrote {path}\n")
        applied.append(op["name"])
    return {"applied": applied, "skipped": skipped}
