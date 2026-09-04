"""
The human-approval half of propose-only curation.

`distill.py` never writes a file. This module is the ONLY place in the
package that does, and it never writes without an explicit yes for that
specific op -- there is no batch "approve all" here on purpose. Read each
op, show the diff, ask.

Every write is resolved against notes_dir before it happens, never taken
on faith from an op's fields: `apply_ops` also runs standalone against a
JSON file (`recall distill-apply proposed.json`), so an op here may never
have passed through `distill.py`'s own `clean_slug`/corpus-matching at
all -- it could be hand-edited or come from an untrusted source. ADD and
UPDATE destinations are both re-derived and validated here, not trusted
from the op.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from .config import Config
from .corpus import load_corpus
from .distill import clean_slug


def _resolve_within_notes_dir(cfg: Config, path: str) -> str:
    """Refuse `path` unless it resolves (real, symlink-following) inside
    `cfg.notes_dir`. Raises ValueError on escape."""
    notes_real = os.path.realpath(os.path.expanduser(cfg.notes_dir))
    path_real = os.path.realpath(path)
    if os.path.commonpath([notes_real, path_real]) != notes_real:
        raise ValueError(f"refusing to write outside notes_dir: {path!r}")
    return path


def _dest_in_notes_dir(cfg: Config, stem: str) -> str:
    """Resolve `<notes_dir>/<stem>.md` for a NEW note name, refusing
    anything that isn't already a clean slug (traversal, absolute paths,
    and anything clean_slug would have to sanitize are all rejected
    outright rather than silently rewritten) -- plus the realpath/
    commonpath check as defense in depth against a symlinked notes_dir."""
    if clean_slug(stem) != stem:
        raise ValueError(f"unsafe note name: {stem!r}")
    path = os.path.join(os.path.expanduser(cfg.notes_dir), f"{stem}.md")
    path = _resolve_within_notes_dir(cfg, path)
    # ADD must never be an implicit overwrite. In particular, ops supplied to
    # `distill-apply` may be hand-authored and therefore have bypassed the
    # advisory duplicate detection in distill.py.
    if os.path.lexists(path):
        raise ValueError(f"refusing to overwrite existing note: {path}")
    return path


def _resolve_update_target(cfg: Config, op: Dict) -> Tuple[Dict, str]:
    """Re-resolve an UPDATE's target BY NAME against the live corpus --
    `op['matched_memory']['path']` is never trusted, it could be forged.
    Raises ValueError if the named target no longer exists in the corpus,
    or if the corpus's own path for it somehow escapes notes_dir."""
    name = op["matched_memory"]["name"]
    docs = load_corpus(cfg.notes_dir)
    target = next((d for d in docs if d["name"] == name), None)
    if target is None:
        raise ValueError(f"update target no longer exists in the corpus: {name!r}")
    dest = _resolve_within_notes_dir(cfg, target["path"])
    return target, dest


def _write_add(dest: str, op: Dict) -> str:
    content = (
        f"---\nname: {op['name']}\ndescription: {op['description']}\n"
        f"priority: normal\n---\n\n{op['body']}\n"
    )
    # Exclusive creation closes the check-then-write race with the preflight
    # check in _dest_in_notes_dir.
    with open(dest, "x", encoding="utf-8") as fh:
        fh.write(content)
    return dest


def _write_update(dest: str, target: Dict, op: Dict) -> str:
    # Carry the target's real priority forward -- never downgrade a
    # hard-rule/high note to normal just because it went through UPDATE.
    priority = target.get("priority", "normal")
    content = (
        f"---\nname: {op['name']}\ndescription: {op['description']}\n"
        f"priority: {priority}\n---\n\n{op['body']}\n"
    )
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write(content)
    return dest


def _describe(op: Dict, dest: str) -> str:
    lines = [f"{op['op']} {op['name']!r} (type={op['type']})"]
    if op.get("description"):
        lines.append(f"  description: {op['description']}")
    if op["op"] == "UPDATE":
        lines.append(f"  target: {dest}")
    else:
        lines.append(f"  destination: {dest}")
    if op["op"] != "UPDATE" and op.get("matched_memory"):
        lines.append(f"  advisory near-duplicate: {op['matched_memory']['name']!r} "
                      f"({op['matched_memory']['path']})")
    body_preview = op["body"].strip().splitlines()[0][:100] if op["body"].strip() else ""
    lines.append(f"  body: {body_preview}{'...' if len(op['body'].strip()) > 100 else ''}")
    return "\n".join(lines)


def apply_ops(ops: List[Dict], cfg: Config, auto_confirm=None) -> Dict:
    """Walk each proposed op, print it, ask for approval, write only on
    yes. `auto_confirm` is a callable(op) -> bool for non-interactive use
    (tests, scripting); interactive stdin y/n otherwise. Returns a summary
    dict. NEVER applies anything the caller didn't explicitly approve.

    Destination resolution (and its ValueError on an unsafe name/target)
    happens before the prompt is shown, on purpose: the human approving
    needs to see where it actually writes, not the name the op claims.

    One op's destination being unsafe/colliding must never abort the rest
    of the batch: earlier ops may already have written, and later ops in
    the same batch are otherwise-independent. So a ValueError here marks
    just that op as an error and moves on to the next one, rather than
    propagating out of the loop."""
    applied, skipped, errors = [], [], []
    for op in ops:
        target_doc: Optional[Dict] = None
        try:
            if op["op"] == "ADD":
                dest = _dest_in_notes_dir(cfg, op["name"])
            else:
                target_doc, dest = _resolve_update_target(cfg, op)
        except ValueError as exc:
            print(f"{op['op']} {op['name']!r}\n  -> error: {exc}\n")
            errors.append({"name": op["name"], "error": str(exc)})
            continue

        print(_describe(op, dest))
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
            path = _write_add(dest, op)
        else:
            path = _write_update(dest, target_doc, op)
        print(f"  -> wrote {path}\n")
        applied.append(op["name"])
    return {"applied": applied, "skipped": skipped, "errors": errors}
