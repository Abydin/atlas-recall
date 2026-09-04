"""
Corpus loading: scan a directory of markdown notes into doc dicts.

Reads a single configured `notes_dir` -- "read a directory you point it
at" is the whole point of this package, no multi-project directory map,
no per-machine hardcoding. Also keeps the dataless-file guard: on macOS, a
directory
synced through iCloud Drive can evict a file's local copy to a network-
only placeholder, and reading one blocks for tens of seconds. That guard
is cheap (an lstat, not a read) and harmless on any other OS.
"""
from __future__ import annotations

import glob
import os
import re
import sys
import time
from typing import Dict, List

SF_DATALESS = 0x40000000  # macOS st_flags bit for an iCloud-evicted (dataless) file


def is_dataless(path: str) -> bool:
    """True if `path` is a macOS dataless (iCloud-evicted) placeholder --
    reading it would block on a network re-download. os.lstat does NOT
    trigger a download, so this check is safe. Always False on non-macOS
    (st_flags absent there)."""
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return bool(getattr(st, "st_flags", 0) & SF_DATALESS)


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.S)


def read_frontmatter(text: str):
    """Parse a note's frontmatter into a flat dict (top-level keys only).
    Returns (frontmatter_dict, body)."""
    fm = {}
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return fm, text
    for line in m.group(1).splitlines():
        s = line.strip()
        if ":" in s and not line.startswith((" ", "\t")):
            k, _, v = s.partition(":")
            fm[k.strip()] = v.strip().strip('"')
    return fm, m.group(2)


PRIORITY_LEVELS = ("hard-rule", "high", "normal")


def load_corpus(notes_dir: str) -> List[Dict]:
    """Scan `notes_dir` recursively for *.md files; return doc dicts:
    {key, name, description, priority, superseded_by, path, mtime, body,
    tokens_text}. `key` is the note's path relative to `notes_dir` without
    the .md suffix, so recursive corpora can safely contain notes with the
    same filename or display name."""
    notes_dir = os.path.expanduser(notes_dir)
    docs = []
    skipped_dataless = 0
    if not os.path.isdir(notes_dir):
        return docs

    for fpath in sorted(glob.glob(os.path.join(notes_dir, "**", "*.md"), recursive=True)):
        base = os.path.basename(fpath)
        if base in ("MEMORY.md", "INDEX.md"):
            continue
        if os.sep + "_archive" + os.sep in fpath:
            continue
        if os.sep + ".snapshots" + os.sep in fpath:
            continue
        if is_dataless(fpath):
            skipped_dataless += 1
            continue
        try:
            text = open(fpath, encoding="utf-8").read()
        except OSError:
            continue

        name = os.path.splitext(base)[0]
        fm, body = read_frontmatter(text)
        name = fm.get("name", name)
        description = fm.get("description", "")
        priority = fm.get("priority", "normal")
        if priority not in PRIORITY_LEVELS:
            priority = "normal"
        superseded_by = fm.get("superseded-by", "")
        try:
            mtime = os.path.getmtime(fpath)
        except OSError:
            mtime = time.time()

        docs.append({
            "key": os.path.splitext(os.path.relpath(fpath, notes_dir))[0].replace(os.sep, "/"),
            "name": name,
            "description": description,
            "priority": priority,
            "superseded_by": superseded_by,
            "path": fpath,
            "mtime": mtime,
            "body": body.strip(),
            "tokens_text": f"{name} {description} {body}",
        })

    if skipped_dataless:
        print(
            f"[recall] {skipped_dataless} note(s) under {notes_dir} skipped "
            f"(iCloud-evicted / dataless) -- run `recall warm` to materialize "
            f"them so they're searched",
            file=sys.stderr,
        )

    return docs


def wikilinks(body: str):
    return set(re.findall(r"\[\[([^\]]+)\]\]", body))
