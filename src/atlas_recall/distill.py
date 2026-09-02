"""
Propose-only memory curation.

An agent (or a person) proposes ADD/UPDATE operations on the notes corpus
as JSON; this module turns those candidates into a reviewable list of
proposed ops -- clean slugs, type classification, an advisory near-
duplicate warning. It never writes to the corpus itself. Nothing here is
allowed to touch a file; `recall distill-apply` is the separate, explicit,
human-gated step that actually writes, and it asks before every single
write.

That split is the whole trust story: no other memory-curation tool ships
propose-then-approve as the default, let alone as the ONLY path.

Dedup degrade path: near-duplicate detection tries dense (semantic) recall
first when it's configured and available, and falls back to BM25 lexical
overlap when it isn't -- never a hard crash. Advisory either way: it only
ever decorates an ADD card with "you might mean this existing note", it
never controls whether something becomes an ADD or an UPDATE. That control
is explicit only -- a candidate opts into UPDATE by naming its target.
"""
from __future__ import annotations

import re
import sys
from typing import Dict, List, Optional

from .bm25 import build_index, rank as bm25_rank
from .config import Config
from .corpus import load_corpus

VALID_TYPES = ("feedback", "project", "user", "reference")

FEEDBACK_KEYWORDS = ["rule", "lesson", "don't", "always", "never", "when", "pattern"]
PROJECT_KEYWORDS = ["project", "status", "planning", "state", "progress"]
USER_KEYWORDS = ["preference", "schedule", "routine", "setting"]


def classify_type(name: str, description: str, body: str) -> str:
    text = f"{name} {description} {body}".lower()
    feedback_hits = sum(1 for kw in FEEDBACK_KEYWORDS if kw in text)
    project_hits = sum(1 for kw in PROJECT_KEYWORDS if kw in text)
    user_hits = sum(1 for kw in USER_KEYWORDS if kw in text)

    if "feedback" in name or "lesson" in name or feedback_hits > project_hits + user_hits:
        return "feedback"
    if "project" in name or project_hits > feedback_hits + user_hits:
        return "project"
    if "routine" in name or "schedule" in name or user_hits > feedback_hits + project_hits:
        return "user"
    return "reference"


def clean_slug(text: str) -> str:
    """Turn a candidate name (possibly dirty: list markers, numbers, bold,
    underscores) into a clean kebab slug. Never returns empty."""
    s = (text or "").strip().lower()
    s = s.replace("**", "").replace("*", "").replace("`", "")
    s = re.sub(r"^[\s\d.)\](_\-*#>+:]+", "", s)
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s[:60].rstrip("-") or "memory"


# ---------------------------------------------------------------------------
# Dedup: dense-first, BM25-or-nothing degrade
# ---------------------------------------------------------------------------
def find_duplicates(query: str, cfg: Config, docs: Optional[List[Dict]] = None, k: int = 3) -> List[Dict]:
    """Return near-duplicate candidates: [{name, description, path, type,
    distance_or_score}]. Tries dense recall first (if configured and
    reachable); degrades to BM25 lexical overlap; degrades to [] (silently
    proposing ADD) if even that finds nothing. Never raises."""
    if docs is None:
        docs = load_corpus(cfg.notes_dir)

    if cfg.dense_enabled:
        try:
            from . import dense as dense_mod
            if dense_mod.ollama_reachable():
                hits = dense_mod.query_dense(query, cfg, topn=k)
                by_key = {d["key"]: d for d in docs}
                out = []
                for h in hits:
                    d = by_key.get(h["name"])
                    out.append({
                        "name": h["name"],
                        "description": d["description"] if d else "",
                        "path": h["path"],
                        "type": (d or {}).get("priority", "reference"),
                        "distance": h["distance"],
                    })
                return out
        except Exception as e:
            print(f"[distill] WARN: dense dedup unavailable ({type(e).__name__}: {e}) -- degrading to BM25",
                  file=sys.stderr)

    # BM25-or-nothing degrade path.
    bm25, keys = build_index(docs)
    hit_keys = bm25_rank(query, bm25, keys, topn=k)
    by_key = {d["key"]: d for d in docs}
    out = []
    for key in hit_keys:
        d = by_key.get(key)
        if not d:
            continue
        out.append({
            "name": d["name"], "description": d["description"], "path": d["path"],
            "type": d.get("priority", "reference"), "distance": None,
        })
    return out


def resolve_update_target(updates_name: str, cfg: Config, docs: List[Dict]) -> Optional[Dict]:
    """Resolve an explicit `updates: "<name>"` opt-in to its note. Only an
    EXACT (slug-insensitive) name match is accepted -- proximity alone is
    never enough to pick a write target."""
    target_slug = clean_slug(updates_name)
    matches = find_duplicates(updates_name, cfg, docs=docs, k=10)
    for m in matches:
        if clean_slug(m["name"]) == target_slug:
            return m
    return None


def propose_op(op, name, description, body, mtype, matched_memory=None, distance=None) -> Dict:
    return {
        "op": op, "name": name, "type": mtype, "description": description, "body": body,
        "matched_memory": matched_memory, "distance": distance,
    }


def distill(candidates: List[Dict], cfg: Config, verbose: bool = False) -> List[Dict]:
    """Turn author-supplied candidate memories into proposed ops (ADD/UPDATE).
    Each candidate: {name?, type?, description, body, updates?}. `body` is
    passed through verbatim -- never truncated, never mangled. An empty
    body is a hard error (stderr + skip)."""
    docs = load_corpus(cfg.notes_dir)
    ops = []
    seen_add_slugs: Dict[str, int] = {}

    for i, cand in enumerate(candidates):
        if not isinstance(cand, dict):
            print(f"[distill] SKIP candidate #{i}: not an object", file=sys.stderr)
            continue

        raw_name = (cand.get("name") or "").strip()
        description = (cand.get("description") or "").strip()
        body = cand.get("body") or ""

        if not body.strip():
            label = raw_name or description or f"#{i}"
            print(f"[distill] SKIP {label!r}: empty body -- candidates must "
                  f"arrive with the full body text authored", file=sys.stderr)
            continue

        query = description or raw_name
        matches = find_duplicates(query, cfg, docs=docs, k=3)
        nearest = matches[0] if matches else None

        updates_name = (cand.get("updates") or "").strip()
        if updates_name:
            target = resolve_update_target(updates_name, cfg, docs)
            if target is None:
                label = raw_name or description or f"#{i}"
                print(f"[distill] SKIP {label!r}: updates target {updates_name!r} not "
                      f"found -- never silently falls back to ADD", file=sys.stderr)
                continue
            op_kind = "UPDATE"
            matched_memory = target
            name = clean_slug(raw_name or description)
            distance = None  # explicit opt-in, not proximity-driven
        else:
            op_kind = "ADD"
            matched_memory = nearest
            distance = nearest["distance"] if nearest else None
            name = clean_slug(raw_name or description)
            if name in seen_add_slugs:
                seen_add_slugs[name] += 1
                name = f"{name}-{seen_add_slugs[name]}"
            else:
                seen_add_slugs[name] = 1

        mtype = cand.get("type")
        if mtype not in VALID_TYPES:
            mtype = classify_type(name, description, body)

        ops.append(propose_op(op_kind, name, description, body, mtype, matched_memory, distance))
        if verbose:
            print(f"[distill] {op_kind} {name!r} (type={mtype})", file=sys.stderr)

    return ops
