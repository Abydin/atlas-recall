"""
Hybrid retrieval core: BM25 lexical search, fused with optional dense
(semantic) search via Reciprocal Rank Fusion (RRF), scored by recency
decay and a priority tiebreak, gated by a conservative admission floor.

This is the differentiator this whole package exists to ship. Adapted from
a private hybrid_recall.py (RRF fusion over a Chroma/bge-m3 corpus) built
across several rounds of measurement against a labeled should-hit eval set.
The headline number, reproduced here because it's the reason RRF (rank-
based fusion) replaced a plain distance cutoff: at a 0.45 cosine cutoff, a
nomic-embed-text dense-only path admitted an average of 16.1 of 108 corpus
docs per query at 100% hit-recall -- a flat junk floor, the cutoff wasn't
discriminating anything. Swapping to bge-m3 at an empirically swept 0.48
cutoff, still dense-only, brought that down to 2.9 of 119 docs at the same
100% hit-recall -- roughly 5.5x fewer false admits. RRF fusion (this file)
is what let a lexical retriever contribute to that number too, without a
BM25 score and a cosine distance needing to live on the same scale -- RRF
only cares about RANK, so "keyword-shaped query wins on BM25" and
"paraphrase wins on dense" can both feed the same fused score.

Admission vs. ranking are deliberately split: `admission_score` (rrf x
recency, NO priority) is what MIN_SCORE gates on -- priority must never be
able to conjure a floor-clearing score out of a weak/spurious match.
`final_score` (admission_score x priority weight) is priority-boosted and
used ONLY to re-rank among candidates that already cleared the floor on
their own merit.

Never raises past top_pointers()/score_docs(): a failure in the optional
dense path (chromadb not installed, Ollama not running, empty index) just
drops fusion to BM25-only. That's the whole point of the "dense path is
optional" design decision -- retrieval quality degrades gracefully, it
never breaks.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from .bm25 import build_index, rank as bm25_rank
from .config import Config
from .corpus import load_corpus, wikilinks

RRF_K = 60              # standard RRF smoothing constant
BM25_TOPN = 15
DENSE_TOPN = 15
RECENCY_FLOOR = 0.6     # oldest notes still keep at least this multiplier
WIKILINK_MAX_EXTRA = 2  # cap on wikilink-expansion additions

PRIORITY_WEIGHT = {"hard-rule": 1.20, "high": 1.10, "normal": 1.0}
# Gentle tiebreak, not a ranking override. Measured on the reference corpus:
# at 2.0/1.5 weights, a tangentially-related high-priority doc outscored and
# displaced the actually-correct match for a real query -- priority must
# nudge among near-ties, not override real relevance signal.


def rrf_fuse(*ranked_lists: List[str], k: int = RRF_K) -> Dict[str, float]:
    """Reciprocal Rank Fusion over any number of ranked key-lists.
    Returns {key: fused_score}, higher is better."""
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for i, key in enumerate(ranked, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + i)
    return scores


def recency_decay(mtime: float, halflife_days: float, now: Optional[float] = None) -> float:
    now = now or time.time()
    age_days = max(0.0, (now - mtime) / 86400.0)
    decay = 0.5 ** (age_days / halflife_days)
    return max(RECENCY_FLOOR, decay)


def _dense_hits(query: str, cfg: Config, topn: int) -> List[str]:
    """Best-effort dense ranking. Returns [] on any failure -- chromadb
    absent, Ollama unreachable, empty collection, whatever. This is the
    degrade point: everything downstream just sees an empty dense list and
    falls back to BM25-only fusion."""
    if not cfg.dense_enabled:
        return []
    try:
        from . import dense as dense_mod
    except ImportError:
        return []
    if not dense_mod.ollama_reachable():
        return []
    try:
        hits = dense_mod.query_dense(query, cfg, topn=topn)
    except Exception:
        return []
    return [h["name"] for h in hits if h["distance"] <= cfg.dense_dist_max]


def score_docs(
    query: str,
    docs: List[Dict],
    cfg: Config,
    bm25=None,
    keys=None,
    now: Optional[float] = None,
) -> List[Tuple[Dict, float]]:
    """Core hybrid scoring. Returns a list of (doc, final_score) sorted
    descending, excluding superseded docs. Never raises."""
    if bm25 is None or keys is None:
        bm25, keys = build_index(docs)

    bm25_hits = bm25_rank(query, bm25, keys, topn=BM25_TOPN)
    dense_hits = _dense_hits(query, cfg, DENSE_TOPN)

    fused = rrf_fuse(bm25_hits, dense_hits)
    by_key = {d["key"]: d for d in docs}

    scored = []
    for key, rrf_score in fused.items():
        doc = by_key.get(key)
        if doc is None or doc.get("superseded_by"):
            continue
        decay = recency_decay(doc["mtime"], cfg.recency_halflife_days, now=now)
        admission_score = rrf_score * decay
        if admission_score < cfg.min_score:
            continue
        weight = PRIORITY_WEIGHT.get(doc["priority"], 1.0)
        final = admission_score * weight
        scored.append((doc, final))

    scored.sort(key=lambda t: -t[1])
    return scored


def top_pointers(
    query: str,
    cfg: Config,
    docs: Optional[List[Dict]] = None,
    now: Optional[float] = None,
) -> List[Dict]:
    """Return the final pointer list for injection: up to cfg.top_k hybrid
    hits plus up to WIKILINK_MAX_EXTRA high-priority 1-hop wikilink
    neighbors. Returns [] if nothing clears cfg.min_score (conservative
    floor -- silent beats noisy)."""
    if docs is None:
        docs = load_corpus(cfg.notes_dir)
    if not docs:
        return []

    bm25, keys = build_index(docs)
    scored = score_docs(query, docs, cfg, bm25=bm25, keys=keys, now=now)
    if not scored:
        return []

    top = scored[: cfg.top_k]
    by_name = {d["name"]: d for d, _ in scored}
    result = []
    seen_names = set()
    for doc, score in top:
        result.append({**doc, "why": "hybrid retrieval (BM25+dense, RRF)", "score": score})
        seen_names.add(doc["name"])

    extra = 0
    for doc, _score in top:
        if extra >= WIKILINK_MAX_EXTRA:
            break
        for link in wikilinks(doc["body"]):
            if extra >= WIKILINK_MAX_EXTRA:
                break
            neighbor = by_name.get(link)
            if not neighbor or neighbor["name"] in seen_names:
                continue
            if neighbor["priority"] not in ("hard-rule", "high"):
                continue
            result.append({**neighbor, "why": f"wikilink from {doc['name']}", "score": 0.0})
            seen_names.add(neighbor["name"])
            extra += 1

    return result


def format_pointer_block(pointers: List[Dict]) -> str:
    """Render the injection block: pointers for normal-priority hits, full
    body for hard-rule/high hits."""
    if not pointers:
        return ""
    lines = ["== LIKELY-RELEVANT NOTES (read before acting) =="]
    for p in pointers:
        if p["priority"] in ("hard-rule", "high"):
            lines.append(f"{p['name']} -- {p['path']} -- {p['why']}")
            lines.append(f"  BODY: {p['body']}")
        else:
            one_liner = p["description"] or (p["body"].splitlines()[0][:160] if p["body"] else "")
            lines.append(f"{p['name']} -- {p['path']} -- {one_liner}")
    return "\n".join(lines)
