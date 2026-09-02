"""
BM25 lexical retrieval -- the default, always-on retriever. Pure Python
(rank_bm25), no external services. This is what makes `recall query` work
immediately after `pip install`, with nothing else running.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Small stopword list. Necessary because a personal-notes corpus is small
# (tens to low hundreds of docs) -- BM25's IDF assumes common words are
# common IN THE CORPUS, and a small corpus doesn't give words like
# "what"/"should"/"before" enough volume to be correctly downweighted, so
# they end up driving false-positive matches on fully off-topic queries.
STOPWORDS = frozenset("""
a an the this that these those is are was were be been being
i you he she it we they me him her us them my your his its our their
what which who whom whose when where why how
do does did doing done
have has had having
will would shall should can could may might must
not no nor
in on at by for with about against between into through during
before after above below to from up down out off over under
again further then once here there all any both each few more most
other some such only own same so than too very just
and or but if
of as
""".split())


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def build_index(docs: List[Dict]):
    """Return (BM25Okapi instance, doc_keys_in_bm25_order), or (None, [])
    if `docs` is empty -- BM25Okapi's constructor divides by corpus_size
    and raises ZeroDivisionError on an empty corpus."""
    if not docs:
        return None, []
    from rank_bm25 import BM25Okapi
    corpus_tokens = [tokenize(d["tokens_text"]) for d in docs]
    bm25 = BM25Okapi(corpus_tokens)
    return bm25, [d["key"] for d in docs]


def rank(query: str, bm25, keys: List[str], topn: int = 15) -> List[str]:
    """Return ranked list of doc keys (best first), zero-score hits dropped."""
    if bm25 is None or not keys:
        return []
    scores = bm25.get_scores(tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return [keys[i] for i in order[:topn] if scores[i] > 0]
