# Contributing

## Setup

```
git clone https://github.com/Abydin/atlas-recall
cd atlas-recall
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dense,dev]"
pytest
```

The `pip install --upgrade pip` step matters on the `>=3.9` floor this
package claims to support: the pip bundled with a stock 3.9 venv (e.g.
21.2.4) predates PEP 660 and rejects `pip install -e .` outright.

`pip install -e ".[dense,dev]"` pulls in `chromadb` too, so the dense-path
tests actually run. If you only care about the default (BM25) path, `pip
install -e ".[dev]"` is enough and the dense tests will skip themselves.

## Ground rules

- **The default path stays dependency-free.** If a change to `retrieval.py`,
  `bm25.py`, or `corpus.py` needs anything beyond the standard library plus
  `rank_bm25`, that's a sign it belongs behind the `dense` extra instead.
- **Nothing here knows about a specific person's machine or corpus.** Every
  path is either a CLI argument or comes from the config file `recall init`
  writes. If you catch yourself hardcoding a directory, stop.
- **Curation stays propose-then-approve.** `distill.py` never writes a file.
  If you're adding a new curation feature, it proposes an op; `apply.py` is
  the only place that writes, and only after an explicit yes for that one
  op.
- **Re-measure before touching `retrieval.py`'s tunables.** `RRF_K`,
  `MIN_SCORE`, `PRIORITY_WEIGHT`, `RECENCY_FLOOR` were set by measuring
  precision/recall against a labeled should-hit corpus (see the README's
  "measured result" section for the methodology), not by feel. No eval
  harness ships in this repo yet -- build a small labeled query set over
  your own notes (query, expected note names) and compare `recall query`
  output before/after your change. Changing a tunable without re-measuring
  is how a "small" tweak quietly makes retrieval worse.

## Tests

```
pytest tests/
```

Tests build a small throwaway markdown corpus under a temp directory --
nothing here reads or depends on any real notes.

## Reporting a bug

Open an issue with: your OS, `recall --version` output (or the installed
package version), the command you ran, and what you expected vs. what
happened. If it's a retrieval-quality issue ("this should have matched but
didn't"), include the query and a redacted excerpt of the note you expected
back -- retrieval bugs are much easier to fix with an example than a
description.
