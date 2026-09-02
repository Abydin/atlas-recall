# Contributing

## Setup

```
git clone https://github.com/Abydin/atlas-recall
cd atlas-recall
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dense,dev]"
pytest
```

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
- **Run the eval before touching `retrieval.py`'s tunables.** `RRF_K`,
  `MIN_SCORE`, `PRIORITY_WEIGHT`, `RECENCY_FLOOR` were set by measuring
  precision/recall against a labeled corpus, not by feel. Changing one
  without re-measuring is how a "small" tweak quietly makes retrieval worse.

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
