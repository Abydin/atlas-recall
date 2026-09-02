# atlas-recall

Deterministic markdown memory and knowledge graph for AI coding agents:
index a directory of notes, retrieve the relevant ones, and search/trace
the same corpus as a wikilink graph. No vector database required, and no
agent lock-in -- the engine is a plain `recall` CLI, so it works with any
agent or workflow that can run a shell command.

Retrieval and search are two interfaces on one engine, not two separate
tools bolted together. The same corpus, the same config, the same index:
`recall query` answers "what's relevant" automatically (wire it into
anything that can invoke a command per turn), and `recall find` / `recall
map` / `recall trace` answer the same question interactively for a human,
plus a wikilink graph over your notes -- as of this writing (2026), mem0,
Letta, and Zep don't ship that combination; check their current docs, this
is the kind of claim that goes stale.

Claude Code is the first integration this package ships: a
`UserPromptSubmit` hook (`recall hook`) that injects retrieval results
straight into the prompt, no copy-paste. More integrations are planned;
until they land, any other agent gets the same retrieval and knowledge
graph by shelling out to the `recall` CLI directly.

## The problem

Claude Code forgets everything between sessions unless you paste context in
by hand. The usual fix is a vector database: embed every note, embed the
query, take the top-K by cosine distance. In practice that ceiling doesn't
discriminate well on a personal notes corpus (tens to low hundreds of
files), it needs an embedding model running somewhere, and it's one more
service that has to be up before the first command works.

## The measured result

A dense-only retriever at a swept 0.45 cosine cutoff, measured against a
15-query labeled should-hit set over a 108-document corpus, admitted an
average of **16.1 of 108 documents per query at 100% hit-recall** (14.9% of
the corpus) - the cutoff wasn't discriminating anything, it was a junk
floor. Swapping the embedding model and re-sweeping the cutoff (0.48, still
dense-only) brought that down to **2.9 of 119 documents at the same 100%
hit-recall** (2.4% of the corpus), roughly 5.5x fewer false admits.

The retriever this package ships (`retrieval.py`) goes further: BM25
lexical search fused with the dense path via Reciprocal Rank Fusion (RRF),
gated by a conservative admission floor, scored by recency decay and a
light priority tiebreak. RRF fuses by RANK, not by raw score, which is what
lets a keyword-shaped query win on BM25 and a paraphrase win on dense
without needing a BM25 score and a cosine distance to live on the same
scale. The 5.5x number above is the dense half of that measurement,
reproduced honestly here because it's the reason RRF replaced a plain
distance cutoff rather than the whole story - re-run `recall query` against
your own corpus and judge the fused result on your own notes; a number
measured on someone else's corpus is a reason to expect a good default, not
a guarantee.

## The design decision that makes this a product, not a demo

**The dense path is optional.** Most retrieval-over-markdown projects need a
Python venv with `chromadb`, plus a running Ollama with an embedding model
pulled, before the first query works. Most people bounce off that step and
the repo dies unused.

- Default: **BM25 lexical search, plus recency decay, plus a priority
  weight. Pure Python, zero external services, works immediately after
  install.**
- `recall index --dense` opts into Chroma + Ollama for semantic retrieval on
  top of BM25.
- Everything that touches the dense path degrades instead of crashing when
  it isn't configured: retrieval fuses BM25-only, curation dedup falls back
  to BM25 lexical overlap, and `recall index --dense` without `chromadb`
  installed prints exactly what to run rather than a stack trace.

## Install

```
pipx install atlas-recall
```

(or `pip install atlas-recall` inside a venv). To track the latest commit
rather than the released version, install from source with
`pipx install git+https://github.com/Abydin/atlas-recall`.

The dense path is a separate extra, not a hard dependency:

```
pip install "atlas-recall[dense]"
```

## The three commands, with real output

```
$ recall init ~/notes
Wrote config to ~/.config/atlas-recall/config.json
notes_dir = ~/notes

Paste this block into your Claude Code settings.json ...
{
  "hooks": {
    "UserPromptSubmit": [
      { "hooks": [ { "type": "command", "command": "recall hook" } ] }
    ]
  }
}

$ recall index
index: discovered=3 changed=3 unchanged=0 removed=0 edges_verified=1 edges_divergent=0

$ recall query "what temperature should the water be for a light roast pour over"
== LIKELY-RELEVANT NOTES (read before acting) ==
water-temperature -- ~/notes/water-temperature.md -- hybrid retrieval (BM25+dense, RRF)
  BODY: 96C for light roasts, 92C for dark roasts. Off the boil by about 30 seconds.
coffee-brewing -- ~/notes/coffee-brewing.md -- pour-over ratio and grind size notes
```

That ran with no Chroma collection built and no Ollama process running -
`recall init` never mentions either. `recall hook` (the piece that actually
wires into Claude Code) reads a `UserPromptSubmit` payload on stdin and
emits the same block as `hookSpecificOutput.additionalContext`; paste the
block `recall init` printed into your `settings.json` and every prompt gets
the relevant notes injected automatically.

## The human path: search and trace the same corpus

```
$ recall find "chain lube"
bike-maintenance | doc | bike-maintenance | chain lube schedule | ~/notes/bike-maintenance.md

$ recall node coffee-brewing
id: coffee-brewing
...
-- 1-hop edges --
  link -> water-temperature [water-temperature]

$ recall trace coffee-brewing
coffee-brewing
  coffee-brewing -[link]-> water-temperature

$ recall map coffee
# coffee

## coffee-brewing  [coffee-brewing]
pour-over ratio and grind size notes
  -link-> water-temperature

$ recall verify
verify: checked=1 divergent=0
```

`find` is full-text search (SQLite FTS5, porter-stemmed). `node` shows one
note plus its outbound `[[wikilinks]]`. `trace` follows those links
recursively. `map` shows the shape of a topic: search hits plus what they
connect to. `verify` reports links that point at a note that no longer
exists, so drift is visible instead of silent.

`recall index` builds the SQLite FTS5 + wikilink graph that `find` /
`node` / `trace` / `map` / `verify` read (and the Chroma collection, with
`--dense`). It does NOT need to run before `recall query` / `recall hook`
work: the BM25 corpus those read is built fresh, in-process, per query --
there's no separate BM25 index step, and that's a real strength, not a
gap: the automatic path works the moment `notes_dir` is configured, no
indexing required. One directory, one config file - `index` and `query`
are still two interfaces on the same engine, not two tools that happen to
sit in the same repo.

## Propose-only curation

As of this writing (2026), I'm not aware of another tool that ships memory
curation where an agent proposes edits and a human approves every single
one, with nothing auto-written -- that's a claim about what I've seen, not
an exhaustive survey, so verify against current alternatives before
repeating it. `recall distill`
reads candidate notes (name, description, body, optionally which existing
note to update) as JSON on stdin and returns a list of proposed ADD/UPDATE
operations - an advisory near-duplicate warning on ADDs, an explicit,
name-based opt-in required for UPDATE (never a fuzzy match deciding to
overwrite something). It never writes a file.

```
$ echo '[{"description": "grind size for espresso", "body": "18g in, 36g out, 25-30s."}]' \
    | recall distill > proposed.json
$ recall distill-apply proposed.json
ADD 'grind-size-for-espresso' (type=reference)
  description: grind size for espresso
  body: 18g in, 36g out, 25-30s.
Apply this op? [y/N] y
  -> wrote ~/notes/grind-size-for-espresso.md
```

`distill-apply` is the only place in this package that writes to your
notes, and it asks before every op, individually - there is no batch
approve.

## Configuration

`recall init <dir>` writes `~/.config/atlas-recall/config.json` (or
`$ATLAS_RECALL_CONFIG`). Everything the engine touches comes from that file
or a CLI flag - nothing in this package points at a specific person's
machine, corpus, or rules. Two fields worth knowing about:

- `rules_path`: an optional file (e.g. a house style guide) injected
  verbatim, undiluted, at the top of every hook block. Empty by default.
- `keyword_rules`: `[[keywords], "rule line"]` pairs - if a prompt contains
  one of the keywords, that one rule line gets echoed inline. Ships empty.
  See `examples/keyword_rules.json` for the shape; copy it, don't inherit
  it, the rules in there aren't yours.

## What's in the box

| Path | What it is |
|---|---|
| `bm25.py` | Pure-Python lexical retrieval, the always-on default. |
| `dense.py` | Optional Chroma + Ollama semantic retrieval, guarded imports. |
| `retrieval.py` | RRF fusion, recency decay, priority tiebreak, admission floor. |
| `knowledge.py` | SQLite FTS5 + wikilink graph: `find`/`node`/`trace`/`map`/`verify`. |
| `hook.py` | The Claude Code `UserPromptSubmit` integration. |
| `distill.py` / `apply.py` | Propose-only curation and its human-gated apply step. |
| `warmer.py` | Optional macOS fix for cloud-synced notes going "dataless". |
| `cli.py` | The `recall` console script. |

## Known limitation, carried over honestly

`recall index` (incremental) only re-derives outbound edges for notes it
re-parses that pass. An edge whose TARGET note was deleted or renamed, but
whose SOURCE note wasn't touched, doesn't get a chance to self-heal - it
stays flagged divergent (`recall verify` will show it) until `recall index
--rebuild` does a full pass. This isn't hidden: it's the tradeoff of
incremental indexing being fast, and `verify` exists specifically so it's
visible rather than silent.

## License

MIT, see `LICENSE`.
