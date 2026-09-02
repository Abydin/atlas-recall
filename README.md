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

Two ways to wire it into an agent, both driven by `recall install` (no
copy-pasting JSON by hand): an **MCP server** (`atlas-recall-mcp`) that
exposes retrieval and the knowledge graph as structured tools over local
stdio, and a **Claude Code hook** (`recall hook`) that injects retrieval
results straight into the prompt on every turn. Any other agent that can
run a shell command gets the same engine by calling the `recall` CLI
directly.

## Supported clients

| Client | How | Config written |
|---|---|---|
| Claude Code | `UserPromptSubmit` hook | `~/.claude/settings.json` |
| Claude Desktop | MCP (stdio) | `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) |
| Cursor | MCP (stdio) | `~/.cursor/mcp.json` |
| Windsurf | MCP (stdio) | `~/.codeium/windsurf/mcp_config.json` |
| Cline (VS Code) | MCP (stdio) | `.../globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json` |
| Codex CLI | MCP (stdio), via `codex mcp add` | `~/.codex/config.toml` |

The Cline path assumes standard VS Code; `recall install --client cline`
refuses (rather than silently creating a file Cline will never read) if
that globalStorage directory doesn't exist -- VS Code Insiders and
VSCodium use a different one, so wire it in by hand in that case.

Every path above was verified against that client's own docs, not
guessed -- `recall install --client <name>` writes to it directly (merges,
backs up, idempotent, `--dry-run` to preview). Zed also takes a local
stdio MCP server the same way the others do; it just isn't wired into
`recall install` yet, so point it at `atlas-recall-mcp` by hand.

**Not supported: ChatGPT.** ChatGPT's web and desktop connectors need a
remote HTTP MCP endpoint, not a local stdio process -- exposing a user's
private notes over a network endpoint is a deployment decision for
whoever runs this, not something this package does on its own, so it
isn't shipped here.

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

The dense retrieval path and the MCP server are both separate extras, not
hard dependencies:

```
pip install "atlas-recall[dense]"   # Chroma + Ollama semantic retrieval
pip install "atlas-recall[mcp]"     # atlas-recall-mcp -- needs Python 3.10+
```

The base package (everything except the MCP server) stays on a Python
3.9 floor; the `mcp` SDK itself requires 3.10+.

## The three commands, with real output

```
$ recall init ~/notes
Wrote config to ~/.config/atlas-recall/config.json
notes_dir = ~/notes

Next: recall install --client <claude-code|claude-desktop|cursor|windsurf|cline|codex>
      wires this into your AI client's config directly (merges, backs up, idempotent;
      pass --dry-run to preview first).

Then: recall index    (builds the search index)
      recall query "..."   (see what would be injected)

$ recall install --client claude-code
{
  "success": true,
  "path": "/Users/you/.claude/settings.json",
  "changed": true,
  "dry_run": false,
  "backup": null,
  "client": "claude-code"
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
`recall init` never mentions either. `recall hook` (the piece that
actually wires into Claude Code) reads a `UserPromptSubmit` payload on
stdin and emits the same block as `hookSpecificOutput.additionalContext`;
`recall install --client claude-code` writes that wiring for you. Prefer
to do it by hand? `recall init` still prints the raw hook block as a
fallback -- paste it into `settings.json` yourself, merging the `"hooks"`
key if one already exists there.

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

## The MCP server

`atlas-recall-mcp` (needs `pip install "atlas-recall[mcp]"` and Python
3.10+) exposes the same engine as MCP tools over stdio, so a client calls
them directly instead of shelling out to `recall`. Every tool returns
structured JSON -- fields, not formatted prose -- so the calling model can
act on `pointers[i].score` or `edges[i].divergent` rather than re-parsing
English.

| Tool | What it does |
|---|---|
| `recall_query` | Hybrid retrieval for a text -- the same ranking the Claude Code hook injects automatically. |
| `recall_search` | Full-text (FTS5/BM25) search, optionally filtered by `doc_type`. |
| `recall_list_notes` | List indexed notes, newest-modified first, optionally filtered by type. |
| `recall_node` | One note by id: full body plus its 1-hop edges. |
| `recall_trace` | Follow the wikilink/frontmatter edge graph outward from a note. |
| `recall_map` | The shape of a topic: search hits plus each hit's neighbors. |
| `recall_verify` | Report edges whose target note no longer exists. |

`recall install --client <claude-desktop|cursor|windsurf|cline|codex>`
wires this in directly; see "Supported clients" above for exactly which
config file it writes.

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
| `server.py` / `_mcp_entry.py` | The MCP server (`atlas-recall-mcp`) and its optional-extra import guard. |
| `install.py` | `recall install`/`uninstall`: merge, backup, atomic write into a client's config. |
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
