"""
Configuration for atlas-recall.

Nothing in this module points at any particular person's machine, project,
or memory corpus -- every path is either passed explicitly or resolved from
a config file the user creates with `recall init`. That config file lives at
$ATLAS_RECALL_CONFIG, or ~/.config/atlas-recall/config.json by default (XDG-
style; falls back gracefully on machines without XDG_CONFIG_HOME set).

`notes_dir` is the one required setting: the directory of markdown notes to
index and retrieve from. Everything else has a sane default and works with
the dense path fully absent.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import List, Tuple


def _default_config_path() -> str:
    override = os.environ.get("ATLAS_RECALL_CONFIG")
    if override:
        return os.path.expanduser(override)
    xdg = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(xdg, "atlas-recall", "config.json")


DEFAULT_CONFIG_PATH = _default_config_path()

# The value `chroma_dir` has always defaulted to, and the value `recall
# init` -> `save_config` has therefore always round-tripped into every
# user's config.json whether or not they ever looked at it -- because
# save_config serializes the whole dataclass, not just the keys someone
# actually set. So this string can't be told apart from "the user typed
# this on purpose" by presence in the file; only by still being exactly
# this value. See resolve_chroma_dir().
CHROMA_DIR_DEFAULT = "~/.atlas-recall/chroma"


@dataclass
class Config:
    # The directory of markdown notes to index and retrieve from. Required.
    notes_dir: str = ""

    # Dense (semantic) retrieval is OFF by default -- see README "why BM25
    # first". Turn it on with `recall index --dense` (writes dense=true
    # here) once Chroma + Ollama are actually available.
    dense_enabled: bool = False
    chroma_dir: str = CHROMA_DIR_DEFAULT
    collection: str = "atlas_recall_notes"
    embed_model: str = "bge-m3"
    dense_dist_max: float = 0.48  # cosine distance ceiling; see README for how
                                  # this number was measured on the reference
                                  # corpus -- recalibrate on your own notes.

    # Optional hard-rules file, injected verbatim (loud, undiluted) at the
    # top of every hook block. Empty by default -- most users don't have one.
    rules_path: str = ""

    # Keyword -> rule-line triggers. Ships EMPTY. See examples/keyword_rules.json
    # for a worked example (someone else's personal rules) -- copy it, don't inherit it.
    keyword_rules: List[Tuple[List[str], str]] = field(default_factory=list)

    top_k: int = 3
    # Conservative admission floor on the fused RRF score -- below this,
    # top_pointers() returns nothing rather than a weak guess. Calibrated
    # for the DEFAULT (BM25-only) case: with a single ranked list feeding
    # rrf_fuse(), a rank-1 hit scores ~1/(RRF_K+1) ~= 0.0164 before recency
    # decay, so the floor has to sit comfortably under that or BM25-only
    # mode would silently admit nothing, ever. Turning on the dense path
    # only raises fused scores (two retrievers agreeing on a hit adds a
    # second 1/(RRF_K+rank) term), so this floor stays valid there too --
    # it was never the thing dense-vs-BM25-only needed different values
    # for. If you index a much larger corpus (thousands of notes) and see
    # too much getting admitted, raise this; it's a knob, not a constant.
    min_score: float = 0.01
    recency_halflife_days: float = 365.0


def resolve_config_path(path: str = None) -> str:
    """The config path `load_config(path)` will actually read from."""
    return os.path.expanduser(path or DEFAULT_CONFIG_PATH)


def load_config(path: str = None) -> Config:
    """Load config from disk; return defaults (notes_dir="") if absent."""
    path = resolve_config_path(path)
    if not os.path.isfile(path):
        return Config()
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    cfg = Config()
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def corpus_key(config_path: str, notes_dir: str) -> str:
    """Stable key identifying one (active config, notes_dir) pairing, so
    derived storage (index db, dense Chroma store) can be namespaced per
    corpus. Keying on both -- not just notes_dir -- means an explicitly
    isolated ATLAS_RECALL_CONFIG (e.g. a scratch config for a test run)
    never shares storage with the real config even in the freak case both
    happen to point at the same notes_dir."""
    cfg_norm = os.path.normpath(os.path.abspath(os.path.expanduser(config_path)))
    notes_norm = os.path.normpath(os.path.abspath(os.path.expanduser(notes_dir)))
    digest = hashlib.sha256(f"{cfg_norm}\x00{notes_norm}".encode("utf-8")).hexdigest()
    return digest[:16]


def default_corpus_dir(config_path: str, notes_dir: str) -> str:
    """Per-corpus directory under ~/.atlas-recall. Two different notes_dir
    values, or two different active configs, always get two different
    directories here -- see corpus_key()."""
    return os.path.join(
        os.path.expanduser("~/.atlas-recall"), "corpora", corpus_key(config_path, notes_dir)
    )


def resolve_chroma_dir(cfg: Config, config_path: str) -> str:
    """Where the dense Chroma store lives for this corpus. A chroma_dir the
    user actually customized is always honoured verbatim. One still sitting
    at CHROMA_DIR_DEFAULT (which is nearly every config -- see the comment
    on that constant) is treated as unset, and a per-corpus directory is
    derived instead."""
    if cfg.chroma_dir and cfg.chroma_dir != CHROMA_DIR_DEFAULT:
        return os.path.expanduser(cfg.chroma_dir)
    return os.path.join(default_corpus_dir(config_path, cfg.notes_dir), "chroma")


def save_config(cfg: Config, path: str = None) -> str:
    path = path or DEFAULT_CONFIG_PATH
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, indent=2)
        fh.write("\n")
    return path
