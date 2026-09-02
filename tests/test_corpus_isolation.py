"""
Reproduction + regression coverage for the corpus-collision bug: the index
db path used to be derived from chroma_dir's parent, and chroma_dir
defaulted to a single fixed path (~/.atlas-recall/chroma) no matter which
config was active or what notes_dir pointed at. Two different corpora on
one machine -- or a config deliberately isolated via ATLAS_RECALL_CONFIG
for testing -- silently shared (and clobbered) one index.

Every test here runs under an isolated HOME (see the `isolated_home`
fixture) so nothing ever touches the real ~/.atlas-recall.
"""
import os
import textwrap
from pathlib import Path

import pytest

from atlas_recall import cli, knowledge
from atlas_recall.config import Config, resolve_chroma_dir


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """A fake HOME so ~/.atlas-recall resolves under tmp_path, never the
    real user's home directory."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


def _make_notes(dirpath: Path, name: str, body: str):
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / f"{name}.md").write_text(
        textwrap.dedent(f"""---
        name: {name}
        description: test note
        priority: normal
        ---

        {body}
        """),
        encoding="utf-8",
    )


def test_two_configs_with_different_notes_dirs_do_not_collide(isolated_home, tmp_path):
    """THE BUG: two distinct configs, each with its own notes_dir, used to
    resolve to the identical index db path (chroma_dir was never set
    explicitly by either, so both fell back to the same
    ~/.atlas-recall/chroma -> ~/.atlas-recall/recall.db)."""
    notes_a = tmp_path / "corpus-a"
    notes_b = tmp_path / "corpus-b"
    _make_notes(notes_a, "note-a", "corpus A content")
    _make_notes(notes_b, "note-b", "corpus B content")

    config_a = str(tmp_path / "config-a.json")
    config_b = str(tmp_path / "config-b.json")

    cfg_a = Config(notes_dir=str(notes_a))  # chroma_dir left at the default
    cfg_b = Config(notes_dir=str(notes_b))  # chroma_dir left at the default

    db_a = cli._db_path(cfg_a, config_path=config_a)
    db_b = cli._db_path(cfg_b, config_path=config_b)

    assert db_a != db_b, (
        "two different notes_dir values resolved to the same index db path "
        f"({db_a}) -- indexing corpus B will clobber corpus A's index"
    )

    # Build both indexes for real and confirm they hold their own docs only.
    conn_a = knowledge.connect(db_a, notes_dir=cfg_a.notes_dir)
    knowledge.run_index_pass(conn_a, Path(cfg_a.notes_dir), full=True)
    conn_a.close()

    conn_b = knowledge.connect(db_b, notes_dir=cfg_b.notes_dir)
    knowledge.run_index_pass(conn_b, Path(cfg_b.notes_dir), full=True)
    conn_b.close()

    conn_a = knowledge.connect(db_a, notes_dir=cfg_a.notes_dir)
    ids_a = knowledge.all_doc_ids(conn_a)
    conn_a.close()

    conn_b = knowledge.connect(db_b, notes_dir=cfg_b.notes_dir)
    ids_b = knowledge.all_doc_ids(conn_b)
    conn_b.close()

    assert ids_a == {"note-a"}
    assert ids_b == {"note-b"}


def test_same_config_and_notes_dir_reuses_its_own_index(isolated_home, tmp_path):
    """Negative control: a single config used twice must resolve to the
    SAME db path (and therefore reuse the index), not build a fresh one
    every run."""
    notes = tmp_path / "corpus"
    _make_notes(notes, "note-1", "content")
    config_path = str(tmp_path / "config.json")
    cfg = Config(notes_dir=str(notes))

    first = cli._db_path(cfg, config_path=config_path)
    second = cli._db_path(cfg, config_path=config_path)
    assert first == second

    conn = knowledge.connect(first, notes_dir=cfg.notes_dir)
    knowledge.run_index_pass(conn, Path(cfg.notes_dir), full=True)
    conn.close()
    assert first.exists()

    # Asking again after the db already exists on disk must return the
    # same path (not silently rebuild elsewhere).
    third = cli._db_path(cfg, config_path=config_path)
    assert third == first


def test_explicit_chroma_dir_is_still_honoured(isolated_home, tmp_path):
    """A user-customized chroma_dir must still control where the dense
    store lives -- corpus namespacing only kicks in for the untouched
    default."""
    notes = tmp_path / "corpus"
    _make_notes(notes, "note-1", "content")
    custom_chroma = tmp_path / "my-custom-chroma"
    cfg = Config(notes_dir=str(notes), chroma_dir=str(custom_chroma))

    resolved = resolve_chroma_dir(cfg, str(tmp_path / "config.json"))
    assert resolved == str(custom_chroma)


def test_default_chroma_dir_is_namespaced_per_corpus(isolated_home, tmp_path):
    """Two configs left at the (serialized-but-never-customized) default
    chroma_dir must not resolve to the same Chroma directory either."""
    notes_a = tmp_path / "corpus-a"
    notes_b = tmp_path / "corpus-b"
    cfg_a = Config(notes_dir=str(notes_a))
    cfg_b = Config(notes_dir=str(notes_b))

    chroma_a = resolve_chroma_dir(cfg_a, str(tmp_path / "config-a.json"))
    chroma_b = resolve_chroma_dir(cfg_b, str(tmp_path / "config-b.json"))
    assert chroma_a != chroma_b


def test_mismatched_corpus_is_refused_not_silently_served(isolated_home, tmp_path):
    """Guard: opening an existing index whose stamped notes_dir doesn't
    match the active config's notes_dir must refuse loudly, never silently
    serve the wrong corpus."""
    notes_a = tmp_path / "corpus-a"
    notes_b = tmp_path / "corpus-b"
    _make_notes(notes_a, "note-a", "corpus A content")
    _make_notes(notes_b, "note-b", "corpus B content")
    db_path = tmp_path / "shared.db"

    conn = knowledge.connect(db_path, notes_dir=str(notes_a))
    knowledge.run_index_pass(conn, notes_a, full=True)
    conn.close()

    with pytest.raises(knowledge.CorpusMismatch):
        knowledge.connect(db_path, notes_dir=str(notes_b))


def test_legacy_shared_index_migrates_when_notes_dir_matches(isolated_home, tmp_path):
    """Migration: an index built before corpus tracking existed (or by the
    0.2.0-era code that always wrote to the shared default path), whose
    stamped notes_dir matches this config's notes_dir, gets claimed and
    moved into the new per-corpus location -- not silently orphaned."""
    notes = tmp_path / "corpus"
    _make_notes(notes, "note-1", "content")
    cfg = Config(notes_dir=str(notes))  # chroma_dir left at the default

    # Simulate a pre-fix index sitting at the old shared path this cfg's
    # (default) chroma_dir would have produced under the old logic.
    legacy_path = Path(os.path.expanduser(cfg.chroma_dir)).parent / "recall.db"
    conn = knowledge.connect(legacy_path, notes_dir=cfg.notes_dir)
    knowledge.run_index_pass(conn, Path(cfg.notes_dir), full=True)
    conn.close()
    assert legacy_path.exists()

    config_path = str(tmp_path / "config.json")
    new_path = cli._db_path(cfg, config_path=config_path)

    assert new_path != legacy_path
    assert new_path.exists()
    assert not legacy_path.exists(), "legacy db should have been moved, not copied/orphaned"

    conn = knowledge.connect(new_path, notes_dir=cfg.notes_dir)
    assert knowledge.all_doc_ids(conn) == {"note-1"}
    conn.close()


def test_legacy_shared_index_left_alone_when_notes_dir_differs(isolated_home, tmp_path):
    """A legacy index at the shared path that belongs to a DIFFERENT
    notes_dir must not be claimed -- it's not this corpus's data."""
    notes_other = tmp_path / "other-corpus"
    _make_notes(notes_other, "note-other", "content")
    other_cfg = Config(notes_dir=str(notes_other))
    legacy_path = Path(os.path.expanduser(other_cfg.chroma_dir)).parent / "recall.db"
    conn = knowledge.connect(legacy_path, notes_dir=other_cfg.notes_dir)
    knowledge.run_index_pass(conn, Path(other_cfg.notes_dir), full=True)
    conn.close()

    notes_mine = tmp_path / "my-corpus"
    _make_notes(notes_mine, "note-mine", "content")
    my_cfg = Config(notes_dir=str(notes_mine))
    config_path = str(tmp_path / "config.json")

    new_path = cli._db_path(my_cfg, config_path=config_path)
    assert legacy_path.exists(), "a different corpus's legacy index must be left in place"
    assert new_path != legacy_path


def test_legacy_index_with_no_meta_is_not_silently_claimed(isolated_home, tmp_path):
    """A true 0.1.0-era db (built before the meta table existed at all)
    can't be verified against any notes_dir. It must be left alone, and a
    fresh index built at the new per-corpus path -- never guessed at."""
    notes = tmp_path / "corpus"
    _make_notes(notes, "note-1", "content")
    cfg = Config(notes_dir=str(notes))
    legacy_path = Path(os.path.expanduser(cfg.chroma_dir)).parent / "recall.db"

    # Build a db with the OLD schema only (no meta table) to simulate a
    # genuine pre-migration 0.1.0 index.
    import sqlite3
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(legacy_path))
    conn.execute(
        "CREATE TABLE docs (id TEXT PRIMARY KEY, type TEXT, title TEXT, "
        "description TEXT, path TEXT, mtime INTEGER, hash TEXT, body TEXT)"
    )
    conn.execute(
        "INSERT INTO docs VALUES ('x', 'note', 'X', '', '/tmp/x.md', 1, 'h', 'body')"
    )
    conn.commit()
    conn.close()

    config_path = str(tmp_path / "config.json")
    new_path = cli._db_path(cfg, config_path=config_path)

    assert legacy_path.exists(), "unverifiable legacy index must not be deleted or moved"
    assert new_path != legacy_path
    # A fresh (empty, unstamped-until-used) db is fine to build at the new path.


def test_cmd_index_end_to_end_for_two_corpora(isolated_home, tmp_path, monkeypatch):
    """Drives the actual shipped composition (`recall index`, i.e.
    cli.cmd_index) for two different corpora under the same default config
    path -- not the individual pieces by hand -- and confirms neither run's
    index clobbers the other's, through the lock/migrate/connect/index path
    users actually exercise."""
    import atlas_recall.cli as cli_mod

    notes_a = tmp_path / "corpus-a"
    notes_b = tmp_path / "corpus-b"
    _make_notes(notes_a, "note-a", "corpus A content")
    _make_notes(notes_b, "note-b", "corpus B content")

    cfg_a = Config(notes_dir=str(notes_a))
    cfg_b = Config(notes_dir=str(notes_b))
    configs = iter([cfg_a, cfg_b])
    monkeypatch.setattr(cli_mod, "load_config", lambda: next(configs))

    class Args:
        rebuild = False
        full = False
        dense = False

    assert cli_mod.cmd_index(Args()) == 0
    assert cli_mod.cmd_index(Args()) == 0

    db_a = cli_mod._db_path(cfg_a)
    db_b = cli_mod._db_path(cfg_b)
    assert db_a != db_b

    conn = knowledge.connect(db_a, notes_dir=cfg_a.notes_dir)
    assert knowledge.all_doc_ids(conn) == {"note-a"}
    conn.close()

    conn = knowledge.connect(db_b, notes_dir=cfg_b.notes_dir)
    assert knowledge.all_doc_ids(conn) == {"note-b"}
    conn.close()
