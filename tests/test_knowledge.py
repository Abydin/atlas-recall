from pathlib import Path
import os
import time

import pytest

from atlas_recall import knowledge


def _fresh_conn(tmp_path):
    return knowledge.connect(tmp_path / "recall.db", fresh=True)


def test_index_pass_discovers_all_notes(notes_dir, tmp_path):
    conn = _fresh_conn(tmp_path)
    stats = knowledge.run_index_pass(conn, Path(notes_dir), full=True)
    assert stats["discovered"] == 5
    assert stats["changed"] == 5


def test_find_matches_body_text(notes_dir, tmp_path):
    conn = _fresh_conn(tmp_path)
    knowledge.run_index_pass(conn, Path(notes_dir), full=True)
    hits = knowledge.find(conn, "descale")
    assert any(h["id"] == "rule-descale-weekly" for h in hits)


def test_node_reports_wikilink_edge(notes_dir, tmp_path):
    conn = _fresh_conn(tmp_path)
    knowledge.run_index_pass(conn, Path(notes_dir), full=True)
    n = knowledge.node(conn, "reference-bike-maintenance-log")
    assert n is not None
    dsts = {e["dst"] for e in n["edges"]}
    assert "gear-inventory" in dsts
    assert all(not e["divergent"] for e in n["edges"])


def test_verify_flags_broken_wikilink(notes_dir, tmp_path):
    # Add a note with a link to a note that doesn't exist.
    Path(notes_dir, "dangling.md").write_text(
        "---\nname: dangling\n---\n\nSee [[nonexistent-target]] for details.\n"
    )
    conn = _fresh_conn(tmp_path)
    knowledge.run_index_pass(conn, Path(notes_dir), full=True)
    result = knowledge.verify(conn, limit=50)
    assert result["divergent"] >= 1
    assert any(e["dst"] == "nonexistent-target" for e in result["divergent_edges"])


def test_trace_follows_links_two_hops(notes_dir, tmp_path):
    conn = _fresh_conn(tmp_path)
    knowledge.run_index_pass(conn, Path(notes_dir), full=True)
    rows = knowledge.trace(conn, "rule-check-tire-pressure", depth=2)
    dsts = {r["dst"] for r in rows}
    assert "rule-descale-weekly" in dsts


def test_map_topic_returns_hits_with_neighbors(notes_dir, tmp_path):
    conn = _fresh_conn(tmp_path)
    knowledge.run_index_pass(conn, Path(notes_dir), full=True)
    result = knowledge.map_topic(conn, "maintenance log")
    assert result["hits"]
    assert result["hits"][0]["id"] == "reference-bike-maintenance-log"
    assert result["hits"][0]["neighbors"]


def test_incremental_index_detects_edits_within_one_second(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    note = notes / "note.md"
    note.write_text("---\nname: note\n---\n\nfirst", encoding="utf-8")
    conn = _fresh_conn(tmp_path)
    knowledge.run_index_pass(conn, notes, full=False)

    note.write_text("---\nname: note\n---\n\nsecond", encoding="utf-8")
    stats = knowledge.run_index_pass(conn, notes, full=False)

    assert stats["changed"] == 1
    assert conn.execute("SELECT body FROM docs WHERE id = 'note'").fetchone()[0] == "\nsecond"


def test_incremental_index_removes_old_id_when_frontmatter_name_changes(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    note = notes / "note.md"
    note.write_text("---\nname: old-id\n---\n\nfirst", encoding="utf-8")
    conn = _fresh_conn(tmp_path)
    knowledge.run_index_pass(conn, notes, full=False)

    note.write_text("---\nname: new-id\n---\n\nsecond", encoding="utf-8")
    # Make this test independent of filesystem timestamp granularity while
    # the prior test exercises the no-sleep same-second path.
    now = time.time() + 2
    os.utime(note, (now, now))
    knowledge.run_index_pass(conn, notes, full=False)

    assert conn.execute("SELECT id FROM docs ORDER BY id").fetchall() == [("new-id",)]


def test_index_rejects_duplicate_frontmatter_ids(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "one.md").write_text("---\nname: same\n---\n\none", encoding="utf-8")
    (notes / "two.md").write_text("---\nname: same\n---\n\ntwo", encoding="utf-8")
    conn = _fresh_conn(tmp_path)

    with pytest.raises(ValueError, match="duplicate note id"):
        knowledge.run_index_pass(conn, notes, full=True)
