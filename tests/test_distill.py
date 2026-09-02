import json

from atlas_recall.distill import distill, clean_slug, find_duplicates
from atlas_recall.apply import apply_ops


def test_clean_slug_strips_markdown_cruft():
    assert clean_slug("5. **Strict MCP Config**") == "strict-mcp-config"
    assert clean_slug("") == "memory"


def test_distill_empty_body_is_skipped(cfg, capsys):
    ops = distill([{"name": "x", "description": "d", "body": "   "}], cfg)
    assert ops == []
    assert "empty body" in capsys.readouterr().err


def test_distill_add_with_advisory_duplicate(cfg):
    ops = distill(
        [{"description": "another note about descaling the espresso machine weekly", "body": "Body text."}],
        cfg,
    )
    assert len(ops) == 1
    assert ops[0]["op"] == "ADD"
    assert ops[0]["matched_memory"] is not None  # advisory nearest neighbor


def test_distill_update_requires_exact_target(cfg, capsys):
    ops = distill(
        [{"description": "d", "body": "b", "updates": "does-not-exist"}],
        cfg,
    )
    assert ops == []
    assert "not found" in capsys.readouterr().err


def test_distill_never_crashes_with_dense_enabled_but_unreachable(cfg):
    cfg.dense_enabled = True
    ops = distill([{"description": "descale rule", "body": "body"}], cfg)
    assert len(ops) == 1  # degraded to BM25, still worked


def test_find_duplicates_bm25_fallback_when_dense_off(cfg):
    hits = find_duplicates("descale weekly buildup", cfg)
    assert hits
    assert hits[0]["name"] == "rule-descale-weekly"


def test_find_duplicates_reports_priority_not_type(cfg):
    """A note's priority (hard-rule|high|normal) and a candidate's type
    (feedback|project|user|reference) are disjoint vocabularies -- the
    BM25 dedup hit dict must label the value it actually carries."""
    hits = find_duplicates("descale weekly buildup", cfg)
    assert hits
    assert "priority" in hits[0]
    assert hits[0]["priority"] == "hard-rule"
    assert "type" not in hits[0]


def test_apply_never_writes_without_confirmation(cfg, tmp_path):
    ops = [{
        "op": "ADD", "name": "new-note", "type": "reference",
        "description": "d", "body": "body text", "matched_memory": None, "distance": None,
    }]
    summary = apply_ops(ops, cfg, auto_confirm=lambda op: False)
    assert summary == {"applied": [], "skipped": ["new-note"]}
    assert not (tmp_path / "notes" / "new-note.md").exists()


def test_apply_writes_only_on_explicit_yes(cfg, tmp_path):
    ops = [{
        "op": "ADD", "name": "new-note", "type": "reference",
        "description": "d", "body": "body text", "matched_memory": None, "distance": None,
    }]
    summary = apply_ops(ops, cfg, auto_confirm=lambda op: True)
    assert summary["applied"] == ["new-note"]
    written = tmp_path / "notes" / "new-note.md"
    assert written.exists()
    assert "body text" in written.read_text()


def test_apply_add_refuses_to_overwrite_existing_note(cfg, tmp_path):
    existing = tmp_path / "notes" / "existing.md"
    existing.write_text("original content", encoding="utf-8")
    ops = [{
        "op": "ADD", "name": "existing", "type": "reference",
        "description": "d", "body": "replacement", "matched_memory": None, "distance": None,
    }]
    import pytest

    with pytest.raises(ValueError, match="overwrite"):
        apply_ops(ops, cfg, auto_confirm=lambda op: True)
    assert existing.read_text(encoding="utf-8") == "original content"


def test_apply_add_refuses_path_escaping_notes_dir(cfg, tmp_path):
    """CWE-22: an ADD op with a dirty/traversal name must never write
    outside notes_dir, and must raise rather than silently sanitize and
    write somewhere unexpected inside it."""
    import pytest

    ops = [{
        "op": "ADD", "name": "../pwn/escaped", "type": "reference",
        "description": "d", "body": "body text", "matched_memory": None, "distance": None,
    }]
    with pytest.raises(ValueError):
        apply_ops(ops, cfg, auto_confirm=lambda op: True)
    assert not (tmp_path / "pwn").exists()
    assert not any(p.name == "escaped.md" for p in (tmp_path / "notes").rglob("*.md"))


def test_apply_update_ignores_forged_matched_memory_path(cfg, tmp_path):
    """UPDATE must re-resolve the target BY NAME against the live corpus,
    never trust matched_memory['path'] verbatim -- a forged absolute path
    must not be touched."""
    import pytest

    outside = tmp_path / "pwned.txt"
    outside.write_text("untouched")

    ops = [{
        "op": "UPDATE", "name": "rule-descale-weekly", "type": "feedback",
        "description": "d", "body": "new body text",
        "matched_memory": {"name": "rule-descale-weekly", "path": str(outside)},
        "distance": None,
    }]
    apply_ops(ops, cfg, auto_confirm=lambda op: True)
    assert outside.read_text() == "untouched"
    real_note = tmp_path / "notes" / "rule-descale-weekly.md"
    assert "new body text" in real_note.read_text()


def test_apply_update_raises_when_target_no_longer_exists(cfg):
    import pytest

    ops = [{
        "op": "UPDATE", "name": "ghost", "type": "feedback",
        "description": "d", "body": "new body text",
        "matched_memory": {"name": "note-that-was-deleted", "path": "/tmp/whatever.md"},
        "distance": None,
    }]
    with pytest.raises(ValueError):
        apply_ops(ops, cfg, auto_confirm=lambda op: True)


def test_apply_update_carries_forward_existing_priority(cfg, tmp_path):
    """A hard-rule note must stay hard-rule after an approved UPDATE --
    apply.py must not hardcode priority: normal on rewrite."""
    ops = [{
        "op": "UPDATE", "name": "rule-descale-weekly", "type": "feedback",
        "description": "d", "body": "rewritten body",
        "matched_memory": {"name": "rule-descale-weekly", "path": "/tmp/ignored.md"},
        "distance": None,
    }]
    apply_ops(ops, cfg, auto_confirm=lambda op: True)
    real_note = tmp_path / "notes" / "rule-descale-weekly.md"
    text = real_note.read_text()
    assert "priority: hard-rule" in text
    assert "rewritten body" in text


def test_apply_describe_prints_resolved_destination(cfg, tmp_path, capsys):
    """The human approving needs to see where an op actually writes --
    for BOTH ADD and UPDATE -- in the pre-approval describe line, not
    only after the fact in the '-> wrote' confirmation. Declining every
    op (auto_confirm=False) means the only way these paths can appear in
    stdout is via _describe itself."""
    ops = [
        {
            "op": "ADD", "name": "new-note", "type": "reference",
            "description": "d", "body": "body text", "matched_memory": None, "distance": None,
        },
        {
            "op": "UPDATE", "name": "rule-descale-weekly", "type": "feedback",
            "description": "d", "body": "rewritten body",
            "matched_memory": {"name": "rule-descale-weekly", "path": "/tmp/ignored.md"},
            "distance": None,
        },
    ]
    apply_ops(ops, cfg, auto_confirm=lambda op: False)
    out = capsys.readouterr().out
    assert str(tmp_path / "notes" / "new-note.md") in out
    assert str(tmp_path / "notes" / "rule-descale-weekly.md") in out
    assert "/tmp/ignored.md" not in out
