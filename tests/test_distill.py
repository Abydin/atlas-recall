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
        [{"description": "another note about em dashes in outward text", "body": "Body text."}],
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
    ops = distill([{"description": "em dash rule", "body": "body"}], cfg)
    assert len(ops) == 1  # degraded to BM25, still worked


def test_find_duplicates_bm25_fallback_when_dense_off(cfg):
    hits = find_duplicates("em dash outward", cfg)
    assert hits
    assert hits[0]["name"] == "feedback-no-em-dashes"


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
