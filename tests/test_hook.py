import json

from atlas_recall.hook import build_context, run_hook


def test_rules_path_missing_warns_but_still_exits_clean(cfg, capsys):
    cfg.rules_path = "/nonexistent/rules/path/does-not-exist.md"
    ctx = build_context("anything at all", cfg)
    err = capsys.readouterr().err
    assert "/nonexistent/rules/path/does-not-exist.md" in err
    assert "HARD RULES NOT INJECTED" in err
    assert "HARD RULES" not in ctx  # still degrades to no block, silently for the transcript


def test_run_hook_still_exits_0_and_emits_json_when_rules_path_bad(cfg, monkeypatch):
    import atlas_recall.hook as hook_mod

    monkeypatch.setattr(hook_mod, "load_config", lambda: cfg)
    cfg.rules_path = "/nonexistent/rules/path/does-not-exist.md"
    stdin = json.dumps({"prompt": "anything at all"})

    rc = run_hook(stdin)
    assert rc == 0
