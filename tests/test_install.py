import json
import subprocess

import pytest

from atlas_recall import install


# --------------------------------------------------------------------------
# mcpServers-shaped clients (cursor here as the representative -- same code
# path as claude-desktop/windsurf/cline)
# --------------------------------------------------------------------------
def test_install_mcp_creates_file_with_our_entry(tmp_path):
    cfg_path = tmp_path / "mcp.json"
    result = install.install("cursor", config_path=cfg_path)
    assert result["success"] is True
    assert result["changed"] is True
    data = json.loads(cfg_path.read_text())
    assert install.SERVER_NAME in data["mcpServers"]


def test_install_mcp_is_idempotent(tmp_path):
    cfg_path = tmp_path / "mcp.json"
    install.install("cursor", config_path=cfg_path)
    first = json.loads(cfg_path.read_text())

    result2 = install.install("cursor", config_path=cfg_path)
    second = json.loads(cfg_path.read_text())

    assert result2["success"] is True
    assert result2["changed"] is False
    assert first == second
    assert len(second["mcpServers"]) == 1


def test_install_mcp_merge_preserves_existing_keys(tmp_path):
    cfg_path = tmp_path / "mcp.json"
    cfg_path.write_text(json.dumps({
        "mcpServers": {"some-other-server": {"command": "foo", "args": []}},
        "unrelatedTopLevelKey": {"nested": True},
    }))

    install.install("cursor", config_path=cfg_path)

    data = json.loads(cfg_path.read_text())
    assert "some-other-server" in data["mcpServers"]
    assert data["mcpServers"]["some-other-server"] == {"command": "foo", "args": []}
    assert install.SERVER_NAME in data["mcpServers"]
    assert data["unrelatedTopLevelKey"] == {"nested": True}


def test_install_mcp_creates_backup_of_existing_file(tmp_path):
    cfg_path = tmp_path / "mcp.json"
    cfg_path.write_text(json.dumps({"mcpServers": {}}))

    result = install.install("cursor", config_path=cfg_path)

    assert result["backup"] is not None
    backup_path = tmp_path / result["backup"].rsplit("/", 1)[-1]
    assert backup_path.exists()
    assert json.loads(backup_path.read_text()) == {"mcpServers": {}}


def test_install_mcp_no_backup_when_file_did_not_exist(tmp_path):
    cfg_path = tmp_path / "mcp.json"
    result = install.install("cursor", config_path=cfg_path)
    assert result["backup"] is None


def test_install_dry_run_writes_nothing(tmp_path):
    cfg_path = tmp_path / "mcp.json"
    result = install.install("cursor", dry_run=True, config_path=cfg_path)

    assert result["success"] is True
    assert result["changed"] is True
    assert result["dry_run"] is True
    assert not cfg_path.exists()
    # no stray backup or temp file either
    assert list(tmp_path.iterdir()) == []


def test_install_dry_run_on_existing_file_leaves_it_untouched(tmp_path):
    cfg_path = tmp_path / "mcp.json"
    original = {"mcpServers": {"other": {"command": "x", "args": []}}}
    cfg_path.write_text(json.dumps(original))

    install.install("cursor", dry_run=True, config_path=cfg_path)

    assert json.loads(cfg_path.read_text()) == original
    # dry run must not leave a backup file behind either
    assert list(tmp_path.iterdir()) == [cfg_path]


def test_install_refuses_malformed_json_and_does_not_write(tmp_path):
    cfg_path = tmp_path / "mcp.json"
    cfg_path.write_text("{ not valid json ][")
    before = cfg_path.read_text()

    result = install.install("cursor", config_path=cfg_path)

    assert result["success"] is False
    assert str(cfg_path) in result["error"]
    assert cfg_path.read_text() == before
    # no backup written either -- we never got past the refusal
    assert list(tmp_path.iterdir()) == [cfg_path]


def test_uninstall_removes_only_our_entry(tmp_path):
    cfg_path = tmp_path / "mcp.json"
    cfg_path.write_text(json.dumps({
        "mcpServers": {
            "some-other-server": {"command": "foo", "args": []},
        },
    }))
    install.install("cursor", config_path=cfg_path)
    data = json.loads(cfg_path.read_text())
    assert set(data["mcpServers"]) == {"some-other-server", install.SERVER_NAME}

    result = install.uninstall("cursor", config_path=cfg_path)

    assert result["success"] is True
    assert result["changed"] is True
    data = json.loads(cfg_path.read_text())
    assert set(data["mcpServers"]) == {"some-other-server"}


def test_uninstall_when_not_installed_is_a_noop(tmp_path):
    cfg_path = tmp_path / "mcp.json"
    cfg_path.write_text(json.dumps({"mcpServers": {}}))

    result = install.uninstall("cursor", config_path=cfg_path)

    assert result["success"] is True
    assert result["changed"] is False


def test_uninstall_dry_run_writes_nothing(tmp_path):
    cfg_path = tmp_path / "mcp.json"
    install.install("cursor", config_path=cfg_path)
    before = cfg_path.read_text()

    result = install.uninstall("cursor", dry_run=True, config_path=cfg_path)

    assert result["changed"] is True
    assert result["dry_run"] is True
    assert cfg_path.read_text() == before


# --------------------------------------------------------------------------
# Claude Code: hooks.UserPromptSubmit shape, not mcpServers
# --------------------------------------------------------------------------
def test_install_claude_code_hook_creates_expected_shape(tmp_path):
    cfg_path = tmp_path / "settings.json"
    result = install.install("claude-code", config_path=cfg_path)

    assert result["success"] is True
    data = json.loads(cfg_path.read_text())
    commands = [
        h["command"]
        for group in data["hooks"]["UserPromptSubmit"]
        for h in group["hooks"]
    ]
    assert commands == [install.HOOK_COMMAND]


def test_install_claude_code_hook_is_idempotent(tmp_path):
    cfg_path = tmp_path / "settings.json"
    install.install("claude-code", config_path=cfg_path)
    install.install("claude-code", config_path=cfg_path)

    data = json.loads(cfg_path.read_text())
    commands = [
        h["command"]
        for group in data["hooks"]["UserPromptSubmit"]
        for h in group["hooks"]
        if h.get("command") == install.HOOK_COMMAND
    ]
    assert commands == [install.HOOK_COMMAND]


def test_install_claude_code_hook_preserves_other_hook_events(tmp_path):
    cfg_path = tmp_path / "settings.json"
    cfg_path.write_text(json.dumps({
        "hooks": {
            "PreToolUse": [{"hooks": [{"type": "command", "command": "some-other-hook"}]}],
        },
        "otherSetting": "keep-me",
    }))

    install.install("claude-code", config_path=cfg_path)

    data = json.loads(cfg_path.read_text())
    assert data["hooks"]["PreToolUse"][0]["hooks"][0]["command"] == "some-other-hook"
    assert data["otherSetting"] == "keep-me"
    assert install.HOOK_COMMAND in [
        h["command"] for g in data["hooks"]["UserPromptSubmit"] for h in g["hooks"]
    ]


def test_install_claude_code_hook_preserves_other_userpromptsubmit_hooks(tmp_path):
    cfg_path = tmp_path / "settings.json"
    cfg_path.write_text(json.dumps({
        "hooks": {
            "UserPromptSubmit": [
                {"hooks": [{"type": "command", "command": "some-other-prompt-hook"}]},
            ],
        },
    }))

    install.install("claude-code", config_path=cfg_path)
    data = json.loads(cfg_path.read_text())
    commands = {h["command"] for g in data["hooks"]["UserPromptSubmit"] for h in g["hooks"]}
    assert commands == {"some-other-prompt-hook", install.HOOK_COMMAND}


def test_uninstall_claude_code_hook_removes_only_ours(tmp_path):
    cfg_path = tmp_path / "settings.json"
    cfg_path.write_text(json.dumps({
        "hooks": {
            "UserPromptSubmit": [
                {"hooks": [{"type": "command", "command": "some-other-prompt-hook"}]},
            ],
        },
    }))
    install.install("claude-code", config_path=cfg_path)

    result = install.uninstall("claude-code", config_path=cfg_path)

    assert result["changed"] is True
    data = json.loads(cfg_path.read_text())
    commands = {h["command"] for g in data["hooks"]["UserPromptSubmit"] for h in g["hooks"]}
    assert commands == {"some-other-prompt-hook"}


def test_uninstall_claude_code_hook_drops_now_empty_group(tmp_path):
    cfg_path = tmp_path / "settings.json"
    install.install("claude-code", config_path=cfg_path)

    install.uninstall("claude-code", config_path=cfg_path)

    data = json.loads(cfg_path.read_text())
    assert data["hooks"]["UserPromptSubmit"] == []


def test_claude_code_malformed_json_refuses_and_does_not_write(tmp_path):
    cfg_path = tmp_path / "settings.json"
    cfg_path.write_text("{{{not json")
    before = cfg_path.read_text()

    result = install.install("claude-code", config_path=cfg_path)

    assert result["success"] is False
    assert cfg_path.read_text() == before


# --------------------------------------------------------------------------
# resolve_server_command
# --------------------------------------------------------------------------
def test_resolve_server_command_falls_back_when_not_on_path(monkeypatch):
    monkeypatch.setattr(install.shutil, "which", lambda name: None)
    entry = install.resolve_server_command()
    assert entry["args"] == ["-m", "atlas_recall.server"]
    assert entry["command"]  # some interpreter path


def test_resolve_server_command_prefers_console_script(monkeypatch):
    monkeypatch.setattr(install.shutil, "which", lambda name: "/usr/local/bin/atlas-recall-mcp")
    entry = install.resolve_server_command()
    assert entry == {"command": "/usr/local/bin/atlas-recall-mcp", "args": []}


# --------------------------------------------------------------------------
# Unknown client
# --------------------------------------------------------------------------
def test_install_unknown_client_fails_cleanly(tmp_path):
    result = install.install("not-a-real-client", config_path=tmp_path / "x.json")
    assert result["success"] is False
    assert "unknown client" in result["error"]


# --------------------------------------------------------------------------
# Codex: driven through the `codex` CLI, not a config_path override --
# mock shutil.which + subprocess.run so this never shells out for real.
# --------------------------------------------------------------------------
def test_codex_install_skips_when_binary_absent(monkeypatch):
    monkeypatch.setattr(install.shutil, "which", lambda name: None)
    result = install.install("codex")
    assert result["success"] is False
    assert "codex CLI not found" in result["error"]


def test_codex_install_calls_mcp_add_when_not_present(monkeypatch, tmp_path):
    monkeypatch.setattr(install.shutil, "which", lambda name: "/usr/local/bin/codex" if name == "codex" else None)
    monkeypatch.setattr(install, "codex_config_path", lambda: tmp_path / "config.toml")

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:3] == ["codex", "mcp", "list"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:3] == ["codex", "mcp", "add"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="added", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(install.subprocess, "run", fake_run)

    result = install.install("codex")

    assert result["success"] is True
    assert result["changed"] is True
    assert any(c[:3] == ["codex", "mcp", "add"] for c in calls)


def test_codex_install_is_idempotent_when_already_present(monkeypatch, tmp_path):
    monkeypatch.setattr(install.shutil, "which", lambda name: "/usr/local/bin/codex" if name == "codex" else None)
    monkeypatch.setattr(install, "codex_config_path", lambda: tmp_path / "config.toml")

    def fake_run(cmd, **kwargs):
        if cmd[:3] == ["codex", "mcp", "list"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{install.SERVER_NAME}\n", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(install.subprocess, "run", fake_run)

    result = install.install("codex")

    assert result["success"] is True
    assert result["changed"] is False


def test_codex_uninstall_removes_when_present(monkeypatch, tmp_path):
    monkeypatch.setattr(install.shutil, "which", lambda name: "/usr/local/bin/codex" if name == "codex" else None)
    monkeypatch.setattr(install, "codex_config_path", lambda: tmp_path / "config.toml")
    (tmp_path / "config.toml").write_text("[mcp_servers.atlas-recall]\n")

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:3] == ["codex", "mcp", "list"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{install.SERVER_NAME}\n", stderr="")
        if cmd[:3] == ["codex", "mcp", "remove"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="removed", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(install.subprocess, "run", fake_run)

    result = install.uninstall("codex")

    assert result["success"] is True
    assert result["changed"] is True
    assert any(c[:3] == ["codex", "mcp", "remove"] for c in calls)
