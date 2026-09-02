import json

import pytest

from atlas_recall import install
from atlas_recall.cli import main
from atlas_recall import __version__


def test_version_flag_prints_version_and_exits(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert __version__ in out


def test_install_command_writes_through_cli(tmp_path, monkeypatch, capsys):
    cfg_path = tmp_path / "mcp.json"
    monkeypatch.setitem(install.CLIENTS["cursor"], "path_fn", lambda: cfg_path)

    code = main(["install", "--client", "cursor"])

    assert code == 0
    data = json.loads(cfg_path.read_text())
    assert install.SERVER_NAME in data["mcpServers"]


def test_install_command_dry_run_writes_nothing_through_cli(tmp_path, monkeypatch):
    cfg_path = tmp_path / "mcp.json"
    monkeypatch.setitem(install.CLIENTS["cursor"], "path_fn", lambda: cfg_path)

    code = main(["install", "--client", "cursor", "--dry-run"])

    assert code == 0
    assert not cfg_path.exists()


def test_uninstall_command_removes_through_cli(tmp_path, monkeypatch):
    cfg_path = tmp_path / "mcp.json"
    monkeypatch.setitem(install.CLIENTS["cursor"], "path_fn", lambda: cfg_path)
    main(["install", "--client", "cursor"])

    code = main(["uninstall", "--client", "cursor"])

    assert code == 0
    data = json.loads(cfg_path.read_text())
    assert install.SERVER_NAME not in data["mcpServers"]


def test_install_command_rejects_unknown_client():
    with pytest.raises(SystemExit):
        main(["install", "--client", "not-a-real-client"])
