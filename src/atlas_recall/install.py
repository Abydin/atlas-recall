"""
`recall install` / `recall uninstall`: write the Claude Code hook block or
an MCP server entry directly into a supported client's config file,
instead of making the user copy-paste JSON by hand.

Every client here has an independently-verified config path and schema
(see the README's "Supported clients" table) -- nothing in this module
guesses. A client whose format could not be confirmed against its own
docs is simply not registered in CLIENTS.

Every write follows the same shape: read-existing-or-empty -> merge into
it (never overwrite unrelated keys) -> back up the existing file (if any)
-> atomic write (temp file + os.replace, so a crash mid-write can't leave
a half-written config). Idempotent: running install twice leaves exactly
one of our entries, not two. `dry_run=True` computes and returns what
would change without touching disk at all -- no write, no backup.

A malformed existing config file is a hard refusal, not a best-effort
merge: this module raises rather than silently discarding content it
can't parse. Better a clear error naming the path than a clobbered file.

Codex CLI is the one client configured through its own first-party `codex
mcp add`/`codex mcp list` subcommands rather than by this module writing
TOML directly -- `~/.codex/config.toml`'s `[mcp_servers.*]` tables are
real syntax to hand-edit correctly (env tables, no JSON-style dicts), and
codex already ships a command that does it right.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

SERVER_NAME = "atlas-recall"
HOOK_COMMAND = "recall hook"


# --------------------------------------------------------------------------
# Shared file helpers
# --------------------------------------------------------------------------
def _now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _backup(path: Path) -> Optional[Path]:
    """Copy `path` to a timestamped sibling before any write. No-op
    (returns None) if `path` doesn't exist yet -- nothing to back up."""
    if not path.exists():
        return None
    backup_path = path.with_name(f"{path.name}.bak-{_now_stamp()}")
    shutil.copy2(path, backup_path)
    return backup_path


def _atomic_write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def _load_json_or_raise(path: Path) -> Dict:
    """Load `path` as JSON, returning {} if it doesn't exist yet or is
    empty. Raises ValueError -- never silently discards -- on malformed
    JSON, naming the path, so a caller refuses to write rather than
    clobber a broken file that might still have content worth keeping."""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not valid JSON ({e}) -- refusing to modify it") from e


def resolve_server_command() -> Dict[str, Any]:
    """Prefer the installed `atlas-recall-mcp` console script by its
    absolute path -- GUI apps (Claude Desktop, Cursor, ...) don't inherit
    the shell's PATH, so a bare command name silently fails to launch.
    Falls back to `<this interpreter> -m atlas_recall.server` if the
    script isn't resolvable on PATH right now."""
    exe = shutil.which("atlas-recall-mcp")
    if exe:
        return {"command": exe, "args": []}
    return {"command": sys.executable, "args": ["-m", "atlas_recall.server"]}


# --------------------------------------------------------------------------
# Config path resolvers -- each one is backed by that client's own docs
# (see README). Anything not confirmed is simply not defined here.
# --------------------------------------------------------------------------
def claude_code_settings_path() -> Path:
    return Path(os.path.expanduser("~/.claude/settings.json"))


def claude_desktop_config_path() -> Path:
    if sys.platform == "darwin":
        base = Path.home() / "Library/Application Support/Claude"
    elif sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA", os.path.expanduser("~"))) / "Claude"
    else:
        raise ValueError(
            "claude-desktop: no confirmed config path for this platform "
            f"({sys.platform!r}) -- only macOS and Windows are documented"
        )
    return base / "claude_desktop_config.json"


def cursor_config_path() -> Path:
    return Path(os.path.expanduser("~/.cursor/mcp.json"))


def windsurf_config_path() -> Path:
    return Path(os.path.expanduser("~/.codeium/windsurf/mcp_config.json"))


def cline_config_path() -> Path:
    if sys.platform == "darwin":
        base = Path.home() / "Library/Application Support/Code/User/globalStorage"
    elif sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA", os.path.expanduser("~"))) / "Code/User/globalStorage"
    else:
        base = Path.home() / ".config/Code/User/globalStorage"
    return base / "saoudrizwan.claude-dev/settings/cline_mcp_settings.json"


def codex_config_path() -> Path:
    return Path(os.path.expanduser("~/.codex/config.toml"))


# --------------------------------------------------------------------------
# mcpServers-shaped clients (Claude Desktop, Cursor, Windsurf, Cline)
# --------------------------------------------------------------------------
def _install_mcp_json(path: Path, dry_run: bool) -> Dict[str, Any]:
    try:
        data = _load_json_or_raise(path)
    except ValueError as e:
        return {"success": False, "path": str(path), "error": str(e)}

    servers = data.setdefault("mcpServers", {})
    entry = resolve_server_command()
    if servers.get(SERVER_NAME) == entry:
        return {"success": True, "path": str(path), "changed": False, "dry_run": dry_run}

    if dry_run:
        return {"success": True, "path": str(path), "changed": True, "dry_run": True, "would_write": entry}

    backup_path = _backup(path)
    servers[SERVER_NAME] = entry
    _atomic_write_json(path, data)
    return {
        "success": True, "path": str(path), "changed": True, "dry_run": False,
        "backup": str(backup_path) if backup_path else None,
    }


def _uninstall_mcp_json(path: Path, dry_run: bool) -> Dict[str, Any]:
    try:
        data = _load_json_or_raise(path)
    except ValueError as e:
        return {"success": False, "path": str(path), "error": str(e)}

    servers = data.get("mcpServers", {})
    if SERVER_NAME not in servers:
        return {"success": True, "path": str(path), "changed": False, "dry_run": dry_run}

    if dry_run:
        return {"success": True, "path": str(path), "changed": True, "dry_run": True}

    backup_path = _backup(path)
    del servers[SERVER_NAME]
    _atomic_write_json(path, data)
    return {
        "success": True, "path": str(path), "changed": True, "dry_run": False,
        "backup": str(backup_path) if backup_path else None,
    }


# --------------------------------------------------------------------------
# Claude Code: hooks.UserPromptSubmit-shaped, not mcpServers-shaped
# --------------------------------------------------------------------------
def _install_claude_code_hook(path: Path, dry_run: bool) -> Dict[str, Any]:
    try:
        data = _load_json_or_raise(path)
    except ValueError as e:
        return {"success": False, "path": str(path), "error": str(e)}

    hooks = data.setdefault("hooks", {})
    prompt_hooks = hooks.setdefault("UserPromptSubmit", [])

    already = any(
        h.get("type") == "command" and h.get("command") == HOOK_COMMAND
        for group in prompt_hooks
        for h in group.get("hooks", [])
    )
    if already:
        return {"success": True, "path": str(path), "changed": False, "dry_run": dry_run}

    if dry_run:
        return {"success": True, "path": str(path), "changed": True, "dry_run": True}

    backup_path = _backup(path)
    prompt_hooks.append({"hooks": [{"type": "command", "command": HOOK_COMMAND}]})
    _atomic_write_json(path, data)
    return {
        "success": True, "path": str(path), "changed": True, "dry_run": False,
        "backup": str(backup_path) if backup_path else None,
    }


def _uninstall_claude_code_hook(path: Path, dry_run: bool) -> Dict[str, Any]:
    try:
        data = _load_json_or_raise(path)
    except ValueError as e:
        return {"success": False, "path": str(path), "error": str(e)}

    hooks = data.get("hooks", {})
    prompt_hooks = hooks.get("UserPromptSubmit", [])

    new_groups = []
    changed = False
    for group in prompt_hooks:
        original = group.get("hooks", [])
        remaining = [
            h for h in original
            if not (h.get("type") == "command" and h.get("command") == HOOK_COMMAND)
        ]
        if remaining == original:
            new_groups.append(group)
            continue
        changed = True
        if remaining:
            new_groups.append({**group, "hooks": remaining})
        # else: this matcher group only ever held our hook -- drop it.

    if not changed:
        return {"success": True, "path": str(path), "changed": False, "dry_run": dry_run}

    if dry_run:
        return {"success": True, "path": str(path), "changed": True, "dry_run": True}

    backup_path = _backup(path)
    hooks["UserPromptSubmit"] = new_groups
    _atomic_write_json(path, data)
    return {
        "success": True, "path": str(path), "changed": True, "dry_run": False,
        "backup": str(backup_path) if backup_path else None,
    }


# --------------------------------------------------------------------------
# Codex CLI: driven through `codex mcp add`/`codex mcp list`, not a
# hand-rolled TOML writer -- see module docstring.
# --------------------------------------------------------------------------
def _codex_available() -> bool:
    return shutil.which("codex") is not None


def _codex_has_server() -> Optional[bool]:
    if not _codex_available():
        return None
    try:
        result = subprocess.run(["codex", "mcp", "list"], capture_output=True, text=True, timeout=10)
    except Exception:
        return None
    return SERVER_NAME in result.stdout


def _install_codex(dry_run: bool) -> Dict[str, Any]:
    if not _codex_available():
        return {"success": False, "path": None, "error": "codex CLI not found on PATH"}
    config_path = codex_config_path()
    if _codex_has_server():
        return {"success": True, "path": str(config_path), "changed": False, "dry_run": dry_run}

    entry = resolve_server_command()
    cmd = ["codex", "mcp", "add", SERVER_NAME, "--", entry["command"], *entry["args"]]
    if dry_run:
        return {
            "success": True, "path": str(config_path), "changed": True, "dry_run": True,
            "would_run": " ".join(cmd),
        }

    backup_path = _backup(config_path)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except Exception as e:  # noqa: BLE001
        return {"success": False, "path": str(config_path), "error": str(e)}
    if result.returncode != 0:
        return {
            "success": False, "path": str(config_path),
            "error": (result.stderr or result.stdout).strip() or "codex mcp add failed",
        }
    return {
        "success": True, "path": str(config_path), "changed": True, "dry_run": False,
        "backup": str(backup_path) if backup_path else None,
    }


def _uninstall_codex(dry_run: bool) -> Dict[str, Any]:
    if not _codex_available():
        return {"success": False, "path": None, "error": "codex CLI not found on PATH"}
    config_path = codex_config_path()
    if not _codex_has_server():
        return {"success": True, "path": str(config_path), "changed": False, "dry_run": dry_run}

    if dry_run:
        return {"success": True, "path": str(config_path), "changed": True, "dry_run": True}

    backup_path = _backup(config_path)
    for sub in (["codex", "mcp", "remove", SERVER_NAME], ["codex", "mcp", "rm", SERVER_NAME]):
        try:
            result = subprocess.run(sub, capture_output=True, text=True, timeout=15)
        except Exception:  # noqa: BLE001
            continue
        if result.returncode == 0:
            return {
                "success": True, "path": str(config_path), "changed": True, "dry_run": False,
                "backup": str(backup_path) if backup_path else None,
            }
    return {
        "success": False, "path": str(config_path),
        "error": (
            f"could not remove via `codex mcp remove`/`codex mcp rm` -- edit {config_path} "
            f"by hand: delete the [mcp_servers.{SERVER_NAME}] table"
        ),
    }


# --------------------------------------------------------------------------
# Client registry + public dispatch
# --------------------------------------------------------------------------
CLIENTS: Dict[str, Dict[str, Any]] = {
    "claude-code": {"kind": "hook", "path_fn": claude_code_settings_path},
    "claude-desktop": {"kind": "mcp", "path_fn": claude_desktop_config_path},
    "cursor": {"kind": "mcp", "path_fn": cursor_config_path},
    "windsurf": {"kind": "mcp", "path_fn": windsurf_config_path},
    "cline": {"kind": "mcp", "path_fn": cline_config_path},
    "codex": {"kind": "codex"},
}


def _dispatch(
    client: str,
    dry_run: bool,
    config_path: Optional[Path],
    hook_fn: Callable,
    mcp_fn: Callable,
    codex_fn: Callable,
) -> Dict[str, Any]:
    if client not in CLIENTS:
        return {"success": False, "error": f"unknown client {client!r} -- choose from {sorted(CLIENTS)}"}
    spec = CLIENTS[client]
    if spec["kind"] == "codex":
        result = codex_fn(dry_run)
    else:
        try:
            path = config_path or spec["path_fn"]()
        except ValueError as e:
            return {"success": False, "client": client, "error": str(e)}
        result = (hook_fn if spec["kind"] == "hook" else mcp_fn)(path, dry_run)
    result["client"] = client
    return result


def install(client: str, dry_run: bool = False, config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Install `client`'s config entry. `config_path` overrides the
    resolved path (tests use this to point at a temp file instead of the
    real one); ignored for `codex`, whose path is fixed by the `codex`
    binary itself."""
    return _dispatch(client, dry_run, config_path, _install_claude_code_hook, _install_mcp_json, _install_codex)


def uninstall(client: str, dry_run: bool = False, config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Remove `client`'s config entry. See `install` for `config_path`."""
    return _dispatch(client, dry_run, config_path, _uninstall_claude_code_hook, _uninstall_mcp_json, _uninstall_codex)
