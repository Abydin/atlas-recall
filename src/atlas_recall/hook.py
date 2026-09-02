"""
Claude Code UserPromptSubmit hook: reads a prompt from stdin JSON, retrieves
the top-K relevant notes for it, and injects them via hookSpecificOutput.
additionalContext.

Invariant: this always exits 0. It never exits with an error code that
would erase the prompt, and any internal failure (missing config, no
notes_dir, retrieval exception, timeout) degrades to injecting nothing --
silent beats a broken turn.

`recall init` prints the settings.json block that wires this in as
`recall hook`.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from typing import Optional

from .config import Config, load_config
from .retrieval import format_pointer_block, top_pointers

LIVE_TIMEOUT_SECS = 2.0  # hard wall-clock budget for retrieval


def _call_with_timeout(fn, timeout, *args, **kwargs):
    """Run `fn` on a daemon thread; stop waiting after `timeout` seconds.
    On timeout the thread is abandoned (Python can't forcibly kill a
    thread) but that's safe: a daemon thread never blocks process exit,
    even mid-syscall -- the reliable way to bound a call that might be
    stuck in blocking I/O a signal can't interrupt (e.g. a cloud-synced
    drive re-downloading an evicted file)."""
    box = {}

    def _run():
        try:
            box["value"] = fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            box["error"] = e

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        return None, True
    if "error" in box:
        raise box["error"]
    return box.get("value"), False


def _keyword_reminders(prompt: str, cfg: Config) -> str:
    """Targeted repetition: if the prompt mentions a configured trigger
    keyword, echo the ONE rule line it governs inline. Empty by default --
    see examples/keyword_rules.json for how to configure your own."""
    p = (prompt or "").lower()
    hits = []
    for keywords, rule_line in cfg.keyword_rules:
        if any(kw in p for kw in keywords) and rule_line not in hits:
            hits.append(rule_line)
    if not hits:
        return ""
    return "== TRIGGERED BY THIS PROMPT ==\n" + "\n".join(hits)


def _rules_block(cfg: Config) -> str:
    if not cfg.rules_path:
        return ""
    try:
        import os
        text = open(os.path.expanduser(cfg.rules_path), encoding="utf-8").read().strip()
    except OSError:
        # A typo'd or iCloud-evicted rules_path must still degrade to
        # exit 0 with nothing injected -- but silently vanishing a block
        # literally labeled "HARD RULES (enforced)" with no diagnostic
        # is its own failure mode. Warn loudly on stderr (stdout stays
        # pure JSON for the hook contract) and still return "".
        print(
            f"[recall hook] WARN: rules_path {cfg.rules_path!r} unreadable -- "
            f"HARD RULES NOT INJECTED this turn",
            file=sys.stderr,
        )
        return ""
    if not text:
        return ""
    return f"== HARD RULES (enforced) ==\n{text}\n== END HARD RULES =="


def build_context(prompt: str, cfg: Config) -> str:
    """Best-effort: retrieve + format. Never raises past this function."""
    if not cfg.notes_dir:
        return ""

    rules = _rules_block(cfg)
    triggers = _keyword_reminders(prompt, cfg)

    live_block = ""
    if len(prompt) >= 8:
        try:
            pointers, timed_out = _call_with_timeout(top_pointers, LIVE_TIMEOUT_SECS, prompt, cfg)
            if not timed_out:
                live_block = format_pointer_block(pointers)
        except Exception:
            live_block = ""

    return "\n\n".join(p for p in (rules, triggers, live_block) if p)


def _emit(ctx: str) -> None:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": ctx,
        }
    }), flush=True)


def run_hook(stdin_text: Optional[str] = None) -> int:
    """Entry point for `recall hook`. Reads stdin JSON (or `stdin_text` in
    tests), emits exactly one context blob, always returns 0."""
    try:
        raw = stdin_text if stdin_text is not None else sys.stdin.read()
        data = json.loads(raw)
        prompt = data.get("prompt", "")
        cfg = load_config()
        ctx = build_context(prompt, cfg)
        _emit(ctx)
    except Exception:
        _emit("")
    return 0
