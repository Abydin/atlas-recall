"""
Console-script entry point for `atlas-recall-mcp`, kept separate from
server.py on purpose.

`[project.scripts]` entries install unconditionally, but the `mcp` SDK is
an optional extra (`pip install atlas-recall[mcp]`) that also requires
Python 3.10+ -- the rest of this package stays on the 3.9 floor. server.py
imports `mcp.server.fastmcp` at module level (required for the
`@mcp.tool()` decorator pattern), so if that import were allowed to fail
inside server.py itself, `atlas-recall-mcp` would print a raw
ModuleNotFoundError traceback for anyone who installed the base package
without the extra. This module exists to turn that into one clear line
instead.
"""
from __future__ import annotations

import sys


def main() -> None:
    try:
        from . import server
    except ImportError as e:
        print(
            "atlas-recall-mcp requires the `mcp` extra and Python 3.10+ -- "
            f"run: pip install 'atlas-recall[mcp]'  (current interpreter: "
            f"Python {sys.version.split()[0]})\n(import error: {e})",
            file=sys.stderr,
        )
        sys.exit(1)
    server.main()


if __name__ == "__main__":
    main()
