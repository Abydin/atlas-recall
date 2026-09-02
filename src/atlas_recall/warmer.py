"""
Keep-downloaded warmer for a notes directory synced through a cloud drive
(iCloud Drive being the common case).

Root cause this protects against: on macOS, a cloud-synced directory can
evict a file's local copy to a network-only ("dataless") placeholder to
save disk space. Reading one blocks the calling thread until it's
re-downloaded -- measured up to ~70s on one such file. `corpus.py`'s
`is_dataless()` check lets retrieval SKIP
a dataless file instead of hanging on it, but a skipped file means a
smaller corpus gets searched. This warmer is the environmental fix: force-
materialize the configured notes_dir on a schedule (e.g. cron/launchd) so
files rarely go dataless in the first place.

`brctl download <path>` (macOS-only; part of the CloudDocs / bird stack)
requests materialization without blocking on the transfer completing --
fire-and-forget is fine here, this just needs to run often enough that the
directory doesn't get a long enough idle gap to be evicted again.

No-op (does nothing, returns 0) on any non-macOS platform or if `brctl`
isn't on PATH -- this is a narrow, optional convenience, not a dependency
of the retrieval or hook path.
"""
from __future__ import annotations

import shutil
import subprocess
import sys

from .config import Config, load_config


def warm(cfg: Config = None) -> int:
    cfg = cfg or load_config()
    if sys.platform != "darwin":
        print("recall warm: no-op (not macOS)")
        return 0
    if not shutil.which("brctl"):
        print("recall warm: no-op (brctl not found)")
        return 0
    if not cfg.notes_dir:
        print("recall warm: no notes_dir configured, run `recall init` first", file=sys.stderr)
        return 1
    import os
    real = os.path.realpath(os.path.expanduser(cfg.notes_dir))
    try:
        subprocess.run(["brctl", "download", real], capture_output=True, timeout=30)
    except Exception as e:  # noqa: BLE001
        print(f"recall warm: brctl failed ({e})", file=sys.stderr)
        return 1
    print(f"recall warm: requested materialization of {real}")
    return 0
