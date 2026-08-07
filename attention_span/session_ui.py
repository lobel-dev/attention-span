"""Per-session presentation state."""

import contextlib
import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Mapping
from typing import Any

from attention_span import render_facts


def state_dir() -> str:
    """Return the private runtime-state directory for this Claude installation."""
    claude_home = os.environ.get("CLAUDE_HOME") or os.path.join(
        os.path.expanduser("~"), ".claude"
    )
    return os.path.join(
        os.path.abspath(os.path.expanduser(claude_home)),
        "hooks",
        "attention-span",
        "cache",
    )


UI_PREFIX = os.path.join(state_dir(), "ui-")
SCHEMA = 1


def session_key(session_id: Any) -> str:
    """Return an opaque, filename-safe key without leaking or trusting an identifier."""
    try:
        raw = str(session_id or "unknown").encode("utf-8", errors="replace")
    except Exception:
        raw = b"unknown"
    return hashlib.sha256(raw).hexdigest()


def ensure_private_directory(path: str) -> bool:
    """Create an owner-only state directory, rejecting a substituted symlink."""
    try:
        os.makedirs(path, mode=0o700, exist_ok=True)
        info = os.stat(path, follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode):
            return False
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            return False
        os.chmod(path, 0o700)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _read(path: str) -> dict[str, Any] | None:
    try:
        with open(path) as f:
            value = json.load(f)
        return (
            value if isinstance(value, dict) and value.get("schema") == SCHEMA else None
        )
    except Exception:
        return None


def _atomic_write(path: str, value: Mapping[str, Any]) -> bool:
    directory = os.path.dirname(path) or "."
    tmp = None
    try:
        if not ensure_private_directory(directory):
            return False
        fd, tmp = tempfile.mkstemp(prefix=".cc-health-", dir=directory, text=True)
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, separators=(",", ":"), allow_nan=False)
        os.replace(tmp, path)
        return True
    except Exception:
        if tmp:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
        return False


def next_notice(
    session_id: Any, done_n: Any, live_n: Any, *, show_working: bool = True
) -> render_facts.Notice | None:
    """Record cohort progress and return this render cycle's ``Notice``, or None.

    The counts cross to the renderer as a record, never as the sentence they used to be
    formatted into: this function decides HOW MANY finished, and the catalog decides
    what to call them.
    """
    try:
        done_n = max(0, int(done_n or 0))
        live_n = max(0, int(live_n or 0))
    except (TypeError, ValueError, OverflowError):
        return None
    path = UI_PREFIX + session_key(session_id)
    old = _read(path)
    previous = old.get("done_n") if old and isinstance(old.get("done_n"), int) else None
    finished = done_n - previous if previous is not None else 0
    written = _atomic_write(path, {"schema": SCHEMA, "done_n": done_n})
    # An unwritten baseline would re-announce the same completions next cycle.
    if not written or finished <= 0:
        return None
    return render_facts.Notice(done_n=finished, live_n=live_n if show_working else 0)
