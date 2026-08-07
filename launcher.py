#!/usr/bin/env python3
"""Stable statusline launcher with a detached, daily release-update trigger."""

import contextlib
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

UPDATE_INTERVAL_SECONDS = 24 * 60 * 60
FALSE_VALUES = {"0", "false", "no", "off"}


def auto_update_enabled() -> bool:
    value = os.environ.get("CLAUDE_HEALTH_AUTO_UPDATE", "1")
    return value.strip().lower() not in FALSE_VALUES


def resolve_release(install_root: str | Path) -> Path:
    root = Path(install_root).resolve()
    release = (root / "current").resolve(strict=True)
    releases = (root / "releases").resolve(strict=True)
    if os.path.commonpath((str(release), str(releases))) != str(releases):
        raise ValueError("current release escapes the install root")
    if not (release / "statusline.py").is_file():
        raise ValueError("current release has no statusline")
    return release


def claim_update_check(install_root: str | Path, now: float | None = None) -> bool:
    root = Path(install_root)
    now = time.time() if now is None else now
    lock_path = root / "update-throttle.lock"
    stamp_path = root / "update-check.stamp"
    try:
        with lock_path.open("a+") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, OSError):
                return False
            try:
                age = now - stamp_path.stat().st_mtime
                if 0 <= age < UPDATE_INTERVAL_SECONDS:
                    return False
            except OSError:
                pass
            stamp_path.touch(exist_ok=True)
            os.utime(str(stamp_path), (now, now))
            return True
    except OSError:
        return False


def spawn_update(install_root: str | Path, release: Path) -> None:
    updater = release / "update.py"
    if not updater.is_file():
        return
    subprocess.Popen(
        [sys.executable, str(updater), "--install-root", str(install_root)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
    )


def main(install_root: Any = None) -> int:
    root = (
        Path(install_root).absolute()
        if install_root is not None
        else Path(__file__).absolute().parent
    )
    try:
        if (root / "current").is_symlink():
            release = resolve_release(root)
            if auto_update_enabled() and claim_update_check(root):
                with contextlib.suppress(Exception):
                    spawn_update(root, release)
            statusline = release / "statusline.py"
        else:
            statusline = root / "statusline.py"
        os.execv(
            sys.executable,
            [sys.executable, str(statusline)] + sys.argv[1:],
        )
        return 0
    except Exception:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
