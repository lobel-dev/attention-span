"""Terminal-safe text, compact token, and visible-width utilities."""

import re
import unicodedata
from collections.abc import Iterable
from typing import Any

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def sanitize(s: Any) -> str:
    """Strip C0/C1/DEL control chars (incl. ESC/BEL) from a host/repo-derived
    string before it is rendered into the statusline — neutralizing ANSI/OSC
    escape injection from a crafted file path or model id. Non-str -> "".

    Applied at the RENDER seams only (the EDIT LOOP basename, the model-swap
    chip), never at capture: a raw ``file_path`` is a dict key for same-file
    blind-loop matching (`_failed_edit_loop`), so it must stay byte-for-byte.
    """
    if not isinstance(s, str):
        return ""
    return _CONTROL_RE.sub("", s)


def fmt_tokens(n: Any) -> str:
    """Compact token count: None->'', <1000->'512', <1M->'206K', else '1.5M'."""
    if n is None:
        return ""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)


_COMPACT_UNITS = (
    (10**15, "Q"),
    (10**12, "T"),
    (10**9, "B"),
    (10**6, "M"),
    (10**3, "K"),
)


def compact_magnitude(value: Any) -> str:
    """A bounded non-negative count for the terminal-pressure fallback."""
    try:
        n = max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return "0"
    if n * 2 >= 1999 * 10**15:  # 999.5Q: no larger named unit is available
        return "999Q+"
    for i, (scale, suffix) in enumerate(_COMPACT_UNITS):
        if n < scale:
            continue
        if n < 100 * scale:
            tenths = (n * 10 + scale // 2) // scale
            whole, decimal = divmod(tenths, 10)
            return "{}{}{}".format(
                whole, ("." + str(decimal)) if decimal else "", suffix
            )
        whole = (n + scale // 2) // scale
        if whole >= 1000 and i > 0:
            return "1" + _COMPACT_UNITS[i - 1][1]
        return f"{whole}{suffix}"
    return str(n)


def display_tokens(value: Any) -> str:
    """Familiar token formatting with a fail-closed ceiling for absurd inputs."""
    try:
        n = max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return "0"
    if n * 2 >= 1999 * 10**15:
        return compact_magnitude(n)
    return fmt_tokens(n)


_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def vis_len(s: str) -> int:
    """Visible width: drop SGR codes; count emoji/wide glyphs as 2 (approx)."""
    s = _ANSI_RE.sub("", s)
    w = 0
    prev_narrow = False
    for ch in s:
        o = ord(ch)
        if o == 0xFE0F:
            if prev_narrow:
                w += 1
                prev_narrow = False
            continue
        if (
            o >= 0x1F000
            or (0x2600 <= o <= 0x27BF and unicodedata.east_asian_width(ch) in "WF")
            or 0x2B00 <= o <= 0x2BFF
            or o in (0x231A, 0x231B, 0x23F0, 0x23F1, 0x23F3, 0x2728)
        ):
            w += 2
            prev_narrow = False
        else:
            w += 1
            prev_narrow = 0x2600 <= o <= 0x27BF
    return w


def _take_visible(chars: Iterable[str], budget: int) -> list[str]:
    """The longest run of ``chars`` whose combined visible width fits ``budget``."""
    taken: list[str] = []
    used = 0
    for char in chars:
        char_width = vis_len(char)
        if used + char_width > budget:
            break
        taken.append(char)
        used += char_width
    return taken


def right_ellipsis(value: str, width: int) -> str:
    """Trim to ``width`` visible columns, marking the cut with a trailing ellipsis."""
    if not value or width <= 0:
        return ""
    if vis_len(value) <= width:
        return value
    return "".join(_take_visible(value, width - 1)) + "…"


def left_ellipsis(value: str, width: int) -> str:
    """Trim to ``width`` visible columns, marking the cut with a leading ellipsis."""
    if not value or width <= 0:
        return ""
    if vis_len(value) <= width:
        return value
    return "…" + "".join(reversed(_take_visible(reversed(value), width - 1)))
