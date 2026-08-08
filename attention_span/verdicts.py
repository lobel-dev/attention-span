"""Verdicts and compact display labels derived from analysis facts."""

import math
import os
import re
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any, cast

from attention_span import health_config, text
from attention_span.analysis import Analysis

DEFAULTS = health_config.ENGINE_DEFAULTS

_sanitize = cast(Callable[[object], str], text.sanitize)
_fmt_tokens = cast(Callable[[object], str], text.fmt_tokens)
_resolve_thresholds = cast(
    Callable[[Mapping[str, Any] | None], Mapping[str, Any]],
    health_config.resolve_thresholds,
)
_context_band_field = cast(
    Callable[[str, str | None], str | None],
    health_config.context_band_field,
)


def blind_loop_alert(analysis: Analysis) -> dict[str, str | int] | None:
    """A live failed-edit loop as ``{state: "red", file, base, count, chip}``, else None.

    The one BEHAVIORAL signal trusted to lead and alarm, because it is deterministic:
    it reads neither the R2E ratio nor context, and is never gated behind the
    low-signal window floor (a short child transcript can still hold a real loop).
    ``base`` is ``_sanitize``-stripped for display (this is a render seam);
    ``file`` stays raw so callers can match on it.
    """
    floop = analysis.failed_edit_loop
    if floop.count < 1:
        return None
    base = _sanitize(os.path.basename(floop.file or "")) or "file"
    return {
        "state": "red",
        "file": floop.file or "",
        "base": base,
        "count": floop.count,
        "chip": f"EDIT LOOP: {base} x{floop.count}",
    }


def r2e_posture(
    analysis: Analysis, th: Mapping[str, Any] | None = None
) -> tuple[str, str]:
    """Provisional read-vs-edit posture from the UNCALIBRATED R2E ratio: (state, chip).

    Diagnostic only - nothing renders it and it drives no action. NEVER red and never
    coupled to context: context saturation is its own signal, so grading it again here
    would double-signal. The ratio stays internal; the chip is the posture word, and an
    insufficient window yields an empty one (light off).
    """
    if analysis.insufficient:
        return "green", ""
    thresholds = _resolve_thresholds(th)
    base = analysis.base_tier
    if base == "red" and analysis.edits >= thresholds["MIN_EDITS_FOR_RED"]:
        return "yellow", "Close Watch"
    if base in ("red", "yellow"):
        return "yellow", "Spot Check"
    return "green", "Monitor Normally"


def context_verdict(
    context_tokens: Any,
    ctx_pct: Any = None,
    th: Mapping[str, Any] | None = None,
    window_class: str | None = None,
) -> tuple[str | None, str]:
    """Context operating tier from absolute token size alone: ``(tier, chip)``.

    ``tier`` is a ``health_config`` context-tier name; grading assumes
    ``CONTEXT_TIER_BANDS`` is ordered worst-first. An unknown size yields
    ``(None, "")``: the caller decides what to show, rather than this function
    reporting full health of nothing.

    ``window_class`` selects WHICH owner-reasoned ladder applies; an unknown class
    grades on the base one. The advertised window never SCALES a band - it only picks
    between absolute ladders reasoned about one at a time.

    ``ctx_pct`` is a DISPLAY parameter only: appended to the chip, grades nothing. It
    is derived from the same token count read here, so grading it too would grade one
    quantity on two ladders and make the verdict depend on the model's window size.

    The bands are broad owner-set workflow triggers, not a quality measurement: no
    symptom is inferred from a token count.
    """
    if not context_tokens:
        return None, ""
    thresholds = _resolve_thresholds(th)
    state = "peak"
    for tier in health_config.CONTEXT_TIER_BANDS:
        band = _context_band_field(tier, window_class)
        if band and context_tokens >= thresholds[band]:
            state = tier
            break
    chip = _fmt_tokens(context_tokens)
    if ctx_pct is not None:
        chip += f" · {int(ctx_pct)}%"
    return state, chip


def window_class(window_size: Any) -> str | None:
    """Class name for an advertised context window size, or None meaning UNKNOWN.

    EXACT match against ``health_config.WINDOW_CLASSES``: anything else - unlisted,
    missing, non-numeric, non-finite, non-positive, non-integral - is None. Doubt
    falls back to the base ladder instead of guessing a nearest class.
    """
    if isinstance(window_size, bool) or not isinstance(window_size, (int, float)):
        return None
    if not math.isfinite(window_size) or window_size <= 0:
        return None
    size = int(window_size)
    if size != window_size:
        return None
    return health_config.WINDOW_CLASSES.get(size)


def model_label(model_id: Any) -> str:
    """Compact display label: tier + version + optional context marker.

        claude-opus-4-8[1m]       -> 'opus-4.8·1M'
        claude-haiku-4-5-20251001 -> 'haiku-4.5'   (release date tag dropped)
        unknown-model-xyz         -> sanitized fallback

    Callers also compare labels to detect a model SWAP, so the label must keep every
    distinction that makes two runs different models - version and context marker
    included, not tier alone. Output is ``_sanitize``-safe.
    """
    if not model_id or not isinstance(model_id, str):
        return ""
    low = model_id.lower()

    tier = None
    for t in ("opus", "sonnet", "haiku", "fable"):
        if t in low:
            tier = t
            break
    if not tier:
        return _sanitize(model_id)

    after = low[low.index(tier) + len(tier) :]  # e.g. '-4-6' or '-4-8[1m]'
    ctx_m = re.search(r"\[(\d+[km])\]", after)
    ctx = ctx_m.group(1).upper() if ctx_m else ""
    parts: list[str] = []

    for seg in re.split(r"[-\[]", after.lstrip("-")):
        if not seg.isdigit():
            break
        try:
            datetime.strptime(seg, "%Y%m%d")
            break  # valid calendar date → release tag, not a version component
        except ValueError:
            pass
        parts.append(seg)

    label = tier + ("-" + ".".join(parts) if parts else "")
    return label + ("·" + ctx if ctx else "")
