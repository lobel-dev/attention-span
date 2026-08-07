"""Derive user-facing verdicts and compact display labels."""

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
    """The one BEHAVIORAL signal trusted enough to lead-and-alarm: a genuine
    failed-edit loop. Returns a dict (state always ``"red"``) or ``None``.

        {"state": "red", "file": <path>, "base": <basename>, "count": <n>,
         "chip": "EDIT LOOP: <base> x<n>"}

    High-confidence and deterministic: ``failed_edit_loop.count >= 1`` means a
    file whose most recent edit GENUINELY failed (``<tool_use_error>``) was
    re-edited with no intervening Read (see ``_failed_edit_loop``). It depends on
    neither the R2E ratio nor context — a blind loop is a blind loop regardless.

    Fires REGARDLESS of ``insufficient``: a 2-edit deterministic loop is real
    even in a short (e.g. subagent child) transcript, so it is never gated behind
    the low-signal window floor. ``basename`` is ``_sanitize``-stripped at this
    render seam (ANSI/OSC-injection defense); the raw ``file`` is also returned
    for callers that need it.
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
    """Internal read-vs-edit posture from the UNCALIBRATED R2E ratio.

    Returns ``(state, chip)`` for this module's debug CLI and engine tests. It has no
    public statusline or ``/health`` exposure and never drives an action.

        INSUFFICIENT                  -> "green",  ""                  (light off)
        R2E < 0.5 (>= MIN_EDITS)      -> "yellow", "Close Watch"
        0.5 <= R2E < 1.0              -> "yellow", "Spot Check"
        R2E < 0.5 (< MIN_EDITS)       -> "yellow", "Spot Check"
        R2E >= 1.0                    -> "green",  "Monitor Normally"

    NEVER red and NEVER coupled to context: the old red "Manual Review" escalation
    (gated on context saturation) is gone — context saturation is now shown
    directly by the lead context light, so re-deriving it here would be redundant
    double-signalling. The worst this provisional signal says is a soft yellow.
    The R2E ratio itself stays internal; the chip is the posture word.
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
    """Context operating tier from absolute size alone. Returns (tier|None, chip).

    The tier is one of ``health_config.CONTEXT_TIERS`` — the ladder is walked
    worst-first and the first band the token count clears wins. Returns (None, "")
    when context size is unknown (0) so the caller can suppress the whole context
    group rather than show a misleading full-health reading of nothing.

    ``window_class`` (M2) selects WHICH owner-reasoned ladder applies — each tier's
    band resolves through ``health_config.context_band_field``, so a class listed in
    ``CONTEXT_CLASS_BAND_OVERRIDES`` grades on its own boundaries. This is not the
    retired percentage axis returning: the advertised window never SCALES a band, it
    only selects between individually-reasoned absolute ladders, and an unknown class
    (None, unlisted, garbage) grades on the base ladder exactly as before.

    ``ctx_pct`` is a DISPLAY parameter only: it is appended to the chip and grades
    nothing. Claude Code derives ``used_percentage`` from the same token count this
    function reads, divided by the model's window, so treating it as an independent
    axis meant grading one quantity on two ladders — the same 190K session came out
    FUNCTIONAL on a 1M model and DEAD on a 200K one. Window SIZE is the percentage's only
    unique information, and the owner's decision is that the heat bar carries it.

    These bands are broad, uncalibrated workflow triggers for gradual context risk,
    NOT a quality measurement. Observable symptoms remain separate signals: this
    function does not infer forgotten constraints, contradictions, or sloppy edits
    from a token count.
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
    """Class name for an advertised context window, or None when unknown.

    ``window_size`` is the payload's ``context_window.context_window_size`` — the
    model-derived advertised window Claude Code reports every render. Classification
    is an EXACT match against ``health_config.WINDOW_CLASSES``: an unlisted size (or
    a missing / non-numeric / non-finite / non-positive one) returns None, meaning
    UNKNOWN. Doubt deliberately falls back to the single shipped ladder rather than
    guessing a nearest class; the caller records the raw size alongside, so the
    taxonomy can learn a new size later without having mis-graded anything meanwhile.
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

    Examples:
        claude-sonnet-4-6         -> 'sonnet-4.6'
        claude-opus-4-8[1m]       -> 'opus-4.8·1M'
        claude-haiku-4-5-20251001 -> 'haiku-4.5'   (date tag dropped)
        claude-fable-5            -> 'fable-5'
        unknown-model-xyz         -> sanitized fallback

    Used for BOTH display and swap detection in the renderer, so a version bump
    (opus-4.7 -> opus-4.8) correctly fires ⇄, not just a tier change. Result
    is _sanitize-safe (no raw file paths reach this function, only model ids).
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
