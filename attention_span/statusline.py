#!/usr/bin/env python3
"""Claude Code ``statusLine`` hook: the orchestrator"""

import contextlib
import json
import math
import os
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple, cast

from attention_span import agent_health, render, render_facts, session_ui, subagents

CACHE_TTL = 10  # seconds — the render cache window
CACHE_SCHEMA = 6  # bump on any change to the entry format or the identity shape
_state_dir = session_ui.state_dir
_session_key = cast(Callable[[Any], str], session_ui.session_key)
_ensure_private_directory = cast(
    Callable[[str], bool], session_ui.ensure_private_directory
)
_next_notice = cast(Callable[..., render_facts.Notice | None], session_ui.next_notice)


def _num(v: Any) -> float | None:
    """Parse to float, or None when missing/unparseable (mirrors the old jq/num())."""
    try:
        n = float(v)
    except (TypeError, ValueError, OverflowError):
        return None
    return n if math.isfinite(n) else None


def _as_dict(value: Any) -> Mapping[str, Any]:
    """A payload sub-object as a mapping; ``{}`` for absent or off-schema values."""
    return value if isinstance(value, dict) else {}


def _flag(name: str, default: bool) -> bool:
    """A CLAUDE_HEALTH_SHOW_* style on/off env flag. Unset -> default; '' or '0' -> off."""
    v = os.environ.get(name)
    if v is None:
        return default
    return v not in ("", "0")


def _no_color() -> bool:
    return bool(os.environ.get("NO_COLOR")) or _flag("CLAUDE_HEALTH_NO_COLOR", False)


def _cold_marker(target_width: int | None = None) -> str:
    """Render the neutral cold-start waiting action (NO_COLOR-aware)."""
    return render.render_standby_row(
        columns=_envi("COLUMNS"), no_color=_no_color(), target_width=target_width
    )


def _identity_row(
    identity: Mapping[str, Any], columns: int | None, target_width: int | None = None
) -> str:
    """The identity row these transient rows align to, or '' when it has no content."""
    return render.render_metadata_row(
        rail_color="cyan",
        columns=columns,
        no_color=_no_color(),
        target_width=target_width,
        **identity,
    )


def _align_rows(
    build_row: Callable[[int | None], str],
    identity: Mapping[str, Any],
    columns: int | None,
) -> str:
    """Join a transient row to the identity row on ONE shared right edge."""
    metadata = _identity_row(identity, columns)
    if not metadata:
        return build_row(None)
    row = build_row(render.vis_len(metadata))
    metadata = _identity_row(
        identity, columns, max(render.vis_len(row), render.vis_len(metadata))
    )
    return "\n".join((row, metadata))


def _transcript_fingerprint(transcript: str) -> list[int]:
    """A cheap cache identity that changes for every transcript append or rewrite."""
    try:
        info = os.stat(transcript)
        return [info.st_size, info.st_mtime_ns]
    except (OSError, TypeError, ValueError):
        return []


def _transcript_session(transcript: str) -> str:
    """The session id a transcript path names: its basename without the extension."""
    return os.path.splitext(os.path.basename(transcript))[0]


def _cache_path(transcript: str) -> str:
    return os.path.join(
        _state_dir(),
        "render-" + _session_key(_transcript_session(transcript)) + ".json",
    )


def _write_cache(cache: str, value: Mapping[str, Any]) -> bool:
    """Atomically publish a private cache entry without following its target."""
    directory = os.path.dirname(cache) or "."
    tmp = None
    try:
        if not _ensure_private_directory(directory):
            return False
        fd, tmp = tempfile.mkstemp(prefix=".render-", dir=directory, text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, separators=(",", ":"), allow_nan=False)
        os.replace(tmp, cache)
        return True
    except Exception:
        if tmp:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
        return False


def _display_cwd(value: Any) -> str:
    """Contract the current user's home prefix without otherwise rewriting the path."""
    if not isinstance(value, str) or not value:
        return ""
    home = os.path.expanduser("~")
    if home and (value == home or value.startswith(home + os.sep)):
        return "~" + value[len(home) :]
    return value


def _session_identity(payload: Mapping[str, Any]) -> dict[str, str]:
    """Normalize the live payload facts that make up the persistent identity row."""
    model = _as_dict(payload.get("model"))
    effort = _as_dict(payload.get("effort"))
    workspace = _as_dict(payload.get("workspace"))
    repo = _as_dict(workspace.get("repo"))

    def text(value: Any) -> str:
        return value if isinstance(value, str) else ""

    return {
        "model_id": text(model.get("id")),
        "effort": text(effort.get("level")),
        "repo_name": text(repo.get("name")),
        "cwd": _display_cwd(payload.get("cwd") or workspace.get("current_dir") or ""),
    }


def _read_cache_if_fresh(cache: str, identity: Mapping[str, Any]) -> str | None:
    """Return the cached render if it is younger than the TTL, else None. A cache hit
    does ZERO filesystem writes (no cost/baseline work happens past this). Live session
    identity is part of the cache contract so model/effort/location changes render now.
    """
    try:
        if os.path.islink(cache):
            return None
        if time.time() - os.path.getmtime(cache) < CACHE_TTL:
            with open(cache) as f:
                value = json.load(f)
            if (
                value.get("schema") == CACHE_SCHEMA
                and value.get("identity") == identity
                and isinstance(value.get("output"), str)
            ):
                return cast(str, value["output"])
    except Exception:
        pass
    return None


def _live_context_tokens(cw: Mapping[str, Any]) -> int | None:
    """Context tokens from the payload's LIVE usage sample, or None when there is none.

    Only a positive reading counts as live: an absent, off-schema or zero sample means
    the harness has nothing to say yet, and the caller falls back to the transcript.
    """
    current_usage = _as_dict(cw.get("current_usage"))
    if not current_usage:
        return None
    try:
        tokens = _num(agent_health.usage_tokens(current_usage))
    except (TypeError, ValueError, OverflowError):
        return None
    if tokens is None or tokens <= 0:
        return None
    return int(tokens)


class ContextFacts(NamedTuple):
    """Everything a payload's ``context_window`` contributes to the drawn panel."""

    tokens: int | None
    pct: float | None
    window_size: int | None


def _context_facts(cw: Mapping[str, Any]) -> ContextFacts:
    """Normalize the payload's ``context_window`` ONCE, for the render and the cache key.

    Both callers read THIS, so the cache identity changes exactly when the rendered
    context could - a second normalization would be free to drift from the first and
    serve a stale panel. Every field is finite/JSON-safe (``_num`` rejects non-finite),
    which the cache write requires (``allow_nan=False``).
    """
    size = _num(cw.get("context_window_size"))
    return ContextFacts(
        tokens=_live_context_tokens(cw),
        pct=_num(cw.get("used_percentage")),
        window_size=int(size) if size is not None and size > 0 else None,
    )


class IdentityFacts(NamedTuple):
    """What the payload puts on the identity row beyond ``_session_identity``."""

    lines_added: float | None
    lines_removed: float | None
    fast_mode: bool


def _identity_facts(payload: Mapping[str, Any]) -> IdentityFacts:
    """Normalize the payload's identity-row facts ONCE, for the render and cache key.

    Same contract as ``_context_facts``: both callers read THIS, so the cache identity
    changes exactly when the drawn row could. Every field is finite/JSON-safe (``_num``
    rejects non-finite), which the cache write requires (``allow_nan=False``).
    """
    cost = _as_dict(payload.get("cost"))
    return IdentityFacts(
        lines_added=_num(cost.get("total_lines_added")),
        lines_removed=_num(cost.get("total_lines_removed")),
        fast_mode=bool(payload.get("fast_mode")),
    )


def build_output(payload: Any) -> str | None:
    """The pure-ish core: stdin payload dict -> the rendered panel string (or None to
    stay silent). Does cost-baseline + subagent I/O but no caching/printing, so it is
    directly unit-testable. Returns None when there is nothing to show."""
    if not isinstance(payload, dict):
        return None
    transcript = payload.get("transcript_path") or ""
    if not transcript:
        return None  # no path -> nothing to render (silent)

    cw = _as_dict(payload.get("context_window"))
    identity = _session_identity(payload)
    identity_facts = _identity_facts(payload)

    session_id = _transcript_session(transcript)
    ui_session_id = payload.get("session_id") or session_id

    if not os.path.isfile(transcript):
        columns = _envi("COLUMNS")
        if columns and columns <= render.NARROW_ACTION_COLS:
            return _cold_marker()
        return _align_rows(_cold_marker, identity, columns)

    analysis = agent_health.analyze_transcript(transcript)
    if analysis.compaction_pending:
        columns = _envi("COLUMNS")

        def compact_row(target_width: int | None = None) -> str:
            return render.render_compact_row(
                columns=columns, no_color=_no_color(), target_width=target_width
            )

        if columns and columns <= render.NARROW_ACTION_COLS:
            return compact_row()
        return _align_rows(compact_row, identity, columns)

    context = _context_facts(cw)
    ctx_pct = context.pct
    window_cls = agent_health.window_class(context.window_size)
    context_is_live = context.tokens is not None
    ctx_tokens = (
        context.tokens if context.tokens is not None else int(analysis.context_tokens)
    )
    blind_loop = agent_health.blind_loop_alert(analysis)
    cx_state, _ = agent_health.context_verdict(
        ctx_tokens, ctx_pct=ctx_pct, window_class=window_cls
    )
    show_agents = _flag("CLAUDE_HEALTH_SHOW_AGENTS", True)

    try:
        cohort = subagents.cohort(
            transcript, session_id, notified=analysis.task_notifications
        )
    except Exception:  # discovery must never take the row down with it
        cohort = render.EMPTY_COHORT

    notice = None
    if not cohort.tokens_warming:
        notice = _next_notice(
            ui_session_id, cohort.done_n, cohort.live_n, show_working=show_agents
        )

    facts = render_facts.RenderFacts(
        context=render_facts.ContextLoad(
            tier=cx_state, tokens=ctx_tokens, pct=ctx_pct, is_live=context_is_live
        ),
        identity=render_facts.Identity(
            lines_added=identity_facts.lines_added,
            lines_removed=identity_facts.lines_removed,
            turns=analysis.turns,
            fast_mode=identity_facts.fast_mode,
            **identity,
        ),
        notice=notice,
        blind_loop=blind_loop,
        cache_health=analysis.cache_health,
        parse_degraded=analysis.parse_health.degraded,
        last_stop_reason=analysis.last_stop_reason,
        subagent_blind_loop=cohort.blind_loop_n > 0,
        show_context=_flag("CLAUDE_HEALTH_SHOW_CONTEXT", True),
        show_agents=show_agents,
        agents_total=cohort.total_n,
        agents_live=cohort.live_n,
        agents_tokens=cohort.tokens_total,
        agents_tokens_complete=cohort.tokens_complete,
        agents_tokens_warming=cohort.tokens_warming,
        agents_models=cohort.models,
    )
    layout = render.LayoutOpts(columns=_envi("COLUMNS"), no_color=_no_color())
    return render.render_panel(facts, layout)


def _envi(name: str) -> int | None:
    try:
        return int(float(os.environ.get(name, "")))
    except (TypeError, ValueError):
        return None


def main() -> None:
    """Read stdin, serve the cache or render, write the cache, print. Silent on every
    failure (the statusline must never crash). The cold-start marker is uncached so real
    data replaces it the instant the first turn lands."""
    try:
        raw_in = sys.stdin.read()
        payload = json.loads(raw_in) if raw_in.strip() else {}
    except Exception:
        return  # unreadable stdin -> silent

    if not isinstance(payload, dict):
        return
    transcript = payload.get("transcript_path") or ""
    if not transcript:
        return

    identity = _session_identity(payload)
    cache_identity: dict[str, Any] = dict(identity)
    cache_identity["columns"] = _envi("COLUMNS")
    cache_identity["no_color"] = _no_color()
    cache_identity["show_context"] = _flag("CLAUDE_HEALTH_SHOW_CONTEXT", True)
    cache_identity["show_agents"] = _flag("CLAUDE_HEALTH_SHOW_AGENTS", True)
    cw = _as_dict(payload.get("context_window"))
    cache_identity["context"] = list(_context_facts(cw))
    cache_identity["identity_facts"] = list(_identity_facts(payload))
    cache_identity["notice_session"] = _session_key(
        payload.get("session_id") or _transcript_session(transcript)
    )
    cache_identity["transcript"] = _transcript_fingerprint(transcript)

    if not os.path.isfile(transcript):
        try:
            out = build_output(payload)
            if out:
                sys.stdout.write(out + "\n")
        except Exception:
            pass
        return

    cache = _cache_path(transcript)
    cached = _read_cache_if_fresh(cache, cache_identity)
    if cached is not None:
        sys.stdout.write(cached)
        return

    try:
        out = build_output(payload)
    except Exception:
        return  # any analysis/render failure -> silent (never a crashed statusline)
    if not out or not out.strip():
        return

    line = out + "\n"
    _write_cache(
        cache, {"schema": CACHE_SCHEMA, "identity": cache_identity, "output": line}
    )
    sys.stdout.write(line)


if __name__ == "__main__":
    main()
