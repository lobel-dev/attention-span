"""Transcript path drivers and compatibility facade for health analysis."""

import dataclasses
import json
import math
import sys
from collections.abc import Iterator, Mapping, Sequence
from os import PathLike
from typing import Any

from attention_span import detectors, events, health_config, text, verdicts
from attention_span.analysis import Analysis as Analysis
from attention_span.analysis import EditLoop as EditLoop
from attention_span.analysis import ParseHealth as ParseHealth
from attention_span.analysis import Perseveration as Perseveration
from attention_span.analysis import Repetition as Repetition
from attention_span.analysis import UsageTotals as UsageTotals
from attention_span.reducer import _append_changed as _append_changed
from attention_span.reducer import _consume as _consume
from attention_span.reducer import _empty_cursor as _empty_cursor
from attention_span.reducer import _finalize as _finalize
from attention_span.reducer import _is_thinking_block as _is_thinking_block
from attention_span.reducer import _new_state as _new_state
from attention_span.reducer import _public_event as _public_event
from attention_span.reducer import _ts_epoch as _ts_epoch
from attention_span.reducer import _usage_int as _usage_int
from attention_span.reducer import usage_tokens as usage_tokens
from attention_span.transcript import iter_lines, iter_objects

DEFAULTS = health_config.ENGINE_DEFAULTS

GLYPH = {"green": "🟢", "yellow": "🟡", "red": "🔴"}


EDIT_TOOLS = events.EDIT_TOOLS
_is_synthetic = events._is_synthetic
classify_bash = events.classify_bash
classify_tool = events.classify_tool
_tool_file_path = events._tool_file_path
_normalize_content = events._normalize_content
classify_error = events.classify_error
_executed_events = detectors._executed_events
_failed_edit_loop = detectors._failed_edit_loop
_percentile = detectors._percentile
_cache_health = detectors._cache_health
_repetition = detectors._repetition
_perseveration = detectors._perseveration
_sanitize = text.sanitize
fmt_tokens = text.fmt_tokens
_thresholds = health_config.resolve_thresholds
blind_loop_alert = verdicts.blind_loop_alert
r2e_posture = verdicts.r2e_posture
context_verdict = verdicts.context_verdict
window_class = verdicts.window_class
model_label = verdicts.model_label


def analyze_transcript(
    path: str | PathLike[str],
    window: int | None = None,
    th: Mapping[str, Any] | None = None,
    include_sidechain: bool = False,
) -> Analysis:
    """Single forward pass over a transcript. Returns an ``Analysis``:

        reads, edits          - read/edit counts within the rolling window
        total_reads, total_edits - lifetime counts (executed only; diagnostics)
        r2e                   - reads/edits within the window (inf if no edits)
        base_tier             - green | yellow | red from R2E alone
        window_used           - read+edit events in the window
        insufficient          - window_used < MIN_WINDOW (engine light off)
        failed_edit_loop      - the ``EditLoop`` (file, count)
        cache_health          - #1 cache-thrash verdict {hit_drop, churn_mult, show,
                                thrash_score, window_turns, suppressed}
        repetition            - #3 read-repetition ``Repetition``
        perseveration         - #4 identical repeated tool calls ``Perseveration``
        parse_health          - #2 self-doubt ``ParseHealth``; analyze_transcript-only
                                (the drift-invariant carve-out)
        last_model            - model id of the most recent real assistant turn
        last_stop_reason      - stop_reason of the last assistant turn ("end_turn" = done)
        compaction_pending    - latest driver compact summary has no later real assistant
        turns                 - count of real (non-synthetic) user turns
        context_tokens        - last assistant usage cache_read+cache_creation+input
        usage_totals          - the ``UsageTotals``: cumulative message.usage sums,
                                deduped LAST-WINS per API call (message.id -> requestId,
                                any-distance; the (cr,cc,inp) tuple fallback
                                consecutive-only)
        max_error_bytes       - largest errored tool_result body (>2KB), else 0
        task_notifications    - subagent task-id -> epoch seconds of its latest
                                parent-side task-notification (a child stop this
                                transcript witnessed; the cohort retires against it)
        thinking_turns        - assistant turns that emitted a signed thinking block
        assistant_turns       - substantive assistant turns (thinking denominator)
        trailing_no_thinking  - consecutive recent substantive turns with no thinking

    Two phases over ONE file read: Phase A accumulates ordered read/edit events
    + result provenance (plus turns/context/error state); Phase B builds the
    rolling window and blind-loop state from that list, EXCLUDING hook-denied
    tool calls (which never executed). Subagent turns (isSidechain) are excluded
    throughout — unless ``include_sidechain=True``, used to analyze a single-agent
    child transcript whose every line is isSidechain.
    """
    state = _new_state()
    line_no = None
    try:
        for line_no, obj in iter_lines(path):
            if obj is None:
                state["decode_failures"] += 1
                continue
            _consume(state, obj, line_no=line_no, include_sidechain=include_sidechain)
    except Exception:
        if line_no is not None:
            state["parse_aborted"] = True
            state["parse_aborted_line"] = line_no

    return _finalize(state, window=window, th=th)


def iter_states(
    path: str | PathLike[str],
    window: int | None = None,
    th: Mapping[str, Any] | None = None,
    at: str = "line",
) -> Iterator[tuple[dict[str, Any], Analysis]]:
    """Yield ``(cursor, snapshot)`` after lines that change analysis state.

    Cursor metadata includes settled read/edit events, so calibration can emit a
    feature row only after execution status is known. PENDING read/edit events
    still appear in live snapshots, matching ``analyze_transcript`` for trailing
    in-flight tool uses.
    """
    if at != "line":
        raise ValueError("iter_states currently supports at='line' only")

    state = _new_state()
    try:
        for line_no, obj in iter_objects(path):
            meta = _consume(state, obj, line_no=line_no)
            if not meta["changed"]:
                continue
            events = [_public_event(ev) for ev in state["ordered"]]
            cursor = {
                "line_no": line_no,
                "event_index": state["event_index"],
                "changed": meta["changed"],
                "settled_tool_use_ids": meta["settled_tool_use_ids"],
                "settled_events": meta["settled_events"],
                "new_events": meta["new_events"],
                "pending_events": [
                    ev for ev in events if ev.get("result_state") == "PENDING"
                ],
                "events": events,
            }
            yield cursor, _finalize(state, window=window, th=th)
    except Exception:
        return


USAGE = "usage: agent_health.py <transcript.jsonl> [ctx_pct] [model_id] [window_size]"


def _cli_float(arguments: Sequence[str], index: int) -> float | None:
    """Positional argument ``index`` as a FINITE float, or None when absent/off-schema.

    "nan"/"inf" parse as floats but are off-schema for both consumers: context_verdict
    formats ctx_pct with int() (ValueError on nan, OverflowError on inf), and a
    non-finite window size is never a real advertised window. Rejected HERE, at the
    argv boundary, so no downstream caller has to re-check.
    """
    if len(arguments) <= index or not arguments[index]:
        return None
    try:
        value = float(arguments[index])
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _model_suffix(stdin_model: str | None, last_model: str | None) -> str:
    """The trailing model chip: '⇄ label' when the live model differs from the last."""
    last_label = model_label(last_model or "")
    label = model_label(stdin_model) if stdin_model else last_label
    if not label:
        return ""
    swapped = bool(stdin_model) and bool(last_label) and label != last_label
    return f"  · {'⇄ ' if swapped else ''}{label}"


def _summary_line(
    analysis: Analysis,
    ctx_pct: float | None,
    win_cls: str | None,
    stdin_model: str | None,
) -> str:
    """The one-line human verdict the debug CLI prints on stdout."""
    # Imported HERE, never in the module body
    from attention_span import status_catalog

    blind_loop = blind_loop_alert(analysis)
    _, r2e_chip = r2e_posture(analysis)
    cx_state, cx_chip = context_verdict(
        analysis.context_tokens, ctx_pct=ctx_pct, window_class=win_cls
    )

    parts = []
    if blind_loop:
        parts.append("{}  {}".format(GLYPH["red"], blind_loop["chip"]))
    if cx_state is not None:
        tier = status_catalog.STATUSES.get(cx_state, status_catalog.FALLBACK)
        parts.append(f"{tier.glyph} {tier.action or cx_state} {cx_chip}")

    line = "  ".join(parts) if parts else "◌ no signal"
    line += f"  · {analysis.turns} turns"
    if r2e_chip:
        line += f"  · {r2e_chip} ? (provisional)"
    if analysis.cache_health["show"]:
        line += "  · ⚠ cache {}%↓".format(int(round(analysis.cache_health["hit_drop"])))
    if analysis.repetition.score >= DEFAULTS["REPEAT_MIN"]:
        line += f"  · ⚠ repeat x{analysis.repetition.score}"
    if analysis.perseveration.score >= DEFAULTS["PERSEV_MIN"]:
        line += f"  · ⚠ persev x{analysis.perseveration.score}"
    if analysis.parse_health.degraded:
        line += "  · ◌ health data stale?"
    return line + _model_suffix(stdin_model, analysis.last_model)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the transcript-analysis debug CLI."""
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if not arguments:
        print(USAGE, file=sys.stderr)
        return 2

    ctx_pct = _cli_float(arguments, 1)
    stdin_model = arguments[2] if len(arguments) > 2 else None
    window_size = _cli_float(arguments, 3)
    win_cls = window_class(window_size) if window_size is not None else None

    analysis = analyze_transcript(arguments[0])
    summary = dataclasses.asdict(analysis)
    summary["r2e"] = "inf" if analysis.r2e == float("inf") else round(analysis.r2e, 3)
    summary["window_class"] = win_cls
    print(json.dumps(summary, default=str), file=sys.stderr)
    print(_summary_line(analysis, ctx_pct, win_cls, stdin_model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
