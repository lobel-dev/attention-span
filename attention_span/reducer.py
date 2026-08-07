"""Reduce transcript objects into immutable analysis snapshots."""

import hashlib
import json
import re
from collections import deque
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from attention_span import detectors, events, health_config
from attention_span.analysis import Analysis, ParseHealth, UsageTotals

TASK_NOTIFICATION_KIND = "task-notification"
_TASK_ID_RE = re.compile(r"<task-id>([^<]*)</task-id>")


def _cache_triple(usage: Mapping[str, Any]) -> tuple[int, int, int]:
    """The (cache_read, cache_creation, input) token trio of a ``message.usage``.

    The ONE place those three keys are read; ``or 0`` guards null-valued keys. The int
    annotation is the SCHEMA, not a coercion: JSON can still smuggle an off-schema value
    through, which is what ``_usage_int`` exists to absorb at the accumulator.
    """
    return (
        usage.get("cache_read_input_tokens", 0) or 0,
        usage.get("cache_creation_input_tokens", 0) or 0,
        usage.get("input_tokens", 0) or 0,
    )


def usage_tokens(usage: Any) -> int:
    """Context-window size from an assistant ``message.usage``: the canonical
    cache_read + cache_creation + input sum (the schema's ``context_tokens``).
    The single home for this formula — calibration imports it instead of
    re-summing the keys."""
    cr, cc, inp = _cache_triple(usage or {})
    return cr + cc + inp


def _usage_int(v: Any) -> int:
    """An off-schema-tolerant token count: a non-bool int passes, anything else -> 0.

    The usage_totals accumulator sums these, so a JSON-permitted-but-off-schema value
    (a string, null, a float) must coerce rather than raise mid-pass (the same
    philosophy as the existing usage read's ``or 0`` guards, one step stricter)."""
    return v if isinstance(v, int) and not isinstance(v, bool) else 0


def _ts_epoch(ts: Any) -> float | None:
    """A transcript ISO-8601 timestamp -> epoch seconds (float), or None.

    The ONE site that parses transcript timestamps (cache-health idle suppression
    needs inter-turn gaps). Parsed HERE in the O(n) `_consume` pass and stored as a
    number so `_finalize` — which re-runs per-emit in the O(n^2) `iter_states` walk —
    only does arithmetic, never re-parses. Handles the Python-3.8 `fromisoformat`
    gotcha (it rejects a trailing `Z` before 3.11) by normalizing `Z` -> `+00:00`.
    Returns None on absent/off-schema input so callers degrade rather than crash.
    """
    if not ts or not isinstance(ts, str):
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _digest(text: str) -> str:
    """A stable identity for a block of transcript text, compared but never read back.

    Digested rather than kept whole because the perseveration call log holds one entry
    per tool call for the whole session, and both the things it identifies - a Write's
    input body, a Read's result - run to kilobytes. ``surrogatepass`` because a
    transcript string can carry a lone surrogate (JSON permits a bare \\udXXX escape)
    that plain UTF-8 encoding refuses, and an encode raised inside `_consume` would
    abort the whole pass and silently freeze every metric at that line.
    """
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _call_signature(name: Any, tool_input: Mapping[str, Any]) -> str:
    """The identity of one tool CALL: its tool name plus a digest of its input.

    Sorted keys so two spellings of the same input compare equal. ``default=str``
    absorbs a JSON-permitted-but-unexpected value rather than raising mid-pass, the
    same off-schema philosophy as ``_usage_int``. Computed HERE in the O(n) `_consume`
    pass, not in the detector, which re-runs per emit in the O(n^2) `iter_states` walk.
    """
    payload = json.dumps(tool_input, sort_keys=True, default=str)
    return f"{name}\x00{_digest(payload)}"


def _new_state() -> dict[str, Any]:
    """Create Phase-A accumulators for transcript analysis."""
    return {
        "ordered": [],
        "calls": [],
        "id_to_call": {},
        "id_to_error": {},
        "id_to_file": {},
        "id_to_event": {},
        "last_model": None,
        "last_stop_reason": None,
        "compaction_pending": False,
        "context_tokens": 0,
        "turns": 0,
        "max_error_bytes": 0,
        "event_index": 0,
        "thinking_turns": 0,
        "assistant_turns": 0,
        "trailing_no_thinking": 0,
        "cache_turns": [],
        "assistant_lines": 0,
        "usage_seen": False,
        "tooluse_seen": False,
        "task_notifications": {},
        "decode_failures": 0,
        "parse_aborted": False,
        "parse_aborted_line": None,
        "usage_totals": {
            "input": 0,
            "cache_creation": 0,
            "cache_read": 0,
            "output": 0,
            "api_calls": 0,
        },
        "_usage_contrib_by_id": {},
        "_last_usage_key": None,
        "_last_usage_contrib": None,
    }


def _append_changed(changed: list[str], name: str) -> None:
    if name not in changed:
        changed.append(name)


def _public_event(ev: Mapping[str, Any]) -> dict[str, Any]:
    """Small, serializable read/edit event view for calibration cursors."""
    return {
        "tool_use_id": ev.get("tool_use_id"),
        "event_index": ev.get("event_index"),
        "line_no": ev.get("line_no"),
        "class": ev.get("class"),
        "tool_name": ev.get("tool_name"),
        "file_path": ev.get("file_path"),
        "result_state": ev.get("result_state", "PENDING"),
        "result_line_no": ev.get("result_line_no"),
    }


def _is_thinking_block(it: Mapping[str, Any]) -> bool:
    """True if a content item is realized reasoning (a signed thinking block).

    Key off the signature, not the `thinking` text: Claude Code stores the text
    redacted (empty) but keeps the cryptographic signature, so signature presence
    is the reliable evidence the model actually reasoned this turn. Encrypted
    `redacted_thinking` blocks count too.
    """
    ty = it.get("type")
    if ty == "thinking":
        return bool(it.get("signature")) or bool(it.get("thinking"))
    return ty == "redacted_thinking"


def _empty_cursor() -> dict[str, Any]:
    """Return fresh metadata for an object that changed no analysis input."""
    return {
        "changed": [],
        "settled_tool_use_ids": [],
        "settled_events": [],
        "new_events": [],
    }


def _consume_compaction_marker(
    state: dict[str, Any], obj: Mapping[str, Any], changed: list[str]
) -> None:
    if obj.get("isCompactSummary") is True and not state["compaction_pending"]:
        state["compaction_pending"] = True
        _append_changed(changed, "compaction_pending")


def _consume_assistant_header(
    state: dict[str, Any], msg: Mapping[str, Any], changed: list[str]
) -> Any:
    state["assistant_lines"] += 1
    model = msg.get("model")
    if (
        state["compaction_pending"]
        and isinstance(model, str)
        and model
        and model != "<synthetic>"
    ):
        state["compaction_pending"] = False
        _append_changed(changed, "compaction_pending")
    if model and model != "<synthetic>" and model != state["last_model"]:
        state["last_model"] = model
        _append_changed(changed, "last_model")

    stop_reason = msg.get("stop_reason")
    if stop_reason and stop_reason != state["last_stop_reason"]:
        state["last_stop_reason"] = stop_reason
        _append_changed(changed, "last_stop_reason")
    return model


def _consume_cache_turn(
    state: dict[str, Any],
    usage: Mapping[str, Any],
    tokens: Any,
    ts_epoch: float | None,
    changed: list[str],
) -> None:
    cr, cc, inp = _cache_triple(usage)
    prev = state["cache_turns"][-1] if state["cache_turns"] else None
    if prev is None or (cr, cc, inp) != (prev["cr"], prev["cc"], prev["inp"]):
        state["cache_turns"].append(
            {"cr": cr, "cc": cc, "inp": inp, "ctx": tokens, "ts_epoch": ts_epoch}
        )
        _append_changed(changed, "cache_turns")


def _apply_usage_contribution(
    totals: dict[str, int],
    contribution: Mapping[str, int],
    previous: Mapping[str, int] | None = None,
) -> None:
    """Fold one API call's usage into the running totals, superseding ``previous``.

    A re-logged streaming line restates a call already counted, so its earlier
    contribution is backed out in the same step (LAST-WINS) rather than double-counted.
    """
    for key, value in contribution.items():
        totals[key] += value - (previous[key] if previous is not None else 0)


def _consume_identified_usage(
    state: dict[str, Any],
    id_key: Any,
    contribution: dict[str, int],
    changed: list[str],
) -> Any:
    totals = state["usage_totals"]
    by_id = state["_usage_contrib_by_id"]
    previous = by_id.get(id_key)
    if previous is None:
        _apply_usage_contribution(totals, contribution)
        totals["api_calls"] += 1
        by_id[id_key] = contribution
        _append_changed(changed, "usage_totals")
    elif contribution != previous:
        _apply_usage_contribution(totals, contribution, previous)
        by_id[id_key] = contribution
        _append_changed(changed, "usage_totals")
    return id_key


def _consume_unidentified_usage(
    state: dict[str, Any], contribution: dict[str, int], changed: list[str]
) -> tuple[str, int, int, int]:
    key = (
        "cr_cc_inp",
        contribution["cache_read"],
        contribution["cache_creation"],
        contribution["input"],
    )
    totals = state["usage_totals"]
    if key == state["_last_usage_key"] and state["_last_usage_contrib"] is not None:
        previous = state["_last_usage_contrib"]
        if contribution != previous:
            _apply_usage_contribution(totals, contribution, previous)
            _append_changed(changed, "usage_totals")
    else:
        _apply_usage_contribution(totals, contribution)
        totals["api_calls"] += 1
        _append_changed(changed, "usage_totals")
    return key


def _consume_usage_totals(
    state: dict[str, Any],
    obj: Mapping[str, Any],
    msg: Mapping[str, Any],
    usage: Mapping[str, Any],
    changed: list[str],
) -> None:
    cr, cc, inp = _cache_triple(usage)
    contribution = {
        "input": _usage_int(inp),
        "cache_creation": _usage_int(cc),
        "cache_read": _usage_int(cr),
        "output": _usage_int(usage.get("output_tokens", 0)),
    }
    id_key = msg.get("id") or obj.get("requestId")
    if id_key is not None:
        key = _consume_identified_usage(state, id_key, contribution, changed)
    else:
        key = _consume_unidentified_usage(state, contribution, changed)
    state["_last_usage_key"] = key
    state["_last_usage_contrib"] = contribution


def _consume_assistant_usage(
    state: dict[str, Any],
    obj: Mapping[str, Any],
    msg: Mapping[str, Any],
    model: Any,
    ts_epoch: float | None,
    changed: list[str],
) -> None:
    usage = msg.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    if usage:
        state["usage_seen"] = True
    if not usage or model == "<synthetic>":
        return

    tokens = usage_tokens(usage)
    if tokens != state["context_tokens"]:
        state["context_tokens"] = tokens
        _append_changed(changed, "context_tokens")
    _consume_cache_turn(state, usage, tokens, ts_epoch, changed)
    _consume_usage_totals(state, obj, msg, usage, changed)


def _consume_thinking(
    state: dict[str, Any], content: list[Any], changed: list[str]
) -> None:
    has_thinking = any(
        isinstance(item, dict) and _is_thinking_block(item) for item in content
    )
    has_output = any(
        isinstance(item, dict) and item.get("type") in ("text", "tool_use")
        for item in content
    )
    if not has_output and not has_thinking:
        return

    state["assistant_turns"] += 1
    _append_changed(changed, "assistant_turns")
    if has_thinking:
        state["thinking_turns"] += 1
        _append_changed(changed, "thinking_turns")
        if state["trailing_no_thinking"] != 0:
            state["trailing_no_thinking"] = 0
            _append_changed(changed, "trailing_no_thinking")
    else:
        state["trailing_no_thinking"] += 1
        _append_changed(changed, "trailing_no_thinking")


def _consume_tool_use(
    state: dict[str, Any],
    item: Mapping[str, Any],
    line_no: int | None,
    cwd: Any,
    ts_epoch: float | None,
    changed: list[str],
    new_events: list[dict[str, Any]],
) -> None:
    state["tooluse_seen"] = True
    name = item.get("name", "")
    tool_input = item.get("input")
    if not isinstance(tool_input, dict):
        tool_input = {}
    event_class = events.classify_tool(name, tool_input)
    tool_use_id = item.get("id")

    # Logged BEFORE the read/edit gate (calls are neutral until they reach `ordered`)
    # result state and body digest arrive later, in `_settle_tool_result`
    call = {
        "tool_use_id": tool_use_id,
        "class": event_class,
        "tool_name": name,
        "sig": _call_signature(name, tool_input),
        "ts_epoch": ts_epoch,
        "result_state": "PENDING",
        "result_hash": None,
    }
    state["calls"].append(call)
    _append_changed(changed, "calls")
    if tool_use_id:
        state["id_to_call"][tool_use_id] = call

    if event_class not in ("read", "edit"):
        return

    file_path = events._tool_file_path(name, tool_input, cwd)
    state["event_index"] += 1
    event = {
        "tool_use_id": tool_use_id,
        "event_index": state["event_index"],
        "line_no": line_no,
        "class": event_class,
        "tool_name": name,
        "file_path": file_path,
        "result_state": "PENDING",
        "result_line_no": None,
    }
    state["ordered"].append(event)
    new_events.append(event)
    _append_changed(changed, "ordered")
    if tool_use_id:
        state["id_to_event"][tool_use_id] = event
    if (
        event_class == "edit"
        and name in events.EDIT_TOOLS
        and file_path
        and tool_use_id
    ):
        state["id_to_file"][tool_use_id] = file_path
        _append_changed(changed, "id_to_file")


def _consume_tool_uses(
    state: dict[str, Any],
    content: list[Any],
    line_no: int | None,
    cwd: Any,
    ts_epoch: float | None,
    changed: list[str],
    new_events: list[dict[str, Any]],
) -> None:
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "tool_use":
            continue
        _consume_tool_use(state, item, line_no, cwd, ts_epoch, changed, new_events)


def _consume_assistant(
    state: dict[str, Any],
    obj: Mapping[str, Any],
    msg: Mapping[str, Any],
    content: Any,
    line_no: int | None,
    changed: list[str],
    new_events: list[dict[str, Any]],
) -> None:
    model = _consume_assistant_header(state, msg, changed)
    ts_epoch = _ts_epoch(obj.get("timestamp"))
    _consume_assistant_usage(state, obj, msg, model, ts_epoch, changed)
    if isinstance(content, list):
        _consume_thinking(state, content, changed)
        _consume_tool_uses(
            state, content, line_no, obj.get("cwd"), ts_epoch, changed, new_events
        )


def _injected_user_cursor(
    obj: Mapping[str, Any], changed: list[str]
) -> dict[str, Any] | None:
    """The cursor for a harness-injected user line, or None when it is a real turn.

    A compact summary or meta line settles nothing and starts nothing; only the
    compaction marker read before this point can have changed anything.
    """
    if not (obj.get("isCompactSummary") or obj.get("isMeta")):
        return None
    return {**_empty_cursor(), "changed": changed}


def _consume_task_notification(
    state: dict[str, Any], obj: Mapping[str, Any], content: Any, changed: list[str]
) -> None:
    """Record the LATEST stop time per task-id from a parent-side task-notification.

    Gated on the ``origin.kind`` marker, never on the text: a compaction digest and a
    pasted conversation log both carry whole notification blocks verbatim (observed
    shapes), and neither says a live agent stopped. ``<status>`` is not read either -
    the notification fires exactly when an agent stops, so a killed run is evidenced
    the same way a finished one is. Keeping TIMES rather than a done bit is what lets
    a resumed agent, which notifies again and writes again, be told apart from a
    retired one.
    """
    origin = obj.get("origin")
    if not isinstance(origin, Mapping) or origin.get("kind") != TASK_NOTIFICATION_KIND:
        return
    if not isinstance(content, str):
        return
    match = _TASK_ID_RE.search(content)
    ts_epoch = _ts_epoch(obj.get("timestamp"))
    if match is None or ts_epoch is None:
        return
    task_id = match.group(1).strip()
    if not task_id:
        return
    notifications = state["task_notifications"]
    if ts_epoch > notifications.get(task_id, float("-inf")):
        notifications[task_id] = ts_epoch
        _append_changed(changed, "task_notifications")


def _settle_tool_result(
    state: dict[str, Any],
    item: Mapping[str, Any],
    line_no: int | None,
    changed: list[str],
    settled: list[Any],
) -> None:
    tool_use_id = item.get("tool_use_id")
    result_state = events.classify_error(item.get("is_error"), item.get("content"))
    body = events._normalize_content(item.get("content"))
    if tool_use_id:
        state["id_to_error"][tool_use_id] = result_state
        _append_changed(changed, "id_to_error")
        event = state["id_to_event"].get(tool_use_id)
        if event is not None:
            event["result_state"] = result_state
            event["result_line_no"] = line_no
            settled.append(tool_use_id)
        call = state["id_to_call"].get(tool_use_id)
        if call is not None:
            call["result_state"] = result_state
            call["result_hash"] = _digest(body)
            _append_changed(changed, "calls")

    if (
        item.get("is_error")
        and len(body) > 2048
        and len(body) > state["max_error_bytes"]
    ):
        state["max_error_bytes"] = len(body)
        _append_changed(changed, "max_error_bytes")


def _consume_user_items(
    state: dict[str, Any],
    content: list[Any],
    line_no: int | None,
    changed: list[str],
    settled: list[Any],
) -> None:
    has_real = False
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "tool_result":
            _settle_tool_result(state, item, line_no, changed, settled)
        elif item_type == "text":
            if not events._is_synthetic(item.get("text", "")):
                has_real = True
        else:
            has_real = True
    if has_real:
        state["turns"] += 1
        _append_changed(changed, "turns")


def _consume_user(
    state: dict[str, Any],
    content: Any,
    line_no: int | None,
    changed: list[str],
    settled: list[Any],
) -> None:
    if isinstance(content, str):
        if content.strip() and not events._is_synthetic(content):
            state["turns"] += 1
            _append_changed(changed, "turns")
    elif isinstance(content, list):
        _consume_user_items(state, content, line_no, changed, settled)


def _finish_cursor(
    state: dict[str, Any],
    changed: list[str],
    settled: list[Any],
    new_events: list[dict[str, Any]],
) -> dict[str, Any]:
    settled_events = [
        _public_event(state["id_to_event"][tool_use_id])
        for tool_use_id in settled
        if tool_use_id in state["id_to_event"]
    ]
    return {
        "changed": changed,
        "settled_tool_use_ids": settled,
        "settled_events": settled_events,
        "new_events": [_public_event(event) for event in new_events],
    }


def _consume(
    state: dict[str, Any],
    obj: Mapping[str, Any],
    line_no: int | None = None,
    include_sidechain: bool = False,
) -> dict[str, Any]:
    """Consume one transcript object into Phase-A state and return cursor metadata."""
    # `json.loads` accepts ANY JSON value, so a transcript line holding `"hello"`, `42`
    # or `[1, 2]` decodes to a real (non-None) NON-mapping and arrives here. Skipping it
    # is the only tolerant option: an AttributeError raised mid-pass aborts the whole
    # walk (parse_aborted) and silently drops every LATER line. Same off-schema
    # philosophy as `_usage_int` and the usage isinstance check.
    if not isinstance(obj, Mapping):
        return _empty_cursor()

    changed: list[str] = []
    settled: list[Any] = []
    new_events: list[dict[str, Any]] = []

    if obj.get("isSidechain") and not include_sidechain:
        return _empty_cursor()

    object_type = obj.get("type")
    message = obj.get("message") or {}
    if not isinstance(message, Mapping):
        message = {}  # an off-schema scalar/list `message` carries none of its fields
    content = message.get("content")
    _consume_compaction_marker(state, obj, changed)

    if object_type == "assistant":
        _consume_assistant(state, obj, message, content, line_no, changed, new_events)
    elif object_type == "user":
        # Read BEFORE the injected-line bail: a stop is terminal evidence whatever
        # harness flags the line happens to carry.
        _consume_task_notification(state, obj, content, changed)
        injected = _injected_user_cursor(obj, changed)
        if injected is not None:
            return injected
        _consume_user(state, content, line_no, changed, settled)

    return _finish_cursor(state, changed, settled, new_events)


def _finalize(
    state: dict[str, Any],
    window: Any = None,
    th: Mapping[str, Any] | None = None,
) -> Analysis:
    """Build the public ``Analysis`` from Phase-A state. The ONE constructor."""
    thresholds = health_config.resolve_thresholds(th)
    if window is None:
        window = int(thresholds["WINDOW"])

    win: deque[Any] = deque(maxlen=window)
    total_reads = 0
    total_edits = 0
    for event in detectors._executed_events(state["ordered"]):
        event_class = event.get("class")
        win.append(event_class)
        if event_class == "read":
            total_reads += 1
        elif event_class == "edit":
            total_edits += 1

    reads = win.count("read")
    edits = win.count("edit")
    window_used = reads + edits
    if edits == 0:
        r2e = float("inf")
    elif reads == 0:
        r2e = 0.0
    else:
        r2e = reads / edits

    if r2e == float("inf") or r2e >= thresholds["HEALTHY"]:
        base = "green"
    elif r2e >= thresholds["BLIND"]:
        base = "yellow"
    else:
        base = "red"

    schema_canary = (
        state["assistant_lines"] >= thresholds["PARSE_CANARY_MIN_ASSISTANT"]
        and not state["usage_seen"]
        and not state["tooluse_seen"]
    )
    parse_degraded = bool(
        state["parse_aborted"]
        or schema_canary
        or state["decode_failures"] >= thresholds["PARSE_DECODE_MIN"]
    )

    return Analysis(
        reads=reads,
        edits=edits,
        total_reads=total_reads,
        total_edits=total_edits,
        r2e=r2e,
        base_tier=base,
        window_used=window_used,
        insufficient=window_used < thresholds["MIN_WINDOW"],
        failed_edit_loop=detectors._failed_edit_loop(state["ordered"]),
        cache_health=detectors._cache_health(state["cache_turns"], thresholds),
        repetition=detectors._repetition(state["ordered"], thresholds),
        perseveration=detectors._perseveration(state["calls"], thresholds),
        parse_health=ParseHealth(
            decode_failures=state["decode_failures"],
            assistant_lines=state["assistant_lines"],
            usage_seen=state["usage_seen"],
            tooluse_seen=state["tooluse_seen"],
            parse_aborted=state["parse_aborted"],
            schema_canary=bool(schema_canary),
            degraded=parse_degraded,
        ),
        last_model=state["last_model"],
        last_stop_reason=state["last_stop_reason"],
        compaction_pending=state["compaction_pending"],
        turns=state["turns"],
        context_tokens=state["context_tokens"],
        usage_totals=UsageTotals(**state["usage_totals"]),
        max_error_bytes=state["max_error_bytes"],
        task_notifications=dict(state["task_notifications"]),
        thinking_turns=state["thinking_turns"],
        assistant_turns=state["assistant_turns"],
        trailing_no_thinking=state["trailing_no_thinking"],
    )
