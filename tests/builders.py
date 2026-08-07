"""Shared transcript builders for tests."""

import datetime as _dt
import json
import os
import tempfile
from collections.abc import Sequence
from os import PathLike
from typing import Any, TypeAlias

from attention_span import agent_health
from attention_span.analysis import Analysis

JsonObject: TypeAlias = dict[str, Any]
ToolUse: TypeAlias = tuple[str, str, JsonObject]
Usage: TypeAlias = dict[str, int]
ToolResult: TypeAlias = tuple[str, bool, str]
CacheSpec: TypeAlias = tuple[int, int, int]  # (cache_read, cache_creation, input)

_T0 = _dt.datetime(2026, 6, 29, 12, 0, 0, tzinfo=_dt.UTC)


def iso_ts(seconds_offset: int | float) -> str:
    """A Z-suffixed ISO-8601 UTC timestamp `seconds_offset` after the fixed base."""
    return (
        (_T0 + _dt.timedelta(seconds=seconds_offset)).isoformat().replace("+00:00", "Z")
    )


def usage(cr: int = 0, cc: int = 0, inp: int = 0, out: int = 0) -> Usage:
    """A message.usage dict: cache_read / creation / input (the cache-health series)
    plus output_tokens (the usage_totals burn accumulator)."""
    return {
        "cache_read_input_tokens": cr,
        "cache_creation_input_tokens": cc,
        "input_tokens": inp,
        "output_tokens": out,
    }


def asst(
    tool_uses: Sequence[ToolUse],
    model: str = "claude-opus-4-8",
    usage: Usage | None = None,
    thinking: str | bool | None = None,
    child: bool = False,
    agent_id: str | None = None,
    stop_reason: str | None = None,
    timestamp: str | None = None,
    msg_id: str | None = None,
    request_id: str | None = None,
    cwd: str | None = None,
) -> JsonObject:
    """An assistant turn carrying a list of (id, name, input) tool_uses.

    thinking: if not None, prepend a signed thinking block — pass a string for
    explicit text, or True for a realistic redacted (empty-text) signed block
    (matches how Claude Code stores thinking: empty text, signature present).
    child: mark as a subagent turn (isSidechain True + optional agentId), as a
    child transcript line is stored. stop_reason: assistant message.stop_reason
    (e.g. "end_turn" for done-detection). timestamp: top-level ISO-8601 string
    (see iso_ts) — the cache-health series parses it for inter-turn idle gaps.
    msg_id / request_id: message.id and the top-level requestId — the
    usage_totals per-API-call dedupe keys (streaming re-log collapse).
    cwd: the record's top-level working directory, as Claude Code stamps on every
    line — what a relative shell-read operand (`cat config.json`) anchors to.
    """
    content: list[JsonObject] = []
    if thinking is not None:
        text = "" if thinking is True else str(thinking)
        content.append({"type": "thinking", "thinking": text, "signature": "sig-test"})
    content.extend(
        {"type": "tool_use", "id": tid, "name": name, "input": inp}
        for (tid, name, inp) in tool_uses
    )
    msg: JsonObject = {"model": model, "content": content}
    if usage is not None:
        msg["usage"] = usage
    if stop_reason is not None:
        msg["stop_reason"] = stop_reason
    if msg_id is not None:
        msg["id"] = msg_id
    obj: JsonObject = {"type": "assistant", "isSidechain": bool(child), "message": msg}
    if timestamp is not None:
        obj["timestamp"] = timestamp
    if cwd is not None:
        obj["cwd"] = cwd
    if request_id is not None:
        obj["requestId"] = request_id
    if child and agent_id:
        obj["agentId"] = agent_id
    return obj


def synthetic_api_error(timestamp: str | None = None) -> JsonObject:
    """The line Claude Code writes for an API error mid-session: `model` is the
    `"<synthetic>"` marker and `usage` is complete but ALL-ZERO (verbatim shape from
    the synthetic-zero-usage PRD, verified against a real transcript). The engine must
    treat its usage as fabricated - schema-valid, value-empty."""
    obj = asst([], model="<synthetic>", usage=usage(), timestamp=timestamp)
    obj["isApiErrorMessage"] = True
    return obj


def results(items: Sequence[ToolResult], child: bool = False) -> JsonObject:
    """A user turn carrying a list of (tool_use_id, is_error, content) results."""
    return {
        "type": "user",
        "isSidechain": bool(child),
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tid,
                    "is_error": is_err,
                    "content": content,
                }
                for (tid, is_err, content) in items
            ]
        },
    }


def user_str(text: str) -> JsonObject:
    """A user turn whose content is a plain string."""
    return {"type": "user", "isSidechain": False, "message": {"content": text}}


def task_notification(
    task_id: str,
    timestamp: str | None = None,
    status: str = "completed",
) -> JsonObject:
    """The PARENT-side line Claude Code writes when a subagent stops.

    Verbatim shape from a real transcript: a ``user`` line whose ``message.content`` is
    a BARE STRING holding the block, authoritatively marked by ``origin.kind`` (present
    on every one of the 203 notifications across Claude Code 2.1.207-2.1.224). It never
    carries isMeta or isCompactSummary, and the harness fires one EACH time the agent
    stops - a resumed agent notifies again under the same task-id.
    """
    if timestamp is None:
        timestamp = iso_ts(0)
    content = (
        "<task-notification>\n"
        f"<task-id>{task_id}</task-id>\n"
        f"<tool-use-id>toolu_{task_id}</tool-use-id>\n"
        f"<output-file>/tmp/tasks/{task_id}.output</output-file>\n"
        f"<status>{status}</status>\n"
        f'<summary>Agent "{task_id}" finished</summary>\n'
        "<note>A task-notification fires each time this agent stops with no live "
        "background children of its own. The user can send it another message and "
        "resume it, so the same task-id may notify more than once.</note>\n"
        "<result>Done.</result>\n"
        "<usage><subagent_tokens>58809</subagent_tokens><tool_uses>18</tool_uses>"
        "<duration_ms>88028</duration_ms></usage>\n"
        "</task-notification>"
    )
    return {
        "parentUuid": f"parent-{task_id}",
        "isSidechain": False,
        "type": "user",
        "message": {"role": "user", "content": content},
        "timestamp": timestamp,
        "origin": {"kind": "task-notification"},
        "promptSource": "system",
        "userType": "external",
    }


def user_compact_summary(text: str | None = None, as_list: bool = False) -> JsonObject:
    """A post-compaction continuation message, as Claude Code injects it after /compact.

    Marked by the authoritative top-level ``isCompactSummary`` flag (verified against a
    real transcript) — it carries the prior-session digest, NOT a real human turn.
    ``as_list`` wraps the digest in a content[] text item instead of a bare string.
    """
    if text is None:
        text = (
            "This session is being continued from a previous conversation that ran "
            "out of context. The conversation is summarized below:\n\nSummary:\n[digest]"
        )
    content = [{"type": "text", "text": text}] if as_list else text
    return {
        "type": "user",
        "isSidechain": False,
        "isCompactSummary": True,
        "isVisibleInTranscriptOnly": True,
        "message": {"role": "user", "content": content},
    }


def system_compact_boundary(
    trigger: str = "manual",
    pre_tokens: int = 240_000,
    post_tokens: int = 12_000,
    preserved: bool = True,
) -> JsonObject:
    """Claude Code's compaction marker line, as written for BOTH `/compact` and auto.

    Verified against real transcripts: every `compact_boundary` there carries
    ``preservedSegment`` — Claude Code keeps a segment of recent messages across a
    compaction. That segment is why the post-compact statusline payload still reports a
    PRE-compact ``current_usage``: the payload is built from the message list sliced at
    this boundary, and the preserved turns sit inside that slice.
    """
    meta: JsonObject = {
        "trigger": trigger,
        "preTokens": pre_tokens,
        "postTokens": post_tokens,
        "cumulativeDroppedTokens": max(0, pre_tokens - post_tokens),
    }
    if preserved:
        meta["preservedSegment"] = {
            "headUuid": "head-uuid",
            "anchorUuid": "anchor-uuid",
            "tailUuid": "tail-uuid",
        }
    return {
        "type": "system",
        "subtype": "compact_boundary",
        "isSidechain": False,
        "content": "Conversation compacted",
        "level": "info",
        "compactMetadata": meta,
    }


THRASH_SPECS: tuple[CacheSpec, ...] = tuple(
    [(5000 + i, 5000, 500) for i in range(3)]
    + [(9000, 500, 500), (9010, 490, 500)]
    + [(400 + i, 9100 - i, 500) for i in range(8)]
)


def cache_objs(specs: Sequence[CacheSpec], gap: int = 5) -> list[JsonObject]:
    """Assistant turns with DISTINCT usage (avoiding the consecutive-dup collapse) and
    incrementing Z-timestamps `gap` seconds apart — the #1 cache-health series."""
    return [
        asst([], usage=usage(cr, cc, inp), timestamp=iso_ts((i + 1) * gap))
        for i, (cr, cc, inp) in enumerate(specs)
    ]


def read_tu(i: int, path: str = "/repo/a.py") -> ToolUse:
    return (f"r{i}", "Read", {"file_path": path})


def edit_tu(i: int, path: str = "/repo/a.py") -> ToolUse:
    return (f"e{i}", "Edit", {"file_path": path})


def bash_tu(i: int, command: str = "npm test") -> ToolUse:
    """A NEUTRAL shell tool_use: neither read nor edit, so it never enters the R2E
    window. The shape a stalled agent re-runs, which only #4 perseveration sees."""
    return (f"b{i}", "Bash", {"command": command})


def write_lines(
    objs: Sequence[JsonObject], path: str | PathLike[str] | None = None
) -> str:
    """Write objects as JSONL, returning the path. ``None`` writes a fresh temp file."""
    if path is None:
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
    output_path = os.fspath(path)
    with open(output_path, "w") as f:
        f.writelines(json.dumps(o) + "\n" for o in objs)
    return output_path


def analyze(objs: Sequence[JsonObject]) -> Analysis:
    """Run the engine over a throwaway JSONL transcript built from ``objs``."""
    path = write_lines(objs)
    try:
        return agent_health.analyze_transcript(path)
    finally:
        os.unlink(path)


GENUINE = "<tool_use_error>String to replace not found</tool_use_error>"
GENUINE_UNREAD = (
    "<tool_use_error>File has not been read yet. Read it first.</tool_use_error>"
)
HOOKDENY = (
    "[Fact-Forcing Gate]\n\nBefore editing /repo/a.py, present these facts:\n1. ..."
)
