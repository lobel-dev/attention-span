"""Immutable value records produced by transcript analysis."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EditLoop:
    """The worst same-file blind-edit run: a file whose failed edit was re-edited
    with no intervening Read, and how many times. ``count`` 0 means no loop."""

    file: str | None = None
    count: int = 0


@dataclass(frozen=True)
class UsageTotals:
    """Cumulative ``message.usage`` sums, deduped LAST-WINS per API call.

    Raw sums only. ``cache_read`` is kept apart from the rest because a re-read of the
    cached prefix is not new spend, which is the distinction consumers grade on.
    """

    input: int = 0
    cache_creation: int = 0
    cache_read: int = 0
    output: int = 0
    api_calls: int = 0


@dataclass(frozen=True)
class Repetition:
    """Read-repetition. Diagnostic only: no part of the panel reads it.

    ``score`` is the worst no-progress re-read run length - monotonic, so a threshold
    can be set against it later.
    """

    score: int = 0
    worst_target: str | None = None
    count: int = 0


@dataclass(frozen=True)
class Perseveration:
    """Identical repeated tool calls. Diagnostic only: no part of the panel reads it.

    ``score`` is the worst same-call/same-result run length - monotonic, so a threshold
    can be set against it later. ``worst_target`` names the TOOL, not the call
    signature, which is a digest a reader could not act on.
    """

    score: int = 0
    worst_target: str | None = None
    count: int = 0


@dataclass(frozen=True)
class ParseHealth:
    """Self-doubt: can this ``Analysis`` be trusted at all?

    ``degraded`` is the one field consumers read - it keeps a half-analyzed transcript
    from being presented as a confident verdict; the rest is the evidence behind it.
    The ONE part of an ``Analysis`` carved out of the ``iter_states`` drift invariant,
    because the fields it counts are only reachable from ``analyze_transcript``'s loop.
    """

    decode_failures: int = 0
    assistant_lines: int = 0
    usage_seen: bool = False
    tooluse_seen: bool = False
    parse_aborted: bool = False
    schema_canary: bool = False
    degraded: bool = False


@dataclass(frozen=True)
class Analysis:
    """The structured facts the engine derives from one pass over a transcript.

    The defaults describe an EMPTY transcript, so ``Analysis()`` is the honest
    nothing-known value rather than a partial one.

    ``cache_health`` is the one detector result still a Mapping: it crosses into
    ``render_facts`` unchanged, and tolerating an off-schema value there belongs to
    that seam rather than here.
    """

    # The rolling read-to-edit window (R2E ratio)
    reads: int = 0
    edits: int = 0
    total_reads: int = 0  # lifetime, executed calls only
    total_edits: int = 0
    r2e: float = float("inf")  # reads/edits in the window; inf when no edits
    base_tier: str = "green"  # green | yellow | red, from R2E alone
    window_used: int = 0  # read+edit events the window actually holds
    insufficient: bool = True  # too few events in the window to grade
    failed_edit_loop: EditLoop = EditLoop()
    cache_health: Mapping[str, Any] = field(default_factory=dict)
    repetition: Repetition = Repetition()
    perseveration: Perseveration = Perseveration()
    parse_health: ParseHealth = ParseHealth()
    last_model: str | None = None  # model of the most recent real assistant turn
    last_stop_reason: str | None = None
    compaction_pending: bool = False  # compact summary with no later assistant turn
    turns: int = 0  # real (non-synthetic) user turns
    context_tokens: int = 0  # context the last real assistant turn occupied
    usage_totals: UsageTotals = UsageTotals()
    max_error_bytes: int = 0  # largest errored tool_result body; small ones ignored
    # child task id -> epoch of the latest child stop this transcript witnessed
    task_notifications: Mapping[str, float] = field(default_factory=dict)
    thinking_turns: int = 0
    assistant_turns: int = 0  # substantive turns: the thinking denominator
    trailing_no_thinking: int = 0  # latest substantive turns with no thinking
