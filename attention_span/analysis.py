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

    Burn is a CONSUMER computation: ``input + cache_creation + output``. ``cache_read``
    is excluded there - it is a re-read of the cached prefix, not new tokens.
    """

    input: int = 0
    cache_creation: int = 0
    cache_read: int = 0
    output: int = 0
    api_calls: int = 0


@dataclass(frozen=True)
class Repetition:
    """Detector #3, read-repetition. Telemetry only - never displayed.

    ``score`` (== the worst no-progress re-read run length) is the monotonic
    calibration column; a run >= REPEAT_MIN is "firing".
    """

    score: int = 0
    worst_target: str | None = None
    count: int = 0


@dataclass(frozen=True)
class Perseveration:
    """Detector #4, identical repeated tool calls. Telemetry only - never displayed.

    ``score`` (== the worst same-call/same-result run length) is the monotonic
    calibration column; a run >= PERSEV_MIN is "firing". ``worst_target`` names the
    TOOL that ran the worst run - the call signature itself is a digest and says
    nothing a reader could act on.
    """

    score: int = 0
    worst_target: str | None = None
    count: int = 0


@dataclass(frozen=True)
class ParseHealth:
    """Detector #2, self-doubt: can this Analysis be trusted at all?

    ``degraded`` is the one field consumers read - it degrades the row to honest
    instead of showing a confident green over a half-analyzed transcript. The rest is
    the evidence behind it. This is the ONE part of the Analysis carved out of the
    ``iter_states`` drift invariant: ``decode_failures`` / ``parse_aborted`` are
    counted in ``analyze_transcript``'s loop, which ``iter_states`` never runs.
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
    nothing-known value (insufficient window, no signal) rather than a partial one.

    Read by the statusline (``context_tokens``, ``compaction_pending``), the renderer
    (``parse_health``, ``cache_health``, ``last_stop_reason``, ``turns``), and the
    cohort rollup (``usage_totals``, ``insufficient``, ``last_stop_reason``). The
    remaining fields are the engine's own reasoning, kept because ``iter_states``
    calibration and the debug CLI read them.

    ``cache_health`` is the one detector result still a Mapping: it crosses into
    ``render_facts.RenderFacts`` unchanged, and that seam's off-schema tolerance is
    the status catalog's to own. Typing it belongs with that seam, not here.
    """

    # The rolling read-to-edit window (R2E ratio)
    reads: int = 0
    edits: int = 0
    total_reads: int = 0
    total_edits: int = 0
    r2e: float = float("inf")
    base_tier: str = "green"
    window_used: int = 0
    insufficient: bool = True
    failed_edit_loop: EditLoop = EditLoop()
    cache_health: Mapping[str, Any] = field(default_factory=dict)
    repetition: Repetition = Repetition()
    perseveration: Perseveration = Perseveration()
    parse_health: ParseHealth = ParseHealth()
    last_model: str | None = None
    last_stop_reason: str | None = None
    compaction_pending: bool = False
    turns: int = 0
    context_tokens: int = 0
    usage_totals: UsageTotals = UsageTotals()
    max_error_bytes: int = 0
    task_notifications: Mapping[str, float] = field(default_factory=dict)
    thinking_turns: int = 0
    assistant_turns: int = 0
    trailing_no_thinking: int = 0
