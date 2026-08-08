"""The catalog: one row per status, holding everything the statusline says about it.

A row declares WHEN it fires as well as what it says: ``fires`` is its own verdict over
a ``RenderFacts``, ``rank`` its seat in the one precedence ladder ``PRECEDENCE`` walks.
Adding a status is one row here, never an edit to a detector chain living elsewhere.

Two rules keep the module honest:

  * **No ANSI, ever.** Builders return structured records (``Instrument``, ``Reason``);
    the renderer owns every colour decision.
  * **No engine tuning.** Bands, gates and the tier taxonomy live in ``health_config``;
    the predicates here read those values, they never restate them.

Builders take one ``RenderFacts`` and are pure.
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from attention_span import health_config, render_facts, text, verdicts

DELEGATION_THRESHOLD = health_config.PRESENTATION_GATES.DELEGATION_THRESHOLD


CALM_STATUS = "Healthy"  # the foot of the ladder: nothing above it fired
WARMING_STATUS = "Warming"  # cold start: context tracking has no reading yet
NOTICE_STATUS = "Notice"  # the harness announced finished subagents
COMPACT_ACK_STATUS = "Compact ack"  # shown by its own renderer entry point
WARMING_ACTION = "WAIT FOR SESSION DATA"
STANDBY_REASON = "No session data yet"  # the reason shown until a tier is read
COMPACT_ACK_ACTION = "COMPACT COMPLETE"
COMPACT_ACK_DETAIL = "CONTEXT UPDATES NEXT TURN"  # sentence-cased in narrow panes
UNKNOWN_STATUS_ACTION = "CHECK STATUS"  # last resort for an unmapped status
NOTICE_ACTION_TEMPLATE = "{} SUBAGENT{} FINISHED"
FILE_DETAIL_COLS = 24
RANK_NEVER = 999  # out of the ladder entirely: PRECEDENCE stops below this


@dataclass(frozen=True)
class StatusSpec:
    """One status, wholly declared: when it fires, how it looks, what it shows.

    ``fires`` is the row's own verdict over a ``RenderFacts``; ``rank`` is its seat in
    the single precedence ladder, lower wins. ``detail`` returns Row 1's instruments in
    render order; ``reason`` returns Row 2's line, or ``None`` for a blank one.

    ``pseudo`` marks a row the renderer shows on its own authority rather than because a
    detector concluded something; that is what keeps it out of ``active_warnings``.

    ``sheds_context`` marks a row whose subject is NOT the context load, so a pane too
    tight to hold everything may drop its context instruments and keep the rest.
    """

    color: str
    glyph: str
    action: str
    narrow: tuple[str, ...]
    detail: Callable[[render_facts.RenderFacts], tuple[render_facts.Instrument, ...]]
    reason: Callable[[render_facts.RenderFacts], render_facts.Reason | None]
    rank: int
    fires: Callable[[render_facts.RenderFacts], bool]
    pseudo: bool = False
    sheds_context: bool = False


def _edit_retry_details(blind_loop: Mapping[str, Any] | None) -> tuple[str, str]:
    """The file and attempt count an engine edit-loop alert already carries.

    Read as fields, never parsed back out of the alert's prose. Sanitized again because
    the alert is an engine dict this seam does not own, and no unscrubbed text may reach
    a terminal.
    """
    alert = blind_loop or {}
    base = text.sanitize(alert.get("base") or "")
    count = alert.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        return base, ""
    return base, str(count)


def _tokens_warming(facts: render_facts.RenderFacts) -> bool:
    """Resolve the warming hint: absent means 'incomplete', complete always wins."""
    warming = facts.agents_tokens_warming
    if warming is None:
        warming = not facts.agents_tokens_complete
    if facts.agents_tokens_complete:
        warming = False
    return bool(warming)


def _cache_drop(facts: render_facts.RenderFacts) -> int | None:
    """The hit-rate drop in whole percentage points, or None when there isn't one."""
    drop = (facts.cache_health or {}).get("hit_drop")
    if isinstance(drop, (int, float)) and math.isfinite(float(drop)):
        return int(round(drop))
    return None


def _context_trusted(facts: render_facts.RenderFacts) -> bool:
    """Whether the context reading may be believed at all.

    THE one definition: a degraded parse invalidates the reading unless the harness
    supplied a live sample. Predicates and instruments both read it here, so no tier can
    fire on a number the same row then refuses to print.
    """
    return (not facts.parse_degraded) or facts.context.is_live


def _context_display(
    facts: render_facts.RenderFacts,
) -> tuple[tuple[render_facts.Instrument, ...], str]:
    """Row 1's context instruments and the joined value Row 2 falls back to.

    Derived together so the two rows cannot contradict each other: an empty value means
    there are no instruments either, and Row 2's context reason disappears with them.
    """
    if not (facts.show_context and _context_trusted(facts)):
        return (), ""
    context = facts.context
    items = []
    value = ""
    if context.tokens:
        value = text.display_tokens(context.tokens)
        items.append(render_facts.Instrument("ctx_load", value))
    if context.pct is not None:
        pct_value = f"{int(context.pct)}%"
        items.append(render_facts.Instrument("window", pct_value))
        value = f"{value} · {pct_value}" if value else pct_value
    return tuple(items), value


def _context_instruments(
    facts: render_facts.RenderFacts,
) -> tuple[render_facts.Instrument, ...]:
    return _context_display(facts)[0]


def _foreign_model_chip(facts: render_facts.RenderFacts) -> str:
    """What the LIVE children run on, when that is news beside the session model.

    One foreign label is named; two or more distinct labels become a count, because
    naming one would be an arbitrary pick the row cannot stand behind. A child with no
    assistant turn yet has no model evidence and claims nothing. Distinctness is judged
    on the public label, so a context-window variant counts as a different model.
    """
    labels = {
        label
        for label in (verdicts.model_label(m) for m in facts.agents_models)
        if label
    }
    if not labels:
        return ""
    if len(labels) > 1:
        return f"{len(labels)} models"
    [label] = labels
    return "" if label == verdicts.model_label(facts.identity.model_id) else label


def _working_instruments(
    facts: render_facts.RenderFacts,
) -> tuple[render_facts.Instrument, ...]:
    """The live-child counter, suppressed wholesale when agents are hidden."""
    if not (facts.show_agents and facts.agents_live):
        return ()
    working: tuple[render_facts.Instrument, ...] = (
        render_facts.Instrument("working", f"{facts.agents_live}"),
    )
    chip = _foreign_model_chip(facts)
    if chip:
        working += (render_facts.Instrument("models", "⇄ " + chip),)
    return working


def _ambient(facts: render_facts.RenderFacts) -> tuple[render_facts.Instrument, ...]:
    """The ambient tail rows share: context, then working."""
    return _context_instruments(facts) + _working_instruments(facts)


def _token_instrument(
    facts: render_facts.RenderFacts,
) -> tuple[render_facts.Instrument, ...]:
    """The child-token reading, or the reason there isn't one."""
    if facts.agents_tokens and facts.agents_tokens > 0:
        prefix = "" if facts.agents_tokens_complete else "≥"
        return (
            render_facts.Instrument(
                "tokens", prefix + text.display_tokens(facts.agents_tokens)
            ),
        )
    if not facts.agents_tokens_complete:
        return (
            render_facts.Instrument(
                "tokens_status", "warming" if _tokens_warming(facts) else "incomplete"
            ),
        )
    return ()


def _file_instruments(
    facts: render_facts.RenderFacts,
) -> tuple[render_facts.Instrument, ...]:
    """The looping file, column-capped with the ``×N`` suffix preserved.

    The count is the actionable half - a name alone does not say the edit is repeating -
    so the path yields columns to it, never the other way round.
    """
    file_name, count = _edit_retry_details(facts.blind_loop)
    count_text = " ×" + count if count else ""
    detail = file_name + count_text
    if text.vis_len(detail) > FILE_DETAIL_COLS:
        detail = (
            text.right_ellipsis(file_name, FILE_DETAIL_COLS - text.vis_len(count_text))
            + count_text
        )
    return (render_facts.Instrument("file", detail),) if detail else ()


def _calm_detail(
    facts: render_facts.RenderFacts,
) -> tuple[render_facts.Instrument, ...]:
    """Nothing to report but the ambient readings."""
    return _ambient(facts)


def _edit_loop_detail(
    facts: render_facts.RenderFacts,
) -> tuple[render_facts.Instrument, ...]:
    return _file_instruments(facts) + _ambient(facts)


def _truncated_detail(
    facts: render_facts.RenderFacts,
) -> tuple[render_facts.Instrument, ...]:
    return (render_facts.Instrument("plain", "last response cut off"),) + _ambient(
        facts
    )


def _refused_detail(
    facts: render_facts.RenderFacts,
) -> tuple[render_facts.Instrument, ...]:
    return (render_facts.Instrument("plain", "last response refused"),) + _ambient(
        facts
    )


def _cache_detail(
    facts: render_facts.RenderFacts,
) -> tuple[render_facts.Instrument, ...]:
    drop = _cache_drop(facts)
    if drop is None:
        return _ambient(facts)
    return (render_facts.Instrument("plain", f"reuse fell {drop}%"),) + _ambient(facts)


def _many_agents_detail(
    facts: render_facts.RenderFacts,
) -> tuple[render_facts.Instrument, ...]:
    """The one detail whose subject is the child burn: context sits last, not first."""
    return (
        (render_facts.Instrument("runs", f"{facts.agents_total}"),)
        + _token_instrument(facts)
        + _working_instruments(facts)
        + _context_instruments(facts)
    )


def _notice_detail(
    facts: render_facts.RenderFacts,
) -> tuple[render_facts.Instrument, ...]:
    """A notice states a completion; only the live count beside it is an instrument."""
    live = facts.notice.live_n if facts.notice else 0
    return (render_facts.Instrument("working", f"{live}"),) if live else ()


def _no_detail(facts: render_facts.RenderFacts) -> tuple[render_facts.Instrument, ...]:
    return ()


def _parse_reason(facts: render_facts.RenderFacts) -> render_facts.Reason | None:
    return render_facts.Reason("Session history could not be checked")


def _edit_loop_reason(facts: render_facts.RenderFacts) -> render_facts.Reason | None:
    # Uncapped on purpose: Row 2 shows the whole path Row 1 had to abbreviate.
    file_name, _ = _edit_retry_details(facts.blind_loop)
    return render_facts.Reason("File: " + file_name) if file_name else None


def _agent_edit_loop_reason(
    facts: render_facts.RenderFacts,
) -> render_facts.Reason | None:
    return render_facts.Reason("A child agent retried a failed edit")


def _many_agents_reason(facts: render_facts.RenderFacts) -> render_facts.Reason | None:
    detail = f"{text.compact_magnitude(facts.agents_total)} subagents"
    if facts.agents_tokens and facts.agents_tokens > 0:
        qualifier = "" if facts.agents_tokens_complete else "at least "
        detail += f"; {qualifier}{text.display_tokens(facts.agents_tokens)} tokens"
    elif not facts.agents_tokens_complete:
        detail += (
            "; token total still loading"
            if _tokens_warming(facts)
            else "; token total incomplete"
        )
    return render_facts.Reason(detail)


def _truncated_reason(facts: render_facts.RenderFacts) -> render_facts.Reason | None:
    return render_facts.Reason("The last response was cut off")


def _refused_reason(facts: render_facts.RenderFacts) -> render_facts.Reason | None:
    return render_facts.Reason("The last response was refused")


def _cache_reason(facts: render_facts.RenderFacts) -> render_facts.Reason | None:
    drop = _cache_drop(facts)
    if drop is None:
        return render_facts.Reason("Claude is re-reading more context")
    return render_facts.Reason(f"Context reuse fell {drop}%")


def _notice_reason(facts: render_facts.RenderFacts) -> render_facts.Reason | None:
    """The live child count outranks the notice's own: it is the fresher fact."""
    if facts.show_agents and facts.agents_live:
        return render_facts.Reason(f"{facts.agents_live} still working")
    live = facts.notice.live_n if facts.notice else 0
    return render_facts.Reason(f"{live} still working") if live else None


def _calm_reason(facts: render_facts.RenderFacts) -> render_facts.Reason | None:
    """Standby, then the context load, then the window fill - the calm cascade.

    Row 2 for any row whose subject is just the context state. No tier means no reading
    has arrived, so the row says so rather than print a load of nothing.
    """
    context = facts.context
    if facts.show_context and context.tier is None:
        return render_facts.Reason(STANDBY_REASON)
    value = _context_display(facts)[1]
    if not value:
        return None
    if context.tokens:
        label = "Context load: "
        shown = text.display_tokens(context.tokens)
        return render_facts.Reason(
            text=label + shown, label=label, value=shown, use_context_color=True
        )
    if context.pct is not None:
        return render_facts.Reason(f"Window used: {int(context.pct)}%")
    return None


def _compact_ack_reason(facts: render_facts.RenderFacts) -> render_facts.Reason | None:
    return render_facts.Reason(COMPACT_ACK_DETAIL)


def _no_reason(facts: render_facts.RenderFacts) -> render_facts.Reason | None:
    return None


def _fires_never(facts: render_facts.RenderFacts) -> bool:
    """A row the ladder never selects; the renderer reaches it directly."""
    return False


def _fires_health_limited(facts: render_facts.RenderFacts) -> bool:
    """Parse health is degraded, but a live sample keeps the context reading usable."""
    return bool(facts.parse_degraded and facts.context.is_live)


def _fires_health_stale(facts: render_facts.RenderFacts) -> bool:
    """Parse health is degraded and nothing live backs the reading up."""
    return bool(facts.parse_degraded and not facts.context.is_live)


def _fires_edit_loop(facts: render_facts.RenderFacts) -> bool:
    """This session retried an edit blind, on a transcript that parsed cleanly."""
    return bool(not facts.parse_degraded and facts.blind_loop)


def _fires_agent_edit_loop(facts: render_facts.RenderFacts) -> bool:
    """A subagent is in the red: its own blind-loop detector fired."""
    return bool(facts.subagent_blind_loop)


def _fires_many_agents(facts: render_facts.RenderFacts) -> bool:
    """Cumulative delegation crossed the presentation gate."""
    return facts.agents_total >= DELEGATION_THRESHOLD


def _fires_truncated(facts: render_facts.RenderFacts) -> bool:
    return bool(not facts.parse_degraded and facts.last_stop_reason == "max_tokens")


def _fires_refused(facts: render_facts.RenderFacts) -> bool:
    return bool(not facts.parse_degraded and facts.last_stop_reason == "refusal")


def _fires_cache(facts: render_facts.RenderFacts) -> bool:
    if facts.parse_degraded:
        return False
    return bool((facts.cache_health or {}).get("show"))


def _fires_notice(facts: render_facts.RenderFacts) -> bool:
    """The harness announced finished subagents; the row IS that announcement."""
    return bool(facts.notice)


def _fires_warming(facts: render_facts.RenderFacts) -> bool:
    """Context tracking is on but has produced no tier yet - a cold start."""
    return bool(facts.show_context and facts.context.tier is None)


def _fires_calm(facts: render_facts.RenderFacts) -> bool:
    """The foot of the ladder: reached only because nothing above it fired."""
    return True


def _tier_fires(tier: str) -> Callable[[render_facts.RenderFacts], bool]:
    """One tier's predicate: graded to THIS tier, on a reading the row would show."""

    def fires(facts: render_facts.RenderFacts) -> bool:
        return bool(
            facts.show_context
            and _context_trusted(facts)
            and facts.context.tier == tier
        )

    return fires


_TIER_ROWS: Mapping[str, tuple[str, str, str]] = MappingProxyType(
    {
        # tier: (colour, moon glyph, public STATUS word)
        "dead": ("magenta_red", "💀", "DEAD"),
        "failing": ("red", "🌑", "FAILING"),
        "degrading": ("orange", "🌘", "DEGRADING"),
        "functional": ("yellow", "🌗", "FUNCTIONAL"),
        "strong": ("green", "🌖", "STILL SHARP"),
        "peak": ("mint", "🌕", "PEAK"),
    }
)


# The calm half of the ladder: a reading, not an alert.
_READING_TIERS: tuple[str, ...] = tuple(
    tier
    for tier in health_config.CONTEXT_TIERS
    if tier not in health_config.CONTEXT_FIRING_TIERS
)

# Every tier's seat, worst first within its half. Alerts sit mid-ladder; readings sit
# below every warning, so a quiet context never masks one, and above the calm floor, so
# a reading still gets to speak.
_TIER_RANKS: Mapping[str, int] = MappingProxyType(
    {tier: 50 + seat for seat, tier in enumerate(health_config.CONTEXT_FIRING_TIERS)}
    | {tier: 91 + seat for seat, tier in enumerate(_READING_TIERS)}
)


def _tier_spec(tier: str) -> StatusSpec:
    color, glyph, word = _TIER_ROWS[tier]
    return StatusSpec(
        color=color,
        glyph=glyph,
        action=word,
        narrow=(UNKNOWN_STATUS_ACTION,),
        detail=_calm_detail,
        reason=_calm_reason,
        rank=_TIER_RANKS[tier],
        fires=_tier_fires(tier),
        pseudo=tier in _READING_TIERS,
    )


_ROWS = {
    "Health data limited": StatusSpec(
        color="cyan",
        glyph="◌",
        action="CAN'T CHECK SESSION",
        narrow=("CHECK DATA", "DATA"),
        detail=_calm_detail,
        reason=_parse_reason,
        rank=10,
        fires=_fires_health_limited,
    ),
    "Health data stale": StatusSpec(
        color="cyan",
        glyph="◌",
        action="CAN'T CHECK SESSION",
        narrow=("CHECK DATA", "DATA"),
        detail=_calm_detail,
        reason=_parse_reason,
        rank=11,
        fires=_fires_health_stale,
    ),
    "Edit loop": StatusSpec(
        color="red",
        glyph="■",
        action="READ FILE, THEN RETRY",
        narrow=("READ FILE", "RETRY"),
        detail=_edit_loop_detail,
        reason=_edit_loop_reason,
        rank=20,
        fires=_fires_edit_loop,
    ),
    "Agent edit loop": StatusSpec(
        color="red",
        glyph="■",
        action="CHECK CHILD AGENT",
        narrow=("CHECK AGENT", "CHECK"),
        detail=_calm_detail,
        reason=_agent_edit_loop_reason,
        rank=30,
        fires=_fires_agent_edit_loop,
    ),
    "Many agent runs": StatusSpec(
        color="yellow",
        glyph="▲",
        action="REVIEW CHILD TOKEN BURN",
        narrow=("REVIEW BURN", "BURN"),
        detail=_many_agents_detail,
        reason=_many_agents_reason,
        rank=40,
        fires=_fires_many_agents,
        sheds_context=True,
    ),
    "Response truncated": StatusSpec(
        color="yellow",
        glyph="▲",
        action="CHECK LAST RESPONSE",
        narrow=("CHECK RESPONSE", "RESPONSE"),
        detail=_truncated_detail,
        reason=_truncated_reason,
        rank=60,
        fires=_fires_truncated,
    ),
    "Response refused": StatusSpec(
        color="yellow",
        glyph="▲",
        action="CHECK LAST RESPONSE",
        narrow=("CHECK RESPONSE", "RESPONSE"),
        detail=_refused_detail,
        reason=_refused_reason,
        rank=61,
        fires=_fires_refused,
    ),
    "Cache problem": StatusSpec(
        color="yellow",
        glyph="▲",
        action="CHECK CACHE - COST RISING",
        narrow=("CHECK CACHE", "CACHE"),
        detail=_cache_detail,
        reason=_cache_reason,
        rank=70,
        fires=_fires_cache,
    ),
    NOTICE_STATUS: StatusSpec(
        color="green",
        glyph="✓",
        action=NOTICE_ACTION_TEMPLATE,
        narrow=("FINISHED",),
        detail=_notice_detail,
        reason=_notice_reason,
        rank=80,
        fires=_fires_notice,
        pseudo=True,
    ),
    WARMING_STATUS: StatusSpec(
        color="cyan",
        glyph="◌",
        action=WARMING_ACTION,
        narrow=("WAIT",),
        detail=_calm_detail,
        reason=_calm_reason,
        rank=90,
        fires=_fires_warming,
        pseudo=True,
    ),
    CALM_STATUS: StatusSpec(
        color="green",
        glyph="🌕",
        action="PEAK",
        narrow=("GO",),
        detail=_calm_detail,
        reason=_calm_reason,
        rank=99,
        fires=_fires_calm,
        pseudo=True,
    ),
    COMPACT_ACK_STATUS: StatusSpec(
        color="cyan",
        glyph="✓",
        action=COMPACT_ACK_ACTION,
        narrow=("COMPACT",),
        detail=_no_detail,
        reason=_compact_ack_reason,
        rank=RANK_NEVER,
        fires=_fires_never,
        pseudo=True,
    ),
}
_ROWS.update({tier: _tier_spec(tier) for tier in _TIER_ROWS})

STATUSES: Mapping[str, StatusSpec] = MappingProxyType(_ROWS)

PRECEDENCE: tuple[tuple[str, StatusSpec], ...] = tuple(
    (name, spec)
    for name, spec in sorted(STATUSES.items(), key=lambda row: row[1].rank)
    if spec.rank < RANK_NEVER
)

FALLBACK = StatusSpec(
    color="white",
    glyph="◌",
    action="",  # unknown status: action_for() derives the text from the name
    narrow=(UNKNOWN_STATUS_ACTION,),
    detail=_no_detail,
    reason=_no_reason,
    rank=RANK_NEVER,
    fires=_fires_never,
    pseudo=True,
)


def active_warnings(facts: render_facts.RenderFacts) -> tuple[str, ...]:
    """Every firing row a detector actually concluded, in precedence order.

    Pseudo rows sit out: they are shown on the renderer's own authority, so counting
    them here would turn a quiet reading into a warning.
    """
    return tuple(
        name for name, spec in PRECEDENCE if not spec.pseudo and spec.fires(facts)
    )


def dominant_status(facts: render_facts.RenderFacts) -> str:
    """The single status the row reports: the first rung of the ladder that fires."""
    for name, spec in PRECEDENCE:
        if spec.fires(facts):
            return name
    return CALM_STATUS  # unreachable while the calm row is the ladder's floor


def resolve(status: str) -> StatusSpec:
    """The row for ``status``, or the fallback row when nothing declares it."""
    spec = STATUSES.get(status) if isinstance(status, str) else None
    return spec if spec is not None else FALLBACK


def notice_action(notice: render_facts.Notice | None) -> str:
    """The action text for a finished-subagent notice, pluralized on the count."""
    done = notice.done_n if notice else 0
    return NOTICE_ACTION_TEMPLATE.format(done, "" if done == 1 else "S")


def action_for(status: str, notice: render_facts.Notice | None = None) -> str:
    """The direct user action for a status - the whole of what Row 1's headline says.

    Only the notice needs the count folded in; every other row states its own action.
    """
    if status == NOTICE_STATUS:
        return notice_action(notice)
    spec = STATUSES.get(status) if isinstance(status, str) else None
    if spec is not None:
        return spec.action
    return text.sanitize(status).upper()
