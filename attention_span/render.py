"""Renderer for the attention-span statusline"""

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from attention_span import agent_health, health_config, status_catalog, subagents, text
from attention_span.render_facts import RenderFacts

EMPTY_COHORT = subagents.Cohort()

vis_len = cast(Callable[[str], int], text.vis_len)
_right_ellipsis = cast(Callable[[str, int], str], text.right_ellipsis)
_left_ellipsis = cast(Callable[[str, int], str], text.left_ellipsis)
_sanitize = cast(Callable[[object], str], text.sanitize)
_dominant_status = cast(Callable[[Any], str], status_catalog.dominant_status)
_resolve_status = cast(Callable[[str], Any], status_catalog.resolve)
_action_for = cast(Callable[..., str], status_catalog.action_for)

_GATES = health_config.PRESENTATION_GATES

NARROW_ACTION_COLS = _GATES.NARROW_ACTION_COLS
MICRO_ACTION_COLS = _GATES.MICRO_ACTION_COLS
STATUS_BAY_WIDTH = _GATES.STATUS_BAY_WIDTH
STATUS_BAY_MIN_FILL = _GATES.STATUS_BAY_MIN_FILL

_RAW = {
    "mint": "38;2;0;210;130",
    "green": "38;2;0;175;80",
    "yellow": "38;2;230;200;0",
    "orange": "38;2;255;176;85",
    "red": "38;2;255;85;85",
    "moss": "38;2;125;160;135",
    "rose": "38;2;185;130;130",
    "magenta_red": "38;2;255;60;140",
    "blue": "38;2;0;153;255",
    "cyan": "38;2;86;182;194",
    "white": "38;2;220;220;220",
    "magenta": "38;2;180;140;255",
}

_NO_COLOR = False


@dataclass(frozen=True)
class LayoutOpts:
    """How the pane draws - not what the session is.

    The two facts about the TERMINAL, kept apart from the session facts they constrain:
    how many columns there are to spend, and whether colour may be spent at all.
    """

    columns: int | None = None
    no_color: bool = False


DEFAULT_LAYOUT = LayoutOpts()


def paint(s: str, c: str) -> str:
    if _NO_COLOR:
        return s
    return "\033[{}m{}\033[0m".format(_RAW.get(c, _RAW["white"]), s)


def dim(s: str) -> str:
    if _NO_COLOR:
        return s
    return f"\033[2m{s}\033[0m"


def _status_bay(headline_text: str, color: str, width: int = STATUS_BAY_WIDTH) -> str:
    if vis_len(headline_text) > width:
        headline_text = _truncate_headline(headline_text, width)
    fill = max(0, width - vis_len(headline_text) - 1)
    headline = paint(headline_text, color)
    return headline + (paint(" " + "─" * fill, color) if fill else "")


def _min_bay_width(headline_text: str) -> int:
    """The narrowest bay that still shows a deliberate dash run."""
    return vis_len(headline_text) + 1 + STATUS_BAY_MIN_FILL


def _bay_width(
    headline_text: str,
    other_width: int,
    target_width: int | None,
    columns: int | None = None,
) -> int:
    """Bay width that lands Row 1's right edge on Row 2's."""
    if not target_width:
        return STATUS_BAY_WIDTH
    minimum = _min_bay_width(headline_text)
    width = target_width - other_width
    if columns and columns > 0:
        width = min(width, columns - other_width)
    return max(minimum, width)


def _tier_spec(tier: str | None) -> Any | None:
    """The catalog row for a context tier, or None when the tier is not one."""
    if tier not in health_config.CONTEXT_TIERS:
        return None
    return status_catalog.STATUSES[tier]


def context_status(tier: str | None) -> str:
    """Return the public wording for a context tier: one bare status word."""
    spec = _tier_spec(tier)
    return spec.action if spec else ""


def context_tier_color(tier: str | None) -> str:
    """Palette key for a context tier; white when the tier is unknown."""
    spec = _tier_spec(tier)
    return spec.color if spec else "white"


def _fit_plain_action(action_text: str, spec: Any, width: int) -> str:
    """Keep a useful worded action in physically tiny panes."""
    for candidate in (action_text,) + spec.narrow:
        if vis_len(candidate) <= width:
            return cast(str, candidate)
    return _right_ellipsis(action_text, max(0, width))


def render_standby_row(
    *,
    columns: int | None = None,
    no_color: bool = False,
    target_width: int | None = None,
) -> str:
    """Render the framed cold-start row without requiring transcript analysis.

    ``target_width`` is the visible width of the identity row the caller will join
    below this one; absent, the bay keeps its fixed fallback width.
    """
    global _NO_COLOR
    _NO_COLOR = no_color
    spec = status_catalog.STATUSES[status_catalog.WARMING_STATUS]
    action_text = spec.action
    if columns and columns <= MICRO_ACTION_COLS:
        return paint(_fit_plain_action(action_text, spec, columns), spec.color)
    if columns and columns <= NARROW_ACTION_COLS:
        return (
            paint("╭ ", spec.color)
            + paint(action_text, spec.color)
            + "\n"
            + paint("╰ ", spec.color)
            + dim(status_catalog.STANDBY_REASON)
        )
    headline = spec.glyph + " " + action_text
    width = _bay_width(headline, 3, target_width, columns)
    return paint("╭─ ", spec.color) + _status_bay(headline, spec.color, width)


def render_compact_row(
    *,
    columns: int | None = None,
    no_color: bool = False,
    target_width: int | None = None,
) -> str:
    """Acknowledge completed compaction while CC waits for the next live usage sample.

    ``target_width`` aligns the row with the identity row joined below it, exactly as
    in ``render_standby_row``; the trailing detail is charged to the same budget.
    """
    global _NO_COLOR
    _NO_COLOR = no_color
    spec = status_catalog.STATUSES[status_catalog.COMPACT_ACK_STATUS]
    action_text = spec.action
    detail = status_catalog.COMPACT_ACK_DETAIL
    if columns and columns <= MICRO_ACTION_COLS:
        return paint(_fit_plain_action(action_text, spec, columns), spec.color)
    if columns and columns <= NARROW_ACTION_COLS:
        return (
            paint("╭ ", spec.color)
            + paint(action_text, spec.color)
            + "\n"
            + paint("╰ ", spec.color)
            + dim(detail[:1] + detail[1:].lower())
        )
    headline = spec.glyph + " " + action_text
    width = _bay_width(headline, 3 + 2 + vis_len(detail), target_width, columns)
    return (
        paint("╭─ ", spec.color)
        + _status_bay(headline, spec.color, width)
        + "  "
        + dim(detail)
    )


def _fail_soft_count(value: Any) -> int:
    """Coerce to a non-negative int; 0 for absent or off-schema values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    if not math.isfinite(float(value)) or value < 0:
        return 0
    return int(value)


def _lines_changed(added: Any, removed: Any) -> str:
    """'+123/-45' for the identity row, or '' when nothing changed.

    The counters are the harness's SESSION-CUMULATIVE totals (`cost.total_lines_*`):
    every line this session ever edited, monotonic, unaffected by commits. Not a git
    diff - a commit does not reset it, only a new session does.
    """
    added, removed = _fail_soft_count(added), _fail_soft_count(removed)
    if not added and not removed:
        return ""
    return paint(f"+{added}", "moss") + dim("/") + paint(f"-{removed}", "rose")


def _turns_segment(turns: Any) -> str:
    """'↺ 12' for the identity row; '' for absent, zero, or off-schema counts."""
    count = _fail_soft_count(turns)
    return dim(f"↺ {count}") if count else ""


SEGMENT_JOIN = "   "

_LOCATION_SEPARATOR_COLS = 5


def _location_budget(
    columns: int,
    identity_plain: str,
    has_location: bool,
    changed: str,
    turns_seg: str,
) -> int:
    """Columns left for the repo/cwd pair once every other segment is paid for."""
    identity_gap = len(SEGMENT_JOIN) if identity_plain and has_location else 0
    changed_width = (vis_len(changed) + len(SEGMENT_JOIN)) if changed else 0
    turns_width = (vis_len(turns_seg) + len(SEGMENT_JOIN)) if turns_seg else 0
    return max(
        0,
        columns
        - 3
        - vis_len(identity_plain)
        - identity_gap
        - changed_width
        - turns_width,
    )


def _fit_location(repo_label: str, cwd_label: str, budget: int) -> tuple[str, str]:
    """Trim the repo and cwd labels into ``budget`` columns, cwd keeping its tail.

    The repo name yields first (it is capped at a third of the budget) because the
    working directory's END - where the session actually is - is the useful half.
    """
    separator = _LOCATION_SEPARATOR_COLS if repo_label and cwd_label else 0
    if vis_len(repo_label) + vis_len(cwd_label) + separator <= budget:
        return repo_label, cwd_label
    repo_budget = min(vis_len(repo_label), max(1, budget // 3)) if repo_label else 0
    cwd_budget = max(0, budget - repo_budget - separator)
    return (
        _right_ellipsis(repo_label, repo_budget),
        _left_ellipsis(cwd_label, cwd_budget),
    )


def render_metadata_row(
    *,
    model_id: Any = "",
    effort: Any = "",
    repo_name: Any = "",
    cwd: Any = "",
    lines_added: Any = None,
    lines_removed: Any = None,
    turns: Any = None,
    fast_mode: bool = False,
    rail_color: str = "green",
    columns: int | None = None,
    no_color: bool = False,
    target_width: int | None = None,
) -> str:
    """Compose the wide session-identity row from live statusline payload facts."""
    global _NO_COLOR
    _NO_COLOR = no_color
    repo_label = _sanitize(repo_name)
    cwd_label = _sanitize(cwd)
    if cwd_label:
        repo_label = ""
    model = _instrument_model_label(model_id)
    effort_label = _sanitize(effort)
    changed = _lines_changed(lines_added, lines_removed)
    turns_seg = _turns_segment(turns)

    if columns and columns > NARROW_ACTION_COLS:
        model = _right_ellipsis(model, max(8, min(32, columns // 3)))
        effort_label = _right_ellipsis(effort_label, max(3, min(10, columns // 8)))
    if fast_mode and model:
        model = "⚡ " + model

    if columns and columns > 0:
        identity_plain = health_config.IDENTITY_JOIN.join(
            v for v in (model, effort_label.upper() if effort_label else "") if v
        )
        repo_label, cwd_label = _fit_location(
            repo_label,
            cwd_label,
            _location_budget(
                columns,
                identity_plain,
                bool(repo_label or cwd_label),
                changed,
                turns_seg,
            ),
        )

    location: list[str] = []
    if repo_label:
        location.append(paint(repo_label, "cyan"))
    if cwd_label:
        if location:
            location.append(dim("›"))
        location.append(dim(cwd_label))
    identity: list[str] = []
    if model:
        identity.append(paint(model, "magenta"))
    if effort_label:
        identity.append(paint(effort_label.upper(), "magenta"))

    segments: list[str] = []
    if location:
        segments.append("  ".join(location))
    if changed:
        segments.append(changed)
    if turns_seg:
        segments.append(turns_seg)
    if identity:
        segments.append(dim(health_config.IDENTITY_JOIN).join(identity))
    if not segments:
        return ""
    if segments == [turns_seg]:
        # Turns alone never summons the row: no location or identity facts, no row.
        return ""

    gaps = [SEGMENT_JOIN] * (len(segments) - 1)
    if gaps and target_width:
        natural = 3 + sum(vis_len(s) for s in segments) + len(SEGMENT_JOIN) * len(gaps)
        pad = target_width - natural
        if columns and columns > 0:
            pad = min(pad, columns - natural)
        if pad > 0:
            gaps[-1] += " " * pad
    content = segments[0] + "".join(
        gap + segment for gap, segment in zip(gaps, segments[1:], strict=True)
    )
    return paint("╰─ ", rail_color) + content


def _instrument_model_label(model_id: Any) -> str:
    """Turn the compact engine label into a deliberate model identity."""
    label = agent_health.model_label(model_id)
    if not label:
        return ""
    model, sep, context = label.partition("·")
    match = re.match(r"^([A-Za-z]+)-(.+)$", model)
    if match:
        model = f"{match.group(1).upper()} {match.group(2)}"
    if sep:
        model += " / " + context
    return model


def _truncate_headline(value: str, width: int) -> str:
    return value if vis_len(value) <= width else _right_ellipsis(value, width)


_GAUGE_RAMP = (
    "38;2;0;175;80",
    "38;2;90;190;55",
    "38;2;160;200;35",
    "38;2;230;200;0",
    "38;2;245;180;30",
    "38;2;255;160;60",
    "38;2;255;176;85",
    "38;2;255;130;80",
    "38;2;255;100;82",
    "38;2;255;85;85",
)


def _paint_raw(s: str, sgr: str) -> str:
    """paint() with a raw SGR parameter string (the gauge ramp isn't in _RAW)."""
    if _NO_COLOR:
        return s
    return f"\033[{sgr}m{s}\033[0m"


def _ramp_sgr(i: int, width: int) -> str:
    """SGR for the i-th filled cell of a width-cell gauge along _GAUGE_RAMP."""
    if width <= 1:
        return _GAUGE_RAMP[-1]
    return _GAUGE_RAMP[i * (len(_GAUGE_RAMP) - 1) // (width - 1)]


def bar(
    pct: float, width: int = 10, muted: bool = False, tier_color: str | None = None
) -> str:
    """A gauge whose LENGTH is always the percentage. What colours the fill depends on
    whether that percentage is itself a severity axis.
    """
    pct = max(0.0, min(100.0, pct))
    filled = max(0, min(width, int(pct * width / 100.0)))
    if muted:
        return "".join(dim("█" if i < filled else "░") for i in range(width))
    if tier_color is not None:
        return "".join(
            paint("█", tier_color) if i < filled else dim("░") for i in range(width)
        )
    return "".join(
        _paint_raw("█", _ramp_sgr(i, width)) if i < filled else dim("░")
        for i in range(width)
    )


_PLAIN_INSTRUMENT: tuple[str, Callable[[str], str]] = ("", str.upper)
_INSTRUMENT_LABELS: dict[str, tuple[str, Callable[[str], str]]] = {
    "working": ("WORKING ", str),
    "models": ("", str),
    "runs": ("SUBAGENTS ", str),
    "tokens": ("TOKENS ", str),
    "tokens_status": ("TOKENS ", str.upper),
    "file": ("FILE ", str),
    "plain": _PLAIN_INSTRUMENT,
}


def render_panel(facts: RenderFacts, layout: LayoutOpts = DEFAULT_LAYOUT) -> str:
    """Compose the framed action-first statusline.

    ``facts`` is the whole session: one ``render_facts.RenderFacts``, built where the
    payload and the transcript are read. ``layout`` is the pane. The engine's Analysis
    stays behind the assembly point in statusline.py - every fact the panel or the
    status decision needs has already been read out of it into the snapshot.
    """
    global _NO_COLOR
    _NO_COLOR = layout.no_color
    columns = layout.columns

    context = facts.context
    context_color = context_tier_color(context.tier)

    status = _dominant_status(facts)
    spec = _resolve_status(status)
    status_color, status_glyph = spec.color, spec.glyph
    action_text = _action_for(status, notice=facts.notice)

    headline_text = f"{status_glyph} {action_text}"
    parts = spec.detail(facts)

    def action_rows() -> str:
        cols = cast(int, columns)
        if cols <= MICRO_ACTION_COLS:
            shown_action = _fit_plain_action(action_text, spec, cols)
            return paint(shown_action, status_color)
        shown_action = _fit_plain_action(action_text, spec, cols - 2)
        action_line = paint("╭ ", status_color) + paint(shown_action, status_color)
        reason = spec.reason(facts)
        if reason is not None and reason.text:
            fitted_reason = _right_ellipsis(reason.text, cols - 2)
            rendered_reason = (
                dim(reason.label) + paint(reason.value, context_color)
                if reason.use_context_color and fitted_reason == reason.text
                else dim(fitted_reason)
            )
            return action_line + "\n" + paint("╰ ", status_color) + rendered_reason
        return action_line

    if columns and columns <= NARROW_ACTION_COLS:
        return action_rows()

    def identity_row(target: int | None = None) -> str:
        identity = facts.identity
        return render_metadata_row(
            model_id=identity.model_id,
            effort=identity.effort,
            repo_name=identity.repo_name,
            cwd=identity.cwd,
            lines_added=identity.lines_added,
            lines_removed=identity.lines_removed,
            turns=identity.turns,
            fast_mode=identity.fast_mode,
            rail_color=status_color,
            columns=columns,
            no_color=layout.no_color,
            target_width=target,
        )

    metadata = identity_row()
    target_width = vis_len(metadata) if metadata else None

    def instrument_tail(items: list[Any]) -> str:
        instruments: list[str] = []
        for item in items:
            if item.kind == "ctx_load":
                instruments.append(
                    dim("CONTEXT LOAD ") + paint(item.text, context_color)
                )
            elif item.kind == "window":
                instruments.append(
                    dim("WINDOW ")
                    + bar(cast(float, context.pct), width=10, tier_color=context_color)
                    + " "
                    + dim(item.text)
                )
            else:
                label, transform = _INSTRUMENT_LABELS.get(item.kind, _PLAIN_INSTRUMENT)
                instruments.append((dim(label) if label else "") + transform(item.text))
        return "  " + "   ".join(instruments) if instruments else ""

    def calm_compose(items: list[Any], status_width: int) -> str:
        tail = instrument_tail(items)
        return (
            paint("╭─ ", status_color)
            + _status_bay(headline_text, status_color, status_width)
            + tail
        )

    fit_width = _min_bay_width(headline_text) if target_width else STATUS_BAY_WIDTH

    display = list(parts)
    if columns and columns > 0:
        if vis_len(calm_compose(display, fit_width)) > columns:
            display = [p for p in display if p.kind != "models"]
        if vis_len(calm_compose(display, fit_width)) > columns:
            display = [p for p in display if p.kind != "window"]
        if vis_len(calm_compose(display, fit_width)) > columns:
            display = [p for p in display if p.kind != "working"]
        if spec.sheds_context and vis_len(calm_compose(display, fit_width)) > columns:
            display = [p for p in display if p.kind not in ("ctx_load", "window")]
    line1 = calm_compose(
        display,
        _bay_width(
            headline_text, 3 + vis_len(instrument_tail(display)), target_width, columns
        ),
    )

    if columns and columns > 0 and vis_len(line1) > columns:
        return action_rows()

    if not metadata:
        return line1
    return line1 + "\n" + identity_row(max(vis_len(line1), vis_len(metadata)))
