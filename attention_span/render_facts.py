"""Structured facts passed from session analysis to status rendering."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Instrument:
    """One Row-1 reading: WHAT it is and its already-formatted value."""

    kind: str
    text: str


@dataclass(frozen=True)
class Reason:
    """Row 2's one-line explanation, in plain text plus styling hints."""

    text: str
    label: str = ""
    value: str = ""
    use_context_color: bool = False


@dataclass(frozen=True)
class ContextLoad:
    """The context verdict and the numbers behind it, as one fact."""

    tier: str | None = None
    tokens: Any = 0
    pct: float | None = None
    is_live: bool = False


@dataclass(frozen=True)
class Notice:
    """Transient announcement that subagents finished since the last render."""

    done_n: int = 0
    live_n: int = 0

    def __bool__(self) -> bool:
        return self.done_n > 0


@dataclass(frozen=True)
class Identity:
    """Row 2's ambient facts: which model is running, where, and what changed."""

    model_id: str = ""
    effort: str = ""
    repo_name: str = ""
    cwd: str = ""
    lines_added: Any = None
    lines_removed: Any = None
    turns: Any = None
    fast_mode: bool = False


@dataclass(frozen=True)
class RenderFacts:
    """The one structured snapshot of session facts the render seam carries."""

    context: ContextLoad = field(default_factory=ContextLoad)
    identity: Identity = field(default_factory=Identity)
    notice: Notice | None = None
    blind_loop: Mapping[str, Any] | None = None
    subagent_blind_loop: bool = False
    cache_health: Mapping[str, Any] | None = None
    parse_degraded: bool = False
    last_stop_reason: Any = ""
    show_context: bool = True
    show_agents: bool = True
    agents_total: int = 0
    agents_live: int = 0
    agents_tokens: Any = 0
    agents_tokens_complete: bool = True
    agents_tokens_warming: bool | None = None
    # Distinct model ids of the unfinished children/subagents
    agents_models: tuple[str, ...] = ()
