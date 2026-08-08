"""Single source of truth for every health-status tuning value.

Numbers and taxonomies only: the bands the engine grades against, the gates the renderer
applies, and the tier vocabulary both share. What a status LOOKS and READS like lives in
``status_catalog``, one row per status; no status wording belongs here.

A literals-only leaf: it imports no project module, so importing it can never introduce
a failure mode into the silent-on-failure render path.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class EngineBands:
    """Bands the engine evaluates. Field names ARE the ``th=`` override keys."""

    WINDOW: int = 10  # rolling window size, in read+edit events (neutrals excluded)
    MIN_WINDOW: int = 5  # below this many read+edit events -> INSUFFICIENT (light off)
    MIN_EDITS_FOR_RED: int = (
        3  # min-activity floor: fewer edits than this cannot escalate the posture
    )
    HEALTHY: float = 1.0  # R2E >= HEALTHY -> green (reads keep pace with edits)
    BLIND: float = 0.5  # R2E <  BLIND   -> red; [BLIND, HEALTHY) -> yellow

    # Base context ladder: a load >= a band opens that band's tier.
    CTX_TIER_STRONG: int = 32_000
    CTX_TIER_FUNCTIONAL: int = 128_000
    CTX_TIER_DEGRADING: int = 200_000
    CTX_TIER_FAILING: int = 300_000
    CTX_TIER_DEAD: int = 600_000

    # The 200K class is reasoned on its own, never a scaling of the base ladder: every
    # tier it grades has to fire before that window's wall.
    CTX_TIER_FUNCTIONAL_200K: int = 80_000
    CTX_TIER_DEGRADING_200K: int = 128_000
    CTX_TIER_FAILING_200K: int = 160_000
    CTX_TIER_DEAD_200K: int = 192_000

    CACHE_WINDOW: int = 12  # trailing assistant turns scored for cache thrash
    CACHE_WARM_TURNS: int = 3  # skip the first N turns - warmup legitimately caches
    CACHE_IDLE_TTL_S: int = (
        300  # inter-turn gap (s) past which a cache miss is legit expiry
    )
    CACHE_HIT_DROP_SHOW: int = (
        25  # show the chip only when the hit-rate drop (pct pts) >= this
    )
    REPEAT_WINDOW: int = 12  # trailing read/edit events scanned for read-repetition
    REPEAT_MIN: int = 5  # >= this many no-progress re-reads of one file -> score fires
    PERSEV_WINDOW: int = 12  # trailing executed tool calls scanned for perseveration
    PERSEV_MIN: int = 3  # >= this many identical no-progress repeats -> score fires
    PERSEV_IDLE_TTL_S: int = (
        300  # gap (s) past which a repeat is a deliberate poll, not a stall
    )
    PARSE_DECODE_MIN: int = (
        2  # >= this many undecodable lines -> degraded (1 = normal tail)
    )
    PARSE_CANARY_MIN_ASSISTANT: int = (
        5  # assistant lines that recognized nothing -> schema canary
    )


@dataclass(frozen=True)
class PresentationGates:
    """Boundaries the renderer applies to already-computed facts."""

    DELEGATION_THRESHOLD: int = 50
    NARROW_ACTION_COLS: int = 74
    MICRO_ACTION_COLS: int = 24
    STATUS_BAY_WIDTH: int = 27
    STATUS_BAY_MIN_FILL: int = 3


ENGINE_BANDS = EngineBands()
PRESENTATION_GATES = PresentationGates()
ENGINE_DEFAULTS: Mapping[str, float] = MappingProxyType(asdict(ENGINE_BANDS))

CONTEXT_TIER_BANDS: Mapping[str, str] = MappingProxyType(
    {
        "dead": "CTX_TIER_DEAD",
        "failing": "CTX_TIER_FAILING",
        "degrading": "CTX_TIER_DEGRADING",
        "functional": "CTX_TIER_FUNCTIONAL",
        "strong": "CTX_TIER_STRONG",
    }
)

# Every tier, worst -> best. Each name is also a ``status_catalog`` row.
CONTEXT_TIERS: tuple[str, ...] = tuple(CONTEXT_TIER_BANDS) + ("peak",)

# The tiers that are an alert; every other tier is a reading, shown without alarm.
CONTEXT_FIRING_TIERS: tuple[str, ...] = ("dead", "failing", "degrading", "functional")

# Advertised window size -> the class name that selects a ladder.
WINDOW_CLASSES: Mapping[int, str] = MappingProxyType(
    {
        200_000: "200k",
        1_000_000: "1m",
    }
)

# Per-class band swaps, tier -> field. An unlisted class or tier keeps the base band.
CONTEXT_CLASS_BAND_OVERRIDES: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "200k": MappingProxyType(
            {
                "functional": "CTX_TIER_FUNCTIONAL_200K",
                "degrading": "CTX_TIER_DEGRADING_200K",
                "failing": "CTX_TIER_FAILING_200K",
                "dead": "CTX_TIER_DEAD_200K",
            }
        ),
    }
)

# The identity row's separator - session identity punctuation, not status copy.
IDENTITY_JOIN = " · "


def engine_defaults() -> dict[str, float]:
    """Return a fresh mutable copy of the engine band defaults."""
    return dict(ENGINE_DEFAULTS)


def resolve_thresholds(
    overrides: Mapping[str, float | None] | None = None,
) -> dict[str, float]:
    """Return complete engine thresholds with known non-None overrides applied."""
    resolved = dict(ENGINE_DEFAULTS)
    if overrides:
        for key in ENGINE_DEFAULTS:
            if key in overrides:
                value = overrides[key]
                if value is not None:
                    resolved[key] = value
    return resolved


def context_band_field(tier: str, window_class: str | None = None) -> str | None:
    """The EngineBands FIELD that opens ``tier`` for ``window_class``."""
    if isinstance(window_class, str):
        overrides = CONTEXT_CLASS_BAND_OVERRIDES.get(window_class)
        if overrides and tier in overrides:
            return overrides[tier]
    return CONTEXT_TIER_BANDS.get(tier)
