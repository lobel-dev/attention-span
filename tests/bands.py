"""Band-derived sample values, so retuning a context band stays a one-file edit.

Almost every context assertion in the suite needs a token count that lands in a particular
tier. Spelling those numbers out makes a `health_config` retune a multi-file hunt through
failing tests - exactly the friction the config module exists to remove. Retuning yellow
from 64K to 110K broke 15 assertions across four files, none of which were testing the
band's VALUE; they only needed "some yellow load".

So these names say which TIER a value belongs to and derive the number from the shipped
bands. A future retune moves them automatically.

The shipped numbers themselves are still pinned, deliberately, in exactly one place:
``tests/test_health_config.py::TestShippedContextBands``. That is the single test a retune
is supposed to update.
"""

from attention_span import health_config, text

BANDS = health_config.ENGINE_BANDS
LOAD_PEAK = BANDS.CTX_TIER_STRONG // 2
LOAD_STRONG = (BANDS.CTX_TIER_STRONG + BANDS.CTX_TIER_FUNCTIONAL) // 2
LOAD_FUNCTIONAL = (BANDS.CTX_TIER_FUNCTIONAL + BANDS.CTX_TIER_DEGRADING) // 2
LOAD_DEGRADING = (BANDS.CTX_TIER_DEGRADING + BANDS.CTX_TIER_FAILING) // 2
LOAD_FAILING = (BANDS.CTX_TIER_FAILING + BANDS.CTX_TIER_DEAD) // 2
LOAD_DEAD = BANDS.CTX_TIER_DEAD
PCT_LOW = 30
PCT_MID = 65
PCT_HIGH = 88
fmt = text.fmt_tokens

LOAD_PEAK_S = fmt(LOAD_PEAK)
LOAD_STRONG_S = fmt(LOAD_STRONG)
LOAD_FUNCTIONAL_S = fmt(LOAD_FUNCTIONAL)
LOAD_DEGRADING_S = fmt(LOAD_DEGRADING)
LOAD_FAILING_S = fmt(LOAD_FAILING)
LOAD_DEAD_S = fmt(LOAD_DEAD)

FUNCTIONAL_BAND_S = fmt(BANDS.CTX_TIER_FUNCTIONAL)
DEGRADING_BAND_S = fmt(BANDS.CTX_TIER_DEGRADING)
FAILING_BAND_S = fmt(BANDS.CTX_TIER_FAILING)
DEAD_BAND_S = fmt(BANDS.CTX_TIER_DEAD)
LOAD_200K_FUNCTIONAL = (
    BANDS.CTX_TIER_FUNCTIONAL_200K + BANDS.CTX_TIER_DEGRADING_200K
) // 2
LOAD_200K_DEGRADING = (BANDS.CTX_TIER_DEGRADING_200K + BANDS.CTX_TIER_FAILING_200K) // 2
LOAD_200K_FAILING = (BANDS.CTX_TIER_FAILING_200K + BANDS.CTX_TIER_DEAD_200K) // 2
LOAD_200K_DEAD = BANDS.CTX_TIER_DEAD_200K

LOAD_200K_DEGRADING_S = fmt(LOAD_200K_DEGRADING)
DEGRADING_200K_BAND_S = fmt(BANDS.CTX_TIER_DEGRADING_200K)
