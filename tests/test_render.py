"""Behavior tests for the pure, action-first statusline renderer."""

import dataclasses
import re
import unittest

from bands import (
    LOAD_DEAD,
    LOAD_DEGRADING,
    LOAD_DEGRADING_S,
    LOAD_FAILING,
    LOAD_FUNCTIONAL,
    LOAD_FUNCTIONAL_S,
    LOAD_PEAK,
    LOAD_PEAK_S,
    LOAD_STRONG,
    LOAD_STRONG_S,
    PCT_HIGH,
    PCT_LOW,
    PCT_MID,
    fmt,
)

from attention_span import (
    agent_health,
    health_config,
    render,
    render_facts,
    status_catalog,
    subagents,
    text,
)

NOW = 1_700_000_000


def cohort(
    live=0,
    done=0,
    *,
    total=None,
    state="green",
    insufficient=False,
    tokens=0,
    known=None,
    untrusted=0,
    models=(),
):
    """A Cohort with `live` working children (each `state`) and `done` finished ones.

    `total` overrides the inventory when it must exceed the health detail the renderer
    was handed; `known` under the inventory leaves the burn still warming.
    """
    total_n = live + done if total is None else total
    return subagents.Cohort(
        live=tuple(
            subagents.Subagent(state=state, insufficient=insufficient)
            for _ in range(live)
        ),
        total_n=total_n,
        done_n=done,
        tokens_total=tokens,
        tokens_known_n=total_n if known is None else known,
        tokens_untrusted_n=untrusted,
        models=tuple(models),
    )


def analysis(**over):
    """A healthy ``Analysis`` the renderer can read, overridable field by field."""
    return dataclasses.replace(
        agent_health.Analysis(
            turns=3,
            context_tokens=50_000,
            insufficient=False,
            last_model="claude-opus-4-8",
            cache_health={"show": False},
            last_stop_reason="end_turn",
        ),
        **over,
    )


def _record(over, record):
    """Lift ``record``'s own fields out of a flat kwargs dict."""
    return {
        f.name: over.pop(f.name) for f in dataclasses.fields(record) if f.name in over
    }


def facts(**over):
    """A RenderFacts from flat keywords, with the cohort counts a live session resolves."""
    context = _record(over, render_facts.ContextLoad)
    identity = _record(over, render_facts.Identity)
    c = over.pop("cohort", None)
    if c is not None:
        over.setdefault("subagents", tuple(c.live))
        over.setdefault("agents_total", c.total_n)
        over.setdefault("agents_live", c.live_n)
        over.setdefault("agents_tokens", c.tokens_total)
        over.setdefault("agents_tokens_complete", c.tokens_complete)
        over.setdefault("agents_tokens_warming", c.tokens_warming)
        over.setdefault("agents_models", c.models)
    subagents = tuple(over.pop("subagents", ()) or ())
    done_n = over.pop("agents_done", 0)
    over.setdefault("agents_total", done_n + len(subagents))
    over.setdefault("agents_live", len(subagents))

    def _state(a):
        return a.get("state") if isinstance(a, dict) else getattr(a, "state", None)

    over.setdefault("subagent_blind_loop", any(_state(a) == "red" for a in subagents))
    return render_facts.RenderFacts(
        context=render_facts.ContextLoad(**context),
        identity=render_facts.Identity(**identity),
        **over,
    )


def panel(a=None, **kw):
    """render_panel driven exactly as statusline.py drives it: analysis, facts, layout."""
    a = a or analysis()
    kw.setdefault("tier", "strong")
    kw.setdefault("pct", 24)
    kw.setdefault("tokens", a.context_tokens)
    kw.setdefault("turns", a.turns)
    kw.setdefault("cache_health", a.cache_health)
    kw.setdefault("parse_degraded", a.parse_health.degraded)
    kw.setdefault("last_stop_reason", a.last_stop_reason)
    layout = render.LayoutOpts(
        columns=kw.pop("columns", None), no_color=kw.pop("no_color", True)
    )
    return render.render_panel(facts(**kw), layout)


class TestAmbientStatus(unittest.TestCase):
    def panel(self, a=None, **kw):
        return panel(a, **kw)

    def test_healthy_panel_uses_the_framed_instrument_layout(self):

        out = self.panel(
            cohort=cohort(5, 4, tokens=500_000),
            effort="max",
            model_id="claude-opus-4-8[1m]",
            repo_name="attention-span",
            cwd="~/dev/barnett-l/attention-span",
            columns=120,
        )
        self.assertEqual(
            out,
            "╭─ 🌕 PEAK ───  CONTEXT LOAD 50K"
            "   WINDOW ██░░░░░░░░ 24%"
            "   WORKING 5\n"
            "╰─ ~/dev/barnett-l/attention-span   ↺ 3          OPUS 4.8 / 1M · MAX",
        )

    def test_no_usage_fact_reaches_row_one_at_any_width(self):

        for columns in (200, 160, 120, 100, 80, 74, 40, 24):
            with self.subTest(columns=columns):
                out = self.panel(
                    cohort=cohort(5, 4, tokens=500_000),
                    effort="max",
                    model_id="claude-fable-5",
                    repo_name="attention-span",
                    cwd="~/dev/barnett-l/attention-span",
                    columns=columns,
                )
                rows = out.splitlines()
                self.assertLessEqual(len(rows), 2, out)
                self.assertTrue(
                    all(render.vis_len(row) <= columns for row in rows), out
                )
                for gone in ("7D", "5H", "PACE USAGE", "resets ", "at this rate", "🚨"):
                    self.assertNotIn(gone, out, gone)

    def test_the_identity_row_still_names_the_model(self):

        out = self.panel(columns=120, model_id="claude-fable-5", effort="max")
        self.assertEqual(
            out.splitlines()[1],
            "╰─ ↺ 3                                     FABLE 5 · MAX",
        )

    def test_fast_mode_marks_the_identity_row_with_a_bolt(self):
        out = self.panel(
            columns=120, model_id="claude-opus-5", effort="max", fast_mode=True
        )
        self.assertTrue(out.splitlines()[1].endswith("⚡ OPUS 5 · MAX"), out)
        off = self.panel(columns=120, model_id="claude-opus-5", effort="max")
        self.assertNotIn("⚡", off)

    def test_session_metadata_renders_on_a_dedicated_second_row(self):
        out = self.panel(
            model_id="claude-opus-4-8[1m]",
            effort="max",
            repo_name="attention-span",
            cwd="~/dev/barnett-l/attention-span",
        )
        self.assertEqual(
            out,
            "╭─ 🌕 PEAK ────────  CONTEXT LOAD 50K"
            "   WINDOW ██░░░░░░░░ 24%\n"
            "╰─ ~/dev/barnett-l/attention-span   ↺ 3   OPUS 4.8 / 1M · MAX",
        )

    def test_session_metadata_uses_repo_as_location_fallback(self):
        out = self.panel(
            model_id="claude-opus-4-8[1m]",
            effort="max",
            repo_name="attention-span",
        )
        self.assertEqual(
            out.splitlines()[1],
            "╰─ attention-span   ↺ 3              OPUS 4.8 / 1M · MAX",
        )

    def test_identity_never_wears_a_context_tier_colour(self):

        row = render.render_metadata_row(
            model_id="claude-fable-5",
            effort="xhigh",
            cwd="~/dev/lobel-dev/claude-code-health",
            lines_added=120,
            lines_removed=30,
            rail_color="green",
            columns=120,
            no_color=False,
        )
        identity_part = row.split("   ", 1)[1]
        self.assertIn(render._RAW["magenta"], identity_part)
        self.assertIn(render._RAW["moss"], identity_part)
        self.assertIn(render._RAW["rose"], identity_part)
        for tier in health_config.CONTEXT_TIERS:
            with self.subTest(tier=tier):
                self.assertNotIn(
                    render._RAW[render.context_tier_color(tier)], identity_part
                )

    def test_no_active_agents_omits_working_count(self):
        self.assertEqual(
            self.panel(),
            "╭─ 🌕 PEAK ───────────────────  CONTEXT LOAD 50K   WINDOW ██░░░░░░░░ 24%",
        )

    def test_the_window_axis_sheds_before_the_token_count(self):

        values = {
            "tier": "degrading",
            "tokens": 213_000,
            "is_live": True,
            "pct": 21,
            "cohort": cohort(5),
        }
        wide = self.panel(columns=120, **values).splitlines()[0]
        self.assertIn("WINDOW", wide)
        self.assertIn("CONTEXT LOAD 213K", wide)

        squeezed = self.panel(columns=80, **values).splitlines()[0]
        self.assertNotIn("WINDOW", squeezed)
        self.assertIn("CONTEXT LOAD 213K", squeezed)
        self.assertIn("WORKING 5", squeezed)

    def test_foreign_child_model_chip_joins_the_working_count(self):
        line = self.panel(
            model_id="claude-fable-5",
            cohort=cohort(2, models=("claude-opus-5",)),
            columns=140,
        ).splitlines()[0]
        self.assertIn("WORKING 2   ⇄ opus-5", line)

    def test_mixed_child_models_render_as_a_count(self):
        line = self.panel(
            model_id="claude-fable-5",
            cohort=cohort(3, models=("claude-opus-5", "claude-sonnet-5")),
            columns=140,
        ).splitlines()[0]
        self.assertIn("WORKING 3   ⇄ 2 models", line)

    def test_the_model_chip_sheds_before_every_other_ambient_fact(self):

        values = {
            "tier": "degrading",
            "tokens": 213_000,
            "is_live": True,
            "pct": 21,
            "model_id": "claude-fable-5",
            "cohort": cohort(5, models=("claude-opus-5",)),
        }
        wide = self.panel(columns=140, **values).splitlines()[0]
        self.assertIn("⇄ opus-5", wide)
        self.assertIn("WINDOW", wide)

        squeezed = self.panel(columns=80, **values).splitlines()[0]
        self.assertNotIn("⇄", squeezed)
        self.assertIn("WINDOW", squeezed)
        self.assertIn("WORKING 5", squeezed)

    def test_a_full_window_no_longer_buys_the_bar_a_reprieve(self):

        line = self.panel(
            tier="strong",
            tokens=40_000,
            is_live=True,
            pct=PCT_HIGH,
            cohort=cohort(5),
            columns=80,
        ).splitlines()[0]
        self.assertNotIn("WINDOW", line)
        self.assertIn("CONTEXT LOAD 40K", line)

    def test_a_firing_tier_states_its_status_beside_both_facts(self):
        out = self.panel(
            analysis(context_tokens=LOAD_FUNCTIONAL),
            tier="functional",
            pct=PCT_LOW,
        )
        self.assertEqual(
            out,
            "╭─ 🌗 FUNCTIONAL ─────────────  CONTEXT LOAD "
            + LOAD_FUNCTIONAL_S
            + "   WINDOW ███░░░░░░░ 30%",
        )

    def test_every_firing_tier_states_its_own_status(self):

        cases = (
            (LOAD_FUNCTIONAL, "functional", "🌗 FUNCTIONAL"),
            (LOAD_DEGRADING, "degrading", "🌘 DEGRADING"),
            (LOAD_FAILING, "failing", "🌑 FAILING"),
            (LOAD_DEAD, "dead", "💀 DEAD"),
        )
        for tokens, tier, headline in cases:
            with self.subTest(tier=tier):
                out = self.panel(
                    analysis(context_tokens=tokens),
                    tier=tier,
                    pct=PCT_LOW,
                    columns=120,
                )
                self.assertTrue(out.startswith("╭─ " + headline), out)
                self.assertIn("CONTEXT LOAD " + fmt(tokens), out)

    def test_the_two_calm_tiers_never_claim_the_bay(self):

        for tokens, tier, shown in (
            (LOAD_PEAK, "peak", LOAD_PEAK_S),
            (LOAD_STRONG, "strong", LOAD_STRONG_S),
        ):
            with self.subTest(tier=tier):
                out = self.panel(
                    analysis(context_tokens=tokens), tier=tier, pct=PCT_LOW
                )
                self.assertTrue(out.startswith("╭─ 🌕 PEAK"), out)
                self.assertIn("CONTEXT LOAD " + shown, out)

    def test_context_cause_survives_from_thirty_two_to_forty_columns(self):
        for columns in (40, 39, 35, 32):
            with self.subTest(columns=columns):
                out = self.panel(
                    analysis(context_tokens=LOAD_FUNCTIONAL),
                    tier="functional",
                    pct=PCT_LOW,
                    columns=columns,
                )
                self.assertTrue(
                    all(render.vis_len(line) <= columns for line in out.splitlines())
                )
                self.assertEqual(
                    out, "╭ FUNCTIONAL\n╰ Context load: " + LOAD_FUNCTIONAL_S
                )

    def test_the_token_count_is_the_reason_even_on_a_nearly_full_window(self):
        wide = self.panel(
            analysis(context_tokens=LOAD_STRONG), tier="strong", pct=PCT_MID
        )
        narrow = self.panel(
            analysis(context_tokens=LOAD_STRONG),
            tier="strong",
            pct=PCT_MID,
            columns=40,
        )
        self.assertEqual(
            wide,
            "╭─ 🌕 PEAK ───────────────────  CONTEXT LOAD "
            + LOAD_STRONG_S
            + "   WINDOW ██████░░░░ "
            + str(PCT_MID)
            + "%",
        )
        self.assertEqual(narrow, "╭ PEAK\n╰ Context load: " + LOAD_STRONG_S)

    def test_the_load_fact_carries_the_tier_colour_and_the_window_never_does(self):
        pct = str(PCT_LOW) + "%"
        wide = self.panel(
            analysis(context_tokens=LOAD_FUNCTIONAL),
            tier="functional",
            pct=PCT_LOW,
            no_color=False,
        )
        self.assertIn(render.paint(LOAD_FUNCTIONAL_S, "yellow"), wide)
        self.assertIn(render.dim(pct), wide)
        self.assertNotIn(render.paint(pct, "yellow"), wide)
        self.assertIn(render.dim("░"), wide)

        narrow = self.panel(
            analysis(context_tokens=LOAD_FUNCTIONAL),
            tier="functional",
            pct=PCT_LOW,
            columns=40,
            no_color=False,
        )
        self.assertIn(
            render.dim("Context load: ") + render.paint(LOAD_FUNCTIONAL_S, "yellow"),
            narrow,
        )

    def test_every_tier_paints_the_load_fact_in_its_own_colour(self):

        cases = (
            (LOAD_PEAK, "peak", "mint"),
            (LOAD_STRONG, "strong", "green"),
            (LOAD_FUNCTIONAL, "functional", "yellow"),
            (LOAD_DEGRADING, "degrading", "orange"),
            (LOAD_FAILING, "failing", "red"),
            (LOAD_DEAD, "dead", "magenta_red"),
        )
        for tokens, tier, colour in cases:
            with self.subTest(tier=tier):
                out = self.panel(
                    analysis(context_tokens=tokens),
                    tier=tier,
                    pct=PCT_LOW,
                    columns=120,
                    no_color=False,
                )
                self.assertIn(render.paint(fmt(tokens), colour), out)

    def test_the_window_bar_wears_the_tier_colour_not_a_fill_keyed_ramp(self):

        out = self.panel(
            analysis(context_tokens=LOAD_DEAD),
            tier="dead",
            pct=PCT_HIGH,
            columns=120,
            no_color=False,
        )
        self.assertIn(render.dim(str(PCT_HIGH) + "%"), out)
        self.assertIn(render.paint("█", render.context_tier_color("dead")), out)
        self.assertNotIn(render.dim("█"), out)

    def test_a_low_fill_high_load_window_bar_is_never_green(self):

        out = self.panel(
            analysis(context_tokens=LOAD_FAILING),
            tier="failing",
            pct=PCT_LOW,
            columns=120,
            no_color=False,
        )
        self.assertIn(render.paint("█", render.context_tier_color("failing")), out)

        for stop in render._GAUGE_RAMP[:5]:
            self.assertNotIn(render._paint_raw("█", stop), out)

    def test_context_guidance_respects_context_visibility(self):
        out = self.panel(
            analysis(context_tokens=LOAD_DEGRADING),
            tier="degrading",
            pct=14,
            show_context=False,
        )
        self.assertTrue(out.startswith("╭─ 🌕 PEAK"))
        self.assertNotIn("CONTEXT", out)

    def test_context_actions_fit_micro_panes_without_extra_rows(self):

        tiers = (
            ("functional", LOAD_FUNCTIONAL, "FUNCTIONAL"),
            ("degrading", LOAD_DEGRADING, "DEGRADING"),
            ("failing", LOAD_FAILING, "FAILING"),
            ("dead", LOAD_DEAD, "DEAD"),
        )
        for tier, tokens, word in tiers:
            for columns in (24, 12):
                with self.subTest(tier=tier, columns=columns):
                    out = self.panel(
                        analysis(context_tokens=tokens),
                        tier=tier,
                        pct=14,
                        columns=columns,
                    )
                    self.assertEqual(out, word)
                    self.assertLessEqual(render.vis_len(out), columns)
                    self.assertEqual(len(out.splitlines()), 1)

    def test_a_pane_narrower_than_the_status_word_still_never_wraps(self):

        for tier, tokens, word in (
            ("functional", LOAD_FUNCTIONAL, "FUNCTIONAL"),
            ("degrading", LOAD_DEGRADING, "DEGRADING"),
            ("failing", LOAD_FAILING, "FAILING"),
            ("dead", LOAD_DEAD, "DEAD"),
        ):
            with self.subTest(tier=tier):
                out = self.panel(
                    analysis(context_tokens=tokens),
                    tier=tier,
                    pct=14,
                    columns=9,
                )
                self.assertEqual(len(out.splitlines()), 1)
                self.assertLessEqual(render.vis_len(out), 9)
                self.assertTrue(word.startswith(out.removesuffix("…")), out)

    def test_the_narrow_reason_is_the_token_count_at_every_tier(self):

        cases = [
            (
                LOAD_FUNCTIONAL,
                "functional",
                "FUNCTIONAL",
                "Context load: " + LOAD_FUNCTIONAL_S,
            ),
            (
                LOAD_DEGRADING,
                "degrading",
                "DEGRADING",
                "Context load: " + LOAD_DEGRADING_S,
            ),
            (LOAD_DEAD, "dead", "DEAD", "Context load: " + fmt(LOAD_DEAD)),
        ]
        for context_tokens, tier, action, reason in cases:
            with self.subTest(tier=tier):
                out = self.panel(
                    analysis(context_tokens=context_tokens),
                    tier=tier,
                    pct=PCT_HIGH,
                    columns=74,
                )
                self.assertEqual(out, "╭ " + action + "\n╰ " + reason)
                self.assertEqual(len(out.splitlines()), 2)
                self.assertTrue(
                    all(render.vis_len(line) <= 74 for line in out.splitlines())
                )

    def test_blind_loop_has_highest_confident_priority(self):
        out = self.panel(
            analysis(context_tokens=160_000),
            blind_loop={"state": "red", "base": "cfg.json", "count": 3},
            tier="red",
            model_id="claude-opus-4-8[1m]",
            effort="max",
            repo_name="attention-span",
            cwd="~/dev/attention-span",
            columns=120,
        )
        lines = out.splitlines()
        self.assertEqual(
            lines[0],
            "╭─ ■ READ FILE, THEN RETRY ───  FILE cfg.json ×3"
            "   CONTEXT LOAD 160K   WINDOW ██░░░░░░░░ 24%",
        )
        self.assertEqual(
            lines[1],
            "╰─ ~/dev/attention-span   ↺ 3"
            "                                            OPUS 4.8 / 1M · MAX",
        )
        self.assertNotIn("Healthy", out)

    def test_blind_loop_detail_is_control_character_sanitized(self):
        out = self.panel(
            blind_loop={"state": "red", "base": "bad\nname.json", "count": 2}
        )
        self.assertNotIn("\nname", out)

    def test_narrow_edit_warning_says_what_to_do_and_names_the_file(self):
        out = self.panel(
            blind_loop={"state": "red", "base": "config.json", "count": 3},
            columns=40,
        )
        self.assertEqual(out, "╭ READ FILE, THEN RETRY\n╰ File: config.json")
        self.assertNotIn("EDIT LOOP", out)

    def test_wide_edit_warning_keeps_an_ordinary_filename_complete(self):
        out = self.panel(
            blind_loop={"state": "red", "base": "config.json", "count": 3},
            columns=120,
        )
        self.assertIn("FILE config.json ×3", out)

    def test_narrow_reason_counts_wide_filename_characters(self):
        out = self.panel(
            blind_loop={
                "state": "red",
                "base": "🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥.json",
                "count": 3,
            },
            columns=25,
        )
        self.assertTrue(all(render.vis_len(line) <= 25 for line in out.splitlines()))

    def test_long_edit_filename_moves_to_a_dedicated_reason_line(self):
        out = self.panel(
            blind_loop={
                "state": "red",
                "base": "packages/a-very-long-component/config.json",
                "count": 12,
            },
            columns=75,
            cohort=cohort(5),
        )
        self.assertEqual(
            out,
            "╭ READ FILE, THEN RETRY\n"
            "╰ File: packages/a-very-long-component/config.json",
        )
        self.assertTrue(all(render.vis_len(line) <= 75 for line in out.splitlines()))

        for columns in (33, 32):
            with self.subTest(columns=columns):
                compact = self.panel(
                    blind_loop={
                        "state": "red",
                        "base": "packages/a-very-long-component/config.json",
                        "count": 12,
                    },
                    columns=columns,
                )
                self.assertTrue(
                    all(
                        render.vis_len(line) <= columns for line in compact.splitlines()
                    )
                )
                self.assertEqual(compact.splitlines()[0], "╭ READ FILE, THEN RETRY")
                self.assertTrue(compact.splitlines()[1].startswith("╰ File: packages/"))
                self.assertIn("…", compact.splitlines()[1])
                self.assertNotIn("C24%", compact)

    def test_child_blind_loop_is_actionable(self):
        out = self.panel(cohort=cohort(1, state="red", insufficient=True))
        self.assertEqual(
            out,
            "╭─ ■ CHECK CHILD AGENT ───────  CONTEXT LOAD 50K"
            "   WINDOW ██░░░░░░░░ 24%   WORKING 1",
        )

    def test_stop_reasons_are_worded(self):
        self.assertIn(
            "▲ CHECK LAST RESPONSE", self.panel(analysis(last_stop_reason="max_tokens"))
        )
        refused = self.panel(analysis(last_stop_reason="refusal"))
        self.assertIn("▲ CHECK LAST RESPONSE", refused)
        self.assertIn("LAST RESPONSE REFUSED", refused)

    def test_unknown_stop_reason_stays_healthy(self):
        self.assertTrue(
            self.panel(analysis(last_stop_reason="future")).startswith("╭─ 🌕 PEAK")
        )

    def test_cache_degradation_is_not_healthy(self):
        out = self.panel(analysis(cache_health={"show": True, "hit_drop": 42}))
        self.assertTrue(out.startswith("╭─ ▲ CHECK CACHE - COST RISING"))
        self.assertIn("REUSE FELL 42%", out)

    def test_parse_degradation_suppresses_confident_claims(self):
        degraded = analysis(parse_health=agent_health.ParseHealth(degraded=True))
        live = self.panel(
            degraded,
            is_live=True,
            blind_loop={"state": "red", "base": "x"},
        )
        stale = self.panel(degraded, is_live=False)
        self.assertTrue(live.startswith("╭─ ◌ CAN'T CHECK SESSION"))
        self.assertTrue(stale.startswith("╭─ ◌ CAN'T CHECK SESSION"))
        self.assertNotIn("Edit loop", live)
        self.assertNotIn("Healthy", live + stale)
        self.assertNotIn("ctx", stale)

    def test_notice_replaces_ambient_line_but_not_alarm(self):
        notice = render_facts.Notice(done_n=3, live_n=2)

        self.assertEqual(
            self.panel(notice=notice), "╭─ ✓ 3 SUBAGENTS FINISHED ────  WORKING 2"
        )
        alarm = self.panel(notice=notice, tier="degrading")
        self.assertTrue(alarm.startswith("╭─ 🌘 DEGRADING"))

    def test_extreme_cumulative_delegation_cannot_be_healthy(self):
        out = self.panel(
            pct=26,
            cohort=cohort(1, 124, tokens=12_886_595),
        )
        self.assertEqual(
            out,
            "╭─ ▲ REVIEW CHILD TOKEN BURN ─  SUBAGENTS 125   TOKENS 12.9M"
            "   WORKING 1   CONTEXT LOAD 50K   WINDOW ██░░░░░░░░ 26%",
        )

    def test_fiftieth_total_child_activates_persistent_warning(self):

        out = self.panel(cohort=cohort(1, 49))
        self.assertTrue(out.startswith("╭─ ▲ REVIEW CHILD TOKEN BURN"))

    def test_uncapped_inventory_drives_warning_and_working_count(self):
        out = self.panel(
            cohort=cohort(16, total=50, tokens=2_000_000),
        )
        self.assertIn("▲ REVIEW CHILD TOKEN BURN", out)
        self.assertIn("SUBAGENTS 50", out)
        self.assertIn("WORKING 50", out)

    def test_hiding_agent_details_does_not_change_health_classification(self):
        out = self.panel(
            cohort=cohort(0, 124, total=125, tokens=13_665_260),
            show_agents=False,
        )
        self.assertTrue(out.startswith("╭─ ▲ REVIEW CHILD TOKEN BURN"))
        self.assertIn("SUBAGENTS 125   TOKENS 13.7M", out)
        self.assertNotIn("WORKING", out)

    def test_hiding_agent_details_does_not_hide_child_blind_loop(self):
        out = self.panel(
            cohort=cohort(1, state="red", insufficient=True), show_agents=False
        )
        self.assertTrue(out.startswith("╭─ ■ CHECK CHILD AGENT"))
        self.assertNotIn("WORKING", out)

    def test_partial_burn_is_qualified(self):
        out = self.panel(
            cohort=cohort(0, 125, tokens=2_191_608, known=0),
        )
        self.assertIn("TOKENS ≥2.2M", out)
        self.assertNotIn("TOKENS 2.2M", out)

    def test_high_delegation_has_compact_form_at_forty_columns(self):
        out = self.panel(
            cohort=cohort(0, 125, tokens=13_665_260),
            pct=26,
            columns=40,
        )
        self.assertEqual(
            out, "╭ REVIEW CHILD TOKEN BURN\n╰ 125 subagents; 13.7M tokens"
        )
        self.assertTrue(all(render.vis_len(line) <= 40 for line in out.splitlines()))

    def test_compact_delegation_bounds_extreme_values_at_forty_columns(self):
        out = self.panel(
            cohort=cohort(0, 10**400, tokens=10**400, known=0),
            pct=100,
            columns=40,
        )
        self.assertTrue(all(render.vis_len(line) <= 40 for line in out.splitlines()))
        self.assertEqual(out.splitlines()[0], "╭ REVIEW CHILD TOKEN BURN")
        self.assertIn("999Q+ subagents", out.splitlines()[1])
        self.assertNotIn("ctx", out.lower())

    def test_delegation_facts_survive_just_above_the_narrow_breakpoint(self):
        out = self.panel(
            cohort=cohort(0, 10**400, tokens=10**400, known=0),
            columns=render.NARROW_ACTION_COLS + 1,
        )
        self.assertEqual(out.splitlines()[0], "╭ REVIEW CHILD TOKEN BURN")
        self.assertIn("subagents", out.splitlines()[1])
        self.assertIn("tokens", out.splitlines()[1])
        self.assertTrue(
            all(
                render.vis_len(line) <= render.NARROW_ACTION_COLS + 1
                for line in out.splitlines()
            )
        )

    def test_delegation_has_terse_fallback_for_very_narrow_terminals(self):
        out = self.panel(
            cohort=cohort(0, 10**30, tokens=10**30, known=0),
            pct=100,
            columns=24,
        )
        self.assertEqual(out, "REVIEW CHILD TOKEN BURN")
        self.assertLessEqual(render.vis_len(out), 24)

    def test_delegation_final_fallback_never_exceeds_declared_width(self):
        out = self.panel(
            cohort=cohort(0, 10**30, tokens=10**30, known=0),
            pct=100,
            columns=9,
        )
        self.assertEqual(out, "BURN")

    def test_the_window_percentage_is_stated_as_a_fact_not_parsed_back(self):

        out = self.panel(pct=18)
        self.assertEqual(
            out,
            "╭─ 🌕 PEAK ───────────────────  CONTEXT LOAD 50K   WINDOW █░░░░░░░░░ 18%",
        )

    def test_without_percentage_falls_back_to_tokens(self):
        out = self.panel(pct=None, tokens=50_000)
        self.assertEqual(out, "╭─ 🌕 PEAK ───────────────────  CONTEXT LOAD 50K")

    def test_width_tiers_preserve_the_visual_grammar_and_never_wrap(self):
        agents = cohort(5)
        wide = self.panel(columns=200, cohort=agents)
        medium = self.panel(columns=43, cohort=agents)
        narrow = self.panel(columns=24, cohort=agents)
        tiny = self.panel(columns=9, cohort=agents)
        self.assertIn("WORKING 5", wide)
        self.assertEqual(medium, "╭ PEAK\n╰ Context load: 50K")
        self.assertEqual(narrow, "PEAK")
        self.assertTrue(all(render.vis_len(line) <= 24 for line in narrow.splitlines()))

        self.assertEqual(tiny, "PEAK")
        for output in (medium, narrow, tiny):
            self.assertNotIn("CTX", output)
            self.assertNotIn("5H", output)
            self.assertNotIn("WK", output)

    def test_micro_panes_keep_complete_plain_words(self):
        cases = [
            (self.panel(cohort=cohort(1, state="red"), columns=9), "CHECK"),
            (self.panel(analysis(cache_health={"show": True}), columns=9), "CACHE"),
            (
                self.panel(
                    analysis(parse_health=agent_health.ParseHealth(degraded=True)),
                    columns=9,
                ),
                "DATA",
            ),
            (self.panel(tier=None, pct=None, columns=9), "WAIT"),
            (self.panel(notice=render_facts.Notice(done_n=3), columns=9), "FINISHED"),
        ]
        for output, expected in cases:
            with self.subTest(output=output):
                self.assertEqual(output, expected)

    def test_fifty_five_columns_gives_the_action_its_own_plain_english_line(self):
        out = self.panel(
            columns=55,
            cohort=cohort(5),
            model_id="claude-opus-4-8[1m]",
            effort="max",
            repo_name="attention-span",
            cwd="~/dev/barnett-l/attention-span",
        )
        self.assertEqual(
            out,
            "╭ PEAK\n╰ Context load: 50K",
        )
        self.assertTrue(all(render.vis_len(line) <= 55 for line in out.splitlines()))
        for packed_label in ("CTX", "5H", "WK", "OPUS4.8"):
            self.assertNotIn(packed_label, out)

    def test_long_identity_is_bounded_at_wide_and_medium_widths(self):
        for columns in (120, 80, 75):
            with self.subTest(columns=columns):
                out = self.panel(
                    columns=columns,
                    model_id="claude-opus-4-8[1m]",
                    effort="max",
                    repo_name="a-very-long-repository-name-for-statusline-testing",
                    cwd="~/dev/organizations/a-very-long-repository-name-for-statusline-testing/packages/deeply/nested/component",
                )
                metadata = out.splitlines()[1]
                self.assertLessEqual(render.vis_len(metadata), columns)
                self.assertIn("OPUS 4.8 / 1M", metadata)
                self.assertIn("MAX", metadata)

        for columns in (120, 80, 70, 2, 1):
            with self.subTest(adversarial_columns=columns):
                out = self.panel(
                    columns=columns,
                    model_id="custom-" + "model" * 100,
                    effort="ultra-effort-level-" * 20,
                    repo_name="repository",
                    cwd="~/dev/repository",
                )
                self.assertTrue(
                    all(render.vis_len(line) <= columns for line in out.splitlines())
                )

    def test_a_wide_glyph_location_is_bounded_in_columns_not_code_points(self):

        cases = (
            ("cwd", "~/dev/" + "🔥" * 25 + "/" + "🔥" * 20),
            ("repo_name", "🔥" * 50),
        )
        for columns in (120, 100, 80, 75):
            for field, value in cases:
                with self.subTest(columns=columns, field=field):
                    out = self.panel(
                        columns=columns,
                        model_id="claude-opus-4-8[1m]",
                        effort="max",
                        lines_added=120,
                        lines_removed=30,
                        **{field: value},
                    )
                    metadata = out.splitlines()[1]
                    self.assertLessEqual(render.vis_len(metadata), columns)
                    self.assertIn("…", metadata)


class TestSharedRightEdge(unittest.TestCase):
    """Row 1 and Row 2 end in ONE column, and BOTH rows stretch to reach it.

    Row 1 flexes its dash fill; Row 2 flexes the join gap before its identity segment,
    so the identity right-aligns. Whichever row is naturally narrower stretches toward
    the wider one. Two clamps bound the edge: Row 1's minimum fill, and the declared
    pane width. A Row 2 with fewer than two segments has no interior gap to widen and
    so stays natural.
    """

    def rows(self, cwd, columns, **kw):
        kw.setdefault("model_id", "claude-opus-4-8[1m]")
        kw.setdefault("effort", "max")
        return panel(columns=columns, cwd=cwd, **kw).splitlines()

    def fill_width(self, row):

        return max(len(run) for run in re.findall(r"─+", row[3:]))

    def segments(self, row):

        return re.split(r" {3,}", row[3:])

    def test_both_rows_end_in_the_same_column(self):
        for columns in (200, 160, 120, 100, 80, 75):
            for depth in range(10):
                cwd = "~/dev/" + "/".join(["segment"] * depth)
                with self.subTest(columns=columns, depth=depth):
                    rows = self.rows(cwd, columns)
                    if len(rows) != 2 or not rows[0].startswith("╭─ "):
                        continue
                    row1, row2 = (render.vis_len(row) for row in rows)
                    self.assertLessEqual(row1, columns)
                    self.assertLessEqual(row2, columns)
                    if row1 == row2:
                        continue
                    if row1 > row2:
                        self.assertLess(len(self.segments(rows[1])), 2, rows[1])
                        self.assertEqual(
                            self.fill_width(rows[0]), render.STATUS_BAY_MIN_FILL
                        )
                    else:
                        self.assertEqual(row1, columns)

    def test_the_identity_row_stretches_under_an_instrument_row_one_cannot_shed(self):

        out = panel(
            analysis(context_tokens=108_000),
            no_color=True,
            tier="strong",
            pct=11,
            columns=120,
            cohort=cohort(1),
            lines_added=37,
            lines_removed=15,
            model_id="claude-fable-5",
            effort="high",
            cwd="~/dev/lobel-dev/health",
        )
        self.assertEqual(
            out,
            "╭─ 🌕 PEAK ───  CONTEXT LOAD 108K   WINDOW █░░░░░░░░░ 11%   WORKING 1\n"
            "╰─ ~/dev/lobel-dev/health   +37/-15   ↺ 3              FABLE 5 · HIGH",
        )
        self.assertEqual(*[render.vis_len(row) for row in out.splitlines()])

    def test_the_stretched_gap_is_plain_spaces(self):

        out = panel(
            analysis(context_tokens=108_000),
            no_color=False,
            tier="strong",
            pct=11,
            columns=120,
            cohort=cohort(1),
            lines_added=37,
            lines_removed=15,
            model_id="claude-fable-5",
            effort="high",
            cwd="~/dev/lobel-dev/health",
        )
        row2 = out.splitlines()[1]
        self.assertRegex(row2, r"\033\[0m {4,}\033\[")
        self.assertEqual(*[render.vis_len(row) for row in out.splitlines()])

    def test_the_fill_stretches_to_reach_a_wide_identity_row(self):
        rows = self.rows("~/dev/organizations/attention-span/packages/renderer", 160)
        self.assertEqual(render.vis_len(rows[0]), render.vis_len(rows[1]))
        self.assertGreater(self.fill_width(rows[0]), render.STATUS_BAY_MIN_FILL)

    def test_the_fill_shrinks_to_a_narrow_identity_row_but_not_past_the_floor(self):

        rows = self.rows("", 120, model_id="claude-fable-5", effort="")
        self.assertEqual(self.fill_width(rows[0]), render.STATUS_BAY_MIN_FILL)
        self.assertEqual(render.vis_len(rows[0]), render.vis_len(rows[1]))

    def test_a_single_segment_identity_row_is_never_padded(self):
        row = render.render_metadata_row(
            model_id="claude-fable-5",
            rail_color="green",
            columns=120,
            no_color=True,
            target_width=90,
        )
        self.assertEqual(row, "╰─ FABLE 5")

    def test_a_lone_row_keeps_the_deliberate_fixed_bay(self):
        out = panel(
            analysis(),
            no_color=True,
            tier="strong",
            pct=24,
            columns=120,
        )
        self.assertEqual(len(out.splitlines()), 1)
        self.assertEqual(
            self.fill_width(out),
            render.STATUS_BAY_WIDTH - render.vis_len("🌕 PEAK") - 1,
        )

    def test_the_standby_row_aligns_with_the_identity_row(self):
        metadata = render.render_metadata_row(
            model_id="claude-opus-4-8[1m]",
            effort="max",
            cwd="~/dev/barnett-l/attention-span",
            rail_color="cyan",
            columns=120,
            no_color=True,
        )
        row = render.render_standby_row(
            columns=120, no_color=True, target_width=render.vis_len(metadata)
        )
        self.assertEqual(render.vis_len(row), render.vis_len(metadata))

    def test_the_standby_row_keeps_the_fixed_bay_without_a_second_row(self):
        row = render.render_standby_row(columns=120, no_color=True)
        self.assertEqual(render.vis_len(row), 3 + render.STATUS_BAY_WIDTH)

    def test_the_compact_row_aligns_with_the_identity_row(self):
        metadata = render.render_metadata_row(
            model_id="claude-opus-4-8[1m]",
            effort="max",
            cwd="~/dev/barnett-l/attention-span",
            rail_color="cyan",
            columns=120,
            no_color=True,
        )
        row = render.render_compact_row(
            columns=120, no_color=True, target_width=render.vis_len(metadata)
        )
        self.assertEqual(render.vis_len(row), render.vis_len(metadata))

    def test_the_compact_row_keeps_the_fixed_bay_without_a_second_row(self):
        row = render.render_compact_row(columns=120, no_color=True)
        self.assertEqual(
            render.vis_len(row),
            3 + render.STATUS_BAY_WIDTH + 2 + len(status_catalog.COMPACT_ACK_DETAIL),
        )


class TestTurnsCounter(unittest.TestCase):
    def metadata(self, **kw):
        kw.setdefault("model_id", "claude-fable-5")
        kw.setdefault("effort", "high")
        kw.setdefault("cwd", "~/dev/lobel-dev/attention-span")
        kw.setdefault("lines_added", 120)
        kw.setdefault("lines_removed", 30)
        kw.setdefault("rail_color", "green")
        kw.setdefault("columns", 120)
        kw.setdefault("no_color", True)
        return render.render_metadata_row(**kw)

    def test_the_identity_row_counts_turns_between_lines_and_identity(self):
        row = self.metadata(turns=12)
        self.assertIn("+120/-30   ↺ 12   FABLE 5 · HIGH", row)

    def test_turns_hide_until_the_first_real_human_turn(self):
        for turns in (None, 0, -3, True, float("nan"), float("inf"), "12"):
            with self.subTest(turns=turns):
                self.assertNotIn("↺", self.metadata(turns=turns))

    def test_turns_wear_the_dim_ambient_style_only(self):
        row = self.metadata(turns=12, no_color=False)
        self.assertIn(render.dim("↺ 12"), row)
        for tier in health_config.CONTEXT_TIERS:
            with self.subTest(tier=tier):
                self.assertNotIn(
                    render.paint("↺ 12", render.context_tier_color(tier)), row
                )

    def test_turns_charge_the_location_budget_not_the_edge(self):
        row = self.metadata(
            turns=1234,
            cwd="~/dev/organizations/attention-span/packages/renderer",
            columns=48,
        )
        self.assertLessEqual(render.vis_len(row), 48)
        self.assertIn("↺ 1234", row)
        self.assertIn("…", row)

    def test_the_turns_counter_never_summons_the_identity_row_alone(self):
        row = self.metadata(
            model_id="",
            effort="",
            cwd="",
            lines_added=None,
            lines_removed=None,
            turns=9,
        )
        self.assertEqual(row, "")

    def test_the_panel_identity_row_carries_the_analyzers_turn_count(self):
        out = panel(
            analysis(turns=7),
            no_color=True,
            tier="strong",
            pct=24,
            columns=120,
            model_id="claude-fable-5",
            effort="high",
            cwd="~/dev/lobel-dev/attention-span",
        )
        rows = out.splitlines()
        self.assertIn("↺ 7", rows[1])
        self.assertEqual(*[render.vis_len(row) for row in rows])


class TestPureHelpers(unittest.TestCase):
    def test_the_seam_carries_only_facts_a_row_can_draw(self):

        retired = ("r2e_chip", "show_r2e", "action", "now", "cx_chip")
        fields = {f.name for f in dataclasses.fields(render_facts.RenderFacts)}
        for name in retired:
            self.assertNotIn(name, fields, name)
        with self.assertRaises(TypeError):
            render_facts.RenderFacts(r2e_chip="Close Watch")

    def test_an_unknown_tier_still_renders_a_calm_panel(self):

        out = panel(tier="green", pct=24)
        self.assertTrue(out.startswith("╭─ 🌕 PEAK"), out)

    def test_the_micro_action_is_measured_in_columns_not_code_points(self):

        spec = status_catalog.STATUSES[status_catalog.CALM_STATUS]
        action = "🔥" * 5
        for width in range(render.MICRO_ACTION_COLS + 1):
            with self.subTest(width=width):
                shown = render._fit_plain_action(action, spec, width)
                self.assertLessEqual(render.vis_len(shown), width)
        self.assertEqual(render._fit_plain_action(action, spec, 0), "")
        self.assertEqual(render._fit_plain_action(action, spec, 1), "…")
        self.assertEqual(render._fit_plain_action(action, spec, 10), action)

    def test_compact_magnitude_is_normalized_and_bounded(self):
        self.assertEqual(text.compact_magnitude(13_665_260), "13.7M")
        self.assertEqual(text.compact_magnitude(999_500_000), "1B")
        self.assertEqual(text.compact_magnitude(10**30), "999Q+")


if __name__ == "__main__":
    unittest.main(verbosity=2)
