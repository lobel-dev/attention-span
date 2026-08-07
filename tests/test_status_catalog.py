"""Behaviour tests for status_catalog - the single declaration of every status.

Every expected string here was hand-traced from the render.py if-chains this module
replaced (the traced line is cited beside each case). That is why the rewire moved no
pixels, and why these cases stay: they pin the rendered wording at its source, where
tests/test_render.py pins the drawn rows the renderer builds from it.
"""

import dataclasses
import unittest

from attention_span import health_config, render, render_facts, text
from attention_span import status_catalog as catalog

EXPECTED_KEYS = {
    "Health data limited",
    "Health data stale",
    "Edit loop",
    "Agent edit loop",
    "Many agent runs",
    "Response truncated",
    "Response refused",
    "Cache problem",
    "Healthy",
    "dead",
    "failing",
    "degrading",
    "functional",
    "strong",
    "peak",
    "Warming",
    "Notice",
    "Compact ack",
}

TIERS = ("dead", "failing", "degrading", "functional", "strong", "peak")


def facts(**over):
    """A RenderFacts built the way statusline.py builds it, from flat keywords.

    The context verdict is one ``ContextLoad``; the flat aliases below exist so the
    table-driven FIRES fixtures can name a tier or a live flag without nesting.
    """
    for alias, name in (
        ("cx_state", "tier"),
        ("context_tokens", "tokens"),
        ("ctx_pct", "pct"),
        ("context_is_live", "is_live"),
    ):
        if alias in over:
            over[name] = over.pop(alias)
    context = {
        f.name: over.pop(f.name)
        for f in dataclasses.fields(render_facts.ContextLoad)
        if f.name in over
    }
    return render_facts.RenderFacts(context=render_facts.ContextLoad(**context), **over)


class TestCatalogCompleteness(unittest.TestCase):
    def test_catalog_declares_exactly_the_expected_statuses(self):
        self.assertEqual(set(catalog.STATUSES), EXPECTED_KEYS)

    def test_every_row_is_fully_populated(self):
        for name, spec in catalog.STATUSES.items():
            self.assertTrue(spec.color, name)
            self.assertTrue(spec.glyph, name)
            self.assertTrue(spec.action, name)
            self.assertTrue(spec.narrow, name)
            self.assertTrue(callable(spec.detail), name)
            self.assertTrue(callable(spec.reason), name)
            self.assertTrue(callable(spec.fires), name)
            self.assertIsInstance(spec.rank, int, name)

    def test_every_row_colour_resolves_in_the_renderer_palette(self):

        for name, spec in catalog.STATUSES.items():
            self.assertIn(spec.color, render._RAW, name)
        self.assertIn(catalog.FALLBACK.color, render._RAW)

    def test_every_builder_survives_a_default_facts_record(self):

        for name, spec in catalog.STATUSES.items():
            self.assertIsInstance(spec.detail(facts()), tuple, name)
            reason = spec.reason(facts())
            self.assertTrue(
                reason is None or isinstance(reason, render_facts.Reason), name
            )

    def test_every_instrument_kind_is_one_the_renderer_can_draw(self):

        drawable = set(render._INSTRUMENT_LABELS) | {"ctx_load", "window"}
        rich = facts(
            blind_loop={"base": "config.json", "count": 3},
            cache_health={"hit_drop": 23.4},
            tokens=50_000,
            pct=24,
            tier="strong",
            notice=render_facts.Notice(done_n=3, live_n=2),
            agents_total=125,
            agents_live=2,
            agents_tokens=2_200_000,
            agents_tokens_complete=False,
        )
        for name, spec in catalog.STATUSES.items():
            for item in spec.detail(rich):
                self.assertIn(item.kind, drawable, name)

    def test_status_specs_are_frozen_and_the_catalog_is_read_only(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            catalog.STATUSES["Healthy"].color = "red"
        with self.assertRaises(TypeError):
            catalog.STATUSES["Healthy"] = None


class TestThePrecedenceLadder(unittest.TestCase):
    """The catalog decides WHICH status fires, so the ladder is pinned here.

    These cases moved out of tests/test_render.py on 2026-08-07 with the if-chain they
    described: precedence used to be an ordered list of appends in ``render``, where
    eleven status names were spelled out a second time with nothing tying them to the
    rows they named.
    """

    def test_the_ladder_is_ordered_by_rank_with_no_two_rows_sharing_a_seat(self):
        ranks = [spec.rank for _, spec in catalog.PRECEDENCE]
        self.assertEqual(ranks, sorted(ranks))
        self.assertEqual(len(set(ranks)), len(ranks))

    def test_a_row_is_in_the_ladder_exactly_when_it_can_be_selected(self):

        seated = {name for name, _ in catalog.PRECEDENCE}
        for name, spec in catalog.STATUSES.items():
            reachable = spec.rank < catalog.RANK_NEVER
            self.assertEqual(name in seated, reachable, name)
            if not reachable:
                self.assertIs(spec.fires, catalog._fires_never, name)

    def test_the_firing_tiers_are_seated_worst_first(self):
        seats = [
            name
            for name, _ in catalog.PRECEDENCE
            if name in health_config.CONTEXT_TIERS
        ]
        self.assertEqual(seats, list(health_config.CONTEXT_FIRING_TIERS))

    def test_the_calm_tiers_never_fire(self):
        for tier in ("strong", "peak"):
            self.assertFalse(catalog.STATUSES[tier].fires(facts(cx_state=tier)), tier)
            self.assertNotEqual(catalog.dominant_status(facts(cx_state=tier)), tier)

    def test_exactly_one_row_sheds_its_context_under_pressure(self):
        shedders = [n for n, s in catalog.STATUSES.items() if s.sheds_context]
        self.assertEqual(shedders, ["Many agent runs"])

    FIRES = (
        ("Health data limited", {"parse_degraded": True, "context_is_live": True}),
        ("Health data stale", {"parse_degraded": True}),
        ("Edit loop", {"blind_loop": {"chip": "EDIT LOOP: a.py x3"}}),
        ("Agent edit loop", {"subagent_blind_loop": True}),
        ("Many agent runs", {"agents_total": catalog.DELEGATION_THRESHOLD}),
        ("dead", {"cx_state": "dead"}),
        ("failing", {"cx_state": "failing"}),
        ("degrading", {"cx_state": "degrading"}),
        ("functional", {"cx_state": "functional"}),
        ("Response truncated", {"last_stop_reason": "max_tokens"}),
        ("Response refused", {"last_stop_reason": "refusal"}),
        ("Cache problem", {"cache_health": {"show": True}}),
        (catalog.WARMING_STATUS, {}),
        (catalog.CALM_STATUS, {"cx_state": "strong"}),
    )

    def test_every_row_fires_on_its_own_facts_and_wins_the_ladder(self):
        for name, over in self.FIRES:
            with self.subTest(name):
                self.assertTrue(catalog.STATUSES[name].fires(facts(**over)))
                self.assertEqual(catalog.dominant_status(facts(**over)), name)

    def test_no_row_fires_on_facts_meant_for_another(self):
        for name, over in self.FIRES:
            for other, _ in self.FIRES:
                if other == name or catalog.STATUSES[other].pseudo:
                    continue
                with self.subTest(f"{other} on {name}"):
                    self.assertFalse(catalog.STATUSES[other].fires(facts(**over)))

    def test_a_notice_answers_with_its_own_text_not_its_row_name(self):
        notice = render_facts.Notice(done_n=3, live_n=2)
        self.assertEqual(
            catalog.dominant_status(facts(notice=notice)), catalog.NOTICE_STATUS
        )
        self.assertIs(
            catalog.resolve(catalog.dominant_status(facts(notice=notice))),
            catalog.STATUSES[catalog.NOTICE_STATUS],
        )
        self.assertEqual(
            catalog.action_for(catalog.NOTICE_STATUS, notice=notice),
            "3 SUBAGENTS FINISHED",
        )

    def test_a_warning_outranks_a_notice_and_a_notice_outranks_a_cold_start(self):
        notice = render_facts.Notice(done_n=3, live_n=2)
        self.assertEqual(
            catalog.dominant_status(facts(notice=notice, cx_state="degrading")),
            "degrading",
        )

        self.assertEqual(
            catalog.dominant_status(facts(notice=notice)), catalog.NOTICE_STATUS
        )

    def test_a_cold_start_needs_context_tracking_switched_on(self):
        self.assertEqual(
            catalog.dominant_status(facts(show_context=False)), catalog.CALM_STATUS
        )

    def test_the_calm_floor_is_reached_only_when_nothing_else_fires(self):
        self.assertEqual(catalog.active_warnings(facts(cx_state="strong")), ())
        self.assertEqual(
            catalog.dominant_status(facts(cx_state="strong")), catalog.CALM_STATUS
        )

    def test_all_concurrent_warnings_are_available_for_inspection(self):

        self.assertEqual(
            catalog.active_warnings(
                facts(
                    cx_state="functional",
                    last_stop_reason="max_tokens",
                    cache_health={"show": True},
                )
            ),
            ("functional", "Response truncated", "Cache problem"),
        )

    def test_parse_doubt_keeps_independent_warnings_for_inspection(self):
        self.assertEqual(
            catalog.active_warnings(
                facts(parse_degraded=True, context_is_live=True, agents_total=125)
            ),
            ("Health data limited", "Many agent runs"),
        )

    def test_parse_doubt_silences_the_detectors_that_need_a_readable_transcript(self):

        silenced = facts(
            parse_degraded=True,
            blind_loop={"chip": "EDIT LOOP: a.py x3"},
            last_stop_reason="max_tokens",
            cache_health={"show": True},
            cx_state="degrading",
        )
        self.assertEqual(catalog.active_warnings(silenced), ("Health data stale",))

    def test_a_live_sample_lets_a_tier_fire_through_parse_doubt(self):
        live = facts(parse_degraded=True, context_is_live=True, cx_state="degrading")
        self.assertEqual(
            catalog.active_warnings(live), ("Health data limited", "degrading")
        )

    def test_a_hidden_context_tier_is_not_a_status(self):
        self.assertEqual(
            catalog.active_warnings(facts(cx_state="dead", show_context=False)), ()
        )

    def test_the_delegation_gate_is_inclusive(self):
        gate = catalog.DELEGATION_THRESHOLD
        self.assertIn(
            "Many agent runs", catalog.active_warnings(facts(agents_total=gate))
        )
        self.assertEqual(catalog.active_warnings(facts(agents_total=gate - 1)), ())

    def test_pseudo_statuses_never_count_as_warnings(self):

        self.assertEqual(catalog.active_warnings(facts()), ())
        self.assertEqual(catalog.dominant_status(facts()), catalog.WARMING_STATUS)


class TestTheRendererBindsTheCatalog(unittest.TestCase):
    """The catalog is the only declaration; the renderer binds it, never a copy."""

    def test_every_tier_this_config_names_is_a_catalog_row(self):
        for tier in health_config.CONTEXT_TIERS:
            self.assertIn(tier, catalog.STATUSES, tier)

    def test_tiers_carry_the_last_resort_ladder_the_renderer_gives_them(self):

        for tier in TIERS:
            self.assertEqual(
                catalog.STATUSES[tier].narrow, (catalog.UNKNOWN_STATUS_ACTION,)
            )

    def test_the_standby_rows_carry_the_standby_copy(self):
        self.assertEqual(catalog.STATUSES["Warming"].action, catalog.WARMING_ACTION)
        self.assertEqual(
            catalog.STATUSES["Compact ack"].action, catalog.COMPACT_ACK_ACTION
        )

    def test_the_renderer_re_exports_the_width_primitives(self):

        self.assertIs(render.vis_len, text.vis_len)
        self.assertIs(render._right_ellipsis, text.right_ellipsis)


class TestResolve(unittest.TestCase):
    def test_known_status_resolves_to_its_own_row(self):
        self.assertIs(
            catalog.resolve("Cache problem"), catalog.STATUSES["Cache problem"]
        )
        self.assertIs(catalog.resolve("degrading"), catalog.STATUSES["degrading"])

    def test_the_notice_resolves_by_name_like_any_other_row(self):

        self.assertIs(catalog.resolve("Notice"), catalog.STATUSES["Notice"])

    def test_unknown_status_falls_back(self):
        for status in ("Bogus", "", None, "PEAK"):
            self.assertIs(catalog.resolve(status), catalog.FALLBACK, status)

    def test_the_fallback_row_shows_nothing_and_advises_a_check(self):
        self.assertEqual(catalog.FALLBACK.narrow, (catalog.UNKNOWN_STATUS_ACTION,))
        self.assertEqual(catalog.FALLBACK.detail(facts()), ())
        self.assertIsNone(catalog.FALLBACK.reason(facts()))


class TestActionText(unittest.TestCase):
    def test_notice_action_counts_one_subagent_singular(self):

        self.assertEqual(
            catalog.notice_action(render_facts.Notice(done_n=1, live_n=2)),
            "1 SUBAGENT FINISHED",
        )

    def test_notice_action_counts_many_subagents_plural(self):
        self.assertEqual(
            catalog.notice_action(render_facts.Notice(done_n=3, live_n=2)),
            "3 SUBAGENTS FINISHED",
        )

    def test_an_absent_notice_still_yields_a_sentence(self):

        self.assertEqual(catalog.notice_action(None), "0 SUBAGENTS FINISHED")

    def test_action_for_is_the_rows_own_action_for_every_declared_status(self):
        for name, spec in catalog.STATUSES.items():
            if name == "Notice":
                continue
            self.assertEqual(catalog.action_for(name), spec.action, name)

    def test_action_for_reads_the_notice_only_on_the_notice_row(self):
        notice = render_facts.Notice(done_n=3, live_n=2)
        self.assertEqual(
            catalog.action_for("Notice", notice=notice), "3 SUBAGENTS FINISHED"
        )
        self.assertEqual(
            catalog.action_for("Healthy", notice=notice),
            catalog.STATUSES["Healthy"].action,
        )

    def test_a_cold_start_reads_its_action_off_its_own_row(self):

        self.assertEqual(
            catalog.action_for(catalog.WARMING_STATUS), catalog.WARMING_ACTION
        )
        self.assertEqual(catalog.dominant_status(facts()), catalog.WARMING_STATUS)

    def test_action_for_falls_back_to_the_sanitized_status(self):
        self.assertEqual(catalog.action_for("Bogus status"), "BOGUS STATUS")
        self.assertEqual(catalog.action_for("all\x1b[31m done"), "ALL[31M DONE")


class TestDetailBuilders(unittest.TestCase):
    """Row 1, traced from render.py:665-727."""

    def detail(self, status, **over):
        return catalog.STATUSES[status].detail(facts(**over))

    def test_healthy_shows_context_then_working(self):

        self.assertEqual(
            self.detail("Healthy", tokens=50_000, pct=24, agents_live=5),
            (
                render_facts.Instrument("ctx_load", "50K"),
                render_facts.Instrument("window", "24%"),
                render_facts.Instrument("working", "5"),
            ),
        )

    def test_working_row_names_one_foreign_child_model(self):

        self.assertEqual(
            self.detail(
                "Healthy",
                agents_live=2,
                agents_models=("claude-opus-5",),
                identity=render_facts.Identity(model_id="claude-fable-5"),
            ),
            (
                render_facts.Instrument("working", "2"),
                render_facts.Instrument("models", "⇄ opus-5"),
            ),
        )

    def test_working_row_counts_mixed_child_models(self):

        self.assertEqual(
            self.detail(
                "Healthy",
                agents_live=4,
                agents_models=("claude-opus-5", "claude-sonnet-5"),
                identity=render_facts.Identity(model_id="claude-fable-5"),
            ),
            (
                render_facts.Instrument("working", "4"),
                render_facts.Instrument("models", "⇄ 2 models"),
            ),
        )

    def test_working_row_stays_silent_when_children_match_the_session(self):
        self.assertEqual(
            self.detail(
                "Healthy",
                agents_live=3,
                agents_models=("claude-fable-5",),
                identity=render_facts.Identity(model_id="claude-fable-5"),
            ),
            (render_facts.Instrument("working", "3"),),
        )

    def test_working_row_counts_context_window_variants_as_distinct(self):

        self.assertEqual(
            self.detail(
                "Healthy",
                agents_live=2,
                agents_models=("claude-opus-5", "claude-opus-5[1m]"),
                identity=render_facts.Identity(model_id="claude-opus-5"),
            ),
            (
                render_facts.Instrument("working", "2"),
                render_facts.Instrument("models", "⇄ 2 models"),
            ),
        )

    def test_unknown_child_models_claim_nothing(self):

        self.assertEqual(
            self.detail(
                "Healthy",
                agents_live=2,
                agents_models=(),
                identity=render_facts.Identity(model_id="claude-fable-5"),
            ),
            (render_facts.Instrument("working", "2"),),
        )

    def test_every_tier_shares_the_calm_row(self):
        for tier in TIERS:
            self.assertEqual(
                self.detail(tier, tokens=160_000, pct=24, agents_live=1),
                (
                    render_facts.Instrument("ctx_load", "160K"),
                    render_facts.Instrument("window", "24%"),
                    render_facts.Instrument("working", "1"),
                ),
                tier,
            )

    def test_hidden_context_and_hidden_agents_empty_the_calm_row(self):

        self.assertEqual(
            self.detail(
                "Healthy",
                tokens=50_000,
                pct=24,
                agents_live=5,
                show_context=False,
                show_agents=False,
            ),
            (),
        )

    def test_a_zero_percentage_still_shows_the_window(self):

        self.assertEqual(
            self.detail("Healthy", pct=0),
            (render_facts.Instrument("window", "0%"),),
        )

    def test_stale_health_data_suppresses_context_but_keeps_working(self):

        self.assertEqual(
            self.detail(
                "Health data stale",
                parse_degraded=True,
                is_live=False,
                tokens=50_000,
                pct=24,
                agents_live=2,
            ),
            (render_facts.Instrument("working", "2"),),
        )

    def test_limited_health_data_keeps_live_context(self):
        self.assertEqual(
            self.detail(
                "Health data limited",
                parse_degraded=True,
                is_live=True,
                tokens=50_000,
            ),
            (render_facts.Instrument("ctx_load", "50K"),),
        )

    def test_agent_edit_loop_has_no_detail_of_its_own(self):
        self.assertEqual(
            self.detail("Agent edit loop", tokens=50_000, agents_live=1),
            (
                render_facts.Instrument("ctx_load", "50K"),
                render_facts.Instrument("working", "1"),
            ),
        )

    def test_edit_loop_names_the_file_with_its_attempt_count(self):

        self.assertEqual(
            self.detail(
                "Edit loop",
                blind_loop={"state": "red", "base": "config.json", "count": 3},
                tokens=160_000,
                pct=24,
            ),
            (
                render_facts.Instrument("file", "config.json ×3"),
                render_facts.Instrument("ctx_load", "160K"),
                render_facts.Instrument("window", "24%"),
            ),
        )

    def test_edit_loop_keeps_the_count_when_it_truncates_the_path(self):

        self.assertEqual(
            self.detail(
                "Edit loop",
                blind_loop={
                    "base": "packages/a-very-long-component/config.json",
                    "count": 12,
                },
            ),
            (render_facts.Instrument("file", "packages/a-very-lon… ×12"),),
        )

    def test_edit_loop_truncation_counts_wide_characters(self):
        detail = self.detail(
            "Edit loop",
            blind_loop={"base": "" + "🔥" * 20 + ".json", "count": 3},
        )
        self.assertEqual(detail, (render_facts.Instrument("file", "🔥" * 10 + "… ×3"),))
        self.assertEqual(render.vis_len(detail[0].text), 24)

    def test_edit_loop_without_a_count_shows_the_bare_name(self):
        self.assertEqual(
            self.detail("Edit loop", blind_loop={"base": "x"}),
            (render_facts.Instrument("file", "x"),),
        )

    def test_edit_loop_with_no_chip_shows_no_file(self):

        self.assertEqual(self.detail("Edit loop", blind_loop={}), ())

    def test_truncated_and_refused_state_the_symptom(self):

        self.assertEqual(
            self.detail("Response truncated"),
            (render_facts.Instrument("plain", "last response cut off"),),
        )
        self.assertEqual(
            self.detail("Response refused"),
            (render_facts.Instrument("plain", "last response refused"),),
        )

    def test_cache_problem_reports_the_reuse_drop(self):

        self.assertEqual(
            self.detail("Cache problem", cache_health={"show": True, "hit_drop": 23.4}),
            (render_facts.Instrument("plain", "reuse fell 23%"),),
        )

    def test_cache_problem_without_a_number_says_nothing_extra(self):
        for health in ({}, {"show": True}, {"hit_drop": None}, {"hit_drop": "x"}):
            self.assertEqual(self.detail("Cache problem", cache_health=health), ())

    def test_many_agent_runs_leads_with_the_burn_and_ends_with_context(self):

        self.assertEqual(
            self.detail(
                "Many agent runs",
                agents_total=125,
                agents_live=1,
                agents_tokens=12_900_000,
                tokens=50_000,
                pct=26,
            ),
            (
                render_facts.Instrument("runs", "125"),
                render_facts.Instrument("tokens", "12.9M"),
                render_facts.Instrument("working", "1"),
                render_facts.Instrument("ctx_load", "50K"),
                render_facts.Instrument("window", "26%"),
            ),
        )

    def test_an_incomplete_token_total_is_marked_at_least(self):

        self.assertEqual(
            self.detail(
                "Many agent runs",
                agents_total=50,
                agents_tokens=2_200_000,
                agents_tokens_complete=False,
            ),
            (
                render_facts.Instrument("runs", "50"),
                render_facts.Instrument("tokens", "≥2.2M"),
            ),
        )

    def test_a_missing_token_total_reports_why(self):

        self.assertEqual(
            self.detail(
                "Many agent runs", agents_total=50, agents_tokens_complete=False
            ),
            (
                render_facts.Instrument("runs", "50"),
                render_facts.Instrument("tokens_status", "warming"),
            ),
        )
        self.assertEqual(
            self.detail(
                "Many agent runs",
                agents_total=50,
                agents_tokens_complete=False,
                agents_tokens_warming=False,
            ),
            (
                render_facts.Instrument("runs", "50"),
                render_facts.Instrument("tokens_status", "incomplete"),
            ),
        )

    def test_a_complete_total_is_never_called_warming(self):

        self.assertEqual(
            self.detail(
                "Many agent runs",
                agents_total=50,
                agents_tokens_complete=True,
                agents_tokens_warming=True,
            ),
            (render_facts.Instrument("runs", "50"),),
        )

    def test_notice_shows_only_the_working_count_it_parsed(self):

        self.assertEqual(
            self.detail(
                "Notice",
                notice=render_facts.Notice(done_n=3, live_n=2),
                tokens=50_000,
                pct=24,
            ),
            (render_facts.Instrument("working", "2"),),
        )

    def test_a_notice_without_a_working_count_shows_nothing(self):
        self.assertEqual(
            self.detail("Notice", notice=render_facts.Notice(done_n=3)), ()
        )

    def test_the_standby_rows_carry_no_instruments(self):
        self.assertEqual(self.detail("Compact ack", tokens=50_000), ())


class TestReasonBuilders(unittest.TestCase):
    """Row 2, traced from render.py:736-787."""

    def reason(self, status, **over):
        return catalog.STATUSES[status].reason(facts(**over))

    def test_unreadable_history_says_so(self):

        for status in ("Health data limited", "Health data stale"):
            self.assertEqual(
                self.reason(status, parse_degraded=True).text,
                "Session history could not be checked",
                status,
            )

    def test_edit_loop_names_the_whole_file(self):

        self.assertEqual(
            self.reason(
                "Edit loop",
                blind_loop={
                    "base": "packages/a-very-long-component/x.json",
                    "count": 12,
                },
            ).text,
            "File: packages/a-very-long-component/x.json",
        )

    def test_edit_loop_without_a_file_has_no_reason(self):
        self.assertIsNone(self.reason("Edit loop", blind_loop={}))

    def test_agent_edit_loop_explains_the_child(self):

        self.assertEqual(
            self.reason("Agent edit loop").text, "A child agent retried a failed edit"
        )

    def test_many_agent_runs_counts_subagents_and_tokens(self):

        self.assertEqual(
            self.reason(
                "Many agent runs", agents_total=125, agents_tokens=13_665_260
            ).text,
            "125 subagents; 13.7M tokens",
        )
        self.assertEqual(
            self.reason(
                "Many agent runs",
                agents_total=125,
                agents_tokens=13_665_260,
                agents_tokens_complete=False,
            ).text,
            "125 subagents; at least 13.7M tokens",
        )

    def test_many_agent_runs_says_why_a_token_total_is_missing(self):
        self.assertEqual(
            self.reason(
                "Many agent runs", agents_total=125, agents_tokens_complete=False
            ).text,
            "125 subagents; token total still loading",
        )
        self.assertEqual(
            self.reason(
                "Many agent runs",
                agents_total=125,
                agents_tokens_complete=False,
                agents_tokens_warming=False,
            ).text,
            "125 subagents; token total incomplete",
        )
        self.assertEqual(
            self.reason("Many agent runs", agents_total=125).text, "125 subagents"
        )

    def test_truncated_and_refused_state_the_symptom(self):

        self.assertEqual(
            self.reason("Response truncated").text, "The last response was cut off"
        )
        self.assertEqual(
            self.reason("Response refused").text, "The last response was refused"
        )

    def test_cache_problem_quantifies_the_drop_when_it_can(self):

        self.assertEqual(
            self.reason("Cache problem", cache_health={"hit_drop": 23.4}).text,
            "Context reuse fell 23%",
        )
        self.assertEqual(
            self.reason("Cache problem", cache_health={}).text,
            "Claude is re-reading more context",
        )

    def test_a_notice_prefers_the_live_agent_count(self):

        self.assertEqual(
            self.reason(
                "Notice", notice=render_facts.Notice(done_n=3, live_n=2), agents_live=4
            ).text,
            "4 still working",
        )

    def test_a_notice_falls_back_to_its_own_prose(self):
        self.assertEqual(
            self.reason(
                "Notice",
                notice=render_facts.Notice(done_n=3, live_n=2),
                show_agents=False,
            ).text,
            "2 still working",
        )
        self.assertIsNone(self.reason("Notice", notice=render_facts.Notice(done_n=3)))

    def test_the_calm_row_stands_by_before_any_context_arrives(self):

        self.assertEqual(self.reason("Healthy", tokens=0).text, catalog.STANDBY_REASON)
        self.assertEqual(self.reason("Warming").text, catalog.STANDBY_REASON)

    def test_the_calm_row_reports_the_context_load_in_the_tier_colour(self):

        reason = self.reason("Healthy", tier="strong", tokens=50_000, pct=24)
        self.assertEqual(reason.text, "Context load: 50K")
        self.assertEqual(reason.label, "Context load: ")
        self.assertEqual(reason.value, "50K")
        self.assertTrue(reason.use_context_color)
        self.assertNotIn("\033", reason.text)

    def test_every_firing_tier_reports_its_context_load(self):
        for tier in TIERS:
            self.assertEqual(
                self.reason(tier, tier=tier, tokens=128_000).text,
                "Context load: 128K",
                tier,
            )

    def test_the_calm_row_falls_back_to_the_window_percentage(self):

        reason = self.reason("Healthy", tier="strong", tokens=0, pct=18)
        self.assertEqual(reason.text, "Window used: 18%")
        self.assertFalse(reason.use_context_color)

    def test_the_calm_row_is_silent_with_no_context_to_report(self):
        self.assertIsNone(self.reason("Healthy", tier="strong", show_context=False))
        self.assertIsNone(self.reason("Healthy", tier="strong"))

    def test_untrusted_context_is_not_reported(self):

        self.assertIsNone(
            self.reason(
                "Healthy",
                tier="strong",
                tokens=50_000,
                parse_degraded=True,
                is_live=False,
            )
        )

    def test_the_compact_row_explains_the_wait(self):
        self.assertEqual(self.reason("Compact ack").text, catalog.COMPACT_ACK_DETAIL)

    def test_no_reason_ever_carries_an_escape_sequence(self):
        rich = facts(
            blind_loop={"base": "config.json", "count": 3},
            cache_health={"hit_drop": 23.4},
            tokens=50_000,
            pct=24,
            tier="strong",
            notice=render_facts.Notice(done_n=3, live_n=2),
            agents_total=125,
            agents_live=2,
            agents_tokens=2_200_000,
        )
        for name, spec in catalog.STATUSES.items():
            reason = spec.reason(rich)
            if reason is not None:
                self.assertNotIn("\033", reason.text, name)


if __name__ == "__main__":
    unittest.main()
