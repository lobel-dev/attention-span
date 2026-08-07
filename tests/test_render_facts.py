import dataclasses
import unittest

from attention_span import render_facts, status_catalog


class RenderFactsRecordsTest(unittest.TestCase):
    def test_catalog_does_not_reexport_fact_records_or_text_utilities(self):
        for name in (
            "Instrument",
            "Reason",
            "ContextLoad",
            "Notice",
            "Identity",
            "RenderFacts",
            "compact_magnitude",
            "display_tokens",
            "vis_len",
            "right_ellipsis",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(status_catalog, name))

    def test_records_are_frozen_and_default_complete(self):
        facts = render_facts.RenderFacts()
        self.assertIsInstance(facts, render_facts.RenderFacts)
        self.assertEqual(facts.context, render_facts.ContextLoad())
        self.assertEqual(facts.identity, render_facts.Identity())
        self.assertEqual(facts.notice, None)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            facts.show_context = False
        with self.assertRaises(dataclasses.FrozenInstanceError):
            render_facts.ContextLoad().pct = 5
        with self.assertRaises(dataclasses.FrozenInstanceError):
            render_facts.Notice().done_n = 5
        with self.assertRaises(dataclasses.FrozenInstanceError):
            render_facts.Identity().cwd = "/tmp"
        with self.assertRaises(dataclasses.FrozenInstanceError):
            render_facts.Instrument("plain", "x").text = "y"
        with self.assertRaises(dataclasses.FrozenInstanceError):
            render_facts.Reason("x").text = "y"

    def test_notice_truth_tracks_finished_count_only(self):
        self.assertEqual(
            [f.name for f in dataclasses.fields(render_facts.Notice)],
            ["done_n", "live_n"],
        )
        self.assertTrue(render_facts.Notice(done_n=1, live_n=0))
        self.assertFalse(render_facts.Notice(done_n=0, live_n=3))

    def test_context_load_field_order_is_stable(self):
        self.assertEqual(
            [f.name for f in dataclasses.fields(render_facts.ContextLoad)],
            ["tier", "tokens", "pct", "is_live"],
        )

    def test_render_facts_field_order_is_stable(self):
        self.assertEqual(
            [f.name for f in dataclasses.fields(render_facts.RenderFacts)],
            [
                "context",
                "identity",
                "notice",
                "blind_loop",
                "subagent_blind_loop",
                "cache_health",
                "parse_degraded",
                "last_stop_reason",
                "show_context",
                "show_agents",
                "agents_total",
                "agents_live",
                "agents_tokens",
                "agents_tokens_complete",
                "agents_tokens_warming",
                "agents_models",
            ],
        )


if __name__ == "__main__":
    unittest.main()
