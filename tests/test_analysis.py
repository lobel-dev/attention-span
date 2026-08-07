import dataclasses
import unittest

import attention_span.analysis as analysis
from attention_span import agent_health


class TestAnalysisValueTypes(unittest.TestCase):
    RECORD_NAMES = (
        "EditLoop",
        "UsageTotals",
        "Repetition",
        "Perseveration",
        "ParseHealth",
        "Analysis",
    )

    def test_records_are_canonical_frozen_dataclasses(self):
        for name in self.RECORD_NAMES:
            with self.subTest(name=name):
                record = getattr(analysis, name)
                value = record()
                self.assertTrue(dataclasses.is_dataclass(record))
                self.assertTrue(record.__dataclass_params__.frozen)
                first_field = dataclasses.fields(record)[0].name
                with self.assertRaises(dataclasses.FrozenInstanceError):
                    setattr(value, first_field, getattr(value, first_field))

    def test_agent_health_reexports_the_canonical_record_identities(self):
        for name in self.RECORD_NAMES:
            with self.subTest(name=name):
                self.assertIs(getattr(agent_health, name), getattr(analysis, name))

    def test_leaf_record_field_order_and_defaults_are_stable(self):
        expected = {
            "EditLoop": (("file", "count"), (None, 0)),
            "UsageTotals": (
                ("input", "cache_creation", "cache_read", "output", "api_calls"),
                (0, 0, 0, 0, 0),
            ),
            "Repetition": (
                ("score", "worst_target", "count"),
                (0, None, 0),
            ),
            "Perseveration": (
                ("score", "worst_target", "count"),
                (0, None, 0),
            ),
            "ParseHealth": (
                (
                    "decode_failures",
                    "assistant_lines",
                    "usage_seen",
                    "tooluse_seen",
                    "parse_aborted",
                    "schema_canary",
                    "degraded",
                ),
                (0, 0, False, False, False, False, False),
            ),
        }

        for name, (field_names, defaults) in expected.items():
            with self.subTest(name=name):
                record = getattr(analysis, name)
                value = record()
                self.assertEqual(
                    tuple(field.name for field in dataclasses.fields(record)),
                    field_names,
                )
                self.assertEqual(tuple(vars(value).values()), defaults)

    def test_analysis_field_order_and_defaults_are_stable(self):
        value = analysis.Analysis()

        self.assertEqual(
            tuple(field.name for field in dataclasses.fields(analysis.Analysis)),
            (
                "reads",
                "edits",
                "total_reads",
                "total_edits",
                "r2e",
                "base_tier",
                "window_used",
                "insufficient",
                "failed_edit_loop",
                "cache_health",
                "repetition",
                "perseveration",
                "parse_health",
                "last_model",
                "last_stop_reason",
                "compaction_pending",
                "turns",
                "context_tokens",
                "usage_totals",
                "max_error_bytes",
                "task_notifications",
                "thinking_turns",
                "assistant_turns",
                "trailing_no_thinking",
            ),
        )
        self.assertEqual(
            vars(value),
            {
                "reads": 0,
                "edits": 0,
                "total_reads": 0,
                "total_edits": 0,
                "r2e": float("inf"),
                "base_tier": "green",
                "window_used": 0,
                "insufficient": True,
                "failed_edit_loop": analysis.EditLoop(),
                "cache_health": {},
                "repetition": analysis.Repetition(),
                "perseveration": analysis.Perseveration(),
                "parse_health": analysis.ParseHealth(),
                "last_model": None,
                "last_stop_reason": None,
                "compaction_pending": False,
                "turns": 0,
                "context_tokens": 0,
                "usage_totals": analysis.UsageTotals(),
                "max_error_bytes": 0,
                "task_notifications": {},
                "thinking_turns": 0,
                "assistant_turns": 0,
                "trailing_no_thinking": 0,
            },
        )

    def test_mapping_defaults_are_not_shared(self):
        first = analysis.Analysis()
        second = analysis.Analysis()

        for name in ("cache_health", "task_notifications"):
            with self.subTest(name=name):
                self.assertEqual(getattr(first, name), {})
                self.assertEqual(getattr(second, name), {})
                self.assertIsNot(getattr(first, name), getattr(second, name))

    def test_debug_asdict_still_recurses_through_nested_records(self):
        value = analysis.Analysis(
            failed_edit_loop=analysis.EditLoop(file="/repo/a.py", count=2),
            usage_totals=analysis.UsageTotals(input=3, output=5, api_calls=1),
        )

        debug_value = dataclasses.asdict(value)

        self.assertEqual(
            debug_value["failed_edit_loop"],
            {"file": "/repo/a.py", "count": 2},
        )
        self.assertEqual(
            debug_value["usage_totals"],
            {
                "input": 3,
                "cache_creation": 0,
                "cache_read": 0,
                "output": 5,
                "api_calls": 1,
            },
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
