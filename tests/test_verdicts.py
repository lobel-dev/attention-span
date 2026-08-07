"""Pure verdict and display-label behavior."""

import os
import subprocess
import sys
import unittest
from pathlib import Path

from bands import (
    LOAD_200K_DEAD,
    LOAD_200K_DEGRADING,
    LOAD_200K_FAILING,
    LOAD_200K_FUNCTIONAL,
    LOAD_DEAD,
    LOAD_DEAD_S,
    LOAD_DEGRADING,
    LOAD_DEGRADING_S,
    LOAD_FAILING,
    LOAD_FAILING_S,
    LOAD_FUNCTIONAL,
    LOAD_FUNCTIONAL_S,
    LOAD_PEAK,
    LOAD_PEAK_S,
    LOAD_STRONG,
    LOAD_STRONG_S,
)

from attention_span import agent_health, health_config, verdicts
from attention_span.analysis import Analysis, EditLoop

ROOT = Path(__file__).resolve().parents[1]
BANDS = health_config.ENGINE_BANDS


class TestCanonicalVerdicts(unittest.TestCase):
    def test_agent_health_reexports_the_canonical_functions(self):
        for name in (
            "blind_loop_alert",
            "r2e_posture",
            "context_verdict",
            "window_class",
            "model_label",
        ):
            with self.subTest(name=name):
                self.assertIs(getattr(agent_health, name), getattr(verdicts, name))

    def test_threshold_defaults_keep_the_engine_identity(self):
        self.assertIs(verdicts.DEFAULTS, health_config.ENGINE_DEFAULTS)
        self.assertIs(agent_health.DEFAULTS, verdicts.DEFAULTS)

    def test_fresh_import_has_no_agent_health_cycle(self):
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import attention_span.verdicts as verdicts; "
                    "assert verdicts.__name__ == 'attention_span.verdicts'; "
                    "assert 'attention_span.agent_health' not in sys.modules"
                ),
            ],
            cwd=ROOT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)


class TestBlindLoopAlert(unittest.TestCase):
    def test_absent_loop_has_no_alert(self):
        self.assertIsNone(verdicts.blind_loop_alert(Analysis()))

    def test_alert_preserves_raw_file_and_sanitizes_the_display_basename(self):
        analysis = Analysis(
            insufficient=True,
            edits=2,
            failed_edit_loop=EditLoop(file="/repo/a\x1b[31m.py", count=2),
        )

        self.assertEqual(
            verdicts.blind_loop_alert(analysis),
            {
                "state": "red",
                "file": "/repo/a\x1b[31m.py",
                "base": "a[31m.py",
                "count": 2,
                "chip": "EDIT LOOP: a[31m.py x2",
            },
        )


class TestR2EPosture(unittest.TestCase):
    def test_insufficient_analysis_leaves_the_posture_off(self):
        self.assertEqual(verdicts.r2e_posture(Analysis()), ("green", ""))

    def test_red_with_enough_edits_is_close_watch(self):
        analysis = Analysis(
            insufficient=False,
            base_tier="red",
            edits=BANDS.MIN_EDITS_FOR_RED,
        )
        self.assertEqual(verdicts.r2e_posture(analysis), ("yellow", "Close Watch"))

    def test_red_below_the_edit_floor_and_yellow_are_spot_check(self):
        cases = (
            Analysis(insufficient=False, base_tier="red", edits=0),
            Analysis(insufficient=False, base_tier="yellow", edits=99),
        )
        for analysis in cases:
            with self.subTest(analysis=analysis):
                self.assertEqual(
                    verdicts.r2e_posture(analysis), ("yellow", "Spot Check")
                )

    def test_green_is_monitor_normally(self):
        analysis = Analysis(insufficient=False, base_tier="green")
        self.assertEqual(verdicts.r2e_posture(analysis), ("green", "Monitor Normally"))


class TestContextVerdict(unittest.TestCase):
    def test_every_tier_is_reachable_from_its_own_sample(self):
        cases = (
            (LOAD_PEAK, "peak", LOAD_PEAK_S),
            (LOAD_STRONG, "strong", LOAD_STRONG_S),
            (LOAD_FUNCTIONAL, "functional", LOAD_FUNCTIONAL_S),
            (LOAD_DEGRADING, "degrading", LOAD_DEGRADING_S),
            (LOAD_FAILING, "failing", LOAD_FAILING_S),
            (LOAD_DEAD, "dead", LOAD_DEAD_S),
        )
        for tokens, tier, chip in cases:
            with self.subTest(tier=tier):
                self.assertEqual(verdicts.context_verdict(tokens), (tier, chip))

    def test_each_boundary_opens_its_tier_and_not_a_token_earlier(self):
        for tier, field in health_config.CONTEXT_TIER_BANDS.items():
            band = getattr(BANDS, field)
            with self.subTest(tier=tier):
                self.assertEqual(verdicts.context_verdict(band)[0], tier)
                self.assertNotEqual(verdicts.context_verdict(band - 1)[0], tier)

    def test_unknown_suppressed(self):
        self.assertEqual(verdicts.context_verdict(0), (None, ""))
        self.assertEqual(verdicts.context_verdict(None), (None, ""))

    def test_window_fill_cannot_change_the_tier_at_any_percentage(self):

        baseline = verdicts.context_verdict(LOAD_FUNCTIONAL)[0]
        for pct in (0, 30, 65, 81, 99, 100):
            with self.subTest(ctx_pct=pct):
                self.assertEqual(
                    verdicts.context_verdict(LOAD_FUNCTIONAL, ctx_pct=pct)[0],
                    baseline,
                )

    def test_pct_chip_appended(self):
        _, chip = verdicts.context_verdict(250_000, ctx_pct=30)
        self.assertIn("250K", chip)
        self.assertIn("30%", chip)

    def test_custom_thresholds_steer_the_whole_ladder(self):
        th = {
            "CTX_TIER_STRONG": 10_000,
            "CTX_TIER_FUNCTIONAL": 80_000,
            "CTX_TIER_DEGRADING": 160_000,
            "CTX_TIER_FAILING": 320_000,
            "CTX_TIER_DEAD": 640_000,
        }
        cases = (
            (5_000, "peak"),
            (70_000, "strong"),
            (90_000, "functional"),
            (200_000, "degrading"),
            (400_000, "failing"),
            (700_000, "dead"),
        )
        for tokens, expected in cases:
            with self.subTest(tokens=tokens):
                self.assertEqual(verdicts.context_verdict(tokens, th=th)[0], expected)

    def test_the_200k_class_grades_on_its_own_ladder(self):

        for tokens, tier_200k in (
            (LOAD_200K_FUNCTIONAL, "functional"),
            (LOAD_200K_DEGRADING, "degrading"),
            (LOAD_200K_FAILING, "failing"),
            (LOAD_200K_DEAD, "dead"),
        ):
            with self.subTest(tokens=tokens):
                self.assertEqual(
                    verdicts.context_verdict(tokens, window_class="200k")[0],
                    tier_200k,
                )
                base = verdicts.context_verdict(tokens)[0]
                for cls in ("1m", None, "bogus"):
                    self.assertEqual(
                        verdicts.context_verdict(tokens, window_class=cls)[0],
                        base,
                        cls,
                    )

    def test_classes_agree_outside_the_divergence_range(self):

        for tokens in (LOAD_PEAK, BANDS.CTX_TIER_FUNCTIONAL_200K - 1, LOAD_DEAD):
            with self.subTest(tokens=tokens):
                self.assertEqual(
                    verdicts.context_verdict(tokens, window_class="200k")[0],
                    verdicts.context_verdict(tokens)[0],
                )

    def test_th_override_of_a_class_field_moves_that_ladder_only(self):
        th = {"CTX_TIER_DEGRADING_200K": LOAD_200K_DEGRADING + 1}
        self.assertEqual(
            verdicts.context_verdict(LOAD_200K_DEGRADING, th=th, window_class="200k")[
                0
            ],
            "functional",
        )
        self.assertEqual(
            verdicts.context_verdict(LOAD_200K_DEGRADING, th=th)[0],
            verdicts.context_verdict(LOAD_200K_DEGRADING)[0],
        )


class TestModelLabel(unittest.TestCase):
    def test_version_extracted(self):
        self.assertEqual(verdicts.model_label("claude-sonnet-4-6"), "sonnet-4.6")
        self.assertEqual(verdicts.model_label("claude-opus-4-8"), "opus-4.8")
        self.assertEqual(verdicts.model_label("claude-fable-5"), "fable-5")

    def test_date_tag_dropped(self):
        self.assertEqual(verdicts.model_label("claude-haiku-4-5-20251001"), "haiku-4.5")

    def test_context_variant_surfaced(self):
        self.assertEqual(verdicts.model_label("claude-opus-4-8[1m]"), "opus-4.8·1M")

    def test_empty_and_none(self):
        self.assertEqual(verdicts.model_label(""), "")
        self.assertEqual(verdicts.model_label(None), "")

    def test_version_bump_gives_distinct_labels(self):
        self.assertNotEqual(
            verdicts.model_label("claude-opus-4-7"),
            verdicts.model_label("claude-opus-4-8"),
        )

    def test_unknown_model_id_is_sanitized(self):
        result = verdicts.model_label("custom-\x1b[31mmodel")
        self.assertNotIn("\x1b", result)


class TestWindowClass(unittest.TestCase):
    """Advertised-window classification: exact-match, any doubt -> None."""

    def test_every_taxonomy_size_classifies(self):
        for size, cls in health_config.WINDOW_CLASSES.items():
            self.assertEqual(verdicts.window_class(size), cls)

    def test_an_integral_float_size_still_classifies(self):
        for size, cls in health_config.WINDOW_CLASSES.items():
            self.assertEqual(verdicts.window_class(float(size)), cls)

    def test_unknown_sizes_and_garbage_are_none(self):
        for value in (None, 0, -1, 500_000, "1m", float("nan"), float("inf"), True):
            self.assertIsNone(verdicts.window_class(value), repr(value))

    def test_a_non_integral_size_is_unknown(self):
        size = next(iter(health_config.WINDOW_CLASSES))
        self.assertIsNone(verdicts.window_class(size + 0.5))
