"""Unit tests for agent_health — stdlib unittest, no third-party deps.

Run: uv run pytest tests/test_agent_health.py   (from repo root)
"""

import contextlib
import dataclasses
import io
import json
import os
import unittest

from builders import (
    GENUINE,
    HOOKDENY,
    analyze,
    asst,
    bash_tu,
    edit_tu,
    iso_ts,
    read_tu,
    results,
    synthetic_api_error,
    user_str,
    write_lines,
)
from builders import usage as mk_usage

from attention_span import agent_health as ah
from attention_span import health_config

BANDS = health_config.ENGINE_BANDS


class TestAnalyzeTranscript(unittest.TestCase):
    def test_reads_only_is_green_infinite(self):
        objs = [asst([read_tu(i, f"/repo/f{i}.py")]) for i in range(6)]
        a = analyze(objs)
        self.assertEqual(a.edits, 0)
        self.assertEqual(a.r2e, float("inf"))
        self.assertEqual(a.base_tier, "green")
        self.assertFalse(a.insufficient)
        self.assertIsNone(ah.blind_loop_alert(a))
        state, chip = ah.r2e_posture(a)
        self.assertEqual(state, "green")
        self.assertEqual(chip, "Monitor Normally")

    def test_insufficient_suppresses_chip(self):
        objs = [asst([read_tu(0)]), asst([read_tu(1)]), asst([edit_tu(0)])]
        a = analyze(objs)
        self.assertTrue(a.insufficient)
        self.assertIsNone(ah.blind_loop_alert(a))
        state, chip = ah.r2e_posture(a)
        self.assertEqual(state, "green")
        self.assertEqual(chip, "")

    def test_low_r2e_is_yellow_close_watch_never_red(self):
        objs = [asst([read_tu(0, "/repo/r.py")])]
        objs += [
            asst([(f"e{i}", "Edit", {"file_path": f"/repo/f{i}.py"})]) for i in range(8)
        ]
        a = analyze(objs)
        self.assertEqual(a.edits, 8)
        self.assertEqual(a.reads, 1)
        self.assertEqual(a.base_tier, "red")
        self.assertEqual(a.failed_edit_loop.count, 0)
        self.assertIsNone(ah.blind_loop_alert(a))
        state, chip = ah.r2e_posture(a)
        self.assertEqual(state, "yellow")
        self.assertEqual(chip, "Close Watch")

    def test_sufficient_red_window_always_has_min_edits(self):
        objs = [
            asst([(f"e{i}", "Edit", {"file_path": f"/repo/f{i}.py"})]) for i in range(5)
        ]
        a = analyze(objs)
        self.assertEqual(a.base_tier, "red")
        self.assertGreaterEqual(a.edits, BANDS.MIN_EDITS_FOR_RED)

    def test_balanced_r2e_is_monitor_normally(self):

        objs = [asst([read_tu(i, f"/repo/r{i}.py")]) for i in range(5)]
        objs += [
            asst([(f"e{i}", "Edit", {"file_path": f"/repo/e{i}.py"})]) for i in range(5)
        ]
        a = analyze(objs)
        self.assertEqual(a.base_tier, "green")
        self.assertEqual(ah.r2e_posture(a), ("green", "Monitor Normally"))

    def test_mild_read_heavy_r2e_is_monitor_normally(self):
        objs = [asst([read_tu(i, f"/repo/r{i}.py")]) for i in range(6)]
        objs += [
            asst([(f"e{i}", "Edit", {"file_path": f"/repo/e{i}.py"})]) for i in range(4)
        ]
        a = analyze(objs)
        self.assertEqual(a.base_tier, "green")
        self.assertEqual(ah.r2e_posture(a), ("green", "Monitor Normally"))

    def test_mild_edit_skew_is_spot_check(self):
        objs = [asst([read_tu(i, f"/repo/r{i}.py")]) for i in range(4)]
        objs += [
            asst([(f"e{i}", "Edit", {"file_path": f"/repo/e{i}.py"})]) for i in range(6)
        ]
        a = analyze(objs)
        self.assertEqual(a.base_tier, "yellow")
        self.assertEqual(ah.r2e_posture(a), ("yellow", "Spot Check"))

    def test_r2e_posture_never_red_and_context_independent(self):
        objs = [asst([read_tu(i, f"/repo/r{i}.py")]) for i in range(3)]
        objs += [
            asst([(f"e{i}", "Edit", {"file_path": f"/repo/f{i}.py"})]) for i in range(7)
        ]
        a = analyze(objs)
        self.assertEqual(a.base_tier, "red")
        self.assertEqual(ah.r2e_posture(a), ("yellow", "Close Watch"))
        for tokens in (600_000, 0):
            loaded = dataclasses.replace(a, context_tokens=tokens)
            self.assertEqual(ah.r2e_posture(loaded), ("yellow", "Close Watch"))
        self.assertIsNone(ah.blind_loop_alert(a))

    def test_genuine_failed_edit_loop_is_red_blind_loop(self):
        objs = [
            asst([read_tu(0, "/repo/a.py")]),
            asst([read_tu(1, "/repo/b.py")]),
            asst([("e0", "Edit", {"file_path": "/repo/config.json"})]),
            results([("e0", True, GENUINE)]),
            asst([("e1", "Edit", {"file_path": "/repo/config.json"})]),
            results([("e1", True, GENUINE)]),
            asst([("e2", "Edit", {"file_path": "/repo/config.json"})]),
        ]
        a = analyze(objs)
        self.assertEqual(a.failed_edit_loop.count, 2)
        self.assertTrue(a.failed_edit_loop.file.endswith("config.json"))
        bl = ah.blind_loop_alert(a)
        self.assertIsNotNone(bl)
        self.assertEqual(bl["state"], "red")
        self.assertIn("EDIT LOOP", bl["chip"])
        self.assertIn("config.json", bl["chip"])
        self.assertIn("x2", bl["chip"])

    def test_blind_loop_fires_even_when_insufficient(self):
        objs = [
            asst([("e0", "Edit", {"file_path": "/repo/cfg.json"})]),
            results([("e0", True, GENUINE)]),
            asst([("e1", "Edit", {"file_path": "/repo/cfg.json"})]),
        ]
        a = analyze(objs)
        self.assertTrue(a.insufficient)
        self.assertEqual(a.failed_edit_loop.count, 1)
        bl = ah.blind_loop_alert(a)
        self.assertIsNotNone(bl)
        self.assertEqual(bl["state"], "red")
        self.assertEqual(ah.r2e_posture(a), ("green", ""))

    def test_shell_reread_of_the_failed_file_breaks_the_loop(self):
        objs = [
            asst([("e0", "Edit", {"file_path": "/repo/config.json"})]),
            results([("e0", True, GENUINE)]),
            asst([("b0", "Bash", {"command": "cat /repo/config.json"})]),
            results([("b0", False, "{...}")]),
            asst([("e1", "Edit", {"file_path": "/repo/config.json"})]),
        ]
        a = analyze(objs)
        self.assertEqual(a.failed_edit_loop.count, 0)
        self.assertIsNone(ah.blind_loop_alert(a))

    def _shell_reread(self, command, cwd=None):
        return analyze(
            [
                asst(
                    [("e0", "Edit", {"file_path": "/repo/config.json"})],
                    cwd=cwd,
                ),
                results([("e0", True, GENUINE)]),
                asst([("b0", "Bash", {"command": command})], cwd=cwd),
                results([("b0", False, "{...}")]),
                asst([("e1", "Edit", {"file_path": "/repo/config.json"})], cwd=cwd),
            ]
        )

    def test_relative_shell_reread_anchored_to_cwd_breaks_the_loop(self):
        a = self._shell_reread("cat config.json", cwd="/repo")
        self.assertEqual(a.failed_edit_loop.count, 0)
        self.assertIsNone(ah.blind_loop_alert(a))

        self.assertEqual(a.repetition.score, 0)

    def test_relative_shell_reread_behind_a_cd_still_alarms(self):

        a = self._shell_reread("cd sub && cat config.json", cwd="/repo")
        self.assertEqual(a.failed_edit_loop.count, 1)
        self.assertIsNotNone(ah.blind_loop_alert(a))

    def test_relative_shell_reread_without_a_record_cwd_still_alarms(self):

        for cwd in (None, ""):
            with self.subTest(cwd=cwd):
                a = self._shell_reread("cat config.json", cwd=cwd)
                self.assertEqual(a.failed_edit_loop.count, 1)
                self.assertIsNotNone(ah.blind_loop_alert(a))

    def test_absolute_shell_reread_is_unaffected_by_the_record_cwd(self):
        for cwd in (None, "/repo", "/elsewhere"):
            with self.subTest(cwd=cwd):
                a = self._shell_reread("cat /repo/config.json", cwd=cwd)
                self.assertEqual(a.failed_edit_loop.count, 0)
                self.assertIsNone(ah.blind_loop_alert(a))

    def test_unrelated_shell_read_still_leaves_the_loop_alarming(self):

        for command in ("cat /repo/other.py", "grep -r needle"):
            with self.subTest(command=command):
                objs = [
                    asst([("e0", "Edit", {"file_path": "/repo/config.json"})]),
                    results([("e0", True, GENUINE)]),
                    asst([("b0", "Bash", {"command": command})]),
                    results([("b0", False, "hit")]),
                    asst([("e1", "Edit", {"file_path": "/repo/config.json"})]),
                ]
                a = analyze(objs)
                self.assertEqual(a.failed_edit_loop.count, 1)
                self.assertIsNotNone(ah.blind_loop_alert(a))

    def test_hook_deny_loop_is_not_red(self):

        objs = [
            asst([read_tu(0, "/repo/a.py")]),
            asst([read_tu(1, "/repo/b.py")]),
            asst([("e0", "Edit", {"file_path": "/repo/config.json"})]),
            results([("e0", True, HOOKDENY)]),
            asst([("e1", "Edit", {"file_path": "/repo/config.json"})]),
            results([("e1", True, HOOKDENY)]),
            asst([("e2", "Edit", {"file_path": "/repo/config.json"})]),
            results([("e2", True, HOOKDENY)]),
        ]
        a = analyze(objs)
        self.assertEqual(a.failed_edit_loop.count, 0)
        self.assertEqual(a.edits, 0)
        self.assertTrue(a.insufficient)
        self.assertIsNone(ah.blind_loop_alert(a))

        self.assertEqual(ah.r2e_posture(a), ("green", ""))

    def test_hook_deny_edits_excluded_from_window(self):

        objs = [asst([read_tu(i, f"/repo/r{i}.py")]) for i in range(3)]
        objs += [
            asst([(f"e{i}", "Edit", {"file_path": f"/repo/e{i}.py"})]) for i in range(2)
        ]
        for i in range(5):
            objs.append(asst([(f"d{i}", "Edit", {"file_path": f"/repo/d{i}.py"})]))
            objs.append(results([(f"d{i}", True, HOOKDENY)]))
        a = analyze(objs)
        self.assertEqual(a.reads, 3)
        self.assertEqual(a.edits, 2)
        self.assertEqual(a.total_edits, 2)
        self.assertEqual(a.base_tier, "green")

    def test_hook_deny_read_excluded_from_window(self):

        objs = [asst([("b0", "Bash", {"command": "ls -la"})])]
        objs.append(results([("b0", True, HOOKDENY)]))
        objs += [asst([read_tu(i, f"/repo/r{i}.py")]) for i in range(5)]
        a = analyze(objs)
        self.assertEqual(a.reads, 5)
        self.assertEqual(a.total_reads, 5)

    def test_inflight_edit_counts(self):

        objs = [asst([read_tu(i, f"/repo/r{i}.py")]) for i in range(4)]
        objs.append(asst([("e0", "Edit", {"file_path": "/repo/x.py"})]))
        a = analyze(objs)
        self.assertEqual(a.edits, 1)
        self.assertEqual(a.window_used, 5)

    def test_neutral_burst_does_not_advance_window(self):
        objs = [asst([read_tu(i, f"/repo/f{i}.py")]) for i in range(4)]
        objs += [asst([(f"n{i}", "TodoWrite", {"todos": []})]) for i in range(20)]
        a = analyze(objs)
        self.assertEqual(a.reads, 4)
        self.assertEqual(a.edits, 0)
        self.assertEqual(a.window_used, 4)
        self.assertTrue(a.insufficient)

    def test_window_is_last_n_events(self):
        objs = [asst([read_tu(i, f"/repo/r{i}.py")]) for i in range(6)]
        objs += [
            asst([(f"e{i}", "Edit", {"file_path": f"/repo/e{i}.py"})])
            for i in range(10)
        ]
        a = analyze(objs)
        self.assertEqual(a.edits, 10)
        self.assertEqual(a.reads, 0)
        self.assertEqual(a.r2e, 0.0)
        self.assertEqual(a.base_tier, "red")

    def test_threshold_window_applies_by_default(self):
        objs = [asst([read_tu(i, f"/repo/r{i}.py")]) for i in range(5)]
        objs += [asst([edit_tu(i, f"/repo/e{i}.py")]) for i in range(5)]
        path = write_lines(objs)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))

        a = ah.analyze_transcript(path, th={"WINDOW": 5})
        self.assertEqual(a.window_used, 5)
        self.assertEqual(a.reads, 0)
        self.assertEqual(a.edits, 5)

    def test_float_threshold_window_still_builds_the_rolling_window(self):

        objs = [asst([read_tu(i, f"/repo/r{i}.py")]) for i in range(5)]
        objs += [asst([edit_tu(i, f"/repo/e{i}.py")]) for i in range(5)]
        path = write_lines(objs)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))

        a = ah.analyze_transcript(path, th={"WINDOW": 5.0})
        self.assertEqual(a.window_used, 5)
        self.assertEqual(a.reads, 0)
        self.assertEqual(a.edits, 5)

    def test_explicit_window_overrides_threshold_window(self):
        objs = [asst([read_tu(i, f"/repo/r{i}.py")]) for i in range(5)]
        objs += [asst([edit_tu(i, f"/repo/e{i}.py")]) for i in range(5)]
        path = write_lines(objs)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))

        a = ah.analyze_transcript(path, window=10, th={"WINDOW": 5})
        self.assertEqual(a.window_used, 10)
        self.assertEqual(a.reads, 5)
        self.assertEqual(a.edits, 5)

    def test_subagent_turns_excluded(self):
        objs = [asst([read_tu(i, f"/repo/f{i}.py")]) for i in range(5)]
        side = asst(
            [(f"se{i}", "Edit", {"file_path": f"/repo/s{i}.py"}) for i in range(10)]
        )
        side["isSidechain"] = True
        objs.append(side)
        a = analyze(objs)
        self.assertEqual(a.edits, 0)
        self.assertEqual(a.reads, 5)

    def test_last_model_ignores_synthetic(self):
        objs = [
            asst([read_tu(0)], model="claude-opus-4-8"),
            asst([read_tu(1)], model="<synthetic>"),
        ]
        a = analyze(objs)
        self.assertEqual(a.last_model, "claude-opus-4-8")

    def test_bash_reads_and_edits_count_in_window(self):
        objs = [
            asst([("b0", "Bash", {"command": "git status"})]),
            asst([("b1", "Bash", {"command": "cat x"})]),
            asst([("b2", "Bash", {"command": "rm y"})]),
            asst([("b3", "Bash", {"command": "sed -i s/a/b/ z"})]),
            asst([("b4", "Bash", {"command": "npm test"})]),
            asst([("b5", "Bash", {"command": "ls"})]),
        ]
        a = analyze(objs)
        self.assertEqual(a.reads, 3)
        self.assertEqual(a.edits, 2)

    def test_context_tokens_capture(self):
        usage = {
            "cache_read_input_tokens": 100_000,
            "cache_creation_input_tokens": 50_000,
            "input_tokens": 6_000,
        }
        objs = [asst([read_tu(0)], usage=usage), user_str("next please")]
        a = analyze(objs)
        self.assertEqual(a.context_tokens, 156_000)

    def test_context_tokens_zero_without_usage(self):
        objs = [asst([read_tu(0)]), asst([edit_tu(0)])]
        a = analyze(objs)
        self.assertEqual(a.context_tokens, 0)

    def test_context_tokens_from_last_assistant(self):
        objs = [
            asst([read_tu(0)], usage={"input_tokens": 100_000}),
            user_str("more"),
            asst([read_tu(1)], usage={"input_tokens": 300_000}),
            user_str("even more"),
        ]
        a = analyze(objs)
        self.assertEqual(a.context_tokens, 300_000)

    def test_context_tokens_survive_synthetic_zero_usage(self):

        objs = [
            asst(
                [read_tu(0)],
                usage={
                    "cache_read_input_tokens": 200_000,
                    "cache_creation_input_tokens": 40_000,
                    "input_tokens": 10,
                },
            ),
            synthetic_api_error(),
        ]
        a = analyze(objs)
        self.assertEqual(a.context_tokens, 240_010)

    def test_synthetic_line_preserves_iter_states_invariant(self):

        objs = [
            asst([read_tu(0)], usage=mk_usage(200_000, 40_000, 10, 5), msg_id="m1"),
            synthetic_api_error(),
            asst([], usage=mk_usage(240_000, 9_990, 10, 7), msg_id="m2"),
        ]
        path = write_lines(objs)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        a = ah.analyze_transcript(path)
        self.assertEqual(a, list(ah.iter_states(path))[-1][1])
        self.assertEqual(a.context_tokens, 250_000)

    def test_synthetic_only_usage_still_proves_schema(self):

        objs = [synthetic_api_error() for _ in range(6)]
        ph = analyze(objs).parse_health
        self.assertTrue(ph.usage_seen)
        self.assertFalse(ph.schema_canary)
        self.assertFalse(ph.degraded)

    def test_turns_counting(self):
        objs = [
            user_str("hello"),
            results([("x", False, "ok")]),
            user_str("<system-reminder> ping"),
            user_str("do the thing"),
        ]
        a = analyze(objs)
        self.assertEqual(a.turns, 2)

    def test_malformed_content_line_does_not_freeze_pass(self):

        objs = [
            user_str("start"),
            asst([read_tu(0)]),
            results([("x", True, [{"type": "text", "text": None}])]),
            asst([read_tu(1)], usage={"input_tokens": 300_000}),
            user_str("peak"),
        ]
        a = analyze(objs)
        self.assertEqual(a.total_reads, 2)
        self.assertEqual(a.context_tokens, 300_000)
        self.assertEqual(a.turns, 2)


class TestSanitize(unittest.TestCase):
    """Control-char stripping at the render seams (ANSI/OSC injection defense)."""

    def test_blind_loop_chip_sanitizes_file(self):

        a = ah.Analysis(
            insufficient=False,
            failed_edit_loop=ah.EditLoop(file="/repo/a\x1b[31m.py", count=2),
            edits=3,
        )
        bl = ah.blind_loop_alert(a)
        self.assertNotIn("\x1b", bl["chip"])
        self.assertIn("EDIT LOOP", bl["chip"])
        self.assertIn("x2", bl["chip"])


class TestDebugCli(unittest.TestCase):
    """argv is untrusted input: the debug CLI must survive whatever it is handed."""

    def _transcript(self):
        objs = [
            asst([read_tu(i, f"/repo/f{i}.py")], usage=mk_usage(cr=50_000, inp=1_000))
            for i in range(6)
        ]
        path = write_lines(objs)
        self.addCleanup(os.unlink, path)
        return path

    def test_cli_float_rejects_non_finite(self):
        for raw in ("nan", "inf", "-inf", "Infinity"):
            with self.subTest(raw=raw):
                self.assertIsNone(ah._cli_float(("t.jsonl", raw), 1))
        self.assertEqual(ah._cli_float(("t.jsonl", "42.5"), 1), 42.5)

    def test_non_finite_ctx_pct_does_not_crash_the_cli(self):

        path = self._transcript()
        for raw in ("nan", "inf", "-inf"):
            with self.subTest(raw=raw):
                with (
                    contextlib.redirect_stdout(io.StringIO()) as out,
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    self.assertEqual(ah.main([path, raw]), 0)
                self.assertNotIn("%", out.getvalue())

    def test_a_stalled_repeat_reaches_the_debug_line(self):

        objs = []
        for i in range(BANDS.PERSEV_MIN):
            objs.append(asst([bash_tu(i)], timestamp=iso_ts(i * 10)))
            objs.append(results([(f"b{i}", True, "1 test failed")]))
        path = write_lines(objs)
        self.addCleanup(os.unlink, path)

        with (
            contextlib.redirect_stdout(io.StringIO()) as out,
            contextlib.redirect_stderr(io.StringIO()) as err,
        ):
            self.assertEqual(ah.main([path]), 0)

        self.assertIn(f"persev x{BANDS.PERSEV_MIN}", out.getvalue())
        self.assertEqual(
            json.loads(err.getvalue())["perseveration"]["score"], BANDS.PERSEV_MIN
        )

    def test_cache_repeat_and_parse_doubt_reach_the_debug_line(self):

        a = ah.Analysis(
            cache_health={"show": True, "hit_drop": 37.4},
            repetition=ah.Repetition(
                score=BANDS.REPEAT_MIN,
                worst_target="/repo/a.py",
                count=BANDS.REPEAT_MIN,
            ),
            parse_health=ah.ParseHealth(degraded=True),
        )

        line = ah._summary_line(a, None, None, None)

        self.assertIn("cache 37%", line)
        self.assertIn(f"repeat x{BANDS.REPEAT_MIN}", line)
        self.assertIn("health data stale?", line)

    def test_non_finite_window_size_grades_on_the_base_ladder(self):
        path = self._transcript()
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()) as err,
        ):
            self.assertEqual(ah.main([path, "50", "claude-opus-4-8", "inf"]), 0)
        self.assertIsNone(json.loads(err.getvalue())["window_class"])


class TestFixtures(unittest.TestCase):
    """If committed fixtures exist, the core must classify them correctly."""

    FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

    def _analyze(self, name):
        p = os.path.join(self.FIX, name)
        if not os.path.exists(p):
            self.skipTest(f"fixture {name} not present")
        return ah.analyze_transcript(p)

    def test_compaction_summary_fixture_excluded_from_turns(self):

        a = self._analyze("compaction_summary.jsonl")
        self.assertEqual(a.turns, 2)

    def test_genuine_blind_loop_fixture(self):
        a = self._analyze("genuine_blind_loop.jsonl")
        self.assertGreaterEqual(a.failed_edit_loop.count, 1)
        bl = ah.blind_loop_alert(a)
        self.assertIsNotNone(bl)
        self.assertEqual(bl["state"], "red")

    def test_hookdeny_not_loop_fixture(self):
        a = self._analyze("hookdeny_not_loop.jsonl")
        self.assertEqual(a.failed_edit_loop.count, 0)
        self.assertIsNone(ah.blind_loop_alert(a))


if __name__ == "__main__":
    unittest.main(verbosity=2)
