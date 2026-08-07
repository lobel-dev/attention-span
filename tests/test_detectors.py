import json
import unittest

from builders import THRASH_SPECS

from attention_span import agent_health, detectors, health_config
from attention_span.analysis import EditLoop, Perseveration, Repetition

THRESHOLDS = health_config.ENGINE_DEFAULTS
BANDS = health_config.ENGINE_BANDS


def event(event_class, path=None, result_state="PENDING", tool_name=None):
    return {
        "class": event_class,
        "file_path": path,
        "result_state": result_state,
        "tool_name": tool_name,
    }


def call(
    sig,
    result_hash="same-answer",
    ts=0.0,
    event_class="neutral",
    result_state="OK",
    tool_name="Bash",
):
    """One entry of the reducer's call log, as ``_consume_tool_use`` builds it."""
    return {
        "tool_use_id": None,
        "class": event_class,
        "tool_name": tool_name,
        "sig": sig,
        "ts_epoch": ts,
        "result_state": result_state,
        "result_hash": result_hash,
    }


def edit_call(i, ts=0.0):
    return call(f"edit-{i}", ts=ts, event_class="edit", tool_name="Edit")


def cache_turns(specs, gap=5, timestamps=True):
    return [
        {
            "cr": cr,
            "cc": cc,
            "inp": inp,
            "ctx": cr + cc + inp,
            "ts_epoch": (i + 1) * gap if timestamps else None,
        }
        for i, (cr, cc, inp) in enumerate(specs)
    ]


class TestDetectorFacade(unittest.TestCase):
    def test_agent_health_reexports_canonical_detector_identities(self):
        names = (
            "_executed_events",
            "_failed_edit_loop",
            "_percentile",
            "_cache_health",
            "_repetition",
            "_perseveration",
        )

        for name in names:
            with self.subTest(name=name):
                self.assertIs(getattr(agent_health, name), getattr(detectors, name))


class TestExecutedEvents(unittest.TestCase):
    def test_excludes_only_hook_denials_and_keeps_pending(self):
        denied = event("edit", "/repo/a.py", "HOOK_DENY")
        genuine = event("edit", "/repo/b.py", "GENUINE")
        successful = event("read", "/repo/c.py", "OK")
        pending = event("edit", "/repo/d.py")

        self.assertEqual(
            detectors._executed_events([denied, genuine, successful, pending]),
            [genuine, successful, pending],
        )


class TestFailedEditLoop(unittest.TestCase):
    def test_genuine_failure_followed_by_pending_reedits_counts(self):
        ordered = [
            event("edit", "/repo/a.py", "GENUINE"),
            event("edit", "/repo/a.py"),
            event("edit", "/repo/a.py"),
        ]

        self.assertEqual(
            detectors._failed_edit_loop(ordered),
            EditLoop(file="/repo/a.py", count=2),
        )

    def test_only_same_file_read_breaks_the_loop(self):
        different_file_read = [
            event("edit", "/repo/a.py", "GENUINE"),
            event("read", "/repo/b.py", "OK"),
            event("edit", "/repo/a.py"),
        ]
        same_file_read = [
            event("edit", "/repo/a.py", "GENUINE"),
            event("read", "/repo/a.py", "OK"),
            event("edit", "/repo/a.py"),
        ]

        self.assertEqual(detectors._failed_edit_loop(different_file_read).count, 1)
        self.assertEqual(detectors._failed_edit_loop(same_file_read).count, 0)

    def test_success_after_failure_resets_the_loop(self):
        ordered = [
            event("edit", "/repo/a.py", "GENUINE"),
            event("edit", "/repo/a.py", "OK"),
            event("edit", "/repo/a.py"),
        ]

        self.assertEqual(detectors._failed_edit_loop(ordered), EditLoop())

    def test_hook_denied_retry_after_failure_never_ran(self):
        ordered = [
            event("edit", "/repo/a.py", "GENUINE"),
            event("edit", "/repo/a.py", "HOOK_DENY"),
        ]

        self.assertEqual(detectors._failed_edit_loop(ordered), EditLoop())


class TestPercentile(unittest.TestCase):
    def test_empty_and_nearest_rank_values(self):
        self.assertEqual(detectors._percentile([], 0.9), 0.0)
        self.assertEqual(detectors._percentile([1, 2, 3, 4, 5], 0.0), 1)
        self.assertEqual(detectors._percentile([1, 2, 3, 4, 5], 0.9), 5)


class TestCacheHealth(unittest.TestCase):
    def test_sustained_thrash_shows(self):
        health = detectors._cache_health(cache_turns(THRASH_SPECS), THRESHOLDS)

        self.assertFalse(health["suppressed"])
        self.assertTrue(health["show"])
        self.assertGreaterEqual(
            health["hit_drop"], health_config.ENGINE_BANDS.CACHE_HIT_DROP_SHOW
        )
        self.assertGreaterEqual(health["churn_mult"], 1.0)
        self.assertGreater(health["thrash_score"], 0)

    def test_long_idle_suppressed(self):
        turns = cache_turns(
            THRASH_SPECS,
            gap=health_config.ENGINE_BANDS.CACHE_IDLE_TTL_S + 60,
        )

        health = detectors._cache_health(turns, THRESHOLDS)

        self.assertTrue(health["suppressed"])
        self.assertFalse(health["show"])

    def test_warmup_only_suppressed(self):
        turns = cache_turns([(5000 + i, 5000, 500) for i in range(3)])

        health = detectors._cache_health(turns, THRESHOLDS)

        self.assertTrue(health["suppressed"])
        self.assertFalse(health["show"])

    def test_healthy_stable_not_shown(self):
        turns = cache_turns([(9000, 500, 500 + i) for i in range(15)])

        health = detectors._cache_health(turns, THRESHOLDS)

        self.assertFalse(health["suppressed"])
        self.assertFalse(health["show"])
        self.assertEqual(health["hit_drop"], 0.0)

    def test_growth_explained_churn_not_thrash(self):
        turns = cache_turns([(3000, 18000, 500 + i * 4000) for i in range(15)])

        health = detectors._cache_health(turns, THRESHOLDS)

        self.assertFalse(health["show"])
        self.assertLess(health["churn_mult"], 1.0)

    def _collapsed_window(self, collapse_specs):

        return cache_turns(
            [(5000 + i, 5000, 500) for i in range(3)]
            + [(9000 + i, 500, 500) for i in range(3)]
            + list(collapse_specs)
        )

    def test_zero_cache_read_window_counts_as_maximum_churn(self):

        window = health_config.ENGINE_BANDS.CACHE_WINDOW
        turns = self._collapsed_window([(0, 9500 - i, 500) for i in range(window)])

        health = detectors._cache_health(turns, THRESHOLDS)

        self.assertEqual(health["churn_mult"], detectors.CHURN_MULT_MAX)
        self.assertTrue(health["show"])

        json.dumps(health, allow_nan=False)

    def test_zero_cache_read_window_without_creation_is_no_churn(self):

        window = health_config.ENGINE_BANDS.CACHE_WINDOW
        turns = self._collapsed_window([(0, 0, 500) for _ in range(window)])

        health = detectors._cache_health(turns, THRESHOLDS)

        self.assertEqual(health["churn_mult"], 0.0)
        self.assertFalse(health["show"])

    def test_missing_timestamps_fail_closed(self):
        turns = cache_turns(THRASH_SPECS, timestamps=False)

        self.assertFalse(detectors._cache_health(turns, THRESHOLDS)["show"])

    def test_partial_timestamps_fail_closed(self):
        warm = cache_turns([(5000 + i, 5000, 500) for i in range(3)])
        collapse = cache_turns(
            [(400 + i, 9100 - i, 500) for i in range(8)],
            timestamps=False,
        )

        self.assertFalse(detectors._cache_health(warm + collapse, THRESHOLDS)["show"])


class TestRepetition(unittest.TestCase):
    def test_repeated_reads_fire(self):
        path = "/repo/a.py"
        ordered = [
            event("read", path) for _ in range(health_config.ENGINE_BANDS.REPEAT_MIN)
        ]

        self.assertEqual(
            detectors._repetition(ordered, THRESHOLDS),
            Repetition(
                score=health_config.ENGINE_BANDS.REPEAT_MIN,
                worst_target=path,
                count=health_config.ENGINE_BANDS.REPEAT_MIN,
            ),
        )

    def test_read_after_edit_is_a_verification_baseline(self):
        ordered = [
            event("read", "/repo/a.py"),
            event("edit", "/repo/a.py", "OK"),
            event("read", "/repo/a.py"),
            event("read", "/repo/a.py"),
        ]

        repetition = detectors._repetition(ordered, THRESHOLDS)

        self.assertLess(repetition.score, health_config.ENGINE_BANDS.REPEAT_MIN)

    def test_hook_denied_reads_are_excluded(self):
        ordered = [
            event("read", "/repo/a.py", "HOOK_DENY")
            for _ in range(health_config.ENGINE_BANDS.REPEAT_MIN + 2)
        ]

        self.assertEqual(detectors._repetition(ordered, THRESHOLDS), Repetition())

    def test_shell_reads_are_not_scored(self):

        ordered = [
            event("read", "/repo/a.py", tool_name="Bash")
            for _ in range(health_config.ENGINE_BANDS.REPEAT_MIN + 2)
        ]

        self.assertEqual(detectors._repetition(ordered, THRESHOLDS), Repetition())

    def test_distinct_files_are_not_repetition(self):
        ordered = [
            event("read", f"/repo/f{i}.py")
            for i in range(health_config.ENGINE_BANDS.REPEAT_MIN + 2)
        ]

        repetition = detectors._repetition(ordered, THRESHOLDS)

        self.assertLess(repetition.score, health_config.ENGINE_BANDS.REPEAT_MIN)

    def test_any_edit_resets_reference_rereads(self):
        ordered = [
            item
            for i in range(health_config.ENGINE_BANDS.REPEAT_MIN + 2)
            for item in (
                event("read", "/repo/spec.md"),
                event("edit", f"/repo/impl{i}.py", "OK"),
            )
        ]

        repetition = detectors._repetition(ordered, THRESHOLDS)

        self.assertLess(repetition.score, health_config.ENGINE_BANDS.REPEAT_MIN)


class TestPerseveration(unittest.TestCase):
    """#4 - the same call producing the same answer, with nothing moving between."""

    def test_identical_call_and_result_run_fires(self):
        calls = [call("npm-test", ts=i) for i in range(BANDS.PERSEV_MIN)]

        self.assertEqual(
            detectors._perseveration(calls, THRESHOLDS),
            Perseveration(
                score=BANDS.PERSEV_MIN,
                worst_target="Bash",
                count=BANDS.PERSEV_MIN,
            ),
        )

    def test_changed_result_resets_the_run(self):

        calls = [
            call("npm-test", result_hash=f"answer-{i}", ts=i)
            for i in range(BANDS.PERSEV_MIN + 2)
        ]

        self.assertEqual(detectors._perseveration(calls, THRESHOLDS).score, 1)

    def test_any_executed_edit_resets_every_run(self):
        calls = [
            call("npm-test", ts=0),
            call("npm-test", ts=1),
            edit_call(0, ts=2),
            call("npm-test", ts=3),
            call("npm-test", ts=4),
        ]

        self.assertEqual(detectors._perseveration(calls, THRESHOLDS).score, 2)

    def test_hook_denied_calls_never_ran(self):
        calls = [
            call("npm-test", ts=i, result_state="HOOK_DENY")
            for i in range(BANDS.PERSEV_MIN + 2)
        ]

        self.assertEqual(detectors._perseveration(calls, THRESHOLDS), Perseveration())

    def test_idle_gap_does_not_extend_a_run(self):

        gap = BANDS.PERSEV_IDLE_TTL_S + 1
        calls = [call("check", ts=i * gap) for i in range(BANDS.PERSEV_MIN + 2)]

        self.assertEqual(detectors._perseveration(calls, THRESHOLDS).score, 1)

    def test_missing_timestamps_fail_closed(self):

        calls = [call("npm-test", ts=None) for _ in range(BANDS.PERSEV_MIN + 2)]

        self.assertEqual(detectors._perseveration(calls, THRESHOLDS).score, 1)

    def test_pending_calls_neither_extend_nor_reset(self):
        settled = [call("npm-test", ts=i) for i in range(BANDS.PERSEV_MIN)]
        pending = call("npm-test", ts=99, result_state="PENDING", result_hash=None)
        pending_edit = call(
            "edit-0",
            ts=99,
            event_class="edit",
            result_state="PENDING",
            result_hash=None,
            tool_name="Edit",
        )

        extended = [*settled[:-1], pending]
        interrupted = [*settled[:-1], pending_edit, settled[-1]]

        self.assertEqual(
            detectors._perseveration(extended, THRESHOLDS).score,
            BANDS.PERSEV_MIN - 1,
        )
        self.assertEqual(
            detectors._perseveration(interrupted, THRESHOLDS).score,
            BANDS.PERSEV_MIN,
        )

    def test_the_tdd_test_edit_test_loop_stays_at_one(self):

        calls = [
            item
            for i in range(BANDS.PERSEV_MIN + 2)
            for item in (call("npm-test", ts=i * 2), edit_call(i, ts=i * 2 + 1))
        ]

        self.assertEqual(detectors._perseveration(calls, THRESHOLDS).score, 1)

    def test_repeats_spaced_beyond_the_window_never_accumulate(self):
        calls = []
        for i in range(BANDS.PERSEV_MIN + 1):
            calls.append(call("npm-test", ts=len(calls)))
            base = len(calls)
            calls.extend(
                call(f"filler-{i}-{j}", ts=base + j) for j in range(BANDS.PERSEV_WINDOW)
            )

        self.assertEqual(detectors._perseveration(calls, THRESHOLDS).score, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
