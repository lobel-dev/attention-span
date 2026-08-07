"""Reducer state-machine integration tests."""

import dataclasses
import json
import os
import tempfile
import unittest

from builders import (
    THRASH_SPECS,
    analyze,
    asst,
    bash_tu,
    cache_objs,
    edit_tu,
    iso_ts,
    read_tu,
    results,
    synthetic_api_error,
    task_notification,
    user_compact_summary,
    user_str,
    write_lines,
)
from builders import usage as mk_usage

from attention_span import agent_health as ah
from attention_span import health_config, reducer

BANDS = health_config.ENGINE_BANDS


class TestReducerFacade(unittest.TestCase):
    def test_agent_health_reexports_canonical_reducer_identities(self):
        names = (
            "usage_tokens",
            "_usage_int",
            "_ts_epoch",
            "_new_state",
            "_append_changed",
            "_public_event",
            "_is_thinking_block",
            "_empty_cursor",
            "_consume",
            "_finalize",
        )

        for name in names:
            with self.subTest(name=name):
                self.assertIs(getattr(ah, name), getattr(reducer, name))


class TestUsageTotals(unittest.TestCase):
    """Cumulative usage_totals — deduped LAST-WINS per API call (message.id ->
    requestId -> the consecutive-identical (cr,cc,inp) tuple), because Claude Code
    re-logs ONE call across several streaming lines with GROWING partial output."""

    def test_streaming_relog_counts_once_last_wins(self):

        objs = [
            asst([], usage=mk_usage(1744563, 163973, 762, out), msg_id="msg_01")
            for out in (100, 2000, 8789)
        ]
        ut = analyze(objs).usage_totals
        self.assertEqual(ut.api_calls, 1)
        self.assertEqual(ut.output, 8789)
        self.assertEqual(ut.input, 762)
        self.assertEqual(ut.cache_creation, 163973)
        self.assertEqual(ut.cache_read, 1744563)

    def test_distinct_ids_sum_and_bump_api_calls(self):
        objs = [
            asst([], usage=mk_usage(100, 10, 1, 5), msg_id="msg_01"),
            asst([], usage=mk_usage(200, 20, 2, 7), msg_id="msg_02"),
        ]
        ut = analyze(objs).usage_totals
        self.assertEqual(ut.api_calls, 2)
        self.assertEqual(ut.input, 3)
        self.assertEqual(ut.cache_creation, 30)
        self.assertEqual(ut.cache_read, 300)
        self.assertEqual(ut.output, 12)

    def test_missing_usage_contributes_nothing(self):

        objs = [asst([read_tu(0)]), user_str("hello"), asst([read_tu(1)])]
        self.assertEqual(analyze(objs).usage_totals, ah.UsageTotals())

    def test_off_schema_usage_values_coerced_to_zero(self):

        u = {
            "cache_read_input_tokens": None,
            "cache_creation_input_tokens": None,
            "input_tokens": 5,
            "output_tokens": 1.5,
        }
        a = analyze([asst([], usage=u, msg_id="m1")])
        self.assertEqual(a.usage_totals, ah.UsageTotals(input=5, api_calls=1))
        self.assertFalse(a.parse_health.parse_aborted)

    def test_fallback_request_id_then_tuple(self):
        objs = [
            asst([], usage=mk_usage(50, 5, 1, 10), request_id="req_1"),
            asst([], usage=mk_usage(50, 5, 1, 20), request_id="req_1"),
            asst([], usage=mk_usage(60, 6, 2, 1)),
            asst([], usage=mk_usage(60, 6, 2, 7)),
        ]
        ut = analyze(objs).usage_totals
        self.assertEqual(ut.api_calls, 2)
        self.assertEqual(ut.output, 20 + 7)
        self.assertEqual(ut.input, 1 + 2)

    def test_distinct_tuples_without_ids_count_separately(self):
        objs = [
            asst([], usage=mk_usage(60, 6, 2, 1)),
            asst([], usage=mk_usage(61, 6, 2, 1)),
        ]
        self.assertEqual(analyze(objs).usage_totals.api_calls, 2)

    def test_synthetic_zero_usage_counts_no_api_call(self):

        objs = [
            asst([], usage=mk_usage(100, 10, 1, 5), msg_id="msg_01"),
            synthetic_api_error(),
        ]
        self.assertEqual(
            analyze(objs).usage_totals,
            ah.UsageTotals(
                input=1, cache_creation=10, cache_read=100, output=5, api_calls=1
            ),
        )

    def test_usage_totals_drift_invariant(self):

        objs = [
            asst([read_tu(0)], usage=mk_usage(100, 10, 1, 100), msg_id="m1"),
            asst([], usage=mk_usage(100, 10, 1, 900), msg_id="m1"),
        ]
        path = write_lines(objs)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        a = ah.analyze_transcript(path)
        states = list(ah.iter_states(path))
        self.assertEqual(a, states[-1][1])
        self.assertEqual(a.usage_totals.output, 900)
        self.assertEqual(a.usage_totals.api_calls, 1)

    def test_snapshots_do_not_alias_mutating_state(self):

        objs = [
            asst([], usage=mk_usage(10, 1, 1, 5), msg_id="m1"),
            asst([], usage=mk_usage(20, 2, 2, 6), msg_id="m2"),
        ]
        path = write_lines(objs)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        snaps = [s for _, s in ah.iter_states(path)]
        self.assertEqual(snaps[0].usage_totals.api_calls, 1)
        self.assertEqual(snaps[-1].usage_totals.api_calls, 2)

    def test_burn_arithmetic_excludes_cache_read(self):

        objs = [
            asst(
                [],
                usage=mk_usage(1744563, 163973, 762, 8789),
                msg_id="m1",
                child=True,
                agent_id="a1",
            )
        ]
        path = write_lines(objs)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        ut = ah.analyze_transcript(path, include_sidechain=True).usage_totals
        burn = ut.input + ut.cache_creation + ut.output
        self.assertEqual(burn, 762 + 163973 + 8789)
        self.assertGreater(ut.cache_read, 10 * burn)


class TestThinkingTracking(unittest.TestCase):
    """Signed-thinking-block presence (evidence of realized reasoning)."""

    def test_counts_signed_thinking_turns(self):
        objs = [
            asst([read_tu(0)], thinking=True),
            asst([read_tu(1)]),
            asst([edit_tu(0)], thinking="reasoning here"),
        ]
        a = analyze(objs)
        self.assertEqual(a.assistant_turns, 3)
        self.assertEqual(a.thinking_turns, 2)

    def test_trailing_no_thinking_accumulates_then_resets(self):
        a = analyze(
            [
                asst([read_tu(0)], thinking=True),
                asst([read_tu(1)]),
                asst([edit_tu(0)]),
            ]
        )
        self.assertEqual(a.trailing_no_thinking, 2)
        b = analyze(
            [
                asst([read_tu(0)], thinking=True),
                asst([read_tu(1)]),
                asst([edit_tu(0)]),
                asst([edit_tu(1)], thinking=True),
            ]
        )
        self.assertEqual(b.trailing_no_thinking, 0)
        self.assertEqual(b.thinking_turns, 2)

    def test_redacted_thinking_block_counts(self):
        turn = {
            "type": "assistant",
            "isSidechain": False,
            "message": {
                "model": "claude-opus-4-8",
                "content": [
                    {"type": "redacted_thinking", "data": "enc"},
                    {
                        "type": "tool_use",
                        "id": "r0",
                        "name": "Read",
                        "input": {"file_path": "/x"},
                    },
                ],
            },
        }
        a = analyze([turn])
        self.assertEqual(a.thinking_turns, 1)
        self.assertEqual(a.assistant_turns, 1)

    def test_unsigned_empty_thinking_block_ignored(self):
        turn = {
            "type": "assistant",
            "isSidechain": False,
            "message": {
                "model": "m",
                "content": [
                    {"type": "thinking", "thinking": ""},
                    {
                        "type": "tool_use",
                        "id": "r0",
                        "name": "Read",
                        "input": {"file_path": "/x"},
                    },
                ],
            },
        }
        a = analyze([turn])
        self.assertEqual(a.thinking_turns, 0)
        self.assertEqual(a.assistant_turns, 1)

    def test_thinking_fields_survive_iter_states_invariant(self):
        objs = [
            asst([read_tu(0)], thinking=True),
            asst([edit_tu(0)]),
            asst([read_tu(1)], thinking=True),
        ]
        path = write_lines(objs)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        a = ah.analyze_transcript(path)
        states = list(ah.iter_states(path))
        self.assertEqual(a, states[-1][1])
        self.assertEqual(a.thinking_turns, 2)


class TestSidechainInclusion(unittest.TestCase):
    """Child (subagent) transcript analysis: every line is isSidechain — skipped by
    default, analyzable with include_sidechain=True (Stage 0 of subagent health)."""

    def _path(self, objs):
        path = write_lines(objs)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        return path

    def test_child_skipped_by_default(self):

        objs = [
            asst([read_tu(i, f"/repo/f{i}.py")], child=True, agent_id="a1")
            for i in range(6)
        ]
        a = ah.analyze_transcript(self._path(objs))
        self.assertEqual(a.reads, 0)
        self.assertEqual(a.edits, 0)
        self.assertTrue(a.insufficient)

    def test_include_sidechain_matches_driver_analysis(self):

        events = [read_tu(i, f"/repo/f{i}.py") for i in range(6)]
        child = ah.analyze_transcript(
            self._path([asst([e], child=True, agent_id="a1") for e in events]),
            include_sidechain=True,
        )
        driver = ah.analyze_transcript(self._path([asst([e]) for e in events]))
        self.assertEqual(child.reads, 6)
        self.assertFalse(child.insufficient)
        for name in (
            "reads",
            "edits",
            "r2e",
            "base_tier",
            "insufficient",
            "failed_edit_loop",
        ):
            self.assertEqual(getattr(child, name), getattr(driver, name))

    def test_last_stop_reason_captured(self):
        objs = [asst([read_tu(0)]), asst([read_tu(1)], stop_reason="end_turn")]
        a = ah.analyze_transcript(self._path(objs))
        self.assertEqual(a.last_stop_reason, "end_turn")

    def test_last_stop_reason_none_when_absent(self):
        a = ah.analyze_transcript(self._path([asst([read_tu(0)])]))
        self.assertIsNone(a.last_stop_reason)


class TestInjectedTurnsExcluded(unittest.TestCase):
    """`turns` = real human turns only. Injected Claude Code content (compaction
    digests, !-bang/slash side-channels) must never inflate it."""

    def _turns(self, objs):
        path = write_lines(objs)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        return ah.analyze_transcript(path).turns

    def test_compaction_summary_string_not_counted(self):
        self.assertEqual(
            self._turns(
                [user_str("real one"), user_compact_summary(), user_str("real two")]
            ),
            2,
        )

    def test_compaction_summary_list_not_counted(self):

        self.assertEqual(
            self._turns([user_str("real one"), user_compact_summary(as_list=True)]), 1
        )

    def test_human_paste_of_compaction_phrase_still_counts(self):

        phrase = (
            "This session is being continued from a previous conversation that ran "
            "out of context."
        )
        self.assertEqual(self._turns([user_str(phrase)]), 1)

    def test_bash_and_local_command_side_channels_not_counted(self):

        objs = [
            user_str("real human turn"),
            user_str("<bash-input>open .</bash-input>"),
            user_str("<bash-stdout>(Bash completed with no output)</bash-stdout>"),
            user_str("<bash-stderr></bash-stderr>"),
            user_str("<local-command-stdout>Login successful</local-command-stdout>"),
        ]
        self.assertEqual(self._turns(objs), 1)

    def test_type_summary_index_record_not_counted(self):

        self.assertEqual(
            self._turns(
                [
                    user_str("real"),
                    {"type": "summary", "summary": "title", "leafUuid": "x"},
                ]
            ),
            1,
        )

    def test_meta_flagged_injection_not_counted(self):

        meta = {
            "type": "user",
            "isSidechain": False,
            "isMeta": True,
            "message": {"role": "user", "content": "# Code Review\n> template body"},
        }
        self.assertEqual(self._turns([user_str("real"), meta]), 1)

    def test_non_meta_human_turn_with_same_text_still_counts(self):

        self.assertEqual(self._turns([user_str("# Code Review")]), 1)

    def test_compaction_preserves_iter_states_invariant(self):
        objs = [
            user_str("real one"),
            user_compact_summary(),
            asst([read_tu(0)]),
            results([("r0", False, "ok")]),
            user_str("real two"),
        ]
        path = write_lines(objs)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        final = ah.analyze_transcript(path)
        states = list(ah.iter_states(path))
        self.assertEqual(final, states[-1][1])
        self.assertEqual(final.turns, 2)

    def test_compaction_pending_until_next_real_driver_assistant(self):
        marker = user_compact_summary()

        self.assertTrue(analyze([marker]).compaction_pending)
        self.assertTrue(
            analyze(
                [
                    marker,
                    asst([], model="<synthetic>"),
                    asst([], child=True, agent_id="child"),
                ]
            ).compaction_pending
        )
        self.assertFalse(analyze([marker, asst([])]).compaction_pending)


class TestTaskNotifications(unittest.TestCase):
    """`task_notifications` = the PARENT-side evidence that a subagent stopped.

    A task-notification fires exactly when an agent stops, so the block itself is the
    evidence and its `<status>` word is never read. The harness can resume a notified
    agent, so the same task-id may notify repeatedly and only the LATEST time counts.
    """

    def _epoch(self, ts):
        return reducer._ts_epoch(ts)

    def test_notification_records_the_lines_own_epoch_timestamp(self):
        notified = analyze([task_notification("a1", timestamp=iso_ts(30))])

        self.assertEqual(notified.task_notifications, {"a1": self._epoch(iso_ts(30))})

    def test_the_latest_notification_for_one_task_id_wins(self):

        objs = [
            task_notification("a1", timestamp=iso_ts(10)),
            task_notification("a1", timestamp=iso_ts(90)),
        ]

        self.assertEqual(
            analyze(objs).task_notifications, {"a1": self._epoch(iso_ts(90))}
        )

    def test_an_out_of_order_older_notification_never_moves_the_time_back(self):
        objs = [
            task_notification("a1", timestamp=iso_ts(90)),
            task_notification("a1", timestamp=iso_ts(10)),
        ]

        self.assertEqual(
            analyze(objs).task_notifications, {"a1": self._epoch(iso_ts(90))}
        )

    def test_distinct_task_ids_are_kept_apart(self):
        objs = [
            task_notification("a1", timestamp=iso_ts(10)),
            task_notification("a2", timestamp=iso_ts(20)),
        ]

        self.assertEqual(
            analyze(objs).task_notifications,
            {"a1": self._epoch(iso_ts(10)), "a2": self._epoch(iso_ts(20))},
        )

    def test_a_block_without_a_parseable_task_id_is_ignored(self):
        obj = task_notification("a1")
        obj["message"]["content"] = "<task-notification>\n<status>completed</status>"

        self.assertEqual(analyze([obj]).task_notifications, {})

    def test_a_block_with_an_empty_task_id_is_ignored(self):
        self.assertEqual(analyze([task_notification("")]).task_notifications, {})

    def test_a_marked_line_with_off_schema_content_is_ignored_not_fatal(self):

        off_schema = task_notification("a1")
        off_schema["message"]["content"] = [{"type": "text", "text": "<task-id>a1</>"}]

        result = analyze([off_schema])

        self.assertEqual(result.task_notifications, {})
        self.assertFalse(result.parse_health.parse_aborted)

    def test_a_line_without_a_parseable_timestamp_is_ignored(self):
        missing = task_notification("a1")
        del missing["timestamp"]
        off_schema = task_notification("a2", timestamp="not-a-timestamp")

        self.assertEqual(analyze([missing, off_schema]).task_notifications, {})

    def test_a_user_line_quoting_the_block_without_the_marker_is_not_evidence(self):

        quoted = task_notification("a1")
        del quoted["origin"]

        self.assertEqual(analyze([quoted]).task_notifications, {})

    def test_an_assistant_line_carrying_the_block_is_not_evidence(self):
        quoted = task_notification("a1")
        quoted["type"] = "assistant"

        self.assertEqual(analyze([quoted]).task_notifications, {})

    def test_a_sidechain_line_carrying_the_block_is_not_evidence(self):

        quoted = task_notification("a1")
        quoted["isSidechain"] = True

        self.assertEqual(analyze([quoted]).task_notifications, {})

    def test_a_transcript_with_no_notification_yields_an_empty_mapping(self):
        self.assertEqual(
            analyze([asst([read_tu(0)]), user_str("hi")]).task_notifications, {}
        )

    def test_notifications_survive_the_iter_states_drift_invariant(self):
        objs = [
            asst([read_tu(0)]),
            task_notification("a1", timestamp=iso_ts(10)),
            asst([edit_tu(0)]),
            task_notification("a1", timestamp=iso_ts(50)),
        ]
        path = write_lines(objs)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))

        final = ah.analyze_transcript(path)
        states = list(ah.iter_states(path))

        self.assertEqual(final, states[-1][1])
        self.assertEqual(final.task_notifications, {"a1": self._epoch(iso_ts(50))})

    def test_snapshots_do_not_alias_the_mutating_notification_state(self):
        objs = [
            task_notification("a1", timestamp=iso_ts(10)),
            task_notification("a2", timestamp=iso_ts(20)),
        ]
        path = write_lines(objs)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))

        snaps = [s for _, s in ah.iter_states(path)]

        self.assertEqual(snaps[0].task_notifications, {"a1": self._epoch(iso_ts(10))})
        self.assertEqual(len(snaps[-1].task_notifications), 2)


class TestCacheHealthIntegration(unittest.TestCase):
    def test_streaming_duplicate_usage_deduped(self):

        warm = [(5000 + i, 5000, 500) for i in range(3)]
        dup_collapse = [(400, 9100, 500)] * 10
        ch = analyze(cache_objs(warm + dup_collapse)).cache_health
        self.assertTrue(ch["suppressed"])
        self.assertFalse(ch["show"])

    def test_missing_timestamp_input_still_fails_closed(self):

        objs = [asst([], usage=mk_usage(*spec)) for spec in THRASH_SPECS]

        self.assertFalse(analyze(objs).cache_health["show"])

    def test_synthetic_line_does_not_eat_a_cache_warmup_slot(self):

        reals = cache_objs([(5000 + i, 5000, 500) for i in range(6)])
        objs = [synthetic_api_error(timestamp=iso_ts(1))] + reals
        ch = analyze(objs).cache_health
        self.assertEqual(ch["window_turns"], 6 - BANDS.CACHE_WARM_TURNS)

        path = write_lines(cache_objs(THRASH_SPECS, gap=5))
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        a = ah.analyze_transcript(path)
        last = list(ah.iter_states(path))[-1][1]
        self.assertEqual(a.cache_health, last.cache_health)
        self.assertTrue(a.cache_health["show"])


class TestRepetitionIntegration(unittest.TestCase):
    def test_repetition_drift_invariant(self):
        path = write_lines(
            [asst([read_tu(i, "/repo/a.py")]) for i in range(BANDS.REPEAT_MIN)]
        )
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        a = ah.analyze_transcript(path)
        last = list(ah.iter_states(path))[-1][1]
        self.assertEqual(a.repetition, last.repetition)


class TestPerseverationIntegration(unittest.TestCase):
    """#4 rides its OWN call log, so a neutral repeat is scored without ever
    reaching `ordered` - the R2E window's substrate."""

    FAILED = "1 test failed"

    def _stalled(self, bodies):
        """`npm test` re-run once per body, ten seconds apart (well inside the TTL)."""
        objs = []
        for i, body in enumerate(bodies):
            objs.append(asst([bash_tu(i)], timestamp=iso_ts(i * 10)))
            objs.append(results([(f"b{i}", True, body)]))
        return objs

    def test_neutral_calls_accrue_to_the_call_log_but_never_to_ordered(self):
        state = reducer._new_state()

        cursor = reducer._consume(state, asst([bash_tu(0)], timestamp=iso_ts(0)))

        self.assertEqual(state["ordered"], [])
        self.assertIn("calls", cursor["changed"])
        self.assertNotIn("ordered", cursor["changed"])
        (call,) = state["calls"]
        self.assertEqual(call["class"], "neutral")
        self.assertEqual(call["tool_name"], "Bash")
        self.assertEqual(call["result_state"], "PENDING")
        self.assertIsNone(call["result_hash"])
        self.assertEqual(call["ts_epoch"], reducer._ts_epoch(iso_ts(0)))

    def test_the_signature_separates_calls_by_tool_input(self):
        state = reducer._new_state()
        for i, command in enumerate(("npm test", "npm test", "npm run build")):
            reducer._consume(state, asst([bash_tu(i, command)]))

        first, repeat, other = (call["sig"] for call in state["calls"])

        self.assertEqual(first, repeat)
        self.assertNotEqual(first, other)

    def test_settling_attaches_the_result_state_and_a_body_identity(self):
        state = reducer._new_state()
        for i, body in enumerate(("1 test failed", "1 test failed", "0 failed")):
            reducer._consume(state, asst([bash_tu(i)], timestamp=iso_ts(i)))
            reducer._consume(state, results([(f"b{i}", True, body)]))

        first, repeat, moved = state["calls"]

        self.assertEqual(first["result_state"], "GENUINE")
        self.assertEqual(first["result_hash"], repeat["result_hash"])
        self.assertNotEqual(first["result_hash"], moved["result_hash"])

    def test_neutral_repeats_score_without_occupying_the_r2e_window(self):
        objs = [asst([read_tu(i, f"/repo/f{i}.py")]) for i in range(2)]
        objs += self._stalled([self.FAILED] * BANDS.PERSEV_MIN)

        a = analyze(objs)

        self.assertGreaterEqual(a.perseveration.score, BANDS.PERSEV_MIN)
        self.assertEqual(a.perseveration.worst_target, "Bash")
        self.assertEqual((a.reads, a.edits, a.window_used), (2, 0, 2))
        self.assertEqual((a.total_reads, a.total_edits), (2, 0))

    def test_a_changed_result_keeps_a_retry_loop_quiet(self):
        objs = self._stalled([self.FAILED] * (BANDS.PERSEV_MIN - 1) + ["0 failed"])

        self.assertLess(analyze(objs).perseveration.score, BANDS.PERSEV_MIN)

    def test_perseveration_drift_invariant(self):
        path = write_lines(self._stalled([self.FAILED] * BANDS.PERSEV_MIN))
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))

        a = ah.analyze_transcript(path)
        last = list(ah.iter_states(path))[-1][1]

        self.assertEqual(a.perseveration, last.perseveration)
        self.assertGreaterEqual(a.perseveration.score, BANDS.PERSEV_MIN)


class TestParseHealth(unittest.TestCase):
    """#2 self-doubt — decode failures / schema canary / aborted pass, precise enough
    not to nag on a single half-written trailing line."""

    def _raw(self, lines):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        return path

    def test_decode_failures_counted_and_degraded(self):
        good = json.dumps(asst([read_tu(0)]))
        a = ah.analyze_transcript(self._raw([good, "not json", "{ also broken", good]))
        self.assertEqual(a.parse_health.decode_failures, 2)
        self.assertTrue(a.parse_health.degraded)

    def test_single_trailing_halfline_not_degraded(self):

        good = json.dumps(asst([read_tu(0)]))
        a = ah.analyze_transcript(self._raw([good, "{ half-written"]))
        self.assertEqual(a.parse_health.decode_failures, 1)
        self.assertFalse(a.parse_health.degraded)

    def test_schema_canary_trips_on_unrecognized_assistant_lines(self):

        objs = [asst([]) for _ in range(BANDS.PARSE_CANARY_MIN_ASSISTANT)]
        a = analyze(objs)
        self.assertTrue(a.parse_health.schema_canary)
        self.assertTrue(a.parse_health.degraded)

    def test_usage_present_does_not_trip_canary(self):
        objs = [
            asst([read_tu(i)], usage=mk_usage(1, 1, 1))
            for i in range(BANDS.PARSE_CANARY_MIN_ASSISTANT + 2)
        ]
        a = analyze(objs)
        self.assertFalse(a.parse_health.schema_canary)
        self.assertFalse(a.parse_health.degraded)

    def test_parse_aborted_set_on_midpass_exception(self):
        from unittest import mock

        path = write_lines([asst([read_tu(0)]), asst([read_tu(1)]), asst([read_tu(2)])])
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        real = ah._consume
        seen = {"n": 0}

        def boom(*a, **k):
            seen["n"] += 1
            if seen["n"] == 2:
                raise RuntimeError("simulated mid-pass failure")
            return real(*a, **k)

        with mock.patch.object(ah, "_consume", side_effect=boom):
            a = ah.analyze_transcript(path)
        self.assertTrue(a.parse_health.parse_aborted)
        self.assertTrue(a.parse_health.degraded)

    def test_open_failure_stays_silent_not_aborted(self):

        a = ah.analyze_transcript("/no/such/transcript.jsonl")
        self.assertFalse(a.parse_health.parse_aborted)
        self.assertFalse(a.parse_health.degraded)

    def test_utf8_corruption_in_first_buffer_not_silently_dropped(self):

        good = json.dumps(asst([read_tu(0, "/repo/a.py")])).encode("utf-8")
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "wb") as f:
            f.write(good + b"\n")
            f.write(b"\xff\xff not valid utf8 or json\n")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        a = ah.analyze_transcript(path)
        self.assertEqual(a.total_reads, 1)
        self.assertGreaterEqual(a.parse_health.decode_failures, 1)

    def test_off_schema_tool_input_handled_gracefully(self):

        bad = {
            "type": "assistant",
            "isSidechain": False,
            "message": {
                "model": "m",
                "usage": mk_usage(1000, 2000, 500),
                "content": [
                    {
                        "type": "tool_use",
                        "id": "x",
                        "name": "Read",
                        "input": ["not", "a", "dict"],
                    }
                ],
            },
        }
        path = write_lines([asst([read_tu(0, "/repo/a.py")]), bad])
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        a = ah.analyze_transcript(path)
        self.assertFalse(a.parse_health.parse_aborted)
        self.assertFalse(a.parse_health.degraded)
        self.assertEqual(a, list(ah.iter_states(path))[-1][1])

    def test_non_mapping_line_skipped_not_aborted(self):

        good = json.dumps(asst([read_tu(0, "/repo/a.py")]))
        path = self._raw([good, '"hello"', "42", "[1, 2]", good])
        a = ah.analyze_transcript(path)
        self.assertFalse(a.parse_health.parse_aborted)
        self.assertEqual(a.parse_health.decode_failures, 0)
        self.assertEqual(a.total_reads, 2)
        self.assertEqual(a, list(ah.iter_states(path))[-1][1])

    def test_non_mapping_message_handled_gracefully(self):

        bad = {"type": "assistant", "isSidechain": False, "message": "just some text"}
        path = write_lines(
            [
                asst([read_tu(0, "/repo/a.py")]),
                bad,
                asst([read_tu(1, "/repo/b.py")]),
            ]
        )
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        a = ah.analyze_transcript(path)
        self.assertFalse(a.parse_health.parse_aborted)
        self.assertEqual(a.total_reads, 2)
        self.assertEqual(a, list(ah.iter_states(path))[-1][1])

    def test_off_schema_usage_handled_gracefully(self):
        bad = {
            "type": "assistant",
            "isSidechain": False,
            "message": {
                "model": "m",
                "usage": ["not", "a", "dict"],
                "content": [{"type": "text", "text": "hi"}],
            },
        }
        path = write_lines([asst([read_tu(0, "/repo/a.py")]), bad])
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        a = ah.analyze_transcript(path)
        self.assertFalse(a.parse_health.parse_aborted)
        self.assertEqual(a, list(ah.iter_states(path))[-1][1])


class TestDriftInvariantCarveout(unittest.TestCase):
    """Verification item 7: the drift invariant holds on all keys EXCEPT parse_health,
    which diverges ONLY on corrupt input (decode_failures counted only in
    analyze_transcript); and iter_states feature rows populate fully (no KeyError)."""

    def _both(self, path):
        a = ah.analyze_transcript(path)
        last = list(ah.iter_states(path))[-1][1]
        return a, last

    def _raw(self, lines):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        return path

    def test_clean_input_identical_including_parse_health(self):
        path = write_lines(
            cache_objs(THRASH_SPECS, gap=5)
            + [asst([read_tu(0, "/repo/a.py")]), asst([edit_tu(0, "/repo/a.py")])]
        )
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        a, last = self._both(path)
        self.assertEqual(a, last)

    def test_corrupt_input_diverges_only_on_parse_health(self):
        good = json.dumps(asst([read_tu(0, "/repo/a.py")]))
        path = self._raw([good, "garbage one", "garbage two", good])
        a, last = self._both(path)

        self.assertEqual(a.parse_health.decode_failures, 2)
        self.assertEqual(last.parse_health.decode_failures, 0)
        for f in dataclasses.fields(ah.Analysis):
            if f.name == "parse_health":
                continue
            self.assertEqual(
                getattr(a, f.name),
                getattr(last, f.name),
                msg="non-parse_health field diverged: " + f.name,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
