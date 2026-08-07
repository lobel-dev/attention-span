"""Unit tests for subagents.py — discovery + per-agent health rollup.

Builds a synthetic on-disk subagents/ layout (no real Claude Code session needed)
and drives child .jsonl mtimes with os.utime so the liveness filter is deterministic.
"""

import dataclasses
import json
import os
import shutil
import stat
import tempfile
import unittest
from unittest import mock

from builders import GENUINE, asst, read_tu, results, write_lines
from builders import usage as mk_usage

from attention_span import agent_health, session_ui, subagents

NOW = 1_700_000_000.0


STALE = 3_600


class SubagentsLayout(unittest.TestCase):
    """A temp <proj>/<session>.jsonl + <session>/subagents/ scaffold."""

    def setUp(self):
        self.proj = tempfile.mkdtemp()
        self.session = "11111111-2222-3333-4444-555555555555"
        self.parent = os.path.join(self.proj, self.session + ".jsonl")
        write_lines([], self.parent)
        self.subdir = os.path.join(self.proj, self.session, "subagents")
        os.makedirs(self.subdir, exist_ok=True)
        self.addCleanup(shutil.rmtree, self.proj, ignore_errors=True)

        patcher = mock.patch.object(
            subagents, "AGENTS_PREFIX", os.path.join(self.proj, "agents-")
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.memo_path = (
            subagents.AGENTS_PREFIX + session_ui.session_key(self.session) + ".json"
        )

    def add_child(self, agent_id, objs, *, meta=None, age=0, nested=None):
        """Write a child agent-<id>.jsonl (+ optional meta) with mtime = NOW - age.

        ``nested="wf_x"`` places it in the nested workflow layout
        ``subagents/workflows/wf_x/agent-<id>.jsonl`` instead of the flat one.
        """
        d = (
            self.subdir
            if not nested
            else os.path.join(self.subdir, "workflows", nested)
        )
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"agent-{agent_id}.jsonl")
        write_lines(objs, path)
        if meta is not None:
            mp = os.path.join(d, f"agent-{agent_id}.meta.json")
            with open(mp, "w") as f:
                json.dump(meta, f)
        os.utime(path, (NOW - age, NOW - age))
        return path

    def child_reads(self, n, agent_id="a1"):
        return [
            asst([read_tu(i, f"/repo/f{i}.py")], child=True, agent_id=agent_id)
            for i in range(n)
        ]

    def blind_loop_child(self, agent_id, *, done=False):
        """A child stuck in a GENUINE failed-edit loop, so its verdict is red."""
        return [
            asst(
                [("e0", "Edit", {"file_path": "/repo/c.json"})],
                child=True,
                agent_id=agent_id,
            ),
            results([("e0", True, GENUINE)], child=True),
            asst(
                [("e1", "Edit", {"file_path": "/repo/c.json"})],
                child=True,
                agent_id=agent_id,
                stop_reason="end_turn" if done else None,
            ),
        ]

    def child_with_usage(self, agent_id, specs, *, done=False, model="claude-opus-4-8"):
        """A child whose lines carry usage: specs = [(cr, cc, inp, out), ...] with
        one distinct message.id per line (each line = one API call)."""
        specs = tuple(specs)
        return [
            asst(
                [read_tu(i, f"/repo/u{i}.py")],
                child=True,
                agent_id=agent_id,
                model=model,
                usage=mk_usage(*spec),
                msg_id=f"{agent_id}-m{i}",
                stop_reason="end_turn" if done and i == len(specs) - 1 else None,
            )
            for i, spec in enumerate(specs)
        ]


class TestDiscovery(SubagentsLayout):
    def test_subagents_dir_derivation(self):
        self.assertEqual(subagents.subagents_dir(self.parent), self.subdir)

    def test_no_subagents_dir_returns_empty(self):
        other = os.path.join(self.proj, "no-such-session.jsonl")
        write_lines([], other)
        self.assertEqual(subagents.discover(other), [])

    def test_discover_finds_children_excludes_meta(self):
        self.add_child("a1", self.child_reads(6), meta={"agentType": "Explore"})
        found = subagents.discover(self.parent)
        self.assertEqual(len(found), 1)
        self.assertTrue(found[0].endswith("agent-a1.jsonl"))

    def test_discover_finds_nested_workflow_children(self):

        self.add_child("flat", self.child_reads(6))
        self.add_child("wf1", self.child_reads(6), nested="wf_9415f65d-369")
        self.add_child(
            "wf2",
            self.child_reads(6),
            nested="wf_9415f65d-369",
            meta={"agentType": "Task"},
        )
        found = sorted(subagents.discover(self.parent))
        self.assertEqual(len(found), 3)
        self.assertTrue(any("workflows" in p for p in found))
        self.assertTrue(all(p.endswith(".jsonl") for p in found))


class TestChildRecord(SubagentsLayout):
    """The ``Subagent`` record cohort() builds from ONE child transcript."""

    def record(self):
        [agent] = subagents.cohort(self.parent, self.session).live
        return agent

    def test_verdict_runs_per_child(self):
        self.add_child("a1", self.child_reads(6), meta={"agentType": "Explore"}, age=5)
        a = self.record()
        self.assertIsInstance(a, subagents.Subagent)
        self.assertEqual(a.agent_id, "a1")
        self.assertEqual(a.agent_type, "Explore")
        self.assertEqual(a.state, "green")
        self.assertFalse(a.insufficient)

    def test_insufficient_child(self):
        self.add_child("tiny", self.child_reads(2), age=5)
        self.assertTrue(self.record().insufficient)

    def test_missing_meta_degrades_gracefully(self):
        self.add_child("nometa", self.child_reads(6), age=5)
        self.assertEqual(self.record().agent_type, "")

    def _identity_for(self, agent_id, meta):
        path = self.add_child(agent_id, self.child_reads(6), meta=meta, age=5)
        return subagents._identity(path)

    def test_off_schema_agent_type_falls_back_to_unknown(self):

        for i, bad in enumerate(({"name": "Explore"}, ["Explore"], 7, True)):
            with self.subTest(bad=bad):
                identity = self._identity_for(f"at{i}", {"agentType": bad})
                self.assertEqual(identity["agent_type"], "")

    def test_off_schema_spawn_depth_is_unknown(self):

        for i, bad in enumerate(("2", 2.5, True, [2], None)):
            with self.subTest(bad=bad):
                identity = self._identity_for(f"sd{i}", {"spawnDepth": bad})
                self.assertIsNone(identity["spawn_depth"])

    def test_well_formed_meta_survives_validation(self):
        identity = self._identity_for("ok", {"agentType": "Explore", "spawnDepth": 2})
        self.assertEqual(identity["agent_type"], "Explore")
        self.assertEqual(identity["spawn_depth"], 2)

    def test_child_model_is_the_transcripts_last_assistant_model(self):

        self.add_child("m1", self.child_with_usage("m1", [(0, 0, 1, 1)]), age=5)
        self.assertEqual(self.record().model, "claude-opus-4-8")

    def test_degraded_child_claims_no_model(self):

        path = self.add_child("bad", self.child_reads(1), age=5)
        with open(path, "a") as f:
            f.write("not json\n{ also broken\n")
        os.utime(path, (NOW - 5, NOW - 5))
        self.assertEqual(self.record().model, "")

    def test_child_red_surfaces(self):

        objs = [
            asst([read_tu(0, "/repo/a.py")], child=True, agent_id="r"),
            asst([read_tu(1, "/repo/b.py")], child=True, agent_id="r"),
            asst(
                [("e0", "Edit", {"file_path": "/repo/c.json"})],
                child=True,
                agent_id="r",
            ),
            results([("e0", True, GENUINE)], child=True),
            asst(
                [("e1", "Edit", {"file_path": "/repo/c.json"})],
                child=True,
                agent_id="r",
            ),
            results([("e1", True, GENUINE)], child=True),
            asst(
                [("e2", "Edit", {"file_path": "/repo/c.json"})],
                child=True,
                agent_id="r",
            ),
        ]
        self.add_child("r", objs, age=5)
        self.assertEqual(self.record().state, "red")

    def test_blind_loop_child_red_even_when_insufficient(self):

        objs = [
            asst(
                [("e0", "Edit", {"file_path": "/repo/c.json"})],
                child=True,
                agent_id="b",
            ),
            results([("e0", True, GENUINE)], child=True),
            asst(
                [("e1", "Edit", {"file_path": "/repo/c.json"})],
                child=True,
                agent_id="b",
            ),
        ]
        self.add_child("b", objs, age=5)
        out = self.record()
        self.assertEqual(out.state, "red")
        self.assertTrue(out.insufficient)

    def test_edit_skewed_child_is_green_not_yellow(self):

        objs = [asst([read_tu(0, "/repo/r.py")], child=True, agent_id="e")]
        objs += [
            asst(
                [(f"x{i}", "Edit", {"file_path": f"/repo/f{i}.py"})],
                child=True,
                agent_id="e",
            )
            for i in range(7)
        ]
        self.add_child("e", objs, age=5)
        out = self.record()
        self.assertFalse(out.insufficient)
        self.assertEqual(out.state, "green")


class TestCohort(SubagentsLayout):
    """cohort(): live/done partition, per-session memo (parse-once), burn totals."""

    SPEC_A = (1000, 500, 10, 20)
    SPEC_B = (2000, 100, 5, 50)

    def cohort(self, **kw):
        return subagents.cohort(self.parent, self.session, **kw)

    def test_partitions_live_vs_done(self):
        self.add_child("live1", self.child_with_usage("live1", [self.SPEC_A]), age=5)
        self.add_child(
            "old1",
            self.child_with_usage("old1", [self.SPEC_B], done=True),
            age=STALE,
        )
        co = self.cohort()
        self.assertEqual([a.agent_id for a in co.live], ["live1"])
        self.assertEqual(co.done_n, 1)
        self.assertEqual(co.tokens_total, 530 + 155)
        self.assertEqual(co.total_n, 2)
        self.assertEqual(co.live_n, 1)
        self.assertEqual(co.tokens_known_n, 2)
        self.assertEqual(co.tokens_untrusted_n, 0)
        self.assertFalse(co.tokens_warming)
        self.assertTrue(co.tokens_complete)

    def test_completion_state_overrides_transcript_age(self):
        self.add_child(
            "fresh-done",
            self.child_with_usage("fresh-done", [self.SPEC_A], done=True),
            age=5,
        )
        self.add_child(
            "stale-working",
            self.child_with_usage("stale-working", [self.SPEC_B]),
            age=STALE,
        )

        co = self.cohort()

        self.assertEqual([agent.agent_id for agent in co.live], ["stale-working"])
        self.assertEqual(co.live_n, 1)
        self.assertEqual(co.done_n, 1)
        self.assertEqual(co.total_n, 2)

    def test_inventory_counts_are_uncapped_from_live_health_list(self):
        for i in range(5):
            self.add_child(
                f"live{i}",
                self.child_with_usage(f"live{i}", [self.SPEC_A]),
                age=5 + i,
            )
        co = self.cohort(cap=2)
        self.assertEqual(len(co.live), 2)
        self.assertEqual(co.live_n, 5)
        self.assertEqual(co.total_n, 5)

    def test_live_records_are_subagents_carrying_identity_and_burn(self):
        self.add_child(
            "a1",
            self.child_with_usage("a1", [self.SPEC_A]),
            meta={"agentType": "Explore"},
            age=5,
        )
        [a] = self.cohort().live
        self.assertIsInstance(a, subagents.Subagent)
        self.assertEqual(a.agent_id, "a1")
        self.assertEqual(a.agent_type, "Explore")
        self.assertEqual(a.burn, 530)

    def test_live_detail_is_capped_at_the_newest_children(self):
        for i in range(subagents.MAX_AGENTS + 3):
            self.add_child(
                f"a{i}",
                self.child_with_usage(f"a{i}", [self.SPEC_A]),
                age=i,
            )
        co = self.cohort()
        ids = [a.agent_id for a in co.live]
        self.assertEqual(len(co.live), subagents.MAX_AGENTS)
        self.assertIn("a0", ids)
        self.assertNotIn(f"a{subagents.MAX_AGENTS + 2}", ids)
        self.assertEqual(co.live_n, subagents.MAX_AGENTS + 3)

    def test_nested_workflow_children_counted(self):
        self.add_child(
            "wf1", self.child_with_usage("wf1", [self.SPEC_A]), nested="wf_x", age=5
        )
        self.add_child(
            "wf2",
            self.child_with_usage("wf2", [self.SPEC_B], done=True),
            nested="wf_x",
            age=STALE,
        )
        co = self.cohort()
        self.assertEqual([a.agent_id for a in co.live], ["wf1"])
        self.assertEqual(co.done_n, 1)
        self.assertEqual(co.tokens_total, 685)

    def test_models_are_the_distinct_models_of_unfinished_children(self):

        self.add_child(
            "l1",
            self.child_with_usage("l1", [self.SPEC_A], model="claude-opus-5"),
            age=5,
        )
        self.add_child(
            "l2",
            self.child_with_usage("l2", [self.SPEC_A], model="claude-fable-5"),
            age=6,
        )
        self.add_child(
            "l3",
            self.child_with_usage("l3", [self.SPEC_A], model="claude-opus-5"),
            age=7,
        )
        self.add_child(
            "d1",
            self.child_with_usage(
                "d1", [self.SPEC_B], done=True, model="claude-sonnet-5"
            ),
            age=STALE,
        )
        co = self.cohort()
        self.assertEqual(co.models, ("claude-fable-5", "claude-opus-5"))

    def test_models_survive_the_memo_round_trip(self):
        self.add_child(
            "l1",
            self.child_with_usage("l1", [self.SPEC_A], model="claude-opus-5"),
            age=5,
        )
        first = self.cohort()
        with mock.patch.object(
            subagents.agent_health,
            "analyze_transcript",
            side_effect=AssertionError("must not re-parse"),
        ):
            second = self.cohort()
        self.assertEqual(second.models, first.models)
        self.assertEqual(second.models, ("claude-opus-5",))

    def test_legacy_memo_entry_without_model_stays_a_hit_then_learns_on_change(self):

        path = self.add_child(
            "a1",
            self.child_with_usage("a1", [self.SPEC_A], model="claude-opus-5"),
            age=5,
        )
        st = os.stat(path)
        legacy = {
            path: {
                "mtime": st.st_mtime,
                "size": st.st_size,
                "burn": 530,
                "burn_trusted": True,
                "context_tokens": 0,
                "blind_loop": False,
                "state": "green",
                "insufficient": False,
                "chip": "",
                "done": False,
            }
        }
        with open(self.memo_path, "w") as f:
            json.dump(legacy, f)
        with mock.patch.object(
            subagents.agent_health,
            "analyze_transcript",
            side_effect=AssertionError("legacy entries must not mass re-parse"),
        ):
            co = self.cohort()
        self.assertEqual(co.models, ())
        self.assertEqual(co.tokens_total, 530)
        self.assertEqual(co.live_n, 1)

        self.add_child(
            "a1",
            self.child_with_usage(
                "a1", [self.SPEC_A, self.SPEC_B], model="claude-opus-5"
            ),
            age=5,
        )
        co = self.cohort()
        self.assertEqual(co.models, ("claude-opus-5",))
        with open(self.memo_path) as f:
            healed = json.load(f)
        self.assertEqual(healed[path]["model"], "claude-opus-5")

    def test_wrong_typed_model_memo_entry_is_a_miss(self):

        path = self.add_child(
            "a1",
            self.child_with_usage("a1", [self.SPEC_A], model="claude-opus-5"),
            age=5,
        )
        st = os.stat(path)
        poisoned = {
            path: {
                "mtime": st.st_mtime,
                "size": st.st_size,
                "burn": 530,
                "burn_trusted": True,
                "context_tokens": 0,
                "blind_loop": False,
                "state": "green",
                "insufficient": False,
                "chip": "",
                "done": False,
                "model": 7,
            }
        }
        with open(self.memo_path, "w") as f:
            json.dump(poisoned, f)
        co = self.cohort()
        self.assertEqual(co.models, ("claude-opus-5",))
        with open(self.memo_path) as f:
            healed = json.load(f)
        self.assertEqual(healed[path]["model"], "claude-opus-5")

    def test_memo_hit_serves_without_reparse(self):

        self.add_child("live1", self.child_with_usage("live1", [self.SPEC_A]), age=5)
        self.add_child(
            "old1",
            self.child_with_usage("old1", [self.SPEC_B], done=True),
            age=STALE,
        )
        first = self.cohort()
        self.assertTrue(os.path.exists(self.memo_path))
        calls = {"n": 0}
        real = agent_health.analyze_transcript

        def counting(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        with mock.patch.object(
            subagents.agent_health, "analyze_transcript", side_effect=counting
        ):
            second = self.cohort()
        self.assertEqual(calls["n"], 0)
        self.assertEqual(second.tokens_total, first.tokens_total)
        self.assertEqual(second.done_n, first.done_n)
        self.assertEqual(
            [a.agent_id for a in second.live],
            [a.agent_id for a in first.live],
        )

    def test_memo_full_hit_does_zero_writes(self):
        self.add_child(
            "old1",
            self.child_with_usage("old1", [self.SPEC_B], done=True),
            age=STALE,
        )
        self.cohort()
        mtime = os.path.getmtime(self.memo_path)
        os.utime(self.memo_path, (mtime - 100, mtime - 100))
        self.cohort()
        self.assertEqual(os.path.getmtime(self.memo_path), mtime - 100)

    def test_changed_child_reparses_and_upserts(self):
        path = self.add_child("g", self.child_with_usage("g", [self.SPEC_A]), age=5)
        self.assertEqual(self.cohort().tokens_total, 530)

        self.add_child(
            "g", self.child_with_usage("g", [self.SPEC_A, self.SPEC_B]), age=5
        )
        calls = {"n": 0}
        real = agent_health.analyze_transcript

        def counting(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        with mock.patch.object(
            subagents.agent_health, "analyze_transcript", side_effect=counting
        ):
            co = self.cohort()
        self.assertEqual(calls["n"], 1)
        self.assertEqual(co.tokens_total, 530 + 155)

        with mock.patch.object(
            subagents.agent_health, "analyze_transcript", side_effect=counting
        ):
            self.cohort()
        self.assertEqual(calls["n"], 1)
        self.assertTrue(path)

    def test_corrupt_memo_rebuilds_without_raise(self):
        self.add_child("a1", self.child_with_usage("a1", [self.SPEC_A]), age=5)
        with open(self.memo_path, "w") as f:
            f.write("{ not json at all")
        co = self.cohort()
        self.assertEqual(co.tokens_total, 530)
        with open(self.memo_path) as f:
            self.assertTrue(isinstance(json.load(f), dict))

    def test_wrong_typed_memo_entry_reparses_and_self_heals(self):

        path = self.add_child("a1", self.child_with_usage("a1", [self.SPEC_A]), age=5)
        st = os.stat(path)
        poisoned = {
            path: {
                "mtime": st.st_mtime,
                "size": st.st_size,
                "burn": "x",
                "context_tokens": 0,
                "blind_loop": False,
                "state": "green",
                "insufficient": False,
                "chip": "",
                "done": False,
            }
        }
        with open(self.memo_path, "w") as f:
            json.dump(poisoned, f)
        co = self.cohort()
        self.assertEqual(co.tokens_total, 530)
        self.assertEqual([a.agent_id for a in co.live], ["a1"])
        with open(self.memo_path) as f:
            healed = json.load(f)
        self.assertEqual(healed[path]["burn"], 530)

    def test_oversized_numeric_memo_entry_reparses_and_self_heals(self):
        path = self.add_child("a1", self.child_with_usage("a1", [self.SPEC_A]), age=5)
        st = os.stat(path)
        poisoned = {
            path: {
                "mtime": st.st_mtime,
                "size": st.st_size,
                "burn": 10**400,
                "burn_trusted": True,
                "context_tokens": 0,
                "blind_loop": False,
                "state": "green",
                "insufficient": False,
                "chip": "",
                "done": False,
            }
        }
        with open(self.memo_path, "w") as f:
            json.dump(poisoned, f)
        co = self.cohort()
        self.assertEqual(co.tokens_total, 530)
        with open(self.memo_path) as f:
            healed = json.load(f)
        self.assertEqual(healed[path]["burn"], 530)

    def test_memo_write_failure_is_silent(self):
        self.add_child("a1", self.child_with_usage("a1", [self.SPEC_A]), age=5)
        unwritable = os.path.join(self.proj, "no-such-dir", "agents-")
        with mock.patch.object(subagents, "AGENTS_PREFIX", unwritable):
            co = self.cohort()
        self.assertEqual(co.tokens_total, 530)

    def test_crashed_memo_write_leaves_previous_memo_intact(self):

        self.add_child("a1", self.child_with_usage("a1", [self.SPEC_A]), age=5)
        self.cohort()
        with open(self.memo_path) as f:
            before = json.load(f)
        self.assertTrue(before)

        self.add_child(
            "a1", self.child_with_usage("a1", [self.SPEC_A, self.SPEC_B]), age=5
        )
        with mock.patch.object(subagents.json, "dump", side_effect=OSError("disk")):
            self.cohort()
        with open(self.memo_path) as f:
            self.assertEqual(json.load(f), before)

    def test_memo_is_private_under_a_typical_shared_host_umask(self):
        self.add_child("a1", self.child_with_usage("a1", [self.SPEC_A]), age=5)
        old_umask = os.umask(0o022)
        self.addCleanup(os.umask, old_umask)

        self.cohort()

        self.assertEqual(stat.S_IMODE(os.stat(self.memo_path).st_mode), 0o600)

    def test_cohort_uses_shared_opaque_session_key(self):
        self.add_child("a1", self.child_with_usage("a1", [self.SPEC_A]), age=5)
        session_id = "../../private/session?name"
        expected = (
            subagents.AGENTS_PREFIX + session_ui.session_key(session_id) + ".json"
        )

        with mock.patch.object(subagents, "_atomic_write_json") as write_memo:
            subagents.cohort(self.parent, session_id)

        write_memo.assert_called_once()
        self.assertEqual(write_memo.call_args.args[0], expected)
        self.assertNotIn(session_id, expected)

    def test_burn_excludes_cache_read(self):

        spec = (1744563, 163973, 762, 8789)
        self.add_child("big", self.child_with_usage("big", [spec]), age=5)
        co = self.cohort()
        self.assertEqual(co.tokens_total, 762 + 163973 + 8789)

    def test_cold_memo_parses_bounded_number_of_finished_children(self):
        total = 7
        for i in range(total):
            self.add_child(
                f"old{i}",
                self.child_with_usage(f"old{i}", [self.SPEC_A], done=True),
                age=STALE + i,
            )
        calls = {"n": 0}
        real = subagents._analyze_child

        def counting(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        with (
            mock.patch.object(subagents, "COHORT_PARSE_BUDGET", 3),
            mock.patch.object(subagents, "_analyze_child", side_effect=counting),
        ):
            co = self.cohort()
        self.assertEqual(calls["n"], 3)
        self.assertEqual(co.done_n, 3)
        self.assertEqual(co.live_n, 4)
        self.assertEqual(co.tokens_total, 3 * 530)
        self.assertEqual(co.tokens_known_n, 3)
        self.assertEqual(co.tokens_untrusted_n, 0)
        self.assertTrue(co.tokens_warming)
        self.assertFalse(co.tokens_complete)
        with open(self.memo_path) as f:
            self.assertEqual(len(json.load(f)), 3)

        with mock.patch.object(subagents, "COHORT_PARSE_BUDGET", 3):
            self.cohort()
            warmed = self.cohort()
        self.assertEqual(warmed.tokens_known_n, total)
        self.assertFalse(warmed.tokens_warming)
        self.assertTrue(warmed.tokens_complete)
        self.assertEqual(warmed.tokens_total, total * 530)

    def test_unparsed_stale_children_are_not_claimed_as_done(self):
        total = 7
        for i in range(total):
            self.add_child(
                f"old-working{i}",
                self.child_with_usage(f"old-working{i}", [self.SPEC_A]),
                age=STALE + i,
            )

        with mock.patch.object(subagents, "COHORT_PARSE_BUDGET", 3):
            co = self.cohort()

        self.assertEqual(co.done_n, 0)
        self.assertEqual(co.live_n, total)
        self.assertTrue(co.tokens_warming)

    def test_parse_degraded_child_is_neutral_and_contributes_no_burn(self):
        path = self.add_child("bad", [], age=5)
        good = asst(
            [read_tu(0)],
            child=True,
            agent_id="bad",
            usage=mk_usage(*self.SPEC_A),
            msg_id="bad-m0",
        )
        with open(path, "w") as f:
            f.write(json.dumps(good) + "\n")
            f.write("not json\n")
            f.write("{ also broken\n")
        os.utime(path, (NOW - 5, NOW - 5))
        co = self.cohort()
        [a] = co.live
        self.assertEqual(a.agent_id, "bad")
        self.assertTrue(a.insufficient)
        self.assertEqual(a.state, "green")
        self.assertEqual(a.burn, 0)
        self.assertEqual(co.tokens_total, 0)
        self.assertEqual(co.tokens_known_n, 1)
        self.assertEqual(co.tokens_untrusted_n, 1)
        self.assertFalse(co.tokens_warming)
        self.assertFalse(co.tokens_complete)
        with mock.patch.object(
            subagents.agent_health,
            "analyze_transcript",
            side_effect=AssertionError("must not re-parse"),
        ):
            cached = self.cohort()
        self.assertFalse(cached.tokens_complete)
        self.assertEqual(cached.tokens_untrusted_n, 1)

    def test_blind_loop_child_red_via_memo(self):

        objs = [
            asst(
                [("e0", "Edit", {"file_path": "/repo/c.json"})],
                child=True,
                agent_id="r",
            ),
            results([("e0", True, GENUINE)], child=True),
            asst(
                [("e1", "Edit", {"file_path": "/repo/c.json"})],
                child=True,
                agent_id="r",
            ),
        ]
        self.add_child("r", objs, age=5)
        self.assertEqual(self.cohort().live[0].state, "red")
        with mock.patch.object(
            subagents.agent_health,
            "analyze_transcript",
            side_effect=AssertionError("must not re-parse"),
        ):
            [a] = self.cohort().live
        self.assertEqual(a.state, "red")
        self.assertIn("EDIT LOOP", a.chip)

    def test_a_red_child_beyond_the_cap_still_warns(self):

        for i in range(2):
            self.add_child(
                f"live{i}", self.child_with_usage(f"live{i}", [self.SPEC_A]), age=5 + i
            )
        self.add_child("old-red", self.blind_loop_child("old-red"), age=STALE)

        co = self.cohort(cap=2)

        self.assertEqual(len(co.live), 2)
        self.assertNotIn("old-red", [a.agent_id for a in co.live])
        self.assertEqual(co.live_n, 3)
        self.assertEqual(co.blind_loop_n, 1)
        with mock.patch.object(
            subagents.agent_health,
            "analyze_transcript",
            side_effect=AssertionError("must not re-parse"),
        ):
            self.assertEqual(self.cohort(cap=2).blind_loop_n, 1)

    def test_a_child_past_the_parse_budget_is_never_claimed_red(self):

        for i in range(3):
            self.add_child(
                f"live{i}", self.child_with_usage(f"live{i}", [self.SPEC_A]), age=5 + i
            )
        self.add_child("old-red", self.blind_loop_child("old-red"), age=STALE)

        with mock.patch.object(subagents, "COHORT_PARSE_BUDGET", 3):
            co = self.cohort()

        self.assertEqual(co.live_n, 4)
        self.assertEqual(co.blind_loop_n, 0)

    def test_a_finished_red_child_stops_warning(self):

        self.add_child("done-red", self.blind_loop_child("done-red", done=True), age=5)

        co = self.cohort()

        self.assertEqual(co.done_n, 1)
        self.assertEqual(co.blind_loop_n, 0)
        with open(self.memo_path) as f:
            [entry] = json.load(f).values()
        self.assertEqual(entry["state"], "red")

    def test_empty_layout_returns_empty_cohort(self):
        self.assertEqual(self.cohort(), subagents.Cohort())
        self.assertFalse(os.path.exists(self.memo_path))


class TestNotifiedRetirement(SubagentsLayout):
    """A child stops when EITHER its own transcript proves end_turn OR the parent
    recorded a task-notification for it after the child's last write.

    The zombie this covers: torn down (or TaskStop'd) once its result was captured, so
    its own transcript ends on a tool_use and never records end_turn. Timestamps
    arbitrate resumption - a resumed child writes past its notification and is live
    again - so nothing here is a permanent done bit.
    """

    SPEC_A = (1000, 500, 10, 20)
    SPEC_B = (2000, 100, 5, 50)

    def cohort(self, **kw):
        return subagents.cohort(self.parent, self.session, **kw)

    def test_a_notified_child_is_retired_without_its_own_end_turn(self):
        self.add_child(
            "zombie",
            self.child_with_usage("zombie", [self.SPEC_A], model="claude-opus-5"),
            age=STALE,
        )
        self.add_child(
            "busy",
            self.child_with_usage("busy", [self.SPEC_B], model="claude-fable-5"),
            age=5,
        )

        co = self.cohort(notified={"zombie": NOW - STALE + 1})

        self.assertEqual([a.agent_id for a in co.live], ["busy"])
        self.assertEqual(co.live_n, 1)
        self.assertEqual(co.done_n, 1)
        self.assertEqual(co.total_n, 2)
        self.assertEqual(co.models, ("claude-fable-5",))
        self.assertEqual(co.tokens_total, 530 + 155)

    def test_a_notification_at_the_childs_own_mtime_retires_it(self):

        self.add_child("edge", self.child_with_usage("edge", [self.SPEC_A]), age=5)

        self.assertEqual(self.cohort(notified={"edge": NOW - 5}).done_n, 1)

    def test_a_child_writing_past_its_notification_is_live_again(self):

        self.add_child(
            "resumed", self.child_with_usage("resumed", [self.SPEC_A]), age=5
        )

        co = self.cohort(notified={"resumed": NOW - 5 - 1})

        self.assertEqual([a.agent_id for a in co.live], ["resumed"])
        self.assertEqual(co.done_n, 0)
        self.assertEqual(co.live_n, 1)

    def test_a_notification_for_an_unknown_agent_id_changes_nothing(self):
        self.add_child("a1", self.child_with_usage("a1", [self.SPEC_A]), age=5)

        co = self.cohort(notified={"never-spawned-here": NOW})

        self.assertEqual([a.agent_id for a in co.live], ["a1"])
        self.assertEqual(co.done_n, 0)
        self.assertEqual(co.total_n, 1)

    def test_retirement_is_derived_per_render_and_never_stored_in_the_memo(self):

        path = self.add_child("z", self.child_with_usage("z", [self.SPEC_A]), age=STALE)

        self.assertEqual(self.cohort(notified={"z": NOW}).done_n, 1)

        with open(self.memo_path) as f:
            entry = json.load(f)[path]
        self.assertFalse(entry["done"])

        with mock.patch.object(
            subagents.agent_health,
            "analyze_transcript",
            side_effect=AssertionError("must not re-parse"),
        ):
            self.assertEqual(self.cohort(notified={"z": NOW}).done_n, 1)
            self.assertEqual(self.cohort().done_n, 0)

    def test_an_end_turn_child_stays_done_with_no_notification_at_all(self):
        self.add_child(
            "d1",
            self.child_with_usage("d1", [self.SPEC_A], done=True),
            age=5,
        )

        self.assertEqual(self.cohort(notified={}).done_n, 1)
        self.assertEqual(self.cohort(notified=None).done_n, 1)

    def test_a_notified_red_child_stops_raising_the_blind_loop_warning(self):

        self.add_child("red-zombie", self.blind_loop_child("red-zombie"), age=STALE)

        self.assertEqual(self.cohort().blind_loop_n, 1)

        co = self.cohort(notified={"red-zombie": NOW - STALE + 1})

        self.assertEqual(co.done_n, 1)
        self.assertEqual(co.blind_loop_n, 0)
        self.assertEqual(co.live, ())


class TestCohortDerivations(unittest.TestCase):
    """The counts every reader shares, derived from the gathered facts in ONE place."""

    def test_live_n_is_every_child_not_proven_done(self):
        self.assertEqual(subagents.Cohort(total_n=7, done_n=3).live_n, 4)

    def test_capped_live_detail_never_stands_in_for_the_count(self):
        co = subagents.Cohort(live=(subagents.Subagent(),) * 2, total_n=50)
        self.assertEqual(len(co.live), 2)
        self.assertEqual(co.live_n, 50)

    def test_burn_warms_until_every_child_is_parsed(self):
        co = subagents.Cohort(total_n=7, tokens_known_n=3)
        self.assertTrue(co.tokens_warming)
        self.assertFalse(co.tokens_complete)

    def test_a_parsed_but_untrusted_burn_is_settled_yet_incomplete(self):
        co = subagents.Cohort(total_n=2, tokens_known_n=2, tokens_untrusted_n=1)
        self.assertFalse(co.tokens_warming)
        self.assertFalse(co.tokens_complete)

    def test_the_empty_cohort_is_idle_and_complete(self):
        co = subagents.Cohort()
        self.assertEqual((co.total_n, co.live_n, co.done_n), (0, 0, 0))
        self.assertFalse(co.tokens_warming)
        self.assertTrue(co.tokens_complete)

    def test_records_are_frozen(self):
        for value in (subagents.Cohort(), subagents.Subagent()):
            with (
                self.subTest(value=value),
                self.assertRaises(dataclasses.FrozenInstanceError),
            ):
                value.total_n = 3


if __name__ == "__main__":
    unittest.main(verbosity=2)
