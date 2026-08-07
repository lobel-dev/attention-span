"""End-to-end regressions for statusline.py — the Python orchestrator.

statusline.py reads the statusLine stdin JSON and runs the whole pipeline (cold-start
marker, 10s cache, per-session cost baseline, engine + renderer) in ONE process. These
tests pipe a synthetic payload to `python3 statusline.py` and assert on the rendered panel.

An account-usage half of this file went on 2026-08-06 with the layer it covered: the
quota-axi snapshot fixtures, the throttled-`npx`-refresh race repro, the account-chip
gatherer unit tests, and every end-to-end assertion that a `rate_limits` payload put a
`7D`/`FABLE`/`5H` fact on Row 1. What replaced them is one guard - the orchestrator must
IGNORE `rate_limits` outright - because the property worth protecting is now absence.
"""

import json
import os
import runpy
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from bands import (
    LOAD_200K_DEGRADING,
    LOAD_DEGRADING,
    LOAD_DEGRADING_S,
    LOAD_FUNCTIONAL,
    LOAD_FUNCTIONAL_S,
)
from builders import (
    THRASH_SPECS,
    asst,
    cache_objs,
    edit_tu,
    iso_ts,
    read_tu,
    synthetic_api_error,
    system_compact_boundary,
    task_notification,
    user_compact_summary,
    write_lines,
)
from builders import usage as mk_usage

from attention_span import (
    agent_health,
    health_config,
    render,
    status_catalog,
    statusline,
    subagents,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ID = "claude-opus-4-8"


class TestStatusline(unittest.TestCase):
    """Pipe a synthetic transcript + payload through `python3 statusline.py`."""

    def test_orchestrator_uses_the_package_renderer(self):
        self.assertEqual(statusline.__name__, "attention_span.statusline")
        self.assertIs(statusline.render, render)

    def test_root_shim_inserts_release_root_and_delegates_to_package_main(self):
        shim_path = os.path.join(ROOT, "statusline.py")
        shim_sys_path = list(sys.path)
        root_count = shim_sys_path.count(ROOT)
        with (
            mock.patch.object(sys, "path", shim_sys_path),
            mock.patch.object(statusline, "main") as package_main,
        ):
            runpy.run_path(shim_path, run_name="__main__")
            self.assertEqual(sys.path[0], ROOT)
            self.assertEqual(sys.path.count(ROOT), root_count + 1)
        package_main.assert_called_once_with()

    def setUp(self):

        self._home = tempfile.mkdtemp()
        self._claude_home = os.path.join(self._home, ".claude")
        self.addCleanup(shutil.rmtree, self._home, ignore_errors=True)

    def _env(self, extra=None):
        env = dict(os.environ)

        for k in [k for k in env if k.startswith("CLAUDE_HEALTH_")]:
            del env[k]
        env.update(
            {
                "HOME": self._home,
                "CLAUDE_HOME": self._claude_home,
                "NO_COLOR": "1",
                "COLUMNS": "200",
            }
        )
        if extra:
            env.update(extra)
        return env

    def _run(self, payload, env_extra=None):
        proc = subprocess.run(
            [sys.executable, os.path.join(ROOT, "statusline.py")],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            cwd=ROOT,
            env=self._env(env_extra),
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc

    def _make_transcript(self, objs):
        transcript = write_lines(objs)
        self.addCleanup(lambda: os.path.exists(transcript) and os.unlink(transcript))
        session_id = os.path.splitext(os.path.basename(transcript))[0]
        cache = self._cache_path(transcript)
        legacy_cache = os.path.join("/tmp", "claude-statusline-" + session_id)
        ui_state = statusline.session_ui.UI_PREFIX + statusline.session_ui.session_key(
            session_id
        )
        for path in (cache, legacy_cache, ui_state):
            self.addCleanup(lambda p=path: os.path.exists(p) and os.unlink(p))
        return transcript

    def _cache_path(self, transcript):
        session_id = os.path.splitext(os.path.basename(transcript))[0]
        return os.path.join(
            self._claude_home,
            "hooks",
            "attention-span",
            "cache",
            "render-" + statusline.session_ui.session_key(session_id) + ".json",
        )

    def _invoke(self, transcript, payload_extra=None, env_extra=None):
        """Run once for an existing transcript. Clears the 10s render cache first (so a
        second invocation in one test re-renders) but PRESERVES the cost baseline across
        invocations — the snapshot-and-subtract cost chip relies on it persisting."""
        session_id = os.path.splitext(os.path.basename(transcript))[0]
        caches = (
            self._cache_path(transcript),
            os.path.join("/tmp", "claude-statusline-" + session_id),
        )
        for cache in caches:
            if os.path.lexists(cache):
                os.unlink(cache)
        payload = {"transcript_path": transcript, "model": {"id": MODEL_ID}}
        if payload_extra:
            payload.update(payload_extra)
        return self._run(payload, env_extra=env_extra)

    def _run_statusline(self, objs, payload_extra=None, env_extra=None):
        return self._invoke(self._make_transcript(objs), payload_extra, env_extra)

    def test_large_transcript_tokens_lead_context_dead(self):

        objs = [asst([read_tu(0, "/repo/r.py")], usage={"input_tokens": 600_000})]
        objs += [asst([edit_tu(i, f"/repo/e{i}.py")]) for i in range(8)]
        proc = self._run_statusline(objs)
        self.assertIn("600K", proc.stdout)
        self.assertIn("DEAD", proc.stdout)
        self.assertNotIn("Close Watch", proc.stdout)

    def test_payload_metadata_populates_second_row(self):
        cwd = os.path.join(self._home, "dev", "barnett-l", "attention-span")
        proc = self._run_statusline(
            [asst([], usage={"input_tokens": 50_000})],
            {
                "model": {"id": "claude-opus-4-8[1m]"},
                "effort": {"level": "max"},
                "cwd": cwd,
                "workspace": {"repo": {"name": "attention-span"}},
                "pr": {
                    "number": 123,
                    "url": "https://example.test/pull/123",
                    "review_state": "approved",
                },
            },
        )
        self.assertEqual(
            proc.stdout.splitlines()[1],
            "╰─ ~/dev/barnett-l/attention-span   OPUS 4.8 / 1M · MAX",
        )

    def test_blind_loop_pins_ahead_of_context(self):

        objs = [
            asst([read_tu(0, "/repo/a.py")], usage={"input_tokens": 60_000}),
            asst([read_tu(1, "/repo/b.py")]),
            asst([("e0", "Edit", {"file_path": "/repo/cfg.json"})]),
        ]
        from builders import GENUINE, results

        objs += [
            results([("e0", True, GENUINE)]),
            asst([("e1", "Edit", {"file_path": "/repo/cfg.json"})]),
            results([("e1", True, GENUINE)]),
            asst(
                [("e2", "Edit", {"file_path": "/repo/cfg.json"})],
                usage={"input_tokens": 60_000},
            ),
        ]
        proc = self._run_statusline(objs)
        self.assertIn("READ FILE, THEN RETRY", proc.stdout)
        self.assertIn("cfg.json", proc.stdout)
        self.assertIn("60K", proc.stdout)

    def test_no_signal_renders_warming_up(self):

        objs = [
            asst([read_tu(0, "/repo/a.py")]),
            asst([read_tu(1, "/repo/b.py")]),
            asst([edit_tu(0, "/repo/a.py")]),
        ]
        proc = self._run_statusline(objs)
        self.assertIn("WAIT FOR SESSION DATA", proc.stdout)

    def test_healthy_row_leads_with_context_light(self):

        objs = [asst([read_tu(i, f"/repo/f{i}.py")]) for i in range(8)]
        objs += [asst([edit_tu(0, "/repo/f0.py")])]
        objs += [asst([edit_tu(1, "/repo/f1.py")], usage={"input_tokens": 50_000})]
        proc = self._run_statusline(
            objs,
            {
                "context_window": {"used_percentage": 18},
                "cost": {"total_cost_usd": 1.20},
            },
        )
        self.assertIn("🌕 PEAK", proc.stdout)
        self.assertIn("18%", proc.stdout)
        self.assertNotIn(" │ ", proc.stdout)

    def _zombie_layout(self, session, notify):
        """A real <proj>/<session>.jsonl + subagents/ tree: one child that never
        recorded end_turn, one that is genuinely still working, and optionally the
        parent's task-notification for the first one."""
        proj = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, proj, ignore_errors=True)
        parent_objs = [asst([read_tu(i, f"/repo/f{i}.py")]) for i in range(8)]
        parent_objs += [asst([edit_tu(0, "/repo/f0.py")], usage=mk_usage(inp=50_000))]
        if notify:
            parent_objs += [task_notification("zombie", timestamp=iso_ts(600))]
        parent = os.path.join(proj, session + ".jsonl")
        write_lines(parent_objs, parent)

        subdir = os.path.join(proj, session, "subagents")
        os.makedirs(subdir)
        for agent_id, offset in (("zombie", 0), ("busy", 900)):
            child = os.path.join(subdir, f"agent-{agent_id}.jsonl")
            write_lines(
                [
                    asst(
                        [read_tu(i, f"/repo/c{i}.py")],
                        child=True,
                        agent_id=agent_id,
                        usage=mk_usage(inp=10, out=20),
                        msg_id=f"{agent_id}-m{i}",
                    )
                    for i in range(6)
                ],
                child,
            )

            stamp = agent_health._ts_epoch(iso_ts(offset))
            os.utime(child, (stamp, stamp))
        return parent

    def test_a_notified_child_is_not_counted_as_working(self):
        without = self._invoke(self._zombie_layout("zombie-control", notify=False))
        with_notice = self._invoke(self._zombie_layout("zombie-notified", notify=True))

        self.assertIn("WORKING 2", without.stdout)
        self.assertIn("WORKING 1", with_notice.stdout)
        self.assertNotIn("WORKING 2", with_notice.stdout)

    def test_cold_start_marker_is_shown_and_never_cached(self):

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        missing = os.path.join(d, "fresh-session.jsonl")
        cost_base = "/tmp/claude-statusline-cost-fresh-session"
        cache = "/tmp/claude-statusline-fresh-session"
        for p in (cost_base, cache):
            self.addCleanup(lambda p=p: os.path.exists(p) and os.unlink(p))
        proc = self._run(
            {
                "transcript_path": missing,
                "model": {"id": "claude-opus-4-8[1m]"},
                "effort": {"level": "max"},
                "cwd": os.path.join(self._home, "dev", "attention-span"),
                "workspace": {"repo": {"name": "attention-span"}},
                "cost": {"total_cost_usd": 7.0},
            }
        )
        self.assertIn("WAIT FOR SESSION DATA", proc.stdout)

        self.assertEqual(
            proc.stdout.splitlines()[0],
            "╭─ ◌ WAIT FOR SESSION DATA ──────────────────",
        )
        self.assertEqual(
            proc.stdout.splitlines()[1],
            "╰─ ~/dev/attention-span   OPUS 4.8 / 1M · MAX",
        )
        self.assertFalse(os.path.exists(cache))
        self.assertFalse(os.path.exists(cost_base))

        tiny = self._run(
            {
                "transcript_path": missing,
                "model": {"id": "claude-opus-4-8[1m]"},
                "effort": {"level": "max"},
            },
            env_extra={"COLUMNS": "9"},
        )
        self.assertTrue(all(len(line) <= 9 for line in tiny.stdout.splitlines()))

        narrow = self._run(
            {"transcript_path": missing, "model": {"id": "claude-opus-4-8[1m]"}},
            env_extra={"COLUMNS": "55"},
        )
        self.assertEqual(
            narrow.stdout, "╭ WAIT FOR SESSION DATA\n╰ No session data yet\n"
        )

    def test_cost_is_removed_from_ambient_line(self):

        objs = [asst([read_tu(i, f"/repo/f{i}.py")]) for i in range(8)]
        objs += [asst([edit_tu(0, "/repo/f0.py")])]
        transcript = self._make_transcript(objs)
        first = self._invoke(transcript, {"cost": {"total_cost_usd": 1.00}})
        self.assertNotIn("since /clear", first.stdout)
        later = self._invoke(transcript, {"cost": {"total_cost_usd": 3.50}})
        self.assertNotIn("$", later.stdout)
        self.assertNotIn("since /clear", later.stdout)
        self.assertNotIn("⟳", later.stdout)

    def test_cost_chip_resets_on_new_session(self):

        objs = [asst([read_tu(i, f"/repo/f{i}.py")]) for i in range(8)]
        objs += [asst([edit_tu(0, "/repo/f0.py")])]
        proc = self._run_statusline(objs, {"cost": {"total_cost_usd": 42.0}})
        self.assertNotIn("since /clear", proc.stdout)

    def test_cost_chip_guards_against_counter_reset(self):

        objs = [asst([read_tu(i, f"/repo/f{i}.py")]) for i in range(8)]
        objs += [asst([edit_tu(0, "/repo/f0.py")])]
        transcript = self._make_transcript(objs)
        self._invoke(transcript, {"cost": {"total_cost_usd": 5.00}})
        proc = self._invoke(transcript, {"cost": {"total_cost_usd": 2.00}})
        self.assertNotIn("since /clear", proc.stdout)
        self.assertNotIn("$-", proc.stdout)

    def test_live_current_usage_preferred_over_legacy_cumulative_total(self):

        objs = [asst([read_tu(i, f"/repo/f{i}.py")]) for i in range(6)]
        objs += [asst([edit_tu(0, "/repo/f0.py")], usage={"input_tokens": 70_000})]
        proc = self._run_statusline(
            objs,
            {
                "context_window": {
                    "total_input_tokens": 600_000,
                    "current_usage": {
                        "input_tokens": 1_000,
                        "cache_creation_input_tokens": 1_000,
                        "cache_read_input_tokens": 18_000,
                    },
                    "used_percentage": 2,
                }
            },
        )
        self.assertIn("🌕 PEAK", proc.stdout)
        self.assertIn("CONTEXT LOAD 20K   WINDOW ░░░░░░░░░░ 2%", proc.stdout)
        self.assertNotIn("70K", proc.stdout)
        self.assertNotIn("600K", proc.stdout)
        self.assertNotIn("DEAD", proc.stdout)

    def test_null_current_usage_without_compact_falls_back_to_transcript_tokens(self):

        objs = [asst([read_tu(i, f"/repo/f{i}.py")]) for i in range(6)]
        objs += [asst([edit_tu(0, "/repo/f0.py")], usage={"input_tokens": 180_000})]
        proc = self._run_statusline(
            objs,
            {
                "context_window": {
                    "total_input_tokens": 250_000,
                    "current_usage": None,
                    "used_percentage": 25,
                }
            },
        )
        self.assertIn("🌗 FUNCTIONAL", proc.stdout)
        self.assertIn("CONTEXT LOAD 180K   WINDOW ██░░░░░░░░ 25%", proc.stdout)
        self.assertNotIn("250K", proc.stdout)

    def test_synthetic_zero_usage_line_keeps_last_real_context(self):

        objs = [asst([read_tu(i, f"/repo/f{i}.py")]) for i in range(6)]
        objs += [
            asst([edit_tu(0, "/repo/f0.py")], usage={"input_tokens": LOAD_DEGRADING})
        ]
        objs += [synthetic_api_error()]
        proc = self._run_statusline(objs)
        self.assertIn("CONTEXT LOAD " + LOAD_DEGRADING_S, proc.stdout)
        self.assertIn("DEGRADING", proc.stdout)
        self.assertNotIn("WAIT FOR SESSION DATA", proc.stdout)

    def test_compact_marker_acknowledges_pending_context_refresh(self):

        objs = [asst([], usage={"input_tokens": 180_000})]
        objs += [
            {
                "type": "user",
                "isCompactSummary": True,
                "isSidechain": False,
                "message": {"role": "user", "content": "compacted summary"},
            }
        ]
        proc = self._run_statusline(
            objs,
            {
                "context_window": {
                    "total_input_tokens": 0,
                    "current_usage": None,
                    "used_percentage": None,
                }
            },
        )
        self.assertIn("✓ COMPACT COMPLETE", proc.stdout)
        self.assertIn("CONTEXT UPDATES NEXT TURN", proc.stdout)
        self.assertNotIn("180K", proc.stdout)

        for tier in health_config.CONTEXT_TIERS:
            word = status_catalog.STATUSES[tier].action
            self.assertNotIn(word, proc.stdout, word)

        narrow = self._run_statusline(
            objs,
            {
                "context_window": {
                    "total_input_tokens": 0,
                    "current_usage": None,
                    "used_percentage": None,
                }
            },
            env_extra={"COLUMNS": "55"},
        )
        self.assertEqual(
            narrow.stdout,
            "╭ COMPACT COMPLETE\n╰ Context updates next turn\n",
        )

    def test_historic_compact_marker_does_not_acknowledge_on_resume(self):

        objs = [
            asst([], usage={"input_tokens": 180_000}),
            {
                "type": "user",
                "isCompactSummary": True,
                "isSidechain": False,
                "message": {"role": "user", "content": "compacted summary"},
            },
            asst([], usage={"input_tokens": 30_000}),
        ]
        proc = self._run_statusline(
            objs,
            {
                "context_window": {
                    "total_input_tokens": 0,
                    "current_usage": None,
                    "used_percentage": None,
                }
            },
        )
        self.assertNotIn("COMPACT COMPLETE", proc.stdout)
        self.assertIn("PEAK", proc.stdout)

    def _compaction_with_preserved_segment(self, trigger):
        """One render immediately after a compaction that preserved recent messages."""
        pre_compact = mk_usage(cr=238_000, cc=2_000, inp=10, out=120)
        objs = [
            asst([], usage=pre_compact),
            system_compact_boundary(
                trigger=trigger, pre_tokens=240_010, post_tokens=12_000
            ),
            user_compact_summary(),
        ]
        return self._run_statusline(
            objs,
            {
                "context_window": {
                    "total_input_tokens": 240_010,
                    "current_usage": dict(pre_compact),
                    "used_percentage": 24,
                }
            },
        )

    def test_manual_compaction_with_preserved_segment_never_shows_pre_compact_load(
        self,
    ):
        proc = self._compaction_with_preserved_segment("manual")
        self.assertNotIn("240K", proc.stdout)
        self.assertIn("✓ COMPACT COMPLETE", proc.stdout)
        self.assertIn("CONTEXT UPDATES NEXT TURN", proc.stdout)

    def test_auto_compaction_with_preserved_segment_never_shows_pre_compact_load(self):
        proc = self._compaction_with_preserved_segment("auto")
        self.assertNotIn("240K", proc.stdout)
        self.assertIn("✓ COMPACT COMPLETE", proc.stdout)
        self.assertIn("CONTEXT UPDATES NEXT TURN", proc.stdout)

    def test_post_compact_shows_no_context_number_including_post_tokens(self):

        proc = self._compaction_with_preserved_segment("manual")
        self.assertNotIn("12K", proc.stdout)
        self.assertNotIn("CONTEXT LOAD", proc.stdout)

    def test_next_real_turn_after_preserved_compaction_restores_live_context(self):

        pre_compact = mk_usage(cr=238_000, cc=2_000, inp=10, out=120)
        objs = [
            asst([], usage=pre_compact),
            system_compact_boundary(pre_tokens=240_010, post_tokens=12_000),
            user_compact_summary(),
            asst([], usage=mk_usage(cr=44_000, cc=1_000, inp=10, out=90)),
        ]
        proc = self._run_statusline(
            objs,
            {
                "context_window": {
                    "total_input_tokens": 45_010,
                    "current_usage": mk_usage(cr=44_000, cc=1_000, inp=10, out=90),
                    "used_percentage": 5,
                }
            },
        )
        self.assertNotIn("COMPACT COMPLETE", proc.stdout)
        self.assertIn("PEAK", proc.stdout)
        self.assertIn("WINDOW ░░░░░░░░░░ 5%", proc.stdout)

    def test_payload_rate_limits_are_ignored_entirely(self):

        objs = [asst([read_tu(i, f"/repo/f{i}.py")]) for i in range(6)]
        objs += [asst([edit_tu(0, "/repo/f0.py")], usage={"input_tokens": 50_000})]
        limits = {
            "rate_limits": {
                "five_hour": {"used_percentage": 95},
                "seven_day": {"used_percentage": 88},
            }
        }
        transcript = self._make_transcript(objs)
        for columns in ("200", "120", "80", "55"):
            with self.subTest(columns=columns):
                env = {"COLUMNS": columns}
                with_limits = self._invoke(transcript, limits, env_extra=env).stdout
                without = self._invoke(transcript, None, env_extra=env).stdout
                self.assertEqual(with_limits, without)
                self.assertLessEqual(len(with_limits.strip().splitlines()), 2)
                for gone in ("7D", "5H", "PACE USAGE", "🚨", "resets "):
                    self.assertNotIn(gone, with_limits, gone)

    def test_no_npx_is_ever_spawned(self):

        fake_bin = os.path.join(self._home, "bin")
        os.makedirs(fake_bin)
        spawn_log = os.path.join(self._home, "npx-spawns.log")
        fake_npx = os.path.join(fake_bin, "npx")
        with open(fake_npx, "w") as f:
            f.write("#!/bin/sh\nprintf 'spawn\\n' >> \"$NPX_SPAWN_LOG\"\n")
        os.chmod(fake_npx, 0o755)
        objs = [asst([], usage={"input_tokens": 50_000})]
        proc = self._run_statusline(
            objs,
            {"rate_limits": {"seven_day": {"used_percentage": 9}}},
            env_extra={
                "PATH": fake_bin + os.pathsep + os.environ.get("PATH", ""),
                "NPX_SPAWN_LOG": spawn_log,
            },
        )
        self.assertEqual(proc.stderr, "")
        self.assertIn("PEAK", proc.stdout)
        self.assertFalse(os.path.exists(spawn_log))

    def test_cache_is_private_and_scoped_to_claude_home(self):
        transcript = self._make_transcript([asst([], usage={"input_tokens": 50_000})])
        session_id = os.path.splitext(os.path.basename(transcript))[0]
        cache = self._cache_path(transcript)

        self._run({"transcript_path": transcript, "model": {"id": MODEL_ID}})

        self.assertTrue(os.path.isfile(cache))
        self.assertEqual(stat.S_IMODE(os.stat(os.path.dirname(cache)).st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(os.stat(cache).st_mode), 0o600)
        self.assertFalse(os.path.exists("/tmp/claude-statusline-" + session_id))

    def test_cache_write_replaces_symlink_without_touching_target(self):
        transcript = self._make_transcript([asst([], usage={"input_tokens": 50_000})])
        cache = self._cache_path(transcript)
        os.makedirs(os.path.dirname(cache), mode=0o700)
        target = os.path.join(self._home, "do-not-overwrite")
        with open(target, "w") as f:
            f.write("sentinel")
        os.symlink(target, cache)

        self._run({"transcript_path": transcript, "model": {"id": MODEL_ID}})

        self.assertFalse(os.path.islink(cache))
        with open(target) as f:
            self.assertEqual(f.read(), "sentinel")

    def test_build_output_reads_non_compacted_transcript_once(self):
        transcript = self._make_transcript([asst([], usage={"input_tokens": 50_000})])
        payload = {"transcript_path": transcript, "model": {"id": MODEL_ID}}
        real_open = open
        transcript_opens = []

        def counting_open(path, *args, **kwargs):
            if os.fspath(path) == transcript:
                transcript_opens.append(path)
            return real_open(path, *args, **kwargs)

        with (
            mock.patch("builtins.open", side_effect=counting_open),
            mock.patch.object(
                statusline.subagents, "cohort", return_value=subagents.Cohort()
            ),
        ):
            self.assertTrue(statusline.build_output(payload))

        self.assertEqual(len(transcript_opens), 1)

    def test_cache_hit_serves_cached_and_does_no_writes(self):

        objs = [asst([read_tu(i, f"/repo/f{i}.py")]) for i in range(6)]
        objs += [asst([edit_tu(0, "/repo/f0.py")], usage={"input_tokens": 90_000})]
        transcript = self._make_transcript(objs)
        cache = self._cache_path(transcript)
        payload = {
            "transcript_path": transcript,
            "model": {"id": MODEL_ID},
            "cost": {"total_cost_usd": 1.00},
        }
        first = self._run(payload)
        cache_mtime = os.path.getmtime(cache)
        second = self._run(dict(payload, cost={"total_cost_usd": 9.99}))
        self.assertEqual(second.stdout, first.stdout)
        self.assertEqual(os.path.getmtime(cache), cache_mtime)

    def test_cache_invalidates_when_live_session_identity_changes(self):
        transcript = self._make_transcript([asst([], usage={"input_tokens": 50_000})])
        first = {
            "transcript_path": transcript,
            "model": {"id": "claude-opus-4-8"},
            "effort": {"level": "high"},
            "cwd": "/repo/one",
            "workspace": {"repo": {"name": "one"}},
        }
        second = dict(
            first,
            model={"id": "claude-opus-4-8[1m]"},
            effort={"level": "max"},
            cwd="/repo/two",
            workspace={"repo": {"name": "two"}},
        )

        self.assertRegex(self._run(first).stdout, r"/repo/one +OPUS 4\.8 · HIGH")
        self.assertRegex(self._run(second).stdout, r"/repo/two +OPUS 4\.8 / 1M · MAX")

    def test_cache_invalidates_when_live_context_tokens_change(self):

        transcript = self._make_transcript([asst([], usage={"input_tokens": 50_000})])
        payload = {
            "transcript_path": transcript,
            "model": {"id": MODEL_ID},
            "context_window": {
                "current_usage": {"input_tokens": 50_000},
                "used_percentage": 25,
            },
        }
        self.assertIn("50K", self._run(payload).stdout)

        grown = dict(
            payload,
            context_window={
                "current_usage": {"input_tokens": 180_000},
                "used_percentage": 90,
            },
        )
        second = self._run(grown).stdout
        self.assertIn("180K", second)
        self.assertNotIn("50K", second)

    def test_cache_invalidates_when_fast_mode_toggles(self):

        transcript = self._make_transcript([asst([], usage={"input_tokens": 50_000})])
        payload = {"transcript_path": transcript, "model": {"id": MODEL_ID}}
        self.assertNotIn("⚡", self._run(payload).stdout)

        fast = self._run(dict(payload, fast_mode=True)).stdout
        self.assertIn("⚡", fast)

    def test_cache_invalidates_when_edit_totals_change(self):

        transcript = self._make_transcript([asst([], usage={"input_tokens": 50_000})])
        payload = {
            "transcript_path": transcript,
            "model": {"id": MODEL_ID},
            "cost": {"total_lines_added": 12, "total_lines_removed": 3},
        }
        self.assertIn("+12/-3", self._run(payload).stdout)

        edited = dict(
            payload, cost={"total_lines_added": 400, "total_lines_removed": 3}
        )
        second = self._run(edited).stdout
        self.assertIn("+400/-3", second)
        self.assertNotIn("+12/-3", second)

    def test_cache_invalidates_when_a_display_flag_toggles(self):

        transcript = self._make_transcript([asst([], usage={"input_tokens": 50_000})])
        payload = {"transcript_path": transcript, "model": {"id": MODEL_ID}}
        self.assertIn("50K", self._run(payload).stdout)

        hidden = self._run(payload, env_extra={"CLAUDE_HEALTH_SHOW_CONTEXT": "0"})
        self.assertNotIn("50K", hidden.stdout)

    def test_cache_invalidates_at_compact_boundary(self):
        transcript = self._make_transcript([asst([], usage={"input_tokens": 180_000})])
        before = {
            "transcript_path": transcript,
            "model": {"id": MODEL_ID},
            "context_window": {
                "total_input_tokens": 180_000,
                "current_usage": {"input_tokens": 180_000},
                "used_percentage": 90,
            },
        }
        self.assertIn("FUNCTIONAL", self._run(before).stdout)

        with open(transcript, "a") as f:
            f.write(
                json.dumps(
                    {
                        "type": "user",
                        "isCompactSummary": True,
                        "isSidechain": False,
                        "message": {"role": "user", "content": "compacted summary"},
                    }
                )
                + "\n"
            )
        after = dict(
            before,
            context_window={
                "total_input_tokens": 0,
                "current_usage": None,
                "used_percentage": None,
            },
        )
        compacted = self._run(after)
        self.assertIn("✓ COMPACT COMPLETE", compacted.stdout)
        self.assertIn("CONTEXT UPDATES NEXT TURN", compacted.stdout)
        self.assertNotIn("180K", compacted.stdout)
        self.assertNotIn("FUNCTIONAL", compacted.stdout)

        with open(transcript, "a") as f:
            f.write(json.dumps(asst([], usage={"input_tokens": 30_000})) + "\n")
        next_turn = dict(
            before,
            context_window={
                "total_input_tokens": 30_000,
                "current_usage": {"input_tokens": 30_000},
                "used_percentage": 15,
            },
        )
        refreshed = self._run(next_turn)
        self.assertIn("PEAK", refreshed.stdout)
        self.assertIn("WINDOW █░░░░░░░░░ 15%", refreshed.stdout)
        self.assertNotIn("COMPACT COMPLETE", refreshed.stdout)

    def test_cache_invalidates_at_preserved_segment_compact_boundary(self):

        usage = {"input_tokens": 180_000}
        transcript = self._make_transcript([asst([], usage=usage)])
        payload = {
            "transcript_path": transcript,
            "model": {"id": MODEL_ID},
            "context_window": {
                "total_input_tokens": 180_000,
                "current_usage": dict(usage),
                "used_percentage": 90,
            },
        }
        self.assertIn("FUNCTIONAL", self._run(payload).stdout)

        with open(transcript, "a") as f:
            f.write(
                json.dumps(
                    system_compact_boundary(pre_tokens=180_000, post_tokens=9_000)
                )
                + "\n"
            )
            f.write(json.dumps(user_compact_summary()) + "\n")
        compacted = self._run(payload)
        self.assertIn("✓ COMPACT COMPLETE", compacted.stdout)
        self.assertNotIn("180K", compacted.stdout)

    def test_cache_invalidates_for_compact_summary_larger_than_old_tail_probe(self):
        usage = {"input_tokens": 180_000}
        transcript = self._make_transcript([asst([], usage=usage)])
        payload = {
            "transcript_path": transcript,
            "model": {"id": MODEL_ID},
            "context_window": {
                "total_input_tokens": 180_000,
                "current_usage": dict(usage),
                "used_percentage": 90,
            },
        }
        self.assertIn("FUNCTIONAL", self._run(payload).stdout)

        base_summary = user_compact_summary()
        summary = dict(
            base_summary,
            message=dict(base_summary["message"], content="x" * 70_000),
        )
        with open(transcript, "a") as f:
            f.write(json.dumps(summary) + "\n")

        compacted = self._run(payload)
        self.assertIn("✓ COMPACT COMPLETE", compacted.stdout)
        self.assertNotIn("180K", compacted.stdout)

    def test_cache_invalidates_when_terminal_width_changes(self):
        transcript = self._make_transcript([asst([], usage={"input_tokens": 50_000})])
        payload = {
            "transcript_path": transcript,
            "model": {"id": "claude-opus-4-8[1m]"},
            "effort": {"level": "max"},
            "cwd": "/repo/attention-span",
            "workspace": {"repo": {"name": "attention-span"}},
            "rate_limits": {
                "five_hour": {"used_percentage": 25},
                "seven_day": {"used_percentage": 33},
            },
        }
        wide = self._run(payload, env_extra={"COLUMNS": "120"}).stdout
        narrow = self._run(payload, env_extra={"COLUMNS": "55"}).stdout
        self.assertNotEqual(narrow, wide)
        self.assertTrue(all(len(line) <= 55 for line in narrow.splitlines()))

    def test_narrow_pane_keeps_only_the_action_and_its_reason(self):
        transcript = self._make_transcript(
            [asst([], usage={"input_tokens": LOAD_FUNCTIONAL})]
        )
        payload = {
            "transcript_path": transcript,
            "model": {"id": "claude-opus-4-8[1m]"},
            "effort": {"level": "max"},
            "cwd": "/repo/attention-span",
            "workspace": {"repo": {"name": "attention-span"}},
            "rate_limits": {
                "five_hour": {"used_percentage": 25},
                "seven_day": {"used_percentage": 33},
            },
        }
        out = self._run(payload, env_extra={"COLUMNS": "55"}).stdout
        self.assertEqual(
            out, "╭ FUNCTIONAL\n╰ Context load: " + LOAD_FUNCTIONAL_S + "\n"
        )
        for packed_label in ("CTX", "5H", "WK", "OPUS"):
            self.assertNotIn(packed_label, out)

    def test_cold_start_baseline_then_plain_dollar(self):

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        transcript = os.path.join(d, "fresh-cost-session.jsonl")
        cost_base = "/tmp/claude-statusline-cost-fresh-cost-session"
        cache = "/tmp/claude-statusline-fresh-cost-session"
        for p in (cost_base, cache):
            self.addCleanup(lambda p=p: os.path.exists(p) and os.unlink(p))
        cold = self._run(
            {
                "transcript_path": transcript,
                "model": {"id": MODEL_ID},
                "cost": {"total_cost_usd": 0.0},
            }
        )
        self.assertIn("WAIT FOR SESSION DATA", cold.stdout)
        objs = [asst([read_tu(i, f"/repo/f{i}.py")]) for i in range(6)]
        objs += [asst([edit_tu(0, "/repo/f0.py")], usage={"input_tokens": 90_000})]
        write_lines(objs, transcript)
        proc = self._run(
            {
                "transcript_path": transcript,
                "model": {"id": MODEL_ID},
                "cost": {"total_cost_usd": 1.50},
            }
        )
        self.assertNotIn("$", proc.stdout)
        self.assertNotIn("since /clear", proc.stdout)


class TestPlatformDegradationE2E(TestStatusline):
    """End-to-end: pipe degradation-shaped transcripts through `python3 statusline.py`
    and confirm the cache chip appears ONLY on thrash and ◌ ONLY on a degraded parse."""

    def test_thrash_transcript_shows_cache_chip(self):
        proc = self._run_statusline(cache_objs(THRASH_SPECS))
        self.assertIn("CHECK CACHE - COST RISING", proc.stdout)

    def test_healthy_transcript_no_cache_chip(self):
        healthy = cache_objs([(9000, 500, 500 + i) for i in range(15)])
        proc = self._run_statusline(healthy)
        self.assertNotIn("⚠ cache", proc.stdout)
        self.assertNotIn("data drift", proc.stdout)
        self.assertNotIn("◌ stale", proc.stdout)

    def test_schema_canary_degraded_leads_with_stale(self):

        proc = self._run_statusline([asst([]) for _ in range(6)])
        self.assertIn("◌ CAN'T CHECK SESSION", proc.stdout)

    def test_degraded_with_live_context_keeps_context_and_drift_tail(self):

        cw = {
            "context_window": {
                "current_usage": {"input_tokens": 800_000},
                "total_input_tokens": 800_000,
                "used_percentage": 80,
            }
        }
        proc = self._run_statusline([asst([]) for _ in range(6)], payload_extra=cw)
        self.assertIn("◌ CAN'T CHECK SESSION", proc.stdout)
        self.assertIn("CONTEXT LOAD 800K   WINDOW ████████░░ 80%", proc.stdout)


class TestWindowClassGrading(unittest.TestCase):
    """The advertised window picks the ladder a session is graded on.

    `context_window_size` is model-derived and unconditional in the payload (verified
    against the shipped 2.1.222 binary). It was RECORDED in the `/health` snapshot until
    that report was deleted on 2026-08-06, so the only remaining place it can be
    observed is the rendered row - which is where it always mattered: an unknown or
    unlisted size must grade byte-identically to the base ladder, and the 200k class
    must diverge from the 1m one at identical token counts.
    """

    def _build(self, session_id, context_window):
        transcript = write_lines([asst([read_tu(0)], usage=mk_usage(inp=90_000))])
        self.addCleanup(lambda: os.path.exists(transcript) and os.unlink(transcript))
        cache = (
            "/tmp/claude-statusline-"
            + os.path.splitext(os.path.basename(transcript))[0]
        )
        self.addCleanup(lambda: os.path.exists(cache) and os.unlink(cache))
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        payload = {
            "transcript_path": transcript,
            "session_id": session_id,
            "model": {"id": MODEL_ID},
        }
        if context_window is not None:
            payload["context_window"] = context_window
        with (
            mock.patch.object(
                statusline.session_ui, "UI_PREFIX", os.path.join(tmp, "ui-")
            ),
            mock.patch.dict(os.environ, {"NO_COLOR": "1", "COLUMNS": "200"}),
        ):
            return statusline.build_output(payload)

    def test_every_taxonomy_size_renders_a_panel(self):
        for size, cls in health_config.WINDOW_CLASSES.items():
            self.assertTrue(
                self._build("window-fact-" + cls, {"context_window_size": size}), cls
            )

    def test_garbage_sizes_fall_back_to_the_base_ladder_without_crashing(self):

        absent = self._build("window-fact-absent", {})
        for i, garbage in enumerate(("1m", float("nan"), 0)):
            out = self._build(
                f"window-fact-garbage-{i}", {"context_window_size": garbage}
            )
            self.assertEqual(out, absent, repr(garbage))

    def test_an_unlisted_size_grades_on_the_base_ladder(self):

        self.assertEqual(
            self._build("window-fact-unlisted", {"context_window_size": 500_000}),
            self._build("window-fact-unlisted-absent", {}),
        )

    def test_a_1m_window_renders_identical_to_an_absent_one(self):

        size = max(health_config.WINDOW_CLASSES)
        self.assertEqual(
            self._build("window-fact-render-a", {"context_window_size": size}),
            self._build("window-fact-render-b", {}),
        )

    def test_the_200k_class_diverges_from_the_1m_class_at_the_same_tokens(self):

        sizes = {cls: size for size, cls in health_config.WINDOW_CLASSES.items()}
        outs = {}
        for cls in ("200k", "1m"):
            outs[cls] = self._build(
                "window-divergence-" + cls,
                {
                    "context_window_size": sizes[cls],
                    "total_input_tokens": LOAD_200K_DEGRADING,
                    "current_usage": {"input_tokens": LOAD_200K_DEGRADING},
                },
            )
        self.assertIn("DEGRADING", outs["200k"])
        self.assertIn("FUNCTIONAL", outs["1m"])


class TestNotices(unittest.TestCase):
    """Per-session presentation state: delegation facts and one-shot completion notices.

    A `--inspect <session-id>` CLI test stood at the head of this class until
    2026-08-06. The subcommand, the snapshot it read, and the `/health` report it
    printed were all deleted; the statusline is the whole product now, so every
    assertion below reads the rendered row.
    """

    def test_build_output_keeps_high_delegation_persistent(self):
        transcript = write_lines([asst([read_tu(0)], usage=mk_usage(inp=90_000))])
        self.addCleanup(lambda: os.path.exists(transcript) and os.unlink(transcript))
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        payload = {
            "transcript_path": transcript,
            "session_id": "notice-e2e",
            "model": {"id": MODEL_ID},
        }
        cohort = subagents.Cohort(
            live=(subagents.Subagent(),) * 5,
            total_n=109,
            done_n=104,
            tokens_total=11_500_000,
            tokens_known_n=109,
        )
        with (
            mock.patch.object(
                statusline.session_ui, "UI_PREFIX", os.path.join(tmp, "ui-")
            ),
            mock.patch.object(statusline.subagents, "cohort", return_value=cohort),
            mock.patch.dict(os.environ, {"NO_COLOR": "1", "COLUMNS": "200"}),
        ):
            first = statusline.build_output(payload)
            second = statusline.build_output(payload)
        expected = (
            "╭─ ▲ REVIEW CHILD TOKEN BURN ───  SUBAGENTS 109   TOKENS 11.5M"
            "   WORKING 5   CONTEXT LOAD 90K\n╰─ OPUS 4.8"
        )
        self.assertEqual(first, expected)
        self.assertEqual(second, expected)

    def test_build_output_surfaces_foreign_child_models(self):

        transcript = write_lines([asst([read_tu(0)], usage=mk_usage(inp=10_000))])
        self.addCleanup(lambda: os.path.exists(transcript) and os.unlink(transcript))
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        payload = {
            "transcript_path": transcript,
            "session_id": "foreign-model-e2e",
            "model": {"id": "claude-fable-5"},
        }
        cohort = subagents.Cohort(
            live=(subagents.Subagent(),) * 2,
            total_n=2,
            tokens_known_n=2,
            models=("claude-opus-5",),
        )
        with (
            mock.patch.object(
                statusline.session_ui, "UI_PREFIX", os.path.join(tmp, "ui-")
            ),
            mock.patch.object(statusline.subagents, "cohort", return_value=cohort),
            mock.patch.dict(os.environ, {"NO_COLOR": "1", "COLUMNS": "200"}),
        ):
            out = statusline.build_output(payload)
        self.assertIn("WORKING 2   ⇄ opus-5", out)

    def test_warming_cohort_does_not_update_completion_notice_state(self):
        transcript = write_lines([asst([read_tu(0)], usage=mk_usage(inp=50_000))])
        self.addCleanup(lambda: os.path.exists(transcript) and os.unlink(transcript))
        payload = {
            "transcript_path": transcript,
            "session_id": "warming-notice-e2e",
            "model": {"id": MODEL_ID},
        }
        cohort = subagents.Cohort(
            live=(subagents.Subagent(),) * 3,
            total_n=7,
            done_n=3,
            tokens_total=1_590,
            tokens_known_n=3,
        )
        with (
            mock.patch.object(statusline.subagents, "cohort", return_value=cohort),
            mock.patch.object(
                statusline, "_next_notice", return_value=None
            ) as next_notice,
            mock.patch.dict(os.environ, {"NO_COLOR": "1", "COLUMNS": "200"}),
        ):
            statusline.build_output(payload)

        next_notice.assert_not_called()

    def test_hiding_agents_keeps_child_run_accounting_in_the_row(self):
        transcript = write_lines([asst([read_tu(0)], usage=mk_usage(inp=90_000))])
        self.addCleanup(lambda: os.path.exists(transcript) and os.unlink(transcript))
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        session_id = "hidden-agents-e2e"
        payload = {
            "transcript_path": transcript,
            "session_id": session_id,
            "model": {"id": MODEL_ID},
        }
        cohort = subagents.Cohort(
            live=(subagents.Subagent(),) * 2,
            total_n=106,
            done_n=104,
            tokens_total=2_191_608,
            tokens_known_n=16,
        )
        with (
            mock.patch.object(
                statusline.session_ui, "UI_PREFIX", os.path.join(tmp, "ui-")
            ),
            mock.patch.object(statusline.subagents, "cohort", return_value=cohort),
            mock.patch.dict(
                os.environ,
                {"NO_COLOR": "1", "COLUMNS": "200", "CLAUDE_HEALTH_SHOW_AGENTS": "0"},
            ),
        ):
            ambient = statusline.build_output(payload)

        self.assertNotIn("working", ambient)
        self.assertTrue(ambient.startswith("╭─ ▲ REVIEW CHILD TOKEN BURN"))
        self.assertIn("SUBAGENTS 106   TOKENS ≥2.2M", ambient)

    def test_compaction_stops_the_row_asserting_the_discarded_load(self):

        transcript = write_lines([asst([read_tu(0)], usage=mk_usage(cr=260_000))])
        self.addCleanup(lambda: os.path.exists(transcript) and os.unlink(transcript))
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        session_id = "post-compact-inspect-e2e"
        payload = {
            "transcript_path": transcript,
            "session_id": session_id,
            "model": {"id": MODEL_ID},
            "rate_limits": {"seven_day": {"used_percentage": 33}},
            "context_window": {
                "total_input_tokens": 260_000,
                "current_usage": mk_usage(cr=260_000),
                "used_percentage": 26,
            },
        }
        cohort = subagents.Cohort(
            live=(subagents.Subagent(),) * 2,
            total_n=5,
            done_n=3,
            tokens_total=1_000,
            tokens_known_n=5,
        )
        with (
            mock.patch.object(
                statusline.session_ui, "UI_PREFIX", os.path.join(tmp, "ui-")
            ),
            mock.patch.object(statusline.subagents, "cohort", return_value=cohort),
            mock.patch.dict(os.environ, {"NO_COLOR": "1", "COLUMNS": "200"}),
        ):
            before = statusline.build_output(payload)
            with open(transcript, "a") as f:
                f.write(
                    json.dumps(
                        system_compact_boundary(pre_tokens=260_000, post_tokens=19_000)
                    )
                    + "\n"
                )
                f.write(json.dumps(user_compact_summary()) + "\n")
            row = statusline.build_output(payload)

        self.assertIn("260K", before)
        self.assertIn("COMPACT COMPLETE", row)
        self.assertNotIn("260K", row)

    def test_hiding_agents_omits_working_count_from_completion_notice(self):
        transcript = write_lines([asst([read_tu(0)], usage=mk_usage(inp=50_000))])
        self.addCleanup(lambda: os.path.exists(transcript) and os.unlink(transcript))
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        payload = {
            "transcript_path": transcript,
            "session_id": "hidden-notice-e2e",
            "model": {"id": MODEL_ID},
        }
        cohorts = [
            subagents.Cohort(
                live=(subagents.Subagent(),),
                total_n=2,
                done_n=1,
                tokens_total=100,
                tokens_known_n=2,
            ),
            subagents.Cohort(
                live=(subagents.Subagent(),),
                total_n=3,
                done_n=2,
                tokens_total=150,
                tokens_known_n=3,
            ),
        ]
        with (
            mock.patch.object(
                statusline.session_ui, "UI_PREFIX", os.path.join(tmp, "ui-")
            ),
            mock.patch.object(statusline.subagents, "cohort", side_effect=cohorts),
            mock.patch.dict(
                os.environ,
                {"NO_COLOR": "1", "COLUMNS": "200", "CLAUDE_HEALTH_SHOW_AGENTS": "0"},
            ),
        ):
            statusline.build_output(payload)
            notice = statusline.build_output(payload)
            self.assertEqual(
                notice,
                "╭─ ✓ 1 SUBAGENT FINISHED ───\n╰─ OPUS 4.8",
            )

    def test_hiding_agents_keeps_child_blind_loop_actionable(self):
        transcript = write_lines([asst([read_tu(0)], usage=mk_usage(inp=90_000))])
        self.addCleanup(lambda: os.path.exists(transcript) and os.unlink(transcript))
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        payload = {
            "transcript_path": transcript,
            "session_id": "hidden-loop-e2e",
            "model": {"id": MODEL_ID},
        }

        cohort = subagents.Cohort(
            live=(subagents.Subagent(state="red", insufficient=True),),
            total_n=1,
            tokens_known_n=1,
            blind_loop_n=1,
        )
        with (
            mock.patch.object(
                statusline.session_ui, "UI_PREFIX", os.path.join(tmp, "ui-")
            ),
            mock.patch.object(statusline.subagents, "cohort", return_value=cohort),
            mock.patch.dict(
                os.environ,
                {"NO_COLOR": "1", "COLUMNS": "200", "CLAUDE_HEALTH_SHOW_AGENTS": "0"},
            ),
        ):
            ambient = statusline.build_output(payload)
        self.assertTrue(ambient.startswith("╭─ ■ CHECK CHILD AGENT"))
        self.assertNotIn("working", ambient)


if __name__ == "__main__":
    unittest.main(verbosity=2)
