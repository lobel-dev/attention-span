"""Tests for settled incremental snapshots from agent_health.iter_states."""

import os
import unittest

from builders import (
    GENUINE,
    HOOKDENY,
    asst,
    edit_tu,
    read_tu,
    results,
    user_str,
    write_lines,
)

from attention_span import agent_health as ah


class TestIterStates(unittest.TestCase):
    def _states(self, objs):
        path = write_lines(objs)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        return path, list(ah.iter_states(path))

    def test_last_snapshot_matches_analyze_transcript(self):
        objs = [
            asst([read_tu(0, "/repo/a.py")], usage={"input_tokens": 1234}),
            results([("r0", False, "ok")]),
            user_str("peak"),
            asst([edit_tu(0, "/repo/a.py")]),
            results([("e0", True, GENUINE)]),
        ]
        path, states = self._states(objs)
        self.assertTrue(states)
        self.assertEqual(ah.analyze_transcript(path), states[-1][1])

    def test_hook_denied_edit_is_removed_when_settled(self):
        objs = [
            asst([edit_tu(0, "/repo/a.py")]),
            results([("e0", True, HOOKDENY)]),
        ]
        path, states = self._states(objs)
        self.assertEqual(ah.analyze_transcript(path), states[-1][1])
        cursor, snapshot = states[-1]
        self.assertEqual(snapshot.edits, 0)
        self.assertEqual(snapshot.total_edits, 0)
        self.assertEqual(cursor["settled_tool_use_ids"], ["e0"])
        self.assertEqual(cursor["settled_events"][0]["result_state"], "HOOK_DENY")

    def test_pending_trailing_edit_matches_live_behavior(self):
        objs = [asst([read_tu(i, f"/repo/r{i}.py")]) for i in range(4)]
        objs.append(asst([edit_tu(0, "/repo/x.py")]))
        path, states = self._states(objs)
        self.assertEqual(ah.analyze_transcript(path), states[-1][1])
        cursor, snapshot = states[-1]
        self.assertEqual(snapshot.edits, 1)
        self.assertEqual(snapshot.window_used, 5)
        self.assertEqual(cursor["pending_events"][-1]["tool_use_id"], "e0")
        self.assertEqual(cursor["pending_events"][-1]["result_state"], "PENDING")

    def test_threshold_window_applies_to_iter_states(self):
        objs = [asst([read_tu(i, f"/repo/r{i}.py")]) for i in range(5)]
        objs += [asst([edit_tu(i, f"/repo/e{i}.py")]) for i in range(5)]
        path = write_lines(objs)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))

        states = list(ah.iter_states(path, th={"WINDOW": 5}))
        self.assertTrue(states)
        snapshot = states[-1][1]
        self.assertEqual(snapshot.window_used, 5)
        self.assertEqual(snapshot.reads, 0)
        self.assertEqual(snapshot.edits, 5)

    def test_hook_denied_retry_after_genuine_failure_not_blind_loop(self):
        objs = [
            asst([edit_tu(0, "/repo/a.py")]),
            results([("e0", True, GENUINE)]),
            asst([("e1", "Edit", {"file_path": "/repo/a.py"})]),
            results([("e1", True, HOOKDENY)]),
        ]
        path, states = self._states(objs)
        self.assertEqual(ah.analyze_transcript(path), states[-1][1])
        self.assertEqual(states[-1][1].failed_edit_loop.count, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
