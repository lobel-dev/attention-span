import os
import shutil
import tempfile
import unittest
from unittest import mock

from attention_span import render_facts, session_ui

Notice = render_facts.Notice


class TestSessionNotices(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.prefix = os.path.join(self.dir, "ui-")
        patch = mock.patch.object(session_ui, "UI_PREFIX", self.prefix)
        patch.start()
        self.addCleanup(patch.stop)

    def test_first_observation_baselines_without_completion_notice(self):
        self.assertIsNone(session_ui.next_notice("s", 12, 5))
        self.assertIsNone(session_ui.next_notice("s", 12, 5))

    def test_completion_delta_appears_once(self):
        session_ui.next_notice("s", 12, 5)
        self.assertEqual(session_ui.next_notice("s", 15, 2), Notice(done_n=3, live_n=2))
        self.assertIsNone(session_ui.next_notice("s", 15, 2))

    def test_the_notice_crosses_as_counts_not_as_a_sentence(self):

        session_ui.next_notice("s", 12, 5)
        notice = session_ui.next_notice("s", 15, 2)
        self.assertIsInstance(notice, render_facts.Notice)
        self.assertEqual((notice.done_n, notice.live_n), (3, 2))

    def test_hidden_agent_details_do_not_leak_working_count_in_notice(self):
        session_ui.next_notice("s", 12, 5, show_working=False)
        self.assertEqual(
            session_ui.next_notice("s", 15, 2, show_working=False),
            Notice(done_n=3, live_n=0),
        )

    def test_high_delegation_is_not_a_transient_notice(self):
        session_ui.next_notice("s", 49, 3)
        self.assertEqual(session_ui.next_notice("s", 50, 2), Notice(done_n=1, live_n=2))
        self.assertEqual(session_ui.next_notice("s", 51, 1), Notice(done_n=1, live_n=1))

    def test_initial_high_count_only_baselines_completion_state(self):
        self.assertIsNone(session_ui.next_notice("s", 104, 5))

    def test_write_failure_omits_notice(self):
        session_ui.next_notice("s", 103, 5)
        with mock.patch.object(session_ui, "_atomic_write", return_value=False):
            self.assertIsNone(session_ui.next_notice("s", 104, 5))

    def test_state_directory_symlink_is_rejected(self):
        target = os.path.join(self.dir, "target")
        link = os.path.join(self.dir, "cache-link")
        os.mkdir(target)
        os.symlink(target, link)

        with mock.patch.object(session_ui, "UI_PREFIX", os.path.join(link, "ui-")):
            self.assertIsNone(session_ui.next_notice("s", 1, 0))

        self.assertEqual(os.listdir(target), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
