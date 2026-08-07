"""Unit tests for the corpus_pins loud-skip contract — stdlib unittest.

The loudness IS the feature (the previous corpus vanished and its guards went inert
silently), so the contract gets its own tests: a missing pin must warn on stderr
exactly once per label per process and skip; CC_HEALTH_REQUIRE_CORPUS=1 must escalate
the skip to a failure; a present pin must stay completely quiet. Uses temp dirs only —
no reliance on the real corpus.

Run: python3 -m unittest discover -s tests   (from repo root)
"""

import contextlib
import io
import os
import tempfile
import unittest

try:
    import corpus_pins
except ImportError:
    raise unittest.SkipTest(
        "local pin registry tests/corpus_pins.py not present"
    ) from None


class TestRequireCorpus(unittest.TestCase):
    def setUp(self):

        self._saved_warned = set(corpus_pins._warned)
        corpus_pins._warned.clear()
        self._saved_env = os.environ.pop(corpus_pins.REQUIRE_ENV, None)

    def tearDown(self):
        corpus_pins._warned.clear()
        corpus_pins._warned.update(self._saved_warned)
        if self._saved_env is not None:
            os.environ[corpus_pins.REQUIRE_ENV] = self._saved_env
        else:
            os.environ.pop(corpus_pins.REQUIRE_ENV, None)

    def _call(self, path, label="test-pin"):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            corpus_pins.require_corpus(self, label, path)
        return stderr.getvalue()

    def test_present_path_is_a_silent_no_op(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self._call(tmp), "")

    def test_missing_path_warns_on_stderr_and_skips(self):
        stderr = io.StringIO()
        with (
            self.assertRaises(unittest.SkipTest) as ctx,
            contextlib.redirect_stderr(stderr),
        ):
            corpus_pins.require_corpus(self, "test-pin", "/nonexistent/pin")
        self.assertIn("pinned corpus missing", str(ctx.exception))
        self.assertIn("test-pin", stderr.getvalue())
        self.assertIn("SKIPPED", stderr.getvalue())

    def test_missing_path_warns_once_per_label_per_process(self):
        stderr = io.StringIO()
        for _ in range(3):
            with (
                self.assertRaises(unittest.SkipTest),
                contextlib.redirect_stderr(stderr),
            ):
                corpus_pins.require_corpus(self, "test-pin", "/nonexistent/pin")
        self.assertEqual(stderr.getvalue().count("SKIPPED"), 1)

    def test_distinct_labels_each_get_their_own_warning(self):
        stderr = io.StringIO()
        for label in ("pin-a", "pin-b"):
            with (
                self.assertRaises(unittest.SkipTest),
                contextlib.redirect_stderr(stderr),
            ):
                corpus_pins.require_corpus(self, label, "/nonexistent/pin")
        self.assertIn("pin-a", stderr.getvalue())
        self.assertIn("pin-b", stderr.getvalue())

    def test_require_env_escalates_missing_path_to_failure(self):
        os.environ[corpus_pins.REQUIRE_ENV] = "1"
        with self.assertRaises(self.failureException) as ctx:
            self._call("/nonexistent/pin")
        self.assertIn("pinned corpus missing", str(ctx.exception))
        self.assertIn(corpus_pins.REQUIRE_ENV, str(ctx.exception))

    def test_require_env_leaves_present_path_a_no_op(self):
        os.environ[corpus_pins.REQUIRE_ENV] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self._call(tmp), "")


class TestRegistryIntegrity(unittest.TestCase):
    """The registry must stay internally consistent so a typo cannot silently detach
    an exception or a positive from the projects the guards actually walk."""

    def test_pinned_projects_nonempty(self):
        self.assertTrue(corpus_pins.PINNED_PROJECTS)

    def test_repetition_exemptions_reference_pinned_projects(self):
        pinned = set(corpus_pins.PINNED_PROJECTS)
        for proj, rel in corpus_pins.KNOWN_REPETITION_PAGINATION:
            self.assertIn(proj, pinned, msg=rel)

    def test_positive_drivers_reference_pinned_projects(self):
        pinned = set(corpus_pins.PINNED_PROJECTS)
        for proj, rel in corpus_pins.POSITIVE_DRIVERS:
            self.assertIn(proj, pinned, msg=rel)


if __name__ == "__main__":
    unittest.main(verbosity=2)
