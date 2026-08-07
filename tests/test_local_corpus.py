"""No-false-red regression over the operator's private transcript corpus.

The test logic is tracked; the pin registry (tests/corpus_pins.py: project paths,
session UUIDs under ~/.claude/projects) is local-only via .gitignore. Without it this
whole module skips, so the suite stays green on machines without the corpus.

Run: python3 -m unittest discover -s tests   (from repo root)
"""

import os
import unittest

try:
    import corpus_pins
except ImportError:
    raise unittest.SkipTest(
        "local pin registry tests/corpus_pins.py not present"
    ) from None

from attention_span import agent_health as ah
from attention_span import health_config

BANDS = health_config.ENGINE_BANDS


class TestLocalCorpusRegression(unittest.TestCase):
    """The no-false-red invariant — extended to EVERY platform-degradation signal.

    On the pinned healthy local corpus (tests/corpus_pins.py: driver sessions + every
    subagent child, analyzed with include_sidechain=True), NONE of the signals may
    fire: no blind loop, no cache-thrash chip, no parse-health doubt, and both
    repetition and perseveration below their fire thresholds - except the named
    KNOWN_REPETITION_PAGINATION files,
    which are exempt from the repetition assertion ONLY (documented offset/limit
    pagination false positive of the shadow column). The corpus proves the negative
    (stays quiet when healthy); the synthetic fixtures carry the positive paths while
    POSITIVE_DRIVERS is empty. Corpus-guarded LOUDLY via corpus_pins.require_corpus:
    absence warns on stderr and skips (CI-safe), and CC_HEALTH_REQUIRE_CORPUS=1
    escalates the skip to a failure. This is the precision gate for the whole
    feature set."""

    def _pinned(self, pattern, recursive=False, exclude_positive=False):

        import glob

        positives = set(corpus_pins.POSITIVE_DRIVERS)
        files = []
        for proj in corpus_pins.PINNED_PROJECTS:
            pdir = corpus_pins.project_dir(proj)
            corpus_pins.require_corpus(self, proj, pdir)
            for path in sorted(
                glob.glob(os.path.join(pdir, pattern), recursive=recursive)
            ):
                rel = os.path.relpath(path, pdir)
                if exclude_positive and (proj, rel) in positives:
                    continue
                files.append(
                    (path, (proj, rel) in corpus_pins.KNOWN_REPETITION_PAGINATION)
                )
        if not files:
            self.skipTest("pinned corpus present but matched no " + pattern)
        return files

    def _driver_files(self):
        return self._pinned("*.jsonl", exclude_positive=True)

    def _child_files(self):
        return self._pinned(
            os.path.join("*", "subagents", "**", "agent-*.jsonl"), recursive=True
        )

    def _assert_quiet(self, a, where, repetition_exempt=False):
        self.assertEqual(a.failed_edit_loop.count, 0, msg="blind-loop in " + where)
        self.assertIsNone(ah.blind_loop_alert(a), msg="blind-loop verdict in " + where)
        self.assertFalse(a.cache_health.get("show"), msg="cache thrash in " + where)
        self.assertFalse(a.parse_health.degraded, msg="parse degraded in " + where)

        self.assertLess(
            a.perseveration.score,
            BANDS.PERSEV_MIN,
            msg="perseveration fired in " + where,
        )
        if not repetition_exempt:
            self.assertLess(
                a.repetition.score,
                BANDS.REPEAT_MIN,
                msg="repetition fired in " + where,
            )

    def test_no_driver_false_red(self):
        for path, exempt in self._driver_files():
            self._assert_quiet(
                ah.analyze_transcript(path), path, repetition_exempt=exempt
            )

    def test_known_driver_retry_stays_a_real_positive(self):
        if not corpus_pins.POSITIVE_DRIVERS:
            self.skipTest(
                "no local positive pinned - synthetic fixtures carry the "
                "positive paths (PRD open question tracks finding a real one)"
            )
        for proj, session_file in corpus_pins.POSITIVE_DRIVERS:
            path = os.path.join(corpus_pins.project_dir(proj), session_file)
            corpus_pins.require_corpus(self, proj + "/" + session_file, path)
            analysis = ah.analyze_transcript(path)
            self.assertGreaterEqual(analysis.failed_edit_loop.count, 1)
            self.assertIsNotNone(ah.blind_loop_alert(analysis))

    def test_no_child_false_red(self):
        for path, exempt in self._child_files():
            self._assert_quiet(
                ah.analyze_transcript(path, include_sidechain=True),
                path,
                repetition_exempt=exempt,
            )

    def test_no_driver_stop_reason_row(self):

        from attention_span import render, render_facts, status_catalog

        for path, _exempt in self._driver_files():
            a = ah.analyze_transcript(path)
            facts = render_facts.RenderFacts(
                context=render_facts.ContextLoad(tier="strong", tokens=50_000),
                cache_health=a.cache_health,
                parse_degraded=a.parse_health.degraded,
                last_stop_reason=a.last_stop_reason,
            )
            warnings = status_catalog.active_warnings(facts)
            for status in ("Response truncated", "Response refused"):
                self.assertNotIn(status, warnings, msg=status + " fired in " + path)
            out = render.render_panel(facts, render.LayoutOpts(no_color=True))
            for text in ("LAST RESPONSE CUT OFF", "LAST RESPONSE REFUSED"):
                self.assertNotIn(text, out, msg="stop-reason row in " + path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
