import json
import os
import tempfile
import unittest

from builders import asst, read_tu

from attention_span import agent_health, transcript


class TestTranscriptPrimitives(unittest.TestCase):
    """The shared JSONL walk used by live analysis and incremental snapshots."""

    def _tmp(self, text):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
        self.addCleanup(os.unlink, path)
        return path

    def _tmp_bytes(self, contents):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "wb") as stream:
            stream.write(contents)
        self.addCleanup(os.unlink, path)
        return path

    def test_agent_health_reexports_the_canonical_reader_identities(self):
        self.assertIs(agent_health.iter_lines, transcript.iter_lines)
        self.assertIs(agent_health.iter_objects, transcript.iter_objects)

    def test_iter_objects_yields_decoded_and_skips_undecodable(self):
        path = self._tmp('{"a": 1}\nnot json\n{"b": 2}\n')

        self.assertEqual(
            list(transcript.iter_objects(path)),
            [(1, {"a": 1}), (3, {"b": 2})],
        )

    def test_iter_lines_surfaces_decode_failures_as_none(self):
        path = self._tmp('{"a": 1}\nnot json\n')

        self.assertEqual(
            list(transcript.iter_lines(path)),
            [(1, {"a": 1}), (2, None)],
        )

    def test_invalid_utf8_is_replaced_without_dropping_the_line(self):
        path = self._tmp_bytes(b'{"text": "\xff"}\n')

        self.assertEqual(
            list(transcript.iter_lines(path)),
            [(1, {"text": "\N{REPLACEMENT CHARACTER}"})],
        )

    def test_open_failure_raises_oserror_on_first_iteration(self):
        objects = transcript.iter_objects("/no/such/transcript.jsonl")

        self.assertIs(iter(objects), objects)
        with self.assertRaises(OSError):
            next(objects)

    def test_analyze_tolerates_corrupt_trailing_line(self):
        path = self._tmp(json.dumps(asst([read_tu(1)])) + "\n{ broken\n")

        self.assertEqual(agent_health.analyze_transcript(path).total_reads, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
