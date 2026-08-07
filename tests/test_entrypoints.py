import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from attention_span import agent_health

ROOT = Path(__file__).resolve().parent.parent


class TestModuleEntrypoints(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.temp), ignore_errors=True)

    def _env(self):
        env = dict(os.environ)
        for key in [key for key in env if key.startswith("CLAUDE_HEALTH_")]:
            del env[key]
        env.update(
            {
                "HOME": str(self.temp / "home"),
                "CLAUDE_HOME": str(self.temp / "claude"),
                "NO_COLOR": "1",
                "COLUMNS": "200",
            }
        )
        return env

    def _run_module(self, module, *args, stdin=None):
        return subprocess.run(
            [sys.executable, "-m", module, *args],
            input=stdin,
            text=True,
            capture_output=True,
            cwd=ROOT,
            env=self._env(),
            check=False,
        )

    def test_agent_health_exposes_callable_main(self):
        self.assertTrue(callable(agent_health.main))

    def test_statusline_package_module_executes(self):
        proc = self._run_module("attention_span.statusline", stdin="")

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout, "")
        self.assertEqual(proc.stderr, "")

    def test_update_package_module_executes(self):
        proc = self._run_module(
            "attention_span.update",
            "--install-root",
            str(self.temp / "missing-install-root"),
        )

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout, "")
        self.assertNotIn("Traceback", proc.stderr)

    def test_agent_health_package_module_executes_debug_cli(self):
        transcript = self.temp / "transcript.jsonl"
        transcript.write_text(
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "id": "msg-1",
                        "model": "claude-opus-4-8",
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 10, "output_tokens": 1},
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "read-1",
                                "name": "Read",
                                "input": {"file_path": "README.md"},
                            }
                        ],
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

        proc = self._run_module("attention_span.agent_health", str(transcript))

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("turns", proc.stdout)
        self.assertIn('"reads": 1', proc.stderr)

    def test_release_builder_module_executes(self):
        outdir = self.temp / "dist"

        proc = self._run_module("scripts.build_release", "--outdir", str(outdir))

        self.assertEqual(proc.returncode, 0, proc.stderr)
        archives = tuple(outdir.glob("attention-span-*.tar.gz"))
        checksums = tuple(outdir.glob("attention-span-*.tar.gz.sha256"))
        self.assertEqual(len(archives), 1)
        self.assertEqual(len(checksums), 1)
