import os
import unittest

from builders import GENUINE, GENUINE_UNREAD, HOOKDENY

from attention_span import agent_health, events


class TestEventFacade(unittest.TestCase):
    def test_agent_health_reexports_canonical_classifier_identities(self):
        names = (
            "classify_bash",
            "classify_tool",
            "classify_error",
            "_normalize_content",
            "_tool_file_path",
            "_is_synthetic",
        )

        for name in names:
            with self.subTest(name=name):
                self.assertIs(getattr(agent_health, name), getattr(events, name))


class TestClassifyBash(unittest.TestCase):
    def test_git_status_is_read(self):
        self.assertEqual(events.classify_bash("git status"), "read")

    def test_git_diff_is_read(self):
        self.assertEqual(events.classify_bash("git diff HEAD~1"), "read")

    def test_git_commit_is_edit(self):
        self.assertEqual(events.classify_bash("git commit -m 'x'"), "edit")

    def test_sed_inplace_is_edit(self):
        self.assertEqual(events.classify_bash("sed -i 's/a/b/' f.txt"), "edit")

    def test_sed_no_inplace_is_read(self):
        self.assertEqual(events.classify_bash("sed -n '1,5p' f.txt"), "read")

    def test_perl_inplace_is_edit_and_noninplace_is_neutral(self):
        self.assertEqual(events.classify_bash("perl -pi -e 's/a/b/' f.txt"), "edit")
        self.assertEqual(
            events.classify_bash("perl -ne 'print if /a/' f.txt"), "neutral"
        )

    def test_wrapper_env_assignment_and_path_basename_are_skipped(self):
        self.assertEqual(
            events.classify_bash("sudo env MODE=read /usr/bin/git status"), "read"
        )
        self.assertEqual(
            events.classify_bash("MODE=write command /bin/rm old.txt"), "edit"
        )

    def test_mutation_is_sticky(self):
        self.assertEqual(events.classify_bash("grep x f && rm y"), "edit")

    def test_devnull_redirect_carveout(self):
        self.assertEqual(events.classify_bash("echo hi 2>/dev/null"), "neutral")

    def test_real_redirect_is_edit(self):
        self.assertEqual(events.classify_bash("cat f > out.txt"), "edit")

    def test_devnull_stdout_redirect_is_not_edit(self):
        self.assertEqual(events.classify_bash("cat f >/dev/null"), "read")

    def test_fd_dup_is_not_edit(self):
        self.assertEqual(events.classify_bash("ls -la 2>&1"), "read")

    def test_npm_test_is_neutral(self):
        self.assertEqual(events.classify_bash("npm test"), "neutral")

    def test_npm_install_is_edit(self):
        self.assertEqual(events.classify_bash("npm install lodash"), "edit")

    def test_plain_ls_is_read(self):
        self.assertEqual(events.classify_bash("ls -la"), "read")

    def test_runner_is_neutral(self):
        self.assertEqual(events.classify_bash("python script.py"), "neutral")
        self.assertEqual(events.classify_bash("cd /tmp"), "neutral")

    def test_heredoc_body_is_data_not_command(self):
        cmd = "cat <<EOF\nrm -rf /\nEOF"
        self.assertEqual(events.classify_bash(cmd), "read")

    def test_heredoc_with_redirect_is_edit(self):
        cmd = "cat <<EOF > out.txt\nhello\nEOF"
        self.assertEqual(events.classify_bash(cmd), "edit")


class TestClassifyTool(unittest.TestCase):
    def test_known_tools_retain_their_classification(self):
        self.assertEqual(events.classify_tool("Read", {"file_path": "a.py"}), "read")
        self.assertEqual(events.classify_tool("Edit", {"file_path": "a.py"}), "edit")
        self.assertEqual(events.classify_tool("Task", {}), "neutral")

    def test_file_paths_are_normalized_and_notebook_path_takes_precedence(self):
        self.assertEqual(
            events._tool_file_path("Read", {"file_path": "src/../README.md"}),
            os.path.normpath("src/../README.md"),
        )
        self.assertEqual(
            events._tool_file_path(
                "NotebookEdit",
                {"notebook_path": "notebooks/../main.ipynb", "file_path": "old"},
            ),
            os.path.normpath("notebooks/../main.ipynb"),
        )
        self.assertIsNone(events._tool_file_path("Bash", {"file_path": "a.py"}))


class TestBashReadPath(unittest.TestCase):
    """The file-scoped shell READ extractor (`cat a.py` is a reread of a.py).

    Without it, an agent that rereads a failed file through the shell and then
    retries the edit still trips the blind-loop alarm.
    """

    def path(self, command, cwd=None):
        return events._tool_file_path("Bash", {"command": command}, cwd)

    def test_single_file_dump_yields_its_normalized_path(self):
        for command in (
            "cat /repo/a.py",
            "head -50 /repo/a.py",
            "tail -n +20 /repo/a.py",
            "bat /repo/a.py",
            "sed -n '1,40p' /repo/a.py",
            "cat /repo/src/../a.py",
        ):
            with self.subTest(command=command):
                self.assertEqual(self.path(command), os.path.normpath("/repo/a.py"))

    def test_one_file_across_a_pipeline_is_still_that_file(self):

        self.assertEqual(self.path("cat /repo/a.py | grep -n foo"), "/repo/a.py")
        self.assertEqual(
            self.path("cat /repo/a.py && head -5 /repo/a.py"), "/repo/a.py"
        )

    def test_multiple_distinct_files_are_ambiguous(self):
        for command in (
            "cat /repo/a.py /repo/b.py",
            "cat /repo/a.py; cat /repo/b.py",
            "head -5 /repo/a.py | tail -2 /repo/b.py",
        ):
            with self.subTest(command=command):
                self.assertIsNone(self.path(command))

    def test_search_and_listing_reads_carry_no_path(self):

        for command in (
            "grep -n needle /repo/a.py",
            "rg needle /repo/a.py",
            "grep -r foo",
            "ls -la /repo/a.py",
            "wc -l /repo/a.py",
        ):
            with self.subTest(command=command):
                self.assertIsNone(self.path(command))

    def test_non_operand_tokens_are_never_paths(self):

        for command in (
            "cat",
            "head -50",
            "cat /repo/*.py",
            "cat /repo/src/",
            "cat .env",
            "cat $FILE",
            "cat 'my file.py'",
            "cat ~/notes.md",
            "sed -n 1,40p",
        ):
            with self.subTest(command=command):
                self.assertIsNone(self.path(command))

    def test_relative_operand_anchors_to_the_record_cwd(self):

        for command in ("cat config.json", "head -50 src/../config.json"):
            with self.subTest(command=command):
                self.assertEqual(
                    self.path(command, cwd="/repo"),
                    os.path.normpath("/repo/config.json"),
                )

    def test_absolute_operand_ignores_the_cwd(self):

        for cwd in (None, "/repo", "/elsewhere"):
            with self.subTest(cwd=cwd):
                self.assertEqual(self.path("cat /repo/a.py", cwd=cwd), "/repo/a.py")

    def test_relative_operand_without_a_usable_cwd_is_refused(self):

        for cwd in (None, "", "repo/sub", 42, {"path": "/repo"}):
            with self.subTest(cwd=cwd):
                self.assertIsNone(self.path("cat config.json", cwd=cwd))

    def test_relative_operand_under_a_chdir_command_is_refused(self):

        for command in (
            "cd sub && cat config.json",
            "cat config.json && cd sub",
            "pushd sub; cat config.json; popd",
            "(cd sub && cat config.json)",
        ):
            with self.subTest(command=command):
                self.assertIsNone(self.path(command, cwd="/repo"))

    def test_absolute_operand_survives_a_chdir_command(self):

        self.assertEqual(
            self.path("cd sub && cat /repo/config.json", cwd="/repo"),
            "/repo/config.json",
        )

    def test_one_file_rule_is_applied_after_anchoring(self):

        self.assertEqual(
            self.path("cat config.json && cat /repo/config.json", cwd="/repo"),
            "/repo/config.json",
        )
        self.assertIsNone(self.path("cat config.json other.json", cwd="/repo"))

    def test_edit_classified_commands_carry_no_path(self):

        for command in (
            "sed -i 's/a/b/' /repo/a.py",
            "cat /repo/a.py > /repo/b.py",
            "cat /repo/a.py && rm /repo/b.py",
        ):
            with self.subTest(command=command):
                self.assertIsNone(self.path(command))


class TestClassifyError(unittest.TestCase):
    def test_not_error_is_ok(self):
        self.assertEqual(events.classify_error(False, "all good"), "OK")
        self.assertEqual(events.classify_error(None, "all good"), "OK")

    def test_hook_gate_is_hook_deny(self):
        self.assertEqual(events.classify_error(True, HOOKDENY), "HOOK_DENY")

    def test_tool_use_error_is_genuine(self):
        self.assertEqual(events.classify_error(True, GENUINE), "GENUINE")
        self.assertEqual(events.classify_error(True, GENUINE_UNREAD), "GENUINE")

    def test_tool_use_error_wins_over_hook_deny_text(self):
        body = "<tool_use_error>failed</tool_use_error>\nblocked by hook"
        self.assertEqual(events.classify_error(True, body), "GENUINE")

    def test_bash_nonzero_exit_defaults_genuine(self):
        self.assertEqual(
            events.classify_error(True, "Exit code 1\nls: no such file"), "GENUINE"
        )

    def test_list_content_form(self):
        body = [{"type": "text", "text": GENUINE}]
        self.assertEqual(events.classify_error(True, body), "GENUINE")

    def test_malformed_text_does_not_crash(self):

        self.assertEqual(
            events.classify_error(True, [{"type": "text", "text": None}]),
            "GENUINE",
        )
        self.assertEqual(
            events.classify_error(True, [{"type": "text", "text": 5}]), "GENUINE"
        )


class TestNormalizeContent(unittest.TestCase):
    def test_str_passthrough(self):
        self.assertEqual(events._normalize_content("plain"), "plain")
        self.assertEqual(events._normalize_content(None), "")

    def test_list_of_text_parts(self):
        self.assertEqual(
            events._normalize_content([{"text": "a"}, {"text": "b"}]), "ab"
        )

    def test_malformed_text_coerced(self):

        self.assertEqual(events._normalize_content([{"text": None}]), "")
        self.assertEqual(events._normalize_content([{"text": 5}]), "5")
        self.assertEqual(
            events._normalize_content([{"text": "a"}, {"text": None}, {"text": "b"}]),
            "ab",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
