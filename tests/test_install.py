import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def payload_files():
    """The file list install.sh copies, read from install.sh itself."""
    with open(os.path.join(ROOT, "install.sh")) as f:
        source = f.read()
    body = source.partition("PAYLOAD_FILES=(")[2].partition(")")[0]
    return tuple(body.split())


class TestInstaller(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.home, ignore_errors=True)

    def run_installer(self, *args, env_overrides=None, cwd=None):
        env = dict(os.environ, CLAUDE_HOME=self.home)
        if env_overrides:
            env.update(env_overrides)
        return subprocess.run(
            ["bash", os.path.join(ROOT, "install.sh"), *args],
            cwd=ROOT if cwd is None else cwd,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_requires_python_3_11_before_changing_the_install(self):
        fake_bin = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, fake_bin, ignore_errors=True)
        fake_python = os.path.join(fake_bin, "python3")
        with open(fake_python, "w") as f:
            f.write("#!/usr/bin/env bash\nexit 1\n")
        os.chmod(fake_python, 0o755)

        proc = self.run_installer(
            env_overrides={"PATH": fake_bin + os.pathsep + os.environ["PATH"]}
        )

        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Python 3.11 or newer is required", proc.stderr)
        self.assertFalse(os.path.exists(self.installed_hooks))

    def test_version_validation_imports_the_contract_from_the_repo_root(self):
        with open(os.path.join(ROOT, "install.sh")) as f:
            installer = f.read()

        self.assertIn(
            "cd -- \"$REPO_DIR\" && PYTHONPATH='' PYTHONSAFEPATH='' python3 - \"$REPO_DIR\"",
            installer,
        )
        self.assertIn(
            "from attention_span.release_contract import parse_version",
            installer,
        )
        self.assertNotIn("sys.path." + "insert", installer)
        self.assertNotIn("sys.path." + "append", installer)

    def test_installer_ignores_a_caller_directory_package(self):
        caller = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, caller, ignore_errors=True)
        poison = os.path.join(caller, "attention_span")
        os.mkdir(poison)
        with open(os.path.join(poison, "__init__.py"), "w") as f:
            f.write("")
        with open(os.path.join(poison, "release_contract.py"), "w") as f:
            f.write('raise RuntimeError("caller package imported")\n')

        proc = self.run_installer(
            cwd=caller,
            env_overrides={
                "PYTHONPATH": caller,
                "PYTHONSAFEPATH": "1",
            },
        )

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("caller package imported", proc.stderr)

    def run_installer_piped(self, *args, env_overrides=None, cwd=None):
        """Run install.sh the way `curl ... | bash -s -- ...` does.

        Nothing puts the script on disk for the shell, so BASH_SOURCE is empty -
        exactly the condition the working-directory trust bug depended on.
        """
        env = dict(os.environ, CLAUDE_HOME=self.home)
        if env_overrides:
            env.update(env_overrides)
        with open(os.path.join(ROOT, "install.sh")) as f:
            script = f.read()
        return subprocess.run(
            ["bash", "-s", "--", *args],
            input=script,
            cwd=ROOT if cwd is None else cwd,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=120,
        )

    def make_checkout(self, license_text, with_installer=False):
        """A directory that looks exactly like a real checkout to the installer."""
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        os.mkdir(os.path.join(root, "attention_span"))
        for relative in payload_files():
            shutil.copy2(os.path.join(ROOT, relative), os.path.join(root, relative))
        if with_installer:
            shutil.copy2(
                os.path.join(ROOT, "install.sh"), os.path.join(root, "install.sh")
            )
        with open(os.path.join(root, "LICENSE"), "w") as f:
            f.write(license_text)
        return root

    def make_tarball(self, source_dir):
        """Pack a checkout the way GitHub's archive endpoint does (one top dir)."""
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        path = os.path.join(directory, "attention-span.tar.gz")
        with tarfile.open(path, "w:gz") as archive:
            archive.add(source_dir, arcname="attention-span-main")
        return path

    def installed_license(self):
        with open(os.path.join(self.installed_hooks, "current", "LICENSE")) as f:
            return f.read()

    def test_piped_run_installs_the_download_not_the_working_directory(self):

        attacker = self.make_checkout("ATTACKER PAYLOAD\n")
        marker = os.path.join(attacker, "attacker-code-ran")
        with open(os.path.join(attacker, "attention_span", "release_contract.py")) as f:
            contract = f.read()
        with open(
            os.path.join(attacker, "attention_span", "release_contract.py"), "w"
        ) as f:
            f.write(f"open({marker!r}, 'a').close()\n" + contract)
        official = self.make_checkout("OFFICIAL PAYLOAD\n", with_installer=True)

        proc = self.run_installer_piped(
            cwd=attacker,
            env_overrides={
                "ATTENTION_SPAN_TARBALL_URL": "file://" + self.make_tarball(official)
            },
        )

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self.installed_license(), "OFFICIAL PAYLOAD\n")
        self.assertFalse(os.path.exists(marker), "working-directory code was executed")

    def test_piped_uninstall_runs_without_downloading_anything(self):

        self.assertEqual(self.run_installer().returncode, 0)
        elsewhere = self.make_checkout("ATTACKER PAYLOAD\n")

        proc = self.run_installer_piped(
            "--uninstall",
            cwd=elsewhere,
            env_overrides={
                "ATTENTION_SPAN_TARBALL_URL": "file:///nonexistent-attention-span.tar.gz"
            },
        )

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertFalse(os.path.exists(self.installed_hooks))

    def test_incomplete_download_fails_instead_of_bootstrapping_forever(self):

        official = self.make_checkout("OFFICIAL PAYLOAD\n", with_installer=True)
        os.unlink(os.path.join(official, "attention_span", "statusline.py"))

        proc = self.run_installer_piped(
            env_overrides={
                "ATTENTION_SPAN_TARBALL_URL": "file://" + self.make_tarball(official)
            },
        )

        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("incomplete", proc.stderr)
        self.assertFalse(os.path.exists(self.installed_hooks))

    @property
    def settings(self):
        return os.path.join(self.home, "settings.json")

    @property
    def installed_hooks(self):
        return os.path.join(self.home, "hooks", "attention-span")

    def read_settings(self):
        with open(self.settings) as f:
            return json.load(f)

    def test_installs_release_and_publishes_stable_aliases(self):

        proc = self.run_installer()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(os.path.islink(os.path.join(self.installed_hooks, "current")))
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    self.installed_hooks,
                    "current",
                    "attention_span",
                    "agent_health.py",
                )
            )
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    self.installed_hooks,
                    "current",
                    "attention_span",
                    "analysis.py",
                )
            )
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    self.installed_hooks,
                    "current",
                    "attention_span",
                    "detectors.py",
                )
            )
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    self.installed_hooks,
                    "current",
                    "attention_span",
                    "events.py",
                )
            )
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    self.installed_hooks,
                    "current",
                    "attention_span",
                    "reducer.py",
                )
            )
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    self.installed_hooks,
                    "current",
                    "attention_span",
                    "transcript.py",
                )
            )
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    self.installed_hooks,
                    "current",
                    "attention_span",
                    "verdicts.py",
                )
            )
        )
        self.assertFalse(
            os.path.exists(
                os.path.join(self.installed_hooks, "current", "agent_health.py")
            )
        )
        self.assertFalse(
            os.path.exists(os.path.join(self.installed_hooks, "current", "reducer.py"))
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    self.installed_hooks,
                    "current",
                    "attention_span",
                    "render.py",
                )
            )
        )
        self.assertFalse(
            os.path.exists(os.path.join(self.installed_hooks, "current", "render.py"))
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    self.installed_hooks,
                    "current",
                    "attention_span",
                    "session_ui.py",
                )
            )
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    self.installed_hooks,
                    "current",
                    "attention_span",
                    "status_catalog.py",
                )
            )
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    self.installed_hooks,
                    "current",
                    "attention_span",
                    "statusline.py",
                )
            )
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(self.installed_hooks, "current", "statusline.py")
            )
        )
        self.assertTrue(
            os.stat(
                os.path.join(self.installed_hooks, "current", "statusline.py")
            ).st_mode
            & 0o111
        )
        self.assertFalse(
            os.stat(
                os.path.join(
                    self.installed_hooks,
                    "current",
                    "attention_span",
                    "statusline.py",
                )
            ).st_mode
            & 0o111
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    self.installed_hooks,
                    "current",
                    "attention_span",
                    "subagents.py",
                )
            )
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    self.installed_hooks,
                    "current",
                    "attention_span",
                    "render_facts.py",
                )
            )
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    self.installed_hooks,
                    "current",
                    "attention_span",
                    "text.py",
                )
            )
        )
        self.assertFalse(
            os.path.exists(
                os.path.join(self.installed_hooks, "current", "session_ui.py")
            )
        )
        self.assertFalse(
            os.path.exists(
                os.path.join(self.installed_hooks, "current", "status_catalog.py")
            )
        )
        self.assertFalse(
            os.path.exists(
                os.path.join(self.installed_hooks, "current", "subagents.py")
            )
        )
        self.assertFalse(
            os.path.exists(
                os.path.join(self.installed_hooks, "current", "render_facts.py")
            )
        )
        self.assertFalse(
            os.path.exists(os.path.join(self.installed_hooks, "current", "text.py"))
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    self.installed_hooks,
                    "current",
                    "attention_span",
                    "__init__.py",
                )
            )
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    self.installed_hooks,
                    "current",
                    "attention_span",
                    "health_config.py",
                )
            )
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    self.installed_hooks,
                    "current",
                    "attention_span",
                    "release_contract.py",
                )
            )
        )
        self.assertFalse(
            os.path.exists(
                os.path.join(self.installed_hooks, "current", "health_config.py")
            )
        )
        self.assertFalse(
            os.path.exists(
                os.path.join(self.installed_hooks, "current", "release_contract.py")
            )
        )
        self.assertTrue(
            os.path.isfile(os.path.join(self.installed_hooks, "current", "LICENSE"))
        )
        self.assertTrue(
            os.path.islink(os.path.join(self.installed_hooks, "launcher.py"))
        )
        self.assertEqual(
            os.readlink(os.path.join(self.installed_hooks, "launcher.py")),
            os.path.join("current", "launcher.py"),
        )
        self.assertTrue(os.path.islink(os.path.join(self.installed_hooks, "VERSION")))
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    self.installed_hooks,
                    "current",
                    "attention_span",
                    "update.py",
                )
            )
        )
        self.assertTrue(
            os.path.isfile(os.path.join(self.installed_hooks, "current", "update.py"))
        )
        self.assertEqual(
            os.stat(os.path.join(self.installed_hooks, "current", "update.py")).st_mode
            & 0o777,
            0o755,
        )
        self.assertEqual(
            os.stat(
                os.path.join(
                    self.installed_hooks,
                    "current",
                    "attention_span",
                    "update.py",
                )
            ).st_mode
            & 0o777,
            0o644,
        )

        self.assertFalse(os.path.exists(os.path.join(self.home, "skills", "health")))

    def test_uninstall_removes_hooks_without_touching_real_tmp_telemetry(self):

        fd, live_cache = tempfile.mkstemp(
            prefix="claude-statusline-user-state-", dir="/tmp"
        )
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(live_cache) and os.unlink(live_cache))
        installed = self.run_installer()
        self.assertEqual(installed.returncode, 0, installed.stderr)
        removed = self.run_installer("--uninstall")
        self.assertEqual(removed.returncode, 0, removed.stderr)
        self.assertFalse(os.path.exists(self.installed_hooks))
        self.assertTrue(os.path.exists(live_cache))

    def test_uninstall_does_not_glob_shared_tmp_state(self):
        with open(os.path.join(ROOT, "install.sh")) as f:
            installer = f.read()

        self.assertNotIn("rm -f /tmp/claude-statusline-*", installer)

    def test_fresh_install_uninstall_removes_owned_setting(self):
        self.assertEqual(self.run_installer().returncode, 0)
        self.assertIn("statusLine", self.read_settings())
        self.assertIn("launcher.py", self.read_settings()["statusLine"]["command"])

        removed = self.run_installer("--uninstall")
        self.assertEqual(removed.returncode, 0, removed.stderr)
        self.assertNotIn("statusLine", self.read_settings())

    def test_reinstall_keeps_one_original_restore_point(self):
        original = {
            "theme": "dark",
            "statusLine": {"type": "command", "command": "old-statusline"},
        }
        with open(self.settings, "w") as f:
            json.dump(original, f)

        self.assertEqual(self.run_installer().returncode, 0)
        self.assertEqual(self.run_installer().returncode, 0)
        removed = self.run_installer("--uninstall")

        self.assertEqual(removed.returncode, 0, removed.stderr)
        self.assertEqual(self.read_settings(), original)

    def test_rerun_migrates_owned_flat_install_to_auto_updating_launcher(self):
        os.makedirs(self.installed_hooks)
        old_statusline = os.path.join(self.installed_hooks, "statusline.py")
        with open(old_statusline, "w") as f:
            f.write("# old flat install\n")
        with open(self.settings, "w") as f:
            json.dump(
                {
                    "statusLine": {
                        "type": "command",
                        "command": "python3 " + old_statusline,
                    }
                },
                f,
            )
        open(self.settings + ".statusline-created-by-attention-span", "a").close()

        migrated = self.run_installer()

        self.assertEqual(migrated.returncode, 0, migrated.stderr)
        self.assertIn("launcher.py", self.read_settings()["statusLine"]["command"])
        self.assertTrue(os.path.islink(os.path.join(self.installed_hooks, "current")))
        self.assertTrue(
            os.path.isfile(os.path.join(self.installed_hooks, "current", "update.py"))
        )
        self.assertIn("Automatic stable-release updates: on", migrated.stdout)

    def test_install_does_not_overwrite_another_hook_file(self):
        hooks = os.path.join(self.home, "hooks")
        os.makedirs(hooks)
        other_hook = os.path.join(hooks, "statusline.py")
        with open(other_hook, "w") as f:
            f.write("# another tool\n")
        original = {
            "statusLine": {"type": "command", "command": "python3 " + other_hook}
        }
        with open(self.settings, "w") as f:
            json.dump(original, f)

        self.assertEqual(self.run_installer().returncode, 0)
        with open(other_hook) as f:
            self.assertEqual(f.read(), "# another tool\n")

        self.assertEqual(self.run_installer("--uninstall").returncode, 0)
        self.assertEqual(self.read_settings(), original)
        with open(other_hook) as f:
            self.assertEqual(f.read(), "# another tool\n")

    def test_uninstall_preserves_unrelated_changes_made_after_install(self):
        with open(self.settings, "w") as f:
            json.dump({"theme": "dark"}, f)
        self.assertEqual(self.run_installer().returncode, 0)

        current = self.read_settings()
        current["newSetting"] = True
        with open(self.settings, "w") as f:
            json.dump(current, f)

        removed = self.run_installer("--uninstall")
        restored = self.read_settings()
        self.assertEqual(removed.returncode, 0, removed.stderr)
        self.assertEqual(restored, {"theme": "dark", "newSetting": True})

    def test_uninstall_does_not_replace_a_new_statusline_choice(self):
        self.assertEqual(self.run_installer().returncode, 0)
        current = self.read_settings()
        current["statusLine"] = {
            "type": "command",
            "command": "python3 /other/tool/statusline.py",
        }
        with open(self.settings, "w") as f:
            json.dump(current, f)

        removed = self.run_installer("--uninstall")

        self.assertEqual(removed.returncode, 0, removed.stderr)
        self.assertEqual(self.read_settings()["statusLine"], current["statusLine"])

    def test_malformed_settings_stop_before_install_changes(self):
        broken = "{not valid json\n"
        with open(self.settings, "w") as f:
            f.write(broken)

        proc = self.run_installer()

        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Cannot safely update", proc.stderr)
        with open(self.settings) as f:
            self.assertEqual(f.read(), broken)

        self.assertFalse(os.path.exists(os.path.join(self.home, "hooks")))
        self.assertFalse(os.path.exists(os.path.join(self.home, "skills")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
