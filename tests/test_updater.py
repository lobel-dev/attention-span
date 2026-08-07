import gzip
import hashlib
import io
import json
import os
import runpy
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from attention_span import update

ROOT = Path(__file__).resolve().parent.parent


class TestLauncher(unittest.TestCase):
    def setUp(self):
        self.install_root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.install_root), ignore_errors=True)
        release = self.install_root / "releases" / "1.0.0-test"
        release.mkdir(parents=True)
        (release / "statusline.py").write_text("# statusline\n", encoding="utf-8")
        (release / "update.py").write_text("# updater\n", encoding="utf-8")
        os.symlink("releases/1.0.0-test", str(self.install_root / "current"))

    def test_resolves_one_physical_release_and_throttles_update_spawn(self):
        import launcher

        release = launcher.resolve_release(self.install_root)
        self.assertEqual(
            release, (self.install_root / "releases" / "1.0.0-test").resolve()
        )
        self.assertTrue(launcher.claim_update_check(self.install_root, now=1_000_000))
        self.assertFalse(launcher.claim_update_check(self.install_root, now=1_000_001))
        self.assertTrue(
            launcher.claim_update_check(
                self.install_root,
                now=1_000_000 + launcher.UPDATE_INTERVAL_SECONDS + 1,
            )
        )

    def test_auto_update_is_on_by_default_with_explicit_opt_out(self):
        import launcher

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(launcher.auto_update_enabled())
        with mock.patch.dict(
            os.environ, {"CLAUDE_HEALTH_AUTO_UPDATE": "0"}, clear=True
        ):
            self.assertFalse(launcher.auto_update_enabled())

    def test_failed_update_spawn_does_not_suppress_rendering(self):
        import launcher

        with (
            mock.patch.dict(
                os.environ, {"CLAUDE_HEALTH_AUTO_UPDATE": "1"}, clear=False
            ),
            mock.patch.object(launcher, "claim_update_check", return_value=True),
            mock.patch.object(
                launcher, "spawn_update", side_effect=OSError("cannot spawn")
            ) as spawn_update,
            mock.patch.object(launcher.os, "execv") as execv,
        ):
            self.assertEqual(launcher.main(self.install_root), 0)

        spawn_update.assert_called_once()
        execv.assert_called_once()
        self.assertEqual(
            Path(execv.call_args.args[1][1]),
            (self.install_root / "releases" / "1.0.0-test" / "statusline.py").resolve(),
        )

    def test_detached_update_spawn_preserves_the_root_shim_argv_contract(self):
        import launcher

        release = (self.install_root / "current").resolve()
        with mock.patch.object(launcher.subprocess, "Popen") as popen:
            launcher.spawn_update(self.install_root, release)

        popen.assert_called_once_with(
            [
                sys.executable,
                str(release / "update.py"),
                "--install-root",
                str(self.install_root),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )

    def test_root_update_shim_inserts_release_root_and_preserves_sys_argv(self):
        shim_path = ROOT / "update.py"
        shim_sys_path = list(sys.path)
        root = str(ROOT)
        root_count = shim_sys_path.count(root)
        shim_argv = [str(shim_path), "--install-root", "/tmp/update-shim-target"]
        observed_argv = []

        def package_main():
            observed_argv.append(tuple(sys.argv))
            return 7

        with (
            mock.patch.object(sys, "path", shim_sys_path),
            mock.patch.object(sys, "argv", shim_argv),
            mock.patch.object(update, "main", side_effect=package_main) as main,
            self.assertRaises(SystemExit) as raised,
        ):
            runpy.run_path(str(shim_path), run_name="__main__")

        self.assertEqual(raised.exception.code, 7)
        self.assertEqual(shim_sys_path[0], root)
        self.assertEqual(shim_sys_path.count(root), root_count + 1)
        self.assertEqual(observed_argv, [tuple(shim_argv)])
        main.assert_called_once_with()


class TestUpdater(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.temp), ignore_errors=True)
        self.install_root = self.temp / ".claude" / "hooks" / "attention-span"
        old_release = self.install_root / "releases" / "0.9.0-old"
        old_release.mkdir(parents=True)
        (old_release / "VERSION").write_text("0.9.0\n", encoding="utf-8")
        (old_release / "statusline.py").write_text("# old\n", encoding="utf-8")
        (old_release / "launcher.py").write_text("# old launcher\n", encoding="utf-8")
        os.symlink("releases/0.9.0-old", str(self.install_root / "current"))
        os.symlink("current/launcher.py", str(self.install_root / "launcher.py"))
        os.symlink("current/VERSION", str(self.install_root / "VERSION"))

        skill = self.temp / ".claude" / "skills" / "health"
        skill.mkdir(parents=True)
        (skill / ".attention-span-owned").touch()
        (skill / "SKILL.md").write_text(
            "python3 {} --inspect\n".format(self.install_root / "launcher.py"),
            encoding="utf-8",
        )

        from scripts import build_release

        result = build_release.build_release(str(self.temp / "dist"))
        self.archive = Path(result.archive_path).read_bytes()
        self.version = result.version
        self.digest = hashlib.sha256(self.archive).hexdigest()

    def archive_with_extra_payload(self, *extras):
        from scripts import build_release

        root_name = f"attention-span-{self.version}/"
        manifest_name = root_name + "release-manifest.json"
        with gzip.GzipFile(fileobj=io.BytesIO(self.archive), mode="rb") as compressed:
            tar_bytes = compressed.read()
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as archive:
            members_by_name = {member.name: member for member in archive.getmembers()}
            manifest = json.load(archive.extractfile(members_by_name[manifest_name]))
            members = tuple(
                (
                    member.name,
                    member.mode,
                    archive.extractfile(member).read(),
                )
                for member in members_by_name.values()
                if member.name != manifest_name
            )

        extra_entries = tuple(
            {
                "path": relative,
                "size": len(contents),
                "sha256": hashlib.sha256(contents).hexdigest(),
                "mode": "0644",
            }
            for relative, contents in extras
        )
        updated_manifest = {
            **manifest,
            "files": (*manifest["files"], *extra_entries),
        }
        extra_members = tuple(
            (root_name + relative, 0o644, contents) for relative, contents in extras
        )
        manifest_bytes = (
            json.dumps(updated_manifest, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        all_members = tuple(
            sorted(
                (
                    *members,
                    *extra_members,
                    (manifest_name, 0o644, manifest_bytes),
                ),
                key=lambda member: member[0],
            )
        )

        compressed = io.BytesIO()
        with gzip.GzipFile(fileobj=compressed, mode="wb", mtime=0) as stream:
            stream.write(build_release._deterministic_tar_bytes(all_members))
        return compressed.getvalue()

    def metadata(self, immutable=True):
        from attention_span import release_contract

        asset_name = f"attention-span-{self.version}.tar.gz"
        return {
            "tag_name": self.version,
            "draft": False,
            "prerelease": False,
            "immutable": immutable,
            "assets": [
                {
                    "name": asset_name,
                    "state": "uploaded",
                    "size": len(self.archive),
                    "digest": "sha256:" + self.digest,
                    "browser_download_url": (
                        f"https://github.com/{release_contract.GITHUB_REPOSITORY}/releases/download/v{self.version}/{asset_name}"
                    ),
                }
            ],
        }

    def test_updater_logic_is_packaged_and_requires_both_entrypoints(self):
        self.assertEqual(update.__name__, "attention_span.update")
        self.assertIn("attention_span/analysis.py", update.REQUIRED_FILES)
        self.assertIn("attention_span/detectors.py", update.REQUIRED_FILES)
        self.assertIn("attention_span/events.py", update.REQUIRED_FILES)
        self.assertIn("attention_span/render_facts.py", update.REQUIRED_FILES)
        self.assertIn("attention_span/reducer.py", update.REQUIRED_FILES)
        self.assertIn("attention_span/text.py", update.REQUIRED_FILES)
        self.assertIn("attention_span/transcript.py", update.REQUIRED_FILES)
        self.assertIn("attention_span/verdicts.py", update.REQUIRED_FILES)
        self.assertIn("update.py", update.REQUIRED_FILES)
        self.assertIn("attention_span/update.py", update.REQUIRED_FILES)

    def test_selects_only_newer_immutable_release_with_exact_asset(self):
        selected = update.select_update(self.metadata(), "0.9.0")
        self.assertEqual(selected.version, self.version)
        self.assertEqual(selected.digest, self.digest)
        self.assertIsNone(update.select_update(self.metadata(), self.version))
        with self.assertRaisesRegex(ValueError, "immutable"):
            update.select_update(self.metadata(immutable=False), "0.9.0")

    def test_installs_verified_archive_and_atomically_switches_current(self):
        old_target = os.readlink(str(self.install_root / "current"))
        installed = update.install_archive(
            self.install_root,
            self.archive,
            expected_version=self.version,
            expected_digest=self.digest,
            expected_size=len(self.archive),
        )

        self.assertEqual(installed, self.version)
        self.assertNotEqual(os.readlink(str(self.install_root / "current")), old_target)
        current = (self.install_root / "current").resolve()
        self.assertEqual((current / "VERSION").read_text().strip(), self.version)
        self.assertTrue((current / "update.py").is_file())
        self.assertTrue((current / "attention_span" / "update.py").is_file())
        self.assertEqual((current / "update.py").stat().st_mode & 0o777, 0o755)
        self.assertEqual(
            (current / "attention_span" / "update.py").stat().st_mode & 0o777,
            0o644,
        )
        self.assertTrue((current / "attention_span" / "agent_health.py").is_file())
        self.assertTrue((current / "attention_span" / "analysis.py").is_file())
        self.assertTrue((current / "attention_span" / "detectors.py").is_file())
        self.assertTrue((current / "attention_span" / "events.py").is_file())
        self.assertTrue((current / "attention_span" / "__init__.py").is_file())
        self.assertTrue((current / "attention_span" / "health_config.py").is_file())
        self.assertTrue((current / "attention_span" / "reducer.py").is_file())
        self.assertTrue((current / "attention_span" / "release_contract.py").is_file())
        self.assertTrue((current / "attention_span" / "render.py").is_file())
        self.assertTrue((current / "attention_span" / "render_facts.py").is_file())
        self.assertTrue((current / "attention_span" / "session_ui.py").is_file())
        self.assertTrue((current / "attention_span" / "status_catalog.py").is_file())
        self.assertTrue((current / "attention_span" / "statusline.py").is_file())
        self.assertTrue((current / "attention_span" / "subagents.py").is_file())
        self.assertTrue((current / "attention_span" / "text.py").is_file())
        self.assertTrue((current / "attention_span" / "transcript.py").is_file())
        self.assertTrue((current / "attention_span" / "verdicts.py").is_file())
        self.assertTrue((current / "statusline.py").is_file())
        self.assertEqual((current / "statusline.py").stat().st_mode & 0o777, 0o755)
        self.assertEqual(
            (current / "LICENSE").read_text(), (ROOT / "LICENSE").read_text()
        )
        self.assertEqual(
            (current / "attention_span" / "statusline.py").stat().st_mode & 0o777,
            0o644,
        )
        self.assertFalse((current / "agent_health.py").exists())
        self.assertFalse((current / "health_config.py").exists())
        self.assertFalse((current / "reducer.py").exists())
        self.assertFalse((current / "release_contract.py").exists())
        self.assertFalse((current / "render.py").exists())
        self.assertFalse((current / "render_facts.py").exists())
        self.assertFalse((current / "session_ui.py").exists())
        self.assertFalse((current / "status_catalog.py").exists())
        self.assertFalse((current / "subagents.py").exists())
        self.assertFalse((current / "text.py").exists())
        self.assertTrue((self.install_root / "launcher.py").is_symlink())
        self.assertEqual(
            os.readlink(str(self.install_root / "launcher.py")), "current/launcher.py"
        )
        self.assertTrue((self.install_root / "VERSION").is_symlink())
        self.assertEqual(
            (self.install_root / "VERSION").read_text().strip(), self.version
        )
        skill = self.temp / ".claude" / "skills" / "health" / "SKILL.md"
        self.assertEqual(
            skill.read_text(),
            "python3 {} --inspect\n".format(self.install_root / "launcher.py"),
        )

    def test_activation_failure_leaves_every_live_entry_point_on_old_release(self):
        old_target = os.readlink(str(self.install_root / "current"))
        launcher_target = os.readlink(str(self.install_root / "launcher.py"))
        version_target = os.readlink(str(self.install_root / "VERSION"))
        skill = self.temp / ".claude" / "skills" / "health" / "SKILL.md"
        old_skill = skill.read_text()

        with (
            mock.patch.object(
                update, "_switch_current", side_effect=OSError("blocked")
            ),
            self.assertRaisesRegex(OSError, "blocked"),
        ):
            update.install_archive(
                self.install_root,
                self.archive,
                expected_version=self.version,
                expected_digest=self.digest,
                expected_size=len(self.archive),
            )

        self.assertEqual(os.readlink(str(self.install_root / "current")), old_target)
        self.assertEqual(
            os.readlink(str(self.install_root / "launcher.py")), launcher_target
        )
        self.assertEqual(
            os.readlink(str(self.install_root / "VERSION")), version_target
        )
        self.assertEqual((self.install_root / "VERSION").read_text().strip(), "0.9.0")
        self.assertEqual(skill.read_text(), old_skill)

    def test_digest_failure_keeps_current_release(self):
        old_target = os.readlink(str(self.install_root / "current"))
        with self.assertRaisesRegex(ValueError, "digest"):
            update.install_archive(
                self.install_root,
                self.archive,
                expected_version=self.version,
                expected_digest="0" * 64,
                expected_size=len(self.archive),
            )
        self.assertEqual(os.readlink(str(self.install_root / "current")), old_target)

    def test_corrupt_existing_release_directory_is_discarded_and_reinstalled(self):

        payload = update._archive_payload(self.archive, self.version)
        corrupt = (
            self.install_root / "releases" / (self.version + "-" + self.digest[:12])
        )
        corrupt.mkdir()
        (corrupt / "VERSION").write_text(self.version + "\n", encoding="utf-8")
        (corrupt / "launcher.py").write_text("# truncated\n", encoding="utf-8")
        self.assertFalse(update._release_matches(corrupt, payload))

        installed = update.install_archive(
            self.install_root,
            self.archive,
            expected_version=self.version,
            expected_digest=self.digest,
            expected_size=len(self.archive),
        )

        self.assertEqual(installed, self.version)
        self.assertTrue(update._release_matches(corrupt, payload))
        self.assertEqual((self.install_root / "current").resolve(), corrupt.resolve())

        self.assertEqual(
            sorted(
                path.name
                for path in (self.install_root / "releases").iterdir()
                if path.name.startswith(".")
            ),
            [],
        )

    def test_archive_paths_cannot_escape_release_directory(self):
        for unsafe in (
            ".",
            "../statusline.py",
            "/statusline.py",
            "a/../../b",
            "a//b",
            "./a",
            "a/",
            "bad\x00.py",
        ):
            with (
                self.subTest(unsafe=unsafe),
                self.assertRaisesRegex(ValueError, "path"),
            ):
                update._safe_relative_path(unsafe)

    def test_installs_verified_archive_with_nested_payload_path(self):
        from scripts import build_release

        source = self.temp / "nested-source"
        for relative, _mode in build_release.PAYLOAD:
            destination = source / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        nested = source / "attention_span" / "nested.py"
        nested.parent.mkdir(exist_ok=True)
        nested.write_text("NESTED = True\n", encoding="utf-8")
        payload = build_release.PAYLOAD + (("attention_span/nested.py", 0o644),)

        with (
            mock.patch.object(build_release, "ROOT", source),
            mock.patch.object(build_release, "PAYLOAD", payload),
        ):
            result = build_release.build_release(str(self.temp / "nested-dist"))
        archive = Path(result.archive_path).read_bytes()
        digest = hashlib.sha256(archive).hexdigest()

        update.install_archive(
            self.install_root,
            archive,
            expected_version=self.version,
            expected_digest=digest,
            expected_size=len(archive),
        )

        installed = self.install_root / "current" / "attention_span" / "nested.py"
        self.assertEqual(installed.read_text(), "NESTED = True\n")

    def test_rejects_payload_paths_that_collide_on_portable_filesystems(self):
        colliding_payloads = (
            (("STATUSLINE.py", b"# case collision\n"),),
            (
                ("caf\N{LATIN SMALL LETTER E WITH ACUTE}.py", b"# composed\n"),
                ("cafe\u0301.py", b"# decomposed\n"),
            ),
        )

        for extras in colliding_payloads:
            with self.subTest(extras=extras):
                archive = self.archive_with_extra_payload(*extras)
                digest = hashlib.sha256(archive).hexdigest()
                with self.assertRaisesRegex(ValueError, "colliding"):
                    update.install_archive(
                        self.install_root,
                        archive,
                        expected_version=self.version,
                        expected_digest=digest,
                        expected_size=len(archive),
                    )

    def test_staged_release_is_reverified_before_activation(self):
        old_target = os.readlink(str(self.install_root / "current"))
        with (
            mock.patch.object(update, "_release_matches", return_value=False),
            self.assertRaisesRegex(ValueError, "staged release"),
        ):
            update.install_archive(
                self.install_root,
                self.archive,
                expected_version=self.version,
                expected_digest=self.digest,
                expected_size=len(self.archive),
            )

        self.assertEqual(os.readlink(str(self.install_root / "current")), old_target)

    def test_archive_missing_required_file_is_never_activated(self):
        from scripts import build_release

        incomplete_payload = tuple(
            item for item in build_release.PAYLOAD if item[0] != "statusline.py"
        )
        with mock.patch.object(build_release, "PAYLOAD", incomplete_payload):
            result = build_release.build_release(str(self.temp / "incomplete-dist"))
        archive = Path(result.archive_path).read_bytes()
        digest = hashlib.sha256(archive).hexdigest()
        old_target = os.readlink(str(self.install_root / "current"))

        with self.assertRaisesRegex(ValueError, "required files"):
            update.install_archive(
                self.install_root,
                archive,
                expected_version=self.version,
                expected_digest=digest,
                expected_size=len(archive),
            )

        self.assertEqual(os.readlink(str(self.install_root / "current")), old_target)

    def test_check_once_fetches_and_installs_release(self):
        with (
            mock.patch.object(update, "fetch_latest", return_value=self.metadata()),
            mock.patch.object(update, "download_asset", return_value=self.archive),
        ):
            result = update.check_once(self.install_root)

        self.assertEqual(result, "updated")
        state = json.loads((self.install_root / "update-state.json").read_text())
        self.assertEqual(state["status"], "updated")
        self.assertEqual(state["installed_version"], self.version)

    def test_background_failure_is_silent_and_records_state(self):
        with mock.patch.object(update, "fetch_latest", side_effect=OSError("offline")):
            self.assertEqual(
                update.main(["--install-root", str(self.install_root)]),
                0,
            )
        state = json.loads((self.install_root / "update-state.json").read_text())
        self.assertEqual(state["status"], "error")
        self.assertIn("offline", state["error"])


class TestInstalledLauncher(unittest.TestCase):
    def test_installed_launcher_renders_with_auto_update_disabled(self):
        claude_home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(claude_home), ignore_errors=True)
        (claude_home / "skills").mkdir()
        installed = subprocess.run(
            ["bash", str(ROOT / "install.sh")],
            cwd=str(ROOT),
            env=dict(os.environ, CLAUDE_HOME=str(claude_home)),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(installed.returncode, 0, installed.stderr)

        settings = json.loads((claude_home / "settings.json").read_text())
        self.assertIn("launcher.py", settings["statusLine"]["command"])
        self.assertTrue(
            (claude_home / "hooks" / "attention-span" / "current").is_symlink()
        )
        self.assertTrue(
            (
                claude_home / "hooks" / "attention-span" / "current" / "update.py"
            ).is_file()
        )
        hook_root = claude_home / "hooks" / "attention-span"
        current = (hook_root / "current").resolve()
        self.assertTrue((current / "attention_span" / "update.py").is_file())
        updater = subprocess.run(
            [
                sys.executable,
                str(current / "update.py"),
                "--install-root",
                str(claude_home / "missing-install-root"),
            ],
            cwd=str(claude_home),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(updater.returncode, 0, updater.stderr)
        self.assertNotIn("Traceback", updater.stderr)
        env = dict(
            os.environ,
            CLAUDE_HOME=str(claude_home),
            CLAUDE_HEALTH_AUTO_UPDATE="0",
            NO_COLOR="1",
            COLUMNS="120",
        )
        rendered = subprocess.run(
            [sys.executable, str(hook_root / "launcher.py")],
            cwd=str(claude_home),
            env=env,
            input=json.dumps(
                {
                    "transcript_path": str(self.temp_transcript(claude_home)),
                    "model": {"id": "claude-opus-4-8"},
                }
            ),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(rendered.returncode, 0, rendered.stderr)
        self.assertIn("WAIT FOR SESSION DATA", rendered.stdout.upper())

    @staticmethod
    def temp_transcript(claude_home):
        return claude_home / "projects" / "not-written-yet.jsonl"


if __name__ == "__main__":
    unittest.main(verbosity=2)
