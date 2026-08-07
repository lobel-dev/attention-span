import ast
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import tokenize
import unittest
from pathlib import Path
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestImportBootstrapPolicy(unittest.TestCase):
    def test_only_executable_shims_mutate_the_import_path(self):
        tracked = subprocess.check_output(
            ["git", "ls-files", "-z", "--", "*.py"],
            cwd=ROOT,
        ).decode("utf-8")
        path_mutations = []
        e402_suppressions = []

        for relative in filter(None, tracked.split("\0")):
            source = Path(ROOT, relative).read_text(encoding="utf-8")
            lines = source.splitlines()
            for node in ast.walk(ast.parse(source, filename=relative)):
                function = node.func if isinstance(node, ast.Call) else None
                path = function.value if isinstance(function, ast.Attribute) else None
                system = path.value if isinstance(path, ast.Attribute) else None
                if (
                    isinstance(function, ast.Attribute)
                    and function.attr in {"append", "insert"}
                    and isinstance(path, ast.Attribute)
                    and path.attr == "path"
                    and isinstance(system, ast.Name)
                    and system.id == "sys"
                ):
                    path_mutations.append((relative, lines[node.lineno - 1].strip()))
            for token in tokenize.generate_tokens(io.StringIO(source).readline):
                if (
                    token.type == tokenize.COMMENT
                    and "noqa:" in token.string
                    and "E402" in token.string
                ):
                    e402_suppressions.append((relative, token.string))

        root_insertion = "sys" + (
            ".path.insert(0, os.path.dirname(os.path.abspath(__file__)))"
        )
        self.assertEqual(
            path_mutations,
            [
                ("statusline.py", root_insertion),
                ("update.py", root_insertion),
            ],
        )
        self.assertEqual(e402_suppressions, [])


class TestReleaseBuilder(unittest.TestCase):
    def setUp(self):
        self.out = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.out, ignore_errors=True)

    def _builder(self):
        from scripts import build_release

        return build_release

    def _archive(self, path):
        with gzip.open(path, "rb") as compressed:
            tar_bytes = compressed.read()
        return tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:")

    def _version_root(self, version):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        with open(os.path.join(root, "VERSION"), "w") as f:
            f.write(version + "\n")
        return Path(root)

    def test_version_is_ascii_strict_semver_and_tag_must_match(self):
        builder = self._builder()
        version = builder.read_version()

        builder.validate_tag(version, version)
        with self.assertRaisesRegex(ValueError, "does not match VERSION"):
            builder.validate_tag("9.9.9", version)
        # The v-prefixed spelling is the OLD contract; the platform tombstoned
        # those names, so shipping one now must fail loudly, not quietly pass.
        with self.assertRaisesRegex(ValueError, "does not match VERSION"):
            builder.validate_tag("v" + version, version)

        for valid in ("0.0.0", "1.2.3", "10.20.30"):
            with self.subTest(valid=valid):
                self.assertEqual(builder.read_version(self._version_root(valid)), valid)

        invalid_versions = (
            "01.2.3",
            "1.02.3",
            "1.2.03",
            "1.2",
            "1.2.3-beta",
            "1.2.3٤",
            "١.2.3",
        )
        for invalid in invalid_versions:
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "strict SemVer"),
            ):
                builder.read_version(self._version_root(invalid))

    def test_build_is_deterministic_and_writes_checksum(self):
        builder = self._builder()

        first = builder.build_release(self.out)
        with open(first.archive_path, "rb") as f:
            first_bytes = f.read()
        second = builder.build_release(self.out)
        with open(second.archive_path, "rb") as f:
            second_bytes = f.read()

        self.assertEqual(first_bytes, second_bytes)
        digest = hashlib.sha256(first_bytes).hexdigest()
        with open(first.checksum_path) as f:
            self.assertEqual(
                f.read(), digest + "  " + os.path.basename(first.archive_path) + "\n"
            )

    def test_manifest_covers_the_installable_payload(self):
        builder = self._builder()
        result = builder.build_release(self.out)
        root = "attention-span-" + result.version

        with self._archive(result.archive_path) as archive:
            members = {member.name: member for member in archive.getmembers()}
            manifest_name = root + "/release-manifest.json"
            manifest = json.load(archive.extractfile(members[manifest_name]))

            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["name"], "attention-span")
            self.assertEqual(manifest["version"], result.version)
            self.assertEqual(
                [entry["path"] for entry in manifest["files"]],
                sorted(path for path, _mode in builder.PAYLOAD),
            )
            payload_paths = {entry["path"] for entry in manifest["files"]}

            self.assertIn("LICENSE", payload_paths)
            self.assertIn("attention_span/agent_health.py", payload_paths)
            self.assertIn("attention_span/analysis.py", payload_paths)
            self.assertIn("attention_span/detectors.py", payload_paths)
            self.assertIn("attention_span/events.py", payload_paths)
            self.assertIn("attention_span/__init__.py", payload_paths)
            self.assertIn("attention_span/health_config.py", payload_paths)
            self.assertIn("attention_span/reducer.py", payload_paths)
            self.assertIn("attention_span/release_contract.py", payload_paths)
            self.assertIn("attention_span/render.py", payload_paths)
            self.assertIn("attention_span/render_facts.py", payload_paths)
            self.assertIn("attention_span/session_ui.py", payload_paths)
            self.assertIn("attention_span/status_catalog.py", payload_paths)
            self.assertIn("attention_span/statusline.py", payload_paths)
            self.assertIn("attention_span/subagents.py", payload_paths)
            self.assertIn("attention_span/text.py", payload_paths)
            self.assertIn("attention_span/transcript.py", payload_paths)
            self.assertIn("attention_span/update.py", payload_paths)
            self.assertIn("attention_span/verdicts.py", payload_paths)
            self.assertIn("statusline.py", payload_paths)
            self.assertIn("update.py", payload_paths)
            self.assertNotIn("agent_health.py", payload_paths)
            self.assertNotIn("health_config.py", payload_paths)
            self.assertNotIn("reducer.py", payload_paths)
            self.assertNotIn("release_contract.py", payload_paths)
            self.assertNotIn("render.py", payload_paths)
            self.assertNotIn("session_ui.py", payload_paths)
            self.assertNotIn("status_catalog.py", payload_paths)
            self.assertNotIn("subagents.py", payload_paths)
            self.assertNotIn("text.py", payload_paths)
            payload_modes = dict(builder.PAYLOAD)
            self.assertEqual(payload_modes["LICENSE"], 0o644)
            self.assertEqual(payload_modes["statusline.py"], 0o755)
            self.assertEqual(payload_modes["attention_span/analysis.py"], 0o644)
            self.assertEqual(payload_modes["attention_span/detectors.py"], 0o644)
            self.assertEqual(payload_modes["attention_span/events.py"], 0o644)
            self.assertEqual(payload_modes["attention_span/render_facts.py"], 0o644)
            self.assertEqual(payload_modes["attention_span/reducer.py"], 0o644)
            self.assertEqual(payload_modes["attention_span/statusline.py"], 0o644)
            self.assertEqual(payload_modes["attention_span/text.py"], 0o644)
            self.assertEqual(payload_modes["attention_span/transcript.py"], 0o644)
            self.assertEqual(payload_modes["attention_span/verdicts.py"], 0o644)
            self.assertEqual(payload_modes["update.py"], 0o755)
            self.assertEqual(payload_modes["attention_span/update.py"], 0o644)

            expected_names = {manifest_name}
            for entry in manifest["files"]:
                member_name = root + "/" + entry["path"]
                expected_names.add(member_name)
                member = members[member_name]
                contents = archive.extractfile(member).read()
                self.assertEqual(entry["size"], len(contents))
                self.assertEqual(entry["sha256"], hashlib.sha256(contents).hexdigest())
                self.assertEqual(entry["mode"], format(member.mode, "04o"))

            self.assertEqual(set(members), expected_names)

    def test_build_supports_nested_payload_paths(self):
        builder = self._builder()
        source = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, source, ignore_errors=True)
        (source / "VERSION").write_text("1.2.3\n", encoding="ascii")
        nested = source / "attention_span" / "__init__.py"
        nested.parent.mkdir()
        nested.write_text('"""Package marker."""\n', encoding="utf-8")
        payload = (("VERSION", 0o644), ("attention_span/__init__.py", 0o644))

        with (
            mock.patch.object(builder, "ROOT", source),
            mock.patch.object(builder, "PAYLOAD", payload),
        ):
            result = builder.build_release(self.out)

        with self._archive(result.archive_path) as archive:
            names = {member.name for member in archive.getmembers()}
        self.assertIn(
            "attention-span-1.2.3/attention_span/__init__.py",
            names,
        )

    def test_build_rejects_unsafe_payload_paths(self):
        builder = self._builder()
        source = self._version_root("1.2.3")

        for unsafe in (
            ".",
            "../escape.py",
            "/absolute.py",
            "a//b.py",
            "./a.py",
            "a.py/",
            "bad\x00.py",
        ):
            with (
                self.subTest(unsafe=unsafe),
                mock.patch.object(builder, "ROOT", source),
                mock.patch.object(builder, "PAYLOAD", ((unsafe, 0o644),)),
                self.assertRaisesRegex(ValueError, "path"),
            ):
                builder.build_release(self.out)

    def test_build_rejects_duplicate_and_reserved_payload_paths(self):
        builder = self._builder()
        source = self._version_root("1.2.3")
        invalid_payloads = (
            (("VERSION", 0o644), ("VERSION", 0o644)),
            (("release-manifest.json", 0o644),),
        )

        for payload in invalid_payloads:
            with (
                self.subTest(payload=payload),
                mock.patch.object(builder, "ROOT", source),
                mock.patch.object(builder, "PAYLOAD", payload),
                self.assertRaisesRegex(ValueError, "reserved or duplicated"),
            ):
                builder.build_release(self.out)

    def test_build_rejects_portably_colliding_payload_paths(self):
        builder = self._builder()
        source = self._version_root("1.2.3")
        invalid_payloads = (
            (("statusline.py", 0o644), ("STATUSLINE.py", 0o644)),
            (
                ("caf\N{LATIN SMALL LETTER E WITH ACUTE}.py", 0o644),
                ("cafe\u0301.py", 0o644),
            ),
        )

        for payload in invalid_payloads:
            with (
                self.subTest(payload=payload),
                mock.patch.object(builder, "ROOT", source),
                mock.patch.object(builder, "PAYLOAD", payload),
                self.assertRaisesRegex(ValueError, "colliding"),
            ):
                builder.build_release(self.out)

    def test_build_rejects_symlinked_payload_sources(self):
        builder = self._builder()
        source = self._version_root("1.2.3")
        outside = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, outside, ignore_errors=True)
        (outside / "secret.py").write_text("secret\n", encoding="utf-8")
        (source / "linked").symlink_to(outside, target_is_directory=True)

        with (
            mock.patch.object(builder, "ROOT", source),
            mock.patch.object(builder, "PAYLOAD", (("linked/secret.py", 0o644),)),
            self.assertRaisesRegex(ValueError, "source"),
        ):
            builder.build_release(self.out)

    def test_bundle_can_install_in_an_isolated_claude_home(self):
        builder = self._builder()
        result = builder.build_release(self.out)
        extract_dir = tempfile.mkdtemp()
        claude_home = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, extract_dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, claude_home, ignore_errors=True)

        with self._archive(result.archive_path) as archive:
            if hasattr(tarfile, "data_filter"):
                archive.extractall(extract_dir, filter="data")
            else:
                archive.extractall(extract_dir)
        os.makedirs(os.path.join(claude_home, "skills"))
        release_root = os.path.join(extract_dir, "attention-span-" + result.version)
        env = dict(os.environ, CLAUDE_HOME=claude_home)
        proc = subprocess.run(
            ["bash", os.path.join(release_root, "install.sh")],
            cwd=release_root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(proc.returncode, 0, proc.stderr)
        installed_version = os.path.join(
            claude_home, "hooks", "attention-span", "VERSION"
        )
        with open(installed_version) as f:
            self.assertEqual(f.read().strip(), result.version)


class TestShippedFileListsAgree(unittest.TestCase):
    """The shipped-file list is restated by hand in three places.

    scripts/build_release.py PAYLOAD is the source of truth for the release tarball,
    attention_span/update.py REQUIRED_FILES gates the archives the updater accepts, and
    install.sh PAYLOAD_FILES is the local-install copy list. Adding a module to one and
    forgetting the others ships a release the updater rejects, or an install missing a
    file the release has, and nothing else notices.
    """

    def _bash_array(self, name):
        source = Path(ROOT, "install.sh").read_text(encoding="utf-8")
        body = source.partition(name + "=(")[2].partition(")")[0]
        return tuple(body.split())

    def test_installer_ships_the_release_payload_apart_from_install_sh(self):
        from attention_span import update
        from scripts import build_release

        release_paths = tuple(path for path, _mode in build_release.PAYLOAD)

        installed = self._bash_array("PAYLOAD_FILES")
        expected = set(release_paths) - {"install.sh"}

        self.assertIn("install.sh", release_paths)
        self.assertEqual(len(installed), len(set(installed)))
        self.assertEqual(set(installed), expected)
        self.assertEqual(update.REQUIRED_FILES, expected)

    def test_installer_marks_the_same_files_executable_as_the_release(self):
        from scripts import build_release

        executable = self._bash_array("EXECUTABLE_FILES")
        release_executables = {
            path for path, mode in build_release.PAYLOAD if mode & 0o111
        }

        self.assertEqual(set(executable), release_executables - {"install.sh"})


class TestWorkflowPolicy(unittest.TestCase):
    """Releases are cut MANUALLY; CI verifies the release path but never publishes.

    The old tag-push release workflow auto-published v1.0.0 the moment the release
    tag landed, racing the manual flow - and under immutable releases, deleting that
    release burned the tag name permanently. These tests hold the door shut: no
    workflow may trigger on tags, hold write permission, or touch gh release.
    """

    CHECKOUT_SHA = "11d5960a326750d5838078e36cf38b85af677262"
    SETUP_UV_SHA = "d4b2f3b6ecc6e67c4457f6d3e41ec42d3d0fcb86"

    def _workflows(self):
        directory = os.path.join(ROOT, ".github", "workflows")
        workflows = {}
        for name in sorted(os.listdir(directory)):
            if name.endswith((".yml", ".yaml")):
                with open(os.path.join(directory, name)) as f:
                    workflows[name] = f.read()
        # The policy assertions below iterate this dict, so an empty directory would
        # pass every one of them vacuously while CI silently stopped existing.
        self.assertTrue(workflows)
        return workflows

    def test_regular_ci_builds_the_release_bundle(self):
        workflow = self._workflows()["test.yml"]

        self.assertIn(
            "PYTHONPATH= PYTHONSAFEPATH= uv run --no-project "
            "python -m scripts.build_release --outdir dist",
            workflow,
        )
        self.assertNotIn("python " + "scripts/build_release.py", workflow)

    def test_no_workflow_publishes_a_release(self):
        for name, workflow in self._workflows().items():
            with self.subTest(name=name):
                self.assertNotIn("tags:", workflow)  # no tag-push trigger
                self.assertNotIn("gh release", workflow)

    def test_no_workflow_can_write_repository_contents(self):
        for name, workflow in self._workflows().items():
            with self.subTest(name=name):
                self.assertIn("contents: read", workflow)
                self.assertNotIn("contents: write", workflow)

    def test_workflows_pin_reviewed_third_party_actions(self):
        for name, workflow in self._workflows().items():
            with self.subTest(name=name):
                self.assertIn("actions/checkout@" + self.CHECKOUT_SHA, workflow)
                self.assertIn("astral-sh/setup-uv@" + self.SETUP_UV_SHA, workflow)
                self.assertNotIn("actions/checkout@v4", workflow)
                self.assertNotIn("astral-sh/setup-uv@v5", workflow)

    def test_workflow_checkouts_do_not_persist_credentials(self):
        for name, workflow in self._workflows().items():
            with self.subTest(name=name):
                self.assertIn("persist-credentials: false", workflow)


if __name__ == "__main__":
    unittest.main(verbosity=2)
