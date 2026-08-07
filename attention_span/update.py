#!/usr/bin/env python3
"""Silent, integrity-checked updater for immutable attention-span releases."""

import argparse
import contextlib
import gzip
import hashlib
import io
import json
import os
import re
import secrets
import shutil
import tarfile
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple, cast
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from attention_span.release_contract import (
    GITHUB_REPOSITORY,
    MANIFEST_SCHEMA_VERSION,
    NAME,
    parse_version,
    payload_path_key,
    tag_for,
    validate_payload_path,
)

_parse_version = cast(Callable[[str], Any], parse_version)
_payload_path_key = cast(Callable[[str], str], payload_path_key)
_tag_for = cast(Callable[[str], str], tag_for)
_validate_payload_path = cast(Callable[[Any], str], validate_payload_path)

LATEST_URL = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/releases/latest"
DOWNLOAD_PATH_PREFIX = f"/{GITHUB_REPOSITORY}/releases/download/"
MAX_ARCHIVE_BYTES = 5 * 1024 * 1024
MAX_UNPACKED_BYTES = 10 * 1024 * 1024
MAX_TAR_BYTES = MAX_UNPACKED_BYTES + 1024 * 1024
MAX_MANIFEST_BYTES = 128 * 1024
MAX_FILES = 64
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
MODE_RE = re.compile(r"^0[0-7]{3}$")
REQUIRED_FILES = {
    "LICENSE",
    "VERSION",
    "attention_span/__init__.py",
    "attention_span/agent_health.py",
    "attention_span/analysis.py",
    "attention_span/detectors.py",
    "attention_span/events.py",
    "attention_span/health_config.py",
    "attention_span/reducer.py",
    "attention_span/release_contract.py",
    "attention_span/render.py",
    "attention_span/render_facts.py",
    "attention_span/session_ui.py",
    "attention_span/status_catalog.py",
    "attention_span/statusline.py",
    "attention_span/subagents.py",
    "attention_span/text.py",
    "attention_span/transcript.py",
    "attention_span/update.py",
    "attention_span/verdicts.py",
    "launcher.py",
    "statusline.py",
    "update.py",
}


class ReleaseAsset(NamedTuple):
    version: str
    url: str
    size: int
    digest: str


class PayloadFile(NamedTuple):
    path: str
    mode: int
    contents: bytes


def _read_url(url: str, limit: int) -> bytes:
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "attention-span-updater",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urlopen(request, timeout=15) as response:
        data = cast(bytes, response.read(limit + 1))
    if len(data) > limit:
        raise ValueError("response exceeds size limit")
    return data


def fetch_latest() -> dict[str, Any]:
    data = _read_url(LATEST_URL, MAX_MANIFEST_BYTES)
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("latest release response is not an object")
    return value


def select_update(metadata: Any, current_version: str) -> ReleaseAsset | None:
    current = _parse_version(current_version)
    if not isinstance(metadata, dict):
        raise ValueError("release metadata is not an object")
    if metadata.get("draft") is not False or metadata.get("prerelease") is not False:
        raise ValueError("latest release is not stable")
    if metadata.get("immutable") is not True:
        raise ValueError("latest release is not immutable")

    tag = metadata.get("tag_name")
    if not isinstance(tag, str):
        raise ValueError("latest release tag is invalid")
    # Tags are BARE semver (see release_contract.tag_for); the tag IS the version.
    version = tag
    parsed = _parse_version(version)
    if tag != _tag_for(version):
        raise ValueError("latest release tag is invalid")
    if parsed <= current:
        return None

    name = f"{NAME}-{version}.tar.gz"
    assets = metadata.get("assets")
    if not isinstance(assets, list):
        raise ValueError("release assets are missing")
    matches = [
        asset
        for asset in assets
        if isinstance(asset, dict) and asset.get("name") == name
    ]
    if len(matches) != 1:
        raise ValueError("release archive asset is not unique")
    asset = matches[0]
    if asset.get("state") != "uploaded":
        raise ValueError("release archive is not uploaded")
    size = asset.get("size")
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or not 0 < size <= MAX_ARCHIVE_BYTES
    ):
        raise ValueError("release archive size is invalid")
    digest_value = asset.get("digest")
    if not isinstance(digest_value, str) or not digest_value.startswith("sha256:"):
        raise ValueError("release archive digest is missing")
    digest = digest_value[7:]
    if not DIGEST_RE.fullmatch(digest):
        raise ValueError("release archive digest is invalid")
    url = asset.get("browser_download_url")
    parsed_url = urlparse(url) if isinstance(url, str) else None
    if (
        parsed_url is None
        or parsed_url.scheme != "https"
        or parsed_url.netloc != "github.com"
        or not parsed_url.path.startswith(DOWNLOAD_PATH_PREFIX)
    ):
        raise ValueError("release archive URL is invalid")
    return ReleaseAsset(version, cast(str, url), size, digest)


def download_asset(asset: ReleaseAsset) -> bytes:
    return _read_url(asset.url, min(MAX_ARCHIVE_BYTES, asset.size))


def _safe_relative_path(value: Any) -> str:
    return _validate_payload_path(value)


def _archive_payload(
    archive_bytes: bytes, expected_version: str
) -> tuple[PayloadFile, ...]:
    root_name = f"{NAME}-{expected_version}/"
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(archive_bytes), mode="rb") as compressed:
            tar_bytes = compressed.read(MAX_TAR_BYTES + 1)
        if len(tar_bytes) > MAX_TAR_BYTES:
            raise ValueError("release archive expands beyond size limit")
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as archive:
            return _payload_from_archive(archive, root_name, expected_version)
    except (gzip.BadGzipFile, tarfile.TarError, OSError, EOFError) as exc:
        raise ValueError("release archive cannot be read") from exc


class ManifestEntry(NamedTuple):
    """One validated ``files[]`` row of a release manifest.

    ``key`` is the path's filesystem-collision key, carried alongside so the caller
    tracking already-seen paths never recomputes the normalization.
    """

    path: str
    key: str
    size: int
    mode: int
    digest: str


def _archive_members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    """The archive's members by name, rejecting anything but unique regular files."""
    members = archive.getmembers()
    if not members or len(members) > MAX_FILES + 1:
        raise ValueError("release archive member count is invalid")
    if any(not member.isfile() for member in members):
        raise ValueError("release archive contains non-file members")
    by_name = {member.name: member for member in members}
    if len(by_name) != len(members):
        raise ValueError("release archive contains duplicate paths")
    return by_name


def _read_manifest(
    archive: tarfile.TarFile,
    manifest_member: tarfile.TarInfo | None,
    expected_version: str,
) -> list[Any]:
    """The manifest's ``files`` rows, once its identity matches what we asked for."""
    if manifest_member is None or manifest_member.size > MAX_MANIFEST_BYTES:
        raise ValueError("release manifest is missing or oversized")
    stream = archive.extractfile(manifest_member)
    if stream is None:
        raise ValueError("release manifest cannot be read")
    try:
        manifest = json.loads(stream.read().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("release manifest is invalid") from exc

    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("name") != NAME
        or manifest.get("version") != expected_version
    ):
        raise ValueError("release manifest identity is invalid")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries or len(entries) > MAX_FILES:
        raise ValueError("release manifest files are invalid")
    return entries


def _manifest_entry(
    entry: Any, seen_keys: frozenset[str], accounted_bytes: int
) -> ManifestEntry:
    """Validate one manifest row against the rows already accepted before it.

    ``seen_keys`` and ``accounted_bytes`` are the running path set and payload size, so
    a colliding path and an over-budget total are refused here with everything else.
    """
    if not isinstance(entry, dict):
        raise ValueError("release manifest entry is invalid")
    relative = _safe_relative_path(entry.get("path"))
    key = _payload_path_key(relative)
    if key in seen_keys:
        raise ValueError("release manifest contains colliding paths")
    size = entry.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("release manifest size is invalid")
    if accounted_bytes + size > MAX_UNPACKED_BYTES:
        raise ValueError("release payload is oversized")
    digest = entry.get("sha256")
    if not isinstance(digest, str) or not DIGEST_RE.fullmatch(digest):
        raise ValueError("release manifest digest is invalid")
    mode_value = entry.get("mode")
    if not isinstance(mode_value, str) or not MODE_RE.fullmatch(mode_value):
        raise ValueError("release manifest mode is invalid")
    mode = int(mode_value, 8)
    if mode & ~0o755:
        raise ValueError("release manifest mode is unsafe")
    return ManifestEntry(relative, key, size, mode, digest)


def _payload_file(
    archive: tarfile.TarFile, member: tarfile.TarInfo | None, entry: ManifestEntry
) -> PayloadFile:
    """Read one archive member, held to the size and digest the manifest promised."""
    if member is None or member.size != entry.size:
        raise ValueError("release payload size does not match manifest")
    stream = archive.extractfile(member)
    if stream is None:
        raise ValueError("release payload cannot be read")
    contents = stream.read()
    if hashlib.sha256(contents).hexdigest() != entry.digest:
        raise ValueError("release payload digest does not match manifest")
    return PayloadFile(entry.path, entry.mode, contents)


def _payload_from_archive(
    archive: tarfile.TarFile, root_name: str, expected_version: str
) -> tuple[PayloadFile, ...]:
    by_name = _archive_members(archive)
    manifest_name = root_name + "release-manifest.json"
    entries = _read_manifest(archive, by_name.get(manifest_name), expected_version)

    payload: tuple[PayloadFile, ...] = ()
    expected_names: frozenset[str] = frozenset({manifest_name})
    seen: frozenset[str] = frozenset()
    seen_keys: frozenset[str] = frozenset()
    total_size = 0
    for raw_entry in entries:
        entry = _manifest_entry(raw_entry, seen_keys, total_size)
        seen = seen | {entry.path}
        seen_keys = seen_keys | {entry.key}
        total_size += entry.size
        member_name = root_name + entry.path
        expected_names = expected_names | {member_name}
        payload = (*payload, _payload_file(archive, by_name.get(member_name), entry))

    if set(by_name) != expected_names or not REQUIRED_FILES.issubset(seen):
        raise ValueError("release payload does not match required files")
    contents_by_path = {item.path: item.contents for item in payload}
    if contents_by_path["VERSION"].decode("ascii").strip() != expected_version:
        raise ValueError("release VERSION does not match manifest")
    return payload


def _atomic_write(path: Path, contents: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + "-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(contents)
        os.chmod(temporary, mode)
        os.replace(temporary, str(path))
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def _switch_current(install_root: Path, release_dir: Path) -> None:
    temporary = install_root / (f".current-{os.getpid()}-{secrets.token_hex(4)}")
    try:
        os.symlink(os.path.relpath(str(release_dir), str(install_root)), str(temporary))
        os.replace(str(temporary), str(install_root / "current"))
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink()


def _release_matches(release_dir: Path, payload: tuple[PayloadFile, ...]) -> bool:
    expected = {item.path: item for item in payload}
    actual = set()
    try:
        for path in release_dir.rglob("*"):
            if path.is_symlink():
                return False
            if path.is_file():
                actual.add(path.relative_to(release_dir).as_posix())
            elif not path.is_dir():
                return False
        if actual != set(expected):
            return False
        for relative, item in expected.items():
            path = release_dir / relative
            if path.read_bytes() != item.contents:
                return False
            if path.stat().st_mode & 0o777 != item.mode:
                return False
        return True
    except OSError:
        return False


def _discard(path: Path) -> None:
    """Rename ``path`` out of the way, then delete it, best effort.

    Renaming first frees the deterministic release name immediately, so a removal
    that only half succeeds cannot leave anything at the path we are about to stage.
    """
    quarantine = path.parent / f".discarded-{path.name}-{secrets.token_hex(4)}"
    os.replace(str(path), str(quarantine))
    shutil.rmtree(str(quarantine), ignore_errors=True)
    with contextlib.suppress(OSError):
        quarantine.unlink()


def install_archive(
    install_root: str | Path,
    archive_bytes: bytes,
    expected_version: str,
    expected_digest: str,
    expected_size: int,
) -> str:
    install_root = Path(install_root).resolve()
    if len(archive_bytes) != expected_size:
        raise ValueError("release archive size does not match metadata")
    archive_digest = hashlib.sha256(archive_bytes).hexdigest()
    if archive_digest != expected_digest:
        raise ValueError("release archive digest does not match metadata")
    _parse_version(expected_version)
    payload = _archive_payload(archive_bytes, expected_version)

    releases = install_root / "releases"
    releases.mkdir(parents=True, exist_ok=True)
    release_dir = releases / (expected_version + "-" + archive_digest[:12])
    if not (release_dir.is_dir() and _release_matches(release_dir, payload)):
        # This path is derived from the version and digest alone, so a partial directory
        # left by a crashed install would otherwise fail every later run at the same
        # place. The updater is detached and silent, so that would block this version
        # forever. Discard whatever is there and re-stage from the verified payload.
        if os.path.lexists(str(release_dir)):
            _discard(release_dir)
        stage = Path(
            tempfile.mkdtemp(prefix="." + release_dir.name + "-", dir=str(releases))
        )
        try:
            for item in payload:
                destination = stage / item.path
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("wb") as stream:
                    stream.write(item.contents)
                destination.chmod(item.mode)
            if not _release_matches(stage, payload):
                raise ValueError("staged release failed verification")
            os.replace(str(stage), str(release_dir))
        finally:
            if stage.exists():
                shutil.rmtree(str(stage), ignore_errors=True)

    # launcher.py and VERSION are stable aliases installed once by install.sh.
    # Switching this one pointer activates the entire verified module set.
    # There are deliberately no fallible mutations after the activation boundary.
    _switch_current(install_root, release_dir)
    return expected_version


def current_version(install_root: str | Path) -> str:
    value = (
        ((Path(install_root) / "current").resolve(strict=True) / "VERSION")
        .read_text(encoding="ascii")
        .strip()
    )
    _parse_version(value)
    return value


def _write_state(
    install_root: str | Path,
    status: str,
    installed_version: str,
    error: str | None = None,
) -> None:
    value = {
        "schema_version": 1,
        "last_checked": int(time.time()),
        "status": status,
        "installed_version": installed_version,
        "error": error,
    }
    _atomic_write(
        Path(install_root) / "update-state.json",
        (json.dumps(value, sort_keys=True) + "\n").encode("utf-8"),
    )


def check_once(install_root: str | Path) -> str:
    install_root = Path(install_root)
    installed = current_version(install_root)
    asset = select_update(fetch_latest(), installed)
    if asset is None:
        _write_state(install_root, "up-to-date", installed)
        return "up-to-date"
    archive = download_asset(asset)
    installed = install_archive(
        install_root,
        archive,
        expected_version=asset.version,
        expected_digest=asset.digest,
        expected_size=asset.size,
    )
    _write_state(install_root, "updated", installed)
    return "updated"


def _safe_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return "".join(ch for ch in text if ch >= " " and ch != "\x7f")[:240]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-root", required=True)
    args = parser.parse_args(argv)
    root = Path(args.install_root)
    try:
        check_once(root)
    except Exception as exc:
        try:
            installed = current_version(root)
        except Exception:
            installed = "unknown"
        with contextlib.suppress(Exception):
            _write_state(root, "error", installed, _safe_error(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
