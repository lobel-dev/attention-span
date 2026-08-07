#!/usr/bin/env python3
"""Build the deterministic, manifest-checked release bundle consumed by update.py."""

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Any, NamedTuple, TypeAlias

from attention_span.release_contract import (
    MANIFEST_SCHEMA_VERSION,
    NAME,
    parse_version,
    payload_path_key,
    tag_for,
    validate_payload_path,
)

ROOT = Path(__file__).resolve().parent.parent

PayloadItem: TypeAlias = tuple[str, int]
Entry: TypeAlias = tuple[str, int, bytes]
ManifestFile: TypeAlias = dict[str, str | int]
ArchiveMember: TypeAlias = tuple[str, int, bytes]

# update.py rejects any archive whose members do not match this manifest exactly.
PAYLOAD = (
    ("LICENSE", 0o644),
    ("VERSION", 0o644),
    ("attention_span/__init__.py", 0o644),
    ("attention_span/agent_health.py", 0o644),
    ("attention_span/analysis.py", 0o644),
    ("attention_span/detectors.py", 0o644),
    ("attention_span/events.py", 0o644),
    ("attention_span/health_config.py", 0o644),
    ("attention_span/reducer.py", 0o644),
    ("attention_span/release_contract.py", 0o644),
    ("attention_span/render.py", 0o644),
    ("attention_span/render_facts.py", 0o644),
    ("attention_span/session_ui.py", 0o644),
    ("attention_span/status_catalog.py", 0o644),
    ("attention_span/statusline.py", 0o644),
    ("attention_span/subagents.py", 0o644),
    ("attention_span/text.py", 0o644),
    ("attention_span/transcript.py", 0o644),
    ("attention_span/update.py", 0o644),
    ("attention_span/verdicts.py", 0o644),
    ("install.sh", 0o755),
    ("launcher.py", 0o755),
    ("statusline.py", 0o755),
    ("update.py", 0o755),
)
MANIFEST_NAME = "release-manifest.json"


class Release(NamedTuple):
    version: str
    archive_path: str
    checksum_path: str


def read_version(root: str | PathLike[str] | None = None) -> str:
    directory = Path(root) if root is not None else ROOT
    version = (directory / "VERSION").read_text(encoding="utf-8").strip()
    try:
        parse_version(version)
    except ValueError as exc:
        raise ValueError(
            f"VERSION {version!r} is not strict SemVer X.Y.Z in ASCII digits"
        ) from exc
    return version


def validate_tag(tag: str, version: str) -> None:
    if tag != tag_for(version):
        raise ValueError(f"tag {tag} does not match VERSION {version}")


def _manifest(entries: Sequence[Entry], version: str) -> dict[str, Any]:
    files: list[ManifestFile] = [
        {
            "path": path,
            "size": len(contents),
            "sha256": hashlib.sha256(contents).hexdigest(),
            "mode": format(mode, "04o"),
        }
        for path, mode, contents in entries
    ]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "name": NAME,
        "version": version,
        "files": files,
    }


def _deterministic_tar_bytes(members: Sequence[ArchiveMember]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, mode, contents in members:
            info = tarfile.TarInfo(name)
            info.type = tarfile.REGTYPE
            info.size = len(contents)
            info.mode = mode
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            archive.addfile(info, io.BytesIO(contents))
    return buffer.getvalue()


def _payload_source(path: str) -> Path:
    root = ROOT.resolve(strict=True)
    candidate = ROOT
    for part in path.split("/"):
        candidate /= part
        if candidate.is_symlink():
            raise ValueError("release payload source cannot be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("release payload source cannot be read") from exc
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("release payload source must be a regular file under ROOT")
    return resolved


def build_release(outdir: str | PathLike[str]) -> Release:
    version = read_version()
    root = f"{NAME}-{version}/"
    payload: tuple[PayloadItem, ...] = tuple(
        (validate_payload_path(raw_path), mode) for raw_path, mode in PAYLOAD
    )
    payload_paths = tuple(path for path, _mode in payload)
    if MANIFEST_NAME in payload_paths or len(payload_paths) != len(set(payload_paths)):
        raise ValueError("release payload path is reserved or duplicated")
    payload_keys = tuple(payload_path_key(path) for path in payload_paths)
    if len(payload_keys) != len(set(payload_keys)):
        raise ValueError("release payload contains filesystem-colliding paths")
    entries = tuple(
        (path, mode, _payload_source(path).read_bytes())
        for path, mode in sorted(payload)
    )
    manifest = json.dumps(_manifest(entries, version), indent=2, sort_keys=True) + "\n"

    members: list[ArchiveMember] = [
        (root + path, mode, contents) for path, mode, contents in entries
    ]
    members.append((root + MANIFEST_NAME, 0o644, manifest.encode("utf-8")))
    members.sort(key=lambda member: member[0])

    compressed = io.BytesIO()
    with gzip.GzipFile(
        fileobj=compressed, mode="wb", compresslevel=9, mtime=0
    ) as stream:
        stream.write(_deterministic_tar_bytes(members))
    archive_bytes = compressed.getvalue()

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    archive_path = outdir / f"{NAME}-{version}.tar.gz"
    archive_path.write_bytes(archive_bytes)
    checksum_path = outdir / (archive_path.name + ".sha256")
    digest = hashlib.sha256(archive_bytes).hexdigest()
    checksum_path.write_text(f"{digest}  {archive_path.name}\n", encoding="ascii")
    return Release(version, str(archive_path), str(checksum_path))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=str(ROOT / "dist"))
    parser.add_argument("--tag", help="release tag that must match VERSION")
    args = parser.parse_args(argv)

    if args.tag is not None:
        validate_tag(args.tag, read_version())
    result = build_release(args.outdir)
    print(result.archive_path)
    print(result.checksum_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
