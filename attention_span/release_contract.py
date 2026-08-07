"""Shared version identity for release building and automatic updates."""

import re
import unicodedata
from pathlib import PurePosixPath

NAME = "attention-span"
GITHUB_REPOSITORY = "lobel-dev/attention-span"
MANIFEST_SCHEMA_VERSION = 1
SEMVER = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


def parse_version(value: object) -> tuple[int, int, int]:
    if not isinstance(value, str) or not SEMVER.fullmatch(value):
        raise ValueError("version must be strict ASCII SemVer X.Y.Z")
    major, minor, patch = value.split(".")
    return int(major), int(minor), int(patch)


def tag_for(version: str) -> str:
    # Bare on purpose. The conventional v-prefixed names v1.0.0 and v1.0.1 were
    # burned before launch: an accidental CI publish under immutable releases
    # tombstones a tag name for the owner/name namespace permanently - surviving
    # release deletion, setting changes, and even repository re-creation. The
    # bare spelling is the only way release 1.0.0 can exist at all.
    parse_version(version)
    return version


def validate_payload_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("manifest path is invalid")
    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or str(path) != value
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise ValueError("manifest path is unsafe")
    return value


def payload_path_key(value: object) -> str:
    return unicodedata.normalize("NFC", validate_payload_path(value)).casefold()
