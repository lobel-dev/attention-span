"""Streaming primitives for Claude Code JSONL transcripts."""

import json
from collections.abc import Iterator
from os import PathLike
from typing import Any


def iter_lines(path: str | PathLike[str]) -> Iterator[tuple[int, Any | None]]:
    """The ONE place the JSONL transcript format is read.

    Yields ``(line_no, obj)`` for every line, 1-indexed, with ``obj`` None for an
    undecodable one, so a caller can COUNT decode failures - a live transcript's
    trailing line is often half-written. Sidechain turns are yielded as-is, and
    ``OSError`` from ``open()`` propagates: both are the caller's policy, not this
    reader's.

    ``errors="replace"`` is deliberate. Strict decoding raises INSIDE the line
    iterator, over a whole read buffer at once: that escapes the json-only ``try``
    here and, for a bad byte in the first buffer, aborts before any line is yielded -
    dropping valid leading lines with no ``line_no`` to report the abort against.
    Replacing keeps decoding, so structural corruption fails ``json.loads`` and is
    counted like any other undecodable line.
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, 1):
            try:
                obj = json.loads(line)
            except Exception:
                obj = None
            yield line_no, obj


def iter_objects(path: str | PathLike[str]) -> Iterator[tuple[int, Any]]:
    """``iter_lines`` with undecodable lines filtered out: the consumer never sees None."""
    for line_no, obj in iter_lines(path):
        if obj is not None:
            yield line_no, obj
