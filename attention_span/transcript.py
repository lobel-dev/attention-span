"""Streaming primitives for Claude Code JSONL transcripts."""

import json
from collections.abc import Iterator
from os import PathLike
from typing import Any


def iter_lines(path: str | PathLike[str]) -> Iterator[tuple[int, Any | None]]:
    """The ONE place the JSONL transcript format is read.

    Opens ``path`` (utf-8) and yields ``(line_no, obj)`` for every line,
    1-indexed, where ``obj`` is the decoded JSON object or ``None`` for an
    undecodable line — so diagnostic callers can *count* decode failures (a live
    transcript's trailing line is often half-written). Sidechain turns are yielded
    as-is; callers account for them, since the engine skips them silently while
    calibration counts them. Propagates ``OSError`` from ``open()`` so the caller
    decides whether to swallow (engine) or report (calibration).

    ``errors="replace"`` is deliberate: strict UTF-8 decoding raises *inside the line
    iterator* (a whole ~8KB read buffer at once), which would escape this generator's
    json-only ``try`` and, if the bad byte sits in the first buffer, abort before ANY
    line is yielded — silently dropping the valid leading lines AND leaving
    ``line_no`` unset so the engine can't mark the pass aborted. Replacing a bad byte
    with U+FFFD keeps decoding: structural corruption then fails ``json.loads`` and is
    counted as a normal ``decode_failure`` (the same path as a half-written tail);
    in-string corruption is absorbed and the line still parses. The fix lives in this
    ONE reader, so analyze_transcript and iter_states get it identically (drift-safe).
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, 1):
            try:
                obj = json.loads(line)
            except Exception:
                obj = None
            yield line_no, obj


def iter_objects(path: str | PathLike[str]) -> Iterator[tuple[int, Any]]:
    """Decoded transcript objects only — the engine's reader of choice.

    Same single walk as ``iter_lines`` with undecodable lines filtered out, so the
    consumer never sees ``None``.
    """
    for line_no, obj in iter_lines(path):
        if obj is not None:
            yield line_no, obj
