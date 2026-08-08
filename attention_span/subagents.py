"""Subagent (child-transcript) discovery + per-agent health rollup."""

import contextlib
import glob
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from os import PathLike
from typing import Any, cast

from attention_span import agent_health, session_ui

MAX_AGENTS = 16
COHORT_PARSE_BUDGET = MAX_AGENTS
_state_dir = session_ui.state_dir
_session_key = cast(Callable[[Any], str], session_ui.session_key)
_ensure_private_directory = cast(
    Callable[[str], bool], session_ui.ensure_private_directory
)

AGENTS_PREFIX = os.path.join(_state_dir(), "agents-")
_STATES = {"green", "yellow", "red"}


@dataclass(frozen=True)
class Subagent:
    """One subagent run as the cohort knows it: identity, verdict, and token burn."""

    agent_id: str = ""
    agent_type: str = ""
    spawn_depth: int | None = None
    state: str = "green"
    chip: str = ""
    context_tokens: int = 0
    insufficient: bool = False
    done: bool = False
    mtime: float = 0.0
    burn: int = 0
    burn_trusted: bool = False
    model: str = ""


@dataclass(frozen=True)
class Cohort:
    """A session's subagents taken together - live and done - plus their token burn.

    Gathered facts only: every count a reader wants is derived HERE, once, so no
    consumer re-derives one of its own. ``live`` carries the expensive per-child health
    detail for at most ``cap`` working children, which is why it is never the count -
    ``live_n`` is the uncapped inventory. ``models`` and ``blind_loop_n`` describe the
    UNFINISHED children and read the whole memo: ``cap`` bounds displayed DETAIL, never
    the truth a warning is drawn from, since an old child in an edit loop still has to
    be actionable. A child with no assistant turn yet claims no model.
    """

    live: tuple[Subagent, ...] = ()
    total_n: int = 0
    done_n: int = 0
    tokens_total: int = 0
    tokens_known_n: int = 0
    tokens_untrusted_n: int = 0
    models: tuple[str, ...] = ()
    blind_loop_n: int = 0

    @property
    def live_n(self) -> int:
        """Children not yet proven finished - completion needs transcript evidence."""
        return self.total_n - self.done_n

    @property
    def tokens_warming(self) -> bool:
        """True while a child's burn is still unparsed, so the total is a lower bound."""
        return self.tokens_known_n < self.total_n

    @property
    def tokens_complete(self) -> bool:
        """True when every child's burn is both parsed and trusted."""
        return not self.tokens_warming and self.tokens_untrusted_n == 0


def subagents_dir(parent_path: str | PathLike[str]) -> str:
    """The sibling ``subagents/`` dir for a driver transcript path (existence not checked)."""
    session = os.path.splitext(os.path.basename(parent_path))[0]
    return os.path.join(os.path.dirname(parent_path), session, "subagents")


def discover(parent_path: str | PathLike[str]) -> list[str]:
    """Child transcript paths for a session, or ``[]`` when none."""
    d = subagents_dir(parent_path)
    if not os.path.isdir(d):
        return []
    return glob.glob(os.path.join(d, "**", "agent-*.jsonl"), recursive=True)


def read_meta(agent_path: str) -> dict[str, Any]:
    """Load the ``.meta.json`` sidecar for a child transcript; ``{}`` on any failure."""
    meta_path = agent_path[: -len(".jsonl")] + ".meta.json"
    try:
        with open(meta_path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _agent_id(path: str) -> str:
    """The agent id from a child path: ``.../agent-<id>.jsonl`` -> ``<id>``."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem[len("agent-") :] if stem.startswith("agent-") else stem


def _identity(path: str) -> dict[str, Any]:
    """The identity fields every ``Subagent`` carries: id plus its sidecar metadata.

    The sidecar is external JSON, so each field is typed HERE rather than trusted: an
    off-schema value becomes the not-known default instead of landing in a field it
    contradicts. bool is excluded from spawn_depth because it is an int subclass.
    """
    meta = read_meta(path)
    agent_type = meta.get("agentType")
    spawn_depth = meta.get("spawnDepth")
    return {
        "agent_id": _agent_id(path),
        "agent_type": agent_type if isinstance(agent_type, str) else "",
        "spawn_depth": (
            spawn_depth
            if isinstance(spawn_depth, int) and not isinstance(spawn_depth, bool)
            else None
        ),
    }


def _analyze_child(
    path: str, mtime: float, th: Mapping[str, Any] | None = None
) -> Subagent:
    """Parse ONE child transcript into its ``Subagent`` record (may raise)."""
    analysis = agent_health.analyze_transcript(path, th=th, include_sidechain=True)
    identity = _identity(path)
    if analysis.parse_health.degraded:
        return Subagent(**identity, insufficient=True, mtime=mtime)
    bl = agent_health.blind_loop_alert(analysis)
    ut = analysis.usage_totals
    return Subagent(
        **identity,
        state="red" if bl else "green",
        chip=str(bl["chip"]) if bl else "",
        context_tokens=analysis.context_tokens,
        insufficient=analysis.insufficient,
        done=analysis.last_stop_reason == "end_turn",
        mtime=mtime,
        # Burn EXCLUDES cache_read: a re-read of the cached prefix is not new tokens.
        burn=ut.input + ut.cache_creation + ut.output,
        burn_trusted=True,
        model=analysis.last_model or "",
    )


def _read_memo(memo_path: str) -> dict[str, dict[str, Any]]:
    """The per-session cohort memo, or ``{}`` on any failure (corrupt -> full rebuild)."""
    try:
        with open(memo_path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return {}
        return {k: v for k, v in data.items() if isinstance(v, dict)}
    except Exception:
        return {}


def _atomic_write_json(path: str, value: Mapping[str, Any]) -> None:
    """Replace ``path`` with ``value`` as JSON in one step; silent on any failure.

    A render may read the memo while another writes it, so a truncating write would
    expose a half-written file. A deliberate twin of ``session_ui``'s writer, kept
    separate so neither module owns the other's state format.
    """
    tmp = None
    try:
        directory = os.path.dirname(path) or "."
        if not _ensure_private_directory(directory):
            return
        fd, tmp = tempfile.mkstemp(prefix=".cc-agents-", dir=directory, text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(value, fh)
        os.replace(tmp, path)
    except Exception:
        if tmp:
            with contextlib.suppress(OSError):
                os.unlink(tmp)


def _nonneg_int(v: Any) -> int | None:
    """A JSON number -> non-negative int, or None when missing/off-schema."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    try:
        finite = math.isfinite(float(v))
    except (TypeError, ValueError, OverflowError):
        return None
    if not finite or v < 0:
        return None
    return int(v)


def _memo_hit(ent: Any, mtime: float, size: int) -> dict[str, Any] | None:
    """Return a normalized memo entry when the stored schema is safe to trust."""
    if (
        not isinstance(ent, dict)
        or ent.get("mtime") != mtime
        or ent.get("size") != size
    ):
        return None
    burn = _nonneg_int(ent.get("burn"))
    context_tokens = _nonneg_int(ent.get("context_tokens"))
    state = ent.get("state")
    if burn is None or context_tokens is None or state not in _STATES:
        return None
    chip = ent.get("chip")
    insufficient = ent.get("insufficient")
    done = ent.get("done")
    burn_trusted = ent.get("burn_trusted")
    model = ent.get("model", "")
    if (
        not isinstance(chip, str)
        or not isinstance(insufficient, bool)
        or not isinstance(done, bool)
        or not isinstance(burn_trusted, bool)
        or not isinstance(model, str)
    ):
        return None
    return {
        "mtime": mtime,
        "size": size,
        "burn": burn,
        "context_tokens": context_tokens,
        "blind_loop": state == "red",
        "state": state,
        "insufficient": insufficient,
        "chip": chip,
        "done": done,
        "burn_trusted": burn_trusted,
        "model": model,
    }


def _entry_from_agent(agent: Subagent, mtime: float, size: int) -> dict[str, Any]:
    return {
        "mtime": mtime,
        "size": size,
        "burn": agent.burn,
        "burn_trusted": agent.burn_trusted,
        "context_tokens": agent.context_tokens,
        "blind_loop": agent.state == "red",
        "state": agent.state,
        "insufficient": agent.insufficient,
        "chip": agent.chip,
        "done": agent.done,
        "model": agent.model,
    }


def _agent_from_entry(path: str, mtime: float, entry: Mapping[str, Any]) -> Subagent:
    return Subagent(
        **_identity(path),
        state=entry["state"],
        chip=entry["chip"],
        context_tokens=entry["context_tokens"],
        insufficient=entry["insufficient"],
        done=entry["done"],
        mtime=mtime,
        burn=entry["burn"],
        burn_trusted=entry["burn_trusted"],
        model=entry["model"],
    )


def _is_done(
    entry: Mapping[str, Any], path: str, notified: Mapping[str, float]
) -> bool:
    """Whether a child has STOPPED: its own transcript proved ``end_turn``, or the
    parent recorded a task-notification for it no earlier than its last write.

    The ONE effective-done predicate, so no two counts can disagree. The memo's
    ``done`` keeps meaning transcript-SELF-evidence only; the notification is applied
    HERE, per render, which is what lets a resumed child - one whose transcript grew
    past its notification - go back to working with no permanent bit to unset.
    """
    if entry["done"]:
        return True
    return bool(notified.get(_agent_id(path), -math.inf) >= entry["mtime"])


def cohort(
    parent_path: str | PathLike[str],
    session_id: Any,
    *,
    cap: int = MAX_AGENTS,
    th: Mapping[str, Any] | None = None,
    notified: Mapping[str, float] | None = None,
) -> Cohort:
    """The PERSISTENT subagent ``Cohort``: live children, done count, cumulative burn.

    Completion is derived from transcript state, never from a transcript's age.
    ``notified`` is the parent's task-id -> latest-stop-time ledger, the only evidence
    for a child torn down before its own transcript could record ``end_turn``. ``cap``
    bounds only the per-child detail in ``live``; every count reads the whole memo.
    """
    stops: Mapping[str, float] = notified or {}
    try:
        children = discover(parent_path)
    except Exception:
        children = []
    memo_path = AGENTS_PREFIX + _session_key(session_id) + ".json"
    memo = _read_memo(memo_path)

    stats: list[tuple[float, int, str]] = []
    for path in children:
        try:
            st = os.stat(path)
        except OSError:
            continue
        stats.append((st.st_mtime, st.st_size, path))
    stats.sort(reverse=True)  # newest first
    total_n = len(stats)
    new_memo: dict[str, dict[str, Any]] = {}
    live: list[Subagent] = []
    done_paths: set[str] = set()
    tokens_total = 0
    parses_left = COHORT_PARSE_BUDGET
    for mtime, size, path in stats:
        agent = None
        entry = _memo_hit(memo.get(path), mtime, size)
        if entry is None:
            if parses_left <= 0:
                continue  # out of budget: unknown, so never done
            parses_left -= 1
            try:
                agent = _analyze_child(path, mtime, th=th)
            except Exception:
                continue  # a deleted or half-written child is likewise unknown
            entry = _entry_from_agent(agent, mtime, size)
        new_memo[path] = entry
        tokens_total += entry["burn"]
        if _is_done(entry, path, stops):
            done_paths.add(path)
            continue
        if len(live) >= cap:
            continue
        if agent is None:
            agent = _agent_from_entry(path, mtime, entry)
        live.append(agent)

    if new_memo != memo:  # zero writes on a full cache hit
        _atomic_write_json(memo_path, new_memo)
    return Cohort(
        live=tuple(live),
        total_n=total_n,
        done_n=len(done_paths),
        tokens_total=tokens_total,
        tokens_known_n=len(new_memo),
        tokens_untrusted_n=sum(
            1 for entry in new_memo.values() if not entry["burn_trusted"]
        ),
        models=tuple(
            sorted(
                {
                    entry["model"]
                    for path, entry in new_memo.items()
                    if entry["model"] and path not in done_paths
                }
            )
        ),
        blind_loop_n=sum(
            1
            for path, entry in new_memo.items()
            if entry["blind_loop"] and path not in done_paths
        ),
    )
