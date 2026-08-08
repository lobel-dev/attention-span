"""Pure detectors derived from ordered transcript events and cache turns."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, TypedDict, cast

from attention_span.analysis import EditLoop, Perseveration, Repetition

Event = Mapping[str, Any]
CacheTurn = Mapping[str, Any]
Call = Mapping[str, Any]

CHURN_MULT_MAX = 10.0


class _FileLoopState(TypedDict):
    pending_fail: bool
    loop: int


class _PersevRun(TypedDict):
    result_hash: Any
    positions: list[int]
    last_ts: float | None


class _JudgedCacheTurn(TypedDict):
    t: CacheTurn
    hit: float
    grew: bool
    ts_ok: bool


def _executed_events(ordered: Iterable[Event]) -> list[Event]:
    """The read/edit events that actually ran: all except HOOK_DENY ones.

    The single home for a load-bearing rule - a gate-denied call never executed, so it
    may advance nothing - which is what keeps its consumers from drifting apart on it.
    PENDING events are kept: a trailing in-flight tool use still counts.
    """
    return [ev for ev in ordered if ev.get("result_state") != "HOOK_DENY"]


def _failed_edit_loop(ordered: Iterable[Event]) -> EditLoop:
    """The worst same-file ``EditLoop`` over the executed events."""
    file_state: dict[str, _FileLoopState] = {}

    def fstate(f: str) -> _FileLoopState:
        return file_state.setdefault(f, {"pending_fail": False, "loop": 0})

    for ev in _executed_events(ordered):
        f = ev.get("file_path")
        if ev.get("class") == "read":
            if f:
                st = fstate(f)
                st["pending_fail"] = False
                st["loop"] = 0
            continue
        if ev.get("class") != "edit" or not f:
            continue
        st = fstate(f)
        if st["pending_fail"]:
            st["loop"] += 1
        if ev.get("result_state") == "GENUINE":
            st["pending_fail"] = True
        elif ev.get("result_state") == "OK":
            st["pending_fail"] = False
            st["loop"] = 0

    floop = EditLoop()
    for f, st in file_state.items():
        if st["loop"] > floop.count:
            floop = EditLoop(file=f, count=st["loop"])
    return floop


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Nearest-rank q-quantile (0..1) of an already-sorted non-empty list."""
    if not sorted_vals:
        return 0.0
    i = int(round(q * (len(sorted_vals) - 1)))
    return sorted_vals[max(0, min(len(sorted_vals) - 1, i))]


def _cache_health(
    cache_turns: Sequence[CacheTurn], thresholds: Mapping[str, Any]
) -> dict[str, float | int | bool]:
    """Cache-thrash health from the per-turn cache series the reducer accrues.

    Targets a SUSTAINED collapse of the cache-hit fraction under continuous creation
    churn; a warm session keeps a high read fraction and creates little new cache.
    Three legitimate confounds are suppressed BEFORE scoring - precision over recall,
    a false anomaly on healthy bursty work being worse than a missed one: warmup turns;
    turns after an idle gap, where an expired cache is a legitimate miss (the gap is
    clamped >= 0, so out-of-order steps never read as idle); and turns whose context
    jumped, whose creation is growth rather than churn.

    Returns ``{hit_drop, churn_mult, window_turns, suppressed, show, thrash_score}``:
      hit_drop     - percentage points the recent hit fraction sits below the session's
                     warm baseline, clamped >= 0,
      churn_mult   - growth-unexplained creation as a multiple of the window's cache
                     reads, saturating at ``CHURN_MULT_MAX`` - which is also what a
                     window with no cache reads at all reports,
      thrash_score - monotonic worse-is-higher combination of the two,
      show         - the display gate: a sustained drop with real churn, on
                     timestamp-verified turns.
    """
    warm = thresholds["CACHE_WARM_TURNS"]
    window = thresholds["CACHE_WINDOW"]
    idle_ttl = thresholds["CACHE_IDLE_TTL_S"]
    show_thr = thresholds["CACHE_HIT_DROP_SHOW"]

    suppressed = {
        "hit_drop": 0.0,
        "churn_mult": 0.0,
        "window_turns": 0,
        "suppressed": True,
        "show": False,
        "thrash_score": 0.0,
    }

    def _hit(t: CacheTurn) -> float | None:
        denom = t["cr"] + t["cc"] + t["inp"]
        return (t["cr"] / denom) if denom > 0 else None

    judged: list[_JudgedCacheTurn] = []
    prev_ts: float | int | None = None
    prev_ctx = None
    for i, t in enumerate(cache_turns):
        ts = cast(float | int | None, t.get("ts_epoch"))
        if prev_ts is not None and ts is not None:
            ts_ok = True  # a real prev->cur delta exists
            idle = (ts - prev_ts) > idle_ttl  # negative out-of-order delta < ttl
        else:
            ts_ok = False
            idle = False
        grew = prev_ctx is not None and t["ctx"] > prev_ctx
        hit = _hit(t)
        if i >= warm and not idle and hit is not None:
            judged.append({"t": t, "hit": hit, "grew": grew, "ts_ok": ts_ok})
        if ts is not None:
            prev_ts = ts
        prev_ctx = t["ctx"]

    if len(judged) < 3:  # too little post-warmup signal to trust a verdict -> suppress
        return dict(suppressed)

    scored = judged[-window:]
    baseline_hit = _percentile(sorted(p["hit"] for p in judged), 0.9)  # warm high-water

    sum_cr = sum(p["t"]["cr"] for p in scored)
    sum_cc = sum(p["t"]["cc"] for p in scored)
    sum_inp = sum(p["t"]["inp"] for p in scored)
    denom = sum_cr + sum_cc + sum_inp
    recent_hit = (sum_cr / denom) if denom else 0.0

    hit_drop = max(0.0, (baseline_hit - recent_hit) * 100.0)
    churn_unexplained = sum(p["t"]["cc"] for p in scored if not p["grew"])
    if sum_cr > 0:
        churn_mult = min(churn_unexplained / sum_cr, CHURN_MULT_MAX)
    elif churn_unexplained > 0:
        churn_mult = CHURN_MULT_MAX
    else:
        churn_mult = 0.0
    thrash_score = hit_drop * (1.0 + churn_mult)

    # Idle suppression is the ONLY guard against a legit cache-expiry reading as thrash
    ts_verified = all(p["ts_ok"] for p in scored)
    show = hit_drop >= show_thr and churn_mult >= 1.0 and ts_verified

    return {
        "hit_drop": round(hit_drop, 1),
        "churn_mult": round(churn_mult, 3),
        "window_turns": len(scored),
        "suppressed": False,
        "show": bool(show),
        "thrash_score": round(thrash_score, 2),
    }


def _repetition(ordered: Iterable[Event], thresholds: Mapping[str, Any]) -> Repetition:
    """Read-repetition: one file re-read inside the window with no edit in between.

    Diagnostic only, and SEPARATE from the blind loop, which sees only failed edits.
    Reads ``_executed_events``, so an agent complying with a gate is never scored as
    forgetful.

    ANY edit is PROGRESS and resets every file's run, so re-consulting a reference file
    between edits elsewhere is not flagged; the first read after editing a given file
    is verification, not forgetfulness, and starts a fresh run. TOOL reads only - shell
    reads are skipped because distinguishing a re-read from a new region of the same
    file needs discrimination this does not have. Sequential PAGINATION of one huge
    file is a known residual false positive.

    ``score`` is the worst no-progress re-read run length: monotonic.
    """
    window = thresholds["REPEAT_WINDOW"]
    # file -> positions of its consecutive no-progress reads still inside the window
    runs: dict[str, list[int]] = {}
    # files whose latest event was an edit (the read-after-edit carve-out)
    just_edited: set[str] = set()
    worst_count = 0
    worst_file = None
    pos = 0
    for ev in _executed_events(ordered):
        pos += 1
        f = ev.get("file_path")
        cls = ev.get("class")
        if cls == "edit":
            runs.clear()  # ANY edit is progress: reset every no-progress read run
            if f:
                just_edited.add(f)
            continue
        if cls != "read" or not f:
            continue
        if ev.get("tool_name") == "Bash":
            continue
        if f in just_edited:  # read-after-edit = verification -> fresh baseline run
            just_edited.discard(f)
            runs[f] = [pos]
            continue
        run = [p for p in runs.get(f, []) if p > pos - window]  # keep in-window reads
        run.append(pos)
        runs[f] = run
        if len(run) > worst_count:
            worst_count = len(run)
            worst_file = f
    return Repetition(score=worst_count, worst_target=worst_file, count=worst_count)


def _persev_run_extends(
    run: _PersevRun, call: Call, ts: float | None, idle_ttl: float
) -> bool:
    """True if ``call`` continues ``run``: same answer, no idle gap in between.

    The two conditions that make a repeat evidence of a STALL rather than of work. A
    different result body means the world moved, so the repeat re-checked a changed
    fact. A gap past the idle TTL means the agent was WAITING - a deliberate poll - not
    looping. That gap needs a VERIFIABLE delta, so a missing or out-of-order timestamp
    fails CLOSED and refuses to extend, as ``_cache_health`` does for its own idle
    suppression.
    """
    if run["result_hash"] != call["result_hash"]:
        return False
    last_ts = run["last_ts"]
    if last_ts is None or ts is None:
        return False
    return 0 <= ts - last_ts <= idle_ttl


def _perseveration(
    calls: Iterable[Call], thresholds: Mapping[str, Any]
) -> Perseveration:
    """Perseveration: one tool call re-run with nothing moving between the runs.

    Diagnostic only. It covers what the others cannot - the blind loop sees only failed
    edits and ``_repetition`` only file reads, so a model re-running one failing test
    command trips neither. Scored over the reducer's CALL log, which holds neutral
    calls; ``ordered`` cannot, since a neutral event there would corrupt the R2E
    window's occupancy.

    A repeat extends a run only when the signature, the result body and the timing all
    allow it (``_persev_run_extends``), it is still inside the trailing window, and no
    executed EDIT has intervened - ANY edit is progress, the rule ``_repetition`` also
    applies, so a healthy test-edit-test loop never accumulates.

    Edit-class calls reset runs but are never THEMSELVES tracked: a repeated failed
    edit is the blind loop's territory, and scoring it here would report one defect
    twice. A PENDING call is skipped outright - an unsettled result can neither match
    nor differ, so it may extend nothing and reset nothing.

    ``score`` is the worst run length: monotonic.
    """
    window = thresholds["PERSEV_WINDOW"]
    idle_ttl = thresholds["PERSEV_IDLE_TTL_S"]
    # call signature -> the run it is currently on, with the positions still in window
    runs: dict[Any, _PersevRun] = {}
    worst_count = 0
    worst_target = None
    pos = 0
    for call in _executed_events(calls):
        if call.get("result_state") == "PENDING":
            continue
        pos += 1
        if call.get("class") == "edit":
            runs.clear()
            continue
        signature = call["sig"]
        ts = cast(float | None, call["ts_epoch"])
        run = runs.get(signature)
        positions = [pos]
        if run is not None and _persev_run_extends(run, call, ts, idle_ttl):
            in_window = [p for p in run["positions"] if p > pos - window]
            if in_window:
                positions = [*in_window, pos]
        runs[signature] = {
            "result_hash": call["result_hash"],
            "positions": positions,
            "last_ts": ts,
        }
        if len(positions) > worst_count:
            worst_count = len(positions)
            worst_target = call["tool_name"]
    return Perseveration(
        score=worst_count, worst_target=worst_target, count=worst_count
    )
