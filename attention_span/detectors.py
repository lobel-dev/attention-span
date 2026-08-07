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

    Single home for the repo's load-bearing rule — a hook/gate-denied call never
    executed, so it must advance neither the R2E window NOR the blind-loop detector.
    Both consumers iterate this view so the exclusion can't drift between them.
    PENDING events are kept (a trailing in-flight tool use still counts), matching
    analyze_transcript.
    """
    return [ev for ev in ordered if ev.get("result_state") != "HOOK_DENY"]


def _failed_edit_loop(ordered: Iterable[Event]) -> EditLoop:
    """Derive the ``EditLoop`` from executed/PENDING events (HOOK_DENY excluded via
    _executed_events)."""
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
    """Cache-thrash health (#1) from the per-turn cache series accrued in `_consume`.

    The April-2026 incident's signature was a SUSTAINED cache-hit collapse with
    continuous creation churn ("continuous cache misses, draining usage limits faster
    than expected"). A warm long session keeps a high cache-read fraction and creates
    little new cache; the bug drove the hit fraction down and re-created every turn.

    THREE legitimate confounds are suppressed BEFORE scoring (precision over recall —
    prefer missing thrash to a false anomaly on healthy bursty work):
      * warmup — the first CACHE_WARM_TURNS turns legitimately create cache (skipped),
      * idle   — a turn taken > CACHE_IDLE_TTL_S after the previous one legitimately
                 missed an expired cache (excluded; uses the per-turn timestamp delta,
                 clamped >= 0 so out-of-order recorded steps never read as idle),
      * growth — a turn whose context jumped (a big Read/paste) legitimately creates
                 cache, so its creation is NOT counted as churn (growth-normalized).

    Returns ``{hit_drop, churn_mult, window_turns, suppressed, show, thrash_score}``:
      hit_drop     — pct-point drop of the recent windowed hit fraction below the
                     session's warm baseline (clamped >= 0),
      churn_mult   — growth-unexplained creation tokens as a multiple of the window's
                     cache-read total (~0 healthy, climbs past 1 under real thrash),
                     saturating at CHURN_MULT_MAX, which is also what a window with no
                     cache reads at all (total collapse) reports,
      thrash_score — the monotonic worse-is-higher calibration column (hit_drop scaled
                     by sustained churn),
      show         — the UNCALIBRATED display gate: a sustained drop past
                     CACHE_HIT_DROP_SHOW *and* real churn (dim chip only; a yellow/red
                     alarm is gated through calibration, not shipped from here).
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
    """Read-repetition score (#3) — a SEPARATE detector from the precise blind-loop.

    Telemetry only (never displayed; a shadow `repetition_score` calibration column).
    Iterates `_executed_events` so HOOK_DENY (gate-denied, never-ran) reads are
    excluded — the same cardinal-rule exclusion the blind-loop uses, so a healthy
    agent complying with a gate is never scored as forgetful.

    Signal: the same file Read repeatedly within REPEAT_WINDOW executed events with NO
    intervening edit AT ALL (the April "forgetful and repetitive" symptom). ANY edit is
    PROGRESS — it resets every file's re-read run, so re-consulting a reference/spec file
    between edits to OTHER files is not flagged; and the first read after editing a given
    file is a productive verification (the read-after-edit carve-out), not forgetfulness,
    so it starts a fresh run. Scored on TOOL reads only: Grep/Glob carry no file_path at
    all, and shell reads are skipped here (see the guard below) because the region
    discrimination they would need is deferred; legitimate sequential PAGINATION of one
    huge file is a known residual false positive of the shadow column (documented in
    CALIBRATION.md).

    Returns a ``Repetition``; ``score`` (== the worst no-progress re-read run length)
    is the monotonic column. A run >= REPEAT_MIN is "firing".
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

    The two conditions that make a repeat evidence of a STALL rather than of work.
    A different result body means the world moved, so the repeat re-checked a changed
    fact and starts a fresh run. A gap past the idle TTL means the agent was WAITING
    (a deliberate `sleep 30 && check` poll), not looping. That gap needs a VERIFIABLE
    delta, so a missing timestamp on either side - or a negative, out-of-order one -
    fails CLOSED and refuses to extend, the same discipline `_cache_health` applies
    to its own idle suppression.
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
    """Perseveration score (#4) - one tool call re-run with nothing moving between.

    Telemetry only (never displayed; a shadow `perseveration_score` calibration
    column). It closes the gap the other detectors leave open: the blind loop covers
    only FAILED edits and #3 covers only file Reads, so a model re-running one failing
    `npm test` five times trips nothing. Scored over the reducer's CALL log, which
    holds neutral calls too - `ordered` cannot, because a neutral event there would
    corrupt the R2E window's occupancy.

    A repeat extends a run only when all of these hold:
      * same signature AND same result body, inside a verifiable non-idle gap
        (both in `_persev_run_extends`),
      * within the trailing PERSEV_WINDOW executed calls,
      * with no executed EDIT since - ANY edit is progress and resets every run, the
        same rule `_repetition` applies, so a healthy `test, edit, test` TDD loop
        never accumulates.

    Edit-class calls reset but are never THEMSELVES tracked: an identical failed edit
    repeated is the blind loop's territory, and scoring it here would report one defect
    twice. HOOK_DENY calls are excluded through `_executed_events` (the cardinal rule:
    they never ran), and a PENDING call is skipped outright - an unsettled result can
    neither match nor differ, so it may extend nothing and reset nothing.

    Returns a ``Perseveration``; ``score`` (== the worst run length) is the monotonic
    column. A run >= PERSEV_MIN is "firing".
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
