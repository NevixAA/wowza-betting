"""Adaptive in-run sampling for the NEAR odds-capture loops (P1 2026-10-22).

    from src.capture_schedule import PassScheduler, due_interval_min, plan_next

WHY. Certified CLV needs an observation inside the final 30 minutes before kickoff
(src/shadow_settlement.CLOSE_MAX_MIN). Measured over fixtures kicking off 2026-10-04..10-22:

    capture archives (Bet365, API-Football)   recorded inside T-30m   2.8%
                                              a capture RAN inside T-30m  30.1%
    v91_shadow certified closes               34 / 449 = 7.6%

Two separate defects, and this module handles the timing one (src/odds_heartbeat handles the
other — a pass that saw an UNCHANGED price wrote nothing, so a change-log could not prove it).

GitHub delivers ~8 NEAR runs/day per capture workflow, whatever the cron asks for (CLAUDE.md,
"GitHub Actions cannot hit a clock"). So the only timing lever is how long a delivered run stays
alive and where inside it the API calls go:

* KEEP-ALIVE. Past its base length a run keeps sampling while any kickoff it has seen lies within
  KEEPALIVE_MIN, up to CAP_MIN. Replayed against the measured run starts this lifts "a capture ran
  inside T-30m" from ~44% to ~76% (new-format) and ~34% to ~90% (standard) of fixtures.
* PER-FIXTURE DUE INTERVALS. A pass no longer re-prices every fixture inside 12h. Far fixtures are
  re-priced hourly, 1-6h every ~15 min, 30-60 min every ~5 min, and inside 30 min on EVERY pass.
  That is what pays for the longer runs: the API spend moves to where the close forms.
* PASS SPACING tightens to 2 minutes when a kickoff is inside 35 minutes.

The scheduler is OFF unless CAPTURE_STATE is set (manual single runs keep the old behaviour:
everything due, no state, no plan printed). It never fetches anything itself and costs no calls.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

#: (upper bound of minutes-to-kickoff, minimum minutes between two fetches of one fixture).
#: Intervals sit slightly under the nominal 5/15/60 so a pass that lands a few seconds early is
#: not skipped. Inside 30 minutes the interval is 0: every pass prices the fixture.
DUE_INTERVALS: tuple[tuple[float, float], ...] = (
    (30.0, 0.0),
    (60.0, 4.0),
    (360.0, 14.0),
    (math.inf, 55.0),
)

#: Pass spacing by nearest upcoming kickoff (minutes) -> seconds to sleep.
SLEEP_STEPS: tuple[tuple[float, int], ...] = ((35.0, 120), (70.0, 180), (math.inf, 300))

BASE_MIN = 40.0         # a run always samples at least this long (unchanged from before)
KEEPALIVE_MIN = 90.0    # ...and then stays alive while a kickoff lies within this many minutes
CAP_MIN = 170.0         # ...but never longer than this (workflow timeout is set above it)


def due_interval_min(minutes_to_ko: float | None) -> float:
    """Minimum minutes between two fetches of a fixture this far from kickoff.

    Unknown kickoff -> 0 (price it every pass): the capture scripts keep such fixtures rather than
    drop them, and an unknown lead time must not be mistaken for a far one."""
    if minutes_to_ko is None or not math.isfinite(minutes_to_ko):
        return 0.0
    for upper, interval in DUE_INTERVALS:
        if minutes_to_ko <= upper:
            return interval
    return DUE_INTERVALS[-1][1]


def sleep_seconds(nearest_ko_min: float | None) -> int:
    if nearest_ko_min is None:
        return SLEEP_STEPS[-1][1]
    for upper, secs in SLEEP_STEPS:
        if nearest_ko_min <= upper:
            return secs
    return SLEEP_STEPS[-1][1]


def plan_next(kickoff_minutes: list[float], elapsed_min: float, *, base_min: float = BASE_MIN,
              keepalive_min: float = KEEPALIVE_MIN, cap_min: float = CAP_MIN
              ) -> tuple[bool, int]:
    """(continue?, seconds to sleep) for the loop, from the kickoffs the last pass saw.

    `kickoff_minutes` are minutes from NOW to each fixture's kickoff (negative = already started;
    those are ignored). Continue while inside the base length; past it, only while a kickoff is
    still ahead after the sleep and within `keepalive_min`; never past `cap_min`."""
    upcoming = sorted(m for m in kickoff_minutes if m is not None and math.isfinite(m) and m > 0)
    nearest = upcoming[0] if upcoming else None
    sleep_s = sleep_seconds(nearest)
    after = elapsed_min + sleep_s / 60.0
    if after >= cap_min:
        return False, sleep_s
    if after < base_min:
        return True, sleep_s
    wanted = [m for m in upcoming if sleep_s / 60.0 < m <= keepalive_min]
    return bool(wanted), sleep_s


class PassScheduler:
    """Per-run state shared by the passes of one NEAR loop (a JSON file in the runner's temp dir).

    `due(fid, hours)` -> should this pass fetch the fixture; `mark(fid)` after a fetch;
    `see(fid, hours)` records a kickoff for the loop plan; `close()` persists and prints
    LOOP_CONTINUE / LOOP_SLEEP_S for the workflow's bash loop.
    """

    def __init__(self, state_path: str | os.PathLike | None, now: float | None = None,
                 loop_start_s: float | None = None, base_min: float = BASE_MIN,
                 keepalive_min: float = KEEPALIVE_MIN, cap_min: float = CAP_MIN):
        self.path = Path(state_path) if state_path else None
        self.now = time.time() if now is None else float(now)
        self.loop_start_s = loop_start_s
        self.base_min, self.keepalive_min, self.cap_min = base_min, keepalive_min, cap_min
        self.last_fetch: dict[str, float] = {}
        self.kickoffs: dict[str, float] = {}
        self.n_due = self.n_skipped = 0
        if self.path and self.path.exists():
            try:
                self.last_fetch = {str(k): float(v) for k, v in
                                   json.loads(self.path.read_text(encoding="utf-8")).get(
                                       "last_fetch", {}).items()}
            except Exception:                                         # noqa: BLE001
                self.last_fetch = {}          # a corrupt state file means "everything due"

    @classmethod
    def from_env(cls) -> "PassScheduler":
        def f(name, default):
            try:
                return float(os.environ[name])
            except (KeyError, ValueError):
                return default
        start = os.getenv("LOOP_START_S")
        return cls(os.getenv("CAPTURE_STATE") or None,
                   loop_start_s=float(start) if start else None,
                   base_min=f("LOOP_BASE_MIN", BASE_MIN),
                   keepalive_min=f("LOOP_KEEPALIVE_MIN", KEEPALIVE_MIN),
                   cap_min=f("LOOP_CAP_MIN", CAP_MIN))

    @property
    def active(self) -> bool:
        return self.path is not None

    def see(self, fid, hours_to_ko: float | None) -> None:
        if fid is not None and hours_to_ko is not None and math.isfinite(hours_to_ko):
            self.kickoffs[str(fid)] = hours_to_ko * 60.0

    def due(self, fid, hours_to_ko: float | None) -> bool:
        """True when the fixture should be fetched on this pass. Always True when inactive."""
        self.see(fid, hours_to_ko)
        if not self.active or fid is None:
            self.n_due += 1
            return True
        mins = None if hours_to_ko is None else hours_to_ko * 60.0
        last = self.last_fetch.get(str(fid))
        ok = last is None or (self.now - last) / 60.0 >= due_interval_min(mins)
        if ok:
            self.n_due += 1
        else:
            self.n_skipped += 1
        return ok

    def mark(self, fid) -> None:
        if fid is not None:
            self.last_fetch[str(fid)] = self.now

    def plan(self) -> tuple[bool, int] | None:
        if self.loop_start_s is None:
            return None
        elapsed = (self.now - self.loop_start_s) / 60.0
        return plan_next(list(self.kickoffs.values()), elapsed, base_min=self.base_min,
                         keepalive_min=self.keepalive_min, cap_min=self.cap_min)

    def close(self) -> None:
        if self.active:
            try:
                self.path.write_text(json.dumps({"last_fetch": self.last_fetch}),
                                     encoding="utf-8")
            except Exception as e:                                    # noqa: BLE001
                print(f"[schedule] state not saved ({e})")
            print(f"[schedule] {self.n_due} fixture(s) due, {self.n_skipped} not yet due")
        p = self.plan()
        if p is not None:
            cont, sleep_s = p
            # Machine-readable, one per line, parsed by the workflow loop.
            print(f"LOOP_CONTINUE={1 if cont else 0}")
            print(f"LOOP_SLEEP_S={sleep_s}")


# ── WIDE passes from inside the loop (owner-approved 2026-10-26) ─────────────────────────────────
# WHY. A WIDE pass (every look-ahead fixture, no --max-hours) used to run only when the WIDE cron
# itself was delivered. Measured 2026-10-05..10-26 from the consensus archives: 3.2-3.3 delivered
# WIDE passes/day per job against 4 requested, landing hours late (00:05 -> ~04:00) and with
# 7-9 h gaps. The market-first engine refuses a book whose pair is older than its 12 h stale limit
# (market_edge stale_hard_minutes=720), so the far horizon cannot depend on a cron landing.
#
# HOW. Every pass of the NEAR loop first asks `wide_due`. A WIDE pass runs when the last one is
# >= WIDE_EVERY_H old, read from a RECORDED timestamp in a committed file
# (output/capture_wide/<writer>.json) — never from file mtime, which `git checkout` resets on every
# CI run. A missing, corrupt or future timestamp is UNKNOWN AGE and treated as infinitely old:
# the pass runs (the first FPL freshness fix fell back otherwise and deadlocked itself).
#
# HARD GUARDS (env-overridable, per UTC day and per job): at most WIDE_MAX_PER_DAY passes and at
# most WIDE_MAX_CALLS_PER_DAY API calls spent by WIDE passes (a pass is refused when calls_today
# plus the last pass's measured cost would exceed it). No WIDE pass starts within WIDE_LATEST_MIN
# of the loop cap, so a late WIDE pass cannot push the job into its timeout. MIN_QUOTA in the
# capture scripts stays the final backstop. Nothing in this module makes an API call.
WIDE_STATE_DIR = Path(__file__).resolve().parents[1] / "output" / "capture_wide"
WIDE_EVERY_H = 6.0
WIDE_MAX_PER_DAY = 4
WIDE_MAX_CALLS_PER_DAY = 2500
WIDE_EST_CALLS = 500            # cost assumed for a pass before one has been measured
WIDE_LATEST_MIN = 15.0
CLOSE_GUARD_MIN = 35.0          # defer a WIDE pass while a kickoff is this close...
WIDE_DEFER_MAX_H = 2.0          # ...but never more than this past due


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _parse_utc(s) -> float | None:
    """Epoch seconds from an ISO 'YYYY-MM-DDTHH:MM:SSZ' string, or None when unusable."""
    from datetime import datetime, timezone
    try:
        return datetime.strptime(str(s), "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None


def _iso(ts: float) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_wide_state(path) -> dict:
    """The recorded WIDE state, or {} when missing / unreadable (= unknown age)."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:                                                 # noqa: BLE001
        return {}


def wide_due(state: dict, now: float, *, every_h: float = WIDE_EVERY_H,
             max_per_day: int = WIDE_MAX_PER_DAY,
             max_calls_per_day: float = WIDE_MAX_CALLS_PER_DAY,
             loop_elapsed_min: float | None = None, loop_cap_min: float | None = None,
             next_ko_min: float | None = None) -> tuple[bool, str]:
    """(run a WIDE pass now?, reason). Pure: no I/O and no clock of its own."""
    if (loop_elapsed_min is not None and loop_cap_min is not None
            and loop_elapsed_min > loop_cap_min - WIDE_LATEST_MIN):
        return False, "LOOP_TOO_LATE"
    last = _parse_utc(state.get("last_wide_utc"))
    if last is None or last > now + 300:
        return True, "UNKNOWN_AGE"                # never recorded / corrupt / future: run now
    age_h = (now - last) / 3600.0
    if age_h < every_h:
        return False, f"FRESH({age_h:.1f}h<{every_h:g}h)"
    # A WIDE pass takes minutes. While a close is forming (a kickoff inside CLOSE_GUARD_MIN, where
    # the loop prices every 2 minutes) it waits — but only up to WIDE_DEFER_MAX_H past due, so a
    # busy Saturday cannot starve the far horizon.
    if (next_ko_min is not None and 0 < next_ko_min <= CLOSE_GUARD_MIN
            and age_h < every_h + WIDE_DEFER_MAX_H):
        return False, f"DEFER_CLOSE_FORMING(ko in {next_ko_min:.0f}m)"
    today = _iso(now)[:10]
    same_day = state.get("day") == today
    passes = int(state.get("passes_today", 0) or 0) if same_day else 0
    calls = float(state.get("calls_today", 0) or 0) if same_day else 0.0
    if passes >= max_per_day:
        return False, f"DAY_PASS_CAP({passes}>={max_per_day})"
    est = float(state.get("last_pass_calls") or WIDE_EST_CALLS)
    if calls + est > max_calls_per_day:
        return False, f"DAY_CALL_CAP({calls:.0f}+{est:.0f}>{max_calls_per_day:.0f})"
    return True, f"DUE({age_h:.1f}h)"


def record_wide(path, now: float, calls: int) -> dict:
    """Record a completed WIDE pass: its timestamp plus per-UTC-day pass and call counters."""
    p = Path(path)
    st = load_wide_state(p)
    today = _iso(now)[:10]
    if st.get("day") != today:
        st = {"day": today, "passes_today": 0, "calls_today": 0}
    st.update(last_wide_utc=_iso(now), day=today,
              passes_today=int(st.get("passes_today", 0) or 0) + 1,
              calls_today=int(st.get("calls_today", 0) or 0) + int(calls),
              last_pass_calls=int(calls))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(st, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return st


def record_wide_from_env(started: float, calls: int) -> None:
    """Called by a capture script at the end of a WIDE pass when CAPTURE_WIDE_STATE is set.
    `started` is the pass's own start time, so the next due check measures from when the far
    prices were actually taken. Never raises into the capture."""
    path = os.getenv("CAPTURE_WIDE_STATE")
    if not path:
        return
    try:
        st = record_wide(path, started, calls)
        print(f"[wide] recorded pass {st['passes_today']} today, {calls} calls "
              f"({st['calls_today']} WIDE calls today)")
    except Exception as e:                                            # noqa: BLE001
        print(f"[wide] state not recorded ({e})")


def main(argv: list[str] | None = None) -> int:
    """`python -m src.capture_schedule wide-due --state PATH` -> WIDE_DUE=0|1 and WIDE_REASON=."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["wide-due"])
    ap.add_argument("--state", required=True)
    ap.add_argument("--next-ko-min", default="",
                    help="minutes to the nearest kickoff from the last NEAR pass (blank = unknown)")
    a = ap.parse_args(argv)
    try:
        next_ko = float(a.next_ko_min) if str(a.next_ko_min).strip() else None
    except ValueError:
        next_ko = None
    now = time.time()
    start = os.getenv("LOOP_START_S")
    try:
        elapsed = (now - float(start)) / 60.0 if start else None
    except ValueError:
        elapsed = None
    ok, why = wide_due(load_wide_state(a.state), now,
                       every_h=_env_float("CAPTURE_WIDE_EVERY_H", WIDE_EVERY_H),
                       max_per_day=int(_env_float("CAPTURE_WIDE_MAX_PER_DAY", WIDE_MAX_PER_DAY)),
                       max_calls_per_day=_env_float("CAPTURE_WIDE_MAX_CALLS_PER_DAY",
                                                    WIDE_MAX_CALLS_PER_DAY),
                       loop_elapsed_min=elapsed,
                       loop_cap_min=_env_float("LOOP_CAP_MIN", CAP_MIN),
                       next_ko_min=next_ko)
    print(f"WIDE_DUE={1 if ok else 0}")
    print(f"WIDE_REASON={why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
