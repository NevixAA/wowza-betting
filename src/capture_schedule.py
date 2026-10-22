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
