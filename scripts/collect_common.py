"""
Shared plumbing for the Data Truth real-data collectors (scripts/collect_*.py, owner decision C,
2026-11-02). Imported by the collectors; not a collector itself.

API-FOOTBALL BUDGET (hard, checked BEFORE every call):
  * the account-wide remaining daily quota, read from x-ratelimit-requests-remaining on every
    response (and from the free /status endpoint before the first call), must stay >= FLOOR
    (25,000 of the 75,000/day plan; normal use is up to ~23,300/day, so the floor protects it);
  * this job's own calls per UTC run-day may not exceed DAILY_CAP (45,000). They are journaled in
    output/collect/af_budget.json, which the workflow commits, so two runs on one day share it.
  Either limit raises BudgetStop; the collectors save what they have and exit 0 (resumable).

JOURNAL. Every collector writes its output parquet incrementally (every FLUSH_EVERY items) and
treats it as its own journal: a key already present - including a recorded EMPTY answer - is never
fetched again. A failed request (network / HTTP error) records nothing and is retried next run.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLLECT_DIR = ROOT / "output" / "collect"
BUDGET_FILE = COLLECT_DIR / "af_budget.json"
BASE_URL = "https://v3.football.api-sports.io"

DAILY_CAP = 45_000          # this job's API-Football calls per UTC day (hard)
FLOOR = 25_000              # stop when the account's remaining daily quota falls below this
# A run-day override is allowed only inside these bounds (owner 2026-11-02: floor 10,000 for one day
# to finish sooner); the workflow sets COLLECT_AF_FLOOR / COLLECT_AF_DAILY_CAP explicitly for it.
_FLOOR_MIN, _CAP_MAX = 10_000, 60_000
for _var, _lo, _hi in (("COLLECT_AF_FLOOR", _FLOOR_MIN, 75_000), ("COLLECT_AF_DAILY_CAP", 1, _CAP_MAX)):
    _v = os.getenv(_var, "").strip()
    if _v:
        if not _v.isdigit() or not _lo <= int(_v) <= _hi:
            raise SystemExit(f"{_var}={_v!r} outside [{_lo:,}, {_hi:,}] - refusing to run")
        if _var == "COLLECT_AF_FLOOR":
            FLOOR = int(_v)
        else:
            DAILY_CAP = int(_v)
FLUSH_EVERY = 200           # items between incremental parquet writes
PAUSE_S = 0.15              # between calls (the plan allows 450/min; ~400/min used elsewhere)


class BudgetStop(BaseException):
    """The API-Football budget for this run-day is spent (or the account floor is reached).
    A BaseException on purpose: wrapped code with a broad `except Exception` (backfill_af_history's
    _get) must not swallow it and carry on as if the call had merely failed."""


def utc_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class AFClient:
    """API-Football GET with the hard budget above. `session` is injectable for tests."""

    def __init__(self, key: str | None = None, session=None, budget_file: Path | None = None,
                 daily_cap: int = DAILY_CAP, floor: int = FLOOR, pause_s: float = PAUSE_S):
        self.key = (os.getenv("APIFOOTBALL_KEY", "") if key is None else key).strip()
        if session is None:
            import requests
            session = requests.Session()
        self.session = session
        self.budget_file = Path(budget_file or BUDGET_FILE)
        self.daily_cap, self.floor, self.pause_s = int(daily_cap), int(floor), float(pause_s)
        self.remaining: int | None = None
        self.calls_run = 0
        self.errors = 0
        self._journal = self._read_journal()
        self._checked_status = False

    # -- journal of this job's calls per UTC day ------------------------------------------------
    def _read_journal(self) -> dict:
        try:
            return json.loads(self.budget_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _write_journal(self) -> None:
        self.budget_file.parent.mkdir(parents=True, exist_ok=True)
        self.budget_file.write_text(json.dumps(self._journal, indent=1, sort_keys=True), encoding="utf-8")

    def calls_today(self) -> int:
        return int(self._journal.get(utc_day(), 0))

    def _count(self) -> None:
        d = utc_day()
        self._journal[d] = int(self._journal.get(d, 0)) + 1
        self.calls_run += 1
        if self.calls_run % 50 == 0:
            self._write_journal()

    def flush(self) -> None:
        self._write_journal()

    # -- the budget gate -------------------------------------------------------------------------
    def _status(self) -> None:
        """Free endpoint (does not count against the quota): the account's day usage."""
        self._checked_status = True
        try:
            r = self.session.get(f"{BASE_URL}/status", headers={"x-apisports-key": self.key}, timeout=20)
            req = ((r.json() or {}).get("response") or {}).get("requests") or {}
            cur, lim = req.get("current"), req.get("limit_day")
            if isinstance(cur, int) and isinstance(lim, int):
                self.remaining = lim - cur
                print(f"[budget] account: {cur:,}/{lim:,} used today, {self.remaining:,} remaining; "
                      f"this job today: {self.calls_today():,}/{self.daily_cap:,}")
        except Exception as e:                                           # noqa: BLE001
            print(f"[budget] /status unreadable ({e}); the first call's header will decide")

    def check(self) -> None:
        if not self._checked_status:
            self._status()
        if self.calls_today() >= self.daily_cap:
            raise BudgetStop(f"job cap reached: {self.calls_today():,} >= {self.daily_cap:,} calls today")
        if self.remaining is not None and self.remaining < self.floor:
            raise BudgetStop(f"account floor reached: {self.remaining:,} < {self.floor:,} remaining today")

    def get(self, endpoint: str, params: dict) -> dict | None:
        """One call. Returns the JSON body, or None on a failed request (nothing recorded).
        Raises BudgetStop BEFORE a call that the budget does not allow."""
        if not self.key:
            raise BudgetStop("APIFOOTBALL_KEY is not set")
        for attempt in range(3):
            self.check()
            try:
                r = self.session.get(f"{BASE_URL}/{endpoint.lstrip('/')}",
                                     headers={"x-apisports-key": self.key}, params=params, timeout=30)
            except Exception as e:                                       # noqa: BLE001
                self.errors += 1
                print(f"[af] {endpoint} {params}: {e}")
                return None
            self._count()
            rem = r.headers.get("x-ratelimit-requests-remaining")
            try:
                if rem is not None:
                    self.remaining = int(rem)
            except (TypeError, ValueError):
                pass
            if r.status_code == 429:
                time.sleep(2.0 * (attempt + 1))
                continue
            if r.status_code != 200:
                self.errors += 1
                print(f"[af] {endpoint} {params}: HTTP {r.status_code}")
                return None
            try:
                body = r.json()
            except ValueError:
                self.errors += 1
                return None
            errs = body.get("errors")
            if errs:                       # API-Football sends [] or {} when there is none
                self.errors += 1
                print(f"[af] {endpoint} {params}: provider error {errs}")
                return None
            if self.pause_s:
                time.sleep(self.pause_s)
            return body
        return None


# -- incremental parquet journal -----------------------------------------------------------------

def read_parquet(path: Path, columns=None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=columns) if columns else pd.DataFrame()
    return pd.read_parquet(path)


def append_parquet(path: Path, rows: list[dict], key: list[str], columns: list[str],
                   keep: str = "last") -> int:
    """Append `rows` to the parquet at `path` (created if absent), de-duplicated on `key` with the
    NEW row winning (`keep="first"`: the STORED row wins - an answer kept as first seen, the
    coaches / transfers refresh of fix8/history2). Returns the stored row count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows, columns=columns)
    old = read_parquet(path)
    if not old.empty:
        for c in columns:
            if c not in old.columns:
                old[c] = None
        new = pd.concat([old[columns], new], ignore_index=True)
    new = new.drop_duplicates(subset=key, keep=keep).reset_index(drop=True)
    tmp = path.with_suffix(".tmp.parquet")
    new.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return len(new)


def history_frame(columns: list[str]) -> pd.DataFrame:
    """Columns of player_history.parquet (the month-partitioned dataset at the repo root)."""
    import pyarrow.dataset as ds
    p = ROOT / "player_history.parquet"
    if not p.exists():
        return pd.DataFrame(columns=columns)
    have = set(ds.dataset(str(p), format="parquet").schema.names)
    return pd.read_parquet(p, columns=[c for c in columns if c in have])
