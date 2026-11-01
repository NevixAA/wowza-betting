"""OddsAPI credit accounting from GitHub Actions run logs (R2-C finding 23).

WHY LOGS. Every OddsAPI consumer already prints what it spent, but nothing added it up, so the plan
ran dry on 2026-10-30/31 with no warning: the BTTS refresh clock was broken and every predict run
bought the whole board (~62% of October's credits). The log is the only place where ALL consumers
(predict's BTTS per-event fetch, the props odds fetcher, the exchange capture, ...) report, and the
account-level `x-requests-remaining` value printed by any of them measures every consumer at once.

Lines parsed (one regex each, all optional — a consumer that prints nothing simply does not count):
  [oddsapi_usage] {json}                                 src/btts_odds.py (R2-C, per league/market)
  [btts] N call(s), M credit(s) spent, ... (skipped K within ...h refresh, ...)   src/btts_odds.py
  [odds_fetcher] ... quota remaining: R                  player_model/odds_fetcher.py
  [exchange] ... credits S ...                           scripts/capture_exchange_odds.py
  x-requests-remaining / requests_remaining = R          anything else that prints it

Health (`build_health`): credits per workflow / league / market, BTTS duplicate-fetch suppression
rate (fixtures skipped by the 2h refresh clock / fixtures considered), account burn per day from the
remaining-credits series, projected days to exhaustion, and WARN/FAIL thresholds before exhaustion.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone

QUOTA_RULE_VERSION = "oddsapi-quota-1.0"
#: Plan allowance (OddsAPI 100k/month plan, src/btts_odds.py docstring). Override via build_health.
DEFAULT_MONTHLY_ALLOWANCE = 100_000
WARN_REMAINING_FRACTION = 0.25        # WARN when fewer than 25% of the month's credits remain ...
WARN_DAYS_TO_EXHAUSTION = 7           # ... or when the current burn exhausts the pool within 7 days
FAIL_REMAINING = 5_000                # FAIL: production predict would run dry within ~2 days

_TS = r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)"
RE_USAGE = re.compile(_TS + r".*\[oddsapi_usage\]\s+(?P<json>\{.*\})")
RE_BTTS = re.compile(_TS + r".*\[btts\]\s+(?P<calls>\d+) call\(s\), (?P<credits>\d+) credit\(s\) spent, "
                     r"filled (?P<filled>\d+)/(?P<rows>\d+) fixtures \(skipped (?P<fresh>\d+) within")
RE_BTTS_OLD = re.compile(_TS + r".*\[btts\]\s+(?P<calls>\d+) call\(s\), filled (?P<filled>\d+)/"
                         r"(?P<rows>\d+) fixtures \(skipped (?P<fresh>\d+) within")
RE_REMAINING = re.compile(_TS + r".*(?:quota remaining:|x-requests-remaining[\"']?[:=]\s*|"
                          r"requests_remaining[\"']?[:=]\s*)\s*(?P<rem>\d+)", re.I)
RE_EXCHANGE = re.compile(_TS + r".*\[exchange\].*credits (?P<credits>\d+)")


def parse_log_text(text: str, workflow: str, run_id: str | int = "") -> dict:
    """All OddsAPI facts in one run's log text."""
    out = {"workflow": workflow, "run_id": str(run_id), "usage": [], "btts": [], "remaining": [],
           "exchange_credits": 0}
    for line in text.splitlines():
        m = RE_USAGE.search(line)
        if m:
            try:
                rec = json.loads(m.group("json"))
                rec["ts"] = m.group("ts")
                out["usage"].append(rec)
                if rec.get("requests_remaining") is not None:
                    out["remaining"].append((m.group("ts"), int(rec["requests_remaining"])))
            except ValueError:
                pass
            continue
        m = RE_BTTS.search(line)
        if m:
            d = {k: int(v) for k, v in m.groupdict().items() if k != "ts"}
            d["ts"] = m.group("ts")
            out["btts"].append(d)
            continue
        m = RE_BTTS_OLD.search(line)
        if m:
            d = {k: int(v) for k, v in m.groupdict().items() if k != "ts"}
            d["credits"] = None                   # pre-2026-11-01 log line had no credit count
            d["ts"] = m.group("ts")
            out["btts"].append(d)
            continue
        m = RE_EXCHANGE.search(line)
        if m:
            out["exchange_credits"] += int(m.group("credits"))
        m = RE_REMAINING.search(line)
        if m:
            out["remaining"].append((m.group("ts"), int(m.group("rem"))))
    return out


def parse_log_zip(blob: bytes, workflow: str, run_id) -> dict:
    texts = []
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for n in z.namelist():
            if n.endswith(".txt") and "/" not in n.strip("/"):      # top-level job logs only
                texts.append(z.read(n).decode("utf-8", "replace"))
        if not texts:
            texts = [z.read(n).decode("utf-8", "replace") for n in z.namelist() if n.endswith(".txt")]
    return parse_log_text("\n".join(texts), workflow, run_id)


def _ts(s: str) -> datetime:
    base = s.rstrip("Z").split(".")[0]
    return datetime.strptime(base, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)


def build_health(runs: list[dict], window_hours: float, now: datetime | None = None,
                 monthly_allowance: int = DEFAULT_MONTHLY_ALLOWANCE, runs_listed: int | None = None,
                 log_failures: int = 0) -> dict:
    now = now or datetime.now(timezone.utc)
    by_wf = defaultdict(lambda: {"runs": 0, "credits": 0, "calls": 0, "runs_with_unknown_credits": 0})
    by_league = defaultdict(int)
    by_market = defaultdict(int)
    considered = fresh = 0
    remaining = []
    for r in runs:
        w = by_wf[r["workflow"]]
        w["runs"] += 1
        if r["usage"]:
            for u in r["usage"]:
                w["credits"] += int(u.get("credits", 0))
                w["calls"] += int(u.get("calls", 0))
                for lg, c in (u.get("credits_by_league") or {}).items():
                    by_league[lg] += int(c)
                for mk, c in (u.get("credits_by_market") or {"btts": u.get("credits", 0)}).items():
                    by_market[mk] += int(c)
                considered += int(u.get("fixtures_considered", 0))
                fresh += int(u.get("suppressed_fresh", 0))
        else:
            for b in r["btts"]:
                w["calls"] += b["calls"]
                if b.get("credits") is None:
                    w["runs_with_unknown_credits"] += 1
                    w["credits"] += 2 * b["calls"]              # 2 credits/call at uk,eu (documented)
                else:
                    w["credits"] += b["credits"]
                by_market["btts"] += b["credits"] if b.get("credits") is not None else 2 * b["calls"]
                considered += b["rows"]
                fresh += b["fresh"]
        if r.get("exchange_credits"):
            w["credits"] += r["exchange_credits"]
            by_market["exchange_h2h"] += r["exchange_credits"]
        remaining += r["remaining"]

    credits_window = sum(w["credits"] for w in by_wf.values())
    per_day_logged = credits_window * 24.0 / window_hours if window_hours else None
    rem_sorted = sorted((_ts(t), v) for t, v in remaining)
    burn = None
    latest_rem = rem_sorted[-1][1] if rem_sorted else None
    if len(rem_sorted) >= 2:
        # Sum only DECREASES between consecutive observations: a monthly reset jumps UP.
        spent = sum(max(0, a[1] - b[1]) for a, b in zip(rem_sorted, rem_sorted[1:]))
        span_h = (rem_sorted[-1][0] - rem_sorted[0][0]).total_seconds() / 3600.0
        if span_h >= 1:
            burn = spent * 24.0 / span_h
    rate = burn if burn is not None else per_day_logged
    days_left = (latest_rem / rate) if (latest_rem is not None and rate) else None
    status, reasons = "PASS", []
    if latest_rem is None:
        status = "UNKNOWN"
        reasons.append("no x-requests-remaining value in any log of the window; account level unknown")
    else:
        if latest_rem < FAIL_REMAINING:
            status = "FAIL"
            reasons.append(f"{latest_rem} credits left (< {FAIL_REMAINING})")
        elif latest_rem < WARN_REMAINING_FRACTION * monthly_allowance:
            status = "WARN"
            reasons.append(f"{latest_rem} credits left (< {WARN_REMAINING_FRACTION:.0%} of {monthly_allowance})")
        if days_left is not None and days_left < WARN_DAYS_TO_EXHAUSTION and status != "FAIL":
            status = "WARN"
            reasons.append(f"exhaustion in {days_left:.1f} days at {rate:.0f} credits/day")
    supp = fresh / considered if considered else None
    return {
        "artifact": "oddsapi_quota_health", "rule_version": QUOTA_RULE_VERSION,
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "window_hours": window_hours,
        "runs_listed": runs_listed, "runs_parsed": len(runs), "log_download_failures": log_failures,
        "credits_logged_in_window": credits_window,
        "credits_per_day_logged_estimate": None if per_day_logged is None else round(per_day_logged),
        "account_remaining_latest": latest_rem,
        "account_remaining_latest_ts": rem_sorted[-1][0].strftime("%Y-%m-%dT%H:%M:%SZ") if rem_sorted else None,
        "account_burn_per_day_from_remaining": None if burn is None else round(burn),
        "projected_days_to_exhaustion": None if days_left is None else round(days_left, 1),
        "monthly_allowance_assumed": monthly_allowance,
        "by_workflow": dict(by_wf), "by_league": dict(by_league), "by_market": dict(by_market),
        "btts_fixtures_considered": considered, "btts_suppressed_by_refresh_clock": fresh,
        "btts_duplicate_fetch_suppression_rate": None if supp is None else round(supp, 4),
        "thresholds": {"warn_remaining_fraction": WARN_REMAINING_FRACTION,
                       "warn_days_to_exhaustion": WARN_DAYS_TO_EXHAUSTION, "fail_remaining": FAIL_REMAINING},
        "status": status, "reasons": reasons,
        "health_check_line": (f"OddsAPI quota {status}: remaining {latest_rem}, burn "
                              f"{'?' if rate is None else round(rate)}/day, "
                              f"{'?' if days_left is None else round(days_left, 1)} days left, BTTS "
                              f"suppression {'?' if supp is None else f'{supp:.0%}'}"),
    }


def fetch_runs(repo: str, token: str, hours: float, workflows: list[str] | None = None,
               max_runs: int = 200, session=None) -> tuple[list[dict], int, int]:
    """Download and parse the logs of every completed run in the window. Returns
    (parsed runs, runs listed, log download failures). Requires a token with actions:read."""
    import requests
    s = session or requests.Session()
    h = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    runs, page = [], 1
    while len(runs) < max_runs:
        r = s.get(f"https://api.github.com/repos/{repo}/actions/runs",
                  params={"created": f">={since}", "status": "completed", "per_page": 100, "page": page},
                  headers=h, timeout=30)
        r.raise_for_status()
        batch = r.json().get("workflow_runs", [])
        if not batch:
            break
        for w in batch:
            name = (w.get("path") or "").split("/")[-1] or w.get("name", "")
            if workflows and name not in workflows:
                continue
            runs.append({"id": w["id"], "workflow": name})
        page += 1
    runs = runs[:max_runs]
    parsed, fails = [], 0
    for run in runs:
        try:
            r = s.get(f"https://api.github.com/repos/{repo}/actions/runs/{run['id']}/logs",
                      headers=h, timeout=60)
            if r.status_code != 200:
                fails += 1
                continue
            parsed.append(parse_log_zip(r.content, run["workflow"], run["id"]))
        except Exception:                                          # noqa: BLE001
            fails += 1
    return parsed, len(runs), fails
