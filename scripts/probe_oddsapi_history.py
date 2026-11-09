"""
OddsAPI historical-data PROBE (owner 2026-11-09: "probe first"; P7 historical menu H1-H4).
=========================================================================================
Before any historical purchase, confirm what soccer history The Odds API really holds for our leagues,
one event per league, and record the EXACT credits spent (response header x-requests-last).

Per league with an OddsAPI sport key, on one probe date inside every gap window (a match day in our own
history between 2024-03-01 and 2025-03-31: after 2023-05-03 for the event markets, before 2026-04 / 06 /
08 / 10 for H2 / H4 / H1 / H3):
  F  /historical/sports/{sport}/odds          markets=totals,h2h regions=eu, 08:00 UTC     (H1 totals, H4 h2h)
  E  /historical/sports/{sport}/events        08:00 UTC -> the day's first event          (1 credit)
  S  /historical/sports/{sport}/events/{id}/odds markets=btts,alternate_totals regions=eu   (H2)
  P  same event, markets=player_goal_scorer_anytime,player_shots_on_target,player_assists,
     player_to_receive_card regions=uk,eu,us,us2 - prop leagues only                     (H3)
Cost: 10 x markets RETURNED x regions per odds call, 1 per events call; empty answers are free.
Hard cap PROBE_CREDIT_CAP (default 6,000) on the summed x-requests-last; the probe stops when reached.
Writes output/collect/oddsapi_history_probe.json (per league: what came back, lines, books, credits).
Nothing else is called, nothing is bought.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "output" / "collect" / "oddsapi_history_probe.json"
BASE = "https://api.the-odds-api.com/v4/historical/sports"
PROPS = "player_goal_scorer_anytime,player_shots_on_target,player_assists,player_to_receive_card"
WINDOW = ("2024-03-01", "2025-03-31")
CAP = int(os.getenv("PROBE_CREDIT_CAP", "6000"))


def probe_dates() -> dict:
    """{league: YYYY-MM-DD}: the busiest match day of the league in WINDOW in our own history."""
    parts = []
    for f in ("output/fd_history.parquet", "output/af_history.parquet"):
        p = ROOT / f
        if p.exists():
            parts.append(pd.read_parquet(p, columns=["league", "date"]))
    # the top leagues and the European cups are props-only leagues: their match days come from
    # player_history (the props history), not from the team banks
    ph = ROOT / "player_history.parquet"
    if ph.exists():
        import pyarrow.dataset as ds
        parts.append(ds.dataset(str(ph), format="parquet").to_table(columns=["league", "date"]).to_pandas())
    h = pd.concat(parts, ignore_index=True)
    h["d"] = pd.to_datetime(h["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    h = h[(h["d"] >= WINDOW[0]) & (h["d"] <= WINDOW[1])]
    c = h.groupby(["league", "d"]).size().reset_index(name="n")
    best = c.sort_values(["league", "n", "d"], ascending=[True, False, False]).drop_duplicates("league")
    return dict(zip(best["league"], best["d"]))


class Meter:
    def __init__(self, key):
        self.key, self.spent, self.calls, self.stopped = key, 0, 0, False
        self.unknown_cost_calls = 0

    def get(self, url, **params):
        if self.spent >= CAP:
            self.stopped = True
            return None, 0
        r = requests.get(url, params={"apiKey": self.key, **params}, timeout=30)
        self.calls += 1
        hdr = r.headers.get("x-requests-last")
        if hdr is None:
            # an unmeasured call: the cost is UNKNOWN, never assumed; stop so nothing is spent unmetered
            self.stopped, self.unknown_cost_calls = True, self.unknown_cost_calls + 1
            return {"_error": "no x-requests-last header: cost unknown, probe stopped"}, None
        last = int(hdr)
        self.spent += last
        time.sleep(0.4)
        if r.status_code != 200:
            return {"_error": r.status_code, "_text": r.text[:200]}, last
        return r.json(), last


def summarise_featured(body):
    ev = (body or {}).get("data") or []
    tot = h2h = tot25 = 0
    books = set()
    for e in ev:
        mk = {m["key"]: m for b in e.get("bookmakers", []) for m in b.get("markets", [])}
        books |= {b["key"] for b in e.get("bookmakers", [])}
        if "totals" in mk:
            tot += 1
            if any(o.get("point") == 2.5 for o in mk["totals"].get("outcomes", [])):
                tot25 += 1
        h2h += "h2h" in mk
    return {"events": len(ev), "with_totals": tot, "with_totals_2_5": tot25, "with_h2h": h2h,
            "books": len(books), "snapshot": (body or {}).get("timestamp")}


def summarise_event(body):
    d = (body or {}).get("data") or {}
    out = {"books": len(d.get("bookmakers", []) or []), "markets": {}}
    for b in d.get("bookmakers", []) or []:
        for m in b.get("markets", []):
            s = out["markets"].setdefault(m["key"], {"books": 0, "points": set(), "outcomes": 0})
            s["books"] += 1
            s["outcomes"] += len(m.get("outcomes", []))
            s["points"] |= {o.get("point") for o in m.get("outcomes", []) if o.get("point") is not None}
    for s in out["markets"].values():
        s["points"] = sorted(p for p in s["points"] if p is not None)[:12]
    return out


def run() -> dict:
    import config
    from player_model.odds_fetcher import PROP_SPORT_KEYS
    key = os.getenv("ODDS_API_KEY", "").strip()
    if not key:
        raise SystemExit("ODDS_API_KEY missing (GitHub Actions secret)")
    sk = {lg: s for lg, s in dict(config.ODDS_API_SPORT_KEYS).items() if s}
    dates = probe_dates()
    m = Meter(key)
    res = {"probed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "cap": CAP, "leagues": {}}
    for lg in sorted(sk):
        if m.stopped:
            break
        d = dates.get(lg)
        r = {"sport_key": sk[lg], "probe_date": d, "credits": {}}
        res["leagues"][lg] = r
        if not d:
            r["skipped"] = "no match day of ours in the probe window"
            continue
        t0 = f"{d}T08:00:00Z"
        body, c = m.get(f"{BASE}/{sk[lg]}/odds", regions="eu", markets="totals,h2h", date=t0, oddsFormat="decimal")
        r["credits"]["featured"] = c
        r["featured"] = summarise_featured(body) if body and "_error" not in body else body
        evs, c = m.get(f"{BASE}/{sk[lg]}/events", date=t0)
        r["credits"]["events"] = c
        later = [e for e in ((evs or {}).get("data") or []) if str(e.get("commence_time", ""))[:10] == d]
        if not later:
            r["event"] = "no event of that day in the provider's event list"
            continue
        e = sorted(later, key=lambda x: x["commence_time"])[0]
        ko = datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
        at = (ko - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        r["event"] = {"id": e["id"], "home": e.get("home_team"), "away": e.get("away_team"), "kickoff": e["commence_time"]}
        body, c = m.get(f"{BASE}/{sk[lg]}/events/{e['id']}/odds", regions="eu", markets="btts,alternate_totals",
                        date=at, oddsFormat="decimal")
        r["credits"]["side_markets"] = c
        r["side_markets"] = summarise_event(body) if body and "_error" not in body else body
        if lg in PROP_SPORT_KEYS:
            body, c = m.get(f"{BASE}/{sk[lg]}/events/{e['id']}/odds", regions="uk,eu,us,us2", markets=PROPS,
                            date=at, oddsFormat="decimal")
            r["credits"]["props"] = c
            r["props"] = summarise_event(body) if body and "_error" not in body else body
    res["credits_spent"], res["calls"], res["stopped"] = m.spent, m.calls, m.stopped
    res["calls_with_unknown_cost"] = m.unknown_cost_calls
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=str, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[probe] credits spent {m.spent} in {m.calls} calls; stopped at cap: {m.stopped}")
    return res


if __name__ == "__main__":
    run()
