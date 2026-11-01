"""ONE-OFF (2026-11-01): regrade player props wrongly voided as 'DNP'.

    python scripts/oneoff_regrade_false_dnp_2026_11.py [--dry-run]

The props settler matched players by name, so 'Armand Laurienté' (ledger) never met 'A. Laurienté'
(API-Football stats) and the prop was voided as "did not play" (fixed in update_results.py, a8a0e6ec).
This regrades the past ones from player_history.parquet - the same API-Football player stats, keyed
by player_id - using the settler's own rules:

  * only rows voided with notes == 'DNP' that carry a player_id;
  * the SAME player_id has a history row within +/-1 day of the match date with minutes > 0
    (0 minutes stays VOID 'DNP (0 min)', exactly as the settler would grade it);
  * result from update_results._resolve_player_market on that row's goals / shots on target /
    assists / yellow cards; P/L only for priced rows (an unpriced row has no P/L, as in the settler);
  * notes = 'REGRADED_FROM_FALSE_DNP_2026-11-01'. Rows not found in history are left untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import update_results as U  # noqa: E402

NOTE = "REGRADED_FROM_FALSE_DNP_2026-11-01"


def main(dry_run: bool = False) -> int:
    led = pd.read_csv(U.PLAYER_LEDGER_FILE, dtype=str, keep_default_na=False)
    h = pd.read_parquet(ROOT / "player_history.parquet",
                        columns=["player_id", "date", "minutes", "goals", "assists", "shots_on_target",
                                 "yellow_cards"])
    h["d"] = pd.to_datetime(h["date"], errors="coerce").dt.normalize()
    h["pid"] = pd.to_numeric(h["player_id"], errors="coerce").astype("Int64").astype(str)
    by_pid = {k: g for k, g in h.groupby("pid")}

    cand = led[(led["result"] == "VOID") & (led["notes"] == "DNP") & (led.get("player_id", "") != "")]
    fixed, still = {"WIN": 0, "LOSS": 0, "0min": 0}, 0
    for i, r in cand.iterrows():
        try:
            pid = str(int(float(r["player_id"])))
            day = pd.Timestamp(str(r["match_date"])[:10])
        except (TypeError, ValueError):
            continue
        g = by_pid.get(pid)
        if g is None:
            still += 1
            continue
        g = g[(g["d"] >= day - pd.Timedelta(days=1)) & (g["d"] <= day + pd.Timedelta(days=1))]
        if g.empty:
            still += 1
            continue
        s = g.sort_values("minutes", ascending=False).iloc[0]
        if not float(s["minutes"] or 0) > 0:
            led.at[i, "notes"] = "DNP (0 min)"
            fixed["0min"] += 1
            continue
        stats = {"goals": s["goals"], "shots_on_target": s["shots_on_target"], "assists": s["assists"],
                 "yellow_card": int(float(s["yellow_cards"] or 0) > 0)}
        res = U._resolve_player_market(r["market"], stats)
        if not res:
            continue
        try:
            odds = float(r.get("market_odds") or "nan")
        except ValueError:
            odds = float("nan")
        led.at[i, "result"] = res
        led.at[i, "pnl"] = (str(round(odds - 1.0, 4) if res == "WIN" else -1.0) if odds > 1.0 else "")
        led.at[i, "notes"] = NOTE
        fixed[res] += 1
    print(f"false-DNP candidates {len(cand)}: regraded WIN {fixed['WIN']} / LOSS {fixed['LOSS']}, "
          f"0-minute {fixed['0min']} (stay VOID), not in history {still} (untouched)")
    if not dry_run:
        led.to_csv(U.PLAYER_LEDGER_FILE, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main("--dry-run" in sys.argv))
