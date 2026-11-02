"""
Sidecars for two props ledger defects (props_truth, round-2 findings 4 and 5, 2026-11-02).
Read-only on output/player_ledger.csv and player_history (column projection); no network. The ledger
is NEVER rewritten - consumers exclude the listed rows by (ledger_row, player_id, fixture_id, market,
match_date).

1. output/corrections/props_sentinel_prob.csv - World Cup tips whose model_prob is the market CAP
   that predict.py wrote in place of the model's probability (`min(p, cap, 0.95)`, removed
   2026-11-02: capped_prob now refuses the tip). The caps of the 2026-06-23 run were goals 0.5,
   sot 0.8, cards 0.4 (top-1% career rate). exclude = True for every row.
2. output/corrections/props_ghost_tips.csv - graded rows (WIN/LOSS) that disagree with
   player_history:
     other_fixture_same_day  the same player_id played a DIFFERENT fixture that day (another club):
                             the grade came from a match the tip was not about. exclude = True.
     stats_disagree          same fixture, but the result recomputed from player_history's stats
                             differs from the ledger's. exclude = False (which source is right is
                             not decided here; listed for review - player_history counts extra time,
                             props settle on 90 minutes, owner decision 2026-11-02).
3. output/corrections/props_truth_audit.json - counts and the P/L of the excluded rows.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from player_model.club_rule import same_fixture_sides  # noqa: E402

CAPS_2026_06 = {"goals": 0.5, "sot": 0.8, "cards": 0.4}
STAT = {"goals": "goals", "sot": "shots_on_target", "cards": "yellow_cards", "assists": "assists"}
KEEP = ["ledger_row", "signal_date", "match_date", "league", "home_team", "away_team", "player_name",
        "team", "player_id", "fixture_id", "market", "tier", "model_prob", "market_odds", "result",
        "pnl", "notes"]


def sentinel_rows(led: pd.DataFrame) -> pd.DataFrame:
    p = pd.to_numeric(led["model_prob"], errors="coerce")
    cap = led["market"].map(CAPS_2026_06)
    wc = led["league"].astype(str).eq("World Cup") & led["tier"].isin(["VALUABLE", "MARKSMAN", "SNIPER"])
    hit = wc & ((p == cap) | (p == 0.5))
    out = led.loc[hit, KEEP].copy()
    out["sentinel_value"] = p[hit]
    out["reason"] = np.where(p[hit] == cap[hit], "model_prob == that run's market cap (min(p, cap) wrote the cap)",
                             "model_prob exactly 0.5 on a non-goals market (goals cap value; may be genuine)")
    out["in_round2_brief_168"] = (p[hit] == 0.5) & led.loc[hit, "tier"].isin(["VALUABLE", "MARKSMAN"])
    out["exclude"] = True
    return out


def _need(market: str):
    m = re.fullmatch(r"([a-z]+?)(\d*)", str(market))
    return (m.group(1), int(m.group(2) or 1)) if m else (None, None)


def ghost_rows(led: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    g = led[led["result"].isin(["WIN", "LOSS"])].copy()
    g["_pid"] = pd.to_numeric(g["player_id"], errors="coerce")
    h = hist.copy()
    h["_d"] = pd.to_datetime(h["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    m = g.merge(h, left_on=["_pid", "match_date"], right_on=["player_id", "_d"], how="inner",
                suffixes=("", "_h"))
    fx = pd.to_numeric(m["fixture_id"], errors="coerce")
    fx_h = pd.to_numeric(m["fixture_id_h"], errors="coerce")
    sides = [same_fixture_sides(a, b, c, d) for a, b, c, d in
             zip(m["home_team"], m["away_team"], m["home_team_h"], m["away_team_h"])]
    other = np.where(fx.notna(), fx != fx_h, ~np.array(sides, dtype=bool))
    hres = []
    for mk, *vals in zip(m["market"], *(m[c] for c in ("goals", "shots_on_target", "yellow_cards", "assists"))):
        base, n = _need(mk)
        v = dict(zip(("goals", "shots_on_target", "yellow_cards", "assists"), vals)).get(STAT.get(base))
        hres.append(None if v is None or pd.isna(v) else ("WIN" if v >= n else "LOSS"))
    m["history_result"] = hres
    m["_other"] = other
    # A ledger row is a ghost only when NONE of the player's history rows that day is its fixture;
    # when one is, only that row is compared (another row that day is not this tip's match).
    any_same = (~m["_other"]).groupby(m["ledger_row"]).transform("any")
    m = m[~(any_same & m["_other"])]
    m["category"] = np.where(m["_other"], "other_fixture_same_day",
                             np.where(pd.notna(m["history_result"]) & (m["history_result"] != m["result"]),
                                      "stats_disagree", ""))
    m = m[m["category"] != ""]
    out = m[KEEP + ["category", "history_result"]].copy()
    out["history_fixture_id"] = m["fixture_id_h"].to_numpy()
    out["history_team"] = m["team_h"].to_numpy()
    out["history_match"] = (m["home_team_h"].astype(str) + " vs " + m["away_team_h"].astype(str)).to_numpy()
    out["history_minutes"] = m["minutes"].to_numpy()
    out["exclude"] = out["category"].eq("other_fixture_same_day")
    # one listing per ledger row (a player can have two history rows on one date)
    out = out.sort_values(["ledger_row", "category"]).drop_duplicates("ledger_row", keep="first")
    return out


def _pnl(s: pd.DataFrame) -> float:
    return round(float(pd.to_numeric(s["pnl"], errors="coerce").sum()), 2)


def main() -> int:
    led = pd.read_csv(ROOT / "output" / "player_ledger.csv", dtype=str).reset_index(names="ledger_row")
    cols = ["fixture_id", "player_id", "date", "team", "home_team", "away_team", "minutes", "goals",
            "shots_on_target", "yellow_cards", "assists"]
    parts = sorted((ROOT / "player_history.parquet").glob("month_2026-*.parquet"))
    hist = pd.concat([pd.read_parquet(p, columns=cols) for p in parts], ignore_index=True)
    out_dir = ROOT / "output" / "corrections"
    out_dir.mkdir(parents=True, exist_ok=True)
    sen = sentinel_rows(led)
    gh = ghost_rows(led, hist)
    sen.to_csv(out_dir / "props_sentinel_prob.csv", index=False)
    gh.to_csv(out_dir / "props_ghost_tips.csv", index=False)
    graded = led[led["result"].isin(["WIN", "LOSS"])]
    staked = graded[graded["tier"].isin(["SNIPER", "MARKSMAN", "VALUABLE"])]
    excl = set(sen["ledger_row"]) | set(gh.loc[gh["exclude"], "ledger_row"])
    st_after = staked[~staked["ledger_row"].isin(excl)]
    ghost_other = gh[gh["category"] == "other_fixture_same_day"]
    audit = {
        "written": "2026-11-02 (props_truth)", "ledger": "output/player_ledger.csv (NOT modified)",
        "ledger_rows": int(len(led)),
        "sentinel": {"rows": int(len(sen)), "in_round2_brief_168": int(sen["in_round2_brief_168"].sum()),
                     "by_value": {str(k): int(v) for k, v in sen["sentinel_value"].value_counts().items()},
                     "results": sen["result"].fillna("pending").value_counts().to_dict(),
                     "pnl_u": _pnl(sen),
                     "writer": "player_model/predict.py `p_model = min(p_model, _MARKET_CAPS[market], 0.95)`; "
                               "replaced by capped_prob (refuses the tip) on fix4/props_truth"},
        "ghost": {"other_fixture_same_day": int(len(ghost_other)),
                  "other_fixture_same_day_regraded_from_false_dnp":
                      int(ghost_other["notes"].astype(str).str.startswith("REGRADED_FROM_FALSE_DNP").sum()),
                  "other_fixture_same_day_pnl_u": _pnl(ghost_other),
                  "other_fixture_same_day_staked_tiers": int(ghost_other["tier"].isin(
                      ["SNIPER", "MARKSMAN", "VALUABLE"]).sum()),
                  "stats_disagree": int((gh["category"] == "stats_disagree").sum()),
                  "graded_rows_joined_to_history": "player_id + match_date, history months 2026-*"},
        "staked_tiers_graded_pnl_u": {"as_stored": _pnl(staked), "excluding_listed_rows": _pnl(st_after),
                                      "rows_as_stored": int(len(staked)), "rows_after": int(len(st_after))},
        "consumers": "exclude rows with exclude == True, keyed on ledger_row + (player_id, fixture_id, "
                     "market, match_date); ledger_row is the 0-based row of player_ledger.csv at the "
                     "commit this file was written on",
    }
    (out_dir / "props_truth_audit.json").write_text(json.dumps(audit, indent=2, default=str) + "\n",
                                                    encoding="utf-8")
    print(json.dumps(audit, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
