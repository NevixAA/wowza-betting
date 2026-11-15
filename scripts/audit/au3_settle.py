"""AU3 Phase 6 step 2: INDEPENDENT settlement of the first-sight selections.

Does NOT import update_results.py, src/ledger.py or any selection code. Results come from the two
committed score tables (origin/main @ AUDIT_REF):
    output/af_history.parquet   API-Football (FTHG/FTAG)
    output/fd_history.parquet   football-data.co.uk (home_goals/away_goals)
Club names are matched LEAGUE-SCOPED with src.team_names.resolve_result (a name resolver, not a
settlement routine - it refuses ambiguous matches). Fixture date may differ by +-1 day (UTC vs local
kickoff). A fixture that cannot be matched is UNRESOLVED and is never counted as a loss.

Settlement (decimal odds, flat 1u):
    WIN  -> odds - 1     LOSS -> -1     VOID -> 0 and excluded from settled stake
    OU25 OVER wins total >= 3; UNDER wins total <= 2
    btts (YES) wins home>0 and away>0; over15 wins total >= 2; over35 wins total >= 4

Writes reports/au3_data/settled_{ou25,side}.csv
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from au3_common import DATA, REPO  # noqa: E402
from src.team_names import resolve_result  # noqa: E402  (name resolution only)


def load_results() -> pd.DataFrame:
    af = pd.read_parquet(REPO / "output" / "af_history.parquet",
                         columns=["league", "date", "home_team", "away_team", "FTHG", "FTAG"])
    af = af.rename(columns={"FTHG": "hg", "FTAG": "ag"}); af["src"] = "af"
    fd = pd.read_parquet(REPO / "output" / "fd_history.parquet",
                         columns=["league", "date", "home_team", "away_team", "home_goals", "away_goals"])
    fd = fd.rename(columns={"home_goals": "hg", "away_goals": "ag"}); fd["src"] = "fd"
    r = pd.concat([af, fd], ignore_index=True)
    r["date"] = pd.to_datetime(r["date"]).dt.normalize()
    r = r[r["date"] >= "2026-04-01"].dropna(subset=["hg", "ag"])
    return r


class Resolver:
    def __init__(self, res: pd.DataFrame):
        self.by_lg = {lg: g for lg, g in res.groupby("league")}
        self.names = {lg: sorted(set(g.home_team) | set(g.away_team)) for lg, g in self.by_lg.items()}

    def find(self, league, date, home, away):
        g = self.by_lg.get(league)
        if g is None:
            return None, "NO_LEAGUE_IN_RESULTS"
        d = pd.Timestamp(date)
        out, tiers, fails = {}, set(), []
        # Each score source is matched on its OWN club spellings (af and fd spell clubs differently).
        for src, gs in g.groupby("src"):
            names = sorted(set(gs.home_team) | set(gs.away_team))
            h = resolve_result(home, names); a = resolve_result(away, names)
            c = gs[(gs.date - d).abs() <= pd.Timedelta(days=1)]
            m = c[(c.home_team == h) & (c.away_team == a)] if (h and a) else c.iloc[0:0]
            tier = "RESOLVER"
            if m.empty:
                # DATE-SCOPED FALLBACK: among this source's fixtures in the league within +-1 day, the
                # UNIQUE fixture whose home AND away names both share an identity token (>= 4 chars,
                # generic words removed) with ours. Two candidates -> refused (AMBIGUOUS).
                hits = [i for i, r in c.iterrows() if _sim(home, r.home_team) and _sim(away, r.away_team)]
                keys = {(c.at[i, "home_team"], c.at[i, "away_team"]) for i in hits}
                if len(keys) == 1:
                    m = c.loc[hits]; tier = "DATE_FUZZY"
                elif len(keys) > 1:
                    fails.append("AMBIGUOUS"); continue
                else:
                    fails.append("NO_SCORE_ROW" if (h and a) else "NAME_UNRESOLVED"); continue
            out[src] = (int(m.hg.iloc[0]), int(m.ag.iloc[0])); tiers.add(tier)
        if not out:
            return None, ("AMBIGUOUS" if "AMBIGUOUS" in fails else "NAME_UNRESOLVED" if "NAME_UNRESOLVED" in fails
                          else "NO_SCORE_ROW")
        return out, "RESOLVER" if tiers == {"RESOLVER"} else "DATE_FUZZY"


_GEN = {"fc", "cf", "sc", "afc", "cd", "sd", "ad", "ac", "ss", "as", "us", "sv", "vfl", "vfb", "fsv", "tsg", "club",
        "if", "ifk", "fk", "sk", "bk", "ca", "cp", "real", "sporting", "united", "city", "town", "athletic", "de",
        "the", "and", "deportivo", "atletico", "club", "ksv", "ff"}


def _toks(s: str) -> set:
    import unicodedata
    s = "".join(ch for ch in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(ch)).lower()
    return {t for t in re.split(r"[^a-z0-9]+", s) if len(t) >= 4 and t not in _GEN}


def _sim(a: str, b: str) -> bool:
    ta, tb = _toks(a), _toks(b)
    return bool(ta & tb) or any(x[:5] == y[:5] for x in ta for y in tb if len(x) >= 5 and len(y) >= 5)


def outcome(market: str, side: str, hg: int, ag: int) -> str:
    t = hg + ag
    m = market.lower()
    if m in ("ou25", "over25"):
        if side == "OVER":
            return "WIN" if t >= 3 else "LOSS"
        if side == "UNDER":
            return "WIN" if t <= 2 else "LOSS"
        return "NO_SIDE"
    if m == "btts":
        return "WIN" if (hg > 0 and ag > 0) else "LOSS"
    if m == "over15":
        return "WIN" if t >= 2 else "LOSS"
    if m == "over35":
        return "WIN" if t >= 4 else "LOSS"
    return "UNKNOWN_MARKET"


def settle(name: str, R: Resolver) -> pd.DataFrame:
    fs = pd.read_csv(DATA / f"first_sight_{name}.csv", dtype=str)
    rows = []
    for r in fs.itertuples(index=False):
        market = "ou25" if name == "ou25" else str(r.market)
        side = str(r.side) if name == "ou25" else "YES" if market == "btts" else "OVER"
        sc, why = R.find(r.league if isinstance(r.league, str) else str(getattr(r, "final_league", "")),
                         r.match_date, r.home_team, r.away_team)
        rec = {"ind_status": "OK" if sc else why, "ind_match_tier": why if sc else "", "ind_hg": np.nan, "ind_ag": np.nan, "ind_src": "", "src_disagree": False}
        if sc:
            src = "af" if "af" in sc else "fd"
            hg, ag = sc[src]
            rec.update(ind_hg=hg, ind_ag=ag, ind_src="+".join(sorted(sc)),
                       src_disagree=len(set(sc.values())) > 1)
            res = outcome(market, side, hg, ag)
            rec["ind_result"] = res
        else:
            rec["ind_result"] = "UNRESOLVED"
        rows.append(rec)
    out = pd.concat([fs.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    out["market_norm"] = "ou25" if name == "ou25" else out["market"].str.lower()
    for c in ("odds", "final_odds"):
        out[c + "_f"] = pd.to_numeric(out[c], errors="coerce")
    w = out["ind_result"] == "WIN"; l = out["ind_result"] == "LOSS"
    out["ind_pnl_fs"] = np.where(w, out["odds_f"] - 1, np.where(l, -1.0, np.nan))
    out["ind_pnl_final"] = np.where(w, out["final_odds_f"] - 1, np.where(l, -1.0, np.nan))
    out["ledger_result"] = out["final_result"]
    out["ledger_pnl"] = pd.to_numeric(out["final_pnl"], errors="coerce")
    out.to_csv(DATA / f"settled_{name}.csv", index=False)
    print(name, out["ind_status"].value_counts().to_dict())
    return out


if __name__ == "__main__":
    R = Resolver(load_results())
    for nm in (sys.argv[1:] or ["ou25", "side"]):
        settle(nm, R)
