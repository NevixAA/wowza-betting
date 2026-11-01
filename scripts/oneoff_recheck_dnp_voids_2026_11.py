"""ONE-OFF (2026-11-01): re-check every props row voided 'DNP' against API-Football's stats.

    python scripts/oneoff_recheck_dnp_voids_2026_11.py [--dry-run] [--limit-fixtures N]

Follow-up to a8a0e6ec / 34267dd2. The settler matched players by NAME, so abbreviated stats names
('A. Laurienté') turned real appearances into 'did not play' voids; 4,334 were regraded from
player_history. The rest (players absent from player_history) can only be checked against the
fixture's own stats, which needs APIFOOTBALL_KEY - so this runs in CI (oneoff_dnp_recheck.yml).

Per row (same rules as update_results.update_player_results after a8a0e6ec):
  * fixture stats empty           -> VOID, notes NO_STATS (honest label, not 'did not play')
  * player found (id, exact name, initial+surname unique), minutes > 0 -> WIN/LOSS from his stats,
    P/L only on priced rows, notes RECHECKED_FROM_DNP_2026-11-01
  * player found with 0 minutes    -> stays VOID, notes 'DNP (0 min)'
  * player not in the fixture's stats -> stays VOID 'DNP' (genuinely did not play)
Updates are applied by row identity onto the LATEST ledger at the end, so concurrent workflows'
writes are not lost.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import update_results as U  # noqa: E402

KEY = ["match_date", "home_team", "away_team", "player_name", "market"]
NOTE = "RECHECKED_FROM_DNP_2026-11-01"


def recheck(led: pd.DataFrame, limit_fixtures: int | None = None) -> dict:
    from player_model.api_football import find_fixture_id, get_fixture_player_stats
    from player_model.config import PROP_LEAGUES
    from src.season_resolver import resolve_api_season, is_known
    cand = led[(led["result"] == "VOID") & (led["notes"] == "DNP")]
    fixtures = list(cand.groupby(["league", "match_date", "home_team", "away_team"]).groups.items())
    if limit_fixtures:
        fixtures = fixtures[:limit_fixtures]
    upd, counts = {}, {"WIN": 0, "LOSS": 0, "NO_STATS": 0, "DNP (0 min)": 0, "DNP": 0, "no_fixture": 0}
    for n, ((league, md, home, away), idx) in enumerate(fixtures, 1):
        lid = PROP_LEAGUES.get(league)
        season = resolve_api_season(league, str(md)[:10]) if lid else None
        fid = find_fixture_id(lid, season, str(md)[:10], home, away) if (lid and is_known(season)) else None
        if not fid:
            counts["no_fixture"] += len(idx)
            continue
        raw = get_fixture_player_stats(fid)
        smap = {U._norm(p["player_name"]): p for p in raw if p.get("player_name")}
        for i in idx:
            r = led.loc[i]
            k = tuple(str(r[c]) for c in KEY)
            if not smap:
                upd[k] = {"result": "VOID", "pnl": "0.0", "notes": "NO_STATS"}
                counts["NO_STATS"] += 1
                continue
            s = None
            pid = str(r.get("player_id", "") or "").split(".")[0]
            if pid:
                s = next((v for v in smap.values() if str(v.get("player_id")) == pid), None)
            s = s or smap.get(U._norm(r["player_name"])) or U._initial_surname_match(U._norm(r["player_name"]), smap)
            if s is None:
                counts["DNP"] += 1
                continue
            if int(s.get("minutes_played") or 0) == 0:
                upd[k] = {"result": "VOID", "pnl": "0.0", "notes": "DNP (0 min)"}
                counts["DNP (0 min)"] += 1
                continue
            res = U._resolve_player_market(r["market"], s)
            if not res:
                continue
            try:
                odds = float(r.get("market_odds") or "nan")
            except ValueError:
                odds = float("nan")
            upd[k] = {"result": res, "notes": NOTE,
                      "pnl": (str(round(odds - 1.0, 4) if res == "WIN" else -1.0) if odds > 1.0 else "")}
            counts[res] += 1
        if n % 100 == 0:
            print(f"  {n}/{len(fixtures)} fixtures: {counts}", flush=True)
    return {"updates": upd, "counts": counts, "fixtures": len(fixtures)}


def apply(led: pd.DataFrame, upd: dict) -> int:
    k = led[KEY].astype(str).agg(tuple, axis=1)
    hit = 0
    for i, kk in zip(led.index, k):
        u = upd.get(kk)
        if u and led.at[i, "result"] == "VOID" and led.at[i, "notes"] == "DNP":
            for c, v in u.items():
                led.at[i, c] = v
            hit += 1
    return hit


def main(argv) -> int:
    import json
    if "--apply-journal" in argv:                    # commit step: apply onto the LATEST ledger
        upd = {tuple(json.loads(k)): v for k, v in
               json.loads(Path(argv[argv.index("--apply-journal") + 1]).read_text(encoding="utf-8")).items()}
        fresh = pd.read_csv(U.PLAYER_LEDGER_FILE, dtype=str, keep_default_na=False)
        n = apply(fresh, upd)
        fresh.to_csv(U.PLAYER_LEDGER_FILE, index=False)
        print(f"applied {n} update(s) to {U.PLAYER_LEDGER_FILE.name}")
        return 0
    dry = "--dry-run" in argv
    lim = int(argv[argv.index("--limit-fixtures") + 1]) if "--limit-fixtures" in argv else None
    led = pd.read_csv(U.PLAYER_LEDGER_FILE, dtype=str, keep_default_na=False)
    out = recheck(led, lim)
    print(f"checked {out['fixtures']} fixture(s): {out['counts']}")
    if "--journal" in argv:
        p = Path(argv[argv.index("--journal") + 1])
        p.write_text(json.dumps({json.dumps(list(k)): v for k, v in out["updates"].items()}), encoding="utf-8")
        print(f"journal: {len(out['updates'])} update(s) -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
