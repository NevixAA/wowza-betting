"""ONE-OFF (2026-11-01): re-check every props row voided 'DNP' against API-Football's stats.

    python scripts/oneoff_recheck_dnp_voids_2026_11.py [--dry-run] [--limit-fixtures N]

Follow-up to a8a0e6ec / 34267dd2. The settler matched players by NAME, so abbreviated stats names
('A. Laurienté') turned real appearances into 'did not play' voids; 4,334 were regraded from
player_history. The rest (players absent from player_history) can only be checked against the
fixture's own stats, which needs APIFOOTBALL_KEY - so this runs in CI (oneoff_dnp_recheck.yml).

Per row (the settler's own helpers in update_results: _find_prop_fixture, _fixture_players,
_appearance, _fill_null_stats, _resolve_player_market):
  * fixture stats empty           -> VOID, notes NO_STATS (honest label, not 'did not play')
  * player found BY player_id ONLY, played -> WIN/LOSS from his stats, P/L only on priced rows,
    notes RECHECKED_FROM_DNP_2026-11-01. Names are never matched (Data Truth: the initial+surname
    name match was removed from update_results; a ledger row without a player_id is not checked).
  * the match sheet proves he did not play -> stays VOID with the settler's note
    ('DNP (0 min)', 'DNP (not in matchday squad)', 'DNP (unused substitute)')
  * no player_id / appearance or stat not published -> stays VOID, notes DNP_NO_PLAYER_ID /
    DNP_UNVERIFIED (labels saying it was not checked, so the batch pool still drains)
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
    from src.league_registry import prop_leagues
    from src.season_resolver import resolve_api_season, is_known
    prop_ids = prop_leagues()
    has_pid, has_odds = "player_id" in led.columns, "market_odds" in led.columns
    cand = led[(led["result"] == "VOID") & (led["notes"] == "DNP")]
    fixtures = list(cand.groupby(["league", "match_date", "home_team", "away_team"]).groups.items())
    if limit_fixtures:
        fixtures = fixtures[:limit_fixtures]
    upd = {}
    counts = {"WIN": 0, "LOSS": 0, "NO_STATS": 0, "DNP": 0, "no_fixture": 0, "no_player_id": 0,
              "unverified": 0}
    void = {"result": "VOID", "pnl": "0.0"}
    for n, ((league, md, home, away), idx) in enumerate(fixtures, 1):
        day = str(md)[:10]
        lg = U._canon_league(league)
        lid = prop_ids.get(lg) if lg else None
        season = resolve_api_season(lg, day) if lid else None
        fx = U._find_prop_fixture(lid, season, day, home, away)[0] if (lid and is_known(season)) else None
        fid = U._as_id((fx.get("fixture") or {}).get("id")) if fx else None
        if fid is None:
            for i in idx:   # no fixture id at all: record it, so batches move on
                upd[tuple(str(led.loc[i, c]) for c in KEY)] = {**void, "notes": "DNP_NO_FIXTURE"}
            counts["no_fixture"] += len(idx)
            continue
        players = U._fixture_players(fid)
        for i in idx:
            r = led.loc[i]
            k = tuple(str(r[c]) for c in KEY)
            if not players["by_id"]:
                upd[k] = {**void, "notes": "NO_STATS"}
                counts["NO_STATS"] += 1
                continue
            pid = U._as_id(r["player_id"]) if has_pid else None
            if pid is None:
                upd[k] = {**void, "notes": "DNP_NO_PLAYER_ID"}
                counts["no_player_id"] += 1
                continue
            rec = players["by_id"].get(pid)
            played, note = U._appearance(rec, players, fid, pid)
            if played == "DNP":
                upd[k] = {**void, "notes": note}
                counts["DNP"] += 1
                continue
            res = (U._resolve_player_market(r["market"], U._fill_null_stats(rec, fid, fx, players))
                   if played == "PLAYED" else "")
            if not res:
                upd[k] = {**void, "notes": "DNP_UNVERIFIED"}
                counts["unverified"] += 1
                continue
            odds = float(pd.to_numeric(r["market_odds"], errors="coerce")) if has_odds else float("nan")
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
