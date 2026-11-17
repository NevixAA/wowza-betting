"""Replay TOP-3 per league per gameweek on the clean retrain predictions - what it WOULD have picked.

    python scripts/top3_league_replay.py --track standard \
        --predictions <run>/predictions.parquet \
        --bars-dir output/threshold_proposals/league_bars/standard \
        --fd-raw <Pro>/experiments/ou_info_inputs_2026_10/fd_raw [--stage holdout] [--out <dir>]

RETROSPECTIVE EVIDENCE, NOT A BACKTEST TO TUNE ON. Nothing is chosen here: the bars come from the bars
proposals (src/league_bars, owner-approved later through src/threshold_apply) and the rule is the production
rule of src/top3_league.py, applied to the retrain's out-of-sample predictions:
  * a tip qualifies when its edge (p_model - de-vigged two-sided bet-time price) clears THAT league's own
    VALUABLE bar for THAT market. A league x market with no bars in the file gets no tip (NO_LEAGUE_BAR) -
    there is no global bar and no bar is ever made up to let this run;
  * per league-week (src/gameweek.league_week_id) the 3 best by value, across O/U 2.5 / BTTS / Over 1.5 /
    Over 3.5; never padded; never opposite goal directions on one fixture (src/direction_guard).

WHAT THE REPLAY CANNOT REPRODUCE (and says so in its output):
  * value ranking uses EV = p_side x odds - 1 (production ranks by market_edge_v2's ev_lower_bound, which
    needs the live engine's uncertainty penalty - not in the retrain predictions);
  * first sight / the week freeze: every fixture of the week is assumed known before the freeze;
  * the retrain rows carry a date, not a kickoff: the gameweek is taken at 12:00 UTC of the match date.
  * a pick is OUT_OF_SAMPLE only when its bars did not see its date: WALK_FORWARD_HOLDOUT bars on
    stage == holdout rows; IN_PERIOD_SPLIT bars only on/after their split's check_from date. Everything else
    is labelled IN_SAMPLE and kept out of the out-of-sample totals.

Prices: the same loaders as the bars (src/league_bars: football-data pre-match same-book pairs for Standard
O/U 2.5, the snapshot pairs for the side markets / New-Format). If src/league_bars is not in this checkout yet
(it lives on branch prep/league-bars until the merge), pass --bars-src <checkout with src/league_bars.py>.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.direction_guard import goal_direction  # noqa: E402
from src.gameweek import league_week_id  # noqa: E402

#: retrain market -> (bars key, direction-guard market)
MARKETS = {"over25": ("ou25", "ou25"), "btts": ("btts", "btts"), "over15": ("over15", "over15"),
           "over35": ("over35", "over35")}
TOP_N = 3


def load_bars(bars_dir: Path, track: str) -> dict:
    """{bars key: {league: params}} from the bars proposals directory of one track."""
    out: dict = {}
    for f in sorted(Path(bars_dir).rglob("best_params_*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        key = f.stem.replace("best_params_", "")
        if key == "standard":
            out.setdefault("ou25", {}).update(data)
        elif key == "side_markets":
            for m, v in data.items():
                if isinstance(v, dict):
                    out.setdefault(m, {}).update(v)
        else:
            m = key[len(track) + 1:] if key.startswith(f"{track}_") else key
            m = "ou25" if m in ("over25", "ou25") else m
            out.setdefault(m, {}).update(data)
    return out


def own_bars(bars: dict, market: str, league: str) -> dict | None:
    """The league's own three bars, or None. No global fallback."""
    lp = (bars.get(MARKETS[market][0]) or {}).get(league)
    if not isinstance(lp, dict):
        return None
    vals = [lp.get(k) for k in ("valuable_th", "marksman_th", "sniper_th")]
    try:
        v, m, s = (float(x) for x in vals)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite([v, m, s]).all() and v <= m <= s):
        return None
    return {"valuable": v, "marksman": m, "sniper": s, "status": str(lp.get("status") or ""),
            "evidence": str(lp.get("evidence") or ""), "split": lp.get("split") or {}}


def _tier(edge: float, b: dict) -> str:
    return "SNIPER" if edge >= b["sniper"] else "MARKSMAN" if edge >= b["marksman"] else \
        "VALUABLE" if edge >= b["valuable"] else "AVOID"


def _oos(b: dict, stage: str, date: pd.Timestamp) -> bool:
    if b["evidence"] == "IN_PERIOD_SPLIT":
        cf = (b["split"] or {}).get("check_from")
        return bool(cf) and date >= pd.Timestamp(cf)
    return stage == "holdout"


def select(bets: pd.DataFrame, bars: dict) -> tuple[pd.DataFrame, dict]:
    """bets: one row per priced prediction with market, league, date, stage, home, away, side ('yes'/'no'),
    p_side, odds_taken, edge, pnl. Returns (picks, {reason: count}). Pure."""
    excl: Counter = Counter()

    def bump(k, n=1):
        excl[k] += n
    rows = []
    for r in bets.to_dict("records"):
        b = own_bars(bars, r["market"], r["league"])
        if b is None:
            bump("NO_LEAGUE_BAR")
            continue
        t = _tier(float(r["edge"]), b)
        if t == "AVOID":
            bump("BELOW_LEAGUE_BAR")
            continue
        d = pd.Timestamp(r["date"])
        side = "YES" if r["side"] == "yes" else "NO"
        gside = {"over25": "OVER" if side == "YES" else "UNDER"}.get(r["market"], "YES" if r["market"] == "btts" else "OVER")
        rows.append({**r, "tier": t, "bar_status": b["status"], "bar_evidence": b["evidence"],
                     "bar_valuable": b["valuable"], "bar_marksman": b["marksman"], "bar_sniper": b["sniper"],
                     "ev": float(r["p_side"]) * float(r["odds_taken"]) - 1.0,
                     "direction": goal_direction(MARKETS[r["market"]][1], gside),
                     "league_week_id": league_week_id(r["league"], d.tz_localize("UTC") + pd.Timedelta(hours=12)),
                     "out_of_sample": _oos(b, r["stage"], d)})
    if not rows:
        return pd.DataFrame(), dict(excl)
    q = pd.DataFrame(rows).sort_values(["league_week_id", "ev", "edge", "date", "home"],
                                       ascending=[True, False, False, True, True], kind="mergesort")
    picks = []
    for _w, g in q.groupby("league_week_id", sort=True):
        dirs, k = {}, 0
        for r in g.to_dict("records"):
            if k >= TOP_N:
                bump("RANKED_BELOW_TOP3")
                continue
            fx = (str(r["date"])[:10], r["home"], r["away"])
            if r["direction"] and dirs.get(fx) and dirs[fx] != r["direction"]:
                bump("DIRECTION_CONFLICT")
                continue
            dirs.setdefault(fx, r["direction"])
            k += 1
            picks.append({**r, "top3_rank": k})
    return pd.DataFrame(picks), dict(excl)


def summarise(picks: pd.DataFrame) -> dict:
    def agg(d):
        if d.empty:
            return {"picks": 0}
        return {"picks": int(len(d)), "won": int((d["pnl"] > 0).sum()), "units": round(float(d["pnl"].sum()), 3),
                "roi": round(float(d["pnl"].mean()), 4), "avg_odds": round(float(d["odds_taken"].mean()), 3),
                "league_weeks": int(d["league_week_id"].nunique())}
    if picks.empty:
        return {"out_of_sample": {"picks": 0}, "in_sample": {"picks": 0}}
    oos = picks[picks["out_of_sample"]]
    return {"out_of_sample": agg(oos), "in_sample": agg(picks[~picks["out_of_sample"]]),
            "oos_by_market": {m: agg(g) for m, g in oos.groupby("market")},
            "oos_by_league": {lg: agg(g) for lg, g in oos.groupby("league")},
            "oos_by_tier": {t: agg(g) for t, g in oos.groupby("tier")},
            "oos_by_bar_status": {s: agg(g) for s, g in oos.groupby("bar_status")},
            "oos_picks_per_league_week": oos.groupby("league_week_id").size().value_counts().sort_index().to_dict()}


def _league_bars_modules(bars_src: str | None):
    """(src.league_bars, scripts/league_bars.py) - from this checkout, or from --bars-src."""
    import src
    base = ROOT
    if not (ROOT / "src" / "league_bars.py").exists():
        if not bars_src:
            raise SystemExit("src/league_bars.py is not in this checkout (branch prep/league-bars): pass --bars-src")
        base = Path(bars_src)
        src.__path__.append(str(base / "src"))         # the bars module only; everything else stays ours
    from src import league_bars as LB
    spec = importlib.util.spec_from_file_location("league_bars_script", base / "scripts" / "league_bars.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return LB, script


def priced_bets(a) -> pd.DataFrame:
    LB, bars_script = _league_bars_modules(a.bars_src)
    pred = pd.read_parquet(a.predictions)
    pred = pred[(pred["track"] == a.track) & (pred["candidate"] == a.candidate) & pred["deployable"]
                & pred["market"].isin(list(MARKETS)) & pred["stage"].isin(a.stages)]
    aliases = bars_script.load_aliases(None)
    parts = []
    for market, pm in pred.groupby("market"):
        if market in LB.TWO_SIDED_BET and a.track == "standard":
            if not a.fd_raw:
                raise SystemExit("--fd-raw is needed for Standard O/U 2.5 bet-time prices")
            prices = LB.fd_ou25_pairs(a.fd_raw)
        elif market in LB.SNAPSHOT_KEYS:
            prices, _note = bars_script.snapshot_prices(a.track, market)
        else:
            continue
        priced, _stats = LB.attach_prices(pm, prices, aliases)
        b = LB.bets_frame(priced, market)
        b["p_side"] = np.where(b["side"] == "yes", b["p"], 1.0 - b["p"])
        parts.append(b.assign(market=market))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--track", required=True, choices=["standard", "new_format"])
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--bars-dir", required=True)
    ap.add_argument("--candidate", default="champion_recipe")
    ap.add_argument("--stages", nargs="+", default=["walk_forward", "holdout"])
    ap.add_argument("--fd-raw")
    ap.add_argument("--bars-src", help="checkout holding src/league_bars.py when this one does not")
    ap.add_argument("--out", help="write picks CSV + summary JSON here")
    a = ap.parse_args(argv)
    bars = load_bars(Path(a.bars_dir), a.track)
    if not bars:
        raise SystemExit(f"no bars proposals under {a.bars_dir} - nothing is replayed without real bars")
    bets = priced_bets(a)
    parts = bets["row_id"].str.split("|", expand=True)
    bets["home"], bets["away"] = parts[2], parts[3]
    picks, excl = select(bets, bars)
    out = {"track": a.track, "predictions": a.predictions, "bars_dir": a.bars_dir, "priced_bets": int(len(bets)),
           "excluded": excl, "summary": summarise(picks),
           "limits": ["ranking by EV point, not ev_lower_bound", "no first-sight/freeze reproduction",
                      "gameweek from match date at 12:00 UTC"]}
    print(json.dumps(out, indent=1, default=str))
    if a.out:
        od = Path(a.out)
        od.mkdir(parents=True, exist_ok=True)
        picks.to_csv(od / f"top3_replay_{a.track}.csv", index=False)
        (od / f"top3_replay_{a.track}.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
