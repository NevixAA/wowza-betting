"""P10 (config/bars_league_calib_protocol_p10.json) - RESEARCH / PROPOSAL ONLY. Statistics only; prices stay local.

    # v6 R2 TOP-3 tips per league-week on the holdout + shortfall causes, for ONE evidence + priced universe
    python scripts/bars_league_calib_p10.py top3 --priced <local>/priced --evidence <evidence json> \
        --fixtures <frame fixtures parquet> --work <local dir> --out <json>

    # calibration slope / league-level gap per league, champion vs variants, from harness predictions
    python scripts/bars_league_calib_p10.py slopes --predictions <run>/predictions.parquet[,...] \
        --candidates champion_recipe,champion_recipe+league+logitcal,champion_recipe+logitcal --out <json>

    # one Holm family over the fresh-holdout cells of several study runs (+ the champion's P9 cells)
    python scripts/bars_league_calib_p10.py holm --fresh <fresh_result.json>[,...] --out <json>

The v6 R2 replay is the same code as scripts/top3_tiers_v7.py `evaluate` (its v6 R2 block): tips clearing the
serving cell's bars (source refitted on development rows < 2026-04-01 only), TOP-3 by R2 per league-week
(src.research.top3_ranking.select, LIVE_RULE), one direction per fixture, never padded. LEAGUE_WIDE cells are
the league's own list and are not in the TOP-3 pool (as P9).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.bars_opt import oddsapi_prices as P  # noqa: E402
from src.bars_opt import serve as BS  # noqa: E402
from src.bars_opt import study as S  # noqa: E402
from src.research import top3_ranking as R  # noqa: E402
from src.research import top3_tiers as T  # noqa: E402

DEV_END = pd.Timestamp("2026-04-01")
H0, H1 = pd.Timestamp("2026-04-01"), pd.Timestamp("2026-11-08")
CAUSES = ("NO_FIXTURES", "ALL_CELLS_OFF", "NO_MODEL", "NO_PRICE", "NO_TIP_CLEARS_BAR", "RULE_CONFLICT")


def _v7():
    spec = importlib.util.spec_from_file_location("t7", ROOT / "scripts" / "top3_tiers_v7.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def v6_tips(b: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(every holdout row whose serving cell issues a tip, the TOP-3 R2 picks) - top3_tiers_v7 evaluate's v6 block."""
    v6 = []
    for c in b["cells"]:
        sc = c["serving"]
        if not sc:
            continue
        fam = b["fams"][(c["track"], c["mside"])]
        f = fam[(fam["league"] == c["league"]) & (fam["date"] >= H0) & (fam["date"] <= H1)]
        if f.empty:
            continue
        dev = fam[fam["date"] < DEV_END]
        cell = dict(sc, source_params=(S.source_params(sc["source"], dev) if sc["source"] != S.LEAGUE_WIDE_SOURCE else {}))
        if sc["source"] == "SHRINK":
            w = cell["source_params"].get("weights", {})
            if c["league"] not in w:
                continue
            cell["source_params"] = {"weights": {c["league"]: w[c["league"]]}}
        s = BS.score(f[["league", "p_side", "fair", "odds"]], cell)
        tip = s["tier"].astype(str) != ""
        if tip.any():
            v6.append(f[tip].assign(p_used=pd.to_numeric(s.loc[tip, "p_used"]), status=c["status"]))
    tips = pd.concat(v6, ignore_index=True) if v6 else pd.DataFrame()
    picks = R.select(tips, b["lam"].assign(n=0, roi=np.nan, se=np.nan, lcb=np.nan, eb=np.nan), R.LIVE_RULE) \
        if len(tips) else pd.DataFrame()
    return tips, picks


def shortfall(b: dict, tips: pd.DataFrame, picks: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    """One row per (track, league, ISO week) of the holdout: v6 R2 tips (0..3) and, when < 3, the CAUSE.
    Precedence: NO_FIXTURES > ALL_CELLS_OFF > the largest blocking slot count among NO_MODEL / NO_PRICE /
    NO_TIP_CLEARS_BAR; 3+ tip slots but < 3 picks -> RULE_CONFLICT (distinct fixture x market, one direction)."""
    weeks = sorted(set(T.gw(pd.Series(pd.date_range(H0, H1, freq="D")))))
    tipk = set(zip(tips["track"], tips["league"], tips["mside"], tips["row_id"])) if len(tips) else set()
    slots = []
    for c in b["cells"]:
        u = b["unis"][c["track"]]
        mk = S.MARKET_SIDES[c["mside"]][0]
        um = u[(u["market"] == mk) & (u["league"] == c["league"])]
        modeled = set(um.loc[um["p"].notna(), "row_id"])
        fam = b["fams"][(c["track"], c["mside"])]
        priced = set(fam.loc[fam["league"] == c["league"], "row_id"])
        f = fx[(fx["track"] == c["track"]) & (fx["league"] == c["league"])]
        if f.empty:
            continue
        rid = f["row_id"].to_numpy()
        slots.append(pd.DataFrame({"track": c["track"], "league": c["league"], "mside": c["mside"], "row_id": rid,
                                   "date": f["date"].to_numpy(), "model": f["row_id"].isin(modeled).to_numpy(),
                                   "price": f["row_id"].isin(priced).to_numpy(),
                                   "tip": [(c["track"], c["league"], c["mside"], r) in tipk for r in rid]}))
    sl = pd.concat(slots, ignore_index=True) if slots else pd.DataFrame(
        columns=["track", "league", "mside", "row_id", "date", "model", "price", "tip"])
    for k in ("model", "price", "tip"):
        sl[k] = sl[k].astype(bool)
    sl["gw"] = T.gw(sl["date"]) if len(sl) else []
    fxw = fx.assign(gw=T.gw(fx["date"]))
    pk = picks.groupby(["track", "league", "gw"]).size() if len(picks) else pd.Series(dtype=int)
    cells_on = {}
    for c in b["cells"]:
        cells_on[(c["track"], c["league"])] = cells_on.get((c["track"], c["league"]), 0) + 1
    rows = []
    for (tr, lg) in sorted(set(zip(fx["track"], fx["league"]))):
        fl, sll = fxw[(fxw["track"] == tr) & (fxw["league"] == lg)], sl[(sl["track"] == tr) & (sl["league"] == lg)]
        for w in weeks:
            f, s = fl[fl["gw"] == w], sll[sll["gw"] == w]
            n = int(pk.get((tr, lg, w), 0))
            r = {"track": tr, "league": lg, "gw": w, "fixtures": int(len(f)), "tips": n, "slots": int(len(s)),
                 "slots_no_model": int((~s["model"]).sum()), "slots_no_price": int((s["model"] & ~s["price"]).sum()),
                 "slots_no_tip": int((s["model"] & s["price"] & ~s["tip"]).sum()), "tip_slots": int(s["tip"].sum())}
            if n >= 3:
                r["cause"] = ""
            elif len(f) == 0:
                r["cause"] = "NO_FIXTURES"
            elif cells_on.get((tr, lg), 0) == 0:
                r["cause"] = "ALL_CELLS_OFF"
            elif r["tip_slots"] >= 3:
                r["cause"] = "RULE_CONFLICT"
            else:
                c = {"NO_MODEL": r["slots_no_model"], "NO_PRICE": r["slots_no_price"], "NO_TIP_CLEARS_BAR": r["slots_no_tip"]}
                r["cause"] = max(c, key=lambda k: (c[k], -list(c).index(k)))
            rows.append(r)
    return pd.DataFrame(rows)


def tier_stats(t: pd.DataFrame) -> dict:
    if t is None or t.empty:
        return {"bets": 0}
    s = S.score(t)
    return {"bets": s["bets"], "hit_rate": round(float(t["won"].mean()), 4), "roi": s["roi"], "ci95": s["ci95"],
            "mean_odds": round(float(t["odds"].mean()), 3), "clv": s["clv"], "clv_n": s["clv_n"]}


def cmd_top3(a) -> int:
    t7 = _v7()
    work = P.assert_outside_repo(a.work)
    work.mkdir(parents=True, exist_ok=True)
    ev = json.loads(Path(a.evidence).read_text(encoding="utf-8"))
    b = t7.build(Path(a.priced), ev, DEV_END)
    tips, picks = v6_tips(b)
    if len(picks):
        picks.to_parquet(work / "v6_picks.parquet", index=False)              # carries prices: LOCAL only
    fx = pd.read_parquet(a.fixtures)
    fx = fx[(fx["date"] >= H0) & (fx["date"] <= H1)]
    fx = fx[fx["league"].isin({c["league"] for c in b["cells"]} | set(b["unis"]["standard"]["league"])
                              | set(b["unis"]["new_format"]["league"]))]
    sf = shortfall(b, tips, picks, fx)
    sf.to_parquet(work / "v6_shortfall.parquet", index=False)

    def shares(x):
        return {str(k): int((x["tips"].clip(upper=3) == k).sum()) for k in (3, 2, 1, 0)} | {"league_weeks": int(len(x))}
    act = sf[sf["fixtures"] > 0]
    res = {"evidence": Path(a.evidence).name, "holdout": [str(H0.date()), str(H1.date())], "rule": R.LIVE_RULE,
           "cells_non_off_model": len(b["cells"]), "cells_with_lam": int(b["lam"]["lam"].notna().sum()),
           "lam": b["lam"].round(4).to_dict("records"),
           "tips_rows": int(len(tips)), "combined": tier_stats(picks), "per_league": {},
           "shortfall": {"all_league_weeks": shares(sf), "league_weeks_with_fixtures": shares(act),
                         "cause_counts_with_fixtures": act.loc[act["tips"] < 3, "cause"].value_counts().to_dict(),
                         "slots_with_fixtures": {k: int(act[k].sum()) for k in
                                                 ("slots", "slots_no_model", "slots_no_price", "slots_no_tip", "tip_slots")}}}
    for (tr, lg), g in sf.groupby(["track", "league"]):
        ga = g[g["fixtures"] > 0]
        pl = picks[(picks["track"] == tr) & (picks["league"] == lg)] if len(picks) else picks
        res["per_league"][f"{tr}|{lg}"] = {
            "weeks_with_fixtures": int(len(ga)), "tips_share": shares(ga),
            "tips_per_week": round(float(ga["tips"].mean()), 3) if len(ga) else None,
            "causes": ga.loc[ga["tips"] < 3, "cause"].value_counts().to_dict(), "picks": tier_stats(pl)}
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("combined", "shortfall", "cells_non_off_model", "cells_with_lam")},
                     indent=1, default=str))
    return 0


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _cal_slope(p: np.ndarray, y: np.ndarray) -> float | None:
    """Logistic regression y ~ a + b logit(p) (Newton, no penalty); b = 1 is calibrated, b < 1 over-confident."""
    if len(y) < 50 or y.min() == y.max():
        return None
    X = np.column_stack([np.ones(len(p)), _logit(p)])
    w = np.zeros(2)
    for _ in range(50):
        mu = 1 / (1 + np.exp(-X @ w))
        H = X.T @ (X * (mu * (1 - mu))[:, None])
        g = X.T @ (y - mu)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return None
        w += step
        if np.abs(step).max() < 1e-9:
            break
    return round(float(w[1]), 4)


def cmd_slopes(a) -> int:
    cands = a.candidates.split(",")
    parts = []
    for f in a.predictions.split(","):
        p = pd.read_parquet(f, columns=["stage", "track", "market", "candidate", "date", "league", "y", "p", "deployable"])
        parts.append(p[p["candidate"].isin(cands) & p["deployable"] & p["y"].notna() & p["p"].notna()])
    p = pd.concat(parts, ignore_index=True)
    out = {"per_league": [], "level_slope": []}
    for (st, tr, mk, cd), g in p.groupby(["stage", "track", "market", "candidate"]):
        lv = g.groupby("league").agg(n=("y", "size"), x=("p", "mean"), y=("y", "mean"))
        if len(lv) >= 2:
            w = lv["n"].to_numpy(float)
            x, y = lv["x"].to_numpy(float), lv["y"].to_numpy(float)
            xm, ym = np.average(x, weights=w), np.average(y, weights=w)
            vx = np.average((x - xm) ** 2, weights=w)
            out["level_slope"].append({"stage": st, "track": tr, "market": mk, "candidate": cd, "leagues": int(len(lv)),
                                       "level_slope": round(float(np.average((x - xm) * (y - ym), weights=w) / vx), 4) if vx > 0 else None,
                                       "mean_abs_gap": round(float(np.average(np.abs(x - y), weights=w)), 4)})
        for lg, gl in g.groupby("league"):
            out["per_league"].append({"stage": st, "track": tr, "market": mk, "candidate": cd, "league": lg,
                                      "n": int(len(gl)), "mean_p": round(float(gl["p"].mean()), 4),
                                      "observed": round(float(gl["y"].mean()), 4),
                                      "gap": round(float(gl["p"].mean() - gl["y"].mean()), 4),
                                      "cal_slope": _cal_slope(gl["p"].to_numpy(float), gl["y"].to_numpy(float))})
    Path(a.out).write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"{len(out['per_league'])} league rows, {len(out['level_slope'])} level-slope rows -> {a.out}")
    return 0


def cmd_holm(a) -> int:
    """Combined Holm (protocol 'multiplicity'): every fresh-holdout cell of every listed run is one hypothesis
    (a cell with no bet or no p-value still counts); PROVEN re-decided on the combined family."""
    from src import league_bars as LB
    cells = []
    for f in a.fresh.split(","):
        name, path = f.split("=", 1)
        for c in json.loads(Path(path).read_text(encoding="utf-8"))["cells"]:
            cells.append(dict(c, run=name))
    m = len(cells)
    tested = [i for i, c in enumerate(cells) if c["final"]["bets"] > 0 and c["final"].get("p_one_sided") is not None]
    ph = LB.holm(np.array([cells[i]["final"]["p_one_sided"] for i in tested], float), extra=m - len(tested))
    rows = []
    for i, c in enumerate(cells):
        f = c["final"]
        adj = ph[tested.index(i)] if i in tested else None
        lo = (f.get("ci95") or [None])[0]
        proven = bool(adj is not None and np.isfinite(adj) and adj < S.HOLM_ALPHA and f["bets"] >= S.MIN_FINAL_BETS
                      and lo is not None and lo > 0)
        rows.append({"run": c["run"], "track": c["track"], "mside": c["mside"], "league": c["league"],
                     "verdict_in_run": c.get("verdict"), "p_holm_in_run": c.get("p_holm"),
                     "p_holm_combined": None if adj is None or not np.isfinite(adj) else round(float(adj), 4),
                     "bets": f["bets"], "roi": f.get("roi"), "ci95": f.get("ci95"), "clv": f.get("clv"),
                     "proven_combined": proven})
    res = {"family": m, "tested_with_p": len(tested), "proven_combined": sum(r["proven_combined"] for r in rows),
           "cells": rows}
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "cells"}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("top3")
    for k in ("--priced", "--evidence", "--fixtures", "--out"):
        t.add_argument(k, required=True)
    t.add_argument("--work", required=True, type=Path)
    s = sub.add_parser("slopes")
    for k in ("--predictions", "--candidates", "--out"):
        s.add_argument(k, required=True)
    h = sub.add_parser("holm")
    h.add_argument("--fresh", required=True, help="name=path[,name=path...] of fresh_result.json files")
    h.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"top3": cmd_top3, "slopes": cmd_slopes, "holm": cmd_holm}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
