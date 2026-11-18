"""P11 - per-league bars + TOP-3 tiers (T1 best value / T2 safest / T3 risky value) on the CLEAN models.
RESEARCH / PROPOSAL ONLY (protocol config/p11_bars_tiers_protocol.json). Nothing is installed or approved.

    # 1) priced universes: own bet-time (backfill stripped) + OddsAPI BET-role history (BT, BTU, OUU)
    python scripts/p11_bars_tiers.py prepare --universe <u> --store <odds-history checkout> --aliases <map> --work <W>
    # 2) bars: selection on the fit window (< 2026-09-01), FINAL once on validation (2026-09-01..2026-11-08)
    python scripts/p11_bars_tiers.py bars --work <W>
    # 3) tiers: per-league tier parameters on the fit window's inner out-of-sample bets
    python scripts/p11_bars_tiers.py tiers-fit --work <W>
    # 4) validation (once): tiers on the validation bets, shortfall, v6 / bade2c0b comparisons
    python scripts/p11_bars_tiers.py validate --work <W> --universe <u>
    # 5) the serving proposal (bars cells, month 2026-11) + the tier config with frozen serving calibration
    python scripts/p11_bars_tiers.py propose --work <W>

Prices are PRIVATE (OddsAPI: no redistribution; this repo is PUBLIC): priced universes, pools and picks stay in
--work (outside the repo). Only counts and statistics are written to output/research/p11_bars_tiers/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.bars_opt import oddsapi_prices as P  # noqa: E402
from src.bars_opt import serve as BS  # noqa: E402
from src.bars_opt import study as S  # noqa: E402
from src.research import top3_ranking as R  # noqa: E402

PROTOCOL = ROOT / "config" / "p11_bars_tiers_protocol.json"
OUT = ROOT / "output" / "research" / "p11_bars_tiers"
TIER_CONFIG = ROOT / "config" / "top3_tiers_p11.json"
NAMESPACE = "bars_p11"
MONTH = "2026-11"
TAG = "p11"
BET_TIERS = ["BT", "BTU", "OUU"]
CLOSE_TIERS = ["B1"]
FIT_END = S.SELECT_END                      # 2026-09-01 (fit = date < FIT_END)
VAL_START, VAL_END = S.FINAL_START, S.FINAL_END
SERVING_LAM_BEFORE = pd.Timestamp("2026-11-09")
TIERS = ("T1_BEST_VALUE", "T2_SAFEST", "T3_RISKY_VALUE")
GATES = ("CAL", "BAR")
T3_FLOORS = ("REL", 2.25, 2.5, 2.75, 3.0)
MIN_FIT = S.MIN_OWN_BETS                    # 40
TRACKS = ("standard", "new_format")


def sha_lf(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def gw(d: pd.Series) -> pd.Series:
    return pd.to_datetime(d).dt.strftime("%G-W%V")


def direction(mside: str) -> str:
    return "OVER" if mside in S.OVER_DIRECTION else "UNDER"


def market_of(mside: str) -> str:
    return S.MARKET_SIDES[mside][0]


def stats(t: pd.DataFrame) -> dict:
    if t is None or t.empty:
        return {"bets": 0}
    s = S.score(t)
    return {"bets": s["bets"], "hit": round(float(t["won"].mean()), 4), "roi": s["roi"], "ci95": s["ci95"],
            "p_one_sided": s["p_one_sided"], "mean_odds": round(float(t["odds"].mean()), 3), "clv": s["clv"],
            "clv_n": s["clv_n"]}


def lcb(pnl: np.ndarray, dates: pd.Series) -> float:
    if len(pnl) == 0:
        return -np.inf
    se = S.max_se(pnl, pd.to_datetime(dates).dt.strftime("%Y-%m-%d").to_numpy())
    return float(pnl.mean() - (se if np.isfinite(se) else 1.0))


# ── 1) prepare ───────────────────────────────────────────────────────────────────────────────

def cmd_prepare(a) -> int:
    work = P.assert_outside_repo(a.work)
    (work / "priced").mkdir(parents=True, exist_ok=True)
    aliases = json.loads(Path(a.aliases).read_text(encoding="utf-8"))
    st = {"bet_tiers": BET_TIERS, "close_tiers": CLOSE_TIERS, "own_backfill_excluded": True}
    for t in TRACKS:
        u = pd.read_parquet(Path(a.universe) / f"universe_{t}.parquet")
        u, sc = P.strip_unproven_own(u, t, aliases)
        r = P.price_track(u, Path(a.store), BET_TIERS, CLOSE_TIERS)
        r["priced"].to_parquet(work / "priced" / f"universe_{t}.parquet", index=False)
        st[t] = {"own_strip": sc, "price_source": r["priced"]["price_source"].value_counts().to_dict(),
                 "close_source": r["priced"]["close_source"].value_counts().to_dict(),
                 "integrity": r["stats"].get("integrity"), "pairs": r["stats"].get("pairs")}
    ok, why = P.tiers_complete(Path(a.store), BET_TIERS)
    st["tiers_complete"] = {"complete": ok, "reason": why}
    (work / "prepare_stats.json").write_text(json.dumps(st, indent=1, default=str), encoding="utf-8")
    cov = coverage(work)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "coverage.json").write_text(json.dumps({"price_tiers": st, "coverage": cov}, indent=1, default=str) + "\n",
                                       encoding="utf-8")
    print(json.dumps({t: {k: st[t][k] for k in ("price_source", "integrity")} for t in TRACKS}, indent=1, default=str))
    return 0


def coverage(work: Path) -> list[dict]:
    """Counts only: priced rows per track x market x league x window (fit / validation) and source."""
    rows = []
    for t in TRACKS:
        u = pd.read_parquet(work / "priced" / f"universe_{t}.parquet")
        u["date"] = pd.to_datetime(u["date"])
        u["win"] = np.where(u["date"] < FIT_END, "fit", np.where((u["date"] >= VAL_START) & (u["date"] <= VAL_END),
                                                                  "validation", "other"))
        u = u[(u["win"] != "other") & u["y"].notna()]
        priced = u["yes"].notna() & u["no"].notna()
        g = u.assign(priced=priced, own=priced & (u["price_source"] == P.SRC_OWN),
                     hist=priced & (u["price_source"] == P.SRC_HIST))
        for (mk, lg, w), x in g.groupby(["market", "league", "win"]):
            rows.append({"track": t, "market": mk, "league": lg, "window": w, "rows": int(len(x)),
                         "priced": int(x["priced"].sum()), "own": int(x["own"].sum()), "history": int(x["hist"].sum())})
    return rows


# ── 2) bars ──────────────────────────────────────────────────────────────────────────────────

def cmd_bars(a) -> int:
    work = P.assert_outside_repo(a.work)
    sd = work / "study"
    r1 = S.run_select(work / "priced", sd, workers=a.workers)
    r2 = S.run_final(work / "priced", sd, workers=a.workers)
    OUT.mkdir(parents=True, exist_ok=True)
    cells = r2["cells"]
    table = [{k: c.get(k) for k in ("track", "mside", "league", "config", "mode", "verdict", "reason",
                                    "valuable_th", "marksman_th", "sniper_th", "wf_bets", "wf_roi", "p_holm",
                                    "family_size")} | {"validation": c.get("final")} for c in cells]
    (OUT / "bars_study.json").write_text(json.dumps({"select": r1, "final_summary": r2["summary"], "cells": table},
                                                    indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"families": r1["families"], "verdicts": dict(Counter(c["verdict"] for c in cells))},
                     indent=1, default=str))
    return 0


def load_choices(work: Path) -> dict:
    raw = json.loads((work / "study" / "choices.json").read_text(encoding="utf-8"))
    return {(c["track"], c["mside"]): c for c in raw}


# ── calibration + pools ──────────────────────────────────────────────────────────────────────

def lam_from(rows: pd.DataFrame) -> pd.DataFrame:
    """Per-cell lam / lam_se (R2 of v6) from scored priced rows (track, league, mside, fair, p_used, won)."""
    r = rows[np.isfinite(rows["p_used"].to_numpy(float))]
    return R._lams(r[R.CELL + ["fair", "p_used", "won"]]) if len(r) else R._lams(None)


def add_cal(b: pd.DataFrame, lam: pd.DataFrame) -> pd.DataFrame:
    b = b.drop(columns=[c for c in ("lam", "lam_se", "lam_rows") if c in b.columns]).merge(
        lam[R.CELL + ["lam", "lam_se"]], on=R.CELL, how="left")
    d = b["p_used"] - b["fair"]
    b["p_cal"] = b["fair"] + b["lam"] * d
    b["cal_ev"] = b["p_cal"] * b["odds"] - 1.0
    b["ev_lo"] = (b["fair"] + (b["lam"] - b["lam_se"].fillna(0.0)) * d) * b["odds"] - 1.0
    b["ev_raw"] = b["p_used"] * b["odds"] - 1.0
    return b


def _model_bets(t: pd.DataFrame, e: pd.DataFrame, choice: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(bets on e of the MODEL-mode leagues of the frozen choice, refitted on t; t scored by the same source)."""
    chosen = S.parse_cfg(choice["chosen"])
    if chosen[0] == S.LEAGUE_WIDE_SOURCE:
        return pd.DataFrame(), pd.DataFrame()
    mine = [lg for lg, b in choice["bars"].items() if b.get("mode") == S.MODE_MODEL and "off" not in b]
    if not mine:
        return pd.DataFrame(), pd.DataFrame()
    bars, scorer = S.fit_cfg(t, chosen)
    use = {lg: bars[lg] for lg in mine if lg in bars and "off" not in bars[lg]}
    ts = scorer(t[t["league"].isin(mine)])
    if not use or e.empty:
        return pd.DataFrame(), ts
    ee = scorer(e[e["league"].isin(list(use))])
    return (S.apply_bars(ee, use, S.MODE_MODEL) if len(ee) else pd.DataFrame()), ts


def fit_pool(work: Path) -> pd.DataFrame:
    """Inner out-of-sample bets of the fit window (per inner block: bars + source + lam refitted on rows before
    the block), MODEL-mode leagues of each family's frozen choice, with calibration columns."""
    ch = load_choices(work)
    out = []
    for f, track, mside in S.load_families(work / "priced"):
        c = ch.get((track, mside))
        if not c or not c.get("chosen"):
            continue
        sel = S.select_rows(f, FIT_END)
        starts = S.inner_blocks(sel["date"])
        for k in range(len(starts) - 1):
            t = sel[sel["date"] < starts[k]]
            e = sel[(sel["date"] >= starts[k]) & (sel["date"] < starts[k + 1])]
            if t.empty or e.empty:
                continue
            bets, ts = _model_bets(t, e, c)
            if bets.empty:
                continue
            out.append(add_cal(bets, lam_from(ts)).assign(fold=k))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def val_pool(work: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validation bets of the frozen bars (refitted on the fit window = run_final's bets, MODEL mode) with lam
    from fit-window rows only; also the lam table."""
    ch = load_choices(work)
    out, lams = [], []
    for f, track, mside in S.load_families(work / "priced"):
        c = ch.get((track, mside))
        if not c or not c.get("chosen"):
            continue
        sel = S.select_rows(f, FIT_END)
        fin = f[(f["date"] >= VAL_START) & (f["date"] <= VAL_END)]
        bets, ts = _model_bets(sel, fin, c)
        lam = lam_from(ts)
        lams.append(lam)
        if len(bets):
            out.append(add_cal(bets, lam))
    lam = pd.concat(lams, ignore_index=True) if lams else R._lams(None)
    return (pd.concat(out, ignore_index=True) if out else pd.DataFrame()), lam


# ── tier picking ─────────────────────────────────────────────────────────────────────────────

def pick(pool: pd.DataFrame, params: dict) -> pd.DataFrame:
    """T1 / T2 / T3 per (track, league, ISO week). params: {"track|league": {"gate", "t3_floor"}}; a league
    without params uses gate CAL and no T3 (fail closed - reported). Never padded."""
    if pool is None or pool.empty:
        return pd.DataFrame()
    p = pool.copy()
    p["gw"] = gw(p["date"])
    p["market"] = p["mside"].map(market_of)
    p["direction"] = p["mside"].map(direction)
    out = []
    for (tr, lg, w), g in p.groupby(["track", "league", "gw"], sort=True):
        prm = params.get(f"{tr}|{lg}") or {"gate": "CAL", "t3_floor": None}
        ok = g["cal_ev"].notna() & np.isfinite(g["cal_ev"].to_numpy(float))
        if prm["gate"] == "CAL":
            ok &= g["cal_ev"] > 0
        else:
            ok &= g["ev_raw"] > 0
        g = g[ok].sort_values(["date", "row_id", "mside"], kind="mergesort")
        if g.empty:
            continue
        used, dirs, taken = set(), {}, []

        def free(r):
            return (r.row_id, r.market) not in used and dirs.get(r.row_id, r.direction) == r.direction

        def first(frame, tier):
            for r in frame.itertuples(index=False):
                if free(r):
                    used.add((r.row_id, r.market))
                    dirs[r.row_id] = r.direction
                    taken.append(r._asdict() | {"tier": tier, "gate": prm["gate"], "t3_floor": prm["t3_floor"]})
                    return

        first(g.sort_values("cal_ev", ascending=False, kind="mergesort"), TIERS[0])
        first(g.sort_values("p_cal", ascending=False, kind="mergesort"), TIERS[1])
        fl = prm.get("t3_floor")
        if fl == "REL":
            first(g.sort_values(["odds", "ev_lo"], ascending=[False, False], kind="mergesort"), TIERS[2])
        elif fl is not None:
            first(g[g["odds"] >= float(fl)].sort_values("ev_lo", ascending=False, kind="mergesort"), TIERS[2])
        out += taken
    return pd.DataFrame(out)


def fit_params(pool: pd.DataFrame) -> dict:
    res = {}
    for (tr, lg), g in pool.groupby(["track", "league"]):
        key = f"{tr}|{lg}"
        grid, gate_score = [], {}
        for gate in GATES:
            t = pick(g, {key: {"gate": gate, "t3_floor": "REL"}})
            n = len(t)
            gate_score[gate] = lcb(t["pnl"].to_numpy(float), t["date"]) if n >= MIN_FIT else -np.inf
            grid.append({"gate": gate, "t3_floor": "REL", "fit_picks": n,
                         "fit_roi": round(float(t["pnl"].mean()), 4) if n else None})
        gate = max(GATES, key=lambda x: (gate_score[x], x == "CAL")) if np.isfinite(max(gate_score.values())) else "CAL"
        best, best_s = None, -np.inf
        for fl in T3_FLOORS:
            t = pick(g, {key: {"gate": gate, "t3_floor": fl}})
            t3 = t[t["tier"] == TIERS[2]] if len(t) else t
            n = len(t3)
            s = lcb(t3["pnl"].to_numpy(float), t3["date"]) if n >= MIN_FIT else -np.inf
            grid.append({"gate": gate, "t3_floor": fl, "fit_t3_picks": n,
                         "fit_t3_roi": round(float(t3["pnl"].mean()), 4) if n else None})
            if s > best_s + 1e-12:
                best, best_s = fl, s
        res[key] = {"gate": gate, "t3_floor": best if best is not None else "REL",
                    "gate_basis": "FIT" if np.isfinite(max(gate_score.values())) else f"UNFITTED (< {MIN_FIT} fit picks): CAL",
                    "t3_basis": "FIT" if best is not None else f"UNFITTED (no floor with >= {MIN_FIT} fit T3 picks): REL",
                    "grid": grid}
    return res


def tier_table(picks: pd.DataFrame) -> dict:
    out = {"ALL": stats(picks)}
    for t in TIERS:
        out[t] = stats(picks[picks["tier"] == t]) if len(picks) else {"bets": 0}
    return out


def cmd_tiers_fit(a) -> int:
    work = P.assert_outside_repo(a.work)
    pool = fit_pool(work)
    pool.to_parquet(work / "fit_pool.parquet", index=False)                     # prices: LOCAL only
    prm = fit_params(pool)
    picks = pick(pool, {k: {"gate": v["gate"], "t3_floor": v["t3_floor"]} for k, v in prm.items()})
    picks.to_parquet(work / "fit_picks.parquet", index=False)
    per_league = {}
    for (tr, lg), g in picks.groupby(["track", "league"]):
        per_league[f"{tr}|{lg}"] = tier_table(g)
    apr = picks[picks["date"] >= S.FINAL_START - pd.DateOffset(months=5)] if len(picks) else picks
    body = {"protocol": json.loads(PROTOCOL.read_text(encoding="utf-8"))["protocol"],
            "protocol_sha256_lf": sha_lf(PROTOCOL), "script_sha256_lf": sha_lf(Path(__file__)),
            "fit_pool_bets": int(len(pool)), "fit_pool_from": str(pool["date"].min().date()) if len(pool) else None,
            "fit_pool_to": str(pool["date"].max().date()) if len(pool) else None,
            "params": prm, "fit_all": tier_table(picks), "fit_2026_04_08": tier_table(apr), "fit_per_league": per_league}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tier_fit.json").write_text(json.dumps(body, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"leagues": len(prm), "gates": dict(Counter(v["gate"] for v in prm.values())),
                      "t3": dict(Counter(str(v["t3_floor"]) for v in prm.values())),
                      "fit_all": body["fit_all"], "fit_2026_04_08": body["fit_2026_04_08"]}, indent=1, default=str))
    return 0


# ── 4) validation ────────────────────────────────────────────────────────────────────────────

def v6_select(bets: pd.DataFrame, lam: pd.DataFrame) -> pd.DataFrame:
    if bets is None or bets.empty:
        return pd.DataFrame()
    st = lam.assign(n=0, roi=np.nan, se=np.nan, lcb=np.nan, eb=np.nan)
    b = bets.drop(columns=[c for c in ("lam", "lam_se", "lam_rows", "p_cal", "cal_ev", "ev_lo") if c in bets.columns])
    s = R.select(b, st, R.LIVE_RULE)
    return add_cal(s.drop(columns=["key", "has_stat", "no_stat", "rank_basis", "lam"], errors="ignore"), lam) if len(s) else s


def approved_v6(work: Path, approved_evidence: Path) -> pd.DataFrame:
    """v6 R2 on the approved cells (as approved: their own fitted source parameters) on the validation rows."""
    ev = json.loads(Path(approved_evidence).read_text(encoding="utf-8"))
    frames = {(t, m): f for f, t, m in S.load_families(work / "priced")}
    bets, fitrows = [], []
    for fam in ev["families"]:
        for c in fam["cells"]:
            if c.get("mode") != S.MODE_MODEL or c.get("source") == S.LEAGUE_WIDE_SOURCE:
                continue
            f = frames.get((c["track"], c["mside"]))
            if f is None:
                continue
            g = f[f["league"] == c["league"]]
            for part, keep in ((g[g["date"] < FIT_END], fitrows), (g[(g["date"] >= VAL_START) & (g["date"] <= VAL_END)], bets)):
                if part.empty:
                    continue
                s = BS.score(part[["league", "p_side", "fair", "odds"]], c)
                pu = pd.to_numeric(s["p_used"], errors="coerce")
                if keep is fitrows:
                    keep.append(part.assign(p_used=pu)[pu.notna() & s["in_range"].astype(bool)])
                else:
                    tip = s["tier"].astype(str) != ""
                    if tip.any():
                        keep.append(part[tip].assign(p_used=pu[tip]))
    lam = lam_from(pd.concat(fitrows, ignore_index=True)) if fitrows else R._lams(None)
    b = pd.concat(bets, ignore_index=True) if bets else pd.DataFrame()
    return v6_select(b, lam)


def shortfall(work: Path, universe: Path, picks: pd.DataFrame, vpool: pd.DataFrame, lam: pd.DataFrame,
              params: dict) -> pd.DataFrame:
    """One row per (track, league, ISO week) of the validation window with >= 1 fixture: tips and the cause."""
    ch = load_choices(work)
    on = {}
    for (tr, ms), c in ch.items():
        for lg, b in (c.get("bars") or {}).items():
            if c.get("chosen") and "off" not in b and b.get("mode") == S.MODE_MODEL \
                    and S.parse_cfg(c["chosen"])[0] != S.LEAGUE_WIDE_SOURCE:
                on.setdefault((tr, lg), set()).add(ms)
    lamk = {tuple(k) for k in lam.loc[lam["lam"].notna(), R.CELL].values}
    priced = {}
    for f, tr, ms in S.load_families(work / "priced"):
        f = f[(f["date"] >= VAL_START) & (f["date"] <= VAL_END)]
        priced[(tr, ms)] = set(f["row_id"])
    vp = vpool.assign(gw=gw(vpool["date"])) if len(vpool) else vpool
    pk = picks.assign(gw=gw(picks["date"])) if len(picks) else picks
    rows = []
    for tr in TRACKS:
        u = pd.read_parquet(Path(universe) / f"universe_{tr}.parquet", columns=["row_id", "date", "league", "market", "p", "y"])
        u["date"] = pd.to_datetime(u["date"])
        u = u[(u["date"] >= VAL_START) & (u["date"] <= VAL_END) & u["y"].notna()]
        fx = u.drop_duplicates("row_id").assign(gw=lambda d: gw(d["date"]))
        for (lg, w), g in fx.groupby(["league", "gw"]):
            cells = on.get((tr, lg), set())
            n = int(((pk["track"] == tr) & (pk["league"] == lg) & (pk["gw"] == w)).sum()) if len(pk) else 0
            ids = set(g["row_id"])
            slots = len(ids) * len(cells)
            no_price = sum(len(ids - priced.get((tr, ms), set())) for ms in cells)
            no_lam = sum(len(ids & priced.get((tr, ms), set())) for ms in cells if (tr, lg, ms) not in lamk)
            vv = vp[(vp["track"] == tr) & (vp["league"] == lg) & (vp["gw"] == w)] if len(vp) else vp
            gate = (params.get(f"{tr}|{lg}") or {}).get("gate", "CAL")
            val = int(((vv["cal_ev"] > 0) if gate == "CAL" else (vv["ev_raw"] > 0)).sum()) if len(vv) else 0
            r = {"track": tr, "league": lg, "gw": w, "fixtures": len(ids), "tips": n, "cells_on": len(cells),
                 "slots": slots, "slots_no_price": no_price, "slots_no_calibration": no_lam,
                 "bar_tips": int(len(vv)), "value_tips": val}
            if n >= 3:
                r["cause"] = ""
            elif not cells:
                r["cause"] = "ALL_CELLS_OFF"
            elif val >= 3:
                r["cause"] = "T3_EMPTY" if n == 2 else "RULE_CONFLICT"
            elif val in (1, 2) and n == val:
                r["cause"] = "FEW_VALUE"
            elif no_price >= max(slots - no_price, 1) and val == 0:
                r["cause"] = "NO_PRICE"
            elif no_lam and no_lam >= slots - no_price - no_lam and val == 0:
                r["cause"] = "NO_CALIBRATION"
            elif val == 0:
                r["cause"] = "NO_VALUE"
            else:
                r["cause"] = "FEW_VALUE"
            rows.append(r)
    return pd.DataFrame(rows)


def cmd_validate(a) -> int:
    work = P.assert_outside_repo(a.work)
    marker = work / "validation_evaluated.json"
    if marker.exists() and not a.again:
        raise SystemExit(f"{marker}: the validation was already evaluated - once only")
    fit = json.loads((OUT / "tier_fit.json").read_text(encoding="utf-8"))
    if fit["protocol_sha256_lf"] != sha_lf(PROTOCOL):
        raise SystemExit("tier fit was made under another protocol text")
    params = {k: {"gate": v["gate"], "t3_floor": v["t3_floor"]} for k, v in fit["params"].items()}
    vpool, lam = val_pool(work)
    vpool.to_parquet(work / "val_pool.parquet", index=False)
    picks = pick(vpool, params)
    marker.write_text(json.dumps({"at": pd.Timestamp.now(tz="UTC").isoformat(), "picks": int(len(picks))}), encoding="utf-8")
    picks.to_parquet(work / "val_picks.parquet", index=False)
    v6p = v6_select(vpool, lam)
    v6p.to_parquet(work / "val_v6_p11bars.parquet", index=False)
    v6a = approved_v6(work, Path(a.approved_evidence)) if a.approved_evidence else pd.DataFrame()
    if len(v6a):
        v6a.to_parquet(work / "val_v6_bade2c0b.parquet", index=False)
    sf = shortfall(work, Path(a.universe), picks, vpool, lam, params)
    sf.to_parquet(work / "val_shortfall.parquet", index=False)

    per_league = {}
    for key in sorted(set(fit["params"]) | {f"{t}|{l}" for t, l in zip(sf["track"], sf["league"])}):
        tr, lg = key.split("|", 1)
        g = picks[(picks["track"] == tr) & (picks["league"] == lg)] if len(picks) else picks
        s = sf[(sf["track"] == tr) & (sf["league"] == lg)]
        v6l = v6p[(v6p["track"] == tr) & (v6p["league"] == lg)] if len(v6p) else v6p
        v6al = v6a[(v6a["track"] == tr) & (v6a["league"] == lg)] if len(v6a) else v6a
        per_league[key] = {"params": fit["params"].get(key, {"gate": None, "t3_floor": None}) and
                           {k: fit["params"].get(key, {}).get(k) for k in ("gate", "t3_floor", "gate_basis", "t3_basis")},
                           "weeks_with_fixtures": int(len(s)),
                           "tips_per_week": round(float(s["tips"].mean()), 2) if len(s) else None,
                           "weeks_3_2_1_0": [int((s["tips"] >= 3).sum()), int((s["tips"] == 2).sum()),
                                             int((s["tips"] == 1).sum()), int((s["tips"] == 0).sum())],
                           "causes_lt3": dict(Counter(s.loc[s["tips"] < 3, "cause"])),
                           "tiers": tier_table(g) if len(g) else {"ALL": {"bets": 0}},
                           "v6_p11_bars": stats(v6l), "v6_bade2c0b": stats(v6al),
                           "fit": fit["fit_per_league"].get(key)}
    body = {"protocol": fit["protocol"], "protocol_sha256_lf": fit["protocol_sha256_lf"],
            "window": [str(VAL_START.date()), str(VAL_END.date())],
            "validation_pool_bets": int(len(vpool)), "tiers": tier_table(picks),
            "v6_p11_bars": stats(v6p), "v6_bade2c0b": stats(v6a),
            "weeks": {"league_weeks_with_fixtures": int(len(sf)),
                      "dist_3_2_1_0": [int((sf["tips"] >= 3).sum()), int((sf["tips"] == 2).sum()),
                                       int((sf["tips"] == 1).sum()), int((sf["tips"] == 0).sum())],
                      "causes_lt3": dict(Counter(sf.loc[sf["tips"] < 3, "cause"])),
                      "slots": int(sf["slots"].sum()), "slots_no_price": int(sf["slots_no_price"].sum()),
                      "slots_no_calibration": int(sf["slots_no_calibration"].sum())},
            "per_league": per_league}
    (OUT / "validation.json").write_text(json.dumps(body, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: body[k] for k in ("validation_pool_bets", "tiers", "v6_p11_bars", "v6_bade2c0b", "weeks")},
                     indent=1, default=str))
    return 0


# ── 5) the proposal ──────────────────────────────────────────────────────────────────────────

def cmd_propose(a) -> int:
    from src import threshold_proposals as TP
    from src.bars_opt import cells as C
    from src.bars_opt import hist_study as H
    work = P.assert_outside_repo(a.work)
    study_cells_path = work / "study" / "cells.json"
    fams = C.build(work / "priced", MONTH, workers=a.workers, verdict_path=study_cells_path)
    study_cells = json.loads(study_cells_path.read_text(encoding="utf-8"))
    key_m = f"{MONTH}_{TAG}"
    cand_dir = TP.CANDIDATE_DIR / NAMESPACE / key_m
    for m, data in C.production_candidates(fams).items():
        TP.write_candidate("standard" if m == "ou25" else m, data, cand_dir)
    table = H.status_table(fams, study_cells)
    counts = dict(Counter(r["status"] for r in table))
    prop_dir = TP.PROPOSAL_DIR / NAMESPACE
    prop_dir.mkdir(parents=True, exist_ok=True)
    meta = {"protocol": "p11-bars-tiers-clean/1", "protocol_sha256_lf": sha_lf(PROTOCOL),
            "predictions": "champion_recipe holdout harness, code 694995f90, frame 9ed237ebc04aee09",
            "price_tiers": {"bet": BET_TIERS + ["own (backfill stripped)", "FD pre-match certified join"],
                            "close": CLOSE_TIERS},
            "validation_window": [str(VAL_START.date()), str(VAL_END.date())]}
    ev = prop_dir / f"evidence_{key_m}.json"
    TP._safe_write(ev, json.dumps({"month": MONTH, "meta": meta, "status_counts": counts,
                                   "league_market_status": table, "families": fams}, indent=1, default=str))
    rel = cand_dir.resolve().relative_to(ROOT.resolve()).as_posix()
    p = TP.build_proposal(TP.load_candidates(cand_dir), month=MONTH, resolve=not a.no_resolve,
                          source={"workflow": "P11 bars + tiers (scripts/p11_bars_tiers.py)", "candidates_dir": rel,
                                  "evidence": f"output/threshold_proposals/{NAMESPACE}/evidence_{key_m}.json"} | meta)
    path = TP.write_proposal(p, prop_dir)

    # serving calibration (frozen with the proposal): lam per cell from every priced row < 2026-11-09 scored
    # by the SERVING cell's own source
    frames = {(t, m): f for f, t, m in S.load_families(work / "priced")}
    rows = []
    for fam in fams:
        for c in fam["cells"]:
            if c.get("mode") != S.MODE_MODEL:
                continue
            f = frames[(c["track"], c["mside"])]
            g = f[(f["league"] == c["league"]) & (f["date"] < SERVING_LAM_BEFORE)]
            if g.empty:
                continue
            s = BS.score(g[["league", "p_side", "fair", "odds"]], c)
            pu = pd.to_numeric(s["p_used"], errors="coerce")
            rows.append(g.assign(p_used=pu)[pu.notna() & s["in_range"].astype(bool)])
    lam = lam_from(pd.concat(rows, ignore_index=True)) if rows else R._lams(None)
    fit = json.loads((OUT / "tier_fit.json").read_text(encoding="utf-8"))
    cfg = {"protocol": "p11-bars-tiers-clean/1", "status": "PROPOSAL - AWAITING OWNER (2026-11-20); install with the go-live",
           "bars_proposal_id": p["proposal_id"], "bars_proposal_hash": p["proposal_hash"],
           "tiers": {"T1_BEST_VALUE": "highest cal_ev", "T2_SAFEST": "highest p_cal of the rest",
                     "T3_RISKY_VALUE": "of the rest, odds >= t3_floor, highest ev_lo (REL: the rest's longest odds)"},
           "gates": {"CAL": "cal_ev > 0", "BAR": "cleared bar and p_used * odds - 1 > 0"},
           "leagues": {k: {kk: v[kk] for kk in ("gate", "t3_floor", "gate_basis", "t3_basis")}
                       for k, v in sorted(fit["params"].items())},
           "calibration": {"rule": "R2: p_cal = fair + lam * (p_used - fair); ev_lo with lam - lam_se",
                           "fitted_on": f"priced rows < {SERVING_LAM_BEFORE.date()}",
                           "cells": lam.round(6).to_dict("records")},
           "stake": "0.5% flat on every TOP-3 tip (owner 2026-11-17); unchanged"}
    TIER_CONFIG.write_text(json.dumps(cfg, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"proposal_id": p["proposal_id"], "proposal_hash": p["proposal_hash"], "entries": len(p["entries"]),
                      "proposal": str(Path(path).resolve().relative_to(ROOT.resolve())), "status_counts": counts,
                      "serving_lam_cells": int(lam["lam"].notna().sum())}, indent=1, default=str))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--universe", required=True)
    p.add_argument("--store", required=True)
    p.add_argument("--aliases", required=True)
    p.add_argument("--work", required=True)
    for name in ("bars", "tiers-fit", "propose"):
        s = sub.add_parser(name)
        s.add_argument("--work", required=True)
        s.add_argument("--workers", type=int, default=4)
        s.add_argument("--no-resolve", action="store_true")
    v = sub.add_parser("validate")
    v.add_argument("--work", required=True)
    v.add_argument("--universe", required=True)
    v.add_argument("--approved-evidence", default=str(ROOT / "output" / "threshold_proposals" / "bars_oddsapi_hist"
                                                       / "evidence_2026-11_v5.json"))
    v.add_argument("--again", action="store_true")
    a = ap.parse_args(argv)
    return {"prepare": cmd_prepare, "bars": cmd_bars, "tiers-fit": cmd_tiers_fit, "validate": cmd_validate,
            "propose": cmd_propose}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
