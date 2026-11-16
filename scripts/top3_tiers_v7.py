"""TOP-3 v7 (three named tiers) - RESEARCH / PROPOSAL ONLY (config/top3_tiers_protocol_v7.json).

    # 1) DEV only: per-league T3 thresholds + per-cell calibration (refuses unless the protocol is committed)
    python scripts/top3_tiers_v7.py fit-dev --priced <local>/priced --evidence <evidence json> --work <local dir>

    # 2) HOLDOUT, once: tiers per league, shortfall diagnostics, v6 R2 comparison (refuses unless the
    #    thresholds file of step 1 is committed unchanged)
    python scripts/top3_tiers_v7.py evaluate --priced <local>/priced --evidence <evidence json> \
        --fixtures <frame fixtures parquet> --work <local dir>

Prices are PRIVATE (OddsAPI history: no redistribution). Per-pick frames stay in --work (outside this repo);
only counts and statistics are written to output/research/top3_tiers_v7/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
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

OUT = ROOT / "output" / "research" / "top3_tiers_v7"
PROTOCOL = ROOT / T.PROTOCOL_FILE
THRESHOLDS = OUT / "t3_thresholds.json"


def sha_lf(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def committed_unchanged(p: Path) -> tuple[bool, str]:
    rel = p.resolve().relative_to(ROOT.resolve()).as_posix()
    if not subprocess.run(["git", "ls-files", "--", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip():
        return False, f"{rel} is not committed"
    if subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip():
        return False, f"{rel} has uncommitted changes"
    return True, ""


def protocol() -> dict:
    ok, why = committed_unchanged(PROTOCOL)
    if not ok:
        raise SystemExit(f"protocol not frozen: {why}")
    pr = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    code = ROOT / pr["code"]["file"]
    if sha_lf(code) != pr["code"]["sha256_lf"]:
        raise SystemExit(f"{pr['code']['file']} differs from the code the protocol froze")
    if sha_lf(Path(__file__)) != pr["code"]["script_sha256_lf"]:
        raise SystemExit(f"{pr['code']['script']} differs from the script the protocol froze")
    return pr


def cells_from_evidence(ev: dict) -> list[dict]:
    """Every non-OFF MODEL cell of the proposal's status table with its configuration (source, odds range),
    status and - for serving cells - the full cell (bars) for the v6 R2 comparison."""
    full = {(c["track"], c["mside"], c["league"]): c for f in ev["families"] for c in f["cells"]}
    out = []
    for r in ev["league_market_status"]:
        if r["status"] == "OFF" or r.get("mode", "MODEL") != S.MODE_MODEL or not r.get("config"):
            continue
        src, meas, hi, lo = S.parse_cfg(r["config"])
        if src == S.LEAGUE_WIDE_SOURCE:
            continue
        out.append({"track": r["track"], "mside": r["mside"], "league": r["league"], "status": r["status"],
                    "source": src, "measure": meas, "odds_max": hi, "odds_min": lo,
                    "serving": full.get((r["track"], r["mside"], r["league"]))})
    return out


def build(priced_dir: Path, ev: dict, dev_end: pd.Timestamp) -> dict:
    unis = {t: pd.read_parquet(Path(priced_dir) / f"universe_{t}.parquet") for t in ("standard", "new_format")}
    cells = cells_from_evidence(ev)
    fams, rows = {}, []
    for c in cells:
        k = (c["track"], c["mside"])
        if k not in fams:
            fams[k] = S.family_frame(unis[c["track"]], c["mside"])
        r = T.cell_rows(fams[k], c, dev_end)
        if len(r):
            rows.append(r.assign(status=c["status"]))
    rows = pd.concat(rows, ignore_index=True)
    dev = rows[rows["date"] < dev_end]
    lam = R._lams(dev[R.CELL + ["fair", "p_used", "won"]])
    cal = T.calibrate(rows, lam)
    return {"unis": unis, "cells": cells, "fams": fams, "rows": cal, "lam": lam}


def cmd_fit_dev(a) -> int:
    pr = protocol()
    dev_end = pd.Timestamp(pr["windows"]["development_before"])
    work = P.assert_outside_repo(a.work)
    work.mkdir(parents=True, exist_ok=True)
    ev = json.loads(Path(a.evidence).read_text(encoding="utf-8"))
    b = build(Path(a.priced), ev, dev_end)
    cal = b["rows"]
    dev_pool = cal[(cal["date"] < dev_end) & (cal["cal_ev"] > 0)]
    fit = T.fit_t3(dev_pool)
    OUT.mkdir(parents=True, exist_ok=True)
    body = {"protocol": pr["protocol"], "protocol_sha256_lf": sha_lf(PROTOCOL),
            "evidence": a.evidence_label or Path(a.evidence).name, "evidence_sha256": sha_lf(Path(a.evidence)),
            "development_before": str(dev_end.date()), "dev_rows": int((cal["date"] < dev_end).sum()),
            "dev_value_rows": int(len(dev_pool)), "cells_non_off_model": len(b["cells"]),
            "cells_with_lam": int(b["lam"]["lam"].notna().sum()),
            "lam": b["lam"].round(4).to_dict("records"),
            "t3": {k: {kk: vv for kk, vv in v.items()} for k, v in sorted(fit.items())}}
    THRESHOLDS.write_text(json.dumps(body, indent=1, default=str) + "\n", encoding="utf-8")
    n_thr = sum(1 for v in fit.values() if v.get("threshold"))
    print(json.dumps({"leagues": len(fit), "with_t3_threshold": n_thr, "cells": len(b["cells"]),
                      "cells_with_lam": body["cells_with_lam"], "file": str(THRESHOLDS)}, indent=1))
    return 0


def tier_stats(t: pd.DataFrame) -> dict:
    if t is None or t.empty:
        return {"bets": 0}
    s = S.score(t)
    return {"bets": s["bets"], "hit_rate": round(float(t["won"].mean()), 4), "roi": s["roi"], "ci95": s["ci95"],
            "mean_odds": round(float(t["odds"].mean()), 3), "clv": s["clv"], "clv_n": s["clv_n"],
            "mean_cal_ev": round(float(t["cal_ev"].mean()), 4) if "cal_ev" in t else None}


def cmd_evaluate(a) -> int:
    pr = protocol()
    ok, why = committed_unchanged(THRESHOLDS)
    if not ok:
        raise SystemExit(f"thresholds not frozen: {why}")
    thr_body = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    if thr_body["protocol_sha256_lf"] != sha_lf(PROTOCOL):
        raise SystemExit("thresholds were fitted under another protocol text")
    work = P.assert_outside_repo(a.work)
    marker = work / "holdout_evaluated.json"
    if marker.exists():
        raise SystemExit(f"{marker}: the holdout was already evaluated - once only")
    dev_end = pd.Timestamp(pr["windows"]["development_before"])
    h0, h1 = pd.Timestamp(pr["windows"]["holdout"][0]), pd.Timestamp(pr["windows"]["holdout"][1])
    ev = json.loads(Path(a.evidence).read_text(encoding="utf-8"))
    if sha_lf(Path(a.evidence)) != thr_body["evidence_sha256"]:
        raise SystemExit("evidence differs from the one the thresholds were fitted with")
    b = build(Path(a.priced), ev, dev_end)
    thr = T.thresholds_of(thr_body["t3"])
    cal = b["rows"]
    ho = cal[(cal["date"] >= h0) & (cal["date"] <= h1)]
    pool = ho[ho["cal_ev"] > 0]
    picks = T.pick_tiers(pool, thr)
    picks["stake"] = picks["status"].map(T.stake_label)
    marker.write_text(json.dumps({"at": pd.Timestamp.now(tz="UTC").isoformat(), "picks": int(len(picks))}),
                      encoding="utf-8")
    picks.to_parquet(work / "v7_picks.parquet", index=False)              # carries prices: LOCAL only

    # v6 R2 (live rule) on the same rows: tips clearing the serving cell's bars (source refitted on DEV only),
    # TOP-3 by R2 per league-week (src.research.top3_ranking.select)
    v6 = []
    for c in b["cells"]:
        sc = c["serving"]
        if not sc:
            continue
        f = b["fams"][(c["track"], c["mside"])]
        f = f[(f["league"] == c["league"]) & (f["date"] >= h0) & (f["date"] <= h1)]
        if f.empty:
            continue
        dev = b["fams"][(c["track"], c["mside"])]
        dev = dev[dev["date"] < dev_end]
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
    v6b = pd.concat(v6, ignore_index=True) if v6 else pd.DataFrame()
    v6p = R.select(v6b, b["lam"].assign(n=0, roi=np.nan, se=np.nan, lcb=np.nan, eb=np.nan), R.LIVE_RULE) \
        if len(v6b) else pd.DataFrame()
    if len(v6p):
        v6p = T.calibrate(v6p.drop(columns=["key", "has_stat", "no_stat", "rank_basis", "lam"], errors="ignore"), b["lam"])
        v6p.to_parquet(work / "v6_picks.parquet", index=False)

    # shortfall: every played fixture of the window x the league's non-OFF MODEL cells
    fx = pd.read_parquet(a.fixtures)
    fx = fx[(fx["date"] >= h0) & (fx["date"] <= h1)]
    fx = fx[fx["league"].isin({c["league"] for c in b["cells"]} | set(b["unis"]["standard"]["league"])
                              | set(b["unis"]["new_format"]["league"]))]
    weeks = sorted(set(T.gw(pd.Series(pd.date_range(h0, h1, freq="D")))))
    lamk = {tuple(k) for k in b["lam"].loc[b["lam"]["lam"].notna(), R.CELL].values}
    slots = []
    for c in b["cells"]:
        u = b["unis"][c["track"]]
        mk = S.MARKET_SIDES[c["mside"]][0]
        um = u[(u["market"] == mk) & (u["league"] == c["league"])]
        modeled = set(um.loc[um["p"].notna(), "row_id"])
        fam = b["fams"][(c["track"], c["mside"])]
        priced = set(fam.loc[fam["league"] == c["league"], "row_id"])
        r = ho[(ho["track"] == c["track"]) & (ho["league"] == c["league"]) & (ho["mside"] == c["mside"])]
        value = set(r.loc[r["cal_ev"] > 0, "row_id"])
        f = fx[(fx["track"] == c["track"]) & (fx["league"] == c["league"])]
        if f.empty:
            continue
        slots.append(pd.DataFrame({"track": c["track"], "league": c["league"], "mside": c["mside"],
                                   "row_id": f["row_id"].to_numpy(), "date": f["date"].to_numpy(),
                                   "model": f["row_id"].isin(modeled).to_numpy(),
                                   "price": f["row_id"].isin(priced).to_numpy(),
                                   "lam": (c["track"], c["league"], c["mside"]) in lamk,
                                   "value": f["row_id"].isin(value).to_numpy()}))
    slots = pd.concat(slots, ignore_index=True) if slots else pd.DataFrame(
        columns=["track", "league", "mside", "row_id", "date", "model", "price", "lam", "value"])
    for c in ("model", "price", "lam", "value"):
        slots[c] = slots[c].astype(bool)
    cells_on = {}
    for c in b["cells"]:
        cells_on[(c["track"], c["league"])] = cells_on.get((c["track"], c["league"]), 0) + 1
    tr_lg = fx[["track", "league"]].drop_duplicates()
    sf = T.shortfall(fx, slots, picks if len(picks) else pd.DataFrame(columns=["track", "league", "gw", "tier"]),
                     cells_on, weeks, thr)
    sf = sf.merge(tr_lg, on=["track", "league"])

    # statistics only
    res = {"protocol": pr["protocol"], "protocol_sha256_lf": sha_lf(PROTOCOL),
           "thresholds_sha256_lf": sha_lf(THRESHOLDS), "evidence_sha256": thr_body["evidence_sha256"],
           "holdout": [str(h0.date()), str(h1.date())], "label": pr["windows"]["holdout_label"],
           "pool_rows": int(len(pool)), "holdout_rows_valued": int(ho["cal_ev"].notna().sum()),
           "tiers": {}, "per_league": {}, "combined": tier_stats(picks), "by_stake": {},
           "v6_r2": {"combined": tier_stats(v6p), "per_league": {}}}
    for tier in T.TIERS:
        res["tiers"][tier] = tier_stats(picks[picks["tier"] == tier]) if len(picks) else {"bets": 0}
    for st in ("STAKE", "PAPER"):
        res["by_stake"][st] = tier_stats(picks[picks["stake"] == st]) if len(picks) else {"bets": 0}
    for (tr, lg), g in (picks.groupby(["track", "league"]) if len(picks) else []):
        res["per_league"][f"{tr}|{lg}"] = {tier: tier_stats(g[g["tier"] == tier]) for tier in T.TIERS} \
            | {"combined": tier_stats(g)}
    for (tr, lg), g in (v6p.groupby(["track", "league"]) if len(v6p) else []):
        res["v6_r2"]["per_league"][f"{tr}|{lg}"] = tier_stats(g)
    # shortfall shares
    def shares(x):
        n = len(x)
        return {str(k): int((x["tips"].clip(upper=3) == k).sum()) for k in (3, 2, 1, 0)} | {"league_weeks": n}
    active = sf[sf["fixtures"] > 0]
    res["shortfall"] = {"all_league_weeks": shares(sf), "league_weeks_with_fixtures": shares(active),
                        "cause_counts_all": sf.loc[sf["tips"] < 3, "cause"].value_counts().to_dict(),
                        "cause_counts_with_fixtures": active.loc[active["tips"] < 3, "cause"].value_counts().to_dict(),
                        "per_league": {}}
    for (tr, lg), g in sf.groupby(["track", "league"]):
        ga = g[g["fixtures"] > 0]
        res["shortfall"]["per_league"][f"{tr}|{lg}"] = {
            "weeks": int(len(g)), "weeks_with_fixtures": int(len(ga)), "tips_share_with_fixtures": shares(ga),
            "mean_tips_per_week_with_fixtures": round(float(ga["tips"].mean()), 3) if len(ga) else None,
            "causes": ga.loc[ga["tips"] < 3, "cause"].value_counts().to_dict(),
            "slots": {k: int(ga[k].sum()) for k in ("slots", "slots_no_model", "slots_no_price", "slots_no_lam",
                                                     "slots_no_value", "value_slots")}}
    sf.to_parquet(work / "shortfall.parquet", index=False)                  # counts only, kept local with the picks
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "holdout_evaluation.json").write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("combined", "tiers", "by_stake")} | {"v6_r2": res["v6_r2"]["combined"],
                                                                                "shortfall": {k: v for k, v in res["shortfall"].items() if k != "per_league"}},
                     indent=1, default=str))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fit-dev", "evaluate"):
        s = sub.add_parser(name)
        s.add_argument("--priced", required=True)
        s.add_argument("--evidence", required=True)
        s.add_argument("--work", required=True, type=Path)
        s.add_argument("--evidence-label")
        if name == "evaluate":
            s.add_argument("--fixtures", required=True)
    a = ap.parse_args(argv)
    return {"fit-dev": cmd_fit_dev, "evaluate": cmd_evaluate}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
