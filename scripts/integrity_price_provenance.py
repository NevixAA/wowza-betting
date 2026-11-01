"""Price provenance health (R2-C findings 14, 15, 24 + newformat_odds_history line mapping).

    python scripts/integrity_price_provenance.py     -> output/price_provenance_health.json

READ-ONLY on every raw file. For each capture archive it labels rows with
src.integrity.price_provenance.label_market_mapping and checks the labels against REALISED
outcomes: for each (period, line) the mean de-vigged implied probability (proportional, last
snapshot per fixture) is compared with the realised hit rate from output/fd_history.parquet.
A valid full-match line has implied ~= realised; a mis-mapped one does not. For the backtest
outputs it counts synthetic (filled) side prices and confirms how many of them sit in the rows a
ROI would read (placed SNIPER/MARKSMAN).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

LINES = {"1.5": ("over15", "under15"), "2.5": ("over25", "under25"), "3.5": ("over35", "under35")}


def _lg(x) -> str:
    from src.league_registry import canonical_name
    return canonical_name(x) or str(x)


def outcomes() -> dict:
    """{(canonical league, YYYY-MM-DD): {(home, away): (total goals, btts)}} from fd_history."""
    import pandas as pd
    f = pd.read_parquet(ROOT / "output" / "fd_history.parquet")
    f["md"] = pd.to_datetime(f["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    f["tg"] = f["home_goals"] + f["away_goals"]
    f["bt"] = ((f["home_goals"] > 0) & (f["away_goals"] > 0)).astype(float)
    idx: dict = {}
    for r in f.dropna(subset=["md"]).itertuples(index=False):
        idx.setdefault((_lg(r.league), r.md), {}).setdefault((str(r.home_team), str(r.away_team)), (r.tg, r.bt))
    return idx


def _result_of(idx: dict, league, d, home, away):
    """(tg, bt) of the SAME fixture: one league, one date, clubs matched by
    src/team_names.resolve_result (the settlement resolver). (nan, nan) when absent/ambiguous."""
    from src.team_names import resolve_result
    pairs = idx.get((_lg(league), str(d)[:10]))
    if not pairs:
        return (float("nan"), float("nan"))
    h = resolve_result(str(home), sorted({p[0] for p in pairs}))
    a = None if h is None else resolve_result(str(away), sorted({p[1] for p in pairs if p[0] == h}))
    return pairs[(h, a)] if a is not None else (float("nan"), float("nan"))


def evidence(lab, out, source):
    import numpy as np
    import pandas as pd
    d = lab.copy()
    d["_ts"] = pd.to_datetime(d["snapshot_ts"], errors="coerce", utc=True, format="mixed")
    d["period"] = np.where(d["price_origin"] == "BACKFILL_HISTORICAL", "backfill",
                           np.where(d["market_mapping_valid"], "forward_valid", "forward_invalid"))
    d = d.sort_values("_ts").drop_duplicates(["league", "match", "match_date", "market", "period"], keep="last")
    w = d.pivot_table(index=["league", "match", "match_date", "period"], columns="market",
                      values="odds", aggfunc="last").reset_index()
    p = w["match"].astype(str).str.split(" vs ", n=1, expand=True)
    if p.shape[1] < 2:
        p[1] = None
    # Resolver join (league + date + resolve_result), not a lower-cased/stripped name key.
    res = [_result_of(out, lg, md, h, a) if a is not None else (float("nan"), float("nan"))
           for lg, md, h, a in zip(w["league"], w["match_date"], p[0], p[1])]
    w["tg"] = [r[0] for r in res]
    w["bt"] = [r[1] for r in res]
    unmatched = int(w["tg"].isna().sum())
    if unmatched:
        print(f"  [{source}] {unmatched} fixture-period row(s) without a resolvable result - left out")
    m = w[w["tg"].notna()]
    rows = []
    for per, g in m.groupby("period"):
        for ln, (o, u) in LINES.items():
            if o not in g or u not in g:
                continue
            gg = g.dropna(subset=[o, u, "tg"])
            if len(gg) < 20:
                continue
            io, iu = 1 / gg[o], 1 / gg[u]
            rows.append({"source": source, "period": per, "line": f"OU{ln}", "n": int(len(gg)),
                         "median_over": round(float(gg[o].median()), 3),
                         "implied_over": round(float((io / (io + iu)).mean()), 4),
                         "realised_over": round(float((gg["tg"] > float(ln)).mean()), 4)})
        if "btts_yes" in g and "btts_no" in g:
            gg = g.dropna(subset=["btts_yes", "btts_no", "bt"])
            if len(gg) >= 20:
                iy, ino = 1 / gg["btts_yes"], 1 / gg["btts_no"]
                rows.append({"source": source, "period": per, "line": "BTTS", "n": int(len(gg)),
                             "median_over": round(float(gg["btts_yes"].median()), 3),
                             "implied_over": round(float((iy / (iy + ino)).mean()), 4),
                             "realised_over": round(float(gg["bt"].mean()), 4)})
    for r in rows:
        se = (r["realised_over"] * (1 - r["realised_over"]) / r["n"]) ** 0.5
        r["gap"] = round(r["implied_over"] - r["realised_over"], 4)
        r["consistent_full_match"] = bool(abs(r["gap"]) <= max(0.05, 3 * se))
    return rows


def synthetic_backtests():
    import pandas as pd
    from src.integrity.price_provenance import label_side_prices, SYNTHETIC_LABEL
    out = {}
    for mk in ("btts", "over15", "over35"):
        p = ROOT / "output" / f"backtest_results_{mk}.csv"
        if not p.exists():
            out[mk] = {"status": "UNKNOWN", "reason": "file absent"}
            continue
        d = pd.read_csv(p, low_memory=False)
        lab = label_side_prices(d, mk)
        syn = lab["price_provenance"] == SYNTHETIC_LABEL
        tier = lab.get("signal_tier", lab.get("tier", pd.Series("", index=lab.index))).astype(str)
        placed = tier.isin(["SNIPER", "MARKSMAN"])
        out[mk] = {"rows": int(len(lab)), "synthetic_rows": int(syn.sum()),
                   "synthetic_share": round(float(syn.mean()), 4) if len(lab) else None,
                   "basis": lab.attrs.get("provenance_basis"),
                   "placed_rows": int(placed.sum()),
                   "placed_rows_synthetic": int((placed & syn).sum()),
                   "label": "SYNTHETIC_PRICE_DIAGNOSTIC_ONLY / INVALID_FOR_PROFITABILITY on synthetic rows"}
    return out


def main(argv=None) -> int:
    import pandas as pd
    import logging
    logging.basicConfig(level=logging.WARNING)
    from src.integrity.price_provenance import label_market_mapping, PROVENANCE_VERSION
    out = outcomes()
    files, ev = {}, []
    for src in ("newformat_odds_history", "standard_sidemarket_odds_history", "standard_odds_history"):
        p = ROOT / "output" / f"{src}.csv"
        if not p.exists():
            continue
        raw = pd.read_csv(p, low_memory=False)
        lab = label_market_mapping(raw, src)
        bad = lab[~lab["market_mapping_valid"]]
        files[src] = {"rows": int(len(lab)), "invalid_rows": int(len(bad)),
                      "invalid_by_reason": {k: int(v) for k, v in bad["mapping_reason"].value_counts().items()},
                      "invalid_by_market": {k: int(v) for k, v in bad["market"].value_counts().items()},
                      "main_line_only_rows": int((lab["line_selection"] == "MAIN_LINE_ONLY").sum()),
                      "raw_file_rewritten": False}
        ev += evidence(lab, out, src)
    doc = {"artifact": "price_provenance_health", "version": PROVENANCE_VERSION,
           "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "capture_files": files, "mapping_evidence_vs_realised": ev,
           "synthetic_side_prices_in_backtests": synthetic_backtests()}
    bad_ev = [e for e in ev if e["period"] != "forward_invalid" and not e["consistent_full_match"]]
    doc["status"] = "WARN" if bad_ev else "PASS"
    doc["unexplained_inconsistent_cells"] = bad_ev
    (ROOT / "output" / "price_provenance_health.json").write_text(
        json.dumps(doc, indent=2, default=str), encoding="utf-8")
    for e in ev:
        print(e)
    print(json.dumps(files, indent=1))
    print(json.dumps(doc["synthetic_side_prices_in_backtests"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
