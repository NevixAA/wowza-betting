"""AU4 Phase 5 - ledger price provenance: decision-time availability and closing-price validity.

A. bets_ledger (O/U 2.5) CLOSING price provenance. update_results._closing_for_market takes the
   last row of the first archive that has the fixture (dense -> nf history -> std history -> std
   side), WITHOUT the price-provenance labels. For every ledger row with closing_odds this script
   finds which archive rows carry exactly that price for the same fixture (date + team tokens) and
   side, and labels the close:
     INVALID_NF_PARSER_WINDOW  - only matching rows are newformat_odds_history forward O/U rows
                                 captured before 2026-08-10T08:00Z (parser took first-half / team
                                 totals; see src/integrity/price_provenance.py)
     VALID_ARCHIVE             - at least one matching row outside that window
     NOT_FOUND                 - no archive row has that price (JSON-sourced close, since purged)
   and recomputes ledger CLV statistics with and without the invalid closes.
B. Same for side_bets_ledger BTTS closes (btts_yes > 3.20 = first-half BTTS).
C. Decision-time availability (live O/U rows, generated_at >= 2026-08-19, book_odds_snapshots
   coverage): per-book last-observation-carried-forward standing price at generated_at for the
   same fixture/side. Reports whether the ledger price was standing at some book, its rank vs the
   best standing price, and the best-price gap.
D. BTTS entry construction: median Yes and median No across books (src/btts_odds.py) vs the
   same-book same-snapshot pair, from book_odds_snapshots.
Writes reports/au4_data/ledger_prices.json + csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from au4_common import OUT, block_bootstrap, heavy_lock, read_rolling, save_csv, save_json, summarise, utc  # noqa: E402
from au4_clv_forensic import same_fixture  # noqa: E402

NF_FIX = pd.Timestamp("2026-08-10T08:00:00Z")
ARCH = ["newformat_odds_dense.csv", "newformat_odds_history.csv", "standard_odds_history.csv",
        "standard_sidemarket_odds_history.csv"]


def load_archives() -> pd.DataFrame:
    parts = []
    for n in ARCH:
        d = read_rolling(n)
        d["archive"] = n
        parts.append(d[["snapshot_ts", "match_date", "league", "match", "market", "odds", "archive"]])
    a = pd.concat(parts, ignore_index=True)
    a["odds"] = pd.to_numeric(a["odds"], errors="coerce")
    a["ts"] = utc(a["snapshot_ts"])
    a["md"] = a["match_date"].astype(str).str[:10]
    bf = a["snapshot_ts"].astype(str).str.endswith("T00:00:00Z")
    a["invalid"] = ((a["archive"] == "newformat_odds_history.csv") & ~bf & (a["ts"] < NF_FIX)
                    & a["market"].isin(["over15", "under15", "over25", "under25", "over35", "under35"]))
    a["invalid"] |= (a["market"] == "btts_yes") & (a["odds"] > 3.20)
    return a


def classify_closes(led: pd.DataFrame, a: pd.DataFrame, market_of) -> pd.DataFrame:
    idx = {k: g for k, g in a.groupby(["md", "market"])}
    out = []
    for i, r in led.iterrows():
        c = pd.to_numeric(r.get("closing_odds"), errors="coerce")
        if pd.isna(c):
            continue
        mk = market_of(r)
        g = idx.get((str(r["match_date"])[:10], mk))
        lab, arch = "NOT_FOUND", None
        if g is not None:
            g = g[np.isclose(g["odds"], c)]
            g = g[[same_fixture(f"{r['home_team']} vs {r['away_team']}", m) for m in g["match"]]]
            if len(g):
                arch = ";".join(sorted(g["archive"].unique()))
                lab = "VALID_ARCHIVE" if (~g["invalid"]).any() else (
                    "INVALID_NF_PARSER_WINDOW" if mk != "btts_yes" else "INVALID_FIRST_HALF_BTTS")
        out.append({"idx": i, "close_label": lab, "close_archives": arch})
    return led.join(pd.DataFrame(out).set_index("idx"), how="left")


def main() -> None:
    R: dict = {}
    a = load_archives()
    # ---- A. bets_ledger closes
    b = pd.read_csv(OUT / "bets_ledger.csv", low_memory=False)
    b = b[b["side"].isin(["OVER", "UNDER"])]
    b = classify_closes(b, a, lambda r: "under25" if r["side"] == "UNDER" else "over25")
    b["clv_pct"] = pd.to_numeric(b["clv_pct"], errors="coerce")
    b["fixture"] = b["home_team"].astype(str) + "|" + b["away_team"].astype(str) + "|" + b["match_date"].astype(str)
    cl = b[b["clv_pct"].notna()]
    R["bets_ledger_close_labels"] = cl.groupby(["source", "model_type", "close_label"]).size().reset_index(name="n").to_dict("records")
    inv = cl[cl["close_label"].astype(str).str.startswith("INVALID")]
    R["bets_ledger_invalid_close"] = {"n": len(inv), "match_date_range": [inv["match_date"].min(), inv["match_date"].max()],
                                      "by_side": inv["side"].value_counts().to_dict(),
                                      "clv_pct": summarise(inv["clv_pct"]),
                                      "results": inv["result"].value_counts().to_dict()}
    live = cl[cl["source"] == "live"]
    ok = live[~live["close_label"].astype(str).str.startswith("INVALID")]
    R["bets_ledger_live_clv"] = {
        "as_stored": {"summary": summarise(live["clv_pct"]), "ci": block_bootstrap(live["clv_pct"], live["fixture"])},
        "excluding_invalid_closes": {"summary": summarise(ok["clv_pct"]), "ci": block_bootstrap(ok["clv_pct"], ok["fixture"])},
        "by_model_type_as_stored": {m: summarise(g["clv_pct"]) for m, g in live.groupby("model_type")},
        "by_model_type_excl_invalid": {m: summarise(g["clv_pct"]) for m, g in ok.groupby("model_type")},
        "by_side_excl_invalid": {m: {"summary": summarise(g["clv_pct"]), "ci": block_bootstrap(g["clv_pct"], g["fixture"], n_boot=1000)} for m, g in ok.groupby("side")},
        "from_2026_09_01_excl_invalid": {m: {"summary": summarise(g["clv_pct"]), "ci": block_bootstrap(g["clv_pct"], g["fixture"], n_boot=1000)}
                                         for m, g in ok[ok["match_date"] >= "2026-09-01"].groupby("model_type")},
    }
    save_csv(cl[["source", "generated_at", "match_date", "league", "home_team", "away_team", "side", "odds", "closing_odds",
                 "clv_pct", "model_type", "close_label", "close_archives", "result"]], "ledger_close_provenance.csv")

    # ---- B. side ledger BTTS / O1.5 / O3.5 closes
    s = pd.read_csv(OUT / "side_bets_ledger.csv", low_memory=False)
    s = classify_closes(s, a, lambda r: {"btts": "btts_yes", "over15": "over15", "over35": "over35"}.get(str(r["market"]), ""))
    s["clv_pct"] = pd.to_numeric(s["clv_pct"], errors="coerce")
    sc = s[s["clv_pct"].notna()]
    R["side_ledger_close_labels"] = sc.groupby(["market", "close_label"]).size().reset_index(name="n").to_dict("records")
    R["side_ledger_close_archive"] = sc["close_archives"].value_counts().to_dict()
    R["side_ledger_clv_by_market"] = {m: summarise(g["clv_pct"]) for m, g in sc.groupby("market")}

    # ---- C. decision-time availability (book_odds_snapshots, OU25)
    q = read_rolling("book_odds_snapshots.csv")
    q = q[q["market"] == "OU25"].copy()
    q["ts"] = utc(q["snapshot_ts"])
    q["odds"] = pd.to_numeric(q["odds"], errors="coerce")
    q = q.sort_values("ts")
    qg = {k: g for k, g in q.groupby(["match_date", "match", "side"])}
    lv = pd.read_csv(OUT / "bets_ledger.csv", low_memory=False)
    lv = lv[(lv["source"] == "live") & lv["side"].isin(["OVER", "UNDER"]) & (lv["generated_at"] >= "2026-08-19 10:00")].copy()
    rows = []
    for i, r in lv.iterrows():
        g = qg.get((str(r["match_date"])[:10], f"{r['home_team']} vs {r['away_team']}", r["side"]))
        t = pd.Timestamp(str(r["generated_at"]), tz="UTC") + pd.Timedelta(seconds=59)
        rec = {"idx": i, "avail_status": "NO_BOOK_SNAPSHOTS"}
        if g is not None:
            st = g[g["ts"] <= t].groupby("bookmaker").tail(1)        # LOCF standing price per book
            if len(st):
                o = float(r["odds"])
                rec.update({"avail_status": "OK", "n_books": len(st), "best": st["odds"].max(), "median": st["odds"].median(),
                            "ledger_price_standing": bool(np.isclose(st["odds"], o).any()),
                            "ledger_gt_best": bool(o > st["odds"].max() + 1e-9),
                            "rank_pct": float((st["odds"] > o + 1e-9).mean()),
                            "best_gap_pct": (st["odds"].max() / o - 1) * 100,
                            "standing_age_min_of_match": float((t - st.loc[np.isclose(st["odds"], o), "ts"].max()).total_seconds() / 60) if np.isclose(st["odds"], o).any() else np.nan})
            else:
                rec["avail_status"] = "NO_SNAPSHOT_BEFORE_DECISION"
        rows.append(rec)
    av = lv.join(pd.DataFrame(rows).set_index("idx"))
    okav = av[av["avail_status"] == "OK"]
    R["decision_time_availability"] = {
        "rows": len(av), "status": av["avail_status"].value_counts().to_dict(),
        "ledger_price_standing_at_some_book": int(okav["ledger_price_standing"].sum()),
        "ledger_price_above_best_standing": int(okav["ledger_gt_best"].sum()),
        "books_standing": summarise(okav["n_books"]),
        "best_gap_pct": summarise(okav["best_gap_pct"]),
        "share_ledger_is_best": float((okav["rank_pct"] == 0).mean()) if len(okav) else None,
        "standing_age_min": summarise(okav["standing_age_min_of_match"])}
    save_csv(av[["generated_at", "match_date", "home_team", "away_team", "side", "odds", "avail_status", "n_books", "best", "median",
                 "ledger_price_standing", "ledger_gt_best", "best_gap_pct", "model_type", "result"]], "ledger_decision_time_availability.csv")

    # ---- D. BTTS median-of-books vs same-book pair
    bt = read_rolling("book_odds_snapshots.csv")
    bt = bt[bt["market"] == "BTTS"].copy()
    bt["odds"] = pd.to_numeric(bt["odds"], errors="coerce")
    w = bt.pivot_table(index=["match_date", "match", "snapshot_ts", "bookmaker"], columns="side", values="odds", aggfunc="last").dropna()
    w["orr_same"] = 1 / w["YES"] + 1 / w["NO"]
    w["fair_same"] = (1 / w["YES"]) / w["orr_same"]
    per = w.groupby(level=["match_date", "match", "snapshot_ts"]).agg(med_yes=("YES", "median"), med_no=("NO", "median"),
                                                                      fair_same_med=("fair_same", "median"), n=("YES", "size"))
    per = per[per["n"] >= 2]
    per["orr_mixed"] = 1 / per["med_yes"] + 1 / per["med_no"]
    per["fair_mixed"] = (1 / per["med_yes"]) / per["orr_mixed"]
    per["fair_diff_pp"] = (per["fair_mixed"] - per["fair_same_med"]) * 100
    R["btts_median_mixed_vs_same_book"] = {"snapshots": len(per), "orr_mixed": summarise(per["orr_mixed"]),
                                           "fair_diff_pp": summarise(per["fair_diff_pp"]),
                                           "abs_fair_diff_gt_1pp": int((per["fair_diff_pp"].abs() > 1).sum())}
    # ---- E. close staleness: production-equivalent close row (first archive in ARCH order with the
    # fixture, last row) vs kickoff (book_odds_snapshots, OddsAPI names == ledger names)
    ko = read_rolling("book_odds_snapshots.csv").drop_duplicates(["match_date", "match"], keep="last")
    kmap = {(r.match_date, r.match): r.kickoff_utc for r in ko.itertuples()}
    aidx = {k: g for k, g in a.groupby(["md", "market"])}
    st_rows = []
    for _, r in cl[cl["source"] == "live"].iterrows():
        k = kmap.get((str(r["match_date"])[:10], f"{r['home_team']} vs {r['away_team']}"))
        if k is None:
            continue
        g = aidx.get((str(r["match_date"])[:10], "under25" if r["side"] == "UNDER" else "over25"))
        if g is None:
            continue
        for arch in ARCH:
            h = g[g["archive"] == arch]
            h = h[[same_fixture(f"{r['home_team']} vs {r['away_team']}", m) for m in h["match"]]]
            if len(h):
                last = h.sort_values("ts").iloc[-1]
                st_rows.append({"archive": arch, "close_ts": last["ts"], "kickoff": pd.Timestamp(k),
                                "odds_match": bool(np.isclose(last["odds"], r["closing_odds"])),
                                "model_type": r["model_type"], "match_date": r["match_date"]})
                break
    sd = pd.DataFrame(st_rows)
    if len(sd):
        sd["min_before_ko"] = (utc(sd["kickoff"]) - utc(sd["close_ts"])).dt.total_seconds() / 60
        R["close_staleness_live"] = {"n": len(sd), "odds_reproduced": int(sd["odds_match"].sum()),
                                     "archive": sd["archive"].value_counts().to_dict(),
                                     "minutes_before_kickoff": summarise(sd["min_before_ko"]),
                                     "share_gt_60min": float((sd["min_before_ko"] > 60).mean()),
                                     "share_gt_360min": float((sd["min_before_ko"] > 360).mean()),
                                     "share_negative_post_ko": float((sd["min_before_ko"] < 0).mean())}
    save_json(R, "ledger_prices.json")
    import json
    print(json.dumps(R, indent=1, default=str)[:12000])


if __name__ == "__main__":
    with heavy_lock("AU4 ledger_prices"):
        main()
