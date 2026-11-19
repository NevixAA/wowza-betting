"""AU4 Phase 5 - bookmaker price integrity (read-only; no production imports).

For every committed price file in output/ this script measures, from the raw rows:
  * schema: which of fixture / league / teams / date / UTC kickoff / book / market / side / line /
    odds / timestamp the file actually carries;
  * odds sanity (<=1.0, <=1.05 placeholder, >50, NaN) and exact / key duplicates;
  * post-kickoff rows (snapshot_ts >= kickoff) where a kickoff is available or joinable;
  * two-sided pairing: same fixture + market + line + BOOK + SAME snapshot_ts -> overround
    distribution; and the cross-book "best Over + best Under" overround, to show what mixing
    books does (overround < 1.0 = an apparent arbitrage that no single book offered);
  * line mapping: same book + same ts, Over 1.5 < Over 2.5 < Over 3.5 (and Under reversed);
    BTTS Yes > 3.20 (first-half signature); prop thresholds sot < sot2 < sot3 < sot4;
  * the NF parser window (forward rows before 2026-08-10T08:00Z) re-derived independently.
Ledger checks (decision-time availability, closing-price provenance) are in
au4_ledger_prices.py. Writes reports/au4_data/price_integrity.json (+ csv tables).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from au4_common import OUT, DATA, heavy_lock, read_dir, read_rolling, save_csv, save_json, summarise, utc  # noqa: E402

FIELDS = {"fixture_id": ["fixture_id", "event_id", "_oa_event_id"], "league": ["league"],
          "teams": ["match", "home_team"], "date": ["match_date", "date"],
          "kickoff_utc": ["kickoff_utc", "commence_time"], "book": ["bookmaker", "bookmaker_title", "bookmaker_id"],
          "market": ["market"], "side": ["side", "outcome"], "line": ["line", "point"], "odds": ["odds", "price"],
          "snapshot_ts": ["snapshot_ts", "captured_at"]}
NF_PARSER_FIX = pd.Timestamp("2026-08-10T08:00:00Z")


def schema(d: pd.DataFrame) -> dict:
    return {k: next((c for c in v if c in d.columns), None) for k, v in FIELDS.items()}


def odds_sanity(o: pd.Series) -> dict:
    o = pd.to_numeric(o, errors="coerce")
    return {"n": int(len(o)), "nan": int(o.isna().sum()), "le_1.00": int((o <= 1.0).sum()),
            "le_1.05": int((o <= 1.05).sum()), "gt_50": int((o > 50).sum()),
            "median": float(o.median()) if o.notna().any() else None}


def pair_stats(d: pd.DataFrame, keys: list, side_col: str, a: str, b: str) -> dict:
    """Same-key (incl. book and ts) two-sided pairs -> overround."""
    x = d[d[side_col].isin([a, b])].copy()
    for k in keys:                                   # NaN keys (e.g. BTTS line) would be dropped
        if x[k].isna().any():
            x[k] = x[k].astype(object).where(x[k].notna(), "NA")
    w = x.pivot_table(index=keys, columns=side_col, values="odds", aggfunc="last")
    if a not in w.columns or b not in w.columns:
        return {"pairs": 0}
    w = w.dropna(subset=[a, b])
    orr = 1 / w[a] + 1 / w[b]
    return {"pairs": int(len(w)), "rows_in_scope": int(len(x)),
            "pair_rate_rows": float(2 * len(w) / max(1, len(x))),
            "overround": summarise(orr), "lt_1.00": int((orr < 1.0).sum()), "gt_1.15": int((orr > 1.15).sum())}


def best_cross_book(d: pd.DataFrame, keys_nobook: list, side_col: str, a: str, b: str) -> dict:
    x = d[d[side_col].isin([a, b])].copy()
    for k in keys_nobook:
        if x[k].isna().any():
            x[k] = x[k].astype(object).where(x[k].notna(), "NA")
    w = x.pivot_table(index=keys_nobook, columns=side_col, values="odds", aggfunc="max").dropna()
    if a not in w.columns or b not in w.columns or w.empty:
        return {"pairs": 0}
    orr = 1 / w[a] + 1 / w[b]
    return {"pairs": int(len(w)), "overround": summarise(orr), "lt_1.00": int((orr < 1.0).sum()),
            "share_lt_1.00": float((orr < 1.0).mean())}


def monotone(d: pd.DataFrame, keys: list, col_by_line: dict, increasing: bool) -> dict:
    """Over prices must increase with the line; Under prices must decrease."""
    w = d.pivot_table(index=keys, columns="_lk", values="odds", aggfunc="last")
    cols = [c for c in col_by_line if c in w.columns]
    bad = pd.Series(False, index=w.index)
    n_cmp = pd.Series(0, index=w.index)
    for i in range(len(cols) - 1):
        lo, hi = w[cols[i]], w[cols[i + 1]]
        ok = lo.notna() & hi.notna()
        n_cmp += ok
        bad |= ok & ((lo >= hi) if increasing else (lo <= hi))
    has = n_cmp > 0
    return {"groups_compared": int(has.sum()), "violations": int((bad & has).sum()),
            "violation_rate": float((bad & has).sum() / max(1, has.sum()))}


def post_ko(ts: pd.Series, ko: pd.Series) -> dict:
    t, k = utc(ts), utc(ko)
    known = t.notna() & k.notna()
    post = known & (t >= k)
    return {"kickoff_known": int(known.sum()), "post_kickoff_rows": int(post.sum()),
            "post_kickoff_share": float(post.sum() / max(1, known.sum()))}


def main() -> None:
    R: dict = {}
    # ---------------- book_quotes (API-Football, per book, both sides) ----------------
    q = read_dir("book_quotes")
    q["odds"] = pd.to_numeric(q["odds"], errors="coerce")
    q["ts"] = utc(q["snapshot_ts"])
    q["mtk"] = pd.to_numeric(q["minutes_to_kickoff"], errors="coerce")
    r = {"rows": len(q), "span": [str(q["ts"].min()), str(q["ts"].max())], "schema": schema(q),
         "odds": odds_sanity(q["odds"]), "post_ko": post_ko(q["snapshot_ts"], q["kickoff_utc"]),
         "minutes_to_kickoff_le_0": int((q["mtk"] <= 0).sum()),
         "ladder_band": q["ladder_band"].value_counts().to_dict(),
         "exact_dups": int(q.duplicated(subset=[c for c in q.columns if c != "_part"]).sum()),
         "conflicting_same_key_ts": int(q.duplicated(subset=["snapshot_ts", "fixture_id", "bookmaker", "market", "side", "line"], keep=False).sum()
                                        - q.duplicated(subset=["snapshot_ts", "fixture_id", "bookmaker", "market", "side", "line", "odds"], keep=False).sum())}
    kb = ["fixture_id", "bookmaker", "market", "line", "snapshot_ts"]
    knb = ["fixture_id", "market", "line", "snapshot_ts"]
    for mk, a, b in (("ou", "over", "under"), ("btts", "yes", "no"), ("ht_ou", "over", "under")):
        s = q[q["market"] == mk]
        r[f"pairs_samebook_{mk}"] = pair_stats(s, kb, "side", a, b)
        r[f"pairs_bestcross_{mk}"] = best_cross_book(s, knb, "side", a, b)
    ou = q[q["market"] == "ou"].copy()
    ou["_lk"] = ou["line"].astype(float)
    r["monotone_over"] = monotone(ou[ou["side"] == "over"], ["fixture_id", "bookmaker", "snapshot_ts"], [1.5, 2.5, 3.5], True)
    r["monotone_under"] = monotone(ou[ou["side"] == "under"], ["fixture_id", "bookmaker", "snapshot_ts"], [1.5, 2.5, 3.5], False)
    bt = q[(q["market"] == "btts") & (q["side"] == "yes")]
    r["btts_yes_gt_3.20"] = int((bt["odds"] > 3.2).sum())
    r["btts_yes_n"] = int(len(bt))
    # by-book overround (same book same ts, OU 2.5)
    s = q[(q["market"] == "ou") & (q["line"] == 2.5)]
    w = s.pivot_table(index=["fixture_id", "bookmaker", "snapshot_ts"], columns="side", values="odds", aggfunc="last").dropna()
    w["orr"] = 1 / w["over"] + 1 / w["under"]
    bybook = w.groupby(level="bookmaker")["orr"].agg(["size", "median", "mean"]).reset_index()
    save_csv(bybook, "price_bookquotes_ou25_overround_by_book.csv")
    R["book_quotes"] = r

    # ---------------- book_odds_snapshots (OddsAPI eu, per book) ----------------
    b = read_rolling("book_odds_snapshots.csv")
    b["odds"] = pd.to_numeric(b["odds"], errors="coerce")
    r = {"rows": len(b), "span": [str(utc(b["snapshot_ts"]).min()), str(utc(b["snapshot_ts"]).max())],
         "schema": schema(b), "odds": odds_sanity(b["odds"]), "post_ko": post_ko(b["snapshot_ts"], b["kickoff_utc"]),
         "books": b["bookmaker"].nunique(), "markets": b["market"].value_counts().to_dict()}
    # change-log: pair same ts is a lower bound; also LOCF pairing per (match, book, market)
    kb = ["match_date", "match", "bookmaker", "market", "snapshot_ts"]
    r["sides"] = {f"{a}|{s}": int(v) for (a, s), v in b.groupby(["market", "side"]).size().items()}
    r["pairs_samebook_same_ts"] = pair_stats(b, kb, "side", "OVER", "UNDER")
    r["pairs_samebook_same_ts_btts"] = pair_stats(b[b["market"] == "BTTS"], kb, "side", "YES", "NO")
    wide = b[b["market"] != "BTTS"].pivot_table(index=kb, columns="side", values="odds", aggfunc="last").dropna()
    wide["orr"] = 1 / wide["OVER"] + 1 / wide["UNDER"]
    save_csv(wide[wide["orr"] > 1.15].reset_index().head(200), "price_bookodds_overround_gt_115.csv")
    r["pairs_bestcross_same_ts"] = best_cross_book(b, ["match_date", "match", "market", "snapshot_ts"], "side", "OVER", "UNDER")
    b2 = b.copy()
    b2["_lk"] = b2["market"].map({"OU15": 1.5, "OU25": 2.5, "OU35": 3.5})
    r["monotone_over"] = monotone(b2[b2["side"] == "OVER"], ["match", "bookmaker", "snapshot_ts"], [1.5, 2.5, 3.5], True)
    R["book_odds_snapshots"] = r
    ko_oa = b.drop_duplicates(["match_date", "match"])[["match_date", "match", "kickoff_utc"]]

    # ---------------- standard_odds_history / newformat_odds_dense (first-listed book) ----------
    for name in ("standard_odds_history.csv", "newformat_odds_dense.csv"):
        d = read_rolling(name)
        d["odds"] = pd.to_numeric(d["odds"], errors="coerce")
        r = {"rows": len(d), "span": [str(d["snapshot_ts"].min()), str(d["snapshot_ts"].max())], "schema": schema(d),
             "odds": odds_sanity(d["odds"]), "markets": d["market"].value_counts().to_dict(),
             "book_identity": "NOT RECORDED - first OddsAPI book listing each price (src/predict.py `not ov25` guards)"}
        d2 = d.copy()
        d2["side"] = d2["market"].str.replace(r"\d+", "", regex=True)
        d2["ln"] = d2["market"].str.extract(r"(\d+)")[0]
        r["pairs_ou25_same_ts"] = pair_stats(d2[d2["ln"] == "25"], ["match_date", "match", "snapshot_ts"], "side", "over", "under")
        j = d.merge(ko_oa, on=["match_date", "match"], how="left")
        r["post_ko_via_book_odds_snapshots"] = post_ko(j["snapshot_ts"], j["kickoff_utc"])
        pk = j[utc(j["snapshot_ts"]) >= utc(j["kickoff_utc"])].copy()
        pk["minutes_after_ko"] = (utc(pk["snapshot_ts"]) - utc(pk["kickoff_utc"])).dt.total_seconds() / 60
        r["post_ko_minutes_after"] = summarise(pk["minutes_after_ko"])
        save_csv(pk, f"price_postko_{Path(name).stem}.csv")
        R[name] = r

    # ---------------- AF Bet365 single-book captures ----------------
    for name in ("standard_sidemarket_odds_history.csv", "newformat_odds_history.csv"):
        d = read_rolling(name)
        d["odds"] = pd.to_numeric(d["odds"], errors="coerce")
        backfill = d["snapshot_ts"].astype(str).str.endswith("T00:00:00Z")
        ts = utc(d["snapshot_ts"])
        r = {"rows": len(d), "schema": schema(d), "odds": odds_sanity(d["odds"]),
             "backfill_rows_T00": int(backfill.sum()), "forward_rows": int((~backfill).sum()),
             "markets": d["market"].value_counts().to_dict(),
             "book_identity": "Bet365 only (API-Football bookmaker=8) for forward rows; backfill = OddsAPI historical main line",
             "post_ko": post_ko(d.loc[~backfill, "snapshot_ts"], d.loc[~backfill, "kickoff_utc"]) if "kickoff_utc" in d.columns else None,
             "kickoff_blank_forward": int(d.loc[~backfill, "kickoff_utc"].isna().sum()) if "kickoff_utc" in d.columns else None}
        ou_m = d["market"].isin(["over15", "under15", "over25", "under25", "over35", "under35"])
        r["forward_ou_rows_before_2026-08-10T08Z"] = int((ou_m & ~backfill & (ts < NF_PARSER_FIX)).sum())
        r["btts_yes_gt_3.20"] = int(((d["market"] == "btts_yes") & (d["odds"] > 3.2)).sum())
        d2 = d[~backfill].copy()
        d2["side"] = d2["market"].str.extract(r"^(over|under|btts_yes|btts_no|ht_over|ht_under)")[0]
        d2["ln"] = d2["market"].str.extract(r"(\d+)$")[0]
        for ln in ("15", "25", "35"):
            r[f"pairs_ou{ln}_same_ts"] = pair_stats(d2[(d2["ln"] == ln) & d2["side"].isin(["over", "under"])],
                                                   ["match_date", "match", "snapshot_ts"], "side", "over", "under")
        bt = d2[d2["market"].isin(["btts_yes", "btts_no"])]
        r["pairs_btts_same_ts"] = pair_stats(bt, ["match_date", "match", "snapshot_ts"], "market", "btts_yes", "btts_no")
        # same pairing but only post parser fix (valid window)
        post = d2[utc(d2["snapshot_ts"]) >= NF_PARSER_FIX]
        r["pairs_ou25_same_ts_post_fix"] = pair_stats(post[(post["ln"] == "25") & post["side"].isin(["over", "under"])],
                                                      ["match_date", "match", "snapshot_ts"], "side", "over", "under")
        pre = d2[utc(d2["snapshot_ts"]) < NF_PARSER_FIX]
        r["pairs_ou25_same_ts_pre_fix"] = pair_stats(pre[(pre["ln"] == "25") & pre["side"].isin(["over", "under"])],
                                                     ["match_date", "match", "snapshot_ts"], "side", "over", "under")
        o = d2[d2["side"] == "over"].copy()
        o["_lk"] = o["ln"].astype(float)
        r["monotone_over"] = monotone(o, ["match_date", "match", "snapshot_ts"], [15.0, 25.0, 35.0], True)
        o_post = o[utc(o["snapshot_ts"]) >= NF_PARSER_FIX]
        r["monotone_over_post_fix"] = monotone(o_post, ["match_date", "match", "snapshot_ts"], [15.0, 25.0, 35.0], True)
        R[name] = r

    # ---------------- exchange quotes ----------------
    e = read_dir("exchange_quotes")
    if len(e):
        e["price"] = pd.to_numeric(e["price"], errors="coerce")
        R["exchange_quotes"] = {"rows": len(e), "schema": schema(e), "odds": odds_sanity(e["price"]),
                                "post_ko": post_ko(e["captured_at"], e["commence_time"]),
                                "markets": e["market"].value_counts().to_dict(),
                                "books": e["bookmaker_key"].value_counts().to_dict(),
                                "last_update_after_commence": int((utc(e["last_update"]) >= utc(e["commence_time"])).sum())}

    # ---------------- props ----------------
    ki = pd.read_csv(DATA / "prop_kickoff_index.csv")
    pq_ = read_dir("prop_quotes")
    pq_["odds"] = pd.to_numeric(pq_["odds"], errors="coerce")
    r = {"rows": len(pq_), "schema": schema(pq_), "odds": odds_sanity(pq_["odds"]),
         "by_market_side": {f"{a}|{b}": int(v) for (a, b), v in pq_.groupby(["market", "side"]).size().items()},
         "one_sided_share": float(1 - pq_["side"].isin(["no", "under"]).mean()),
         "le_1.05_by_market": pq_[pq_["odds"] <= 1.05]["market"].value_counts().to_dict()}
    # threshold monotonicity sot < sot2 < sot3 < sot4 per player/book/ts
    s = pq_[pq_["market"].isin(["sot", "sot2", "sot3", "sot4"]) & (pq_["side"] == "over")].copy()
    s["_lk"] = s["market"].map({"sot": 1, "sot2": 2, "sot3": 3, "sot4": 4})
    r["monotone_sot"] = monotone(s, ["match", "player", "bookmaker", "snapshot_ts"], [1, 2, 3, 4], True)
    # cross-check point->market mapping
    r["point_market_mismatch"] = int(((pq_["market"] == "sot") & (pq_["point"] != 0.5)).sum()
                                     + ((pq_["market"] == "sot2") & (pq_["point"] != 1.5)).sum()
                                     + ((pq_["market"] == "sot3") & (pq_["point"] != 2.5)).sum()
                                     + ((pq_["market"] == "sot4") & (pq_["point"] != 3.5)).sum())
    j = pq_.merge(ki[["match", "date", "kickoff_utc"]].rename(columns={"date": "match_date"}), on=["match", "match_date"], how="left")
    r["post_ko_via_kickoff_index"] = post_ko(j["snapshot_ts"], j["kickoff_utc"])
    R["prop_quotes"] = r
    h = read_rolling("player_prop_odds_history.csv")
    h["odds"] = pd.to_numeric(h["odds"], errors="coerce")
    h["source"] = h["source"].fillna("oddsapi")
    r = {"rows": len(h), "schema": schema(h), "odds": odds_sanity(h["odds"]),
         "by_source": h["source"].value_counts().to_dict(),
         "bookmaker_blank_by_source": h[h["bookmaker"].isna()]["source"].value_counts().to_dict(),
         "dedup_key": "match_date, league, match, player, market, odds  (NO bookmaker, NO side; a price that reverts A->B->A is not re-recorded)",
         "le_1.05_by_market": h[h["odds"] <= 1.05]["market"].value_counts().to_dict()}
    hs = h[h["market"].isin(["sot", "sot2", "sot3", "sot4"])].copy()
    hs["_lk"] = hs["market"].map({"sot": 1, "sot2": 2, "sot3": 3, "sot4": 4})
    r["monotone_sot_same_ts"] = monotone(hs, ["match", "player", "snapshot_ts"], [1, 2, 3, 4], True)
    j = h.merge(ki[["match", "date", "kickoff_utc"]].rename(columns={"date": "match_date"}), on=["match", "match_date"], how="left")
    r["post_ko_via_kickoff_index"] = post_ko(j["snapshot_ts"], j["kickoff_utc"])
    R["player_prop_odds_history"] = r

    # ---------------- historical backtest price provenance ----------------
    bk = {}
    for nm, col in (("backtest_results_over15", "odds_over15"), ("backtest_results_over35", "odds_over35"),
                    ("backtest_results_btts", "odds_btts"), ("backtest_results_standard", "odds_over25"),
                    ("backtest_results_newformat", "odds_over25")):
        p = OUT / f"{nm}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p, low_memory=False)
        x = {"rows": len(d)}
        if "price_synthetic" in d.columns:
            syn = d["price_synthetic"].astype(str).str.lower().isin(("true", "1"))
            x["price_synthetic_true"] = int(syn.sum())
            if "bet" in d.columns:
                placed = d["bet"].astype(str).str.upper().isin(("TRUE", "1", "YES", "OVER", "UNDER")) | (pd.to_numeric(d.get("bet_stake"), errors="coerce") > 0)
                x["placed"] = int(placed.sum()); x["placed_synthetic"] = int((placed & syn).sum())
        x["modal_price"] = float(pd.to_numeric(d[col], errors="coerce").mode().iloc[0])
        x["modal_price_share"] = float((pd.to_numeric(d[col], errors="coerce") == x["modal_price"]).mean())
        bk[nm] = x
    R["backtest_prices"] = bk
    save_json(R, "price_integrity.json")
    import json
    print(json.dumps(R, indent=1, default=str)[:20000])


if __name__ == "__main__":
    with heavy_lock("AU4 price_integrity"):
        main()
