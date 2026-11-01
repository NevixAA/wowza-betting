"""AU4 Phase 8 - player-prop CLV forensic (read-only, independent of production code).

Question: why did output/clv_records.csv (mean clv_pct ~+1.03%, beat-close 30.3%) produce the
verdict "POSITIVE CLV - genuine edge signal" (src/clv_capture.clv_report: verdict = mean > 0)?

Steps
 1. Reproduce the production aggregate from the committed file.
 2. Distribution: mean / median / trimmed / winsorized / %>0 / %=0 / %<0, plus two metrics that
    do not suffer the odds-ratio Jensen bias: mean log(odds_bet/odds_close) and the implied-prob
    move (1/odds_close - 1/odds_bet, positive = market moved toward us).
 3. Breakdowns: market, tier, league, odds band, source of the closing row, price age (entry
    hours before kickoff), month.  Fixture-block and day-block bootstrap CIs.
 4. Independent close reconstruction from the full prop history (live + archive):
      - same (player, market) and same FIXTURE (team-token match, +-1 day), as production;
      - production close  = last row by snapshot_ts (any time, any book);
      - strict close      = last row with snapshot_ts < kickoff (kickoff from git-history index);
      - same-book close   = for AF rows (bookmaker known): last pre-KO row of the entry book.
    Flags: production close row at/after kickoff, entry row after kickoff, entry price never seen
    in the history at or before ts_bet, close from a different source/book than the entry.
 5. Settled ROI of CLV records (flat 1u at odds_bet) with fixture-block CI, by tier.

Writes reports/au4_data/clv_*.csv|json. Usage: python scripts/audit/au4_clv_forensic.py
"""
from __future__ import annotations

import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from au4_common import (OUT, DATA, block_bootstrap, heavy_lock, read_rolling, save_csv,  # noqa: E402
                        save_json, summarise, utc)

STOP = {"fc", "cf", "sc", "ac", "afc", "cd", "ud", "sv", "fk", "bk", "if", "sk", "club", "de", "the",
        "1.", "1", "real", "atletico", "athletic", "sporting", "united", "city", "deportivo"}


def pnorm(s) -> str:
    t = unicodedata.normalize("NFKD", str(s or ""))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", t.lower().strip())


def team_tokens(s) -> set:
    t = re.sub(r"[^a-z0-9 ]+", " ", pnorm(s))
    return {w for w in t.split() if w not in STOP and len(w) > 1}


def same_team(a, b) -> bool:
    if pnorm(a) == pnorm(b):
        return True
    ta, tb = team_tokens(a), team_tokens(b)
    if ta and tb and (ta <= tb or tb <= ta):
        return True
    return SequenceMatcher(None, pnorm(a), pnorm(b)).ratio() >= 0.82


def split_match(m):
    p = str(m).split(" vs ")
    return (p[0].strip(), p[1].strip()) if len(p) == 2 else None


def same_fixture(a, b) -> bool:
    if str(a) == str(b):
        return True
    x, y = split_match(a), split_match(b)
    if not x or not y:
        return False
    return (same_team(x[0], y[0]) and same_team(x[1], y[1])) or (same_team(x[0], y[1]) and same_team(x[1], y[0]))


def odds_band(o):
    return pd.cut(o, [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 1e9],
                  labels=["<1.5", "1.5-2", "2-3", "3-5", "5-10", ">=10"], right=False)


def main() -> None:
    c = pd.read_csv(OUT / "clv_records.csv", low_memory=False)
    c["odds_bet"] = pd.to_numeric(c["odds_bet"], errors="coerce")
    c["odds_close"] = pd.to_numeric(c["odds_close"], errors="coerce")
    c["clv_pct"] = pd.to_numeric(c["clv_pct"], errors="coerce")
    parts = c["bet_id"].astype(str).str.split("|", n=3, expand=True)
    c["d"] = parts[1].str[:10]
    c["ts_bet_utc"] = utc(c["ts_bet"])
    c["tier"] = c["notes"].astype(str)
    closed = c[c["clv_pct"].notna()].copy()

    # 1. reproduce production aggregate
    prod = {"n": int(len(closed)), "mean_clv_pct": round(closed["clv_pct"].mean(), 4),
            "beat_close_rate": round((closed["clv_pct"] > 0).mean(), 3),
            "verdict_rule": "mean_clv_pct > 0  (src/clv_capture.py:113)",
            "records_total": int(len(c)), "records_never_closed": int(c["clv_pct"].isna().sum())}

    # 2. distribution + bias-free metrics
    closed["log_clv"] = np.log(closed["odds_bet"] / closed["odds_close"])
    closed["dprob"] = 1 / closed["odds_close"] - 1 / closed["odds_bet"]   # + = moved toward us
    dist = {"clv_pct": summarise(closed["clv_pct"]), "log_clv": summarise(closed["log_clv"]),
            "implied_prob_move": summarise(closed["dprob"])}
    moved = closed[closed["clv_pct"] != 0]
    dist["clv_pct_moved_only"] = summarise(moved["clv_pct"])
    dist["log_clv_moved_only"] = summarise(moved["log_clv"])

    # kickoff index
    ki = pd.read_csv(DATA / "prop_kickoff_index.csv")
    ki["kickoff_utc"] = utc(ki["kickoff_utc"])
    kmap = {}
    for r in ki.itertuples():
        kmap.setdefault(str(r.match), []).append((str(r.date)[:10], r.kickoff_utc, r.league))

    def ko_for(match, d):
        best = None
        for dd, ko, lg in kmap.get(str(match), []):
            if dd == d:
                return ko, lg
            if best is None and abs((pd.Timestamp(dd) - pd.Timestamp(d)).days) <= 1:
                best = (ko, lg)
        return best if best else (pd.NaT, None)

    kl = [ko_for(m, d) for m, d in zip(c["match"], c["d"])]
    c["kickoff_utc"] = [k for k, _ in kl]
    c["league"] = [lg for _, lg in kl]
    c["kickoff_utc"] = utc(c["kickoff_utc"])

    # 4. independent close reconstruction
    h = read_rolling("player_prop_odds_history.csv")
    h["ts"] = utc(h["snapshot_ts"])
    h["pk"] = h["player"].map(pnorm)
    h["odds"] = pd.to_numeric(h["odds"], errors="coerce")
    h["md"] = h["match_date"].astype(str).str[:10]
    h["source"] = h["source"].fillna("oddsapi")
    hg = {k: g for k, g in h.groupby(["pk", "market"])}
    fx_cache: dict = {}

    def fx_ok(tip_match, hist_match):
        k = (tip_match, hist_match)
        if k not in fx_cache:
            fx_cache[k] = same_fixture(tip_match, hist_match)
        return fx_cache[k]

    recs = []
    for r in c.itertuples():
        g = hg.get((pnorm(r.player), r.market))
        out = {"bet_id": r.bet_id}
        if g is None:
            out["hist_status"] = "NO_HISTORY_PLAYER_MARKET"; recs.append(out); continue
        dd = pd.Timestamp(r.d)
        g = g[(pd.to_datetime(g["md"], errors="coerce") - dd).abs() <= pd.Timedelta(days=1)]
        g = g[[fx_ok(r.match, m) for m in g["match"]]]
        if g.empty:
            out["hist_status"] = "NO_HISTORY_SAME_FIXTURE"; recs.append(out); continue
        g = g.sort_values("ts", kind="mergesort")
        exact_day = g[g["md"] == r.d]
        out["hist_status"] = "OK"
        out["prod_like_close"] = float(exact_day["odds"].iloc[-1]) if len(exact_day) else np.nan
        out["prod_like_close_ts"] = exact_day["ts"].iloc[-1] if len(exact_day) else pd.NaT
        out["prod_like_close_source"] = exact_day["source"].iloc[-1] if len(exact_day) else None
        out["n_hist_rows"] = len(g)
        out["n_sources"] = g["source"].nunique()
        out["n_books_af"] = g["bookmaker"].dropna().nunique()
        # entry availability: price equal to odds_bet seen at/before ts_bet (+10 min slack)
        tb = r.ts_bet_utc
        pre = g[g["ts"] <= tb + pd.Timedelta(minutes=10)] if pd.notna(tb) else g.iloc[0:0]
        out["entry_seen_by_ts_bet"] = bool((np.isclose(pre["odds"], r.odds_bet)).any())
        out["entry_seen_ever"] = bool((np.isclose(g["odds"], r.odds_bet)).any())
        out["best_pre_entry"] = float(pre["odds"].max()) if len(pre) else np.nan
        first_eq = g[np.isclose(g["odds"], r.odds_bet)]
        out["entry_row_source"] = first_eq["source"].iloc[0] if len(first_eq) else None
        out["entry_row_book"] = first_eq["bookmaker"].iloc[0] if len(first_eq) else None
        ko = r.kickoff_utc
        if pd.notna(ko):
            preko = g[g["ts"] < ko]
            out["n_post_ko_rows"] = int((g["ts"] >= ko).sum())
            out["strict_close"] = float(preko["odds"].iloc[-1]) if len(preko) else np.nan
            out["strict_close_ts"] = preko["ts"].iloc[-1] if len(preko) else pd.NaT
            out["strict_close_source"] = preko["source"].iloc[-1] if len(preko) else None
            out["prod_close_post_ko"] = bool(pd.notna(out["prod_like_close_ts"]) and out["prod_like_close_ts"] >= ko)
            # same-book (AF rows only; OddsAPI rows carry no book)
            eb = out["entry_row_book"]
            if isinstance(eb, str) and eb:
                sb = preko[preko["bookmaker"] == eb]
                out["samebook_close"] = float(sb["odds"].iloc[-1]) if len(sb) else np.nan
            # same-source close
            es = out["entry_row_source"]
            if isinstance(es, str):
                ss = preko[preko["source"] == es]
                out["samesource_close"] = float(ss["odds"].iloc[-1]) if len(ss) else np.nan
        recs.append(out)
    R = pd.DataFrame(recs)
    c = c.merge(R, on="bet_id", how="left")
    c["hours_before_ko"] = (c["kickoff_utc"] - c["ts_bet_utc"]).dt.total_seconds() / 3600
    c["entry_after_ko"] = c["hours_before_ko"] < 0
    for col, nm in (("strict_close", "clv_strict"), ("samebook_close", "clv_samebook"),
                    ("samesource_close", "clv_samesource"), ("prod_like_close", "clv_prodlike")):
        if col in c.columns:
            c[nm] = c["odds_bet"] / c[col] - 1
    c["prod_close_matches_recon"] = np.isclose(c["odds_close"], c.get("prod_like_close"))
    save_csv(c.drop(columns=["ts_bet_utc"]).assign(), "clv_records_reconstructed.csv")

    cl = c[c["clv_pct"].notna()].copy()
    cl["log_clv"] = np.log(cl["odds_bet"] / cl["odds_close"])
    cl["dprob"] = 1 / cl["odds_close"] - 1 / cl["odds_bet"]
    cl["fixture"] = cl["match"].astype(str) + "|" + cl["d"]
    cl["band"] = odds_band(cl["odds_bet"]).astype(str)
    cl["age"] = pd.cut(cl["hours_before_ko"], [-1e9, 0, 3, 12, 24, 72, 1e9],
                       labels=["post-KO", "0-3h", "3-12h", "12-24h", "24-72h", ">72h"]).astype(str)
    cl["month"] = cl["d"].str[:7]

    integ = {
        "closed_records": int(len(cl)),
        "kickoff_known": int(cl["kickoff_utc"].notna().sum()),
        "hist_status": cl["hist_status"].value_counts(dropna=False).to_dict(),
        "prod_close_reproduced": int(cl["prod_close_matches_recon"].sum()),
        "prod_close_row_at_or_after_kickoff": int(cl["prod_close_post_ko"].fillna(False).astype(bool).sum()),
        "entry_logged_after_kickoff": int(cl["entry_after_ko"].fillna(False).sum()),
        "entry_price_not_seen_by_ts_bet": int((cl["entry_seen_by_ts_bet"] == False).sum()),  # noqa: E712
        "entry_price_never_in_history": int((cl["entry_seen_ever"] == False).sum()),  # noqa: E712
        "close_source_differs_from_entry_source": int(((cl["prod_like_close_source"].notna()) & (cl["entry_row_source"].notna()) & (cl["prod_like_close_source"] != cl["entry_row_source"])).sum()),
        "records_with_multiple_sources": int((cl["n_sources"] > 1).sum()),
        "records_with_multiple_af_books": int((cl["n_books_af"] > 1).sum()),
        "strict_close_differs_from_prod": int((cl["strict_close"].notna() & ~np.isclose(cl["strict_close"], cl["odds_close"])).sum()),
        "clv_zero_share": float((cl["clv_pct"] == 0).mean()),
    }

    # 3. breakdowns with fixture-block CIs
    def tab(by):
        rows = []
        for k, g in cl.groupby(by, dropna=False):
            m, lo, hi, G = block_bootstrap(g["clv_pct"], g["fixture"], n_boot=1000)
            lm, llo, lhi, _ = block_bootstrap(g["log_clv"], g["fixture"], n_boot=1000)
            rows.append({by: k, "n": len(g), "fixtures": G, "mean_clv_pct": m, "ci_lo": lo, "ci_hi": hi,
                         "median": g["clv_pct"].median(), "pct_pos": (g["clv_pct"] > 0).mean(),
                         "pct_zero": (g["clv_pct"] == 0).mean(), "pct_neg": (g["clv_pct"] < 0).mean(),
                         "mean_log_clv": lm, "log_ci_lo": llo, "log_ci_hi": lhi,
                         "mean_dprob": g["dprob"].mean()})
        return pd.DataFrame(rows).sort_values("n", ascending=False)

    tabs = {}
    for by in ("market", "tier", "league", "band", "age", "month", "prod_like_close_source", "entry_row_source"):
        t = tab(by)
        tabs[by] = t
        save_csv(t, f"clv_by_{by}.csv")

    overall = {}
    for nm, col in (("clv_pct", "clv_pct"), ("log_clv", "log_clv"), ("dprob", "dprob")):
        overall[nm] = {"fixture_block": block_bootstrap(cl[col], cl["fixture"]),
                       "day_block": block_bootstrap(cl[col], cl["d"])}
    # alternative closes
    alt = {}
    for nm in ("clv_strict", "clv_samesource", "clv_samebook"):
        s = cl[[nm, "fixture"]].dropna()
        alt[nm] = {"summary": summarise(s[nm]), "fixture_block": block_bootstrap(s[nm], s["fixture"])}
    # clean subset: kickoff known, prod close pre-KO, entry pre-KO, entry seen by ts_bet, same source
    clean = cl[cl["kickoff_utc"].notna() & ~cl["entry_after_ko"].fillna(True)
               & (cl["entry_seen_by_ts_bet"] == True) & cl["clv_strict"].notna()]  # noqa: E712
    alt["clean_strict"] = {"n": len(clean), "summary": summarise(clean["clv_strict"]),
                           "fixture_block": block_bootstrap(clean["clv_strict"], clean["fixture"]),
                           "log": block_bootstrap(np.log1p(clean["clv_strict"]), clean["fixture"])}

    # outliers
    out = cl.reindex(cl["clv_pct"].abs().sort_values(ascending=False).index).head(25)[
        ["bet_id", "match", "market", "tier", "odds_bet", "odds_close", "clv_pct", "prod_like_close_source",
         "entry_row_source", "hours_before_ko", "result"]]
    save_csv(out, "clv_top_outliers.csv")
    top_share = cl["clv_pct"].clip(lower=0).sort_values(ascending=False)
    contrib = {"sum_clv_pct": cl["clv_pct"].sum(),
               "sum_top1pct_positive": top_share.head(max(1, len(cl) // 100)).sum(),
               "mean_without_top1pct_abs": cl.loc[~cl.index.isin(cl["clv_pct"].abs().sort_values(ascending=False).head(max(1, len(cl)//100)).index), "clv_pct"].mean()}

    # 5. settled ROI of CLV records
    st = c[c["result"].isin(["WIN", "LOSS"])].copy()
    st["pnl"] = pd.to_numeric(st["pnl"], errors="coerce")
    st["fixture"] = st["match"].astype(str) + "|" + st["d"]
    roi = {"all": {"n": len(st), "roi": st["pnl"].mean(), "fixture_block": block_bootstrap(st["pnl"], st["fixture"])}}
    for t, g in st.groupby("tier"):
        roi[t] = {"n": len(g), "roi": g["pnl"].mean(), "fixture_block": block_bootstrap(g["pnl"], g["fixture"], n_boot=1000)}
    stc = st[st["clv_pct"].notna()]
    roi["clv_pos_vs_neg"] = {
        "pos": {"n": int((stc.clv_pct > 0).sum()), "roi": stc.loc[stc.clv_pct > 0, "pnl"].mean()},
        "zero": {"n": int((stc.clv_pct == 0).sum()), "roi": stc.loc[stc.clv_pct == 0, "pnl"].mean()},
        "neg": {"n": int((stc.clv_pct < 0).sum()), "roi": stc.loc[stc.clv_pct < 0, "pnl"].mean()}}
    # settlement sanity: implied-prob vs hit rate
    st["imp"] = 1 / st["odds_bet"]
    roi["hit_vs_implied"] = {"hit_rate": (st["result"] == "WIN").mean(), "mean_implied": st["imp"].mean()}

    save_json({"production_reproduction": prod, "distribution": dist, "integrity": integ,
               "overall_ci": overall, "alternative_closes": alt, "outlier_contribution": contrib,
               "settled_roi": roi}, "clv_forensic.json")
    print(pd.Series(prod)); print(pd.Series(integ))
    for k, v in alt.items():
        print(k, v.get("fixture_block"))
    print("overall", overall)
    print("roi", {k: (v.get("n"), v.get("roi")) for k, v in roi.items() if isinstance(v, dict) and "roi" in v})


if __name__ == "__main__":
    with heavy_lock("AU4 clv_forensic"):
        main()
