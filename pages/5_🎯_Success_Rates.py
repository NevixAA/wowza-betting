"""Success rates by MODEL x MARKET x TIER — the breakdown a single headline number hides.

WHY THIS PAGE EXISTS. Performance was only ever shown as one aggregate, or filtered one dimension
at a time. That pools things which behave nothing alike: a SNIPER on the main O/U line and a
VALUABLE player-props card sit in the same average, and props are paper-only by invariant 2 while
O/U is real money. A single "% success" across all of it cannot inform any decision.

THREE LEDGERS, ONE SCHEMA. Main O/U has no `market` column (it is implicitly OU25); props have no
`model_type`. Both are filled in explicitly here rather than left to a silent NaN that would form
its own bucket.

WHAT IT REFUSES TO DO:
  * No bare percentages. Every rate carries its n, because "67% success" on three bets is noise
    wearing a number's clothes.
  * STAKED (SNIPER/MARKSMAN) is separated from RECORDED (VALUABLE and below). VALUABLE is tracked
    for measurement and never bet, so folding it into a win rate misstates what the money did.
  * Player props are labelled PAPER throughout. Invariant 2: the props model is accurate but has no
    betting edge, so its rates are research, not performance.
  * PERFORMANCE_CUTOFF_DATE is applied by default and named on screen. Pre-cutoff rows exist to
    contribute CLV, not P&L.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

st.set_page_config(page_title="Success Rates | Wowza", page_icon="🎯", layout="wide")
st.title("🎯 Success rates — model × market × tier")

STAKED = ("SNIPER", "MARKSMAN")
_WIN = ("WIN",)
_LOSS = ("LOSS",)


@st.cache_data(ttl=120)
def _load() -> pd.DataFrame:
    """Union the ledgers into (model, market, tier, result, odds, date)."""
    out = []

    def _grab(path, *, model_col=None, model_const=None, market_col=None,
              market_const=None, tier_col="signal_tier", label=""):
        p = config.OUTPUT_DIR / path
        if not p.exists():
            return
        try:
            d = pd.read_csv(p, low_memory=False)
        except Exception:
            return
        if tier_col not in d.columns or "result" not in d.columns:
            return
        res = d["result"].astype(str).str.upper()
        d = d[res.isin(_WIN + _LOSS)].copy()
        if d.empty:
            return
        d["_res"] = d["result"].astype(str).str.upper()
        d["_tier"] = d[tier_col].astype(str)
        d["_model"] = (d[model_col].astype(str) if model_col and model_col in d.columns
                       else model_const or "unknown")
        d["_market"] = (d[market_col].astype(str) if market_col and market_col in d.columns
                        else market_const or "unknown")
        dc = next((c for c in ("match_date", "date", "signal_date") if c in d.columns), None)
        d["_date"] = pd.to_datetime(d[dc], errors="coerce") if dc else pd.NaT
        oc = next((c for c in ("odds", "entry_odds", "market_odds") if c in d.columns), None)
        d["_odds"] = pd.to_numeric(d[oc], errors="coerce") if oc else np.nan
        d["_source"] = label
        out.append(d[["_source", "_model", "_market", "_tier", "_res", "_odds", "_date"]])

    # Main O/U carries no `market` column — it IS the OU25 line. Named explicitly so it never
    # lands in an "unknown" bucket next to a genuinely missing value.
    _grab("bets_ledger.csv", model_col="model_type", market_const="OU25", label="team O/U")
    _grab("side_bets_ledger.csv", model_col="model_type", market_col="market", label="side market")
    # Props have no model_type. Labelled PAPER because invariant 2 makes them permanently unbet.
    _grab("player_ledger.csv", model_const="player_props (PAPER)", market_col="market",
          tier_col="tier", label="player props")
    _grab("ht_ledger.csv", model_col="model_type", market_const="HT O/U", label="half-time")
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)
    df["win"] = (df["_res"] == "WIN").astype(int)
    # Flat 1u return, matching v9's "P/L in units" convention rather than an invented stake plan.
    df["ret"] = np.where(df["win"] == 1, df["_odds"].fillna(2.0) - 1.0, -1.0)
    return df


df = _load()
if df.empty:
    st.info("No settled rows in any ledger yet.")
    st.stop()

cut = getattr(config, "PERFORMANCE_CUTOFF_DATE", None)
c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    use_cut = st.checkbox(
        f"Apply PERFORMANCE_CUTOFF_DATE ({cut})", value=bool(cut),
        help="v9 counts P&L only from this date. Earlier rows still contribute CLV but predate "
             "fixes, so pooling them misstates current performance.")
with c2:
    scope = st.radio("Scope", ["STAKED only (SNIPER/MARKSMAN)", "All tiers"], index=0,
                     help="VALUABLE is recorded for measurement and never bet. Including it in a "
                          "win rate describes something other than what the money did.")
with c3:
    min_n = st.number_input("Hide cells with fewer than n", 1, 100, 10, 1,
                            help="A rate on a handful of bets is noise. Hidden, not zeroed.")

d = df.copy()
if use_cut and cut:
    d = d[d["_date"].isna() | (d["_date"] >= pd.to_datetime(cut))]
if scope.startswith("STAKED"):
    d = d[d["_tier"].isin(STAKED)]

st.caption(f"{len(d):,} settled observation(s) after filters · "
           f"sources: {d['_source'].value_counts().to_dict()}")
if d.empty:
    st.warning("Nothing left after these filters.")
    st.stop()


def _table(g: pd.DataFrame) -> pd.DataFrame:
    t = g.groupby(["_model", "_market", "_tier"]).agg(
        n=("win", "size"), wins=("win", "sum"),
        pl_u=("ret", "sum"), roi=("ret", "mean"),
        avg_odds=("_odds", "mean")).reset_index()
    t["win_pct"] = (100 * t["wins"] / t["n"]).round(1)
    t["roi_pct"] = (100 * t["roi"]).round(1)
    # Standard error on the mean return: the honest width around every ROI figure.
    se = g.groupby(["_model", "_market", "_tier"])["ret"].sem().reset_index(name="se")
    t = t.merge(se, on=["_model", "_market", "_tier"], how="left")
    t["roi_ci95"] = t.apply(
        lambda r: "—" if pd.isna(r["se"]) or r["n"] < 5
        else f"{100*(r['roi']-1.96*r['se']):+.0f}% … {100*(r['roi']+1.96*r['se']):+.0f}%", axis=1)
    t = t.rename(columns={"_model": "model", "_market": "market", "_tier": "tier"})
    return t[["model", "market", "tier", "n", "wins", "win_pct", "pl_u",
              "roi_pct", "roi_ci95", "avg_odds"]].sort_values("n", ascending=False)


full = _table(d)
shown = full[full["n"] >= min_n]
hidden = full[full["n"] < min_n]

st.subheader("Model × market × tier")
st.dataframe(shown, use_container_width=True, hide_index=True,
             column_config={
                 "win_pct": st.column_config.NumberColumn("win %", format="%.1f"),
                 "pl_u": st.column_config.NumberColumn("P/L (u)", format="%.1f"),
                 "roi_pct": st.column_config.NumberColumn("ROI %", format="%.1f"),
                 "roi_ci95": st.column_config.TextColumn("ROI 95% CI"),
                 "avg_odds": st.column_config.NumberColumn("avg odds", format="%.2f"),
             })
if not hidden.empty:
    st.caption(f"{len(hidden)} cell(s) hidden for n < {min_n} "
               f"({int(hidden['n'].sum())} observations). Hidden rather than shown at zero — an "
               f"empty cell reads as 'no data', a 0% reads as 'it lost'.")
    with st.expander("Show the thin cells anyway"):
        st.dataframe(hidden, use_container_width=True, hide_index=True)

# The CI column is the point of the page, so say what it means rather than assuming it is read.
_wide = shown[shown["roi_ci95"].str.contains("…", na=False)]
if not _wide.empty:
    st.info("**Read the CI, not the ROI.** With this few settled bets most intervals span zero, "
            "which means the data cannot yet distinguish a winning cell from a losing one. A +20% "
            "ROI on n=30 and a −20% on n=30 are frequently the same underlying truth.")

st.subheader("Roll-ups")
r1, r2 = st.columns(2)
with r1:
    st.markdown("**By market**")
    m = d.groupby("_market").agg(n=("win", "size"), win_pct=("win", "mean"),
                                 pl_u=("ret", "sum"), roi=("ret", "mean")).reset_index()
    m["win_pct"] = (100 * m["win_pct"]).round(1)
    m["roi_pct"] = (100 * m["roi"]).round(1)
    st.dataframe(m[["_market", "n", "win_pct", "pl_u", "roi_pct"]].sort_values("n", ascending=False),
                 use_container_width=True, hide_index=True)
with r2:
    st.markdown("**By tier**")
    t = d.groupby("_tier").agg(n=("win", "size"), win_pct=("win", "mean"),
                               pl_u=("ret", "sum"), roi=("ret", "mean")).reset_index()
    t["win_pct"] = (100 * t["win_pct"]).round(1)
    t["roi_pct"] = (100 * t["roi"]).round(1)
    st.dataframe(t[["_tier", "n", "win_pct", "pl_u", "roi_pct"]].sort_values("n", ascending=False),
                 use_container_width=True, hide_index=True)

st.caption(
    "Flat 1u stakes, matching v9's 'P/L in units' convention. Player props are labelled PAPER: "
    "invariant 2 makes them permanently unbet, so their rates are research rather than "
    "performance. Missing odds default to 2.00 for the return calculation, which is stated here "
    "because it flatters nothing but is still an assumption.")
