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
    # Numeric CI bounds kept ALONGSIDE the display string. The string is right for the table (it
    # is a range, not a sortable quantity) but a chart needs numbers, and recomputing them
    # somewhere else would let the table and the chart disagree about the same cell.
    t["roi_lo"] = (100 * (t["roi"] - 1.96 * t["se"])).where(t["n"] >= 5)
    t["roi_hi"] = (100 * (t["roi"] + 1.96 * t["se"])).where(t["n"] >= 5)
    t = t.rename(columns={"_model": "model", "_market": "market", "_tier": "tier"})
    return t[["model", "market", "tier", "n", "wins", "win_pct", "pl_u",
              "roi_pct", "roi_ci95", "roi_lo", "roi_hi",
              "avg_odds"]].sort_values("n", ascending=False)


full = _table(d)
shown = full[full["n"] >= min_n]
hidden = full[full["n"] < min_n]

st.subheader("Model × market × tier")
st.dataframe(shown.drop(columns=["roi_lo", "roi_hi"], errors="ignore"),
             width="stretch", hide_index=True,
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
        st.dataframe(hidden, width="stretch", hide_index=True)

# The CI column is the point of the page, so say what it means rather than assuming it is read.
# ── ROI with its confidence interval, as a picture ────────────────────────────
# The CI is the whole point of this page and it was only ever a text column, which is the one
# format that makes an interval hard to compare across rows. Drawn with whiskers and a zero line,
# "most of these intervals span zero" stops being a caption the reader has to take on trust and
# becomes the obvious feature of the chart.
#
# Cells whose interval CLEARS zero are highlighted; everything else is muted, so the eye is drawn
# to the only cells the data can actually distinguish. With the current sample that is usually
# none, and showing that plainly is the honest outcome.
_ch = shown.dropna(subset=["roi_lo", "roi_hi"]).copy()
if len(_ch):
    _ch["cell"] = (_ch["model"].astype(str) + " · " + _ch["market"].astype(str)
                   + " · " + _ch["tier"].astype(str))
    _ch["decisive"] = (_ch["roi_lo"] > 0) | (_ch["roi_hi"] < 0)
    _ch = _ch.sort_values("roi_pct")
    try:
        import plotly.graph_objects as _go
        _fig = _go.Figure()
        for _dec, _grp in _ch.groupby("decisive"):
            _fig.add_trace(_go.Scatter(
                x=_grp["roi_pct"], y=_grp["cell"], mode="markers",
                marker=dict(size=11, color="#00c896" if _dec else "#7a7a7a",
                            symbol="diamond" if _dec else "circle"),
                error_x=dict(type="data", symmetric=False,
                             array=_grp["roi_hi"] - _grp["roi_pct"],
                             arrayminus=_grp["roi_pct"] - _grp["roi_lo"],
                             thickness=1.4, width=4,
                             color="#00c896" if _dec else "#6a6a6a"),
                name=("interval clears zero" if _dec else "interval spans zero"),
                hovertemplate="%{y}<br>ROI %{x:.1f}%<extra></extra>",
            ))
        _fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#999")
        _fig.update_layout(
            height=max(240, 34 * len(_ch)), margin=dict(l=8, r=8, t=8, b=8),
            xaxis_title="ROI % (95% CI)", yaxis_title=None,
            legend=dict(orientation="h", y=1.02, x=0),
            # No hardcoded background: the viewer's light/dark theme must win.
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(_fig, width="stretch")
        _n_dec = int(_ch["decisive"].sum())
        st.caption(f"{_n_dec} of {len(_ch)} cells have an interval that clears zero. "
                   f"The rest cannot yet be told apart from break-even, whatever their point "
                   f"estimate says.")
    except Exception as _e:                                  # noqa: BLE001
        st.caption(f"ROI interval chart unavailable ({type(_e).__name__}: {_e})")

_wide = shown[shown["roi_ci95"].str.contains("…", na=False)]
if not _wide.empty:
    st.info("**Read the CI, not the ROI.** With this few settled bets most intervals span zero, "
            "which means the data cannot yet distinguish a winning cell from a losing one. A +20% "
            "ROI on n=30 and a −20% on n=30 are frequently the same underlying truth.")

def _rollup(g: pd.DataFrame, by: str) -> None:
    """Roll-up table plus the one comparison that makes a hit rate mean anything.

    A win % on its own is unreadable. 39.8% is excellent at 3.00 and ruinous at 1.90 — the
    number carries no information until you know what price it was struck at. So the chart puts
    the ACHIEVED win rate next to the BREAK-EVEN win rate implied by the average odds actually
    taken (1 / avg_odds), which is the level the bets had to clear to not lose money.

    That comparison is also exactly what the P/L column already says, just in a form you can read
    at a glance and per group: bar above the marker means the group made money, below means it
    lost. Nothing here is a new claim — it is the existing arithmetic drawn instead of tabulated.
    """
    t = g.groupby(by).agg(n=("win", "size"), win_pct=("win", "mean"),
                          pl_u=("ret", "sum"), roi=("ret", "mean"),
                          avg_odds=("_odds", "mean"),
                          n_odds=("_odds", "count")).reset_index()
    t["win_pct"] = (100 * t["win_pct"]).round(1)
    t["roi_pct"] = (100 * t["roi"]).round(1)
    # Break-even is the requirement at the price paid, not a target we chose. But it is only
    # meaningful where prices actually EXIST, and coverage here is very uneven: the staked tiers
    # are 100% priced, while `assists` has 3 quotes across 13 settled bets. A mean over those 3
    # drawn as the break-even line for all 13 would be an invented reference level, so the marker
    # is suppressed below 80% coverage instead of being drawn from a fragment.
    #
    # `_odds.mean()` also skips NaN whereas `ret` substitutes 2.00 for a missing price, so on a
    # thinly-priced group the two would disagree about the same bets. Showing coverage makes that
    # visible rather than letting it quietly shift the comparison.
    t["odds_cov"] = (100.0 * t["n_odds"] / t["n"]).round(0)
    _ok = t["odds_cov"] >= 80
    t["need_pct"] = (100.0 / t["avg_odds"]).round(1).where(_ok)
    t = t.sort_values("n", ascending=False)
    st.dataframe(t[[by, "n", "win_pct", "need_pct", "odds_cov", "pl_u", "roi_pct"]],
                 width="stretch", hide_index=True,
                 column_config={
                     by: st.column_config.TextColumn(by.lstrip("_")),
                     "win_pct": st.column_config.NumberColumn(
                         "win %", format="%.1f", help="Actually achieved."),
                     "need_pct": st.column_config.NumberColumn(
                         "break-even %", format="%.1f",
                         help="1 / average odds taken. Below this the group loses money. "
                              "Blank where under 80% of the bets have a recorded price."),
                     "odds_cov": st.column_config.NumberColumn(
                         "priced %", format="%.0f%%",
                         help="Share of settled bets in this group with a recorded price. "
                              "Where this is low, P/L uses a 2.00 stand-in."),
                     "pl_u": st.column_config.NumberColumn("P/L (u)", format="%.1f"),
                     "roi_pct": st.column_config.NumberColumn("ROI %", format="%.1f"),
                 })
    try:
        import plotly.graph_objects as _go
        _f = _go.Figure()
        _f.add_trace(_go.Bar(x=t[by].astype(str), y=t["win_pct"], name="achieved",
                             # Grey where there is no trustworthy break-even to compare against:
                             # a red bar would assert "this lost" on a group whose reference
                             # level we just declined to compute.
                             marker_color=["#7a7a7a" if b != b
                                           else ("#00c896" if a >= b else "#e05c5c")
                                           for a, b in zip(t["win_pct"], t["need_pct"])],
                             hovertemplate="%{x}<br>achieved %{y:.1f}%<extra></extra>"))
        _be = t.dropna(subset=["need_pct"])
        _f.add_trace(_go.Scatter(x=_be[by].astype(str), y=_be["need_pct"], name="break-even",
                                 mode="markers",
                                 marker=dict(symbol="line-ew", size=26,
                                             line=dict(width=3, color="#d0d0d0")),
                                 hovertemplate="%{x}<br>needs %{y:.1f}%<extra></extra>"))
        _f.update_layout(height=260, margin=dict(l=8, r=8, t=8, b=8),
                         yaxis_title="win %", xaxis_title=None, bargap=0.35,
                         legend=dict(orientation="h", y=1.04, x=0),
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(_f, width="stretch", key=f"rollup_{by}")
    except Exception as _e:                                  # noqa: BLE001
        st.caption(f"chart unavailable ({type(_e).__name__})")

    # The roll-ups deliberately IGNORE the min-n filter above — the point of a roll-up is the
    # complete picture — but that lets a 9-bet group show a +100% ROI next to an 87-bet group at
    # -5%, with nothing on screen saying which one is real. Naming the thin groups costs one line
    # and stops the largest number on the chart being read as the best market.
    _thin = t[t["n"] < 20]
    if not _thin.empty:
        st.caption("Thin: " + ", ".join(f"{r[by]} n={int(r['n'])}" for _, r in _thin.iterrows())
                   + " — a single result moves these by a full unit.")


st.subheader("Roll-ups")
r1, r2 = st.columns(2)
with r1:
    st.markdown("**By market**")
    _rollup(d, "_market")
with r2:
    st.markdown("**By tier**")
    _rollup(d, "_tier")

st.caption(
    "Flat 1u stakes, matching v9's 'P/L in units' convention. Player props are labelled PAPER: "
    "invariant 2 makes them permanently unbet, so their rates are research rather than "
    "performance. Missing odds default to 2.00 for the return calculation, which is stated here "
    "because it flatters nothing but is still an assumption.")
