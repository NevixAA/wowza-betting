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


def _read_csv(*args, **kwargs):
    """pd.read_csv with paper leagues removed — they are collected, never counted.

    See config.PAPER_LEAGUES. Every KPI on this page reads through here so a paper
    league cannot reach a total by way of a path that forgot the rule.
    """
    import config as _cfg
    return _cfg.drop_paper_leagues(pd.read_csv(*args, **kwargs))
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import dashboard_ui as ui  # noqa: E402
from src.tier_labels import PAPER_TIER_LABEL, STAKED_LABEL, display_frame  # noqa: E402
from dashboard_data import freshness as _fresh, success_view as V  # noqa: E402
import importlib  # noqa: E402
V = importlib.reload(V)              # page + helper always the same version (see 15_League_Scout)

st.set_page_config(page_title="Success Rates | Wowza", page_icon="🎯", layout="wide")

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
            d = _read_csv(p, low_memory=False)
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
_SUB = ("Every rate with its n, staked tiers apart from recorded ones, and props labelled PAPER — the breakdown a "
        "single headline number hides.")
_EYEBROW = "Performance · success rates"
if df.empty:
    ui.page_header("Success rates", _SUB, eyebrow_text=_EYEBROW)
    ui.empty_state("NO_DATA", "No settled rows in any ledger yet.")
    st.stop()

cut = getattr(config, "PERFORMANCE_CUTOFF_DATE", None)
_newest = df["_date"].max()
_, _chip = _fresh.ts_badge(_newest if pd.notna(_newest) else None, "weekly",
                           f"newest settled {_newest:%d/%m}" if pd.notna(_newest) else "newest settled")
ui.page_header("Success rates", _SUB, eyebrow_text=_EYEBROW,
               freshness=ui.status_chip("NO_DATA", label="FLAT 1U · SETTLED ONLY", tone="NEUTRAL") + " " + _chip)

# ── tier windows (dashboard_access.WINDOWS, 2026-10-18): free = last 7 days of STAKED bets only;
# pro = 7/14/30/60/90 days; admin also "all time". The window is applied to the settled match date.
from dashboard_access import WINDOWS, current_tier  # noqa: E402
_tier = current_tier()
_wins = WINDOWS[_tier]
c1, c2 = st.columns([1.4, 1])
if len(_wins) > 1:
    with c1:
        window = st.radio("Period", _wins, horizontal=True, index=0,
                          format_func=lambda w: "All time" if w == 0 else f"Last {w} days")
else:
    window = _wins[0]
    st.caption(f"Last {window} days · Wowza Pro shows 7 / 14 / 30 / 60 / 90 days.")

if _tier == "free":
    use_cut, scope = bool(cut), f"STAKED only ({STAKED_LABEL})"
    with st.expander("Advanced filters"):
        min_n = st.number_input("Hide cells with fewer than n", 1, 100, 5, 1,
                                help="A rate on a handful of bets is noise. Hidden, not zeroed.")
else:
    with c2:
        scope = st.radio("Scope", [f"STAKED only ({STAKED_LABEL})", "All tiers"], index=0,
                         help=f"{PAPER_TIER_LABEL} is recorded for measurement and never bet. Including it in a "
                              "win rate describes something other than what the money did.")
    with st.expander("Advanced filters"):
        a1, a2 = st.columns(2)
        with a1:
            use_cut = st.checkbox(
                f"Apply PERFORMANCE_CUTOFF_DATE ({cut})", value=bool(cut),
                help="v9 counts P&L only from this date. Earlier rows still contribute CLV but predate "
                     "fixes, so pooling them misstates current performance.")
        with a2:
            min_n = st.number_input("Hide cells with fewer than n", 1, 100, 10, 1,
                                    help="A rate on a handful of bets is noise. Hidden, not zeroed.")

d = df.copy()
if use_cut and cut:
    d = d[d["_date"].isna() | (d["_date"] >= pd.to_datetime(cut))]
if window:
    _since = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=int(window))
    d = d[d["_date"] >= _since]
if scope.startswith("STAKED"):
    d = d[d["_tier"].isin(STAKED)]

st.caption(f"{len(d):,} settled observation(s) after filters · "
           f"sources: {d['_source'].value_counts().to_dict()}")
if d.empty:
    ui.empty_state("NO_DATA", "Nothing left after these filters.")
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
_ch = V.ci_frame(shown)

# ═══ STATUS — what this view can and cannot tell apart ═══════════════════════════════════════════
_n_dec = int(_ch["decisive"].sum()) if len(_ch) else 0
cc = st.columns(3)
ui.metric_card("Settled observations", f"{len(d):,}", title="In this view",
               supporting=[(k, f"{v:,}") for k, v in d["_source"].value_counts().to_dict().items()],
               n=len(d), container=cc[0])
ui.metric_card("Cells shown", f"{len(shown):,}", title="Model × market × tier",
               supporting=[("Hidden (n < " + str(min_n) + ")", f"{len(hidden):,}"),
                           ("Observations hidden", f"{int(hidden['n'].sum()):,}")], container=cc[1])
ui.metric_card("Intervals clearing zero", f"{_n_dec} of {len(_ch)}", title="Read the CI, not the ROI",
               supporting=[("Spanning zero", f"{len(_ch) - _n_dec}")],
               status=ui.status_chip("NO_DATA", tone="NEUTRAL",
                                     label="NOT YET DISTINGUISHABLE FROM BREAK-EVEN" if _n_dec == 0 else
                                     f"{_n_dec} CELL(S) DECISIVE"), container=cc[2])

ui.section_header("Model × market × tier")
ui.dataframe(shown.drop(columns=["roi_lo", "roi_hi"], errors="ignore"),
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
        ui.dataframe(hidden, width="stretch", hide_index=True)

# ── ROI with its confidence interval, as a picture ────────────────────────────
# The CI is the whole point of this page and it was only ever a text column, which is the one
# format that makes an interval hard to compare across rows. Drawn with whiskers and a zero line,
# "most of these intervals span zero" stops being a caption the reader has to take on trust and
# becomes the obvious feature of the chart. Cells whose interval CLEARS zero are drawn as filled
# diamonds in the primary colour, the rest as grey circles — shape and the legend word carry the
# reading, so colour is never the only cue.
if len(_ch):
    try:
        st.altair_chart(V.ci_chart(_ch), width="stretch")
        st.caption(f"{_n_dec} of {len(_ch)} cells have an interval that clears zero. "
                   f"The rest cannot yet be told apart from break-even, whatever their point "
                   f"estimate says.")
    except Exception as _e:                                  # noqa: BLE001
        st.caption(f"ROI interval chart unavailable ({type(_e).__name__}: {_e})")

_wide = shown[shown["roi_ci95"].str.contains("…", na=False)]
if not _wide.empty:
    st.caption("**Read the CI, not the ROI.** With this few settled bets most intervals span zero, "
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
    try:
        st.altair_chart(V.rollup_chart(display_frame(t), by), width="stretch", key=f"rollup_{by}")
    except Exception as _e:                                  # noqa: BLE001
        st.caption(f"chart unavailable ({type(_e).__name__})")
    ui.dataframe(t[[by, "n", "win_pct", "need_pct", "odds_cov", "pl_u", "roi_pct"]],
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

    # The roll-ups deliberately IGNORE the min-n filter above — the point of a roll-up is the
    # complete picture — but that lets a 9-bet group show a +100% ROI next to an 87-bet group at
    # -5%, with nothing on screen saying which one is real. Naming the thin groups costs one line
    # and stops the largest number on the chart being read as the best market.
    _thin = t[t["n"] < 20]
    if not _thin.empty:
        st.markdown(ui.status_chip("SMALL_SAMPLE", label=f"{len(_thin)} THIN GROUP(S) · n < 20"))
        st.caption("Thin: " + ", ".join(f"{r[by]} n={int(r['n'])}" for _, r in _thin.iterrows())
                   + " — a single result moves these by a full unit.")


ui.section_header("Roll-ups", "Achieved win rate against the break-even the prices demanded. Bar above its tick = "
                  "the group made money; the roll-ups ignore the min-n filter, so thin groups are named.")
r1, r2 = st.columns(2)
with r1:
    ui.eyebrow("By market")
    _rollup(d, "_market")
with r2:
    ui.eyebrow("By tier")
    _rollup(d, "_tier")

st.caption(
    "Flat 1u stakes, matching v9's 'P/L in units' convention. Player props are labelled PAPER: "
    "invariant 2 makes them permanently unbet, so their rates are research rather than "
    "performance. Missing odds default to 2.00 for the return calculation, which is stated here "
    "because it flatters nothing but is still an assumption.")
