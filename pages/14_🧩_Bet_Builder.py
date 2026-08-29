"""
Bet Builder & Multi — combo candidates from Pro, with results.

WHERE THE DATA COMES FROM, AND WHY THIS PAGE IS IN v9

The combos are generated in v9-Pro (`wowzaV9-Pro`), which is where all the joint-probability
research lives. This page only READS Pro's committed output; it computes nothing and writes
nothing. It sits in v9 because that is where the dashboard is and where anyone would look for
it, and because the dashboard is referenced by NONE of v9's ~20 workflows — so a page here
cannot affect notify, predict or collect. That is the same reasoning behind every other
presentation change in this app.

Pro is expected as a sibling checkout (`../wowzaV9-Pro` or `../v10`). If it is absent the page
says so plainly rather than rendering an empty grid that looks like "no combos today".

WHAT THE NUMBERS MEAN, STATED ON SCREEN

* `fair_odds` is OUR price. No bookmaker sells a same-game-builder price we collect, so there is
  no market price to compare against and no EV. The comparison a reader makes is against their
  own book.
* `naive` is what multiplying the legs would ask. The gap between the two IS the research
  finding — on a four-leg builder it reached 6.7x.
* P/L is settled at the FAIR price, so it measures whether the probabilities were right, not
  whether anyone could have won that much.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

import dashboard_ui as ui

st.set_page_config(page_title="Bet Builder | Wowza", page_icon="🧩", layout="wide")
ui.autorefresh(minutes=5, key="builder_refresh")

BASE_DIR = Path(__file__).resolve().parents[1]
# Pro checkout, tried under both names it goes by locally.
_PRO_CANDIDATES = [BASE_DIR.parent / "wowzaV9-Pro", BASE_DIR.parent / "v10"]
PRO = next((p for p in _PRO_CANDIDATES if (p / "output").exists()), None)

ui.page_header("🧩 Bet Builder & Multi",
               "Same-match builders and cross-match multiples, priced with measured correlation",
               badge="RESEARCH · PAPER")


@st.cache_data(ttl=300, show_spinner=False)
def _load(name: str) -> pd.DataFrame:
    if PRO is None:
        return pd.DataFrame()
    p = PRO / "output" / name
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p, low_memory=False)
    except Exception:
        return pd.DataFrame()


if PRO is None:
    st.warning(
        "**v9-Pro checkout not found.** This page reads combo research produced by "
        "`NevixAA/wowzaV9-Pro`, expected as a sibling folder (`../wowzaV9-Pro` or `../v10`). "
        "Nothing is broken in v9 — the data simply lives in the other repo.")
    st.stop()

cand = _load("bet_builder_candidates.csv")
settled = _load("bet_builder_settled.csv")
dep = _load("combo_dependency_matrix.csv")

if cand.empty and settled.empty:
    st.info("No combo output yet. Generate it in Pro with `python -m src.combo.generate`.")
    st.stop()

# ── Settled record ────────────────────────────────────────────────────────────
# Results first. A candidate list with no record of how the last ones went is a wish list, and
# this page should not be one.
st.markdown("## 📊 Settled record")
if settled.empty:
    st.caption("Nothing settled yet — results appear once the fixtures finish and Pro's "
               "settlement pass has run.")
else:
    r = settled["combo_result"]
    won, lost = int((r == "WON").sum()), int((r == "LOST").sum())
    decided = won + lost
    odds = pd.to_numeric(settled.get("fair_combo_odds", settled.get("fair_odds")),
                         errors="coerce")
    pnl = float(((odds - 1.0).where(r == "WON", -1.0).where(r != "VOID", 0.0)
                 .where(r != "UNKNOWN", 0.0)).sum())
    ui.metric_row([
        {"label": "Settled", "value": f"{decided:,}",
         "help": "VOID and UNKNOWN are excluded — a player who never came on has not lost."},
        {"label": "Won", "value": f"{won:,}"},
        {"label": "Hit rate", "value": f"{100*won/decided:.1f}%" if decided else "—",
         "help": "Compare against the average joint probability below, not against 50%."},
        {"label": "P/L at fair odds", "value": f"{pnl:+.2f}u",
         "help": "Staked at OUR fair price. No bookmaker builder price exists, so this measures "
                 "whether the probabilities were right — not what anyone could have won."},
    ])
    exp = pd.to_numeric(settled.get("joint_probability"), errors="coerce").mean()
    if decided and exp == exp:
        st.caption(f"Model expected **{100*exp:.1f}%** of these to win; **{100*won/decided:.1f}%** "
                   f"did. Void {int((r=='VOID').sum())}, unverifiable {int((r=='UNKNOWN').sum())}.")
    ui.table(settled.sort_values("combo_result").head(60)[
        [c for c in ("match", "legs", "leg_probs", "joint_probability", "fair_odds",
                     "final_score", "combo_result", "leg_results") if c in settled.columns]])
st.divider()

# ── Live candidates ───────────────────────────────────────────────────────────
st.markdown("## 🧩 Current candidates")
if cand.empty:
    st.caption("No current candidates.")
else:
    c1, c2, c3 = st.columns([1, 1, 1.4])
    n_legs = c1.multiselect("Legs", sorted(cand["n_legs"].dropna().unique().tolist()),
                            default=sorted(cand["n_legs"].dropna().unique().tolist()))
    odds_col = "fair_combo_odds" if "fair_combo_odds" in cand.columns else "fair_odds"
    lo, hi = c2.slider("Fair odds", 1.0, 50.0, (2.0, 15.0), 0.5)
    only_corr = c3.checkbox(
        "Only where correlation matters (ratio ≥ 1.15)", value=True,
        help="Combos whose true joint differs materially from multiplying the legs. Where the "
             "ratio is ~1.00 we are offering no insight the market does not already have.")

    d = cand[cand["n_legs"].isin(n_legs)] if n_legs else cand
    o = pd.to_numeric(d.get(odds_col), errors="coerce")
    d = d[o.between(lo, hi)]
    if only_corr and "dependency_ratio" in d.columns:
        d = d[pd.to_numeric(d["dependency_ratio"], errors="coerce") >= 1.15]
    st.caption(f"{len(d):,} of {len(cand):,} candidates after filters.")

    show = [c for c in ("match", "league", "match_date", "legs", "leg_probs",
                        "joint_probability", "independence_probability", "dependency_ratio",
                        odds_col, "independence_fair_odds", "leg_flags", "data_quality")
            if c in d.columns]
    sort_col = "dependency_ratio" if "dependency_ratio" in d.columns else odds_col
    ui.table(d.sort_values(sort_col, ascending=False).head(200)[show], height=520)

    st.info(
        "**`fair_odds` is our price, not a market price.** No bookmaker same-game-builder price "
        "is collected anywhere, so there is no EV column — the comparison to make is against "
        "your own book. **`independence_fair_odds`** is what naively multiplying the legs would "
        "ask; the gap between the two is the whole point of this system, and on a four-leg "
        "builder it has reached **6.7×**.")
st.divider()

# ── The measured dependence ───────────────────────────────────────────────────
if not dep.empty:
    st.markdown("## 🔗 Why combos are not multiplication")
    a = dep[dep["segment"] == "ALL"].copy()
    a["independence_error_pp"] = (100 * (a["p_joint"] - a["independent_joint"])).round(2)
    top = pd.concat([a.nlargest(6, "dependency_ratio"), a.nsmallest(4, "dependency_ratio")])
    ui.table(top[["market_a", "market_b", "n", "p_joint", "independent_joint",
                  "dependency_ratio", "independence_error_pp"]].round(4))
    st.caption(
        f"Measured on **{int(a['n'].max()):,} settled fixtures**. A ratio of 1.00 would mean "
        f"multiplication is safe; independence is statistically defensible for only "
        f"**{int(a.get('independence_within_ci', pd.Series(dtype=bool)).sum())} of {len(a)}** "
        f"pairs. Positive `independence_error_pp` means multiplying UNDERSTATES the true chance.")

st.caption("Generated in v9-Pro · PAPER research · Pro places no stakes and this page places none.")
