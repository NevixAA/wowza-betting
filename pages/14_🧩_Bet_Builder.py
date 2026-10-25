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

# Visual redesign 2026-10-25 (stage 3): settled record first as one card (hit rate beside the model's own
# expected rate and n), then the strongest-dependence candidates as cards, the full candidate table, and
# the measured dependence as one signed chart. PRESENTATION ONLY — same files, same arithmetic, same
# filters; components in dashboard_data/builder_view.py.
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

import dashboard_ui as ui

st.set_page_config(page_title="Bet Builder | Wowza", page_icon="🧩", layout="wide")
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from dashboard_access import require as _require_tier  # noqa: E402
_require_tier("pro")  # tier guard (dashboard_access.PAGES), 2026-10-18

ui.autorefresh(minutes=5, key="builder_refresh")

from dashboard_data.core import read_csv as _read_any, reachable, resolve_file  # noqa: E402
from dashboard_data import builder_view as V, freshness as _fresh  # noqa: E402
import importlib  # noqa: E402
V = importlib.reload(V)              # page + helper always the same version (see 15_League_Scout)

_TITLE, _SUB = "Bet Builder & Multi", "Same-match builders and cross-match multiples, priced with measured correlation"
_EYEBROW = "Tools · combo research"
_STATE = ui.status_chip("RESEARCH", label="RESEARCH · PAPER")


@st.cache_data(ttl=300, show_spinner=False)
def _load(name: str) -> pd.DataFrame:
    # Local sibling checkout if there is one, else Pro's public repo over HTTP — the same path
    # the Pro page uses. This page used to look for ../wowzaV9-Pro only, which exists on a
    # laptop and never on Streamlit Cloud, so the deployed page always said "not found".
    return _read_any(resolve_file("pro", f"output/{name}"), low_memory=False)


if not reachable("pro"):
    ui.page_header(_TITLE, _SUB, eyebrow_text=_EYEBROW, freshness=_STATE)
    ui.empty_state("NOT_CONNECTED",
                   "Pro could not be reached. This page reads combo research produced by `NevixAA/wowzaV9-Pro` — "
                   "from a sibling checkout or over HTTP — and neither answered. Nothing is broken in v9; the data "
                   "lives in the other repo.")
    st.stop()

cand = _load("bet_builder_candidates.csv")
settled = _load("bet_builder_settled.csv")
dep = _load("combo_dependency_matrix.csv")

_gen = pd.to_datetime(cand.get("generated_at"), errors="coerce", utc=True).max() if len(cand) else None
_, _chip = _fresh.ts_badge(_gen if _gen is not None and pd.notna(_gen) else None, "daily", "candidates generated")
ui.page_header(_TITLE, _SUB, eyebrow_text=_EYEBROW, freshness=_STATE + " " + _chip)

if cand.empty and settled.empty:
    ui.empty_state("NO_DATA", "No combo output yet. Generate it in Pro with `python -m src.combo.generate`.")
    st.stop()

# ═══ PERFORMANCE — settled record first ══════════════════════════════════════════════════════════
# Results first. A candidate list with no record of how the last ones went is a wish list, and
# this page should not be one.
ui.section_header("Settled record", "P/L is settled at OUR fair price — no bookmaker builder price exists — so it "
                  "measures whether the probabilities were right, not what anyone could have won.")
if settled.empty:
    ui.empty_state("NO_DATA", "Nothing settled yet — results appear once the fixtures finish and Pro's "
                              "settlement pass has run.")
else:
    s = V.settled_summary(settled)
    r = settled["combo_result"]
    decided, won, exp = s["decided"], s["won"], s["exp"]
    left, right = st.columns([1, 2])
    V.settled_card(s, container=left)
    with right:
        # THE EXPECTATION MUST COME FROM THE DECIDED ROWS ONLY. Averaging joint_probability over the
        # whole file compares two different populations: the undecided rows are mostly 3- and 4-leg
        # combos, which are far less likely by construction, while the hit rate can only be measured
        # on the settled ones. Taken over everything it read "expected 26.4%, 50.3% did" — a model
        # apparently twice as good as it is, produced entirely by mixing the groups.
        if decided and exp is not None and exp == exp:
            st.caption(f"Model expected **{100*exp:.1f}%** of these to win; **{100*won/decided:.1f}%** "
                       f"did — measured on the {decided:,} DECIDED combos only, since the "
                       f"{int((r=='UNKNOWN').sum()):,} unverifiable ones skew longer and less likely. "
                       f"Void {int((r=='VOID').sum())}.")
        st.caption("VOID and UNKNOWN are excluded — a player who never came on has not lost.")
    with st.expander(f"Settled combos — {min(len(settled), 60)} shown"):
        ui.compact_table(settled.sort_values("combo_result").head(60)[
            [c for c in ("match", "legs", "leg_probs", "joint_probability", "fair_odds",
                         "final_score", "combo_result", "leg_results") if c in settled.columns]], V.SETTLED_CONFIG)

# ═══ RESEARCH — current candidates ═══════════════════════════════════════════════════════════════
ui.section_header("Current candidates", "`fair_odds` is our price, not a market price. The naive price is what "
                  "multiplying the legs would ask; the gap between them is the whole point of this system.")
if cand.empty:
    ui.empty_state("NO_DATA", "No current candidates.")
else:
    _legs = sorted(cand["n_legs"].dropna().unique().tolist())
    n_legs = st.multiselect("Legs", _legs, default=_legs)
    odds_col = "fair_combo_odds" if "fair_combo_odds" in cand.columns else "fair_odds"
    with st.expander("Advanced filters"):
        c2, c3 = st.columns([1, 1.4])
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
    ordered = d.sort_values(sort_col, ascending=False)
    if ordered.empty:
        ui.empty_state("NO_DATA", "No candidates match the filters.")
    else:
        for col, (_, row) in zip(st.columns(3), ordered.head(3).iterrows()):
            V.candidate_card(row, odds_col, container=col)
        ui.compact_table(ordered.head(200)[show], V.CANDIDATE_CONFIG, height=520)
    st.caption(
        "No bookmaker same-game-builder price is collected anywhere, so there is no EV column — the comparison to "
        "make is against your own book. **`independence_fair_odds`** is what naively multiplying the legs would "
        "ask; on a four-leg builder the gap has reached **6.7×**.")

# ═══ RESEARCH — the measured dependence ══════════════════════════════════════════════════════════
if not dep.empty:
    ui.section_header("Why combos are not multiplication")
    a, top = V.dependence_frame(dep)
    left, right = st.columns([3, 2])
    with left:
        st.altair_chart(V.dependence_chart(top), width="stretch")
    with right:
        st.caption(
            f"Measured on **{int(a['n'].max()):,} settled fixtures**. A ratio of 1.00 would mean "
            f"multiplication is safe; independence is statistically defensible for only "
            f"**{int(a.get('independence_within_ci', pd.Series(dtype=bool)).sum())} of {len(a)}** "
            f"pairs. Positive `independence_error_pp` means multiplying UNDERSTATES the true chance.")
    ui.compact_table(top[["market_a", "market_b", "n", "p_joint", "independent_joint",
                          "dependency_ratio", "independence_error_pp"]].round(4), V.DEP_CONFIG)

st.caption("Generated in v9-Pro · PAPER research · Pro places no stakes and this page places none.")
