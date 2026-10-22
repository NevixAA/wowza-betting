"""Bankroll — the approved staking policy, Pro's monthly recommendation, and the evidence behind it.

DISPLAY ONLY. Evidence is computed in wowzaV9-Pro (src/bankroll); the production policy is
config/bankroll_policy.json and changes only through a reviewed PR. Nothing on this page edits
either. Read-only, light: two JSON files and a few small CSVs.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard_access import require as _require_tier  # noqa: E402
_require_tier("pro")  # tier guard (dashboard_access.PAGES), 2026-10-18

from dashboard_data import pro, ui  # noqa: E402
import dashboard_ui as UI  # noqa: E402  (metric_row lives here)

import importlib  # noqa: E402
pro = importlib.reload(pro)          # keep page + helper the same version (see 15_League_Scout)

st.set_page_config(page_title="Wowza | Bankroll", page_icon="💰", layout="wide")
st.title("💰 Bankroll policy")
ui.state_banner("REVIEW", "How much bankroll each track deserves. Display only — policies change "
                          "through a reviewed PR, never here.")

prod_path = Path(__file__).resolve().parents[1] / "config" / "bankroll_policy.json"
prod = json.loads(prod_path.read_text(encoding="utf-8")) if prod_path.exists() else {}
b = pro.bankroll() if pro.available() else {}
rec = b.get("recommendation") or {}
review = rec.get("review", {})
from dashboard_data import freshness as _fresh  # noqa: E402  (P1-3: never show stale evidence as current)
_man = _fresh.manifest()
_fresh.banner(st, "output/bankroll/current_policy_recommendation.json", "Pro bankroll recommendation", _man)

def pct(x):
    return "—" if x is None or pd.isna(x) else f"{100 * float(x):.2f}%"

# ── headline ───────────────────────────────────────────────────────────────────────────────────
c = st.columns(4)
c[0].metric("Production policy", prod.get("policy_version", "—"), help=prod.get("status", ""))
c[1].metric("Pro recommends", review.get("RECOMMENDED_POLICY", "—").replace("_", " ").title()[:28],
            help=rec.get("policy_version", ""))
c[2].metric("Change recommended", review.get("POLICY_CHANGE_RECOMMENDED", "—"))
asof = pd.Timestamp(review["as_of"]) if review.get("as_of") else None
nxt = (asof + pd.offsets.MonthBegin(1) + pd.Timedelta(days=1)) if asof is not None else None
c[3].metric("Last review", review.get("as_of", "—"), delta=f"next ~{nxt:%d %b}" if nxt is not None else None,
            delta_color="off")
if review:
    wf = review.get("walk_forward", {})
    st.caption(f"Allocator walk-forward: **{wf.get('status', '?')}** ({wf.get('months_evaluated', 0)} of "
               f"{wf.get('required', 6)} months). Until validated, Pro only ever recommends the current policy "
               f"with risk REDUCTIONS. Best growth: {review.get('BEST_GROWTH_POLICY')} · best survival: "
               f"{review.get('BEST_SURVIVAL_POLICY')}.")

# ── tracks ─────────────────────────────────────────────────────────────────────────────────────
st.subheader("Base stake by track")
dec = b.get("track_decisions", pd.DataFrame())
rows = []
for k, t in (prod.get("tracks") or {}).items():
    r = (rec.get("tracks") or {}).get(k, {})
    d = dec[dec["track"] == k].iloc[0] if len(dec) and (dec["track"] == k).any() else None
    rows.append({"Track": k.replace("|", " · "), "State": t.get("state"), "Base (production)": pct(t.get("base_fraction")),
                 "Recommended": pct(r.get("base_fraction")), "Decision": None if d is None else d["decision"],
                 "Max odds": str(t.get("max_odds", "—")), "n": None if d is None else d.get("n"),
                 "ROI": None if d is None else pct(d.get("roi")), "Shrunk ROI": None if d is None else pct(d.get("shrunk_roi")),
                 "Why": None if d is None else str(d.get("reasons", ""))[:160]})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
pf = prod.get("portfolio", {})
st.caption(f"Caps: single bet {pct(pf.get('single_bet_cap'))} · fixture {pct(pf.get('fixture_cap'))} · day "
           f"{pct(pf.get('daily_cap'))} · league/day {pct(pf.get('league_daily_cap'))} · league×market/day "
           f"{pct(pf.get('league_market_daily_cap'))} · one real position per fixture · tiers "
           f"{prod.get('tier_multipliers')} · convergence leagues: "
           f"{(prod.get('convergence') or {}).get('leagues') or 'none'}")

# ── stake plan (src/stake_plan.py, wired 2026-10-14) ──────────────────────────────────────────
st.subheader("Stake plan — the policy applied to the current tips")
_fresh.banner(st, "output/stake_plan.csv", "stake plan", _man)
_sp_path = Path(__file__).resolve().parents[1] / "output" / "stake_plan.csv"
try:
    sp = pd.read_csv(_sp_path) if _sp_path.exists() else pd.DataFrame()
except Exception:                                                    # noqa: BLE001
    sp = pd.DataFrame()
if sp.empty:
    st.info("No stake plan yet — predict writes `output/stake_plan.csv` on its next run.")
else:
    staked = sp[sp["final_fraction"] > 0]
    c = st.columns(3)
    c[0].metric("Tips sized", f"{len(sp):,}", help=f"generated {sp['generated_at'].iloc[0]} UTC")
    c[1].metric("With a stake", f"{len(staked):,}")
    c[2].metric("Days covered", f"{sp['date'].nunique()}")
    st.caption("Recommendations only: nothing here places a bet, changes a tip or sends a message. Stakes are % of a "
               "notional bankroll (1.00 unit = 1% of 100); the drawdown throttle is NOT applied because there is no "
               "real-bankroll ledger. Every zero carries its reason.")
    by_day = (staked.groupby("date")["final_fraction"].sum() * 100).round(2).rename("% of bankroll").reset_index()
    show = staked.assign(stake_pct=(staked["final_fraction"] * 100).round(3))[
        ["date", "fixture", "league", "track", "signal_tier", "odds", "stake_pct"]].sort_values(["date", "stake_pct"],
                                                                                              ascending=[True, False])
    st.dataframe(show.rename(columns={"stake_pct": "stake % of bankroll"}), hide_index=True, width="stretch")
    st.caption("Total staked per day (cap " + pct(pf.get("daily_cap")) + "): "
               + " · ".join(f"{r['date']} {r['% of bankroll']}%" for _, r in by_day.iterrows()))
    with st.expander("Why tips got no stake"):
        st.dataframe(sp[sp["final_fraction"] <= 0]["reason"].value_counts().rename("tips").reset_index(),
                     hide_index=True, width="stretch")

# ── drawdown ───────────────────────────────────────────────────────────────────────────────────
path = b.get("production_policy_path", pd.DataFrame())
if len(path):
    st.subheader("Drawdown throttle — production policy replayed on the prospective record")
    last = path.iloc[-1]
    th = prod.get("drawdown_throttle", [])
    UI.metric_row([{"label": "Bankroll (start 100)", "value": f"{last['bankroll']:.1f}"},
                   {"label": "Peak", "value": f"{last['peak']:.1f}"},
                   {"label": "Current drawdown", "value": pct(last["drawdown"])},
                   {"label": "Throttle", "value": f"×{last['throttle']:.2f}",
                    "help": f"Steps: {th} (drawdown -> multiplier)"}])
    st.line_chart(path.set_index("date")[["bankroll", "peak"]], height=240)
    st.caption("Simulated with the production policy file and v9's own stake reader. Not a real-money ledger.")

# ── growth vs survival ─────────────────────────────────────────────────────────────────────────
cand = b.get("policy_candidates", pd.DataFrame())
if len(cand):
    st.subheader("Growth vs survival — candidate policies on the same record")
    show = cand[["policy", "end", "max_dd", "bets", "risk_end_median", "risk_end_p05", "risk_p_maxdd_gt_10",
                 "risk_p_maxdd_gt_20", "passes_survival", "score"]].rename(columns={
        "end": "Ending", "max_dd": "Max DD", "risk_end_median": "Median (resampled)", "risk_end_p05": "5th pct",
        "risk_p_maxdd_gt_10": "P(DD>10%)", "risk_p_maxdd_gt_20": "P(DD>20%)", "passes_survival": "Survives"})
    st.dataframe(show, hide_index=True, width="stretch")
st_ = b.get("stress_tests", pd.DataFrame())
if len(st_):
    with st.expander("Stress tests (recommended policy)"):
        st.dataframe(st_[["scenario", "end", "return_pct", "max_dd", "bets"]], hide_index=True, width="stretch")
wfd = b.get("walk_forward", pd.DataFrame())
if len(wfd):
    with st.expander("Walk-forward test of the allocator (each month chosen with earlier data only)"):
        st.dataframe(wfd, hide_index=True, width="stretch")
if not rec:
    ui.absent("No Pro bankroll review yet - it runs on the 2nd of each month.")
