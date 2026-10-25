"""Bankroll — the approved staking policy, Pro's monthly recommendation, and the evidence behind it.

DISPLAY ONLY. Evidence is computed in wowzaV9-Pro (src/bankroll); the production policy is
config/bankroll_policy.json and changes only through a reviewed PR. Nothing on this page edits
either. Read-only, light: two JSON files and a few small CSVs.

Visual redesign 2026-10-25 (stage 3): decision first — is a policy change recommended, then one card per
track (state, base stake, Pro's view, the evidence n beside every ROI), then today's stake plan, the
drawdown path and the research tables. PRESENTATION ONLY: components in dashboard_data/bankroll_view.py;
every value is read from the same files as before. Pro's review is cached for five minutes (it was re-read
on every widget interaction); the values are the same call's values.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard_access import require as _require_tier  # noqa: E402
_require_tier("pro")  # tier guard (dashboard_access.PAGES), 2026-10-18

import dashboard_ui as ui  # noqa: E402
from dashboard_data import bankroll_view as V, pro  # noqa: E402
from dashboard_data import freshness as _fresh  # noqa: E402  (P1-3: never show stale evidence as current)

import importlib  # noqa: E402
pro = importlib.reload(pro)          # keep page + helper the same version (see 15_League_Scout)
V = importlib.reload(V)

st.set_page_config(page_title="Wowza | Bankroll", page_icon="💰", layout="wide")


@st.cache_data(ttl=300, show_spinner="Reading Pro's bankroll review…")
def _pro_bankroll() -> dict:
    b = pro.bankroll() if pro.available() else {}
    b.pop("freshness", None)                       # not displayed on this page; keeps the cache picklable
    return b


prod_path = Path(__file__).resolve().parents[1] / "config" / "bankroll_policy.json"
prod = json.loads(prod_path.read_text(encoding="utf-8")) if prod_path.exists() else {}
b = _pro_bankroll()
rec = b.get("recommendation") or {}
review = rec.get("review", {})
_man = _fresh.manifest()
pct = V.pct

_rstate, _rchip = _fresh.header_badge("output/bankroll/current_policy_recommendation.json",
                                      "Pro bankroll recommendation", _man)
ui.page_header("Bankroll policy", "How much bankroll each track deserves. Display only — policies change through a "
               "reviewed PR, never here.", eyebrow_text="Performance · staking policy",
               freshness=ui.status_chip("REVIEW", label="DISPLAY ONLY · REVIEW", tone="INFO") + " " + _rchip)
if _rstate != "FRESH":                              # the long STALE / HISTORICAL / dated-study explanation
    _fresh.banner(st, "output/bankroll/current_policy_recommendation.json", "Pro bankroll recommendation", _man)

# ═══ ACTION — is a policy change recommended? ════════════════════════════════════════════════════
ui.section_header("Decision", "What production runs, what Pro recommends, and when it last looked.")
asof = pd.Timestamp(review["as_of"]) if review.get("as_of") else None
nxt = (asof + pd.offsets.MonthBegin(1) + pd.Timedelta(days=1)) if asof is not None else None
wf = review.get("walk_forward", {}) if review else {}
change = review.get("POLICY_CHANGE_RECOMMENDED", "—")
c = st.columns(3)
ui.metric_card("Production policy", prod.get("policy_version", "—"), title="Running now", help=prod.get("status", ""),
               supporting=[("Status", prod.get("status", "—"))], container=c[0])
ui.metric_card("Pro recommends", review.get("RECOMMENDED_POLICY", "—").replace("_", " ").title()[:28],
               title="Pro's monthly review", help=rec.get("policy_version", ""),
               supporting=[("Change recommended", change), ("Best growth", review.get("BEST_GROWTH_POLICY", "—")),
                           ("Best survival", review.get("BEST_SURVIVAL_POLICY", "—"))],
               status=ui.status_chip("NO_DATA", label=f"CHANGE RECOMMENDED · {change}",
                                     tone="WARNING" if str(change).upper() == "YES" else "NEUTRAL"),
               container=c[1])
ui.metric_card("Last review", review.get("as_of", "—"), title="Cadence",
               delta=f"next ~{nxt:%d %b}" if nxt is not None else None, delta_color="off",
               supporting=[("Walk-forward", f"{wf.get('months_evaluated', 0)} of {wf.get('required', 6)} months")],
               status=ui.status_chip("NO_DATA", label=str(wf.get("status", "?")).replace("_", " "), tone="RESEARCH"),
               container=c[2])
if review:
    st.caption(f"Allocator walk-forward: **{wf.get('status', '?')}** ({wf.get('months_evaluated', 0)} of "
               f"{wf.get('required', 6)} months). Until validated, Pro only ever recommends the current policy "
               f"with risk REDUCTIONS. Best growth: {review.get('BEST_GROWTH_POLICY')} · best survival: "
               f"{review.get('BEST_SURVIVAL_POLICY')}.")
if not rec:
    ui.empty_state("NO_DATA", "No Pro bankroll review yet - it runs on the 2nd of each month.")

# ═══ STATUS — one card per track ═════════════════════════════════════════════════════════════════
ui.section_header("Base stake by track", "Production base stake is the main number. ROI is shown only with its n "
                  "and match-days beside it — an ROI alone is never evidence.")
dec = b.get("track_decisions", pd.DataFrame())
rows = V.track_rows(prod, rec, dec)
V.track_cards(rows)
with st.expander("All tracks — table"):
    ui.compact_table(rows, V.TRACK_CONFIG)
pf = prod.get("portfolio", {})
st.caption(f"Caps: single bet {pct(pf.get('single_bet_cap'))} · fixture {pct(pf.get('fixture_cap'))} · day "
           f"{pct(pf.get('daily_cap'))} · league/day {pct(pf.get('league_daily_cap'))} · league×market/day "
           f"{pct(pf.get('league_market_daily_cap'))} · one real position per fixture · tiers "
           f"{prod.get('tier_multipliers')} · convergence leagues: "
           f"{(prod.get('convergence') or {}).get('leagues') or 'none'}")

# ═══ ACTION — the policy applied to the current tips (src/stake_plan.py, wired 2026-10-14) ═══════
ui.section_header("Stake plan — the policy applied to the current tips")
_sstate, _schip = _fresh.header_badge("output/stake_plan.csv", "stake plan", _man)
st.markdown(_schip)
if _sstate != "FRESH":
    _fresh.banner(st, "output/stake_plan.csv", "stake plan", _man)
_sp_path = Path(__file__).resolve().parents[1] / "output" / "stake_plan.csv"
try:
    sp = pd.read_csv(_sp_path) if _sp_path.exists() else pd.DataFrame()
except Exception:                                                    # noqa: BLE001
    sp = pd.DataFrame()
if sp.empty:
    ui.empty_state("NO_DATA", "No stake plan yet — predict writes `output/stake_plan.csv` on its next run.")
else:
    show, by_day = V.plan_rows(sp)
    left, right = st.columns([1, 2])
    ui.metric_card("With a stake", f"{len(show):,}", title="Today and upcoming",
                   help=f"generated {sp['generated_at'].iloc[0]} UTC",
                   supporting=[("Tips sized", f"{len(sp):,}"), ("Days covered", f"{sp['date'].nunique()}")],
                   status=ui.status_chip("NO_DATA", label="RECOMMENDATION ONLY", tone="INFO"), container=left)
    with right:
        if len(by_day):
            st.altair_chart(V.day_bars(by_day, pf.get("daily_cap")), width="stretch")
    st.caption("Recommendations only: nothing here places a bet, changes a tip or sends a message. Stakes are % of a "
               "notional bankroll (1.00 unit = 1% of 100); the drawdown throttle is NOT applied because there is no "
               "real-bankroll ledger. Every zero carries its reason.")
    st.caption("Total staked per day (cap " + pct(pf.get("daily_cap")) + "): "
               + " · ".join(f"{r['date']} {r['% of bankroll']}%" for _, r in by_day.iterrows()))
    ui.compact_table(show, V.PLAN_CONFIG)
    with st.expander("Why tips got no stake"):
        ui.compact_table(sp[sp["final_fraction"] <= 0]["reason"].value_counts().rename("tips").reset_index())

# ═══ PERFORMANCE — drawdown ══════════════════════════════════════════════════════════════════════
path = b.get("production_policy_path", pd.DataFrame())
if len(path):
    ui.section_header("Drawdown throttle — production policy replayed on the prospective record",
                      "Simulated with the production policy file and v9's own stake reader. Not a real-money ledger.")
    last = path.iloc[-1]
    th = prod.get("drawdown_throttle", [])
    left, right = st.columns([1, 2])
    ui.metric_card("Bankroll (start 100)", f"{last['bankroll']:.1f}", title="Replayed policy",
                   supporting=[("Peak", f"{last['peak']:.1f}"), ("Current drawdown", pct(last["drawdown"])),
                               ("Throttle", f"×{last['throttle']:.2f}")],
                   help=f"Steps: {th} (drawdown -> multiplier)",
                   status=ui.status_chip("NO_DATA", label="SIMULATED · NOT A MONEY LEDGER", tone="INFO"),
                   container=left)
    with right:
        st.altair_chart(V.drawdown_chart(path), width="stretch")

# ═══ RESEARCH — growth vs survival ═══════════════════════════════════════════════════════════════
cand = b.get("policy_candidates", pd.DataFrame())
if len(cand):
    ui.section_header("Growth vs survival — candidate policies on the same record")
    show = cand[["policy", "end", "max_dd", "bets", "risk_end_median", "risk_end_p05", "risk_p_maxdd_gt_10",
                 "risk_p_maxdd_gt_20", "passes_survival", "score"]].rename(columns={
        "end": "Ending", "max_dd": "Max DD", "risk_end_median": "Median (resampled)", "risk_end_p05": "5th pct",
        "risk_p_maxdd_gt_10": "P(DD>10%)", "risk_p_maxdd_gt_20": "P(DD>20%)", "passes_survival": "Survives"})
    ui.compact_table(show, V.CANDIDATE_CONFIG)
st_ = b.get("stress_tests", pd.DataFrame())
if len(st_):
    with st.expander("Stress tests (recommended policy)"):
        ui.compact_table(st_[["scenario", "end", "return_pct", "max_dd", "bets"]])
wfd = b.get("walk_forward", pd.DataFrame())
if len(wfd):
    with st.expander("Walk-forward test of the allocator (each month chosen with earlier data only)"):
        ui.compact_table(wfd)
