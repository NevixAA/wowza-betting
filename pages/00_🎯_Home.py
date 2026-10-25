"""Home (Wowza Pro) — the first screen. Was app.py until the tiered router (2026-10-18).

Visual redesign 2026-10-25 (owner brief): the first viewport answers three questions, one card each —
  A  Today's action        what production stakes today, the next kickoff, the top Best-bets pick,
                            and paper / shadow counts kept SEPARATE from the staked count;
  B  Current performance   staked P/L, ROI, drawdown from peak, last 7 days;
  C  System state          data freshness, settlement, CLV, production state (system_health.json).
Below: best opportunities (the Best-bets list), recent performance, and what needs attention.

PRESENTATION ONLY. Every number comes from dashboard_data.today_view (Home's original staked-basis
logic, moved there verbatim), dashboard_data.production_view (the stake plan, as the Production page
reads it), src.top_picks (the same lists the Tips page and the morning album show) and the CI-written
system_health.json / artifact_manifest.json. tests/test_home_tips_values.py checks the rendered values
against those functions.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from dashboard_access import require  # noqa: E402
require("pro")

import dashboard_ui as ui  # noqa: E402
from dashboard_data import production_view, today_view  # noqa: E402

st.set_page_config(page_title="Wowza | Home", page_icon="🎯", layout="wide", initial_sidebar_state="expanded")
ui.autorefresh(5, key="main_refresh")

_TZ = "Asia/Jerusalem"


def _best_bets(day: str) -> tuple[pd.DataFrame, bool]:
    from src import top_picks as TP
    fz = TP.frozen(day)
    lists, frozen = (fz, True) if not fz.empty else (TP.todays_lists(day), False)
    if lists is None or lists.empty:
        return pd.DataFrame(), frozen
    return lists[lists["list"] == "BEST_BETS"].sort_values("rank").reset_index(drop=True), frozen


@st.cache_data(ttl=60)
def _home(today: str, local_day: str) -> dict:
    """Everything Home reads, in ONE cached call (one cache-key computation per rerun, not six).

    * performance: Home's original staked basis (today_view);
    * the stake plan (same reader as the Production page), its counts, and the kickoff of every active
      tip - the 'next' kickoff is picked against the clock on each run, not cached;
    * the Best-bets list (src.top_picks, the Tips page's lists); system health, manifest, API budget."""
    plan = production_view.stake_plan(today=today)
    ko = today_view.kickoffs()
    active = plan[plan["final_fraction"] > 0] if len(plan) else plan
    times = (pd.to_datetime(active["fixture"].map(ko.set_index("fixture")["kickoff_utc"]), errors="coerce", utc=True)
             if len(active) and len(ko) else pd.Series(dtype="datetime64[ns, UTC]"))
    sh, man = today_view.health_files()
    bb, bb_frozen = _best_bets(today)
    return {"perf": today_view.performance(today_view.load_settled(), today=local_day),
            "upcoming_ou": today_view.upcoming_staked_ou(today_view.load_bets()),
            "act": today_view.action(plan), "ko_times": times,
            "gen": plan["generated_at"].dropna().astype(str).max() if len(plan) else None,
            "bb": bb, "bb_frozen": bb_frozen, "sh": sh, "man": man, "budget": today_view.api_budget()}


today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
H = _home(today, pd.Timestamp.now().normalize().strftime("%Y-%m-%d"))
perf, act, _gen, bb, bb_frozen = H["perf"], H["act"], H["gen"], H["bb"], H["bb_frozen"]
sh, man, budget = H["sh"], H["man"], H["budget"]
_ko_times = H["ko_times"]
_now = pd.Timestamp.now(tz="UTC")
nxt = _ko_times[_ko_times > _now].min() if len(_ko_times) else pd.NaT
_cut = config.PERFORMANCE_CUTOFF_DATE

# ── header — the ONE freshness statement on the page ─────────────────────────────────────────
_h_age = today_view.age_hours(sh.get("generated_at")) if sh else None
_fresh = (ui.freshness_badge("FRESH" if _h_age is not None and _h_age <= 36 else "STALE",
                             f"system health {_h_age:.1f} h old" if _h_age is not None else "no timestamp")
          if sh else ui.freshness_badge("UNKNOWN", "system health missing"))
_tip_line = f"stake plan {pd.Timestamp(_gen):%d/%m %H:%M} UTC" if _gen else ""
ui.page_header("Home", "What production is doing today, how the staked book stands, and whether the "
               "data behind it is fit for decisions.", eyebrow_text="Today · Wowza Pro",
               freshness=_fresh + (f" :gray[{_tip_line}]" if _tip_line else ""))

# ── hero row ─────────────────────────────────────────────────────────────────────────────────
a, b, c = st.columns(3)

with a:
    _nk = f"{nxt.tz_convert(_TZ):%a %H:%M}" if pd.notna(nxt) else "—"
    _paper_bits = ", ".join(f"{v} {k}" for k, v in sorted(act["by_state"].items())) or "none"
    ui.metric_card(
        "Active tips", f"{act['active']}", title="A · Today's action",
        help="Tips dated today or later that the approved bankroll policy stakes (stake plan, final stake > 0) "
             "— the same count as 'With a stake' on the Production page.",
        supporting=[("Next active kickoff", _nk),
                    ("Upcoming staked tips (O/U 2.5)", f"{H['upcoming_ou']}"),
                    ("Paper / no stake", f"{act['paper']}"),
                    ("Shadow engine BET", f"{act['shadow_bet']} of {act['shadow_n']}")],
        status=ui.status_chip("LIVE", label="PRODUCTION") + " " + ui.status_chip("RESEARCH", label="SHADOW SEPARATE"))
    if len(bb):
        r = bb.iloc[0]
        st.caption(f"Best bets #1: **{r['home_team']} v {r['away_team']}** · {r['pick']} @ {ui.fmt_odds(r['odds'])}"
                   f" · chance {ui.fmt_prob(r['p_win'])}")
    st.caption(f"Upcoming staked tips = SNIPER + MARKSMAN on O/U 2.5 in the current slate. "
               f"Not staked by state: {_paper_bits}. Kickoff in Israel time.")

with b:
    if perf["n"]:
        _dd = perf["drawdown"]
        ui.metric_card(
            "Staked P/L", f"{perf['pnl']:+.2f}u", title="B · Current performance",
            help="Flat 1u, SNIPER + MARKSMAN, every staked market (O/U 2.5, side markets, HT), settled only. "
                 "Same number as the Tips page.",
            supporting=[("ROI", ui.fmt_roi(perf["roi"])),
                        ("Win rate — staked", f"{perf['win_rate'] * 100:.1f}%"),
                        ("Settled staked bets", f"{perf['n']}"),
                        ("Drawdown from peak", ui.fmt_units(_dd)),
                        ("Last 7 days", f"{ui.fmt_units(perf['last7_pnl'])} · n={perf['last7_n']}")],
            evidence=ui.evidence_badge("settled since", None, span=str(_cut), state="flat 1u"), n=perf["n"],
            chart_data=perf["curve"]["cum"].round(2).tolist() if len(perf["curve"]) > 1 else None)
    else:
        with st.container(border=True):
            ui.eyebrow("B · Current performance")
            st.metric("Staked P/L", "—")
            ui.stat_line([("Win rate — staked", "—"), ("Settled staked bets", "0")])
    st.caption("Paper VALUABLE P/L: "
               + (f"**{perf['paper_pnl']:+.2f}u** · {perf['paper_n']} tips, not bets" if perf["paper_n"] else "—")
               + " — tracked, never staked, never in the P/L.")

with c:
    overall = str(sh.get("overall", "UNKNOWN")) if sh else "UNKNOWN"
    _stale = today_view.check(sh, "Stale artifacts")
    _set = today_view.check(sh, "Shadow settlement coverage")
    _clv = today_view.check(sh, "Certified CLV coverage")
    _prot = today_view.check(sh, "Protected tracks (research lock)")
    _sel = ((sh.get("versions") or {}).get("selection_engine_version") or {}) if sh else {}

    def _v(chk: dict) -> str:
        return f"{chk.get('value', '—')} ({chk.get('status', 'NO_DATA')})" if chk else "— (NO_DATA)"

    with st.container(border=True):
        ui.eyebrow("C · System state")
        st.metric("System health", overall, help="Overall PASS / WARN / FAIL as CI recorded it in "
                  "output/system_health.json; every check carries its rule on the System Health page.")
        ui.stat_line([("Data freshness", _v(_stale)), ("Settlement", _v(_set)),
                      ("CLV coverage", _v(_clv)),
                      ("Production engine", f"{_sel.get('production', '—')} · shadow {_sel.get('shadow', '—')}")])
        _cnt = sh.get("counts") or {}
        st.markdown(ui.status_chip(overall) + " "
                    + (ui.status_chip("PROTECTED", label="PROTECTED TRACKS UNCHANGED")
                       if _prot.get("status") == "GREEN" else ui.status_chip(_prot.get("status", "NO_DATA"),
                                                                             label="PROTECTED TRACKS: CHECK"))
                    + f" :gray[{_cnt.get('GREEN', 0)} green · {_cnt.get('AMBER', 0)} amber · {_cnt.get('RED', 0)} red]")

st.caption(f"Settled since {_cut} (by match date), flat 1u stakes. Per-market breakdown on the "
           f"Tips page — one market carries most of the total.")

# ── best opportunities today ─────────────────────────────────────────────────────────────────
ui.section_header("Best opportunities today",
                  ("Frozen this morning and scored." if bb_frozen else "Preview — frozen when the morning album goes out.")
                  + " Best bets = most expected profit among tips with at least a 45% chance (learned from past "
                    "results). The tier is shown as metadata only.")
if bb.empty:
    ui.empty_state("NO_DATA", "No Best-bets list for today yet.")
else:
    _cols = st.columns(3)
    for i, r in bb.head(3).iterrows():
        with _cols[i % 3]:
            ui.selection_card(league=f"#{int(r['rank'])} · {r['league']}", kickoff="", home=r["home_team"],
                              away=r["away_team"], pick=str(r["pick"]), odds=r["odds"],
                              production_state="", decision_state="", tier=r["signal_tier"], show_plan=False,
                              supporting=[("Chance", ui.fmt_prob(r["p_win"])),
                                          ("Expected profit", ui.fmt_roi(r["ev"]))])
    if len(bb) > 3:
        st.caption("Also on the list: " + " · ".join(
            f"#{int(r['rank'])} {r['home_team']} v {r['away_team']} ({r['pick']} @ {ui.fmt_odds(r['odds'])})"
            for _, r in bb.iloc[3:].iterrows()) + ". All three lists are on the Tips page.")

# ── recent performance ───────────────────────────────────────────────────────────────────────
ui.section_header("Recent performance", "Staked P/L per market, flat 1u, SNIPER + MARKSMAN, settled since the "
                  "cutoff. The line on the P/L card above is the cumulative total by match date. A sample, "
                  "not a track record.")
if perf["n"]:
    ui.compact_table(perf["by_market"], {"Bets": ui.num_col("Bets", fmt="%d"),
                                         "P/L": ui.num_col("P/L", fmt="%+.2f"),
                                         "Run": ui.spark_col("Run", help="Running total in units, one point "
                                                             "per settled bet; each row on its own scale — "
                                                             "compare the shape, read P/L for the level.")})
    _thin = perf["by_market"][perf["by_market"]["Bets"] < ui.SMALL_SAMPLE_FLOOR]
    if len(_thin):
        st.markdown(" ".join(ui.sample_marker(int(r["Bets"])) + f" :gray[{r['Market']}]"
                             for _, r in _thin.iterrows()))
else:
    ui.empty_state("NO_DATA", "No settled staked bets since the cutoff yet.")

# ── needs attention ──────────────────────────────────────────────────────────────────────────
ui.section_header("Needs attention", "Only what is not healthy, worst first — from system_health.json, the "
                  "artifact manifest and the API usage log. Full rules on the System Health page.")
ui.needs_attention(today_view.attention(sh, man, budget))

# ── Disclaimer & Terms (shown on every dashboard page) ──
from utils.disclaimer import disclaimer_footer  # noqa: E402
disclaimer_footer()
