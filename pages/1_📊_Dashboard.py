"""Tips — every live tip, what production does with it, and what the shadow engine thinks of it.

Visual redesign 2026-10-25 (owner brief), PRESENTATION ONLY:
  * a compact filter toolbar (date, market, league, model, production state, view mode); legacy tier,
    drift and ordering sit in "Advanced filters";
  * "Next up" selection cards above the table: production state (stake plan) and the market-first
    shadow view (fair / decision probability, EV and EV lower bound, engine, evidence) side by side,
    the legacy tier last and neutral;
  * a CURRENT (legacy tier ladder) vs SHADOW (market-first) strip that never implies SNIPER = strongest;
  * "All selections" as a sortable dataframe (numbers stay numeric, formatted by column_config);
  * top picks, settled performance, CLV, half-time and suppressed fixtures in tabs below.

The board rows are telegram_bot.tips_table.tips_rows (the Telegram image's rows); production and shadow
columns come from output/stake_plan.csv and output/edge_engine_shadow.csv via dashboard_data.tips_view.
Every figure the page showed before is computed exactly as before (tests/test_home_tips_values.py).
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _read_csv(*args, **kwargs):
    """pd.read_csv with paper leagues removed — they are collected, never counted.

    See config.PAPER_LEAGUES. Every KPI on this page reads through here so a paper
    league cannot reach a total by way of a path that forgot the rule.
    """
    import config as _cfg
    return _cfg.drop_paper_leagues(pd.read_csv(*args, **kwargs))
import streamlit as st  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard_access import require as _require_tier  # noqa: E402
_require_tier("pro")  # tier guard (dashboard_access.PAGES), 2026-10-18

import config  # noqa: E402
import dashboard_ui as ui  # noqa: E402
from src.tier_labels import GAP_EXPLAINER, PAPER_TIER_LABEL, STAKED_LABEL, tier_label  # noqa: E402
from dashboard_data import tips_view as TV  # noqa: E402

st.set_page_config(page_title="Tips | Wowza", page_icon="📊", layout="wide")
ui.autorefresh(5, key="dash_refresh")

PYTHON = str(Path(sys.executable))
V9_DIR = str(Path(__file__).resolve().parents[1])
_TZ_NAME = TV.TZ_NAME


@st.cache_data(ttl=30)
def load_bets():
    f = config.OUTPUT_DIR / "bets.csv"
    if not f.exists():
        return pd.DataFrame()
    df = _read_csv(f)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Ensure optional columns exist
    for col in ["model_type", "drift_signal", "signal_tier", "best_edge", "best_side"]:
        if col not in df.columns:
            df[col] = ""
    df["model_type"] = df["model_type"].fillna("").astype(str)
    # A blank tag must never default to one model — derive it from the league so new-format
    # rows can't show up under 'standard' (canonical map in config.model_type_for_league).
    _blank_mt = df["model_type"].str.strip().isin(["", "nan"])
    if _blank_mt.any() and "league" in df.columns:
        df.loc[_blank_mt, "model_type"] = df.loc[_blank_mt, "league"].map(config.model_type_for_league)
    df["drift_signal"] = df["drift_signal"].fillna("New").astype(str)
    df["signal_tier"] = df["signal_tier"].fillna("").astype(str)
    return df


@st.cache_data(ttl=60)
def load_board(date_from: str, date_to: str) -> pd.DataFrame:
    """The SAME rows the Telegram tips image draws (telegram_bot.tips_table): live tips from the
    two committed ledgers, first side per fixture, paper leagues out."""
    from telegram_bot.tips_table import tips_rows
    return tips_rows(date_from, date_to)


@st.cache_data(ttl=60)
def load_board_attached(date_from: str, date_to: str) -> pd.DataFrame:
    """load_board's rows (unchanged, same order) with the stake plan's and the shadow engine's recorded
    columns attached by left join. Filtering after the join keeps exactly the rows filtering before it would."""
    return TV.attach(load_board(date_from, date_to), TV.load_plan(), TV.load_shadow())


@st.cache_data(ttl=300)
def _top_picks(day: str):
    from src import top_picks as _TP
    fz = _TP.frozen(day)
    return (fz, True) if not fz.empty else (_TP.todays_lists(day), False)


@st.cache_data(ttl=300)
def _top_scoreboard(days):
    from src import top_picks as _TP
    return _TP.scoreboard(days)


# ── load ───────────────────────────────────────────────────────────────────────────────────────
from utils.access import is_public  # noqa: E402

df_all = load_bets()
df = df_all[df_all["date"] >= pd.Timestamp.now().normalize()].copy() if not df_all.empty else df_all
_today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
_all_board = load_board(_today, "2099-12-31")
# File mtime is NOT a freshness signal: a git checkout resets it on every deploy. The newest recorded
# tip is a real timestamp.
_last = (pd.to_datetime(_all_board["generated_at"], errors="coerce").max() if len(_all_board) else pd.NaT)

# ── header (the ONE freshness statement) ───────────────────────────────────────────────────────
_fresh = (ui.freshness_badge("CURRENT", f"newest tip recorded {_last:%d/%m/%Y %H:%M} UTC")
          if pd.notna(_last) else ui.freshness_badge("NO_DATA", "no tip recorded"))
ui.page_header("Tips", "Every live tip — one side per fixture, the same rows as the Telegram image — with what "
               "production stakes and what the market-first shadow engine thinks of it.",
               eyebrow_text="Today · Tips", freshness=_fresh + f" :gray[{len(df)} fixtures priced]")

# The admin actions run the pipeline in place, so they need a FULL checkout. The slim deployment
# (deploy/README_DASHBOARD_DEPLOY.md) carries only the dashboard code - no pipeline.py - so they are hidden.
if not is_public() and (Path(V9_DIR) / "pipeline.py").exists():
    with st.popover("Admin actions"):
        if st.button("🔄 Run Predict Now", width="stretch"):
            with st.spinner("Fetching fixtures & computing predictions..."):
                result = subprocess.run([PYTHON, "pipeline.py", "--mode", "predict"],
                                        cwd=V9_DIR, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                st.success("Predictions updated!")
                st.cache_data.clear()
            else:
                st.error(f"Error: {result.stderr[-500:]}")
        if st.button("📥 Update Results Now", width="stretch"):
            with st.spinner("Fetching match results..."):
                result = subprocess.run([PYTHON, "update_results.py"],
                                        cwd=V9_DIR, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                st.success("Results updated!")
                st.cache_data.clear()
            else:
                st.error(f"Error: {result.stderr[-500:]}")

if df_all.empty:
    ui.empty_state("NO_DATA", "No predictions yet — predict writes bets.csv on its next run.")
elif df.empty:
    ui.empty_state("NO_DATA", "No upcoming fixtures right now. Past matches are hidden — see Live for results.")

# ── filter toolbar ─────────────────────────────────────────────────────────────────────────────
_MARKETS = ["O/U 2.5", "BTTS", "O/U 1.5", "O/U 3.5"]
_STATES = ["ACTIVE", "ACTIVE_LOW", "PAPER", "BLOCKED", TV.NOT_IN_PLAN]
f1, f2, f3, f4 = st.columns([2, 2, 3, 2])
with f1:
    _hi = df["date"].max() if len(df) else pd.Timestamp(_today)
    if len(_all_board):
        _hi = max(_hi, pd.Timestamp(_all_board["date"].max())) if pd.notna(_hi) else pd.Timestamp(_all_board["date"].max())
    date_range = st.date_input("Date range", value=(pd.Timestamp(_today).date(), _hi.date()), key="dash_dates")
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        date_from, date_to = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    else:
        date_from = date_to = pd.Timestamp(date_range[0]) if date_range else pd.Timestamp(_today)
with f2:
    market_filter = st.multiselect("Market", _MARKETS, default=_MARKETS, key="dash_market")
with f3:
    all_leagues = sorted(df["league"].dropna().unique().tolist()) if "league" in df.columns else []
    league_filter = st.multiselect("League", all_leagues, default=all_leagues, key="dash_league")
with f4:
    available_models = sorted(df["model_type"].unique().tolist()) if len(df) else []
    available_models = available_models or ["standard", "new_format"]
    model_filter = st.multiselect("Model", available_models, default=available_models)
g1, g2 = st.columns([3, 2])
with g1:
    state_filter = st.multiselect("Production state", _STATES, default=_STATES, key="dash_state",
                                  help="From the stake plan production wrote for each tip. NOT IN PLAN = the tip "
                                       "is in the ledger but not in the current stake plan (sent on an earlier run).")
with g2:
    mode = st.segmented_control("View", ["Production", "Research"], default="Production", key="dash_mode",
                                help="Research adds the market-first shadow columns (fair and decision "
                                     "probability, EV, EV lower bound, engine) to the table.") or "Production"
with st.expander("Advanced filters"):
    a1, a2, a3 = st.columns(3)
    with a1:
        tier_filter = st.multiselect("Gap size", list(TV.TIER_ORDER), default=list(TV.TIER_ORDER),
                                     format_func=tier_label, help=GAP_EXPLAINER)
    with a2:
        drift_filter = st.multiselect("Drift signal", ["Confirmed", "Neutral", "New", "Conflicted"],
                                      default=["Confirmed", "Neutral", "New", "Conflicted"],
                                      help="O/U 2.5 only; a side-market tip has no drift and is never hidden by it.")
    with a3:
        _order = st.radio("Order within a day", ["Kickoff time", "Best first"], key="dash_order")

# ── board (same rows + same filters as before; production/shadow columns attached) ─────────────
board = load_board_attached(date_from.strftime("%Y-%m-%d"), date_to.strftime("%Y-%m-%d"))
if len(board):
    _keep = board["tier"].isin(tier_filter) & board["market"].map(TV.market_group).isin(market_filter)
    if model_filter:
        _keep &= board["model"].isin(model_filter)
    if league_filter:
        _keep &= board["league"].isin(league_filter)
    _keep &= board["drift"].isna() | board["drift"].isin(drift_filter)
    board = board[_keep & board["production_state"].isin(state_filter)]

if board.empty:
    ui.empty_state("NO_DATA", "No tips match the current filters.")
else:
    _staked = int(board["tier"].isin(["SNIPER", "MARKSMAN"]).sum())
    _future = pd.to_datetime(board["kickoff_utc"], errors="coerce", utc=True)
    _next = _future[_future > pd.Timestamp.now(tz="UTC")].min()
    ui.metric_grid([
        {"label": "Tips shown", "value": f"{len(board)}",
         "help": "Every live tip in the date range, all markets, one side per fixture."},
        {"label": f"Staked ({STAKED_LABEL})", "value": f"{_staked}",
         "help": f"On O/U 2.5 every tier, {PAPER_TIER_LABEL} included, carries the same paper stake (the gap "
                 f"size does not rank outcomes there); on other markets {PAPER_TIER_LABEL} is tracked but not "
                 f"staked. {GAP_EXPLAINER}"},
        {"label": "Fixtures", "value": f"{board[['date', 'home_team', 'away_team']].drop_duplicates().shape[0]}"},
        {"label": "Next kickoff", "value": (f"{_next.tz_convert(_TZ_NAME):%a %H:%M}" if pd.notna(_next) else "—"),
         "help": f"Israel time ({_TZ_NAME})."},
    ], per_row=4)

    loc = TV.order(TV.localise(board), _order)

    # ── next up: selection cards ───────────────────────────────────────────────────────────────
    _up = loc[~loc["started"]].sort_values(["_ko", "_t"], na_position="last").head(3)
    ui.section_header("Next up", "The soonest kickoffs still to play — ordered by time, not by tier. Production "
                      "state is what the bankroll policy did; the shadow view is the market-first engine, which "
                      "never stakes.")
    if _up.empty:
        ui.empty_state("NO_DATA", "Every tip in the current filter has already kicked off.")
    else:
        for i, (_, r) in enumerate(_up.iterrows()):
            if i % 3 == 0:                      # one row of 3 at a time, so a phone stacks them in time order
                cols = st.columns(3)
            _ko = r["_local"].strftime("%a %H:%M") if pd.notna(r["_local"]) else "–"
            _ev = (f"{ui.fmt_roi(r['ev_point'])} / {ui.fmt_roi(r['ev_lower_bound'])}"
                   if pd.notna(r.get("ev_point")) else ui.fmt_roi(r["ev_lower_bound"]))
            _stake = (f"{r['final_fraction'] * 100:.2f}%" if pd.notna(r["final_fraction"]) else "—")
            _evid = []
            if isinstance(r.get("clv_evidence_state"), str):
                _evid.append(ui.evidence_badge("CLV", r.get("clv_track_n") if pd.notna(r.get("clv_track_n")) else None,
                                               state=r["clv_evidence_state"]))
            if isinstance(r.get("engine_version"), str):
                _evid.append(f":gray[engine {r['engine_version']}]")
            ui.selection_card(
                league=r["league"], kickoff=_ko, home=r["home_team"], away=r["away_team"],
                pick=TV.tip_text(r["market"]), odds=pd.to_numeric(r["odds"], errors="coerce"),
                production_state=r["deployment_state"] if r["in_plan"] else "",
                decision_state=r["decision_state"] if isinstance(r["decision_state"], str) else "",
                tier=r["tier"],
                supporting=[("Fair prob", ui.fmt_prob(r["p_market_fair"])),
                            ("Decision prob", ui.fmt_prob(r["p_decision"])),
                            ("EV / EV lower bound" if pd.notna(r.get("ev_point")) else "EV lower bound", _ev),
                            ("Policy stake", _stake),
                            ("Edge (recorded)", "–" if pd.isna(r["edge"]) else f"+{r['edge']:.1f}%")],
                evidence=" ".join(_evid), container=cols[i % 3])

    # ── CURRENT vs SHADOW ──────────────────────────────────────────────────────────────────────
    ui.section_header("Current vs shadow", "Two different questions. CURRENT is the legacy tier ladder production "
                      "sends — tier = size of the model-vs-price gap. SHADOW is the market-first engine's verdict "
                      "on the same tips. A higher tier is not stronger evidence.")
    _counts = " · ".join(f"{int((board['tier'] == t).sum())} {tier_label(t)}" for t in TV.TIER_ORDER
                         if (board["tier"] == t).any())
    _sh = TV.shadow_summary(board)
    cur, sha = st.columns(2)
    with cur.container(border=True):
        ui.eyebrow("Current · legacy tier ladder")
        st.markdown(ui.status_chip("LIVE", label="PRODUCTION") + f" &nbsp;{_counts}")
        _ps = board["production_state"].value_counts()
        ui.stat_line([(s, f"{int(_ps.get(s, 0))}") for s in _STATES if int(_ps.get(s, 0))])
        st.caption("Production state per tip, from the stake plan. O/U 2.5 tiers are staked equally "
                   f"({tier_label('SNIPER')} −9.9%, {tier_label('MARKSMAN')} −5.8%, {tier_label('VALUABLE')} "
                   "−0.8% since 26 Aug); on BTTS and Over 1.5 the edge "
                   "does rank outcomes, so the ladder stays.")
    with sha.container(border=True):
        ui.eyebrow(f"Shadow · market-first ({_sh['engine'] or 'engine n/a'})")
        st.markdown(ui.status_chip("SHADOW") + " &nbsp;" + " · ".join(
            f"{v} {ui.status_meta(k)[1]}" for k, v in sorted(_sh["by_state"].items()))
            + (f" · {_sh['not_evaluated']} NOT EVALUATED" if _sh["not_evaluated"] else ""))
        ui.stat_line([("Median EV lower bound", ui.fmt_roi(_sh["ev_lb_median"])),
                      ("EV lower bound > 0", f"{_sh['ev_lb_positive']} of {_sh['evaluated']}")])
        st.caption("Shadow only — logs and never stakes. NOT EVALUATED = the tip is not in the current stake plan, "
                   "so no shadow verdict was recorded beside it.")

    # ── all selections ─────────────────────────────────────────────────────────────────────────
    ui.section_header("All selections", "Kickoff in Israel time · the same tips as the Telegram image (from the "
                      "ledgers, first side per fixture, paper leagues excluded) · Edge is as recorded when the tip "
                      "was sent · not betting advice.")
    _tab = TV.selections_table(loc, research=(mode == "Research"))
    _emax = max(float(pd.to_numeric(board["edge"], errors="coerce").max() or 0), 1.0)
    _cfg = {
        "Day": st.column_config.DateColumn("Day", format="ddd D MMM", width="small"),
        "Kickoff": st.column_config.DatetimeColumn("Kickoff", format="HH:mm", width="small"),
        "Started": st.column_config.CheckboxColumn("Started", width="small"),
        "Match": ui.text_col("Match", width="medium"),
        "Rating": ui.num_col("Rating", fmt="%d/10", help="League rating 1-10 (Pro), the number the stake plan "
                             "sizes by.", width="small"),
        "Odds": ui.num_col("Odds", fmt="%.2f", width="small"),
        "Edge %": ui.bar_col("Edge %", max_value=_emax, fmt="+%.1f%%",
                             help="Model-vs-price edge when the tip was sent (this is what the legacy tier sizes)."),
        "Model %": ui.num_col("Model %", fmt="%.0f%%", help="Model probability (side markets)."),
        "Stake %": ui.num_col("Stake %", fmt="%.2f%%", help="Policy stake, % of the notional bankroll."),
        "Fair prob %": ui.num_col("Fair prob %", fmt="%.1f%%", help="De-vigged market probability (shadow)."),
        "Decision prob %": ui.num_col("Decision prob %", fmt="%.1f%%",
                                      help="Market + capped model residual (shadow)."),
        "EV %": ui.num_col("EV %", fmt="%+.1f%%", help="Shadow point EV per unit staked."),
        "EV lower bound %": ui.num_col("EV lower bound %", fmt="%+.1f%%",
                                       help="Shadow EV after the uncertainty penalty; > 0 is what a shadow BET needs."),
    }
    ui.compact_table(_tab, {k: v for k, v in _cfg.items() if k in _tab.columns}, height=420)
    st.download_button("Download as CSV", board.drop(columns=["generated_at"], errors="ignore")
                       .to_csv(index=False).encode("utf-8"), "wowza_tips.csv", "text/csv")

# ── detail tabs ────────────────────────────────────────────────────────────────────────────────
t_top, t_perf, t_clv, t_ht, t_sup = st.tabs(["Top picks", "Settled performance", "Closing line value",
                                             "Half-time O/U", "Suppressed"])

# Top picks: 3 lists of 5 (src/top_picks.py, 2026-10-18). People take a few tips, not 60.
with t_top:
    try:
        from src.top_picks import LIST_LABEL as _TPL, LISTS as _TPLISTS
        _tp, _is_frozen = _top_picks(_today)
        st.caption(("Frozen this morning and scored." if _is_frozen else "Preview — frozen when the morning album goes out.")
                   + " Best bets: most expected profit with at least a 45% chance · Safest: the 5 shortest prices · "
                     "Value: most expected profit. Chance = honest win estimate learned from past results.")
        if _tp is None or _tp.empty:
            ui.empty_state("NO_DATA", "No tips for today yet.")
        else:
            _cols = st.columns(3)
            for _c, _name in zip(_cols, _TPLISTS):
                _x = _tp[_tp["list"] == _name].sort_values("rank")
                with _c:
                    ui.eyebrow(_TPL[_name])
                    if _x.empty:
                        st.caption("nothing qualifies today")
                        continue
                    ui.compact_table(pd.DataFrame({
                        "Match": _x["home_team"] + " v " + _x["away_team"], "Pick": _x["pick"],
                        "Odds": pd.to_numeric(_x["odds"], errors="coerce").round(2),
                        "Chance": pd.to_numeric(_x["p_win"], errors="coerce") * 100}),
                        {"Odds": ui.num_col("Odds", fmt="%.2f"), "Chance": ui.num_col("Chance", fmt="%.0f%%")})
        _win = st.radio("Top-picks record", [7, 30, 0], horizontal=True,
                        format_func=lambda w: "All time" if w == 0 else f"Last {w} days", key="tp_window")
        _sb = _top_scoreboard(_win or None)
        if _sb.empty:
            st.caption("The record starts with the first frozen day.")
        else:
            ui.compact_table(_sb.assign(list=_sb["list"].map(_TPL), hit_rate=(_sb["hit_rate"] * 100).round(1),
                                        roi=(_sb["roi"] * 100).round(1), zero_days=(_sb["zero_days"] * 100).round(0))
                             .rename(columns={"list": "List", "days": "Days", "picks": "Picks", "wins": "Wins",
                                              "hit_rate": "Hit %", "roi": "ROI %", "zero_days": "0/5 days %"}),
                             {"Hit %": ui.num_col("Hit %", fmt="%.1f%%"), "ROI %": ui.num_col("ROI %", fmt="%+.1f%%"),
                              "0/5 days %": ui.num_col("0/5 days %", fmt="%.0f%%")})
        with st.expander("Weekly table — live record and backtest, week by week"):
            from dashboard_data import pro as _pro
            _wk = _pro.read_csv(_pro._f("output/studies/top_picks_weekly.csv"))
            if _wk is None or _wk.empty:
                st.caption("No weekly table yet — Pro writes it daily.")
            else:
                _lab = {"BEST_BETS": "Best bets", "SAFEST": "Safest", "VALUE": "Value", "RANDOM_5": "Random 5",
                        "CURRENT": "Current order"}
                _kind = {"live_this_week": "Live · this week", "live_to_date": "Live · to date", "backtest": "Backtest"}
                _w = _wk.assign(List=_wk["list"].map(_lab), Kind=_wk["kind"].map(_kind),
                                **{"Hit %": (_wk["hit_rate"] * 100).round(0), "ROI %": (_wk["roi"] * 100).round(0),
                                   "0/5 days %": (_wk["zero_days"] * 100).round(0)})
                if "avg_odds" not in _w.columns:
                    _w["avg_odds"] = float("nan")
                ui.compact_table(_w.sort_values(["week", "kind", "list"], ascending=[False, True, True])[
                    ["week", "Kind", "List", "Hit %", "avg_odds", "ROI %", "0/5 days %", "days", "picks"]].rename(
                    columns={"week": "Week", "avg_odds": "Avg odds", "days": "Days", "picks": "Picks"}),
                    {"Hit %": ui.num_col("Hit %", fmt="%.0f%%"), "ROI %": ui.num_col("ROI %", fmt="%+.0f%%"),
                     "0/5 days %": ui.num_col("0/5 days %", fmt="%.0f%%"), "Avg odds": ui.num_col("Avg odds")})
                st.caption("Live = the lists users were shown, frozen each morning and graded. Backtest = the "
                           "walk-forward test re-run on all history each day. Earlier weeks are never rewritten.")
        with st.expander("Best leagues — results by league"):
            from dashboard_data import pro as _pro
            _lg = _pro.read_csv(_pro._f("output/studies/league_results_weekly.csv"))
            if _lg is None or _lg.empty:
                st.caption("No league table yet — Pro writes it daily.")
            else:
                _wnd = st.radio("Window", ["season", "last_30_days"], horizontal=True, key="lg_window",
                                format_func=lambda w: "Season to date" if w == "season" else "Last 30 days")
                _t = _lg[(_lg["week"] == _lg["week"].max()) & (_lg["window"] == _wnd)].sort_values("rank")
                ui.compact_table(_t.assign(**{"Hit %": (_t["hit_rate"] * 100).round(0), "ROI %": (_t["roi"] * 100).round(0),
                                              "Shrunk ROI %": (_t["shrunk_roi"] * 100).round(0)})[
                    ["rank", "league", "tips", "Hit %", "avg_odds", "ROI %", "Shrunk ROI %", "best_market", "label"]].rename(
                    columns={"rank": "#", "league": "League", "tips": "Tips", "avg_odds": "Avg odds",
                             "best_market": "Best market", "label": "Label"}),
                    {"Hit %": ui.num_col("Hit %", fmt="%.0f%%"), "ROI %": ui.num_col("ROI %", fmt="%+.0f%%"),
                     "Shrunk ROI %": ui.num_col("Shrunk ROI %", fmt="%+.0f%%"), "Avg odds": ui.num_col("Avg odds")})
                st.caption("Ranked by ROI shrunk toward the average (a lucky handful cannot top it). PROVEN / LOSING = "
                           "clear of break-even after correcting for testing many leagues at once.")
    except Exception as _e:                                              # noqa: BLE001 - a viewer never dies
        ui.empty_state("ERROR", f"Top picks unavailable ({type(_e).__name__})")

# ── Settled performance ──────────────────────────────────────────────────────────────────────
# FOUR FILTERS, EACH OF WHICH CHANGES THE ANSWER, so all are stated on screen: STAKED TIERS ONLY
# (VALUABLE is not a bet), FROM PERFORMANCE_CUTOFF_DATE, SETTLED ONLY, and broken out PER MARKET
# (a single blended number is either the wrong sign or the wrong story). n is shown beside the total.
_STAKED_TIERS = ("SNIPER", "MARKSMAN")
_MKT_SOURCES = [("bets_ledger.csv", None, "O/U 2.5"),
                ("side_bets_ledger.csv", "market", None),
                ("ht_ledger.csv", None, "HT O/U")]
_cut = pd.Timestamp(getattr(config, "PERFORMANCE_CUTOFF_DATE", "1970-01-01"))


@st.cache_data(ttl=60)
def _settled_staked():
    """Settled staked rows per market (the ledgers are read once a minute, not on every rerun)."""
    _parts, _missing, _unreadable = [], [], []
    for _f, _mkc, _const in _MKT_SOURCES:
        _fp = config.OUTPUT_DIR / _f
        if not _fp.exists():
            _missing.append(_f)
            continue
        try:
            _d = _read_csv(_fp, low_memory=False)
            _d["_dt"] = pd.to_datetime(_d.get("match_date"), errors="coerce")
            _d["_pnl"] = pd.to_numeric(_d.get("pnl"), errors="coerce")
            _oc = "odds" if "odds" in _d.columns else "entry_odds"
            _d["_odds"] = pd.to_numeric(_d.get(_oc), errors="coerce")
            _d["_mkt"] = (_d[_mkc].astype(str) if _mkc and _mkc in _d.columns else _const)
            _keep = (_d.get("signal_tier").isin(_STAKED_TIERS)
                     & _d["_pnl"].notna() & (_d["_dt"] >= _cut))
            if _keep.any():
                _parts.append(_d[_keep][["_mkt", "_dt", "_pnl", "_odds", "signal_tier"]])
        except Exception as _e:                             # noqa: BLE001
            _unreadable.append(f"{_f} unreadable ({type(_e).__name__}: {_e})")
    return ((pd.concat(_parts, ignore_index=True).sort_values("_dt") if _parts else pd.DataFrame()),
            _missing, _unreadable)


@st.cache_data(ttl=60)
def _clv_rows() -> pd.DataFrame:
    _cl = _read_csv(config.OUTPUT_DIR / "bets_ledger.csv", low_memory=False)
    _cl["_clv"] = pd.to_numeric(_cl.get("clv_pct"), errors="coerce")
    _cl["_o"] = pd.to_numeric(_cl.get("odds"), errors="coerce")
    _cl["_c"] = pd.to_numeric(_cl.get("closing_odds"), errors="coerce")
    _cl["_dt"] = pd.to_datetime(_cl.get("match_date"), errors="coerce")
    return _cl[_cl["_clv"].notna()][["league", "_clv", "_o", "_c", "_dt"]]


_s, _missing, _unreadable = _settled_staked()

with t_perf:
    for _msg in _unreadable:
        st.caption(_msg)
    if not len(_s):
        ui.empty_state("NO_DATA", "No settled staked bets since the cutoff.")
    else:
        _tot = float(_s["_pnl"].sum())
        _hit = float((_s["_pnl"] > 0).mean())
        _THIN = 20
        _m = (_s.groupby("_mkt")
              .agg(Bets=("_pnl", "size"), PL=("_pnl", "sum"),
                   Hit=("_pnl", lambda x: 100.0 * (x > 0).mean()),
                   _avg_odds=("_odds", "mean"), _n_odds=("_odds", "count"))
              .reset_index().rename(columns={"_mkt": "Market"}))
        _m["Per bet"] = _m["PL"] / _m["Bets"]
        # Break-even at the price actually paid; suppressed when a fifth of the market's prices are missing.
        _m["Need"] = (100.0 / _m["_avg_odds"]).where(_m["_n_odds"] >= 0.8 * _m["Bets"])
        _seq = {k: [round(v, 3) for v in g.sort_values("_dt")["_pnl"].cumsum()]
                for k, g in _s.groupby("_mkt")}
        _m["Run"] = _m["Market"].map(lambda k: _seq.get(k) if len(_seq.get(k, [])) >= 3 else None)
        _m = _m.sort_values("Bets", ascending=False)

        _biggest = _m.iloc[_m["PL"].abs().values.argmax()]
        _ex = _s[_s["_mkt"] != _biggest["Market"]]
        ui.metric_grid([
            {"label": "P/L — all staked markets", "value": f"{_tot:+.2f}u",
             "delta": f"{_tot / len(_s):+.3f}u per bet",
             "help": f"Flat 1u stakes, {STAKED_LABEL} only. Read the per-market table below "
                     "before this number — it is a sum of very unequal samples."},
            {"label": f"Excluding {_biggest['Market']}",
             "value": f"{float(_ex['_pnl'].sum()):+.2f}u" if len(_ex) else "—",
             "delta": f"{len(_ex)} bets", "delta_color": "off",
             "help": f"{_biggest['Market']} carries {float(_biggest['PL']):+.2f}u on "
                     f"{int(_biggest['Bets'])} bets — the largest single contribution. Shown so the "
                     "total cannot be read as broad-based when it is not."},
            {"label": "Settled bets", "value": f"{len(_s):,}",
             "help": f"Since {_cut.date()} (PERFORMANCE_CUTOFF_DATE). Small samples move a lot."},
            {"label": "Days covered", "value": f"{_s['_dt'].dt.date.nunique()}",
             "help": f"{_s['_dt'].min().date()} to {_s['_dt'].max().date()}. Hit rate across all "
                     f"markets is {_hit:.1%}, but it is only meaningful per market — each has a "
                     f"different break-even price."},
        ], per_row=4)

        ui.table(_m[["Market", "Bets", "PL", "Per bet", "Hit", "Need", "Run"]].round(2),
                 sparks={"Run": {"help": "Running total in units, one point per settled bet, oldest "
                                         "to newest. Each row is scaled to its own range — compare "
                                         "the SHAPE, not the height, and read P/L for the level."}},
                 column_config={"PL": ui.num_col("PL", fmt="%+.2f"), "Per bet": ui.num_col("Per bet", fmt="%+.2f"),
                                "Hit": ui.num_col("Hit", fmt="%.2f%%"), "Need": ui.num_col("Need", fmt="%.2f%%")})
        _thin = _m[_m["Bets"] < _THIN]
        if len(_thin):
            st.markdown(" ".join(ui.sample_marker(int(r["Bets"]), floor=_THIN) + f" :gray[{r['Market']}]"
                                 for _, r in _thin.iterrows()))
        st.caption(
            f"Flat 1u stakes · **staked tiers only** ({STAKED_LABEL} — {PAPER_TIER_LABEL} is not a bet) · "
            f"settled only · from {_cut.date()}. **Hit** is the share won; **Need** is the "
            f"break-even rate implied by the average price paid, so Hit above Need is a profitable "
            f"market and below it is not."
            + (f" Thin: " + ", ".join(f"{r['Market']} n={int(r['Bets'])}" for _, r in _thin.iterrows())
               + " — at these counts one result moves P/L by a whole unit." if len(_thin) else ""))

        # Cumulative curve per market rather than one blended line.
        _cur = (_s.assign(_c=_s.groupby("_mkt")["_pnl"].cumsum())
                .pivot_table(index="_dt", columns="_mkt", values="_c", aggfunc="last")
                .ffill())
        if len(_cur) > 1:
            st.line_chart(_cur, height=280)
            st.caption(f"Cumulative units by market. Flat after a market's last settled bet — the "
                       f"line is carried forward, not still running. n = {len(_s)} across "
                       f"{_s['_mkt'].nunique()} markets over {_s['_dt'].dt.date.nunique()} days, "
                       f"which is a sample and not a track record.")

        _by = (_s.groupby(["signal_tier", "_mkt"])["_pnl"].agg(Bets="size", **{"P/L": "sum"})
               .reset_index().rename(columns={"signal_tier": "Tier", "_mkt": "Market"}))
        _by["Per bet"] = (_by["P/L"] / _by["Bets"]).round(3)
        with st.expander("By tier and market"):
            ui.table(_by.sort_values("Bets", ascending=False),
                     column_config={"P/L": ui.num_col("P/L", fmt="%+.2f"), "Per bet": ui.num_col("Per bet", fmt="%+.3f")})
            if _missing:
                st.caption("Not present, so contributing nothing: " + ", ".join(_missing) + ".")

# ── Closing line value ───────────────────────────────────────────────────────────────────────
# Reported as a MEDIAN: the raw mean is inflated by closing prices that cannot belong to the same
# selection. |CLV| <= 25% is the same cut v11 applies (CLV_PLAUSIBLE_ABS), so the two repos cannot
# quote different CLV for the same rows. The exact-zero share is shown, not hidden: drift.py stores
# price CHANGES only, so a flat row is a genuine flat line OR a missing observation.
with t_clv:
    if not len(_s):
        ui.empty_state("NO_DATA", "No settled staked bets since the cutoff.")
    else:
        try:
            _raw = _clv_rows()
            _PLAUS = 25.0
            _ok = _raw[_raw["_clv"].abs() <= _PLAUS].copy()
            if not len(_ok):
                ui.empty_state("NO_DATA", "No plausible CLV observations yet.")
            else:
                _flat = (_ok["_c"] == _ok["_o"])
                _moved = _ok[~_flat]
                ui.metric_grid([
                    {"label": "Median CLV", "value": f"{_ok['_clv'].median():+.2f}%",
                     "help": "Median, not mean. The mean over these rows is "
                             f"{_ok['_clv'].mean():+.2f}% and the raw unfiltered mean is "
                             f"{_raw['_clv'].mean():+.2f}% — inflated by impossible closing prices."},
                    {"label": "Clean observations", "value": f"{len(_ok):,}",
                     "delta": f"−{len(_raw) - len(_ok)} rejected", "delta_color": "off",
                     "help": f"Of {len(_raw):,} rows with a CLV. Rejected where |CLV| > {_PLAUS:.0f}% "
                             "— the same cut v11 uses."},
                    {"label": "Beat the close", "value": f"{100 * (_moved['_clv'] > 0).mean():.1f}%",
                     "help": f"Share of the {len(_moved):,} rows that actually moved. Including the "
                             f"{int(_flat.sum())} flat rows it reads "
                             f"{100 * (_ok['_clv'] > 0).mean():.1f}%."},
                    {"label": "No recorded move", "value": f"{100 * _flat.mean():.0f}%",
                     "help": f"{int(_flat.sum())} rows where closing == entry. drift.py stores price "
                             "CHANGES only, so this is a genuine flat line for a polled fixture and a "
                             "missing observation for an unpolled one. Not separable from the "
                             "committed data."},
                ], per_row=4)
                _wk = (_moved.dropna(subset=["_dt"])
                       .assign(_w=lambda x: x["_dt"].dt.to_period("W").dt.start_time)
                       .groupby(["league", "_w"])["_clv"].median().reset_index())
                # A 1-point series reads as "steady" when it means "one week of data": blank under 3.
                _MIN_PTS = 3
                _tr = _wk.groupby("league")["_clv"].apply(
                    lambda s: [float(round(v, 2)) for v in s] if len(s) >= _MIN_PTS else None)
                _lg = (_moved.groupby("league")["_clv"]
                       .agg(**{"n": "size", "Median CLV %": "median",
                               "Beat close %": lambda s: 100 * (s > 0).mean()})
                       .reset_index().rename(columns={"league": "League"}))
                _lg["Trend"] = _lg["League"].map(_tr)
                _lg = _lg[_lg["n"] >= 5].sort_values("Median CLV %", ascending=False)
                if len(_lg):
                    # A SHARED y-scale, so a ±1% league never draws like a ±40% one.
                    _lim = float(max(5.0, _wk["_clv"].abs().max()))
                    ui.table(_lg.round(2),
                             sparks={"Trend": {"y_min": -_lim, "y_max": _lim,
                                               "help": "Weekly median CLV %, blank under "
                                                       f"{_MIN_PTS} weeks of data. Shared scale "
                                                       f"±{_lim:.0f}% across all rows."}},
                             column_config={"Median CLV %": ui.num_col("Median CLV %", fmt="%+.2f%%"),
                                            "Beat close %": ui.num_col("Beat close %", fmt="%.2f%%")})
                    st.caption(
                        f"Leagues with at least 5 rows that moved, of {_moved['league'].nunique()} "
                        f"with any. Flat rows excluded from these figures — see 'no recorded move'. "
                        f"n is small everywhere (largest {int(_lg['n'].max())}), and v11 wants 150 "
                        f"per segment before CLV gates a bet, so nothing here is settled.")
        except Exception as _e:                                  # noqa: BLE001
            ui.empty_state("ERROR", f"CLV panel unavailable ({type(_e).__name__}: {_e})")


# ── HT O/U Predictions ───────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_predictions_ht():
    f = config.OUTPUT_DIR / "predictions.csv"
    if not f.exists():
        return pd.DataFrame()
    preds = _read_csv(f)
    if "p_ht_over05" not in preds.columns:
        return pd.DataFrame()
    preds["date"] = pd.to_datetime(preds["date"], errors="coerce")
    today = pd.Timestamp.now().normalize()
    preds = preds[preds["date"] >= today].copy()
    # Only standard-format leagues (have HT data)
    preds = preds[preds["p_ht_over05"].notna()]
    return preds


with t_ht:
    ht_preds = load_predictions_ht()
    if ht_preds.empty:
        ui.empty_state("NO_DATA", "No half-time predictions for upcoming fixtures.")
    else:
        st.caption("Standard-format leagues only · Based on HT rolling features (HTHG/HTAG)")
        ht_rows = []
        for _, row in ht_preds.iterrows():
            p05 = float(row["p_ht_over05"])
            p15 = float(row["p_ht_over15"]) if "p_ht_over15" in row and pd.notna(row["p_ht_over15"]) else None
            ht_rows.append({
                "Date": str(row["date"])[:10],
                "League": row.get("league", ""),
                "Match": f"{row['home_team']} vs {row['away_team']}",
                "P(HT OVER 0.5)": p05 * 100,
                "Fair OVER 0.5": round(1 / max(p05, 0.01), 2),
                "Fair UNDER 0.5": round(1 / max(1 - p05, 0.01), 2),
                "P(HT OVER 1.5)": p15 * 100 if p15 else None,
                "Fair OVER 1.5": round(1 / max(p15, 0.01), 2) if p15 else None,
            })
        ui.compact_table(pd.DataFrame(ht_rows), {
            "P(HT OVER 0.5)": ui.num_col("P(HT OVER 0.5)", fmt="%.0f%%"),
            "P(HT OVER 1.5)": ui.num_col("P(HT OVER 1.5)", fmt="%.0f%%"),
            "Fair OVER 0.5": ui.num_col("Fair OVER 0.5"), "Fair UNDER 0.5": ui.num_col("Fair UNDER 0.5"),
            "Fair OVER 1.5": ui.num_col("Fair OVER 1.5")})
        st.caption("Compare fair prices against your bookmaker's HT market to find value")

# ── Suppressed fixtures ──────────────────────────────────────────────────────────────────────
with t_sup:
    avoided = df[df["bet"] == "AVOID"] if len(df) and "bet" in df.columns else pd.DataFrame()
    if avoided.empty:
        ui.empty_state("NO_DATA", "No fixtures suppressed in the current slate.")
    else:
        st.caption(f"{len(avoided)} fixtures suppressed (AVOID / both-losing guard)")
        ui.compact_table(avoided[["date", "league", "home_team", "away_team", "signal_tier", "model_type"]],
                         {"date": st.column_config.DateColumn("date", format="YYYY-MM-DD")})


# ── Disclaimer & Terms (shown on every dashboard page) ──
from utils.disclaimer import disclaimer_footer  # noqa: E402
disclaimer_footer()
