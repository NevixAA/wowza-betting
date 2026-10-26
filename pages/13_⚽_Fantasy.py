"""
Fantasy (FPL) — Wowza Fantasy V2.

A VIEWER. Everything on this page is precomputed by `python -m player_model.fantasy_v2` inside
fantasy_refresh.yml and read here from small CSV/JSON artifacts with column projection. No model
is trained, no parquet is read, and no file mtime is consulted at render time — the deployment has
~1 GB and has died twice from doing that work here.

THE MEAN FORECAST IS FPL's OWN ep_next unless the residual has passed its acceptance gate. FPL beat
the old standalone Wowza projection in every settled gameweek, and the Model lab says so on every
render. Wowza's job is the residual ("FPL baseline + Wowza residual", shadow until it earns the
mean), minutes, ceiling, role, multi-gameweek planning, captaincy risk and transfers. Fantasy is
scored in FPL points, never units or ROI — it is a prediction contest, not betting.

Layout (visual redesign stage 2a, 2026-10-25): organised around FPL tasks — Overview / Players /
Transfers / Captaincy / Fixtures / Squad / Model lab. Model and research detail lives in Model lab
so it never competes with a decision. Presentation only: every value comes from the same artifacts
as before; card helpers live in dashboard_data/fantasy_view.py. Streamlit renders every tab body on
each run, so nothing heavy may run unconditionally anywhere on this page (the squad solver and the
club squads stay behind buttons).
"""
from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dashboard_ui as ui  # noqa: E402
from dashboard_data import fantasy as fdata  # noqa: E402
from dashboard_data import fantasy_view as fv  # noqa: E402

# STALE-MODULE GUARD (2026-10-22). Streamlit Cloud re-runs page files on every deploy but keeps
# modules it already imported, so after the V2 update the long-running process served the OLD
# dashboard_data.fantasy and the page died with AttributeError on v2_projections. Reload once if the
# readers this page needs are missing.
if not all(hasattr(fdata, f) for f in ("v2_projections", "v2_summary", "v2_benchmark", "v2_transfers",
                                       "v2_fixture_matrix", "v2_scoreboard",
                                       "v2_scoreboard_reliability", "v2_pooled_reliability")):
    import importlib  # noqa: E402
    fdata = importlib.reload(fdata)
if not all(hasattr(fv, f) for f in ("player_card", "captain_card", "transfer_card", "fixture_heatmap_spec",
                                    "recommendations", "transfer_candidates", "fixture_agg")):
    import importlib  # noqa: E402
    fv = importlib.reload(fv)

st.set_page_config(page_title="Fantasy | Wowza", page_icon="⚽", layout="wide")
from dashboard_access import require as _require_tier  # noqa: E402
_require_tier("pro")  # tier guard (dashboard_access.PAGES), 2026-10-18

ui.autorefresh(minutes=10, key="fantasy_refresh")

POS_ORDER = fv.POS_ORDER


@st.cache_data(ttl=600, show_spinner=False)
def _load():
    return (fdata.v2_projections(), fdata.v2_summary(), fdata.v2_benchmark(),
            fdata.v2_transfers(), fdata.v2_fixture_matrix())


proj, summ, bench, trans, fxm = _load()


def _num(x, fmt="{:.2f}", none="—"):
    try:
        if x is None or x != x:
            return none
        return fmt.format(x)
    except Exception:                                                # noqa: BLE001
        return none


# ══ HEADER: gameweek, deadline, freshness, projection health ══════════════════════════════════
age_now = fdata.v2_fpl_age_hours()
fresh = ("UNKNOWN" if age_now is None else "CURRENT" if age_now <= 30 else
         "AGING" if age_now <= 54 else "STALE")
ui.page_header(
    "Fantasy", "Premier League Fantasy projections anchored on FPL's own expected points, with Wowza's "
    "minutes, role, fixture and uncertainty layer. Scored in **FPL points** — not betting tips.",
    eyebrow_text="FPL · Fantasy V2",
    freshness=ui.freshness_badge(fresh, f"FPL data {age_now:.0f}h old" if age_now is not None
                                 else "FPL data age unknown"))

if proj.empty or not summ:
    ui.empty_state("NO_DATA", "Fantasy V2 artifacts are not built yet. They are produced daily by the "
                   "Fantasy Refresh workflow (`python -m player_model.fantasy_v2`).")
    st.stop()

bs = summ.get("benchmark_status") or {}
ph = summ.get("projection_health") or {}
dl = pd.to_datetime(summ.get("deadline"), utc=True, errors="coerce")
now = pd.Timestamp.now(tz="UTC")
if pd.notna(dl):
    hrs = (dl - now).total_seconds() / 3600
    dl_item = {"label": "Deadline (UTC)", "value": dl.strftime("%a %d %b %H:%M"),
               "delta": f"{hrs:.0f}h to go" if hrs > 0 else "passed", "delta_color": "off"}
else:
    dl_item = {"label": "Deadline", "value": "—"}
ui.metric_grid([
    {"label": "Gameweek", "value": f"GW{summ.get('gameweek')}" if summ.get("gameweek") else "—",
     "help": "The gameweek these projections are FOR (FPL's next deadline)."},
    dl_item,
    {"label": "Projection health", "value": ph.get("state", "—"),
     "delta": f"{ph.get('players', 0)} players · {ph.get('minutes_disagreements', 0)} minutes flags",
     "delta_color": "off",
     "help": "FPL data age comes from the RECORDED fetch time in fpl_cache_meta.json, never from file "
             "mtime. The daily job runs once a day, so up to ~30h is normal."},
], per_row=3)
dec = bs.get("decision_residual")
if bs.get("residual_applied"):
    st.markdown(ui.status_chip("LIVE", label="RESIDUAL APPLIED") +
                f" **Mean forecast: {bs.get('mean_forecast')}.** The residual passed its out-of-sample "
                "acceptance gate and is applied.")
else:
    st.markdown(ui.status_chip("SHADOW") +
                f" **Mean forecast: {bs.get('mean_forecast', 'FPL ep_next')}.** FPL baseline + Wowza "
                f"residual is a *shadow* adjustment — acceptance gate **{dec or '—'}** (mode "
                f"`{bs.get('residual_mode', 'shadow')}`); not added to the headline number until it beats "
                "FPL over enough completed gameweeks.")
if fresh in ("STALE", "UNKNOWN"):
    ui.empty_state("STALE", "FPL snapshot is stale or undatable — injuries, prices and the squad list "
                   "may be out of date.")
rc1, rc2 = st.columns([4, 1])
rc1.caption(f"Archive: GWs {ph.get('archived_gameweeks')} · final {ph.get('final_gameweeks')} · "
            f"provisional {ph.get('provisional_gameweeks')} · built {summ.get('generated_at')}")
if rc2.button("Reload artifacts"):
    st.cache_data.clear()
    st.rerun()

import dashboard_fantasy_tools as tools  # noqa: E402
if not all(hasattr(tools, f) for f in ("board", "differentials", "prop_leaderboards", "set_pieces",
                                       "transfer_helper", "alerts", "chip_advisor", "lineup",
                                       "mini_league", "club_squads")):
    import importlib  # noqa: E402
    tools = importlib.reload(tools)

v1 = tools.board()

t_over, t_play, t_tr, t_cap, t_fix, t_sq, t_lab = st.tabs(
    ["Overview", "Players", "Transfers", "Captaincy", "Fixtures", "Squad", "Model lab"])

agg = fv.fixture_agg(fxm) if not fxm.empty else pd.DataFrame()

# ══ OVERVIEW ═══════════════════════════════════════════════════════════════════════════════════
with t_over:
    ui.section_header("Captaincy", "Headline mean. The shadow ranking (FPL baseline + Wowza residual) "
                      "is in the Captaincy tab.")
    cap_h = summ.get("captaincy") or {}
    for col, key, title in zip(st.columns(3), ("SAFE", "BALANCED", "HIGH_UPSIDE"),
                               ("SAFE", "BALANCED", "HIGH UPSIDE")):
        c = cap_h.get(key)
        if not c:
            ui.empty_state("NO_DATA", f"{title}: no eligible starter", container=col)
            continue
        fv.captain_card(c, fv.lookup(proj, c), title, container=col)

    ui.section_header("Top transfers", "Best precomputed moves among the most-owned players, ranked "
                      "by 5-gameweek gain. Plan your own in the Transfers tab.")
    if trans.empty:
        ui.empty_state("NO_DATA", "No precomputed transfers in the artifact.")
    else:
        for col, (_, t) in zip(st.columns(3), trans.head(3).iterrows()):
            fv.precomputed_move_card(t, container=col)

    ui.section_header("Best value", "Startable players (P(start) ≥ 60 %, not doubtful) with the most "
                      "5-gameweek expected points per £m.")
    bv = fv.best_value(proj, 3)
    for col, (_, r) in zip(st.columns(3), bv.iterrows()):
        fv.player_card(r, eyebrow_text=f"BEST VALUE · {r['value_5gw_per_m']:.2f} pts/£m (5 GW)",
                       container=col)

    ui.section_header("Fixture swing", "Easiest and hardest runs over the next gameweeks by mean "
                      "official FDR. Full matrix in the Fixtures tab.")
    if agg.empty:
        ui.empty_state("NO_DATA", "No upcoming fixtures in the artifact.")
    else:
        easy, hard = fv.fixture_swing(agg, 5)
        runs = fv.team_runs(agg)
        for col, frame, title in zip(st.columns(2), (easy, hard), ("EASIEST RUNS", "HARDEST RUNS")):
            with col.container(border=True):
                ui.eyebrow(title)
                st.markdown("  \n".join(f"**{r.team}** &nbsp; avg FDR {r.avg_fdr:.1f} &nbsp; " +
                                        fv.run_chips(runs.get(r.team, [])) for r in frame.itertuples()))

# ══ PLAYERS ════════════════════════════════════════════════════════════════════════════════════
with t_play:
    ui.section_header("Recommendations", "One pick per question. Best captain = the published BALANCED "
                      "captaincy pick; differential = owned by under 10 %.")
    recs = fv.recommendations(proj, summ)
    rec_items = [("best_overall", "BEST OVERALL"), ("best_value", "BEST VALUE"),
                 ("best_differential", "BEST DIFFERENTIAL"), ("best_captain", "BEST CAPTAIN"),
                 ("best_defender", "BEST DEFENDER")]
    for i in range(0, len(rec_items), 3):
        for col, (k, title) in zip(st.columns(3), rec_items[i:i + 3]):
            r = recs.get(k)
            if r is None:
                ui.empty_state("NO_DATA", f"{title}: no eligible player", container=col)
            elif k == "best_captain":
                fv.captain_card(r, fv.lookup(proj, r), title, container=col)
            else:
                fv.player_card(r, eyebrow_text=title, container=col)

    ui.section_header("Top projected", "Highest final xPts among players not ruled out.")
    pos_pick = st.radio("Position", ["All"] + POS_ORDER, horizontal=True, key="card_pos")
    cards = fv.top_projected(proj, pos_pick, 6)
    cols = st.columns(3)
    for i, (_, r) in enumerate(cards.iterrows()):
        fv.player_card(r, container=cols[i % 3])

    ui.section_header("Player explorer", "Every projected player. Numbers sort numerically.")
    f1, f2 = st.columns([1, 2])
    poss = f1.multiselect("Position", POS_ORDER, default=POS_ORDER)
    q = f2.text_input("Search player", "")
    sort_by = st.radio("Sort by", ["final_xpts", "xpts_5gw", "value_5gw_per_m", "ceiling",
                                   "wowza_residual", "ownership"], horizontal=True)
    with st.expander("Advanced filters"):
        a1, a2, a3 = st.columns(3)
        teams = sorted(proj["team"].dropna().unique())
        tsel = a1.multiselect("Team", teams, default=[])
        pmin, pmax = float(proj["price"].min()), float(proj["price"].max())
        prng = a2.slider("Price £m", pmin, pmax, (pmin, pmax), step=0.1)
        smin = a3.slider("Min P(start)", 0.0, 1.0, 0.0, step=0.05)
    v = proj[proj["position"].isin(poss) & proj["price"].between(*prng) & (proj["p_start"] >= smin)]
    if tsel:
        v = v[v["team"].isin(tsel)]
    if q:
        v = v[v["player_name"].str.contains(q, case=False, na=False)
              | v["web_name"].str.contains(q, case=False, na=False)]
    v = v.sort_values(sort_by, ascending=False)
    show = ["web_name", "team", "position", "price", "final_xpts", "fpl_ep_next", "wowza_residual",
            "p_start", "exp_minutes", "goal_prob", "assist_prob", "cs_prob", "median", "p90",
            "xpts_3gw", "xpts_5gw", "value_5gw_per_m", "ownership", "next_fixture", "risk",
            "confidence"]
    ui.dataframe(v[show], hide_index=True, width="stretch", height=520, column_config={
        "web_name": "Player", "team": "Team", "position": "Pos",
        "price": st.column_config.NumberColumn("£m", format="%.1f"),
        "final_xpts": st.column_config.ProgressColumn("Final xPts", min_value=0,
                                                      max_value=float(max(proj["final_xpts"].max(), 1)),
                                                      format="%.2f"),
        "fpl_ep_next": st.column_config.NumberColumn("FPL ep", format="%.1f"),
        "wowza_residual": st.column_config.NumberColumn("Wowza adj (shadow)", format="%+.2f"),
        "p_start": st.column_config.NumberColumn("P(start)", format="percent"),
        "exp_minutes": st.column_config.NumberColumn("xMin", format="%.0f"),
        "goal_prob": st.column_config.NumberColumn("P(goal)", format="percent"),
        "assist_prob": st.column_config.NumberColumn("P(assist)", format="percent"),
        "cs_prob": st.column_config.NumberColumn("P(CS)", format="percent"),
        "median": st.column_config.NumberColumn("Median", format="%.0f"),
        "p90": st.column_config.NumberColumn("P90", format="%.0f"),
        "xpts_3gw": st.column_config.NumberColumn("3-GW", format="%.1f"),
        "xpts_5gw": st.column_config.NumberColumn("5-GW", format="%.1f"),
        "value_5gw_per_m": st.column_config.NumberColumn("5-GW/£m", format="%.2f"),
        "ownership": st.column_config.NumberColumn("Own %", format="%.1f"),
        "next_fixture": "Next", "risk": "Risk", "confidence": "Confidence"})
    st.caption(f"{len(v)} of {len(proj)} players.")

    ui.section_header("More player tools", "On the V1 projection board.")
    p_diff, p_lead, p_sp = st.tabs(["Differentials", "Prop leaderboards", "Set-pieces & pens"])
    with p_diff:
        tools.differentials(v1)
    with p_lead:
        tools.prop_leaderboards()
    with p_sp:
        tools.set_pieces()

# ══ TRANSFERS ══════════════════════════════════════════════════════════════════════════════════
with t_tr:
    ui.section_header("Transfer planner", "Pick the player you would sell. Replacements are the same "
                      "position, affordable within your bank, ranked by projected gain over the horizon "
                      "— a delta, never a raw score.")
    owned = proj.sort_values("ownership", ascending=False)
    t1, t2, t3 = st.columns([2, 1, 1])
    out_name = t1.selectbox("Transfer OUT", owned["player_name"].tolist(), index=0)
    bank = t2.number_input("Bank £m", 0.0, 20.0, 0.5, step=0.1)
    horizon = t3.radio("Horizon", ["next GW", "3 GW", "5 GW"], index=2, horizontal=True)
    hcol = fv.HORIZON_COL[horizon]
    o, cand = fv.transfer_candidates(proj, out_name, bank, horizon)
    lo, ro = st.columns([1, 2])
    fv.player_card(o, eyebrow_text="TRANSFER OUT", primary_label=f"{horizon} projection",
                   primary_value=f"{o[hcol]:.2f}", container=lo)
    with ro:
        if cand.empty:
            ui.empty_state("NO_DATA", "No affordable replacement in this position.")
        for i in range(0, len(cand), 2):
            for col, (_, c) in zip(st.columns(2), cand.iloc[i:i + 2].iterrows()):
                fv.transfer_card(o, c, horizon=horizon, gain=float(c["gain"]), container=col)
    st.caption("Expected gain = IN minus OUT on the chosen horizon (the ranking number). Price change, "
               "next/3/5-GW gains and fixture swing (mean FDR of the next five, IN minus OUT; negative = "
               "easier) are the same subtraction on the other published columns.")

    if not trans.empty:
        ui.section_header("Best moves among the most-owned", "Precomputed by the daily job, 5-GW horizon.")
        for col, (_, t) in zip(st.columns(3), trans.iloc[3:6].iterrows()):
            fv.precomputed_move_card(t, container=col)
        with st.expander("All precomputed moves (table)"):
            ui.dataframe(trans.head(40)[["out_web", "out_team", "position", "out_price", "in_web",
                                         "in_team", "in_price", "gain_next", "gain_3gw", "gain_5gw",
                                         "net_gain_5gw_with_hit", "in_p_start", "in_risk"]],
                         hide_index=True, width="stretch", column_config={
                             "out_web": "OUT", "out_team": "OUT team", "position": "Pos",
                             "out_price": ui.money_col("OUT £m"), "in_web": "IN", "in_team": "IN team",
                             "in_price": ui.money_col("IN £m"),
                             "gain_next": ui.num_col("Gain next", fmt="%+.2f"),
                             "gain_3gw": ui.num_col("Gain 3-GW", fmt="%+.2f"),
                             "gain_5gw": ui.num_col("Gain 5-GW", fmt="%+.2f"),
                             "net_gain_5gw_with_hit": ui.num_col("5-GW after hit", fmt="%+.2f"),
                             "in_p_start": ui.pct_col("IN P(start)"), "in_risk": "IN risk"})

    ui.section_header("More transfer tools", "On the V1 projection board.")
    tr_help, tr_alert = st.tabs(["Transfer helper (your team ID)", "Alerts / watchlist"])
    with tr_help:
        tools.transfer_helper(v1)
    with tr_alert:
        tools.alerts(v1)

# ══ CAPTAINCY ══════════════════════════════════════════════════════════════════════════════════
with t_cap:
    view = st.radio("Mean used", ["Headline", "Shadow (FPL + residual)"], horizontal=True,
                    help="Headline = the board's mean. Shadow = ranked with the Wowza residual "
                         "added, so you can see where the not-yet-accepted model disagrees.")
    if view != "Headline":
        st.markdown(ui.status_chip("SHADOW") + " FPL baseline + Wowza residual — not the headline mean.")
    cap = summ.get("captaincy" if view == "Headline" else "captaincy_shadow") or {}
    for col, key, title in zip(st.columns(3), ("SAFE", "BALANCED", "HIGH_UPSIDE"),
                               ("SAFE", "BALANCED", "HIGH UPSIDE")):
        c = cap.get(key)
        if not c:
            ui.empty_state("NO_DATA", f"{title}: no eligible starter", container=col)
            continue
        fv.captain_card(c, fv.lookup(proj, c), title, container=col)

    ui.section_header("Expected points vs upside", "Startable pool (P(start) ≥ 60 %, not doubtful), "
                      "top 40.")
    pool = proj[(proj["p_start"] >= 0.6) & ~proj["risk"].isin(["Unavailable", "Doubtful"])]
    pool = pool.nlargest(40, "final_xpts" if view == "Headline" else "v2_candidate_xpts").copy()
    pool["EV"] = pool["final_xpts"] if view == "Headline" else pool["v2_candidate_xpts"]
    pool["P(15+)"] = pool["p15"] if view == "Headline" else pool["cand_p15"]
    # only the columns the chart encodes are shipped (was every projection column, ~70)
    pool = pool[["web_name", "team", "position", "EV", "P(15+)", "p_start", "p_blank", "next_fixture"]]
    top10 = pool.nlargest(10, "EV").index                  # the ten labelled dots, as before
    pool = pool.assign(lab=pool["web_name"].where(pool.index.isin(top10), ""))
    st.vega_lite_chart(pool, fv.upside_scatter_spec(), width="stretch")
    st.caption("Right = higher expected score, up = bigger haul chance, bigger dot = more likely to "
               "start. Upside probabilities come from the minutes mixture; on the evaluated "
               "gameweeks P(15+) ran ~2–4× too high around the FPL mean (see Model lab), so read it as "
               "a ranking, not a literal probability.")

# ══ FIXTURES ═══════════════════════════════════════════════════════════════════════════════════
with t_fix:
    ui.section_header("Fixture matrix", "Green = easy, red = hard in every view; the number is printed "
                      "in each cell. Sorted easiest run first.")
    if agg.empty:
        ui.empty_state("NO_DATA", "No upcoming fixtures in the artifact.")
    else:
        lens = st.radio("Colour by", list(fv.LENS), horizontal=True, key="fx_lens",
                        help=" · ".join(f"{k}: {v}" for k, v in fv.LENS_HELP.items()))
        st.vega_lite_chart(spec=fv.fixture_heatmap_spec(agg, lens), width="stretch")
        st.caption(f"{fv.LENS_HELP[lens]}. Attack/defence ease are Wowza's 1–5 quintiles from a "
                   "point-in-time team model (player xG + goals, shrunk toward the league mean, "
                   "home edge); FDR is FPL's official rating. An opponent can be easy to score "
                   "against and dangerous at the same time — that is why there are two.")
    ui.section_header("Chip advisor")
    tools.chip_advisor(v1)

# ══ SQUAD ══════════════════════════════════════════════════════════════════════════════════════
with t_sq:
    s_best, s_line, s_league, s_club = st.tabs(["Best legal squad", "My lineup · your squad",
                                                "Mini-league", "Club squads"])
    with s_best:
        st.caption("A legal 15 (£100m, max 3 per club, exactly 2/5/5/3), solved by integer "
                   "programming on the chosen horizon. Computed on demand from the small projection "
                   "table — no model is loaded.")
        hz = st.radio("Optimise", ["final_xpts", "xpts_3gw", "xpts_5gw"], horizontal=True, key="opt_h")
        if st.button("Solve squad"):
            from player_model.fantasy_optimizer import optimal_squad
            df_o = proj.assign(availability=proj["risk"].map(
                {"Unavailable": "unavailable"}).fillna("available"))
            sq = optimal_squad(df_o, points_col=hz)
            if not sq or not sq.get("feasible"):
                st.warning(f"No feasible squad ({(sq or {}).get('reason', 'no data')}).")
            else:
                capn = sq.get("captain")
                cap_name = capn.get("web_name") if capn is not None else "—"
                st.success(f"Formation {sq.get('formation')} · cost £{sq.get('total_cost', 0):.1f}m · "
                           f"XI points {sq.get('xi_points', 0):.1f} · captain {cap_name}")
                for lab in ("xi", "bench"):
                    part = sq.get(lab)
                    if isinstance(part, pd.DataFrame) and not part.empty:
                        st.markdown(f"**{'Starting XI' if lab == 'xi' else 'Bench'}**")
                        ui.dataframe(part[[c for c in ("web_name", "team", "position", "price", hz,
                                                       "p_start", "next_fixture") if c in part.columns]],
                                     hide_index=True, width="stretch", column_config={
                                         "web_name": "Player", "price": ui.money_col("£m"),
                                         hz: ui.num_col(hz, fmt="%.2f"), "p_start": ui.pct_col("P(start)")})
    with s_line:
        tools.lineup(v1)
    with s_league:
        tools.mini_league(v1)
    with s_club:
        tools.club_squads()

# ══ MODEL LAB ══════════════════════════════════════════════════════════════════════════════════
with t_lab:
    st.markdown(ui.status_chip("RESEARCH") + " How the forecast is built and how it has scored. "
                "Fantasy V2 = **FPL baseline + Wowza residual** (shadow). Charts first, tables second.")
    m_bench, m_sb, m_cal, m_dec = st.tabs(["Benchmark", "Gameweek scoreboard", "Calibration & method",
                                           "Decomposition"])

    # ── FPL vs Wowza benchmark ──
    with m_bench:
        ui.section_header("FPL vs Wowza benchmark", "Wowza vs FPL ep_next. Walk-forward: each gameweek "
                          "is predicted only by models fitted on earlier gameweeks.")
        bc1, bc2 = st.columns([1, 1])
        pop = bc1.radio("Population", ["ACTIVE", "LEDGER"], horizontal=True,
                        help="ACTIVE = every player with PL minutes or ep_next ≥ 0.5. LEDGER = the players "
                             "the old Wowza projection covered (the published head-to-head).")
        fin = bc2.toggle("Completed gameweeks only", value=False,
                         help="Excludes the provisional (partly played) gameweek.")
        bt = fdata.v2_benchmark_table(bench, pop, fin)
        if bt.empty:
            st.warning("No out-of-sample gameweeks scored yet.")
        else:
            sc = ((bench.get("scores") or {}).get(pop) or {}).get("by_gw") or []
            long = []
            for r in sc:
                if fin and r.get("status") != "FINAL":
                    continue
                for arm, lab in (("FPL", "FPL ep_next"), ("LINEAR", "V2 residual"), ("WOWZA", "Old Wowza")):
                    m = r.get(arm) or {}
                    if "mae" in m:
                        long.append({"gw": f"GW{r['gw']}{'*' if r['status'] != 'FINAL' else ''}",
                                     "series": lab, "MAE": m["mae"]})
            if long:
                ch = alt.Chart(pd.DataFrame(long)).mark_bar().encode(
                    x=alt.X("gw:N", title=None), xOffset="series:N",
                    y=alt.Y("MAE:Q", title="MAE (points)"),
                    color=alt.Color("series:N", legend=alt.Legend(orient="top", title=None)),
                    tooltip=["gw", "series", alt.Tooltip("MAE:Q", format=".3f")]).properties(height=240)
                st.altair_chart(ch, width="stretch")
                st.caption("MAE per gameweek, lower is better. * provisional gameweek (some fixtures were "
                           "unplayed at the last snapshot; only finished teams are scored).")
            best = bt.loc[bt["MAE"].idxmin(), "model"]
            fpl_row = bt[bt["arm"] == "FPL"].iloc[0]
            ui.metric_grid([
                {"label": "Lowest MAE", "value": str(best)},
                {"label": "FPL ep_next MAE", "value": f"{fpl_row['MAE']:.3f}"},
                {"label": "Gameweeks scored", "value": str(bt["GWs"].iloc[0]) if len(bt) else "—"},
            ], per_row=3)
            ui.dataframe(bt.drop(columns=["arm"]), hide_index=True, width="stretch",
                         column_config={"MAE": st.column_config.NumberColumn(format="%.3f"),
                                        "RMSE": st.column_config.NumberColumn(format="%.3f"),
                                        "Spearman": st.column_config.NumberColumn(format="%.3f"),
                                        "bias": st.column_config.NumberColumn(format="%+.3f")})
            st.caption(f"Lowest MAE here: **{best}**. FPL ep_next MAE {fpl_row['MAE']:.3f}. Walk-forward: "
                       "each gameweek is predicted only by models fitted on earlier gameweeks.")
        pub = bench.get("published_ledger") or {}
        if pub.get("available"):
            st.caption(f"Published ledger recomputed (GWs {pub.get('gameweeks')}, n={pub['FPL'].get('n')}): "
                       f"FPL MAE {pub['FPL'].get('mae')} / ρ {pub['FPL'].get('spearman')} / bias "
                       f"{pub['FPL'].get('bias'):+} vs old Wowza MAE {pub['WOWZA'].get('mae')} / ρ "
                       f"{pub['WOWZA'].get('spearman')} / bias {pub['WOWZA'].get('bias'):+}. FPL won every "
                       f"gameweek: {pub.get('fpl_won_every_gw')}.")

    # ── Gameweek-by-gameweek scoreboard (precomputed, ~20-60 rows) ──
    with m_sb:
        sb = fdata.v2_scoreboard() if hasattr(fdata, "v2_scoreboard") else pd.DataFrame()
        ui.section_header("Gameweek-by-gameweek scoreboard", "MAE/RMSE/Spearman/bias, top-k and captain "
                          "hits, and minutes calibration per gameweek and model.")
        if sb.empty:
            ui.empty_state("NO_DATA", "No scoreboard yet — it is written by the Fantasy Refresh workflow "
                           "once a gameweek has been scored out of sample.")
        else:
            ARM_LABEL = {"FPL": "FPL baseline", "WOWZA": "Old Wowza", "LINEAR": "LINEAR (V2)",
                         "RIDGE": "RIDGE", "HGB": "HGB"}
            sbview = st.radio("View", ["Accuracy", "Top-10/15 & captain", "Minutes calibration"],
                              horizontal=True, key="sb_view")
            s = sb.copy()
            s["GW"] = "GW" + s["gw"].astype(str) + s["status"].ne("FINAL").map({True: "*", False: ""})
            s["model"] = s["arm"].map(ARM_LABEL).fillna(s["arm"])
            fin_sb = s[s["status"] == "FINAL"]
            lin_, fpl_ = fin_sb[fin_sb["arm"] == "LINEAR"], fin_sb[fin_sb["arm"] == "FPL"]
            if len(lin_) and len(fpl_):
                won = int((lin_.set_index("gw")["mae"] < fpl_.set_index("gw")["mae"]).sum())
                ui.metric_grid([{"label": "Residual beat FPL (final GWs, MAE)",
                                 "value": f"{won} of {len(lin_)}"}], per_row=3)
            rel = fdata.v2_scoreboard_reliability() if hasattr(fdata, "v2_scoreboard_reliability") \
                else pd.DataFrame()
            if sbview == "Minutes calibration" and not rel.empty:
                gw_opts = ["All scored GWs"] + [f"GW{g}" for g in sorted(rel["gw"].unique())]
                pick = st.selectbox("Reliability of P(60+ minutes)", gw_opts, key="sb_rel_gw")
                gsel = None if pick == gw_opts[0] else [int(pick[2:])]
                pr = fdata.v2_pooled_reliability(rel, gsel)
                if not pr.empty:
                    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
                        strokeDash=[4, 4], color=ui.token("BORDER_STRONG")).encode(x="x:Q", y="y:Q")
                    pts = alt.Chart(pr).mark_line(point=True).encode(
                        x=alt.X("pred_mean:Q", title="Predicted P(60+)", scale=alt.Scale(domain=[0, 1])),
                        y=alt.Y("obs_rate:Q", title="Observed 60+ rate", scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color("source:N", legend=alt.Legend(orient="top", title=None)),
                        tooltip=["source", "lo", "hi", "n", alt.Tooltip("pred_mean:Q", format=".3f"),
                                 alt.Tooltip("obs_rate:Q", format=".3f")]).properties(height=260)
                    st.altair_chart(diag + pts, width="stretch")
                    st.caption("Fixed bins 0-0.1-0.3-0.5-0.7-0.9-1. On the diagonal = calibrated. "
                               "Old Wowza p_start is a P(start) number, so it is read here as P(60+).")
            base = ["GW", "status", "model", "n"]
            if sbview == "Accuracy":
                cols_ = base + ["mae", "rmse", "spearman", "bias", "fpl_mae_same_rows", "mae_minus_fpl",
                                "train_gws"]
            elif sbview == "Top-10/15 & captain":
                cols_ = base + ["top10_overlap", "top10_actual_mean", "top10_best_mean",
                                "top15_overlap", "top15_actual_mean", "top15_best_mean", "captain",
                                "captain_actual", "best_captain", "best_captain_actual",
                                "captain_shortfall"]
            else:
                cols_ = base + ["minutes_source", "pred_p0", "obs_p0", "pred_p1_59", "obs_p1_59",
                                "pred_p60", "obs_p60", "p60_brier", "p60_ece"]
            ui.dataframe(s[[c for c in cols_ if c in s.columns]], hide_index=True, width="stretch")
            if len(lin_) and len(fpl_):
                st.caption(f"FPL baseline + Wowza residual (LINEAR) had the lower MAE in **{won} of "
                           f"{len(lin_)}** final gameweeks. Each gameweek is scored only by models "
                           "fitted on earlier FINAL gameweeks (`train_gws`). A handful of gameweeks is "
                           "evidence, not a promotion: the residual stays behind its pre-registered "
                           "gate. Old Wowza is scored only on the players it published, so compare it "
                           "with `fpl_mae_same_rows`, not with the FPL row. Top-k columns: overlap with "
                           "the actual top-k, mean actual points of the predicted top-k, and the best "
                           "possible (`_best_mean`). Captain points are single (not doubled). "
                           "LINEAR/RIDGE/HGB share one minutes forecast (the residual moves only the "
                           "mean). * = provisional gameweek.")

    # ── Method & calibration ──
    with m_cal:
        decd = bench.get("decision") or {}
        dist = (bench.get("distribution") or {}).get("mean_fpl") or {}
        if dist:
            rel6 = pd.DataFrame(dist.get("reliability_p6") or [])
            if not rel6.empty:
                rc = alt.Chart(rel6).mark_line(point=True).encode(
                    x=alt.X("pred:Q", title="predicted P(6+)", axis=alt.Axis(format="%")),
                    y=alt.Y("obs:Q", title="observed", axis=alt.Axis(format="%")),
                    tooltip=["pred", "obs", "n"])
                diag = alt.Chart(pd.DataFrame({"x": [0, 0.4], "y": [0, 0.4]})).mark_line(
                    strokeDash=[4, 4], color=ui.token("BORDER_STRONG")).encode(x="x", y="y")
                st.altair_chart((diag + rc).properties(height=240, title="Reliability of P(6+)"),
                                width="stretch")
            ll = dist.get("minutes_log_loss", {})
            bw = dist.get("brier_play_vs_wowza_p_start", {})
            ic = dist.get("interval_coverage", {})
            ui.metric_grid([
                {"label": "Minutes log-loss (mixture)", "value": _num(ll.get("mixture_model"), "{:.3f}"),
                 "delta": f"heuristic {_num(ll.get('heuristic_chance_x_rate'), '{:.3f}')}", "delta_color": "off"},
                {"label": "Brier P(plays) vs old Wowza p_start", "value": _num(bw.get("mixture_model"), "{:.3f}"),
                 "delta": f"old {_num(bw.get('wowza_p_start'), '{:.3f}')}", "delta_color": "off"},
                {"label": "Actual inside P25–P75", "value": _num(ic.get("p25_p75_inclusive"), "{:.0%}"),
                 "delta": f"above P90 {_num(ic.get('above_p90'), '{:.0%}')}", "delta_color": "off"},
            ], per_row=3)
        st.markdown(f"**Decision.** Replace the old standalone projection with the FPL-anchored V2: "
                    f"**{decd.get('replace_old_wowza_projection_with_fpl_anchored_v2', '—')}**. "
                    f"Apply the residual to the mean: **{decd.get('apply_residual_to_mean', '—')}**.")
        for why in (decd.get("detail") or {}).get("reasons", []):
            st.caption(f"• {why}")
        od = pd.DataFrame(bench.get("fpl_overdispersion") or [])
        if not od.empty:
            st.markdown("**Where FPL ep_next goes wrong** (out-of-sample, ACTIVE population)")
            ui.dataframe(od.rename(columns={"band": "ep_next band", "ep_next": "mean ep_next",
                                            "actual": "mean actual", "v2": "mean V2"}),
                         hide_index=True, width="stretch")
        imp = (bench.get("importance") or {}).get("LINEAR") or []
        if imp:
            st.markdown(f"**What the residual uses** — permutation importance on GW"
                        f"{(bench.get('importance') or {}).get('permutation_test_gw')} (RMSE increase when shuffled)")
            ui.dataframe(pd.DataFrame(imp), hide_index=True, width="stretch")
        wb = bench.get("wowza_bias") or {}
        if wb:
            st.markdown(f"**Old Wowza bias {wb.get('bias'):+.2f}** (FPL {wb.get('fpl_bias'):+.2f}, n={wb.get('n')}) "
                        "by actual minutes band:")
            ui.dataframe(pd.DataFrame(wb.get("by", {}).get("minutes_band", [])), hide_index=True,
                         width="stretch")
        st.caption("Full write-up: docs/FANTASY_V2_MODEL_AND_PRODUCT_REPORT.md")

    # ── Projection decomposition ──
    with m_dec:
        st.caption("How a projection is built: **FPL baseline + Wowza adjustments = candidate**. "
                   "Adjustments are measured against the average player in the training gameweeks, "
                   "so they are exact and add up.")
        names = proj.sort_values("final_xpts", ascending=False)["player_name"].tolist()
        who = st.selectbox("Player", names, index=0, key="dec_player")
        r = proj[proj["player_name"] == who].iloc[0]
        parts = [("FPL ep_next", float(r["fpl_ep_next"]))]
        labels = {"adj_calibration": "FPL calibration", "adj_availability": "Availability",
                  "adj_minutes": "Minutes", "adj_attacking_role": "Attacking role",
                  "adj_fixture": "Fixture", "adj_wowza_projection": "Old Wowza signal",
                  "adj_base": "Base"}
        for c, lab in labels.items():
            if c in r and pd.notna(r[c]) and abs(float(r[c])) > 1e-4:
                parts.append((lab, float(r[c])))
        rows, run = [], 0.0
        for k, (lab, val) in enumerate(parts):
            start = 0.0 if k == 0 else run
            end = val if k == 0 else run + val
            rows.append({"step": f"{k}. {lab}", "start": start, "end": end, "value": val,
                         "kind": "baseline" if k == 0 else ("up" if val >= 0 else "down")})
            run = end
        rows.append({"step": f"{len(parts)}. Candidate (FPL + residual)", "start": 0.0, "end": run,
                     "value": run, "kind": "total"})
        wf = pd.DataFrame(rows)
        ch = alt.Chart(wf).mark_bar().encode(
            y=alt.Y("step:N", sort=None, title=None), x=alt.X("start:Q", title="points"), x2="end:Q",
            color=alt.Color("kind:N", scale=alt.Scale(
                domain=["baseline", "up", "down", "total"],
                range=[ui.token("NO_DATA"), ui.token("POSITIVE"), ui.token("NEGATIVE"), ui.token("PRIMARY")]),
                legend=alt.Legend(orient="top", title=None)),
            tooltip=["step", alt.Tooltip("value:Q", format="+.2f")]).properties(height=300)
        st.altair_chart(ch, width="stretch")
        st.caption(f"Headline (shown on the board): **{r['final_xpts']:.2f}** — "
                   f"{'residual applied' if bool(r['residual_applied']) else 'FPL ep_next; the residual is in shadow'}. "
                   "‘FPL calibration’ is large for players with a big ep_next: across every evaluated "
                   "gameweek, ep_next above 6 overshot actual points by roughly 2.7–4.7.")
        ui.metric_grid([
            {"label": "P(0 min)", "value": f"{r['p_0']:.0%}"},
            {"label": "P(1–59)", "value": f"{r['p_1_59']:.0%}"},
            {"label": "P(60+)", "value": f"{r['p_60']:.0%}"},
            {"label": "Expected minutes", "value": f"{r['exp_minutes']:.0f}"},
            {"label": "P(goal)", "value": f"{r['goal_prob']:.0%}"},
            {"label": "P(assist)", "value": f"{r['assist_prob']:.0%}"},
            {"label": "P(clean sheet pts)", "value": _num(r["cs_prob"], "{:.0%}")},
            {"label": "Exp. bonus", "value": _num(r["exp_bonus"])},
        ], per_row=4)
        st.caption(f"Distribution: P25 {r['p25']:.0f} · median {r['median']:.0f} · P75 {r['p75']:.0f} · "
                   f"P90 {r['p90']:.0f} · P(blank ≤2) {r['p_blank']:.0%} · uncertainty {r['uncertainty']} "
                   f"({r['uncertainty_score']:.0f}/100).")

from utils.disclaimer import disclaimer_footer  # noqa: E402
disclaimer_footer()
