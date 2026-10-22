"""
Fantasy (FPL) — Wowza Fantasy V2.

A VIEWER. Everything on this page is precomputed by `python -m player_model.fantasy_v2` inside
fantasy_refresh.yml and read here from small CSV/JSON artifacts with column projection. No model
is trained, no parquet is read, and no file mtime is consulted at render time — the deployment has
~1 GB and has died twice from doing that work here.

THE MEAN FORECAST IS FPL's OWN ep_next unless the residual has passed its acceptance gate. FPL beat
the old standalone Wowza projection in every settled gameweek, and the benchmark panel below says
so on every render. Wowza's job is the residual (shadow until it earns the mean), minutes, ceiling,
role, multi-gameweek planning, captaincy risk and transfers. Fantasy is scored in FPL points,
never units or ROI — it is a prediction contest, not betting.
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

# STALE-MODULE GUARD (2026-10-22). Streamlit Cloud re-runs page files on every deploy but keeps
# modules it already imported, so after the V2 update the long-running process served the OLD
# dashboard_data.fantasy and the page died with AttributeError on v2_projections. Reload once if the
# readers this page needs are missing.
if not all(hasattr(fdata, f) for f in ("v2_projections", "v2_summary", "v2_benchmark", "v2_transfers",
                                       "v2_fixture_matrix", "v2_scoreboard",
                                       "v2_scoreboard_reliability", "v2_pooled_reliability")):
    import importlib  # noqa: E402
    fdata = importlib.reload(fdata)

st.set_page_config(page_title="Fantasy | Wowza", page_icon="⚽", layout="wide")
from dashboard_access import require as _require_tier  # noqa: E402
_require_tier("pro")  # tier guard (dashboard_access.PAGES), 2026-10-18

ui.autorefresh(minutes=10, key="fantasy_refresh")

POS_ORDER = ["GKP", "DEF", "MID", "FWD"]
POS_LABEL = {"GKP": "Goalkeeper", "DEF": "Defender", "MID": "Midfielder", "FWD": "Forward"}
#: official FPL FDR colours, always printed with the number (colour is never the only cue)
FDR_BG = {1: "#257D5A", 2: "#00FF86", 3: "#E7E7E7", 4: "#FF1751", 5: "#80072D"}
FDR_FG = {1: "#FFFFFF", 2: "#0B3D2E", 3: "#333333", 4: "#FFFFFF", 5: "#FFFFFF"}


@st.cache_data(ttl=600, show_spinner=False)
def _load():
    return (fdata.v2_projections(), fdata.v2_summary(), fdata.v2_benchmark(),
            fdata.v2_transfers(), fdata.v2_fixture_matrix())


proj, summ, bench, trans, fxm = _load()

st.title("⚽ Fantasy V2 — FPL analytics")
st.caption("Premier League Fantasy projections anchored on FPL's own expected points, with Wowza's "
           "minutes, role, fixture and uncertainty layer. Predictions in **FPL points** — not "
           "betting tips.")
if st.button("🔄 Reload artifacts"):
    st.cache_data.clear()
    st.rerun()

if proj.empty or not summ:
    st.info("Fantasy V2 artifacts are not built yet. They are produced daily by the Fantasy "
            "Refresh workflow (`python -m player_model.fantasy_v2`).")
    st.stop()


def _num(x, fmt="{:.2f}", none="—"):
    try:
        if x is None or x != x:
            return none
        return fmt.format(x)
    except Exception:                                                # noqa: BLE001
        return none


def _fdr_strip(fixture_run: str, fdr_run: str) -> str:
    """Next-5 fixture chips: opponent + FDR number on the official FDR colour."""
    labs = str(fixture_run or "").split(" | ")
    fdrs = str(fdr_run or "").split(" | ")
    chips = []
    for lab, f in zip(labs, fdrs + [""] * (len(labs) - len(fdrs))):
        try:
            v = int(str(f).split("+")[0])
        except ValueError:
            v = 3
        bg, fg = FDR_BG.get(v, "#E7E7E7"), FDR_FG.get(v, "#333")
        txt = "BLANK" if lab == "BLANK" else f"{lab} · {f}"
        chips.append(f"<span style='display:inline-block;margin:1px 2px;padding:1px 6px;"
                     f"border-radius:4px;font-size:0.72rem;background:{bg};color:{fg}'>{txt}</span>")
    return "".join(chips)


# ══ 1. GAMEWEEK COMMAND CENTER ══════════════════════════════════════════════════════════════
st.subheader("🧭 Gameweek command center")
bs = summ.get("benchmark_status") or {}
ph = summ.get("projection_health") or {}
age_now = fdata.v2_fpl_age_hours()
fresh = ("UNKNOWN" if age_now is None else "CURRENT" if age_now <= 30 else
         "AGING" if age_now <= 54 else "STALE")
dl = pd.to_datetime(summ.get("deadline"), utc=True, errors="coerce")
now = pd.Timestamp.now(tz="UTC")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Gameweek", f"GW{summ.get('gameweek')}" if summ.get("gameweek") else "—",
          help="The gameweek these projections are FOR (FPL's next deadline).")
if pd.notna(dl):
    left = dl - now
    hrs = left.total_seconds() / 3600
    c2.metric("Deadline (UTC)", dl.strftime("%a %d %b %H:%M"),
              f"{hrs:.0f}h to go" if hrs > 0 else "passed", delta_color="off")
else:
    c2.metric("Deadline", "—")
c3.metric("FPL data", f"{age_now:.0f}h old" if age_now is not None else "age unknown", fresh,
          delta_color="off",
          help="Age from the RECORDED fetch time in fpl_cache_meta.json, never from file mtime. "
               "The daily job runs once a day, so up to ~30h is normal.")
c4.metric("Projection health", ph.get("state", "—"),
          f"{ph.get('players', 0)} players · {ph.get('minutes_disagreements', 0)} minutes flags",
          delta_color="off")
dec = bs.get("decision_residual")
if bs.get("residual_applied"):
    st.success(f"**Mean forecast: {bs.get('mean_forecast')}.** The residual passed its "
               "out-of-sample acceptance gate and is applied.", icon="✅")
else:
    st.info(f"**Mean forecast: {bs.get('mean_forecast', 'FPL ep_next')}.** Wowza's residual "
            f"acceptance gate: **{dec or '—'}** (mode `{bs.get('residual_mode', 'shadow')}`) — the "
            "residual is shown everywhere as a *shadow* adjustment but not added to the headline "
            "number until it beats FPL over enough completed gameweeks.", icon="ℹ️")
if fresh in ("STALE", "UNKNOWN"):
    st.error("FPL snapshot is stale or undatable — injuries, prices and the squad list may be out "
             "of date.", icon="🚨")
st.caption(f"Archive: GWs {ph.get('archived_gameweeks')} · final {ph.get('final_gameweeks')} · "
           f"provisional {ph.get('provisional_gameweeks')} · built {summ.get('generated_at')}")

# ══ 2. MODEL BENCHMARK (always visible) ═════════════════════════════════════════════════════
st.subheader("📏 Model benchmark — Wowza vs FPL ep_next")
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
    st.dataframe(bt.drop(columns=["arm"]), hide_index=True, width="stretch",
                 column_config={"MAE": st.column_config.NumberColumn(format="%.3f"),
                                "RMSE": st.column_config.NumberColumn(format="%.3f"),
                                "Spearman": st.column_config.NumberColumn(format="%.3f"),
                                "bias": st.column_config.NumberColumn(format="%+.3f")})
    best = bt.loc[bt["MAE"].idxmin(), "model"]
    fpl_row = bt[bt["arm"] == "FPL"].iloc[0]
    st.caption(f"Lowest MAE here: **{best}**. FPL ep_next MAE {fpl_row['MAE']:.3f}. Walk-forward: "
               "each gameweek is predicted only by models fitted on earlier gameweeks.")
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
            tooltip=["gw", "series", alt.Tooltip("MAE:Q", format=".3f")]).properties(height=220)
        st.altair_chart(ch, width="stretch")
        st.caption("* provisional gameweek (some fixtures were unplayed at the last snapshot; "
                   "only finished teams are scored).")
pub = bench.get("published_ledger") or {}
if pub.get("available"):
    st.caption(f"Published ledger recomputed (GWs {pub.get('gameweeks')}, n={pub['FPL'].get('n')}): "
               f"FPL MAE {pub['FPL'].get('mae')} / ρ {pub['FPL'].get('spearman')} / bias "
               f"{pub['FPL'].get('bias'):+} vs old Wowza MAE {pub['WOWZA'].get('mae')} / ρ "
               f"{pub['WOWZA'].get('spearman')} / bias {pub['WOWZA'].get('bias'):+}. FPL won every "
               f"gameweek: {pub.get('fpl_won_every_gw')}.")

# Gameweek-by-gameweek scoreboard: precomputed (output/fantasy_v2_scoreboard*.csv), ~20-60 rows.
sb = fdata.v2_scoreboard() if hasattr(fdata, "v2_scoreboard") else pd.DataFrame()
with st.expander("🗓️ Gameweek-by-gameweek scoreboard", expanded=True):
    if sb.empty:
        st.info("No scoreboard yet — it is written by the Fantasy Refresh workflow once a gameweek "
                "has been scored out of sample.")
    else:
        ARM_LABEL = {"FPL": "FPL baseline", "WOWZA": "Old Wowza", "LINEAR": "LINEAR (V2)",
                     "RIDGE": "RIDGE", "HGB": "HGB"}
        view = st.radio("View", ["Accuracy", "Top-10/15 & captain", "Minutes calibration"],
                        horizontal=True, key="sb_view")
        s = sb.copy()
        s["GW"] = "GW" + s["gw"].astype(str) + s["status"].ne("FINAL").map({True: "*", False: ""})
        s["model"] = s["arm"].map(ARM_LABEL).fillna(s["arm"])
        base = ["GW", "status", "model", "n"]
        if view == "Accuracy":
            cols_ = base + ["mae", "rmse", "spearman", "bias", "fpl_mae_same_rows", "mae_minus_fpl",
                            "train_gws"]
        elif view == "Top-10/15 & captain":
            cols_ = base + ["top10_overlap", "top10_actual_mean", "top10_best_mean",
                            "top15_overlap", "top15_actual_mean", "top15_best_mean", "captain",
                            "captain_actual", "best_captain", "best_captain_actual",
                            "captain_shortfall"]
        else:
            cols_ = base + ["minutes_source", "pred_p0", "obs_p0", "pred_p1_59", "obs_p1_59",
                            "pred_p60", "obs_p60", "p60_brier", "p60_ece"]
        st.dataframe(s[[c for c in cols_ if c in s.columns]], hide_index=True, width="stretch")
        fin_sb = s[s["status"] == "FINAL"]
        lin_, fpl_ = fin_sb[fin_sb["arm"] == "LINEAR"], fin_sb[fin_sb["arm"] == "FPL"]
        if len(lin_) and len(fpl_):
            won = int((lin_.set_index("gw")["mae"] < fpl_.set_index("gw")["mae"]).sum())
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
        rel = fdata.v2_scoreboard_reliability() if hasattr(fdata, "v2_scoreboard_reliability") \
            else pd.DataFrame()
        if view == "Minutes calibration" and not rel.empty:
            gw_opts = ["All scored GWs"] + [f"GW{g}" for g in sorted(rel["gw"].unique())]
            pick = st.selectbox("Reliability of P(60+ minutes)", gw_opts, key="sb_rel_gw")
            gsel = None if pick == gw_opts[0] else [int(pick[2:])]
            pr = fdata.v2_pooled_reliability(rel, gsel)
            if not pr.empty:
                diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
                    strokeDash=[4, 4], color="#999").encode(x="x:Q", y="y:Q")
                pts = alt.Chart(pr).mark_line(point=True).encode(
                    x=alt.X("pred_mean:Q", title="Predicted P(60+)", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("obs_rate:Q", title="Observed 60+ rate", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("source:N", legend=alt.Legend(orient="top", title=None)),
                    tooltip=["source", "lo", "hi", "n", alt.Tooltip("pred_mean:Q", format=".3f"),
                             alt.Tooltip("obs_rate:Q", format=".3f")]).properties(height=260)
                st.altair_chart(diag + pts, width="stretch")
                st.caption("Fixed bins 0-0.1-0.3-0.5-0.7-0.9-1. On the diagonal = calibrated. "
                           "Old Wowza p_start is a P(start) number, so it is read here as P(60+).")

# ══ 3. TOP RECOMMENDATIONS (cards) ══════════════════════════════════════════════════════════
st.subheader("⭐ Top recommendations")
pos_pick = st.radio("Position", ["All"] + POS_ORDER, horizontal=True, key="card_pos")
avail = proj[~proj["risk"].isin(["Unavailable"])]
cards = avail if pos_pick == "All" else avail[avail["position"] == pos_pick]
cards = cards.sort_values("final_xpts", ascending=False).head(6)
cols = st.columns(3)
for i, r in enumerate(cards.itertuples()):
    with cols[i % 3].container(border=True):
        st.markdown(f"**{r.web_name}** · {r.team}")
        st.caption(f"{POS_LABEL.get(r.position, r.position)} · £{r.price:.1f}m · "
                   f"owned {_num(r.ownership, '{:.1f}%')} · confidence **{r.confidence}**")
        a, b, c = st.columns(3)
        a.metric("Final xPts", _num(r.final_xpts))
        b.metric("FPL baseline", _num(r.fpl_ep_next, "{:.1f}"))
        c.metric("Wowza adj.", _num(r.wowza_residual, "{:+.2f}"),
                 "applied" if bool(r.residual_applied) else "shadow", delta_color="off")
        st.caption(f"P(start) **{r.p_start:.0%}** · ~{r.exp_minutes:.0f} min · ceiling (P90) "
                   f"**{r.ceiling:.0f}** · P(10+) {r.p10:.0%} · next **{r.next_fixture}**")
        st.markdown(_fdr_strip(r.fixture_run, r.fdr_run), unsafe_allow_html=True)
        if r.risk != "Low":
            st.caption(f"⚠️ Risk: {r.risk}" + (f" — {r.news}" if isinstance(r.news, str) and r.news else ""))

tabs = st.tabs(["🧮 Decomposition", "🔎 Player explorer", "🔄 Transfer planner",
                "🏆 Captaincy lab", "📅 Fixture matrix", "⭐ Best legal squad", "🔬 Method & calibration"])

# ══ 4. PROJECTION DECOMPOSITION ═════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("How a projection is built: **FPL baseline + Wowza adjustments = candidate**. "
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
    for k, (lab, v) in enumerate(parts):
        start = 0.0 if k == 0 else run
        end = v if k == 0 else run + v
        rows.append({"step": f"{k}. {lab}", "start": start, "end": end, "value": v,
                     "kind": "baseline" if k == 0 else ("up" if v >= 0 else "down")})
        run = end
    rows.append({"step": f"{len(parts)}. Candidate (FPL + residual)", "start": 0.0, "end": run,
                 "value": run, "kind": "total"})
    wf = pd.DataFrame(rows)
    ch = alt.Chart(wf).mark_bar().encode(
        y=alt.Y("step:N", sort=None, title=None), x=alt.X("start:Q", title="points"), x2="end:Q",
        color=alt.Color("kind:N", scale=alt.Scale(domain=["baseline", "up", "down", "total"],
                                                  range=["#6B7280", "#0894AF", "#C2410C", "#1E6FD9"]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["step", alt.Tooltip("value:Q", format="+.2f")]).properties(height=300)
    st.altair_chart(ch, width="stretch")
    st.caption(f"Headline (shown on the board): **{r['final_xpts']:.2f}** — "
               f"{'residual applied' if bool(r['residual_applied']) else 'FPL ep_next; the residual is in shadow'}. "
               "‘FPL calibration’ is large for players with a big ep_next: across every evaluated "
               "gameweek, ep_next above 6 overshot actual points by roughly 2.7–4.7.")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("P(0 min)", f"{r['p_0']:.0%}")
    d2.metric("P(1–59)", f"{r['p_1_59']:.0%}")
    d3.metric("P(60+)", f"{r['p_60']:.0%}")
    d4.metric("Expected minutes", f"{r['exp_minutes']:.0f}")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("P(goal)", f"{r['goal_prob']:.0%}")
    e2.metric("P(assist)", f"{r['assist_prob']:.0%}")
    e3.metric("P(clean sheet pts)", _num(r["cs_prob"], "{:.0%}"))
    e4.metric("Exp. bonus", _num(r["exp_bonus"]))
    st.caption(f"Distribution: P25 {r['p25']:.0f} · median {r['median']:.0f} · P75 {r['p75']:.0f} · "
               f"P90 {r['p90']:.0f} · P(blank ≤2) {r['p_blank']:.0%} · uncertainty {r['uncertainty']} "
               f"({r['uncertainty_score']:.0f}/100).")

# ══ 5. PLAYER EXPLORER ══════════════════════════════════════════════════════════════════════
with tabs[1]:
    f1, f2, f3, f4 = st.columns(4)
    poss = f1.multiselect("Position", POS_ORDER, default=POS_ORDER)
    teams = sorted(proj["team"].dropna().unique())
    tsel = f2.multiselect("Team", teams, default=[])
    pmin, pmax = float(proj["price"].min()), float(proj["price"].max())
    prng = f3.slider("Price £m", pmin, pmax, (pmin, pmax), step=0.1)
    smin = f4.slider("Min P(start)", 0.0, 1.0, 0.0, step=0.05)
    q = st.text_input("Search player", "")
    sort_by = st.radio("Sort by", ["final_xpts", "xpts_5gw", "value_5gw_per_m", "ceiling",
                                   "wowza_residual", "ownership"], horizontal=True)
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
    st.dataframe(v[show], hide_index=True, width="stretch", height=520, column_config={
        "web_name": "Player", "price": st.column_config.NumberColumn("£m", format="%.1f"),
        "final_xpts": st.column_config.ProgressColumn("Final xPts", min_value=0,
                                                      max_value=float(max(proj["final_xpts"].max(), 1)),
                                                      format="%.2f"),
        "fpl_ep_next": st.column_config.NumberColumn("FPL ep", format="%.1f"),
        "wowza_residual": st.column_config.NumberColumn("Wowza adj", format="%+.2f"),
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
        "next_fixture": "Next"})
    st.caption(f"{len(v)} of {len(proj)} players.")

# ══ 6. TRANSFER PLANNER ═════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("Pick the player you would sell. Replacements are the same position, affordable "
                "within your bank, ranked by **projected gain** over the horizon — a delta, never "
                "a raw score.")
    owned = proj.sort_values("ownership", ascending=False)
    t1, t2, t3 = st.columns([2, 1, 1])
    out_name = t1.selectbox("Transfer OUT", owned["player_name"].tolist(), index=0)
    bank = t2.number_input("Bank £m", 0.0, 20.0, 0.5, step=0.1)
    horizon = t3.radio("Horizon", ["next GW", "3 GW", "5 GW"], index=2, horizontal=True)
    hcol = {"next GW": "final_xpts", "3 GW": "xpts_3gw", "5 GW": "xpts_5gw"}[horizon]
    o = proj[proj["player_name"] == out_name].iloc[0]
    cand = proj[(proj["position"] == o["position"]) & (proj["element"] != o["element"])
                & (proj["price"] <= o["price"] + bank) & (proj["risk"] != "Unavailable")].copy()
    cand["gain"] = cand[hcol] - o[hcol]
    cand = cand.sort_values("gain", ascending=False).head(5)
    lo, ro = st.columns([1, 2])
    with lo.container(border=True):
        st.markdown(f"**OUT · {o['web_name']}** ({o['team']}, £{o['price']:.1f}m)")
        st.metric(f"{horizon} projection", f"{o[hcol]:.2f}")
        st.caption(f"FPL ep {o['fpl_ep_next']:.1f} · residual {o['wowza_residual']:+.2f} · "
                   f"P(start) {o['p_start']:.0%} · risk {o['risk']}")
        st.markdown(_fdr_strip(o["fixture_run"], o["fdr_run"]), unsafe_allow_html=True)
    with ro:
        if cand.empty:
            st.info("No affordable replacement in this position.")
        for c in cand.itertuples():
            with st.container(border=True):
                a, b = st.columns([3, 1])
                a.markdown(f"**IN · {c.web_name}** ({c.team}, £{c.price:.1f}m) — "
                           f"{horizon}: {getattr(c, hcol):.2f}")
                a.caption(f"FPL ep {c.fpl_ep_next:.1f} · residual {c.wowza_residual:+.2f} · final "
                          f"{c.final_xpts:.2f} · 3-GW {c.xpts_3gw:.1f} · 5-GW {c.xpts_5gw:.1f} · "
                          f"P(start) {c.p_start:.0%} · ceiling {c.ceiling:.0f} · "
                          f"{c.value_5gw_per_m:.2f}/£m · owned {c.ownership:.1f}% · risk {c.risk}")
                a.markdown(_fdr_strip(c.fixture_run, c.fdr_run), unsafe_allow_html=True)
                b.metric("Expected gain", f"{c.gain:+.2f}",
                         f"{c.gain - 4:+.2f} after a -4 hit", delta_color="off")
    if not trans.empty:
        with st.expander("Best moves among the most-owned players (precomputed, 5-GW)"):
            st.dataframe(trans.head(40)[["out_web", "out_team", "position", "out_price", "in_web",
                                         "in_team", "in_price", "gain_next", "gain_3gw", "gain_5gw",
                                         "net_gain_5gw_with_hit", "in_p_start", "in_risk"]],
                         hide_index=True, width="stretch")

# ══ 7. CAPTAINCY LAB ════════════════════════════════════════════════════════════════════════
with tabs[3]:
    view = st.radio("Mean used", ["Headline", "Shadow (FPL + residual)"], horizontal=True,
                    help="Headline = the board's mean. Shadow = ranked with the Wowza residual "
                         "added, so you can see where the not-yet-accepted model disagrees.")
    cap = summ.get("captaincy" if view == "Headline" else "captaincy_shadow") or {}
    cc = st.columns(3)
    for col, key, title in zip(cc, ("SAFE", "BALANCED", "HIGH_UPSIDE"),
                               ("🛡️ SAFE", "⚖️ BALANCED", "🚀 HIGH-UPSIDE")):
        c = cap.get(key)
        if not c:
            col.info(f"{title}: no eligible starter")
            continue
        with col.container(border=True):
            st.markdown(f"**{title} · {c['web_name']}** ({c['team']})")
            m1, m2 = st.columns(2)
            m1.metric("Captain EV", f"{c['captain_ev']:.1f}")
            m2.metric("P(start)", f"{c['p_start']:.0%}")
            st.caption(f"P(6+) {c['p6']:.0%} · P(10+) {c['p10']:.0%} · P(15+) {c['p15']:.0%} · "
                       f"~{c['exp_minutes']:.0f} min · ceiling {c['ceiling']:.0f} · "
                       f"vs {c['next_fixture']} · pens {'yes' if c['pen_first'] else 'no'} · "
                       f"set pieces {'yes' if c['set_piece'] else 'no'}")
            st.caption(c.get("reasoning", ""))
            if c.get("note"):
                st.caption(f"ℹ️ {c['note']}")
    pool = proj[(proj["p_start"] >= 0.6) & ~proj["risk"].isin(["Unavailable", "Doubtful"])]
    pool = pool.nlargest(40, "final_xpts" if view == "Headline" else "v2_candidate_xpts").copy()
    pool["EV"] = pool["final_xpts"] if view == "Headline" else pool["v2_candidate_xpts"]
    pool["P(15+)"] = pool["p15"] if view == "Headline" else pool["cand_p15"]
    sc_ = alt.Chart(pool).mark_circle(opacity=0.8).encode(
        x=alt.X("EV:Q", title="expected points (one game)"),
        y=alt.Y("P(15+):Q", title="P(15+ points) — upside", axis=alt.Axis(format="%")),
        size=alt.Size("p_start:Q", title="P(start)", scale=alt.Scale(range=[30, 300])),
        color=alt.Color("position:N", legend=alt.Legend(orient="top", title=None)),
        tooltip=["web_name", "team", alt.Tooltip("EV:Q", format=".2f"),
                 alt.Tooltip("P(15+):Q", format=".1%"), alt.Tooltip("p_start:Q", format=".0%"),
                 alt.Tooltip("p_blank:Q", format=".0%", title="P(blank)"), "next_fixture"]
    ).properties(height=380)
    labels_ = alt.Chart(pool.nlargest(10, "EV")).mark_text(dy=-10, fontSize=10).encode(
        x="EV:Q", y="P(15+):Q", text="web_name:N")
    st.altair_chart(sc_ + labels_, width="stretch")
    st.caption("Right = higher expected score, up = bigger haul chance, bigger dot = more likely to "
               "start. Upside probabilities come from the minutes mixture; on the evaluated "
               "gameweeks P(15+) ran ~2–4× too high around the FPL mean (see Method), so read it as "
               "a ranking, not a literal probability.")

# ══ 8. FIXTURE MATRIX ═══════════════════════════════════════════════════════════════════════
with tabs[4]:
    if fxm.empty:
        st.info("No upcoming fixtures in the artifact.")
    else:
        lens = st.radio("Colour by", ["Attack ease (team goal expectancy)",
                                      "Defence ease (clean-sheet chance)", "Official FDR"],
                        horizontal=True)
        f = fxm.copy()
        f["GW"] = "GW" + f["gw"].astype(str)
        agg = f.groupby(["team", "GW"], as_index=False).agg(
            label=("label", lambda s: " + ".join(s)), attack_ease=("attack_ease", "mean"),
            defence_ease=("defence_ease", "mean"), fdr=("fdr", "mean"),
            lam_for=("lam_for", "sum"), cs_prob=("cs_prob", "mean"))
        key = {"Attack ease (team goal expectancy)": "attack_ease",
               "Defence ease (clean-sheet chance)": "defence_ease", "Official FDR": "fdr"}[lens]
        order = (agg.groupby("team")[key].mean().sort_values(ascending=(key == "fdr")).index.tolist())
        scale = (alt.Scale(domain=[1, 5], range=["#257D5A", "#E7E7E7", "#80072D"]) if key == "fdr"
                 else alt.Scale(domain=[1, 5], range=["#80072D", "#E7E7E7", "#257D5A"]))
        base = alt.Chart(agg).encode(x=alt.X("GW:N", title=None, sort=sorted(agg["GW"].unique(),
                                                                          key=lambda g: int(g[2:]))),
                                     y=alt.Y("team:N", sort=order, title=None))
        heat = base.mark_rect().encode(
            color=alt.Color(f"{key}:Q", scale=scale, legend=alt.Legend(title=key)),
            tooltip=["team", "GW", "label", alt.Tooltip("lam_for:Q", format=".2f", title="team xG"),
                     alt.Tooltip("cs_prob:Q", format=".0%", title="CS prob"),
                     alt.Tooltip("fdr:Q", format=".0f", title="FDR")])
        txt = base.mark_text(fontSize=10).encode(text="label:N")
        st.altair_chart(heat + txt, width="stretch")
        st.caption("Sorted easiest run first. Attack/defence ease are Wowza's 1–5 quintiles from a "
                   "point-in-time team model (player xG + goals, shrunk toward the league mean, "
                   "home edge); FDR is FPL's official rating. An opponent can be easy to score "
                   "against and dangerous at the same time — that is why there are two.")

# ══ 9. BEST LEGAL SQUAD ═════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("A legal 15 (£100m, max 3 per club, exactly 2/5/5/3), solved by integer "
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
                    st.dataframe(part[[c for c in ("web_name", "team", "position", "price", hz,
                                                   "p_start", "next_fixture") if c in part.columns]],
                                 hide_index=True, width="stretch")

# ══ 10. METHOD & CALIBRATION ════════════════════════════════════════════════════════════════
with tabs[6]:
    decd = bench.get("decision") or {}
    st.markdown(f"**Decision.** Replace the old standalone projection with the FPL-anchored V2: "
                f"**{decd.get('replace_old_wowza_projection_with_fpl_anchored_v2', '—')}**. "
                f"Apply the residual to the mean: **{decd.get('apply_residual_to_mean', '—')}**.")
    for why in (decd.get("detail") or {}).get("reasons", []):
        st.caption(f"• {why}")
    dist = (bench.get("distribution") or {}).get("mean_fpl") or {}
    if dist:
        m1, m2, m3 = st.columns(3)
        ll = dist.get("minutes_log_loss", {})
        m1.metric("Minutes log-loss (mixture)", _num(ll.get("mixture_model"), "{:.3f}"),
                  f"heuristic {_num(ll.get('heuristic_chance_x_rate'), '{:.3f}')}", delta_color="off")
        bw = dist.get("brier_play_vs_wowza_p_start", {})
        m2.metric("Brier P(plays) vs old Wowza p_start", _num(bw.get("mixture_model"), "{:.3f}"),
                  f"old {_num(bw.get('wowza_p_start'), '{:.3f}')}", delta_color="off")
        ic = dist.get("interval_coverage", {})
        m3.metric("Actual inside P25–P75", _num(ic.get("p25_p75_inclusive"), "{:.0%}"),
                  f"above P90 {_num(ic.get('above_p90'), '{:.0%}')}", delta_color="off")
        rel = pd.DataFrame(dist.get("reliability_p6") or [])
        if not rel.empty:
            rc = alt.Chart(rel).mark_line(point=True).encode(
                x=alt.X("pred:Q", title="predicted P(6+)", axis=alt.Axis(format="%")),
                y=alt.Y("obs:Q", title="observed", axis=alt.Axis(format="%")),
                tooltip=["pred", "obs", "n"])
            diag = alt.Chart(pd.DataFrame({"x": [0, 0.4], "y": [0, 0.4]})).mark_line(
                strokeDash=[4, 4], color="#9CA3AF").encode(x="x", y="y")
            st.altair_chart((diag + rc).properties(height=240, title="Reliability of P(6+)"),
                            width="stretch")
    od = pd.DataFrame(bench.get("fpl_overdispersion") or [])
    if not od.empty:
        st.markdown("**Where FPL ep_next goes wrong** (out-of-sample, ACTIVE population)")
        st.dataframe(od.rename(columns={"band": "ep_next band", "ep_next": "mean ep_next",
                                        "actual": "mean actual", "v2": "mean V2"}),
                     hide_index=True, width="stretch")
    imp = (bench.get("importance") or {}).get("LINEAR") or []
    if imp:
        st.markdown(f"**What the residual uses** — permutation importance on GW"
                    f"{(bench.get('importance') or {}).get('permutation_test_gw')} (RMSE increase when shuffled)")
        st.dataframe(pd.DataFrame(imp), hide_index=True, width="stretch")
    wb = bench.get("wowza_bias") or {}
    if wb:
        st.markdown(f"**Old Wowza bias {wb.get('bias'):+.2f}** (FPL {wb.get('fpl_bias'):+.2f}, n={wb.get('n')}) "
                    "by actual minutes band:")
        st.dataframe(pd.DataFrame(wb.get("by", {}).get("minutes_band", [])), hide_index=True,
                     width="stretch")
    st.caption("Full write-up: docs/FANTASY_V2_MODEL_AND_PRODUCT_REPORT.md")

# ── tools from the original page, restored (owner 2026-10-22) ──────────────────────────────────
import dashboard_fantasy_tools  # noqa: E402
dashboard_fantasy_tools.render()

from utils.disclaimer import disclaimer_footer  # noqa: E402
disclaimer_footer()
