"""Fantasy planning tools restored under the V2 page (owner 2026-10-22: "why u removed all of this").

The Fantasy V2 rewrite (6b814744) dropped these tools to make room; nothing had asked for them to go.
They come back here, unchanged in behaviour, reading the V1 board (output/fantasy_tips.csv) that the
untouched V1 pipeline still writes daily. The V2 page owns the forecast, captaincy, transfers-by-gain,
fixture matrix and the legal squad; these are the tools V2 did not replace:

    Differentials · Prop leaderboards · Set-pieces & pens · Transfer helper (by team ID)
    Chip advisor · My lineup (your squad) · Mini-league · Alerts / watchlist · Club squads

Viewer rules kept: the board is read from a 37 KB CSV; prop leaderboards stay opt-in
(FANTASY_DASHBOARD_LIVE_BUILD=1) because scoring them needs the 152 MB player history, which took
the deployment over its memory limit twice.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

import dashboard_ui as ui

BASE_DIR = Path(__file__).resolve().parent
TIPS_FILE = BASE_DIR / "output" / "fantasy_tips.csv"
_ALLOW_LIVE_BUILD = os.getenv("FANTASY_DASHBOARD_LIVE_BUILD", "").lower() in {"1", "true", "yes"}
_POS_ALIAS = {"F": "FWD", "M": "MID", "D": "DEF", "G": "GKP", "GK": "GKP",
              "FWD": "FWD", "MID": "MID", "DEF": "DEF", "GKP": "GKP"}


@st.cache_data(ttl=600, show_spinner=False)
def _board() -> pd.DataFrame:
    try:
        d = pd.read_csv(TIPS_FILE)
    except Exception:                                                   # noqa: BLE001
        return pd.DataFrame()
    if d.empty:
        return d
    if "position" in d.columns:
        d["position"] = d["position"].map(lambda p: _POS_ALIAS.get(str(p), str(p)))
    fx_on = bool(d["fixtures_available"].iloc[0]) if "fixtures_available" in d.columns else False
    d["disp_pts"] = d["fixture_adj_pts"] if fx_on and "fixture_adj_pts" in d.columns else d["fantasy_pts"]
    from player_model.fantasy import rank_and_captains
    return rank_and_captains(d, base_col="disp_pts")


@st.cache_data(ttl=600, show_spinner=False)
def _leaderboards():
    if not _ALLOW_LIVE_BUILD:
        return {}
    from player_model.fantasy_features import market_leaderboards
    return market_leaderboards(top_n=15)


def render() -> None:
    st.divider()
    st.subheader("🧰 More FPL tools")
    df = _board()
    if df.empty:
        st.info("The V1 board (output/fantasy_tips.csv) is not available yet — Fantasy Refresh writes it daily.")
        return
    st.caption("Planning tools from the original Fantasy page, on the V1 projection board.")

    t_diff, t_lead, t_sp, t_tr, p_chip, p_line, p_league, p_alert, t_sq = st.tabs(
        ["💎 Differentials", "📊 Prop leaderboards", "🎯 Set-pieces & pens", "🔄 Transfer helper",
         "🃏 Chip advisor", "📋 My lineup · your squad", "🏆 Mini-league", "🔔 Alerts / watchlist",
         "🏟️ Club squads"])

    with t_diff:
        st.caption("High projected points **and** low ownership — the picks that win mini-leagues.")
        max_own = st.slider("Max ownership %", 1.0, 30.0, 10.0, 0.5)
        try:
            from player_model.fantasy_features import differentials
            dd = differentials(df, max_owned=max_own, top_n=20)
            if dd.empty:
                st.info("Ownership data unavailable (needs the live FPL feed) — run Fantasy Refresh.")
            else:
                cols = [c for c in ["player_name", "team", "position", "price", "owned_pct", "fantasy_pts"] if c in dd.columns]
                ui.table(dd[cols].rename(columns={"player_name": "Player", "team": "Team", "position": "Pos",
                                                  "price": "£m", "owned_pct": "Owned %", "fantasy_pts": "Exp pts"}),
                         bars=("Exp pts",))
        except Exception as e:                                          # noqa: BLE001
            st.warning(f"Differentials unavailable: {e}")

    with t_lead:
        st.caption("Per-market probabilities from the **calibrated** prop models (this game).")
        try:
            lbs = _leaderboards()
            if not lbs:
                st.info("Leaderboards are computed by the pipeline, not the dashboard — scoring every PL player "
                        "here needs the 152 MB player history and five prop models, which exceeded the "
                        "deployment's memory. Run locally with `FANTASY_DASHBOARD_LIVE_BUILD=1` to see them.")
            else:
                lbl = {"goals": "⚽ Anytime scorer", "goals2": "🎯 2+ goals", "assists": "🅰️ Assist",
                       "sot2": "🎯 2+ SOT", "cards": "🟨 Booked"}
                lcols = st.columns(len(lbs))
                for col, (mkt, board) in zip(lcols, lbs.items()):
                    col.markdown(f"**{lbl.get(mkt, mkt)}**")
                    for r in board.head(10).itertuples():
                        col.write(f"{getattr(r, 'player_name', '')} — {r.prob:.0%}")
        except Exception as e:                                          # noqa: BLE001
            st.warning(f"Leaderboards unavailable: {e}")

    with t_sp:
        st.caption("Likely **penalty takers** + **set-piece goal threats** per club. Current squad only.")
        try:
            # Precomputed by fantasy_refresh.yml (player_model.fantasy_features --set-pieces). Computing it
            # here read the whole player_history parquet at render time - +1.35 GB on a ~1 GB app, and
            # Streamlit renders every tab's body, so merely opening Fantasy crashed the app (2026-10-25).
            @st.cache_data(ttl=1800, show_spinner=False)
            def _spt():
                p = Path(__file__).resolve().parent / "output" / "fantasy_set_pieces.csv"
                return pd.read_csv(p) if p.exists() else pd.DataFrame()

            sp = _spt()
            if sp.empty:
                st.info("No set-piece table yet - Fantasy Refresh writes output/fantasy_set_pieces.csv daily.")
            else:
                pick = st.selectbox("Club", ["All"] + sorted(sp["team"].dropna().unique()), key="sp_club")
                view = sp if pick == "All" else sp[sp["team"] == pick]
                ui.table(view.rename(columns={"player_name": "Player", "team": "Team", "position": "Pos",
                                              "pens": "Pens (career)", "sp_goals": "Set-piece goals"}), height=480)
        except Exception as e:                                          # noqa: BLE001
            st.warning(f"Set-pieces unavailable: {e}")

    with t_tr:
        st.caption("Enter your FPL team ID → sell/buy suggestions by projected points + availability. "
                   "(Find it in your FPL 'Points' page URL: /entry/**ID**/event/…)")
        tid = st.text_input("FPL team ID", placeholder="e.g. 1234567", key="v1_transfer_id")
        if tid.strip().isdigit():
            try:
                from player_model.fantasy_features import transfer_suggestions
                res = transfer_suggestions(int(tid), df)
                if res.get("error"):
                    st.warning(res["error"])
                else:
                    for s in res.get("sells", []):
                        st.markdown(f"**OUT:** {s['out']}")
                        for opt in s["in_options"]:
                            st.write(f"   → IN: {opt}")
                        st.write("")
                    st.caption(res.get("note", ""))
            except Exception as e:                                      # noqa: BLE001
                st.warning(f"Transfer helper unavailable: {e}")

    with p_chip:
        st.caption("Wildcard / Bench Boost / Triple Captain / Free Hit timing from the fixture calendar "
                   "(double & blank gameweeks + fixture difficulty).")
        try:
            from player_model.fantasy_features import chip_advisor

            @st.cache_data(ttl=1800, show_spinner=False)
            def _chips(_d):
                return chip_advisor(_d, horizon=8)

            ca = _chips(df)
            if not ca:
                st.info("No fixture calendar available yet (pre-season / FPL not published).")
            else:
                for r in ca.get("recommendations", []):
                    st.markdown("• " + r)
                gw = pd.DataFrame(ca.get("gameweeks", []))
                if not gw.empty:
                    ui.table(gw[["gw", "n_dgw", "n_bgw", "avg_fdr"]].rename(columns={
                        "gw": "GW", "n_dgw": "Double-GW teams", "n_bgw": "Blank teams", "avg_fdr": "Avg FDR"}))
                if ca.get("triple_captain"):
                    st.markdown("**Triple-captain shortlist:** " + " · ".join(ca["triple_captain"]))
        except Exception as e:                                          # noqa: BLE001
            st.warning(f"Chip advisor unavailable: {e}")

    with p_line:
        st.caption("Enter your FPL team ID → the best XI, bench order and (vice-)captain from **the players you "
                   "already own**. For the best squad regardless of ownership, use **⭐ Best legal squad** above.")
        ltid = st.text_input("FPL team ID ", placeholder="e.g. 1234567", key="lineup_id")
        if ltid.strip().isdigit():
            try:
                from player_model.fantasy_features import auto_lineup
                res = auto_lineup(int(ltid), df)
                if res.get("error"):
                    st.warning(res["error"])
                else:
                    st.success(f"**{res['formation']}**  ·  XI projected {res['xi_pts']:.1f} pts  ·  "
                               f"(C) {res['captain']} · (VC) {res['vice_captain']}")
                    ren = {"player_name": "Player", "team": "Team", "position": "Pos", "fantasy_pts": "xPts"}
                    keep = ["player_name", "team", "position", "fantasy_pts"]
                    st.markdown("**Starting XI**")
                    ui.table(res["xi"][[c for c in keep if c in res["xi"].columns]].rename(columns=ren), bars=("xPts",))
                    st.markdown("**Bench** (autosub order)")
                    ui.table(res["bench"][[c for c in keep if c in res["bench"].columns]].rename(columns=ren), bars=("xPts",))
            except Exception as e:                                      # noqa: BLE001
                st.warning(f"Auto-lineup unavailable: {e}")

    with p_league:
        st.caption("Enter your classic mini-league ID → the league template (most-owned) + differentials "
                   "(high projected points, low ownership *in your league*).")
        lid = st.text_input("Mini-league ID", placeholder="e.g. 314", key="league_id_in")
        if lid.strip().isdigit():
            try:
                from player_model.fantasy_features import mini_league_analysis
                res = mini_league_analysis(int(lid), df)
                if res.get("error"):
                    st.warning(res["error"])
                else:
                    st.caption(f"{res['n_managers']} managers · GW{res['gw']}")
                    c1, c2 = st.columns(2)
                    ren = {"player": "Player", "team": "Team", "league_own_pct": "Own %", "xpts": "xPts"}
                    c1.markdown("**League template (most-owned)**")
                    ui.table(res["template"][["player", "team", "league_own_pct", "xpts"]].rename(columns=ren),
                             container=c1, bars=("xPts",))
                    c2.markdown("**Differentials (≤25% owned)**")
                    ui.table(res["differentials"][["player", "team", "league_own_pct", "xpts"]].rename(columns=ren),
                             container=c2, bars=("xPts",))
            except Exception as e:                                      # noqa: BLE001
                st.warning(f"Mini-league unavailable: {e}")

    with p_alert:
        st.caption("Movers to act on: injuries/doubts, and price-change candidates (net FPL transfers this "
                   "gameweek). Optionally filter to a watchlist.")
        wl_txt = st.text_input("Watchlist (comma-separated names, optional)", key="watchlist_in")
        wl = [w.strip() for w in wl_txt.split(",") if w.strip()] or None
        try:
            from player_model.fantasy_features import fantasy_alerts
            al = fantasy_alerts(df, watchlist=wl)
            a1, a2, a3 = st.columns(3)
            a1.markdown("**🚑 Injuries / doubts**")
            ui.table(al["injuries"], container=a1, height=320)
            a2.markdown("**📈 Likely price risers**")
            ui.table(al["price_risers"], container=a2, height=320)
            a3.markdown("**📉 Likely price fallers**")
            ui.table(al["price_fallers"], container=a3, height=320)
        except Exception as e:                                          # noqa: BLE001
            st.warning(f"Alerts unavailable: {e}")

    with t_sq:
        n_opts = {"Current form (5)": 5, "Last 10": 10, "Last 20": 20, "Full season (38)": 38,
                  "~2 seasons (76)": 76, "~3 seasons (114)": 114, "~5 seasons (190)": 190}

        @st.cache_data(ttl=3600, show_spinner="Recomputing form window…")
        def _squads(n):
            try:
                from player_model.fantasy import build_squads
                return build_squads(n)
            except Exception:                                           # noqa: BLE001
                return pd.DataFrame()

        wsel = st.selectbox("Form window", list(n_opts.keys()), index=0, key="sq_window",
                            help="Per-game stats = average over each player's last N games.")
        # ON DEMAND ONLY: build_squads reads the whole player_history parquet (every column). Streamlit
        # runs every tab body on each render, so an unconditional call loads it on EVERY page view -
        # the exact pattern that took the deployment over its ~1 GB twice. It now loads on request.
        if not (st.session_state.get("sq_loaded") or st.button("Load club squads", key="sq_load")):
            st.caption("Club squads read the full player history, so they load only when asked.")
            return
        st.session_state["sq_loaded"] = True
        sq = _squads(n_opts[wsel])
        if sq.empty:
            st.info("Squad data unavailable (no PL parquet / models).")
        else:
            official = (BASE_DIR / "output" / "pl_squads_official.csv").exists()
            st.caption(("✅ Official current squads — daily transfer-window refresh" if official
                        else "⚠️ Rosters from latest parquet form") + f" · {sq['team'].nunique()} clubs")
            club = st.selectbox("Club", sorted(sq["team"].dropna().unique()), key="sq_club")
            csq = sq[sq["team"] == club].copy()
            csq["Role"] = csq["position"].map({"G": "GK", "D": "DEF", "M": "MID", "F": "FWD"}).fillna(csq["position"])
            disp = csq.rename(columns={"player_name": "Player", "games_used": "Games", "minutes_pg": "Min/g",
                                       "goals_pg": "Goals/g", "assists_pg": "Ast/g", "sot_pg": "SOT/g",
                                       "shots_pg": "Shots/g", "cards_pg": "Cards/g", "rating_pg": "Rating",
                                       "saves_pg": "Saves/g"})
            order = ["Player", "Role", "Games", "Min/g", "Goals/g", "Ast/g", "SOT/g", "Shots/g", "Cards/g",
                     "Rating", "Saves/g"]
            st.dataframe(disp[[c for c in order if c in disp.columns]], width="stretch", hide_index=True, height=520)
