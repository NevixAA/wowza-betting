"""
Fantasy (FPL) Dashboard — the FANTASY signal family.
Model expected-points projections for Premier League players. This is a PREDICTION
product (no odds / no betting edge) — separate from the SNIPER/MARKSMAN betting tips.
"""
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Fantasy | Wowza", page_icon="⚽", layout="wide")
components.html("<script>setTimeout(()=>window.location.reload(),120000)</script>", height=0)

BASE_DIR  = Path(__file__).resolve().parents[1]
TIPS_FILE = BASE_DIR / "output" / "fantasy_tips.csv"

POS_LABEL = {"FWD": "Forward", "MID": "Midfielder", "DEF": "Defender", "GKP": "Goalkeeper",
             "F": "Forward", "M": "Midfielder", "D": "Defender", "G": "Goalkeeper"}
POS_CODES = ["FWD", "MID", "DEF", "GKP"]

st.title("⚽ Fantasy (FPL) Tips")
st.caption("Premier League expected-points projections — captaincy, transfers, per-position. "
           "Predictions, **not** betting tips.")

if st.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()


FIX_OPTS = {"Next 5": 5, "Next 8": 8, "Next 10": 10}
wsel_fx = st.selectbox(
    "Fixture window (opponent-adjusted)", list(FIX_OPTS.keys()), index=0,
    help="Projected points are scaled by the difficulty of each team's next N fixtures "
         "(opponent goals-conceded rate × home/away). Off-season this falls back to form-only "
         "until fixtures publish.")
next_n = FIX_OPTS[wsel_fx]


@st.cache_data(ttl=600, show_spinner="Computing fixture-adjusted projections…")
def _load(nfx):
    try:
        from player_model.fantasy import build_fantasy_projections_fixtures
        d = build_fantasy_projections_fixtures(next_n=nfx)
        if not d.empty:
            return d
    except Exception:
        pass
    if TIPS_FILE.exists():          # fallback: pre-built base CSV
        try:
            return pd.read_csv(TIPS_FILE)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


df = _load(next_n)

if df.empty:
    st.info("No fantasy projections yet. They populate once the model runs on Premier League "
            "data. Run `python -m player_model.fantasy` to generate.")
    st.stop()

# Normalise position codes to FPL (FWD/MID/DEF/GKP) so filters/tables work whether the data
# is fresh (FPL codes) or a stale fallback CSV (old F/M/D/G codes).
_POS_ALIAS = {"F": "FWD", "M": "MID", "D": "DEF", "G": "GKP", "GK": "GKP",
              "FWD": "FWD", "MID": "MID", "DEF": "DEF", "GKP": "GKP"}
if "position" in df.columns:
    df["position"] = df["position"].map(lambda p: _POS_ALIAS.get(str(p), str(p)))

# Opponent-adjusted points when fixtures are available; else form-based. Re-rank on the shown points.
_fx_on = bool(df["fixtures_available"].iloc[0]) if "fixtures_available" in df.columns else False

# Points view: per single game, or TOTAL across the next-N fixtures (double/blank-GW aware).
_view = st.radio("Points view", ["Per game", f"Total next {next_n}"], horizontal=True,
                 help="Per game = one match. Total next N = summed across each team's actual "
                      "upcoming fixtures in the window (a double gameweek ~doubles it, a blank = 0).")
if _view.startswith("Total") and "total_xpts_next" in df.columns:
    df["disp_pts"] = df["total_xpts_next"]
elif _fx_on and "fixture_adj_pts" in df.columns:
    df["disp_pts"] = df["fixture_adj_pts"]
else:
    df["disp_pts"] = df["fantasy_pts"]
df = df.sort_values("disp_pts", ascending=False).reset_index(drop=True)
df["overall_rank"] = range(1, len(df) + 1)
df["pos_rank"] = df.groupby("position")["disp_pts"].rank(ascending=False, method="first").astype(int)
# captain = top 3 among AVAILABLE players (never captain an injured one)
_healthy = ~df["injured"].astype(bool) if "injured" in df.columns else pd.Series(True, index=df.index)
df["captain_pick"] = False
df.loc[df[_healthy].head(3).index, "captain_pick"] = True

if _fx_on:
    st.success(f"**Live FPL data** — current-squad only (transferred-out players removed), official "
               f"injury flags, and clean-sheet points from each team's **{wsel_fx.lower()}** fixture "
               "difficulty (official FDR 1–5; 1 = easiest, 5 = hardest).", icon="✅")
else:
    st.info("**Current-squad filter + injury flags + defensive-contribution & clean-sheet points are "
            "live** (from the official FPL API). Showing form-based points; the per-fixture FDR layer "
            "engages when FPL publishes upcoming fixtures. If you still see a departed player, the FPL "
            "feed hasn't refreshed yet — run the *Fantasy Refresh* action or wait for the daily job.",
            icon="ℹ️")

# ── Captaincy picks ───────────────────────────────────────────────────────────
st.subheader("🏆 Captaincy picks")
cap = df[df.get("captain_pick", False) == True] if "captain_pick" in df.columns else df.head(3)
cols = st.columns(max(len(cap), 1))
for c, r in zip(cols, cap.itertuples()):
    fx = getattr(r, "next_fixtures", "") or ""
    price = getattr(r, "price", None)
    avail = getattr(r, "availability", "") or ""
    help_txt = f"P(goal) {getattr(r,'p_goal',0):.0%} · P(assist) {getattr(r,'p_assist',0):.0%}"
    if price is not None and price == price:
        help_txt += f" · £{price}m"
    if avail and avail not in ("available", "unknown"):
        help_txt += f" · ⚠️ {avail}"
    if fx:
        help_txt += f" · next: {fx}"
    c.metric(f"{r.player_name} ({r.position})", f"{r.disp_pts:.2f} pts", help=help_txt)

# ── Per-position top picks ────────────────────────────────────────────────────
st.subheader("📋 Top by position")
pcols = st.columns(4)
for col, pos in zip(pcols, POS_CODES):
    sub = df[df["position"] == pos].head(5)
    col.markdown(f"**{POS_LABEL.get(pos, pos)}**")
    for r in sub.itertuples():
        flag = "🚑 " if getattr(r, "injured", False) else ("⚠️ " if getattr(r, "doubtful", False) else "")
        col.write(f"{r.pos_rank}. {flag}{r.player_name} — {r.disp_pts:.2f}")

# ── Full ranked table ─────────────────────────────────────────────────────────
st.subheader("📊 Full projections")
posf = st.multiselect("Filter position", POS_CODES, default=POS_CODES)
show = df[df["position"].isin(posf)].copy()
# injury/doubt badge + penalty-taker (⚽) marker prefixed to the player name
if "player_name" in show.columns:
    def _badge(r):
        pre = "🚑 " if r.get("injured") else "⚠️ " if r.get("doubtful") else ""
        pen = " ⚽" if r.get("is_pen_taker") else ""
        return pre + str(r["player_name"]) + pen
    show["Player"] = show.apply(_badge, axis=1)
disp_cols = [c for c in ["overall_rank", "Player", "team", "position", "price", "value",
                         "p_goal", "p_assist", "p_sot2", "dc_pts", "cs_pts", "bonus_pts",
                         "disp_pts", "p_start", "xpts_rot", "avg_fdr", "next_fixtures"]
             if c in show.columns]
show = show[disp_cols].rename(columns={
    "overall_rank": "#", "team": "Team", "position": "Pos", "price": "£m", "value": "Pts/£",
    "p_goal": "P(goal)", "p_assist": "P(assist)", "p_sot2": "P(SOT2+)",
    "dc_pts": "Def", "cs_pts": "CS", "bonus_pts": "Bon", "disp_pts": "Exp pts",
    "p_start": "Start%", "xpts_rot": "xPts·rot", "avg_fdr": "FDR", "next_fixtures": "Next fixtures",
})
for c in ["P(goal)", "P(assist)", "P(SOT2+)"]:
    if c in show.columns:
        show[c] = (show[c] * 100).round(0).astype("Int64").astype(str) + "%"
if "Start%" in show.columns:
    show["Start%"] = (show["Start%"] * 100).round(0).astype("Int64").astype(str) + "%"
st.dataframe(show, width="stretch", hide_index=True, height=560)
st.caption("🚑 = injured/unavailable · ⚠️ = doubtful · ⚽ = penalty taker. "
           "Def = defensive-contribution pts (approx — source lacks clearances/recoveries) · "
           "CS = clean-sheet pts (DEF/GK/MID) · Bon = expected bonus (BPS drivers) · "
           "Start% = P(start) · xPts·rot = rotation-adjusted expected points (Exp pts × Start%).")

st.caption(f"{len(df)} players · source: output/fantasy_tips.csv · FANTASY family (prediction, not betting)")

# ── Advanced tools (differentials / best XI / leaderboards / fixtures / transfers) ──
st.divider()
st.subheader("🧰 FPL tools")


@st.cache_data(ttl=600, show_spinner=False)
def _leaderboards():
    from player_model.fantasy_features import market_leaderboards
    return market_leaderboards(top_n=15)


@st.cache_data(ttl=600, show_spinner=False)
def _ticker(nfx):
    from player_model.fantasy_features import fixture_ticker
    return fixture_ticker(next_n=nfx)


t_diff, t_xi, t_lead, t_fix, t_sp, t_tr = st.tabs(
    ["💎 Differentials", "⭐ Best XI", "📊 Prop leaderboards", "📅 Fixture ticker",
     "🎯 Set-pieces & pens", "🔄 Transfer helper"])

with t_diff:
    st.caption("High projected points **and** low ownership — the picks that win mini-leagues.")
    max_own = st.slider("Max ownership %", 1.0, 30.0, 10.0, 0.5)
    try:
        from player_model.fantasy_features import differentials
        dd = differentials(df, max_owned=max_own, top_n=20)
        if dd.empty:
            st.info("Ownership data unavailable (needs the live FPL feed) — run Fantasy Refresh.")
        else:
            cols = [c for c in ["player_name", "team", "position", "price", "owned_pct",
                                "fantasy_pts"] if c in dd.columns]
            st.dataframe(dd[cols].rename(columns={"player_name": "Player", "team": "Team",
                         "position": "Pos", "price": "£m", "owned_pct": "Owned %",
                         "fantasy_pts": "Exp pts"}), hide_index=True, width="stretch")
    except Exception as e:
        st.warning(f"Differentials unavailable: {e}")

with t_xi:
    st.caption("Points-maximising legal starting XI (injured players excluded).")
    try:
        from player_model.fantasy_features import best_xi
        xi = best_xi(df)
        if not xi:
            st.info("Not enough players to build an XI yet.")
        else:
            c1, c2 = st.columns(2)
            c1.metric("Formation", xi["formation"])
            c2.metric("Total projected pts", f"{xi['total_pts']:.1f}"
                      + (f"  ·  £{xi['total_cost']:.1f}m" if xi.get("total_cost") else ""))
            px = xi["players"]
            cols = [c for c in ["player_name", "team", "position", "price", "fantasy_pts"] if c in px.columns]
            st.dataframe(px[cols].rename(columns={"player_name": "Player", "team": "Team",
                         "position": "Pos", "price": "£m", "fantasy_pts": "Exp pts"}),
                         hide_index=True, width="stretch")
    except Exception as e:
        st.warning(f"Best XI unavailable: {e}")

with t_lead:
    st.caption("Per-market probabilities from the **calibrated** prop models (this game).")
    try:
        lbs = _leaderboards()
        if not lbs:
            st.info("No leaderboards (models/PL data unavailable).")
        else:
            LBL = {"goals": "⚽ Anytime scorer", "goals2": "🎯 2+ goals", "assists": "🅰️ Assist",
                   "sot2": "🎯 2+ SOT", "cards": "🟨 Booked"}
            lcols = st.columns(len(lbs))
            for col, (mkt, board) in zip(lcols, lbs.items()):
                col.markdown(f"**{LBL.get(mkt, mkt)}**")
                for r in board.head(10).itertuples():
                    col.write(f"{getattr(r,'player_name','')} — {r.prob:.0%}")
    except Exception as e:
        st.warning(f"Leaderboards unavailable: {e}")

with t_fix:
    st.caption("Each club's next fixtures + official FDR (1 = easiest, 5 = hardest). Sorted easiest run first.")
    try:
        tk = _ticker(next_n)
        if tk.empty:
            st.info("No upcoming fixtures published in FPL yet (pre-season).")
        else:
            st.dataframe(tk.rename(columns={"team": "Team", "avg_fdr": "Avg FDR"}),
                         hide_index=True, width="stretch", height=520)
    except Exception as e:
        st.warning(f"Fixture ticker unavailable: {e}")

with t_sp:
    st.caption("Likely **penalty takers** + **set-piece goal threats** per club (from career "
               "penalties won/scored + set-piece/free-kick/headed goals). Current squad only.")
    try:
        from player_model.fantasy_features import set_piece_penalty_takers

        @st.cache_data(ttl=1800, show_spinner=False)
        def _spt():
            return set_piece_penalty_takers()

        sp = _spt()
        if sp.empty:
            st.info("No set-piece data (needs the live FPL feed + parquet history).")
        else:
            clubs = sorted(sp["team"].dropna().unique())
            pick = st.selectbox("Club", ["All"] + clubs)
            view = sp if pick == "All" else sp[sp["team"] == pick]
            st.dataframe(view.rename(columns={"player_name": "Player", "team": "Team",
                         "position": "Pos", "pens": "Pens (career)", "sp_goals": "Set-piece goals"}),
                         hide_index=True, width="stretch", height=480)
    except Exception as e:
        st.warning(f"Set-pieces unavailable: {e}")

with t_tr:
    st.caption("Enter your FPL team ID → sell/buy suggestions by projected points + availability. "
               "(Find it in your FPL 'Points' page URL: /entry/**ID**/event/…)")
    tid = st.text_input("FPL team ID", placeholder="e.g. 1234567")
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
        except Exception as e:
            st.warning(f"Transfer helper unavailable: {e}")

# ── Club squads ───────────────────────────────────────────────────────────────
st.divider()
st.subheader("🏟️ Club squads")

# Dynamic form window — per-game stats recomputed over the last N games from raw data.
N_OPTS = {"Current form (5)": 5, "Last 10": 10, "Last 20": 20, "Full season (38)": 38,
          "~2 seasons (76)": 76, "~3 seasons (114)": 114, "~5 seasons (190)": 190}


@st.cache_data(ttl=3600, show_spinner="Recomputing form window…")
def _squads(n):
    try:
        from player_model.fantasy import build_squads
        return build_squads(n)
    except Exception:
        return pd.DataFrame()


wsel = st.selectbox("Form window", list(N_OPTS.keys()), index=0,
                    help="Per-game stats = average over each player's last N games. "
                         "We have ~1 season/player of data, so windows beyond that show all available games.")
n = N_OPTS[wsel]
sq = _squads(n)
if sq.empty:
    st.info("Squad data unavailable (no PL parquet / models).")
else:
    official = (BASE_DIR / "output" / "pl_squads_official.csv").exists()
    st.caption(("✅ Official current squads — daily transfer-window refresh" if official
                else "⚠️ Rosters from latest parquet form; official daily squad refresh activates at season start")
               + f" · {sq['team'].nunique()} clubs")
    club = st.selectbox("Club", sorted(sq["team"].dropna().unique()))
    csq = sq[sq["team"] == club].copy()
    csq["Role"] = csq["position"].map({"G": "GK", "D": "DEF", "M": "MID", "F": "FWD"}).fillna(csq["position"])
    disp = csq.rename(columns={
        "player_name": "Player", "games_used": "Games", "minutes_pg": "Min/g", "goals_pg": "Goals/g",
        "assists_pg": "Ast/g", "sot_pg": "SOT/g", "shots_pg": "Shots/g", "cards_pg": "Cards/g",
        "rating_pg": "Rating", "saves_pg": "Saves/g",
    })
    order = ["Player", "Role", "Games", "Min/g", "Goals/g", "Ast/g", "SOT/g", "Shots/g",
             "Cards/g", "Rating", "Saves/g"]
    st.dataframe(disp[[c for c in order if c in disp.columns]],
                 width="stretch", hide_index=True, height=520)
    avg_games = int(csq["games_used"].dropna().mean()) if "games_used" in csq.columns and csq["games_used"].notna().any() else 0
    st.caption(f"{club}: {len(csq)} players · window: {wsel} · avg {avg_games} games/player used "
               "(new signings blank until they have PL history)")
