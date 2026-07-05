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

POS_LABEL = {"F": "Forward", "M": "Midfielder", "D": "Defender", "G": "Goalkeeper"}

st.title("⚽ Fantasy (FPL) Tips")
st.caption("Premier League expected-points projections — captaincy, transfers, per-position. "
           "Predictions, **not** betting tips.")

if st.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()


@st.cache_data(ttl=120)
def _load():
    if not TIPS_FILE.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(TIPS_FILE)
    except Exception:
        return pd.DataFrame()


df = _load()

if df.empty:
    st.info("No fantasy projections yet. They populate once the model runs on Premier League "
            "fixtures (in-season). Run `python -m player_model.fantasy` to generate.")
    st.stop()

st.warning("**v1 (pre-season):** projections use each player's latest form and are attack-focused. "
           "At season start these upgrade to opponent-adjusted per-GW projections, a current-squad "
           "filter (no departed players), and clean-sheet/save points for defenders & keepers.", icon="⚠️")

# ── Captaincy picks ───────────────────────────────────────────────────────────
st.subheader("🏆 Captaincy picks")
cap = df[df.get("captain_pick", False) == True] if "captain_pick" in df.columns else df.head(3)
cols = st.columns(max(len(cap), 1))
for c, r in zip(cols, cap.itertuples()):
    c.metric(f"{r.player_name} ({r.position})",
             f"{r.fantasy_pts:.2f} pts",
             help=f"P(goal) {getattr(r,'p_goal',0):.0%} · P(assist) {getattr(r,'p_assist',0):.0%}")

# ── Per-position top picks ────────────────────────────────────────────────────
st.subheader("📋 Top by position")
pcols = st.columns(4)
for col, pos in zip(pcols, ["F", "M", "D", "G"]):
    sub = df[df["position"] == pos].head(5)
    col.markdown(f"**{POS_LABEL.get(pos, pos)}**")
    for r in sub.itertuples():
        col.write(f"{r.pos_rank}. {r.player_name} — {r.fantasy_pts:.2f}")

# ── Full ranked table ─────────────────────────────────────────────────────────
st.subheader("📊 Full projections")
posf = st.multiselect("Filter position", ["F", "M", "D", "G"], default=["F", "M", "D", "G"])
show = df[df["position"].isin(posf)].copy()
disp_cols = [c for c in ["overall_rank", "player_name", "team", "position",
                         "p_goal", "p_assist", "p_sot2", "fantasy_pts"] if c in show.columns]
show = show[disp_cols].rename(columns={
    "overall_rank": "#", "player_name": "Player", "team": "Team", "position": "Pos",
    "p_goal": "P(goal)", "p_assist": "P(assist)", "p_sot2": "P(SOT2+)", "fantasy_pts": "Exp pts",
})
for c in ["P(goal)", "P(assist)", "P(SOT2+)"]:
    if c in show.columns:
        show[c] = (show[c] * 100).round(0).astype("Int64").astype(str) + "%"
st.dataframe(show, width="stretch", hide_index=True, height=560)

st.caption(f"{len(df)} players · source: output/fantasy_tips.csv · FANTASY family (prediction, not betting)")

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
