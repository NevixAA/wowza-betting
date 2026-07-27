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
df["disp_pts"] = (df["fixture_adj_pts"] if (_fx_on and "fixture_adj_pts" in df.columns)
                  else df["fantasy_pts"])
df = df.sort_values("disp_pts", ascending=False).reset_index(drop=True)
df["overall_rank"] = range(1, len(df) + 1)
df["pos_rank"] = df.groupby("position")["disp_pts"].rank(ascending=False, method="first").astype(int)
# captain = top 3 among AVAILABLE players (never captain an injured one)
_healthy = ~df["injured"].astype(bool) if "injured" in df.columns else pd.Series(True, index=df.index)
df["captain_pick"] = False
df.loc[df[_healthy].head(3).index, "captain_pick"] = True

if _fx_on:
    st.success(f"**Opponent-adjusted** — points scaled by each team's **{wsel_fx.lower()}** fixtures "
               "(opponent goals-conceded × home/away). FDR > 1 = easy run, < 1 = hard run.", icon="✅")
else:
    st.warning("**Pre-season / no fixtures published yet:** showing form-based points. The opponent-"
               "adjusted fixture layer is wired and engages automatically once PL fixtures publish. "
               "(Season start also adds a current-squad filter + clean-sheet/save points for DEF/GK.)",
               icon="⚠️")

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
# injury/doubt badge prefixed to the player name
if "player_name" in show.columns:
    show["Player"] = show.apply(
        lambda r: ("🚑 " if r.get("injured") else "⚠️ " if r.get("doubtful") else "") + str(r["player_name"]),
        axis=1)
disp_cols = [c for c in ["overall_rank", "Player", "team", "position", "price", "value",
                         "p_goal", "p_assist", "p_sot2", "dc_pts", "cs_pts", "disp_pts",
                         "avg_fdr", "next_fixtures"]
             if c in show.columns]
show = show[disp_cols].rename(columns={
    "overall_rank": "#", "team": "Team", "position": "Pos", "price": "£m", "value": "Pts/£",
    "p_goal": "P(goal)", "p_assist": "P(assist)", "p_sot2": "P(SOT2+)",
    "dc_pts": "Def", "cs_pts": "CS", "disp_pts": "Exp pts", "avg_fdr": "FDR",
    "next_fixtures": "Next fixtures",
})
for c in ["P(goal)", "P(assist)", "P(SOT2+)"]:
    if c in show.columns:
        show[c] = (show[c] * 100).round(0).astype("Int64").astype(str) + "%"
st.dataframe(show, width="stretch", hide_index=True, height=560)
st.caption("🚑 = injured/unavailable · ⚠️ = doubtful (from official FPL status). "
           "Def = defensive-contribution pts (approx — source lacks clearances/recoveries). "
           "CS = expected clean-sheet pts (DEF/GK/MID, from fixture difficulty).")

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
