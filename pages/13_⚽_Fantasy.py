"""
Fantasy (FPL) Dashboard — the FANTASY signal family.
Model expected-points projections for Premier League players. This is a PREDICTION
product (no odds / no betting edge) — separate from the SNIPER/MARKSMAN betting tips.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

import dashboard_ui as ui

st.set_page_config(page_title="Fantasy | Wowza", page_icon="⚽", layout="wide")
# Was components.v1.html with a JS reload — an API whose announced removal date (2026-06-01) has
# passed, and which reloaded the whole tab and so DISCARDED every filter the user had set.
ui.autorefresh(minutes=2, key="fantasy_refresh")

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
st.caption("Top three available players by expected points. Injured players are never suggested "
           "as captain.")
cols = st.columns(max(len(cap), 1))
for c, r in zip(cols, cap.itertuples()):
    avail = getattr(r, "availability", "") or ""
    ui.player_card(
        c,
        name=r.player_name,
        position=getattr(r, "position", ""),
        points=float(r.disp_pts),
        team=getattr(r, "team", "") or "",
        price=getattr(r, "price", None),
        fixture=getattr(r, "next_fixtures", "") or "",
        p_goal=getattr(r, "p_goal", None),
        p_assist=getattr(r, "p_assist", None),
        flag=("🚑 " if getattr(r, "injured", False)
              else "⚠️ " if getattr(r, "doubtful", False) else ""),
    )
    if avail and avail not in ("available", "unknown"):
        c.caption(f"⚠️ {avail}")

# ── Per-position top picks ────────────────────────────────────────────────────
st.subheader("📋 Top by position")
# A compact table per position rather than `col.write(f"{rank}. {name} — {pts}")`, which gave
# four columns of unformatted text with no team, no price and no way to compare down a column.
pcols = st.columns(4)
for col, pos in zip(pcols, POS_CODES):
    sub = df[df["position"] == pos].head(5).copy()
    col.markdown(f"**{POS_LABEL.get(pos, pos)}**")
    if sub.empty:
        col.caption("—")
        continue
    sub["Player"] = [
        ("🚑 " if r.get("injured") else "⚠️ " if r.get("doubtful") else "") + str(r["player_name"])
        for _, r in sub.iterrows()]
    cols_show = ["Player"] + [c for c in ("team", "price", "disp_pts") if c in sub.columns]
    tbl = sub[cols_show].rename(columns={"team": "Team", "price": "£m", "disp_pts": "Pts"})
    _pmax = float(tbl["Pts"].max()) if "Pts" in tbl.columns and len(tbl) else 1.0
    col.dataframe(
        tbl, hide_index=True, width="stretch",
        column_config={
            "Player": st.column_config.TextColumn("Player", width="medium"),
            "Team": st.column_config.TextColumn("Team", width="small"),
            "£m": ui.money_col(),
            "Pts": ui.bar_col("Pts", max_value=max(_pmax, 0.1)),
        })

# ── Minutes history, for the sparkline in the projections table ───────────────
# The table shows Start% — the model's probability that a player starts — with nothing behind it.
# Two players on 70% look identical when one has played 90 minutes eight times running and the
# other alternates 90 and 12. That difference is the whole of rotation risk, and it is the FPL
# decision this page exists to inform.
#
# MINUTES, not goals or points. Measured on the 480-day club window: minutes has 0% zero rows
# (mean 73), while goals are 91.7% zeros and goals+assists 86.5%. A returns sparkline would be a
# flat line at zero for six cells in seven — decoration that looks like information.
#
# CLUB ROWS ONLY, which is invariant 12 and not optional. player_history is a match-level log
# where `team` is whoever the player turned out for that day, INCLUDING internationals: the most
# recent row for Saka is England, for Doku is Belgium, for Haaland is Norway. Joining on name
# alone and taking the latest rows agreed with the player's FPL club for only 37.5% of the squad.
# Joining on (name, resolved FPL club) is club-only and current-club by construction.
#
# Club names are resolved through src/team_names.resolve (invariant 11), which handles
# Coventry City->Coventry, Ipswich Town->Ipswich, Man City->Manchester City and
# Nott'm Forest->Nottingham Forest, and correctly REFUSES 'Man Utd' and 'Spurs' rather than
# guessing. Those two are the only hardcoded aliases, and both targets were checked to exist.
_FPL_TEAM_ALIAS = {"Man Utd": "Manchester United", "Spurs": "Tottenham"}
_MINS_WINDOW_DAYS = 365      # genuine recency: the unbounded window reached back to 2023
_MINS_POINTS = 8
_MINS_MIN_POINTS = 3         # below this the cell stays blank rather than drawing a fake trend


@st.cache_data(ttl=1800, show_spinner=False)
def _minutes_history(pairs: tuple) -> dict:
    """{(normalised name, fpl team): [minutes, oldest->newest]} for recent CLUB matches."""
    import unicodedata as _ud
    import re as _re
    from src.team_names import resolve as _resolve

    # BASE_DIR is this page's own (v9 root), not config's — this page never imports config, and
    # the first version referenced config.BASE_DIR and failed with a NameError that the caller's
    # except swallowed, leaving an empty column that looked exactly like "no history exists".
    fp = BASE_DIR / "player_history.parquet"
    if not fp.exists():
        return {}
    h = pd.read_parquet(fp, columns=["player_name", "team", "date", "minutes"])
    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    h = h.dropna(subset=["date"])
    if h.empty:
        return {}

    def _n(x):
        nf = _ud.normalize("NFKD", str(x or ""))
        a = "".join(c for c in nf if not _ud.combining(c)).lower()
        return _re.sub(r"[^a-z ]", "", a).strip()

    cands = sorted(h["team"].dropna().astype(str).unique())
    tmap = {}
    for _, t in pairs:
        if t not in tmap:
            tmap[t] = _FPL_TEAM_ALIAS.get(t) or _resolve(t, cands)
    # Key on the HISTORY club name; an unresolved club simply yields no sparkline.
    want = {(_n(nm), tmap.get(t)) for nm, t in pairs if tmap.get(t)}
    h["_n"] = h["player_name"].map(_n)
    h = h[[(n, t) in want for n, t in zip(h["_n"], h["team"])]]
    if h.empty:
        return {}
    h = h[h["date"] >= h["date"].max() - pd.Timedelta(days=_MINS_WINDOW_DAYS)]
    h = h.sort_values("date").groupby(["_n", "team"]).tail(_MINS_POINTS)
    out = {}
    back = {v: k for k, v in tmap.items() if v}
    for (n, t), g in h.groupby(["_n", "team"]):
        if len(g) >= _MINS_MIN_POINTS:
            out[(n, back.get(t, t))] = [float(x) for x in
                                        pd.to_numeric(g["minutes"], errors="coerce").fillna(0)]
    return out


def _norm_name(x):
    import unicodedata as _ud
    import re as _re
    nf = _ud.normalize("NFKD", str(x or ""))
    a = "".join(c for c in nf if not _ud.combining(c)).lower()
    return _re.sub(r"[^a-z ]", "", a).strip()


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
_mins_map = {}
try:
    _pairs = tuple(sorted({(str(a), str(b)) for a, b in
                           zip(show.get("player_name", pd.Series(dtype=str)),
                               show.get("team", pd.Series(dtype=str)))}))
    if _pairs:
        _mins_map = _minutes_history(_pairs)
except Exception as _e:                                      # noqa: BLE001
    # Surfaced, not swallowed. A silently-empty sparkline column looks identical to "no history
    # exists", which is how the first version of this appeared to work while doing nothing.
    _mins_map = {}
    st.caption(f"Minutes history unavailable ({type(_e).__name__}: {_e})")
if _mins_map and "player_name" in show.columns and "team" in show.columns:
    show["_mins"] = [
        _mins_map.get((_norm_name(a), str(b)))
        for a, b in zip(show["player_name"], show["team"])
    ]
    disp_cols = disp_cols + ["_mins"]

show = show[disp_cols].rename(columns={
    "_mins": "Minutes (last 8)",
    "overall_rank": "#", "team": "Team", "position": "Pos", "price": "£m", "value": "Pts/£",
    "p_goal": "P(goal)", "p_assist": "P(assist)", "p_sot2": "P(SOT2+)",
    "dc_pts": "Def", "cs_pts": "CS", "bonus_pts": "Bon", "disp_pts": "Exp pts",
    "p_start": "Start%", "xpts_rot": "xPts·rot", "avg_fdr": "FDR", "next_fixtures": "Next fixtures",
})

# NUMBERS STAY NUMBERS. This block used to do
#     show[c] = (show[c] * 100).round(0).astype(str) + "%"
# which rendered "73%" and silently BROKE SORTING: strings sort lexicographically, so clicking
# P(goal) descending gave 9%, 8%, 73%, 45%, 100% in that order. The most useful sort on the page
# did not work. Formatting now happens at RENDER time through column_config, so the stored value
# stays numeric and the header sorts correctly.
#
# Bars need an explicit max_value. An auto-scaled bar changes meaning as the filter changes — the
# same player would render half-full or full depending on who else is on screen — so the scale is
# pinned to the data actually being shown.
_pts_max = float(show["Exp pts"].max()) if "Exp pts" in show.columns and len(show) else 1.0
_colcfg = {
    "#": st.column_config.NumberColumn("#", width="small"),
    "Player": st.column_config.TextColumn("Player", width="medium"),
    "£m": ui.money_col(help="FPL price"),
    "Pts/£": ui.num_col("Pts/£", fmt="%.2f", help="Expected points per £m — value, not raw points"),
    "P(goal)": ui.pct_col("P(goal)", help="Calibrated probability of scoring"),
    "P(assist)": ui.pct_col("P(assist)"),
    "P(SOT2+)": ui.pct_col("P(SOT2+)", help="Two or more shots on target"),
    "Def": ui.num_col("Def", fmt="%.2f",
                      help="Defensive-contribution points (approximate — the source lacks "
                           "clearances and recoveries)"),
    "CS": ui.num_col("CS", fmt="%.2f", help="Clean-sheet points (DEF/GK/MID)"),
    "Bon": ui.num_col("Bon", fmt="%.2f", help="Expected bonus from BPS drivers"),
    "Exp pts": ui.bar_col("Exp pts", max_value=max(_pts_max, 0.1),
                          help="The headline projection — bar is relative to the top player shown"),
    "Start%": ui.pct_col("Start%", help="Probability of starting"),
    "xPts·rot": ui.num_col("xPts·rot", fmt="%.2f",
                           help="Rotation-adjusted: Exp pts × P(start)"),
    "FDR": ui.num_col("FDR", fmt="%.1f", help="Fixture difficulty, 1 easiest to 5 hardest"),
    "Next fixtures": st.column_config.TextColumn("Next fixtures", width="medium"),
    # Fixed 0-90 scale, NOT autoscaled per row. Left to autoscale, a player who went
    # 88-90-89 draws the same alarming zigzag as one who went 12-90-9, because each cell
    # would be normalised to its own range — turning the most useful column on the page into
    # the most misleading one.
    "Minutes (last 8)": ui.spark_col(
        "Minutes (last 8)", y_min=0, y_max=90,
        help="Minutes in the last 8 CLUB matches within a year, oldest to newest. Fixed 0-90 "
             "scale, so rows are comparable. Blank where fewer than 3 such matches exist. "
             "Read it next to Start%: a flat line near 90 is a nailed-on starter, a sawtooth "
             "is rotation risk."),
}
st.dataframe(show, width="stretch", hide_index=True, height=560,
             column_config={k: v for k, v in _colcfg.items() if k in show.columns})
st.caption("🚑 injured/unavailable · ⚠️ doubtful · ⚽ penalty taker — "
           "hover any column header for what it means.")

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
            ui.table(dd[cols].rename(columns={"player_name": "Player", "team": "Team",
                     "position": "Pos", "price": "£m", "owned_pct": "Owned %",
                     "fantasy_pts": "Exp pts"}), bars=("Exp pts",))
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
            ui.table(px[cols].rename(columns={"player_name": "Player", "team": "Team",
                     "position": "Pos", "price": "£m", "fantasy_pts": "Exp pts"}),
                     bars=("Exp pts",))
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
            ui.table(tk.rename(columns={"team": "Team", "avg_fdr": "Avg FDR"}),
                     height=520)
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
            ui.table(view.rename(columns={"player_name": "Player", "team": "Team",
                     "position": "Pos", "pens": "Pens (career)",
                     "sp_goals": "Set-piece goals"}), height=480)
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

# ── Planning tools (chip advisor / auto-lineup / mini-league / alerts) ──────────
st.divider()
st.subheader("🧭 Planning tools")
p_chip, p_line, p_league, p_alert = st.tabs(
    ["🃏 Chip advisor", "📋 Auto-lineup", "🏆 Mini-league", "🔔 Alerts / watchlist"])

with p_chip:
    st.caption("Wildcard / Bench Boost / Triple Captain / Free Hit timing from the fixture "
               "calendar (double & blank gameweeks + fixture difficulty).")
    try:
        from player_model.fantasy_features import chip_advisor

        @st.cache_data(ttl=1800, show_spinner=False)
        def _chips():
            return chip_advisor(df, horizon=8)

        ca = _chips()
        if not ca:
            st.info("No fixture calendar available yet (pre-season / FPL not published).")
        else:
            for r in ca.get("recommendations", []):
                st.markdown("• " + r)
            gw = pd.DataFrame(ca.get("gameweeks", []))
            if not gw.empty:
                ui.table(gw[["gw", "n_dgw", "n_bgw", "avg_fdr"]].rename(columns={
                    "gw": "GW", "n_dgw": "Double-GW teams", "n_bgw": "Blank teams",
                    "avg_fdr": "Avg FDR"}))
            if ca.get("triple_captain"):
                st.markdown("**Triple-captain shortlist:** " + " · ".join(ca["triple_captain"]))
    except Exception as e:
        st.warning(f"Chip advisor unavailable: {e}")

with p_line:
    st.caption("Enter your FPL team ID → optimal starting XI, bench order, and (vice-)captain "
               "by projected points.")
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
                xi = res["xi"][[c for c in ["player_name", "team", "position", "fantasy_pts"] if c in res["xi"].columns]]
                st.markdown("**Starting XI**")
                ui.table(xi.rename(columns={"player_name": "Player", "team": "Team",
                         "position": "Pos", "fantasy_pts": "xPts"}), bars=("xPts",))
                st.markdown("**Bench** (autosub order)")
                bn = res["bench"][[c for c in ["player_name", "team", "position", "fantasy_pts"] if c in res["bench"].columns]]
                ui.table(bn.rename(columns={"player_name": "Player", "team": "Team",
                         "position": "Pos", "fantasy_pts": "xPts"}), bars=("xPts",))
        except Exception as e:
            st.warning(f"Auto-lineup unavailable: {e}")

with p_league:
    st.caption("Enter your classic mini-league ID → the league template (most-owned) + "
               "differentials (high projected points, low ownership *in your league*).")
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
                c1.markdown("**League template (most-owned)**")
                ui.table(res["template"][["player", "team", "league_own_pct", "xpts"]].rename(
                    columns={"player": "Player", "team": "Team", "league_own_pct": "Own %",
                             "xpts": "xPts"}), container=c1, bars=("xPts",))
                c2.markdown("**Differentials (≤25% owned)**")
                ui.table(res["differentials"][["player", "team", "league_own_pct", "xpts"]].rename(
                    columns={"player": "Player", "team": "Team", "league_own_pct": "Own %",
                             "xpts": "xPts"}), container=c2, bars=("xPts",))
        except Exception as e:
            st.warning(f"Mini-league unavailable: {e}")

with p_alert:
    st.caption("Movers to act on: injuries/doubts, and price-change candidates (net FPL "
               "transfers this gameweek). Optionally filter to a watchlist.")
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
    except Exception as e:
        st.warning(f"Alerts unavailable: {e}")

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


# ── Disclaimer & Terms (shown on every dashboard page) ──
from utils.disclaimer import disclaimer_footer  # noqa: E402
disclaimer_footer()

# ── Projected vs actual ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🎯 Projected vs actual")
st.caption("Is the projection any good — and does it beat the number FPL publishes for free?")

_have_ep = "fpl_ep_next" in df.columns and "fpl_ppg" in df.columns
if not _have_ep:
    st.info("Actuals not in this projection frame yet. They appear once the fantasy refresh runs "
            "on the current code (fpl_ep_next / fpl_ppg are now retained).")
else:
    _c = df.copy()
    for _col in ("fantasy_pts", "fpl_ep_next", "fpl_ppg", "minutes_pg"):
        _c[_col] = pd.to_numeric(_c.get(_col), errors="coerce")
    # Players who barely feature are untestable and drag both error terms toward zero for the
    # wrong reason, so they are excluded rather than quietly averaged in.
    _c = _c[_c["minutes_pg"].fillna(0) >= 30].dropna(subset=["fantasy_pts", "fpl_ppg"])
    if _c.empty:
        st.info("No player with 30+ minutes per game and an actual PPG yet.")
    else:
        _c["err_ours"] = (_c["fantasy_pts"] - _c["fpl_ppg"]).abs()
        _c["err_fpl"] = (_c["fpl_ep_next"] - _c["fpl_ppg"]).abs()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Players compared", f"{len(_c):,}",
                  help="30+ minutes per game. Fringe players are untestable.")
        m2.metric("Our MAE", f"{_c['err_ours'].mean():.2f}",
                  help="Mean absolute error vs actual points per game, in FPL points.")
        _fpl_mae = _c["err_fpl"].mean()
        _delta = _fpl_mae - _c["err_ours"].mean()
        m3.metric("FPL's own MAE", f"{_fpl_mae:.2f}",
                  delta=f"{_delta:+.2f} vs ours", delta_color="normal",
                  help="FPL publishes ep_next free. If we are not closer than this, the model "
                       "adds nothing over a number anyone can read off their website.")
        m4.metric("Mean bias", f"{(_c['fantasy_pts'] - _c['fpl_ppg']).mean():+.2f}",
                  help="Positive = we project higher than players actually score.")
        if _delta > 0:
            st.success(f"Our projection is **{_delta:.2f} points closer** to actual PPG than "
                       f"FPL's own expected points.")
        else:
            st.warning(f"**FPL's free number is {-_delta:.2f} points closer** than ours. Until "
                       f"that flips, the projection is not earning its keep — the honest baseline "
                       f"for any fantasy model is the one the platform already gives you.")
        st.caption("`fpl_ppg` is season-to-date ACTUAL points per game, so comparing it to a "
                   "forward projection is only fair in aggregate and only once several gameweeks "
                   "exist. Early season it will look worse than it is.")
        with st.expander("Biggest misses"):
            _cols = [c for c in ["player_name", "team", "position", "fantasy_pts",
                                 "fpl_ep_next", "fpl_ppg", "err_ours", "err_fpl"]
                     if c in _c.columns]
            st.dataframe(_c.nlargest(15, "err_ours")[_cols].round(2),
                         width="stretch", hide_index=True)

    # History, once the append-only log has more than one day in it.
    try:
        from player_model.fantasy_log import calibration
        _hist = calibration()
        if len(_hist) > 1:
            st.markdown("**Calibration over time** — one row per day the projections ran")
            st.dataframe(_hist, width="stretch", hide_index=True)
            st.caption("`ours_beats_fpl` positive means our projection was closer to actual PPG "
                       "than FPL's ep_next that day.")
        elif len(_hist) == 1:
            st.caption("Projection log has one day so far. A trend needs several — it accumulates "
                       "one row per player per day from now on.")
    except Exception:
        pass
