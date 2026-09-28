"""
Fantasy (FPL) Dashboard — the FANTASY signal family.
Model expected-points projections for Premier League players. This is a PREDICTION
product (no odds / no betting edge) — separate from the SNIPER/MARKSMAN betting tips.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

import dashboard_ui as ui
# The shared three-repo chart grammar. Imported under a separate name because this page
# already binds `ui` to the legacy dashboard_ui helper.
from dashboard_data import fantasy as _fdata, ui as dui
# One shared ranking rule for the page and the CSV. The page used to recompute its own ordering
# and captain picks, which silently overrode whatever fantasy.py had written, so the two could
# disagree and a fix in one place did nothing in the other.
from player_model.fantasy import rank_and_captains, MIN_CAPTAIN_START

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
# RANK BY UNCONDITIONAL POINTS, and pick captains who will actually start.
#
# This block used to sort by `disp_pts` -- CONDITIONAL points, what a player scores IF he plays
# -- and mark the top three as captains after excluding only `injured`. Two consequences,
# measured on the 2026-09-27 board: 8 of the top 20 could not play (Ekitiké #2, Romero #3, both
# p_start 0.00), and BOTH captain suggestions, João Pedro and Pedro Porro, were `doubtful` at a
# 75% chance of playing. A captain is doubled, so a blank costs twice.
#
# It also silently overrode the ranking fantasy.py had already written, so fixing the CSV alone
# would have changed nothing on the page. One shared rule now serves both.
df = rank_and_captains(df, base_col="disp_pts")

# ── Data freshness, stated rather than assumed ───────────────────────────────
# Departed players are already filtered out by fantasy.py, which keeps only rows matching the
# current FPL squad. That filter is only as current as the FPL snapshot behind it, so the one
# way a player who has left can still appear is a STALE bootstrap -- which is exactly how a
# two-month-old snapshot once offered an injured Doku as a captaincy pick with nothing saying so.
# Age is read from output/fantasy_health.json, which derives it from the recorded-fetch sidecar
# and never from file mtime (git checkout resets mtime, so mtime always reads as fresh).
_health = {}
try:
    import json as _json
    _hp = BASE_DIR / "output" / "fantasy_health.json"
    if _hp.exists():
        _health = _json.loads(_hp.read_text(encoding="utf-8"))
except Exception:
    _health = {}

if _health:
    # AGE IS RECOMPUTED HERE, NOT READ FROM THE HEALTH FILE. fantasy_health.json records the age
    # AT THE MOMENT IT WAS GENERATED; reading that field later and showing it as "now" is the
    # same stale-number-presented-as-live bug this banner exists to warn about. Caught in
    # testing: the file said 24h while the feed was actually 43.5h old. The gameweek and
    # coverage fields are safe to reuse -- they do not decay by the hour -- but the age is not.
    _feeds = _health.get("feeds", {})
    _boot = _feeds.get("fpl_bootstrap", {})
    _age = _boot.get("age_hours")
    try:
        import json as _json2, time as _time
        _meta = _json2.loads((BASE_DIR / "output" / "fpl_cache_meta.json").read_text(encoding="utf-8"))
        _ts = _meta.get("fpl_bootstrap.json")
        if _ts:
            _age = (_time.time() - float(_ts)) / 3600.0
    except Exception:
        pass  # keep whatever the health file said; an unknown age is handled below
    _state = ("UNAVAILABLE" if _age is None
              else "CURRENT" if _age <= 12 else "AGING" if _age <= 36 else "STALE")
    _gw = _health.get("current_gameweek")
    _age_txt = f"{_age:.0f}h old" if isinstance(_age, (int, float)) else "age unknown"
    _hdr = f"**Gameweek {_gw}**" if _gw else "**Gameweek unknown**"
    if _state == "CURRENT":
        st.success(f"{_hdr} · FPL data {_age_txt} · squad filter current", icon="✅")
    elif _state == "AGING":
        st.warning(f"{_hdr} · FPL data **{_age_txt}** — past its 12h refresh window. A player who "
                   "has since left a club may still appear.", icon="⚠️")
    else:
        st.error(f"{_hdr} · FPL data **{_state}** ({_age_txt}). Treat the squad list as unreliable: "
                 "departed players are filtered using this snapshot, so a stale one lets them "
                 "through.", icon="🚨")
    if not _health.get("settlement_ready", False):
        st.caption("Forecast accuracy cannot be measured yet — projections only began carrying a "
                   "gameweek tag on 2026-09-27, so there is no settled history to score against.")

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

# ── Who is actually startable ─────────────────────────────────────────────────
# SHOWN, NOT ASSUMED. "Nobody is unavailable" and "the availability check returned nothing" look
# identical on a page that only renders the survivors — and the board did once carry eight
# unplayable players in its top twenty. The split is published so the check is visibly alive.
_split = _fdata.availability_split()
if _split.get("available"):
    _sd = pd.DataFrame([
        {"group": "Nailed (start ≥ 85%)", "players": _split["nailed"]},
        {"group": "Rotation risk (60–85%)", "players": _split["rotation_risk"]},
        {"group": "Doubtful (< 60%)", "players": _split["doubtful"]},
        {"group": "Ruled out (0%)", "players": _split["unavailable"]},
    ])
    _a1, _a2 = st.columns([3, 2])
    with _a1:
        st.altair_chart(dui.coverage_bars(_sd, "group", "players", h=150,
                                          title=f"Startability of all {_split['total']} "
                                                f"ranked players"),
                        use_container_width=False)
    with _a2:
        _bad = _split.get("unavailable_in_top20")
        if _bad == 0:
            st.success(f"**0 unplayable players in the top 20.** The ranking is the "
                       f"availability-gated one (points × start probability, hard zero for "
                       f"anyone ruled out), not the raw projection.", icon="✅")
        elif _bad:
            st.error(f"**{_bad} unplayable player(s) in the top 20.** The board is being sorted "
                     f"on a conditional points column somewhere. Ranking must use overall_rank.",
                     icon="🚨")
        st.caption(f"{_split['unavailable']} of {_split['total']} ranked players are ruled out "
                   f"entirely. They stay in the file — a projection for an injured player is "
                   f"still correct about what he would score if he played — but they cannot "
                   f"reach the top of a list that is meant to be picked from.")

# ── Captaincy picks ───────────────────────────────────────────────────────────
st.subheader("🏆 Captaincy picks")
cap = df[df.get("captain_pick", False) == True] if "captain_pick" in df.columns else df.head(3)
st.caption("Ranked on points a player can actually score — projection × start probability. "
           "Anyone doubtful, injured or under a 60% chance of starting is never suggested: "
           "a captain is doubled, so a blank costs twice.")

# SAFE FLOOR OR HIGH CEILING — a mean cannot tell them apart. Two players on 5.5 xPts can be
# completely different bets: a nailed defender is appearance points plus a clean sheet, while a
# forward is a 45% chance of a goal and a long tail. The distribution is SIMULATED from the
# model's own event probabilities under FPL's scoring rules, never assumed — there is no normal
# approximation and no invented standard deviation here.
_dist = None
try:
    from player_model.fantasy_distribution import simulate as _simulate
    _dist = _simulate(df, n_sims=8000).set_index("player_name")
except Exception:
    _dist = None

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
    _conf = getattr(r, "start_confidence", "") or ""
    if _conf:
        _dot = {"Nailed": "🟢", "Likely starter": "🟡", "Rotation risk": "🟠",
                "Doubtful": "🔴", "Unavailable": "⚫"}.get(_conf, "")
        c.caption(f"{_dot} {_conf} · {float(getattr(r, 'p_start', 0)):.0%} start")
    if _dist is not None and r.player_name in _dist.index:
        _d = _dist.loc[r.player_name]
        _style = ("🚀 high ceiling" if float(_d.get("p_10plus", 0)) >= 0.15
                  else "🛡️ safe floor" if float(_d.get("p_blank", 1)) <= 0.10
                  else "balanced")
        c.caption(f"**{_style}** — median {float(_d['sim_median']):.0f}, "
                  f"ceiling {float(_d['sim_ceiling']):.0f} · "
                  f"blank {float(_d['p_blank']):.0%} · 10+ {float(_d['p_10plus']):.0%}")

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
    ["💎 Differentials", "⭐ Dream XI · all players", "📊 Prop leaderboards",
     "📅 Fixture ticker", "🎯 Set-pieces & pens", "🔄 Transfer helper"])

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
    st.caption("The best **legal 15** from every player in the projections — a target squad, not "
               "your current one. Budget, the 2-5-5-3 split and the three-per-club rule are all "
               "hard constraints, solved together. Ranked on points a player can actually score "
               "(projection × start probability), so nobody unavailable is picked. To optimise "
               "the squad you already own, use **📋 My lineup** under Planning tools.")
    # A REAL SQUAD, SOLVED RATHER THAN SORTED.
    #
    # This used best_xi(), which takes the top N per position and tries each formation. Its own
    # docstring admitted "budget is reported (not hard-constrained)", and it had no per-club
    # limit at all -- so what it returned could not be called a legal FPL team. On 2026-09-27 it
    # came in at £60.4m with at most three per club by luck, not because anything stopped it,
    # and it started two doubtful players plus a keeper with a 30% start probability.
    #
    # Budget and the three-per-club rule are COUPLING constraints -- whether a £12m forward
    # belongs depends on what the other fourteen cost -- which no per-position ranking can see.
    # That is what integer programming is for. Solves in ~0.03s.
    try:
        from player_model.fantasy_optimizer import (optimal_squad, verify_legal,
                                                    BUDGET, MAX_PER_CLUB)
        _bud = st.slider("Budget (£m)", 70.0, 110.0, float(BUDGET), 0.5,
                         help="FPL's own budget is £100m. Squeeze it to see what the model "
                              "gives up first.")
        sq = optimal_squad(df, budget=_bud)
        if not sq.get("feasible"):
            st.warning(f"No legal squad possible: {sq.get('reason', 'infeasible')}")
        else:
            bad = verify_legal(sq, _bud)
            if bad:
                st.error("Solver returned an ILLEGAL squad — " + "; ".join(bad))
            else:
                st.success(f"Legal FPL squad · 15 players · 2-5-5-3 · ≤£{_bud:.0f}m · "
                           f"max {MAX_PER_CLUB} per club — all enforced, not assumed.", icon="✅")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Formation", sq["formation"])
            m2.metric("XI points", f"{sq['xi_points']:.1f}")
            m3.metric("Cost", f"£{sq['total_cost']:.1f}m")
            m4.metric("In the bank", f"£{sq['money_remaining']:.1f}m")

            _cap = sq["captain"]; _vc = sq["vice_captain"]
            st.markdown(f"**(C) {_cap['player_name']}** · {_cap.get('team','')} — "
                        f"doubles to **{sq['captain_points'] * 2:.1f}** pts"
                        + (f"  ·  **(VC) {_vc['player_name']}**" if _vc is not None else ""))

            # ── The pitch ────────────────────────────────────────────────────
            # A lineup is a shape, and a table cannot show a shape. Reading "5-3-2" off a
            # formation cell is not the same as seeing five across the back. Rows are laid out
            # keeper-at-the-bottom the way a team sheet is drawn, and each tile carries only
            # what you decide on: who, where, and what he is worth.
            _cap_name = str(_cap.get("player_name", ""))
            _vc_name = str(_vc["player_name"]) if _vc is not None else ""

            def _tile(r):
                mark = " **(C)**" if r["player_name"] == _cap_name else (
                       " **(VC)**" if r["player_name"] == _vc_name else "")
                conf = str(r.get("start_confidence", "") or "")
                dot = {"Nailed": "🟢", "Likely starter": "🟡", "Rotation risk": "🟠",
                       "Doubtful": "🔴", "Unavailable": "⚫"}.get(conf, "")
                return (f"<div style='text-align:center;padding:6px 2px;line-height:1.25'>"
                        f"<b>{r['player_name']}</b>{mark}<br>"
                        f"<span style='opacity:.65;font-size:.85em'>{r.get('team','')} · "
                        f"£{float(r.get('price',0)):.1f}m</span><br>"
                        f"<span style='font-size:1.05em'><b>{float(r['xpts']):.2f}</b></span> "
                        f"<span style='font-size:.85em'>{dot}</span></div>")

            st.markdown("**Starting XI**")
            _xi = sq["xi"]
            for _pos in ("FWD", "MID", "DEF", "GKP"):
                _row = _xi[_xi["position"] == _pos]
                if _row.empty:
                    continue
                # Pad each row so it stays centred regardless of how many play there.
                _pad = max(0, (5 - len(_row)))
                _cols_row = st.columns([1] * (_pad // 2) + [2] * len(_row) + [1] * (_pad - _pad // 2))
                _slots = [c for c, w in zip(_cols_row, [1] * (_pad // 2) + [2] * len(_row)
                                            + [1] * (_pad - _pad // 2)) if w == 2]
                for _c, (_, _r) in zip(_slots, _row.iterrows()):
                    _c.markdown(_tile(_r), unsafe_allow_html=True)
            st.caption("🟢 nailed · 🟡 likely · 🟠 rotation risk — from start probability, "
                       "not a vibe. (C) captain, (VC) vice.")

            _cols = [c for c in ["player_name", "team", "position", "price", "xpts",
                                 "start_confidence"] if c in sq["xi"].columns]
            _ren = {"player_name": "Player", "team": "Team", "position": "Pos",
                    "price": "£m", "xpts": "xPts", "start_confidence": "Minutes"}
            with st.expander("Starting XI as a table"):
                ui.table(sq["xi"][_cols].rename(columns=_ren), bars=("xPts",))
            st.markdown("**Bench** — outfield in order, keeper last (that is how FPL autosubs).")
            ui.table(sq["bench"][_cols].rename(columns=_ren))
    except Exception as e:
        st.warning(f"Squad optimiser unavailable: {e}")

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
    # "Best XI" and "Auto-lineup" both read as "pick my XI", which is why they looked like two
    # tools in the wrong groups. They answer different questions and the grouping is right: Dream
    # XI picks from EVERY player (who to own), My lineup picks from YOUR squad (who to start this
    # week). The labels now say which pool each one draws from.
    ["🃏 Chip advisor", "📋 My lineup · your squad", "🏆 Mini-league",
     "🔔 Alerts / watchlist"])

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
    st.caption("Enter your FPL team ID → the best XI, bench order and (vice-)captain from **the "
               "players you already own**. For the best XI in the game regardless of ownership, "
               "use **⭐ Dream XI** under FPL tools.")
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

# ── REAL SETTLEMENT: pre-deadline forecast vs the points actually scored ──────
# The section below this one compares a forward projection against SEASON-TO-DATE points per
# game, which is not a settlement -- it scores a forecast against an average that already
# contains the matches being forecast. This block is the real test: what was projected before
# the deadline, against what the player went on to score that gameweek.
try:
    import json as _json3
    _perf_p = BASE_DIR / "output" / "fantasy_performance.json"
    _perf = _json3.loads(_perf_p.read_text(encoding="utf-8")) if _perf_p.exists() else {}
except Exception:
    _perf = {}

_ov = (_perf or {}).get("overall") or {}
if _ov:
    _gws = (_perf.get("meta") or {}).get("settled_gameweeks") or []
    st.markdown(f"**Settled gameweeks: {', '.join('GW'+str(g) for g in _gws)}** — "
                f"pre-deadline forecast against points actually scored.")
    _w, _f = _ov.get("wowza", {}), _ov.get("fpl_ep_next", {})
    _wc = _ov.get("wowza_conditional", {})
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Player-gameweeks", f"{_w.get('n', 0):,}")
    s2.metric("Wowza MAE", f"{_w.get('mae', float('nan')):.3f}",
              help="Mean absolute error in FPL points, on the unconditional projection "
                   "(points × start probability) — the number the board now ranks by.")
    s3.metric("FPL ep_next MAE", f"{_f.get('mae', float('nan')):.3f}",
              delta=f"{_w.get('mae', 0) - _f.get('mae', 0):+.3f} vs Wowza",
              delta_color="inverse")
    s4.metric("Rank correlation", f"{_w.get('spearman', float('nan')):.3f}",
              help=f"FPL's is {_f.get('spearman', float('nan')):.3f}. Spearman on actual points.")

    if _w.get("mae", 9e9) < _f.get("mae", 9e9):
        st.success(f"Wowza beats FPL's free number by "
                   f"{_f['mae'] - _w['mae']:.3f} MAE on settled gameweeks.", icon="✅")
    else:
        st.warning(f"**FPL's free number is still {_w.get('mae', 0) - _f.get('mae', 0):.3f} MAE "
                   f"closer.** That is the honest state of it, and it stays on this page until "
                   f"it flips.", icon="⚠️")

    # A chart, not another row of metrics. Two gameweeks of MAE side by side on ONE scale says
    # "we lose, consistently" in a glance; four st.metric tiles make the reader do the compare.
    _bm = _fdata.benchmark()
    if _bm.get("available") and len(_bm.get("table") or []):
        _bd = pd.DataFrame(_bm["table"])
        _long = _bd.melt(id_vars=["gw", "n"], value_vars=["wowza_mae", "fpl_mae"],
                         var_name="series", value_name="value")
        _c1, _c2 = st.columns(2)
        with _c1:
            st.altair_chart(
                dui.compare_bars(_long, "gw", "value", "series",
                                 order=["wowza_mae", "fpl_mae"],
                                 title="Error by gameweek — lower is better",
                                 y_title="points of error"),
                use_container_width=False)
        with _c2:
            _lr = _bd.melt(id_vars=["gw", "n"],
                           value_vars=["wowza_spearman", "fpl_spearman"],
                           var_name="series", value_name="value")
            # Rank correlation gets its OWN chart rather than a second axis on the one above.
            # Error and correlation are different scales pointing in opposite directions, and a
            # twin axis would let the picture be tuned to say whatever the author wanted.
            st.altair_chart(
                dui.compare_bars(_lr, "gw", "value", "series",
                                 order=["wowza_spearman", "fpl_spearman"],
                                 title="Rank correlation — higher is better (separate scale)",
                                 y_title="Spearman"),
                use_container_width=False)
        st.caption(f"Sample: " + " · ".join(f"GW{int(r['gw'])} n={int(r['n']):,}"
                                            for _, r in _bd.iterrows()) +
                   ". Every point on both charts is the same set of players in the same week.")

    if _wc:
        st.caption(f"Scoring the CONDITIONAL projection instead gives MAE "
                   f"{_wc.get('mae', float('nan')):.3f} and rank correlation "
                   f"{_wc.get('spearman', float('nan')):.3f} — much worse, because "
                   f"{_perf.get('actual_zero_share', 0):.0%} of real gameweek scores are zero "
                   f"(the player did not play) and a conditional number never claimed to "
                   f"predict those. Both are shown so the choice is not a silent one.")

    _un = (_perf.get("meta") or {}).get("unsettleable") or []
    if _un:
        with st.expander(f"Why {len(_un)} gameweek(s) are not counted"):
            for _u in _un:
                st.write(f"**GW{_u.get('gw')}** — {_u.get('why')}")
else:
    st.info("No settled gameweeks yet. Projections began carrying a gameweek tag on "
            "2026-09-27; settlement needs a snapshot either side of a deadline.", icon="ℹ️")

st.markdown("##### Proxy check — projection vs season-to-date PPG")
st.caption("Kept because it covers every player every day, but it is a PROXY, not a settlement: "
           "season-to-date PPG includes the matches being forecast.")

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
    #
    # The empty case USED TO RENDER NOTHING. With zero rows neither branch fired and the except
    # swallowed anything else, so the section just ended — indistinguishable from a broken page.
    # That is the state it is actually in right now, and for a reason worth naming on screen:
    # append() was called as `_log_proj(out)` where the variable is `proj`, so it raised NameError
    # into a handler that logged a warning nobody read, and the log wrote zero rows across three
    # days of green workflow runs. Fixed 2026-08-27; the first rows land on the next daily
    # fantasy_refresh, which the workflow does commit per-file.
    st.markdown("**Calibration over time** — one row per day the projections ran")
    try:
        from player_model.fantasy_log import calibration
        _hist = calibration()
        if len(_hist) > 1:
            # Charted, not tabulated. The question is whether the gap to FPL's free number is
            # closing or widening, and that is a direction — a table of daily MAEs makes the
            # reader compute differences in their head.
            _h = _hist.copy()
            _h["snapshot_date"] = pd.to_datetime(_h["snapshot_date"], errors="coerce")
            _h = _h.dropna(subset=["snapshot_date"]).sort_values("snapshot_date")
            st.line_chart(_h.set_index("snapshot_date")[["mae_ours", "mae_fpl"]],
                          height=260)
            st.caption("Mean absolute error against actual PPG, in FPL points. **Lower is "
                       "better**, and the line that matters is whether `mae_ours` sits below "
                       "`mae_fpl` — FPL publishes ep_next for free, so that is the baseline the "
                       "model has to beat to be worth running.")
            _last = _h.iloc[-1]
            _won = int((_h["ours_beats_fpl"] > 0).sum())
            st.caption(f"Ours closer on {_won} of {len(_h)} day(s). Latest: ours "
                       f"{_last['mae_ours']:.2f} vs FPL {_last['mae_fpl']:.2f} "
                       f"({_last['ours_beats_fpl']:+.2f}).")
            ui.table(_hist)
        elif len(_hist) == 1:
            st.info("Projection log has **one day** so far, so there is no trend to draw yet. "
                    "It accumulates one row per player per day from here.")
        else:
            st.info("**No projection history yet.** The log is written by `fantasy_log.append` "
                    "during the daily `fantasy_refresh` workflow and committed per-file. It "
                    "recorded zero rows until 2026-08-27 because `append` was called with an "
                    "undefined name, so the first rows arrive on the next scheduled run.")
    except Exception as _e:                                  # noqa: BLE001
        # Surfaced. A bare `pass` here is what let the empty log look like a design choice.
        st.caption(f"Calibration history unavailable ({type(_e).__name__}: {_e})")
