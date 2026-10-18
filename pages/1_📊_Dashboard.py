"""Dashboard — upcoming tips with drift signals."""
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd


def _read_csv(*args, **kwargs):
    """pd.read_csv with paper leagues removed — they are collected, never counted.

    See config.PAPER_LEAGUES. Every KPI on this page reads through here so a paper
    league cannot reach a total by way of a path that forgot the rule.
    """
    import config as _cfg
    return _cfg.drop_paper_leagues(pd.read_csv(*args, **kwargs))
import streamlit as st
from streamlit_autorefresh import st_autorefresh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
import dashboard_ui as ui

st.set_page_config(page_title="Dashboard | Wowza", page_icon="📊", layout="wide")
st_autorefresh(interval=5 * 60 * 1000, key="dash_refresh")

PYTHON = str(Path(sys.executable))
V9_DIR = str(Path(__file__).resolve().parents[1])


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
    df["model_type"]  = df["model_type"].fillna("").astype(str)
    # A blank tag must never default to one model — derive it from the league so new-format
    # rows can't show up under 'standard' (canonical map in config.model_type_for_league).
    _blank_mt = df["model_type"].str.strip().isin(["", "nan"])
    if _blank_mt.any() and "league" in df.columns:
        import config as _cfg
        df.loc[_blank_mt, "model_type"] = df.loc[_blank_mt, "league"].map(_cfg.model_type_for_league)
    df["drift_signal"] = df["drift_signal"].fillna("New").astype(str)
    df["signal_tier"]  = df["signal_tier"].fillna("").astype(str)
    return df


def drift_badge(signal: str) -> str:
    colors = {"Confirmed": "#00c896", "Conflicted": "#e94560", "Neutral": "#aaa", "New": "#666"}
    icons  = {"Confirmed": "✅", "Conflicted": "⚠️", "Neutral": "➡️", "New": "🆕"}
    c = colors.get(signal, "#666")
    i = icons.get(signal, "")
    return f'<span style="color:{c};font-weight:bold">{i} {signal}</span>'


def odds_badge(side: str, odds: float) -> str:
    color = "#4fc3f7" if side == "OVER" else "#ce93d8"
    return f'<span style="color:{color};font-weight:bold">{side} @ {odds:.2f}</span>'


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("## 📊 Today's Tips")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.access import is_public

col_run, col_upd, col_time = st.columns([2, 2, 6])

with col_run:
    if not is_public() and st.button("🔄 Run Predict Now", width="stretch"):
        with st.spinner("Fetching fixtures & computing predictions..."):
            result = subprocess.run(
                [PYTHON, "pipeline.py", "--mode", "predict"],
                cwd=V9_DIR, capture_output=True, text=True, timeout=120
            )
        if result.returncode == 0:
            st.success("Predictions updated!")
            st.cache_data.clear()
        else:
            st.error(f"Error: {result.stderr[-500:]}")

with col_upd:
    if not is_public() and st.button("📥 Update Results Now", width="stretch"):
        with st.spinner("Fetching match results..."):
            result = subprocess.run(
                [PYTHON, "update_results.py"],
                cwd=V9_DIR, capture_output=True, text=True, timeout=120
            )
        if result.returncode == 0:
            st.success("Results updated!")
            st.cache_data.clear()
        else:
            st.error(f"Error: {result.stderr[-500:]}")

# ── Top picks: 3 lists of 5 (src/top_picks.py, 2026-10-18) ────────────────────
# People take a few tips, not 60. These are the lists the morning album shows, frozen each morning and
# scored; before the album has frozen today's lists the same lists are shown as a preview.
@st.cache_data(ttl=300)
def _top_picks(day: str):
    from src import top_picks as _TP
    fz = _TP.frozen(day)
    return (fz, True) if not fz.empty else (_TP.todays_lists(day), False)


@st.cache_data(ttl=300)
def _top_scoreboard(days):
    from src import top_picks as _TP
    return _TP.scoreboard(days)


try:
    from src.top_picks import LIST_LABEL as _TPL, LISTS as _TPLISTS
    _today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    _tp, _is_frozen = _top_picks(_today)
    st.subheader("⭐ Top picks today")
    st.caption(("Frozen this morning and scored." if _is_frozen else "Preview — frozen when the morning album goes out.")
               + " Best bets: most expected profit with at least a 45% chance · Safest: the 5 shortest prices · "
                 "Value: most expected profit. Chance = honest win estimate learned from past results.")
    if _tp is None or _tp.empty:
        st.info("No tips for today yet.")
    else:
        _cols = st.columns(3)
        for _c, _name in zip(_cols, _TPLISTS):
            _x = _tp[_tp["list"] == _name].sort_values("rank")
            with _c:
                st.markdown(f"**{_TPL[_name]}**")
                if _x.empty:
                    st.caption("nothing qualifies today")
                    continue
                st.dataframe(pd.DataFrame({
                    "Match": _x["home_team"] + " v " + _x["away_team"], "Pick": _x["pick"],
                    "Odds": _x["odds"].map(lambda v: f"{float(v):.2f}"),
                    "Chance": _x["p_win"].map(lambda v: "–" if pd.isna(v) else f"{float(v):.0%}")}),
                    hide_index=True, width="stretch")
    _win = st.radio("Top-picks record", [7, 30, 0], horizontal=True,
                    format_func=lambda w: "All time" if w == 0 else f"Last {w} days", key="tp_window")
    _sb = _top_scoreboard(_win or None)
    if _sb.empty:
        st.caption("The record starts with the first frozen day.")
    else:
        st.dataframe(_sb.assign(list=_sb["list"].map(_TPL), hit_rate=(_sb["hit_rate"] * 100).round(1),
                                roi=(_sb["roi"] * 100).round(1), zero_days=(_sb["zero_days"] * 100).round(0))
                     .rename(columns={"list": "List", "days": "Days", "picks": "Picks", "wins": "Wins",
                                      "hit_rate": "Hit %", "roi": "ROI %", "zero_days": "0/5 days %"}),
                     hide_index=True, width="stretch")
    # Weekly table, stored week by week by Pro (src/studies/top_picks_weekly.py): live record + backtest.
    with st.expander("📅 Weekly table — live record and backtest, week by week"):
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
            st.dataframe(_w.sort_values(["week", "kind", "list"], ascending=[False, True, True])[
                ["week", "Kind", "List", "Hit %", "ROI %", "0/5 days %", "days", "picks"]].rename(
                columns={"week": "Week", "days": "Days", "picks": "Picks"}), hide_index=True, width="stretch")
            st.caption("Live = the lists users were shown, frozen each morning and graded. Backtest = the walk-forward "
                       "test re-run on all history each day. Earlier weeks are never rewritten.")
    st.divider()
except Exception as _e:                                              # noqa: BLE001 - a viewer never dies
    st.caption(f"Top picks unavailable ({type(_e).__name__})")

# ── Load data ──────────────────────────────────────────────────────────────────
df = load_bets()

if df.empty:
    st.warning("No predictions yet. Click 'Run Predict Now' to fetch tips.")
    st.stop()

# Upcoming fixtures only — hide matches that have already been played
df = df[df["date"] >= pd.Timestamp.now().normalize()].copy()
if df.empty:
    st.info("No upcoming fixtures right now. (Past matches are hidden — see Live History for results.)")
    st.stop()


@st.cache_data(ttl=60)
def load_board(date_from: str, date_to: str) -> pd.DataFrame:
    """The SAME rows the Telegram tips image draws (telegram_bot.tips_table): live tips from the
    two committed ledgers, first side per fixture, paper leagues out. The old board read
    side_bets.csv, which is not committed, so side markets never appeared on the deployed app."""
    from telegram_bot.tips_table import tips_rows
    return tips_rows(date_from, date_to)


# File mtime is NOT a freshness signal here: a git checkout resets it on every deploy, so it
# always read "just now". The newest recorded tip is a real timestamp.
_today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
_all_board = load_board(_today, "2099-12-31")
_last = (pd.to_datetime(_all_board["generated_at"], errors="coerce").max()
         if len(_all_board) else pd.NaT)
st.caption(f"Newest tip recorded: {_last:%d/%m/%Y %H:%M} UTC  •  {len(df)} fixtures priced"
           if pd.notna(_last) else f"{len(df)} fixtures priced")

# ── Filters ────────────────────────────────────────────────────────────────────
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    tier_filter = st.multiselect("Tier", ["SNIPER", "MARKSMAN", "VALUABLE"], default=["SNIPER", "MARKSMAN", "VALUABLE"])
with col_f2:
    available_models = sorted(df["model_type"].unique().tolist()) or ["standard", "new_format"]
    model_filter = st.multiselect("Model", available_models, default=available_models)
with col_f3:
    drift_filter = st.multiselect(
        "Drift Signal", ["Confirmed", "Neutral", "New", "Conflicted"],
        default=["Confirmed", "Neutral", "New", "Conflicted"]
    )

col_f4, col_f5, col_f6 = st.columns([2, 1, 1])
with col_f6:
    _MARKETS = ["O/U 2.5", "BTTS", "O/U 1.5", "O/U 3.5"]
    market_filter = st.multiselect("Market", _MARKETS, default=_MARKETS, key="dash_market")
with col_f4:
    all_leagues = sorted(df["league"].dropna().unique().tolist()) if "league" in df.columns else []
    league_filter = st.multiselect("League", all_leagues, default=all_leagues, key="dash_league")
with col_f5:
    _hi = df["date"].max()
    if len(_all_board):
        _hi = max(_hi, pd.Timestamp(_all_board["date"].max()))
    date_range = st.date_input(
        "Date range",
        value=(pd.Timestamp(_today).date(), _hi.date()),
        key="dash_dates",
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        date_from, date_to = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    else:
        date_from = date_to = pd.Timestamp(date_range[0]) if date_range else df["date"].min()

# ── Tips board ─────────────────────────────────────────────────────────────────
# Owner request 2026-10-12: "tips should be more readable". One board for every market instead
# of SNIPER cards + two plain tables + a side-market section that never rendered: grouped by
# match day, one line per tip, tier as a coloured pill, market AND side spelled out.
import html as _html
from zoneinfo import ZoneInfo

_TZ_NAME = "Asia/Jerusalem"
_TIER_CLR = {"SNIPER": "#c42b2b", "MARKSMAN": "#d97706", "VALUABLE": "#2563eb"}
_TIER_ICON = {"SNIPER": "🎯", "MARKSMAN": "🔫", "VALUABLE": "💎"}
_STAKE = {"SNIPER": "staked · full", "MARKSMAN": "staked · ¾", "VALUABLE": "paper"}
_SIDE_CLR = {"OVER": "#0e7490", "UNDER": "#7c3aed", "YES": "#15803d"}


def _market_group(m: str) -> str:
    return " ".join(m.split()[:-1]) if m else m       # "O/U 2.5 UNDER" -> "O/U 2.5"


def _board(rows: pd.DataFrame, order: str) -> str:
    try:
        tz = ZoneInfo(_TZ_NAME)
    except Exception:                                   # noqa: BLE001 — no tzdata: stay in UTC
        tz = ZoneInfo("UTC")
    now = pd.Timestamp.now(tz="UTC")
    rows = rows.copy()
    ko = pd.to_datetime(rows["kickoff_utc"], errors="coerce", utc=True)
    rows["_ko"] = ko
    rows["_local"] = ko.dt.tz_convert(tz)
    # Group by the LOCAL kickoff date, so a 00:30 kickoff sits under the day it is played on
    # here; the ledger's match_date is the fallback when no kickoff time is known.
    rows["_day"] = rows["_local"].dt.strftime("%Y-%m-%d").fillna(rows["date"].astype(str))
    rows["_t"] = rows["tier"].map({t: i for i, t in enumerate(_TIER_CLR)})
    css = """
<style>
.wz-wrap{overflow-x:auto;margin-bottom:6px}
.wz{border-collapse:collapse;width:100%;font-size:0.92rem;min-width:760px;table-layout:fixed}
.wz th{text-align:left;font-weight:600;font-size:0.72rem;letter-spacing:.04em;text-transform:uppercase;
       opacity:.65;padding:6px 8px;border-bottom:1px solid rgba(128,128,128,.35)}
.wz td{padding:7px 8px;border-bottom:1px solid rgba(128,128,128,.15);vertical-align:middle}
.wz tr.done td{opacity:.45}
.wz .num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.wz .ko{font-variant-numeric:tabular-nums;font-weight:600;white-space:nowrap}
.wz .lg{opacity:.7;font-size:.82rem}
.wz .pill{display:inline-block;padding:2px 9px;border-radius:999px;color:#fff;font-weight:700;
          font-size:.74rem;letter-spacing:.03em;white-space:nowrap}
.wz .sub{display:block;font-size:.7rem;opacity:.6;margin-top:2px;white-space:nowrap}
.wz .mk{font-weight:700;white-space:nowrap}
.wz .bar{display:inline-block;height:6px;border-radius:3px;vertical-align:middle;margin-left:6px}
.wz-day{font-weight:700;font-size:1.02rem;margin:18px 0 4px}
.wz-day small{font-weight:400;opacity:.65}
</style>"""
    out = [css]
    emax = max(float(rows["edge"].max() or 0), 1.0)
    for day, g in rows.groupby("_day", sort=True):
        d = pd.Timestamp(day)
        rel = {0: "Today", 1: "Tomorrow"}.get((d.date() - pd.Timestamp.now(tz=tz).date()).days, "")
        counts = " · ".join(f"{int((g['tier'] == t).sum())} {t}" for t in _TIER_CLR
                            if (g["tier"] == t).any())
        label = f"{rel + ' · ' if rel else ''}{d:%a %d %b}"
        out.append(f"<div class='wz-day'>{label} <small>— {len(g)} tip{'s' if len(g) != 1 else ''}"
                   f" ({counts})</small></div>")
        g = (g.sort_values(["_ko", "_t", "edge"], ascending=[True, True, False], na_position="last")
             if order == "Kickoff time" else g.sort_values(["_t", "edge"], ascending=[True, False]))
        out.append("<div class='wz-wrap'><table class='wz'><colgroup><col style='width:8%'><col style='width:33%'>"
                   "<col style='width:15%'><col style='width:15%'><col style='width:8%'>"
                   "<col style='width:12%'><col style='width:9%'></colgroup>"
                   "<tr><th>Kickoff</th><th>Match</th>"
                   "<th>Tip</th><th>Tier</th><th class='num'>Odds</th><th class='num'>Edge</th>"
                   "<th>Model</th></tr>")
        for _, r in g.iterrows():
            started = pd.notna(r["_ko"]) and r["_ko"] <= now
            kot = r["_local"].strftime("%H:%M") if pd.notna(r["_local"]) else "–"
            tier = str(r["tier"])
            clr = _TIER_CLR.get(tier, "#6b7280")
            parts = str(r["market"]).split()
            side = parts[-1] if parts else ""
            mk_txt = " ".join(parts[:-1]).replace("O/U ", "") if parts else ""
            side_word = {"OVER": "Over", "UNDER": "Under", "YES": "Yes"}.get(side, side.title())
            tip = (f"BTTS — {side_word}" if mk_txt == "BTTS" else f"{side_word} {mk_txt}")
            odds = pd.to_numeric(r["odds"], errors="coerce")
            edge = r["edge"]
            bar_w = 0 if pd.isna(edge) else int(46 * min(max(float(edge), 0) / emax, 1))
            extra = []
            if pd.notna(r.get("model_prob")):
                extra.append(f"model {float(r['model_prob']) * 100:.0f}%")
            if isinstance(r.get("drift"), str) and r["drift"] and r["drift"] != "nan":
                extra.append(f"drift {r['drift']}")
            model = "Standard" if str(r["model"]) == "standard" else "New-format"
            out.append(
                f"<tr class='{'done' if started else ''}'>"
                f"<td class='ko'>{kot}{'<span class=sub>started</span>' if started else ''}</td>"
                f"<td><b>{_html.escape(str(r['home_team']))}</b> v "
                f"<b>{_html.escape(str(r['away_team']))}</b>"
                f"<span class='sub lg'>{_html.escape(str(r['league']))}</span></td>"
                f"<td><span class='mk' style='color:{_SIDE_CLR.get(side, 'inherit')}'>{tip}</span>"
                f"{'<span class=sub>' + ' · '.join(extra) + '</span>' if extra else ''}</td>"
                f"<td><span class='pill' style='background:{clr}'>{_TIER_ICON.get(tier, '')} {tier}"
                f"</span><span class='sub'>{_STAKE.get(tier, '')}</span></td>"
                f"<td class='num'><b>{'–' if pd.isna(odds) else f'{odds:.2f}'}</b></td>"
                f"<td class='num'>{'–' if pd.isna(edge) else f'+{edge:.1f}%'}"
                f"<span class='bar' style='width:{bar_w}px;background:{clr}'></span></td>"
                f"<td class='lg'>{model}</td></tr>")
        out.append("</table></div>")
    return "\n".join(out)


board = load_board(date_from.strftime("%Y-%m-%d"), date_to.strftime("%Y-%m-%d"))
if len(board):
    _keep = board["tier"].isin(tier_filter) & board["market"].map(_market_group).isin(market_filter)
    if model_filter:
        _keep &= board["model"].isin(model_filter)
    if league_filter:
        _keep &= board["league"].isin(league_filter)
    # Drift exists for O/U 2.5 only; a side-market row has none and is never hidden by it.
    _keep &= board["drift"].isna() | board["drift"].isin(drift_filter)
    board = board[_keep]

st.markdown("## 🎯 Tips")
if board.empty:
    st.info("No tips match the current filters.")
else:
    _staked = int(board["tier"].isin(["SNIPER", "MARKSMAN"]).sum())
    _future = pd.to_datetime(board["kickoff_utc"], errors="coerce", utc=True)
    _next = _future[_future > pd.Timestamp.now(tz="UTC")].min()
    ui.metric_row([
        {"label": "Tips shown", "value": f"{len(board)}",
         "help": "Every live tip in the date range, all markets, one side per fixture."},
        {"label": "Staked (SNIPER + MARKSMAN)", "value": f"{_staked}",
         "help": "VALUABLE is recorded and tracked but is not a bet."},
        {"label": "Fixtures", "value": f"{board[['date', 'home_team', 'away_team']].drop_duplicates().shape[0]}"},
        {"label": "Next kickoff", "value": (f"{_next.tz_convert(_TZ_NAME):%a %H:%M}"
                                            if pd.notna(_next) else "—"),
         "help": f"Israel time ({_TZ_NAME})."},
    ])
    _order = st.radio("Order within a day", ["Kickoff time", "Best first"], horizontal=True,
                      key="dash_order", label_visibility="collapsed")
    st.markdown(_board(board, _order), unsafe_allow_html=True)
    st.caption(f"Kickoff in Israel time · the same tips as the Telegram image (from the ledgers, "
               f"first side per fixture, paper leagues excluded) · Edge is as recorded when the tip "
               f"was sent · not betting advice.")
    st.download_button("Download as CSV", board.drop(columns=["generated_at"], errors="ignore")
                       .to_csv(index=False).encode("utf-8"), "wowza_tips.csv", "text/csv")

st.divider()

# ── Settled performance ───────────────────────────────────────────────────────
# The dashboard had NO performance display at all: five tables, no chart, no metric. The one
# number a betting dashboard exists to show — are we up or down — was nowhere on the front page.
#
# THREE FILTERS, EACH OF WHICH CHANGES THE ANSWER, so all three are stated on screen:
#
#   1. STAKED TIERS ONLY. VALUABLE is not a bet. Including it here would report -28.30u instead
#      of -4.42u, because 123 unstaked VALUABLE rows carry -23.88u of notional loss. That single
#      filter is the difference between a small loss and a disaster.
#   2. FROM PERFORMANCE_CUTOFF_DATE. Earlier rows predate fixes and are excluded from P&L by
#      design (they still contribute to CLV).
#   3. SETTLED ONLY. An unsettled bet has no pnl and must not read as a zero.
#
# And n is displayed next to the total, because 88 bets over 13 days is not a track record and a
# curve drawn without its sample size invites exactly that reading.
#   4. WHICH MARKET. This was the filter the first version left unstated, and it is the one that
#      changes the answer most. Reading bets_ledger.csv alone reports the main O/U 2.5 line only,
#      so -4.42u sat on the front page looking like the system's record while over15 (+1.48u) and
#      btts (+9.07u) were not in it at all. Every staked market is now loaded and broken out
#      per market, because a single blended number cannot be right here: the total across all
#      three is +6.14u, and +9.07u of that is btts on NINE bets. Ex-btts it is -2.94u. One
#      headline figure would be either the wrong sign or the wrong story, whichever way it went.
_STAKED_TIERS = ("SNIPER", "MARKSMAN")
# (ledger file, column holding the market name, constant market name when there is no column)
_MKT_SOURCES = [("bets_ledger.csv", None, "O/U 2.5"),
                ("side_bets_ledger.csv", "market", None),
                ("ht_ledger.csv", None, "HT O/U")]
_cut = pd.Timestamp(getattr(config, "PERFORMANCE_CUTOFF_DATE", "1970-01-01"))
_parts, _missing = [], []
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
    except Exception as _e:                                 # noqa: BLE001
        st.caption(f"{_f} unreadable ({type(_e).__name__}: {_e})")
_s = (pd.concat(_parts, ignore_index=True).sort_values("_dt")
      if _parts else pd.DataFrame())

if len(_s):
    st.markdown("## 📈 Settled performance by market")
    _tot = float(_s["_pnl"].sum())
    _hit = float((_s["_pnl"] > 0).mean())

    # Per-market first, because the blend is not a fact about the system. Ordered by bet count so
    # the market with something to say about it is at the top and the 9-bet one is at the bottom.
    _THIN = 20
    _m = (_s.groupby("_mkt")
          .agg(Bets=("_pnl", "size"), PL=("_pnl", "sum"),
               Hit=("_pnl", lambda x: 100.0 * (x > 0).mean()),
               _avg_odds=("_odds", "mean"), _n_odds=("_odds", "count"))
          .reset_index().rename(columns={"_mkt": "Market"}))
    _m["Per bet"] = _m["PL"] / _m["Bets"]
    # Break-even at the price actually paid. Suppressed when prices are missing for a fifth of
    # the market's bets, since a mean over the priced fraction is not the level these bets faced.
    _m["Need"] = (100.0 / _m["_avg_odds"]).where(_m["_n_odds"] >= 0.8 * _m["Bets"])
    # Sparkline over the BET SEQUENCE, not weekly buckets: 15 days gives 2-3 weekly points,
    # which is not a line, whereas the running total has one point per bet (88 / 33 / 9).
    _seq = {k: [round(v, 3) for v in g.sort_values("_dt")["_pnl"].cumsum()]
            for k, g in _s.groupby("_mkt")}
    _m["Run"] = _m["Market"].map(lambda k: _seq.get(k) if len(_seq.get(k, [])) >= 3 else None)
    _m = _m.sort_values("Bets", ascending=False)

    _biggest = _m.iloc[_m["PL"].abs().values.argmax()]
    _ex = _s[_s["_mkt"] != _biggest["Market"]]
    ui.metric_row([
        {"label": "P/L — all staked markets", "value": f"{_tot:+.2f}u",
         "delta": f"{_tot / len(_s):+.3f}u per bet",
         "help": "Flat 1u stakes, SNIPER + MARKSMAN only. Read the per-market table below "
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
    ])

    ui.table(_m[["Market", "Bets", "PL", "Per bet", "Hit", "Need", "Run"]].round(2),
             sparks={"Run": {"help": "Running total in units, one point per settled bet, oldest "
                                     "to newest. Each row is scaled to its own range — compare "
                                     "the SHAPE, not the height, and read P/L for the level."}})
    st.caption(
        f"Flat 1u stakes · **staked tiers only** (SNIPER + MARKSMAN — VALUABLE is not a bet) · "
        f"settled only · from {_cut.date()}. **Hit** is the share won; **Need** is the "
        f"break-even rate implied by the average price paid, so Hit above Need is a profitable "
        f"market and below it is not."
        + (f" Thin: " + ", ".join(f"{r['Market']} n={int(r['Bets'])}"
                                  for _, r in _m[_m['Bets'] < _THIN].iterrows())
           + " — at these counts one result moves P/L by a whole unit."
           if (_m["Bets"] < _THIN).any() else ""))

    # Cumulative curve per market rather than one blended line, for the same reason the table is
    # split: a single curve here averaged a 39.8% market with an 88.9% one and showed neither.
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
        ui.table(_by.sort_values("Bets", ascending=False))
        if _missing:
            st.caption("Not present, so contributing nothing: " + ", ".join(_missing) + ".")

    # ── Closing line value ────────────────────────────────────────────────────
    # CLV was displayed NOWHERE in the dashboard, while three separate places carried comments
    # promising that rows excluded from P&L "still contribute to CLV" and ARCHITECTURE.md calls it
    # the real measure of whether the model knows something the market doesn't. The promise was
    # kept in the data and broken in the presentation.
    #
    # Reported as a MEDIAN, and this is the whole reason the panel is written carefully.
    # Raw mean CLV over this ledger is +17.94%. The median is +0.96%. The mean is fiction: 37% of
    # rows carry a closing price that cannot belong to the same selection - 128 rows close BELOW
    # 1.60 (median 1.28) against entry prices near 2.50, and 47 close near 4.33 against entries
    # near 2.00. An O/U 2.5 line does not travel that far. Reporting the mean would advertise a
    # 40% edge on top of a losing P/L.
    #
    # |CLV| <= 25% is the same cut v11 applies (CLV_PLAUSIBLE_ABS in scripts/v11_shadow.py), and
    # it is used here deliberately so the two repos cannot quote different CLV for the same rows.
    # On this ledger it rejects all 175 implausible rows with nothing left over.
    #
    # The exact-zero share is shown rather than hidden. 28% of surviving rows have
    # closing_odds == odds exactly. Because drift.py stores price CHANGES only, that reads as "no
    # change was ever recorded" - which is a genuine flat line for a fixture that was polled and a
    # missing observation for one that wasn't. I tried to separate them against
    # odds_history_v9.json and could not: the file is pruned (only 90 of 210 NON-zero rows appear
    # in it either), so absence proves nothing. Stripping the zeros moves the positive rate from
    # 36.5% to 51.0%, which is too large a swing to resolve silently in either direction.
    try:
        _cl = _read_csv(config.OUTPUT_DIR / "bets_ledger.csv", low_memory=False)
        _cl["_clv"] = pd.to_numeric(_cl.get("clv_pct"), errors="coerce")
        _cl["_o"] = pd.to_numeric(_cl.get("odds"), errors="coerce")
        _cl["_c"] = pd.to_numeric(_cl.get("closing_odds"), errors="coerce")
        _cl["_dt"] = pd.to_datetime(_cl.get("match_date"), errors="coerce")
        _raw = _cl[_cl["_clv"].notna()]
        _PLAUS = 25.0
        _ok = _raw[_raw["_clv"].abs() <= _PLAUS].copy()
        if len(_ok):
            _flat = (_ok["_c"] == _ok["_o"])
            _moved = _ok[~_flat]
            st.markdown("## 🎯 Closing line value")
            ui.metric_row([
                {"label": "Median CLV", "value": f"{_ok['_clv'].median():+.2f}%",
                 "help": "Median, not mean. The mean over these rows is "
                         f"{_ok['_clv'].mean():+.2f}% and the raw unfiltered mean is "
                         f"{_raw['_clv'].mean():+.2f}% — inflated by impossible closing prices."},
                {"label": "Clean observations", "value": f"{len(_ok):,}",
                 "delta": f"−{len(_raw) - len(_ok)} rejected",
                 "delta_color": "off",
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
            ])
            # Per-league, with the trend as a line so a direction is visible at all.
            _wk = (_moved.dropna(subset=["_dt"])
                   .assign(_w=lambda x: x["_dt"].dt.to_period("W").dt.start_time)
                   .groupby(["league", "_w"])["_clv"].median().reset_index())
            # Coerced to a real list, and only where there are enough points to BE a trend.
            # groupby-apply hands back numpy arrays, and a 1-point series renders as a flat line
            # that reads identically to "this league has been steady" when it means "we have one
            # week of data". Blank is the honest cell there; 3 weeks is the floor for a direction.
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
                # A SHARED y-scale: per-cell autoscaling would draw a league moving between
                # -1% and +1% with the same dramatic zigzag as one swinging -40% to +40%.
                _lim = float(max(5.0, _wk["_clv"].abs().max()))
                ui.table(_lg.round(2),
                         sparks={"Trend": {"y_min": -_lim, "y_max": _lim,
                                           "help": "Weekly median CLV %, blank under "
                                                   f"{_MIN_PTS} weeks of data. Shared scale "
                                                   f"±{_lim:.0f}% across all rows."}})
                st.caption(
                    f"Leagues with at least 5 rows that moved, of {_moved['league'].nunique()} "
                    f"with any. Flat rows excluded from these figures — see 'no recorded move'. "
                    f"n is small everywhere (largest {int(_lg['n'].max())}), and v11 wants 150 "
                    f"per segment before CLV gates a bet, so nothing here is settled.")
    except Exception as _e:                                  # noqa: BLE001
        st.caption(f"CLV panel unavailable ({type(_e).__name__}: {_e})")
    st.divider()

# ── HT O/U Predictions ────────────────────────────────────────────────────────
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

ht_preds = load_predictions_ht()
if not ht_preds.empty:
    st.markdown("---")
    st.markdown("### ⏱ Half-Time O/U Predictions")
    st.caption("Standard-format leagues only · Based on HT rolling features (HTHG/HTAG)")
    ht_rows = []
    for _, row in ht_preds.iterrows():
        p05 = float(row["p_ht_over05"])
        p15 = float(row["p_ht_over15"]) if "p_ht_over15" in row and pd.notna(row["p_ht_over15"]) else None
        fair05 = round(1 / max(p05, 0.01), 2)
        fair_u05 = round(1 / max(1 - p05, 0.01), 2)
        fair15 = round(1 / max(p15, 0.01), 2) if p15 else "—"
        ht_rows.append({
            "Date":          str(row["date"])[:10],
            "League":        row.get("league", ""),
            "Match":         f"{row['home_team']} vs {row['away_team']}",
            "P(HT OVER 0.5)":  f"{p05*100:.0f}%",
            "Fair OVER 0.5":   fair05,
            "Fair UNDER 0.5":  fair_u05,
            "P(HT OVER 1.5)":  f"{p15*100:.0f}%" if p15 else "—",
            "Fair OVER 1.5":   fair15,
        })
    st.dataframe(pd.DataFrame(ht_rows), width="stretch", hide_index=True)
    st.caption("Compare fair prices against your bookmaker's HT market to find value")

# ── Suppressed bets note ────────────────────────────────────────────────────────
avoided = df[df["bet"] == "AVOID"]
if not avoided.empty:
    with st.expander(f"🚫 {len(avoided)} fixtures suppressed (AVOID / both-losing guard)"):
        st.dataframe(
            avoided[["date", "league", "home_team", "away_team", "signal_tier", "model_type"]],
            width="stretch", hide_index=True
        )


# ── Disclaimer & Terms (shown on every dashboard page) ──
from utils.disclaimer import disclaimer_footer  # noqa: E402
disclaimer_footer()
