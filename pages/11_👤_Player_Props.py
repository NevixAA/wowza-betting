"""
Player Props — upcoming prop signals, split by whether a bookmaker priced them.

Rebuilt 2026-10-12 (owner: "we know the odds for player props — upgrade the visuality"):

* Two boards, by PRICE, not by tier. A prop with a bookmaker price can be compared with the
  model (edge, EV). One without can only show the model's view. Invariant 13: an unpriced prop is
  AVOID by construction, never a model rejection, so the two are never mixed.
* The old "World Cup 2026" section held every unpriced CLUB signal months after the tournament
  ended, and its cards printed "nan" for a missing lazy-factor list. Both gone.
* Day-grouped tables (kickoff in Israel time) in the same style as the Dashboard tips board,
  instead of one tall dark card per player.

Props are PAPER-ONLY, permanently (CLAUDE.md invariant 2): accurate model, no betting edge.
Nothing on this page is a bet.
"""
import html as _html
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

import dashboard_ui as ui

st.set_page_config(page_title="Player Props | Wowza", page_icon="👤", layout="wide")
ui.autorefresh(minutes=2, key="11_player_props_refresh")
BASE_DIR = Path(__file__).resolve().parents[1]
TIPS_FILE = BASE_DIR / "output" / "player_tips.csv"
LEDGER_FILE = BASE_DIR / "output" / "player_ledger.csv"
TZ_NAME = "Asia/Jerusalem"

ui.page_header("👤 Player Props", "Model vs bookmaker on every upcoming prop", badge="PAPER · RESEARCH")

MARKET_LABEL = {
    "goals": "⚽ Anytime scorer", "goals2": "⚽ 2+ goals", "goals3": "⚽ Hat-trick",
    "assists": "🅰️ Assist",
    "sot": "🎯 1+ shot on target", "sot2": "🎯 2+ shots on target",
    "sot3": "🎯 3+ shots on target", "sot4": "🎯 4+ shots on target",
    "cards": "🟨 Booked",
}
TIER_CLR = {"SNIPER": "#c42b2b", "MARKSMAN": "#d97706", "VALUABLE": "#2563eb",
            "PAPER": "#6b7280", "AVOID": "#9ca3af"}
TIER_ORDER = ["SNIPER", "MARKSMAN", "VALUABLE", "PAPER", "AVOID"]


@st.cache_data(ttl=120)
def load_tips() -> pd.DataFrame:
    if not TIPS_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(TIPS_FILE, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"] >= pd.Timestamp.now().normalize()].copy()
    for c in ("model_prob", "fair_odds", "market_odds", "ev", "edge_abs", "confidence"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["_priced"] = df["market_odds"].notna() & (df["market_odds"] > 1.0)
    # One row per player, market and day. Two feeds can name the same fixture differently
    # ("Oviedo vs SD Eibar" / "Oviedo vs Eibar"), which showed the same prop twice; keep the
    # priced row, then the one with a kickoff time.
    key = [c for c in ("player_id", "market") if c in df.columns]
    if len(key) == 2:
        df["_has_ko"] = pd.to_datetime(df.get("kickoff_utc"), errors="coerce").notna()
        df = (df.sort_values(["_priced", "_has_ko"], ascending=False, kind="mergesort")
                .assign(_d=df["date"].dt.date)
                .drop_duplicates(["_d", *key], keep="first")
                .drop(columns=["_d", "_has_ko"]))
    return df


df = load_tips()
if df.empty:
    st.info("⏳ No upcoming player-prop signals yet.")
    st.stop()

priced = df[df["_priced"]]
unpriced = df[~df["_priced"]]

# ── Headline ──────────────────────────────────────────────────────────────────
_beats = priced[priced["model_prob"] > 1 / priced["market_odds"]]
ui.metric_row([
    {"label": "Upcoming props", "value": f"{len(df):,}",
     "help": f"{df['match'].nunique()} fixtures, {df['player_name'].nunique()} players."},
    {"label": "Priced by a bookmaker", "value": f"{len(priced):,}",
     "delta": f"{100 * len(priced) / len(df):.0f}% of props", "delta_color": "off",
     "help": "Only these can be compared with the market."},
    {"label": "Model above the price", "value": f"{len(_beats):,}",
     "help": "Priced props where the model's probability exceeds the bookmaker's implied one. "
             "Paper only — props have no proven betting edge (tested 8 ways)."},
    {"label": "Model only (no price)", "value": f"{len(unpriced):,}",
     "help": "No bookmaker offers this prop yet. Not a model rejection."},
])
st.caption(
    "Why so many are unpriced: bookmakers price **anytime scorer** deeply but rarely offer "
    "**1+ shot on target**, which is where most of the model's strongest signals sit. Markets "
    "deepen toward kickoff, so prices appear over the last 1–3 days.")

# ── Filters ───────────────────────────────────────────────────────────────────
f1, f2, f3 = st.columns([2, 2, 1])
with f1:
    _mk = [m for m in MARKET_LABEL if m in set(df["market"])] + \
          sorted(set(df["market"]) - set(MARKET_LABEL))
    mkt_filter = st.multiselect("Market", _mk, default=_mk,
                                format_func=lambda m: MARKET_LABEL.get(m, m), key="pp_mkt")
with f2:
    _lg = sorted(df["league"].dropna().unique())
    league_filter = st.multiselect("League", _lg, default=[], placeholder="All leagues",
                                   key="pp_lg")
with f3:
    min_p = st.slider("Min model %", 0, 95, 0, 5, key="pp_minp")


def _filt(d: pd.DataFrame) -> pd.DataFrame:
    d = d[d["market"].isin(mkt_filter) & (d["model_prob"] * 100 >= min_p)]
    if league_filter:
        d = d[d["league"].isin(league_filter)]
    return d


_CSS = """
<style>
.pp-wrap{overflow-x:auto;margin-bottom:6px}
.pp{border-collapse:collapse;width:100%;font-size:.92rem;min-width:820px;table-layout:fixed}
.pp th{text-align:left;font-weight:600;font-size:.72rem;letter-spacing:.04em;text-transform:uppercase;
       opacity:.65;padding:6px 8px;border-bottom:1px solid rgba(128,128,128,.35)}
.pp td{padding:7px 8px;border-bottom:1px solid rgba(128,128,128,.15);vertical-align:middle}
.pp .num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.pp .ko{font-variant-numeric:tabular-nums;font-weight:600;white-space:nowrap}
.pp .sub{display:block;font-size:.74rem;opacity:.65;margin-top:2px;white-space:nowrap;
         overflow:hidden;text-overflow:ellipsis}
.pp .pill{display:inline-block;padding:2px 9px;border-radius:999px;color:#fff;font-weight:700;font-size:.72rem}
.pp .pos{color:#15803d;font-weight:700}.pp .neg{color:#b91c1c}
.pp-day{font-weight:700;font-size:1.02rem;margin:18px 0 4px}
.pp-day small{font-weight:400;opacity:.65}
</style>"""


def _board(d: pd.DataFrame, priced_view: bool, limit: int) -> str:
    try:
        tz = ZoneInfo(TZ_NAME)
    except Exception:                                       # noqa: BLE001
        tz = ZoneInfo("UTC")
    d = d.copy()
    ko = pd.to_datetime(d.get("kickoff_utc"), errors="coerce", utc=True)
    d["_local"] = ko.dt.tz_convert(tz)
    d["_day"] = d["_local"].dt.strftime("%Y-%m-%d").fillna(d["date"].dt.strftime("%Y-%m-%d"))
    if priced_view:
        d["_edge"] = (d["model_prob"] - 1 / d["market_odds"]) * 100
        d = d.sort_values(["_day", "_edge"], ascending=[True, False])
    else:
        d = d.sort_values(["_day", "model_prob"], ascending=[True, False])
    shown = d.groupby("_day", group_keys=False).head(limit)
    today = pd.Timestamp.now(tz=tz).date()
    cols = ("<colgroup><col style='width:6%'><col style='width:20%'><col style='width:22%'>"
            "<col style='width:16%'><col style='width:7%'>"
            + ("<col style='width:9%'><col style='width:9%'><col style='width:11%'>" if priced_view
               else "<col style='width:8%'><col style='width:14%'>") + "</colgroup>")
    head = ("<tr><th>Kickoff</th><th>Player</th><th>Match</th><th>Market</th>"
            "<th class='num'>Model</th>"
            + ("<th class='num'>Book odds</th><th class='num'>Edge</th><th>Tier</th>" if priced_view
               else "<th class='num'>Fair odds</th><th>Note</th>") + "</tr>")
    out = [_CSS]
    for day, g in shown.groupby("_day", sort=True):
        dd = pd.Timestamp(day)
        rel = {0: "Today", 1: "Tomorrow"}.get((dd.date() - today).days, "")
        total = int((d["_day"] == day).sum())
        more = f" · top {len(g)} shown" if total > len(g) else ""
        out.append(f"<div class='pp-day'>{rel + ' · ' if rel else ''}{dd:%a %d %b} "
                   f"<small>— {total} prop{'s' if total != 1 else ''}{more}</small></div>")
        out.append(f"<div class='pp-wrap'><table class='pp'>{cols}{head}")
        for _, r in g.iterrows():
            kot = r["_local"].strftime("%H:%M") if pd.notna(r["_local"]) else "–"
            pos = "" if pd.isna(r.get("position")) else f" · {_html.escape(str(r['position']))}"
            player = (f"<b>{_html.escape(str(r['player_name']))}</b>"
                      f"<span class='sub'>{_html.escape(str(r.get('team', '')))}{pos}</span>")
            match = (f"{_html.escape(str(r['match']))}"
                     f"<span class='sub'>{_html.escape(str(r['league']))}</span>")
            mkt = MARKET_LABEL.get(r["market"], str(r["market"]))
            model = f"<b>{r['model_prob'] * 100:.0f}%</b>"
            if priced_view:
                impl = 100 / r["market_odds"]
                e = r["_edge"]
                tier = str(r.get("tier", ""))
                cells = (f"<td class='num'><b>{r['market_odds']:.2f}</b>"
                         f"<span class='sub'>{impl:.0f}% implied</span></td>"
                         f"<td class='num'><span class='{'pos' if e > 0 else 'neg'}'>{e:+.1f} pts</span>"
                         + (f"<span class='sub'>EV {r['ev']:+.0%}</span>" if pd.notna(r.get("ev")) else "")
                         + f"</td><td><span class='pill' style='background:{TIER_CLR.get(tier, '#6b7280')}'>"
                         f"{_html.escape(tier)}</span></td>")
            else:
                fair = r.get("fair_odds")
                cells = (f"<td class='num'>{'–' if pd.isna(fair) else f'{float(fair):.2f}'}</td>"
                         f"<td><span class='sub' style='opacity:.6'>no bookmaker price yet</span></td>")
            out.append(f"<tr><td class='ko'>{kot}</td><td>{player}</td><td>{match}</td>"
                       f"<td>{mkt}</td><td class='num'>{model}</td>{cells}</tr>")
        out.append("</table></div>")
    return "\n".join(out)


tab_p, tab_u = st.tabs([f"💰 Priced by bookmakers ({len(priced)})",
                        f"🧠 Model only — no price yet ({len(unpriced)})"])
with tab_p:
    p = _filt(priced)
    if p.empty:
        st.info("No priced props match the filters. Prices usually appear 1–3 days before kickoff.")
    else:
        st.markdown(_board(p, True, limit=60), unsafe_allow_html=True)
        st.caption("**Edge** = model probability minus the bookmaker's implied probability, in "
                   "percentage points; green means the model rates the player higher than the price "
                   "does. Bookmaker prices include their margin. Kickoff in Israel time. Paper only.")
with tab_u:
    u = _filt(unpriced)
    if u.empty:
        st.info("Nothing unpriced matches the filters.")
    else:
        st.markdown(_board(u, False, limit=25), unsafe_allow_html=True)
        st.caption("Strongest model signals per day (top 25). **Fair odds** = 1 / model probability "
                   "— compare with your own bookmaker. Kickoff in Israel time. Paper only.")

with st.expander("📊 Raw data — every upcoming prop"):
    show_cols = ["date", "kickoff_utc", "league", "match", "player_name", "position", "team",
                 "market", "model_prob", "fair_odds", "market_odds", "ev", "tier", "confidence",
                 "n_games"]
    st.dataframe(df.sort_values("model_prob", ascending=False)[[c for c in show_cols if c in df.columns]],
                 width="stretch", hide_index=True)

# ── Results / track record ────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📒 Results — settled player-prop signals")


@st.cache_data(ttl=120)
def load_results():
    if not LEDGER_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(LEDGER_FILE, low_memory=False)


led = load_results()
# The all-time record counts SNIPER + MARKSMAN only; the raw ledger logs every tier for data.
if not led.empty and "tier" in led.columns:
    led = led[led["tier"].astype(str).str.upper().isin(["SNIPER", "MARKSMAN"])].copy()
if led.empty or "result" not in led.columns:
    st.info("No SNIPER/MARKSMAN prop results yet (`output/player_ledger.csv`).")
else:
    settled = led[led["result"].astype(str).str.upper().isin(["WIN", "LOSS"])].copy()
    if settled.empty:
        st.info("No settled (WIN/LOSS) signals yet — pending/void only.")
    else:
        wins = int((settled["result"].astype(str).str.upper() == "WIN").sum())
        n = len(settled)
        roi = (pd.to_numeric(settled["pnl"], errors="coerce").mean() * 100
               if "pnl" in settled.columns else float("nan"))
        ui.metric_row([
            {"label": "Settled", "value": f"{n}"},
            {"label": "Record", "value": f"{wins}W – {n - wins}L"},
            {"label": "Win rate", "value": f"{wins / n * 100:.0f}%"},
            {"label": "Flat ROI (1u, paper)", "value": f"{roi:+.1f}%" if roi == roi else "—"},
        ])
        cols = [c for c in ["match_date", "league", "player_name", "market", "tier",
                            "market_odds", "result", "pnl"] if c in settled.columns]
        st.dataframe(settled.sort_values("match_date", ascending=False)[cols].head(150),
                     width="stretch", hide_index=True)

st.caption("Player props are paper research only — the model is accurate but has shown no betting "
           "edge against the market.")

# ── Disclaimer & Terms (shown on every dashboard page) ──
from utils.disclaimer import disclaimer_footer  # noqa: E402
disclaimer_footer()
