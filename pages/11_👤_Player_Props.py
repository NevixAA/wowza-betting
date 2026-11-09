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

Visual redesign 2026-10-25 (stage 3): the day boards are sortable st.dataframe tables (the hand-built HTML
table, its stylesheet and its hardcoded tier colours are gone), the three priced props where the model sits
furthest above the price lead the page as cards, and secondary filters live under "Advanced filters".
PRESENTATION ONLY — the same rows, filters and limits; components in dashboard_data/props_view.py.

Props are PAPER-ONLY, permanently (CLAUDE.md invariant 2): accurate model, no betting edge.
Nothing on this page is a bet.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

import dashboard_ui as ui
from src.tier_labels import tier_label

st.set_page_config(page_title="Player Props | Wowza", page_icon="👤", layout="wide")
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from dashboard_access import require as _require_tier  # noqa: E402
_require_tier("pro")  # tier guard (dashboard_access.PAGES), 2026-10-18

from dashboard_data import props_view as V  # noqa: E402
import importlib  # noqa: E402
V = importlib.reload(V)              # page + helper always the same version (see 15_League_Scout)

ui.autorefresh(minutes=2, key="11_player_props_refresh")
BASE_DIR = Path(__file__).resolve().parents[1]
TIPS_FILE = BASE_DIR / "output" / "player_tips.csv"
LEDGER_FILE = BASE_DIR / "output" / "player_ledger.csv"
MARKET_LABEL = V.MARKET_LABEL
_SUB = "Model vs bookmaker on every upcoming prop. Paper research — props have no proven betting edge."
_EYEBROW = "Tools · player props"


@st.cache_data(ttl=120)
def load_tips() -> pd.DataFrame:
    if not TIPS_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(TIPS_FILE, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"] >= pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()].copy()
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
    df, _ = V.board_rows(df)              # owner serving rules 2026-11-08 (hidden markets, GK)
    return df


df = load_tips()
_state_chip = ui.status_chip("PAPER", label="PAPER · RESEARCH")
if df.empty:
    ui.page_header("Player Props", _SUB, eyebrow_text=_EYEBROW, freshness=_state_chip)
    ui.empty_state("NO_DATA", "⏳ No upcoming player-prop signals yet.")
    st.stop()

ui.page_header("Player Props", _SUB, eyebrow_text=_EYEBROW,
               freshness=_state_chip + " " + ui.status_chip(
                   "NO_DATA", tone="NEUTRAL",
                   label=f"{df['match'].nunique()} fixtures · {df['player_name'].nunique()} players upcoming"))

priced = df[df["_priced"]]
unpriced = df[~df["_priced"]]

# ═══ STATUS — the headline counts ═══════════════════════════════════════════════════════════════
_beats = priced[priced["model_prob"] > 1 / priced["market_odds"]]
ui.metric_grid([
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
], per_row=2)
st.caption(
    "Why so many are unpriced: bookmakers price **anytime scorer** deeply but rarely offer "
    "**1+ shot on target**, which is where most of the model's strongest signals sit. Markets "
    "deepen toward kickoff, so prices appear over the last 1–3 days.")
st.caption(V.hidden_markets_note())

# ═══ Filters — market visible, the rest under Advanced filters ═══════════════════════════════════
_mk = [m for m in MARKET_LABEL if m in set(df["market"])] + \
      sorted(set(df["market"]) - set(MARKET_LABEL))
mkt_filter = st.multiselect("Market", _mk, default=_mk,
                            format_func=lambda m: MARKET_LABEL.get(m, m), key="pp_mkt")
with st.expander("Advanced filters"):
    f2, f3 = st.columns([2, 1])
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


# ═══ RESEARCH — where the model sits furthest above the price ════════════════════════════════════
p = _filt(priced)
ui.section_header("Model furthest above the price", "Priced props only, after the filters. Paper research: a "
                  "gap to the price is not an edge — props have been tested eight ways and none held.")
top = V.top_priced(p)
if top.empty:
    ui.empty_state("NO_DATA", "No priced props match the filters. Prices usually appear 1–3 days before kickoff.")
else:
    for col, (_, r) in zip(st.columns(3), top.iterrows()):
        V.prop_card(r, container=col)

# ═══ DETAIL — the day boards ═════════════════════════════════════════════════════════════════════
ui.section_header("All upcoming props")
tab_p, tab_u = st.tabs([f"Priced by bookmakers ({len(priced)})",
                        f"Model only — no price yet ({len(unpriced)})"])
with tab_p:
    if p.empty:
        ui.empty_state("NO_DATA", "No priced props match the filters. Prices usually appear 1–3 days before kickoff.")
    else:
        V.render_board(p, True, limit=60)
        st.caption("**Edge** = model probability minus the bookmaker's implied probability, in "
                   "percentage points; positive means the model rates the player higher than the price "
                   "does. Bookmaker prices include their margin. **Fair price** says where the fair "
                   "probability behind EV and tier came from: *two-sided price* (both sides of one "
                   "bookmaker's bet, de-vigged) or *margin assumed* (one side quoted, margin assumed). "
                   "Model = the model's own probability. Kickoff in Israel time. Paper only.")
with tab_u:
    u = _filt(unpriced)
    if u.empty:
        ui.empty_state("NO_DATA", "Nothing unpriced matches the filters.")
    else:
        V.render_board(u, False, limit=25)
        st.caption("Strongest model signals per day (top 25). **Fair odds** = 1 / model probability "
                   "— compare with your own bookmaker. No bookmaker price yet. Kickoff in Israel time. Paper only.")

with st.expander("Raw data — every upcoming prop"):
    show_cols = ["date", "kickoff_utc", "league", "match", "player_name", "position", "team",
                 "market", "model_prob", "fair_odds", "market_odds", "fair_basis", "ev", "tier",
                 "confidence", "n_games", "model_prob_hand_adjusted", "prob_adjustment"]
    ui.compact_table(df.sort_values("model_prob", ascending=False)[[c for c in show_cols if c in df.columns]],
                     V.RAW_CONFIG)

# ═══ PERFORMANCE — settled record (rebuilt 2026-11-09, see dashboard_data/props_view.py) ═══════════════════
ui.section_header("Results — settled player-prop tips",
                  f"Every tip tier ({tier_label('VALUABLE')} included), flat 1u paper stakes, for the period you "
                  "pick. ROI and "
                  "profit use only tips with a real bookmaker price; a small sample is marked.")


@st.cache_data(ttl=120)
def load_results():
    if not LEDGER_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(LEDGER_FILE, low_memory=False)


led = load_results()
if led.empty or "result" not in led.columns or "tier" not in led.columns:
    ui.empty_state("NO_DATA", "No prop results yet (`output/player_ledger.csv`).")
else:
    import config as _cfg  # noqa: E402
    clean_from = getattr(_cfg, "PROPS_CLEAN_MODELS_LIVE_DATE", None)
    allr = V.prepare_ledger(led)
    allr["era"] = V.model_era(allr, clean_from)
    period = st.radio("Period", list(V.PERIODS), index=V.PERIODS.index(V.DEFAULT_PERIOD), horizontal=True,
                      key="pp_results_period")
    today = pd.Timestamp.now(tz="UTC").tz_convert(None).normalize()
    d = V.in_period(allr, period, today)
    start = V.period_start(period, today)
    ui.caption(f"{period}: matches from {start:%d %b %Y} on" if start is not None else "All time")
    if clean_from is None:
        ui.caption("Every result here comes from the **pre-repair props models**. The clean retrained props "
                   "models go live at go-live; their results start the Clean Record and will be shown apart.")
    if V.tips(d).empty:
        ui.empty_state("NO_DATA", f"No prop tips with a match date in this period ({period.lower()}).")
    else:
        for era in [e for e in ("clean models", "pre-repair models") if (d["era"] == e).any()]:
            de = d[d["era"] == era]
            tt = V.tier_table(de)
            if tt.empty:
                continue
            tot = tt[tt["Tier"] == "All tips"].iloc[0]
            ui.metric_card("Flat ROI (1u, paper)", ui.fmt_roi(tot["ROI %"] / 100) if tot["Priced"] else "—",
                           title=f"All tips · {era} · {period.lower()}",
                           supporting=[("Settled", f"{int(tot['Settled'])}"),
                                       ("Record", f"{int(tot['Wins'])}W – {int(tot['Settled'] - tot['Wins'])}L"),
                                       ("Profit", ui.fmt_units(tot["Profit (u)"]) if tot["Priced"] else "—"),
                                       ("Void / pending", f"{int(tot['Void'])} / {int(tot['Pending'])}")],
                           status=ui.status_chip("PAPER", label="PAPER · NEVER STAKED")
                           + (" " + ui.sample_marker(int(tot["Priced"])) if ui.small_sample(int(tot["Priced"])) else ""),
                           n=int(tot["Priced"]))
            # dashtruth 2026-11-05: rows a correction sidecar flags are IN this record; say how many
            _st = V.tips(de)[V.tips(de)["R"].isin(V.SETTLED)]
            _fl = {k: v for k, v in V.flagged(_st, BASE_DIR / "output" / "corrections").items() if v}
            if _fl:
                ui.caption("Includes rows flagged in output/corrections (kept, not removed): " + "; ".join(
                    f"{v} of {len(_st)} in {k} ({V.FLAG_FILES[k]})" for k, v in _fl.items()) + ".")
            ui.eyebrow(f"By tier · {era}")
            ui.compact_table(tt, V.TIER_CONFIG)
            ui.eyebrow(f"By market · {era}")
            ui.compact_table(V.market_table(de), V.TIER_CONFIG)
            ui.eyebrow(f"Month by month · {era}")
            ui.compact_table(V.month_table(de), V.TIER_CONFIG)
        n_avoid = V.avoid_count(d)
        if n_avoid:
            ui.caption(f"Logged, not tipped: {n_avoid:,} AVOID rows in this period are recorded for research and "
                       "are not part of the tip record above.")
        with st.expander("Settled tips — newest first"):
            sett = V.tips(d)[V.tips(d)["R"].isin(V.SETTLED)]
            cols = [c for c in ["match_date", "league", "player_name", "market", "tier",
                                "market_odds", "result", "pnl", "era"] if c in sett.columns]
            ui.compact_table(sett.sort_values("match_date", ascending=False)[cols].head(300), V.RESULTS_CONFIG)

st.caption("Player props are paper research only — the model is accurate but has shown no betting "
           "edge against the market.")

# ── Disclaimer & Terms (shown on every dashboard page) ──
from utils.disclaimer import disclaimer_footer  # noqa: E402
disclaimer_footer()
