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
from src.tier_labels import STAKED_LABEL

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
                   "does. Bookmaker prices include their margin. Kickoff in Israel time. Paper only.")
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
                 "market", "model_prob", "fair_odds", "market_odds", "ev", "tier", "confidence",
                 "n_games"]
    ui.compact_table(df.sort_values("model_prob", ascending=False)[[c for c in show_cols if c in df.columns]],
                     V.RAW_CONFIG)

# ═══ PERFORMANCE — settled record ════════════════════════════════════════════════════════════════
ui.section_header("Results — settled player-prop signals", f"{STAKED_LABEL} only, flat 1u paper stakes. A rate is "
                  "shown with its n; a small n is marked.")


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
    ui.empty_state("NO_DATA", f"No {STAKED_LABEL} prop results yet (`output/player_ledger.csv`).")
else:
    settled = led[led["result"].astype(str).str.upper().isin(["WIN", "LOSS"])].copy()
    if settled.empty:
        ui.empty_state("NO_DATA", "No settled (WIN/LOSS) signals yet — pending/void only.")
    else:
        wins = int((settled["result"].astype(str).str.upper() == "WIN").sum())
        n = len(settled)
        roi = (pd.to_numeric(settled["pnl"], errors="coerce").mean() * 100
               if "pnl" in settled.columns else float("nan"))
        left, right = st.columns([1, 2])
        ui.metric_card("Win rate", f"{wins / n * 100:.0f}%", title="Settled record · paper",
                       supporting=[("Settled", f"{n}"), ("Record", f"{wins}W – {n - wins}L"),
                                   ("Flat ROI (1u, paper)", f"{roi:+.1f}%" if roi == roi else "—")],
                       status=ui.status_chip("PAPER", label="PAPER · NEVER STAKED"), n=n, container=left)
        with right:
            cols = [c for c in ["match_date", "league", "player_name", "market", "tier",
                                "market_odds", "result", "pnl"] if c in settled.columns]
            ui.compact_table(settled.sort_values("match_date", ascending=False)[cols].head(150), V.RESULTS_CONFIG)

st.caption("Player props are paper research only — the model is accurate but has shown no betting "
           "edge against the market.")

# ── Disclaimer & Terms (shown on every dashboard page) ──
from utils.disclaimer import disclaimer_footer  # noqa: E402
disclaimer_footer()
