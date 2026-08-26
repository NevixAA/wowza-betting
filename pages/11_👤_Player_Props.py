"""
Player Props Dashboard
Shows SNIPER/MARKSMAN/VALUABLE player prop signals.
When odds aren't available yet, shows WC/PROP_LEAGUE signals by model probability.
"""
import textwrap
from pathlib import Path

import pandas as pd
import streamlit as st

import dashboard_ui as ui

st.set_page_config(page_title="Player Props | Wowza", page_icon="👤", layout="wide")
# Was components.v1.html with a JS reload: an API whose announced removal date
# (2026-06-01) has passed, and which reloaded the whole browser tab and so DISCARDED
# every filter the user had set.
ui.autorefresh(minutes=2, key="11_player_props_refresh")
BASE_DIR   = Path(__file__).resolve().parents[1]
TIPS_FILE  = BASE_DIR / "output" / "player_tips.csv"

st.title("👤 Player Props")
st.caption("SNIPER/MARKSMAN/VALUABLE signals + WC2026 model signals — Goals (1+/2+) · SOT (1-3+) · Assists")

if st.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()

TIER_META = {
    "SNIPER":   ("🎯", "#e94560", "EV > 40% · rel_edge ≥ 20% · 2+ lazy factors"),
    "MARKSMAN": ("🔫", "#00aaff", "EV > 25% · rel_edge ≥ 12% · 1+ lazy factor"),
    "VALUABLE": ("💎", "#f5a623", "EV > 15% · rel_edge > 0%"),
    "WATCH":    ("👁",  "#4caf50", "Model signal · add odds to compute EV"),
}

MARKET_EMOJI = {
    "goals": "⚽", "goals2": "⚽⚽",
    "assists": "🎯",
    "sot": "🔫", "sot2": "🔫", "sot3": "🔫",
    "cards": "🟨",
}
MARKET_LABEL = {
    "goals": "Anytime Scorer", "goals2": "Score 2+",
    "assists": "Assist",
    "sot": "SOT 1+", "sot2": "SOT 2+", "sot3": "SOT 3+",
    "cards": "Yellow Card",
}

@st.cache_data(ttl=120)
def load_tips():
    if not TIPS_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(TIPS_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    today = pd.Timestamp.now().normalize()
    return df[df["date"] >= today].copy()

df = load_tips()

if df.empty:
    st.info("⏳ No player prop signals yet. The player props model runs 30 min after each predict.")
    st.stop()

# ── Split: priced signals vs model-only signals ────────────────────────────────
# SPLIT ON THE PRICE, not on a tier name. This was `df["tier"] != "WATCH"`, and "WATCH" is not a
# tier this pipeline ever emits — the values are AVOID / PAPER / VALUABLE. So has_odds captured
# EVERY row and no_odds was always empty, which is why the priced/unpriced distinction never
# appeared and the page looked broken.
#
# The distinction is the whole point, per invariant 13: an unpriced prop is AVOID **by
# construction** because `enrich_with_odds` does `if not mkt_odds: continue`, so no edge is ever
# computed for it. Reading those AVOIDs as model rejections is wrong, and on 2026-08-16 only 5 of
# 213 scored players had a price at all.
_mo = pd.to_numeric(df.get("market_odds"), errors="coerce")
df["_priced"] = _mo.notna() & (_mo > 1.0)
has_odds  = df[df["_priced"]].copy()
no_odds   = df[~df["_priced"]].copy()

# Say it before anything else, so an empty table is never mistaken for a broken page.
_c1, _c2, _c3 = st.columns(3)
_c1.metric("Prop rows upcoming", f"{len(df):,}")
_c2.metric("Priced by the market", f"{len(has_odds):,}",
           help="Only these can have an edge computed at all.")
_c3.metric("Never priced", f"{len(no_odds):,}",
           help="AVOID by construction, not a model rejection (invariant 13).")
if has_odds.empty:
    st.warning(
        f"**No prop is priced right now** — all {len(df):,} upcoming rows are unpriced, so no edge "
        f"can be computed for any of them. This is a COVERAGE state, not a model opinion. Coverage "
        f"is per FIXTURE rather than per league and deepens toward kickoff, so an empty board days "
        f"out is normal. See output/prop_odds_coverage.json for what the bookmakers actually "
        f"returned per league.")
elif len(no_odds):
    st.caption(
        f"{100*len(no_odds)/max(1,len(df)):.0f}% of upcoming props carry no price. Those rows are "
        f"AVOID by construction and are excluded from everything below — they are not rejections.")

# WC / strong model signals (≥ 55% for PROP_LEAGUE, ≥ 60% for team-model matches)
wc_signals = no_odds[
    ((no_odds["match_tier"] == "PROP_LEAGUE") & (no_odds["model_prob"] >= 0.55)) |
    ((no_odds["match_tier"].isin(["SNIPER","MARKSMAN"])) & (no_odds["model_prob"] >= 0.60))
].copy()

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Priced signals", len(has_odds[has_odds["tier"] != "WATCH"]))
c2.metric("🎯 SNIPER",   len(has_odds[has_odds["tier"] == "SNIPER"]))
c3.metric("🔫 MARKSMAN", len(has_odds[has_odds["tier"] == "MARKSMAN"]))
c4.metric("💎 VALUABLE", len(has_odds[has_odds["tier"] == "VALUABLE"]))
c5.metric("🌍 WC signals", len(wc_signals[wc_signals["match_tier"] == "PROP_LEAGUE"]))

st.markdown("---")


def _render_signal(row, *, show_tier_label=True):
    """Render a single signal card."""
    tier      = row.get("tier", "WATCH")
    emoji, color, desc = TIER_META.get(tier, ("📌","#888",""))
    mkt_emoji = MARKET_EMOJI.get(row["market"], "📊")
    fair_odds = row.get("fair_odds", "—")
    mkt_odds  = row.get("market_odds")
    ev_val    = row.get("ev")
    conf      = float(row.get("confidence", 0))
    lazy      = str(row.get("lazy_factors", ""))
    n_games   = int(row.get("n_games", 0))
    match_tier = row.get("match_tier", "")

    ev_str   = f"EV: {ev_val:+.1%}" if ev_val is not None and not pd.isna(ev_val) else "EV: add odds"
    odds_str = f"@ {mkt_odds:.2f}"  if mkt_odds and not pd.isna(mkt_odds) else f"fair {fair_odds}"
    tier_label = f"{emoji} {tier}" if show_tier_label else f"👁 {match_tier}"
    lazy_badges = " ".join(
        f'<span style="background:#333;color:#aaa;padding:1px 6px;border-radius:8px;font-size:0.75em">{f}</span>'
        for f in lazy.split("|") if f
    )
    ges_str = ""
    ges_val = row.get("ges")
    if ges_val is not None and not pd.isna(ges_val):
        ges_str = f" · GES: {float(ges_val):.2f}"

    st.markdown(textwrap.dedent(f"""
    <div style="border-left:4px solid {color};padding:12px 16px;margin:6px 0;
                background:#111827;border-radius:6px">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="color:{color};font-weight:bold">{tier_label}</span>
            <span style="color:#aaa;font-size:0.85em">📅 {str(row['date'])[:10]}</span>
        </div>
        <b style="color:white;font-size:1.05em">{row['player_name']}</b>
        <span style="color:#aaa"> · {row['position']} · {row['team']}</span><br/>
        <span style="color:#90caf9">{row['league']} — {row['match']}</span><br/>
        <div style="margin:6px 0">
            <span style="color:white">{mkt_emoji} <b>{MARKET_LABEL.get(row['market'], row['market'].upper())}</b></span>
            &nbsp;{odds_str}
            &nbsp;|&nbsp;<span style="color:#00cc88"><b>{ev_str}</b></span>
            &nbsp;|&nbsp;Model P: <b style="color:white">{row['model_prob']:.0%}</b>
            {ges_str}
        </div>
        <div>{lazy_badges}</div>
        <div style="color:#555;font-size:0.8em;margin-top:4px">
            Confidence: {conf:.0%} · Data: {n_games} games · {row.get('data_source','')}
        </div>
    </div>
    """), unsafe_allow_html=True)


# ── WC2026 section (always shown when WC signals exist) ───────────────────────
if not wc_signals.empty:
    st.subheader(f"🌍 World Cup 2026 — {len(wc_signals)} signal(s)")
    st.caption("Model probability signals — no odds plugged in yet. Check your bookmaker for props.")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        mkt_wc = st.multiselect("Market (WC)", sorted(wc_signals["market"].unique()),
                                default=list(wc_signals["market"].unique()),
                                key="mkt_wc")
    with col_f2:
        min_p_wc = st.slider("Min model prob (WC)", 0.50, 0.90, 0.55, 0.01, key="mp_wc")

    wc_filtered = wc_signals[
        wc_signals["market"].isin(mkt_wc) &
        (wc_signals["model_prob"] >= min_p_wc)
    ].sort_values("model_prob", ascending=False)

    for _, row in wc_filtered.iterrows():
        _render_signal(row, show_tier_label=False)

    st.markdown("---")


# ── Priced signals section (when odds are available) ──────────────────────────
if not has_odds.empty:
    st.subheader("📊 Priced Signals")

    col1, col2, col3 = st.columns(3)
    with col1:
        # Options and defaults come from the DATA, not a hardcoded list. The default was
        # ["SNIPER","MARKSMAN"] — tiers the props pipeline never emits (it produces PAPER /
        # VALUABLE / AVOID), so the table was empty even when priced signals existed.
        _tiers_present = [x for x in ["SNIPER", "MARKSMAN", "VALUABLE", "PAPER"]
                          if x in set(has_odds["tier"].dropna())]
        tier_filter = st.multiselect(
            "Tier", _tiers_present or sorted(has_odds["tier"].dropna().unique()),
            default=_tiers_present or None,
            help="Props are PAPER-only permanently (invariant 2): the model is accurate but has no "
                 "betting edge, so these are research signals, never staked.")
    with col2:
        mkt_filter = st.multiselect("Market", sorted(has_odds["market"].unique()),
                                    default=list(has_odds["market"].unique()))
    with col3:
        min_confidence = st.slider("Min confidence", 0.0, 1.0, 0.50, 0.05)

    col4, col5 = st.columns([2, 1])
    with col4:
        all_leagues = sorted(has_odds["league"].dropna().unique().tolist()) if "league" in has_odds.columns else []
        league_filter = st.multiselect("League", all_leagues, default=all_leagues, key="pp_league")
    with col5:
        date_range = st.date_input(
            "Date range",
            value=(has_odds["date"].min().date(), has_odds["date"].max().date()),
            key="pp_dates",
        )
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            date_from, date_to = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        else:
            date_from = date_to = pd.Timestamp(date_range[0]) if date_range else has_odds["date"].min()

    filtered = has_odds[
        has_odds["tier"].isin(tier_filter) &
        has_odds["market"].isin(mkt_filter) &
        (has_odds["confidence"] >= min_confidence) &
        (has_odds["league"].isin(league_filter) if league_filter and "league" in has_odds.columns else True) &
        (has_odds["date"] >= date_from) &
        (has_odds["date"] <= date_to + pd.Timedelta(days=1))
    ].sort_values(["tier", "model_prob"], ascending=[True, False])

    st.subheader(f"📋 {len(filtered)} signal(s)")
    for _, row in filtered.iterrows():
        _render_signal(row)

elif wc_signals.empty:
    st.info("No signals with odds yet. Add odds via `enrich_with_odds()` or wait for the next odds update.")

st.markdown("---")
with st.expander("📊 Raw data — all today's signals"):
    show_cols = ["date","league","match","match_tier","player_name","position","team",
                 "market","model_prob","fair_odds","market_odds","ev",
                 "tier","confidence","lazy_factors","n_games","ges"]
    show_df = df.sort_values("model_prob", ascending=False)
    st.dataframe(show_df[[c for c in show_cols if c in show_df.columns]],
                 width="stretch", hide_index=True)

# ── Results / track record (settled bets from the ledger) ──────────────────────
st.markdown("---")
st.subheader("📒 Results — settled player-prop bets")

LEDGER_FILE = BASE_DIR / "output" / "player_ledger.csv"


@st.cache_data(ttl=120)
def load_results():
    if not LEDGER_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(LEDGER_FILE, low_memory=False)


led = load_results()
# VALUABLE / WATCH are NOT signals (never sent; ~-8% ROI) — the all-time track record
# counts SNIPER + MARKSMAN only. (The raw ledger still logs all tiers for data.)
if not led.empty and "tier" in led.columns:
    led = led[led["tier"].astype(str).str.upper().isin(["SNIPER", "MARKSMAN"])].copy()
if led.empty or "result" not in led.columns:
    st.info("No SNIPER/MARKSMAN results yet (`output/player_ledger.csv`). Populate as graded signals resolve.")
else:
    settled = led[led["result"].astype(str).str.upper().isin(["WIN", "LOSS"])].copy()
    if settled.empty:
        st.info("No settled (WIN/LOSS) bets yet — pending/void only.")
    else:
        wins = int((settled["result"].astype(str).str.upper() == "WIN").sum())
        n = len(settled)
        roi = (pd.to_numeric(settled["pnl"], errors="coerce").mean() * 100
               if "pnl" in settled.columns else float("nan"))
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Settled bets", n)
        r2.metric("Record", f"{wins}W – {n - wins}L")
        r3.metric("Win rate", f"{wins / n * 100:.0f}%")
        r4.metric("Flat ROI (1u)", f"{roi:+.1f}%" if roi == roi else "—")
        st.caption("⚠️ Live props so far are mostly World Cup longshots — small, net-negative sample. "
                   "Treat as data, not proof.")
        cols = [c for c in ["match_date", "league", "player_name", "market", "tier",
                            "market_odds", "result", "pnl"] if c in settled.columns]
        st.dataframe(
            settled.sort_values("match_date", ascending=False)[cols].head(150),
            width="stretch", hide_index=True,
        )

st.caption("Player props · 9 markets · API-Football rolling stats · WC2026 national team data")


# ── Disclaimer & Terms (shown on every dashboard page) ──
from utils.disclaimer import disclaimer_footer  # noqa: E402
disclaimer_footer()
