"""Portfolio Manager — week-by-week bankroll simulation, one 1000U bankroll PER MARKET.

Tests the staking strategy (tier-based fractional Kelly) market-by-market on the
real ledgers, so you can watch each equity curve and tune the rules live.

Markets (separate 1000U each): Standard O/U · New-Format · Player Props · Side/BTTS.
Tier sizing default: SNIPER = 1/8 Kelly, MARKSMAN = 1/16 Kelly, VALUABLE = paper (0).

Read-only / public-safe: pure pandas on committed CSVs, no local dependencies.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

st.set_page_config(page_title="Portfolio | Wowza", page_icon="💼", layout="wide")

OUT = config.OUTPUT_DIR
NF_LEAGUES = set(getattr(config, "NEW_FORMAT_LEAGUES", []))
STD_LEAGUES = {"Championship", "League One", "League Two",
               "Bundesliga 2", "Ligue 2", "La Liga 2", "Serie B"}


# ── Data loading ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_main_ledger() -> pd.DataFrame:
    f = OUT / "bets_ledger.csv"
    if not f.exists():
        return pd.DataFrame()
    d = pd.read_csv(f, low_memory=False)
    d["match_date"] = pd.to_datetime(d["match_date"], errors="coerce")
    return d


@st.cache_data(ttl=60)
def load_player_ledger() -> pd.DataFrame:
    f = OUT / "player_ledger.csv"
    if not f.exists():
        return pd.DataFrame()
    d = pd.read_csv(f, low_memory=False)
    d["match_date"] = pd.to_datetime(d["match_date"], errors="coerce")
    return d


def _market(league: str) -> str:
    if league in NF_LEAGUES:
        return "New-Format"
    if league in STD_LEAGUES:
        return "Standard O/U"
    return "Other"


def normalize(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Return a uniform frame: match_date, odds, edge, tier, result, market."""
    if df.empty:
        return df
    out = pd.DataFrame()
    out["match_date"] = df["match_date"]
    if kind == "main":
        out["odds"] = pd.to_numeric(df["odds"], errors="coerce")
        out["edge"] = pd.to_numeric(df.get("edge_pct", 0), errors="coerce") / 100.0
        out["tier"] = df.get("signal_tier", "").astype(str).str.upper()
        out["result"] = df.get("result", "").astype(str).str.upper()
        out["market"] = df["league"].map(_market)
        out["source"] = df.get("source", "live").astype(str)
    else:  # player props
        odds = pd.to_numeric(df["market_odds"], errors="coerce")
        mp = pd.to_numeric(df.get("model_prob", np.nan), errors="coerce")
        out["odds"] = odds
        # edge = model_prob - implied (fallback to ev if model_prob missing)
        out["edge"] = (mp - 1.0 / odds).fillna(pd.to_numeric(df.get("ev", 0), errors="coerce"))
        out["tier"] = df.get("tier", "").astype(str).str.upper()
        out["result"] = df.get("result", "").astype(str).str.upper()
        out["market"] = "Player Props"
        out["source"] = "live"
    return out


# ── The week-by-week simulation ───────────────────────────────────────────────────
def simulate(bets: pd.DataFrame, start_bank: float, tier_frac: dict,
             cap_pct: float, mode: str):
    """Walk settled bets week by week. Stakes sized off the bankroll at each week's
    start (no intra-week compounding). Returns (equity_df, stats) or None."""
    b = bets.dropna(subset=["match_date", "odds"]).copy()
    b = b[b["result"].isin(["WIN", "LOSS"])]          # drop VOID/unsettled
    b = b[b["odds"] > 1.0]
    if b.empty:
        return None
    b["week"] = b["match_date"].dt.to_period("W").apply(lambda p: p.start_time)

    bank = float(start_bank)
    peak = bank
    rows = []
    for wk, g in b.sort_values("match_date").groupby("week"):
        wk_open = bank
        staked = pnl = 0.0
        n_real = 0
        for _, bt in g.iterrows():
            frac = tier_frac.get(bt["tier"], 0.0)
            if frac <= 0:
                continue                               # paper tier → skip
            odds = float(bt["odds"])
            if mode == "Fractional Kelly":
                fk = max(float(bt["edge"]) / (odds - 1.0), 0.0)
                stake_frac = min(frac * fk, cap_pct / 100.0)
            else:                                      # Flat % per tier
                stake_frac = min(frac, cap_pct / 100.0)
            stake = wk_open * stake_frac
            if stake <= 0:
                continue
            staked += stake
            pnl += stake * (odds - 1.0) if bt["result"] == "WIN" else -stake
            n_real += 1
        bank += pnl
        peak = max(peak, bank)
        rows.append({"week": wk, "bets": n_real, "staked": staked,
                     "pnl": pnl, "bank": bank, "drawdown": (bank - peak) / peak})
    if not rows:
        return None
    eq = pd.DataFrame(rows)

    settled = b[b["tier"].map(lambda t: tier_frac.get(t, 0) > 0)]
    wins = (settled["result"] == "WIN").sum()
    n = len(settled)
    total_staked = eq["staked"].sum()
    stats = {
        "final": bank,
        "roi": (bank - start_bank) / start_bank,
        "yield": (bank - start_bank) / total_staked if total_staked else 0.0,
        "n_bets": int(n),
        "win_pct": wins / n if n else 0.0,
        "best_week": eq["pnl"].max(),
        "worst_week": eq["pnl"].min(),
        "max_dd": eq["drawdown"].min(),
        "weeks": len(eq),
    }
    return eq, stats


# ── UI ────────────────────────────────────────────────────────────────────────────
st.markdown("## 💼 Portfolio Manager")
st.caption("Separate 1000U bankroll per market · week-by-week · real ledgers")

with st.sidebar:
    st.markdown("### ⚙️ Strategy settings")
    start_bank = st.number_input("Starting bankroll (U) per market", 100, 100000, 1000, 100)
    mode = st.radio("Staking", ["Fractional Kelly", "Flat % per tier"], index=0)
    st.markdown("**Per-tier size** " +
                ("(× full Kelly)" if mode == "Fractional Kelly" else "(flat % of bank)"))
    sniper = st.slider("SNIPER", 0.0, 0.5, 0.125, 0.005,
                       help="1/8 Kelly = 0.125")
    marksman = st.slider("MARKSMAN", 0.0, 0.5, 0.0625, 0.005,
                         help="1/16 Kelly = 0.0625")
    valuable = st.slider("VALUABLE", 0.0, 0.5, 0.0, 0.005,
                         help="paper / data-only by default = 0")
    cap_pct = st.slider("Max stake cap (% of bank)", 0.5, 10.0, 5.0, 0.5)
    src = st.radio("Data source", ["Live only", "Backtest (⚠ inflated)", "Live + Backtest"], index=0)
    discount = st.checkbox("Reality-discount edges (÷5)", value=False,
                           help="Backtest edges run ~5× hot vs live — discount to approximate reality.")

tier_frac = {"SNIPER": sniper, "MARKSMAN": marksman, "VALUABLE": valuable}

# Load + normalize
main = normalize(load_main_ledger(), "main")
players = normalize(load_player_ledger(), "player")
allbets = pd.concat([main, players], ignore_index=True) if not main.empty else players

if allbets.empty:
    st.warning("No ledger data found.")
    st.stop()

# Source filter (player ledger is all live)
if src == "Live only":
    allbets = allbets[allbets["source"] == "live"]
elif src.startswith("Backtest"):
    allbets = allbets[allbets["source"] == "backtest"]
if discount:
    allbets = allbets.copy()
    allbets["edge"] = allbets["edge"] / 5.0

# Honesty banner
n_live = (allbets["result"].isin(["WIN", "LOSS"])).sum()
st.info(
    f"**{n_live} settled bets** in this view. ⚠️ Live samples are thin "
    f"(Standard ~17, New-format ~56, Props ~247 mostly losing/void, Side/BTTS = none yet). "
    "Treat curves as illustrative, not proof. Backtest data runs ~5× hot — use the discount toggle."
)

MARKETS = ["Standard O/U", "New-Format", "Player Props", "Side/BTTS"]
tabs = st.tabs([f"📈 {m}" for m in MARKETS])

for tab, mkt in zip(tabs, MARKETS):
    with tab:
        if mkt == "Side/BTTS":
            st.warning("No Side/BTTS bets logged yet. This track populates after the "
                       "July BTTS odds backfill + live logging.")
            continue
        sub = allbets[allbets["market"] == mkt]
        res = simulate(sub, start_bank, tier_frac, cap_pct, mode)
        if res is None:
            st.warning(f"No settled bets for {mkt} under the current tier/source filters.")
            continue
        eq, s = res

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final bankroll", f"{s['final']:,.0f}U", f"{s['roi']*100:+.1f}%")
        c2.metric("Bets / Win%", f"{s['n_bets']}", f"{s['win_pct']*100:.0f}% win")
        c3.metric("Best / Worst week", f"+{s['best_week']:,.0f}U", f"{s['worst_week']:,.0f}U")
        c4.metric("Max drawdown", f"{s['max_dd']*100:.1f}%", f"{s['weeks']} wks")

        curve = eq.set_index("week")["bank"]
        curve.loc[curve.index.min() - pd.Timedelta(days=7)] = start_bank  # seed start point
        st.line_chart(curve.sort_index(), height=240)

        st.dataframe(
            eq.assign(
                week=eq["week"].dt.strftime("%Y-%m-%d"),
                staked=eq["staked"].round(1),
                pnl=eq["pnl"].round(1),
                bank=eq["bank"].round(0),
                drawdown=(eq["drawdown"] * 100).round(1),
            )[["week", "bets", "staked", "pnl", "bank", "drawdown"]],
            use_container_width=True, hide_index=True,
        )

st.caption("Sizing recomputed off bank at each week's open. Kelly f* = edge ÷ (odds−1), "
           "× tier fraction, capped. VALUABLE defaults to paper. "
           "Edge is unproven live (n small) — this is a testing tool, not a guarantee.")
