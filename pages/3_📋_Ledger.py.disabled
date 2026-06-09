"""Full bet history ledger with filters and export."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from utils.access import is_public

st.set_page_config(page_title="Ledger | Wowza", page_icon="📋", layout="wide")
st_autorefresh(interval=5 * 60 * 1000, key="ledger_refresh")


@st.cache_data(ttl=30)
def load_ledger():
    f = config.OUTPUT_DIR / "bets_ledger.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f, dtype=str)
    if "source" not in df.columns:
        df["source"] = "live"
    else:
        df["source"] = df["source"].fillna("live")
    df["pnl"]        = pd.to_numeric(df.get("pnl", pd.Series()), errors="coerce")
    df["edge_pct"]   = pd.to_numeric(df.get("edge_pct", pd.Series()), errors="coerce")
    df["odds"]       = pd.to_numeric(df.get("odds", pd.Series()), errors="coerce")
    df["match_date"] = pd.to_datetime(df.get("match_date", pd.Series()), errors="coerce")
    return df


st.markdown("## 📋 Bet Ledger")

df = load_ledger()
if df.empty:
    st.warning("No ledger data yet.")
    st.stop()

# ── Summary stats ──────────────────────────────────────────────────────────────
live_scored     = df[(df["source"] == "live") & df["pnl"].notna()]
backtest_scored = df[(df["source"] == "backtest") & df["pnl"].notna()]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total entries", len(df))
c2.metric("Live bets",     len(df[df["source"] == "live"]))
c3.metric("Backtest bets", len(df[df["source"] == "backtest"]))
c4.metric("Pending results",
          len(df[(df["source"] == "live") & df["pnl"].isna() & df["result"].isna()
                 if "result" in df.columns else pd.Series([], dtype=bool)]))

st.divider()

# ── Filters ────────────────────────────────────────────────────────────────────
st.markdown("### Filters")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    source_f = st.multiselect("Source", ["live", "backtest"], default=["live", "backtest"])
with col2:
    tier_f = st.multiselect("Tier", ["SNIPER", "MARKSMAN", "VALUABLE"], default=["SNIPER", "MARKSMAN", "VALUABLE"])
with col3:
    leagues = sorted(df["league"].dropna().unique().tolist()) if "league" in df else []
    league_f = st.multiselect("League", leagues, default=leagues)
with col4:
    result_opts = ["WIN", "LOSS", "VOID", "(pending)"]
    result_f = st.multiselect("Result", result_opts, default=result_opts)
with col5:
    if df["match_date"].notna().any():
        min_d = df["match_date"].min().date()
        max_d = df["match_date"].max().date()
        date_range = st.date_input("Date range", value=(min_d, max_d),
                                   min_value=min_d, max_value=max_d)
    else:
        date_range = None

# Apply filters
filtered = df[
    df["source"].isin(source_f) &
    df["signal_tier"].isin(tier_f) &
    df["league"].isin(league_f)
].copy()

if date_range and len(date_range) == 2:
    filtered = filtered[
        (filtered["match_date"].dt.date >= date_range[0]) &
        (filtered["match_date"].dt.date <= date_range[1])
    ]

# Result filter
if "(pending)" in result_f:
    pending_mask = filtered["pnl"].isna()
else:
    pending_mask = pd.Series(False, index=filtered.index)

result_vals = [r for r in result_f if r != "(pending)"]
result_mask = filtered.get("result", pd.Series("", index=filtered.index)).isin(result_vals)
filtered = filtered[result_mask | pending_mask]

st.caption(f"Showing {len(filtered)} of {len(df)} entries")

# ── Table ──────────────────────────────────────────────────────────────────────
display_cols = [c for c in [
    "source", "match_date", "league", "home_team", "away_team",
    "side", "odds", "edge_pct", "signal_tier", "drift_signal",
    "model_type", "result", "pnl", "clv_pct", "notes"
] if c in filtered.columns]

# Color-code result
def style_result(val):
    if val == "WIN":
        return "color: #00c896; font-weight: bold"
    elif val == "LOSS":
        return "color: #e94560; font-weight: bold"
    return ""

def style_pnl(val):
    try:
        v = float(val)
        return f"color: {'#00c896' if v > 0 else '#e94560'}; font-weight: bold"
    except:
        return ""

styled = filtered[display_cols].sort_values("match_date", ascending=False)

st.dataframe(
    styled,
    use_container_width=True,
    hide_index=True,
    column_config={
        "match_date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
        "odds":       st.column_config.NumberColumn("Odds", format="%.2f"),
        "edge_pct":   st.column_config.NumberColumn("Edge %", format="%.1f"),
        "pnl":        st.column_config.NumberColumn("PnL", format="%+.2f"),
        "clv_pct":    st.column_config.NumberColumn("CLV %", format="%.1f"),
        "source":     st.column_config.TextColumn("Source"),
    },
    height=500,
)

# ── Export ─────────────────────────────────────────────────────────────────────
st.divider()
col_exp1, col_exp2 = st.columns([2, 8])
with col_exp1:
    csv_data = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Export to CSV",
        data=csv_data,
        file_name="wowza_ledger_export.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ── Pending results section ────────────────────────────────────────────────────
pending = df[(df["source"] == "live") & df["pnl"].isna()]
pending = pending[pending["match_date"].notna() &
                  (pending["match_date"] < pd.Timestamp("today"))]

if not pending.empty:
    st.divider()
    st.markdown(f"### ⏳ {len(pending)} Live Bets Awaiting Results")
    st.caption("These past-date bets haven't been resolved yet. Run 'Update Results' from the Dashboard, or fill manually below.")

    for idx, row in pending.iterrows():
        with st.expander(f"{str(row['match_date'])[:10]}  |  {row.get('home_team','')} vs {row.get('away_team','')}  |  {row.get('side','')} @ {row.get('odds','')}"):
            col_r, col_p, col_n = st.columns([2, 2, 4])
            with col_r:
                result = st.selectbox("Result", ["", "WIN", "LOSS", "VOID"],
                                      key=f"res_{idx}")
            with col_p:
                pnl_manual = st.number_input("PnL (units)", value=0.0,
                                              step=0.01, key=f"pnl_{idx}")
            with col_n:
                note = st.text_input("Notes", key=f"note_{idx}")

            if not is_public() and st.button("Save", key=f"save_{idx}") and result:
                full = pd.read_csv(config.OUTPUT_DIR / "bets_ledger.csv", dtype=str)
                mask = (
                    (full["match_date"] == str(row["match_date"])[:10]) &
                    (full["home_team"]  == str(row.get("home_team", ""))) &
                    (full["away_team"]  == str(row.get("away_team", ""))) &
                    (full["side"]       == str(row.get("side", "")))
                )
                if "result" not in full.columns:
                    full["result"] = ""
                full.loc[mask, "result"] = result
                full.loc[mask, "pnl"]    = pnl_manual
                if note:
                    full.loc[mask, "notes"] = note
                full.to_csv(config.OUTPUT_DIR / "bets_ledger.csv", index=False)
                st.success("Saved!")
                st.cache_data.clear()
                st.rerun()
