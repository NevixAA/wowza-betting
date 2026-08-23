"""Dashboard — upcoming tips with drift signals."""
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

st.set_page_config(page_title="Dashboard | Wowza", page_icon="📊", layout="wide")
st_autorefresh(interval=5 * 60 * 1000, key="dash_refresh")

PYTHON = str(Path(sys.executable))
V9_DIR = str(Path(__file__).resolve().parents[1])


@st.cache_data(ttl=30)
def load_bets():
    f = config.OUTPUT_DIR / "bets.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
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
    if not is_public() and st.button("🔄 Run Predict Now", use_container_width=True):
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
    if not is_public() and st.button("📥 Update Results Now", use_container_width=True):
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

bets_file = config.OUTPUT_DIR / "bets.csv"
mod_time  = datetime.fromtimestamp(bets_file.stat().st_mtime).strftime("%d/%m/%Y %H:%M")
st.caption(f"Last updated: {mod_time}  •  {len(df)} fixtures fetched")

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

col_f4, col_f5 = st.columns([2, 1])
with col_f4:
    all_leagues = sorted(df["league"].dropna().unique().tolist()) if "league" in df.columns else []
    league_filter = st.multiselect("League", all_leagues, default=all_leagues, key="dash_league")
with col_f5:
    date_range = st.date_input(
        "Date range",
        value=(df["date"].min().date(), df["date"].max().date()),
        key="dash_dates",
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        date_from, date_to = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    else:
        date_from = date_to = pd.Timestamp(date_range[0]) if date_range else df["date"].min()

# Active tips only
tips = df[
    df["bet"].isin(["OVER", "UNDER"]) &
    df["signal_tier"].isin(tier_filter) &
    (df["model_type"].isin(model_filter) if model_filter else True) &
    df["drift_signal"].isin(drift_filter) &
    (df["league"].isin(league_filter) if league_filter and "league" in df.columns else True) &
    (df["date"] >= date_from) &
    (df["date"] <= date_to + pd.Timedelta(days=1))
].copy()

st.divider()

# ── SNIPER section ─────────────────────────────────────────────────────────────
snipers   = tips[tips["signal_tier"] == "SNIPER"].sort_values("best_edge", ascending=False)
marksmen  = tips[tips["signal_tier"] == "MARKSMAN"].sort_values("best_edge", ascending=False)
valuables = tips[tips["signal_tier"] == "VALUABLE"].sort_values("best_edge", ascending=False)

if not snipers.empty:
    st.markdown(f"### 🎯 SNIPER Tips &nbsp; <small style='color:#e94560'>({len(snipers)} bets)</small>",
                unsafe_allow_html=True)
    for _, row in snipers.iterrows():
        side  = row.get("best_side") or row.get("bet", "")
        odds  = row["odds_under25"] if side == "UNDER" else row["odds_over25"]
        edge  = float(row.get("best_edge", 0)) * 100
        model = "🏴󠁧󠁢󠁥󠁮󠁧󠁿" if row.get("model_type") == "standard" else "🌍"
        drift = str(row.get("drift_signal", "New"))

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);
                    border:2px solid #e94560;border-radius:10px;
                    padding:14px 18px;margin-bottom:8px">
          <span style="color:#e94560;font-size:1.1em;font-weight:bold">🎯 SNIPER</span>
          &nbsp;&nbsp;{model}
          <span style="float:right;color:#aaa">{str(row['date'])[:10]}</span><br>
          <span style="color:white;font-size:1.05em">
            <b>{row['home_team']}</b> vs <b>{row['away_team']}</b>
          </span><br>
          <span style="color:#90caf9">{row.get('league','')}</span>
          &nbsp;&nbsp;
          {odds_badge(side, odds)}
          &nbsp;&nbsp;
          <span style="color:#00c896;font-weight:bold">Edge: {edge:.1f}%</span>
          &nbsp;&nbsp;
          {drift_badge(drift)}
        </div>
        """, unsafe_allow_html=True)
else:
    if "SNIPER" in tier_filter:
        st.info("No SNIPER tips matching current filters.")

# ── MARKSMAN section ───────────────────────────────────────────────────────────
if not marksmen.empty:
    st.markdown(f"### 🔫 MARKSMAN Tips &nbsp; <small style='color:#00aaff'>({len(marksmen)} bets — 3/4 stake)</small>",
                unsafe_allow_html=True)
    marksman_rows = []
    for _, row in marksmen.iterrows():
        side  = row.get("best_side") or row.get("bet", "")
        odds  = row["odds_under25"] if side == "UNDER" else row["odds_over25"]
        edge  = float(row.get("best_edge", 0)) * 100
        drift = str(row.get("drift_signal", "New"))
        model = "Standard 🏴󠁧󠁢󠁥󠁮󠁧󠁿" if row.get("model_type") == "standard" else "New-Format 🌍"
        marksman_rows.append({
            "Date": str(row["date"])[:10], "League": row.get("league", ""),
            "Match": f"{row['home_team']} vs {row['away_team']}",
            "Side": side, "Odds": f"{odds:.2f}", "Edge": f"{edge:.1f}%",
            "Drift": drift, "Model": model,
        })
    st.dataframe(pd.DataFrame(marksman_rows), use_container_width=True, hide_index=True)
else:
    if "MARKSMAN" in tier_filter:
        st.info("No MARKSMAN tips matching current filters.")

# ── VALUABLE section ───────────────────────────────────────────────────────────
if not valuables.empty:
    st.markdown(f"### 💎 VALUABLE Tips &nbsp; <small style='color:#f5a623'>({len(valuables)} bets — half stake)</small>",
                unsafe_allow_html=True)

    rows_display = []
    for _, row in valuables.iterrows():
        side  = row.get("best_side") or row.get("bet", "")
        odds  = row["odds_under25"] if side == "UNDER" else row["odds_over25"]
        edge  = float(row.get("best_edge", 0)) * 100
        drift = str(row.get("drift_signal", "New"))
        model = "Standard 🏴󠁧󠁢󠁥󠁮󠁧󠁿" if row.get("model_type") == "standard" else "New-Format 🌍"
        rows_display.append({
            "Date":     str(row["date"])[:10],
            "League":   row.get("league", ""),
            "Match":    f"{row['home_team']} vs {row['away_team']}",
            "Side":     side,
            "Odds":     f"{odds:.2f}",
            "Edge":     f"{edge:.1f}%",
            "Drift":    drift,
            "Model":    model,
        })

    st.dataframe(
        pd.DataFrame(rows_display),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Odds":  st.column_config.NumberColumn(format="%.2f"),
        }
    )

if snipers.empty and marksmen.empty and valuables.empty:
    st.warning("No tips match your filters.")

# ── HT O/U Predictions ────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_predictions_ht():
    f = config.OUTPUT_DIR / "predictions.csv"
    if not f.exists():
        return pd.DataFrame()
    preds = pd.read_csv(f)
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
    st.dataframe(pd.DataFrame(ht_rows), use_container_width=True, hide_index=True)
    st.caption("Compare fair prices against your bookmaker's HT market to find value")

# ── Side Markets: BTTS / Over 1.5 / Over 3.5 ──────────────────────────────────
@st.cache_data(ttl=60)
def load_side_bets():
    f = config.OUTPUT_DIR / "side_bets.csv"
    if not f.exists():
        return pd.DataFrame()
    sb = pd.read_csv(f)
    sb["date"] = pd.to_datetime(sb["date"], errors="coerce")
    today = pd.Timestamp.now().normalize()
    return sb[sb["date"] >= today].copy()

side_bets = load_side_bets()
if not side_bets.empty:
    st.markdown("---")
    st.markdown(
        "### 📌 Side Market Tips &nbsp;"
        "<small style='color:#aaa'>(BTTS · Over 1.5 · Over 3.5)</small>",
        unsafe_allow_html=True,
    )
    st.caption("Standard-format leagues only · Trained separately from O/U 2.5")

    # "BTTS" alone is ambiguous about the side; this path only ever produces YES.
    SIDE_LABELS = {"btts": "BTTS — YES", "over15": "Over 1.5", "over35": "Over 3.5"}
    TIER_EMOJI  = {"SNIPER": "🎯", "MARKSMAN": "🔫", "VALUABLE": "💎"}

    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        side_tier_filter = st.multiselect(
            "Tier (side markets)", ["SNIPER", "MARKSMAN", "VALUABLE"],
            default=["SNIPER", "MARKSMAN", "VALUABLE"], key="sb_tier",
        )
    sb_filtered = side_bets[side_bets["signal_tier"].isin(side_tier_filter)].copy() \
                  if "signal_tier" in side_bets.columns else side_bets

    for mkt, mkt_label in SIDE_LABELS.items():
        ms = sb_filtered[sb_filtered["market"] == mkt] if "market" in sb_filtered.columns else pd.DataFrame()
        if ms.empty:
            continue
        with st.expander(
            f"**{mkt_label}** — {len(ms)} tip{'s' if len(ms) != 1 else ''}",
            expanded=len(ms) <= 5,
        ):
            rows = []
            for _, row in ms.sort_values("edge", ascending=False).iterrows():
                tier = row.get("signal_tier", "")
                ev = row.get("ev")
                rows.append({
                    "Tier":    f"{TIER_EMOJI.get(tier,'')} {tier}",
                    "Date":    str(row["date"])[:10],
                    "League":  row.get("league", ""),
                    "Match":   f"{row['home_team']} vs {row['away_team']}",
                    "Odds":    f"{float(row['market_odds']):.2f}",
                    "Model P": f"{float(row['model_prob'])*100:.0f}%",
                    "Edge":    f"{float(row['edge'])*100:.1f}%",
                    "EV":      f"{float(ev):+.1%}" if pd.notna(ev) else "—",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Suppressed bets note ────────────────────────────────────────────────────────
avoided = df[df["bet"] == "AVOID"]
if not avoided.empty:
    with st.expander(f"🚫 {len(avoided)} fixtures suppressed (AVOID / both-losing guard)"):
        st.dataframe(
            avoided[["date", "league", "home_team", "away_team", "signal_tier", "model_type"]],
            use_container_width=True, hide_index=True
        )


# ── Disclaimer & Terms (shown on every dashboard page) ──
from utils.disclaimer import disclaimer_footer  # noqa: E402
disclaimer_footer()
