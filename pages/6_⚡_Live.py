"""
Live Center — in-play signals, half-time predictions, and signal history in one place.
Merged from the former Live / HalfTime / Live History pages (three tabs).
"""
from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Live Center | Wowza", page_icon="⚡", layout="wide")

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
from utils.access import is_public

TIPS_FILE    = BASE_DIR / "output" / "live_tips.csv"
GAMES_FILE   = BASE_DIR / "output" / "live_games.csv"
PREDS_FILE   = BASE_DIR / "output" / "predictions.csv"
HISTORY_FILE = BASE_DIR / "output" / "live_signals_history.csv"

# Auto-refresh every 2 minutes (in-play data moves fast)
components.html("<script>setTimeout(()=>window.location.reload(),120000)</script>", height=0)

SIGNAL_META = {
    "UNDER_HOLD":     ("🔒", "#00cc88", "UNDER 2.5 — model prediction holding, time running out"),
    "SLEEPING_GAME":  ("😴", "#44aaff", "UNDER 2.5 — both teams low scoring, game going nowhere"),
    "UNDER_RECOVERY": ("📉", "#ffaa00", "UNDER 2.5 — 2 goals scored, 1 more needed for over, time left"),
    "STRONG_STUCK":   ("💪", "#ff6600", "OVER 2.5 — strong attack team can't score, will push harder ⚠️ unvalidated"),
    "COMEBACK":       ("🔥", "#ff4444", "OVER 2.5 — losing team with strong attack pushing for goals ⚠️ unvalidated"),
    "HT_UNDER_0.5":   ("⏱", "#00eeff", "HT UNDER 0.5 — 0-0 late first half, barely any time left"),
    "HT_UNDER_1.5":   ("🧊", "#aaddff", "HT UNDER 1.5 — low scoring first half, fair price attractive"),
    "HT_OVER_0.5":    ("⚡", "#ffdd00", "HT OVER 0.5 — strong attack teams, 0-0, time still left"),
}


# ── Loaders ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=120)
def load_tips():
    return pd.read_csv(TIPS_FILE) if TIPS_FILE.exists() else pd.DataFrame()


@st.cache_data(ttl=120)
def load_games():
    return pd.read_csv(GAMES_FILE) if GAMES_FILE.exists() else pd.DataFrame()


@st.cache_data(ttl=120)
def load_ht():
    if not PREDS_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(PREDS_FILE)
    if "p_ht_over05" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"] >= pd.Timestamp.now().normalize()]
    df = df[df["p_ht_over05"].notna()].copy()
    df["p_ht_over05"] = pd.to_numeric(df["p_ht_over05"], errors="coerce")
    df["p_ht_over15"] = pd.to_numeric(df.get("p_ht_over15"), errors="coerce")
    return df.sort_values("p_ht_over05", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=120)
def load_history():
    if not HISTORY_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(HISTORY_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.sort_values("date", ascending=False).reset_index(drop=True)


def _norm(s: str) -> str:
    return "".join(c for c in str(s).lower() if c.isalnum())


@st.cache_data(ttl=1800, show_spinner="Loading final scores…")
def load_results_lookup():
    try:
        from src.data_loader import load_all_matches
        m = load_all_matches()
    except Exception:
        return {}
    if m is None or m.empty:
        return {}
    m = m.copy()
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    if "total_goals" not in m.columns and {"home_goals", "away_goals"}.issubset(m.columns):
        m["total_goals"] = m["home_goals"] + m["away_goals"]
    look = {}
    for r in m.itertuples(index=False):
        d = getattr(r, "date", None)
        if pd.isna(d):
            continue
        key = (_norm(getattr(r, "home_team", "")), _norm(getattr(r, "away_team", "")),
               pd.Timestamp(d).strftime("%Y-%m-%d"))
        ht = getattr(r, "ht_total_goals", None)
        look[key] = (getattr(r, "total_goals", None),
                     ht if ht is not None and not pd.isna(ht) else None)
    return look


def _settle(bet: str, total, ht_total):
    b = str(bet).upper()
    if "HT" in b:
        if ht_total is None or pd.isna(ht_total):
            return None
        if "OVER 0.5"  in b: return "WIN" if ht_total >= 1 else "LOSS"
        if "UNDER 0.5" in b: return "WIN" if ht_total == 0 else "LOSS"
        if "OVER 1.5"  in b: return "WIN" if ht_total >= 2 else "LOSS"
        if "UNDER 1.5" in b: return "WIN" if ht_total <= 1 else "LOSS"
        return None
    if total is None or pd.isna(total):
        return None
    if "OVER 2.5"  in b: return "WIN" if total >= 3 else "LOSS"
    if "UNDER 2.5" in b: return "WIN" if total <= 2 else "LOSS"
    return None


# ── Tab 1: Live Now ───────────────────────────────────────────────────────────
def render_live():
    st.caption("In-play signals from Poisson recalculation. Check your bookmaker's live screen to compare odds.")

    col_ref, col_run = st.columns([1, 4])
    with col_ref:
        if st.button("🔄 Refresh", key="live_refresh"):
            st.cache_data.clear(); st.rerun()
    with col_run:
        if not is_public() and st.button("▶ Run Live Scan Now", key="live_run"):
            import subprocess
            subprocess.run([sys.executable, str(BASE_DIR / "src" / "live_scanner.py")],
                           cwd=BASE_DIR, capture_output=True, text=True, timeout=60)
            st.cache_data.clear(); st.rerun()

    df, games = load_tips(), load_games()

    with st.expander("ℹ️ How this works"):
        st.markdown("""
        **No live odds API needed.** We compute the *fair price* with Poisson statistics:
        pre-match model → expected goals (λ) → adjust for current score + time → live P(under)/P(over).
        If your bookmaker's live odds are higher than our fair price → **value**.
        """)

    if df.empty or "signal_type" not in df.columns:
        import os
        last = ""
        if TIPS_FILE.exists():
            last = f" (last scan: {pd.Timestamp.fromtimestamp(os.path.getmtime(TIPS_FILE)).strftime('%H:%M')})"
        st.info(f"⏳ No live value signals right now. Scanner runs every ~10 min during match hours.{last}")
        return

    under_tips = df[df["bet"].str.contains("UNDER", na=False)]
    over_tips  = df[df["bet"].str.contains("OVER",  na=False)]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Live Signals", len(df))
    c2.metric("UNDER Value",  len(under_tips))
    c3.metric("OVER Value",   len(over_tips))
    c4.metric("Last Scan",    df["updated_at"].iloc[0] if "updated_at" in df.columns and not df.empty else "—")

    st.markdown("---")
    st.subheader("🎯 Active Live Signals")
    for _, row in df.iterrows():
        emoji, color, _desc = SIGNAL_META.get(row["signal_type"], ("📌", "#888", row["signal_type"]))
        bet = row["bet"]; is_under = "UNDER" in bet
        fair_odds = row["fair_under_odds"] if is_under else row["fair_over_odds"]
        live_p    = row["live_p_under"]    if is_under else row["live_p_over"]
        elapsed = int(row["elapsed_mins"]); bar_pct = min(elapsed / 90 * 100, 100)
        st.markdown(f"""
        <div style="border-left:4px solid {color}; padding:14px 18px; margin:10px 0;
                    background:#111827; border-radius:6px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:1.15em; font-weight:bold; color:white">{emoji} {row['match']}</span>
                <span style="background:{color}22; color:{color}; padding:3px 10px;
                             border-radius:12px; font-size:0.85em; font-weight:bold">{row['signal_type']}</span>
            </div>
            <div style="color:#aaa; font-size:0.9em; margin:4px 0">
                🏆 {row['league']} &nbsp;|&nbsp; ⏱ {elapsed}' &nbsp;|&nbsp;
                ⚽ Score: <b style="color:white">{row['score']}</b>
            </div>
            <div style="background:#1f2937; border-radius:4px; height:6px; margin:8px 0">
                <div style="background:{color}; width:{bar_pct:.0f}%; height:6px; border-radius:4px"></div>
            </div>
            <div style="display:flex; gap:24px; margin:8px 0">
                <div><span style="color:#aaa; font-size:0.85em">Bet</span><br>
                     <b style="color:white; font-size:1.1em">{bet}</b></div>
                <div><span style="color:#aaa; font-size:0.85em">Fair Price</span><br>
                     <b style="color:{color}; font-size:1.1em">{fair_odds}</b></div>
                <div><span style="color:#aaa; font-size:0.85em">Live P({('UNDER' if is_under else 'OVER')})</span><br>
                     <b style="color:white; font-size:1.1em">{live_p*100:.0f}%</b></div>
                <div><span style="color:#aaa; font-size:0.85em">Pre-match P(over)</span><br>
                     <b style="color:#888; font-size:1.0em">{row['pre_p_over']*100:.0f}%</b></div>
            </div>
            <div style="color:#4b5563; font-size:0.78em; margin-top:4px">
                ⚠️ Bet only if your bookmaker's live odds &gt; {fair_odds}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("📊 Raw data"):
        cols = ["league", "match", "score", "elapsed_mins", "signal_type", "bet",
                "fair_under_odds", "fair_over_odds", "live_p_under", "live_p_over",
                "pre_p_over", "lam_remaining"]
        st.dataframe(df[[c for c in cols if c in df.columns]], use_container_width=True)

    st.markdown("---")
    st.subheader(f"👁 All Monitored Games ({len(games)})")
    if games.empty:
        st.info("No live games right now in our tracked leagues.")
        return
    signal_matches = set(df["match"].tolist()) if not df.empty else set()
    for _, g in games.iterrows():
        has_signal = g["match"] in signal_matches
        has_pred   = str(g.get("has_prediction", "")).lower() == "true"
        border      = "#ff4444" if has_signal else ("#00cc88" if has_pred else "#444")
        badge       = "🚨 SIGNAL" if has_signal else ("📊 tracked" if has_pred else "👁 watching")
        badge_color = "#ff4444" if has_signal else ("#00cc88" if has_pred else "#666")
        p_under = g.get("live_p_under"); fair = g.get("fair_under_odds")
        p_str = (f"P(under): <b>{p_under}%</b> | Fair UNDER: <b>{fair}</b>"
                 if pd.notna(p_under) and p_under else "No pre-match prediction")
        st.markdown(f"""
        <div style="border-left:3px solid {border}; padding:10px 14px; margin:6px 0;
                    background:#111827; border-radius:4px; display:flex; justify-content:space-between; align-items:center;">
            <div><span style="color:white; font-weight:bold">{g['match']}</span>
                <span style="color:#888; font-size:0.85em"> · {g['league']}</span><br/>
                <span style="color:#ccc">⏱ {g['elapsed_mins']}' &nbsp;|&nbsp; ⚽ <b>{g['score']}</b>
                &nbsp;|&nbsp; {p_str}</span></div>
            <span style="background:{badge_color}22; color:{badge_color}; padding:3px 10px;
                         border-radius:10px; font-size:0.8em; white-space:nowrap">{badge}</span>
        </div>
        """, unsafe_allow_html=True)


# ── Tab 2: Half-Time ────────────────────────────────────────────────────────────
def render_ht():
    st.caption("Model predictions for HT OVER/UNDER 0.5 and 1.5 · Standard-format leagues only")
    df = load_ht()
    if df.empty:
        st.info("⏳ No HT predictions available. Only standard-format leagues have half-time data.")
        return

    strong_over05  = df[df["p_ht_over05"] >= 0.75]
    strong_under05 = df[df["p_ht_over05"] <= 0.30]
    strong_over15  = df[df["p_ht_over15"] >= 0.60] if "p_ht_over15" in df.columns else pd.DataFrame()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Fixtures",      len(df))
    c2.metric("⚡ Strong OVER 0.5",  len(strong_over05), help="P(HT OVER 0.5) ≥ 75%")
    c3.metric("🧊 Strong UNDER 0.5", len(strong_under05), help="P(HT OVER 0.5) ≤ 30%")
    c4.metric("🔥 Strong OVER 1.5",  len(strong_over15), help="P(HT OVER 1.5) ≥ 60%")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        show = st.selectbox("Show", ["All signals", "Strong OVER 0.5 only", "Strong UNDER 0.5 only",
                                     "Strong OVER 1.5 only", "All fixtures"], key="ht_show")
    with col2:
        leagues = sorted(df["league"].unique())
        league_filter = st.multiselect("League", leagues, default=leagues, key="ht_league")
    df = df[df["league"].isin(league_filter)]

    has15 = "p_ht_over15" in df.columns
    if show == "Strong OVER 0.5 only":
        df = df[df["p_ht_over05"] >= 0.75]
    elif show == "Strong UNDER 0.5 only":
        df = df[df["p_ht_over05"] <= 0.30]
    elif show == "Strong OVER 1.5 only" and has15:
        df = df[df["p_ht_over15"] >= 0.60]
    elif show == "All signals":
        mask = (df["p_ht_over05"] >= 0.75) | (df["p_ht_over05"] <= 0.30)
        if has15:
            mask = mask | (df["p_ht_over15"] >= 0.60)
        df = df[mask]
    if df.empty:
        st.info("No fixtures match your filters.")
        return

    st.subheader(f"📋 {len(df)} Fixture(s)")
    for _, row in df.iterrows():
        p05 = float(row["p_ht_over05"])
        p15 = float(row["p_ht_over15"]) if pd.notna(row.get("p_ht_over15")) else None
        fair_over05  = round(1 / max(p05, 0.01), 2)
        fair_under05 = round(1 / max(1 - p05, 0.01), 2)
        fair_over15  = round(1 / max(p15, 0.01), 2) if p15 else None
        fair_under15 = round(1 / max(1 - p15, 0.01), 2) if p15 else None
        signals = []
        if p05 >= 0.75: signals.append(("⚡", "#ffdd00", f"HT OVER 0.5  ({p05*100:.0f}%) — fair {fair_over05}"))
        if p05 <= 0.30: signals.append(("🧊", "#44aaff", f"HT UNDER 0.5 ({(1-p05)*100:.0f}%) — fair {fair_under05}"))
        if p15 and p15 >= 0.60: signals.append(("🔥", "#ff6600", f"HT OVER 1.5  ({p15*100:.0f}%) — fair {fair_over15}"))
        if p15 and p15 <= 0.25: signals.append(("🔒", "#00cc88", f"HT UNDER 1.5 ({(1-p15)*100:.0f}%) — fair {fair_under15}"))
        border = signals[0][1] if signals else "#444"
        badges = (" ".join(f'<span style="background:{c}22;color:{c};padding:2px 8px;border-radius:10px;'
                           f'font-size:0.82em">{e} {t}</span>' for e, c, t in signals)
                  if signals else '<span style="color:#666">No strong signal</span>')
        p15_html = ""
        if p15:
            p15_html = ('<div><span style="color:#aaa;font-size:0.8em">P(HT OVER 1.5)</span><br>'
                        f'<b style="color:white">{p15*100:.0f}%</b></div>'
                        '<div><span style="color:#aaa;font-size:0.8em">Fair OVER 1.5</span><br>'
                        f'<b style="color:white">{fair_over15}</b></div>')
        st.markdown(f"""
        <div style="border-left:4px solid {border};padding:12px 16px;margin:6px 0;
                    background:#111827;border-radius:6px;">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <b style="color:white;font-size:1.05em">{row['home_team']} vs {row['away_team']}</b>
                <span style="color:#aaa;font-size:0.85em">📅 {str(row['date'])[:10]}</span>
            </div>
            <div style="color:#90caf9;font-size:0.85em;margin:3px 0">🏆 {row.get('league','')}</div>
            <div style="margin:8px 0">{badges}</div>
            <div style="display:flex;gap:24px;margin-top:6px">
                <div><span style="color:#aaa;font-size:0.8em">P(HT OVER 0.5)</span><br>
                    <b style="color:{'#ffdd00' if p05>=0.75 else '#44aaff' if p05<=0.30 else 'white'}">{p05*100:.0f}%</b></div>
                <div><span style="color:#aaa;font-size:0.8em">Fair OVER 0.5</span><br><b style="color:white">{fair_over05}</b></div>
                <div><span style="color:#aaa;font-size:0.8em">Fair UNDER 0.5</span><br><b style="color:white">{fair_under05}</b></div>
                {p15_html}
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.caption("AUC 0.567 (HT OVER 0.5) · AUC 0.569 (HT OVER 1.5) · standard-format leagues only")


# ── Tab 3: History ──────────────────────────────────────────────────────────────
def render_history():
    st.caption("All past in-play signals — which types fire most and whether they'd have won.")
    df = load_history()
    if df.empty:
        st.info("⏳ No live signal history yet. Signals are logged automatically when the scanner fires.")
        return

    lookup = load_results_lookup()

    def _lookup_result(row):
        m = str(row.get("match", ""))
        if " vs " not in m or pd.isna(row["date"]):
            return (None, None)
        h, a = m.split(" vs ", 1)
        for off in (0, -1, 1):
            key = (_norm(h), _norm(a), (pd.Timestamp(row["date"]) + pd.Timedelta(days=off)).strftime("%Y-%m-%d"))
            if key in lookup:
                return lookup[key]
        return (None, None)

    _res = df.apply(_lookup_result, axis=1, result_type="expand")
    df["final_total"] = _res[0]
    _ht = _res[1]
    df["result"] = [_settle(b, t, h) for b, t, h in zip(df["bet"], df["final_total"], _ht)]

    settled = df[df["result"].isin(["WIN", "LOSS"])]
    pending = int((~df["result"].isin(["WIN", "LOSS"])).sum())
    st.subheader("✅ Results — did the live signals win?")
    if settled.empty:
        st.info(f"No signals settled yet — {pending} awaiting final scores.")
    else:
        n = len(settled); wins = int((settled["result"] == "WIN").sum())
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Settled", n)
        r2.metric("Record", f"{wins}W – {n - wins}L")
        r3.metric("Hit rate", f"{wins / n * 100:.0f}%")
        r4.metric("Pending", pending)
        by_type = (settled.assign(win=(settled["result"] == "WIN").astype(int))
                   .groupby("signal_type").agg(bets=("win", "size"), wins=("win", "sum")).reset_index())
        by_type["hit_rate_%"] = (by_type["wins"] / by_type["bets"] * 100).round(0)
        st.dataframe(by_type.sort_values("bets", ascending=False), use_container_width=True, hide_index=True)
        st.caption("⚠️ Small sample — directional only. U/O 2.5 settled on full-time; HT bets on half-time goals.")

    st.markdown("---")
    total = len(df); matches = df["match"].nunique() if "match" in df.columns else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Signals",  total)
    c2.metric("Unique Matches", matches)
    c3.metric("First Signal",   df["date"].min().strftime("%b %d") if not df.empty else "—")
    c4.metric("Latest Signal",  df["date"].max().strftime("%b %d, %Y") if not df.empty else "—")
    st.markdown("---")

    st.subheader("📊 Signal Type Breakdown")
    if "signal_type" in df.columns:
        sig_counts = df["signal_type"].value_counts().reset_index()
        sig_counts.columns = ["Signal Type", "Count"]
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(sig_counts, use_container_width=True, hide_index=True)
        with col2:
            try:
                import plotly.express as px
                fig = px.bar(sig_counts, x="Signal Type", y="Count", color="Count",
                             color_continuous_scale=["#1a1a2e", "#e94560"], title="Signal Frequency")
                fig.update_layout(template="plotly_dark", coloraxis_showscale=False,
                                  plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.bar_chart(sig_counts.set_index("Signal Type")["Count"])
    st.markdown("---")

    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        sig_types = sorted(df["signal_type"].unique()) if "signal_type" in df.columns else []
        sig_filter = st.multiselect("Signal type", sig_types, default=sig_types, key="hist_sig")
    with col_f2:
        leagues = sorted(df["league"].unique()) if "league" in df.columns else []
        league_filter = st.multiselect("League", leagues, default=leagues, key="hist_league")
    filtered = df.copy()
    if sig_filter and "signal_type" in filtered.columns:
        filtered = filtered[filtered["signal_type"].isin(sig_filter)]
    if league_filter and "league" in filtered.columns:
        filtered = filtered[filtered["league"].isin(league_filter)]

    st.subheader(f"📋 {len(filtered)} Signal(s)")
    display_cols = ["date", "league", "match", "score", "elapsed_mins", "signal_type", "bet",
                    "result", "final_total", "fair_under_odds", "fair_over_odds",
                    "live_p_under", "live_p_over", "pre_p_over"]
    show_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(
        filtered[show_cols].rename(columns={
            "elapsed_mins": "min", "signal_type": "signal", "final_total": "FT goals",
            "fair_under_odds": "fair_U", "fair_over_odds": "fair_O",
            "live_p_under": "P(U)", "live_p_over": "P(O)", "pre_p_over": "pre P(O)",
        }),
        use_container_width=True, hide_index=True,
    )


# ── Layout ──────────────────────────────────────────────────────────────────────
st.title("⚡ Live Center")
tab_live, tab_ht, tab_hist = st.tabs(["⚡ Live Now", "⏱ Half-Time", "📜 History"])
with tab_live:
    render_live()
with tab_ht:
    render_ht()
with tab_hist:
    render_history()


# ── Disclaimer & Terms (shown on every dashboard page) ──
from utils.disclaimer import disclaimer_footer  # noqa: E402
disclaimer_footer()
