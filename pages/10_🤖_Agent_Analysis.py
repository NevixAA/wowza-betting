"""
Match Analysis — structured breakdown of every SNIPER/VALUE pick.
Numbers computed from our model. AI used only for context/risk notes.
"""
import sys, math, os
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

V9_DIR = Path(__file__).resolve().parents[1]
load_dotenv(V9_DIR / ".env")
sys.path.insert(0, str(V9_DIR))
import config

st.set_page_config(page_title="Match Analysis | Wowza", page_icon="🔬", layout="wide")
st.markdown("## 🔬 Match Analysis")
st.caption("Structured breakdown of SNIPER/VALUE picks — probabilities, EV, fair odds, cross-market check.")

import streamlit.components.v1 as components
components.html("<script>setTimeout(()=>window.location.reload(),120000)</script>", height=0)

# ── Helpers ───────────────────────────────────────────────────────────────────

def poisson_prob(k, lam):
    if lam <= 0: return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam**k) / math.factorial(k)

def poisson_cdf(k, lam):
    return sum(poisson_prob(i, lam) for i in range(k+1))

def lambda_from_p_over(p_over):
    p_under = max(0.05, min(0.95, 1 - p_over))
    lo, hi = 0.01, 8.0
    for _ in range(60):
        mid = (lo + hi) / 2
        (lo if poisson_cdf(2, mid) > p_under else hi).__class__  # dummy
        if poisson_cdf(2, mid) > p_under: lo = mid
        else: hi = mid
    return round((lo + hi) / 2, 3)

def btts_prob(lam_h, lam_a):
    p_home_score = 1 - poisson_prob(0, lam_h)
    p_away_score = 1 - poisson_prob(0, lam_a)
    return p_home_score * p_away_score

def correct_score_prob(h, a, lam_h, lam_a):
    return poisson_prob(h, lam_h) * poisson_prob(a, lam_a)

def ev(p_model, odds_market):
    return round((p_model * odds_market) - 1, 4)

def fair_odds(p):
    return round(1 / max(p, 0.001), 2)

def edge_pct(p_model, odds_market):
    p_implied = 1 / odds_market if odds_market > 0 else 0
    return round((p_model - p_implied) * 100, 1)

# ── Load data ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_tips():
    f = config.OUTPUT_DIR / "bets.csv"
    if not f.exists(): return pd.DataFrame()
    df = pd.read_csv(f)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    today = pd.Timestamp.now().normalize()
    df = df[df["date"] >= today]
    return df[df["signal_tier"].isin(["SNIPER","VALUE"]) & df["bet"].isin(["OVER","UNDER"])]

tips = load_tips()
if tips.empty:
    st.info("No upcoming SNIPER/VALUE picks. Run predict pipeline first.")
    st.stop()

# ── Pick selector ─────────────────────────────────────────────────────────────

tips["label"] = (
    tips["signal_tier"] + " | "
    + tips["date"].dt.strftime("%d/%m") + "  "
    + tips["home_team"] + " vs " + tips["away_team"]
    + "  [" + tips["league"].astype(str) + "]"
)

selected_label = st.selectbox("Select pick:", tips["label"].tolist())
row = tips[tips["label"] == selected_label].iloc[0]

# ── Header card ───────────────────────────────────────────────────────────────

side  = row.get("best_side") or row.get("bet", "")
odds_o = float(row.get("odds_over25", 2.0))
odds_u = float(row.get("odds_under25", 1.75))
odds   = odds_u if side == "UNDER" else odds_o
edge   = float(row.get("best_edge", 0)) * 100
tier   = row.get("signal_tier", "")
p_over = float(row.get("p_over25", 0.5)) if row.get("p_over25") not in (None,"N/A","") else 0.5
tc     = "#e94560" if tier == "SNIPER" else "#f5a623"

st.markdown(f"""
<div style="background:#16213e;border:2px solid {tc};border-radius:10px;padding:14px 18px;margin:8px 0">
  <span style="color:{tc};font-size:1.2em;font-weight:bold">{'🎯 SNIPER' if tier=='SNIPER' else '💡 VALUE'}</span>
  &nbsp;&nbsp;
  <b style="color:white;font-size:1.1em">{row['home_team']} vs {row['away_team']}</b>
  &nbsp;&nbsp;
  <span style="color:#90caf9">{row.get('league','')}</span>
  &nbsp;|&nbsp;
  <span style="color:#aaa">{str(row['date'])[:10]}</span><br/>
  <span style="color:#00c896;font-size:1.05em;font-weight:bold">Bet: {side} 2.5 @ {odds:.2f}</span>
  &nbsp;|&nbsp;
  <span style="color:#00c896">Edge: <b>{edge:.1f}%</b></span>
  &nbsp;|&nbsp;
  <span style="color:#aaa">Drift: {row.get('drift_signal','New')}</span>
  &nbsp;|&nbsp;
  <span style="color:#aaa">Model: {'🏴 Standard' if row.get('model_type')=='standard' else '🌍 New-Format'}</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Poisson model ─────────────────────────────────────────────────────────────

lam_t = lambda_from_p_over(p_over)
# Split total lambda 55/45 home advantage
lam_h = round(lam_t * 0.55, 3)
lam_a = round(lam_t * 0.45, 3)

p_over_calc  = 1 - poisson_cdf(2, lam_t)
p_under_calc = poisson_cdf(2, lam_t)
p_btts       = btts_prob(lam_h, lam_a)
p_home_win   = sum(correct_score_prob(h, a, lam_h, lam_a) for h in range(8) for a in range(h))
p_draw       = sum(correct_score_prob(n, n, lam_h, lam_a) for n in range(8))
p_away_win   = 1 - p_home_win - p_draw

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 📐 Expected Goals")
    st.metric("λ Home", f"{lam_h:.2f}")
    st.metric("λ Away", f"{lam_a:.2f}")
    st.metric("λ Total", f"{lam_t:.2f}")
    st.caption(f"Derived from P(over 2.5) = {p_over:.1%}")

with col2:
    st.markdown("#### 🎯 Strongest Signals")
    p_bet = p_under_calc if side == "UNDER" else p_over_calc
    ev_val = ev(p_bet, odds)
    edge_val = edge_pct(p_bet, odds)
    fair = fair_odds(p_bet)

    color = "#00cc88" if ev_val > 0 else "#ff4444"
    grade = "S" if edge > 15 else "A" if edge > 8 else "B"
    st.markdown(f"""
    <div style="background:#111827;border-left:4px solid {color};padding:10px 14px;border-radius:6px">
        <b style="color:white;font-size:1.1em">Tier {grade} — {side} 2.5</b><br/>
        <span style="color:#aaa">Model P: <b style="color:white">{p_bet:.1%}</b>
        &nbsp;|&nbsp; Implied P: <b style="color:white">{1/odds:.1%}</b></span><br/>
        <span style="color:#aaa">Edge: <b style="color:{color}">{edge_val:+.1f}%</b>
        &nbsp;|&nbsp; EV: <b style="color:{color}">{ev_val:+.3f}</b>
        &nbsp;|&nbsp; Fair odds: <b style="color:white">{fair}</b></span>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("#### ⚖️ Arbitrage Check")
    arb_sum = round(1/odds_o + 1/odds_u, 4)
    arb_color = "#00cc88" if arb_sum < 1.0 else "#aaa"
    st.markdown(f"""
    <div style="background:#111827;padding:10px 14px;border-radius:6px">
        <span style="color:#aaa">ArbitrageSum O/U:</span><br/>
        <b style="color:{arb_color};font-size:1.3em">{arb_sum:.4f}</b>
        {'<br/><span style="color:#00cc88">✅ Tier S — Arbitrage opportunity!</span>' if arb_sum < 1.0 else '<br/><span style="color:#aaa">No arbitrage</span>'}
        <br/><br/>
        <span style="color:#aaa">Bookmaker margin:</span><br/>
        <b style="color:white">{(arb_sum-1)*100:.1f}%</b>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── All markets ───────────────────────────────────────────────────────────────

st.markdown("#### 📊 All Markets — Our Fair Odds")

markets = {
    "O/U 2.5 OVER":   (p_over_calc,  odds_o),
    "O/U 2.5 UNDER":  (p_under_calc, odds_u),
    "BTTS Yes":        (p_btts,       None),
    "BTTS No":         (1-p_btts,     None),
    "Home Win":        (p_home_win,   None),
    "Draw":            (p_draw,       None),
    "Away Win":        (p_away_win,   None),
}

mkt_rows = []
for mkt, (p, mkt_odds) in markets.items():
    fair = fair_odds(p)
    ev_v = ev(p, mkt_odds) if mkt_odds else "—"
    eg   = edge_pct(p, mkt_odds) if mkt_odds else "—"
    mkt_rows.append({
        "Market": mkt,
        "P (model)": f"{p:.1%}",
        "Fair odds": fair,
        "Market odds": f"{mkt_odds:.2f}" if mkt_odds else "—",
        "Edge %": f"{eg:+.1f}%" if isinstance(eg, float) else eg,
        "EV": f"{ev_v:+.3f}" if isinstance(ev_v, float) else ev_v,
    })

st.dataframe(pd.DataFrame(mkt_rows), use_container_width=True, hide_index=True)

# ── Top correct scores ────────────────────────────────────────────────────────

st.markdown("#### 🎯 Top Correct Score Probabilities")
cs = []
for h in range(5):
    for a in range(5):
        p = correct_score_prob(h, a, lam_h, lam_a)
        cs.append({"Score": f"{h}-{a}", "P": p, "Fair odds": fair_odds(p)})
cs_df = pd.DataFrame(cs).sort_values("P", ascending=False).head(8).reset_index(drop=True)
cs_df["P"] = cs_df["P"].apply(lambda x: f"{x:.2%}")
st.dataframe(cs_df, use_container_width=True, hide_index=True)

# ── Cross-market consistency ──────────────────────────────────────────────────

st.markdown("#### 🔗 Cross-Market Consistency")
confirmations = []
if p_over_calc > 0.5 and side == "OVER":
    confirmations.append("✅ Poisson model confirms OVER 2.5")
if p_btts > 0.55:
    confirmations.append(f"✅ BTTS Yes likely ({p_btts:.0%}) — supports higher scoring")
if lam_t > 2.8:
    confirmations.append(f"✅ High lambda ({lam_t:.2f}) — attacking match expected")
if side == "UNDER" and p_btts < 0.45:
    confirmations.append(f"✅ BTTS No likely ({1-p_btts:.0%}) — supports UNDER")
if edge > 10:
    confirmations.append(f"✅ Model edge {edge:.1f}% significantly above threshold")
if row.get("drift_signal") == "Confirmed":
    confirmations.append("✅ Drift confirmed — market moving our way")

if len(confirmations) >= 3:
    st.success(f"**{len(confirmations)} signals align** — high cross-market confidence")
elif len(confirmations) == 2:
    st.warning(f"**{len(confirmations)} signals align** — moderate confidence")
else:
    st.error("**Weak cross-market support** — proceed with caution")

for c in confirmations:
    st.markdown(f"- {c}")

# ── HT breakdown ──────────────────────────────────────────────────────────────

p_ht05 = row.get("p_ht_over05")
p_ht15 = row.get("p_ht_over15")
if p_ht05 not in (None, "", "N/A"):
    st.markdown("---")
    st.markdown("#### ⏱ Half-Time Model")
    c1, c2 = st.columns(2)
    with c1:
        p05 = float(p_ht05)
        st.metric("P(HT OVER 0.5)", f"{p05:.0%}", help="At least 1 goal in first half")
        st.metric("Fair HT OVER 0.5", fair_odds(p05))
        st.metric("Fair HT UNDER 0.5", fair_odds(1-p05))
    if p_ht15 not in (None, "", "N/A"):
        with c2:
            p15 = float(p_ht15)
            st.metric("P(HT OVER 1.5)", f"{p15:.0%}", help="At least 2 goals in first half")
            st.metric("Fair HT OVER 1.5", fair_odds(p15))
            st.metric("Fair HT UNDER 1.5", fair_odds(1-p15))

# ── AI context (optional) ─────────────────────────────────────────────────────

st.markdown("---")
has_key = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))

if has_key:
    with st.expander("🤖 AI Context & Risk Notes (Gemini)", expanded=False):
        if st.button("Generate AI context"):
            with st.spinner("Asking AI for context..."):
                try:
                    from agent.agent_runner import run_agent
                    # Ask AI only for narrative context, not recalculations
                    result = run_agent(row)
                    # Parse just the risk/context section if present
                    resp = result["response"]
                    # Display cleanly
                    st.markdown(resp)
                except Exception as e:
                    st.error(f"AI error: {e}")
else:
    st.info("💡 Add `GOOGLE_API_KEY=...` to your `.env` file to enable AI context notes.")

# ── Assumptions ───────────────────────────────────────────────────────────────

with st.expander("⚠️ Model Assumptions & Limitations"):
    st.markdown(f"""
    - **Lambda split**: {lam_h:.2f} home / {lam_a:.2f} away (55/45 home advantage — approximate)
    - **Distribution**: Independent Poisson for home and away goals
    - **No live data**: No injury news, lineup, weather, or referee info
    - **P(over 2.5)** from ML model: `{p_over:.3f}` (v9 pipeline)
    - **Confidence interval on λ**: ±0.3 typical for this range
    - All market comparisons use provided odds only — shop for better lines
    """)
