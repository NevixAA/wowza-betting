"""
Match Validation Agent
======================
Uses formulas from agent/formulas.md to independently validate each SNIPER/VALUE signal.

Steps per match:
  1. Load team features from predictions.csv (attack_str, defense_str, league_avg)
  2. Compute λ_H and λ_A using Team Goal Expectancy Model
  3. Build full Poisson + Dixon-Coles probability matrix
  4. Calculate O/U, BTTS, AH, Correct Score, EV, Kelly
  5. Compare against ML model signal → APPROVED / WEAK / DISAGREE
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

st.set_page_config(page_title="Match Validator | Wowza", page_icon="🔬", layout="wide")
st.markdown("## 🔬 Match Validation Agent")
st.caption("Independent model validation using formulas.md — APPROVES or DISAGREES with each signal.")

import streamlit.components.v1 as components
components.html("<script>setTimeout(()=>window.location.reload(),120000)</script>", height=0)


# ── Math library (from formulas.md) ──────────────────────────────────────────

def poisson(k, lam):
    if lam <= 0: return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam**k) / math.factorial(k)

def poisson_cdf(k, lam):
    return sum(poisson(i, lam) for i in range(k+1))

def dixon_coles_tau(x, y, lam_h, lam_a, rho=-0.10):
    """Dixon-Coles correction factor for low-scoring cells."""
    if x == 0 and y == 0: return 1 - lam_h * lam_a * rho
    if x == 1 and y == 0: return 1 + lam_a * rho
    if x == 0 and y == 1: return 1 + lam_h * rho
    if x == 1 and y == 1: return 1 - rho
    return 1.0

def score_matrix(lam_h, lam_a, max_goals=8, rho=-0.10):
    """Full Dixon-Coles corrected scoreline matrix."""
    matrix = {}
    total = 0
    for h in range(max_goals):
        for a in range(max_goals):
            tau = dixon_coles_tau(h, a, lam_h, lam_a, rho)
            p = tau * poisson(h, lam_h) * poisson(a, lam_a)
            matrix[(h, a)] = max(p, 0)
            total += max(p, 0)
    # Normalize
    return {k: v/total for k, v in matrix.items()}

def calc_markets(lam_h, lam_a, rho=-0.10):
    """Compute all markets from λ_H and λ_A."""
    mat = score_matrix(lam_h, lam_a, rho=rho)

    p_over25 = sum(v for (h,a),v in mat.items() if h+a > 2.5)
    p_under25 = 1 - p_over25
    p_btts    = sum(v for (h,a),v in mat.items() if h >= 1 and a >= 1)
    p_home    = sum(v for (h,a),v in mat.items() if h > a)
    p_draw    = sum(v for (h,a),v in mat.items() if h == a)
    p_away    = sum(v for (h,a),v in mat.items() if a > h)
    p_ah_home = p_home  # AH -0.5

    # FH model (first-half goals share: ~44% for most leagues)
    fh_pct = 0.44
    lam_fh = (lam_h + lam_a) * fh_pct
    p_ht_over05 = 1 - poisson(0, lam_fh)
    p_ht_over15 = 1 - poisson_cdf(1, lam_fh)

    top_scores = sorted(mat.items(), key=lambda x: -x[1])[:6]

    return {
        "p_over25":     p_over25,
        "p_under25":    p_under25,
        "p_btts":       p_btts,
        "p_home":       p_home,
        "p_draw":       p_draw,
        "p_away":       p_away,
        "p_ah_home":    p_ah_home,
        "p_ht_over05":  p_ht_over05,
        "p_ht_over15":  p_ht_over15,
        "top_scores":   top_scores,
    }

def ev(p, odds):
    return round(p * odds - 1, 4)

def kelly(p, odds):
    b = odds - 1
    q = 1 - p
    k = (b * p - q) / b if b > 0 else 0
    return max(0, round(k, 4))

def fair(p):
    return round(1 / max(p, 0.001), 2)

def verdict(ml_side, ml_p_over, formula_p_over, ml_edge):
    """Compare ML model vs formula model and return verdict."""
    ml_p = ml_p_over if ml_side == "OVER" else 1 - ml_p_over
    formula_p = formula_p_over if ml_side == "OVER" else 1 - formula_p_over
    diff = abs(ml_p - formula_p)

    if ml_side == "OVER":
        formula_agrees = formula_p_over > 0.50
    else:
        formula_agrees = formula_p_over < 0.50

    if formula_agrees and ml_edge >= 15:
        return "✅ APPROVED", "#00cc88", "Both models agree. Strong edge confirmed."
    elif formula_agrees and ml_edge >= 7:
        return "✅ APPROVED", "#00cc88", "Models agree. Moderate edge."
    elif formula_agrees and ml_edge > 0:
        return "⚠️ WEAK", "#ffaa00", "Models agree direction but edge is thin — monitor."
    elif diff < 0.05:
        return "⚠️ NEUTRAL", "#ffaa00", "Formula model is near 50/50 — insufficient independent confirmation."
    else:
        return "❌ DISAGREE", "#ff4444", f"Formula model says opposite direction (formula P={formula_p:.0%}). Consider skipping."


# ── Load data ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_tips():
    f = config.OUTPUT_DIR / "bets.csv"
    p = config.OUTPUT_DIR / "predictions.csv"
    if not f.exists() or not p.exists():
        return pd.DataFrame()
    bets  = pd.read_csv(f)
    preds = pd.read_csv(p)
    bets["date"] = pd.to_datetime(bets["date"], errors="coerce")
    preds["date"] = pd.to_datetime(preds["date"], errors="coerce")
    today = pd.Timestamp.now().normalize()
    bets  = bets[bets["date"] >= today]
    bets  = bets[bets["signal_tier"].isin(["SNIPER","VALUE"]) & bets["bet"].isin(["OVER","UNDER"])]
    # Merge to get team features
    merged = bets.merge(
        preds[["date","home_team","away_team","home_attack_str","away_attack_str",
               "home_defense_str","away_defense_str","league_avg_goals",
               "home_scored_last5","away_scored_last5","p_over25",
               "p_ht_over05","p_ht_over15"]],
        on=["date","home_team","away_team"], how="left"
    )
    return merged

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

snipers_only = st.checkbox("SNIPER only", value=True)
filtered = tips[tips["signal_tier"] == "SNIPER"] if snipers_only else tips

col_sel, col_all = st.columns([4, 1])
with col_sel:
    selected_label = st.selectbox("Select pick:", filtered["label"].tolist())
with col_all:
    run_all = st.button("▶ Validate All", use_container_width=True)

def analyse_row(row):
    """Run full validation on one row."""
    side       = row.get("best_side") or row.get("bet", "OVER")
    odds_o     = float(row.get("odds_over25",  2.0) or 2.0)
    odds_u     = float(row.get("odds_under25", 1.75) or 1.75)
    odds_bet   = odds_u if side == "UNDER" else odds_o
    edge       = float(row.get("best_edge", 0) or 0) * 100
    tier       = row.get("signal_tier", "VALUE")
    ml_p_over  = float(row.get("p_over25", 0.5) or 0.5)
    lga        = float(row.get("league_avg_goals", 2.5) or 2.5)
    has        = float(row.get("home_attack_str",  1.0) or 1.0)
    aas        = float(row.get("away_attack_str",  1.0) or 1.0)
    hds        = float(row.get("home_defense_str", 1.0) or 1.0)
    ads        = float(row.get("away_defense_str", 1.0) or 1.0)

    # ── Team Goal Expectancy (formulas.md §1) ─────────────────────────────
    # λ_H = LHG × HAS × ADS   (home scores against away defense)
    # λ_A = LAG × AAS × HDS   (away scores against home defense)
    lhg = lga * 0.55
    lag = lga * 0.45
    lam_h = round(lhg * has * ads, 3)
    lam_a = round(lag * aas * hds, 3)
    lam_t = round(lam_h + lam_a, 3)

    mkts = calc_markets(lam_h, lam_a)
    vrd, vrd_color, vrd_note = verdict(side, ml_p_over, mkts["p_over25"], edge)
    ml_p_bet = ml_p_over if side == "OVER" else 1 - ml_p_over
    formula_p_bet = mkts["p_over25"] if side == "UNDER" else mkts["p_under25"]
    formula_p_bet = mkts["p_over25"] if side == "OVER" else mkts["p_under25"]

    return {
        "side": side, "odds_bet": odds_bet, "odds_o": odds_o, "odds_u": odds_u,
        "edge": edge, "tier": tier,
        "ml_p_over": ml_p_over, "ml_p_bet": ml_p_bet,
        "lam_h": lam_h, "lam_a": lam_a, "lam_t": lam_t,
        "lhg": lhg, "lag": lag, "has": has, "aas": aas, "hds": hds, "ads": ads,
        "mkts": mkts,
        "verdict": vrd, "verdict_color": vrd_color, "verdict_note": vrd_note,
        "ev_val": ev(ml_p_bet, odds_bet),
        "kelly_val": kelly(ml_p_bet, odds_bet),
        "kelly_half": round(kelly(ml_p_bet, odds_bet) / 2, 4),
    }


def render_analysis(row, data):
    """Render one match analysis."""
    tc = "#e94560" if data["tier"] == "SNIPER" else "#f5a623"

    # Header
    st.markdown(f"""
    <div style="background:#16213e;border:2px solid {tc};border-radius:10px;padding:12px 18px;margin:4px 0">
      <span style="color:{tc};font-weight:bold;font-size:1.1em">{'🎯 SNIPER' if data['tier']=='SNIPER' else '💡 VALUE'}</span>
      &nbsp;&nbsp;<b style="color:white">{row['home_team']} vs {row['away_team']}</b>
      &nbsp;|&nbsp;<span style="color:#90caf9">{row.get('league','')}</span>
      &nbsp;|&nbsp;<span style="color:#aaa">{str(row['date'])[:10]}</span><br/>
      <span style="color:#00c896;font-weight:bold">Bet: {data['side']} 2.5 @ {data['odds_bet']:.2f}</span>
      &nbsp;|&nbsp;ML Edge: <b style="color:#00c896">{data['edge']:.1f}%</b>
      &nbsp;|&nbsp;ML P(over): <b style="color:white">{data['ml_p_over']:.1%}</b>
    </div>
    """, unsafe_allow_html=True)

    # Verdict banner
    st.markdown(f"""
    <div style="background:{data['verdict_color']}22;border:1.5px solid {data['verdict_color']};
                border-radius:8px;padding:10px 16px;margin:6px 0">
      <b style="color:{data['verdict_color']};font-size:1.15em">{data['verdict']}</b>
      &nbsp;&nbsp;<span style="color:#ccc">{data['verdict_note']}</span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**📐 Expected Goals (formula)**")
        st.metric("λ_H (home)", data["lam_h"],
                  help=f"LHG({data['lhg']:.2f}) × HAS({data['has']:.2f}) × ADS({data['ads']:.2f})")
        st.metric("λ_A (away)", data["lam_a"],
                  help=f"LAG({data['lag']:.2f}) × AAS({data['aas']:.2f}) × HDS({data['hds']:.2f})")
        st.metric("λ_T (total)", data["lam_t"])

    with c2:
        st.markdown("**📊 Formula Probabilities**")
        mkts = data["mkts"]
        st.metric("P(OVER 2.5)",  f"{mkts['p_over25']:.1%}",
                  delta=f"{mkts['p_over25']-data['ml_p_over']:+.1%} vs ML")
        st.metric("P(BTTS Yes)",   f"{mkts['p_btts']:.1%}")
        st.metric("P(Home Win)",   f"{mkts['p_home']:.1%}")
        st.metric("P(Draw)",       f"{mkts['p_draw']:.1%}")
        st.metric("P(Away Win)",   f"{mkts['p_away']:.1%}")

    with c3:
        st.markdown("**💰 EV & Kelly**")
        ev_c = "#00cc88" if data["ev_val"] > 0 else "#ff4444"
        st.metric("EV", f"{data['ev_val']:+.3f}")
        st.metric("Kelly (full)",   f"{data['kelly_val']:.1%}")
        st.metric("Kelly (½ — recommended)", f"{data['kelly_half']:.1%}")
        st.markdown("---")
        st.markdown("**⏱ HT Formula**")
        st.metric("P(HT OVER 0.5)", f"{mkts['p_ht_over05']:.1%}")
        st.metric("P(HT OVER 1.5)", f"{mkts['p_ht_over15']:.1%}")

    with c4:
        st.markdown("**🎯 Top Correct Scores (Dixon-Coles)**")
        for (h, a), p in mkts["top_scores"]:
            bar = "█" * max(1, int(p * 100))
            vc = "#00cc88" if h + a > 2 else "#aaa"
            st.markdown(
                f"<span style='color:white'>{h}-{a}</span> "
                f"<span style='color:{vc}'>{bar}</span> "
                f"<span style='color:#aaa'>{p:.1%} → {fair(p)}</span>",
                unsafe_allow_html=True,
            )

    # Fair odds comparison
    st.markdown("##### Fair Odds vs Market")
    rows = [
        {"Market": f"{data['side']} 2.5",  "Formula P": f"{mkts['p_over25'] if data['side']=='OVER' else mkts['p_under25']:.1%}",
         "Fair odds": fair(mkts["p_over25"] if data["side"]=="OVER" else mkts["p_under25"]),
         "Market odds": data["odds_bet"],
         "EV": f"{data['ev_val']:+.3f}", "Edge vs fair": f"{(data['odds_bet'] - fair(mkts['p_over25'] if data['side']=='OVER' else mkts['p_under25'])):.2f}"},
        {"Market": "BTTS Yes", "Formula P": f"{mkts['p_btts']:.1%}",
         "Fair odds": fair(mkts["p_btts"]), "Market odds": "—", "EV": "—", "Edge vs fair": "—"},
        {"Market": "AH -0.5 Home", "Formula P": f"{mkts['p_ah_home']:.1%}",
         "Fair odds": fair(mkts["p_ah_home"]), "Market odds": "—", "EV": "—", "Edge vs fair": "—"},
        {"Market": "HT OVER 0.5", "Formula P": f"{mkts['p_ht_over05']:.1%}",
         "Fair odds": fair(mkts["p_ht_over05"]), "Market odds": "—", "EV": "—", "Edge vs fair": "—"},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── Single match view ─────────────────────────────────────────────────────────

if not run_all:
    row = filtered[filtered["label"] == selected_label].iloc[0]
    data = analyse_row(row)
    render_analysis(row, data)

    # Optional AI context
    has_key = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
    if has_key:
        with st.expander("🤖 AI Context — injuries, tactical notes (optional)", expanded=False):
            if st.button("Ask AI for context"):
                with st.spinner("Asking AI..."):
                    try:
                        from agent.agent_runner import _load_system_prompt, _run_gemini, _run_claude
                        context_prompt = (
                            f"Match: {row['home_team']} vs {row['away_team']} — {row.get('league','')} — {str(row['date'])[:10]}\n"
                            f"Formula model: λ_H={data['lam_h']}, λ_A={data['lam_a']}, P(OVER)={data['mkts']['p_over25']:.1%}\n"
                            f"Verdict: {data['verdict']}\n\n"
                            f"In 3-4 sentences: are there any injury, tactical, weather, or lineup factors "
                            f"that could significantly affect this prediction? Focus only on factors NOT captured by historical stats."
                        )
                        system = _load_system_prompt()
                        if os.getenv("GOOGLE_API_KEY"):
                            text = _run_gemini(system, context_prompt)
                        else:
                            text = _run_claude(system, context_prompt)
                        st.markdown(text)
                    except Exception as e:
                        st.error(f"AI error: {e}")

# ── Bulk validation ───────────────────────────────────────────────────────────

else:
    st.markdown("---")
    st.markdown(f"### 🎯 Validating {len(filtered)} picks...")
    for _, row in filtered.iterrows():
        data = analyse_row(row)
        with st.expander(f"{data['verdict']}  {row['home_team']} vs {row['away_team']}  [{data['tier']}]",
                         expanded=(data["verdict"].startswith("✅") and data["tier"]=="SNIPER")):
            render_analysis(row, data)
        st.markdown("---")
