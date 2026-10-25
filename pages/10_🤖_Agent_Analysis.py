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

import dashboard_ui as ui
from dotenv import load_dotenv

V9_DIR = Path(__file__).resolve().parents[1]
load_dotenv(V9_DIR / ".env")
sys.path.insert(0, str(V9_DIR))
import config

st.set_page_config(page_title="Match Validator | Wowza", page_icon="🔬", layout="wide")
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from dashboard_access import require as _require_tier  # noqa: E402
_require_tier("admin")  # tier guard (dashboard_access.PAGES), 2026-10-18

ui.page_header("Match validation agent", "Independent model validation using formulas.md — APPROVES or DISAGREES "
               "with each signal. A second opinion, never a bet.", eyebrow_text="Research Lab · agent analysis",
               freshness=ui.status_chip("RESEARCH", label="RESEARCH · SECOND OPINION"))

# Was components.v1.html with a JS reload: an API whose announced removal date
# (2026-06-01) has passed, and which reloaded the whole browser tab and so DISCARDED
# every filter the user had set.
ui.autorefresh(minutes=2, key="10_agent_analysis_refresh")
# ── How to read this page ─────────────────────────────────────────────────────
with st.expander("ℹ️ How to read this page", expanded=False):
    st.markdown("""
This is a **second opinion**. The betting tip comes from the **ML model**; this page
re-derives the same match from scratch with **independent maths** (Poisson + Dixon-Coles
scoreline model from `formulas.md`) and checks whether the two *agree*. Agreement between
two different methods is stronger evidence than either alone.

**The numbers**
- **λ_H / λ_A (lambda)** — expected goals for the home / away team, from
  `league_avg × attack_strength × opponent_defense_strength`. **λ_T** = total expected goals.
  *(If team-strength data is missing, it falls back to league average = 1.0 and says so.)*
- **Formula P** — the probability this independent model gives (e.g. P(OVER 2.5)).
  **ML P** — the machine-learning model's probability. The `Δ vs ML` shows how far apart they are.
- **EV** — expected value per 1 unit staked: `P × odds − 1`. Positive = +EV at that price.
- **Kelly** — mathematically optimal stake fraction; **½-Kelly** is the recommended (safer) version.
- **Correct scores** — most likely exact scorelines (Dixon-Coles corrects the low-scoring cells).

**The verdict**
| Badge | Meaning |
|---|---|
| ✅ **APPROVED** | Independent model agrees on direction AND the ML edge is solid (≥7%). Highest confidence. |
| ⚠️ **WEAK** | Both agree on direction but the edge is thin — monitor, don't over-stake. |
| ⚠️ **NEUTRAL** | Formula model is near 50/50 — no independent confirmation either way. |
| ❌ **DISAGREE** | Formula model points the *opposite* way — consider skipping the tip. |

**AI Context** (optional, needs a key) asks an LLM for injury/tactical/weather factors that
stats can't see. The **9-Agent Pipeline** at the bottom runs the full multi-agent player-props
flow (data → minutes → goal-expectancy → prop-modeling → market-intel → confidence →
simulation → portfolio) for one match.
""")


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

    # The second value is a semantic TONE for the verdict chip (visual redesign 2026-10-25: it was a hex
    # colour). The verdict word and the rule that picks it are unchanged.
    if formula_agrees and ml_edge >= 15:
        return "✅ APPROVED", "POSITIVE", "Both models agree. Strong edge confirmed."
    elif formula_agrees and ml_edge >= 7:
        return "✅ APPROVED", "POSITIVE", "Models agree. Moderate edge."
    elif formula_agrees and ml_edge > 0:
        return "⚠️ WEAK", "WARNING", "Models agree direction but edge is thin — monitor."
    elif diff < 0.05:
        return "⚠️ NEUTRAL", "WARNING", "Formula model is near 50/50 — insufficient independent confirmation."
    else:
        return "❌ DISAGREE", "NEGATIVE", f"Formula model says opposite direction (formula P={formula_p:.0%}). Consider skipping."


# ── Load data ─────────────────────────────────────────────────────────────────

FEAT_COLS = ["home_attack_str","away_attack_str","home_defense_str","away_defense_str",
             "league_avg_goals","home_scored_last5","away_scored_last5",
             "p_over25","p_ht_over05","p_ht_over15"]

@st.cache_data(ttl=60)
def load_tips():
    f = config.OUTPUT_DIR / "bets.csv"
    p = config.OUTPUT_DIR / "predictions.csv"
    if not f.exists() or not p.exists():
        return pd.DataFrame()
    bets  = pd.read_csv(f)
    preds = pd.read_csv(p)

    bets["date"]  = pd.to_datetime(bets["date"],  errors="coerce")
    preds["date"] = pd.to_datetime(preds["date"], errors="coerce")
    today = pd.Timestamp.now().normalize()
    bets  = bets[bets["date"] >= today]
    bets  = bets[bets["signal_tier"].isin(["SNIPER","MARKSMAN","VALUABLE"]) & bets["bet"].isin(["OVER","UNDER"])].copy()

    # Ensure feature columns exist as NaN first
    for col in FEAT_COLS:
        bets[col] = float("nan")

    # Fuzzy row-by-row lookup — handles minor name/date differences
    def _norm(s):
        return str(s).lower().strip()

    for idx, row in bets.iterrows():
        h8 = _norm(row["home_team"])[:8]
        a8 = _norm(row["away_team"])[:8]
        d  = row["date"].date()
        candidates = preds[
            (preds["home_team"].str.lower().str[:8] == h8) &
            (preds["away_team"].str.lower().str[:8] == a8) &
            (preds["date"].dt.date == d)
        ]
        if candidates.empty:
            # Try date ±1 day
            candidates = preds[
                (preds["home_team"].str.lower().str[:8] == h8) &
                (preds["away_team"].str.lower().str[:8] == a8)
            ]
        if not candidates.empty:
            src = candidates.iloc[0]
            for col in FEAT_COLS:
                if col in src.index:
                    bets.at[idx, col] = src[col]

    return bets

tips = load_tips()
if tips.empty:
    ui.empty_state("NO_DATA", "No upcoming SNIPER/VALUE picks. Run predict pipeline first.")
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

if filtered.empty:
    tier_label = "SNIPER" if snipers_only else "SNIPER/MARKSMAN/VALUABLE"
    ui.empty_state("NO_DATA", f"No {tier_label} picks for today. Try unchecking 'SNIPER only' or wait for the next "
                              "predict run.")
    st.stop()

col_sel, col_all = st.columns([4, 1])
with col_sel:
    selected_label = st.selectbox("Select pick:", filtered["label"].tolist())
with col_all:
    run_all = st.button("▶ Validate All", width="stretch", type="secondary")

def _safe_float(val, default=1.0):
    """Return float or default if NaN/None/missing."""
    try:
        v = float(val)
        return default if (v != v) else v  # v != v is True for NaN
    except (TypeError, ValueError):
        return default

def analyse_row(row):
    """Run full validation on one row."""
    side       = row.get("best_side") or row.get("bet", "OVER")
    odds_o     = _safe_float(row.get("odds_over25"),  2.0)
    odds_u     = _safe_float(row.get("odds_under25"), 1.75)
    odds_bet   = odds_u if side == "UNDER" else odds_o
    edge       = _safe_float(row.get("best_edge"), 0.0) * 100
    tier       = row.get("signal_tier", "VALUE")
    ml_p_over  = _safe_float(row.get("p_over25"),         0.5)
    lga        = _safe_float(row.get("league_avg_goals"), 2.5)
    # Default to 1.0 (league average) when team strength data is missing
    has        = _safe_float(row.get("home_attack_str"),  1.0)
    aas        = _safe_float(row.get("away_attack_str"),  1.0)
    hds        = _safe_float(row.get("home_defense_str"), 1.0)
    ads        = _safe_float(row.get("away_defense_str"), 1.0)

    # ── Team Goal Expectancy (formulas.md §1) ─────────────────────────────
    using_defaults = (has == 1.0 and aas == 1.0 and hds == 1.0 and ads == 1.0)
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
        "using_defaults": using_defaults,
    }


_VERDICT_STATE = {"POSITIVE": "PASS", "WARNING": "WARN", "NEGATIVE": "FAIL"}


def render_analysis(row, data):
    """Render one match analysis (visual redesign 2026-10-25: shared cards and chips, no inline HTML).
    Every number is the one analyse_row computed, formatted exactly as before."""
    mkts = data["mkts"]

    # The tip under review: the shared selection card (legacy tier last and neutral)
    ui.selection_card(league=str(row.get("league", "")), kickoff=str(row["date"])[:10],
                      home=row["home_team"], away=row["away_team"], pick=f"Bet: {data['side']} 2.5",
                      odds=f"{data['odds_bet']:.2f}", tier=str(data["tier"]),
                      show_plan=False,
                      supporting=[("ML Edge", f"{data['edge']:.1f}%"), ("ML P(over)", f"{data['ml_p_over']:.1%}")])
    # The verdict: icon + word chip, the note beside it
    st.markdown(ui.status_chip(_VERDICT_STATE.get(data["verdict_color"], "WARN"),
                               label=data["verdict"].split(" ", 1)[-1], tone=data["verdict_color"])
                + f" &nbsp;**{data['verdict']}** — {data['verdict_note']}")

    c1, c2, c3 = st.columns(3)
    ui.metric_card("λ_T (total)", data["lam_t"], title="Expected goals (formula)",
                   supporting=[("λ_H (home)", data["lam_h"]), ("λ_A (away)", data["lam_a"])],
                   help=f"λ_H = LHG({data['lhg']:.2f}) × HAS({data['has']:.2f}) × ADS({data['ads']:.2f}) · "
                        f"λ_A = LAG({data['lag']:.2f}) × AAS({data['aas']:.2f}) × HDS({data['hds']:.2f})",
                   status=ui.status_chip("WARN", label="TEAM STRENGTH MISSING · LEAGUE AVERAGES USED")
                   if data.get("using_defaults") else "", container=c1)
    ui.metric_card("P(OVER 2.5)", f"{mkts['p_over25']:.1%}", title="Formula probabilities",
                   delta=f"{mkts['p_over25']-data['ml_p_over']:+.1%} vs ML",
                   supporting=[("P(BTTS Yes)", f"{mkts['p_btts']:.1%}"), ("P(Home Win)", f"{mkts['p_home']:.1%}"),
                               ("P(Draw)", f"{mkts['p_draw']:.1%}"), ("P(Away Win)", f"{mkts['p_away']:.1%}")],
                   container=c2)
    ui.metric_card("EV", f"{data['ev_val']:+.3f}", title="EV & Kelly",
                   supporting=[("Kelly (full)", f"{data['kelly_val']:.1%}"),
                               ("Kelly (½ — recommended)", f"{data['kelly_half']:.1%}"),
                               ("P(HT OVER 0.5)", f"{mkts['p_ht_over05']:.1%}"),
                               ("P(HT OVER 1.5)", f"{mkts['p_ht_over15']:.1%}")], container=c3)
    if data.get("using_defaults"):
        st.caption("⚠️ Team strength data missing — using league averages (HAS=AAS=HDS=ADS=1.0)")

    left, right = st.columns([2, 3])
    with left:
        ui.eyebrow("Top correct scores (Dixon-Coles)")
        cs = pd.DataFrame([{"Score": f"{h}-{a}", "Probability": p, "Fair odds": fair(p),
                            "Goals": "3+" if h + a > 2 else ("2" if h + a == 2 else "0–1")}
                           for (h, a), p in mkts["top_scores"] if p and p == p])
        ui.compact_table(cs, {"Probability": ui.bar_col("Probability", max_value=0.25, fmt="%.3f",
                                                         help="Bar scale 0–25%: one exact scoreline is rarely "
                                                              "likelier than that."),
                              "Fair odds": ui.num_col("Fair odds")})
    with right:
        ui.eyebrow("Fair odds vs market")
        side_p = mkts["p_over25"] if data["side"] == "OVER" else mkts["p_under25"]
        rows = [
            {"Market": f"{data['side']} 2.5", "Formula P": side_p, "Fair odds": fair(side_p),
             "Market odds": data["odds_bet"], "EV": data["ev_val"],
             "Edge vs fair": round(data["odds_bet"] - fair(side_p), 2)},
            {"Market": "BTTS Yes", "Formula P": mkts["p_btts"], "Fair odds": fair(mkts["p_btts"]),
             "Market odds": None, "EV": None, "Edge vs fair": None},
            {"Market": "AH -0.5 Home", "Formula P": mkts["p_ah_home"], "Fair odds": fair(mkts["p_ah_home"]),
             "Market odds": None, "EV": None, "Edge vs fair": None},
            {"Market": "HT OVER 0.5", "Formula P": mkts["p_ht_over05"], "Fair odds": fair(mkts["p_ht_over05"]),
             "Market odds": None, "EV": None, "Edge vs fair": None},
        ]
        ui.compact_table(pd.DataFrame(rows), {"Formula P": ui.num_col("Formula P", fmt="%.3f"),
                                              "Fair odds": ui.num_col("Fair odds"),
                                              "Market odds": ui.num_col("Market odds"),
                                              "EV": ui.num_col("EV", fmt="%+.3f"),
                                              "Edge vs fair": ui.num_col("Edge vs fair", fmt="%.2f")})
        st.caption("No collected market price for BTTS, AH or HT — those rows show the formula's fair price only.")


# ── Single match view ─────────────────────────────────────────────────────────

if not run_all:
    row = filtered[filtered["label"] == selected_label].iloc[0]
    data = analyse_row(row)
    render_analysis(row, data)

    # Optional AI context
    has_key = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
    if has_key:
        with st.expander("🤖 AI Context — injuries, tactical notes (optional)", expanded=False):
            if st.button("Ask AI for context", type="secondary"):
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
    ui.section_header(f"Validating {len(filtered)} picks")
    for _, row in filtered.iterrows():
        data = analyse_row(row)
        with st.expander(f"{data['verdict']}  {row['home_team']} vs {row['away_team']}  [{data['tier']}]",
                         expanded=(data["verdict"].startswith("✅") and data["tier"]=="SNIPER")):
            render_analysis(row, data)

# ── Agent Pipeline (9-agent flow) ─────────────────────────────────────────────
ui.space("medium")
with st.expander("🤖 9-Agent Player Props Pipeline", expanded=False):
    from agent.agent_runner import PIPELINE_STAGES, AGENTS, run_pipeline

    st.caption(
        "Full pipeline: football-data-collection → minutes-projection → team-goal-expectancy "
        "→ player-prop-modeling → market-intelligence → confidence-scoring → simulation → portfolio-manager"
    )

    # Show available agents
    col_a, col_b = st.columns(2)
    with col_a:
        ui.eyebrow("Pipeline agents")
        for s in PIPELINE_STAGES:
            exists = s in AGENTS
            st.markdown(f"{'✅' if exists else '❌'} `{s}`")
    with col_b:
        ui.eyebrow("Other agents")
        other = [k for k in AGENTS if k not in PIPELINE_STAGES and k]
        for k in sorted(other):
            st.markdown(f"🔹 `{k}`")

    has_key = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
    if not has_key:
        ui.empty_state("NOT_CONNECTED", "No API key configured (GOOGLE_API_KEY or ANTHROPIC_API_KEY). Add it to .env "
                                        "to run the pipeline.")
    else:
        st.markdown("**Run pipeline on a match:**")
        pipeline_match = st.selectbox(
            "Select match for pipeline run:",
            tips["label"].tolist(),
            key="pipeline_match_select",
        )
        if st.button("▶ Run 9-Agent Pipeline", key="run_pipeline_btn", type="primary"):
            row = tips[tips["label"] == pipeline_match].iloc[0]
            match_data = {
                "match":      f"{row['home_team']} vs {row['away_team']}",
                "home_team":  row["home_team"],
                "away_team":  row["away_team"],
                "league":     row.get("league", ""),
                "date":       str(row["date"])[:10],
                "signal_tier": row.get("signal_tier", ""),
                "p_over25":   row.get("p_over25", ""),
                "odds_over25":row.get("odds_over25", ""),
                "odds_under25":row.get("odds_under25", ""),
                "best_edge":  row.get("best_edge", ""),
            }
            with st.spinner("Running 8-stage pipeline... (this takes ~30s per agent)"):
                try:
                    result = run_pipeline(match_data)
                    st.success("Pipeline complete")
                    for stage in PIPELINE_STAGES:
                        with st.expander(f"📋 {stage}", expanded=(stage == "portfolio-manager")):
                            st.markdown(result["stages"].get(stage, "No output"))
                except Exception as e:
                    st.error(f"Pipeline error: {e}")


# ── Disclaimer & Terms (shown on every dashboard page) ──
from utils.disclaimer import disclaimer_footer  # noqa: E402
disclaimer_footer()
