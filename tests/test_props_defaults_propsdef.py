"""propsdef (Phase 2, 2026-11-05): props / Fantasy invented values that do not need a retrain.

Each test fails on data-truth-100 f66ebbe5:
  * a props tip did not carry the model's own probability: model_prob was written AFTER two typed
    hand adjustments (team-strength multiplier, quality-mismatch discount) with nothing beside it;
  * a priced tip did not say its fair probability rests on an ASSUMED overround;
  * the dashboard's Fantasy availability split read an unknown start probability as 0 = unavailable;
  * a dead legacy Fantasy fixture path still carried a typed x1.10 / x0.92 home factor;
  * the Fantasy defaults register still listed fixed defaults as open.
See reports/P2_PROPS_DEFAULTS_AND_FANTASY.md.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NAN = float("nan")


# ── 1. model_prob_raw / prob_adjustment beside model_prob ────────────────────────────────────────
def _old_inline(p, win_prob, market, career_q, opp_q):
    """The arithmetic run_player_predictions ran inline before propsdef (kept here as the oracle)."""
    from player_model.predict import _team_strength_mult
    if win_prob == win_prob:
        p = p * _team_strength_mult(win_prob, market)
    if career_q == career_q and opp_q == opp_q:
        d = float(career_q) / max(float(opp_q), 0.1)
        if d < 0.60 and market in ("goals", "goals2", "sot", "sot2", "sot3"):
            p = p * d
    return p


def test_hand_adjusted_prob_reports_the_factor_and_keeps_the_arithmetic():
    from player_model.predict import hand_adjusted_prob
    cases = [(0.30, NAN, "goals", NAN, NAN), (0.30, NAN, "goals", 0.3, 1.0), (0.30, NAN, "cards", 0.3, 1.0),
             (0.30, 0.85, "goals", NAN, NAN), (0.30, 0.10, "assists", 0.5, 1.0), (0.42, 0.20, "sot", 0.2, 0.05),
             (0.25, 0.50, "sot2", 0.9, 1.0)]
    for p, w, m, cq, oq in cases:
        got, factor, q_ok = hand_adjusted_prob(p, w, m, cq, oq)
        assert got == _old_inline(p, w, m, cq, oq)                   # bit-identical, same order
        assert np.isclose(got, p * factor)
        assert q_ok == (cq == cq and oq == oq)
    assert hand_adjusted_prob(0.3, NAN, "goals", NAN, NAN) == (0.3, 1.0, False)   # nothing applied
    _, f, _ = hand_adjusted_prob(0.3, NAN, "goals", 0.3, 1.0)
    assert np.isclose(f, 0.3)                                        # quality discount alone
    _, f, _ = hand_adjusted_prob(0.3, NAN, "cards", 0.3, 1.0)
    assert f == 1.0                                                  # discount is goals/SOT only


def test_tip_record_carries_the_raw_probability_and_the_factor():
    from player_model import predict as P
    src = inspect.getsource(P.run_player_predictions)
    assert "hand_adjusted_prob(" in src
    assert '"model_prob_raw":' in src and '"prob_adjustment":' in src
    # the raw value is taken BEFORE the adjustment, the inline arithmetic is gone
    assert src.index("_p_raw = p_model") < src.index("hand_adjusted_prob(")
    assert "p_model * _ctx_disc" not in src and "p_model * _team_strength_mult" not in src


# ── 2. a priced tip says its fair probability rests on an assumed overround ───────────────────────
def test_priced_tip_names_the_assumed_overround_basis():
    from player_model import predict as P
    tips = pd.DataFrame([
        {"player_name": "A", "market": "sot", "date": "2026-11-08", "match": "X vs Y", "model_prob": 0.40,
         "confidence": 0.5, "lazy_factors": "", "ges": 0.6, "position": "F"},
        {"player_name": "B", "market": "sot", "date": "2026-11-08", "match": "X vs Y", "model_prob": 0.40,
         "confidence": 0.5, "lazy_factors": "", "ges": 0.6, "position": "F"}])
    for c in ("market_odds", "fair_implied", "edge_abs", "edge_rel", "ev", "tier", "kelly_stake"):
        tips[c] = pd.Series([None, None], dtype=object)   # the columns run_player_predictions writes
    out = P.enrich_with_odds(tips.copy(), {"A|sot|2026-11-08|X vs Y": 3.10})
    assert out.loc[0, "fair_basis"] == P.FAIR_BASIS_ASSUMED_OVERROUND
    assert out.loc[0, "market_odds"] == 3.10
    assert out.loc[1, "fair_basis"] is None or pd.isna(out.loc[1, "fair_basis"])   # unpriced: no basis
    assert pd.isna(out.loc[1, "market_odds"]) if "market_odds" in out else True


# ── 3. dashboard Fantasy availability: unknown is not unavailable ─────────────────────────────────
def test_availability_split_counts_unknown_start_probability_as_unknown(monkeypatch):
    import dashboard_data.fantasy as DF
    frame = pd.DataFrame({"p_start": [0.9, 0.7, 0.3, 0.0, np.nan, np.nan]})
    monkeypatch.setattr(DF, "tips", lambda: frame)
    s = DF.availability_split()
    assert s["unavailable"] == 1                     # was 3: two unknowns read as 0.0
    assert s["unknown"] == 2 and s["unknown_in_top20"] == 2
    assert s["nailed"] + s["rotation_risk"] + s["doubtful"] + s["unavailable"] + s["unknown"] == s["total"]
    assert s["unavailable_in_top20"] == 1


# ── 4. the dead legacy Fantasy fixture path with a typed home factor is gone ──────────────────────
def test_no_typed_home_factor_in_fantasy_v1():
    from player_model import fantasy as F
    assert not hasattr(F, "build_fantasy_projections_fixtures")
    assert not hasattr(F, "_team_defense_map")
    code = "\n".join(ln.split("#", 1)[0] for ln in inspect.getsource(F).splitlines())
    assert "1.10 if is_h else 0.92" not in code


# ── 5. the Fantasy defaults register matches the code ────────────────────────────────────────────
def test_fantasy_defaults_register_is_current():
    from player_model import fantasy_defaults as FD
    by = {d["id"]: d for d in FD.DEFAULTS}
    for fixed in ("referee_cards_3_5", "season_hardcoded"):
        assert by[fixed]["note"].startswith("FIXED"), fixed
    assert by["v1_home_factor_legacy"]["reach"] == "DEAD"
    for new in ("v1_bonus_formula", "v1_fdr_step_home_edge", "v2_fixture_mult_clip",
                "v2_availability_half_life", "v2_m60_mean_clip", "v2_attack_fixture_clip", "v2_lambda_clip"):
        assert new in by and by[new]["class"] == "DOMAIN_PRIOR", new
