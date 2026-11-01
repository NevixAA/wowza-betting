"""R2-C price contract (findings 14, 15, 24, 25 + the 1.08 side-edge finding). Must fail if:
a synthetic price enters ROI; a one-sided price is called de-vigged; a post-kickoff quote is used as
entry; closing/market prices enter a football-only feature set; first-half BTTS enters a full-match
analysis; a pre-fix mis-mapped O/U line is treated as a full-match price.

    python -m pytest tests/test_integrity_price_contract.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.integrity import price_provenance as PP  # noqa: E402
from src.integrity import side_edge_devig as SE  # noqa: E402


# ── synthetic prices ────────────────────────────────────────────────────────────────────────────
def test_synthetic_side_prices_are_labelled_and_refused_for_roi():
    d = pd.DataFrame({"odds_over15": [1.40, 1.40, 1.33], "price_synthetic": [True, False, False]})
    lab = PP.label_side_prices(d, "over15")
    assert list(lab["price_provenance"]) == [PP.SYNTHETIC_LABEL, PP.REAL_LABEL, PP.REAL_LABEL]
    assert list(lab["profitability_label"]) == [PP.INVALID_PROFIT, "VALID", "VALID"]
    with pytest.raises(PP.ProvenanceContractError):
        PP.assert_profitability_inputs(lab)
    PP.assert_profitability_inputs(lab[lab["profitability_valid"]])        # real rows pass


def test_without_flag_the_fill_constant_fails_closed():
    lab = PP.label_side_prices(pd.DataFrame({"odds_btts": [1.85, 1.91, np.nan]}), "btts")
    assert list(lab["profitability_valid"]) == [False, True, False]


def test_raw_frame_is_never_modified():
    d = pd.DataFrame({"odds_btts": [1.85]})
    PP.label_side_prices(d, "btts")
    assert list(d.columns) == ["odds_btts"]


@pytest.mark.parametrize("mk", ["btts", "over15", "over35"])
def test_committed_backtests_never_place_a_synthetic_price(mk):
    p = ROOT / "output" / f"backtest_results_{mk}.csv"
    if not p.exists():
        pytest.skip("no backtest output")
    d = pd.read_csv(p, usecols=lambda c: c in ("price_synthetic", "signal_tier"), low_memory=False)
    if "price_synthetic" not in d:
        pytest.skip("pre-flag backtest output")
    placed = d["signal_tier"].astype(str).isin(["SNIPER", "MARKSMAN"])
    syn = d["price_synthetic"].astype(str).str.lower().isin(("true", "1"))
    assert not (placed & syn).any()


def test_market_edge_guard_drops_fill_constants():
    from src.market_edge import exclude_synthetic_prices
    d = pd.DataFrame({"odds": [1.40, 1.36]})
    assert list(exclude_synthetic_prices(d, "OU15", "odds")["odds"]) == [1.36]


# ── market mapping ──────────────────────────────────────────────────────────────────────────────
def _caps():
    return pd.DataFrame({
        "snapshot_ts": ["2025-09-01T00:00:00Z", "2026-07-20T10:00:00", "2026-08-15T10:00:00Z",
                        "2026-07-20T10:00:00", "2026-09-01T10:00:00Z", "2026-09-01T10:00:00Z"],
        "market": ["over15", "over25", "over25", "btts_yes", "btts_yes", "btts_yes"],
        "odds": [1.66, 3.60, 1.80, 4.00, 1.75, 5.50]})


def test_newformat_mapping_labels():
    raw = _caps()
    lab = PP.label_market_mapping(raw, "newformat_odds_history")
    assert list(lab["market_mapping_valid"]) == [True, False, True, False, True, False]
    assert lab.loc[0, "line_selection"] == "MAIN_LINE_ONLY"                 # backfill selection
    assert lab.loc[1, "mapping_reason"].startswith("PARSER_PRE_FIX_OU")
    assert lab.loc[3, "mapping_reason"].startswith("FIRST_HALF_BTTS")
    assert list(raw.columns) == ["snapshot_ts", "market", "odds"]           # raw untouched
    with pytest.raises(PP.ProvenanceContractError):
        PP.assert_profitability_inputs(lab)


def test_first_half_btts_rejected_from_standard_full_match_reads():
    lab = PP.label_market_mapping(_caps(), "standard_sidemarket_odds_history")
    assert not lab.loc[5, "market_mapping_valid"] and lab.loc[4, "market_mapping_valid"]


def test_loader_mapping_filter_drops_only_with_flag(monkeypatch):
    from src import data_loader as DL
    raw = _caps()
    monkeypatch.delenv("WOWZA_PRICE_MAPPING_FILTER", raising=False)
    assert len(DL._drop_mismapped(raw, "newformat_odds_history", "t")) == len(raw)
    monkeypatch.setenv("WOWZA_PRICE_MAPPING_FILTER", "1")
    assert len(DL._drop_mismapped(raw, "newformat_odds_history", "t")) == 3


# ── de-vig / entry / closing ────────────────────────────────────────────────────────────────────
def _quotes(rows):
    return pd.DataFrame(rows, columns=["snapshot_ts", "match", "bookmaker", "market", "side", "odds"])


def _row(**kw):
    r = {"league": "League Two", "home_team": "H", "away_team": "A",
         "kickoff_utc": "2026-11-01T15:00:00Z", "market": "btts", "p_model": 0.54, "odds": 1.90,
         "asof": "2026-11-01T12:00:00Z", "model_type": "standard"}
    r.update(kw)
    return pd.DataFrame([r])


def test_one_sided_price_is_never_called_devigged():
    q = _quotes([["2026-11-01T10:00:00Z", "H vs A", "b1", "BTTS", "YES", 1.90]])
    ev = SE.evaluate_rows(_row(), q, {})
    assert ev.loc[0, "devig_state"] == "ONE_SIDED_NOT_DEVIGGED"
    assert np.isnan(ev.loc[0, "p_fair_devig"]) and not PP.is_devigged(ev.loc[0, "devig_state"])
    assert ev.loc[0, "classification"] == "UNVERIFIABLE_ONE_SIDED"


def test_post_kickoff_quote_is_never_an_entry_price():
    q = _quotes([["2026-11-01T15:00:00Z", "H vs A", "b1", "BTTS", "YES", 1.90],
                 ["2026-11-01T15:05:00Z", "H vs A", "b1", "BTTS", "NO", 1.98]])
    ev = SE.evaluate_rows(_row(asof="2026-11-01T16:00:00Z"), q, {})
    assert ev.loc[0, "devig_state"] == "ONE_SIDED_NOT_DEVIGGED"
    assert not PP.entry_price_valid("2026-11-01T15:00:00Z", "2026-11-01T15:00:00Z")
    assert PP.entry_price_valid("2026-11-01T14:59:00Z", "2026-11-01T15:00:00Z")
    assert not PP.entry_price_valid(None, "2026-11-01T15:00:00Z")           # unknown fails closed


def test_closing_or_market_prices_never_in_football_only_standard_model():
    import config
    from src.model import load_models
    from src.features_repaired import feature_cols_for
    assert config.MARKET_FEATURE_MODES[0] == "none"
    assert PP.price_features_in(feature_cols_for("none")) == []
    p = config.MODEL_FILE_STANDARD_REPAIRED
    if p.exists():
        assert PP.price_features_in(load_models(model_file=p)["feature_cols"]) == []


def test_nf_model_price_feature_exception_cannot_grow_silently():
    """The NF model is NOT football-only: it carries bookmaker_overround (finding 15, logged WARN in
    model_integrity_health). Any additional price feature must be a deliberate, reviewed change."""
    import config
    from src.model import load_models
    if not config.MODEL_FILE_NEWFORMAT.exists():
        pytest.skip("no NF payload")
    assert PP.price_features_in(load_models(model_file=config.MODEL_FILE_NEWFORMAT)["feature_cols"]) \
        in ([], ["bookmaker_overround"])
