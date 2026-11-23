"""P11 v3 price inputs (owner 2026-11-23: HT prices = "Wire in + re-measure (Recommended)"; fallback books =
"Extend to all markets (Recommended)"). Synthetic frames only: no store, no network, no provider price."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import oddsapi_history as OH  # noqa: E402
from src import oddsapi_history_adapter as AD  # noqa: E402
from src.bars_opt import oddsapi_prices as P  # noqa: E402

LG, SK, D = "USA MLS", "soccer_usa_mls", "2024-06-01"
KO = "2024-06-02T00:30:00Z"
EARLY = "2024-05-26T12:10:00Z"           # BET-role lead time (days before kickoff)
O1 = ("williamhill", "onexbet", "ladbrokes_uk")


def _q(book, market, name, price, point=None, ts=EARLY, tier="BT"):
    return {"unit_key": f"u-{tier}-{ts}", "endpoint": "event_odds", "tier": tier, "snapshot_label": "bet",
            "event_id": "e1", "sport_key": SK, "league": LG, "league_date": D, "requested_at": ts,
            "snapshot_ts": ts, "commence_time": KO, "home_team": "Austin FC", "away_team": "FC Dallas",
            "bookmaker": book, "market": market, "outcome_name": name, "outcome_description": None, "point": point,
            "price": price}


def _pair(book, market, point, over, under, **kw):
    return [_q(book, market, "Over", over, point, **kw), _q(book, market, "Under", under, point, **kw)]


def _universe(markets=("over15", "over25", "over35", "btts", "ht_over05", "ht_over15")):
    return pd.DataFrame([{"stage": "walk_forward", "fold": 0, "track": "new_format", "market": m,
                          "candidate": "champion_recipe", "row_id": f"{D}|{LG}|Austin|Dallas",
                          "date": pd.Timestamp(D), "league": LG, "y": 1.0, "p": 0.5, "deployable": True,
                          "home": "Austin", "away": "Dallas", "yes": np.nan, "no": np.nan, "yes_c": np.nan,
                          "no_c": np.nan} for m in markets])


def _priced(quotes, **kw):
    return P.price_frames(_universe(), pd.DataFrame(quotes), None, bet_role=True, **kw)


def _get(r, market):
    x = r["priced"][r["priced"]["market"] == market].iloc[0]
    return None if pd.isna(x["yes"]) else (float(x["yes"]), float(x["no"]), x["oh_book"])


# ── HT lines + HTB ─────────────────────────────────────────────────────────────────────────────

def test_htb_is_a_bet_role_tier():
    assert OH.snapshot_role("HTB", "bet") == "BET"
    assert {"BT", "BTU", "OUU", "HTB"} <= set(OH.BET_TIERS)


def test_first_half_lines_price_ht_markets_only():
    q = _pair("pinnacle", "totals_h1", 0.5, 1.4, 2.9, tier="HTB") + _pair("pinnacle", "totals_h1", 1.5, 2.6, 1.5, tier="HTB")
    r = _priced(q)
    assert _get(r, "ht_over05") == (1.4, 2.9, "pinnacle")
    assert _get(r, "ht_over15") == (2.6, 1.5, "pinnacle")
    assert _get(r, "over15") is None                     # a first-half line never prices a full-match market


def test_full_match_lines_never_price_ht_markets():
    q = _pair("pinnacle", "alternate_totals", 0.5, 1.05, 9.0) + _pair("pinnacle", "alternate_totals", 1.5, 1.3, 3.4)
    r = _priced(q)
    assert _get(r, "ht_over05") is None and _get(r, "ht_over15") is None
    assert _get(r, "over15") == (1.3, 3.4, "pinnacle")


def test_adapter_names_first_half_lines():
    assert AD.classify("totals_h1", "Over", None, 0.5, "A", "B")["market"] == "ht_over05"
    assert AD.classify("alternate_totals_h1", "Under", None, 1.5, "A", "B")["market"] == "ht_over15"


# ── O1 fallback extended (owner 2026-11-23) ─────────────────────────────────────────────────────

def test_fallback_only_when_bet365_pinnacle_have_no_pair():
    q = _pair("williamhill", "alternate_totals", 1.5, 1.3, 3.4) + _pair("pinnacle", "alternate_totals", 3.5, 2.9, 1.4) \
        + _pair("williamhill", "alternate_totals", 3.5, 3.0, 1.35)
    fb = {"over15": O1, "over35": O1}
    r = _priced(q, fallback=fb)
    assert _get(r, "over15") == (1.3, 3.4, "williamhill")          # no primary book: fallback
    assert _get(r, "over35") == (2.9, 1.4, "pinnacle")              # primary book quotes: it wins
    # without the extension (v2 rules) William Hill never prices O1.5
    assert _get(_priced(q, fallback={"btts": O1}), "over15") is None


def test_fallback_order_william_hill_then_1xbet_then_ladbrokes():
    q = _pair("ladbrokes_uk", "totals_h1", 0.5, 1.4, 2.8, tier="HTB") + _pair("onexbet", "totals_h1", 0.5, 1.45, 2.7, tier="HTB")
    assert _get(_priced(q, fallback={"ht_over05": O1}), "ht_over05")[2] == "onexbet"
    q += _pair("williamhill", "totals_h1", 0.5, 1.42, 2.75, tier="HTB")
    assert _get(_priced(q, fallback={"ht_over05": O1}), "ht_over05")[2] == "williamhill"


def test_fallback_is_per_call_so_tracks_differ():
    q = _pair("williamhill", "alternate_totals", 2.5, 1.9, 1.9)
    assert _get(_priced(q, fallback={"over25": O1}), "over25")[2] == "williamhill"     # New-Format O/U 2.5
    assert _get(_priced(q, fallback={"over15": O1}), "over25") is None                  # Standard: no O/U 2.5 fallback


def test_only_the_owners_books_in_the_owners_order():
    with pytest.raises(ValueError):
        P.check_fallback({"over15": ("betfair_ex_uk",)})
    with pytest.raises(ValueError):
        P.check_fallback({"over15": ("onexbet", "williamhill")})
    P.check_fallback({"over15": O1, "btts": ("williamhill",)})


def test_fallback_never_without_bet_role_tiers():
    q = _pair("williamhill", "alternate_totals", 1.5, 1.3, 3.4)
    with pytest.raises(ValueError):
        P.price_frames(_universe(), pd.DataFrame(q), None, bet_role=False, fallback={"over15": O1})


# ── bet-time proof ───────────────────────────────────────────────────────────────────────────

def test_midnight_stamped_history_pair_never_prices_when_dropped():
    q = _pair("pinnacle", "alternate_totals", 1.5, 1.3, 3.4, ts="2024-05-27T00:00:00Z")
    assert _get(_priced(q), "over15") is not None                    # v2 rules: no such check
    r = _priced(q, drop_midnight=True)
    assert _get(r, "over15") is None
    assert r["stats"]["pairs"]["pairs_midnight_stamp_dropped"] == 1


def test_snapshot_after_kickoff_never_prices_a_fallback():
    q = _pair("williamhill", "alternate_totals", 1.5, 1.3, 3.4, ts="2024-06-02T00:40:00Z")
    assert _get(_priced(q, fallback={"over15": O1}), "over15") is None


def test_module_default_stays_empty_unless_a_protocol_sets_it():
    src = (ROOT / "src" / "bars_opt" / "oddsapi_prices.py").read_text(encoding="utf-8")
    assert "BOOK_FALLBACK_BY_MARKET: dict = {}" in src.splitlines()


# ── the v3 script: never touches the served tier config, owner quotes recorded ─────────────────

def test_v3_protocol_records_the_owner_quotes_and_books():
    proto = json.loads((ROOT / "config" / "p11_bars_tiers_protocol_v3.json").read_text(encoding="utf-8"))
    assert "Wire in + re-measure (Recommended)" in proto["amendment_v3"]["owner_decision"]
    assert "Extend to all markets (Recommended)" in proto["amendment_v3"]["owner_decision"]
    assert "D1" in proto["predictions"]["not_used"]
    b = proto["prices"]["books_by_track_market"]
    assert "no fallback" in b["standard"]["over25"] and "William Hill" in b["new_format"]["over25"]


def test_v3_script_points_away_from_served_outputs():
    sys.path.insert(0, str(ROOT / "scripts"))
    import p11v3_bars_prices as V
    assert V.M.TIER_CONFIG != ROOT / "config" / "top3_tiers_p11.json"
    assert V.M.NAMESPACE == "bars_p11v3" and V.M.OUT.name == "p11_bars_tiers_v3"
    assert "HTB" in V.M.BET_TIERS and V.M.PROTOCOL.name == "p11_bars_tiers_protocol_v3.json"
    assert "over25" not in V.V3_FALLBACK["standard"] and "over25" in V.V3_FALLBACK["new_format"]
    for t in V.TRACKS:
        P.check_fallback(V.V3_FALLBACK[t])


def test_d1_addendum_names_the_d1_frame_and_keeps_protocol_v3():
    add = json.loads((ROOT / "config" / "p11_bars_tiers_protocol_v3_addendum_d1.json").read_text(encoding="utf-8"))
    assert add["protocol"] == "p11-bars-tiers-clean/3"
    assert "Standard predictions = D1 serving models" in add["predictions"]["standard"]
    assert "57382c902c57328b" in add["predictions"]["standard"]
    sys.path.insert(0, str(ROOT / "scripts"))
    import p11v3_bars_prices as V
    pred, rec = V.predictions_rule()
    assert rec["addendum"] == "d1-standard-predictions" and pred["standard"] == add["predictions"]["standard"]
