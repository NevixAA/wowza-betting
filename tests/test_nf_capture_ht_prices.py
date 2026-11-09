"""P7 item 3 (2026-11-09): the new-format capture parses the first-half goals O/U market from the
same API-Football /odds answer it already fetches (0 extra calls), with the standard capture's keys."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NF = _load("capture_nf_odds_forward")
STD = _load("capture_std_sidemarket_odds_forward")


def _answer(bets):
    return {"response": [{"bookmakers": [{"id": 8, "name": "Bet365", "bets": bets}]}]}


FT = {"id": 5, "name": "Goals Over/Under", "values": [
    {"value": "Over 1.5", "odd": "1.30"}, {"value": "Under 1.5", "odd": "3.40"},
    {"value": "Over 2.5", "odd": "1.90"}, {"value": "Under 2.5", "odd": "1.90"},
    {"value": "Over 3.5", "odd": "3.10"}, {"value": "Under 3.5", "odd": "1.36"}]}
HT = {"id": 6, "name": "Goals Over/Under First Half", "values": [
    {"value": "Over 0.5", "odd": "1.40"}, {"value": "Under 0.5", "odd": "2.75"},
    {"value": "Over 1.5", "odd": "3.00"}, {"value": "Under 1.5", "odd": "1.36"}]}
H2 = {"id": 26, "name": "Goals Over/Under - Second Half", "values": [
    {"value": "Over 0.5", "odd": "1.25"}, {"value": "Under 0.5", "odd": "3.75"}]}
CORNERS_HT = {"id": 77, "name": "Total Corners First Half Over/Under", "values": [
    {"value": "Over 0.5", "odd": "1.01"}]}


def test_first_half_prices_are_captured_with_the_standard_keys():
    out = NF._parse_all_odds(_answer([FT, HT]))
    assert (out["ht_over05"], out["ht_under05"]) == (1.40, 2.75)
    assert (out["ht_over15"], out["ht_under15"]) == (3.00, 1.36)
    std = STD._parse_all_odds(_answer([FT, HT]))
    for k in ("ht_over05", "ht_under05", "ht_over15", "ht_under15"):
        assert out[k] == std[k]


def test_full_time_lines_are_unchanged_by_the_half_time_market():
    a = NF._parse_all_odds(_answer([FT]))
    b = NF._parse_all_odds(_answer([HT, FT]))
    c = NF._parse_all_odds(_answer([FT, HT]))
    for k in ("over15", "under15", "over25", "under25", "over35", "under35"):
        assert a[k] == b[k] == c[k]


def test_second_half_and_corner_half_markets_are_not_first_half_goals():
    out = NF._parse_all_odds(_answer([FT, H2, CORNERS_HT]))
    assert not any(k.startswith("ht_") for k in out)


def test_no_half_time_market_means_no_half_time_price():
    out = NF._parse_all_odds(_answer([FT]))
    assert not any(k.startswith("ht_") for k in out)
