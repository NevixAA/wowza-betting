"""P7 item 2 (2026-11-09): the captured first-half two-sided prices are joined into the bars."""
import pandas as pd

from src import league_bars as LB
from src.bars_opt import study as S, universe as U


def test_ht_markets_are_priced_markets_with_their_captured_keys():
    assert LB.SNAPSHOT_KEYS["ht_over05"] == ("ht_over05", "ht_under05")
    assert LB.SNAPSHOT_KEYS["ht_over15"] == ("ht_over15", "ht_under15")
    assert {"ht_over05", "ht_over15"} <= set(U.PRICED_MARKETS)
    assert S.MARKET_SIDES["ht_over05_over"] == ("ht_over05", "yes") and "ht_over05_over" in S.OVER_DIRECTION


def test_ht_pairs_follow_the_same_pre_kickoff_same_snapshot_rule():
    raw = pd.DataFrame([
        # same snapshot, before kickoff -> a pair
        {"snapshot_ts": "2026-09-01T10:00:00Z", "match_date": "2026-09-01", "kickoff_utc": "2026-09-01T18:00:00Z",
         "league": "Championship", "match": "A vs B", "market": "ht_over05", "odds": 1.40},
        {"snapshot_ts": "2026-09-01T10:00:00Z", "match_date": "2026-09-01", "kickoff_utc": "2026-09-01T18:00:00Z",
         "league": "Championship", "match": "A vs B", "market": "ht_under05", "odds": 2.75},
        # after kickoff -> never a price
        {"snapshot_ts": "2026-09-01T19:00:00Z", "match_date": "2026-09-01", "kickoff_utc": "2026-09-01T18:00:00Z",
         "league": "Championship", "match": "C vs D", "market": "ht_over05", "odds": 1.01},
        {"snapshot_ts": "2026-09-01T19:00:00Z", "match_date": "2026-09-01", "kickoff_utc": "2026-09-01T18:00:00Z",
         "league": "Championship", "match": "C vs D", "market": "ht_under05", "odds": 15.0},
    ])
    p = LB.snapshot_pairs(raw, "standard_sidemarket_odds_history", "ht_over05")
    assert list(p["match"]) == ["A vs B"] and (p["yes"].iloc[0], p["no"].iloc[0]) == (1.40, 2.75)
