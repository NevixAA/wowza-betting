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


def test_ht_families_are_study_only_never_served(tmp_path, monkeypatch):
    """Production tips no HT market: the HT families stay out of the serving cells (no KeyError, no proposal)."""
    import pandas as pd
    from src.bars_opt import cells as C, study as S
    fams = [(pd.DataFrame({"date": []}), "standard", "over25_over"),
            (pd.DataFrame({"date": []}), "standard", "ht_over05_over"),
            (pd.DataFrame({"date": []}), "standard", "ht_over15_over")]
    monkeypatch.setattr(S, "load_families", lambda d: fams)
    assert C.not_served_families(tmp_path) == [("standard", "ht_over05_over"), ("standard", "ht_over15_over")]
    seen = []
    monkeypatch.setattr(C, "family_cells", lambda a: seen.append(a[2]) or {"mside": a[2]})
    C.build(tmp_path, "2026-11", workers=1, verdict_path=tmp_path / "none.json")
    assert seen == ["over25_over"]
