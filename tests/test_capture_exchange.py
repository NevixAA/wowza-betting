"""Exchange capture: our leagues only, both sides of each price kept, book key logged."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import capture_exchange_odds as X  # noqa: E402


def test_only_enabled_leagues_are_asked():
    import config
    lg = X.our_leagues()
    assert lg and set(lg) <= set(config.ENABLED_LEAGUES)
    assert "World Cup" not in lg


def test_back_and_lay_rows_keep_book_key_and_side():
    ev = {"id": "e1", "commence_time": "2026-10-21T18:45:00Z", "home_team": "Luton", "away_team": "Barnsley",
          "bookmakers": [{"key": "betfair_ex_uk", "title": "Betfair", "markets": [
              {"key": "h2h", "outcomes": [{"name": "Luton", "price": 2.1}]},
              {"key": "h2h_lay", "outcomes": [{"name": "Luton", "price": 2.16}]},
              {"key": "alternate_totals_lay", "outcomes": [{"name": "Under", "point": 2.5, "price": 2.0}]}]}]}
    rows = X.rows_from_event(ev, "League One", "soccer_england_league1", "2026-10-20T09:00:00Z")
    assert len(rows) == 3
    assert {r["market"] for r in rows} == {"h2h", "h2h_lay", "alternate_totals_lay"}
    assert all(r["bookmaker_key"] == "betfair_ex_uk" for r in rows)       # "Betfair" title alone is ambiguous
    lay = next(r for r in rows if r["market"] == "alternate_totals_lay")
    assert lay["outcome"] == "Under" and lay["point"] == 2.5
    assert set(rows[0]) == set(X.FIELDS)
