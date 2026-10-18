"""Less spammy Telegram (owner 2026-10-18): new tips go in their day's morning album, not one by one;
Bet Builder moves into the album too."""
from datetime import datetime, timezone

import pandas as pd

from telegram_bot import daily_pack as dp
from telegram_bot import notifier as nt


def _t(h):
    return datetime(2026, 10, 18, h, 0, tzinfo=timezone.utc)


def test_instant_rule(monkeypatch):
    monkeypatch.delenv("TIPS_INSTANT_ALERTS", raising=False)
    assert not nt.instant_tip_allowed("2026-10-19", set(), _t(8))                    # future day -> album
    assert not nt.instant_tip_allowed("2026-10-18", set(), _t(8))                    # today, album still to come
    assert nt.instant_tip_allowed("2026-10-18", {"DIGEST|2026-10-18"}, _t(8))        # found after today's album
    assert nt.instant_tip_allowed("2026-10-18 19:30", set(), _t(13))                 # album never went out
    assert not nt.instant_tip_allowed("2026-10-19", {"DIGEST|2026-10-18"}, _t(13))
    monkeypatch.setenv("TIPS_INSTANT_ALERTS", "1")
    assert nt.instant_tip_allowed("2026-10-25", set(), _t(8))                        # old behaviour on request


def test_album_keys_match_the_instant_senders_keys(tmp_path, monkeypatch):
    import config
    from telegram_bot import tips_table as tt
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    pd.DataFrame([{"source": "live", "generated_at": "2026-10-18T05:00", "match_date": "2026-10-18",
                   "league": "Championship", "home_team": "Hull City", "away_team": "Leeds", "side": "OVER",
                   "odds": 1.9, "edge_pct": 6.0, "signal_tier": "SNIPER", "model_type": "standard"}]
                 ).to_csv(tmp_path / "bets_ledger.csv", index=False)
    pd.DataFrame([{"match_date": "2026-10-18", "league": "Brazil Serie A", "home_team": "Santos",
                   "away_team": "Fluminense", "market": "btts", "side": "YES", "odds": 1.95, "edge_pct": 5.0,
                   "signal_tier": "SNIPER", "model_type": "new_format", "model_prob": 0.6}]
                 ).to_csv(tmp_path / "side_bets_ledger.csv", index=False)
    keys = set(tt.todays_rows("2026-10-18")["notify_key"])
    # exactly the f-strings notify_new_snipers / notify_side_bets build for the same tips
    assert keys == {"2026-10-18|Hull City|Leeds|OVER", "SIDE|2026-10-18|Santos|Fluminense|btts"}


def test_bet_builder_image(monkeypatch):
    sel = pd.DataFrame([{"match_date": "2026-10-18", "league": "Brazil Serie A", "match": "Santos vs Fluminense",
                         "legs": "Over 3.5 goals (29%) · Hulk to score (52%)", "n_legs": 2,
                         "joint_probability": 0.0569, "fair_odds": 17.57, "independence_edge": 3.99, "combo_id": "x"}])
    monkeypatch.setattr(dp, "_pro_bet_builder", lambda: sel)
    pngs = dp.bet_builder_image("2026-10-18")
    assert len(pngs) == 1 and pngs[0][:4] == b"\x89PNG"
    assert dp.bet_builder_image("2026-10-19") == []                                  # other days: nothing
    monkeypatch.setattr(dp, "_pro_bet_builder", lambda: pd.DataFrame())
    assert dp.bet_builder_image("2026-10-18") == []
