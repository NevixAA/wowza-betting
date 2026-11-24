"""ONE table for all leagues (owner 2026-11-24: "why it's a message? why no one table for all today's top 3?").
TOP-3 notify sends every newly frozen pick of every league in ONE Telegram table per run, grouped by kickoff day
(Israel time); split only above 4096 characters; once per pick; never after kickoff. Fake sender only."""
import pandas as pd

from src import top3_league as T
from tests.test_top3_league import NOW, _cand, _ctx, _frame, _freeze

ENV = {T.FLAG_ENV: "1"}


def _two_leagues(tmp_path):
    p1, _ = _freeze(_frame([_cand(1, home="H1", away="A1"), _cand(2, "BTTS", "YES", ev_lb=0.01)]), _ctx())
    other = "Sweden Allsvenskan"
    p2, _ = _freeze(_frame([_cand(5, home="S1", away="S2")]).assign(league=other, league_week_id=f"{other}|2026-W47"),
                    _ctx(league=other))
    T.picks_store(tmp_path).append(pd.concat([p1, p2], ignore_index=True))
    return pd.concat([p1, p2], ignore_index=True)


def test_one_message_for_every_league(tmp_path):
    picks = _two_leagues(tmp_path)
    sent = []
    r = T.notify(out_dir=tmp_path, env=ENV, sender=lambda t: sent.append(t) or True, now=NOW)
    assert r["sent"] == 1 and len(sent) == 1
    msg = sent[0]
    assert msg.count("<pre>") == 1 and "Israel time" in msg
    assert "La Liga 2" in msg and "Sweden Allsvenskan"[:13] in msg
    for h in picks["home_team"]:
        assert f"{h} v " in msg
    for col in ("Time", "League", "Match", "Tier", "Pick", "Odds", "Stake"):
        assert col in msg
    assert "paper" not in msg.lower()
    # once per pick: nothing new -> nothing sent
    assert T.notify(out_dir=tmp_path, env=ENV, sender=lambda t: sent.append(t) or True, now=NOW)["sent"] == 0
    assert len(sent) == 1


def test_never_sent_after_kickoff(tmp_path):
    _two_leagues(tmp_path)
    sent = []
    late = NOW + pd.Timedelta(days=30)
    assert T.notify(out_dir=tmp_path, env=ENV, sender=lambda t: sent.append(t) or True, now=late)["sent"] == 0
    assert sent == []


def test_a_failed_send_is_retried_next_run(tmp_path):
    _two_leagues(tmp_path)
    assert T.notify(out_dir=tmp_path, env=ENV, sender=lambda t: False, now=NOW)["sent"] == 0
    sent = []
    assert T.notify(out_dir=tmp_path, env=ENV, sender=lambda t: sent.append(t) or True, now=NOW)["sent"] == 1


def test_split_only_above_the_telegram_limit_and_grouped_by_day():
    base = pd.Timestamp("2026-11-28T12:00:00Z")
    rows = [{"ko": base + pd.Timedelta(hours=6 * i), "league": f"League {i % 7}", "match": f"Home {i} v Away {i}",
             "tier": "T1", "pick": "BTTS Yes", "odds": "2.00", "stake": "0.5% stake", "pick_id": f"p{i}",
             "week": "w"} for i in range(200)]
    msgs = T.combined_messages(rows, foot="legend")
    assert len(msgs) > 1
    assert all(len(t) <= T.TG_LIMIT for t, _ids in msgs)
    assert sorted(i for _t, ids in msgs for i in ids) == sorted(r["pick_id"] for r in rows)   # every pick, once
    assert "legend" in msgs[-1][0] and "legend" not in msgs[0][0]
    assert all("──" in t.split("<pre>")[1].splitlines()[1] for t, _ in msgs)                 # day header on top
    one = T.combined_messages(rows[:5])
    assert len(one) == 1 and "Sat 28 Nov" in one[0][0]                                      # Israel day


def test_israel_time_and_af_kickoff_preferred():
    r = {"af_kickoff_utc": "2026-11-29T00:00:00Z", "kickoff_utc": "2026-11-28T15:00:00Z"}
    assert T.pick_kickoff(r) == pd.Timestamp("2026-11-29T00:00:00Z")
    rows = [{"ko": T.pick_kickoff(r), "league": "Argentina Primera Division", "match": "Rosario Central v Barracas",
             "tier": "T1", "pick": "BTTS Yes", "odds": "2.00", "stake": "-", "pick_id": "x", "week": "w"}]
    text = T.combined_messages(rows)[0][0]
    assert "Sun 29 Nov" in text and "02:00" in text                                         # UTC+2


# ── Admin Board: the all-leagues table on top of the TOP-3 tab ──────────────────────────────────────────────────
def test_admin_board_upcoming_table_sorted_by_kickoff_with_day_filter():
    from dashboard_data import admin_board as AB
    led = pd.DataFrame([
        {"pick_id": "a", "league": "La Liga 2", "league_week_id": "La Liga 2|2026-W48", "home_team": "H1",
         "away_team": "A1", "kickoff_utc": "2026-11-26T19:00:00Z", "market": "BTTS", "side": "YES",
         "slot": "T1_BEST_VALUE", "top3_rank": 1, "stake": "0.005", "stake_reason": "STAKED", "result": "",
         "settle_status": ""},
        {"pick_id": "b", "league": "Brazil Serie A", "league_week_id": "Brazil Serie A|2026-W48", "home_team": "H2",
         "away_team": "A2", "kickoff_utc": "2026-11-24T23:00:00Z", "market": "OU25", "side": "OVER",
         "slot": "T2_SAFEST", "top3_rank": 2, "stake": "0", "stake_reason": "NO_STAKE_BELOW_BAR", "result": "WIN",
         "settle_status": "SETTLED"},
        {"pick_id": "c", "league": "Argentina Primera Division", "league_week_id": "x|2026-W48", "home_team": "H3",
         "away_team": "A3", "kickoff_utc": "2026-11-28T15:00:00Z", "af_kickoff_utc": "2026-12-09T00:00:00Z",
         "market": "BTTS", "side": "YES", "slot": "T3_RISKY_VALUE", "top3_rank": 3, "stake": "0.005",
         "stake_reason": "STAKED", "result": "", "settle_status": ""}])
    now = pd.Timestamp("2026-11-24T10:00:00Z")                            # Tue 24 Nov, Israel 12:00
    week = AB.upcoming_top3(led, {"a": 2.1}, now, "Next 7 days")
    assert list(week["Match"]) == ["H2 v A2", "H1 v A1"]                  # c: AF kickoff 9 Dec, outside 7 days
    assert list(week.columns) == ["Kickoff (Israel)", "League", "Match", "Tier", "Pick", "Odds at tip", "Stake",
                                  "Result"]
    assert week.iloc[0]["Kickoff (Israel)"] == "Wed 25 Nov 01:00" and week.iloc[0]["Result"] == "WIN"
    today = AB.upcoming_top3(led, None, now, "Today")
    assert today.empty                                                  # 23:00Z = 01:00 Israel next day
    assert AB.upcoming_top3(pd.DataFrame(), None, now).empty
