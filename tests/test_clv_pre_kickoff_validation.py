"""P1 item 7 (2026-11-22) - a closing price is certified only when it was OBSERVED strictly before
kickoff, within 30 minutes of it, at a real decimal price (> 1.0). Never a post-kickoff quote, never
a stale quote relabelled as a close, never a suspended book's 1.0, never a default.

Covers the three places a close is judged: the shadow certifier (shadow_settlement.closing_price /
settle_frame), the last-seen heartbeat writer (odds_heartbeat.record) and the coverage measurement
(clv_collection_health.collection_section, ladder), plus the capture chain decision that keeps a
run alive across GitHub's cron gaps (capture_schedule.chain_due).
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
import yaml

from src import capture_schedule as cs
from src import clv_collection_health as cch
from src import odds_heartbeat as hb
from src import shadow_settlement as ss

KO = pd.Timestamp("2026-11-21T15:00:00Z")
MATCH = "Leeds United vs Norwich City"
LEAGUE = "Championship"


def _arch(rows, name="standard_sidemarket_odds_history.csv"):
    """[(name, frame)] from (minutes_before_kickoff, odds) pairs; negative = after kickoff."""
    recs = [{"snapshot_ts": (KO - pd.Timedelta(minutes=m)).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "match_date": "2026-11-21", "kickoff_utc": "2026-11-21T15:00:00Z",
             "league": LEAGUE, "match": MATCH, "market": "over25", "odds": o} for m, o in rows]
    return [(name, pd.DataFrame(recs))]


def _close(rows, **kw):
    idx = ss.index_archives(_arch(rows, **kw))
    return ss.closing_price(idx, "Leeds United", "Norwich City", "2026-11-21", KO, "over25",
                            league=LEAGUE, max_min=ss.CLOSE_MAX_MIN)


# ── the certifier ────────────────────────────────────────────────────────────────────────────
def test_a_post_kickoff_quote_is_never_the_close():
    c = _close([(45, 1.90), (-5, 3.40)])                 # in-play quote 5 min after kickoff
    assert c["minutes"] == 45.0 and c["odds"] == 1.90     # the last PRE-kickoff observation
    assert c["minutes"] > ss.CLOSE_MAX_MIN                # ...which settle_frame labels STALE


def test_only_post_kickoff_quotes_is_post_kickoff_only_not_a_close():
    c = _close([(-1, 1.90), (-20, 2.10)])
    assert c["status"] == ss.POST_KICKOFF_ONLY and np.isnan(c["odds"])


def test_a_quote_at_the_kickoff_second_is_not_pre_kickoff():
    c = _close([(0, 1.90)])
    assert c["status"] == ss.POST_KICKOFF_ONLY


def test_inside_thirty_minutes_certifies():
    c = _close([(240, 2.00), (8, 1.85)])
    assert c["status"] == ss.CLOSE_OK and c["minutes"] == 8.0 and c["odds"] == 1.85


def test_a_suspended_one_point_zero_quote_is_skipped_not_taken():
    # The last row is a suspended market (1.0) at T-2m: the close is the last REAL quote.
    c = _close([(25, 1.88), (2, 1.0)])
    assert c["odds"] == 1.88 and c["minutes"] == 25.0


def test_only_suspended_quotes_give_no_close():
    c = _close([(5, 1.0), (2, 1.0)])
    assert c["status"] == ss.NO_CLOSE and np.isnan(c["odds"])


def test_no_kickoff_no_close():
    idx = ss.index_archives(_arch([(5, 1.9)]))
    c = ss.closing_price(idx, "Leeds United", "Norwich City", "2026-11-21", None, "over25",
                         league=LEAGUE, max_min=ss.CLOSE_MAX_MIN)
    assert c["status"] == ss.NO_KICKOFF and np.isnan(c["odds"])


def test_a_stale_dense_archive_does_not_hide_a_certified_capture_close():
    stale = _arch([(180, 2.05)], name="standard_odds_history.csv")
    fresh = _arch([(4, 1.92)], name="standard_sidemarket_odds_history.csv")
    idx = ss.index_archives(stale + fresh)
    c = ss.closing_price(idx, "Leeds United", "Norwich City", "2026-11-21", KO, "over25",
                         league=LEAGUE, max_min=ss.CLOSE_MAX_MIN)
    assert c["status"] == ss.CLOSE_OK and c["source"] == "standard_sidemarket_odds_history.csv"


def test_settle_frame_never_certifies_a_stale_close():
    """End to end: a T-3h last observation is kept as last_pre_kickoff_odds only."""
    row = {"fixture_id": "e1", "market": "ou25", "logged_at": "2026-11-20T09:00:00+00:00",
           "match_date": "2026-11-21", "kickoff_utc": "2026-11-21 15:00:00", "league": LEAGUE,
           "model_type": "standard", "v9_side": "OVER", "entry_odds": "2.0",
           "home_team": "Leeds United", "away_team": "Norwich City", "result": "",
           "settle_status": ""}
    ev = [{"home_team": "Leeds", "away_team": "Norwich", "home_score": 2, "away_score": 1,
           "date_str": "2026-11-21", "event_id": ""}]
    ups, _ = ss.settle_frame(pd.DataFrame([row]).astype(str), providers=[("t", lambda lg: ev)],
                             archives=_arch([(180, 1.95)]),
                             now=datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc))
    assert len(ups) == 1
    f = ups[0]["fields"]
    assert f["close_status"] == ss.STALE_CLOSE
    assert f["closing_odds"] == "" and f["clv"] == ""
    assert f["last_pre_kickoff_odds"] != ""


# ── the heartbeat writer ─────────────────────────────────────────────────────────────────────
def _hb_row(ts, ko="2026-11-21T15:00:00Z", odds=1.9):
    return {"snapshot_ts": ts, "match_date": "2026-11-21", "kickoff_utc": ko, "league": LEAGUE,
            "match": MATCH, "market": "over25", "odds": odds}


def test_heartbeat_refuses_post_kickoff_unparseable_kickoff_and_suspended(tmp_path):
    n = hb.record("std_capture", [
        _hb_row("2026-11-21T15:01:00Z"),                  # after kickoff
        _hb_row("2026-11-21T14:55:00Z", ko="not-a-time"),  # kickoff unknown
        _hb_row("2026-11-21T14:56:00Z", odds=1.0),         # suspended
        _hb_row("2026-11-21T07:00:00Z"),                  # 8 h out: outside HB_MAX_H
    ], out_dir=tmp_path)
    assert n == 0 and not any(tmp_path.rglob("*.csv"))


def test_heartbeat_keeps_the_newest_pre_kickoff_observation(tmp_path):
    hb.record("std_capture", [_hb_row("2026-11-21T14:52:00Z")], out_dir=tmp_path)
    hb.record("std_capture", [_hb_row("2026-11-21T14:20:00Z")], out_dir=tmp_path)  # older
    got = hb.load(tmp_path)["std_capture"]
    assert list(got["snapshot_ts"]) == ["2026-11-21T14:52:00Z"]


# ── the coverage measurement and the ladder ─────────────────────────────────────────────────
def test_collection_counts_only_strictly_pre_kickoff_real_prices():
    arch = _arch([(400, 1.95), (3, 1.0), (-2, 2.6)])      # T-3m is suspended, T+2m is in-play
    c = cch.collection_section(arch, now=KO + pd.Timedelta(hours=5), window_days=7)
    assert c["fixture_markets"] == 1
    assert c["t30_pct"] == 0.0                            # neither bad quote counts as coverage
    assert c["ladder"] == {"fixture_markets": 1, "T-6h": 0, "T-3h": 0, "T-1h": 0,
                           "T-30m": 0, "T-10m": 0}


def test_ladder_by_league_market_is_cumulative_counts():
    a = _arch([(400, 1.95), (45, 1.9)]) + [("newformat_odds_history.csv", pd.DataFrame([
        {"snapshot_ts": (KO - pd.Timedelta(minutes=7)).strftime("%Y-%m-%dT%H:%M:%SZ"),
         "match_date": "2026-11-21", "kickoff_utc": "2026-11-21T15:00:00Z",
         "league": "Serie B", "match": "Modena vs Juve Stabia", "market": "btts_yes",
         "odds": 1.8}]))]
    c = cch.collection_section(a, now=KO + pd.Timedelta(hours=5), window_days=7)
    lad = c["ladder_by_league_market"]
    assert lad[LEAGUE]["over25"] == {"fixture_markets": 1, "T-6h": 1, "T-3h": 1, "T-1h": 1,
                                     "T-30m": 0, "T-10m": 0}
    assert lad["Serie B"]["btts_yes"]["T-10m"] == 1
    assert c["ladder"]["fixture_markets"] == 2 and c["ladder"]["T-30m"] == 1
    # counts only: no price anywhere in the ladder
    assert all(isinstance(v, int) for d in lad.values() for m in d.values() for v in m.values())


def test_shadow_section_reports_the_kickoff_week_timeline():
    def r(ko, mins, st="OK"):
        return {"result": "WIN", "close_status": st, "close_minutes_to_kickoff": mins,
                "kickoff_utc": ko, "league": LEAGUE, "market": "ou25",
                "model_type": "standard", "close_source": "standard_sidemarket_odds_history.csv"}
    df = pd.DataFrame([r("2026-10-10 15:00:00", 200.0, ss.STALE_CLOSE),
                       r("2026-11-21 15:00:00", 3.0)])
    s = cch.shadow_section(df)
    assert s["by_kickoff_week"]["2026-W41"]["with_t30_close"] == 0
    assert s["by_kickoff_week"]["2026-W47"]["with_t30_close"] == 1
    assert s["close_feed_identity"] == {ss.CROSS_FEED: 1, ss.NO_CLOSE_FEED: 1}


# ── the chain that keeps a capture alive across cron gaps ───────────────────────────────────
@pytest.mark.parametrize("ko,depth,want", [
    (19, 0, True),             # cap hit at T-19m of a kickoff (nf 2026-11-22 15:41)
    (213, 0, True),            # a slot a fresh run can still reach (std 2026-11-20 13:57)
    (cs.CAP_MIN + 1, 0, False),  # beyond a fresh run's reach: cron is the fallback
    (None, 0, False),          # a failed pass printed no NEXT_KO_MIN: never chain blind
    (0, 0, False),
    (30, cs.CHAIN_MAX_DEPTH, False),  # depth cap: no runaway chain
])
def test_chain_due(ko, depth, want):
    ok, why = cs.chain_due(ko, depth)
    assert ok is want, why


def test_chain_cli_prints_machine_readable_lines(capsys):
    cs.main(["chain-due", "--next-ko-min", "40", "--depth", "2"])
    out = capsys.readouterr().out.splitlines()
    assert "CHAIN=1" in out and "CHAIN_DEPTH_NEXT=3" in out
    cs.main(["chain-due", "--next-ko-min", "", "--depth", ""])
    assert "CHAIN=0" in capsys.readouterr().out.splitlines()


@pytest.mark.parametrize("wf", ["std_odds_capture.yml", "nf_odds_capture.yml"])
def test_chained_dispatch_is_near_mode_and_switchable(wf):
    """The chain step dispatches THIS workflow in NEAR mode (max_hours=12 - never a forced WIDE
    pass), after the commit, behind a kill switch, with the actions permission it needs."""
    text = open(f".github/workflows/{wf}", encoding="utf-8").read()
    y = yaml.safe_load(text)
    job = next(iter(y["jobs"].values()))
    assert job["permissions"].get("actions") == "write"
    names = [s.get("name", "") for s in job["steps"]]
    chain = job["steps"][names.index("Chain the next NEAR run")]
    assert names.index("Chain the next NEAR run") == len(names) - 1
    assert "CAPTURE_CHAIN_DISABLED != 'true'" in chain["if"]
    assert f"gh workflow run {wf}" in chain["run"] and "-f max_hours=12" in chain["run"]
    cap = next(s for s in job["steps"] if s.get("id") == "capture")
    assert "chain-due" in cap["run"]
