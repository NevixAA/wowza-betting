"""A player who drops off the board after the deadline must never be settled at 0 (GW7 2026-10: Foden 9 -> 0)."""
import json

import pandas as pd

from player_model import fantasy_settle as fs


def test_player_without_a_post_match_snapshot_is_dropped_not_zeroed(tmp_path, monkeypatch):
    out = tmp_path
    (out / "fpl_bootstrap.json").write_text(json.dumps({"events": [
        {"id": 7, "deadline_time": "2026-10-17T10:00:00Z", "finished": True, "data_checked": True},
        {"id": 8, "deadline_time": "2026-10-24T10:00:00Z", "finished": False, "data_checked": False}]}), encoding="utf-8")
    (out / "fpl_fixtures.json").write_text(json.dumps([
        {"event": 7, "kickoff_time": "2026-10-17T11:30:00Z"}, {"event": 7, "kickoff_time": "2026-10-19T19:00:00Z"}]),
        encoding="utf-8")
    rows = []
    for ts in ("2026-10-16T12:00:00Z", "2026-10-17T11:09:00Z"):          # before deadline, then after it but pre-match
        rows += [{"snapshot_ts": ts, "player_name": "Phil Foden", "fpl_total_points": 9},
                 {"snapshot_ts": ts, "player_name": "Erling Haaland", "fpl_total_points": 30}]
    rows.append({"snapshot_ts": "2026-10-21T12:00:00Z", "player_name": "Erling Haaland", "fpl_total_points": 38})
    pd.DataFrame(rows).to_csv(out / "fantasy_projection_log.csv", index=False)
    monkeypatch.setattr(fs, "OUT", out)
    monkeypatch.setattr(fs, "LOG", out / "fantasy_projection_log.csv")
    led, meta = fs.build_ledger()
    got = dict(zip(led.player_name, led.actual_points))
    assert got == {"Erling Haaland": 8}                                  # Foden dropped, not 0
    assert any("dropped" in n for n in meta["notes"])
