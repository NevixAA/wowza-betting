"""Standard-league team cards (scripts/collect_standard_team_cards.py), owner 2026-11-16 (audit gap G6),
and the af_history fixture_id column (audit G11)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _mod(name):
    spec = importlib.util.spec_from_file_location(name + "_tc", ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


TC = _mod("collect_standard_team_cards")


def _fx(fid, lid=40, status="FT", date="2026-11-16T15:00:00+00:00"):
    return {"fixture": {"id": fid, "date": date, "status": {"short": status}},
            "league": {"id": lid, "season": 2026},
            "teams": {"home": {"id": 1, "name": "Leeds"}, "away": {"id": 2, "name": "Hull"}},
            "goals": {"home": 2, "away": 1}}


def _block(tid, yellow, red, corners=5):
    return {"team": {"id": tid}, "statistics": [
        {"type": "Corner Kicks", "value": corners}, {"type": "Yellow Cards", "value": yellow},
        {"type": "Red Cards", "value": red}, {"type": "Ball Possession", "value": "55%"}]}


def test_17_standard_leagues():
    lg = TC.leagues()
    assert len(lg) == 17 and lg[40] == "Championship" and lg[43] == "National League"


def test_owner_null_rule_exactly():
    # answered block: null count -> 0; sent value kept
    r = TC.parse_row(_fx(1), [_block(1, 3, None), _block(2, None, 1)], "Championship", "2026-11-17T06:00:00Z")
    assert (r["HY"], r["HR"], r["AY"], r["AR"]) == (3, 0, 0, 1)
    assert r["fixture_id"] == 1 and r["fetched_at"] == "2026-11-17T06:00:00Z"
    assert r["home_answered"] and r["away_answered"]
    # one side not answered (all nulls) -> that side missing, never 0
    r = TC.parse_row(_fx(1), [_block(1, 2, 0), {"team": {"id": 2}, "statistics": [
        {"type": "Yellow Cards", "value": None}, {"type": "Red Cards", "value": None}]}], "Championship", "t")
    assert (r["HY"], r["HR"]) == (2, 0) and r["AY"] is None and r["AR"] is None and not r["away_answered"]
    # a type the answered block does not list -> missing
    r = TC.parse_row(_fx(1), [{"team": {"id": 1}, "statistics": [{"type": "Corner Kicks", "value": 4}]},
                              _block(2, 1, 0)], "Championship", "t")
    assert r["HY"] is None and r["HR"] is None
    # nothing answered -> no row at all
    assert TC.parse_row(_fx(1), [], "Championship", "t") is None
    assert TC.parse_row(_fx(1), None, "Championship", "t") is None


def test_window_never_before_collect_from():
    assert TC.window("2026-11-16") == ["2026-11-16"]
    assert TC.window("2026-11-20") == ["2026-11-18", "2026-11-19", "2026-11-20"]


class _FakeClient:
    def __init__(self, by_date, stats, max_calls=100):
        self.by_date, self.stats, self.calls, self.stopped, self.max_calls = by_date, stats, 0, "", max_calls
        self.log = []

    def get(self, endpoint, params):
        if self.calls >= self.max_calls:
            self.stopped = "call cap"
            return None
        self.calls += 1
        self.log.append((endpoint, dict(params)))
        if endpoint == "fixtures":
            return {"response": self.by_date.get(params["date"], [])}
        return {"response": self.stats.get(params["fixture"], [])}


def test_run_collects_only_new_finished_standard_fixtures(monkeypatch, tmp_path):
    out = tmp_path / "cards.parquet"
    monkeypatch.setattr(TC, "OUTPUT", out)
    by_date = {"2026-11-17": [_fx(11), _fx(12, lid=71), _fx(13, status="NS"), _fx(14, lid=179), _fx(15)],
               "2026-11-18": [_fx(16, lid=62)]}
    stats = {11: [_block(1, 2, None), _block(2, 1, 0)], 14: [_block(1, None, None), _block(2, 4, 1)],
             16: [_block(1, 0, 0), _block(2, 0, 0)]}   # 15: no statistics answered
    c = _FakeClient(by_date, stats)
    r = TC.run(today="2026-11-18", client=c)
    # window 11-16..11-18 -> 3 date calls, then 1 stats call per new finished Standard fixture
    assert [e for e, _ in c.log].count("fixtures") == 3
    assert sorted(p["fixture"] for e, p in c.log if e == "fixtures/statistics") == [11, 14, 15, 16]
    assert r["rows"] == 3 and r["unanswered"] == 1
    d = pd.read_parquet(out)
    assert sorted(d["fixture_id"]) == [11, 14, 16]
    assert set(d["league"]) == {"Championship", "Scottish Premiership", "Ligue 2"}
    assert d.set_index("fixture_id").loc[14, ["HY", "HR", "AY", "AR"]].tolist() == [0, 0, 4, 1]
    assert d["fetched_at"].notna().all() and list(d.columns) == TC.COLUMNS
    # second run: stored fixtures are never re-fetched; the unanswered one is retried
    c2 = _FakeClient(by_date, stats)
    TC.run(today="2026-11-18", client=c2)
    assert [p["fixture"] for e, p in c2.log if e == "fixtures/statistics"] == [15]
    assert len(pd.read_parquet(out)) == 3


def test_call_cap_respected(monkeypatch, tmp_path):
    out = tmp_path / "cards.parquet"
    monkeypatch.setattr(TC, "OUTPUT", out)
    monkeypatch.setattr(TC, "stored_ids", lambda path=out: set())
    by_date = {"2026-11-18": [_fx(i) for i in range(100, 120)]}
    c = _FakeClient(by_date, {i: [_block(1, 1, 0), _block(2, 1, 0)] for i in range(100, 120)}, max_calls=6)
    TC.run(today="2026-11-18", client=c, dry_run=True)
    assert c.calls == 6 and not out.exists()


def test_workflow_gated_and_staged():
    t = (ROOT / ".github" / "workflows" / "af_history_extend.yml").read_text(encoding="utf-8")
    assert "if: vars.STD_TEAM_CARDS_ENABLED == 'true'" in t
    assert "python scripts/collect_standard_team_cards.py --max-calls" in t
    assert "git add -f output/af_team_cards_std.parquet" in t


def test_af_history_rows_carry_fixture_id():
    B = _mod("backfill_af_history")
    row = B.parse_fixture("Brazil Serie A", "2026", {
        "fixture": {"id": 777, "date": "2026-11-15T22:00:00+00:00"},
        "teams": {"home": {"name": "A"}, "away": {"name": "B"}}, "goals": {"home": 1, "away": 0}}, None)
    assert row["fixture_id"] == 777
    assert row["HY"] is None                          # unanswered stats stay missing
