"""Go-live rehearsal 2026-11-15, defects 3 and 4 (src/top3_league.py).

3. No go-live gate: the first run decided 44 past league-weeks back to W43, issuing picks long after their
   freeze from candidates first seen before go-live. Now only league-weeks whose FREEZE is at/after the
   TOP-3 start (config/top3_paper_tracks.json start_utc) are decided, and a candidate first seen before the
   start is never a member. run() without a start freezes nothing.
4. notify sent a USA MLS (paper league) pick, and a blanket "Paper track" footer sat above the staked
   Argentina league-rule line. Paper leagues are dropped before sending; each line is labelled by its cell.
"""
from __future__ import annotations

import json

import pandas as pd

import config
from src import top3_league as T
from tests.test_top3_league import (LEAD, LOCK, NOW, WEEK, _cand, _ctx, _frame, _freeze, _lw_ctx,  # noqa: F401
                                    _stats)

FREEZE = LOCK - pd.Timedelta(minutes=LEAD)          # 2026-11-20T17:00Z


def _gated(rows, start, ctx=None, now=NOW, stats=None):
    return T.freeze_due(_frame(rows), set(), ctx or _ctx(), now, LEAD, {}, {}, None, stats, start)


def test_week_freezing_before_the_start_is_never_decided():
    picks, weeks = _gated([_cand(1)], start=FREEZE + pd.Timedelta(minutes=1))
    assert picks.empty and weeks.empty
    # a run days later still cannot back-fill it
    picks, weeks = _gated([_cand(1)], start=FREEZE + pd.Timedelta(minutes=1), now=NOW + pd.Timedelta(days=3))
    assert picks.empty and weeks.empty


def test_week_freezing_at_or_after_the_start_is_decided():
    picks, weeks = _gated([_cand(1)], start=pd.Timestamp("2026-11-19T00:00:00Z"))
    assert list(picks["fixture_id"]) == ["fx1"] and len(weeks) == 1
    assert (pd.to_datetime(picks["issued_at"], utc=True) >= pd.to_datetime(picks["freeze_ts"], utc=True)).all()


def test_candidate_first_seen_before_the_start_is_not_a_member():
    rows = [_cand(1, ev_lb=0.09, fs="2026-11-18T10:00:00Z"), _cand(2, fs="2026-11-19T12:00:00Z")]
    picks, weeks = _gated(rows, start=pd.Timestamp("2026-11-19T00:00:00Z"))
    assert list(picks["fixture_id"]) == ["fx2"]
    assert json.loads(weeks["excluded"].iloc[0])[T.R_PRE_START] >= 1


def test_start_is_read_from_the_pinned_top3_config_without_default(tmp_path):
    cfg = json.loads((T.ROOT / T.START_CONFIG).read_text(encoding="utf-8"))
    assert T.top3_start() == pd.Timestamp(cfg["start_utc"])
    assert T.top3_start(tmp_path) is None                              # no config -> no start, no default


def test_run_without_a_start_freezes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "top3_start", lambda root=None: None)
    monkeypatch.setattr(T, "load_t3_stats", lambda root=None: _stats({}))
    shadow = pd.DataFrame([{"market": "OU25", "selection_engine_version": "market_edge_v2"}])
    shadow.to_csv(tmp_path / "edge_engine_shadow.csv", index=False)
    env = {T.FLAG_ENV: "true", T.LEAD_ENV: "120"}
    res = T.run(now=NOW, out_dir=tmp_path, env=env, committed={})
    assert res.get("status") == "TOP3_START_NOT_SET" or "error" in res, res
    assert not (tmp_path / T.PICKS_NAME).exists()


# ── defect 4 ────────────────────────────────────────────────────────────────────────────────────
def test_notify_never_sends_a_paper_league(tmp_path, monkeypatch):
    paper = sorted(config.PAPER_LEAGUES)[0]
    p1, _ = _freeze(_frame([_cand(1, home="H1", away="A1")]), _ctx())
    pl_rows = [_cand(5, league=paper, home="P1", away="P2")]
    p2, _ = _freeze(_frame(pl_rows).assign(league=paper, league_week_id=f"{paper}|2026-W47"), _ctx(league=paper))
    assert len(p2) == 1
    T.picks_store(tmp_path).append(pd.concat([p1, p2], ignore_index=True))
    sent: list[str] = []
    monkeypatch.setattr("src.direction_guard.real_track_checker", lambda: (lambda *a, **k: False))
    r = T.notify(out_dir=tmp_path, env={T.FLAG_ENV: "1"}, sender=lambda t: sent.append(t) or True)
    assert r["sent"] == 1 and len(sent) == 1
    assert paper not in sent[0] and "P1" not in sent[0]


def test_each_line_is_labelled_by_its_cells_stake_not_a_blanket_footer():
    rows = [_cand(i, "BTTS", "YES", pm=0.5, pf=0.5) for i in range(1, 3)] + [_cand(10, "OU25", "OVER")]
    picks, _ = _freeze(_frame(rows), _lw_ctx(), stats=_stats({"over25_over": 1.0}))
    real = lambda mt, mk, side, league=None: mk == "btts"            # noqa: E731 - the league rule's cell stakes
    msg = T.message(picks.assign(league_week_id=WEEK), real)
    rule_line = next(ln for ln in msg.splitlines() if "League rule" in ln)
    top_line = next(ln for ln in msg.splitlines() if ln.startswith("1."))
    assert "Staked" in rule_line and "Paper" not in rule_line
    assert "Paper" in top_line and "Staked" not in top_line
    assert not any(ln.startswith("<i>Paper track") for ln in msg.splitlines())
