"""TOP-3 re-seed (owner 2026-11-20 "Re-seed for this weekend"; src/top3_league.reseed).

The old candidates are archived with their bytes unchanged, the live edge shadow keeps every line except the
unplayed fixtures' rows first seen before the re-seed instant (archived byte-for-byte first), the re-seed runs
once, and it fails closed (writes nothing) without the flag, a coherent config, or before its instant.
Synthetic files in tmp_path only."""
from __future__ import annotations

import json

import pandas as pd

from src import top3_league as T

AT = "2026-11-20T13:00:00Z"
ENV = {T.FLAG_ENV: "true"}
HEAD = b"selection_engine_version,prediction_ts,fixture_id,kickoff_utc,market,books_used\n"
PLAYED = b"market_edge_v2,2026-11-14T09:00:00Z,f1,2026-11-19T19:00:00Z,OU25,\"a;b\"\n"
UNPLAYED_OLD = b"market_edge_v2,2026-11-15T09:00:00Z,f2,2026-11-21T15:00:00Z,OU25,a\n"
UNPLAYED_NEW = b"market_edge_v2,2026-11-20T13:20:00Z,f3,2026-11-21T15:00:00Z,BTTS,a\n"
CANDS = b"fixture_id,first_seen_at\nf2,2026-11-15T09:00:00Z\n"


def _setup(tmp_path, start=AT, at=AT, tag="pre_reseed_2026-11-20"):
    root, od = tmp_path / "root", tmp_path / "out"
    (root / "config").mkdir(parents=True)
    od.mkdir()
    cfg = {"start_utc": start}
    if at is not None:
        cfg["reseed"] = {"at_utc": at, "archive_tag": tag}
    (root / T.START_CONFIG).write_text(json.dumps(cfg), encoding="utf-8")
    (od / T.SHADOW_NAME).write_bytes(HEAD + PLAYED + UNPLAYED_OLD + UNPLAYED_NEW)
    (od / T.CANDIDATES_NAME).write_bytes(CANDS)
    return root, od


def _snapshot(od):
    return {p.name: p.read_bytes() for p in od.iterdir()}


def test_reseed_archives_bytes_and_drops_only_unplayed_pre_instant_rows(tmp_path):
    root, od = _setup(tmp_path)
    before = (od / T.SHADOW_NAME).read_bytes()
    res = T.reseed(now="2026-11-20T13:30:00Z", out_dir=od, root=root, env=ENV)
    assert res["status"] == "RESEEDED" and res["shadow_rows_archived_out"] == 1 and res["shadow_rows_kept"] == 2
    assert (od / "edge_engine_shadow_pre_reseed_2026-11-20.csv").read_bytes() == before
    assert (od / T.SHADOW_NAME).read_bytes() == HEAD + PLAYED + UNPLAYED_NEW      # kept lines byte-identical
    assert (od / "top3_league_candidates_pre_reseed_2026-11-20.csv").read_bytes() == CANDS
    assert not (od / T.CANDIDATES_NAME).exists()


def test_reseed_runs_once(tmp_path):
    root, od = _setup(tmp_path)
    T.reseed(now="2026-11-20T13:30:00Z", out_dir=od, root=root, env=ENV)
    (od / T.CANDIDATES_NAME).write_bytes(b"fixture_id,first_seen_at\nf2,2026-11-20T13:35:00Z\n")
    snap = _snapshot(od)
    assert T.reseed(now="2026-11-20T14:00:00Z", out_dir=od, root=root, env=ENV)["status"] == "ALREADY_DONE"
    assert _snapshot(od) == snap


def test_reseed_fails_closed(tmp_path):
    cases = [({}, {}, "2026-11-20T13:30:00Z", None),                              # flag off
             ({"at": None}, ENV, "2026-11-20T13:30:00Z", "NO_RESEED_CONFIGURED"),
             ({"at": "2026-11-20T12:00:00Z"}, ENV, "2026-11-20T13:30:00Z", "RESEED_NOT_AT_START"),
             ({"tag": "../x"}, ENV, "2026-11-20T13:30:00Z", "RESEED_CONFIG_INVALID"),
             ({}, ENV, "2026-11-20T12:59:00Z", "BEFORE_RESEED_INSTANT")]
    for i, (kw, env, now, status) in enumerate(cases):
        root, od = _setup(tmp_path / str(i), **kw)
        snap = _snapshot(od)
        res = T.reseed(now=now, out_dir=od, root=root, env=env)
        assert res.get("status") == status
        assert _snapshot(od) == snap                                              # nothing written


def test_reseed_refuses_a_row_it_cannot_read(tmp_path):
    root, od = _setup(tmp_path)
    (od / T.SHADOW_NAME).write_bytes(HEAD + b"market_edge_v2,\"2026-11-15T09:00:00Z\n,f2,x\n")
    snap = _snapshot(od)
    try:
        T.reseed(now="2026-11-20T13:30:00Z", out_dir=od, root=root, env=ENV)
        raised = False
    except ValueError:
        raised = True
    assert raised and _snapshot(od) == snap


def test_committed_reseed_config_is_at_the_top3_start():
    cfg = json.loads((T.ROOT / T.START_CONFIG).read_text(encoding="utf-8"))
    rs = cfg["reseed"]
    assert pd.Timestamp(rs["at_utc"]) == pd.Timestamp(cfg["start_utc"]) == T.top3_start()
    assert pd.Timestamp(cfg["freeze_window_from_utc"]) < pd.Timestamp(cfg["start_utc"])
    assert pd.Timestamp(cfg["stats_as_of_utc"]) < pd.Timestamp(cfg["start_utc"])
    assert "Re-seed for this weekend" in rs["decision"]
