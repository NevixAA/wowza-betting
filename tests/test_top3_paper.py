"""TOP-3 paper track R2 (src/research/top3_paper.py; owner 2026-11-11 "Drop T1") and the statistics freeze (scripts/top3_paper_freeze.py).

The owner's pool rule (clean retrained models + approved per-league cells only), the selection rules (cap 3,
never padded, direction rule, started-before-issue), hash pinning, and the guarantee that nothing reaches
staking, execution or the public tips."""
from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.research import top3_paper as P
from src.research import top3_ranking as R

ROOT = Path(__file__).resolve().parents[1]
CLEAN = "4025384d6cb2a2f3201aec60f3ade16e59a152f4369352a48be9ce77ec3fc3b9"     # second_refit standard
CLEAN_BTTS = "d7cd27622674f7670867148c01397dfbd3ce20386c24b8e8fe8466b601d7a7e1"


def _cell(mside, league="L", lo=0.02):
    return {"mside": mside, "league": league, "track": "standard", "config": "RAW|EDGE|max-|min-", "source": "RAW",
            "measure": "EDGE", "mode": "MODEL", "valuable_th": lo, "marksman_th": lo + 0.02, "sniper_th": lo + 0.04,
            "odds_min": None, "odds_max": None, "source_params": {}, "verdict": "UNPROVEN"}


CELLS = {("ou25", "L", "over"): _cell("over25_over"), ("ou25", "L", "under"): _cell("over25_under"),
         ("btts", "L", "yes"): _cell("btts_yes")}


def _ctx(cells=CELLS):
    std, side = {}, {}
    for (mk, lg, sd), c in cells.items():
        (std.setdefault(lg, {"bars_opt": {}}) if mk == "ou25" else
         side.setdefault(mk, {}).setdefault(lg, {"bars_opt": {}}))["bars_opt"][sd] = c
    return {"standard": std, "side": side, "applied": {f"{k[0]}|{k[1]}" for k in cells}}


def _guards(cells=CELLS, pid="TP-X"):
    return {"clean": {"standard": {CLEAN}, "btts": {CLEAN_BTTS}},
            "champions": {"standard": {"model_version": "standard_20261109T211851Z_4025384d", "artifact_sha256": CLEAN}},
            "approved": {k: (dict(c), pid) for k, c in cells.items()},
            "applied": {f"{k[0]}|{k[1]}": pid for k in cells}}


CFG = {"tracks": list(R.PAPER_RULES), "start_utc": "2026-11-21T00:00:00Z", "protocol_sha256": "p" * 64}


def _cand(fx, market, side, p, fair, odds=2.0, ko="2026-11-21T15:00:00Z", ver="standard_20261109T211851Z_4025384d",
          fam="OU25", seen="2026-11-20T10:00:00Z", league="L", home=None):
    return {"league_week_id": f"{league}|2026-W47", "league": league, "model_type": "standard", "engine_market": fam,
            "prediction_engine_version": ver, "market": market, "side": side, "p_model_raw": p, "p_market_fair": fair,
            "odds": odds, "price_quality_state": "TWO_SIDED_MULTI_BOOK", "fixture_id": fx, "match_date": ko[:10],
            "home_team": home or f"H{fx}", "away_team": f"A{fx}", "kickoff_utc": ko, "first_seen_at": seen}


NOW = pd.Timestamp("2026-11-21T13:05:00Z")       # first kickoff 15:00, lead 120 -> freeze 13:00


def _stats():
    return pd.DataFrame([
        {"track": "standard", "league": "L", "mside": "over25_over", "n": 60, "roi": 0.1, "se": 0.05, "lcb": 0.002,
         "eb": 0.1, "lam": 0.9, "lam_se": 0.05, "lam_rows": 300},
        {"track": "standard", "league": "L", "mside": "btts_yes", "n": 60, "roi": 0.05, "se": 0.02, "lcb": 0.011,
         "eb": 0.05, "lam": 0.2, "lam_se": 0.05, "lam_rows": 300}])


def _run(cands, guards=None, committed=None, now=NOW, cfg=CFG):
    return P.freeze_due(pd.DataFrame(cands), set(), _ctx(), now, 120.0, committed or {}, _stats(),
                        guards or _guards(), cfg, "s" * 64)


# ── the owner's pool rule ───────────────────────────────────────────────────────────────────────
def test_only_clean_models_reach_the_pool():
    picks, weeks = _run([_cand("1", "OU25", "OVER", 0.60, 0.45, ver="standard_20261018T080423Z_64313878"),
                         _cand("2", "OU25", "OVER", 0.60, 0.45, ver="standard_repaired_20261028T123040Z_1f5805e6"),
                         _cand("3", "OU25", "OVER", 0.60, 0.45)])
    assert set(picks["fixture_id"]) == {"3"}
    ex = json.loads(weeks.iloc[0]["excluded"])
    assert ex[P.R_NOT_CLEAN] == 2


def test_champion_version_must_match_its_full_listed_hash():
    g = _guards()
    g["champions"]["standard"]["artifact_sha256"] = "4025384d" + "0" * 56      # same prefix, not the listed file
    picks, _ = _run([_cand("3", "OU25", "OVER", 0.60, 0.45)], guards=g)
    assert picks.empty


def test_only_the_approved_cell_reaches_the_pool():
    # the served bar differs from the approved one -> refused, never a stand-in
    g = _guards()
    g["approved"][("ou25", "L", "over")] = (_cell("over25_over", lo=0.03), "TP-X")
    picks, weeks = _run([_cand("3", "OU25", "OVER", 0.60, 0.45)], guards=g)
    assert picks.empty and json.loads(weeks.iloc[0]["excluded"])[P.R_CELL] == 1
    # applied from another proposal -> refused
    g = _guards()
    g["applied"]["ou25|L"] = "TP-OLD"
    picks, weeks = _run([_cand("3", "OU25", "OVER", 0.60, 0.45)], guards=g)
    assert picks.empty and json.loads(weeks.iloc[0]["excluded"])[P.R_NOT_APPLIED_FROM] == 1


def test_off_or_legacy_bar_gives_no_candidate():
    ctx = _ctx({("btts", "L", "yes"): _cell("btts_yes")})
    ctx["standard"] = {"L": {"valuable_th": 0.0, "marksman_th": 0.01, "sniper_th": 0.02}}   # legacy, non-cell bar
    ctx["applied"].add("ou25|L")
    picks, weeks = P.freeze_due(pd.DataFrame([_cand("3", "OU25", "OVER", 0.60, 0.45)]), set(), ctx, NOW, 120.0, {},
                                _stats(), _guards(), CFG, "s")
    # owner 2026-11-11 "Clean rebuild": production's league_bars already refuses a legacy (non-cell) entry
    # (NO_LEAGUE_BAR), before the paper module's own cell guard would (CELL_NOT_APPROVED)
    ex = json.loads(weeks.iloc[0]["excluded"])
    assert picks.empty and (ex.get(P.R_CELL) == 1 or ex.get("NO_LEAGUE_BAR") == 1)


def test_approved_proposal_file_must_verify(tmp_path):
    src = ROOT / "output/threshold_proposals/bars_oddsapi_hist/2026-11.json"
    d = json.loads(src.read_text(encoding="utf-8"))
    cfg = {"approved_proposals": [{"proposal_id": d["proposal_id"], "proposal_hash": d["proposal_hash"], "file": "p.json"}]}
    (tmp_path / "p.json").write_text(json.dumps(d), encoding="utf-8")
    cells, why = P.approved_cells(cfg, tmp_path)
    assert not why and cells
    d["entries"][0]["entry_hash"] = "0" * 64                        # tampered
    (tmp_path / "p.json").write_text(json.dumps(d), encoding="utf-8")
    cells, why = P.approved_cells(cfg, tmp_path)
    assert why and not cells


# ── selection ───────────────────────────────────────────────────────────────────────────────────
def test_cap_three_never_pad_direction_and_tracks_rank_differently():
    c = [_cand("1", "OU25", "OVER", 0.60, 0.45), _cand("1", "OU25", "UNDER", 0.70, 0.55),
         _cand("2", "BTTS", "YES", 0.60, 0.50, fam="BTTS", ver="btts_20261109T211852Z_d7cd2762"),
         _cand("4", "OU25", "OVER", 0.55, 0.50), _cand("5", "OU25", "OVER", 0.58, 0.50)]
    picks, weeks = _run(c)
    assert set(picks["track"]) == {"R2_CAL_EV"} and len(weeks) == 1         # T1 dropped: no R1 ledger
    for t, g in picks.groupby("track"):
        assert len(g) <= 3 and list(g["rank"]) == list(range(1, len(g) + 1))
        f1 = g[g["fixture_id"] == "1"]
        assert f1["side"].nunique() <= 1                            # never Over and Under on one fixture
    r2 = picks[picks["track"] == "R2_CAL_EV"].sort_values("rank")
    assert set(picks["rank_basis"]) <= set(R.FALLBACK_BASES)
    assert r2.iloc[0]["market"] == "OU25"                           # calibrated EV ranks the O/U gap first
    one, _ = _run([_cand("9", "OU25", "OVER", 0.60, 0.45)])
    assert one.groupby("track").size().max() == 1                   # never padded


def test_late_seen_started_and_before_start_are_not_picked():
    c = [_cand("1", "OU25", "OVER", 0.60, 0.45, seen="2026-11-21T13:01:00Z"),            # after freeze
         _cand("2", "OU25", "OVER", 0.60, 0.45, ko="2026-11-21T13:03:00Z")]              # kicked off before issue
    picks, _ = _run(c)
    assert picks.empty
    early = [_cand("3", "OU25", "OVER", 0.60, 0.45, ko="2026-11-15T15:00:00Z")]
    picks, weeks = _run(early, now=pd.Timestamp("2026-11-15T13:05:00Z"))
    assert picks.empty and weeks.empty                               # protocol start 2026-11-21


def test_direction_against_the_live_production_tip():
    com = {("2026-11-21", "H1", "A1"): {"direction": "UNDER"}}
    picks, _ = _run([_cand("1", "OU25", "OVER", 0.60, 0.45)], committed=com)
    assert picks.empty


def test_every_row_is_labelled_paper_and_never_staked():
    picks, weeks = _run([_cand("3", "OU25", "OVER", 0.60, 0.45)])
    assert (picks["label"] == P.LABEL).all() and (weeks["label"] == P.LABEL).all()
    assert picks["stake_effect"].str.startswith("NONE").all()
    assert "stake" not in [c for c in P.PICK_COLS if c != "stake_effect"]


# ── settlement and summary ──────────────────────────────────────────────────────────────────────
def test_settlement_grades_only_grading_columns_and_summary_has_no_prices():
    picks, _ = _run([_cand("3", "OU25", "OVER", 0.60, 0.45, odds=2.1)])
    v91 = pd.DataFrame({"fixture_id": ["3"], "match_date": ["2026-11-21"], "home_team": ["H3"], "away_team": ["A3"],
                        "home_goals": [2], "away_goals": [1], "result_source": ["test"], "settle_status": ["SETTLED"]})
    out, n = P.settle_picks(picks.astype(str), v91, pd.Timestamp("2026-11-23T00:00:00Z"))
    assert n == 1 and set(out["result"]) == {"WIN"}
    assert np.isclose(pd.to_numeric(out["flat_pnl"]).iloc[0], 1.1)
    s = P.summary(out, NOW, CFG, "OK")
    assert list(s["tracks"]) == ["R2_CAL_EV"] and s["tracks"]["R2_CAL_EV"]["settled"] == 1 and s["label"] == P.LABEL
    assert "odds" not in json.dumps(s).replace("odds_", "")


# ── nothing reaches money or the public ─────────────────────────────────────────────────────────
def test_flag_off_writes_nothing(tmp_path):
    assert P.run(out=tmp_path, env={}) == {"enabled": False}
    assert P.settle(out=tmp_path, env={}) == {"enabled": False}
    assert not any(tmp_path.rglob("*"))


def test_run_writes_only_under_the_research_folder(tmp_path):
    res = P.run(out=tmp_path, env={P.FLAG_ENV: "true", "TOP3_FREEZE_LEAD_MIN": "120"}, cands=pd.DataFrame(),
                committed={})
    assert res["enabled"] is True
    written = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert written and all(P.OUT_SUBDIR.as_posix() in p.relative_to(tmp_path).as_posix() for p in written)


FORBIDDEN_IMPORTS = ("telegram_bot", "src.stake", "src.cloudbet", "src.exec", "src.bankroll", "src.ledger",
                     "src.notifier", "src.betfair", "src.bet_builder")


def test_paper_module_imports_nothing_that_stakes_executes_or_publishes():
    tree = ast.parse((ROOT / "src/research/top3_paper.py").read_text(encoding="utf-8"))
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module)
            mods.update(f"{n.module}.{a.name}" for a in n.names)
    assert not [m for m in mods if m.startswith(FORBIDDEN_IMPORTS)]


def test_no_production_reader_touches_the_paper_files():
    allowed = {"src/research/top3_paper.py", "scripts/top3_paper_freeze.py", "scripts/top3_paper_dryrun.py",
               "tests/test_top3_paper.py",
               "src/research/top3_ranking.py",                      # names the paper module in its docstring only
               # the LIVE TOP-3 reads the ONE statistics freeze (top3_paper.load_freeze, hash-checked) - never a paper
               # ledger; the dry run reuses the freeze computation (owner 2026-11-11: one freeze, no fork)
               "src/top3_league.py", "scripts/top3_t3_dry_run.py"}
    hits = []
    for base in ("src", "telegram_bot", "scripts", "dashboard", "app.py", "streamlit_app.py"):
        p = ROOT / base
        files = [p] if p.is_file() else (list(p.rglob("*.py")) if p.is_dir() else [])
        for f in files:
            rel = f.relative_to(ROOT).as_posix()
            if rel in allowed:
                continue
            t = f.read_text(encoding="utf-8", errors="ignore")
            if "top3_paper" in t or "paper_ledger" in t:
                hits.append(rel)
    assert hits == []


LIVE_READERS = ("src/top3_league.py", "scripts/top3_t3_dry_run.py")
#: the only paper-module names a live reader may use: the pinned config and the hash-checked statistics freeze
LIVE_READER_ALLOWED = {"load_config", "load_freeze", "sha256_lf", "hash_problems"}


def test_live_readers_use_only_the_hash_checked_freeze():
    """The allow-list above is per file; this pins WHAT those files may use: the frozen statistics through
    load_freeze (protocol, ranking code and record hashes all checked) - never a paper ledger, a paper store,
    the paper run / settle, or the paper folder."""
    import re
    for rel in LIVE_READERS:
        f = ROOT / rel
        if not f.is_file():
            continue
        t = f.read_text(encoding="utf-8", errors="ignore")
        for bad in ("paper_ledger", "paper_weeks", "top3_paper/", "research/top3_paper"):
            assert bad not in t, (rel, bad)
        alias = set(re.findall(r"from src\.research import top3_paper as (\w+)", t))
        for a in alias:
            used = set(re.findall(rf"{a}\.(\w+)", t))
            assert used <= LIVE_READER_ALLOWED, (rel, used - LIVE_READER_ALLOWED)


def test_workflows_only_run_and_commit_the_paper_folder():
    for wf in (".github/workflows/predict.yml", ".github/workflows/update_results.yml"):
        t = (ROOT / wf).read_text(encoding="utf-8")
        for line in t.splitlines():
            if "top3_paper" in line and "git add" in line:
                assert "output/research/top3_paper/" in line


# ── hashes and the freeze ───────────────────────────────────────────────────────────────────────
def test_pinned_protocol_and_code_hashes_match():
    cfg = P.load_config()
    assert P.hash_problems(cfg) == []
    proto = json.loads((ROOT / cfg["protocol_file"]).read_text(encoding="utf-8"))
    assert [t["id"] for t in proto["tracks"]] == ["T2_R2_CAL_EV", "T3_R3_CAL_EV_LCB"]
    assert R.LIVE_RULE == "R3_CAL_EV_LCB" and R.PAPER_RULES == ("R2_CAL_EV",)
    assert R.TRACK_RULES == ("R2_CAL_EV", "R3_CAL_EV_LCB") and cfg["tracks"] == ["R2_CAL_EV"]


def test_a_changed_protocol_or_code_is_refused(tmp_path):
    cfg = P.load_config()
    for rel in (cfg["protocol_file"], "src/research/top3_ranking.py"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    assert P.hash_problems(cfg, tmp_path) == []
    with open(tmp_path / "src/research/top3_ranking.py", "a", encoding="utf-8") as f:
        f.write("\n# edited\n")
    assert P.hash_problems(cfg, tmp_path)


def test_unfrozen_or_tampered_statistics_stop_every_track(tmp_path):
    cfg = P.load_config() | {"freeze_ledger": "led.json", "freeze_record": "rec.json"}
    for rel in (cfg["protocol_file"], "src/research/top3_ranking.py"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    st, _, _, why = P.load_freeze(cfg, tmp_path)
    assert st is None and any("not frozen" in w for w in why)
    rec = {"protocol_sha256": cfg["protocol_sha256"], "cells": _stats().to_dict("records")}
    (tmp_path / "rec.json").write_text(json.dumps(rec), encoding="utf-8")
    (tmp_path / "led.json").write_text(json.dumps({"record": "rec.json", "sha256": P.sha256_lf(tmp_path / "rec.json")}),
                                       encoding="utf-8")
    st, _, _, why = P.load_freeze(cfg, tmp_path)
    assert why == [] and len(st) == 2
    (tmp_path / "rec.json").write_text(json.dumps(rec | {"cells": []}), encoding="utf-8")
    st, _, _, why = P.load_freeze(cfg, tmp_path)
    assert st is None and why


def test_real_freeze_refuses_outside_its_window():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/top3_paper_freeze.py"), "--now", "2026-11-11T12:00:00Z"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 2 and "window" in r.stdout


def test_frozen_statistics_roundtrip():
    st = _stats()
    rec = {"cells": [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}
                     for r in st.to_dict("records")]}
    back = R.stats_from_freeze(json.loads(json.dumps(rec)))
    pd.testing.assert_frame_equal(back[R.CELL + R.STAT_COLS].reset_index(drop=True),
                                  st[R.CELL + R.STAT_COLS].astype({"n": int}).reset_index(drop=True), check_dtype=False)


# ── owner 2026-11-11 "Own list": league-wide cells are outside the TOP-3 pool ─────────────────
def _lw_cell(mside, league="L"):
    return _cell(mside, league) | {"config": "NONE|ALL|max-|min-", "source": "NONE", "measure": "ALL",
                                   "mode": "LEAGUE_WIDE", "wf_roi_shrunk": 0.05}


def test_in_pool_is_model_cells_only():
    assert R.in_pool(_cell("btts_yes")) is True
    assert R.in_pool(_lw_cell("btts_yes")) is False
    assert R.in_pool(None) is False
    assert R.in_pool(_cell("btts_yes") | {"source": "NONE"}) is False


def test_league_wide_cell_never_enters_any_paper_track():
    cells = dict(CELLS)
    cells[("btts", "L", "yes")] = _lw_cell("btts_yes")
    c = [_cand("2", "BTTS", "YES", 0.60, 0.50, fam="BTTS", ver="btts_20261109T211852Z_d7cd2762"),
         _cand("3", "OU25", "OVER", 0.60, 0.45)]
    picks, weeks = P.freeze_due(pd.DataFrame(c), set(), _ctx(cells), NOW, 120.0, {}, _stats(), _guards(cells), CFG, "s")
    assert len(picks) == 1 and set(picks["market"]) == {"OU25"}           # the O/U pick only, no BTTS
    for w in weeks["excluded"]:
        assert json.loads(w)[P.R_LEAGUE_WIDE] == 1


def test_protocol_records_own_list_no_ev_fallback_and_history():
    cfg = P.load_config()
    proto = json.loads((ROOT / cfg["protocol_file"]).read_text(encoding="utf-8"))
    assert proto["protocol"] == "top3-ranking-paper/5" and proto["written_before_any_live_pick"] is True
    assert "never ev_est" in proto["rules_common"]["missing_statistic"]
    assert any("Drop T1" in w for w in proto["amendment"]["owner_words"])
    assert [t["id"] for t in proto["tracks"]] == ["T2_R2_CAL_EV", "T3_R3_CAL_EV_LCB"]
    assert "2 tracks" in proto["evaluation"]["family"] and "no_pick_exists" in proto
    assert "LEAGUE_WIDE_OWN_LIST" in proto["pool"]["refusal_reasons"]
    assert any("Own list" in w for w in proto["amendment"]["owner_words"])
    for h in proto["amends"]["history"]:
        assert P.sha256_lf(ROOT / h["file"]) == h["sha256_lf"]            # v1 and v2 kept byte-for-byte


def test_r1_is_research_code_only():
    assert "R1_CELL_LCB" in R.RESEARCH_RULES and "R1_CELL_LCB" not in R.TRACK_RULES + R.PAPER_RULES
    o = R.rank_order(pd.DataFrame([_cand("1", "OU25", "OVER", .6, .45) | {"track": "standard", "mside": "over25_over",
                                                                          "fair": .45, "p_used": .6},
                                   _cand("2", "BTTS", "YES", .6, .5) | {"track": "standard", "mside": "btts_yes",
                                                                        "fair": .5, "p_used": .6}]),
                     _stats(), "R1_CELL_LCB", tiebreak=("kickoff_utc", "fixture_id"))
    assert o.iloc[0]["mside"] == "btts_yes"                         # still computable for research
