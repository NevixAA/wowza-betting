"""Phase 2 part 11 framework: three performance views that are never mixed (src/trustworthy_performance).

Synthetic ledgers only, under tmp_path. Nothing here reads output/ or produces a claim about real data."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import staked as S
from src import trustworthy_performance as TP

START = "2026-11-10T00:00:00Z"


def _cfg(start=None, **kw):
    cfg = TP.load_config()
    cfg = {**cfg, "clean_record_start_utc": start, **kw}
    return cfg


def _bets(rows):
    cols = ["source", "generated_at", "match_date", "league", "home_team", "away_team", "side", "odds",
            "signal_tier", "result", "pnl", "closing_odds", "clv_pct", "model_type"]
    return pd.DataFrame(rows, columns=cols)


def _bet(day, home, result, odds=2.0, gen=None, clv=None, side="OVER"):
    pnl = round(odds - 1, 4) if result == "WIN" else (-1.0 if result == "LOSS" else 0.0)
    return ["live", gen or f"{day} 09:00", day, "L1", home, "Away FC", side, odds, "SNIPER", result, pnl,
            "", clv, "new_format"]


def _write(d, tmp_path, name):
    d.to_csv(tmp_path / name, index=False)


def _plan(tmp_path, rows):
    cols = ["date", "fixture", "league", "model_type", "market", "side", "policy_version", "deployment_state",
            "base_fraction", "final_fraction", "stake_amount", "reason", "generated_at"]
    pd.DataFrame(rows, columns=cols).to_csv(tmp_path / "stake_plan.csv", index=False)


def _days(n, start="2026-10-01"):
    return [str(d.date()) for d in pd.date_range(start, periods=n, freq="D")]


@pytest.fixture
def ledgers(tmp_path):
    rows = [_bet(d, f"Home{i}", "WIN" if i % 3 else "LOSS", clv=1.5) for i, d in enumerate(_days(40))]
    rows += [_bet(d, f"Late{i}", "LOSS", gen=f"{d} 10:00") for i, d in enumerate(_days(5, "2026-11-12"))]
    _write(_bets(rows), tmp_path, "bets_ledger.csv")
    _plan(tmp_path, [[d, f"{d}|Home{i}|Away FC", "L1", "new_format", "OU25", "OVER", "p1", "ACTIVE", 0.01,
                      0.01, 1.0, "OK", f"{d} 08:00"] for i, d in enumerate(_days(40)[:10])])
    props = pd.DataFrame([{"signal_date": d, "match_date": d, "player_name": f"P{i}", "market": "sot",
                           "tier": "VALUABLE", "market_odds": 2.5, "result": "LOSS", "pnl": -1.0}
                          for i, d in enumerate(_days(12))])
    _write(props, tmp_path, "player_ledger.csv")
    (tmp_path / "corrections").mkdir()
    pd.DataFrame([{"ledger_row": 0, "player_name": "P0", "exclude": True}]).to_csv(
        tmp_path / "corrections" / "props_sentinel_prob.csv", index=False)
    return tmp_path


def test_config_minimums_are_the_existing_rules():
    from src.research import league_weekly_eval as LW
    from src import nf_calibration
    cfg = TP.load_config()
    m = TP.minimums(cfg)
    assert (m["settled"], m["matchdays"]) == (LW.MIN_N, LW.MIN_WEEKS)
    assert m["clv"] == nf_calibration.MIN_N == json.load(open(TP.ROOT / "config" / "market_edge_v2.json"))["strong_min_n"]
    # the owner set the Clean Record start on 2026-11-11 (2026-11-21); claims stay off until he enables them
    assert cfg["clean_record_start_utc"] == "2026-11-21T00:00:00Z" and cfg["performance_claims_enabled"] is False
    assert all(v["source"] for v in cfg["minimums"].values())


def test_without_a_start_every_live_tip_is_historical(ledgers):
    v, _ = TP.frames(ledgers, _cfg())
    a = {n: d for n, _k, d, *_ in v[TP.A]}
    assert len(a["bets_ledger.csv"]) == 45 and all(len(d) == 0 for _n, _k, d, *_ in v[TP.C])


def test_start_splits_live_tips_and_research_stays_research(ledgers):
    v, _ = TP.frames(ledgers, _cfg(START))
    a = {n: d for n, _k, d, *_ in v[TP.A]}
    c = {n: d for n, _k, d, *_ in v[TP.C]}
    b = {n: d for n, _k, d, *_ in v[TP.B]}
    assert len(a["bets_ledger.csv"]) == 40 and len(c["bets_ledger.csv"]) == 5
    assert set(a["bets_ledger.csv"]["view"]) == {TP.A} and set(c["bets_ledger.csv"]["view"]) == {TP.C}
    assert "player_ledger.csv" in b and "player_ledger.csv" not in a and "player_ledger.csv" not in c


def test_tip_without_generation_time_is_unassigned():
    d = _bets([_bet("2026-11-12", "X", "WIN")]).assign(generated_at="")
    assert list(TP.assign_live(d, TP.clean_record_start(_cfg(START)))) == [TP.UNASSIGNED]


def test_a_tip_generated_before_the_start_stays_historical_even_if_played_after():
    d = _bets([_bet("2026-11-12", "X", "WIN", gen="2026-11-09 22:00")])
    assert list(TP.assign_live(d, TP.clean_record_start(_cfg(START)))) == [TP.A]


def test_metrics_refuse_two_views(ledgers):
    v, _ = TP.frames(ledgers, _cfg(START))
    a = next(d for n, _k, d, *_ in v[TP.A] if n == "bets_ledger.csv")
    c = next(d for n, _k, d, *_ in v[TP.C] if n == "bets_ledger.csv")
    with pytest.raises(TP.MixedViewsError):
        TP.metrics(pd.concat([a, c]), _cfg(START), "mixed")


def test_flat_and_staked_figures_use_the_one_definition(ledgers):
    res = TP.build(ledgers, _cfg())
    t = res["views"][TP.A]["sources"]["bets_ledger.csv"]["total"]
    assert t["flat_1u"]["label"] == S.FLAT_LABEL and t["staked"]["label"] == S.DEFINITION
    assert t["flat_1u"]["n"] == 45 and t["flat_1u"]["status"] == TP.OK
    # 10 positions have a recorded stake > 0 -> STAKED; 10 matchdays < 30 settled -> NOT_ENOUGH_DATA
    assert t["staked"]["n"] == 10 and t["staked"]["status"] == TP.NOT_ENOUGH
    assert t["staked"]["by_stake_status"][S.STAKED] == 10
    lo, hi = t["flat_1u"]["roi_ci"]
    assert lo <= t["flat_1u"]["roi"] <= hi
    assert t["clv"]["n"] == 40 and t["clv"]["status"] == TP.NOT_ENOUGH        # below the 150 minimum


def test_view_c_is_not_started_without_the_owner_start(ledgers):
    res = TP.build(ledgers, _cfg())
    assert res["views"][TP.C]["status"].startswith("NOT_STARTED")
    assert res["views"][TP.C]["sources"]["bets_ledger.csv"]["settled"] == 0


def test_research_rows_get_sidecar_exclusions_and_are_never_staked(ledgers):
    res = TP.build(ledgers, _cfg())
    p = res["views"][TP.B]["sources"]["player_ledger.csv"]
    assert p["excluded_by_corrections"] == 1 and p["settled"] == 11
    assert p["total"]["staked"]["n"] == 0
    assert p["total"]["staked"]["by_stake_status"][S.NOT_POLICY] == 11
    listed = {c["sidecar"]: c for c in res["corrections_view_A_and_B"]}
    assert listed["props_sentinel_prob.csv"]["rows_changed"] == 1
    assert "stake_audit_reconciled.csv" in listed                               # listed, not applied


def test_extra_time_restatement_is_applied_beside_the_original(ledgers):
    d = pd.read_csv(ledgers / "bets_ledger.csv")
    r = d.iloc[[1]]                                                             # a WIN
    pd.DataFrame([{"ledger": "bets_ledger", "match_date": r["match_date"].iloc[0], "league": "L1",
                   "home_team": r["home_team"].iloc[0], "away_team": "Away FC", "market": "OVER",
                   "restated_result": "LOSS", "restated_pnl": -1.0}]).to_csv(
        ledgers / "corrections" / "team_ledgers_extra_time_restated.csv", index=False)
    v, corr = TP.frames(ledgers, _cfg())
    a = next(x for n, _k, x, *_ in v[TP.A] if n == "bets_ledger.csv")
    row = a[a["home_team"] == r["home_team"].iloc[0]].iloc[0]
    assert (row["result"], row["pnl"], row["result_stored"]) == ("LOSS", -1.0, "WIN")
    et = [c for c in corr if c["sidecar"] == "team_ledgers_extra_time_restated.csv"
          and c["applied_to"] == "bets_ledger.csv"][0]
    assert et["rows_changed"] == 1


def test_block_ci_needs_two_matchdays_and_is_deterministic():
    assert TP.block_ci(np.array([1.0, -1.0]), np.ones(2), np.array(["d", "d"]), "k", 200, 0.95) == (None, None)
    v, days = np.array([1.0, -1.0, 0.5, -0.5]), np.array(["a", "b", "c", "d"])
    assert TP.block_ci(v, np.ones(4), days, "k", 200, 0.95) == TP.block_ci(v, np.ones(4), days, "k", 200, 0.95)


def test_cli_refuses_the_full_build_until_the_owner_enables_it(ledgers, capsys):
    assert TP.main(["--out-dir", str(ledgers)]) == 2
    assert "refused" in capsys.readouterr().err


def test_dry_run_prints_counts_only(ledgers, capsys):
    assert TP.main(["--dry-run", "--out-dir", str(ledgers)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["views"][TP.A]["bets_ledger.csv"]["settled"] == 45
    def keys(o):
        if isinstance(o, dict):
            return set(o) | set().union(*(keys(v) for v in o.values()))
        if isinstance(o, list):
            return set().union(*(keys(v) for v in o)) if o else set()
        return set()
    assert not keys(out) & {"roi", "pnl", "stake_pnl", "mean", "flat_1u", "staked", "clv", "roi_ci"}
