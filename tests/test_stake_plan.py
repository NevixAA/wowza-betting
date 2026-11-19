"""Stake plan (wired 2026-10-14): policy applied per match DAY, output only, first-sight audit."""
import pandas as pd
import pytest

from src import stake_plan as sp


@pytest.fixture(autouse=True)
def _rated_and_evidenced(monkeypatch):
    """These tests are about per-day sizing and the audit: every league is rated (no network) and
    has its own threshold. The no-rating / no-evidence rules are tested below and in
    tests/test_league_rating_sizing.py."""
    import src.betting as betting
    monkeypatch.setattr(sp, "load_league_ratings", lambda *a, **k: _AllRated())
    monkeypatch.setattr(betting, "league_threshold_evidence", lambda league, market="ou25": True)


class _AllRated(dict):
    def get(self, k, d=None):
        return 10


def _bets():
    rows = []
    for day in ("2026-10-17", "2026-10-18"):
        for i in range(8):
            rows.append({"date": day, "league": f"L{i}", "home_team": f"H{i}", "away_team": f"A{i}",
                         "best_side": "OVER", "odds_over25": 2.2, "odds_under25": 1.7,
                         "signal_tier": "SNIPER", "model_type": "new_format"})
    return pd.DataFrame(rows)


def _side():
    # the canonical contract (sp.SIDE_SELECTION_COLUMNS); tests/test_side_stake_contract.py drives
    # the REAL producer, this one only exercises per-day sizing and the audit
    base = {"date": "2026-10-17", "market": "btts", "odds": 2.05, "signal_tier": "SNIPER",
            "model_type": "new_format", "side": "YES", "model_prob": 0.62, "fair_prob": 0.45,
            "edge": 0.17, "ev": 0.27}
    return pd.DataFrame([{**base, "league": "Brazil Serie A", "home_team": "X", "away_team": "Y"},
                         {**base, "league": "USA MLS", "home_team": "P", "away_team": "Q"}])


def test_sized_per_day_and_caps(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "PLAN", tmp_path / "plan.csv")
    monkeypatch.setattr(sp, "AUDIT", tmp_path / "audit.csv")
    from src.bankroll_policy import load_policy
    cap = load_policy()["portfolio"]["daily_cap"]
    p = sp.write(_bets(), _side())
    per_day = p.groupby("date")["final_fraction"].sum()
    assert len(per_day) == 2 and (per_day <= cap + 1e-9).all()
    # both days get stakes of their own (a week squeezed into one daily cap would starve them)
    assert (per_day > 0).all()
    assert p.loc[p["league"] == "USA MLS", "final_fraction"].sum() == 0          # paper league
    assert set(p["bankroll_basis"]) == {sp.BASIS}


def test_audit_first_sight(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "PLAN", tmp_path / "plan.csv")
    monkeypatch.setattr(sp, "AUDIT", tmp_path / "audit.csv")
    sp.write(_bets(), _side())
    n1 = len(pd.read_csv(tmp_path / "audit.csv"))
    sp.write(_bets(), _side())
    assert len(pd.read_csv(tmp_path / "audit.csv")) == n1


def test_never_raises():
    assert sp.write(pd.DataFrame({"nonsense": [1]}), None).empty


def test_ou25_frame_without_model_type_is_a_contract_error():
    """No getattr defaults: a bets frame without model_type used to stake on `unknown|OU25`."""
    with pytest.raises(sp.SelectionSchemaError):
        sp.selections(_bets().drop(columns=["model_type"]), None)


def test_unknown_side_gets_no_odds():
    """A tip without OVER/UNDER used to be priced at the UNDER odds."""
    b = _bets().head(1).assign(best_side="")
    assert sp.selections(b, None)["odds"].isna().all()


def test_threshold_evidence_comes_from_the_real_gates(monkeypatch):
    """Owner 2026-11-09: evidence = the league's OWN owner-applied bars. Championship O/U 2.5 has them
    (installed here); Brazil Serie A O/U 2.5 has none, so it gets no tier and, if planned, NO stake."""
    monkeypatch.undo()                                   # the real league_threshold_evidence
    from tests.own_bars import install
    install(monkeypatch, ["Championship"], markets=("ou25",))
    b =pd.DataFrame([{**_bets().iloc[0].to_dict(), "league": lg, "home_team": lg}
                      for lg in ("Championship", "Brazil Serie A")])
    s = sp.selections(b, None).set_index("league")
    assert bool(s.loc["Championship", "league_threshold_evidence"]) is True
    assert bool(s.loc["Brazil Serie A", "league_threshold_evidence"]) is False


@pytest.fixture(autouse=True)
def _per_track_policy(monkeypatch):
    """These tests pin the TRACK mechanics: per-track staking (tests/policy_helpers). The live per-cell
    staking (owner 2026-11-09) is pinned in tests/test_cell_staking.py."""
    import sys
    from tests.policy_helpers import install_per_track
    install_per_track(monkeypatch, sys.modules[__name__])
