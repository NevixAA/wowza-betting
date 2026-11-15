"""fix/history-dupes (2026-11-10): one provider fixture, one history row.

The OddsAPI history matcher found China Super League 2025-02-23 twice in af_history: 'Tianjin Teda v
Meizhou Kejia' (provider row, fixture 1340905) next to 'Tianjin Jinmen Tiger v Meizhou Hakka'. Both
clubs renamed at once, so no side is the same club by name and src.integrity.duplicate_matches cannot
pair them; src.integrity.fixture_claims pairs them by the provider fixture both rows claim. Data truth
rule 5 now goes RED on such a pair in the served frame or in the served af_history provider rows.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.integrity import data_truth as DT
from src.integrity import fixture_claims as FC

ROOT = Path(__file__).resolve().parents[1]
CSL = "China Super League"
ALIASES = {CSL: {"Meizhou Hakka": "Meizhou Kejia"}}


def _pool(rows):
    """rows: (fixture_id, league, date, home, away, ft_home, ft_away[, status])"""
    recs = [dict(zip(["fixture_id", "league", "date", "home_name", "away_name", "ft_home", "ft_away", "status"],
                     r + (("FT",) if len(r) == 7 else ()))) for r in rows]
    return FC.provider_pool(ht=pd.DataFrame(recs), comp=pd.DataFrame())


def _rows(rows):
    return pd.DataFrame([{"league": lg, "date": pd.Timestamp(d), "season": "2025", "home_team": h,
                          "away_team": a, "home_goals": hg, "away_goals": ag}
                         for lg, d, h, a, hg, ag in rows])


POOL = _pool([(1340905, CSL, "2025-02-23", "Tianjin Teda", "Meizhou Kejia", 2, 2),
              (1340906, CSL, "2025-02-23", "Beijing Guoan", "Wuhan Three Towns", 2, 2),
              (1340910, CSL, "2025-03-01", "Meizhou Kejia", "Henan Jianye", 2, 1)])


def test_both_clubs_renamed_copy_claims_the_same_provider_fixture():
    rows = _rows([(CSL, "2025-02-23", "Tianjin Teda", "Meizhou Kejia", 2.0, 2.0),
                  (CSL, "2025-02-23", "Tianjin Jinmen Tiger", "Meizhou Hakka", 2.0, 2.0),
                  (CSL, "2025-02-23", "Beijing Guoan", "Wuhan Three Towns", 2.0, 2.0)])
    c = FC.claims(rows, POOL, aliases=ALIASES)
    assert dict(zip(c["idx"], c["rule"])) == {0: "both_clubs", 1: "forced_opponent+score", 2: "both_clubs"}
    d = FC.duplicate_claims(c)
    assert sorted(d["idx"]) == [0, 1] and set(d["fixture_id"]) == {1340905}


def test_the_name_rule_alone_cannot_see_that_pair():
    from src.integrity.duplicate_matches import duplicate_pairs
    rows = _rows([(CSL, "2025-02-23", "Tianjin Teda", "Meizhou Kejia", 2.0, 2.0),
                  (CSL, "2025-02-23", "Tianjin Jinmen Tiger", "Meizhou Hakka", 2.0, 2.0)])
    assert duplicate_pairs(rows).empty


def test_forced_opponent_needs_the_providers_score():
    rows = _rows([(CSL, "2025-02-23", "Tianjin Jinmen Tiger", "Meizhou Hakka", 1.0, 2.0),   # wrong score
                  (CSL, "2025-02-23", "Tianjin Jinmen Tiger", "Meizhou Hakka", None, None)])  # no score
    assert FC.claims(rows, POOL, aliases=ALIASES).empty


def test_no_proven_side_claims_nothing():
    rows = _rows([(CSL, "2025-02-23", "Tianjin Jinmen Tiger", "Some Other Club", 2.0, 2.0)])
    assert FC.claims(rows, POOL, aliases={}).empty


def test_adjacent_day_needs_both_clubs_and_the_score():
    ok = _rows([(CSL, "2025-02-24", "Tianjin Teda", "Meizhou Kejia", 2.0, 2.0)])
    c = FC.claims(ok, POOL, aliases={})
    assert list(c["rule"]) == ["both_clubs_adjacent_day"] and list(c["day_off"]) == [-1]
    assert FC.claims(_rows([(CSL, "2025-02-24", "Tianjin Teda", "Meizhou Kejia", 0.0, 0.0)]), POOL,
                     aliases={}).empty
    # one club only on the adjacent day: never
    assert FC.claims(_rows([(CSL, "2025-02-24", "Tianjin Jinmen Tiger", "Meizhou Kejia", 2.0, 2.0)]), POOL,
                     aliases={}).empty


def test_a_stored_fixture_id_is_the_claim_and_two_real_matches_never_pair():
    rows = _rows([(CSL, "2025-02-23", "Tianjin Teda", "Meizhou Kejia", 2.0, 2.0),
                  (CSL, "2025-03-01", "Meizhou Kejia", "Henan Jianye", 2.0, 1.0)])
    rows["fixture_id"] = [1340905, None]
    c = FC.claims(rows, POOL, aliases={}, fixture_col="fixture_id")
    assert list(c["rule"]) == ["fixture_id", "both_clubs"]
    assert FC.duplicate_claims(c).empty


def test_proven_alias_segments_are_chosen_by_date():
    al = {"Bundesliga 2": {"Regensburg": [{"provider": "Jahn Regensburg", "valid_from": None, "valid_to": "2024-08-03"},
                                          {"provider": "SSV Jahn Regensburg", "valid_from": "2024-08-03",
                                           "valid_to": None}]}}
    assert FC._lookup(al, "Bundesliga 2", "Regensburg", "2023-01-01") == "Jahn Regensburg"
    assert FC._lookup(al, "Bundesliga 2", "Regensburg", "2024-08-03") == "SSV Jahn Regensburg"
    assert FC._lookup(al, "Bundesliga 2", "Unknown", "2024-08-03") is None


def test_unfinished_provider_fixtures_are_not_in_the_pool():
    p = _pool([(1, CSL, "2025-02-23", "A", "B", 1, 0, "AWD"), (2, CSL, "2025-02-23", "C", "D", 1, 0, "FT")])
    assert list(p["fixture_id"]) == [2]


# ── data truth rule 5 ──────────────────────────────────────────────────────────────────────────
def _bank(rows):
    return _rows(rows).drop(columns=[])


def test_rule5_red_when_a_both_renamed_copy_reaches_the_served_frame():
    bank = _bank([(CSL, "2025-02-23", "Tianjin Teda", "Meizhou Kejia", 2.0, 2.0),
                  (CSL, "2025-02-23", "Tianjin Jinmen Tiger", "Meizhou Hakka", 2.0, 2.0)])
    c = DT.one_match({"fd_cache": bank}, pool=POOL)
    assert c["status"] == DT.RED, c["value"]
    assert "1 provider fixture(s) claimed by 2+ served rows" in c["value"]
    assert "provider fixture 1340905" in c["detail"]


def test_rule5_green_for_one_row_per_fixture():
    bank = _bank([(CSL, "2025-02-23", "Tianjin Teda", "Meizhou Kejia", 2.0, 2.0),
                  (CSL, "2025-02-23", "Beijing Guoan", "Wuhan Three Towns", 2.0, 2.0)])
    c = DT.one_match({"fd_cache": bank}, pool=POOL)
    assert c["status"] == DT.GREEN, c["value"]


def test_rule5_red_when_two_af_history_provider_rows_claim_one_fixture():
    bank = _bank([(CSL, "2025-02-23", "Tianjin Teda", "Meizhou Kejia", 2.0, 2.0)])
    af = pd.DataFrame({"league": [CSL, CSL], "date": pd.to_datetime(["2025-02-23", "2025-02-23"]),
                       "home_team": ["Tianjin Teda", "Tianjin Jinmen Tiger"],
                       "away_team": ["Meizhou Kejia", "Meizhou Hakka"], "FTHG": [2.0, 2.0], "FTAG": [2.0, 2.0],
                       "fixture_id": [1340905, None]})
    c = DT.one_match({"fd_cache": bank}, pool=POOL, af=af)
    assert c["status"] == DT.RED and "1 by 2+ af_history provider rows" in c["value"]


def test_rule5_red_when_the_provider_pool_is_missing():
    bank = _bank([(CSL, "2025-02-23", "Tianjin Teda", "Meizhou Kejia", 2.0, 2.0)])
    c = DT.one_match({"fd_cache": bank}, pool=pd.DataFrame())
    assert c["status"] == DT.RED and "NOT RUN" in c["value"]


# ── the committed data (fix/history-dupes findings) ──────────────────────────────────────────────
_HAVE = all((ROOT / p).exists() for p in ("output/af_history.parquet", "output/fd_history.parquet",
                                          "output/strength/competition_fixtures.parquet"))


@pytest.mark.skipif(not _HAVE, reason="committed banks not present")
def test_committed_china_pair_is_one_provider_fixture_and_never_served():
    from src.af_history_corrections import apply_af_history_corrections
    from src.af_history_provenance import written_back_mask
    af = pd.read_parquet(ROOT / "output" / "af_history.parquet")
    day = af[(af["league"] == CSL) & (pd.to_datetime(af["date"]) == "2025-02-23")]
    prov = day[(day["home_team"] == "Tianjin Teda") & (day["away_team"] == "Meizhou Kejia")]
    shadow = day[(day["home_team"] == "Tianjin Jinmen Tiger") & (day["away_team"] == "Meizhou Hakka")]
    assert len(prov) == 1 and int(prov["fixture_id"].iloc[0]) == 1340905
    # the other copy is a written-back row: no fixture id, no score - never served by any reader
    assert len(shadow) == 1 and bool(written_back_mask(shadow).iloc[0])
    served = apply_af_history_corrections(af, tag="test")
    served = served[~written_back_mask(served).to_numpy()]
    s = served[(served["league"] == CSL) & (pd.to_datetime(served["date"]) == "2025-02-23")]
    assert set(zip(s["home_team"], s["away_team"])) >= {("Tianjin Teda", "Meizhou Kejia")}
    assert ("Tianjin Jinmen Tiger", "Meizhou Hakka") not in set(zip(s["home_team"], s["away_team"]))


@pytest.mark.skipif(not _HAVE, reason="committed banks not present")
def test_committed_argentina_rows_are_independiente_rivadavia_by_the_provider_record():
    pool = FC.provider_pool()
    for fid, day, home, away, hid, aid, ft in ((1158665, "2024-05-12", "Lanus", "Independ. Rivadavia", 446, 473, (0, 2)),
                                               (1158682, "2024-05-18", "Independ. Rivadavia", "Godoy Cruz", 473, 439, (0, 0))):
        f = pool[pool["fixture_id"] == fid].iloc[0]
        assert (f["_d"], f["home_name"], f["away_name"]) == (day, home, away)
        assert (int(f["home_id"]), int(f["away_id"])) == (hid, aid)
        assert (f["ft_home"], f["ft_away"]) == ft
    # Independiente (provider team 453) played the day before and the day after, never on those days
    ind = pool[(pool["league"] == "Argentina Primera Division")
               & ((pd.to_numeric(pool["home_id"]) == 453) | (pd.to_numeric(pool["away_id"]) == 453))]
    assert not set(ind["_d"]) & {"2024-05-12", "2024-05-18"}
    assert {"2024-05-11", "2024-05-19"} <= set(ind["_d"])
