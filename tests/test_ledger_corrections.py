"""2026-11-11 certification: live-ledger hygiene corrected BESIDE the ledgers (src/ledger_corrections.py)
and data truth rule 9 (no recurrence from the Clean Record start)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import ledger_corrections as LC
from src.integrity import data_truth as DT
from src.integrity import fixture_claims as FC

ROOT = Path(__file__).resolve().parents[1]
LG = "Sweden Allsvenskan"


def _bets(rows):
    return pd.DataFrame([{"source": "live", "generated_at": g, "match_date": d, "league": LG, "home_team": h,
                          "away_team": a, "side": s, "odds": o, "closing_odds": c, "clv_pct": v,
                          "model_type": "new_format", "result": "WIN", "pnl": 1.0}
                         for g, d, h, a, s, o, c, v in rows])


BETS = _bets([("2026-11-22 03:00", "2026-11-23", "Kalmar FF", "Hammarby IF", "UNDER", 2.0, 1.9, 5.0),
              ("2026-11-22 09:00", "2026-11-23", "Kalmar FF", "Hammarby IF", "OVER", 1.9, 1.8, 5.0),
              ("2026-11-22 03:00", "2026-11-23", "IF Elfsborg", "AIK", "OVER", 2.2, 1.0, 120.0),
              ("2026-11-23 16:00", "2026-11-23", "Malmo FF", "Djurgarden", "OVER", 2.1, 2.0, 5.0)])


def test_multi_tip_rows_flag_first_sight_and_opposite_pairs():
    m = LC.multi_tip_rows(BETS)
    assert len(m) == 2 and m["opposite_tip_pair"].all()
    assert list(m["first_sight"]) == [True, False]
    assert list(m["side"]) == ["UNDER", "OVER"]


def test_post_kickoff_rows_use_the_provider_kickoff():
    pool = FC.provider_pool(ht=pd.DataFrame([
        {"fixture_id": 7, "league": LG, "date": "2026-11-23", "home_name": "Malmo FF", "away_name": "Djurgarden",
         "ft_home": 1, "ft_away": 0, "status": "FT"},
        {"fixture_id": 8, "league": LG, "date": "2026-11-23", "home_name": "Kalmar FF", "away_name": "Hammarby IF",
         "ft_home": 1, "ft_away": 0, "status": "FT"}]), comp=pd.DataFrame())
    ko = pd.Series({7: "2026-11-23T15:00:00+00:00", 8: "2026-11-23T15:00:00+00:00"})
    p = LC.post_kickoff_rows(BETS, pool, ko)
    assert list(p["home_team"]) == ["Malmo FF"] and list(p["provider_fixture_id"]) == [7]


def test_invalid_close_is_quarantined_for_every_clv_consumer(tmp_path):
    from src import clv_corrections as CC
    assert len(LC.invalid_close_rows(BETS)) == 1
    out = CC.exclude_contaminated_clv(BETS, tmp_path)          # no sidecars at all: the rule still applies
    row = out[out["home_team"] == "IF Elfsborg"].iloc[0]
    assert row["clv_quarantined"] and pd.isna(row["clv_pct"]) and row["clv_pct_stored"] == 120.0
    assert row["pnl"] == 1.0                                    # P/L kept


def _players(rows):
    return pd.DataFrame([{"signal_date": s, "match_date": d, "league": LG, "home_team": h, "away_team": "Hammarby IF",
                          "player_name": n, "team": "Kalmar FF", "market": "sot", "tier": "VALUE", "result": "WIN",
                          "pnl": 1.0, "player_id": pid, "fixture_id": fid}
                         for s, d, h, n, pid, fid in rows])


def test_player_duplicates_first_sight_wins_reschedule_is_not_a_duplicate():
    pl = _players([("2026-11-20", "2026-11-23", "Kalmar FF", "A One", 1, 9),
                   ("2026-11-21", "2026-11-23", "Kalmar FF", "A One", 1, 9),      # re-logged: duplicate
                   ("2026-11-24", "2026-11-26", "Kalmar FF", "A One", 1, 9),      # rescheduled: kept
                   ("2026-11-20", "2026-11-23", "Kalmar FF", "B Two", 2, 9)])
    d = LC.player_duplicate_rows(pl, pool=FC.provider_pool(ht=pd.DataFrame(), comp=pd.DataFrame()))
    assert len(d) == 1
    r = d.iloc[0]
    assert (r["signal_date"], r["duplicate_of_signal_date"], r["duplicate_rule"]) == \
        ("2026-11-21", "2026-11-20", "same_fixture_player_id_market_date")


def test_v91_duplicate_keeps_the_copy_on_the_kickoff_date():
    v = pd.DataFrame({"fixture_id": [5, 5, 6], "market": ["ou25"] * 3,
                      "match_date": ["2026-11-07", "2026-11-08", "2026-11-08"],
                      "logged_at": ["2026-11-06T10:00Z", "2026-11-07T10:00Z", "2026-11-07T10:00Z"],
                      "kickoff_utc": ["2026-11-08T00:30Z", "2026-11-08T00:30Z", "2026-11-08T18:00Z"],
                      "league": [LG] * 3, "home_team": ["A", "A", "C"], "away_team": ["B", "B", "D"]})
    d = LC.v91_duplicate_rows(v)
    assert list(d["match_date"]) == ["2026-11-07"] and list(d["kept_match_date"]) == ["2026-11-08"]


def test_readers_apply_the_sidecars(tmp_path):
    corr = tmp_path / "corrections"
    corr.mkdir()
    LC.multi_tip_rows(BETS).to_csv(corr / LC.MULTI, index=False)
    BETS[BETS["home_team"] == "Malmo FF"][LC.BETS_KEY].to_csv(corr / LC.POST_KO, index=False)
    d, applied = LC.apply_bets(BETS, tmp_path)
    # owner 2026-11-11 "first tip only": the later tip of the opposite pair is excluded, never regraded
    assert set(d.loc[d["_excluded"], "home_team"]) == {"Malmo FF", "Kalmar FF"}
    later = d[(d["home_team"] == "Kalmar FF") & d["_excluded"]].iloc[0]
    assert later["side"] == "OVER" and "first tip only" in later["_excluded_reason"]
    assert d["opposite_tip_pair"].sum() == 2 and d["multi_tip_first_sight"].sum() == 1
    assert (d["result"] == BETS["result"]).all()                 # never regraded
    v = pd.DataFrame({"fixture_id": ["5", "5"], "market": ["ou25"] * 2, "match_date": ["2026-11-07", "2026-11-08"],
                      "logged_at": ["a", "b"]})
    v.iloc[[0]].to_csv(corr / LC.V91_DUP, index=False)
    assert list(LC.drop_v91_duplicates(v, tmp_path)["match_date"]) == ["2026-11-08"]


# ── data truth rule 9 ───────────────────────────────────────────────────────────────────────────
EMPTY_POOL = pd.DataFrame(columns=FC.POOL_COLS + ["_d"])


def test_rule6_red_on_a_recurrence_after_the_start():
    c = DT.ledgers(start="2026-11-21", bets=BETS, players=pd.DataFrame(), v91=pd.DataFrame(), pool=EMPTY_POOL)
    assert c["status"] == DT.RED
    assert "1 fixture(s) with more than one live tip (1 opposite)" in c["value"]
    assert "1 close(s) <= 1.0" in c["value"] and "1 new-format row(s) with |CLV| > 50" in c["value"]


def test_rule6_ignores_rows_before_the_start_and_waits_for_go_live(monkeypatch):
    c = DT.ledgers(start="2026-12-01", bets=BETS, players=pd.DataFrame(), v91=pd.DataFrame(), pool=EMPTY_POOL)
    assert c["status"] == DT.GREEN
    monkeypatch.setattr(DT, "_clean_record_start", lambda: None)
    assert DT.ledgers()["status"] == DT.GREY


def test_rule6_is_reported_but_not_gated():
    assert not "Data truth 9".startswith(DT.GATED)


# ── the committed sidecars (built from main's live ledgers, 2026-11-11) ──────────────────────────
@pytest.mark.parametrize("name,n", [(LC.POST_KO, 6), (LC.BAD_CLOSE, 1), (LC.V91_DUP, 9)])
def test_committed_sidecar_counts(name, n):
    assert len(pd.read_csv(ROOT / "output" / "corrections" / name)) == n


def test_committed_multi_tip_sidecar_flags_38_opposite_fixtures_of_67():
    m = pd.read_csv(ROOT / "output" / "corrections" / LC.MULTI)
    fx = m.drop_duplicates(["match_date", "league", "home_team", "away_team"])
    assert len(fx) == 67 and int(fx["opposite_tip_pair"].sum()) == 38
    assert int(m["first_sight"].sum()) == 67


def test_first_tip_only_in_the_dashboard_and_telegram_readers(tmp_path):
    corr = tmp_path / "corrections"
    corr.mkdir()
    LC.multi_tip_rows(BETS).to_csv(corr / LC.MULTI, index=False)
    kept = LC.drop_historical_excluded(BETS, tmp_path)
    k = kept[kept["home_team"] == "Kalmar FF"]
    assert list(k["side"]) == ["UNDER"]                         # the first-sight tip only
    assert len(kept) == len(BETS) - 1


def test_first_tip_only_on_the_committed_record():
    import pandas as pd
    m = pd.read_csv(ROOT / "output" / "corrections" / LC.MULTI)
    later = m[(m["opposite_tip_pair"].astype(str) == "True") & (m["first_sight"].astype(str) == "False")]
    d, _ = LC.apply_bets(m.assign(source="live"))
    assert int(d["_excluded"].sum()) >= len(later) > 0
    assert d.loc[d["_excluded"] & d["opposite_tip_pair"], "multi_tip_first_sight"].eq(False).all()
