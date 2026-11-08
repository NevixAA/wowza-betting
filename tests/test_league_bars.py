"""Per-league tip bars (src/league_bars.py) - synthetic data only."""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import league_bars as LB
from src import threshold_proposals as TP

ROOT = Path(__file__).resolve().parents[1]


def _rows(league: str, n: int, market_noise: float, model_noise: float, seed: int,
          start="2023-01-01", holdout_start="2026-04-01", n_hold: int | None = None) -> pd.DataFrame:
    """A model that knows the true probability (up to model_noise) against a market whose fair price
    is the truth plus market_noise. Big market noise + small model noise = a real, provable edge."""
    rng = np.random.default_rng(seed)
    n_hold = n if n_hold is None else n_hold
    frames = []
    for stage, s, k in (("walk_forward", start, n), ("holdout", holdout_start, n_hold)):
        dates = pd.Timestamp(s) + pd.to_timedelta(np.arange(k) // 4, unit="D")
        q = rng.uniform(0.3, 0.7, k)
        m = np.clip(q + rng.normal(0, market_noise, k), 0.05, 0.95)
        p = np.clip(q + rng.normal(0, model_noise, k), 0.01, 0.99)
        y = (rng.uniform(size=k) < q).astype(float)
        frames.append(pd.DataFrame({
            "stage": stage, "fold": 0, "track": "standard", "market": "over25", "candidate": "champion_recipe",
            "row_id": [f"{d:%Y-%m-%d}|{league}|H{i}|A{i}" for i, d in enumerate(dates)],
            "date": dates, "league": league, "y": y, "p": p, "deployable": True,
            "yes": 1 / (m * 1.04), "no": 1 / ((1 - m) * 1.04)}))
    return pd.concat(frames, ignore_index=True)


def _run(frames) -> list[dict]:
    return LB.finalise(LB.run_market(pd.concat(frames, ignore_index=True), "over25", "standard"))


def test_no_global_bar_is_read_anywhere():
    src = inspect.getsource(LB)
    for name in ("SIDE_GLOBAL_BARS", "VALUABLE_THRESHOLD", "SNIPER_THRESHOLD", "MARKSMAN_THRESHOLD",
                 "side_thresholds", "global_bars"):
        assert name not in src, name


def test_every_league_gets_its_own_bars_and_they_differ_when_the_data_differ():
    rows = _run([_rows("Sharp", 3000, 0.01, 0.01, 1), _rows("Soft", 3000, 0.12, 0.01, 2),
                 _rows("Mid", 3000, 0.06, 0.02, 3)])
    bars = {r["league"]: (r["valuable_th"], r["marksman_th"], r["sniper_th"]) for r in rows}
    assert set(bars) == {"Sharp", "Soft", "Mid"}
    assert all(all(v is not None for v in b) for b in bars.values())
    assert len(set(bars.values())) > 1


@pytest.mark.parametrize("seed", range(6))
def test_bars_are_ordered(seed):
    rows = _run([_rows(f"L{i}", 1500, 0.02 + 0.03 * i, 0.02, seed * 10 + i) for i in range(4)])
    for r in rows:
        if r.get("valuable_th") is not None:
            assert r["valuable_th"] <= r["marksman_th"] <= r["sniper_th"]


def test_the_holdout_never_chooses_a_bar():
    base = pd.concat([_rows("A", 2000, 0.08, 0.02, 7), _rows("B", 2000, 0.03, 0.02, 8)], ignore_index=True)
    a = LB.run_market(base, "over25", "standard")
    flipped = base.copy()
    ho = flipped["stage"] == "holdout"
    flipped.loc[ho, "y"] = 1 - flipped.loc[ho, "y"]                 # destroy the holdout
    flipped.loc[ho, "p"] = 1 - flipped.loc[ho, "p"]
    b = LB.run_market(flipped, "over25", "standard")
    for x, y in zip(a, b):
        assert (x["valuable_th"], x["marksman_th"], x["sniper_th"]) == (y["valuable_th"], y["marksman_th"], y["sniper_th"])
    assert [x["ho_roi"] for x in a] != [y["ho_roi"] for y in b]


def test_a_league_without_a_two_sided_price_is_no_price_data():
    f = _rows("Priced", 1500, 0.08, 0.02, 4)
    g = _rows("Unpriced", 1500, 0.08, 0.02, 5)
    g["no"] = np.nan                                              # one side only: never de-vigged
    rows = _run([f, g])
    r = {x["league"]: x for x in rows}["Unpriced"]
    assert r["status"] == LB.STATUS_NO_PRICE and "valuable_th" not in r
    c = LB.candidate_params(rows)["ou25"]["Unpriced"]
    assert c["sniper_th"] is None and c["marksman_th"] is None and c["valuable_th"] is None
    assert c["approved"] is False and c["drop"] is True


def test_holm_matches_the_textbook_and_runs_across_the_whole_family():
    p = np.array([0.01, 0.04, 0.03, 0.005, np.nan])
    np.testing.assert_allclose(LB.holm(p)[:4], [0.03, 0.06, 0.06, 0.02])
    assert np.isnan(LB.holm(p)[4])
    rows = _run([_rows(f"L{i}", 1500, 0.05, 0.02, 20 + i) for i in range(5)])
    tested = [r for r in rows if r["status"] in (LB.STATUS_PROVEN, LB.STATUS_UNPROVEN)]
    assert tested and all(r["family_size"] == sum(r2["p_one_sided"] is not None for r2 in tested) for r in tested)


def test_strict_rule_proves_a_real_edge_and_refuses_a_thin_one():
    rows = _run([_rows("Edge", 6000, 0.15, 0.005, 31), _rows("Thin", 6000, 0.15, 0.005, 32, n_hold=60)])
    r = {x["league"]: x for x in rows}
    assert r["Edge"]["status"] == LB.STATUS_PROVEN, r["Edge"]
    assert r["Thin"]["status"] == LB.STATUS_UNPROVEN and "out-of-sample bets" in r["Thin"]["reason"]
    c = LB.candidate_params(rows)["ou25"]
    assert c["Edge"]["approved"] is True and c["Thin"]["approved"] is False


def test_no_market_edge_is_not_proven():
    rows = _run([_rows(f"L{i}", 3000, 0.0, 0.03, 40 + i) for i in range(3)])
    assert all(r["status"] != LB.STATUS_PROVEN for r in rows)


def test_snapshot_pairs_take_the_earliest_pre_kickoff_same_snapshot_pair():
    raw = pd.DataFrame([
        # match 1: a pair the day before, a pair at kickoff+5m (post-kickoff, refused)
        ("2026-09-04T10:00:00Z", "2026-09-05", "2026-09-05T14:00:00Z", "L", "A vs B", "btts_yes", 1.80),
        ("2026-09-04T10:00:00Z", "2026-09-05", "2026-09-05T14:00:00Z", "L", "A vs B", "btts_no", 2.00),
        ("2026-09-05T14:05:00Z", "2026-09-05", "2026-09-05T14:00:00Z", "L", "A vs B", "btts_yes", 1.50),
        ("2026-09-05T14:05:00Z", "2026-09-05", "2026-09-05T14:00:00Z", "L", "A vs B", "btts_no", 2.60),
        # match 2: only one side in the snapshot -> no pair
        ("2026-09-05T09:00:00Z", "2026-09-05", "2026-09-05T18:00:00Z", "L", "C vs D", "btts_yes", 1.70),
        # match 3: a first-half BTTS price mislabelled (> 3.20) -> mapping invalid -> no pair
        ("2026-09-05T09:00:00Z", "2026-09-05", "2026-09-05T18:00:00Z", "L", "E vs F", "btts_yes", 5.50),
        ("2026-09-05T09:00:00Z", "2026-09-05", "2026-09-05T18:00:00Z", "L", "E vs F", "btts_no", 1.15),
        # match 4: same-day forward stamp with no kickoff -> unprovable -> dropped
        ("2026-09-05T09:00:00Z", "2026-09-05", None, "L", "G vs H", "btts_yes", 1.70),
        ("2026-09-05T09:00:00Z", "2026-09-05", None, "L", "G vs H", "btts_no", 2.10),
        # match 5: a backfill stamp at 00:00 UTC on the match day -> provably before kickoff
        ("2026-09-05T00:00:00Z", "2026-09-05", None, "L", "I vs J", "btts_yes", 1.90),
        ("2026-09-05T00:00:00Z", "2026-09-05", None, "L", "I vs J", "btts_no", 1.90),
    ], columns=["snapshot_ts", "match_date", "kickoff_utc", "league", "match", "market", "odds"])
    pr = LB.snapshot_pairs(raw, "standard_sidemarket_odds_history", "btts").set_index("match")
    assert sorted(pr.index) == ["A vs B", "I vs J"]
    assert (pr.at["A vs B", "yes"], pr.at["A vs B", "no"]) == (1.80, 2.00)


def test_unmatched_clubs_stay_unpriced():
    pred = _rows("L", 10, 0.05, 0.02, 50).drop(columns=["yes", "no"])
    prices = pd.DataFrame({"date": pd.to_datetime(pred["date"]), "league": "L",
                           "home": ["Nobody FC"] * len(pred), "away": ["Nowhere United"] * len(pred),
                           "yes": 1.9, "no": 1.9})
    out, st = LB.attach_prices(pred, prices)
    assert out["yes"].isna().all() and st["L"] == {"rows": len(pred), "priced": 0}


def _digest(paths):
    return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}


def test_no_production_file_is_written(tmp_path, monkeypatch):
    prod = [ROOT / "models" / "best_params_standard.json", ROOT / "models" / "best_params_side_markets.json",
            ROOT / "config" / "threshold_approvals.json"]
    before = _digest(prod)
    rows = _run([_rows("A", 1500, 0.08, 0.02, 60), _rows("B", 1500, 0.03, 0.02, 61)])
    with pytest.raises(PermissionError):
        LB.write_outputs(rows, "standard", ROOT / "models" / "league_bars", {})
    with pytest.raises(PermissionError):                         # outside output/threshold_proposals/
        LB.write_outputs(rows, "standard", tmp_path / "elsewhere", {})
    monkeypatch.setattr(TP, "PROPOSAL_DIR", tmp_path)
    written = LB.write_outputs(rows, "standard", tmp_path / "league_bars", {"track": "standard"})
    assert written and all(tmp_path in Path(p).resolve().parents for p in written)
    assert _digest(prod) == before


# ── IN_PERIOD_SPLIT (owner 2026-11-08: real bars wherever a real price exists) ─────────────────────

def _holdout_only(league, seed, n=1200, market_noise=0.08):
    f = _rows(league, n, market_noise, 0.02, seed, holdout_start="2026-04-01")
    wf = f["stage"] == "walk_forward"
    f.loc[wf, ["yes", "no"]] = np.nan                            # priced only from 2026-04-01
    return f


def test_a_league_priced_only_in_the_holdout_gets_in_period_split_bars():
    rows = _run([_rows("Old", 1500, 0.06, 0.02, 70), _holdout_only("New", 71)])
    r = {x["league"]: x for x in rows}
    assert r["Old"]["evidence"] == LB.EVIDENCE_WF
    n = r["New"]
    assert n["evidence"] == LB.EVIDENCE_SPLIT and n["status"] in (LB.STATUS_PROVEN, LB.STATUS_UNPROVEN)
    assert n["valuable_th"] <= n["marksman_th"] <= n["sniper_th"]
    sp = n["split"]
    assert sp["select_to"] < sp["check_from"] and sp["select_from"] >= "2026-04-01"
    c = LB.candidate_params(rows)["ou25"]["New"]
    assert c["evidence"] == LB.EVIDENCE_SPLIT and c["split"] == sp and c["valuable_th"] is not None


def test_the_split_check_part_never_chooses_a_bar():
    base = pd.concat([_holdout_only("A", 72), _holdout_only("B", 73)], ignore_index=True)
    a = {r["league"]: r for r in LB.run_market(base, "over25", "standard")}
    flipped = base.copy()
    for lg, r in a.items():
        chk = (flipped["league"] == lg) & (pd.to_datetime(flipped["date"]) >= pd.Timestamp(r["split"]["check_from"]))
        flipped.loc[chk, "y"] = 1 - flipped.loc[chk, "y"]
        flipped.loc[chk, "p"] = 1 - flipped.loc[chk, "p"]
    b = {r["league"]: r for r in LB.run_market(flipped, "over25", "standard")}
    for lg in a:
        assert (a[lg]["valuable_th"], a[lg]["marksman_th"], a[lg]["sniper_th"]) == \
               (b[lg]["valuable_th"], b[lg]["marksman_th"], b[lg]["sniper_th"])
        assert a[lg]["split"] == b[lg]["split"]


def test_the_split_leaves_about_40_percent_of_matchdays_to_check():
    days = pd.Series(pd.date_range("2026-04-01", "2026-10-31", freq="D"))
    s = LB.in_period_split(days)
    share = float((days >= s).mean())
    assert s.day == 1 and abs(share - LB.IN_PERIOD_CHECK_SHARE) < 0.1
    assert LB.in_period_split(pd.Series([pd.Timestamp("2026-05-01")])) is None


def test_no_price_data_only_when_no_price_exists_at_any_time():
    none = _rows("None", 800, 0.05, 0.02, 74)
    none[["yes", "no"]] = np.nan
    one_day = _holdout_only("OneDay", 75)
    keep = (one_day["stage"] == "walk_forward") | (one_day.groupby("stage").cumcount() < 3)
    one_day = one_day[keep]                                       # holdout priced on a single matchday
    r = {x["league"]: x for x in _run([_rows("Old", 1500, 0.06, 0.02, 76), none, one_day])}
    assert r["None"]["status"] == LB.STATUS_NO_PRICE
    assert r["OneDay"]["status"] == LB.STATUS_THIN and r["OneDay"]["evidence"] == LB.EVIDENCE_SPLIT
