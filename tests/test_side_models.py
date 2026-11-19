"""Side count models (corners / cards) -> prediction ledger (src/side_models/, owner 2026-11-18).

Synthetic data for the rules (verification, crosswalk, as-of window, NB probabilities, ledger rows) and
checks on the committed config / artifact. No network, no write outside tmp_path.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import prediction_ledger as PL
from src.side_models import data as D
from src.side_models import features as F
from src.side_models import ledger as L
from src.side_models import model as M
from src.side_models import train as T

ROOT = Path(__file__).resolve().parents[1]
SIDE_MARKETS = ("corners_total", "corners_team", "cards_total", "cards_team")


# ── verification ──────────────────────────────────────────────────────────────────────────────────

def _j(**kw):
    base = {"fixture_id": 1, "af_hc": 5.0, "af_ac": 4.0, "af_hy": 2.0, "af_ay": 1.0, "af_hr": 0.0, "af_ar": 0.0,
            "fd_hc": 5.0, "fd_ac": 4.0, "fd_hy": 2.0, "fd_ay": 1.0, "fd_hr": 0.0, "fd_ar": 0.0}
    base.update(kw)
    return pd.DataFrame([base])


def test_corners_verified_only_when_both_sources_agree_exactly():
    assert D.verify(_j())["corners_verified"].iloc[0]
    v = D.verify(_j(fd_hc=6.0))
    assert not v["corners_verified"].iloc[0] and np.isnan(v["home_corners"].iloc[0])
    assert not D.verify(_j(fd_hc=np.nan))["corners_verified"].iloc[0]          # one source only


def test_cards_rule_reds_exact_yellows_exact_without_red_band_with_red():
    v = D.verify(_j())
    assert v["cards_verified"].iloc[0] and v["home_cards"].iloc[0] == 2 and v["away_cards"].iloc[0] == 1
    assert not D.verify(_j(fd_hy=3.0))["cards_verified"].iloc[0]             # no red: yellows must match
    # a second-yellow dismissal: AF lists both yellows (+2), fd does not -> inside the band, AF count kept
    v = D.verify(_j(af_hy=4.0, af_hr=1.0, fd_hy=2.0, fd_hr=1.0))
    assert v["cards_verified"].iloc[0] and v["home_cards"].iloc[0] == 5
    assert not D.verify(_j(af_hy=5.0, af_hr=1.0, fd_hy=2.0, fd_hr=1.0))["cards_verified"].iloc[0]  # +3 > 2x1
    assert not D.verify(_j(af_hy=1.0, af_hr=1.0, fd_hy=2.0, fd_hr=1.0))["cards_verified"].iloc[0]  # AF below fd
    assert not D.verify(_j(af_hr=1.0))["cards_verified"].iloc[0]             # red counts differ


def test_all_zero_af_row_is_refused_as_suspected_zero_fill():
    z = dict(af_hc=0.0, af_ac=0.0, af_hy=0.0, af_ay=0.0, af_hr=0.0, af_ar=0.0,
             fd_hc=0.0, fd_ac=0.0, fd_hy=0.0, fd_ay=0.0, fd_hr=0.0, fd_ar=0.0)
    v = D.verify(_j(**z))
    assert v["zero_fill_suspect"].iloc[0]
    assert not v["corners_verified"].iloc[0] and not v["cards_verified"].iloc[0]


# ── crosswalk ─────────────────────────────────────────────────────────────────────────────────────

def _cw_data(n=8):
    fd, af = [], []
    for i in range(n):
        d = (pd.Timestamp("2024-08-03") + pd.Timedelta(days=7 * i)).strftime("%Y-%m-%d")
        fd.append({"league": "L", "fd_file": "x", "date": d, "fd_home": "Alpha", "fd_away": "Beta",
                   "fd_fthg": float(i % 4), "fd_ftag": 1.0})
        af.append({"fixture_id": 100 + i, "league": "L", "date": d, "home_id": 11, "away_id": 22,
                   "ft_home": i % 4, "ft_away": 1})
    return pd.DataFrame(fd), pd.DataFrame(af)


def test_crosswalk_is_learned_from_score_anchors_with_a_vote_floor():
    fd, af = _cw_data(8)
    xw, audit = D.crosswalk(fd, af)
    assert xw == {"L": {"Alpha": 11, "Beta": 22}}
    fd4, af4 = _cw_data(D.MIN_VOTES - 1)
    assert D.crosswalk(fd4, af4)[0] == {}                                       # too few anchors: unmapped


def test_join_exact_date_then_unique_adjacent_day():
    fd, af = _cw_data(8)
    xw, _ = D.crosswalk(fd, af)
    af2 = af.copy()
    af2.loc[0, "date"] = "2024-08-04"                                          # kickoff-date convention, +1 day
    af2.loc[1, "date"] = "2024-08-12"                                          # +2 days: not this fd row
    j = D.join(fd, af2, xw)
    assert 100 in set(j["fixture_id"]) and 101 not in set(j["fixture_id"])
    assert j["fixture_id"].is_unique and len(j) == 7


# ── features ──────────────────────────────────────────────────────────────────────────────────────

def _table(n=14, unverified=()):
    rows = []
    for i in range(n):
        h, a = (1, 2) if i % 2 == 0 else (2, 1)
        rows.append({"fixture_id": i, "league": "L", "date": f"2025-01-{i + 1:02d}", "home_id": h, "away_id": a,
                     "corners_verified": i not in unverified, "cards_verified": True,
                     "home_corners": float(i), "away_corners": 1.0, "home_cards": 2.0, "away_cards": 1.0})
    t = pd.DataFrame(rows)
    t.loc[~t["corners_verified"], ["home_corners", "away_corners"]] = np.nan
    return t


def test_features_use_strictly_earlier_dates_and_the_window_keeps_unverified_slots():
    t = _table(14, unverified=(10, 11, 12))
    tg = pd.DataFrame([{"fixture_id": 99, "league": "L", "date": "2025-01-14", "home_id": 1, "away_id": 2}])
    f = F.build(t, tg, "corners", window=10, max_age_days=400, min_verified=6)
    # team 1's last 10 before the 14th: fixtures 3..12 (the one ON the 14th, id 13, is excluded);
    # 10, 11, 12 are unverified and still occupy their slots -> 7 verified
    assert f["home_n_window"].iloc[0] == 10 and f["home_n_verified"].iloc[0] == 7
    exp_for = np.mean([i if i % 2 == 0 else 1.0 for i in range(3, 10)])
    assert f["home_for"].iloc[0] == pytest.approx(exp_for)
    assert f["not_scored_reason"].iloc[0] == ""


def test_too_few_verified_matches_is_not_scored_with_a_reason():
    t = _table(14, unverified=tuple(range(4, 13)))
    tg = pd.DataFrame([{"fixture_id": 99, "league": "L", "date": "2025-01-14", "home_id": 1, "away_id": 2}])
    f = F.build(t, tg, "corners", window=10, max_age_days=400, min_verified=6)
    assert np.isnan(f["home_for"].iloc[0]) and "needs >= 6" in f["not_scored_reason"].iloc[0]


def test_old_matches_beyond_max_age_do_not_count():
    t = _table(14)
    tg = pd.DataFrame([{"fixture_id": 99, "league": "L", "date": "2026-06-01", "home_id": 1, "away_id": 2}])
    f = F.build(t, tg, "corners", window=10, max_age_days=400, min_verified=6)
    assert f["home_n_window"].iloc[0] == 0 and f["not_scored_reason"].iloc[0]


# ── model ─────────────────────────────────────────────────────────────────────────────────────────

def test_nb_probabilities_are_monotone_and_match_poisson_at_zero_dispersion():
    mu = np.array([3.0, 5.0, 10.0])
    for a in (0.0, 0.1):
        p = np.vstack([M.p_over(mu, a, line) for line in (2.5, 4.5, 9.5)])
        assert (np.diff(p, axis=0) <= 0).all() and ((p >= 0) & (p <= 1)).all()
    from scipy import stats
    assert M.p_over(np.array([4.0]), 0.0, 4.5)[0] == pytest.approx(1 - stats.poisson.cdf(4, 4.0))


def test_poisson_irls_recovers_a_known_model():
    rng = np.random.default_rng(7)
    n = 4000
    f = pd.DataFrame({"home_for": rng.uniform(2, 8, n), "away_against": rng.uniform(2, 8, n),
                      "away_for": rng.uniform(2, 8, n), "home_against": rng.uniform(2, 8, n)})
    X = M.design(f, "home")
    y = rng.poisson(np.exp(X @ np.array([0.2, 0.5, 0.3])))
    fit = M.fit_part(f, y, "home")
    assert fit["coef"] == pytest.approx([0.2, 0.5, 0.3], abs=0.15)
    assert fit["alpha"] < 0.05


# ── config + artifact ─────────────────────────────────────────────────────────────────────────────

def test_ledger_entries_are_active_and_match_the_side_config():
    cfg = T.load_config()
    m = PL.load_markets()
    by = {(e["model_id"], e["market"]): e for e in m["entries"] if e["repo"] == "v9"}
    for fam, fc in cfg["families"].items():
        for market, mc in fc["markets"].items():
            e = by[(fc["model_id"], market)]
            assert e["status"] == "active"
            assert e["lines"] == mc["lines"]
            want = cfg["team_selections"] if mc["part"] == "team" else cfg["total_selections"]
            assert e["selections"] == want
            assert e["model_file"] == {"*": cfg["artifact"]}
    assert {mk for (_, mk) in by if mk in SIDE_MARKETS} == set(SIDE_MARKETS)


def test_artifact_covers_only_leagues_passing_their_truth_rule_and_lists_every_tracked_league():
    art = L.load_artifact()
    cfg = T.load_config()
    tracked = T.tracked_leagues()
    names = {e["name"] for e in tracked}
    fmt = {e["name"]: e.get("football_data_format") for e in tracked}
    for fam in cfg["families"]:
        cov = art["coverage"][fam]
        assert set(cov) == names                                     # every tracked league, with a verdict
        for lg, c in cov.items():
            if c["covered"] and c["basis"] == "two_source":
                assert fmt[lg] == "mmz4281"
                assert c["overlap_pct"] >= cfg["verification"]["min_overlap_pct"]
                assert c["agree_pct"] >= cfg["verification"]["min_agree_pct"]
            elif c["covered"]:
                assert c["basis"] == "single_source" and fmt[lg] != "mmz4281"
                assert c["stats_pct"] >= cfg["single_source"]["min_stats_pct"]
                assert c["sane_pct"] >= cfg["single_source"]["min_sane_pct"]
            if c["covered"]:
                assert set(art["models"][fam][lg]["params"]) == set(M.PARTS)
            else:
                assert c["reason"] and lg not in art["models"][fam]
    body = {k: art[k] for k in ("config", "data_end", "coverage", "data_audit", "crosswalk", "models")}
    assert art["model_version"].endswith(M.params_sha(body))                  # not hand-edited
    assert "price" not in json.dumps(art["config"]).lower()


# ── ledger rows ───────────────────────────────────────────────────────────────────────────────────

NOW = datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc)


def _art(cov_leagues=("Championship",)):
    p = {"coef": [1.5, 0.1, 0.1], "alpha": 0.05, "n": 100, "mean_y": 5.0}
    fams = ("corners", "cards")
    return {"model_version": "side_test_1",
            "coverage": {f: {lg: {"covered": True} for lg in cov_leagues} for f in fams},
            "models": {f: {lg: {"params": {pt: p for pt in M.PARTS}} for lg in cov_leagues} for f in fams}}


def _champ_table():
    t = _table(14)
    t["league"] = "Championship"
    for c in ("home_cards", "away_cards"):
        t[c] = 1.0
    return t


def _targets(ko="2025-01-15T15:00:00Z", home=1, away=2):
    return pd.DataFrame([{"fixture_id": 555, "league": "Championship", "date": ko[:10], "home_id": home,
                          "away_id": away, "kickoff_utc": pd.Timestamp(ko)}])


def test_rows_every_market_line_selection_registered_and_price_free(tmp_path):
    cfg = T.load_config()
    art = _art()
    scored = L.score(_champ_table(), _targets(), art, cfg)
    rows = L.rows_for(scored, art, cfg, now=NOW, slugs=PL.LeagueSlugs(), names={1: "Home FC", 2: "Away FC"})
    per = {"corners_total": 4 * 2, "corners_team": 3 * 4, "cards_total": 3 * 2, "cards_team": 2 * 4}
    got = pd.Series([r["market"] for r in rows]).value_counts().to_dict()
    assert got == per                                               # one snapshot due (first_sight)
    assert all(r["scored"] == 1 and r["fixture_ref"] == "af:555" for r in rows)
    team = [r for r in rows if r["market"] == "corners_team" and r["selection"] == "HOME_OVER"]
    assert team and all(r["team"] == "Home FC" for r in team)
    # OVER + UNDER = 1 for the same line
    tot = pd.DataFrame([r for r in rows if r["market"] == "cards_total"])
    s = tot.groupby("line")["probability"].sum()
    assert np.allclose(s.to_numpy(float), 1.0, atol=2e-6)
    st = PL.append(rows, repo="v9", writer="side_models", root=tmp_path, now=NOW)
    assert st["written"] == len(rows) and st["refused"] == 0 and st.get("violations", 0) == 0
    assert PL.validate(tmp_path) == []
    st = PL.append(rows, repo="v9", writer="side_models", root=tmp_path, now=NOW)  # first write wins
    assert st["written"] == 0 and st["duplicate_skipped"] == len(rows)


def test_unscorable_fixture_is_stored_unscored_with_its_reason(tmp_path):
    cfg = T.load_config()
    art = _art()
    scored = L.score(_champ_table(), _targets(home=7, away=8), art, cfg)    # teams with no history
    rows = L.rows_for(scored, art, cfg, now=NOW, slugs=PL.LeagueSlugs())
    assert rows and all(r["scored"] == 0 and r["probability"] == "" and r["not_scored_reason"] for r in rows)
    assert PL.append(rows, repo="v9", writer="side_models", root=tmp_path, now=NOW)["violations"] == 0


def test_uncovered_league_gets_no_rows_and_final_snapshot_only_near_kickoff():
    cfg = T.load_config()
    art = _art(cov_leagues=("League One",))
    assert L.rows_for(L.score(_champ_table(), _targets(), art, cfg), art, cfg, now=NOW,
                      slugs=PL.LeagueSlugs()) == []
    art = _art()
    near = datetime(2025, 1, 15, 14, 0, tzinfo=timezone.utc)                 # 60 min before kickoff
    rows = L.rows_for(L.score(_champ_table(), _targets(), art, cfg), art, cfg, now=near, slugs=PL.LeagueSlugs())
    assert {r["snapshot"] for r in rows} == {"first_sight", "final"}
    late = datetime(2025, 1, 15, 15, 0, tzinfo=timezone.utc)
    assert L.rows_for(L.score(_champ_table(), _targets(), art, cfg), art, cfg, now=late,
                      slugs=PL.LeagueSlugs()) == []


def test_flag_off_writes_nothing(monkeypatch):
    monkeypatch.delenv("SIDE_MODELS_LEDGER_ENABLED", raising=False)
    monkeypatch.setenv("PREDICTION_LEDGER_ENABLED", "true")
    assert not L.enabled()
    assert L.board()["disabled"]
    monkeypatch.setenv("SIDE_MODELS_LEDGER_ENABLED", "true")
    monkeypatch.setenv("PREDICTION_LEDGER_ENABLED", "false")
    assert not L.enabled()
    monkeypatch.setenv("PREDICTION_LEDGER_ENABLED", "true")
    assert L.enabled()


def test_predict_workflow_step_is_gated_by_both_variables():
    y = (ROOT / ".github" / "workflows" / "predict.yml").read_text(encoding="utf-8")
    assert "python -m src.side_models.ledger board" in y
    assert "vars.SIDE_MODELS_LEDGER_ENABLED == 'true' && vars.PREDICTION_LEDGER_ENABLED == 'true'" in y


@pytest.mark.skipif(not (D.FILL.exists() and D.FIXTURES.exists() and (D.FD_RAW / "manifest.json").exists()),
                    reason="verified source files not in this checkout")
def test_real_data_crosswalk_maps_every_football_data_name():
    leagues = ["Championship", "Bundesliga 2"]
    fd = D.load_fd_raw(leagues)
    af = D.load_af(leagues)
    xw, audit = D.crosswalk(fd, af)
    for lg in leagues:
        assert audit[lg]["unmapped"] == [], audit[lg]["unmapped"]
        assert len(set(xw[lg].values())) == len(xw[lg])                     # one-to-one
