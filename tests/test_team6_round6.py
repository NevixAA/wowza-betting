"""Data Truth round 6 (team6): Standard xG join, inside-box shot collection, HT-history indicator.

    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/test_team6_round6.py -q

No network: the API-Football client is a counting fake; collector outputs go to tmp_path.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import config  # noqa: E402
from src import data_loader as DL  # noqa: E402
from src import feature_engineering as FE  # noqa: E402
from src.features_repaired import build_features_repaired, build_upcoming_features_repaired  # noqa: E402

STD = "Championship"


def _m(date, h, a, hg, ag, league=STD, **kw):
    r = {"date": pd.Timestamp(date), "league": league, "home_team": h, "away_team": a,
         "home_goals": hg, "away_goals": ag, "total_goals": hg + ag, "over25": float(hg + ag > 2.5),
         "home_shots": 10, "away_shots": 8, "home_sot": 4, "away_sot": 3,
         "home_corners": 5, "away_corners": 4, "home_fouls": 11, "away_fouls": 12,
         "odds_over25": 1.9, "odds_under25": 1.95}
    r.update(kw)
    return r


# ── 1. Standard xG: af_history -> loader join -> xg_last5, training and serving ──────────────────

@pytest.fixture()
def no_key(monkeypatch):
    monkeypatch.delenv("APIFOOTBALL_KEY", raising=False)
    monkeypatch.delenv("WOWZA_FULL_ENRICH", raising=False)
    monkeypatch.setattr(config, "API_KEY", "")


def test_standard_af_history_xg_reaches_training_and_serving_xg_last5(no_key):
    af = ROOT / "output" / "af_history.parquet"
    if not af.exists():
        pytest.skip("af_history.parquet not in this checkout")
    h = pd.read_parquet(af)
    h = h[h["league"].isin(config.STANDARD_FORMAT_LEAGUES) & h["HXG"].notna() & h["AXG"].notna()]
    h = h[~h.duplicated(["league", "date", "home_team"], keep=False)]
    team = h.groupby(["league", "home_team"]).size().sort_values().index[-1]
    rows = h[(h["league"] == team[0]) & (h["home_team"] == team[1])].sort_values("date").tail(3)
    left = pd.DataFrame([_m(r.date, r.home_team, r.away_team, r.FTHG, r.FTAG, league=r.league)
                         for r in rows.itertuples()])
    out = DL._enrich_with_api_shots(left.copy())
    assert np.allclose(out["home_xg"].to_numpy(float), rows["HXG"].to_numpy(float))
    nxt = pd.Timestamp(rows["date"].max()) + pd.Timedelta(days=7)
    tr = FE.build_features(pd.concat([out, pd.DataFrame([_m(nxt, team[1], "ZZ", 0, 0, league=team[0])])],
                                     ignore_index=True))
    row = tr[tr["date"] == nxt].iloc[0]
    assert row["home_xg_last5"] == pytest.approx(rows["HXG"].mean())
    up = pd.DataFrame([{"date": nxt, "league": team[0], "home_team": team[1], "away_team": "ZZ",
                        "odds_over25": 1.9, "odds_under25": 1.95}])
    sv = FE.build_upcoming_features(up, out).iloc[0]
    assert sv["home_xg_last5"] == pytest.approx(row["home_xg_last5"])


# ── 2. inside-box shots: collection + loader fill ───────────────────────────────────────────────

class _Resp:
    def __init__(self, body):
        self.status_code, self._b, self.headers = 200, body, {"x-ratelimit-requests-remaining": "70000"}

    def json(self):
        return self._b


class _Session:
    """Counts calls; /status is free."""

    def __init__(self, bodies):
        self.bodies, self.calls = bodies, []

    def get(self, url, headers=None, params=None, timeout=None):
        ep = url.rsplit("/v3.football.api-sports.io/", 1)[-1]
        if ep.endswith("status"):
            return _Resp({"response": {"requests": {"current": 0, "limit_day": 75000}}})
        self.calls.append((ep, dict(params or {})))
        return _Resp(self.bodies(ep, params))


def _stats_body(home_id, away_id, hib, aib, hxg="1.20", axg="0.40"):
    def side(tid, ib, xg):
        st = [{"type": "Total Shots", "value": 10}, {"type": "expected_goals", "value": xg}]
        if ib is not None:
            st.append({"type": "Shots insidebox", "value": ib})
        return {"team": {"id": tid}, "statistics": st}
    # AWAY first on purpose: home/away must come from the team ids, not the response order
    return {"errors": [], "response": [side(away_id, aib, axg), side(home_id, hib, hxg)]}


@pytest.fixture()
def gap(tmp_path, monkeypatch):
    import collect_af_history_gap as G
    import collect_common as CC
    cdir = tmp_path / "collect"
    for name, val in (("STATS_STORE", cdir / "af_fixture_stats.parquet"),
                      ("COVERAGE", cdir / "af_stats_coverage.parquet"),
                      ("FILL", cdir / "af_history_stats_fill.parquet"),
                      ("AUDIT", cdir / "af_stats_fill_audit.json")):
        monkeypatch.setattr(G, name, val)
    fx = pd.DataFrame([
        {"fixture_id": 1001, "league_id": 40, "season": "2025", "date": "2026-03-07", "home_id": 1,
         "home_name": "Leeds", "away_id": 2, "away_name": "Hull City", "status": "FT"},
        {"fixture_id": 1002, "league_id": 40, "season": "2025", "date": "2026-03-08", "home_id": 3,
         "home_name": "Stoke City", "away_id": 4, "away_name": "Millwall", "status": "FT"},
        {"fixture_id": 2001, "league_id": 40, "season": "2017", "date": "2018-03-07", "home_id": 1,
         "home_name": "Leeds", "away_id": 2, "away_name": "Hull City", "status": "FT"},
    ])
    fxp = tmp_path / "competition_fixtures.parquet"
    fx.to_parquet(fxp)
    monkeypatch.setattr(G, "FIXTURE_SOURCES", (fxp,))
    monkeypatch.setattr(CC, "BUDGET_FILE", cdir / "af_budget.json")
    return G, tmp_path


def _af():
    base = {"FTHG": 1.0, "FTAG": 0.0, "HS": 10.0, "AS": 8.0, "HXG": np.nan, "AXG": np.nan,
            "fixture_id": np.nan, "home_team_id": np.nan, "away_team_id": np.nan}
    return pd.DataFrame([
        # Standard league, banked by the loader write-back with football-data spellings, HIB empty
        {**base, "league": STD, "date": pd.Timestamp("2026-03-07"), "home_team": "Leeds",
         "away_team": "Hull", "season": "2025/26", "HIB": np.nan, "AIB": np.nan},
        # already carries HIB -> not a target
        {**base, "league": STD, "date": pd.Timestamp("2026-03-08"), "home_team": "Stoke",
         "away_team": "Millwall", "season": "2025/26", "HIB": 9.0, "AIB": 5.0},
        # a season the provider holds no statistics for
        {**base, "league": STD, "date": pd.Timestamp("2018-03-07"), "home_team": "Leeds",
         "away_team": "Hull", "season": "2017/18", "HIB": np.nan, "AIB": np.nan},
    ])


def _coverage_body():
    return {"errors": [], "response": [{"seasons": [
        {"year": 2025, "coverage": {"fixtures": {"statistics_fixtures": True}}},
        {"year": 2017, "coverage": {"fixtures": {"statistics_fixtures": False}}}]}]}


def test_insidebox_targets_every_league_and_banked_rows_without_hib(gap):
    G, _ = gap
    lack = G.lacking_insidebox(_af())
    assert list(lack["home_team"]) == ["Leeds", "Leeds"]          # Standard rows, HIB empty
    assert set(lack["league_id"]) == {40}


def test_insidebox_collects_by_team_id_and_skips_seasons_without_statistics(gap):
    G, tmp = gap
    from collect_common import AFClient
    sess = _Session(lambda ep, p: _coverage_body() if ep == "leagues" else _stats_body(1, 2, 11, None))
    cl = AFClient(key="k", session=sess, budget_file=tmp / "collect" / "af_budget.json", pause_s=0)
    st = G.run_insidebox(client=cl, af=_af(), cache_dir=tmp / "cache")
    assert [c[0] for c in sess.calls] == ["leagues", "fixtures/statistics"]   # 2017 season: no call
    assert sess.calls[1][1] == {"fixture": 1001}
    fill = pd.read_parquet(G.FILL)
    assert len(fill) == 1
    r = fill.iloc[0]
    assert (r["home_team"], r["away_team"], r["fixture_id"]) == ("Leeds", "Hull", 1001)
    assert r["HIB"] == 11 and pd.isna(r["AIB"])                 # not given -> missing, never 0
    assert r["HXG"] == pytest.approx(1.2) and r["AXG"] == pytest.approx(0.4)
    assert st["calls"] == 2
    audit = json.loads(G.AUDIT.read_text(encoding="utf-8"))
    assert audit["per_league"][STD]["no_provider_statistics"] == 1
    # second run: journal + cache -> no call at all
    sess2 = _Session(lambda ep, p: pytest.fail("re-fetched"))
    G.run_insidebox(client=AFClient(key="k", session=sess2, budget_file=tmp / "b.json", pause_s=0),
                    af=_af(), cache_dir=tmp / "cache")
    assert sess2.calls == []


def test_insidebox_reads_cached_fixture_statistics_without_a_call(gap):
    G, tmp = gap
    from collect_common import AFClient
    cache = tmp / "cache"
    cache.mkdir()
    (cache / "stats_1001.json").write_text(json.dumps(_stats_body(1, 2, 7, 3)), encoding="utf-8")
    sess = _Session(lambda ep, p: _coverage_body() if ep == "leagues" else pytest.fail("stats re-fetched"))
    G.run_insidebox(client=AFClient(key="k", session=sess, budget_file=tmp / "b.json", pause_s=0),
                    af=_af(), cache_dir=cache)
    assert [c[0] for c in sess.calls] == ["leagues"]
    r = pd.read_parquet(G.FILL).iloc[0]
    assert (r["HIB"], r["AIB"]) == (7, 3)


def test_insidebox_dry_run_counts_calls_and_makes_none(gap):
    G, tmp = gap
    st = G.run_insidebox(dry_run=True, af=_af(), cache_dir=tmp / "cache")
    assert st["planned_coverage_calls"] == 1
    assert st["planned_stats_calls_upper_bound"] == 2 and st["calls"] == 0


def test_loader_fills_insidebox_from_the_sidecar_nan_only(tmp_path):
    side = tmp_path / DL.AF_STATS_FILL
    side.parent.mkdir(parents=True)
    pd.DataFrame([{"league": STD, "date": "2026-03-07", "home_team": "Leeds United",
                   "away_team": "Hull City", "HIB": 11.0, "AIB": np.nan, "HXG": 1.2, "AXG": 0.4,
                   "fixture_id": 1001}]).to_parquet(side)
    df = pd.DataFrame([_m("2026-03-07", "Leeds", "Hull", 1, 0, home_insidebox=np.nan,
                          away_insidebox=np.nan, home_xg=0.9, away_xg=np.nan),
                       _m("2026-03-07", "Leeds", "Hull", 1, 0, league="League One",
                          home_insidebox=np.nan, away_insidebox=np.nan, home_xg=np.nan, away_xg=np.nan)])
    out = DL._enrich_af_stats_fill(df, root=tmp_path)
    assert out.loc[0, "home_insidebox"] == 11.0 and pd.isna(out.loc[0, "away_insidebox"])
    assert out.loc[0, "home_xg"] == 0.9                                 # never overwritten
    assert out.loc[0, "away_xg"] == 0.4
    assert out.loc[1, ["home_insidebox", "home_xg"]].isna().all()       # other league untouched


def test_loader_calls_the_sidecar_fill_on_both_exits():
    import inspect
    src = inspect.getsource(DL._enrich_with_api_shots)
    assert src.count("_enrich_af_stats_fill(df)") == 2


def test_workflow_has_a_guarded_insidebox_step_and_no_run_day_override():
    y = (ROOT / ".github" / "workflows" / "data_truth_collect.yml").read_text(encoding="utf-8")
    assert "python scripts/collect_af_history_gap.py --insidebox" in y
    assert "contains(env.STEPS, 'insidebox')" in y
    assert "afgap,insidebox," in y
    # item 4: the one-day override (floor 10,000 / cap 55,000) was removed on run-day 2 (2026-11-03):
    # the job runs on collect_common's defaults again
    assert "COLLECT_AF_FLOOR" not in y and "COLLECT_AF_DAILY_CAP" not in y


# ── 3. has_ht_history next to the HT form features ──────────────────────────────────────────────

def _ht_hist():
    rows, d0 = [], pd.Timestamp("2026-04-04")
    for i in range(8):
        d = d0 + pd.Timedelta(days=7 * i)
        # A: real HT scores only in its first two matches; then 5 matches without one
        ht = (1, 0) if i < 2 else (np.nan, np.nan)
        rows.append(_m(d, "A", f"X{i}", 1, 1, ht_home_goals=ht[0], ht_away_goals=ht[1]))
        rows.append(_m(d, "B", f"Y{i}", 2, 0, ht_home_goals=1, ht_away_goals=0))
        rows.append(_m(d, "C", f"W{i}", 0, 0, ht_home_goals=np.nan, ht_away_goals=np.nan))
    return pd.DataFrame(rows)


def _train_serve(hist, home, away, date):
    nxt = _m(date, home, away, 0, 0, ht_home_goals=np.nan, ht_away_goals=np.nan)
    tr = FE.build_features(pd.concat([hist, pd.DataFrame([nxt])], ignore_index=True))
    tr = tr[(tr["date"] == pd.Timestamp(date)) & (tr["home_team"] == home)].iloc[0]
    up = pd.DataFrame([{k: nxt[k] for k in ("date", "league", "home_team", "away_team",
                                            "odds_over25", "odds_under25")}])
    return tr, FE.build_upcoming_features(up, hist).iloc[0]


def test_has_ht_history_true_only_with_a_real_ht_score_in_the_window():
    tr, sv = _train_serve(_ht_hist(), "A", "B", "2026-06-06")
    # A's last 5: no half-time score -> HT form unknown, indicator 0; B's window has them -> 1
    assert pd.isna(tr["home_ht_scored_last5"])
    assert tr["home_has_ht_history"] == 0.0 and sv["home_has_ht_history"] == 0.0
    assert tr["away_has_ht_history"] == 1.0 and sv["away_has_ht_history"] == 1.0


def test_has_ht_history_is_zero_without_any_history_and_matches_serving():
    hist = _ht_hist()
    tr, sv = _train_serve(hist, "NEW", "A", "2026-06-06")
    assert tr["home_has_ht_history"] == 0.0 and sv["home_has_ht_history"] == 0.0
    early = hist[hist["date"] <= "2026-04-11"]
    tr2, sv2 = _train_serve(early, "A", "B", "2026-04-18")
    assert tr2["home_has_ht_history"] == 1.0 == sv2["home_has_ht_history"]


def test_has_ht_history_when_no_ht_column_exists():
    hist = _ht_hist().drop(columns=["ht_home_goals", "ht_away_goals"])
    f = FE.build_features(hist)
    assert set(f["home_has_ht_history"].unique()) == {0.0}
    assert set(f["away_has_ht_history"].unique()) == {0.0}


def test_repaired_recipe_has_ht_history_train_and_serve():
    hist = _ht_hist()
    f = build_features_repaired(hist)
    a = f[f["home_team"] == "A"].sort_values("date")
    assert a["home_has_ht_history"].tolist()[0] == 0.0                   # first match: no history
    assert (a["home_has_ht_history"] == a["home_ht_scored_last5"].notna().astype(float)).all()
    up = pd.DataFrame([{"date": pd.Timestamp("2026-06-06"), "league": STD, "home_team": "C",
                        "away_team": "B"}])                  # C: history, never an HT score
    sv = build_upcoming_features_repaired(up, hist).iloc[0]
    assert sv["home_has_ht_history"] == 0.0 and sv["away_has_ht_history"] == 1.0
