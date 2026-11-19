"""Fix round 9 (owner decisions 2026-11-04, KEY=bank): the af_history.parquet write-back.

Round 8 found 90,876 of 93,381 af_history rows written back by the loader's 'BANK THE ENRICHMENT'
step; where a fixture-proven provider answer exists their total shots agree in 14.3% of 1,583 pairs,
and the write-back dropped corners and fouls. Decisions tested here:
  1. provider-proven values (the fullstats / insidebox sidecars, provider rows) are served;
  2. a written-back row (no fixture_id AND no FT score) serves none of its statistics;
  3. the loader never writes af_history.parquet (an original), and retrain does not stage it;
  4. --insidebox targets every match whose SERVED shots / corners / fouls / inside-box are missing;
  5. the TEAM null rule: an answered block's null count is 0, no answer is missing, non-counts and
     passes never zero-filled, 'Free Kicks' stored;
  6. identity pairs proven at fixture level; _enrich_af_stats_fill uses IDENTITY_ALIASES.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import config  # noqa: E402
from src import data_loader as DL  # noqa: E402
from src.af_history_provenance import WRITTEN_BACK_STAT_COLS, written_back_mask  # noqa: E402
from src import af_team_stats as TS  # noqa: E402

LG = "USA MLS"
AF_COLS = ["league", "date", "home_team", "away_team", "season", "FTHG", "FTAG", "FTR", "HS", "AS", "HST",
           "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR", "HXG", "AXG", "HIB", "AIB", "fixture_id",
           "home_team_id", "away_team_id"]


def _af(rows):
    out = []
    for r in rows:
        x = {c: np.nan for c in AF_COLS}
        x.update(r)
        out.append(x)
    d = pd.DataFrame(out, columns=AF_COLS)
    d["date"] = pd.to_datetime(d["date"])
    return d


# ── 2. the provenance rule ──────────────────────────────────────────────────────────────────────

def test_written_back_is_no_fixture_id_and_no_ft_score():
    h = _af([{"fixture_id": 1, "FTHG": 1, "FTAG": 0},          # provider row
             {"FTHG": 2, "FTAG": 2},                           # old provider row without an id
             {"HS": 10, "AS": 5},                              # written back
             {"fixture_id": 7}])                               # an id is provider proof
    assert written_back_mask(h).tolist() == [False, False, True, False]
    assert written_back_mask(h.drop(columns=["fixture_id"])).tolist() == [False, False, True, True]
    assert {"HS", "HST", "HXG", "HIB", "HC", "HF"} <= set(WRITTEN_BACK_STAT_COLS)


@pytest.mark.skipif(not (ROOT / "output" / "af_history.parquet").exists(), reason="no af_history")
def test_rule_on_the_committed_file():
    """The definition rechecked on real data: a fixture id never comes without a score, and a
    written-back row never carries a score, an id, corners, fouls or cards (the bank's map)."""
    h = pd.read_parquet(ROOT / "output" / "af_history.parquet")
    wb = written_back_mask(h)
    assert int((h["fixture_id"].notna() & h["FTHG"].isna()).sum()) == 0
    assert wb.sum() > 0 and h.loc[wb, ["FTHG", "FTAG", "fixture_id", "HC", "AC", "HF", "AF", "HY", "HR",
                                       "home_team_id"]].notna().sum().sum() == 0


# ── 1-3. the loader ─────────────────────────────────────────────────────────────────────────────

def _hist():
    base = {"league": LG, "home_goals": 1.0, "away_goals": 0.0, "season": "2025"}
    rows = [dict(base, date=pd.Timestamp("2025-05-03"), home_team="Inter Miami", away_team="Charlotte FC"),
            dict(base, date=pd.Timestamp("2025-05-10"), home_team="Charlotte FC", away_team="Inter Miami"),
            dict(base, date=pd.Timestamp("2025-05-17"), home_team="Inter Miami", away_team="Orlando City")]
    d = pd.DataFrame(rows)
    for c in ("home_shots", "away_shots", "home_sot", "away_sot", "home_corners", "away_corners",
              "home_fouls", "away_fouls", "home_sot_ratio", "away_sot_ratio"):
        d[c] = np.nan
    return d


def _side(root, rows):
    import collect_af_history_gap as G
    p = root / DL.AF_FULLSTATS_FILL
    p.parent.mkdir(parents=True, exist_ok=True)
    full = [{**{c: np.nan for c in G.FULL_COLS}, **r} for r in rows]
    pd.DataFrame(full, columns=G.FULL_COLS).to_parquet(p, index=False)


@pytest.fixture
def bank(tmp_path, monkeypatch):
    """A tmp af_history: a provider row, two written-back rows; sidecars under tmp_path."""
    af = _af([
        {"league": LG, "date": "2025-05-03", "home_team": "Inter Miami", "away_team": "Charlotte FC",
         "FTHG": 1, "FTAG": 0, "fixture_id": 11, "HS": 14, "AS": 6, "HST": 5, "AST": 2, "HC": 7, "AC": 3,
         "HF": 10, "AF": 12},
        {"league": LG, "date": "2025-05-10", "home_team": "Charlotte FC", "away_team": "Inter Miami",
         "HS": 99, "AS": 98, "HST": 9, "AST": 8, "HXG": 3.3, "AXG": 0.1, "HIB": 20, "AIB": 1},
        {"league": LG, "date": "2025-05-17", "home_team": "Inter Miami", "away_team": "Orlando City",
         "HS": 50, "AS": 40, "HST": 20, "AST": 10}])
    p = tmp_path / "af_history.parquet"
    af.to_parquet(p, index=False)
    monkeypatch.setattr(DL, "AF_HISTORY_PATH", p)
    monkeypatch.delenv("APIFOOTBALL_KEY", raising=False)
    monkeypatch.setattr(config, "API_KEY", "")
    monkeypatch.setattr(DL, "_enrich_ht_scores", lambda df, root=None: df)
    real_s, real_f = DL._enrich_af_stats_fill, DL._enrich_af_fullstats_fill
    monkeypatch.setattr(DL, "_enrich_af_stats_fill", lambda df, root=None: real_s(df, root=tmp_path))
    monkeypatch.setattr(DL, "_enrich_af_fullstats_fill", lambda df, root=None: real_f(df, root=tmp_path))
    # the provider answer for the 2025-05-10 match only (sidecar keyed on af_history's spelling)
    _side(tmp_path, [{"league": LG, "date": "2025-05-10", "home_team": "Charlotte FC",
                      "away_team": "Inter Miami", "fixture_id": 12, "home_shots": 13.0, "away_shots": 8.0,
                      "home_sot": 4.0, "away_sot": 3.0, "home_corners": 6.0, "away_corners": 2.0,
                      "home_fouls": 11.0, "away_fouls": 9.0}])
    return p


def test_written_back_values_are_not_served_and_the_provider_replaces_them(bank):
    before = hashlib.sha256(bank.read_bytes()).hexdigest()
    out = DL._enrich_with_api_shots(_hist())
    r0, r1, r2 = out.iloc[0], out.iloc[1], out.iloc[2]
    # provider row (fixture id + score): served as stored
    assert (r0.home_shots, r0.away_shots, r0.home_corners, r0.home_fouls) == (14, 6, 7, 10)
    # written back, provider answer exists: the provider's values, never 99 / 98
    assert (r1.home_shots, r1.away_shots, r1.home_sot, r1.away_sot) == (13.0, 8.0, 4.0, 3.0)
    assert (r1.home_corners, r1.away_fouls) == (6.0, 9.0)
    # written back, no provider answer: MISSING (shots, SoT, xG, inside-box)
    assert pd.isna(r2.home_shots) and pd.isna(r2.away_sot)
    assert pd.isna(r1.home_xg) and pd.isna(r1.home_insidebox)          # 3.3 / 20 unproven
    a = out.attrs["af_written_back"]
    assert a["total"]["home_shots"] == {"blanked": 2, "provider_replaced": 1, "missing": 1}
    assert a["total"]["home_xg"] == {"blanked": 1, "provider_replaced": 0, "missing": 1}
    assert a["per_league"][LG]["home_sot"]["blanked"] == 2
    assert DL.AF_WRITTEN_BACK_AUDIT["total"]["home_shots"]["provider_replaced"] == 1
    # 3. the original file is untouched
    assert hashlib.sha256(bank.read_bytes()).hexdigest() == before


def test_the_api_exit_never_writes_af_history(bank, monkeypatch):
    """With a key the live loop runs (patched, no network) - and still nothing is banked."""
    import src.api_football_ou as OU
    monkeypatch.setenv("APIFOOTBALL_KEY", "test-not-a-key")
    monkeypatch.setattr(OU, "fetch_shots_for_league", lambda lid, season, name="": pd.DataFrame(
        [{"date": pd.Timestamp("2025-05-17"), "home_team": "Inter Miami", "away_team": "Orlando City",
          "home_shots": 17, "away_shots": 4, "home_sot": 6, "away_sot": 1}]))
    monkeypatch.setattr(config, "NEW_FORMAT_LEAGUES", [LG])
    before = hashlib.sha256(bank.read_bytes()).hexdigest()
    writes = []
    real = pd.DataFrame.to_parquet
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, *a, **k: writes.append(str(path))
                        or real(self, path, *a, **k))
    out = DL._enrich_with_api_shots(_hist())
    assert out.iloc[2].home_shots == 17                               # a live provider value is served
    assert not [w for w in writes if "af_history" in w]
    assert hashlib.sha256(bank.read_bytes()).hexdigest() == before


def test_no_writer_of_af_history_in_the_loader_and_retrain_does_not_stage_it():
    src = inspect.getsource(DL._enrich_with_api_shots) + inspect.getsource(DL._finish_af_enrichment)
    assert "to_parquet" not in src and "BANKED enrichment" not in src
    wf = (ROOT / ".github" / "workflows" / "retrain.yml").read_text(encoding="utf-8")
    staged = [ln for ln in wf.splitlines() if "af_history.parquet" in ln and not ln.strip().startswith("#")]
    assert staged == []


# ── 4. collection targets ───────────────────────────────────────────────────────────────────────

def _cf(fid, date, h, a, hid, aid, lid):
    return {"fixture_id": fid, "league_id": lid, "season": "2025", "date": date, "home_id": hid,
            "home_name": h, "away_id": aid, "away_name": a,
            "_d": pd.Timestamp(date).normalize()}


def test_targets_are_matches_whose_served_stats_are_missing(monkeypatch):
    import collect_af_history_gap as G
    monkeypatch.setattr(G, "_af_ids", lambda: {LG: 253})
    af = _af([
        # provider row with everything: not a target
        {"league": LG, "date": "2025-05-03", "home_team": "A", "away_team": "B", "FTHG": 1, "FTAG": 0,
         "fixture_id": 1, "HS": 1, "AS": 1, "HC": 1, "AC": 1, "HF": 1, "AF": 1, "HIB": 1, "AIB": 1},
        # written back with shots + inside-box, sidecar has everything: not a target
        {"league": LG, "date": "2025-05-10", "home_team": "B", "away_team": "A", "HS": 9, "AS": 9,
         "HIB": 2, "AIB": 2},
        # written back with shots + inside-box, no sidecar: a target (its values are not served)
        {"league": LG, "date": "2025-05-17", "home_team": "A", "away_team": "C", "HS": 9, "AS": 9,
         "HIB": 2, "AIB": 2},
        # provider row lacking corners only: a served_stats target
        {"league": LG, "date": "2025-05-24", "home_team": "C", "away_team": "A", "FTHG": 0, "FTAG": 0,
         "fixture_id": 4, "HS": 1, "AS": 1, "HF": 1, "AF": 1, "HIB": 1, "AIB": 1}])
    full = pd.DataFrame([{"league": LG, "date": "2025-05-10", "home_team": "B", "away_team": "A",
                          "fixture_id": 2, "home_shots": 3.0, "away_shots": 4.0, "home_corners": 1.0,
                          "away_corners": 0.0, "home_fouls": 5.0, "away_fouls": 6.0,
                          "home_insidebox": 2.0, "away_insidebox": 1.0}])
    t = G.stats_targets(af, full, statsfill=pd.DataFrame())
    got = {str(d.date()): r for d, r in zip(t["date"], t["reason"])}
    assert got == {"2025-05-17": "insidebox+served_stats", "2025-05-24": "served_stats"}


def test_dry_run_reads_only_fixtures_the_sidecar_lacks(monkeypatch, tmp_path):
    import collect_af_history_gap as G
    monkeypatch.setattr(G, "_af_ids", lambda: {LG: 253})
    monkeypatch.setattr(G, "STATS_STORE", tmp_path / "store.parquet")
    monkeypatch.setattr(G, "COVERAGE", tmp_path / "cov.parquet")
    pd.DataFrame([{"fixture_id": 31, "league_id": 253, "season": "2025", "home_id": 1, "away_id": 2,
                   "HIB": None, "AIB": None, "HXG": None, "AXG": None, "n_teams": 0, "read_from": "api",
                   "fetched_at": "x"}]).to_parquet(G.STATS_STORE)
    pd.DataFrame([{"league_id": 253, "season": "2025", "statistics_fixtures": True, "fetched_at": "x"}]
                 ).to_parquet(G.COVERAGE)
    af = _af([{"league": LG, "date": d, "home_team": h, "away_team": a, "HS": 9, "AS": 9}
              for d, h, a in (("2025-05-17", "A", "C"), ("2025-05-24", "C", "A"), ("2025-05-31", "A", "D"))])
    fx = pd.DataFrame([_cf(30, "2025-05-17", "A", "C", 1, 3, 253), _cf(31, "2025-05-24", "C", "A", 3, 1, 253),
                       _cf(32, "2025-05-31", "A", "D", 1, 4, 253)])
    full = pd.DataFrame([{"league": LG, "date": "2025-05-31", "home_team": "X", "away_team": "Y",
                          "fixture_id": 32}])                           # answer already read
    s = G.run_insidebox(dry_run=True, af=af, cache_dir=tmp_path, fullstats=full, fixtures=fx,
                        statsfill=pd.DataFrame())
    # 30: to read; 31: provider answered empty (journal n_teams 0); 32: already in the sidecar
    assert s["fixtures_to_read"] == 1 and s["planned_stats_calls_upper_bound"] == 1
    assert s["calls"] == 0


# ── 5. the team null rule ───────────────────────────────────────────────────────────────────────

def _block(**vals):
    return [{"type": k.replace("_", " "), "value": v} for k, v in vals.items()]


def test_team_null_rule():
    blk = [{"type": "Red Cards", "value": None}, {"type": "Corner Kicks", "value": 5},
           {"type": "Offsides", "value": None}, {"type": "Ball Possession", "value": None},
           {"type": "expected_goals", "value": None}, {"type": "goals_prevented", "value": None},
           {"type": "Total passes", "value": None}, {"type": "Passes %", "value": None},
           {"type": "Free Kicks", "value": None}, {"type": "Fouls", "value": "x"}]
    v = TS.team_values(blk)
    assert v["Red Cards"] == 0 and v["Offsides"] == 0 and v["Free Kicks"] == 0 and v["Corner Kicks"] == 5
    assert v["Ball Possession"] is None and v["expected_goals"] is None and v["goals_prevented"] is None
    assert v["Total passes"] is None and v["Passes %"] is None             # passes: not zero-filled
    assert v["Fouls"] is None                                               # unparseable: missing
    assert TS.team_value(blk, "Yellow Cards") is None                      # type not listed: missing
    # no answer: an empty block or a block of only nulls -> everything missing
    assert TS.team_values([]) == {} and TS.team_values(None) == {}
    assert TS.team_value([{"type": "Red Cards", "value": None}], "Red Cards") is None
    assert TS.team_count([{"type": "Corner Kicks", "value": "0"}, {"type": "Fouls", "value": None}],
                         "Fouls") == 0


def _resp(h_stats, a_stats, hid=1, aid=2):
    return [{"team": {"id": hid, "name": "H"}, "statistics": h_stats},
            {"team": {"id": aid, "name": "A"}, "statistics": a_stats}]


def test_fullstats_parse_applies_the_rule_and_stores_free_kicks():
    import collect_af_history_gap as G
    assert G.FULL_STATS["Free Kicks"] == "free_kicks"
    assert {"home_free_kicks", "away_free_kicks"} <= set(G.FULL_COLS)
    h = [{"type": "Red Cards", "value": None}, {"type": "Corner Kicks", "value": 4},
         {"type": "Free Kicks", "value": 12}, {"type": "expected_goals", "value": None}]
    a = [{"type": "Red Cards", "value": 1}, {"type": "Corner Kicks", "value": None}]
    p = G.parse_full_stats(_resp(a, h, hid=2, aid=1), 1, 2)                # order reversed: by team id
    assert p["home_red"] == 0 and p["home_corners"] == 4 and p["home_free_kicks"] == 12
    assert p["away_red"] == 1 and p["away_corners"] == 0
    assert p["home_xg"] is None and p["home_fouls"] is None                # non-count / not listed
    assert p["_unknown"] == set()
    assert G.parse_full_stats(_resp(h, [], 1, 2), 1, 2)["away_corners"] is None   # away: no answer
    assert G.parse_full_stats(_resp(h, a, 1, 9), 1, 2) is None


def test_backfill_stat_and_parse_fixture_by_team_id():
    import backfill_af_history as B
    assert B._stat([{"type": "Red Cards", "value": None}, {"type": "Fouls", "value": 3}], "Red Cards") == 0
    assert B._stat([{"type": "Red Cards", "value": None}], "Red Cards") is None
    assert B._statf([{"type": "expected_goals", "value": None}, {"type": "Fouls", "value": 3}],
                    "expected_goals") is None
    f = {"fixture": {"id": 5, "date": "2025-05-03T18:00:00+00:00"},
         "teams": {"home": {"id": 1, "name": "H"}, "away": {"id": 2, "name": "A"}},
         "goals": {"home": 1, "away": 0}}
    h = [{"type": "Total Shots", "value": 10}, {"type": "Corner Kicks", "value": None}]
    a = [{"type": "Total Shots", "value": 4}, {"type": "Corner Kicks", "value": 3}]
    row = B.parse_fixture("USA MLS", "2025", f, _resp(a, h, hid=2, aid=1))  # away block first
    assert (row["HS"], row["AS"], row["HC"], row["AC"]) == (10, 4, 0, 3)
    row = B.parse_fixture("USA MLS", "2025", f, _resp(h, a, hid=1, aid=9))  # not the fixture's clubs
    assert row["HS"] is None and row["AC"] is None


def test_live_team_stat_parsers_use_the_rule(monkeypatch):
    import src.api_football_ou as OU
    import src.api_football as AFm
    body = {"response": _resp([{"type": "Total Shots", "value": 9}, {"type": "Fouls", "value": None},
                               {"type": "Ball Possession", "value": None},
                               {"type": "expected_goals", "value": None}],
                              [{"type": "Total Shots", "value": None}, {"type": "Corner Kicks", "value": 2}])}
    monkeypatch.setattr(OU, "_get", lambda *a, **k: body)
    s = OU.get_fixture_statistics(1)
    assert s["home_shots"] == 9 and s["home_fouls"] == 0 and s["away_shots"] == 0
    assert s["home_possession"] is None and s["home_xg"] is None and s["home_corners"] is None
    monkeypatch.setattr(AFm, "_get", lambda *a, **k: body)
    t = AFm.get_fixture_statistics(1)
    assert t["home"]["shots"] == 9 and t["home"]["fouls"] == 0 and t["home"]["possession"] is None
    assert t["away"]["shots"] == 0 and t["away"]["corners"] == 2 and t["away"]["offsides"] is None


def test_player_side_team_stats_and_cup_corners_use_the_rule(tmp_path, monkeypatch):
    from player_model import data_fetcher as DF
    import collect_cup_player_stats as CUP
    monkeypatch.setattr(DF, "FIXTURE_STATS_CACHE_DIR", tmp_path)
    monkeypatch.setattr(DF, "_api_get", lambda *a: {"response": [
        {"team": {"name": "H"}, "statistics": [{"type": "Corner Kicks", "value": None},
                                                {"type": "Total Shots", "value": 7},
                                                {"type": "Ball Possession", "value": None}]}]})
    assert DF._fetch_fixture_statistics(8) == {"H": {"corners": 0, "shots_total": 7, "possession": None}}
    assert json.loads((tmp_path / "stats_8.json").read_text())["schema"] == 3
    (tmp_path / "stats_9.json").write_text(json.dumps({"schema": 2, "teams": {"H": {"corners": None}}}))
    assert DF._fetch_fixture_statistics(9) == {"H": {"corners": None}}     # v2: served as written
    assert CUP._corner_value([{"type": "Corner Kicks", "value": None}, {"type": "Fouls", "value": 4}]) == 0.0
    assert np.isnan(CUP._corner_value([{"type": "Corner Kicks", "value": None}]))
    assert np.isnan(CUP._corner_value([]))


# ── 6. identity ─────────────────────────────────────────────────────────────────────────────────

PROVEN = [  # (league pool, spelling, the club's history spelling) - fixture counts in src/team_names.py
    (["A. Lustenau", "Rapid Vienna"], "Austria Lustenau", "A. Lustenau"),
    (["Orgryte", "AIK"], "Orgryte IS", "Orgryte"),
    (["UCD", "Bohemians"], "UC Dublin", "UCD"),
    (["Athletico-PR", "Atletico-MG"], "Atletico Paranaense", "Athletico-PR"),
    (["Atletico Goianiense", "Atletico-MG"], "Atletico GO", "Atletico Goianiense"),
    (["America Mineiro", "Atletico-MG"], "America MG", "America Mineiro"),
    (["Machida", "Kashima Antlers"], "FC Machida Zelvia", "Machida"),
    (["Shanghai Port", "Shanghai Shenhua"], "Shanghai SIPG FC", "Shanghai Port"),
]


@pytest.mark.parametrize("pool,name,want", PROVEN)
def test_proven_identity_pairs_resolve(pool, name, want):
    from src.team_names import resolve
    assert resolve(name, pool) == want
    assert resolve(want, [name] + [p for p in pool if p != want]) == name


def test_identity_pairs_never_cross_clubs():
    from src.team_names import resolve
    assert resolve("America MG", ["Atletico-MG"]) is None
    assert resolve("Atletico GO", ["Atletico-MG", "Atletico Paranaense"]) is None
    assert resolve("Shanghai SIPG FC", ["Shanghai Shenhua"]) is None
    assert resolve("Orgryte IS", ["IFK Goteborg", "GAIS"]) is None


def test_stats_fill_places_a_row_keyed_on_a_renamed_spelling(tmp_path, monkeypatch):
    p = tmp_path / DL.AF_STATS_FILL
    p.parent.mkdir(parents=True)
    pd.DataFrame([{"league": LG, "date": "2025-05-03", "home_team": "Herons", "away_team": "Charlotte FC",
                   "HIB": 9.0, "AIB": 4.0, "HXG": 1.2, "AXG": 0.3}]).to_parquet(p)
    h = _hist()
    for c in ("home_insidebox", "away_insidebox", "home_xg", "away_xg"):
        h[c] = np.nan
    monkeypatch.setattr(DL, "IDENTITY_ALIASES", {})
    assert pd.isna(DL._enrich_af_stats_fill(h.copy(), root=tmp_path).loc[0, "home_insidebox"])
    monkeypatch.setattr(DL, "IDENTITY_ALIASES", {LG: {"Herons": "Inter Miami"}})
    out = DL._enrich_af_stats_fill(h.copy(), root=tmp_path)
    assert out.loc[0, "home_insidebox"] == 9.0 and out.loc[0, "away_xg"] == 0.3
    monkeypatch.setattr(DL, "IDENTITY_ALIASES", {"Japan J-League": {"Herons": "Inter Miami"}})
    assert pd.isna(DL._enrich_af_stats_fill(h.copy(), root=tmp_path).loc[0, "home_insidebox"])
