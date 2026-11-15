"""fix16/readers: every DIRECT reader of output/fd_history.parquet that uses its results applies the
owner's history corrections exactly as the loader serves them (src.history_corrections.serve_history:
removal lists - proven cup rows, awarded results, duplicate copies -, 90-minute scores, provider
dates), and every af_history results reader applies src.af_history_corrections. The builders and
collectors of the lists and the raw-bank audits stay raw (listed below with the reason).

All frames here are built in the test; the committed lists are swapped for tmp lists, nothing under
output/ is written.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from src import af_history_corrections as AC  # noqa: E402
from src import history_corrections as HC  # noqa: E402

LG = "USA MLS"
NAN = np.nan

#            date          home         away       FT     HT
HISTORY = [("2026-05-03", "Austin", "Dallas", 1, 0, 0, 0),        # proven cup row      -> out
           ("2026-06-01", "Miami", "Orlando", 2, 2, NAN, NAN),    # kept copy (no HT)   -> HT 1-1 filled
           ("2026-06-02", "Miami", "Orlando", 2, 2, 1, 1),        # duplicate copy      -> out
           ("2026-07-01", "Houston", "Seattle", 3, 0, 1, 0),      # awarded result      -> out
           ("2026-08-01", "LA Galaxy", "LAFC", 3, 2, 1, 1),       # AET                 -> 2-2
           ("2026-09-02", "Portland", "Seattle", 1, 0, 0, 0),     # one day off         -> 2026-09-01
           ("2026-09-10", "Austin", "Miami", 0, 0, 0, 0)]         # untouched


def _history() -> pd.DataFrame:
    return pd.DataFrame([{"league": LG, "date": pd.Timestamp(d), "season": "2026", "home_team": h,
                          "away_team": a, "home_goals": float(hg), "away_goals": float(ag),
                          "ht_home_goals": hh, "ht_away_goals": ha,
                          "home_shots": NAN, "away_shots": NAN, "home_sot": NAN, "away_sot": NAN}
                         for d, h, a, hg, ag, hh, ha in HISTORY])


@pytest.fixture()
def lists(tmp_path, monkeypatch):
    """The five owner lists as tmp files, swapped in for the committed ones; no af list."""
    c = tmp_path / "corr"
    c.mkdir()
    row = lambda d, h, a, hg, ag: {"league": LG, "date": d, "home_team": h, "away_team": a,  # noqa: E731
                                   "home_goals": hg, "away_goals": ag}
    pd.DataFrame([row("2026-05-03", "Austin", "Dallas", 1, 0)]).to_csv(c / "cup.csv", index=False)
    pd.DataFrame([{**row("2026-06-02", "Miami", "Orlando", 2, 2), "kept_date": "2026-06-01",
                   "kept_home_team": "Miami", "kept_away_team": "Orlando"}]).to_csv(c / "dup.csv", index=False)
    pd.DataFrame([row("2026-07-01", "Houston", "Seattle", 3, 0)]).to_csv(c / "awd.csv", index=False)
    pd.DataFrame([{"league": LG, "date": "2026-08-01", "home_team": "LA Galaxy", "away_team": "LAFC",
                   "history_ft_home": 3, "history_ft_away": 2, "ft90_home": 2, "ft90_away": 2}]
                 ).to_csv(c / "s90.csv", index=False)
    pd.DataFrame([{"league": LG, "history_date": "2026-09-02", "home_team": "Portland",
                   "away_team": "Seattle", "home_goals": 1, "away_goals": 0,
                   "provider_date": "2026-09-01", "fixture_id": 77}]).to_csv(c / "dates.csv", index=False)
    monkeypatch.setattr(HC, "CUP_EXCLUDED", c / "cup.csv")
    monkeypatch.setattr(HC, "DUP_REMOVED", c / "dup.csv")
    monkeypatch.setattr(HC, "AWARDED_EXCLUDED", c / "awd.csv")
    monkeypatch.setattr(HC, "SCORE_90", c / "s90.csv")
    monkeypatch.setattr(HC, "DATE_CORR", c / "dates.csv")
    monkeypatch.setattr(AC, "AF_DUP_REMOVED", c / "no_af_list.csv")
    return c


def _rows(df, hg="home_goals", ag="away_goals"):
    return sorted((pd.Timestamp(d).strftime("%Y-%m-%d"), str(h), str(a), float(x), float(y))
                  for d, h, a, x, y in zip(df["date"], df["home_team"], df["away_team"], df[hg], df[ag]))


SERVED = [("2026-06-01", "Miami", "Orlando", 2.0, 2.0), ("2026-08-01", "LA Galaxy", "LAFC", 2.0, 2.0),
          ("2026-09-01", "Portland", "Seattle", 1.0, 0.0), ("2026-09-10", "Austin", "Miami", 0.0, 0.0)]


def _write_fd(tmp_path) -> Path:
    out = tmp_path / "output"
    out.mkdir(exist_ok=True)
    _history().to_parquet(out / "fd_history.parquet", index=False)
    return out


# ---- the shared helper -----------------------------------------------------------------------------

def test_serve_history_applies_every_list_in_the_loader_order(lists):
    h = _history()
    out = HC.serve_history(h, tag="t")
    assert _rows(out) == SERVED
    kept = out[out["home_team"] == "Miami"].iloc[0]
    assert (kept["ht_home_goals"], kept["ht_away_goals"]) == (1, 1)     # filled from the removed copy
    rep = out.attrs["history_corrections"]
    assert (rep["cup_rows_excluded"], rep["awarded_rows_excluded"], rep["duplicate_rows_removed"],
            rep["score_rows_corrected_90min"]) == (1, 1, 1, 1)
    assert out.attrs["history_date_corrections"]["rows_redated"] == 1
    assert len(h) == 7 and h["date"].iloc[5] == pd.Timestamp("2026-09-02")   # input untouched


def test_serve_history_matches_the_loader(lists):
    """The same rows the loader's identity pipeline starts from (identity_before_one_match runs the
    two lists, then renames clubs: the 4 rows and their scores/dates are the same)."""
    import src.data_loader as DL
    loader = DL._apply_date_corrections(DL._apply_history_corrections(_history()))
    assert _rows(loader) == _rows(HC.serve_history(_history())) == SERVED


def test_serve_history_without_the_score_columns_never_guesses(lists):
    h = _history().drop(columns=["home_goals", "away_goals"])
    out = HC.serve_history(h)
    assert len(out) == len(h)                                          # nothing recognised, nothing removed
    assert out.attrs["history_corrections"]["listed_rows_changed_since_listing"] == 3


# ---- the fixed readers -----------------------------------------------------------------------------

def test_strength_reads_fd_history_under_the_corrections(lists, tmp_path):
    from src.strength import results as R
    out = _write_fd(tmp_path)
    # af_history rows of other pairings teach the club index the names (no overlap with the fd rows)
    af = pd.DataFrame([{"league": LG, "date": pd.Timestamp(f"2025-0{i + 3}-05"), "season": "2025",
                        "home_team": h, "away_team": a, "FTHG": 1.0, "FTAG": 1.0}
                       for i, (h, a) in enumerate([("Miami", "LA Galaxy"), ("Orlando", "LAFC"),
                                                   ("Portland", "Austin"), ("Seattle", "Dallas"),
                                                   ("Houston", "Miami")])])
    af.to_parquet(out / "af_history.parquet", index=False)
    m, _, rep = R.load_results(collected=tmp_path / "none.parquet", af=out / "af_history.parquet",
                               fd=out / "fd_history.parquet", verbose=False)
    fd = m[m["source"] == "fd"]
    got = sorted((d.strftime("%Y-%m-%d"), h.split("|")[-1], a.split("|")[-1], x, y)
                 for d, h, a, x, y in zip(fd["date"], fd["home_key"], fd["away_key"],
                                          fd["home_goals"], fd["away_goals"]))
    assert got == SERVED
    assert rep["fd_history_corrections"]["rows_removed"] == 3
    assert rep["fd_history_corrections"]["rows_redated"] == 1


def test_edge_replay_settles_on_the_corrected_history(lists, tmp_path):
    from src.research import edge_replay as ER
    r = ER.load_results(_write_fd(tmp_path))
    assert _rows(r, "hg", "ag") == SERVED
    e = pd.DataFrame([{"league": LG, "date": d, "home_team": h, "away_team": a}
                      for d, h, a in [("2026-05-03", "Austin", "Dallas"), ("2026-08-01", "LA Galaxy", "LAFC"),
                                      ("2026-07-01", "Houston", "Seattle")]])
    s = ER.settle(e, r)
    assert np.isnan(s["hg"].iloc[0])                                   # a cup row settles nothing
    assert (s["hg"].iloc[1], s["ag"].iloc[1]) == (2.0, 2.0)            # the 90-minute score
    assert np.isnan(s["hg"].iloc[2])                                   # an awarded result is not a score


def test_update_results_ht_graders_read_the_corrected_history(lists, tmp_path):
    import update_results as U
    d = U._read_fd_ht(_write_fd(tmp_path) / "fd_history.parquet", "t")
    assert list(d.columns) == ["league", "date", "home_team", "away_team", "hh", "ha"]
    got = sorted((pd.Timestamp(x).strftime("%Y-%m-%d"), h, a, hh, ha)
                 for _, x, h, a, hh, ha in d.itertuples(index=False, name=None))
    assert [g[:3] for g in got] == [s[:3] for s in SERVED]
    assert got[0][3:] == (1, 1)                                        # the kept copy's HT, filled
    src = (ROOT / "update_results.py").read_text(encoding="utf-8")
    assert src.count("_read_fd_ht(") == 3                              # def + both graders


def test_price_provenance_outcomes_use_the_corrected_history(lists, tmp_path, monkeypatch):
    cwd = os.getcwd()
    try:
        import integrity_price_provenance as PP
    finally:
        os.chdir(cwd)
    _write_fd(tmp_path)
    monkeypatch.setattr(PP, "ROOT", tmp_path)
    o = PP.outcomes()
    flat = sorted((d, h, a, tg) for (lg, d), m in o.items() for (h, a), (tg, _) in m.items())
    assert flat == [(d, h, a, x + y) for d, h, a, x, y in SERVED]


def test_join_replay_base_uses_the_corrected_history(lists, tmp_path, monkeypatch):
    cwd = os.getcwd()
    try:
        import integrity_af_join_replay as JR
    finally:
        os.chdir(cwd)
    import src.data_loader as DL
    _write_fd(tmp_path)
    monkeypatch.setattr(JR, "ROOT", tmp_path)
    monkeypatch.setattr(DL, "_load_backtest_archive", lambda: pd.DataFrame())
    b = JR.base_history()
    assert _rows(b) == SERVED
    assert b.loc[b["home_team"] == "LA Galaxy", "total_goals"].iloc[0] == 4.0


def test_dashboard_pages_read_both_banks_under_the_corrections():
    live = next((ROOT / "pages").glob("6_*Live.py")).read_text(encoding="utf-8")
    assert "serve_history(fd" in live and "SERVE_COLS" in live
    assert "apply_af_history_corrections(af" in live and "FILL_COLS" in live
    agent = next((ROOT / "pages").glob("10_*Agent_Analysis.py")).read_text(encoding="utf-8")
    assert "serve_history(h" in agent and "SERVE_COLS" in agent


# ---- the inventory -----------------------------------------------------------------------------------

#: production readers of the history banks and how each reads them. A NEW reader fails this test
#: until it is classified here.
CORRECTED = {          # fd_history through serve_history (and af_history through the af rule)
    "src/strength/results.py", "src/research/edge_replay.py", "update_results.py",
    "pages/6_⚡_Live.py", "pages/10_🤖_Agent_Analysis.py",
    "scripts/integrity_price_provenance.py", "scripts/integrity_af_join_replay.py",
}
RAW = {                # why the raw file is the right input
    "src/data_loader.py": "the loader itself: applies the lists in identity_before_one_match / local path",
    "src/integrity/data_truth.py": "audits the RAW banks (names, freshness, misfiled leagues); its "
                                   "one-match check runs data_loader.one_club_one_match, which applies them",
    "scripts/build_history_corrections.py": "builds the fd lists FROM the raw rows",
    "scripts/build_af_history_corrections.py": "builds the af list FROM the raw rows",
    "scripts/collect_ht_scores.py": "collector keyed on the history's OWN key (drop_corrected_rows; "
                                    "the loader re-keys its sidecar onto the provider date)",
    "scripts/collect_af_history_gap.py": "collector keyed on af_history's own key (applies the af rule)",
    "scripts/backfill_af_history.py": "writer of af_history; reads league/date spans only to plan fetches",
    "scripts/collect_competition_fixtures.py": "reads the earliest date only to plan fetches",
    "scripts/audit_data_files.py": "profiles the raw committed files (not run by a workflow)",
    "scripts/dashboard_manifest.py": "lists the files the pages open (an audit hook, no results)",
    "scripts/prove_af_history_aliases.py": ("P7: proves aliases from the history names AS STORED, the same raw "
                                            "rows the collectors' matchers read (no result is served)"),
    "scripts/sync_results_from_main.py": ("P7: copies main's newer rows into the raw bank file itself; "
                                          "corrections stay beside it (sidecar), nothing is served"),
    "scripts/audit_history_fixture_claims.py": ("fix/history-dupes: counts provider fixtures claimed twice "
                                                "in the RAW, corrected (apply_af_history_corrections / "
                                                "apply_history_corrections) and served frames; nothing is served"),
    "scripts/build_history_supplement.py": ("fix/history-supplement: reads the banks through data_loader."
                                            "one_club_one_match (which applies the lists) to find the provider "
                                            "fixtures the served history lacks"),
    "scripts/probe_oddsapi_history.py": ("P7: reads league / date only to pick one probe match day per "
                                         "league for the OddsAPI history probe (no result is served)"),
}


def test_every_history_bank_reader_is_classified():
    found = set()
    files = [p for d in ("src", "player_model", "dashboard_data", "pages", "scripts")
             for p in (ROOT / d).rglob("*.py")]
    files += list(ROOT.glob("*.py"))
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        if rel.startswith("scripts/audit/"):
            continue
        t = p.read_text(encoding="utf-8-sig", errors="replace")
        if re.search(r"(fd|af)_history\.parquet", t) and re.search(r"read_parquet|_read\(|_bank_matches", t):
            found.add(rel)
    assert found - CORRECTED - set(RAW) == set(), "unclassified history-bank reader(s)"
    for rel in CORRECTED:
        t = (ROOT / rel).read_text(encoding="utf-8-sig")
        assert "serve_history" in t, rel
    for rel in ("src/strength/results.py", "src/research/edge_replay.py", "pages/6_⚡_Live.py"):
        assert "apply_af_history_corrections" in (ROOT / rel).read_text(encoding="utf-8-sig"), rel
