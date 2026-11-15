"""identity/1.1: Ireland Premier Division 'Bray Wanderers' -> 'Bray' (owner-accepted evidence merge,
2026-11-02). fd_history and af_history hold 2024-11-16 Bray Wanderers 1-3 Drogheda United ==
Bray 1-3 Drogheda: one fixture under two spellings.

    python -m pytest tests/test_identity_bray.py -q
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from src.identity_repair import IDENTITY_VERSION, canonicalise_and_dedupe, load_alias_map

ROOT = Path(__file__).resolve().parents[1]
ALIASES = ROOT / "config" / "standard_ou25_identity_aliases.json"
L = "Ireland Premier Division"


def test_map_holds_the_pair_and_the_version_moved():
    m = load_alias_map(ALIASES)[L]
    assert m["Bray Wanderers"] == "Bray" and m["Drogheda United"] == "Drogheda"
    d = json.loads(ALIASES.read_text(encoding="utf-8"))
    assert d["version"] == "identity/1.1" and IDENTITY_VERSION == "v9r-identity-1.1"
    assert not [p for p in d["rejected_pairs"] if {p["a"], p["b"]} == {"Bray", "Bray Wanderers"}]


def test_the_two_spellings_collapse_to_one_fixture():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-11-16", "2024-11-16", "2025-11-07"]),
        "league": [L] * 3,
        "home_team": ["Bray Wanderers", "Bray", "Bray Wanderers"],
        "away_team": ["Drogheda United", "Drogheda", "Waterford"],
        "home_goals": [1, 1, 1], "away_goals": [3, 3, 2],
    })
    out, rep = canonicalise_and_dedupe(df, load_alias_map(ALIASES), derive=False)
    assert len(out) == 2 and rep["quarantined_fixtures"] == 0
    assert set(out["home_team"]) == {"Bray"}
    assert rep["identity_version"] == "v9r-identity-1.1"


@pytest.mark.skipif(not (ROOT / "output" / "fd_history.parquet").exists()
                    or not (ROOT / "output" / "af_history.parquet").exists(), reason="no history bank")
def test_no_two_different_fixtures_collide_on_the_committed_history():
    fd = pd.read_parquet(ROOT / "output" / "fd_history.parquet")
    af = pd.read_parquet(ROOT / "output" / "af_history.parquet").rename(
        columns={"FTHG": "home_goals", "FTAG": "away_goals"})
    cols = ["date", "league", "home_team", "away_team", "home_goals", "away_goals"]
    d = pd.concat([fd[cols], af[cols]], ignore_index=True)
    d = d[d["league"].astype(str) == L].copy()
    raw = d.copy()
    m = load_alias_map(ALIASES)[L]
    for c in ("home_team", "away_team"):
        d[c] = d[c].map(lambda t: m.get(t, t))
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    assert not (d["home_team"] == d["away_team"]).any()
    # the two spellings never met each other
    pair = {"Bray", "Bray Wanderers"}
    assert not any({h, a} == pair for h, a in zip(raw["home_team"], raw["away_team"]))
    bray = d[(d["home_team"] == "Bray") | (d["away_team"] == "Bray")]
    assert (bray.groupby("date").apply(
        lambda g: len(g[["home_team", "away_team"]].drop_duplicates())) <= 1).all()
    sc = (d.dropna(subset=["home_goals", "away_goals"])
            .groupby(["date", "home_team", "away_team"])[["home_goals", "away_goals"]].nunique())
    assert (sc.max(axis=1) <= 1).all()
