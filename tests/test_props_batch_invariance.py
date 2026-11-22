"""Owner P0 item 5 (2026-11-22): every installed props artifact is batch-invariant.

The same player scored alone, with 10 others, with 1,000 others, in a shuffled batch and beside
unrelated rows with missing inputs gets the IDENTICAL probability (|diff| <= 1e-12, the float rounding
of a BLAS dot product in the recipe's logistic part; the tree and base-rate models are bit-identical) - no statistic of the scoring batch
(the legacy `_prep` batch median) reaches an installed model. Probe rows are real player_history rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from player_model import model as M
from src.integrity import props_certification as PCERT

ARTS = PCERT.installed_artifacts()


@pytest.fixture(scope="module")
def rows():
    return PCERT.probe_frame()


def test_every_owner_listed_artifact_is_installed():
    want = {"goals": "5de94714", "assists": "abf39bb5", "sot": "571b102c", "sot2": "5b5fad60",
            "cards": "bfd5e1cc", "goals2": "81b5b864", "goals3": "89f588a0", "sot3": "f11626d9",
            "sot4": "e0d074b8"}
    chal = {"goals2": "e218cb6c", "goals3": "62e5b2bb", "sot3": "a3646f1d", "sot4": "8ed4b805"}
    got = {a["key"]: a["version"] for a in ARTS}
    for m, s in want.items():
        assert got[f"{m}:champion"].endswith(s)
    for m, s in chal.items():
        assert got[f"{m}:challenger"].endswith(s)


@pytest.mark.parametrize("art", ARTS, ids=[a["key"] for a in ARTS])
def test_installed_artifact_is_batch_invariant(art, rows):
    pl = PCERT.load_raw(PCERT.read_bytes(art["path"]))
    r = PCERT.batch_invariance(pl, rows)
    assert r["targets_scored"] == r["targets"] > 0
    for k in ("with_10_others", "with_1000_others", "shuffled", "with_unrelated_missing_rows"):
        assert r[k]["rows_differing"] == 0 and r[k]["max_abs_diff"] <= PCERT.FLOAT_TOL, (art["key"], k, r[k])
        if pl["candidate"] != "recipe_retrained":                  # no BLAS dot product: bit-identical
            assert r[k]["rows_not_bit_identical"] == 0, (art["key"], k, r[k])
    assert r["invariant"]


def test_base_rate_reads_the_player_not_the_batch(rows):
    a = next(a for a in ARTS if a["key"] == "goals2:champion")
    pl = PCERT.load_raw(PCERT.read_bytes(a["path"]))
    m = pl["model"]
    known = next(iter(m.rates_))
    df = rows.head(50).copy()
    df.loc[0, "player_id"] = known
    df.loc[1, "player_id"] = -7                                    # never seen in training
    p = M.predict_proba(df, pl).to_numpy()
    assert p[0] == m.rates_[known] and p[1] == m.market_rate_
    assert M.predict_proba(df.iloc[[0]], pl).iloc[0] == p[0]


def test_the_probe_catches_a_batch_median_model():
    """Teeth: a legacy-recipe payload (batch-median _prep) is NOT invariant under the same probe."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.normal(size=400), "f2": rng.normal(size=400)})
    y = (X["f1"] + rng.normal(size=400) > 0).astype(int)
    pl = {"market": "probe", "feature_cols": ["f1", "f2"], "models": {"lr": LogisticRegression().fit(X, y)}}
    rows = pd.DataFrame({"f1": rng.normal(size=1300), "f2": rng.normal(size=1300) + 3.0,
                         "player_id": np.arange(1, 1301), "fixture_id": 1})
    rows.loc[:24, "f2"] = np.nan                                   # the targets miss f2 -> batch median
    r = PCERT.batch_invariance(pl, rows)
    assert not r["invariant"]
