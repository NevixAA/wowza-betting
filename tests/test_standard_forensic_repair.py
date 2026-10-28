"""Standard v9 forensic repair (2026-10-28) — required automated tests 1-11, 13, 14 (+ R2 items).

Each defect is tested twice where it makes sense: the REPAIRED recipe must not have it, and a
"documents legacy" test pins that the LEGACY recipe (what serves, byte-identical by default) still
does — so nobody can mistake the legacy numbers for clean ones, and a silent legacy change shows up.
All data is synthetic; nothing touches the network or a serving file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src import model as M
from src.features_repaired import (STANDARD_SERVING_FEATURES, build_features_repaired,
                                   build_upcoming_features_repaired, feature_cols_for)
from src.imputer import apply_imputer, fit_imputer

STD_A, STD_B = "Championship", "League One"
NF = "Brazil Serie A"


def _rounds(teams):
    """Double round robin (circle method): every club plays at most once per matchday."""
    t = list(teams) + ([None] if len(teams) % 2 else [])
    n, out = len(t), []
    for r in range(n - 1):
        pairs = [(t[i], t[n - 1 - i]) for i in range(n // 2)]
        out.append([(h, a) for h, a in pairs if h is not None and a is not None])
        t = [t[0]] + [t[-1]] + t[1:-1]
    return out + [[(a, h) for h, a in rd] for rd in out]


def _season(league, teams, start, seed):
    rng = np.random.default_rng(seed)
    rows, d = [], pd.Timestamp(start)
    games = []
    for rd in _rounds(teams):
        d += pd.Timedelta(days=7)
        games += [(d, h, a) for h, a in rd]
    for d, h, a in games:
        hg, ag = int(rng.poisson(1.5)), int(rng.poisson(1.1))
        hth, hta = min(hg, int(rng.binomial(hg, 0.45))), min(ag, int(rng.binomial(ag, 0.45)))
        rows.append({"date": d, "league": league, "season": "x", "home_team": h, "away_team": a,
                     "home_goals": hg, "away_goals": ag, "ht_home_goals": hth,
                     "ht_away_goals": hta, "home_shots": rng.integers(5, 20),
                     "away_shots": rng.integers(4, 18), "home_sot": rng.integers(1, 8),
                     "away_sot": rng.integers(1, 7), "home_corners": rng.integers(2, 10),
                     "away_corners": rng.integers(1, 9), "home_fouls": rng.integers(8, 16),
                     "away_fouls": rng.integers(8, 16),
                     "odds_over25": rng.uniform(1.7, 2.3), "odds_under25": rng.uniform(1.6, 2.2)})
    df = pd.DataFrame(rows)
    df["total_goals"] = df.home_goals + df.away_goals
    df["over25"] = (df["total_goals"] > 2.5).astype(float)
    df["btts"] = ((df.home_goals > 0) & (df.away_goals > 0)).astype(float)
    return df


def _frame():
    ta = [f"A{i}" for i in range(10)] + ["Rovers"]
    tb = [f"B{i}" for i in range(10)] + ["Rovers"]          # same club name, other league
    parts = []
    for k, start in enumerate(("2023-08-01", "2024-08-01")):
        parts.append(_season(STD_A, ta, start, 10 + k))
        parts.append(_season(STD_B, tb, start, 20 + k))
    df = pd.concat(parts, ignore_index=True)
    # League B's "Rovers" score a lot at half-time; League A's never do.
    a_rov = (df.league == STD_A)
    df.loc[a_rov & (df.home_team == "Rovers"), "ht_home_goals"] = 0
    df.loc[a_rov & (df.away_team == "Rovers"), "ht_away_goals"] = 0
    b_rov = (df.league == STD_B)
    df.loc[b_rov & (df.away_team == "Rovers"), "ht_away_goals"] = 3
    df.loc[b_rov & (df.home_team == "Rovers"), "ht_home_goals"] = 3
    return df.sort_values("date", kind="mergesort").reset_index(drop=True)


@pytest.fixture(scope="module")
def raw():
    return _frame()


@pytest.fixture(scope="module")
def feat(raw):
    return build_features_repaired(raw)


FEATS = feature_cols_for("none")


@pytest.fixture(scope="module")
def repaired(feat):
    rows = feat.dropna(subset=["over25", "home_scored_last5"])
    return M.train_repaired(rows, "over25", feature_cols=FEATS)


@pytest.fixture(scope="module")
def legacy(feat):
    rows = feat.dropna(subset=["over25", "home_scored_last5"])
    return M.train(rows, "over25", feature_cols=FEATS)


def _payload(res):
    return M.build_payload(res, "over25")


# ── 1 batch-composition invariance ──────────────────────────────────────────────────────────

def test_01_repaired_probability_is_batch_invariant(feat, repaired):
    pay = _payload(repaired)
    rows = feat.dropna(subset=["home_scored_last5"]).tail(150).copy()
    rows.iloc[0, rows.columns.get_loc("home_corners_pg_roll")] = np.nan   # a row that needs imputing
    one = rows.iloc[[0]]
    p1 = M.predict_proba(one, payload=pay).iloc[0]
    p11 = M.predict_proba(rows.iloc[:11], payload=pay).iloc[0]
    p101 = M.predict_proba(rows.iloc[:101], payload=pay).iloc[0]
    assert p1 == pytest.approx(p11, abs=1e-12) == pytest.approx(p101, abs=1e-12)


def test_01b_documents_legacy_batch_variance(feat, legacy):
    pay = _payload(legacy)
    assert "imputer" not in pay                        # legacy payload shape is unchanged
    rows = feat.dropna(subset=["home_scored_last5"]).tail(150).copy()
    rows.iloc[0, rows.columns.get_loc("home_scored_last5")] = np.nan
    p1 = M.predict_proba(rows.iloc[[0]], payload=pay).iloc[0]
    p101 = M.predict_proba(rows.iloc[:101], payload=pay).iloc[0]
    assert abs(p1 - p101) > 1e-6, "legacy imputes from the batch: the board changes the probability"


# ── 2 + 3 imputer fitted on training rows only, never on validation/holdout ─────────────────

def test_02_imputer_fitted_on_base_block_only(feat, repaired):
    imp = repaired.meta["imputer"]
    base = repaired.meta["blocks"]["base"]
    assert imp["fitted_on_rows"] == base["rows"]
    assert imp["fitted_on_dates"][1] <= base["dates"][1]
    assert imp["fitted_on_dates"][1] < repaired.meta["blocks"]["holdout"]["dates"][0]


def test_03_no_holdout_medians(feat, repaired):
    pay = _payload(repaired)
    rows = feat.dropna(subset=["over25", "home_scored_last5"])
    blocks = M.four_block_split(rows, "over25")
    base_med = blocks["base"]["home_scored_last5"].median()
    assert pay["imputer"]["features"]["home_scored_last5"]["median"] == pytest.approx(base_med)
    hold = blocks["holdout"].copy()
    hold["home_scored_last5"] = np.nan                      # every holdout row needs imputing
    X, _, _ = apply_imputer(hold, pay["imputer"], FEATS)
    assert np.allclose(X["home_scored_last5"], base_med)
    other = hold.copy()
    other.loc[other.index[1:], "away_scored_last5"] += 2.0  # perturb OTHER rows only
    assert M.predict_proba(hold, payload=pay).iloc[0] == pytest.approx(
        M.predict_proba(other, payload=pay).iloc[0], abs=1e-12)


# ── 4 historical replay == upcoming builder at the same cutoff ─────────────────────────────

def test_04_replay_equals_upcoming_builder(raw, feat):
    last = feat["date"].max()
    fx = feat[feat["date"] == last]
    up = build_upcoming_features_repaired(
        fx[["date", "league", "home_team", "away_team", "odds_over25", "odds_under25"]],
        raw, cutoff=last)
    a = fx[STANDARD_SERVING_FEATURES].reset_index(drop=True).astype(float)
    b = up[STANDARD_SERVING_FEATURES].astype(float)
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-12)


# ── 5 home_advantage varies live when it varies in training ─────────────────────────────────

def test_05_home_advantage_not_constant_live(raw, feat):
    assert feat["home_advantage"].nunique() > 1
    nxt = feat["date"].max() + pd.Timedelta(days=7)
    teams = sorted(raw.loc[raw.league == STD_A, "home_team"].unique())
    up = pd.DataFrame({"date": nxt, "league": STD_A, "home_team": teams[:5],
                       "away_team": teams[5:10], "odds_over25": 2.0, "odds_under25": 1.8})
    f = build_upcoming_features_repaired(up, raw)
    assert f["home_advantage"].nunique() > 1


def test_05b_documents_legacy_constant_home_advantage(raw):
    from src.feature_engineering import build_upcoming_features
    nxt = raw["date"].max() + pd.Timedelta(days=7)
    up = pd.DataFrame({"date": nxt, "league": STD_A, "home_team": ["A0", "A1"],
                       "away_team": ["A2", "A3"], "odds_over25": 2.0, "odds_under25": 1.8})
    f = build_upcoming_features(up, raw)
    assert set(f["home_advantage"]) == {1.0}


# ── 6 away HT features league-scoped (same club name in two leagues) ────────────────────────

def test_06_away_ht_is_league_scoped(raw):
    nxt = raw["date"].max() + pd.Timedelta(days=7)
    up = pd.DataFrame({"date": [nxt], "league": [STD_A], "home_team": ["A0"],
                       "away_team": ["Rovers"], "odds_over25": [2.0], "odds_under25": [1.8]})
    f = build_upcoming_features_repaired(up, raw)
    assert f["away_ht_scored_last5"].iloc[0] == 0.0      # only League A's Rovers (never score at HT)
    from src.feature_engineering import build_upcoming_features
    g = build_upcoming_features(up, raw)
    assert g["away_ht_scored_last5"].iloc[0] > 0.0       # legacy: League B's Rovers leak in


def test_06b_training_is_league_scoped_too(feat):
    a = feat[(feat.league == STD_A) & (feat.away_team == "Rovers")]
    assert a["away_ht_scored_last5"].dropna().eq(0.0).all()


# ── 7 + 8 (bug 8) weights stay with their fixture after sort ────────────────────────────────

def test_07_weights_stay_with_fixture_after_sort(feat):
    rows = feat.dropna(subset=["over25"]).sample(frac=1.0, random_state=3)   # NOT date-sorted
    w = pd.Series(np.arange(len(rows), dtype=float), index=rows.index)
    out = M._rows_with_weights(rows.assign(_id=w), "over25", w)
    assert (out["_w"] == out["_id"]).all()
    with pytest.raises(ValueError):
        M._rows_with_weights(rows, "over25", np.ones(len(rows) - 1))   # no silent fallback


def test_07b_documents_legacy_weight_misalignment(feat, monkeypatch):
    rows = feat.dropna(subset=["over25", "home_scored_last5"]).sample(frac=1.0, random_state=4)
    rows = rows.assign(rid=np.arange(len(rows), dtype=float))
    seen = []
    from sklearn.pipeline import Pipeline
    orig = Pipeline.fit

    def spy(self, X, y=None, **kw):
        if "clf__sample_weight" in kw and not seen:
            seen.append((X["rid"].to_numpy().copy(), np.asarray(kw["clf__sample_weight"]).copy()))
        return orig(self, X, y, **kw)
    monkeypatch.setattr(Pipeline, "fit", spy)
    M.train(rows, "over25", sample_weight=rows["rid"].to_numpy() + 1.0, feature_cols=["rid"] + FEATS[:4])
    rid, w = seen[0]
    assert not np.allclose(w, rid + 1.0), "legacy reorders rows but not their weights"
    seen.clear()
    M.train_repaired(rows, "over25", sample_weight=rows["rid"] + 1.0, feature_cols=["rid"] + FEATS[:4])
    rid, w = seen[0]
    assert np.allclose(w, rid + 1.0)


# ── 8 meta-model evaluated only on untouched rows ───────────────────────────────────────────

def test_08_meta_evaluated_on_untouched_rows(repaired):
    b = repaired.meta["blocks"]
    order = ["base", "calibration", "meta", "holdout"]
    for x, y in zip(order, order[1:]):
        assert b[x]["dates"][1] < b[y]["dates"][0], (x, y)       # strictly earlier, no shared day
    assert repaired["__meta__"]["metrics"]["evaluated_on"] == "holdout"
    assert repaired["__meta__"]["metrics"]["n_test"] == b["holdout"]["rows"]


def test_08b_documents_legacy_meta_fit_on_its_own_test_block(feat, legacy):
    from sklearn.linear_model import LogisticRegression
    rows = feat.dropna(subset=["over25", "home_scored_last5"])
    _, test = M.chronological_split(rows, "over25")
    X, _ = M._prep(test, FEATS)
    mx = np.column_stack([legacy[k]["model"].predict_proba(X)[:, 1]
                          for k in legacy if k != "__meta__"])
    refit = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs").fit(mx, test["over25"])
    assert np.allclose(refit.coef_, legacy["__meta__"]["model"].coef_, atol=1e-6)


# ── 9 promotion comparison scores the SERVED probability ────────────────────────────────────

def test_09_served_score_is_what_predict_serves(feat, legacy, repaired):
    rows = feat.dropna(subset=["over25", "home_scored_last5"]).tail(120)
    for res in (legacy, repaired):
        pay = _payload(res)
        s = M.score_served(pay, rows, "over25")
        p = np.clip(M.predict_proba(rows, payload=pay).to_numpy(), 1e-15, 1 - 1e-15)
        y = rows["over25"].to_numpy()
        assert s["log_loss"] == pytest.approx(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    pay = _payload(legacy)
    assert M.score_payload(pay, rows, "over25") != pytest.approx(
        M.score_served(pay, rows, "over25")["log_loss"], abs=1e-9), \
        "score_payload (sub-model mean) is not the served probability"
    cmp = M.compare_served(_payload(legacy), _payload(repaired), rows, "over25")
    assert cmp["n"] == len(rows)


# ── 10 + 11 closing odds never become an early feature; entry != closing ────────────────────

def test_10_closing_odds_cannot_enter_asof_feature(raw):
    f = build_features_repaired(raw, price_mode="asof")       # only odds_over25 (closing) present
    assert f["bookmaker_overround"].isna().all()
    assert "bookmaker_overround" not in feature_cols_for("none")
    from src.standard_repair import fd_entry_closing
    only_c = pd.DataFrame({"AvgC>2.5": [1.9, 2.1], "AvgC<2.5": [1.95, 1.75]})
    px = fd_entry_closing(only_c)
    assert px["odds_over25_entry"].isna().all() and px["odds_over25_close"].notna().all()


def test_11_entry_and_closing_are_distinct_fields():
    from src.standard_repair import fd_entry_closing, ENTRY_CLASS, CLOSING_CLASS
    raw = pd.DataFrame({"Avg>2.5": [2.00], "Avg<2.5": [1.80], "AvgC>2.5": [1.70], "AvgC<2.5": [2.15]})
    px = fd_entry_closing(raw)
    assert px.loc[0, "odds_over25_entry"] == 2.00 and px.loc[0, "odds_over25_close"] == 1.70
    assert px.loc[0, "entry_price_class"] == ENTRY_CLASS == "FD_PREMATCH_APPROX"
    assert px.loc[0, "close_price_class"] == CLOSING_CLASS
    assert px.loc[0, "entry_price_source"] == "Avg>2.5"


# ── 13 Standard and NF side rows cannot silently pool in a track-specific model ─────────────

def test_13_track_purity():
    from src.standard_repair import assert_track_pure, track_rows
    mixed = pd.DataFrame({"league": [STD_A, NF], "over25": [1.0, 0.0],
                          "home_scored_last5": [1.0, 1.0], "btts": [1.0, 0.0]})
    with pytest.raises(ValueError):
        assert_track_pure(mixed, "standard")
    with pytest.raises(ValueError):
        assert_track_pure(mixed, "newformat")
    assert set(track_rows(mixed, "standard", "btts")["league"]) == {STD_A}
    assert set(track_rows(mixed, "newformat", "btts")["league"]) == {NF}
    assert set(track_rows(mixed, "pooled", "btts")["league"]) == {STD_A, NF}


# ── 14 aliases cannot split rolling history ─────────────────────────────────────────────────

def test_14_aliases_cannot_split_rolling_history():
    from src.identity_repair import canonicalise_and_dedupe
    df = _season(STD_B, ["Sheffield Weds", "Hull", "Stoke", "Wigan"], "2024-08-01", 7)
    # the other source stores every Sheffield Weds match AGAIN under its long name
    dup = df[(df.home_team == "Sheffield Weds") | (df.away_team == "Sheffield Weds")].copy()
    dup = dup.replace({"Sheffield Weds": "Sheffield Wednesday", "Hull": "Hull City"})
    both = pd.concat([df, dup], ignore_index=True)
    fixed, rep = canonicalise_and_dedupe(
        both, {STD_B: {"Sheffield Wednesday": "Sheffield Weds", "Hull City": "Hull"}}, derive=False)
    assert rep["duplicate_rows_removed"] == len(dup)
    clubs = set(fixed.home_team) | set(fixed.away_team)
    assert "Sheffield Wednesday" not in clubs and "Hull City" not in clubs
    f = build_features_repaired(fixed)
    sw = f[(f.home_team == "Sheffield Weds")].iloc[-1]
    prior = fixed[((fixed.home_team == "Sheffield Weds") | (fixed.away_team == "Sheffield Weds"))
                  & (fixed.date < sw.date)].tail(5)
    goals = np.where(prior.home_team == "Sheffield Weds", prior.home_goals, prior.away_goals)
    assert sw["home_scored_last5"] == pytest.approx(goals.mean())
    # serve side: the OddsAPI spelling resolves onto the one history club
    up = pd.DataFrame({"date": [fixed.date.max() + pd.Timedelta(days=7)], "league": [STD_B],
                       "home_team": ["Sheffield Wednesday"], "away_team": ["Hull City"],
                       "odds_over25": [2.0], "odds_under25": [1.8]})
    u = build_upcoming_features_repaired(up, fixed)
    assert u["history_team_home"].iloc[0] == "Sheffield Weds"
    assert u["home_scored_last5"].notna().iloc[0]


def test_14b_derived_aliases_reject_two_clubs_that_meet():
    from src.identity_repair import derive_same_day_aliases
    rows = []
    for k in range(4):                                    # recurring same-day same-score pairs
        d = pd.Timestamp("2024-01-06") + pd.Timedelta(days=7 * k)
        rows += [{"date": d, "league": NF, "home_team": "Yokohama FC", "away_team": f"X{k}",
                  "home_goals": 1, "away_goals": 0},
                 {"date": d, "league": NF, "home_team": "Yokohama F. Marinos", "away_team": f"Y{k}",
                  "home_goals": 1, "away_goals": 0}]
    rows.append({"date": pd.Timestamp("2024-03-01"), "league": NF, "home_team": "Yokohama FC",
                 "away_team": "Yokohama F. Marinos", "home_goals": 2, "away_goals": 2})
    m = derive_same_day_aliases(pd.DataFrame(rows))
    assert "Yokohama FC" not in m.get(NF, {}) and "Yokohama F. Marinos" not in m.get(NF, {})


# ── R2 additions: rolling window never contains a later match; weights from the date ────────

def test_rolling_window_never_contains_a_later_match():
    # a club that moves League One -> Championship: legacy groups by team in [team, LEAGUE, date]
    # order, so its League One rows (alphabetically after "Championship") follow later rows.
    s1 = _season(STD_B, ["Mover", "B1", "B2", "B3"], "2023-08-01", 1)
    s2 = _season(STD_A, ["Mover", "A1", "A2", "A3"], "2024-08-01", 2)
    df = pd.concat([s1, s2], ignore_index=True)
    for scope in ("league", "club"):
        f = build_features_repaired(df, team_scope=scope)
        mv = f[(f.home_team == "Mover")].sort_values("date")
        for _, r in mv.iterrows():
            prior = df[((df.home_team == "Mover") | (df.away_team == "Mover")) & (df.date < r.date)]
            if scope == "league":
                prior = prior[prior.league == r.league]
            prior = prior.tail(5)
            if prior.empty:
                assert np.isnan(r.home_scored_last5)
                continue
            g = np.where(prior.home_team == "Mover", prior.home_goals, prior.away_goals)
            assert r.home_scored_last5 == pytest.approx(g.mean())
    from src.feature_engineering import _build_team_centric
    tc = _build_team_centric(df.sort_values("date").reset_index(drop=True))
    m = tc[tc.team == "Mover"]
    assert (m["date"].diff().dt.days < 0).any(), "legacy order: a later match precedes an earlier one"


def test_weights_are_derived_from_match_date():
    from src.standard_repair import derived_weights, legacy_weights
    df = pd.DataFrame({"date": pd.to_datetime(["2026-10-01", "2025-11-01", "2024-11-01",
                                               "2023-11-01", "2020-11-01"]),
                       "league": STD_A,
                       "season": ["2026/27", "2025", "2024/25", "2023/2024", "2020/21"]})
    w = derived_weights(df)
    assert list(w) == [4.0, 3.0, 2.0, 1.5, 0.0]
    lw = legacy_weights(df)
    assert lw.iloc[0] == 1.0 and lw.iloc[2] == 4.0       # legacy: newest season under-weighted


# ── safety: the legacy defaults are untouched ───────────────────────────────────────────────

def test_default_recipe_is_legacy_and_payload_shape_unchanged(feat, legacy):
    assert config.MODEL_RECIPE in config.MODEL_RECIPES
    pay = M.build_payload(legacy, "over25")
    assert list(pay) == ["target", "models", "feature_cols", "metrics"]


def test_research_payload_refuses_serving_paths(repaired, tmp_path):
    from src.standard_repair import save_research_payload
    with pytest.raises(PermissionError):
        save_research_payload(_payload(repaired), config.MODEL_FILE_STANDARD)
    with pytest.raises(PermissionError):
        save_research_payload(_payload(repaired), config.MODELS_DIR / "anything.pkl")
    assert save_research_payload(_payload(repaired), tmp_path / "r.pkl").exists()
