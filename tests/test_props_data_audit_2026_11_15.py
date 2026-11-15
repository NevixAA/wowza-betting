"""fix/props-data-audit (owner 2026-11-15): the props data problems of the 128-feature audit.

1. `started` from the collected lineups (provider flag drifted from 2026-08); no lineup -> missing.
2. dribbles_past / penalty_won (and penalty_saved): a provider null is missing in an answer that sends
   explicit zeros, the provider's "none" in one that never does; stored null-read-as-0 blanked.
3. Injury features from the fixture-level injury lists of EARLIER fixtures, injuries only, missing
   where the league-season is not covered or was fetched before the match.
4. Cards serves hgb_nan_native at the next retrain (config only).
5. age_peak_delta / height_advantage_score / height_aerial_interaction withheld from training.
6. shots_total-based features: league-relative counterparts (*_lgrel) on which a definition change
   cancels out; the raw ones withheld.
+ AMBER naming documented in the contract; the new Data truth 7d-7g checks catch each recurrence.
SYNTHETIC DATA ONLY.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from player_model import config as C  # noqa: E402
from player_model import data_fetcher as DF  # noqa: E402
from player_model import feature_engineering as FE  # noqa: E402
from player_model import history_refetch as HR  # noqa: E402
from player_model import injury_features as IF  # noqa: E402
from player_model import lineup_started as LS  # noqa: E402
from player_model import player_stat_nulls as PSN  # noqa: E402
from player_model import props_contract as PC  # noqa: E402
from src.integrity import props_feature_truth as PFT  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import promote_props_clean as PROMO  # noqa: E402


# ── 1. started from the lineups ──────────────────────────────────────────────────────────────────

def _roles():
    return pd.DataFrame({"fixture_id": [1.0, 1.0, 2.0], "player_id": [10.0, 11.0, 10.0],
                         "started": [1.0, 0.0, 0.0]})


def test_started_is_the_lineup_role_and_the_provider_flag_stays_beside():
    raw = pd.DataFrame({"fixture_id": [1, 1, 2, 3, 1], "player_id": [10, 11, 10, 10, 12],
                        "started": [1, 1, 1, 1, 1]})
    out = LS.with_lineup_started(raw, _roles())
    assert out["started"].tolist()[:3] == [1.0, 0.0, 0.0]          # the lineup decides, not the flag
    assert np.isnan(out.loc[3, "started"])                          # fixture without a lineup: missing
    assert np.isnan(out.loc[4, "started"])                          # player not listed: missing
    assert out[LS.ORIG_COL].tolist() == [1, 1, 1, 1, 1]             # provider value kept beside
    assert out[LS.SOURCE_COL].tolist() == [LS.FROM_LINEUP_START, LS.FROM_LINEUP_SUB, LS.FROM_LINEUP_SUB,
                                           LS.NO_PROOF, LS.NO_PROOF]


def test_a_player_listed_under_both_roles_has_no_proof(tmp_path):
    d = tmp_path / "lineups"
    d.mkdir()
    pd.DataFrame({"fixture_id": [5, 5, 5], "player_id": [1.0, 1.0, 2.0], "role": ["start", "sub", "sub"]}) \
        .to_parquet(d / "players_1_2026.parquet")
    r = LS.lineup_roles(d)
    assert r.set_index("player_id")["started"].to_dict() == {2.0: 0.0}


def test_starter_rate_reads_the_corrected_started():
    rows = []
    for k in range(6):
        rows.append({"player_id": 7, "date": f"2026-10-{k + 1:02d}", "started": float(k % 2), "minutes": 90})
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    grp = df.groupby("player_id", group_keys=False)
    sr = FE._roll(grp, "started", 5)
    assert sr.iloc[5] == pytest.approx(np.mean([0, 1, 0, 1, 0]))   # previous five, shift(1)


# ── 2. dribbles_past / penalty_won null ─────────────────────────────────────────────────────────

def _st(minutes=90, **groups):
    base = {"games": {"minutes": minutes}, "shots": {"total": None, "on": None},
            "dribbles": {"attempts": None, "success": None, "past": None},
            "penalty": {"won": None, "commited": None, "scored": None, "missed": None, "saved": None},
            "tackles": {"total": None}}
    for k, v in groups.items():
        base[k] = {**base.get(k, {}), **v}
    return base


def test_new_columns_are_null_is_missing():
    assert {"dribbles_past", "penalty_won"} <= PSN.NULL_IS_MISSING
    assert PSN.NULL_IS_NONE_WITHOUT_EXPLICIT_ZEROS <= PSN.NULL_IS_MISSING


def test_null_dribbles_past_is_missing_in_an_explicit_zero_answer_and_none_in_an_old_one():
    s = _st()
    new = PSN.player_counts(s, answered=True, group_unpublished=True)
    old = PSN.player_counts(s, answered=True, group_unpublished=False)
    unknown = PSN.player_counts(s, answered=True)
    for col in ("dribbles_past", "penalty_won", "penalty_saved"):
        assert new[col] is None and unknown[col] is None
        assert old[col] == 0
    assert new["shots_total"] == 0 and new["tackles_total"] == 0        # the other counts keep the rule
    assert new["goals_conceded"] is None and old["goals_conceded"] is None
    assert PSN.player_counts(_st(dribbles={"past": 2}), True, True)["dribbles_past"] == 2


def test_the_group_is_unpublished_only_when_no_player_carries_any_of_it():
    def ans(*stats):
        return [{"players": [{"statistics": [st]} for st in stats]}]
    assert PSN.fixture_group_unpublished(ans(_st(), _st(tackles={"total": 0})))
    assert not PSN.fixture_group_unpublished(ans(_st(dribbles={"past": 0}), _st(tackles={"total": 0})))
    assert not PSN.fixture_group_unpublished(ans(_st(), _st(shots={"total": 2})))   # no explicit zero


def test_fixture_sends_zeros_needs_a_sent_zero_of_a_played_player():
    def ans(*stats):
        return [{"players": [{"statistics": [st]} for st in stats]}]
    assert not PSN.fixture_sends_zeros(ans(_st(), _st(shots={"total": 2})))
    assert PSN.fixture_sends_zeros(ans(_st(), _st(tackles={"total": 0})))
    assert not PSN.fixture_sends_zeros(ans(_st(minutes=0, tackles={"total": 0})))   # did not play
    assert not PSN.fixture_sends_zeros(ans(_st(penalty={"won": 0})))               # not evidence


def test_parser_writes_the_null_provenance():
    meta = {"fixture_id": 9, "date": "2026-11-01", "home_team": "A", "away_team": "B", "team": "A",
            "is_home": 1, "season": "2026"}
    entry = {"player": {"id": 123, "name": "X"}, "statistics": [_st()]}
    row = DF._parse_fixture_player(entry, meta, "Premier League", True, True)
    assert row["dribbles_past"] is None and row["penalty_won"] is None
    assert set(row[PSN.NULL_MISSING_COL].split(",")) >= {"dribbles_past", "penalty_won", "penalty_saved"}
    row_old = DF._parse_fixture_player(entry, meta, "Premier League", True, False)
    assert row_old["dribbles_past"] == 0 and "dribbles_past" not in row_old[PSN.NULL_MISSING_COL]


def test_reader_blanks_null_read_as_zero_only_in_explicit_zero_fixtures():
    hist = pd.DataFrame({"fixture_id": [1, 1, 2, 2], "player_id": [10, 11, 20, 21],
                         "dribbles_past": [0, 1, 0, 2], "penalty_won": [0, 0, 0, 0],
                         "tackles_total": [3, 1, 2, 0], "is_home": [1, 0, 1, 0]})
    # fixture 1 sends explicit zeros (player 11 SENT tackles 0); fixture 2 never does
    rf = pd.DataFrame({"fixture_id": [1, 1, 2, 2], "player_id": [10, 11, 20, 21],
                       "minutes_refetch": [90, 90, 90, 90],
                       "dribbles_past": [0, 1, 0, 2], "penalty_won": [0, 0, 0, 0], "tackles_total": [3, 0, 2, 0],
                       "filled_cols": ["dribbles_past,penalty_won", "penalty_won", "dribbles_past,penalty_won",
                                       "penalty_won,tackles_total"]})
    out = HR.with_refetched(hist, rf).set_index("player_id")
    # fixture 1 PUBLISHED the group (player 11 sent dribbles_past 1): its nulls are the provider's none
    assert out.loc[10, "dribbles_past"] == 0 and out.loc[10, "penalty_won"] == 0
    assert out.loc[11, "penalty_won"] == 0 and out.loc[11, "dribbles_past"] == 1
    assert out.loc[20, "dribbles_past"] == 0 and out.loc[20, "penalty_won"] == 0   # old answer: none = 0
    assert out.loc[21, "tackles_total"] == 0                                          # not a missing count
    assert out.loc[20, HR.NULL_BLANKED_COL] == ""
    # an explicit-zero fixture that published none of the group: every null of it is missing
    g = "dribbles_past,penalty_won,penalty_saved"
    rf3 = rf.assign(dribbles_past=[0, 0, 0, 2], filled_cols=[g, g, g, "penalty_won,tackles_total"])
    out3 = HR.with_refetched(hist, rf3).set_index("player_id")
    assert np.isnan(out3.loc[10, "dribbles_past"]) and np.isnan(out3.loc[11, "penalty_won"])
    assert "dribbles_past" in out3.loc[10, HR.NULL_BLANKED_COL]


def test_rows_parsed_before_the_fix_keep_a_zero_only_where_the_fixture_published_the_field():
    hist = pd.DataFrame({"fixture_id": [1, 1, 2, 2], "player_id": [10, 11, 20, 21],
                         "dribbles_past": [0, 2, 0, 0], "penalty_won": [0, 0, 0, 0],
                         "count_source": ["provider_rule"] * 4})
    out = HR.with_refetched(hist, pd.DataFrame({"fixture_id": [99], "player_id": [99]})).set_index("player_id")
    assert out.loc[10, "dribbles_past"] == 0                 # fixture 1 published dribbles_past
    assert np.isnan(out.loc[20, "dribbles_past"])            # fixture 2: every 0 unproven -> missing
    assert out.loc[20, "dribbles_past_stored"] == 0          # the stored value kept beside
    assert out.loc[10, "penalty_won"] == 0                   # the group was published in fixture 1
    # a row parsed AFTER the fix carries its provenance and is kept as parsed
    hist2 = hist.assign(**{PSN.NULL_MISSING_COL: ["", "", "", ""]})
    out2 = HR.with_refetched(hist2, pd.DataFrame({"fixture_id": [99], "player_id": [99]})).set_index("player_id")
    assert out2.loc[20, "dribbles_past"] == 0


def test_penalties_won_pg_is_no_longer_semantic_zero():
    c = PC.load_contract()
    assert c["features"]["penalties_won_pg"]["class"] != "SEMANTIC_ZERO"


# ── 3. injury features ──────────────────────────────────────────────────────────────────────────

def test_reason_classes():
    assert IF.reason_class("Knee Injury") == "injury"
    assert IF.reason_class("Illness") == "injury"
    assert IF.reason_class("Red Card") == "suspension"
    assert IF.reason_class("Yellow Cards") == "suspension"
    assert IF.reason_class("International duty") == "other"
    assert IF.reason_class("Coach's decision") == "other"
    assert IF.reason_class("Something new") == "unclassified"


def test_every_collected_reason_is_classified():
    inj = FE._collected("injuries")
    if inj.empty:
        pytest.skip("injuries not collected in this tree")
    cls = inj["reason"].map(IF.reason_class)
    named = inj["reason"].notna()                      # a missing reason is unknown: never an injury
    left = inj.loc[named & (cls == "unclassified"), "reason"].astype(str)
    assert left.empty, sorted(left.unique())[:20]


def _inj_world():
    inj = pd.DataFrame({
        "player_id": [1, 1, 1, 1, 2, 3, 1],
        "date": ["2025-09-01", "2025-09-08", "2025-10-20", "2026-02-01", "2025-09-08", "2025-09-08",
                 "2026-03-01"],
        "type": ["Missing Fixture", "Missing Fixture", "Missing Fixture", "Questionable", "Missing Fixture",
                 "Missing Fixture", "Missing Fixture"],
        "reason": ["Knee Injury", "Knee Injury", "Hamstring Injury", "Knock", "Red Card", "Illness",
                   "Knee Injury"],
        "league_id": [39] * 7})
    seasons = pd.DataFrame({"league_id": [39, 39, 40], "season": ["2024", "2025", "2025"],
                            "status": ["ok", "ok", "empty"], "coverage_injuries": [True, True, False],
                            "fetched_at": ["2026-11-05T07:00:00+00:00"] * 2 + ["2026-11-05T07:00:00+00:00"]})
    fx = pd.DataFrame({"fixture_id": [100.0, 200.0], "league_id": [39.0, 40.0],
                       "competition": ["Premier League", "Championship"]})
    played = pd.DataFrame({"player_id": [1.0, 1.0], "_day": pd.to_datetime(["2025-09-15", "2026-03-01"])})
    return inj, seasons, fx, played


def _q(rows):
    q = pd.DataFrame(rows, columns=["player_id", "date", "fixture_id", "league", "season"])
    q["date"] = pd.to_datetime(q["date"])
    return q


def test_injury_features_read_earlier_fixtures_only_and_count_episodes():
    inj, seasons, fx, played = _inj_world()
    q = _q([(1, "2025-09-08", 100, "Premier League", "2025"),     # own fixture's list is not read
            (1, "2025-09-20", 100, "Premier League", "2025"),
            (1, "2025-11-30", 100, "Premier League", "2025"),
            (2, "2025-09-20", 100, "Premier League", "2025"),     # suspension: not an injury
            (3, "2025-09-20", 100, "Premier League", "2025")])
    out = IF.injury_features(q, played, inj, seasons, fx)
    assert out.loc[0, "days_since_last_injury"] == 7          # 09-01 listing, not the 09-08 one
    assert out.loc[0, "return_from_injury_flag"] == 1
    assert out.loc[1, "days_since_last_injury"] == 12 and out.loc[1, "return_from_injury_flag"] == 1
    assert out.loc[1, "chronic_injury_risk"] == 1             # 09-01 and 09-08 are one episode
    assert out.loc[2, "chronic_injury_risk"] == 2             # played 09-15, listed again 10-20: new episode
    assert out.loc[2, "days_since_last_injury"] == 41 and out.loc[2, "return_from_injury_flag"] == 0
    assert np.isnan(out.loc[3, "days_since_last_injury"]) and out.loc[3, "chronic_injury_risk"] == 0
    assert out.loc[4, "return_from_injury_flag"] == 1         # illness is an injury
    assert (out[IF.SOURCE_COL] == IF.FROM_LISTS).all()


def test_questionable_entries_and_uncovered_or_stale_rows_stay_missing():
    inj, seasons, fx, played = _inj_world()
    q = _q([(1, "2026-02-10", 100, "Premier League", "2025"),     # 02-01 'Questionable' is not read
            (1, "2025-09-20", 200, "Championship", "2025"),       # league-season without coverage
            (1, "2026-11-20", 100, "Premier League", "2025"),     # after the fetch: unknown
            (1, "2025-09-20", 100, "Premier League", "2024/25")])  # previous season (2023) not answered
    out = IF.injury_features(q, played, inj, seasons, fx)
    assert out.loc[0, "days_since_last_injury"] == (pd.Timestamp("2026-02-10") - pd.Timestamp("2025-10-20")).days
    for i in (1, 2):
        assert out.loc[i, ["chronic_injury_risk", "days_since_last_injury", "return_from_injury_flag"]].isna().all()
    assert out.loc[1, IF.SOURCE_COL] == IF.NOT_COVERED and out.loc[2, IF.SOURCE_COL] == IF.STALE
    assert np.isnan(out.loc[3, "chronic_injury_risk"]) and out.loc[3, "return_from_injury_flag"] == 1
    assert out.loc[3, IF.SOURCE_COL] == IF.FROM_LISTS_NO_CHRONIC


def test_build_features_never_reads_the_sidelined_source():
    src = (ROOT / "player_model" / "feature_engineering.py").read_text(encoding="utf-8")
    body = src.split("def build_features")[1].split("def upcoming_rows")[0]
    assert "collected_sidelined()" not in body and "apply_injury_pit" not in body
    assert "_injury_features(" in body


# ── 4. cards candidate ──────────────────────────────────────────────────────────────────────────

def test_cards_is_fixed_to_hgb_nan_native_for_the_next_retrain():
    plan = json.loads((ROOT / "config" / "experiments" / "props_clean_retrain" / "plan.json").read_text(encoding="utf-8"))
    assert plan["candidate_by_market"] == {"cards": "hgb_nan_native"}
    assert PROMO.candidate_by_market(plan)["cards"] == "hgb_nan_native"


def test_a_fixed_candidate_never_falls_back_to_the_recipe():
    rep = {"holdout": {"log_loss_minus_base_rate": {"hgb_nan_native": {"ci95": [-0.02, -0.01]},
                                                    "recipe_retrained": {"ci95": [-0.05, -0.03]}}}}
    h2h = {"ci95_hi": 0.01}                             # hgb not better than the recipe
    assert PROMO.choose(rep, h2h)[0] == "recipe_retrained"
    assert PROMO.choose(rep, h2h, "hgb_nan_native")[0] == "hgb_nan_native"
    rep_bad = {"holdout": {"log_loss_minus_base_rate": {"hgb_nan_native": {"ci95": [-0.01, 0.01]},
                                                        "recipe_retrained": {"ci95": [-0.05, -0.03]}}}}
    assert PROMO.choose(rep_bad, h2h, "hgb_nan_native")[0] is None


# ── 5. typed-constant profile composites withheld ───────────────────────────────────────────────

def test_profile_constant_composites_are_withheld_from_training():
    w = {"age_peak_delta", "height_advantage_score", "height_aerial_interaction"}
    assert w <= C.PROFILE_CONSTANT_FEATURES_WITHHELD
    assert not (w & set(C.PLAYER_FEATURE_COLS))
    assert {"age", "height_cm", "height_diff_vs_opp_def", "opp_def_mean_height"} <= set(C.PLAYER_FEATURE_COLS)
    f = pd.DataFrame({"height_cm": [190.0], "opp_def_mean_height": [180.0], "aerial_won_rate": [0.5], "age": [30.0]})
    FE.profile_composites(f)                          # serving keeps computing them for old payloads
    assert f.loc[0, "age_peak_delta"] == 3.0


# ── 6. league-relative shots ────────────────────────────────────────────────────────────────────

def _league(n_fix, start, shots_per_player, league="L", fid0=0, players=11):
    rows = []
    for k in range(n_fix):
        d = pd.Timestamp(start) + pd.Timedelta(days=k)
        for side in (1, 0):
            for p in range(players):
                rows.append({"fixture_id": fid0 + k, "league": league, "date": d, "is_home": side,
                             "player_id": 1000 * side + p + 1, "shots_total": shots_per_player(k),
                             "shots_on_target": 0.4})
    return pd.DataFrame(rows)


def test_league_shots_level_is_strictly_before_the_day_and_nan_below_the_window():
    df = _league(30, "2026-01-01", lambda k: float(k % 3))
    lvl = FE.league_shots_level(df)
    n = C.SHOTS_LEVEL_TEAM_MATCHES
    days = df["date"].dt.normalize()
    # team-matches before day k: 2k; NaN while fewer than n
    first_ok = days.min() + pd.Timedelta(days=n // 2)
    assert lvl[days < first_ok].isna().all()
    k = n // 2 + 3
    exp_team = [11 * float(j % 3) for j in range(k) for _ in (0, 1)][-n:]
    got = lvl[days == days.min() + pd.Timedelta(days=k)].unique()
    assert len(got) == 1 and got[0] == pytest.approx(np.mean(exp_team))


def test_league_shots_level_reads_no_other_league_and_no_upcoming_row():
    a = _league(25, "2026-01-01", lambda k: 1.0, "A")
    b = _league(5, "2026-01-01", lambda k: 5.0, "B", fid0=500)
    up = pd.DataFrame({"fixture_id": [999], "league": ["A"], "date": [pd.Timestamp("2026-01-26")], "is_home": [1],
                       "player_id": [1], "shots_total": [np.nan], FE.UPCOMING_COL: [True]})
    df = pd.concat([a, b, up], ignore_index=True)
    df[FE.UPCOMING_COL] = df[FE.UPCOMING_COL].fillna(False).astype(bool)
    lvl = FE.league_shots_level(df)
    assert lvl[df["league"] == "B"].isna().all()                   # B: 10 team-matches < window, no pooling
    assert lvl.iloc[-1] == pytest.approx(11.0)                      # A only, upcoming row adds nothing
    hist = df[df["date"] < pd.Timestamp("2026-01-26")]
    serve = FE.league_shots_level(pd.concat([hist, up], ignore_index=True))
    assert serve.iloc[-1] == lvl.iloc[-1]                           # bit-identical on the prefix


def test_a_definition_change_cancels_out_of_the_relative_shots():
    # shots double from day 40 (a definition change); SOT unchanged
    df = _league(80, "2026-01-01", lambda k: 1.0 if k < 40 else 2.0)
    df["_lvl"] = FE.league_shots_level(df)
    df["rel"] = df["shots_total"] / df["_lvl"]
    k = (df["date"] - df["date"].min()).dt.days
    before = df.loc[(k >= 30) & (k < 40), "rel"].mean()
    after = df.loc[k >= 40 + C.SHOTS_LEVEL_TEAM_MATCHES // 2, "rel"].mean()
    assert before == pytest.approx(1 / 11) and after == pytest.approx(1 / 11)
    assert df.loc[k >= 40, "shots_total"].mean() == 2 * df.loc[(k >= 30) & (k < 40), "shots_total"].mean()


def test_every_shots_feature_has_its_league_relative_counterpart_in_training():
    assert len(C.SHOTS_DEFINITION_WITHHELD) == 17
    feats = set(C.PLAYER_FEATURE_COLS)
    assert not (set(C.SHOTS_DEFINITION_WITHHELD) & feats)
    assert set(C.SHOTS_LEAGUE_RELATIVE) <= feats
    c = PC.load_contract()
    assert set(c["features"]) == feats


# ── AMBER naming documented ─────────────────────────────────────────────────────────────────────

def test_all_duel_features_are_documented_in_the_contract():
    c = PC.load_contract()["features"]
    for f in ("aerial_won_rate", "opp_def_aerial_win_rate", "aerial_matchup_score", "box_actions_per90_lgrel"):
        assert "NAME NOTE" in c[f]["reason"], f


# ── the checks catch a recurrence ───────────────────────────────────────────────────────────────

def _started_frame(n_months=14, drift_from=None, provider=None):
    rows = []
    for mi in range(n_months):
        m = pd.Timestamp("2025-01-01") + pd.DateOffset(months=mi)
        for i in range(300):
            s = 1.0 if i < 225 else 0.0
            if drift_from is not None and mi >= drift_from:
                s = 1.0 if i < 290 else 0.0
            rows.append({"fixture_id": mi * 1000 + i // 22, "player_id": i, "date": m, "league": "L",
                         "minutes": 90 if s else 30, "started": s,
                         LS.ORIG_COL: (provider if provider is not None else s)})
    raw = pd.DataFrame(rows)
    roles = raw[["fixture_id", "player_id", "started"]].astype(float)
    return raw, roles


def test_started_drift_green_on_stable_lineups_and_red_on_a_drift():
    raw, roles = _started_frame()
    assert PFT.started_drift(raw, roles)["status"] == PFT.GREEN
    bad, broles = _started_frame(drift_from=13)
    assert PFT.started_drift(bad, broles)["status"] == PFT.RED          # league-month share jumps
    raw2, roles2 = _started_frame()
    roles2.loc[0, "started"] = 0.0                                      # features no longer the lineup
    assert PFT.started_drift(raw2, roles2)["status"] == PFT.RED


def test_started_drift_shows_the_provider_flag_drift_as_amber():
    raw, roles = _started_frame(provider=1.0)
    r = PFT.started_drift(raw, roles)
    assert r["status"] == PFT.AMBER and min(r["provider_agreement_by_month"].values()) < 0.9


def _null_world(frame_value):
    rf = pd.DataFrame({"fixture_id": [1] * 12, "player_id": list(range(12)), "minutes_refetch": [90] * 12,
                       "tackles_total": [0] + [1] * 11, "interceptions": [1] * 12,
                       "filled_cols": ["interceptions"] * 12})
    raw = pd.DataFrame({"fixture_id": [1] * 12, "player_id": list(range(12)),
                        "date": [pd.Timestamp("2026-11-01")] * 12, "interceptions": [frame_value] * 12,
                        "tackles_total": [0] + [1] * 11})
    return raw, rf


def test_null_as_zero_catches_an_unpublished_count_read_as_zero():
    raw, rf = _null_world(0.0)
    r = PFT.null_as_zero(raw, rf)
    assert r["status"] == PFT.RED and r["read_as_zero_fixtures"] == {"interceptions": 1}
    raw, rf = _null_world(np.nan)
    assert PFT.null_as_zero(raw, rf)["status"] == PFT.GREEN


def test_stale_sources_catches_a_source_that_stopped():
    raw = pd.DataFrame({"fixture_id": [1, 2], "player_id": [1, 1], "league": ["Premier League"] * 2,
                        "date": pd.to_datetime(["2026-10-01", "2026-11-14"])})
    seasons = pd.DataFrame({"league_id": [39], "season": ["2026"], "status": ["ok"], "coverage_injuries": [True],
                            "fetched_at": ["2026-11-14T00:00:00+00:00"]})
    fx = pd.DataFrame({"fixture_id": [1.0, 2.0], "league_id": [39.0, 39.0], "competition": ["Premier League"] * 2})
    fresh_inj = pd.DataFrame({"league_id": [39], "date": ["2026-11-10"]})
    old_inj = pd.DataFrame({"league_id": [39], "date": ["2026-10-01"]})
    roles = pd.DataFrame({"fixture_id": [1.0, 2.0], "player_id": [1.0, 1.0], "started": [1.0, 1.0]})
    spells = pd.DataFrame({"start": []})
    ok = PFT.stale_sources(raw, fresh_inj, seasons, roles, spells, fx)
    assert ok["status"] == PFT.GREEN, ok
    assert PFT.stale_sources(raw, old_inj, seasons, roles, spells, fx)["status"] == PFT.RED
    assert PFT.stale_sources(raw, fresh_inj, seasons, roles.iloc[:1], spells, fx)["status"] == PFT.RED


def _shots_raw(change_day, factor):
    df = _league(240, "2026-01-01", lambda k: 1.0 if k < change_day else factor)
    df["shots_on_target"] = 0.4
    return df


def test_shots_level_sees_a_definition_jump_and_stays_in_band():
    r = PFT.shots_level(_shots_raw(10_000, 1.0))
    assert r["status"] == PFT.GREEN
    r = PFT.shots_level(_shots_raw(91, 1.5))                         # +50% shots, SOT flat, from April
    assert r["status"] == PFT.AMBER and r["raw_jumps"]               # seen, and cancelled in the features


def test_shots_level_red_when_the_relative_level_does_not_cancel(monkeypatch):
    def broken(df):                                                   # e.g. a pooled / frozen level
        return pd.Series(11.0, index=df.index)
    monkeypatch.setattr(FE, "league_shots_level", broken)
    assert PFT.shots_level(_shots_raw(91, 1.5))["status"] == PFT.RED


# ── found by the new checks: placeholder answers and fixture-wide unpublished counts ─────────────

def test_a_placeholder_answer_holds_no_detailed_count():
    n = 12
    raw = pd.DataFrame({"fixture_id": [1] * n + [2] * n, "player_id": list(range(2 * n)),
                        "minutes": [90] * (2 * n), "passes_total": [0] * n + [30] * n,
                        "shots_total": [0] * n + [1] * n, "goals": [1] + [0] * (n - 1) + [0] * n,
                        "rating": [0.0] * n + [7.0] * n})
    out = HR.with_placeholder_answers_blanked(raw)
    ph = out[out["fixture_id"] == 1]
    assert ph[HR.PLACEHOLDER_COL].all() and ph["shots_total"].isna().all() and ph["passes_total"].isna().all()
    assert ph["goals"].tolist()[0] == 1                       # event counts kept
    assert not out.loc[out["fixture_id"] == 2, HR.PLACEHOLDER_COL].any()
    assert (out.loc[out["fixture_id"] == 2, "shots_total"] == 1).all()


def test_a_count_unpublished_for_a_whole_explicit_zero_fixture_is_missing():
    hist = pd.DataFrame({"fixture_id": [1, 1, 1], "player_id": [10, 11, 12], "saves": [0, 0, 0],
                         "tackles_total": [0, 2, 1]})
    rf = pd.DataFrame({"fixture_id": [1, 1, 1], "player_id": [10, 11, 12], "minutes_refetch": [90, 90, 90],
                       "saves": [0, 0, 0], "tackles_total": [0, 2, 1],
                       "filled_cols": ["saves", "saves", "saves"]})          # tackles 0 SENT: explicit zeros
    out = HR.with_refetched(hist, rf)
    assert out["saves"].isna().all() and out["tackles_total"].tolist() == [0, 2, 1]
    rf2 = rf.assign(filled_cols=["saves,tackles_total", "saves", "saves"])   # no explicit zero: old answer
    out2 = HR.with_refetched(hist, rf2)
    assert out2["saves"].tolist() == [0, 0, 0]                # the provider's none


def test_stale_sources_ignores_listings_of_fixtures_still_to_play():
    raw = pd.DataFrame({"fixture_id": [1, 2], "player_id": [1, 1], "league": ["Premier League"] * 2,
                        "date": pd.to_datetime(["2026-10-01", "2026-11-14"])})
    seasons = pd.DataFrame({"league_id": [39], "season": ["2026"], "status": ["ok"], "coverage_injuries": [True],
                            "fetched_at": ["2026-10-02T00:00:00+00:00"]})
    fx = pd.DataFrame({"fixture_id": [1.0, 2.0], "league_id": [39.0, 39.0], "competition": ["Premier League"] * 2})
    inj = pd.DataFrame({"league_id": [39, 39], "date": ["2026-10-01", "2027-05-01"],
                        "fetched_at": ["2026-10-02T00:00:00+00:00"] * 2})
    roles = pd.DataFrame({"fixture_id": [1.0, 2.0], "player_id": [1.0, 1.0], "started": [1.0, 1.0]})
    assert PFT.stale_sources(raw, inj, seasons, roles, pd.DataFrame({"start": []}), fx)["status"] == PFT.RED


def test_null_as_zero_catches_the_dribbles_past_pattern_again():
    g = "dribbles_past,penalty_won,penalty_saved"
    rf = pd.DataFrame({"fixture_id": [1] * 12, "player_id": list(range(12)), "minutes_refetch": [90] * 12,
                       "tackles_total": [0] + [1] * 11, "dribbles_past": [0] * 12, "filled_cols": [g] * 12})
    raw = pd.DataFrame({"fixture_id": [1] * 12, "player_id": list(range(12)),
                        "date": [pd.Timestamp("2026-11-01")] * 12, "dribbles_past": [0.0] * 12})
    assert PFT.null_as_zero(raw, rf)["status"] == PFT.RED              # an unpublished group read as 0
    assert PFT.null_as_zero(raw.assign(dribbles_past=np.nan), rf)["status"] == PFT.GREEN
