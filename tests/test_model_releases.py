"""Readable model-version names (owner 2026-11-23) and the one-row-per-market default view of the Predictions page.

The registry config/model_release_registry.json is generated from the install history
(scripts/build_model_release_registry.py); these tests read the committed file.
"""
import json

import pandas as pd

from src.model_releases import CANDIDATE, RETIRED, SERVING, UNKNOWN, Releases

KO = "2026-11-21T12:30:00Z"                          # Blackpool v Wigan (League One)


def test_registry_is_generated_and_every_named_key_exists():
    doc = json.loads(open("config/model_release_registry.json", encoding="utf-8").read())
    assert doc["schema_version"] == "mrr/1.0" and "GENERATED" in doc["about"]
    names = json.loads(open("config/model_release_names.json", encoding="utf-8").read())["names"]
    for r in doc["versions"]:
        assert r["release_name"] == names.get(r["release_key"], "")
        assert r["served_from"] and (r["served_to"] is None or r["served_to"] > r["served_from"])


def test_owner_examples():
    R = Releases()
    assert R.display("v9_standard", "btts", "btts_20261119T134019Z_44c67049", KO) == \
        "Go-live refit · 19 Nov · serving at kickoff"
    assert R.display("v9_standard", "btts", "btts_20261118T172022Z_a8ddb87e", KO) == "Pre-live refit · 18 Nov · retired"
    assert R.display("v9_standard", "btts", "btts_20261005T101905Z_238e3419", KO) == "October model · 5 Oct · retired"


def test_props_match_on_market_and_artifact_digest():
    R = Releases()
    # the ledger writes 'props_clean_harness'; the champion record says 'clean_hgb_nan_native' - same bytes (5de94714)
    assert R.display("v9_props", "goals", "props_goals_props_clean_harness_5de94714", KO).endswith(SERVING)
    assert R.status("v9_props", "goals", "props_goals_legacy_1592478c", KO) == RETIRED
    assert R.status("v9_props_challenger", "goals2", "props_goals2_clean_hgb_nan_native_e218cb6c", KO) == CANDIDATE
    # the digest alone never crosses markets
    assert R.records("v9_props", "sot", "props_goals_props_clean_harness_5de94714") == []


def test_unknown_version_is_never_guessed():
    R = Releases()
    assert R.display("v9_standard", "btts", "btts_20991231T000000Z_deadbeef", KO) == \
        f"{UNKNOWN} · btts_20991231T000000Z_deadbeef"


def test_pro_and_side_versions():
    R = Releases()
    assert R.display("pro_bb_dc", "1x2", "pro_bb_dc_1.0.0_66ed8add", KO).startswith("Pro score grid 1.0 · 19 Nov")
    assert R.display("v9_corners", "corners_total", "side_nb_v1_2026-11-09_c9e06a6a9c92", KO).startswith(
        "Side models v1")


def _q(model, version, sel, p, pick, y, at, market="btts", line="", q=2):
    return dict(model_id=model, model_version=version, market=market, line=line, selection=sel, probability=p,
                pick=pick, y=y, q_size=q, outcome_status="SETTLED", hit=(y if pick else None), player_id="",
                player_name="", team="", snapshot="first_sight", canonical_fixture_ref="af:1564416",
                kickoff_utc=pd.Timestamp(KO), match_kickoff_utc=pd.Timestamp(KO), predicted_at=pd.Timestamp(at))


def test_default_view_is_one_row_per_market():
    from dashboard_data import predictions_view as V
    rows = pd.DataFrame([
        _q("v9_standard", "btts_20261005T101905Z_238e3419", "YES", 0.52, True, 1, "2026-11-18T09:00:00Z"),
        _q("v9_standard", "btts_20261005T101905Z_238e3419", "NO", 0.48, False, 0, "2026-11-18T09:00:00Z"),
        _q("v9_standard", "btts_20261118T172022Z_a8ddb87e", "YES", 0.55, True, 1, "2026-11-19T08:00:00Z"),
        _q("v9_standard", "btts_20261118T172022Z_a8ddb87e", "NO", 0.45, False, 0, "2026-11-19T08:00:00Z"),
        _q("v9_standard", "btts_20261119T134019Z_44c67049", "YES", 0.4, False, 1, "2026-11-20T08:00:00Z"),
        _q("v9_standard", "btts_20261119T134019Z_44c67049", "NO", 0.6, True, 0, "2026-11-20T08:00:00Z"),
    ])
    t = V.pick_table(V.pick_rows(V.serving_rows(rows)))
    assert len(t) == 1                                                  # 3 versions x YES/NO -> 1 row
    assert (t["Pick"].iloc[0], t["Probability"].iloc[0], t["Result"].iloc[0]) == ("NO", 0.6, "miss")
    assert t["Model version"].iloc[0] == "Go-live refit · 19 Nov · serving at kickoff"
    full = V.prediction_table(rows)
    assert len(full) == 6 and "Version id" in full.columns             # the toggle view keeps everything


def test_single_selection_pick_below_half_shows_the_no_side():
    from dashboard_data import predictions_view as V
    r = pd.DataFrame([_q("v9_props", "props_goals_props_clean_harness_5de94714", "YES", 0.2, True, 0,
                         "2026-11-20T08:00:00Z", market="goals", line="0.5", q=1)])
    r["player_name"], r["player_id"] = "P", "7"
    t = V.pick_table(r)
    assert t["Pick"].iloc[0] == "not YES" and abs(t["Probability"].iloc[0] - 0.8) < 1e-9
    assert t["Outcome"].iloc[0] == "happened"                           # the NO side happened
