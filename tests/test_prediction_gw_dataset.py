"""Success by gameweek dataset (owner 2026-11-23) + one fixture, one identity + one serving version per question.

Synthetic ledgers (the Predictions page tests' writer): exact numbers, totals reconcile with the ledger questions,
closed gameweeks are never rewritten (a difference goes to the corrections file), idempotent reruns, Clean Record
rows never mixed, no default rate, v11 / prices kept out, and the identity resolver merges accent / name-variant /
kickoff-drift splits but refuses two API-Football fixtures.
"""
import json
import math

import pandas as pd
import pytest

from scripts import build_predictions_index as B
from src import fixture_identity as FI
from src import prediction_gw_dataset as G
from src.model_version_label import version_label
from tests.test_predictions_page import START, _row, _write

CLOSED_NOW = "2026-11-24T12:00:00Z"      # after 2026-W47 ended (Mon 2026-11-23 10:00 Israel = 08:00 UTC)
OPEN_NOW = "2026-11-21T20:00:00Z"        # inside 2026-W47


def _ledger(tmp_path, rows):
    _write(tmp_path, rows)
    cfg = tmp_path / "trust.json"
    cfg.write_text(json.dumps({"clean_record_start_utc": START}), encoding="utf-8")
    return cfg


def _base():
    return [
        _row("v9", "v9_standard", "ou25", "OVER", 0.7, line="2.5", y=1),
        _row("v9", "v9_standard", "ou25", "UNDER", 0.3, line="2.5", y=0),
        _row("pro", "pro_bb_dc", "1x2", "H", 0.5, fx="oa:e1", y=0),
        _row("pro", "pro_bb_dc", "1x2", "D", 0.3, fx="oa:e1", y=0),
        _row("pro", "pro_bb_dc", "1x2", "A", 0.2, fx="oa:e1", y=1),
        _row("v9", "v9_props", "sot", "YES", 0.1, line="0.5", player="7", y=0),
        _row("v9", "v9_props", "sot", "YES", 0.2, line="0.5", player="8", status="VOID_DID_NOT_PLAY"),
        _row("v9", "v9_props", "sot", "YES", 0.6, line="0.5", player="9", y=0),
        # before the Clean Record start: its own row
        _row("v9", "v9_standard", "btts", "YES", 0.6, at="2026-11-18T10:00:00Z", y=1),
        _row("v9", "v9_standard", "btts", "NO", 0.4, at="2026-11-18T10:00:00Z", y=0),
        _row("v9", "v9_standard", "btts", "YES", 0.55, fx="af:3", y=1),
        _row("v9", "v9_standard", "btts", "NO", 0.45, fx="af:3", y=0),
        # private: never read
        _row("v11", "v11_market", "ou25", "OVER", 0.5, line="2.5", y=1),
    ]


def _build(tmp_path, cfg, now, out=None):
    out = out or tmp_path / "out"
    return G.run(v9_root=tmp_path, pro_root=None, out=out, now=now, clean_cfg=cfg), out


def _ds(out):
    return G.read_dataset(out / G.DATASET.name)


def test_gw_bounds_are_the_tip_day_week():
    assert G.gw_bounds("2026-W47") == ("2026-11-16", "2026-11-22", "2026-11-23T08:00:00Z")


def test_exact_numbers_and_reconciliation(tmp_path):
    cfg = _ledger(tmp_path, _base())
    res, out = _build(tmp_path, cfg, CLOSED_NOW)
    d = _ds(out)
    assert set(d["repo"]) == {"v9", "pro"}                            # v11 never read
    assert set(d["gw"]) == {"2026-W47"}
    ou = d[d["market"] == "ou25"].iloc[0]
    assert (ou["predictions"], ou["settled"], ou["hits"]) == (1, 1, 1)
    assert ou["brier"] == pytest.approx(0.09) and ou["log_loss"] == pytest.approx(-math.log(0.7), abs=1e-6)
    x = d[d["market"] == "1x2"].iloc[0]
    assert x["hits"] == 0 and x["brier"] == pytest.approx(0.25 + 0.09 + 0.64)
    p = d[d["market"] == "sot"].iloc[0]
    assert (p["predictions"], p["settled"], p["void"], p["hits"]) == (3, 2, 1, 1)   # 0.1 NO hit, 0.6 YES miss
    assert p["hit_rate"] == pytest.approx(0.5)
    # Clean Record rows never mixed
    bt = d[d["market"] == "btts"]
    assert sorted(bt["clean_record"]) == [False, True] and set(bt["predictions"]) == {1}
    # totals reconcile exactly with the ledger questions (public repos)
    rows, _ = B.with_identity(B.load(tmp_path / "output" / "prediction_ledger",
                                     tmp_path / "output" / "prediction_ledger_settled"))
    sc = B.score(rows)
    assert d["predictions"].sum() + d["duplicate_refs_excluded"].sum() == sc["_qid"].nunique()
    assert d["hits"].sum() == int(sc["hit"].sum())
    assert (d["status"] == "CLOSED").all() and (d["closed_at"] == CLOSED_NOW).all()
    assert G.verify(d, G.read_links(out / G.LINKS_DIR.name)) == []
    meta = json.loads((out / G.META.name).read_text(encoding="utf-8"))
    assert meta["schema_version"] == G.SCHEMA_VERSION and "v9:prediction_ledger" in meta["sources_sha256"]
    for f in (out / G.DATASET.name, out / G.DATASET_CSV.name):
        cols = pd.read_parquet(f).columns if f.suffix == ".parquet" else pd.read_csv(f).columns
        assert not [c for c in cols if B.PL.PRICE_LIKE.search(str(c))]


def test_no_default_rate_without_settled(tmp_path):
    cfg = _ledger(tmp_path, [_row("v9", "v9_props", "sot", "YES", 0.3, line="0.5", player="9", status="PENDING")])
    _, out = _build(tmp_path, cfg, CLOSED_NOW)
    d = _ds(out)
    assert d["settled"].iloc[0] == 0 and d[["hit_rate", "brier", "log_loss"]].isna().all(axis=None)
    assert d["status"].iloc[0] == "OPEN"                                # pending -> never closed


def test_rerun_is_idempotent(tmp_path):
    cfg = _ledger(tmp_path, _base())
    _build(tmp_path, cfg, CLOSED_NOW)
    res, _ = _build(tmp_path, cfg, "2026-11-25T12:00:00Z")
    assert res["changed"] == []


def test_closed_gw_is_never_rewritten_and_a_change_is_a_correction(tmp_path):
    cfg = _ledger(tmp_path, _base())
    _, out = _build(tmp_path, cfg, CLOSED_NOW)
    before = _ds(out)
    links_before = G.read_links(out / G.LINKS_DIR.name)
    # the settled view is re-graded after the GW closed: the OVER did NOT happen after all
    rows = _base()
    rows[0] = _row("v9", "v9_standard", "ou25", "OVER", 0.7, line="2.5", y=0)
    rows[1] = _row("v9", "v9_standard", "ou25", "UNDER", 0.3, line="2.5", y=1)
    _ledger(tmp_path, rows)
    res, _ = _build(tmp_path, cfg, "2026-11-25T12:00:00Z")
    after = _ds(out)
    pd.testing.assert_frame_equal(before, after)                       # closed rows unchanged, byte for byte
    pd.testing.assert_frame_equal(links_before.reset_index(drop=True),
                                  G.read_links(out / G.LINKS_DIR.name).reset_index(drop=True), check_dtype=False)
    corr = G.read_corrections(out / "corrections" / G.CORRECTIONS.name)
    rid = before.loc[before["market"] == "ou25", "row_id"].iloc[0]
    hit = corr[(corr["row_id"] == rid) & (corr["field"] == "hits")]
    assert len(hit) == 1 and hit["reason"].iloc[0] == "LEDGER_CHANGED_AFTER_CLOSE"
    assert (hit["original"].iloc[0], hit["recomputed"].iloc[0]) == ("1", "0")
    # idempotent corrections: a rerun adds nothing
    _build(tmp_path, cfg, "2026-11-26T12:00:00Z")
    assert len(G.read_corrections(out / "corrections" / G.CORRECTIONS.name)) == len(corr)


def test_open_gw_follows_the_ledger(tmp_path):
    rows = _base()
    rows[0] = _row("v9", "v9_standard", "ou25", "OVER", 0.7, line="2.5", status="PENDING")
    rows[1] = _row("v9", "v9_standard", "ou25", "UNDER", 0.3, line="2.5", status="PENDING")
    cfg = _ledger(tmp_path, rows)
    _, out = _build(tmp_path, cfg, OPEN_NOW)
    d = _ds(out)
    assert (d["status"] == "OPEN").all()
    assert d.loc[d["market"] == "ou25", "pending"].iloc[0] == 1
    _ledger(tmp_path, _base())
    _build(tmp_path, cfg, CLOSED_NOW)
    d2 = _ds(out)
    ou = d2[d2["market"] == "ou25"].iloc[0]
    assert ou["status"] == "CLOSED" and ou["hits"] == 1 and ou["pending"] == 0


def test_verify_catches_a_tampered_row(tmp_path):
    cfg = _ledger(tmp_path, _base())
    _, out = _build(tmp_path, cfg, CLOSED_NOW)
    d = _ds(out)
    links = G.read_links(out / G.LINKS_DIR.name)
    d.loc[d["market"] == "ou25", "hits"] = 0
    assert any("hits does not reconcile" in i for i in G.verify(d, links))
    d = _ds(out)
    d.loc[d["market"] == "sot", "hit_rate"] = 0.0
    d.loc[d["market"] == "sot", "settled"] = 0
    assert G.verify(d, links)


def test_versions_are_never_pooled_and_flagged(tmp_path):
    rows = _base()
    old = _row("v9", "v9_props", "sot", "YES", 0.1, line="0.5", player="7", at="2026-11-18T09:00:00Z", y=0)
    old[0]["model_version"] = "props_sot_legacy_8e31d4d8"
    cfg = _ledger(tmp_path, rows + [old])
    _, out = _build(tmp_path, cfg, CLOSED_NOW)
    d = _ds(out)
    s = d[d["market"] == "sot"]
    assert len(s) == 2 and set(s["version_label"]) == {"legacy · 8e31d4d8", "v9_props_v1"}
    assert s.loc[s["model_version"] == "props_sot_legacy_8e31d4d8", "version_pre_clean_record"].iloc[0]


# ── one fixture, one identity ───────────────────────────────────────────────────────────────────────────────────
def _fx(ref, ko, home, away, league="Mexico Liga MX", slug="mexico_liga_mx"):
    return {"fixture_ref": ref, "league_slug": slug, "league": league, "kickoff_utc": ko, "home_team": home,
            "away_team": away}


def test_identity_merges_accent_variant_and_kickoff_drift():
    fx = pd.DataFrame([
        _fx("af:1551039", "2026-11-21T01:00:00Z", "Puebla", "Atletico San Luis"),
        _fx("oa:40846f0c", "2026-11-21T01:00:00Z", "Puebla", "Atlético San Luis"),           # accent
        _fx("af:1551038", "2026-11-21T03:00:00Z", "Club Tijuana", "Atlante FC"),
        _fx("oa:tij", "2026-11-21T03:10:00Z", "Tijuana", "Atlante FC"),                      # variant + drift
        _fx("oa:juarez", "2026-11-21T03:00:00Z", "FC Juárez", "Atlas"),
        _fx("oa:juarez2", "2026-11-21T03:00:00Z", "FC Juarez", "Atlas"),                     # accent, no AF
        _fx("af:9", "2026-11-28T01:00:00Z", "Puebla", "Atletico San Luis"),                  # another week: apart
    ])
    ident = FI.resolve_identities(fx).set_index("fixture_ref")
    assert ident.loc["oa:40846f0c", "canonical_fixture_ref"] == "af:1551039"
    assert ident.loc["oa:tij", "canonical_fixture_ref"] == "af:1551038"
    assert ident.loc["oa:tij", "kickoff_source"] == "api-football"
    assert ident.loc["oa:tij", "canonical_kickoff_utc"] == pd.Timestamp("2026-11-21T03:00:00Z")
    assert ident.loc["oa:tij", "kickoff_drift_min"] == 10
    assert ident.loc["oa:juarez2", "canonical_fixture_ref"] == "oa:juarez"
    assert ident.loc["af:9", "canonical_fixture_ref"] == "af:9"
    rep = FI.split_report(fx, ident.reset_index())
    assert rep["page_keys_before"] == 7 and rep["canonical_fixtures_after"] == 4
    assert rep["kinds"]["accent"] == 2 and rep["kinds"]["name_variant"] == 1 and rep["kinds"]["kickoff_drift"] == 1


def test_identity_kickoff_conflict_without_af_stays_missing_and_two_af_ids_are_refused():
    fx = pd.DataFrame([
        _fx("oa:a", "2026-11-28T15:00:00Z", "Rosario Central", "Barracas Central", "Argentina Primera Division",
            "argentina_primera_division"),
        _fx("oa:b", "2026-11-29T21:00:00Z", "Rosario Central", "Barracas Central", "Argentina Primera Division",
            "argentina_primera_division"),
        _fx("af:1", "2026-11-21T01:00:00Z", "Puebla", "Atletico San Luis"),
        _fx("af:2", "2026-11-21T01:00:00Z", "Puebla", "Atlético San Luis"),
    ])
    ident = FI.resolve_identities(fx).set_index("fixture_ref")
    assert ident.loc["oa:b", "canonical_fixture_ref"] == "oa:a"
    assert ident.loc["oa:b", "kickoff_source"] == "conflict" and pd.isna(ident.loc["oa:b", "canonical_kickoff_utc"])
    assert ident.loc["af:2", "canonical_fixture_ref"] == "af:2"
    assert ident.loc["af:2", "identity_note"] == "AF_ID_CONFLICT"


def test_dataset_counts_a_match_once_across_refs(tmp_path):
    """One model holding a match under two refs (two OddsAPI events) counts it once; the later ref is excluded."""
    rows = [_row("v9", "v9_standard", "ou25", "OVER", 0.7, line="2.5", fx="oa:x1", y=1),
            _row("v9", "v9_standard", "ou25", "UNDER", 0.3, line="2.5", fx="oa:x1", y=0),
            _row("v9", "v9_standard", "ou25", "OVER", 0.6, line="2.5", fx="oa:x2", at="2026-11-20T11:00:00Z", y=1),
            _row("v9", "v9_standard", "ou25", "UNDER", 0.4, line="2.5", fx="oa:x2", at="2026-11-20T11:00:00Z", y=0)]
    cfg = _ledger(tmp_path, rows)
    _, out = _build(tmp_path, cfg, CLOSED_NOW)
    d = _ds(out)
    assert (d["predictions"].iloc[0], d["duplicate_refs_excluded"].iloc[0], d["fixtures"].iloc[0]) == (1, 1, 1)
    assert d["log_loss"].iloc[0] == pytest.approx(-math.log(0.7), abs=1e-6)       # the first-predicted ref


# ── page helpers ───────────────────────────────────────────────────────────────────────────────────────────────
def test_version_labels():
    assert version_label("props_goals_props_clean_harness_5de94714") == "clean · 5de94714"
    assert version_label("props_goals_legacy_1592478c") == "legacy · 1592478c"
    assert version_label("btts_20261119T134019Z_44c67049") == "trained 2026-11-19 · 44c67049"
    assert version_label("standard_repaired_20261028T123040Z_1f5805e6") == "repaired · 1f5805e6"
    assert version_label("pro_bb_dc_1.0.0_66ed8add") == "v1.0.0 · 66ed8add"
    assert version_label("side_nb_v1_2026-11-09_c9e06a6a9c92") == "trained 2026-11-09 · c9e06a6a"


def test_serving_rows_keep_one_version_per_question():
    from dashboard_data import predictions_view as V
    ko = pd.Timestamp("2026-11-21T01:00:00Z")
    base = dict(model_id="v9_props", market="goals", line="0.5", player_id="7", snapshot="first_sight",
                canonical_fixture_ref="af:1551039", kickoff_utc=ko, selection="YES")
    r = pd.DataFrame([{**base, "model_version": "props_goals_legacy_1592478c",
                       "predicted_at": pd.Timestamp("2026-11-18T00:31:00Z")},
                      {**base, "model_version": "props_goals_props_clean_harness_5de94714",
                       "predicted_at": pd.Timestamp("2026-11-19T22:09:00Z")}])
    s = V.serving_rows(r)
    assert len(s) == 1 and s["model_version"].iloc[0].endswith("5de94714")
    assert len(r) == 2                                                   # the input (ledger) is untouched


# ── data truth ─────────────────────────────────────────────────────────────────────────────────────────────────
def test_data_truth_rules_registered():
    from src.integrity import data_truth as DT
    assert DT.is_gated("Data truth 14: prediction GW dataset")
    assert not DT.is_gated("Data truth 15: one fixture, one identity")
    assert not DT.is_gated("Data truth 16: prediction GW dataset vs ledger")
    c = DT.prediction_gw_dataset()
    assert c["status"] in (DT.GREEN, DT.GREY), c


def test_data_truth_15_flags_a_stored_split(tmp_path):
    from src.integrity import data_truth as DT
    fx = pd.DataFrame([_fx("af:1551039", "2026-11-21T01:00:00Z", "Puebla", "Atletico San Luis"),
                       _fx("oa:40846f0c", "2026-11-21T01:00:00Z", "Puebla", "Atlético San Luis")])
    fx["predicted_at"] = "2026-11-20T10:00:00Z"
    fx["kickoff_source"] = "api-football"
    fx["canonical_fixture_ref"] = fx["fixture_ref"]                       # stored split: wrong
    p = tmp_path / "idx.parquet"
    fx.to_parquet(p)
    assert DT.one_fixture_identity(p)["status"] == DT.RED
    fx["canonical_fixture_ref"] = "af:1551039"
    fx.to_parquet(p)
    assert DT.one_fixture_identity(p)["status"] == DT.GREEN


def test_unreachable_pro_ledger_keeps_pro_rows(tmp_path):
    v9, pro = tmp_path / "v9", tmp_path / "pro"
    cfg = _ledger(v9, [r for r in _base() if r[0]["repo"] == "v9"])
    _write(pro, [r for r in _base() if r[0]["repo"] == "pro"])
    out = tmp_path / "out"
    G.run(v9_root=v9, pro_root=pro, out=out, now=CLOSED_NOW, clean_cfg=cfg)
    before = _ds(out)
    assert "pro" in set(before["repo"])
    res = G.run(v9_root=v9, pro_root=None, out=out, now="2026-11-25T12:00:00Z", clean_cfg=cfg)
    pd.testing.assert_frame_equal(before, _ds(out))
    assert res["corrections_new"] == 0 and set(res["changed"]) <= {G.META.name}   # meta: the sources read


def test_index_builder_writes_one_identity_per_match(tmp_path):
    a = _row("v9", "v9_props", "goals", "YES", 0.2, line="0.5", player="7", fx="af:1551039", y=0)
    b = _row("v9", "v9_new_format", "btts", "YES", 0.5, fx="oa:40846f0c", y=1)
    c = _row("v9", "v9_new_format", "btts", "NO", 0.5, fx="oa:40846f0c", y=0)
    b[0]["away_team"] = c[0]["away_team"] = "Awáy FC"                       # the OddsAPI spelling (accent)
    cfg = _ledger(tmp_path, [a, b, c])
    out = tmp_path / "dash"
    res = B.run(v9_root=tmp_path, pro_root=None, out_dir=out, clean_cfg=cfg)
    idx = pd.read_parquet(out / B.INDEX_NAME)
    assert set(idx["canonical_fixture_ref"].astype(str)) == {"af:1551039"}
    assert set(idx["fixture_ref"].astype(str)) == {"af:1551039", "oa:40846f0c"}     # originals kept
    doc = json.loads((out / B.IDENTITY_NAME).read_text(encoding="utf-8"))
    assert (doc["page_keys_before"], doc["canonical_fixtures_after"], doc["kinds"]["accent"]) == (2, 1, 1)
    assert res["matches_split_before"] == 1
    from dashboard_data import predictions_view as V
    assert len(V.matches(V.load_index(out / B.INDEX_NAME))) == 1
