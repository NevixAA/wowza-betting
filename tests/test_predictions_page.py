"""Predictions page (owner 2026-11-22): the compact index builder (scripts/build_predictions_index.py) and the page.

The builder is checked on a tiny synthetic ledger (scoring of two-way, multi-outcome and single-selection questions,
voids, the Clean Record cut, v11 kept out, prices refused, byte-identical reruns); the page is rendered headless on
the committed files and without them.
"""
import json
import math
import shutil
from pathlib import Path

import pandas as pd
import pytest

from scripts import build_predictions_index as B
from src import prediction_ledger as PL

ROOT = Path(__file__).resolve().parents[1]
START = "2026-11-19T06:41:48Z"


def _row(repo, model, market, sel, p, *, line="", fx="af:1", player="", snap="first_sight",
         at="2026-11-20T10:00:00Z", league="championship", pv="v1", y=None, status="SETTLED"):
    r = {c: "" for c in PL.COLUMNS}
    r.update(schema_version="pl/1.0", repo=repo, model_id=model, model_version=f"{model}_v1", market=market,
             line=line, selection=sel, fixture_ref=fx, league="Championship", league_slug=league, season="2026",
             kickoff_utc="2026-11-21T15:00:00Z", home_team="Home FC", away_team="Away FC", player_id=player,
             player_name=f"P{player}" if player else "", probability=str(p), snapshot=snap, predicted_at=at,
             payload_version=pv, scored="1")
    return r, ({"outcome_status": status, "y": "" if y is None else str(y)})


def _write(root: Path, rows):
    raw, settled = {}, {}
    for r, o in rows:
        k = (r["repo"], r["model_id"], r["market"], r["league_slug"], r["season"])
        raw.setdefault(k, []).append(r)
        settled.setdefault(k, []).append({**r, **o})
    for (repo, model, market, league, season), rs in raw.items():
        for sub, data in (("prediction_ledger", rs), ("prediction_ledger_settled", settled[(repo, model, market,
                                                                                               league, season)])):
            p = root / "output" / sub / repo / model / market / league / f"{season}.csv"
            p.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(data).to_csv(p, index=False)


@pytest.fixture
def ledger(tmp_path):
    rows = [
        # two-way: OVER 0.7 happened -> hit, log loss -ln 0.7, binary Brier 0.09
        _row("v9", "v9_standard", "ou25", "OVER", 0.7, line="2.5", y=1),
        _row("v9", "v9_standard", "ou25", "UNDER", 0.3, line="2.5", y=0),
        # 1X2: H most likely, A happened -> miss, -ln 0.2, multi-class Brier .25+.09+.64
        _row("pro", "pro_bb_dc", "1x2", "H", 0.5, y=0),
        _row("pro", "pro_bb_dc", "1x2", "D", 0.3, y=0),
        _row("pro", "pro_bb_dc", "1x2", "A", 0.2, y=1),
        # single selections: a prop at 0.1 that did not happen (pick NO -> hit), one that is void, one pending
        _row("v9", "v9_props", "sot", "YES", 0.1, line="0.5", player="7", y=0),
        _row("v9", "v9_props", "sot", "YES", 0.2, line="0.5", player="8", status="VOID_DID_NOT_PLAY"),
        _row("v9", "v9_props", "sot", "YES", 0.3, line="0.5", player="9", status="PENDING"),
        # before the Clean Record start: in the index, never in the summary
        _row("v9", "v9_standard", "btts", "YES", 0.6, at="2026-11-18T10:00:00Z", y=1),
        _row("v9", "v9_standard", "btts", "NO", 0.4, at="2026-11-18T10:00:00Z", y=0),
        # ledger-only league
        _row("v9", "v9_standard", "ou15", "OVER", 0.8, line="1.5", fx="af:2", league="scottish_premiership",
             pv="v1|ledger_only:TRAINING_ONLY", y=1),
        _row("v9", "v9_standard", "ou15", "UNDER", 0.2, line="1.5", fx="af:2", league="scottish_premiership",
             pv="v1|ledger_only:TRAINING_ONLY", y=0),
        # private: must never be read
        _row("v11", "v11_market", "ou25", "OVER", 0.5, line="2.5", y=1),
    ]
    _write(tmp_path, rows)
    cfg = tmp_path / "trust.json"
    cfg.write_text(json.dumps({"clean_record_start_utc": START}), encoding="utf-8")
    return tmp_path, cfg


def _run(ledger, out):
    root, cfg = ledger
    return B.run(v9_root=root, pro_root=None, out_dir=out, clean_cfg=cfg)


def test_scores_each_kind_of_question(ledger, tmp_path):
    out = tmp_path / "out"
    _run(ledger, out)
    idx = pd.read_parquet(out / B.INDEX_NAME)
    assert set(idx["repo"].astype(str)) == {"v9", "pro"}                         # v11 never read
    s = pd.read_csv(out / B.SUMMARY_NAME).set_index(["model_id", "market"])
    ou = s.loc[("v9_standard", "ou25")]
    assert ou["settled"] == 1 and ou["hits"] == 1
    assert ou["log_loss"] == pytest.approx(-math.log(0.7), abs=1e-4)
    assert ou["brier"] == pytest.approx(0.09, abs=1e-4)
    x = s.loc[("pro_bb_dc", "1x2")]
    assert x["settled"] == 1 and x["hits"] == 0
    assert x["log_loss"] == pytest.approx(-math.log(0.2), abs=1e-4)
    assert x["brier"] == pytest.approx(0.25 + 0.09 + 0.64, abs=1e-4)
    pr = s.loc[("v9_props", "sot")]
    assert (pr["predictions"], pr["settled"], pr["void"], pr["pending"], pr["hits"]) == (3, 1, 1, 1, 1)
    assert pr["log_loss"] == pytest.approx(-math.log(0.9), abs=1e-4)
    assert (pr["single_settled"], pr["single_y_sum"]) == (1, 0)
    assert pr["single_p_sum"] == pytest.approx(0.1)
    # pick flags: one per two-way / multi question, every single row
    one = idx[(idx["market"] == "1x2")]
    assert one.loc[one["pick"], "selection"].astype(str).tolist() == ["H"]


def test_clean_record_cut_and_ledger_only_label(ledger, tmp_path):
    out = tmp_path / "out"
    _run(ledger, out)
    idx = pd.read_parquet(out / B.INDEX_NAME)
    s = pd.read_csv(out / B.SUMMARY_NAME)
    assert "btts" in set(idx["market"].astype(str))                               # in the index ...
    assert "btts" not in set(s["market"])                                         # ... not in the record
    assert s.loc[s["league_slug"] == "scottish_premiership", "ledger_only"].all()
    assert not s.loc[s["league_slug"] == "championship", "ledger_only"].any()
    assert (s["clean_record_start"] == START).all()


def test_rerun_is_byte_identical(ledger, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _run(ledger, a)
    _run(ledger, b)
    for n in (B.INDEX_NAME, B.SUMMARY_NAME):
        assert (a / n).read_bytes() == (b / n).read_bytes(), n


def test_a_price_column_is_refused(ledger, tmp_path):
    root, cfg = ledger
    p = next((root / "output" / "prediction_ledger" / "v9" / "v9_standard" / "ou25").rglob("*.csv"))
    d = pd.read_csv(p, dtype=str)
    d["odds"] = "2.0"
    d.to_csv(p, index=False)
    with pytest.raises(PL.LedgerError):
        B.run(v9_root=root, out_dir=tmp_path / "out", clean_cfg=cfg)


def test_question_sides():
    assert B.question_side("ou25", "OVER") == B.question_side("ou25", "UNDER") == ""
    assert B.question_side("cards_team", "HOME_OVER") == B.question_side("cards_team", "HOME_UNDER") == "HOME"
    assert B.question_side("team_ge4", "HOME") != B.question_side("team_ge4", "AWAY")
    assert B.question_side("correct_score", "1-0") == ""
    assert B.question_side("combo", "a|b") != B.question_side("combo", "a|c")


# ── the committed files ───────────────────────────────────────────────────────────────────────────────────────────
IDX = ROOT / "reports" / "dashboard" / B.INDEX_NAME
SUM = ROOT / "reports" / "dashboard" / B.SUMMARY_NAME


def test_committed_files_hold_no_price_and_no_v11():
    idx = pd.read_parquet(IDX)
    s = pd.read_csv(SUM)
    for cols in (idx.columns, s.columns):
        assert not [c for c in cols if PL.PRICE_LIKE.search(str(c))]
    assert set(idx["repo"].astype(str)) <= set(B.PUBLIC_REPOS)
    assert set(s["repo"]) <= set(B.PUBLIC_REPOS)


# ── the page ──────────────────────────────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def _admin(monkeypatch):
    monkeypatch.setenv("WOWZA_TIER_OVERRIDE", "admin")
    import streamlit as st
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


def _render():
    from streamlit.testing.v1 import AppTest
    page = next((ROOT / "pages").glob("28_*Predictions.py"))
    at = AppTest.from_file(str(page), default_timeout=300).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _text(at) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [t.value for t in at.title] \
        + [s.value for s in at.subheader]
    return "\n".join(str(p) for p in parts)


def test_page_renders_with_the_committed_files(_admin):
    at = _render()
    t = _text(at)
    assert "Predictions" in t and "v11: private" in t and "not bets" in t
    assert at.dataframe, "no table drawn"
    for df in at.dataframe:
        assert not [c for c in df.value.columns if PL.PRICE_LIKE.search(str(c))]
    assert "paper" not in t.lower()


def test_page_renders_an_error_without_its_index(_admin, tmp_path):
    bak = tmp_path / IDX.name
    shutil.move(IDX, bak)
    try:
        at = _render()
        assert "cannot be read" in _text(at)
    finally:
        shutil.move(bak, IDX)
