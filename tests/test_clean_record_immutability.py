"""Clean Record immutability (owner deep-repair brief 2026-11-22, P1 item 8).

1. A record generated before the Clean Record start never moves into the Clean Record - whatever its match date,
   whatever later settlement fills in, under every split the repo has (src/clean_record_eval.record_of,
   dashboard_data/clean_record.assign, src/trustworthy_performance.assign_live).
2. The identity fields of every pre-start live row are frozen: the committed ledgers still hash to the fingerprint
   committed in output/clean_record_evaluation.json, and no pre-start row is counted as Clean Record.
3. Frozen first-sight fields never change: every output/*.first_sight.json sidecar still matches its file.
4. The committed summary carries no price and echoes the owner's flags unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import clean_record_eval as E

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
SUMMARY = OUT / E.OUT_FILE
CELLS = OUT / E.CELLS_FILE


def _cfg():
    return json.loads((ROOT / "config" / "trustworthy_performance.json").read_text(encoding="utf-8"))


def _synthetic(start: pd.Timestamp, n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gen = start + pd.to_timedelta(rng.integers(-20 * 24 * 60, 20 * 24 * 60, n), unit="min")
    play = gen + pd.to_timedelta(rng.integers(-3 * 24 * 60, 10 * 24 * 60, n), unit="min")
    return pd.DataFrame({"generated_at": gen.strftime("%Y-%m-%d %H:%M"), "match_date": play.strftime("%Y-%m-%d"),
                         "pnl": rng.normal(0, 1, n), "result": rng.choice(["WIN", "LOSS", "VOID", ""], n)})


# ── 1. the split ──────────────────────────────────────────────────────────────────────────────────
def test_rows_generated_before_the_start_never_become_clean_record():
    from dashboard_data import clean_record as CR
    from src import trustworthy_performance as TP
    cfg = _cfg()
    start = E.start_ts(cfg)
    d = _synthetic(start)
    pre = pd.to_datetime(d["generated_at"], utc=True) < start
    assert pre.any() and (~pre).any()
    ours = E.record_of(d["generated_at"], d["match_date"], start)
    assert not (ours[pre] == "CLEAN").any()
    assert (ours[pre] == "BEFORE").all()
    assert not (CR.assign(d, "generated_at", "match_date")[pre] == CR.CLEAN).any()
    assert not (TP.assign_live(d, start)[pre] == TP.C).any()
    # one rule: the evaluation's split is the dashboard's split
    assert ours.tolist() == CR.assign(d, "generated_at", "match_date").tolist()


def test_settling_or_appending_rows_never_moves_a_pre_start_row(tmp_path):
    start = E.start_ts(_cfg())
    d = _synthetic(start, seed=11)
    before = E.record_of(d["generated_at"], d["match_date"], start)
    later = d.copy()
    later["result"], later["pnl"] = "WIN", 0.9                         # settlement fills grading columns only
    grown = pd.concat([later, _synthetic(start + pd.Timedelta(days=30), n=50, seed=3)], ignore_index=True)
    after = E.record_of(grown["generated_at"], grown["match_date"], start).iloc[:len(d)]
    assert after.tolist() == before.tolist()


def test_a_date_only_generation_on_the_start_day_is_never_clean():
    start = E.start_ts(_cfg())
    day = start.strftime("%Y-%m-%d")
    d = pd.DataFrame({"g": [day], "m": [day]})
    assert E.record_of(d["g"], d["m"], start).iloc[0] != "CLEAN"


# ── 2. the identity fingerprint of pre-start rows ──────────────────────────────────────────────────
def _ledger(tmp_path, rows):
    cols = E.IDENTITY["bets_ledger.csv"] + ["result", "pnl", "closing_odds", "clv_pct"]
    pd.DataFrame(rows, columns=cols).to_csv(tmp_path / "bets_ledger.csv", index=False)


def _row(gen, day, tier="SNIPER", result="", pnl=""):
    return [gen, day, "L", "H", "A", "UNDER", "standard", tier, "m1", result, pnl, "", ""]


def test_fingerprint_ignores_settlement_and_new_rows_but_catches_rewrites(tmp_path):
    cfg = _cfg()
    start = E.start_ts(cfg)
    old = (start - pd.Timedelta(days=2)).strftime("%Y-%m-%d %H:%M")
    new = (start + pd.Timedelta(days=1)).strftime("%Y-%m-%d %H:%M")
    day = (start + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    _ledger(tmp_path, [_row(old, day)])
    base = E.pre_start_fingerprint(tmp_path, cfg)["bets_ledger.csv"]
    assert base["pre_start_rows"] == 1 and base["clean_rows_generated_before_start"] == 0
    _ledger(tmp_path, [_row(old, day, result="WIN", pnl="0.9"), _row(new, day)])   # settled + a new clean row
    assert E.pre_start_fingerprint(tmp_path, cfg)["bets_ledger.csv"]["identity_sha256"] == base["identity_sha256"]
    _ledger(tmp_path, [_row(old, day, tier="VALUABLE")])                         # a frozen field rewritten
    assert E.pre_start_fingerprint(tmp_path, cfg)["bets_ledger.csv"]["identity_sha256"] != base["identity_sha256"]
    _ledger(tmp_path, [_row(old, day), _row(old, day, tier="MARKSMAN")])         # a back-dated row added
    assert E.pre_start_fingerprint(tmp_path, cfg)["bets_ledger.csv"]["identity_sha256"] != base["identity_sha256"]


def test_committed_ledgers_still_match_the_committed_pre_start_fingerprint():
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["clean_record_start_utc"] == _cfg()["clean_record_start_utc"]
    now = E.pre_start_fingerprint(OUT, _cfg())
    for name, pinned in summary["immutability"].items():
        cur = now[name]
        assert cur["present"] == pinned["present"], name
        if not pinned["present"]:
            continue
        assert cur["clean_rows_generated_before_start"] == 0, name
        assert cur["pre_start_rows"] == pinned["pre_start_rows"], \
            f"{name}: pre-start rows {pinned['pre_start_rows']} -> {cur['pre_start_rows']} (removed or back-dated)"
        assert cur["identity_sha256"] == pinned["identity_sha256"], \
            f"{name}: an identity field of a row generated before the Clean Record start changed"


# ── 3. first-sight sidecars ───────────────────────────────────────────────────────────────────────
SIDECARS = sorted(OUT.rglob("*.first_sight.json"))


@pytest.mark.parametrize("sidecar", SIDECARS, ids=[p.name for p in SIDECARS])
def test_frozen_first_sight_fields_never_change(sidecar):
    from src.research.first_sight import frozen_digest
    sc = json.loads(sidecar.read_text(encoding="utf-8"))
    data = sidecar.with_name(sidecar.name[:-len(".first_sight.json")])
    d = pd.read_csv(data, dtype=str, keep_default_na=False)
    n = int(sc["rows"])
    assert len(d) >= n, f"{data.name}: {n} frozen rows recorded, {len(d)} in the file"
    assert frozen_digest(d, sc["frozen_cols"], n) == sc["frozen_digest"], f"{data.name}: a frozen field changed"


def test_there_are_first_sight_sidecars_to_check():
    assert any(p.name == "top3_league_ledger.csv.first_sight.json" for p in SIDECARS)


# ── 4. the committed summary ──────────────────────────────────────────────────────────────────────
def test_summary_has_no_price_and_keeps_the_owner_flags():
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["performance_claims_enabled"] == bool(_cfg()["performance_claims_enabled"])
    keys = set()

    def walk(o):
        if isinstance(o, dict):
            keys.update(o)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(summary)
    assert not {k for k in keys if any(w in k.lower() for w in ("odds", "price"))} - {
        "real_price_columns", "real_odds_coverage", "priced_settled", "real_price_backtests",
        "synthetic_price_diagnostics"}
    cells = pd.read_csv(CELLS, nrows=0).columns
    assert not [c for c in cells if "odds" in c.lower() and c != "real_odds_coverage" or "price" in c.lower()
                and c != "priced_settled"]
    assert summary["headline"]["verdict"]


def test_verdict_is_never_proven_and_history_is_never_a_candidate():
    mins = {"settled": 30, "matchdays": 8, "clv": 150}
    strong = {"priced_settled": 500, "settled": 500, "matchdays": 60, "roi_ci": [0.05, 0.2],
              "clv": {"status": "OK", "mean_ci": [0.5, 1.5]}}
    assert E.verdict(strong, mins, bets=True, forward=False) == E.POS_UNCONFIRMED
    assert E.verdict(strong, mins, bets=True, forward=True) == E.CANDIDATE
    weak = {**strong, "priced_settled": 29}
    assert E.verdict(weak, mins, bets=True, forward=True) == E.NOT_ENOUGH
    assert "PROVEN" not in {E.verdict(strong, mins, bets=True, forward=f) for f in (True, False)}
