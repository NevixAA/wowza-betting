"""Quality layers (2026-10-26): walk-forward without look-ahead, deterministic layers, an immutable first-sight
shadow log, and no effect on any production output."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
from src.research import quality_tiers as Q

ROOT = Path(__file__).resolve().parents[1]


def _record(n_days=60, per_day=8, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    days = pd.date_range("2026-08-15", periods=n_days, freq="D").strftime("%Y-%m-%d")
    for d in days:
        for i in range(per_day):
            track = "new_format" if i % 2 else "standard"
            market = ["OU25", "BTTS"][i % 2 == 0 and i % 4 == 0]
            side = "OVER" if market == "OU25" else "YES"
            odds = float(rng.uniform(1.6, 2.6))
            p_model = min(0.9, 1 / odds + rng.uniform(0.03, 0.12))
            rows.append({"key": f"L{i % 3}|{d}|H{i}|A{i}|{market}|{side}", "day": d, "gen_day": d, "match_day": d,
                         "league": f"L{i % 3}", "model_type": track, "market": market, "side": side, "odds": odds,
                         "opp_odds": 1 / max(0.05, 1.06 - 1 / odds), "p_model": p_model,
                         "result": "WIN" if rng.uniform() < 1 / odds else "LOSS"})
    return pd.DataFrame(rows)


# ── 1. walk-forward has no look-ahead ───────────────────────────────────────────────────────────
def test_walk_forward_fits_only_on_matches_played_before_the_day(monkeypatch):
    rec = _record()
    seen = []
    real = Q.fit_track

    def spy(train, track):
        seen.append((train["match_day"].max() if len(train) else None, track))
        return real(train, track)

    monkeypatch.setattr(Q, "fit_track", spy)
    out = Q.walk_forward(rec)
    assert out["p_win"].notna().any(), "nothing was ever scored"
    for day in sorted(rec["gen_day"].unique()):
        sub = out[out["gen_day"] == day]
        ft = sub["fitted_through"].replace("", np.nan).dropna()
        assert (ft < day).all(), f"a model used for {day} saw a result from {ft.max()}"


def test_future_results_cannot_move_todays_score():
    rec = _record()
    day = sorted(rec["gen_day"].unique())[45]
    a = Q.walk_forward(rec)
    flipped = rec.copy()
    later = flipped["match_day"] >= day
    flipped.loc[later, "result"] = np.where(flipped.loc[later, "result"] == "WIN", "LOSS", "WIN")
    b = Q.walk_forward(flipped)
    m = rec["gen_day"] <= day
    pd.testing.assert_series_equal(a.loc[m, "score"], b.loc[m, "score"])


def test_league_rating_uses_only_rows_settled_before_each_row():
    rec = Q.add_basic(_record(n_days=10))
    r1 = Q._train_rating(rec)
    rec2 = rec.copy()
    last = rec2["match_day"] == rec2["match_day"].max()
    rec2.loc[last, "pnl"] = 99.0                                       # the last day's own results
    r2 = Q._train_rating(rec2)
    np.testing.assert_allclose(r1, r2)


# ── 2. layers are deterministic ─────────────────────────────────────────────────────────────────
def test_tercile_layers_do_not_depend_on_row_order():
    rec = _record()
    rec["score"] = np.round(np.random.default_rng(3).normal(0, 0.1, len(rec)), 2)      # ties on purpose
    a = Q.layer_tercile(rec).set_axis(rec["key"])
    shuf = rec.sample(frac=1.0, random_state=7)
    b = Q.layer_tercile(shuf).set_axis(shuf["key"]).reindex(a.index)
    assert (a == b).all()
    assert set(a.dropna()) <= set(Q.LAYERS)


def test_fixed_layers_use_the_frozen_cuts_inclusively():
    d = pd.DataFrame({"model_type": ["new_format"] * 4, "market": ["BTTS"] * 4, "score": [0.30, 0.20, 0.10, np.nan]})
    lay = Q.layer_fixed(d, {"new_format|BTTS": [0.10, 0.20]})
    assert lay.tolist()[:3] == ["BEST", "BEST", "SECOND"] and pd.isna(lay.iloc[3])      # >= cut is the upper layer
    assert Q.layer_fixed(d.assign(score=0.05), {"new_format|BTTS": [0.10, 0.20]}).iloc[0] == "THIRD"


def test_frozen_cuts_file_matches_the_study():
    import json
    cuts = json.loads(Q.CUTS_FILE.read_text(encoding="utf-8"))
    res = json.loads((Q.STUDY_DIR / "study_results.json").read_text(encoding="utf-8"))
    assert cuts["cuts"] == res["meta"]["cuts_fixed_from_discovery"]
    assert cuts["version"] == Q.VERSION


# ── 3. the shadow log is first sight and immutable ──────────────────────────────────────────────
@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(Q, "SHADOW_FILE", tmp_path / "quality_tier_shadow.csv")
    seed = _record(n_days=50)
    seed_path = tmp_path / "seed.csv"
    seed.to_csv(seed_path, index=False)
    monkeypatch.setattr(Q, "SEED_FILE", seed_path)
    ko = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=2)).normalize() + pd.Timedelta(hours=15)
    md = ko.strftime("%Y-%m-%d")
    side = pd.DataFrame([{"source": "live", "generated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M"),
                          "signal_date": md, "match_date": md, "league": "L1", "home_team": "Home", "away_team": "Away",
                          "market": "btts", "odds": 2.0, "edge_pct": 6.0, "ev_pct": 5.0, "model_prob": 0.55,
                          "signal_tier": "MARKSMAN", "model_type": "new_format", "result": ""}])
    side.to_csv(tmp_path / "side_bets_ledger.csv", index=False)
    pd.DataFrame([{"date": md, "kickoff_utc": ko.strftime("%Y-%m-%dT%H:%M:%SZ"), "league": "L1", "home_team": "Home",
                   "away_team": "Away", "model_type": "new_format", "p_btts": 0.55, "odds_btts": 2.0,
                   "odds_btts_no": 1.8, "p_over25": 0.5, "odds_over25": 1.9, "odds_under25": 1.9}]).to_csv(
        tmp_path / "predictions.csv", index=False)
    return tmp_path


def _hashes(d: Path, skip=()):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir())
            if p.is_file() and p.name not in skip}


def test_shadow_logs_once_and_never_revises(env):
    assert Q.log_shadow() == 1
    first = (env / "quality_tier_shadow.csv").read_bytes()
    row = pd.read_csv(env / "quality_tier_shadow.csv").iloc[0]
    assert row["layer_fixed"] in Q.LAYERS and row["quality_version"] == Q.VERSION and row["p_win"] > 0
    assert row["fitted_through"] < row["observed_at"][:10]
    # the ledger upgrades the tip in place (tier + odds) - the logged opinion must not move
    led = pd.read_csv(env / "side_bets_ledger.csv")
    led.loc[0, ["odds", "signal_tier", "edge_pct"]] = [2.4, "SNIPER", 15.0]
    led.to_csv(env / "side_bets_ledger.csv", index=False)
    assert Q.log_shadow() == 0
    assert (env / "quality_tier_shadow.csv").read_bytes() == first
    assert Q.shadow_store().verify()["ok"] is True


def test_shadow_refuses_to_append_after_tampering(env):
    Q.log_shadow()
    p = env / "quality_tier_shadow.csv"
    d = pd.read_csv(p, dtype=str, keep_default_na=False)
    d.loc[0, "layer_fixed"] = "BEST" if d.loc[0, "layer_fixed"] != "BEST" else "THIRD"
    d.to_csv(p, index=False, lineterminator="\n")
    assert Q.shadow_store().verify()["ok"] is False
    led = pd.read_csv(env / "side_bets_ledger.csv")
    led = pd.concat([led, led.assign(home_team="Home2", away_team="Away2")], ignore_index=True)
    led.to_csv(env / "side_bets_ledger.csv", index=False)
    assert Q.log_shadow() == 0, "appended to a tampered first-sight log"


def test_shadow_skips_tips_not_on_the_pre_match_board(env):
    pd.read_csv(env / "predictions.csv").iloc[0:0].to_csv(env / "predictions.csv", index=False)
    assert Q.log_shadow() == 0


# ── 4. production outputs are unchanged ─────────────────────────────────────────────────────────
def test_shadow_step_changes_no_other_output(env):
    before = _hashes(env, skip=("quality_tier_shadow.csv", "quality_tier_shadow.csv.first_sight.json"))
    Q.log_shadow()
    after = _hashes(env, skip=("quality_tier_shadow.csv", "quality_tier_shadow.csv.first_sight.json"))
    assert before == after


def test_predict_code_path_does_not_know_about_quality_layers():
    """The layer is research: no tip, tier, stake, ledger or Telegram code may read it."""
    for rel in ("pipeline.py", "src/predict.py", "src/ledger.py", "src/stake_plan.py", "src/top_picks.py",
                "telegram_bot/notifier.py"):
        p = ROOT / rel
        if p.exists():
            assert "quality_tier" not in p.read_text(encoding="utf-8"), f"{rel} references the research layer"


def test_workflow_step_is_tolerant_and_committed():
    wf = (ROOT / ".github" / "workflows" / "predict.yml").read_text(encoding="utf-8")
    assert "python -m src.research.quality_tiers --shadow ||" in wf
    assert "output/quality_tier_shadow.csv" in wf
    i_step, i_commit = wf.index("quality_tiers --shadow"), wf.index("Commit and push outputs")
    i_notify = wf.index("telegram_bot/notifier.py")
    assert i_notify < i_step < i_commit, "the shadow must run after tips are sent and before the commit"
