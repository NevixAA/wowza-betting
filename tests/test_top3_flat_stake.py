"""Owner 2026-11-17 (final): from Nov 21 EVERY TOP-3 tip carries a real stake of 0.5% of bankroll, flat; the owner
places the bets MANUALLY; exec and Cloudbet stay off; nothing else stakes.

Proves, on the committed config/bankroll_policy.json and on synthetic picks:
  1. every TOP-3 pick (TOP and TIER2 slots) records and shows 0.5%;
  2. nothing that is not a TOP-3 pick carries a stake (every cell is PAPER, the stake plan sizes 0, a league-rule
     line keeps its cell's label);
  3. nothing can execute automatically (the TOP-3 stake never reaches the stake plan, Cloudbet keeps no real
     candidate and its client stays DRY by default, no workflow sets CLOUDBET_MODE).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from src import bankroll_policy as BP
from src import top3_league as T
from tests.test_top3_league import (LEAD, NOW, _cand, _ctx, _frame, _lw_ctx, _stats)

ROOT = Path(__file__).resolve().parents[1]
POLICY = json.loads((ROOT / "config" / "bankroll_policy.json").read_text(encoding="utf-8"))


def _freeze_with(policy, cands, ctx, stats=None):
    return T.freeze_due(cands, set(), ctx, NOW, LEAD, {}, policy, None, stats)


# ── 1. every TOP-3 pick shows 0.5% ──────────────────────────────────────────────────────────────
def test_committed_policy_sets_a_flat_manual_half_percent_top3_stake():
    assert T.top3_stake(POLICY) == 0.005
    st = POLICY["top3_league"]["stake"]
    assert st["rule"] == "FLAT" and st["execution"] == "MANUAL"
    assert T.stake_text(0.005) == "0.5%"


def test_every_top3_pick_records_and_shows_the_flat_stake():
    rows = [_cand(1), _cand(2, "OVER15", "OVER"), _cand(3, "OVER35", "OVER"), _cand(4, "OU25", "OVER", pm=0.58)]
    picks, _ = _freeze_with(POLICY, _frame(rows), _ctx(),
                            _stats({"over25_over": 1.0, "over15_over": 0.5, "over35_over": 0.8}))
    top = picks[picks["selection_mode"] == T.SELECTION_MODE]
    assert 1 <= len(top) <= 3
    assert top["stake_effect"].str.startswith("REAL 0.5% of bankroll, flat").all()
    assert top["stake_effect"].str.contains("MANUALLY").all()
    assert top["track_is_real"].astype(bool).all()
    msg = T.message(top.assign(league_week_id="La Liga 2|2026-W47"), real_checker=lambda *a, **k: False,
                    policy=POLICY)
    tip_lines = [ln for ln in msg.splitlines() if ln[:2] in {f"{i}." for i in range(1, 4)}]
    assert len(tip_lines) == len(top)
    assert all(ln.endswith(" · 0.5% stake") for ln in tip_lines)     # owner 2026-11-17 wording
    assert "Paper" not in msg and "PAPER" not in msg            # owner: TOP-3 tips carry no paper label


def test_tier2_slot_is_a_top3_tip_too():
    picks, _ = _freeze_with(POLICY, _frame([_cand(1)]), _ctx(), _stats({"over25_over": 1.0}))
    top = picks[picks["selection_mode"] == T.SELECTION_MODE]
    assert set(top["slot"]) <= {T.SLOT_TOP, T.SLOT_TIER2}
    assert top["stake_effect"].str.startswith("REAL 0.5%").all()


def test_a_paper_league_pick_is_never_staked():
    """USA MLS is a PAPER league (config.PAPER_LEAGUES): its TOP-3 picks are ledgered and graded but never sent,
    so the owner cannot place them - no stake (11-03..09 dry run: 1 of 10 picks was USA MLS)."""
    import config
    lg = "USA MLS"
    assert config.is_paper_league(lg)
    rows = [_cand(1, league=lg)]
    picks, _ = _freeze_with(POLICY, _frame_for(rows, lg), _ctx(league=lg), _stats({"over25_over": 1.0}, league=lg))
    top = picks[picks["selection_mode"] == T.SELECTION_MODE]
    assert len(top) == 1 and top["stake_effect"].str.startswith("NONE").all()
    assert "0.5%" not in T._stake_label(lambda *a, **k: False, top.iloc[0].to_dict(), 0.005)


def _frame_for(rows, league):
    lock = _cand(99, ko="2026-11-20T19:00:00Z", pm=0.5, pf=0.5, ev_lb=-0.1, league=league)
    return pd.concat([pd.DataFrame(rows), pd.DataFrame([lock])], ignore_index=True).astype(object)


@pytest.mark.parametrize("bad", [
    {}, {"rule": "FLAT", "fraction": 0.005}, {"rule": "FLAT", "fraction": 0.005, "execution": "AUTO"},
    {"rule": "KELLY", "fraction": 0.005, "execution": "MANUAL"}, {"rule": "FLAT", "fraction": float("nan"),
                                                                    "execution": "MANUAL"},
    {"rule": "FLAT", "fraction": 0.05, "execution": "MANUAL"}, {"rule": "FLAT", "fraction": 0, "execution": "MANUAL"},
    {"rule": "FLAT", "fraction": "x", "execution": "MANUAL"}])
def test_no_default_an_invalid_stake_block_is_no_stake(bad):
    pol = {"top3_league": {"stake": bad}}
    assert T.top3_stake(pol) is None
    picks, _ = _freeze_with(pol, _frame([_cand(1)]), _ctx(), _stats({"over25_over": 1.0}))
    assert picks["stake_effect"].str.startswith("NONE").all()


# ── 2. nothing outside TOP-3 carries a stake ────────────────────────────────────────────────────
def test_every_cell_is_paper_and_no_track_stakes_under_cell_staking():
    assert BP.cell_staking_on(POLICY)
    for key, cell in POLICY["cell_staking"]["cells"].items():
        assert cell["state"] == "PAPER" and float(cell["base_fraction"]) == 0.0, key
        assert not (cell["state"] in BP.REAL_STATES and BP.positive_fraction(cell)), key
    # any track x league: no cell entry -> PAPER (a track's own state is never inherited)
    for tk in POLICY["tracks"]:
        e = BP.stake_entry(POLICY, tk, "Any League")
        assert not (e.get("state") in BP.REAL_STATES and BP.positive_fraction(e)), tk


def test_league_rule_line_is_not_a_top3_tip_and_keeps_its_cell_label():
    rows = [_cand(i, "BTTS", "YES", pm=0.5, pf=0.5) for i in range(1, 4)] + [_cand(10, "OU25", "OVER")]
    picks, _ = _freeze_with(POLICY, _frame(rows), _lw_ctx(), _stats({"over25_over": 1.0}))
    rule = picks[picks["selection_mode"] == T.OWN_LIST_MODE]
    assert len(rule) == 3 and rule["stake_effect"].str.startswith("NONE").all()
    msg = T.message(picks.assign(league_week_id="La Liga 2|2026-W47"), real_checker=lambda *a, **k: False,
                    policy=POLICY)
    rule_line = [ln for ln in msg.splitlines() if ln.startswith("📏 League rule")]
    assert rule_line and "0.5%" not in rule_line[0]


def test_stake_plan_sizes_nothing_on_the_committed_policy():
    """The sizer behind output/stake_plan.csv (Cloudbet / exec read it): every former staked cell, as a tip,
    gets stake 0."""
    def _sel(league, model_type, market, side, odds):
        return {"date": "2026-11-21", "fixture": f"{league} {market} H v A", "league": league, "home_team": "H",
                "away_team": "A", "model_type": model_type, "market": market, "side": side, "odds": odds,
                "signal_tier": "QUALIFIED", "edge": 0.05}
    sel = pd.DataFrame([_sel("Argentina Primera Division", "new_format", "BTTS", "YES", 1.9),
                        _sel("Finland Veikkausliiga", "new_format", "OU25", "OVER", 1.9),
                        _sel("Finland Veikkausliiga", "new_format", "OVER35", "OVER", 2.4),
                        _sel("La Liga 2", "standard", "OU25", "OVER", 2.0)])
    out = BP.size_day(sel, POLICY, bankroll=1000.0, peak=1000.0)
    stakes = pd.to_numeric(out["stake_amount"], errors="coerce").fillna(0.0)
    assert len(out) == len(sel) and (stakes == 0.0).all(), out[["league", "track", "stake_amount", "reason"]]
    assert (pd.to_numeric(out["final_fraction"], errors="coerce").fillna(0.0) == 0.0).all()


# ── 3. nothing executes automatically ───────────────────────────────────────────────────────────
def test_top3_stake_never_reaches_the_stake_plan_or_execution_code():
    import inspect
    src = inspect.getsource(T)
    assert "stake_plan.csv\", " not in src.replace("LW._read(od / \"stake_plan.csv\")", "")
    for mod in ("src/stake_plan.py", "src/cloudbet/bot.py", "src/cloudbet/client.py"):
        txt = (ROOT / mod).read_text(encoding="utf-8")
        assert "top3_league" not in txt and "top3_stake" not in txt, mod


def test_cloudbet_keeps_no_real_candidate_and_the_client_is_dry_by_default(monkeypatch):
    from src.cloudbet import bot as B
    cands = pd.DataFrame([{"model_type": "new_format", "market": "BTTS", "side": "YES",
                           "league": "Argentina Primera Division", "bet_kind": "team"},
                          {"model_type": "standard", "market": "OU25", "side": "OVER", "league": "La Liga 2",
                           "bet_kind": "team"}])
    assert B._drop_without_real_track(cands).empty
    src = (ROOT / "src" / "cloudbet" / "client.py").read_text(encoding="utf-8")
    assert 'MODE = os.getenv("CLOUDBET_MODE", DRY)' in src
    setters = [p.name for p in (ROOT / ".github" / "workflows").glob("*.yml")
               if "CLOUDBET_MODE" in p.read_text(encoding="utf-8", errors="replace")]
    assert setters == [], setters


def test_research_lock_watches_the_stakes():
    from src import research_lock as RL
    blk = RL._stakes_block(POLICY)
    assert blk["top3_stake"] == {"rule": "FLAT", "fraction": 0.005, "execution": "MANUAL"}
    assert all(c["state"] == "PAPER" for c in blk["cells"].values())
    assert "stakes" in RL.WATCHED_GLOBAL_BLOCKS
    assert math.isclose(sum(c["base_fraction"] for c in blk["cells"].values()), 0.0)
