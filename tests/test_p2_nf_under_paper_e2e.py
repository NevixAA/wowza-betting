"""Phase 2 / P1_UNDER25_RELEASE_GATE: new-format O/U 2.5 UNDER is PAPER end to end (owner 2026-11-02).

"Paper" means: shown and recorded, never staked, never presented as a staked bet. Every test below drives
the REAL code path (betting.evaluate_value -> generate_bets -> stake_plan.write; the Cloudbet bot's run();
the Telegram instant-tip builder; the dashboard Tips table; ledger.append_tips) on a synthetic board, and
every negative check has a POSITIVE CONTROL that proves the check can fail:

  * the same board with the track switched to ACTIVE_LOW in memory DOES get a stake, so a zero stake is
    caused by the PAPER track state and nothing else (league-threshold evidence is forced True, because
    on 2026-11-03 no NF league has per-league O/U evidence, which would zero every NF stake by itself);
  * an NF OVER tip on the same board (ACTIVE_LOW) is staked / placed / labelled as staked.

Nothing here changes the policy file, a threshold, a stake or a track state; nothing is sent or placed
(Telegram _send and the Cloudbet client are replaced or DRY; every file goes to tmp_path).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

import config
from src import bankroll_policy as BP
from src import betting

@pytest.fixture(autouse=True)
def _own_bars(monkeypatch):
    """Owner 2026-11-09: no global bars. These tests predate the cutover and assert the tiers the
    removed global path served, so every league gets exactly those values as its OWN owner-applied
    bars (tests/own_bars.install_historical)."""
    from tests.own_bars import install_historical
    install_historical(monkeypatch)


ROOT = Path(__file__).resolve().parents[1]
TRACK = "new_format|OU25_UNDER"
NF_LEAGUE = next(lg for lg in sorted(config.ENABLED_LEAGUES)
                 if config.model_type_for_league(lg) == "new_format" and not config.is_paper_league(lg))
DAY = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=2)).strftime("%Y-%m-%d")


def _policy() -> dict:
    """The committed policy with PER-TRACK staking: these tests flip the NF Under TRACK and pin its
    mechanics; the live per-cell staking (owner 2026-11-09) is pinned in tests/test_cell_staking.py."""
    p = json.loads((ROOT / "config" / "bankroll_policy.json").read_text(encoding="utf-8"))
    p.setdefault("cell_staking", {})["enabled"] = False
    return p


def _board(day: str = DAY) -> pd.DataFrame:
    """Two NF fixtures through the REAL tiering: a strong UNDER (p_over 0.25 @ 2.20) and a strong OVER
    (p_over 0.78 @ 1.95). Prices sit inside every odds cap of both side tracks."""
    raw = pd.DataFrame([
        {"date": day, "kickoff_utc": f"{day}T18:00:00Z", "league": NF_LEAGUE, "home_team": "Under FC",
         "away_team": "Low Town", "p_over25": 0.25, "odds_over25": 1.70, "odds_under25": 2.20,
         "model_type": "new_format", "drift_signal": "Neutral"},
        {"date": day, "kickoff_utc": f"{day}T20:00:00Z", "league": NF_LEAGUE, "home_team": "Over FC",
         "away_team": "High Town", "p_over25": 0.78, "odds_over25": 1.95, "odds_under25": 1.90,
         "model_type": "new_format", "drift_signal": "Neutral"},
    ])
    bets = betting.generate_bets(betting.evaluate_value(raw))
    assert set(bets["best_side"]) == {"OVER", "UNDER"}, bets
    assert set(bets["signal_tier"]) <= {"QUALIFIED"}, "board must hold tips (owner 2026-11-11: one bar per cell)"
    return bets


@pytest.fixture
def evidence(monkeypatch):
    """Per-league threshold evidence forced True, so ONLY the track state can zero a stake."""
    monkeypatch.setattr(betting, "league_threshold_evidence", lambda *a, **k: True)


# ── 1. policy file, research lock and track registry agree ────────────────────────────────────────
def test_policy_file_nf_under_is_paper_with_zero_base():
    t = _policy()["tracks"][TRACK]
    assert t["state"] == "PAPER" and float(t["base_fraction"]) == 0.0
    assert t["state"] not in BP.REAL_STATES
    assert t.get("paper_reason"), "a PAPER track must carry the owner's reason"
    # the sibling OVER track is still real money (the positive control below depends on it)
    assert _policy()["tracks"]["new_format|OU25_OVER"]["state"] in BP.REAL_STATES


def test_research_lock_and_track_registry_agree_with_policy():
    lock = json.loads((ROOT / "config" / "research_lock_snapshot.json").read_text(encoding="utf-8"))
    locked = lock["fingerprint"]["global"]["bankroll_policy"]["tracks"][TRACK]
    assert locked["state"] == "PAPER" and float(locked["base_fraction"]) == 0.0
    reg = json.loads((ROOT / "config" / "track_registry.json").read_text(encoding="utf-8"))
    rows = [e for e in reg["entries"] if e.get("track") == TRACK]
    assert rows, "the registry lists no NF Under entry"
    for e in rows:
        assert e["track_state"] == "PAPER" and e["real_money"] is False, e["key"]
        assert float(e["stake_policy"]["base_fraction"]) == 0.0, e["key"]
    assert TRACK not in reg["global"]["real_money_tracks"]


# ── 2. the real sizing path: evaluate_value -> generate_bets -> stake_plan.write ───────────────────
def _write_plan(tmp_path, monkeypatch, policy: dict | None = None) -> pd.DataFrame:
    from src import stake_plan as SP
    monkeypatch.setattr(SP, "PLAN", tmp_path / "stake_plan.csv")
    monkeypatch.setattr(SP, "AUDIT", tmp_path / "stake_audit.csv")
    if policy is not None:
        monkeypatch.setattr(BP, "load_policy", lambda path=None: copy.deepcopy(policy))
    plan = SP.write(_board(), None)
    assert len(plan) == 2, "stake plan not written (SelectionSchemaError or skipped)"
    return plan.set_index("side")


def test_stake_plan_gives_nf_under_zero_and_records_it(tmp_path, monkeypatch, evidence):
    plan = _write_plan(tmp_path, monkeypatch)
    u, o = plan.loc["UNDER"], plan.loc["OVER"]
    assert u["track"] == TRACK and u["deployment_state"] == "PAPER"
    assert float(u["final_fraction"]) == 0.0 and float(u["stake_amount"]) == 0.0
    assert u["reason"] == "NO_REAL_STAKE_PAPER"
    assert float(o["final_fraction"]) > 0.0, "positive control: the NF OVER tip must be staked"
    audit = pd.read_csv(tmp_path / "stake_audit.csv")
    rec = audit[audit["track"] == TRACK]
    assert len(rec) == 1 and float(rec["final_fraction"].iloc[0]) == 0.0      # audited, with its zero


def test_same_board_is_staked_when_track_is_real(tmp_path, monkeypatch, evidence):
    """Meaningful failure condition: flip ONLY the track (in memory) and the Under is staked."""
    pol = _policy()
    pol["tracks"][TRACK] = {**pol["tracks"][TRACK], "state": "ACTIVE_LOW", "base_fraction": 0.005}
    plan = _write_plan(tmp_path, monkeypatch, pol)
    assert float(plan.loc["UNDER", "final_fraction"]) > 0.0


def test_size_day_never_stakes_nf_under_whatever_the_tier_or_odds():
    pol = _policy()
    rows = [{"fixture": f"f{i}", "league": NF_LEAGUE, "model_type": "new_format", "market": "OU25",
             "side": side, "signal_tier": tier, "odds": odds, "league_threshold_evidence": True}
            for i, (side, tier, odds) in enumerate(
                [(s, t, o) for s in ("UNDER", "under", " Under ") for t in ("SNIPER", "MARKSMAN", "VALUABLE")
                 for o in (1.80, 2.10, 2.45)])]
    out = BP.size_day(pd.DataFrame(rows), pol, bankroll=100.0, peak=100.0)
    assert (out["track"] == TRACK).all()
    assert (out["final_fraction"] == 0.0).all() and (out["stake_amount"] == 0.0).all()


# ── 3. alternative route: the v9 Cloudbet bot (manual, DRY by default) ─────────────────────────────
class _Quote:
    def __init__(self, odds):
        self.odds, self.market_url = odds, "soccer.total_goals/x?total=2.5"


def _bot_run(tmp_path, monkeypatch, policy: dict | None = None) -> pd.DataFrame:
    from src.cloudbet import bot, candidates
    today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    bets = _board(today)
    led = pd.DataFrame({
        "source": "live", "generated_at": f"{today} 06:00", "match_date": today, "league": bets["league"],
        "home_team": bets["home_team"], "away_team": bets["away_team"], "side": bets["best_side"],
        "odds": [r.odds_under25 if r.best_side == "UNDER" else r.odds_over25 for r in bets.itertuples()],
        "edge_pct": (bets["best_edge"] * 100).round(2), "signal_tier": bets["signal_tier"],
        "model_type": bets["model_type"], "kickoff_utc": bets["kickoff_utc"], "result": ""})
    out = tmp_path / "out"
    out.mkdir()
    led.to_csv(out / "bets_ledger.csv", index=False)
    monkeypatch.setattr(config, "OUTPUT_DIR", out)
    monkeypatch.setattr(bot, "LEDGER", out / "cloudbet_bets.csv")
    monkeypatch.setattr(candidates, "prop_candidates", lambda **k: pd.DataFrame())
    monkeypatch.setattr(bot, "find_event", lambda events, h, a: ({"id": f"{h}-{a}", "markets": {"x": 1}}, ""))
    monkeypatch.setattr(bot, "price_for", lambda ev, m, s: (_Quote(2.30 if s == "UNDER" else 2.05), ""))
    if policy is not None:
        monkeypatch.setattr(BP, "load_policy", lambda path=None: copy.deepcopy(policy))
    return bot.run(mode="DRY", bankroll=1000.0, events_by_league={NF_LEAGUE: [{}]})


def test_cloudbet_bot_never_stakes_nf_under(tmp_path, monkeypatch, evidence):
    placed = _bot_run(tmp_path, monkeypatch)
    sides = set(placed["side"]) if len(placed) else set()
    assert "UNDER" not in sides, "the Cloudbet bot staked a PAPER-track NF Under"
    assert "OVER" in sides, "positive control: the NF OVER candidate must still be staked"


def test_cloudbet_bot_stakes_nf_under_when_track_is_real(tmp_path, monkeypatch, evidence):
    pol = _policy()
    pol["tracks"][TRACK] = {**pol["tracks"][TRACK], "state": "ACTIVE_LOW", "base_fraction": 0.005}
    placed = _bot_run(tmp_path, monkeypatch, pol)
    assert "UNDER" in set(placed["side"])


# ── 4. presentation: Telegram instant tip and the dashboard Tips table ─────────────────────────────
def _instant_tips(tmp_path, monkeypatch, policy: dict | None = None) -> dict:
    from telegram_bot import notifier as N
    out = tmp_path / "tg"
    out.mkdir()
    _board().to_csv(out / "bets.csv", index=False)
    sent: list[str] = []
    monkeypatch.setattr(N.app_config, "OUTPUT_DIR", out)
    monkeypatch.setattr(N, "_load_config", lambda: {"token": "t", "chat_id": "c"})
    monkeypatch.setattr(N, "_send", lambda token, chat, msg, *a, **k: sent.append(msg) or True)
    monkeypatch.setattr(N, "_load_notified", lambda: set())
    monkeypatch.setattr(N, "_save_notified", lambda s: None)
    monkeypatch.setattr(N, "instant_tip_allowed", lambda *a, **k: True)
    if policy is not None:
        monkeypatch.setattr(BP, "load_policy", lambda path=None: copy.deepcopy(policy))
    N.notify_new_snipers()
    tips = [m for m in sent if "2.5</b>" in m]
    assert len(tips) == 2, sent
    return {("UNDER" if "UNDER 2.5" in m else "OVER"): m for m in tips}


def test_telegram_instant_tip_labels_nf_under_as_paper(tmp_path, monkeypatch):
    msg = _instant_tips(tmp_path, monkeypatch)
    assert "PAPER" in msg["UNDER"] and "not staked" in msg["UNDER"]
    assert "PAPER" not in msg["OVER"], "positive control: a real-money track is not labelled paper"


def test_telegram_label_follows_the_policy_not_the_side(tmp_path, monkeypatch):
    pol = _policy()
    pol["tracks"][TRACK] = {**pol["tracks"][TRACK], "state": "ACTIVE_LOW", "base_fraction": 0.005}
    msg = _instant_tips(tmp_path, monkeypatch, pol)
    assert "PAPER" not in msg["UNDER"]


def test_dashboard_tips_table_shows_nf_under_as_paper():
    from dashboard_data import tips_view as TV
    assert TV.stake_label("SNIPER", "O/U 2.5 UNDER", "PAPER") == TV.STAKE_PAPER_TRACK
    assert TV.stake_label("SNIPER", "O/U 2.5 OVER", "ACTIVE_LOW") == TV.STAKE_OU     # positive control
    assert TV.stake_label("SNIPER", "O/U 2.5 OVER") == TV.STAKE_OU                     # old call unchanged
    b = pd.DataFrame({"_day": [DAY, DAY], "_local": pd.to_datetime([f"{DAY} 20:00"] * 2).tz_localize("UTC"),
                      "started": [False, False], "home_team": ["Under FC", "Over FC"],
                      "away_team": ["Low Town", "High Town"], "league": [NF_LEAGUE] * 2, "rating": [None, None],
                      "market": ["O/U 2.5 UNDER", "O/U 2.5 OVER"], "odds": [2.2, 1.95], "edge": [29.5, 26.7],
                      "model_prob": [0.75, 0.78], "drift": [None, None], "production_state": ["PAPER", "ACTIVE_LOW"],
                      "final_fraction": [0.0, 0.005], "tier": ["SNIPER", "SNIPER"], "model": ["new_format"] * 2,
                      "decision_state": [None, None]})
    t = TV.selections_table(b, research=False)
    assert t.loc[0, "Stake note"] == TV.STAKE_PAPER_TRACK and t.loc[0, "Stake %"] == 0.0
    assert t.loc[1, "Stake note"] == TV.STAKE_OU


def test_free_tier_never_offers_a_zero_stake_tip():
    import dashboard_access as DA
    plan = pd.DataFrame({"date": [DAY, DAY], "fixture": ["a", "b"], "league": [NF_LEAGUE] * 2,
                         "track": [TRACK, "new_format|OU25_OVER"], "market": ["OU25", "OU25"],
                         "side": ["UNDER", "OVER"], "signal_tier": ["SNIPER", "SNIPER"], "odds": [2.2, 1.95],
                         "final_fraction": [0.0, 0.005], "deployment_state": ["PAPER", "ACTIVE_LOW"]})
    tips, day, n = DA.free_tip_pick(plan, _policy(), DAY)
    assert set(tips["track"]) == {"new_format|OU25_OVER"} and n == 1     # Under out, Over in (control)


# ── 5. forward tracking still records NF Under predictions ─────────────────────────────────────────
def test_ledger_still_records_nf_under_tips(tmp_path, monkeypatch):
    from src import ledger as L
    monkeypatch.setattr(L, "LEDGER_FILE", tmp_path / "bets_ledger.csv")
    monkeypatch.setattr(L, "_opening_odds", lambda *a, **k: "")
    L.append_tips(_board())
    d = pd.read_csv(tmp_path / "bets_ledger.csv")
    u = d[(d["side"] == "UNDER") & (d["model_type"] == "new_format")]
    assert len(u) == 1 and u["signal_tier"].iloc[0] in ("QUALIFIED",)
    assert float(u["odds"].iloc[0]) == 2.20 and pd.notna(u["edge_pct"].iloc[0])


@pytest.fixture(autouse=True)
def _per_track_policy(monkeypatch):
    """These tests pin the TRACK mechanics: per-track staking (tests/policy_helpers). The live per-cell
    staking (owner 2026-11-09) is pinned in tests/test_cell_staking.py."""
    import sys
    from tests.policy_helpers import install_per_track
    install_per_track(monkeypatch, sys.modules[__name__])
