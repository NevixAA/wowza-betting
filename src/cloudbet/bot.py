"""The runner: select, price at Cloudbet, size, place, record.

    python -m src.cloudbet.bot            # DRY by default — builds requests, sends nothing
    CLOUDBET_MODE=PLAY python -m src.cloudbet.bot

ORDER OF OPERATIONS, and each step can only ever REMOVE candidates:

    candidates (ledger, today)  ->  price at Cloudbet  ->  selection rules  ->  stake  ->  place

PRICING COMES BEFORE SELECTION, not after. The edge that decides a bet must be the edge against
the price we can actually take; selecting on our own book's edge and then looking up Cloudbet
would bet a 9%-at-Bet365 selection that is 4% at Cloudbet. The selector's `cloudbet_price`
column is filled here, and its `no_cloudbet_price` filter is what drops anything unpriced.

THE DAILY LEDGER IS THE SAFETY MECHANISM, not the cap in the rules. `output/cloudbet_bets.csv`
is read before every run and holds every reference id and fixture already placed. The 20/day cap
and the one-side-per-fixture rule are both enforced against it, so a workflow that fires twice,
or a process that dies mid-run and restarts, cannot re-stake. Cloudbet's own referenceId dedup
is the second layer; neither is trusted alone.

NO STAKE WITHOUT REAL INPUTS (Data Truth 2026-11-01). The bankroll is the live Cloudbet balance or
an explicit CLOUDBET_BANKROLL - never a typed default; without one the run refuses to stake. A
candidate tiered on a GLOBAL threshold (no per-league evidence; every prop) is never staked.

NO STAKE ON A PAPER TRACK (Phase 2, 2026-11-03). A candidate whose config/bankroll_policy.json track
is not a real-money track (PAPER: new_format|OU25_UNDER since 2026-11-02 and every standard track;
BLOCKED; undefined) is never staked here either - see _drop_without_real_track.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pandas as pd

import config
from src.cloudbet.candidates import build_candidates
from src.cloudbet.client import DRY, CloudbetClient, Selection
from src.cloudbet.feed import find_event, price_for
from src.cloudbet.selection import SelectionRules, select
from src.cloudbet.staking import StakingRules, describe, stake_for

log = logging.getLogger(__name__)

LEDGER = config.OUTPUT_DIR / "cloudbet_bets.csv"

LEDGER_COLUMNS = [
    "placed_at", "mode", "reference_id", "fixture_id", "match_key", "league", "market", "side",
    "line", "selection_label", "kickoff_utc", "model_type", "bet_kind", "settlement_status",
    "our_odds", "cloudbet_odds", "edge_at_our_price", "edge_at_cloudbet_price",
    "stake", "currency", "status", "filled_price", "accepted", "detail",
]


def _load_ledger() -> pd.DataFrame:
    if not LEDGER.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    d = pd.read_csv(LEDGER)
    for c in LEDGER_COLUMNS:
        if c not in d.columns:
            d[c] = pd.NA
    return d


def _today_utc() -> str:
    return pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")


def _placed_today(led: pd.DataFrame) -> pd.DataFrame:
    if led.empty:
        return led
    day = pd.to_datetime(led["placed_at"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
    # Only ACCEPTED bets count toward the cap. A rejection consumed no money and no slot; an
    # UNKNOWN_SEND_FAILED does NOT count either, because re-placing it is the safe response to a
    # timeout and the deterministic referenceId is what prevents a double stake.
    return led[(day == _today_utc()) & (led["accepted"].astype(str).str.lower() == "true")]


def _append(rows: list[dict]) -> None:
    if not rows:
        return
    d = pd.DataFrame(rows)
    for c in LEDGER_COLUMNS:
        if c not in d.columns:
            d[c] = pd.NA
    d = d[LEDGER_COLUMNS]
    out = pd.concat([_load_ledger(), d], ignore_index=True) if LEDGER.exists() else d
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(LEDGER, index=False)


def price_candidates(cands: pd.DataFrame, client: CloudbetClient,
                     events_by_league: dict) -> pd.DataFrame:
    """Attach Cloudbet's executable price and RE-COMPUTE the edge against it."""
    px, urls, evids, edges, reasons = [], [], [], [], []
    for _, r in cands.iterrows():
        events = events_by_league.get(str(r.get("league"))) or []
        ev, why = find_event(events, str(r.get("home_team")), str(r.get("away_team")))
        if ev is None:
            px.append(None); urls.append(None); evids.append(None); edges.append(None)
            reasons.append(f"no event: {why}"); continue
        # Only fetch when the listing did not already carry markets. The first version called
        # event_odds for EVERY candidate, which is one HTTP round trip per selection against a
        # rate-limited endpoint — and in tests it reached for the network at all.
        full = ev if (ev.get("markets") or {}) else (client.event_odds(str(ev.get("id"))) or ev)
        q, why = price_for(full, str(r["market"]), str(r["side"]))
        if q is None:
            px.append(None); urls.append(None); evids.append(str(ev.get("id")))
            edges.append(None); reasons.append(f"no price: {why}"); continue
        px.append(q.odds); urls.append(q.market_url); evids.append(str(ev.get("id")))
        # EDGE AT THEIR PRICE. Our stored edge was computed against our own book, so the
        # difference between the two implied probabilities is how much of it their price eats.
        ours = pd.to_numeric(pd.Series([r.get("our_odds")]), errors="coerce").iloc[0]
        if not (pd.notna(ours) and float(ours) > 1):
            # Without our own price the stored edge cannot be moved onto Cloudbet's price. It used
            # to be carried over unchanged (shift 0.0), i.e. an edge against a price we never saw.
            edges.append(None); reasons.append("no our_odds: edge at Cloudbet price unknown")
            continue
        edges.append(float(r["edge"]) + (1.0 / float(ours) - 1.0 / q.odds))
        reasons.append("")
    out = cands.copy()
    out["cloudbet_price"] = px
    out["cloudbet_market_url"] = urls
    out["cloudbet_event_id"] = evids
    out["edge_at_our_price"] = out["edge"]
    # The selector sorts and filters on `edge`; from here on it IS the Cloudbet edge.
    out["edge"] = [e if e is not None else float("nan") for e in edges]
    out["price_reason"] = reasons
    return out


def resolve_bankroll(client: CloudbetClient, bankroll: float | None = None) -> tuple[float | None, str]:
    """(bankroll, source), or (None, why) when there is no real figure to size stakes on.

    Order: an explicit argument, the LIVE Cloudbet balance, then the CLOUDBET_BANKROLL environment
    variable. There is no default: a typed 2000 used to size every stake when none of these existed,
    so a bot pointed at an empty or unknown account staked as if it held 2000.
    """
    if bankroll is not None:
        return float(bankroll), "argument"
    live = client.balance()
    if live is not None:
        return live, f"live Cloudbet balance ({client.currency})"
    env = os.getenv("CLOUDBET_BANKROLL", "").strip()
    if not env:
        return None, ("no live Cloudbet balance (no key, or the account API did not answer) and "
                      "CLOUDBET_BANKROLL is not set")
    try:
        v = float(env)
    except ValueError:
        return None, f"CLOUDBET_BANKROLL={env!r} is not a number"
    if not (v == v):
        return None, "CLOUDBET_BANKROLL is NaN"
    return v, "CLOUDBET_BANKROLL"


def _drop_without_league_evidence(cands: pd.DataFrame) -> pd.DataFrame:
    """Only bets tiered on their league's OWN threshold may be staked (owner 2026-11-01).

    Team markets: betting.league_threshold_evidence (approved optimiser value or calibrated
    per-league threshold). Props have no per-league threshold anywhere in the estate - one global
    "positive EV" bar - so no prop has league evidence and none is staked. Every drop is logged."""
    if cands.empty:
        return cands
    from src.betting import league_threshold_evidence
    cache: dict = {}

    def _ok(r) -> bool:
        if str(r.get("bet_kind")) == "prop":
            return False
        k = (str(r.get("league")), str(r.get("market")))
        if k not in cache:
            cache[k] = league_threshold_evidence(*k)
        return cache[k]

    keep = cands.apply(_ok, axis=1).astype(bool)
    if (~keep).any():
        cells = sorted({(str(lg), str(m)) for lg, m in
                        zip(cands.loc[~keep, "league"], cands.loc[~keep, "market"])})
        log.info(f"no stake: {int((~keep).sum())} candidate(s) have no per-league threshold "
                 f"evidence (global bar only): {cells[:20]}")
    return cands[keep]


#: candidate market (ledger spelling) -> bankroll-policy market (src/stake_plan.SIDE_MARKET + OU25)
_POLICY_MARKET = {"ou25": "OU25", "btts": "BTTS", "over15": "OVER15", "over35": "OVER35"}


def _drop_without_real_track(cands: pd.DataFrame) -> pd.DataFrame:
    """Only tips on a REAL-money bankroll-policy track may be staked (Phase 2, 2026-11-03).

    The bot sized every SNIPER/MARKSMAN candidate at its own flat fraction and never read
    config/bankroll_policy.json, so a track the owner set to PAPER (new_format|OU25_UNDER since
    2026-11-02; every standard track) was still staked here. The track key is THE policy key
    (src/bankroll_policy.track_key: model type, market and, for O/U 2.5, side); a track that is not
    in REAL_STATES or has base_fraction 0 is dropped, and so is a market or model type the policy
    does not define. An unreadable policy drops everything (fail closed). Every drop is logged."""
    if cands.empty:
        return cands
    from src.bankroll_policy import REAL_STATES, load_policy, positive_fraction, track_key
    try:
        tracks = load_policy().get("tracks") or {}
    except Exception as e:                                            # noqa: BLE001 - fail closed
        log.error(f"no stake: bankroll policy unreadable ({type(e).__name__}) - every candidate dropped")
        return cands.iloc[0:0]

    def _real(r) -> bool:
        mk = _POLICY_MARKET.get(str(r.get("market")).lower())
        if mk is None:
            return False
        t = tracks.get(track_key(r.get("model_type"), mk, r.get("side"))) or {}
        return t.get("state") in REAL_STATES and positive_fraction(t)

    keep = cands.apply(_real, axis=1).astype(bool)
    if (~keep).any():
        cells = sorted({f"{mt}|{m}|{s}" for mt, m, s in zip(cands.loc[~keep, "model_type"],
                                                          cands.loc[~keep, "market"], cands.loc[~keep, "side"])})
        log.info(f"no stake: {int((~keep).sum())} candidate(s) on a PAPER / BLOCKED / unknown policy "
                 f"track: {cells[:20]}")
    return cands[keep]


def run(mode: str | None = None, bankroll: float | None = None,
        days_ahead: int = 0, rules: SelectionRules | None = None,
        staking: StakingRules | None = None,
        events_by_league: dict | None = None) -> pd.DataFrame:
    client = CloudbetClient(mode=mode)
    rules = rules or SelectionRules()
    staking = staking or StakingRules()
    bankroll, source = resolve_bankroll(client, bankroll)
    if bankroll is None:
        log.error(f"REFUSING TO STAKE: {source}. Set CLOUDBET_BANKROLL or provide a key whose "
                  f"account balance can be read.")
        return pd.DataFrame()

    led = _load_ledger()
    today = _placed_today(led)
    log.info(f"mode={client.mode} | bankroll from {source} | {describe(bankroll, staking)}")
    log.info(f"already placed today: {len(today)} of {rules.max_bets_per_day}")

    cands = build_candidates(days_ahead=days_ahead)
    if cands.empty:
        log.info("no candidates on the board")
        return pd.DataFrame()

    if events_by_league is None:
        log.warning("no Cloudbet event feed supplied — every candidate will be dropped as "
                    "unpriced, which is correct: we do not bet into a price we cannot see")
        events_by_league = {}
    priced = _drop_without_real_track(
        _drop_without_league_evidence(price_candidates(cands, client, events_by_league)))

    sel, rej = select(
        priced, rules,
        already_placed=set(led["reference_id"].dropna().astype(str)),
        placed_fixtures=set(today["fixture_id"].dropna().astype(str)))
    log.info(f"selected {len(sel)}; rejected: {rej}")

    # The cap counts what is ALREADY placed today, not just this run's selections.
    room = max(0, rules.max_bets_per_day - len(today))
    if len(sel) > room:
        log.info(f"daily cap: taking the best {room} of {len(sel)}")
        sel = sel.head(room)

    stakes = pd.to_numeric(today["stake"], errors="coerce")
    if stakes.isna().any():
        # An unreadable stake on an ACCEPTED bet is money at risk we cannot count. Reading it as 0
        # would let the daily exposure cap pass bets it should stop: fail closed instead.
        log.error(f"REFUSING TO STAKE: {int(stakes.isna().sum())} of today's accepted bet(s) in "
                  f"{LEDGER.name} have an unreadable stake, so today's exposure is unknown")
        return pd.DataFrame()
    staked_today = float(stakes.sum())
    rows = []
    for _, r in sel.iterrows():
        stake, why = stake_for(bankroll, staking, staked_today=float(staked_today),
                               is_prop=str(r.get("bet_kind")) == "prop")
        if stake <= 0:
            log.info(f"  SKIP {r['selection_label'][:40]}: {why}")
            continue
        s = Selection(
            fixture_id=str(r["fixture_id"]), event_id=str(r["cloudbet_event_id"]),
            market_url=str(r["cloudbet_market_url"]), league=str(r.get("league")),
            home_team=str(r.get("home_team")), away_team=str(r.get("away_team")),
            kickoff_utc=str(r.get("kickoff_utc")), market=str(r["market"]),
            side=str(r["side"]), price=float(r["cloudbet_price"]), edge=float(r["edge"]),
            tier=str(r.get("signal_tier")), model_type=str(r.get("model_type")))
        res = client.place(s, stake)
        # EXPOSURE ACCUMULATES IN DRY TOO. It only counted accepted bets, and nothing is
        # accepted in DRY — so a dry run reported stakes that blew straight through the daily
        # exposure cap and looked fine. A dry run that does not honour the caps is not a
        # rehearsal. A genuine REJECTION still costs nothing and still consumes nothing.
        if res.accepted or client.mode == DRY:
            staked_today += stake
        rows.append({
            "placed_at": pd.Timestamp.now(tz="UTC").isoformat(), "mode": client.mode,
            "reference_id": res.reference_id, "fixture_id": s.fixture_id,
            "match_key": r.get("match_key"), "league": s.league, "market": s.market,
            "side": s.side, "line": r.get("line"), "selection_label": r.get("selection_label"),
            "kickoff_utc": s.kickoff_utc, "model_type": s.model_type,
            "bet_kind": r.get("bet_kind"), "settlement_status": r.get("settlement_status"),
            "our_odds": r.get("our_odds"), "cloudbet_odds": s.price,
            "edge_at_our_price": r.get("edge_at_our_price"), "edge_at_cloudbet_price": s.edge,
            "stake": stake, "currency": res.currency, "status": res.status,
            "filled_price": res.filled_price, "accepted": res.accepted,
            "detail": res.detail})
        log.info(f"  {res.status:<22} {r['selection_label'][:38]:<40} @{s.price} "
                 f"stake {stake:.2f} {res.currency}")

    # Written even in DRY, so a dry run leaves an auditable record of what it WOULD have done.
    _append(rows)
    return pd.DataFrame(rows)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["DRY", "PLAY", "REAL"])
    ap.add_argument("--bankroll", type=float)
    ap.add_argument("--days-ahead", type=int, default=0)
    a = ap.parse_args()
    out = run(mode=a.mode, bankroll=a.bankroll, days_ahead=a.days_ahead)
    print(f"\n{len(out)} bet(s) recorded -> {LEDGER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
