"""Cloudbet trading API client — feed and bet placement.

    from src.cloudbet.client import CloudbetClient
    c = CloudbetClient()                 # mode comes from CLOUDBET_MODE
    c.place(sel, stake_usd=40.0)

THREE MODES, and the default is the safe one:

    DRY     build the request, log it, send nothing            <- default
    PLAY    send it for real against PLAY_EUR (Cloudbet's own play currency)
    REAL    send it against USDT

PLAY is not a simulation we wrote — it is Cloudbet's own play-money currency going through the
same endpoint, the same market ids, the same price checks and the same rejections. So the only
difference between PLAY and REAL is the `currency` field. That is the whole reason to run PLAY
first: a bug in our market mapping or our stake rounding shows up identically, for nothing.

AUTH. `CLOUDBET_API_KEY` from the environment only. It is never logged, never written to an
output file, and never included in an exception message — a key in a traceback ends up in a CI
log, which is public on this repo.

IDEMPOTENCY IS THE WHOLE SAFETY STORY. Cloudbet dedups on `referenceId`, so a retry after a
timeout cannot place a second bet IF the id is stable. Generating a fresh uuid on retry would
turn every network blip into a double stake, which is the one bug in a betting bot that costs
real money silently. The id is derived DETERMINISTICALLY from the selection, so the same
selection always produces the same id, even across process restarts.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field

import requests

log = logging.getLogger(__name__)

BASE = os.getenv("CLOUDBET_BASE", "https://sports-api.cloudbet.com/pub/v2")
TRADING = os.getenv("CLOUDBET_TRADING_BASE", "https://sports-api.cloudbet.com/pub/v3")

DRY, PLAY, REAL = "DRY", "PLAY", "REAL"
#: Default is DRY. Promoting to PLAY or REAL is an explicit, deliberate environment change.
MODE = os.getenv("CLOUDBET_MODE", DRY).upper()
#: Cloudbet's own play-money currency. Same endpoint, same everything else.
PLAY_CURRENCY = "PLAY_EUR"
REAL_CURRENCY = os.getenv("CLOUDBET_CURRENCY", "USDT")

#: Namespace for deterministic reference ids. Changing this would make every previously placed
#: bet look new to the dedup, so it is a constant and not a config value.
_NS = uuid.UUID("6f1a9b2c-0d3e-4f5a-8b6c-7d8e9f0a1b2c")


@dataclass(frozen=True)
class Selection:
    """One thing we want to bet. Carries everything needed to place it AND to audit it later."""
    fixture_id: str
    event_id: str                 # Cloudbet's event id
    market_url: str               # e.g. "soccer.total_goals/over?total=2.5"
    league: str
    home_team: str
    away_team: str
    kickoff_utc: str
    market: str                   # our name: ou25 / btts / over15 / over35
    side: str
    price: float                  # the price WE saw and decided on
    edge: float
    tier: str
    model_type: str

    @property
    def reference_id(self) -> str:
        """Deterministic UUIDv5 over the identity of the bet.

        Deliberately excludes price and edge: a retry after a price refresh is still the SAME
        bet, and including price would let a one-tick move defeat the dedup and double the stake.
        """
        key = "|".join([self.fixture_id, self.market_url, self.side, str(self.kickoff_utc)[:10]])
        return str(uuid.uuid5(_NS, key))


@dataclass
class PlaceResult:
    accepted: bool
    status: str
    reference_id: str
    requested_price: float
    filled_price: float | None = None
    stake: float = 0.0
    currency: str = ""
    mode: str = ""
    detail: str = ""
    raw: dict = field(default_factory=dict)


class CloudbetClient:
    def __init__(self, mode: str | None = None, timeout: int = 20):
        self.mode = (mode or MODE).upper()
        self.timeout = timeout
        self._key = os.getenv("CLOUDBET_API_KEY", "").strip()
        if self.mode in (PLAY, REAL) and not self._key:
            # Deliberately does not echo anything about the key's value or length.
            raise RuntimeError("CLOUDBET_API_KEY is not set in the environment; "
                               "cannot run in PLAY or REAL mode")

    # ── plumbing ──────────────────────────────────────────────────────────────────────────────
    @property
    def _headers(self) -> dict:
        return {"X-API-Key": self._key, "Content-Type": "application/json"}

    def _get(self, url: str, params: dict | None = None) -> dict:
        for attempt in range(3):
            try:
                r = requests.get(url, headers=self._headers, params=params, timeout=self.timeout)
                if r.status_code == 429:
                    time.sleep(2 ** attempt); continue
                if r.status_code != 200:
                    log.warning(f"cloudbet GET {url.rsplit('/', 1)[-1]} -> {r.status_code}")
                    return {}
                return r.json()
            except Exception as e:                                    # noqa: BLE001
                # The message is logged WITHOUT the request object, because the headers on it
                # carry the key.
                if attempt == 2:
                    log.warning(f"cloudbet GET failed: {type(e).__name__}")
                    return {}
                time.sleep(1 + attempt)
        return {}

    # ── feed ──────────────────────────────────────────────────────────────────────────────────
    def competitions(self, sport: str = "soccer") -> dict:
        return self._get(f"{BASE}/sports/{sport}")

    def fixtures(self, competition_key: str, from_iso: str | None = None,
                 to_iso: str | None = None) -> dict:
        params = {k: v for k, v in (("from", from_iso), ("to", to_iso)) if v}
        return self._get(f"{BASE}/competitions/{competition_key}", params)

    def event_odds(self, event_id: str) -> dict:
        """Current prices for one event — this is the EXECUTABLE price.

        Why this matters more than it sounds: today the estate computes edge against Bet365 and
        a cross-book consensus, and Cloudbet is in neither. An edge against a price you cannot
        take is not an edge. Everything placed through this client must be priced from here.
        """
        return self._get(f"{BASE}/events/{event_id}")

    # ── betting ───────────────────────────────────────────────────────────────────────────────
    def place(self, sel: Selection, stake: float,
              accept_price_change: str = "NONE") -> PlaceResult:
        """Place one single. Returns a result; never raises on a rejection.

        `accept_price_change` defaults to NONE — fill at our price or not at all. "BETTER" is
        also safe (fills only if the price improved). It is never set to accept a WORSE price:
        the measured execution advantage on this estate is +1.99pp, and silently accepting
        slippage is the fastest way to give that back.
        """
        currency = PLAY_CURRENCY if self.mode == PLAY else REAL_CURRENCY
        body = {
            "referenceId": sel.reference_id,
            "eventId": sel.event_id,
            "marketUrl": sel.market_url,
            "price": str(sel.price),
            "stake": str(round(float(stake), 2)),
            "currency": currency,
            "acceptPriceChange": accept_price_change,
        }

        if self.mode == DRY:
            log.info(f"[DRY] would place {sel.market}/{sel.side} {sel.home_team} v "
                     f"{sel.away_team} @ {sel.price} stake {stake:.2f} ({sel.edge:+.1%} edge)")
            return PlaceResult(accepted=False, status="DRY_RUN",
                               reference_id=sel.reference_id, requested_price=sel.price,
                               stake=stake, currency=currency, mode=self.mode,
                               detail="dry run — nothing sent", raw={"body": body})

        try:
            r = requests.post(f"{TRADING}/bets/place/straight", headers=self._headers,
                              data=json.dumps(body), timeout=self.timeout)
            payload = r.json() if r.content else {}
        except Exception as e:                                        # noqa: BLE001
            # A TIMEOUT IS NOT A REJECTION. The bet may well have landed. Because referenceId is
            # deterministic, the safe move is to report unknown and let the caller re-place the
            # SAME selection — Cloudbet will dedup it rather than stake twice.
            log.error(f"cloudbet place failed ({type(e).__name__}) — status UNKNOWN, "
                      f"ref {sel.reference_id}; re-placing the same selection is safe")
            return PlaceResult(accepted=False, status="UNKNOWN_SEND_FAILED",
                               reference_id=sel.reference_id, requested_price=sel.price,
                               stake=stake, currency=currency, mode=self.mode,
                               detail=type(e).__name__)

        status = str(payload.get("status", "")).upper() or f"HTTP_{r.status_code}"
        accepted = status in ("ACCEPTED", "PENDING_ACCEPTANCE", "WON", "LOSS", "PLACED")
        filled = payload.get("price")
        return PlaceResult(
            accepted=accepted, status=status, reference_id=sel.reference_id,
            requested_price=sel.price,
            filled_price=float(filled) if filled not in (None, "") else None,
            stake=stake, currency=currency, mode=self.mode,
            detail=str(payload.get("message", ""))[:200], raw=payload)

    def bet_status(self, reference_id: str) -> dict:
        return self._get(f"{TRADING}/bets/{reference_id}/status")


def fingerprint(obj) -> str:
    """Short stable hash, for logging what a run saw without dumping the payload."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]
