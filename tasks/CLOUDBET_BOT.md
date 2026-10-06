# TASK — Cloudbet auto-betting bot

Self-contained brief. Everything needed to continue is here; no other document is required.

**Scope: the Cloudbet bot only.** The rest of the Wowza estate is being worked on elsewhere —
do not change models, thresholds, tiers, notifications, league routing, or anything in
`src/betting.py`, `src/predict.py`, `config.py` or the workflows. The bot reads v9's output and
places bets; it never writes back into the prediction path.

---

## 0. Status

Built, pushed, **273 tests passing**, and **nothing can place a bet** — `CLOUDBET_MODE` defaults
to `DRY`.

| file | what it does |
|---|---|
| `src/cloudbet/client.py` | API client. `DRY` / `PLAY` / `REAL`. Deterministic `referenceId`. |
| `src/cloudbet/feed.py` | Prices a selection at Cloudbet; matches our fixture to their event. |
| `src/cloudbet/candidates.py` | Builds the candidate board from v9's ledgers (team + props). |
| `src/cloudbet/selection.py` | The owner's rules: tier, edge, ordering, caps. |
| `src/cloudbet/staking.py` | 4.5% of current bankroll, with a daily exposure cap. |
| `src/cloudbet/bot.py` | The runner: price → select → stake → place → record. |
| `scripts/cloudbet_probe.py` | READ-ONLY diagnostic. Needs a key. |
| `tests/test_cloudbet_*.py` | 39 tests. All run offline with no key. |

```bash
PYTHONPATH=. .venv/Scripts/python -m pytest tests/test_cloudbet_*.py -q     # expect 39 passed
PYTHONPATH=. .venv/Scripts/python -m src.cloudbet.bot                       # DRY, sends nothing
```

---

## 1. THE BLOCKER — do this first

**`src/cloudbet/feed.py::MARKET_MAP` is a guess, not a fact.** Cloudbet documents the shape
(`soccer.market/outcome?params`) but their wiki is unreachable over TLS from here and the exact
keys for total goals and both-teams-to-score are not pinned in anything verifiable. The current
map assumes:

```python
"ou25":   ("soccer.total_goals", "total")
"over15": ("soccer.total_goals", "total")
"over35": ("soccer.total_goals", "total")
"btts":   ("soccer.both_teams_to_score", None)
```

Until a real payload confirms these, the bot will price **nothing** and correctly refuse to bet.

### Task 1 — run the probe

Put `CLOUDBET_API_KEY=...` in the gitignored `.env` (never on a command line — it lands in shell
history). Then:

```bash
PYTHONPATH=. .venv/Scripts/python scripts/cloudbet_probe.py
PYTHONPATH=. .venv/Scripts/python scripts/cloudbet_probe.py --competition <key-from-above>
```

It places no bets and runs in DRY regardless of `CLOUDBET_MODE`.

**Acceptance:** the probe prints `MARKET KEYS ACTUALLY RETURNED`. Copy those into `MARKET_MAP`
and the line format into `MARKET_LINE` / `SIDE_MAP`. Add a test with a trimmed real payload in
`tests/test_cloudbet_feed.py` so the map can never silently drift again.

**Why it is built this way:** a classifier that matched on a *name* rather than a verified key
once swallowed seven unrelated bet types in this estate and reported a market probability of
0.254 where the truth was 0.63. `price_for()` therefore returns `(None, reason)` naming the keys
the payload actually contained. A map that matches nothing is recoverable; one that matches the
wrong thing is not.

### Task 2 — answer the question the probe exists to answer

The probe's second half prices our live board and prints, per selection:

```
PRICED   Velez v Platense   ours 2.00 -> cloudbet 2.10   edge shift +0.024
```

**Our edge is computed against Bet365 and a cross-book consensus. Cloudbet is in neither.** A 9%
edge at Bet365 may be 4% at Cloudbet, or nothing.

**Acceptance:** record, in this file, (a) what share of our board Cloudbet prices at all, and
(b) the median edge shift. If most selections survive, the bot is viable. If the shift is
systematically negative and large, say so — that is a finding, not a failure, and it would mean
the edge lives at books we are not betting.

---

## 2. Ordered tasks after the probe

### Task 2b — CLOUDBET AGAINST THE CONSENSUS (added 2026-10-06, owner's idea)

`src/cloudbet/consensus.py` is built and tested (12 tests). It answers a **different question**
from Task 2 and the two must never be summed:

```
edge_vs_model      p_model      - 1/cloudbet_odds    "our model disagrees with them"
edge_vs_consensus  p_consensus  - 1/cloudbet_odds    "they are cheap against the market"
```

The second does not depend on the model being right. Given that the de-vigged market beats every
model this estate has (log loss 0.67702 vs 0.68752 standard, 0.67899 vs 0.68714 new-format, on
identical rows), an edge from Cloudbet being soft has a better-founded mechanism than one from
the model disagreeing with the market: a smaller book pricing a thin second division more slowly.

**Leave-one-out is mandatory and the effect is measured, not assumed.** Simulated over 400
markets per book count, books scattered N(0.50, 0.03), Cloudbet soft by 6pp — the shift in the
consensus from including Cloudbet in its own comparison:

| other books | median | p95 |
|---|---|---|
| 3 | 1.11pp | 3.49pp |
| 6 (our typical) | 0.52pp | 1.80pp |
| 8 | 0.39pp | 1.34pp |

A soft-price edge worth betting is roughly 3–5pp, so self-comparison eats 10–17% of it at six
books and can eat all of it at three — and it bites hardest on thin markets, which is exactly
where softness is most likely. `soft_price_edge()` always excludes the book it is pricing.

Our existing coverage supports this today: **293 of 297 fixtures already have ≥3 two-sided
books, median 6.**

**Task:** record `edge_vs_model`, `edge_vs_consensus`, `consensus_books`, `consensus_dispersion`
and `agree` on every row of `output/cloudbet_bets.csv` via `consensus.both_edges()`.

**Do NOT filter on `agree` yet.** Which of model-only / market-only / both actually performs is
an empirical question, and there is no data to answer it. Record all three cases and let the
play period decide. A test asserts `selection.py` does not reference `agree`, so switching it on
is a deliberate act rather than a drift.

Adding Cloudbet to the consensus for *other* purposes (a better market anchor for research) is
fine and `consensus_probability(..., exclude=())` does it — but that number must never be the
one used to price Cloudbet.

### Task 3 — wire the event feed into the runner

`bot.run()` takes `events_by_league` and currently warns and drops everything when it is absent.
Build the fetch: competitions → our leagues → fixtures in the next N days → `{league: [events]}`.

* Cache the competition list; it changes rarely.
* One fixtures call per league per run, not per fixture.
* A league that cannot be mapped to a Cloudbet competition must be **logged by name and
  skipped**, never silently dropped.

**Acceptance:** `python -m src.cloudbet.bot` in DRY prints a real selection list with Cloudbet
prices attached, and `output/cloudbet_bets.csv` records them with `mode=DRY`.

### Task 4 — settle and grade

Nothing currently reads results back. Add a `--mode settle` pass that polls
`client.bet_status(reference_id)` for unsettled rows and writes `result`, `pnl` and the realised
price into `output/cloudbet_bets.csv`.

* Only bets with `accepted=True` are settled.
* `UNKNOWN_SEND_FAILED` rows must be **re-queried, not assumed lost** — the bet may have landed.

**Acceptance:** a settled row carries `result`, `pnl`, and the actual filled price; a test covers
the `UNKNOWN_SEND_FAILED` path.

### Task 5 — the workflow

A GitHub Actions workflow running the bot on a schedule. **Not before Tasks 1–4 are done.**

* `CLOUDBET_MODE` from a repo variable, default `DRY`.
* `CLOUDBET_API_KEY` from repo secrets.
* **Must stage `output/cloudbet_bets.csv`.** Capturing without committing is this estate's most
  repeated failure — every step green, nothing persisted. It has happened at least five times.
* Run it on `PLAY` for at least two weeks before anyone discusses `REAL`.

---

## 3. Rules that must not be broken

These are not preferences. Each exists because of a measured failure.

1. **Never accept a worse price.** `acceptPriceChange` stays `NONE` or `BETTER`. Best-price
   execution is worth **+1.99pp CI [+1.37, +2.67]** on this estate — larger than any model edge
   measured here. Accepting slippage hands it straight back.
2. **`referenceId` must stay deterministic** — UUIDv5 over (fixture, market_url, side, date). It
   deliberately excludes price, because a retry after a price refresh is the *same bet* and
   keying on price would let a one-tick move defeat the dedup and double the stake.
3. **A send timeout is `UNKNOWN`, never "rejected".** The bet may have landed. Re-placing the
   same selection is safe *because* the reference id is deterministic.
4. **One side per fixture, first tip wins.** 14 fixtures since the August cutoff carry tips on
   *both* sides, 1–6 days apart, because the model changes its mind as the price moves. A human
   ignores the second; a bot would place both and pay the spread twice.
5. **No Cloudbet price → no bet.** An edge against a price nobody will fill is not an edge.
6. **The daily ledger is the cap**, not an in-memory counter. A workflow that fires twice, or a
   process that dies mid-run, must not re-stake.
7. **Never log, echo or commit the API key** — not in an exception message either. A key in a
   traceback ends up in a public CI log.

---

## 4. Decisions needed from the owner

| # | decision | why it matters |
|---|---|---|
| 1 | **20 bets/day × 4.5% = 90% of bankroll at risk in one day.** The 30% daily exposure cap currently binds at ~6 full-size bets. | Both settings came from the owner and they conflict. Lower the per-bet %, lower the cap, or accept the concentration. |
| 2 | **Props are enabled at "any positive edge".** All six prop markets are `UNVERIFIED` in `registry/settlement_alignment.json` — we have never checked our label matches how the book settles. | Measured on one board: 1,296 prop rows, 33 priced, 4 +EV, all `goals`, two of them longshots at 13.0 and 19.5 with ~3.7× model/market probability ratios. Recorded, not filtered, per the owner's instruction. |
| 3 | **The triple (top-3 as an acca) is not built.** | Measured Sept 2026: leg EV is below 1.0 in every league, so folding loses faster. Three legs at 0.95 is 0.86. Needs an explicit decision before building. |
| 4 | **`PLAY` → `REAL`.** | Should not happen until the play period has run and the Argentina BTTS preregistered window (4 of 150 as of 2026-10-06) has data. |

---

## 5. Context worth having

* **Bankroll $2,000 USDT. Staking 4.5% of *current* bankroll** (owner's figure). The measured
  growth-optimal fraction on this estate's own results is **2.2%**, and log-growth turns negative
  above **4.5%** — so 4.5% is exactly the break-even line, where the bankroll neither compounds
  nor decays. Full numbers in `output/staking_study.json`.
* **Selection rules** (owner's): SNIPER or MARKSMAN, edge strictly above 5%, sorted descending by
  edge, hard cap 20/day, markets `ou25` / `btts` / `over15` / `over35`, singles. Props use a
  separate bar: any positive edge.
* **The bot reads the LEDGER, not `bets.csv`.** That was a real bug: `bets.csv` is the current
  7-day board, while the Telegram digest reports the ledger filtered to today. On 2026-10-05 the
  board held 76 tips dated Oct 7–11 and **zero** for that day, while the digest sent 3, all OVER.
  Both correct, different days — but a bot on the board would place bets the owner has never been
  shown. `days_ahead=0` keeps the bot and the digest in agreement.
* **USA MLS is paper-only since 2026-10-06** and no longer appears in candidates.

---

## 6. Working offline

Every Cloudbet test runs with no API key and no network — they build payloads by hand. If a new
test needs the network, the design is wrong: fetch in `bot.py` / the probe, keep `feed.py` pure.

```bash
PYTHONPATH=. .venv/Scripts/python -m pytest tests/test_cloudbet_feed.py -q      # pricing + matching
PYTHONPATH=. .venv/Scripts/python -m pytest tests/test_cloudbet_selection.py -q # the rules
PYTHONPATH=. .venv/Scripts/python -m pytest tests/test_cloudbet_bot.py -q       # the runner
```

Test suite runtime for the whole bot is ~2 seconds. If it jumps to a minute, something is
reaching for the network — that happened once already and the fix is in `price_candidates`:
only fetch when the listing did not already carry markets.
