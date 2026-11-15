# SF-PAPER-1: sharp-agreement paper track, live capture, and the tracker fix

**Owner approval:** 2026-11-15, "Paper track + capture".
**Basis:** research sharp-filter/1, branch `research/sharp-filter`, `reports/SHARP_FILTER_RESEARCH.md`, protocol sha256 `aeb0ab95...`.
**What this is:** a PAPER track only. Nothing here stakes, places an exec order, or publishes a tip.

## 1. The protocol (registered before any pick)

**Protocol file:** `config/sharp_paper/protocol.json`. Its sha256 is in `config/sharp_paper/protocol.sha256`, and `tests/test_sharp_paper.py` fails if the two differ.

**Universe.** Every live model tip of an approved O/U MODEL cell of TP-2026-11-15-bade2c0b. That covers 42 cells across over25_over, over25_under, over15_over and over35_over.
- **Start:** match date on or after 2026-11-21.
- **Tier:** `QUALIFIED`.
- **Dedup:** one pick per fixture and market-side.
- **Excluded:** BTTS (it has no O/U sharp signal) and the league-wide cells.

**Signal.** Read at T-6 h, as in the research's W72_6 window. It is the move in the fair Over 2.5 probability from the T72 snapshot to the T6 snapshot.
- **CONS (primary):** the median over at least 3 non-Pinnacle books.
- **PIN and BOTH (secondary).**
- **Thresholds:** frozen per league. Each is the 75th percentile of the move size over every fixture of that league in the licensed B4 history, 2023-06 to 2026-06. A league with fewer than 100 measured fixtures gets NO_DATA. The thresholds are statistics, not prices.

**Bet.**
- **Price:** Pinnacle's exact-line price for the pick's own side in the T6 snapshot. Without it the pick is NO_PRICE, and no other book stands in.
- **Never the close:** a CLOSE snapshot is never used as the bet price.
- **Stake and settlement:** flat 1 unit, settled from the v9 ledger's own WIN or LOSS.
- **CLV:** measured against the price in the CLOSE snapshot.

**Review.** Each cell is reviewed at 150 CONS AGREE bets, or at the Clean Record milestone (2027-01-21), whichever comes first.
- **Test:** AGREE ROI > 0 and AGREE minus NONE > 0, with Holm over the reviewed cells.
- **PROVEN** = at least 150 bets, Holm p < 0.025 and CI > 0.
- **Decision:** the owner's.

## 2. Live capture

**Script:** `scripts/sharp_paper_capture.py`. It runs from the PRIVATE repo wowza-exec (`.github/workflows/sharp_paper_capture.yml`) at a pinned wowza-betting commit, and writes only to exec's orphan branch `sharp-paper`.

**What each pass does:**
- **Picks.** Reads the live tips from wowza-betting `main`'s ledgers.
- **Matching.** Lists each league's upcoming provider events, which is a free call. Each fixture is matched within its own league, inside its day window, with both clubs proven, one to one.
- **Snapshots.** Buys one event-odds snapshot per due slot: `totals,alternate_totals`, region eu, every book. It stores every book's exact 1.5 / 2.5 / 3.5 pair.

**Slots:**

| Slot | When it is due | Use |
|---|---|---|
| T72 | [60, 72] h before kickoff | open |
| T6 | [4, 6] h | signal and bet price |
| CLOSE | 2-15 min | CLV only |

**Cost:** 2 credits per snapshot, so at most 6 per tipped fixture.

**Credit rules:**
- **Floor:** no paid call when x-requests-remaining minus its cost would fall below **260,000**. The script refuses any lower floor.
- **Caps:** per run (400 by default), per UTC day (1,500) and per month (30,000), journaled in `ledger/usage.json`.
- **Logging:** each run prints one `[oddsapi_usage]` line for the quota monitor.

**Expected credits per day:**
- **Estimate basis:** v9's live tips in the protocol's cells from 2026-09-01 to 2026-11-08, counted with the test-mode universe (all tiers), so this is an upper bound.
- **Volume:** 597 distinct fixtures over 69 days, a mean of 8.7 tipped fixtures per day (p90 24, max 32).
- **Credits:** about **52 per day on average** (p90 144, max 192), or about **1,600 per month**.

**Floor conflict:** the history collector stops at the same 260,000 floor, and live production burns credits below it. Once the history spend has drained the account to about 260k, this capture can make no call until the plan resets on 2026-12-01. **Owner decision needed:** a separate, lower floor for this capture, or a reserved history budget.

## 3. Live sharp tracker fix (`src/sharp_tracker.py`)

**What changed:** the global constants 5% / 10% decimal drift and 3% steam are gone.

**How a move is measured and judged:**
- **The move:** a side's fair-probability change from the snapshot closest to 72 h ([60, 84] h) to the current snapshot.
- **The comparison:** it is judged against the same league's moves (same market, side for 1X2, and lead-time bucket) on fixtures dated strictly earlier that have already kicked off. These are learned from the tracker's own history every run.

**Labels:**
- **SHARP:** above the league's 75th percentile.
- **STRONG:** above the 90th percentile.
- **FADING:** against the side, above the 75th percentile.
- **STEAM_:** the last window's move is above the league's 90th percentile of single-window moves, with at least 4 books.

**Too little history:** a league with fewer than 100 earlier fixtures gets NO signal. It is counted in the log as `no_league_history`, and there is never a global fallback. The label names and the `sharp_tips.csv` columns are kept, with new columns added.

**Replay on the committed history (2026-11-01 18:00):**
- 36 signals, plus 30 rows with no signal because of too little league history.
- Across the 380 league × market × side × bucket groups, 188 have at least 100 earlier fixtures today.

## 4. For the go-live runbook (switch on at go-live)

1. Merge `feat/sharp-paper-capture` into data-truth-100 with the rest of go-live. Nothing runs from it in wowza-betting: no wowza-betting workflow references the paper track, and a test enforces this.
2. Pin the code. Run `gh variable set SHARP_PAPER_WOWZA_REF --body <merged go-live sha> --repo NevixAA/wowza-exec`.
3. Switch on. Run `gh variable set SHARP_PAPER_CAPTURE_ENABLED --body true --repo NevixAA/wowza-exec`. The flag is **SHARP_PAPER_CAPTURE_ENABLED**. To switch off, run `gh variable delete SHARP_PAPER_CAPTURE_ENABLED --repo NevixAA/wowza-exec`.
4. At +24 h, check the latest exec run log:
   - `universe_picks > 0`, which shows the go-live ledgers write `QUALIFIED`;
   - matched fixtures;
   - credits_today in line with section 2.
5. The tracker fix needs no switch. It takes effect with the merge, through the existing sharp_tracker workflow.
