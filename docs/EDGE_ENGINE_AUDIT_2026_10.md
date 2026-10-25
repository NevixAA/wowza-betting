# Edge engine audit and market_edge_v2 shadow (2026-10-25)

Scope: owner brief 2026-10-25 ("complete the incomplete migration"). **SHADOW ONLY.** The legacy
engine stays the production/reference selection path. No served tip, tier, threshold, stake,
Telegram message or ledger row changes. Before/after proof is in section 6.

## 1. What production actually does (verified)

| Path | Formula | Where |
|---|---|---|
| O/U 2.5 | `edge = p_model - 1/odds` per side, tier by `_base_tier(edge)` | `src/betting.py:246-249`, `:140-195`, called twice from `src/predict.py:437,444` |
| BTTS / O1.5 / O3.5 | `fair = (1/odds)/1.08`, `edge = p_model - fair`, fixed 0.10/0.08/0.04 bars | `pipeline.py:93,125-126,143-176` |
| board O/U price | first book that quotes each side (`not ov25` / `not un25`) | `src/predict.py:139-140` |
| board BTTS price | `median(yes)` and `median(no)` taken **independently** across books | `src/btts_odds.py:201-202` |

Notes that matter for reading the legacy numbers:

* `p - 1/odds > 0` is exactly the sign of point EV at that price (`EV = odds*(p - 1/odds)`), so the
  vig maths is not the defect. The defect is that `p_model` is treated as the truth: no market
  anchor, no uncertainty, and the tier is the SIZE of the model's disagreement with one book.
  On the committed board (179 fixtures) the legacy OU25 `best_edge` and the new engine's point EV
  correlate at **-0.06** over the 147 OU25 rows with a valid two-sided consensus.
* The BTTS board pair is a median-yes from one set of books and a median-no from another; it is
  **not** a two-sided market and must not be de-vigged as one. The new engine never uses it as a
  market (only as the execution price).
* The 1.08 side-market margin is close to the measured BTTS overround (per-book median 0.0795),
  so for BTTS the legacy fair value differs from a proportional de-vig of the board pair by only
  +0.15pp mean. The problem there is the same as O/U: the model is the decision.

## 2. Audit of `src/edge_engine.py` (the dormant candidate)

Treated as a candidate, not as correct. Only `power_devig` / `proportional_devig` are used anywhere
(`src/clv_capture.py:24,38`); `classify` has never run. It is left untouched (clv_capture depends
on it); the rebuild lives in `src/market_edge.py`.

| # | Defect | Where | Consequence | Fixed in market_edge_v2 by |
|---|---|---|---|---|
| D1 | `p_model` defaults to **0.5** | `edge_engine.py:139` | a missing model silently becomes a coin flip and is blended in | `MISSING_MODEL` -> NO_BET |
| D2 | No kickoff, timestamp or staleness input at all; `stale_penalty` is a parameter `classify` never passes | `:117-118,173-174` | post-kickoff and day-old prices accepted | per-book `ts`, `POST_KICKOFF`, `STALE_PRICE`, stale penalty |
| D3 | Lineup gate returns OBSERVE before anything else; `lineup_confirmed` defaults False | `:144,163-164` | every team-market candidate is OBSERVE - the engine can never emit anything; a props concept copied over | removed (no lineup input exists for O/U) |
| D4 | `lineup_penalty` computed after the gate already returned | `:173-174` | dead code | removed |
| D5 | Linear probability blend `w*p_model + (1-w)*p_market`, model weight up to **0.30**, one weight for every track and market | `:98-113` | `abs_edge = w*(p_model - p_market)` grows linearly with raw disagreement | logit residual, track x family weight from config, clipped |
| D6 | Floors gate on `abs_edge = p_blend - p_market` and SNIPER needs `ae >= 1.5*abs_floor` | `:176,185-192` | **larger disagreement strengthens the tier** - selects exactly on the anti-predictive quantity | disagreement only ever LOWERS the state (penalty + OBSERVE cap); STRONG independent of the residual |
| D7 | Uncertainty `se = sqrt(p(1-p)/(n_eff + tau))` with `tau=400` | `:121-122` | a pseudo-count makes uncertainty SMALLER with zero evidence (n_eff=0 still behaves like 400 observations) | the unproven residual is removed from the lower bound in proportion `n0/(n0+n_eff)` |
| D8 | `ece` default 0.02, constant, never measured | `:143` | calibration penalty is a guess | penalties are explicit, listed per row (`penalty_breakdown`) |
| D9 | `n_eff` is caller-supplied with no definition; default 0 makes `w=0` and `abs_edge=0`, so the floors can never pass | `:142,105-106` | with defaults the engine is NO_BET always; with a made-up n_eff it is anything | `effective_sample_size` = replay-fitted evidence per track x family (config), reported on every row |
| D10 | CLV gate `rolling_clv > 0` with no n, no CI, no source | `:146,190` | one positive CLV observation unlocks SNIPER/MARKSMAN. The committed `bets_ledger.csv` CLV reaches **+287%** (garbage tails) | QUALIFIED needs track CLV n>=50 with clipped mean>0 AND median>0; STRONG needs PROSPECTIVE evidence for this engine version (n>=150, CI lower bound>0) |
| D11 | `market_baseline(...) or p_market`: exchange price used raw (no validation, no commission/spread), a 0.0 consensus falls through `or` | `:89-90,169` | an unvalidated exchange quote overrides the books | exchange skipped unless `exchange_validated` (false today) |
| D12 | Consensus is a median of a caller-supplied dict; nothing requires each book to be two-sided from one snapshot | `:86-94` | nothing stops best-over(A) + best-under(B) | consensus built only from same-book pairs; one-sided books counted, never used |
| D13 | No overround / arbitrage validation | `:166` | an arb pair (overround < 0) or a garbage pair de-vigs to nonsense | per-book overround in [0, 0.15] |
| D14 | `power_devig` bisects on a fixed bracket [0.5, 5] and never checks it brackets the root | `:72-82` | an out-of-bracket pair silently returns the endpoint as a probability | bracket checked (0.05-20), otherwise `None` -> `DEVIG_FAILED` |
| D15 | Side convention implicit: probabilities are always for OVER; no side argument | `:135-137,166` | UNDER / BTTS-NO require the caller to flip odds; easy to get wrong | both sides evaluated explicitly, `side` recorded |
| D16 | `best_odds` (best across books) is the EV price with no check it is fresh or in the consensus | `:138,152,175` | the classic stale-outlier "edge" | execution price = the board price legacy decides at (`price_source=row`), best valid book recorded separately |
| D17 | `MAX_BET_ODDS = 6.0` vs bankroll policy `hard_max_odds = 3.0` | `:32`, `config/bankroll_policy.json` | inconsistent caps | `max_odds = 3.0` (OBSERVE cap) |
| D18 | `RARE_EVENT_MARKETS` are props markets | `:33` | irrelevant to O/U and BTTS | removed; props are `UNSUPPORTED_MARKET` |
| D19 | `settlement_aligned` defaults True and nothing computes it | `:147,157` | a gate that always passes | removed (no input exists); documented |
| D20 | Flat `tx_cost = 0.02` subtracted from EV | `:36,128-129` | not odds-scaled | `odds_haircut` parameter (0.0 default, documented); EV_LB per brief = `p_lb*odds - 1` |
| D21 | Returns legacy tier names (SNIPER/MARKSMAN/VALUABLE) as decisions | `:153,192-196` | presentation conflated with decision | states NO_BET/OBSERVE/QUALIFIED/STRONG; `display_tier` is presentation only |
| D22 | A one-sided market is OBSERVE (good) but the longshot cap is checked first using `over_odds` as `best` | `:152,159-162` | minor ordering issue | fail-closed gates first, caps can only lower a state |

## 3. Two-sided price data that actually exists (counted 2026-10-25 on committed `output/`)

"two-sided" = same book, both sides, same snapshot (for the change-log archives: at least one
snapshot where the book wrote both sides; with carry-forward every book that ever quoted both
sides has a pair). Per-book = usable for consensus.

| Source | Market | Track | Rows | Fixtures | Two-sided fixtures | Books | Per-book | Dates |
|---|---|---|---|---|---|---|---|---|
| `book_odds_snapshots.csv` + `_archive/` (OddsAPI, change-log) | OU25 | new_format | 84,224 | 1,034 | 1,034 | 18 | yes | 2026-08-19 .. 11-01 |
| | OU25 | standard | 82,585 | 818 | 818 | 17 | yes | 2026-08-20 .. 10-31 |
| | BTTS | new_format | 72,744 | 996 | 996 | 13 | yes | 2026-08-22 .. 11-01 |
| | BTTS | standard | 58,706 | 809 | 809 | 10 | yes | 2026-08-22 .. 10-31 |
| | OU15 | new_format | 3,253 | 110 | 110 | 16 | yes | 2026-08-21 .. 11-01 |
| | OU15 | standard | 311 | 21 | 21 | 12 | yes | 2026-08-22 .. 10-25 |
| | OU35 | new_format | 34,924 | 454 | 454 | 17 | yes | 2026-08-19 .. 11-01 |
| | OU35 | standard | 7,668 | 180 | 180 | 16 | yes | 2026-08-22 .. 10-31 |
| `book_quotes/` (API-Football, daily parts, ladder) | ou@2.5 / 1.5 / 3.5 | new_format | ~29-31k each | 462 | 462 | 8 | yes | 2026-10-04 .. 11-01 |
| | ou@2.5 / 1.5 / 3.5 | standard | ~30-33k each | 504 | 504 | 7-8 | yes | 2026-10-04 .. 11-08 |
| | ht_ou@0.5 / 1.5 | both | ~31-37k each | 462 / 504 | all | 8-9 | yes | 2026-10-04 .. 11-08 |
| `standard_sidemarket_odds_history.csv` | btts | standard | 18,292 | 7,304 | 7,304 | 1 | no (Bet365) | 2024-08-10 .. 2026-11-08 |
| | ou15 / ou25 / ou35 | standard | 6-7k each | ~1,220-1,250 | 1,218 / 1,208 / 1,235 | 1 | no | 2026-05 .. 11-08 |
| `standard_odds_history.csv` | ou25 | standard | 9,159 | 876 | 876 | 1 | no | 2026-08-09 .. 10-31 |
| | ou15 / ou35 | standard | 37 / 682 | 21 / 185 | **0** (over only) | 1 | no | |
| `newformat_odds_history.csv` | btts | new_format | 12,942 | 4,371 | 4,371 | 1 | no | 2025-06-23 .. 2026-11-01 |
| | ou15 / ou25 / ou35 | new_format | 8-14k | 1,957 / 4,343 / 2,310 | 1,957 / 4,307 / 2,310 | 1 | no | 2025-06 .. 2026-11-01 |
| `newformat_odds_dense.csv` | ou25 | new_format | 11,640 | 1,106 | 1,106 | 1 | no | 2026-08-15 .. 11-01 |
| | ou15 / ou35 | new_format | 657 / 2,596 | 122 / 469 | **0** (over only) | 1 | no | |
| `odds_heartbeat/` (std 10-23.., nf 10-22..) | ou15/25/35, btts | both | ~200 each | ~95-109 | 83-108 | 1 | no | 2026-10-22 .. 10-25 |
| `exchange_quotes/2026-10.csv` | h2h, btts, alternate_totals (+lay) | - | 5,843 | - | - | 4 exchanges | yes | 2026-10-20 .. 10-24 (NOT validated, not used) |
| `predictions.csv` (board, 179 fixtures) | OU25 | both | over 179 / under 179 | | | | **board price, book unknown** | |
| | BTTS | both | yes 179 / no 179 | | | | **median-yes / median-no, NOT a pair** | |
| | OU15 / OU35 | both | over 7 / 55, **under 0** | | | | one-sided | |

Per-book overround (last pair per book, all fixtures): OU25 median 0.075 (p10 0.042, p90 0.094);
BTTS 0.080; OU15 0.070; OU35 0.070. Arbitrage pairs (<0): BTTS 2. Pairs >0.15: OU25 5, BTTS 6,
OU35 2. Power vs proportional on the live board: median absolute gap 0.76pp, max 3.2pp - larger
than most real edges, which is why both are recorded and the default is left to replay.

**Synthetic O1.5 (and BTTS/O3.5) prices.** `src/backtest.py:233` `_DEFAULT_ODDS = {"btts": 1.85,
"over15": 1.40, "over35": 2.60}` fills missing prices and flags `price_synthetic`. In the
committed `backtest_results_over15.csv`, **22,441 of 22,516** rows are the 1.40 fill (BTTS 21,335,
O3.5 22,439); 6 genuine 1.40 quotes are unflagged. Guard: `market_edge.exclude_synthetic_prices`
drops flagged rows, and when no flag column exists drops every row at the fill constant (fail
closed). A `BookQuote(synthetic=True)` or `Candidate(price_synthetic=True)` is never a market.
The constants are pinned against `backtest._DEFAULT_ODDS` by a test.

## 4. market_edge_v2 as implemented (`src/market_edge.py`, config `config/market_edge_v2.json`)

`selection_engine_version = "market_edge_v2"`; config `status = UNVALIDATED_DEFAULTS`.

1. **Fail closed (NO_BET):** unsupported market (props), track not standard/new_format, missing
   model, missing kickoff, post-kickoff, synthetic price, no price, stale price, de-vig failure.
2. **Per-book validation:** same book both sides, odds > 1, quote time <= now and < kickoff,
   overround in [0, 0.15], age <= 720 min (180 min inside 6h of kickoff). One-sided books are
   counted (`n_one_sided_books`), never used.
3. **Consensus:** validated exchange (skipped: `exchange_validated=false`) > `MEDIAN_TWO_SIDED_BOOKS`
   (>=2) > `SINGLE_TWO_SIDED_BOOK` > `ONE_SIDED` (OBSERVE, reason `ONE_SIDED_PRICE`, no fair prob)
   > NO_BET. Recorded: `market_source, books_used, n_books, market_overround (median per-book),
   price_timestamp (oldest book used), price_age_minutes (oldest), minutes_to_kickoff`.
4. **De-vig per book, median across books:** proportional `p = (1/a)/(1/a+1/b)`; Power
   `p = (1/a)^k` with `(1/a)^k + (1/b)^k = 1`. Both recorded (`p_market_fair_power`,
   `p_market_fair_proportional`); `devig_method` (default `power`) picks `p_market_fair`.
5. **Decision probability:** `market_residual_raw = logit(p_model) - logit(p_market_fair)`;
   `market_residual_adjusted = w[track|family] * clip(raw, +-0.35)`;
   `p_decision = sigmoid(logit(p_market_fair) + adjusted)`. Default `w = 0.10` everywhere,
   separate keys for standard/new_format x OU25/BTTS/OU15/OU35, **unvalidated**.
6. **Lower bound:** `p_lower_bound = p_decision - uncertainty_penalty`, penalty = sum of
   * books: `1.2816 * max(sd_books, 0.02) / sqrt(n_books)`
   * de-vig: `|p_power - p_proportional|`
   * residual: `|p_decision - p_market_fair| * 300/(300 + n_eff)`; `n_eff = 0` today, so the whole
     residual is removed - with no evidence, `p_lower_bound <= p_market_fair` always (tested)
   * league: `0.02 * 200/(200 + league_history_n)`
   * disagreement: `0.25 * max(0, |p_model - p_market_fair| - 0.08)`
   * long odds: `0.02 * max(0, odds - 2.5)`
   * stale: `0.002 per hour beyond 60 min`
   * retrain: `0.01` if trained < 7 days ago (time parsed from `model_version`, never mtime; unknown = recent)
   * CLV: `0.01` if the track CLV record is weak/absent
7. **EV:** `ev_point = p_decision*odds - 1`, `ev_lower_bound = p_lower_bound*odds - 1`; odds = the
   board price legacy decides at (`price_source=row`), median of valid books if absent; best valid
   book recorded as `best_book_odds`. Both sides evaluated; the side with the higher EV_LB is kept.
8. **States** (value path first, caps can only lower):
   * EV_LB > 0, >=2 books, track CLV n>=50 with clipped mean>0 and median>0 -> QUALIFIED;
     without the CLV record -> OBSERVE `NO_TRACK_EVIDENCE`; <2 books -> OBSERVE `INSUFFICIENT_BOOKS`
   * + PROSPECTIVE evidence for `market_edge_v2` and this track (file `output/edge_engine_evidence.json`,
     n>=150, CI lower bound>0) -> STRONG. **The file does not exist: STRONG is unreachable.**
   * EV_LB <= 0 < EV_point -> OBSERVE; else NO_BET
   * caps: odds > 3.0 -> at most OBSERVE; `|p_model - p_market| > 0.20` -> at most OBSERVE;
     OU35 -> NO_BET always (`MARKET_BLOCKED(OU35); engine_would_be=...`)
9. `display_tier`: STRONG->SNIPER, QUALIFIED->MARKSMAN, OBSERVE->VALUABLE. Presentation only.

Track CLV context today (`clv_track_stats` over the committed ledgers, clipped +-50%):
new_format|OU25 n=910 mean +4.69 median 0.00; standard|OU25 n=421 mean +0.45 median 0.00;
new_format|BTTS n=188 mean -1.49 median -0.79; standard|BTTS n=61 mean +1.87 median 0.00;
new_format|OU15 n=112 mean -0.18 median +0.64. **No track passes the QUALIFIED evidence gate
today**, so the shadow will show OBSERVE/NO_BET only; a would-be QUALIFIED is still visible in
`decision_reason` (`EV_LB>0 but NO_TRACK_EVIDENCE`) for replay.

## 5. Shadow wiring (production predict)

* `pipeline.py:1155` - `src.edge_engine_shadow.run(preds, bets, side_bets, historical)` after the
  legacy `generate_bets` / `_generate_side_bets` and before the stake plan; wrapped, cannot raise.
* `output/edge_engine_shadow.csv` - first sight per `(selection_engine_version, fixture_id, market)`,
  append-only, earliest row wins, pre-kickoff only; legacy columns (`legacy_raw_edge, legacy_tier,
  legacy_side, legacy_threshold` via `gate_resolver`, `legacy_odds`) beside every engine column and
  `new_state / new_side / new_reason`.
* `output/edge_engine_health.json` - recorded `generated_at`, runtime, coverage per market/track
  (two-sided share, multi-book, one-sided), state counts, legacy tipped counts.
* Quotes: this run's OddsAPI per-book rows (`src/predict._LAST_BOOK_ROWS`) and per-event BTTS rows
  (`src/btts_odds._LAST_BOOK_ROWS`) - recording-only module globals - plus `book_odds_snapshots.csv`
  carried forward per book and side.
* `src/stake_plan.py:159,206` - trailing columns on `stake_plan.csv` only, attached after sizing:
  `selection_engine_version="legacy_raw_edge", decision_state, engine_version, p_market_fair,
  p_decision, p_lower_bound, ev_lower_bound, market_source, clv_evidence_state`.
  `stake_audit.csv` keeps its header. Not added to `bets_ledger.csv` / `side_bets_ledger.csv`:
  they are whole-file rewrites shared with `update_results.py`, and the value would be a constant.
* `.github/workflows/predict.yml` stages both new files per file.

## 6. Proof the served outputs are unchanged

`pipeline.mode_predict` was run twice on identical committed inputs (the 179-fixture
`output/predictions.csv` as the scored board, no network, no models): once with the HEAD versions
of `pipeline.py`, `src/stake_plan.py`, `src/predict.py`, `src/btts_odds.py`, once with this change.
Every output compared on all pre-existing columns, run timestamps excluded:

bets.csv (57 rows), side_bets.csv (31), predictions.csv (179), bets_ledger.csv (5,818),
side_bets_ledger.csv (609), ht_ledger.csv, ht_observations.csv, stake_audit.csv (461),
v91_shadow.csv, shadow_h2h.csv, shadow_experiments.csv, postponed.csv: **identical, same columns,
same order**. stake_plan.csv (99 rows): identical on every original column and order; only the 9
trailing shadow columns added. Shadow runtime 1.9 s (predict median ~3 min, timeout 25 min).

## 7. Open questions for the owner

1. Default de-vig (power vs proportional) and the residual weights / n_eff: to be fitted by a
   chronological replay. Defaults are hand-set and labelled `UNVALIDATED_DEFAULTS`.
2. BTTS freshness: BTTS is refetched every 2 h, and the change-log stores changes only, so a
   carried-forward BTTS price's age is an upper bound (last change, not last fetch). Fixtures not
   refetched in a run can read STALE. A per-fixture fetch timestamp would fix the measurement.
   (Same root cause: `btts_odds._last_seen` uses the last change as its refresh clock, so an
   unmoving market is refetched on every run after 2 h - a cost observation, not changed here.)
3. O/U 1.5 has almost no two-sided coverage on OddsAPI EU totals (board: 7 over, 0 under of 179).
   API-Football `book_quotes/` has 7-8 books two-sided for 1.5/2.5/3.5 since 2026-10-04 and could
   feed the consensus; not wired here (separate capture path, different fixture ids).
4. Exchange quotes exist (Smarkets / Betfair / Matchbook) but are unvalidated; validation would
   need commission and back/lay spread handling before they can top the hierarchy.
5. QUALIFIED evidence gate uses RETROSPECTIVE legacy CLV; no track passes it today. Whether a
   prospective record for this engine should replace it is a policy decision.
6. `predict.yml` rebases with `-X theirs`; predict is serialised (`concurrency: predict`), so the
   append-only shadow is safe today, but any second writer of this file would need the same care.
