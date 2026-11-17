# 08 — Player props and CLV audit (AU4)

Audit team AU4, 2026-11-01. Read-only. v9 origin/main `fb0e9697ec8556d4dd3df57f31f8ed5415a5012d`
(branch `audit/au4`); Pro origin/main `e0ccc27b98f57c60ee28eedba81812e9b5c5c25b` (read only).

Commands (worktree root, v9 venv; every heavy step under `HEAVY.lock`):

```
python scripts/audit/au4_kickoff_index.py          # 1,748 prop fixtures with kickoff_utc, from 377 sampled
                                                   # commits of output/player_tips.csv (19 rescheduled)
python scripts/audit/au4_clv_forensic.py           # -> clv_forensic.json, clv_by_*.csv, clv_records_reconstructed.csv
python scripts/audit/au4_props_regrade.py          # -> props_regrade.json/.csv (independent settlement)
python scripts/audit/au4_props_model_vs_market.py  # -> props_mvm.json, props_mvm_calibration.csv
python -m pytest tests/audit/test_au4_props_clv.py -q
```

## 1. Answer: why did +1.03% with 30.3% beat-close become "POSITIVE CLV — genuine edge"?

AU4 reproduced the production aggregate exactly from `output/clv_records.csv`: **n = 4,125 closed
records, mean clv_pct = +0.0103, beat-close = 0.303**. Of all 8,180 records, 4,055 never closed.
The verdict comes from one line:

```python
# src/clv_capture.py:113
"verdict": "POSITIVE CLV — genuine edge signal" if mean_pct > 0 else "non-positive CLV — ..."
```

Five independent causes combine to make that verdict wrong:

1. **The rule is the sign of a mean, with no uncertainty.** The fixture-block bootstrap 95% CI of
   the same mean is **−0.19% … +2.40%** (866 fixtures); the day-block CI is −0.72% … +2.94% (73 days).
   Zero is inside both.
2. **`clv_pct = odds_bet/odds_close − 1` weights every record by its odds.** It equals
   (p_close − p_bet) / p_bet, so a move on a longshot counts many times more than the same move on a
   favourite. By odds band the summed clv_pct is <1.5: −47.9; 1.5–2: −9.5; 2–3: +4.7; 3–5: +24.2;
   5–10: +24.4; **≥10: +46.4 from 12 records**. The 12 records at odds ≥ 10 contribute more than
   the whole positive total (+42.3). Measures that do not weight by odds are negative:
   * mean **log(odds_bet/odds_close) = −1.68%**, fixture-block CI **−2.65% … −0.68%**;
   * mean implied-probability move (1/close − 1/entry; positive = market moved toward the pick)
     = **−0.85 pp**, CI **−1.30 … −0.39 pp**;
   * median 0.000, 5%-trimmed −0.34%, 5%-winsorized −0.25%;
   * 36.8% exactly zero, 30.3% positive, **33.0% negative**. "Beat close 30.3%" means the market
     moved against the pick more often than toward it.
3. **Borrowed entry prices.** For 128 closed records the entry price `odds_bet` never appears in the
   same fixture's price history at or before `ts_bet`; 116 never appear for that fixture at all.
   These are mostly the namesake / other-fixture prices that production fixed on 2026-10-25
   (`bb385e0e`, rows not rewritten). Example: "Ousmane Camara goals 21.00", close 2.05,
   clv_pct +924%. These 128 records have **mean clv_pct +52.5%** and sum to +69.7; the 3,778
   records with a verified entry sum to −27.4 (**mean −0.73%**). Dates: 2026-08-27 → 2026-10-31,
   **9 of them after the fix** (see P8-02).
4. **Closes are not clean.** The close is the last row for (date, player, market) in
   `player_prop_odds_history`, with no kickoff filter and no book filter (`clv_tracker.close_out`):
   * **82** of 3,878 closes with a known kickoff are rows stamped at or after kickoff (in-play);
   * OddsAPI rows (257,233) are **best-of-books with no bookmaker**; API-Football rows are per book,
     but the close ignores the book. 273 closed records have several AF books in their history;
   * the history dedup key is `(match_date, league, match, player, market, odds)`: a price is
     recorded only the first time it is seen, so A → B → A leaves B as the "last" row (magnitude
     NOT IDENTIFIABLE; the reverted observation was never stored);
   * `snapshot_ts` is write time, while prices can come from a 2-hour response cache
     (`CACHE_TTL_PRICED = 7200`);
   * placeholder quotes of 1.01 for "to be carded" are taken at face value (2,120 history rows of
     cards ≤ 1.05).
5. **The CLV sample is a non-random half.** 4,055 of 8,180 records never close: 3,494 because no
   history row has that exact normalised player name and market (the tip's name and the bookmaker's
   name differ), 267 because nothing in history is for that fixture, and 294 for other reasons
   such as dates. The reported CLV describes the players whose names match exactly, not the tip slate.

### Corrected CLV (same records, cleaner closes)

| definition | n | mean | fixture-block 95% CI | log-mean (CI) |
|---|---|---|---|---|
| production (as stored) | 4,125 | +1.03% | −0.19 … +2.40 | −1.68% (−2.65 … −0.68) |
| strict close: last row < kickoff | 3,675 | +0.76% | −0.56 … +2.37 | — |
| same-source strict close (entry row's source) | 3,581 | **−0.96%** | **−1.77 … −0.10** | — |
| same-book strict close (AF rows with a known book) | 753 | −0.39% | −1.13 … +0.38 | — |
| **clean**: kickoff known, entry before KO, entry price seen by ts_bet, strict close | 3,537 | **−0.97%** | **−1.80 … −0.11** | **−2.84% (−3.87 … −1.88)** |

On cleanly defined data, prop CLV is **negative**: the market moves away from the tipped side
between first sight and kickoff.

### CLV breakdowns (production clv_pct, fixture-block CIs; `reports/au4_data/clv_by_*.csv`)

| by | group | n | mean | 95% CI | log-mean | %>0 / =0 / <0 |
|---|---|---|---|---|---|---|
| market | sot | 2,433 | +1.25% | +0.35 … +2.38 | +0.17% | 27 / 47 / 26 |
| market | goals | 975 | +5.67% | +2.83 … +9.19 | +1.59% | 44 / 13 / 43 |
| market | cards | 474 | **−10.1%** | −14.1 … −5.8 | −19.4% | 17 / 31 / 52 |
| market | sot2 | 196 | +1.90% | +0.16 … +3.78 | +1.19% | 35 / 35 / 30 |
| tier | AVOID | 3,792 | −0.66% | −1.49 … +0.17 | −2.63% | 29 / 38 / 33 |
| tier | VALUABLE | 169 | +32.1% | +12.3 … +61.8 | +12.7% | 44 / 29 / 27 |
| tier | PAPER | 160 | +5.42% | +2.09 … +9.30 | +3.88% | 44 / 25 / 31 |
| odds band | <1.5 | 639 | −7.50% | −9.69 … −5.39 | −13.0% | |
| odds band | 5–10 | 234 | +10.4% | +5.9 … +16.2 | +6.5% | |
| odds band | ≥10 | 12 | +387% | +187 … +577 | +132% | |
| price age | >72 h | 1,245 | +2.02% | +0.14 … +4.17 | −0.09% | |
| price age | 12–24 h | 904 | −0.02% | −3.48 … +5.13 | −4.66% | |
| price age | 0–3 h | 106 | +0.27% | −0.81 … +1.36 | +0.14% | 10 / 82 / 8 |
| month | 2026-08 | 898 | −3.59% | −5.46 … −1.44 | −6.1% | |
| month | 2026-09 | 1,653 | +2.47% | +1.29 … +3.88 | +0.95% | |
| month | 2026-10 | 1,554 | +2.18% | −0.41 … +5.64 | −1.93% | |
| league | Premier League | 795 | −1.12% | −2.77 … +0.67 | −2.26% | |
| league | La Liga | 657 | −2.29% | −5.16 … +0.83 | |
| league | Brazil Serie A | 191 | +8.15% | +2.86 … +14.7 | |

The VALUABLE tier's positive CLV disappears on verified entries. With the clean definition: AVOID
−0.30% (−0.95 … +0.45, n = 3,436), PAPER +2.99% (+0.20 … +5.99, n = 141), VALUABLE **+3.69%
(+1.08 … +6.51, n = 145)**, log-mean +2.65% (+0.20 … +5.23). That is a small, borderline-significant
CLV signal on 145 records / 100 fixtures, inside a family whose aggregate is negative and after
choosing among about 15 subgroups. It is a **hypothesis for a pre-registered forward test, not
evidence of edge**: no multiple-testing correction, and CLV is not profitability (§3).

By book: the OddsAPI rows record no book, so CLV by book is not identifiable for 257,233 history
rows. For AF per-book rows the same-book CLV is −0.39% (above).

## 2. Model vs market (props), out of time

Data: `player_ledger.csv` rows with a price, a model probability and a WIN/LOSS result
(VOID/DNP/POSTPONED/unsettled excluded) = **2,907** of 5,468 priced rows, months 2026-06 → 2026-10.
Prices are **one-sided** (99.94% of prop quotes have no No/Under side), so a two-sided de-vig is
impossible. The market model is therefore a logistic recalibration of logit(1/odds), which also
absorbs the margin. Walk-forward by month (train ≤ m−1, test m; the first two months train only):
2,400 test rows, 602 fixtures.

| model (OOT, pooled) | log-loss | Brier | AUC |
|---|---|---|---|
| base rate per market | 0.6418 | 0.2229 | 0.562 |
| **MARKET** recal(1/odds) | **0.6148** | **0.2098** | **0.644** |
| MODEL recal(model_prob) | 0.6641 | 0.2263 | 0.599 |
| BOTH | 0.6436 | 0.2211 | 0.630 |
| raw model_prob (no refit) | 0.6214 | 0.2139 | 0.622 |
| raw 1/odds (vigged) | 0.6543 | 0.2274 | 0.645 |

* Incremental value of the model given the price: ΔLL(BOTH − MARKET) = **+0.0288 (CI +0.018 …
  +0.041)**, ΔBrier +0.0113 (+0.007 … +0.015). Adding the model makes out-of-time forecasts **worse**.
  On verified entries only (n = 1,088): ΔLL +0.051 (+0.032 … +0.070).
* By market (OOT): goals ΔLL −0.0007 (−0.0096 … +0.0081), market AUC 0.585 vs model 0.529;
  sot +0.0428 (+0.029 … +0.058), AUC 0.656 vs 0.606; cards +0.040 (−0.010 … +0.131).
* Upper bound (not chronological; 5-fold cross-fit grouped by fixture): ΔLL goals +0.0014
  (−0.0003 … +0.0032), sot −0.0020 (−0.0056 … +0.0013), cards −0.0095 (−0.028 … +0.011).
  **Even with this optimistic design, no market shows information beyond the price that excludes
  zero.**
* The raw model beats the *raw* implied probability on log-loss (0.621 vs 0.654) only because the
  implied probability carries the margin. Once the margin is removed by recalibration, the price
  wins (0.615). Raw model AUC is 0.622 vs price 0.644.

**Conditional calibration** (`props_mvm_calibration.csv`, deciles of implied probability, test
rows): realised is below implied in every decile (the one-sided margin). The model is
**under-confident at the top**: implied 0.80, model 0.48, realised 0.60. It sits close to realised
in the bottom deciles.

**Tail calibration**: in the top decile of model edge (model_prob − 1/odds), n = 240, the model
says 0.407, the price implies 0.327 and the realised rate is **0.313**. The model is over-confident
by 9.4 pp exactly where it would bet. Flat ROI on that decile is +7.1% (CI −17.8% … +36.4%), which
is noise. Top decile of model probability: model 0.626, realised 0.573, implied 0.681.

## 3. Executable ROI (paper; flat 1u at the logged entry price)

Independent re-grade (`au4_props_regrade.py`): matched by normalised name + date, **and the stats
row must be the record's own fixture**; minutes 0 → VOID; a missing stat → UNRESOLVED, never LOSS.

* Agreement with production on records both can grade: 2,639 LOSS/LOSS + 1,353 WIN/WIN; **9
  disagreements** (6 LOSS→WIN, 3 WIN→LOSS); 1,312 unresolved by AU4. AU4's name join is stricter;
  production uses player_id first. 46 production-graded rows had their only stats row in **another
  fixture** (namesake risk).
* Settlement is therefore **not** what makes props lose. Hit rate 34.0% vs mean implied 46.4%
  (one-sided margin plus selection).

| population | n | ROI | fixture-block 95% CI |
|---|---|---|---|
| clv_records, production grading, all | 5,313 | **−25.2%** | −29.1 … −21.5 |
| AU4 re-grade, all graded | 4,001 | −23.4% | −27.8 … −18.5 |
| AU4 re-grade, entry **verified** | 2,024 | **−28.1%** | −33.3 … −23.0 |
| VALUABLE, verified entry | 78 | −20.7% | −55.1 … +21.7 |
| VALUABLE, unverifiable entry (no same-name history) | 103 | +75.7% | +14.4 … +140.1 |
| VALUABLE, entry not in fixture history (borrowed) | 7 | +214% | −100 … +633 |
| player_ledger, bet tiers (SNIPER/MARKSMAN/VALUABLE), all | 633 | −32.7% | −47.7 … −14.9 |
| player_ledger, bet tiers, verified entry | 38 | −59.1% | −86.4 … −31.4 |
| player_ledger, all priced settled | 2,907 | −27.7% | −33.0 … −21.9 |

The only positive prop ROI is in rows whose price cannot be verified, at odds that were not on offer
for that fixture. **Positive mean CLV does not imply profit here**: records with positive clv_pct
still lose (−17.6% ROI, n = 924); zero-CLV −30.1%; negative-CLV −32.2%.

## 4. Reproduction of the earlier props tests on current data

| earlier claim (2026-07-09 / CLAUDE.md inv. 2) | current data, AU4 | status |
|---|---|---|
| model genuinely accurate (AUC 0.62–0.85) | raw model AUC 0.622 overall OOT (sot 0.606, goals 0.529, cards 0.547 recalibrated) | holds at the bottom of the range; goals discrimination is weak |
| market AUC matches or beats the model | price AUC 0.644 > model 0.622; per market price ≥ model | **confirmed** |
| no betting edge | every verified-price ROI CI is below zero; no incremental LL over the price | **confirmed** |
| "POSITIVE CLV — genuine edge signal" (props_health) | artefact of odds weighting, borrowed prices and unclean closes; clean CLV −0.97% (−1.80 … −0.11) | **refuted** |

## 5. Bugs (Phase-11 fields)

| Bug ID | Sev | Repo | File/line | Root cause | Affected dates | Affected rows | Production impact | Probability impact | Selection impact | ROI impact (before/after, same sample) | Status | Smallest safe fix | Regression test | Class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AU4-P8-01 | P1 | v9 | `src/clv_capture.py:113` | Verdict = sign of mean odds-ratio CLV; no CI, odds-weighted, no data-quality gate | since 2026-06 | the props_health verdict line; 4,125 records | false "genuine edge" message in health / research | none | none (props paper) | n/a; the metric itself flips: +1.03% → clean −0.97% | ACTIVE | report log-CLV and implied-prob move with fixture-block CI on verified same-source pre-KO closes; verdict POSITIVE only if the CI lower bound > 0, else INCONCLUSIVE/NEGATIVE | `test_clv_verdict_not_positive_when_market_moved_against` (xfail strict) | statistical / evaluation bug |
| AU4-P8-02 | P1 | v9 | `player_model/odds_fetcher.py` tip pricing (fixed `bb385e0e` 2026-10-25); rows in `clv_records.csv` and `player_ledger.csv` never relabelled | Entry price borrowed from a namesake / other fixture | 2026-08-27 → 2026-10-31 (9 rows after the fix) | 128 closed CLV records; 42 ledger rows NOT_IN_FIXTURE_HISTORY; unverifiable 1,777 ledger rows | inflated CLV and VALUABLE ROI | none | VALUABLE/MARKSMAN tiers assigned on fake odds | CLV +1.03% → −0.73% without them; VALUABLE ROI +40.9% (re-graded) → −20.7% on verified | HISTORICAL (data), possibly ACTIVE (9 post-fix rows, needs evidence) | add an `entry_verified` label (price present in that fixture's history by ts_bet) and exclude unverified rows from CLV and ROI readers; do not rewrite ledgers | audit script check; production test `tests/test_prop_price_fixture_scope.py` covers the code path | historical fixed + suspicious (post-fix rows) |
| AU4-P8-03 | P2 | v9 | `player_model/clv_tracker.py:117` | Close = last history row with no kickoff filter (history stores no kickoff) | all | 82 of 3,878 closes; 2,420 post-KO history rows | CLV uses in-play prices | none | none | small (those 82: mean +2.4%) | ACTIVE | store `commence_time` in the history rows; close = last row with ts < kickoff | `test_prop_close_is_strictly_pre_kickoff` (xfail strict) | true production bug (evaluation) |
| AU4-P8-04 | P2 | v9 | `player_model/clv_tracker.py:106-117`; `odds_fetcher.py:729-740` | OddsAPI history rows are best-of-books with no book; the close ignores book; entry is best of books at first sight | all | 257,233 history rows without book; 273 records with multiple AF books | CLV mixes books (best-of-books entry vs any-book close) | none | none | same-book −0.39% (n = 753) vs production +1.03% | ACTIVE | write per-book rows to history (prop_quotes already has them) and compute a same-book close | `test_prop_close_is_same_book_as_entry` (xfail strict) | true production bug (evaluation) |
| AU4-P8-05 | P2 | v9 | `player_model/odds_fetcher.py:178,460` | Dedup on (…, odds) keeps the first time a price is seen, so a reverted price is never re-recorded and "last row" is not the last observed price | all | NOT IDENTIFIABLE | close can be an intermediate price | none | none | NOT IDENTIFIABLE | ACTIVE | consecutive-distinct dedup per (fixture, player, market, book), as `drift._write_archive` already does | — | true production bug (evaluation) |
| AU4-P8-06 | P3 | v9 | `player_model/odds_fetcher.py:88,686-694,430` | Response cached up to 2 h, but rows are stamped with the write time | all | NOT IDENTIFIABLE | quote timestamps up to 2 h late; can turn a pre-KO price into a "post-KO" row and vice versa | none | none | NOT IDENTIFIABLE | ACTIVE | stamp the cached `fetched_at` | — | true production bug (data timestamp) |
| AU4-P8-07 | P3 | v9 | `odds_fetcher._best_price` / `prop_odds_af` | Placeholder quotes (cards 1.01) accepted as prices | all | history ≤1.05: 2,639 (cards 2,120); prop_quotes 8,054 | entry 1.01 / close 1.01 records (Hakimi cards 1.01) | model edge vs a fake price | can create AVOID/VALUABLE noise | small | ACTIVE | reject cards/goals quotes ≤ 1.05 as non-executable | — | data-quality |
| AU4-P8-08 | P3 | v9 | `player_model/clv_tracker.py:27,103-117` | `bet_id = PLAYER|date|player|market` (no fixture, book or side); close requires an exact normalised name and exact date | all | 4,055 never closed (3,494 name mismatch) | CLV computed on a non-random half | none | none | NOT IDENTIFIABLE (selection) | ACTIVE | key by fixture_id + player_id; resolve the bookmaker name once at pricing time | — | data-coverage limitation |
| AU4-P8-09 | P3 | Pro | `src/market/clv_schema.py:209-212` | `minutes_close_before_kickoff` is computed against `match_date` midnight, not kickoff | — | Pro CLV schema rows | research metric mislabelled | none | none | n/a | ACTIVE (research) | use the fixture kickoff | — | research-only flaw |

## 6. Verdict inputs (props track)

* **Track: player props (paper).** Claimed: "POSITIVE CLV — genuine edge". Independently verified
  CLV: production +1.03% (CI −0.19 … +2.40). **Corrected CLV −0.97% (CI −1.80 … −0.11)**; log −2.84%.
  Verified-entry ROI −28.1% (CI −33.3 … −23.0, n = 2,024). The model adds no information beyond the
  price (OOT ΔLL +0.029, worse; cross-fit Δ CIs include 0). → **NO DEMONSTRATED EDGE / NEGATIVE
  PERFORMANCE**, with the CLV claim **INVALID EVALUATION**.
* PROMISING-UNCONFIRMED candidate, for pre-registration only: VALUABLE-tier clean CLV +3.69%
  (CI +1.08 … +6.51, n = 145 / 100 fixtures), with verified-entry ROI −20.7% (n = 78). It needs an
  untouched forward period, verified same-book pre-KO closes, and n ≥ 150 fixtures before any reading.
* Props stay paper (CLAUDE.md invariant 2). Nothing here argues otherwise.

## 7. Limitations

* Kickoff times come from 377 sampled commits of `player_tips.csv`; 247 closed records (6%) have no
  kickoff and are excluded from the strict and clean definitions.
* AU4's name join for re-grading leaves 1,312 records unresolved, where production graded them via
  player_id. ROI on the AU4-graded subset is reported alongside production's, not in place of it.
* The props model probabilities in `player_ledger.csv` are as served. The props serving on
  2026-10-26 were trained on leaked injury values (CLAUDE.md, `fb43970c`), so earlier rows may be
  optimistic. That bias would favour the model, and the model still does not beat the price.
* The walk-forward has 5 months, and June is dominated by the World Cup, so recalibration
  coefficients shift between months. The cross-fit upper bound is reported for that reason.
