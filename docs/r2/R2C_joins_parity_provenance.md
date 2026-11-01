# R2-C: enrichment joins, train/serve parity, price provenance, side-edge shadow, OddsAPI quota

Branch `r2/joins-parity-health` (v9). Written 2026-11-01 on top of origin/main `4e9c93357`.
Scope: owner-brief findings 12, 14, 15, 18, 23 and 24, the side-market edge-rule finding, and the
shared integrity health artifacts. **Production selection is unchanged.** Every behaviour change
in `src/data_loader.py` sits behind a flag that defaults OFF, pending owner review.

## 1. Enrichment joins (finding 12)

### The defect, proven on committed data

`_enrich_with_api_shots` merged `output/af_history.parquet` on (home, away, date) with no league
and copied the result back with a positional `zip`. af_history holds duplicate keys:

* 17,016 rows in 8,420 duplicate (home, away, date) groups;
* 3,592 of those groups span two leagues: Dutch Eredivisie fixtures also filed as Norway
  Eliteserien (1,798), and Bundesliga 2 fixtures also filed as Denmark 1st Div (1,794);
* the rest are one club with two spellings that normalise alike (`Argentinos JRS` and
  `Argentinos Jrs`). 1,473 of these groups disagree on a value; for example, the two copies
  carry different xG figures.

The left merge returned **98,531 rows for an 85,467-row frame (+13,064)**. Because `zip` stops at
the shorter input, every row after the first duplicate received a neighbour's value:
**85,454 of 85,467 rows are positionally misaligned**. The first one is 2019-02-25, Dundalk v UCD.

I spot-checked the replay against af_history. On 2026-10-29 Remo v Botafogo was served
9/4 shots and 2/4 shots on target; af_history holds 11/15 and 4/2. Club America v Tijuana was
served 12/5; the bank holds 28/6. The safe join returns the bank's values.

A second defect has the same effect. `_canonicalise_clubs` leaves the history frame with a
non-RangeIndex (85,467 rows, maximum label 107,524). `_enrich_with_standard_sidemarket_odds` and
the first pass of `_enrich_ou25_from_captures` combine `df[col].isna() & matched[col].notna()`
across two different indexes, so the mask aligns by label rather than by position. Their filled
prices also land on the wrong rows.

### Replay (RETROSPECTIVE, `scripts/integrity_af_join_replay.py`, committed caches, no network)

These counts are history cells that differ between the legacy join (what serves) and the safe join:

| column | shifted (both present, different) | phantom (legacy value, no correct value) | dropped (correct value, legacy NaN) |
|---|---|---|---|
| home_shots | 19,066 | 934 | 13 |
| home_sot | 17,773 | 886 | 14 |
| home_xg | 3,142 | 1,682 | 9,008 |
| home_insidebox | 4,546 | 1,496 | 9,888 |
| odds_btts | 724 | 4,283 | 5,996 |
| odds_over15 / over35 | 0 | 534 / 541 | 748 / 765 |
| odds_over25 / under25 | 1 | 1,729 | 166 |

The away columns are symmetric. In total **38,701 history rows** have at least one enriched value
changed. The largest effects are in the National League, Argentina, MLS, Brazil, Liga MX, Ireland
and Japan.

### Serving effect: changes, so the safe path stays behind a flag

I ran the deployed predict pass on the committed board (153 fixtures, `v9_repaired`) once with
each history:

* **Serving features that change:** home/away `xg_last5` (148 fixtures, 82 new-format and
  66 standard), home/away `insidebox_last5` (142–144), `combined_sot_ratio` (82 new-format),
  home/away `corners_pg_roll` (36–43 new-format) and home/away `fouls_pg_roll` (40–45 new-format).
* **Probabilities:** new-format `p_over25` changes on 84 fixtures (mean |Δ| 0.009, max 0.060).
  `p_btts`, `p_over15` and `p_over35` change on all 153 (mean |Δ| about 0.009, max 0.049).
  `p_ht_*` changes on 66–67 standard fixtures. **Standard OU2.5 `p_over25` does not change.**
* **Tips:** 35 under each join, with 4 different on each side. Legacy only: ARG Belgrano v Barracas
  O1.5 MARKSMAN (VALUABLE under safe), JPN Shimizu v Verdy O2.5, SWE Degerfors v GAIS U2.5,
  SWE Malmo v Hammarby U2.5. Safe only: BRA Cruzeiro v Bahia U2.5, BRA Vitoria v Fluminense
  O2.5, JPN Machida v Hiroshima U2.5.

As the brief requires, the safe joins serve **only with `WOWZA_SAFE_ENRICH_JOINS=1`**, which
defaults to 0. The legacy joins are **audited on every run either way**, and the result is
recorded in `output/join_integrity_health.json`. The current status is FAIL, with
`serving_join_mode=legacy`.

**Owner decision needed.** The models now serving were trained on the shifted values. Turning
the flag on changes serving features at once, and changes training data at the next retrain.
The safe values are the correct ones.

### `safe_left_enrich` (`src/integrity/joins.py`)

* The key must carry `league` or `fixture_id`; otherwise the join raises ValueError.
* The right side is collapsed to one row per key. Duplicates that agree become one row. A cell
  where duplicates disagree is AMBIGUOUS and stays unfilled; there is no first-wins or last-wins.
* The merge uses `validate="many_to_one"`. Merged rows must equal left rows, and the left row
  number travels through the merge and is checked, so the left index and order are preserved.
* The fill is NaN-only by default. A JoinReport records rows, duplicate right keys, ambiguous
  cells, unmapped rows and filled cells per column.
* `audit_legacy_merge` measures expansion and misalignment of a legacy merge.
  `audit_league_free_lookup` measures cross-league values in pivot lookups. For example, 532 rows
  of `Denmark 1st Div <- Bundesliga 2` reach the standard side-market read.

### Inventory of fixture-level enrichment joins in v9

The machine-readable copy is in `join_integrity_health.json → fixture_level_enrichment_joins_in_v9`.

| site | legacy key | defect | fix |
|---|---|---|---|
| data_loader `_enrich_with_api_shots` step 1 (af_history) | h+a+date, no league; positional zip | expands and shifts (proven) | safe join, flag |
| data_loader step 1b (af_ht_history; not committed, CI cache only) | same | same shape | safe join, flag |
| data_loader live per-league shots merge | h+a+date within one league; positional | expands if the fetch holds a duplicate | safe join, flag |
| data_loader `_enrich_with_standard_sidemarket_odds` | h+a+date pivot, no league; mask across two indexes | misaligned (proven); cross-league | safe join, flag |
| data_loader `_enrich_ou25_from_captures` pass 1 | same | misaligned (proven) | safe join, flag; pass 2 (resolver) was already index-safe |
| data_loader `_enrich_with_af_odds` | h+a+date reindex | permanent no-op (file can never exist) | audited only |
| data_loader `_nf_real_odds` | (date, league, match) dict | no expansion; reads mis-mapped pre-fix O/U rows | `WOWZA_PRICE_MAPPING_FILTER`, flag |
| feature_engineering / predict lookups | league-scoped team lookups | not edited; outside R2-C's ownership | other workstreams |

## 2. Train/serve parity (finding 18)

`scripts/integrity_feature_parity.py` builds `output/feature_parity_health.json` (all tracks) and
`output/nf_feature_parity_health.json`. The training statistics come from an offline rebuild of
the training frame from the committed caches, using the serving joins and the recipe each model
was trained with. The live statistics come from the committed `output/predictions.csv`. For each
serving feature the report gives the train and live mean, std and missing rate, a constant-live
flag, PSI on training deciles, the expected rule, and a PASS, WARN or FAIL verdict.
The rules are listed in `src/integrity/parity.py` under `PARITY_RULE_VERSION`.

**The 2026-11-01 new-format fixes hold on the fresh board** (board generated 2026-11-01 09:43Z,
`git_sha b0a1204ad4c6`, 86 new-format and 67 standard rows):

| NF feature | train mean / sd | live mean / sd | PSI | verdict |
|---|---|---|---|---|
| home_advantage | 0.475 / 0.276 | 0.509 / 0.229 | 0.10 | PASS (no longer a constant 1.0) |
| away_home_adv_factor | 0.473 / 0.279 | 0.520 / 0.242 | 0.10 | PASS |
| bookmaker_overround | 0.043 / 0.028, **83% missing** | 0.051 / 0.019, 0% missing | 2.07 | WARN (market_derived) |

Summary by model: new-format OU2.5 is WARN (16 PASS, 11 WARN, 0 FAIL). Every standard-track
report is **FAIL on `home_rest_days` and `away_rest_days`**: 63% and 57% of live standard fixtures
have no rest-days value, against 0% in training, so most live values are imputed. New-format
fixtures are at 34–38%, which is a WARN. This is a serving-feature defect for the
feature-engineering owner; R2-C did not touch `feature_engineering.py`.

The many WARNs are mostly PSI on 67–86 live rows, which is noisy at that sample size. Real shifts
also appear: `away_conceded_last5` and `away_defense_str` sit about 0.65 train sd above training,
and the HT rolling features differ in missingness.

There is a regression test for this:
`tests/test_integrity_parity.py::test_constant_live_home_advantage_is_fail`. Any feature that
varies in training but is constant across five or more live fixtures is FAIL, even when the
constant is plausible. Calendar features are capped at WARN by a declared rule.

## 3. Price provenance (findings 14, 15, 24)

`src/integrity/price_provenance.py` and `output/price_provenance_health.json` cover this. Raw
files are never rewritten.

* **Synthetic side prices.** Committed backtests contain 94.8% synthetic BTTS rows, 99.7% O1.5
  and 99.7% O3.5. Those rows are labelled `SYNTHETIC_PRICE_DIAGNOSTIC_ONLY` and
  `INVALID_FOR_PROFITABILITY`. **0 placed SNIPER/MARKSMAN rows use a synthetic price** (209, 10
  and 59 placed rows respectively). `assert_profitability_inputs` refuses any synthetic or
  mis-mapped row.
* **`newformat_odds_history.csv` origin.** I checked each line against realised outcomes from
  `fd_history`, using the de-vigged implied rate against the realised rate:
  * BACKFILL rows (`...T00:00:00Z`, 2025-06 to 2026-06) are valid full-match lines. O1.5 implied
    0.590 vs realised 0.621; O2.5 0.517/0.530; O3.5 0.438/0.442; BTTS 0.533/0.561.
    **The "NF over15 median 1.57–1.66" is selection, not mis-mapping.** The historical OddsAPI
    totals carry only each book's main line, so an O1.5 price exists only where 1.5 was the main
    line: 393 of 2,820 fixtures, the low-scoring ones. These rows are labelled
    `line_selection=MAIN_LINE_ONLY`.
  * FORWARD O/U rows captured before `ac6406cb4` (2026-08-10) are **invalid**. The parser matched
    any bet whose name contained "Over/Under" (first half, corners, team totals), and the last
    match won. This is where the summer-2026 over25 medians of 2.20 to 3.6 come from: implied
    0.318 vs realised 0.490 at O2.5, and 0.546 vs 0.749 at O1.5. 4,074 rows are marked
    `market_mapping_valid=False`.
  * FORWARD BTTS rows before `8d9962c68` (2026-08-23) contain first-half BTTS. 933 rows priced
    above 3.20 are invalid (the gap rule). Standard side-market file: 271 rows.
  * After the fixes, all lines are consistent: O2.5 implied 0.551 vs realised 0.551.
* **Contract tests** (`tests/test_integrity_price_contract.py`): a synthetic price never enters
  ROI; a one-sided price is never called de-vigged; a quote at or after kickoff is never an entry
  price, and an unknown time fails closed; the standard football-only model carries no price
  feature; the new-format model's only price feature is `bookmaker_overround`, which is pinned so
  the exception cannot grow silently; first-half BTTS and pre-fix O/U lines are rejected.
* **Training reads.** `_nf_real_odds` and `_enrich_ou25_from_captures` now count mis-mapped rows on
  every run. With `WOWZA_PRICE_MAPPING_FILTER=1` (default 0, owner review) they also drop them,
  which affects roughly 360 summer-2026 new-format fixtures' training prices.

## 4. Side-market decision rule: 1.08 vs de-vig (SHADOW)

`src/integrity/side_edge_devig.py` and `scripts/integrity_side_edge_shadow.py` compute the shadow
rule beside production. For each bookmaker, both sides of the same market from the same book
must be quoted before kickoff and at or before the decision time. Each pair is de-vigged with the
power method (`src.market_edge.power_devig`), and the consensus is the median across books. The
shadow uses the same thresholds as production, which is proven against
`pipeline._generate_side_bets` in a test. The production rule is not changed.

| scope | tips | confirmed by de-vig | **only because of 1.08** | unverifiable (one-sided) |
|---|---|---|---|---|
| current board (09:43Z) | 2 | 2 | 0 | 0 |
| open ledger tips (match date today or later) | 30 | 16 | **8** | 6 |
| ledger since quotes exist (2026-08-19+, RETROSPECTIVE) | 617 | 424 | **92** | 101 |

The 92 break down as BTTS 77, O1.5 8 and O3.5 7. The median pair overround of the UK/EU books we
capture is about 8%, so on average the 1.08 rule is close to proportional de-vig: the median
inflation is 0.1pp and the mean 0.47pp. The defect is in the tails. The 90th percentile is
+2.5pp; 83 tips were inflated by at least 2pp; and where the real overround is below 6% (sharp
books), inflation averages **+2.67pp**. Because many tips sit just above the 4pp VALUABLE bar,
15% of logged side tips do not survive a real de-vig. The append-only shadow log is
`output/side_edge_devig_shadow.csv`; the summary is `output/side_edge_devig_health.json`.
Cutover is the owner's decision.

## 5. OddsAPI quota (finding 23)

`src/btts_odds.py` gains monitoring only, and the fetch clock behaves exactly as before
(`tests/test_btts_fetch_clock.py` passes unchanged). Each run emits one
`[oddsapi_usage] {json}` line with calls, credits by league and market, fixtures considered,
fixtures suppressed by the 2h clock, capped and near-kickoff counts, and `x-requests-remaining`.
The key is never logged.

`scripts/integrity_oddsapi_quota.py` reads the run logs through the Actions API and writes
`output/oddsapi_quota_health.json`, including a one-line health check. Measured over the last
48h on 2026-11-01:

* account remaining **98,948** (the plan reset on 11-01);
* burn **1,352 credits/day**, measured from the remaining-credits series across all consumers;
* **73 days** to exhaustion at that rate;
* BTTS fetches logged by predict: 390 credits in 24h from 195 calls;
* BTTS duplicate-fetch suppression **58%** (264 of 459 fixtures skipped by the refresh clock), so
  the 11-01 clock fix is working.

Status is PASS. The rule is WARN below 25% of the allowance or under 7 days to exhaustion, and
FAIL below 5,000 credits. An unmeasured quota reports UNKNOWN, never PASS. Per-league numbers
start once this branch's log line ships.

## 6. `output/model_integrity_health.json` (skeleton)

The file has one record per serving track: standard/new-format OU25, BTTS, O1.5 and O3.5;
standard HT O0.5 and O1.5; and props. It carries every field the spec lists. Fields known now
are filled: recipe and version, payload hash, serving version, training cutoff, code SHA, data
hash and identity, feature and imputer versions for `v9_repaired`, chronology and holdout blocks
for repaired, parity, batch invariance (legacy uses the batch median, so FAIL), price provenance,
missing-feature policy, serving-fallback policy (fails OPEN: findings 8 and 17), and join
integrity.

Fields other workstreams own are `UNKNOWN`: weight alignment, legacy chronology, the
season-resolver and league-registry versions, and props. They are picked up automatically when
their health files appear. **UNKNOWN never counts as PASS.** All tracks are currently FAIL, from
the parity rest-days FAIL, the legacy-join FAIL and the legacy batch-median imputation.

## 7. Workflow

`.github/workflows/integrity_health.yml` runs daily (`20 8 * * *`), on push to its own paths, and
on dispatch. It runs the contract tests, then the six generators, each non-fatal and labelled
INFRA_FAILURE if it breaks, then commits the health files one by one. Permissions are
`contents: write` and `actions: read`. It is read-only with respect to production.

## Rollback

Delete `.github/workflows/integrity_health.yml`. Production behaviour already equals main while
`WOWZA_SAFE_ENRICH_JOINS` and `WOWZA_PRICE_MAPPING_FILTER` are unset. The only always-on additions
are the audit calls, which are wrapped in try/except and add about one merge per enrichment, and
one extra log line in `btts_odds.fetch`.

## Open risks / owner decisions

1. **Turn on `WOWZA_SAFE_ENRICH_JOINS`?** It changes serving features and four tips per side on
   today's board. The served models were trained on shifted shot, xG and inside-box data;
   retraining on corrected data is a further decision.
2. **Turn on `WOWZA_PRICE_MAPPING_FILTER`?** It changes training prices for about 360 new-format
   fixtures.
3. Standard rest-days are 63% missing live (a feature-engineering workstream defect).
4. The 1.08 side rule: 8 of 30 open side tips exist only because of it. Cutover to the de-vig
   shadow is an owner decision.
5. af_history itself still holds the Dutch-as-Norway and Bundesliga 2-as-Denmark 1st Div copies
   and the two-spelling duplicates. They are left in place, because the raw bank is not
   rewritten; the safe join ignores them by league key and ambiguity rules.
6. Parity training statistics are unweighted and cover all seasons, so a PSI WARN can reflect
   season drift rather than a defect.
