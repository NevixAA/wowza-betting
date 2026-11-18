# 06 — Settlement and ROI reconciliation (AU3)

Audited: v9 `origin/main` **fb0e9697ec8556d4dd3df57f31f8ed5415a5012d**; Pro `origin/main` e0ccc27b98f5 (registry only).
Scripts: `scripts/audit/au3_first_sight.py`, `au3_settle.py`, `au3_phase6.py` (numbers in
`reports/au3_data/phase6_numbers.json`, row files `first_sight_*.csv`, `settled_*.csv`, `ledger_mutations_*.csv`).
Windows: **since 2026-09-01** (counts) and **all live** (context), labelled on every number.

## 1. Method — immutable first-sight selections

`bets_ledger.csv` and `side_bets_ledger.csv` were not first-sight before 2026-10-26: a later run could rewrite a logged
tip's tier/edge (O/U) and also its **odds** (side markets). The first-sight row of every tip was therefore rebuilt from
**every committed version** of the two ledgers (798 and 399 versions, oldest first, via `git log --raw` +
`git cat-file --batch`). For each key — `(match_date, home, away, side, source)` O/U, `(…, market, source)` side — the
row in the first commit that contains it is the selection; every later change is logged in `ledger_mutations_*.csv`.

Settlement is **independent** (`au3_settle.py`, no import of `update_results.py`/`src/ledger.py`): scores from
`output/af_history.parquet` (API-Football) and `output/fd_history.parquet` (football-data), matched per source,
league-scoped, ±1 day, names via `src.team_names.resolve_result` then a date-scoped unique-token fallback that refuses
ambiguity. WIN = odds−1, LOSS = −1, VOID = 0 and excluded from settled stake, ROI = profit / settled stake (flat 1u — no
variable stake exists on the ledgers; `kelly_pct` is advisory and never applied). An unmatched fixture is UNRESOLVED and
is **never** counted as a loss. 22 adversarial tests: `tests/audit/test_au3_settlement.py` (pass).

## 2. Row categories

| | O/U 2.5 (`bets_ledger`) | side (`side_bets_ledger`) |
|---|---|---|
| live keys ever committed | 1,829 | 652 |
| in the final ledger | 1,806 | 652 |
| **vanished from the ledger** (history rewrite) | **23** | 0 |
| unbettable `AVOID` rows (result VOID, "no side cleared MIN_ODDS") — not bets | 135 | — |
| ledger WIN/LOSS | 1,551 | 602 |
| ledger VOID (real voids) | 0 | 0 |
| ledger unsettled (all) / with match date already past | 143 / 61 | 50 / 18 |
| independently settled WIN/LOSS | 1,475 | 579 |
| ledger-settled but not independently verifiable (unresolved/ambiguous names, results not yet in af/fd) | 96 | 33 |
| **result mismatch ledger vs independent** | **0** | **0** |
| independently settled but ledger still pending | 20 | 10 |
| settled rows with missing odds / missing pnl | 0 / 0 | 0 / 0 |
| **odds changed after first sight** | **0** | **92** (91 settled) |
| tier changed after first sight (excl. VALUE→VALUABLE rename) | 403 | 135 |
| duplicate keys inside a version | 0 | 0 |

Same-fixture correlation: 1,860 fixtures carry at least one tip; **408 carry 2+** (894 tips). 39 fixtures have both
O/U sides logged (pre flip-guard, 2026-10-11). 20 fixtures carry UNDER 2.5 plus an Over 1.5/3.5 (opposite goal
direction); **0** of them had their second leg logged after the direction guard went live (2026-10-25 12:17 UTC). 94
fixtures carry UNDER 2.5 + BTTS Yes, which the owner rule defines as directionless (allowed) though the outcomes are
negatively correlated — the two are not independent bets in any portfolio CI.

**History rewrite:** all 23 vanished O/U rows and all 8 results reset from settled to blank happened in one manual
commit, `075eeb3136` (2026-06-22 "feat: v9.2 — side markets…"): fixtures 2026-06-17 → 06-28 (13 `AVOID`, 10 `OVER`;
one vanished OVER had been settled WIN). Pre-2026-09-01, so outside the counting window, but it shows the ledger was
not append-only then.

## 3. Reconstructed P&L vs committed ledger

A = committed ledger as it stands; B = independent settlement at the **first-sight** price; C = independent settlement
at the ledger's (possibly rewritten) price; D = the identical row set settled both ways.

| market | track | window | A committed ledger | B independent, first-sight price | D same rows: ledger → first-sight |
|---|---|---|---|---|---|
| OU25 | all | since 09-01 | n 877, −45.02u, **−5.13%** [−13.0, +2.1] | n 845, −41.15u, **−4.87%** [−13.0, +2.4] | n 844: −42.87u = −42.87u |
| OU25 | standard | since 09-01 | n 346, −35.18u, −10.17% | n 326, −32.01u, −9.82% [−20.1, +2.7] | identical |
| OU25 | new_format | since 09-01 | n 531, −9.84u, −1.85% | n 519, −9.14u, −1.76% [−14.2, +9.8] | identical |
| OU25 | all | all live | n 1,551, −115.64u, −7.46% | n 1,475, −96.27u, −6.53% [−12.6, −1.2] | identical |
| BTTS | new_format | since 09-01 | n 232, +24.15u, **+10.41%** | n 234, +18.64u, **+7.97%** [−5.2, +21.5] | n 227: +22.92u → +21.67u |
| BTTS | new_format | all live | n 270, +38.18u, +14.14% | n 267, +35.03u, +13.12% [+0.9, +26.2] | n 259: +38.80u → +37.06u |
| BTTS | standard | since 09-01 | n 154, −1.78u, −1.16% | n 145, −0.38u, −0.26% | n 144: −0.74u → −1.15u |
| Over 1.5 | new_format | since 09-01 | n 79, +9.20u, +11.65% | n 76, +8.84u, +11.63% [−4.1, +27.7] | n 75: +8.41u → +8.32u |
| Over 1.5 | standard | since 09-01 | n 9, −1.87u | n 9, −1.87u | identical |
| Over 3.5 | all | since 09-01 | n 24, −3.43u, −14.3% | n 23, −5.24u, −22.8% | — |

(Day-block bootstrap 95% CIs, resampling whole match dates.)

* **O/U 2.5:** the ledger price was never rewritten, so first-sight P&L equals ledger P&L row for row. Differences
  between A and B come only from which rows each side can settle (A includes 96 rows AU3 cannot verify; B includes 20
  the ledger has not settled).
* **Side markets:** 92 tips were settled at a later run's price. On the same rows this flatters NF BTTS by **+1.74u
  (all live)** / +1.25u (since 09-01) and all side markets by **+2.35u** (matches the `src/ledger.py` docstring's
  "+2.26u", which was measured on an earlier date).
* **Tier labels.** O/U, since 09-01, independent settlement, first-sight price — first-sight tiers: SNIPER n 91 −10.5%,
  MARKSMAN n 154 **−16.1%** [−31.6, −0.2], VALUABLE n 600 −1.1%; ledger tiers: SNIPER n 153 −13.8%, MARKSMAN n 305 −4.5%,
  VALUABLE n 387 −1.7%. Side markets: first-sight SNIPER n 102 **+23.5%** [+3.4, +43.4] vs ledger SNIPER n 163 +5.8%.
  Every tier-level number published before 2026-10-26 from the ledger mixes tiers that were never sent.

## 4. Health-contract violations, recounted on origin/main fb0e9697e

`python -m src.contracts` (read-only) on `output/stake_audit.csv` (687 rows, written 2026-10-14 → 2026-11-01):

| violation | brief's count | **recount** | what the rows are | effect |
|---|---|---|---|---|
| `side` not in OVER/UNDER/YES/NO (`YES/OVER`) | 110 | **110** (BTTS 89, OVER15 18, OVER35 3) | rows written 2026-10-14 → 10-22 10:07, before the side-market staking fix b4ce9e22 | all `deployment_state=UNSUPPORTED`, reason `NO_REAL_STAKE_UNSUPPORTED`, **final_fraction 0**: presentation only; no stake, no ROI effect. The only reader that splits the side (`src/research/league_weekly._stake_lookup`) maps it to two keys with stake 0 — harmless |
| null `odds` | 110 | **110** — the *same* 110 rows | odds lost before the fix | none on stakes/ROI; these rows cannot be priced in any stake-audit replay |
| `one_position_per_fixture` | 64 | **66 rows / 33 fixtures** (34 rows written after 2026-10-25) | see below | see below |

The 66 one-position rows decompose as:
* **62 rows / 31 fixtures — the same NF UNDER 2.5 tip sized twice.** Each was first audited under track
  `new_format|OU25` (2026-10-20 → 10-24) and again under `new_format|OU25_UNDER` after commit ab23726d5 (2026-10-26
  "split O/U 2.5 tracks by side") renamed the track. `stake_audit` is first-sight per `(policy_version, fixture, track)`,
  so the rename defeated de-duplication. In 13 of the 31 the second sizing used different odds, in 7 a different
  tier. Effect: **paper allocation double-counted** — 10.22 units of a 100-unit paper bankroll (Σ final_fraction 0.103)
  sized a second time for positions already planned, consuming daily/league caps on 2026-10-26. No ledger, no settled
  ROI and no Telegram tip changed (the ledger key does not include the track). Classification: true (paper-staking)
  bug, P2, HISTORICAL-in-effect (one-off on the rename day) but the audit file still carries both rows.
* **2 fixtures — genuine same-direction multi-positions:** San Lorenzo v Newell's (BTTS + Over 1.5, in the baseline)
  and one BTTS + OU25_OVER. Owner rule 2026-10-25 explicitly allows same-direction positions, so the contract rule
  "one real position per fixture" is **stricter than the approved policy** — intentional owner-approved config, not a
  bug. Recommend the contract text be aligned to "one goal direction per fixture" (owner decision).
* Opposite-direction staked fixtures in the audit: **0**.

`config/contract_baseline.json` acknowledges max 2 rows for this rule; the 64 excess rows are what keeps the system
health check AMBER ("1 new violation").

## 5. Player-prop settlement — id alignment and provider equivalence

`output/player_ledger.csv`: 31,882 rows; 14,705 WIN/LOSS; 2,907 of those priced. Settlement code
(`update_results.update_player_results`, read, not imported) matches the ledger row to API-Football fixture stats **by
normalised player name**, with a fallback "first 5 characters contained in the other name", and a name that is not found
is settled **VOID (DNP)**. `player_id` (present on 13,918 settled rows) and `fixture_id` (10,765) are **not used** in
settlement.

Measured consequences (same `player_id`, same fixture, same market, two spellings, e.g. "Yan Valery"/"Y. Valery"):
* 1,056 duplicate groups (100% differ only in name spelling; 2,013 of the rows from October).
* **12 groups settled WIN for one spelling and LOSS for the other** — the same event graded both ways (e.g. Álvaro
  Marín / A. Marin, Celta Vigo II v Córdoba 2026-10-31, goals and SOT, both priced; Andrea Silipo / A. Silipo, Ascoli v
  Sampdoria). At least one side of each pair matched a different player.
* 441 LOSS/VOID and 250 VOID/WIN groups: one spelling found, the other silently VOIDed as "DNP".
* 120 priced settled rows sit in these groups (−25.40u); 6 groups have both rows priced and settled (double count in
  any priced ROI).
* **`sot2`/`sot3` never settle:** `_resolve_player_market` has no branch for them, returns "", and the row is skipped
  forever. 1,420 rows (233 priced): 0 ever WIN/LOSS, 635 permanently pending, 785 VOID via the DNP/postponed paths.
  Missing outcomes are left blank (correctly not losses), but every sot2/sot3 evaluation is empty.
* Cards are settled on `yellow_card >= 1` only; a straight red is a LOSS here and typically a WIN at a bookmaker.

Provider equivalence: `registry/settlement_alignment.json` already marks every prop market `UNVERIFIED`/`BLOCKED`
(OddsAPI does not expose the settling provider). Nothing in AU3 changes that, and the internal contradictions above make
it worse: **SETTLEMENT_UNVERIFIED** for all player markets; no prop ROI or CLV figure is quotable. Props remain paper
(invariant 2), so no real money was affected.

## 6. Bugs (Phase-11 fields)

| Bug ID | Sev | Repo | File/line | Root cause | Dates | Rows | Prod impact | Prob | Selection | ROI impact (same sample) | Status | Smallest safe fix | Regression test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AU3-06-A | P2 | v9 | `src/ledger.py` append_side_market_tips (pre 10-26) | re-see overwrote `odds`/`edge_pct`/`ev_pct` | 2026-08 → 10-26 | 92 side tips (91 settled; 70 BTTS) | settled & CLV'd at a later price | none | none | +2.35u flattering (NF BTTS +1.74u all-live, +1.25u since 09-01) | HISTORICAL (fixed 10-26; rows deliberately not rewritten) | report side ROI from first-sight history | `au3_first_sight.py` (0 odds mutations after 10-26) |
| AU3-06-B | P2 | v9 | `src/stake_plan.py` first-sight key `(policy_version, fixture, track)` + ab23726d5 track rename | renamed track = new key → same tip re-sized | 2026-10-26 (rename day); rows persist | 31 fixtures / 62 rows | paper allocation double counted (10.22u of 100) | none | none (ledger/Telegram unaffected) | none on ledger ROI; any stake_audit-based bankroll replay double counts | ACTIVE in the file / HISTORICAL in cause | key prior-stake lookup by `(policy_version, fixture, market, side)`; annotate the 31 pairs in readers, never rewrite rows | contract rule + a test that a track rename cannot re-stake a fixture already staked on the same side |
| AU3-06-C | P3 | v9 | `src/contracts/schemas.py:405` | contract says one position per fixture; owner policy allows same direction | since 10-25 | 2 fixtures | AMBER health noise | — | — | — | INTENTIONAL CONFIG mismatch | align contract wording to one goal direction (owner) | existing test_direction_guard |
| AU3-06-D | P2 (paper) | v9 | `update_results.py` update_player_results (name match + 5-char fallback; DNP on miss) | settlement by name, not player_id/fixture_id | all prop history | 1,056 dup groups; 12 contradictory WIN/LOSS; ~691 spurious VOIDs | props paper only | none | none | prop ROI/CLV NOT IDENTIFIABLE | ACTIVE | settle by `player_id` within API-Football `fixture_id`; treat no-id-match as UNRESOLVED not VOID | test: two spellings of one player_id settle identically; unknown name → UNRESOLVED |
| AU3-06-E | P3 (paper) | v9 | `update_results._resolve_player_market` | no branch for `sot2`/`sot3` | 2026-06-24 → now | 1,420 rows (233 priced) | none (paper) | none | none | evaluation of those markets impossible | ACTIVE | add `sot2`/`sot3` thresholds (>=2, >=3) | parametrised test over every market in the ledger |
| AU3-06-F | P3 | v9 | manual commit 075eeb3136 | rows deleted / results blanked by hand | 2026-06-22 | 23 vanished, 8 reset | ledger history not append-only then | none | none | pre-window | HISTORICAL | none (pre-window) | `au3_first_sight.py` ROW_VANISHED check |
| AU3-06-G | P3 | v9 | `update_results._names_match` (O/U and side settlement) | substring / first-word (≥4 chars) club matching, the trap invariant 11 forbids | all history | 0 measured errors in 2,024 independently verified rows | latent | none | none | none measured | SUSPICIOUS — NEEDS EVIDENCE (latent) | use `src.team_names.resolve_result` league-scoped | test: "Real X" never settles "Real Y" |

## 7. Verdict inputs

* O/U 2.5 and side-market settlement is **correct on every row AU3 could verify** (2,024 rows, 0 mismatches); the 129
  unverifiable rows are a coverage gap of the free score tables, not evidence of error.
* Reported O/U ROI is not inflated by settlement or price rewriting; reported **tier-level** ROI before 2026-10-26 is
  invalid (upgraded tiers). Side-market ROI is inflated by +2.35u from price rewriting.
* Player-prop settlement: **SETTLEMENT_UNVERIFIED**, with demonstrated internal contradictions.
