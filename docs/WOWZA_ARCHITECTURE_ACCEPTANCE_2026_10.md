# WOWZA architecture — final acceptance report (section 47)

Date 2026-10-25. Evidence: v9 `958a126e`, Pro `9b32e77`. **Documentation only.** Every answer
is drawn from code (file:line), committed artifacts (with their recorded `generated_at`) or commits.

## Verdict in one paragraph

The platform's governance layer is real and working:

* one model gate (PG-1.0) and one props gate (PG-PROPS-1.0);
* a research lock with an owner-approved baseline;
* first-sight logs with integrity digests;
* a daily health verdict;
* a freshness manifest;
* data contracts.

The **selection layer is not yet the designed one**. Production still selects with the legacy
raw-edge engine, and the market-first engine runs in shadow with unvalidated defaults. The track
registry is a read-model that production does not read, and the code is not physically organised by
layer. One automatic path still changes selection thresholds with no gate: the monthly
`backtest_matrix`. One real contract violation exists: a fixture staked twice across two runs. The
research lock passes (`protected_tracks_modified=False`), but it fingerprints policy, env, thresholds
and eligibility, **not selection code**. A code change such as the direction guard (`47716cb9`) is
therefore proven by its own non-regression run, not by the lock.

---

## Q1. The exact canonical production path, fixture → stake

1. **`.github/workflows/predict.yml`** step `python pipeline.py --mode predict` (line 225). The env
   (188–192) is `LEAGUE_SNIPER_CAP=0.12`, `MARKSMAN_THRESHOLD=0.08`, `VALUABLE_THRESHOLD=0.03`.
2. **`pipeline.py:main`** (1157–1183) calls **`mode_predict`** (750).
3. `mode_predict` does three things:
   * `load_all_matches()` (759);
   * `load_models(config.MODEL_FILE_STANDARD / MODEL_FILE_NEWFORMAT)` + side slots (761–769). It reads the serving `.pkl` (`config.py:35-36`), **not** `current_champions.json`;
   * calls **`src/predict.py::predict_upcoming`** (270).
4. `predict_upcoming` runs, in order:
   1. **`_fetch_odds_api`** (46): OddsAPI totals, pre-match only (96), first book's price per line (139–142), BTTS via `src/btts_odds` (181–185);
   2. `build_upcoming_features` (327);
   3. league routing (333–358). An unknown league falls back to the standard model;
   4. side and HT models;
   5. **`src/betting.py::evaluate_value`** (219), called at 437 and again after drift at 441–444.
5. Inside `evaluate_value`:
   * The edge is `edge_over = p_over − 1/ov_odds` (246–249).
   * `_base_tier` (140) assigns the tier. Its thresholds come from:
     * the env;
     * `config.LEAGUE_*` tables;
     * `models/best_params_standard.json` for `approved` leagues only (154–157).
   * Gates applied after tiering:
     * min odds 1.75;
     * NF cap 3.00 (292–301);
     * AVOID below `VALUE_THRESHOLD`;
     * blind-fixture guard (366–421);
     * drift up/downgrade `_apply_drift_adjustment` (198–214). *Removed 2026-10-26: since then this step only records `drift_shadow_tier` (`config.DRIFT_TIER_ADJUSTMENT = False`).*
6. Back in `mode_predict`, the selection and logging steps:
   1. `shadow_experiments.score` / `append_first_sight` (803–804);
   2. `_log_shadow` (809);
   3. `_log_h2h` (810);
   4. **`generate_bets`** (`betting.py:441`, called at 819);
   5. **`_generate_side_bets`** (78, called at 822). Here the edge is `p − (1/odds)/1.08` (125–126), the bars are 0.10/0.08 and VALUABLE is fixed at 0.04 (171–175);
   6. **`src/direction_guard.py::apply`** (134, called at 834).
7. `src/edge_engine_shadow.run` (857) runs here. It is shadow only and sits off the path.
8. **`_write_stake_plan` → `src/stake_plan.py::write`** (226, called at 869). It does three things:
   * `selections()` (122);
   * `prior_directions()` (196);
   * per match date, **`src/bankroll_policy.py::size_day`** (90).
9. `size_day` sizes each selection:
   * base fraction × tier multiplier (per-track override, 126–130);
   * zeroing reasons (131–144);
   * league-rating scale (145);
   * opposite direction (6b);
   * one position per fixture (178–186);
   * caps.

   It writes `output/stake_plan.csv` and `output/stake_audit.csv` (249–264).
10. `bets.csv` / `side_bets.csv` and the ledgers are written (837–847, 884–899). They go to Telegram via
    `telegram_bot/notifier.py` (instant, gated) and `daily_pack` (album by `src/tip_day`).

The stake is an **output**. It never changes a tip (`stake_plan.py:12-13`) and is not sent on
Telegram.

## Q2. Research-only modules

Each of these is verified to write only research artifacts and to have no reader on the predict,
stake or notify path:

* **market-first engine:** `src/market_edge.py`, `src/edge_engine_shadow.py`, `src/edge_af_quotes.py`;
* **research package:** `src/research/edge_replay.py`, `src/research/league_weekly.py`, `src/research/league_weekly_eval.py`, `src/research/first_sight.py` (shared infrastructure);
* **challenger:** `src/shadow_experiments.py`;
* **shadow logs and settlement:** `src/shadow_compare.py`, `src/shadow_settlement.py`;
* **allocation:** `src/allocation.py` (via `scripts/allocation_registry.py`);
* **keys:** `src/gameweek.py`;
* **governance (detectors, not selectors):** `src/research_lock.py`, `src/system_health.py`, `src/artifact_freshness.py`, `src/contracts/`;
* **builders and replay:** `scripts/build_league_regimes.py`, `scripts/build_track_registry.py`, `scripts/edge_engine_replay.py`;
* **all of Pro `src/studies/*`, `src/player_v3/*`, `src/price_trader/*`, `src/bankroll/*`, `src/scout/*`** (paper only, per their docstrings);
* **all of v11.**

Player props (`player_model/*`) are production code that is **paper-only by policy**
(`player_props` BLOCKED_FROM_REAL).

## Q3. PROTECTED tracks

One entry out of 100 (`config/league_market_regimes.json`, REGIME-1, evidence cutoff 2026-10-20):

* **Argentina Primera Division | BTTS | YES | new_format**
  * production ACTIVE, base fraction 0.0125;
  * n 81;
  * ROI +36.4%, 90% CI [+15.8%, +56.4%], +29.51 u.

The lock also covers these global facts, so they are effectively protected:

* the bankroll policy;
* the deployed env;
* resolved thresholds;
* eligibility;
* Cloudbet DRY mode.

## Q4. REPAIR

There are 6 REPAIR entries; none is in `repair_priority_high`. All are zero-stake already.

| Entry | Score | Reason |
|---|---|---|
| League One OU25 UNDER std | 0.351 | market beats model, dLL CI lower 0.01456 on 114 fixtures |
| League One OU25 OVER std | 0.265 | same |
| USA MLS BTTS YES nf | 0.196 | market beats model, dLL CI lower 0.00634 on 133 fixtures |
| League Two OU25 UNDER std | 0.165 | market beats model, dLL CI lower 0.00595 on 115 fixtures |
| League Two OU25 OVER std | 0.159 | same |
| League One BTTS YES std | 0.110 | market beats model, dLL CI lower 0.00081 on 101 fixtures |

## Q5. LEARNING

There are 93 LEARNING entries, broken down as follows:

* **43** have no evidence at all.
* **45** have some evidence but are below the adequacy bar (n ≥ 30 and ≥ 8 matchdays).
* **5** have adequate evidence without a credible result:

  | Entry | Production | n | ROI | 90% CI |
  |---|---|---|---|---|
  | Argentina OVER15 nf | ACTIVE_LOW | 69 | +0.3% | [−14.3%, +14.9%] |
  | MLS OU25 UNDER nf | PAPER_LEAGUE | 56 | −4.0% | — |
  | Bundesliga 2 OU25 UNDER std | PAPER | 34 | −20.6% | — |
  | La Liga 2 OU25 OVER std | PAPER | 34 | −13.7% | — |
  | Liga MX OU25 UNDER nf | ACTIVE_LOW | 30 | +21.3% | [−5.4%, +47.7%] |

* Serie B OU25 OVER std (PAPER; n 28 across 14 matchdays; ROI +31.3%, CI [+1.0%, +64.1%]) is the
  nearest to a PROTECTED-style result. It is still 2 signals short of the adequacy bar, and it is a
  standard track, which is PAPER by policy.

## Q6. Legacy components still in production

1. **The legacy raw-edge tier engine.** It covers both O/U 2.5 (`betting.py`) and side markets (`pipeline._generate_side_bets`). This is *the* selection engine.
2. ~~**Drift tier adjustment**~~ (`betting.py:198-214`). The earlier "drift gate miscalibrated" study found the rule was never backtested and points the wrong way (Confirmed −10.8% vs Neutral +8.0%). **REMOVED 2026-10-26** by owner decision (live settled tips: Confirmed −9.1% vs Neutral +13.6%); see `DEPRECATION_REGISTER.md` #15.
3. **`src/edge_engine.py` de-vig helpers** in props CLV (`src/clv_capture.py:24`).
4. **Synthetic `_DEFAULT_ODDS`** still filling side-market backtests. The rows are NO_PRICE, so they are never certified.
5. **The monthly per-league optimizer** (`backtest_matrix.yml` → `best_params_*.json`, auto-`approved`).
6. **`EDGE_CEILING`** config, which is dormant.

## Q7. What can change production automatically

| Path | What it changes | Guard | Can it fire now? |
|---|---|---|---|
| `retrain.yml` (`0 3 * * *`, Sundays only after 2026-10-05) → `pipeline._train_one` | serving team `.pkl` + pointer | PG-1.0. `submit_candidate` never promotes; PROMOTE only via `review_parked` after **holdout** (n ≥ 1000, ΔLL ≤ −0.001, CI upper < 0, Brier, ECE, slope, canary, league checks) **and a prospective period** (n ≥ 1000 per track, ΔLL < 0; early fail at 300; 180-day expiry) | Not for weeks. The only parked candidate (`newformat_20261025T082056Z_cfa68e5c`) started its prospective period on 2026-10-25. |
| `player_props_train.yml` (Sun–Tue 04:00) | serving props models | PG-PROPS-1.0 (same two periods, per market, n ≥ 1000 and ≥ 100 events) | No: all 8 markets KEEP_CHAMPION on 2026-10-25 and nothing is parked. Props remain PAPER regardless. |
| `backtest_matrix.yml` (`0 3 1 * *`) | **nothing since 2026-10-26.** It writes a PROPOSAL (`output/threshold_proposals/<YYYY-MM>.json/.md`); `models/best_params_*.json` change only via `python -m src.threshold_apply` on owner-approved entries (`config/threshold_approvals.json`, id + content hash). The same applies to the weekly `retrain.py` refit, which now writes a candidate only. | Owner approval + `research_lock` (threshold files are digest-checked against `config/threshold_applied.json`; any other change is RED) | Proposes on 2026-11-01; serves nothing without approval. |
| Drift (`odds_history_v9.json`), every predict run | **nothing since 2026-10-26** (owner decision). Was: tier up/downgrade per fixture. `drift_signal` is now information only; the old rule's answer is logged as `drift_shadow_tier` (`predictions.csv`, first-sight `output/drift_shadow.csv`). | the code constant `config.DRIFT_TIER_ADJUSTMENT = False` (`config.py:357`); a same-named env var is ignored with a warning (`betting.py:442`). Return only via a frozen Pro forward test + owner approval (`DEPRECATION_REGISTER.md` #15) | no |
| `auto_onboard_leagues.yml` (Mon 09:20) | `config/auto_leagues.json`, merged into ENABLED/NEW_FORMAT leagues (`config.py:721-756`) | its own criteria; `research_lock` fingerprints `AUTO_LEAGUES` | yes (list empty today) |
| Pro `league_ratings.json` fetched live (`stake_plan.py:96-119`) | stake scale per league | can only shrink a stake (min 0.2) | yes |
| Date-armed config | `REQUIRE_FORM_DATA_UNTIL=2026-09-15` re-armed the blind-fixture guard (`betting.py:401-410`); the retrain cadence guard switched daily → Sunday on 2026-10-05 | in code | already fired |
| `research_lock` | **nothing.** It is a detector that turns health RED and writes only on `--approve` (`research_lock.py:39-41,231-239`) | — | — |
| Pro bankroll allocator / learned tiers | nothing in v9 (v9 does not read `v9_bankroll_policy.json` or `learned_tiers`; grep) | — | — |
| `config/bankroll_policy.json` | nothing automatic; no workflow writes it | PR + owner | — |

## Q8. Proof that protected tracks are unchanged

* **Lock check**, run 2026-10-25 on this worktree:

  ```
  $ python -m src.research_lock --check
  protected_tracks_modified=False (approved by Nevo (owner) - approved in chat 2026-10-25 at 2026-10-25T12:09:26Z)
  exit=0
  ```

* **Approved baseline:**
  * `config/research_lock_snapshot.json`;
  * `approved_by "Nevo (owner) - approved in chat 2026-10-25"`, `approved_at 2026-10-25T12:09:26Z`;
  * `fingerprint_sha256 cbd42d57…7803`;
  * approved in `acc2ec4b` ("Fingerprint unchanged; only approved_by/approved_at updated"). The baseline created in `b7430a35` was explicitly an agent baseline, not an owner one.
* **System health** (`output/system_health.json`, 2026-10-25T13:28:49Z): `protected_tracks_modified false`; Protected tracks check GREEN.
* **Non-regression runs recorded in commits** after the lock existed:

  | Commit | What it proved |
  |---|---|
  | `065209bc` | v91_shadow and shadow logs identical on every column; stake_plan identical plus 9 trailing shadow columns |
  | `f08b699e` | shadow_experiments and shadow_h2h byte-identical; stake_plan differs only in shadow columns |
  | `23577263` | 6 migrated output files identical sha256 |
  | `47716cb9` | the only differences are 2 suppressed Over 3.5 tips; bets.csv, ledgers and stake_audit byte-identical |
  | `0ae781c7` | bets, side_bets, both ledgers, stake_audit and HT ledgers byte-identical; stake_plan's 25 served columns identical; lock check False |
  | `f67c3928`, `57f734c7` | "Nothing on the predict path changed"; lock check False |

* **Limit of this proof.** The fingerprint covers:
  * bankroll policy;
  * deployed env;
  * resolved thresholds;
  * eligibility;
  * Cloudbet mode;
  * each PROTECTED track's state, stake and gates.

  It does **not** hash selection code. `47716cb9` (direction guard) and `9c4ae829` (a new predict.yml
  step) landed after the approval without tripping the lock. They are covered by their own
  non-regression runs above, not by the lock.

## Q9. Research datasets accumulating prospectively

Counts on 2026-10-25:

| Dataset | Start | Rows now | Settled |
|---|---|---|---|
| `output/v91_shadow.csv` (incl. VALUABLE) | 2026-10-04 | 790 | 598 (VALUABLE 138 / 182) |
| `output/model_lineage/shadow_h2h.csv` (PG-1.0 period 2) | 2026-10-18 | 2,389 | via v91_shadow |
| `output/model_lineage/shadow_experiments.csv` (H-MODEL-01) | 2026-10-22T13:45Z | 324 (NF 173, std 151) | 134 (72 / 62) |
| `output/edge_engine_shadow.csv` (market_edge_v2) | 2026-10-25T09:25Z | 752 (188 fixtures × 4 markets) | 0 |
| `output/league_weekly_candidates.csv` / `league_weekly_research.csv` | 2026-10-25 | 752 / 60 | 0 |
| `output/stake_audit.csv` | 2026-10-14 | 472 | n/a |
| `output/top_picks_log.csv` | 2026-10-19 | 100 | — |
| `output/direction_conflicts.csv` | 2026-10-25 | 2 | — |
| `output/book_quotes/<date>.csv` (API-Football per-book) | 2026-10-04 | 22 day parts | — |
| `output/book_odds_snapshots.csv` (OddsAPI per-book, rolling) | 2026-10-16 (window) | 37,467 | — |
| `output/newformat_odds_dense.csv` | 2026-08-15 | 14,970 | — |
| `output/exchange_quotes/2026-10.csv` | October 2026 | 1 month part | — |
| `output/clv_records.csv` (props CLV) | 2026-07-13 | 6,351 | 3,469 with close |
| `output/fantasy_ledger.csv` / projection log | GW4 (2026-09-12) | 858 / 10,115 | 858 |
| Pro `edge2_forward_progress.json` (H-EDGE2) | 2026-10-26T00:00Z | 0 counted | 0 |
| Pro `candidate_forward_progress.json` (H-GATE) | 2026-10-20 | 6 forward bets of 1,200 | — |
| Pro props segments | 2026-10-14 | 80 tips | — |
| Pro Player V3 forward | 2026-10-15 | 16,760 props in window | — |
| v11 `v11_shadow_log.csv` | — | 1,875 (local, last 2026-10-20) | — |

## Q10. Next three highest-value research priorities

1. **Make the market-first forward test decidable, quickly and cleanly.** This means H-EDGE2 plus the market_edge_v2 shadow.
   * **Why:** it is the only route to retiring the legacy engine. The evidence against legacy is strong:
     * the model is overconfident by +13.64 pp;
     * the bookmaker's vigged price beats the model on Brier on its own selections;
     * OU25 tiers rank in reverse (SNIPER −9.9% < VALUABLE −0.8%).
   * **What blocks it:**
     * O/U 1.5 two-sided coverage is thin: NF 53/99 and std 12/84 after `0ae781c7`, so H-EDGE2-NF-O15-01 needs 150 settled rows;
     * there is no per-row config hash.

     The owner decisions D2 (config hash) and D3 (O/U 1.5 capture cadence) decide how fast and how
     clean this accrues.
2. **Fix H-MODEL-01 coverage and let it run.**
   * Challenger probability coverage is 58% (AMBER), so a third of the prospective sample is being lost.
   * On the first 72 / 62 settled rows, NF leans challenger (ΔLL −0.0043) and std leans champion (+0.0107). Neither has a CI yet.
   * Calibration is the biggest known lever across the estate (Player V3 finding; team-model overconfidence). An honest per-track result here decides the model layer's direction for next season.
3. **Separate league regime from selection skill on the money tracks.** These are NF BTTS (incl. the PROTECTED Argentina entry) and NF OU25.
   * Argentina BTTS is the only track with a credible positive ROI. Earlier analysis attributed it to the league, not to model selection.
   * The weekly league panel (0 settled so far) and H-EDGE2-NF-BTTS-01 are the instruments.
   * The result decides whether the one protected track is protected for a reason the model controls.

---

## Appendix — `python -m src.contracts` (2026-10-25, read-only)

The full output is in `docs/WOWZA_PLATFORM_ARCHITECTURE_2026_10.md` §5. In summary, there are 7
violations in 2 of 14 files:

* `book_odds_snapshots.csv` is missing 4 columns. The writer predates the contract.
* `stake_audit.csv` has two problems:
  * 110 pre-`b4ce9e22` zero-stake rows;
  * **one real double position** (rows 354/375, San Lorenzo v Newell's 2026-10-24).
