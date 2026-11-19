# Prediction ledger: Phase 1 audit (2026-11-17)

**Owner rule (Nevo, 2026-11-17):**

> "The models need to predict every match from the tracked leagues — not as a bet (that's the TOP-3 job), but every match result/props for every market. We need more predictions to learn from. If learning works only on the tips we give, we don't see if our bars are correct, because the other matches' predictions maybe hit better."

**What this audit covers.** It records, for every model × market in v9 (NevixAA/wowza-betting) and Pro (NevixAA/wowzaV9-Pro), plus v11:
- what is predicted, and when relative to kickoff;
- whether the prediction is persisted, append-only and settled;
- how much of the last 14 days it covers.

**Evidence.**
- Committed files on GitHub `origin/main`: v9 at `dcf8604a4`, Pro at `265cc78a`. Fetched 2026-11-17.
- Window: fixtures played **2026-11-03 .. 2026-11-16**.
- The overwritten board (`output/predictions.csv`) was unioned over **every committed version** in its git history: 101 commits since 2026-10-28, giving 11,852 pre-kickoff fixture-snapshots.

**Tracked leagues.** Registry `config/leagues.json`, `production_state` ∈ {ENABLED, ENABLED_PROVIDER_UNSUPPORTED, PAPER}. These 20 have the same ids on main and on data-truth-100:
- Championship, League One, League Two
- Bundesliga 2, Ligue 2, La Liga 2, Serie B
- Denmark Superliga, Austrian Bundesliga, Romanian Superliga
- Sweden Allsvenskan, Norway Eliteserien, Finland Veikkausliiga, Ireland Premier Division
- Argentina Primera Division, Brazil Serie A, Japan J-League, Mexico Liga MX
- China Super League, USA MLS

**Played fixtures.** From the committed history banks (`output/fd_history.parquet`, plus `output/af_history.parquet` for leagues football-data does not carry): **204 fixtures**. Board names were matched with `src/team_names.resolve_result`. Scripts and per-fixture CSVs are listed in the appendix.

## 1. Inventory: what is predicted, when, and what is kept

### v9 team models (Standard and New-Format)

| Model × market | Leagues | Predicted when | Persisted? | Append-only? | Settled? |
|---|---|---|---|---|---|
| v9_standard / v9_new_format **O/U 2.5** (`p_over25`) | Every board fixture | `predict.yml`: every 15–30 min (hourly overnight). Each run re-scores the whole board, horizon to about 7 days | `output/predictions.csv`: **overwritten every run**. First-sight copies are kept in `v91_shadow.csv` and `edge_engine_shadow.csv` | v91_shadow: yes, first sight only | v91_shadow: yes (202 of 222 window rows SETTLED). edge_engine_shadow: no result columns |
| **O/U 1.5** (`p_over15`), **O/U 3.5** (`p_over35`), **BTTS** (`p_btts`); each its own model and file | Every board fixture, both tracks | Same runs | predictions.csv (overwritten). edge_engine_shadow (first sight; carries prices) | edge_engine_shadow: first sight | **No.** Only the tipped sides reach `side_bets_ledger.csv` (757 rows, all time) |
| **Unders / BTTS-No** | Same | Same | Not stored as a selection anywhere. The value is 1 − p of the same model | — | Only when tipped |
| **HT O/U 0.5, HT O/U 1.5** (`p_ht_over05/15`) | Standard leagues only. New-Format has no HT model | Same runs | `ht_observations.csv` (first sight; carries prices) and `ht_ledger.csv` (tips only) | First sight | Tips only |
| **v9_props** (goals, goals2, assists, sot, sot2, sot3, sot4, cards) | Every props league, including the five COLLECTION_ONLY top leagues | `player_props.yml` (hourly); each run covers the near slate | `player_tips.csv`: **overwritten**, holding only players past the tip filters. `props_r2_shadow.csv`: **every scored player × market at first sight** (`p_served`) | r2_shadow: first sight, keyed for the R2 challenger | player_ledger: tips only. r2_shadow: settled by its own report |

### Pro and v11

The Pro and v11 inventory was traced from Pro `origin/main@265cc78a` and v11 `origin/main`.

| Model × market | Script / workflow | Coverage rule | Persisted / settled | Key |
|---|---|---|---|---|
| **paper_1x2** (Dixon-Coles `model_1x2.json`, trained to 2026-04-27) | `src/pipelines/paper_1x2.py` / `pro_paper_1x2.yml` (daily) | **Gated:** board names must match the model's training names and each team needs ≥8 matches. "Unrated team" skips 75–90% of fixtures | `paper_1x2.csv`, append by row_id; settled in-file. **Carries prices** | Pro fixture_key v1 (sha1 of names) |
| **Bet Builder Dixon-Coles grid** (fitted to v9's O1.5/O2.5/O3.5/BTTS) | `bet_builder.py generate()` / `pro_bet_builder.yml` (Fri–Sun ×3, Mon–Thu ×1) | A grid exists for **every** board fixture ("no model probs 0, inconsistent 0, unfittable 0"). **Nothing from the grid is stored** | — | — |
| **Bet Builder combos** | Same | **Gated:** leg/combo thresholds, top 12 per leg count, then the v9-direction gate (drops 20–80%) | `bet_builder_candidates.csv` **overwritten**; season store `combo_candidates` (run partitions); `bet_builder_settled.csv` | Pro fixture_key |
| **Specials** (full correct-score grid, 0–1 / 6+ / 7+, side 4+) | `src/specials/tracker.py`. **data-truth-100 only**, live since 2026-11-09 | Every rateable fixture | `specials_log.csv` (data-truth-100). **0 settled:** it reads v9 data-truth-100 history, which ends 2026-11-06 | Pro fixture_key v2 |
| valuable_shadow, ou_challenger | `pro_valuable_shadow.yml`, `pro_studies.yml` | v9's own probabilities. Sides with raw edge ≥ 0 only / ≥3 books only | Rewritten-cumulative / first sight, **not settled per row** | sha1 / fixture_key |
| Standard repair forward shadow | `pro_standard_repair_shadow.yml` | **Workflow disabled since 2026-11-01** | — | — |
| **v11 shadow O/U 2.5** (market-first blend of v9 p_over25 and de-vigged consensus) | `scripts/v11_shadow.py` / `v11_collect.yml` (hourly 08–23) | Every board fixture | `v11_shadow_snapshots.csv` append-only; graded in `v11_graded.csv` | sha1(date\|league\|match) |

**What no store holds today:**
- No store holds every model × market × fixture with the model version, fixed snapshots and a settlement, keyed so v9, Pro and props join.
- `predictions.csv` loses everything except the latest board; only git history keeps the rest.
- Unders, BTTS-No, the O1.5/O3.5/BTTS non-tips, HT non-tips and Pro's 1X2, correct-score and extreme-goals grids are never settled per prediction.

## 2. Coverage, 2026-11-03 .. 2026-11-16 (204 played fixtures in 20 tracked leagues)

### v9 team models: fixtures with a pre-kickoff prediction on some committed board

Each column is the number of played fixtures (out of "played") that had that probability on a pre-kickoff board.

| League | Played | On board | O2.5 | BTTS | O1.5 | O3.5 | HT 0.5 | HT 1.5 | Final ≤120 min (O2.5) | Median first-sight lead |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Championship | 25 | 25 | 25 | 25 | 25 | 25 | 25 | 25 | 8 | 155 h |
| League One | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 5 | 164 h |
| League Two | 11 | 11 | 11 | 11 | 11 | 11 | 11 | 11 | 8 | 164 h |
| Bundesliga 2 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 3 | 166 h |
| Ligue 2 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 1 | 138 h |
| La Liga 2 | 19 | 19 | 19 | 19 | 19 | 19 | 19 | 19 | 11 | 165 h |
| Serie B | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 7 | 160 h |
| Denmark Superliga | 6 | 6 | 6 | 6 | 6 | 6 | n/a | n/a | 2 | 159 h |
| Austrian Bundesliga | 6 | 6 | 6 | 6 | 6 | 6 | n/a | n/a | 4 | 164 h |
| **Romanian Superliga** | 8 | **0** | 0 | 0 | 0 | 0 | n/a | n/a | 0 | — |
| Sweden Allsvenskan | 9 | 9 | 9 | 9 | 9 | 9 | n/a | n/a | 6 | 160 h |
| Norway Eliteserien | 8 | 8 | 8 | 8 | 8 | 8 | n/a | n/a | 6 | 167 h |
| Finland Veikkausliiga | 3 | 3 | 3 | 3 | 3 | 3 | n/a | n/a | 0 | 139 h |
| **Ireland Premier Division** | 1 | **0** | 0 | 0 | 0 | 0 | n/a | n/a | 0 | — |
| Argentina Primera Division | 17 | 17 | 17 | 17 | 17 | 17 | n/a | n/a | 9 | 153 h |
| Brazil Serie A | 11 | 11 | 11 | 11 | 11 | 11 | n/a | n/a | 7 | 163 h |
| Japan J-League | 10 | 10 | 10 | 10 | 10 | 10 | n/a | n/a | 4 | 149 h |
| Mexico Liga MX | 10 | 10 | 10 | 10 | 10 | 10 | n/a | n/a | 5 | 164 h |
| China Super League | 8 | 8 | 8 | 8 | 8 | 8 | n/a | n/a | 8 | 166 h |
| USA MLS | 15 | 15 | 15 | 15 | 15 | 15 | n/a | n/a | 15 | 142 h |
| **Total** | **204** | **195 (95.6%)** | **95.6%** | **95.6%** | **95.6%** | **95.6%** | **92/92 (100%)** | **92/92** | **109/195 (56%)** | 159 h |

**How much of that is stored and settled per prediction today:**

| Market | Stored | Settled |
|---|---|---|
| O2.5 | First sight only (v91_shadow, 214 fixtures in the window) | Yes |
| O1.5 / O3.5 / BTTS | First sight only (edge_engine_shadow, 214 fixtures each, with prices) | **Not** settled |
| HT | First sight (ht_observations) | Tips only |
| Unders / No | Never as their own row | — |
| Every market | **No fixed final snapshot anywhere** | — |
| Every market | **No per-market model version** in any first-sight log except v91 | — |

### v9 props: players who actually played vs players the model predicted

| | Fixtures | Players who played | Scored by the model (any market) | In player_tips (after tip filters) |
|---|---:|---:|---:|---:|
| 20 tracked leagues | 202 | 5,614 | **5,498 (97.9%)** | 2,097 (37.4%) |
| 5 COLLECTION_ONLY top leagues (props only) | 48 | 1,351 | 1,335 (98.8%) | 632 (46.8%) |

"Scored by the model" counts the props model's own probability for every player × all 8 markets, from `props_r2_shadow.csv`, first sight only.

**player_tips by market, tracked leagues** (players who played):

| Market | Players in player_tips |
|---|---:|
| sot | 1,416 |
| cards | 898 |
| goals | 428 |
| sot2 | 179 |
| assists | 115 |
| sot3 | 3 |
| **goals2** | **0** |
| **sot4** | **0** |

Per league, model coverage of players who played runs from 95.0% (Ligue 2) to 100% (Ireland); tip coverage runs from 7.7% to 50.4%. The appendix CSV has every row.

### Pro and v11: fixtures with a stored prediction (out of the same 204 played)

| Output | Coverage | Main gap |
|---|---:|---|
| Pro board mirror (model_snapshots) | 96% | Romania absent (as on v9's board); Ireland is a name/home-away mismatch |
| paper_1x2 | **19%** | Unrated-team gate and a stale model |
| Bet Builder combos (any) | 65% | Thresholds and v9-direction gate; the Mon–Thu single pass sometimes runs after kickoff |
| ou_challenger | 95% | Needs 3+ books |
| valuable_shadow (any side) | 96% | Only raw edge ≥ 0 sides. Per market: BTTS 193, O2.5 160, O1.5 96, O3.5 47 |
| Standard repair shadow | 25% | Workflow disabled since 11-01 |
| Specials (data-truth-100) | 15% over the window; **97% (28/29) from its 11-09 start** | Only on data-truth-100; 0 settled |
| v11 shadow O/U 2.5 | = board (Romania absent) | — |

## 3. Every gap, named

| # | Gap | Where | Effect | Ledger answer |
|---|---|---|---|---|
| G1 | **No odds, so no board row: Romanian Superliga** (ENABLED_PROVIDER_UNSUPPORTED). OddsAPI has no events for it, and v9 builds its board from OddsAPI events | v9 team, and every Pro output that mirrors the board | 8 of 204 fixtures (3.9%) never scored by any team model | Recorded in the registry as `tracked_not_on_board` with the reason. Props still score it (218 of 226 players) |
| G2 | **Ireland: home/away reversed between sources** (board "Drogheda United v UC Dublin", banks "UC Dublin v Drogheda") | Board vs results | 1 fixture unmatched | Ledger rows are written. Settlement reports `NO_RESULT` with the reason instead of guessing a swap |
| G3 | **Overwritten board**: `predictions.csv` keeps only the latest run | v9 team | No append-only record of the T-final opinion, nor of O1.5/O3.5/BTTS/HT beyond first sight | Ledger: `first_sight` + `final` snapshot per model × market × selection, append-only |
| G4 | **Final pre-kickoff snapshot missing** for 86 of 195 fixtures: no committed run inside the last 120 min. The cause is run cadence (hourly overnight, and GitHub drops schedules), not code | v9 team | 44% have no near-kickoff opinion | `final` is written when a run lands inside 120 min, never back-filled; its absence is measurable in the registry |
| G5 | **Tier gate / tips-only settlement**: O1.5, O3.5, BTTS and HT are settled only when tipped (side_bets_ledger, ht_ledger) | v9 team | Bars judged only on the selected fixtures (the owner's selection-bias point) | Every row settled in `prediction_ledger_settled/` (log loss, Brier) |
| G6 | **Unders / BTTS-No never their own row** | v9 team | Under bars cannot be calibrated per selection | Both selections stored for every binary market |
| G7 | **No per-market model version** in the first-sight logs except v91; v9's `model_sha` stamp hashes size+mtime (changed 95× while the models changed 4×) | v9 team | Rows from different models pool silently | `model_version` from the board's per-market version column; `model_sha` = sha256 of the serving pickle's content |
| G8 | **HT only for Standard** (New-Format has no HT model) | v9 | No HT probability on any New-Format fixture | Correct by design: NF HT is not a market the model serves, so no rows |
| G9 | **Props tip filters**: MIN_SIGNAL_PROB 0.15 (removes almost all goals2/sot3/sot4), MIN_GAMES_SIGNAL, the starter proxy (minutes_pg), the GK skip for non-card markets, the injured list, the confirmed-lineup filter, the team-strength multiplier and caps | `player_model/predict.py` | Only 37% of players who played reach player_tips; goals2 and sot4 have 0 | Ledger records the model's own probability for **every scored player × every market**, before those filters |
| G10 | **Props players the model never scores (~2%)**: stale club (not in the current squad list, or unseen for STALE_DAYS+), no feature row, a fixture with no API-Football id | `player_model/predict.py` | 116 of 5,614 tracked players | Not written. There is no probability to store, and the player list never reaches the recorder. Counted here as the residual gap |
| G11 | **Props first sight only** (r2_shadow) | props | No lineup-time opinion | `final` snapshot for props (≤120 min, when lineups exist) |
| G12 | **Pro grids not logged**: the Bet Builder computes a full Dixon-Coles grid for every board fixture and keeps none of it | Pro | No 1X2 / correct score / extreme goals on 100% of the board | `pro_bb_dc`: 1X2, full correct score (0-0..6-6 + other), 0, 0–1, 5+, 6+, 7+, side 4+ for every board fixture; unfittable fixtures recorded with the reason |
| G13 | **Pro combos overwritten and gated**: only post-gate candidates survive, in an overwritten CSV | Pro | Gate effect unmeasurable | `pro_bb_combo`: every candidate before the v9-direction gate, settled by the builder's own combo grading |
| G14 | **paper_1x2 unrated-team gate** (19% coverage, stale model) | Pro | — | Not fixed here (a model problem, not a logging one). 1X2 is now logged for 100% of the board from the grid model |
| G15 | **Specials only on data-truth-100, 0 settled** | Pro | — | Same markets are in `pro_bb_dc` on main now. The specials tracker's own model is a follow-up entry after the go-live merge |
| G16 | **Not logged at all**: corners / cards models | v9 | No model exists yet | Pre-registered as `pending_model` (no rows). Turning one on is a config entry |
| G17 | **v11 shadow** keyed on its own sha1, O/U 2.5 only | v11 | — | Registered `pending_writer`. The v11 writer is a follow-up in wowza_v11 (no rows) |
| G18 | **Cross-repo key**: v9 uses OddsAPI event ids, Pro a name hash, props API-Football ids | all | No join | `fixture_ref`: `oa:<OddsAPI id>` for v9 team and Pro (Pro maps its fixture_key to v9's board id), `af:<id>` for props. API-Football ids for team rows need the data-truth-100 fixture index (`output/collect/upcoming_fixtures.parquet`), which reaches main at the go-live merge |

## Appendix: reproducing the numbers

All under the session scratchpad, at `C:\Users\nevo\AppData\Local\Temp\claude\1ee106cff9d5ca11\84b5f38e-4e7e-4236-9e42-6479e0aff4a9\scratchpad\pl\`:
- `audit_team.py`: unions every committed `predictions.csv` and matches played fixtures. Output: `aud\team_cov_by_league.csv`, `aud\team_cov_fixtures.csv`.
- `aud\props_cov.csv`: players who played (player_history, minutes > 0) vs `props_r2_shadow.csv` and every committed `player_tips.csv`.
- Pro: `..\pl_pro\scripts\cov2.py`, `coverage_pct.csv`.
