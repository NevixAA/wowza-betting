# P0 train/serve parity: validation of the rolling-form fix

Branch `phase2/parity` was started from `origin/data-truth-100` at **e07a4591**. Repo: wowza-betting. Fix under test: **1ea46192** (main), which is an ancestor of data-truth-100. Main was checked at **12bd71a3**. Date: 2026-11-03.

## Verdict

**The fix holds on data-truth-100.** For the same next match, serving now gives every rolling-form feature exactly the value training gives it. That covers all 26 `_FORM_FEATURES` columns plus `has_ht_history`, with missing values (NaN) in the same places. This held in all 30 adversarial cases and on 164 real matches (4,592 values, 0 mismatches). The fix did not change training at all.

**It is not finished everywhere.**
1. **Main still has a gap.** Main's training groups form by club name across leagues, while main's serving keeps form inside one league. For a newly promoted or relegated club, main's training sees its old league's games but serving does not. 2 of the 26 cases on main fail.
2. **The committed models were trained before either fix.** Until a retrain, they are still being fed slightly different numbers than they learned from.
3. **The board moves a lot.** On the current board, all 174 fixtures change. For New Format O/U 2.5, 24 of 102 bet decisions and 41 of 102 signal tiers change.

**Historical impact: UNIDENTIFIABLE.** The repo has no frozen history snapshot for each prediction run, so past boards cannot be replayed.

| Check | Result | Source |
|---|---|---|
| Adversarial cases on data-truth-100 | **30 / 30 pass** (plus the 4 existing P0-1 tests) | `tests/test_p2_parity_adversarial.py` |
| Same cases on main 12bd71a3 | 21 pass, **2 real failures** (promoted club), 3 cases test a feature main does not have | same file, run on main |
| Do the cases catch the old bug? | 8 of 10 sampled cases **fail on the pre-fix code** (1ea46192^) | mutation run, see §3 |
| Real matches, last 6 match days (2026-10-25..30) | **164 fixtures, 4,592 values, 0 mismatches** | `reports/p0_parity_evidence/real_history_meta.json` |
| Does the fix change training? | **No**: 0 of 71,146 rows differ on the 26 form columns (1ea46192^ vs 1ea46192) | §4 |
| Does data-truth-100 training differ from main's? | Yes, **547 rows**: P0-2 league scoping (clubs that played in 2 leagues) | §4 |
| Current board: fixtures with any form change | **174 / 174** | `reports/p0_parity_evidence/board_rescore_meta.json` |
| Current board: New Format O/U 2.5 bet decision changes | **24 / 102** (12 UNDER→AVOID, 6 AVOID→OVER, 6 AVOID→UNDER) | `reports/p0_parity_evidence/per_fixture.csv` |
| Current board: Standard O/U 2.5 | **0 changes** (served by the v9_repaired recipe, a separate path) | same |
| Historical replay | **UNIDENTIFIABLE**, see §6 | n/a |

## 1. What was tested

**The rule:** pick a match. Training sees it as a played row appended to the history (`build_features`). Serving sees it as an upcoming fixture (`build_upcoming_features`). Both must give the same value for:
- every column in `src.feature_engineering._FORM_FEATURES`: goals for and against, over-2.5 rate, shots, SoT, corners, fouls, xG, inside-box shots, HT goals for and against, HT over-0.5 and over-1.5 rates (13 stats × home/away = 26 columns);
- `home_has_ht_history` and `away_has_ht_history`.

**Tolerance:** absolute difference ≤ 1e-9. NaN must match NaN; NaN against a number is a failure. When training's frame has no HT data, it does not create the HT columns, so an absent column counts as NaN.

## 2. Adversarial synthetic cases (`tests/test_p2_parity_adversarial.py`)

Every stat in the synthetic histories differs by venue, so a window built from the wrong matches gives a different number.

| Case | What makes it hostile | data-truth-100 |
|---|---|---|
| Alternating home/away, 1/4/5/6/11 prior matches, served as home and as away (10 cases) | Venue-specific stats; fewer, exactly, and more than n=5 matches | pass. The value also equals a hand-computed chronological mean |
| Uneven counts + a club with no history | 2 vs 17 matches; newcomer | pass. Every form value is NaN and has_ht_history is 0 on both sides |
| One club name in two leagues, interleaved dates (2 cases) | 4-0 club in one league, 0-3 in the other | pass. Each league sees only its own games |
| Sparse xG, shots and HT | No xG in the whole window; 2 of 5 shot counts missing; half an HT score; one side with no HT in its window | pass. Missing stays NaN, never 0 |
| xG, inside-box and HT columns absent from history | Whole columns missing | pass |
| HT known only outside the window | HT scores only older than the last 5 games | pass. has_ht_history = 0 |
| Postponed (unscored) rows in the past | Two all-NaN rows inside the window | pass |
| Board name ≠ history spelling (3 cases) | Wolverhampton Wanderers/Wolves, Queens Park Rangers/QPR, Cadiz CF/Cadiz | pass |
| Newly promoted club (2 cases) | Last season in the Championship; "Sheffield Wednesday" next to "Sheffield United" | pass. No borrowing from another league, no near-name match |
| Two matches on one date in history | A second match for the club on the same day | pass |
| First match of a season | 3-month break | pass. Both paths carry form across the break |
| Unsorted history with shared dates | Shuffled rows | pass |
| **Known limit A (pinned):** a history row dated AFTER the fixture | Rescheduled unscored row dated later | **diverges.** Serving takes the club's last 5 rows whatever their date |
| **Known limit B (pinned, 4 cases):** history match on the fixture's own day | Training orders same-day rows home-first | **diverges only** when history is away and the fixture is home |

The two known limits are pinned. A test fails if the behaviour changes, so they cannot be relied on silently. Neither occurs in today's data. The merged history (71,147 rows, offline load of the committed caches) contains:
- **0 rows dated after 2026-10-30**, the latest match date; the board starts 2026-11-02;
- **0 (league, club, day) doubles.**

Run: `python -m pytest tests/test_p2_parity_adversarial.py tests/test_form_train_serve_parity.py -q` → **34 passed**.

## 3. Do the tests catch the old bug?

I swapped in `build_upcoming_features` from **1ea46192^** (pre-fix, read with `git show`) and ran 10 cases:
- **8 fail.** Example: alternating k=11, away side: training 1.8 against old serving 0.0. That is the "last 5 away games" bug.
- **2 pass on the old code too**, with good reason: with only 4 prior matches the old code's 4+4 list covers the same games, and the interleaved case uses constant values.

So the checks have a real failure condition.

**On main 12bd71a3:** the 2 promoted-club cases fail. Training gives 1.2 goals (from the Championship games); serving gives NaN. This is the P0-2 gap, fixed only on data-truth-100. The other 3 failures are only because main has no `has_ht_history` column.

## 4. Training was not changed by the fix

I built training on the same offline history (71,146 scored rows) three times:

| Comparison | Rows differing on any of the 26 form columns |
|---|---|
| 1ea46192^ (pre-fix) vs 1ea46192 (fix) | **0** |
| 1ea46192 vs data-truth-100 e07a4591 | **547** |

The 547 rows are P0-2: form is grouped by (league, club) instead of club name alone. They sit in League One (169), League Two (133), National League (82), Scottish League One (66), Scottish League Two (60) and Scottish Premiership (37). These are clubs with rows in two leagues.

**Consequence:** the committed models (`model_v9_newformat.pkl` 9187ca69, `btts`, `over15`, `over35`, `ht_*`) were trained with the main definition. Until a retrain, a promoted or relegated club's first five games in its new league are served differently from how those models were trained.

## 5. Current-board rescore (NOT history)

This is a **current-board rescore**, not a replay of what was served.

**Inputs:**
- the committed board `output/predictions.csv`: 174 fixtures, generated 2026-11-02T05:59Z by 68825e3c with the pre-fix code;
- the committed models;
- the merged history loaded offline from the committed caches (download skipped, network blocked), last match 2026-10-30.

**Method:** two arms, identical except for the form window:
- **NEW:** data-truth-100 serving.
- **OLD:** the same output, with the 26 form columns and the 13 columns computed only from them replaced by the 1ea46192^ code's values. Same history, same fixtures.

Both arms go through `src.predict.predict_upcoming` unchanged, with the env vars from predict.yml (`STANDARD_OU25_SERVING_RECIPE=v9_repaired`, thresholds 0.12/0.08/0.03). Script: `scripts/p2_parity_board_rescore.py`.

**How close is this to what was actually served?** The OLD arm matches the committed board's goals form on 161–165 of 174 fixtures, but shots, xG and HT form on only 81–117. Stats were collected after the board was made (round-6 inside-box and HT fill). That is why this is labelled a rescore on today's data.

| Output (New Format, 102 fixtures) | Changed | Mean abs change | Max |
|---|---|---|---|
| goals-for form (home / away) | 82 / 88 | 0.54 / 0.45 goals | 2.2 / 1.8 |
| p_over25 (served) | 102 | 3.8 pp (mean change +1.2 pp) | 13.4 pp |
| p_btts / p_over15 / p_over35 | 102 each | 2.1 / 2.0 / 2.9 pp | 8.5 / 9.9 / 13.4 pp |
| best side (OVER/UNDER) | 11 (8 UNDER→OVER, 3 OVER→UNDER) | | |
| signal_tier | 41 | | |
| bet | 24: 12 UNDER→AVOID, 6 AVOID→OVER, 6 AVOID→UNDER | | |
| bet totals | old: 37 UNDER, 3 OVER; new: 31 UNDER, 9 OVER | | |

| Output (Standard, 72 fixtures) | Changed |
|---|---|
| p_over25, edges, tiers, bets | **0** (v9_repaired path, not the legacy form) |
| p_btts / p_over15 / p_over35 | 72 each, mean 2.3 / 2.3 / 2.0 pp |
| p_ht_over05 / p_ht_over15 | 72 each, mean 0.35 / 0.63 pp |

The changed New Format bets are in Argentina (9), Sweden (5), Mexico (4), MLS (3), Brazil (2) and Japan (1).

New Format O/U Unders remain PAPER, as the owner rule requires. Nothing on the live board was changed by this work. Full per-column table: `reports/p0_parity_evidence/column_summary.csv`. Per fixture: `per_fixture.csv`.

## 6. Historical impact: UNIDENTIFIABLE

A point-in-time replay needs each run's exact history plus its exact model. The repo has neither in a usable form:
- **No frozen history per run.** `output/fd_history.parquet` is a rolling cache that later collection runs enriched. CI also merged a fresh football-data download and an API-Football cache that is not committed.
- **Unknown training cutoff.** The New Format champion is stamped `training_cutoff = UNKNOWN_PRE_GATE` on the board.
- **Registry gaps.** `models/registry` keeps 23–26 version JSONs per market, but those hold metadata only. Model files exist for just 1–2 champions per market: New Format has `newformat_20261018T080423Z_9eb6b4b4` and `newformat_20261025T082056Z_cfa68e5c`. The models that served most past boards are gone.

So how much the bug changed past tips and ROI **cannot be measured**. No figure is given.

## 7. Open issues

| # | Pri | Issue | Owner action |
|---|---|---|---|
| 1 | P0 | Committed models were trained before P0-2 league scoping. Serving now differs from their training for clubs that changed league; training at data-truth-100 differs from main on 547 rows. | Approve the clean retrain (GitHub Actions only) once data-truth-100 merges. Do not retrain locally. |
| 2 | P1 | Main 12bd71a3 still has the P0-2 gap: 2 adversarial cases fail there (promoted club gets cross-league form in training, none when served). | Merge data-truth-100 (P0-2) into main with the retrain in #1. |
| 3 | P1 | The fix moves 24 of 102 New Format bet decisions and 41 tiers on today's board. Past New Format results came from the old feature path, and their size is UNIDENTIFIABLE. | Treat New Format performance before 1ea46192 as a different system. Do not pool it with later results in any ROI/CLV claim (handled in P1_TRUSTWORTHY_PERFORMANCE). |
| 4 | P3 | Known limit A: serving ignores the fixture date. A history row dated after the fixture shifts the window. 0 such rows today. | Decide whether serving should cut history at each fixture's date. Pinned by `test_serving_window_ignores_the_fixture_date`. |
| 5 | P3 | Known limit B: a history match on the fixture's own day diverges in one of 4 role orders. 0 such rows today. | None unless same-day doubles appear. Pinned by `test_history_match_on_the_fixture_day`. |
| 6 | P3 | Unscored rows occupy a window slot in both paths. Parity holds, but the window then covers fewer than 5 played games. 1 such row in the merged history (2025-12-05). | Decide whether the loader should drop unscored past rows. |

## 8. Reproduce

```
python -m pytest tests/test_p2_parity_adversarial.py tests/test_form_train_serve_parity.py -q
python scripts/p2_parity_real_history.py <out_dir> 6      # real-data parity + training neutrality (writes real_history_meta.json)
python scripts/p2_parity_board_rescore.py <out_dir>       # current-board rescore (offline; restore output/*.json after: git checkout -- output/)
```

Both scripts block the network and skip the football-data download, using only the committed caches. They are light, single-process jobs (about 2–4 minutes each on the laptop).
