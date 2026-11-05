# Dashboard truth validation (Phase 2, KEY=dashtruth)

Run 2026-11-05 (UTC). Every value below was measured today on committed data by
`scripts/dashboard_truth.py`, with nothing copied from an earlier report. Raw output:
`reports/dashboard_truth/data_truth_100.json` and `reports/dashboard_truth/origin_main.json`.
No training, backtest, threshold change or new performance claim was made. The figures are what the
pages show, and the reproducer checks only that each one is computed correctly. A figure that
reproduces is still not evidence of edge.

## Verdict

- **data-truth-100 (with this branch): 58 of 58 figures MATCH.** On the first run, before the System Health label fix, it was 51 of 52 (figures were added afterwards).
  The one MISMATCH was a label: System Health showed "Contract violations 2" while its own table listed 8.
  That is fixed (section 5).
- **origin/main (what users see today, f9127e10): 22 MATCH, 27 MISMATCH, 9 UNIDENTIFIABLE.**
  - **24 MISMATCHes reproduce exactly on the old tier basis.** That basis is SNIPER + MARKSMAN, any track,
    and it is what main still calls "Staked". None of them is an arithmetic bug in that basis.
  - **The real staked book is much smaller.** THE definition (`src/staked.py`, owner 2026-11-04) applied to
    main's own committed ledgers gives **107 staked bets, -16.89u flat 1u**. Main shows **815 bets, -10.46u**
    under the word "Staked".
  - **The other 3 MISMATCHes are main-only code defects** that data-truth-100 already fixes. They are
    Model Performance's side-market claim, Overview's void-row n, and Home's tier count of upcoming tips
    (section 3).
  - **The 9 UNIDENTIFIABLE are labels main does not have yet** (NOT_SHOWN).
- **Main is the production dashboard. Its "Staked" figures overstate the money book until data-truth-100 is merged.**
  Nothing here changes main.

## 1. Method

`scripts/dashboard_truth.py` puts two values side by side for each figure:

- **Shown.** The page is rendered headlessly (`streamlit.testing` AppTest) in a child process. The render uses the
  target tree's own code and its committed `output/`. The value is read from the rendered metric, stat line or
  caption, which is the text a user sees.
- **Independent.** The same figure is recomputed from the same committed files with plain pandas.
  - No `dashboard_data`, page or `telegram_bot` module is imported.
  - The only shared code is `src/staked.py`, which is THE "Staked" definition and must be shared, plus
    `REAL_STATES` through it.
  - The cutoff (2026-08-10), the paper leagues and the thresholds are read from the target tree's `config.py`.
- **Comparison.** Each figure gets one of three results:
  - MATCH: the two agree within the display's own rounding (half a unit of the last shown digit).
  - MISMATCH: they disagree beyond that.
  - UNIDENTIFIABLE: the label is absent (NOT_SHOWN) or the input file is missing. This is never a pass.
- **Tier basis on main.** Every MISMATCHing "Staked" figure on main is also recomputed on the old tier basis.
  This tells a definition problem apart from a computation bug.
- **Clocks.** Clock-bound figures use the child's recorded `now`: stake-plan rows dated today or later, the
  last 7 days, and the Success Rates default window (7 days).
- **Trees.** data-truth-100 is the worktree `phase2/dashtruth` (base f66ebbe5 plus this branch). main is
  `git archive origin/main` (f9127e10) unpacked into a temp folder.

## 2. data-truth-100: every figure

| Page | Label | Shown | Independent | Source | Result |
|---|---|---|---|---|---|
| Home | Staked P/L (flat 1u) | -15.18u | -15.175 | bets_ledger.csv + side_bets_ledger.csv (live, settled, >= cutoff, pape | MATCH |
| Home | Settled staked bets | 97 | 97 | bets_ledger.csv + side_bets_ledger.csv (live, settled, >= cutoff, pape | MATCH |
| Home | Win rate — staked | 40.2% | 40.2062 | bets_ledger.csv + side_bets_ledger.csv (live, settled, >= cutoff, pape | MATCH |
| Home | ROI | -15.6% | -15.6443 | bets_ledger.csv + side_bets_ledger.csv (live, settled, >= cutoff, pape | MATCH |
| Home | P/L at recorded stakes | -7.92u on 40.88u staked | -7.9213 | bets_ledger.csv + side_bets_ledger.csv (live, settled, >= cutoff, pape | MATCH |
| Home | P/L at recorded stakes | -7.92u on 40.88u staked | 40.88 | bets_ledger.csv + side_bets_ledger.csv (live, settled, >= cutoff, pape | MATCH |
| Home | Drawdown from peak | -21.02u | -21.015 | bets_ledger.csv + side_bets_ledger.csv (live, settled, >= cutoff, pape | MATCH |
| Home | Last 7 days | -1.56u · n=32 | -1.56 | bets_ledger.csv + side_bets_ledger.csv (live, settled, >= cutoff, pape | MATCH |
| Home | Last 7 days | -1.56u · n=32 | 32 | bets_ledger.csv + side_bets_ledger.csv (live, settled, >= cutoff, pape | MATCH |
| Home | Active tips | 18 | 18 | stake_plan.csv rows dated today (UTC) or later | MATCH |
| Home | Upcoming staked tips (O/U 2.5) | 15 | 15 | stake_plan.csv rows dated today (UTC) or later | MATCH |
| Home | Paper / no stake | 38 | 38 | stake_plan.csv rows dated today (UTC) or later | MATCH |
| Home | Shadow engine BET | 0 of 56 | 0 | stake_plan.csv rows dated today (UTC) or later | MATCH |
| Home | Shadow engine BET | 0 of 56 | 56 | stake_plan.csv rows dated today (UTC) or later | MATCH |
| Home | System health | WARN | WARN | system_health.json checks[] recounted | MATCH |
| Home | Not staked | Not staked (every other settled tip, any tier) P/L: -8.19... | -8.19 | bets_ledger.csv + side_bets_ledger.csv (live, settled, >= cutoff, pape | MATCH |
| Home | Not staked | Not staked (every other settled tip, any tier) P/L: -8.19... | 1,552 | bets_ledger.csv + side_bets_ledger.csv (live, settled, >= cutoff, pape | MATCH |
| Home | Not staked | Not staked (every other settled tip, any tier) P/L: -8.19... | 1,204 | bets_ledger.csv + side_bets_ledger.csv (live, settled, >= cutoff, pape | MATCH |
| Home | Player rows zeros_unverified | 949,250 | 949,250 | output/corrections/history_cup_unproven_rows.csv, history_score_refuse | MATCH |
| Home | Unproven cup rows | 410 | 410 | output/corrections/history_cup_unproven_rows.csv, history_score_refuse | MATCH |
| Home | Refused scores | 6 | 6 | output/corrections/history_cup_unproven_rows.csv, history_score_refuse | MATCH |
| Overview | Staked P/L | -15.18u | -15.175 | bets_ledger.csv + side_bets_ledger.csv (live, settled, >= cutoff, pape | MATCH |
| Overview | Staked bets | 97 | 97 | bets_ledger.csv + side_bets_ledger.csv (live, settled, >= cutoff, pape | MATCH |
| Overview | P/L at recorded stakes | -7.92u | -7.9213 | bets_ledger.csv + side_bets_ledger.csv (live, settled, >= cutoff, pape | MATCH |
| Overview | new-format | -15.18u | -15.175 | bets_ledger.csv + side_bets_ledger.csv (live, settled, >= cutoff, pape | MATCH |
| Overview | standard | +0.00u | 0 | bets_ledger.csv + side_bets_ledger.csv (live, settled, >= cutoff, pape | MATCH |
| Overview | Staked | 97 | 97 | bets_ledger.csv + side_bets_ledger.csv (live, settled, >= cutoff, pape | MATCH |
| Overview | Hit | 40.2% | 40.2062 | bets_ledger.csv + side_bets_ledger.csv (live, settled, >= cutoff, pape | MATCH |
| Overview | Break-even | 47.2% | 47.1867 | bets_ledger.csv + side_bets_ledger.csv (live, settled, >= cutoff, pape | MATCH |
| Overview | ROI | -15.6% | -15.6443 | bets_ledger.csv + side_bets_ledger.csv (live, settled, >= cutoff, pape | MATCH |
| Dashboard | P/L — all staked markets (flat 1u) | -15.18u | -15.175 | bets_ledger.csv + side_bets_ledger.csv (live, settled, >= cutoff, pape | MATCH |
| Dashboard | Excluding O/U 2.5 | -2.75u | -2.745 | bets_ledger.csv + side_bets_ledger.csv (live, settled, >= cutoff, pape | MATCH |
| Dashboard | Settled bets | 97 | 97 | bets_ledger.csv + side_bets_ledger.csv (live, settled, >= cutoff, pape | MATCH |
| Success_Rates | Settled observations | 39 | 39 | bets_ledger.csv + side_bets_ledger.csv (live, settled, >= cutoff, pape | MATCH |
| Model_Performance | Overconfident by | +6.41pp | 6.4108 | bets_ledger (edge_pct/100 + 1/odds) + side_bets_ledger (model_prob), W | MATCH |
| Model_Performance | Model claimed | 0.5437 | 0.5437 | bets_ledger (edge_pct/100 + 1/odds) + side_bets_ledger (model_prob), W | MATCH |
| Model_Performance | Actually happened | 0.4796 | 0.4796 | bets_ledger (edge_pct/100 + 1/odds) + side_bets_ledger (model_prob), W | MATCH |
| Model_Performance | Settled sample | 1,641 | 1,641 | bets_ledger (edge_pct/100 + 1/odds) + side_bets_ledger (model_prob), W | MATCH |
| Player_Props | Win rate | 9% | 8.6207 | player_ledger.csv SNIPER + MARKSMAN, WIN/LOSS | MATCH |
| Player_Props | Settled | 116 | 116 | player_ledger.csv SNIPER + MARKSMAN, WIN/LOSS | MATCH |
| Player_Props | Flat ROI (1u, paper) | -57.8% | -57.7586 | player_ledger.csv SNIPER + MARKSMAN, WIN/LOSS | MATCH |
| Player_Props | Includes rows flagged | Includes rows flagged in output/corrections (kept, not re... | 79 | player_ledger.csv SNIPER + MARKSMAN, WIN/LOSS | MATCH |
| Leagues | P/L | -15.18u | -15.175 | bets_ledger.csv + side_bets_ledger.csv (live, settled, >= cutoff, pape | MATCH |
| Leagues | Staked tips settled | 97 | 97 | bets_ledger.csv + side_bets_ledger.csv (live, settled, >= cutoff, pape | MATCH |
| Leagues | Hit | 40.2% | 40.2062 | bets_ledger.csv + side_bets_ledger.csv (live, settled, >= cutoff, pape | MATCH |
| System_Health | GREEN | 20 | 20 | system_health.json checks[] recounted | MATCH |
| System_Health | AMBER | 1 | 1 | system_health.json checks[] recounted | MATCH |
| System_Health | RED | 0 | 0 | system_health.json checks[] recounted | MATCH |
| System_Health | NO_DATA | 0 | 0 | system_health.json checks[] recounted | MATCH |
| System_Health | Contract violations (new) | 2 | 2 | system_health.json checks[] recounted | MATCH |
| System_Health | Known (baseline) | 6 | 6 | system_health.json checks[] recounted | MATCH |
| Production | Tips sized | 56 | 56 | stake_plan.csv rows dated today (UTC) or later | MATCH |
| Production | With a stake | 18 | 18 | stake_plan.csv rows dated today (UTC) or later | MATCH |
| Production | Days covered | 5 | 5 | stake_plan.csv rows dated today (UTC) or later | MATCH |
| Market_Intelligence | Suppressed tips (all time) | 5 | 5 | direction_conflicts.csv rows | MATCH |
| Model_Info | Global Big gap | 12% | 12 | config.py thresholds | MATCH |
| Model_Info | OVER threshold | 12% | 12 | config.py thresholds | MATCH |
| Model_Info | UNDER threshold | 12% | 12 | config.py thresholds | MATCH |

Not every figure on these pages is in the reproducer:

- **Not reproduced: the per-league cards and the per-market tables.** The headline totals these sum to are
  reproduced above.
- **Not reproduced: the Tips board's own "Staked (policy stake > 0)" count.** It depends on the tip-day window
  of `telegram_bot.tips_table`.
- **Not reproduced: the CLV block** (median CLV, beat-the-close).
- **Covered through Home: the Tips page's settled section.** It reads the same function as Home
  (`today_view.staked_settled`), so the Home rows cover it.

## 3. origin/main (production today): every figure

"Independent (policy Staked)" is THE definition on main's committed ledgers and stake records. "Tier basis" is
SNIPER + MARKSMAN on any track, the rule main's pages still apply.

| Page | Label | Shown | Independent (policy Staked) | Tier basis (SNIPER+MARKSMAN) | Result |
|---|---|---|---|---|---|
| Home | Staked P/L | -10.46u | -16.885 | -10.465 | MISMATCH (shown -10.46 vs -16.885; REPRODUCES on the old tier basis (SNIPER + MARKSMAN, any track)) |
| Home | Settled staked bets | 815 | 107 | 815 | MISMATCH (shown 815.0 vs 107; REPRODUCES on the old tier basis (SNIPER + MARKSMAN, any track)) |
| Home | Win rate — staked | 47.2% | 40.1869 | 47.2393 | MISMATCH (shown 47.2 vs 40.1869; REPRODUCES on the old tier basis (SNIPER + MARKSMAN, any track)) |
| Home | ROI | -1.3% | -15.7804 | -1.284 | MISMATCH (shown -1.3 vs -15.7804; REPRODUCES on the old tier basis (SNIPER + MARKSMAN, any track)) |
| Home | P/L at recorded stakes | None | -8.6061 | - | UNIDENTIFIABLE (NOT_SHOWN: the page shows no figure under this label) |
| Home | P/L at recorded stakes | None | 46.62 | - | UNIDENTIFIABLE (NOT_SHOWN: the page shows no figure under this label) |
| Home | Drawdown from peak | -45.47u | -22.725 | -45.47 | MISMATCH (shown -45.47 vs -22.725; REPRODUCES on the old tier basis (SNIPER + MARKSMAN, any track)) |
| Home | Last 7 days | -11.57u · n=47 | -3.27 | -11.57 | MISMATCH (shown -11.57 vs -3.27; REPRODUCES on the old tier basis (SNIPER + MARKSMAN, any track)) |
| Home | Last 7 days | -11.57u · n=47 | 42 | 47 | MISMATCH (shown 47.0 vs 42; REPRODUCES on the old tier basis (SNIPER + MARKSMAN, any track)) |
| Home | Active tips | 11 | 11 | - | MATCH |
| Home | Upcoming staked tips (O/U 2.5) | 13 | 4 | - | MISMATCH (shown 13.0 vs 4) |
| Home | Paper / no stake | 53 | 53 | - | MATCH |
| Home | Shadow engine BET | 0 of 64 | 0 | - | MATCH |
| Home | Shadow engine BET | 0 of 64 | 64 | - | MATCH |
| Home | System health | WARN | WARN | - | MATCH |
| Home | Paper Small gap P/L | Paper Small gap P/L: -10.14u · 876 tips, not bets — track... | -3.72 | -10.14 | MISMATCH (shown -10.14 vs -3.72; REPRODUCES on the old tier basis (SNIPER + MARKSMAN, any track)) |
| Home | Paper Small gap P/L | Paper Small gap P/L: -10.14u · 876 tips, not bets — track... | 1,584 | 876 | MISMATCH (shown 876.0 vs 1584; REPRODUCES on the old tier basis (SNIPER + MARKSMAN, any track)) |
| Home | Not staked tips before the first policy record (caption) | None | 1204 | - | UNIDENTIFIABLE (NOT_SHOWN: the page shows no figure under this label) |
| Home | Player rows zeros_unverified | None | - | - | UNIDENTIFIABLE (NOT_SHOWN: the page shows no figure under this label) |
| Home | Unproven cup rows | None | - | - | UNIDENTIFIABLE (NOT_SHOWN: the page shows no figure under this label) |
| Home | Refused scores | None | - | - | UNIDENTIFIABLE (NOT_SHOWN: the page shows no figure under this label) |
| Overview | Staked P/L | -10.46u | -16.885 | -10.465 | MISMATCH (shown -10.46 vs -16.885; REPRODUCES on the old tier basis (SNIPER + MARKSMAN, any track)) |
| Overview | Staked bets | 814 | 107 | 815 | MISMATCH (shown 814.0 vs 107; does not reproduce on the tier basis either) |
| Overview | P/L at recorded stakes | None | -8.6061 | - | UNIDENTIFIABLE (NOT_SHOWN: the page shows no figure under this label) |
| Overview | new-format | +23.11u | -16.885 | 23.105 | MISMATCH (shown 23.11 vs -16.885; REPRODUCES on the old tier basis (SNIPER + MARKSMAN, any track)) |
| Overview | standard | -33.57u | 0 | -33.57 | MISMATCH (shown -33.57 vs 0; REPRODUCES on the old tier basis (SNIPER + MARKSMAN, any track)) |
| Overview | Staked | 607 | 107 | 608 | MISMATCH (shown 607.0 vs 107; does not reproduce on the tier basis either) |
| Overview | Hit | 49.8% | 40.1869 | 49.6711 | MISMATCH (shown 49.8 vs 40.1869; does not reproduce on the tier basis either) |
| Overview | Break-even | 47.5% | 47.3724 | 47.5054 | MISMATCH (shown 47.5 vs 47.3724; REPRODUCES on the old tier basis (SNIPER + MARKSMAN, any track)) |
| Overview | ROI | 3.8% | -15.7804 | 3.8002 | MISMATCH (shown 3.8 vs -15.7804; REPRODUCES on the old tier basis (SNIPER + MARKSMAN, any track)) |
| Dashboard | P/L — all staked markets | -10.47u | -16.885 | -10.465 | MISMATCH (shown -10.47 vs -16.885; REPRODUCES on the old tier basis (SNIPER + MARKSMAN, any track)) |
| Dashboard | Excluding O/U 2.5 | +31.53u | -2.695 | 31.525 | MISMATCH (shown 31.53 vs -2.695; REPRODUCES on the old tier basis (SNIPER + MARKSMAN, any track)) |
| Dashboard | Settled bets | 815 | 107 | 815 | MISMATCH (shown 815.0 vs 107; REPRODUCES on the old tier basis (SNIPER + MARKSMAN, any track)) |
| Success_Rates | Settled observations | 53 | 49 | 53 | MISMATCH (shown 53.0 vs 49; REPRODUCES on the old tier basis (SNIPER + MARKSMAN, any track)) |
| Model_Performance | Overconfident by | +7.99pp | 6.3433 | - | MISMATCH (shown 7.99 vs 6.34331) |
| Model_Performance | Model claimed | 0.5600 | 0.5435 | - | MISMATCH (shown 0.56 vs 0.543528) |
| Model_Performance | Actually happened | 0.4801 | 0.4801 | - | MATCH |
| Model_Performance | Settled sample | 1,683 | 1,683 | - | MATCH |
| Player_Props | Win rate | 9% | 8.6207 | - | MATCH |
| Player_Props | Settled | 116 | 116 | - | MATCH |
| Player_Props | Flat ROI (1u, paper) | -57.8% | -57.7586 | - | MATCH |
| Player_Props | Includes rows flagged (caption) | None | - | - | UNIDENTIFIABLE (NOT_SHOWN: the page shows no figure under this label) |
| Leagues | P/L | -10.46u | -16.885 | -10.465 | MISMATCH (shown -10.46 vs -16.885; REPRODUCES on the old tier basis (SNIPER + MARKSMAN, any track)) |
| Leagues | Staked tips settled | 815 | 107 | 815 | MISMATCH (shown 815.0 vs 107; REPRODUCES on the old tier basis (SNIPER + MARKSMAN, any track)) |
| Leagues | Hit | 47.2% | 40.1869 | 47.2393 | MISMATCH (shown 47.2 vs 40.1869; REPRODUCES on the old tier basis (SNIPER + MARKSMAN, any track)) |
| System_Health | GREEN | 20 | 20 | - | MATCH |
| System_Health | AMBER | 1 | 1 | - | MATCH |
| System_Health | RED | 0 | 0 | - | MATCH |
| System_Health | NO_DATA | 0 | 0 | - | MATCH |
| System_Health | Contract violations | 2 | 2 | - | MATCH |
| System_Health | Known (baseline) | None | 6 | - | UNIDENTIFIABLE (NOT_SHOWN: the page shows no figure under this label) |
| Production | Tips sized | 64 | 64 | - | MATCH |
| Production | With a stake | 11 | 11 | - | MATCH |
| Production | Days covered | 4 | 4 | - | MATCH |
| Market_Intelligence | Suppressed tips (all time) | 6 | 6 | - | MATCH |
| Model_Info | Global Big gap | 12% | 12 | - | MATCH |
| Model_Info | OVER threshold | 12% | 12 | - | MATCH |
| Model_Info | UNDER threshold | 12% | 12 | - | MATCH |

Why each main MISMATCH happens. Each cause is already fixed on data-truth-100, none is merged, and none was changed here:

| Group | Cause | Status on data-truth-100 |
|---|---|---|
| 24 "Staked" figures (Home, Overview, Tips, Leagues, Success Rates) | Main counts tiers (SNIPER + MARKSMAN, any track, PAPER tracks included) as "Staked". Every figure reproduces exactly on that basis. On THE definition the book is 107 bets, -16.89u flat 1u, and -8.61u at the recorded stakes (46.62u staked). | Fixed: one definition, `src/staked.py` |
| Overview: Staked bets 814 vs 815, track n 607 vs 608, Hit 49.8% vs 49.67% | `v9.performance` takes n from WIN/LOSS rows only. The one VOID row is in the P/L but not in n. | Fixed: staked_only reads `is_staked` with the recorded pnl |
| Home -10.46u vs Tips -10.47u for the same exact -10.465u | Home sums with floats (issue #19). Both values are within display rounding of -10.465, so the reproducer cannot flag this. It was found by reading the two pages side by side. | Fixed: both pages read `today_view` + `src.units` |
| Model Performance: "Model claimed" 0.5600 and "+7.99pp", vs 0.5435 and +6.34pp | Side-market claims were rebuilt as edge/100 + 1/odds, which adds the bookmaker margin. | Fixed 2026-11-02: `v9.calibration` uses the recorded `model_prob` |
| Home "Upcoming staked tips (O/U 2.5)": 13 shown vs 4 | It counts tiers in bets.csv, not policy stakes in the stake plan. | Fixed: `upcoming_staked_ou` reads `plan_staked` |

## 4. Labels that overstate (part 2)

| Where | Label | Problem | Done |
|---|---|---|---|
| main: Home, Overview, Tips, Leagues, Success Rates | "Staked", "Staked P/L", "Staked bets" | Tier basis with PAPER tracks included: 815 bets shown vs 107 policy-staked. | Fixed on data-truth-100. Production shows it until the merge. |
| System Health (both branches) | "Contract violations 2" | Counted only NEW violations, while the expander lists 8. | **Fixed here**: now "Contract violations (new) 2" plus "Known (baseline) 6". |
| Tips page (both) | Caption "Big gap -9.9%, Medium gap -5.8%, Small gap -0.8% since 26 Aug" | Typed numbers that nothing recomputes; the source is UNIDENTIFIABLE. | **Removed here**. The caption now points to the measured edge bands on the Pro page. |
| Home (data-truth-100) | "Settled since 2026-08-10" beside the staked figures | No tip before the first recorded policy decision (**2026-10-14 13:39 UTC**) can be staked. | **Fixed here**: the caption states the first recorded policy decision. |
| Home (data-truth-100) | "Not staked ... 1,552 tips" | The line mixes **1,204** tips that predate the policy with 348 policy-era NOT_STAKED tips. | **Fixed here**: the split is shown. |
| Home (data-truth-100) | "P/L at recorded stakes -7.92u on **+**40.88u staked" | A stake size was signed like a P/L. | **Fixed here**: now "40.88u staked". |
| Player Props (both) | "Win rate 9%, Settled 116, Flat ROI -57.8% (paper)" | **79 of the 116** rows are flagged in `output/corrections/props_sentinel_prob.csv`: the model probability was a cap value. | **Fixed here**: the page says so beside the record. The rows are kept, not removed, and no corrected rate is computed (no new performance claim while collection runs). |
| Tips page "Bar" column | "GLOBAL bar" | Now reads **Bar: GLOBAL** or **Bar: LEAGUE**. On committed data it is **UNIDENTIFIABLE** which tips are global-bar: no committed ledger row records `threshold_source`. The column is absent from the data-truth-100 snapshot and main's code does not write it, so no tip is shown as global-bar and none is guessed. | Label fixed; data pending |
| Home "Best bets" cards (both) | Tier chip with no bar | `output/top_picks_log.csv` and `todays_lists` have no `threshold_source` column, so a global-bar pick cannot be labelled on the card. Adding the column changes an append-only log's schema, so it was left out. | Open (question 2) |
| Paper tracks | Home and Leagues "Paper league", Props "PAPER · NEVER STAKED", Success Rates "player_props (PAPER)", Leagues VALUABLE shown as paper | Correct on data-truth-100. On main, PAPER *tracks* are inside "Staked" (first row). | - |
| Backtest vs live | Tips "Weekly table - live record and backtest" | Rows are labelled per kind (Live / Backtest), which is correct. Portfolio is a labelled simulation with a source filter. | - |
| Bankroll "Since-review ROI" | Pro's review rule (O/U 2.5 includes VALUABLE) | Not called "Staked": evidence_panels says so. Retrospective, and not validated by this audit. | - |

Every staked P/L on data-truth-100 is labelled "flat 1u" or sits beside a "P/L at recorded stakes" line.

## 5. Fixes in this branch (before and after, on committed data-truth-100 data)

| # | File | Before | After |
|---|---|---|---|
| F1 | `pages/19_🩺_System_Health.py` | "Contract violations" = 2 while the table lists 8 (MISMATCH) | "Contract violations (new)" = 2 (MATCH) and "Known (baseline)" = 6 (MATCH) |
| F2 | `pages/1_📊_Dashboard.py` | Typed ROIs -9.9% / -5.8% / -0.8% | Removed; no typed figure left |
| F3 | `dashboard_data/production_view.py` | `final_fraction.fillna(0.0)`: an unrecorded stake showed as 0.000% | The value stays missing. 0 rows are affected in today's committed plan, so this was latent. |
| F4 | `dashboard_data/tips_view.py` | Bar cell "GLOBAL bar" | "GLOBAL" or "LEAGUE" under the "Bar" header |
| F5 | `pages/00_🎯_Home.py`, `dashboard_data/today_view.py` | Cutoff-only caption; not-staked line unsplit; "+40.88u staked" | First policy decision (2026-10-14 13:39 UTC) stated; "1204 of them predate the first recorded policy decision"; "40.88u staked" |
| F6 | `pages/11_👤_Player_Props.py`, `dashboard_data/props_view.py` | No mention of flagged rows | "79 of 116 in props_sentinel_prob.csv" (independent recount: 79) |

## 6. Data truth panel (part 3)

`dashboard_data/truth_view.py` shows the full panel on System Health and a compact card, "D · Data truth", on Home.
It reads recorded artifacts only and computes nothing live. A missing input shows NO_DATA.

| Item | Value on data-truth-100 today | Source |
|---|---|---|
| Stoplight | **GREEN (STALE)**: the worst recorded check status, recorded 2026-11-01T19:25:27Z (about 91 h old). That artifact still says "35 known (baseline)"; today's gate run says 0 known and GREEN. | `output/data_truth_health.json`, which is newer than the Data truth lines of `system_health.json` |
| Player rows still zeros_unverified | **949,250**: 949,250 to verify, 0 re-fetched and stored, 6,354 refused by the plan. The run stopped at the 45,000-call cap. | `output/collect/player_history_refetch_audit.json` |
| Unproven cup rows (kept in history) | **410**: 342 with no cup fixture within ±1 day, 68 with the club not resolved | `output/corrections/history_cup_unproven_rows.csv` |
| Refused scores | **6** (ft_status_scores_differ) | `output/corrections/history_score_refused.csv` |
| Collection | HT fill: 33,992 rows. Shot stats: 67,915 filled (lacking_rows 32,014). Full stats: 75,528 (history_rows 93,385). Player re-fetch: 0 stored. | `output/collect/*audit*.json` |
| Last collection | **2026-11-05** | The newest af_budget.json day with calls recorded, or an audit's generated_at |

`python -m src.integrity.data_truth` now also writes `stoplight` and `gate_reasons` (the gate verdict) into
`data_truth_health.json`, so a reader never has to re-derive the verdict:

- **Today:** the panel shows the worst recorded status, labelled as such, until evidence_refresh writes a new
  artifact.
- **After that:** the panel uses the recorded gate verdict.
- **On main:** the panel would show NO_DATA for the collect and correction inputs, which exist only on
  data-truth-100.

## 7. Automated check (part 4)

`tests/test_dashboard_truth.py` has 13 tests:

- **9 tests on a synthetic output/ with known answers.** They compare the functions the pages call against the
  reproducer:
  - `today_view`: performance, action and upcoming.
  - `v9`: performance by track, and calibration.
  - `leagues.totals` and `production_view.stake_plan`.
  - `truth_view`: recorded values, NO_DATA when inputs are missing, and the labelled worst-status fallback.

  The synthetic data covers a PAPER-track row, a pre-policy row, a paper-league row, an unsettled row, and a
  VALUABLE-tier row that the policy staked.
- **3 tests of the checker itself.**
  - Drift beyond the display rounding is MISMATCH.
  - An absent label is UNIDENTIFIABLE.
  - Half-cent rounding is MATCH.
  - A figure that reproduces on the tier basis says so.
- **1 smoke test on committed data.** Home and System Health are rendered with AppTest, and every figure must
  MATCH.

To reproduce all 11 pages: `python scripts/dashboard_truth.py [--root <tree>] [--json <file>]`. It exits 1 on any MISMATCH.

## 8. Pages not reproduced here (source outside v9's committed data)

| Page | Figures come from | Status |
|---|---|---|
| Pro, Bet Builder, League Research, Bankroll (Pro review part) | wowzaV9-Pro committed JSON/CSV | UNIDENTIFIABLE here. The Pro 1X2 scoreboard and the Bankroll review figures were reproduced in P1_CROSS_REPOSITORY_CONTRACTS.md, and `pro.tip_scoreboard()` refuses a payload it cannot reproduce. |
| V11 Market | wowza_v11 committed files | UNIDENTIFIABLE here |
| Fantasy, Model Performance (FPL part) | player_model fantasy artifacts | UNIDENTIFIABLE here. These are FPL points, never betting units. |
| Model Lab | models/registry JSON | Display of recorded values; not reproduced |
| Live | Live API at render time | UNIDENTIFIABLE: not reproducible from committed data |
| Agent Analysis, Portfolio | User input, or a labelled simulation | Not a reported figure |

## 9. Open questions for the owner

1. **Merge order.** Production (main) shows 815 tier-basis bets as "Staked", while THE definition gives 107. The fix is on
   data-truth-100 and waits for the planned merge. No action was taken here.
2. **Global-bar label on Home's Best bets.** Showing the bar there needs a `threshold_source` column in
   `top_picks_log.csv`, which changes an append-only log's columns. Should I add the column, or keep the tier chip
   unlabelled until the ledgers record the bar?
