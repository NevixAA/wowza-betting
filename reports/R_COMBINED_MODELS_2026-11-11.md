# Combined models: team x props x Pro (research, v9)

Date: 2026-11-11. Branch `research/combined-models`. **Research only. Nothing changes in serving.** The owner decides any use.

* Code: `scripts/research_combined_models.py`
* Tests: `tests/test_research_combined_models.py`
* Development pre-registration: `config/experiments/combined_models/protocol.json`, sha256 `bc3db3b5116bbe65589cb13b483066efd8d11cc188598bac0125bba268338153`. It was committed in 78150ce4c before any residual, coherence or contradiction number was computed.
* Forward pre-registration: `config/experiments/combined_models/forward_protocol.json` (sha256 in the commit message of the commit that adds it).
* Result tables (counts and estimates only, no prices, no player rows): `reports/combined_models_data/`.

Owner's idea (2026-11-11): "combine all the models together to find an edge in matches. Maybe in a game with a higher % of many goals, it's more realistic to bet on SoT?"

## In plain language

1. **The idea as stated does not hold up.** When the team model expects many goals in the match, players do **not** hit SoT, SoT 2+ or score more often than the props model already says.
   * Per league, 0 of 120 cells for the match-total conditioners (team model P(O2.5), Pro Dixon-Coles total; 99 readable) survive Holm, and none has a CI above 0.
   * Pooled over leagues (context only), the team model's P(O2.5) points slightly the **wrong** way for SoT: -0.5 pp, CI [-0.9, -0.1].
2. **What does carry information is the player's own side, not the match total.** Pro's Dixon-Coles expected goals for the player's own team (`dc_lam_side`) adds to the props probability.
   * In the top third of side expectations, players beat the props model by about +1 pp more than in the bottom third. Pooled (context): SoT +1.3 pp, SoT 2+ +0.7 pp, anytime scorer +0.9 pp. Every pooled CI is above 0.
   * The direction holds in 50 of 57 readable league x market cells.
   * Only **one** per-league cell survives Holm: **Serie B, anytime scorer, +1.8 pp, 95% CI [+0.9, +2.7], Holm p 0.019.**
   * The effect sits in midfielders and defenders. For forwards it is about 0 (pooled SoT +0.3 pp, CI [-0.6, +1.3]), and forwards are where most props prices are.
3. **The likely reason is a props feature bug, not a betting edge.** The props feature `team_goals_pg_roll` is joined on the **opponent's** key, so it measures the opponent's scoring.
   * Its Spearman correlation is 0.35 with the opponent's expected goals and -0.03 with the own team's.
   * The props model therefore has no current-season own-team attack feature. Only `prev_season_team_goals_pg` covers own-team attack (0.43).
   * The composite `team_momentum_forward_matchup` ("team goals x opp conceded x forward") inherits the error. It is essentially uncorrelated with both sides' strength (0.01 / 0.02).
   * The side-expectation residual is what you would expect when this input is missing. The clean fix is a props feature fix, which is an owner decision. A combined-model bet would only patch around it.
4. **Coherence (do the models agreeing improve tips?).** No cell survives Holm.
   * The closest: La Liga 2, team O2.5 tips with Pro's 6+ flag on: +26.6 pp residual, CI [+11.5, +41.0], Holm p 0.074 (52 vs 154 fixtures).
   * Pooled (context): team O2.5 tips do better when the props models also see high attacking volume: +9.5 pp, CI [+3.6, +15.4].
   * Props tips do **not** do better when the team model agrees: -0.4 pp for SoT.
5. **The bug-finder (question 3) found four things worth fixing before any of this is used:**
   * **(a)** The goalkeeper `goals_conceded` reads **0** while the real score shows goals in 14% (Sep), 47% (Oct) and 84% (Nov, to the 9th) of 2026 fixtures. The figures are measured after the re-fetched counts are applied, which is what the props features read. It was at most 1.4% from 2025-10 to 2026-05. It feeds the **served** feature `opp_gk_save_rate`.
   * **(b)** `team_goals_pg_roll` is keyed to the opponent (point 3).
   * **(c)** The team model's O2.5 calibration is off by up to 12 pp in some leagues on the walk-forward months (Finland -11.7 pp, China -8.3, Argentina +7.2).
   * **(d)** When the team model strongly contradicts the others, it is usually because a promoted or relegated team plays in August (35% of the August cases vs 11% of normal August fixtures).
6. **Prices.** No development row has a props price. Prices start 2026-10-14, inside the spent window. So "does it beat the bookmaker" is **not testable on development data**. It can only be tested forward.
   * A forward paper protocol is registered for the one surviving cell (Serie B anytime scorer).
   * It is **void the day the props champion changes**, and fixing (a) or (b) changes it.
   * **My recommendation to the owner: fix (a) and (b) first and retrain, then re-run this study.** The forward protocol is there in case the owner prefers to watch the current champion.

## Inputs (exact versions)

| Input | Version | Clock |
|---|---|---|
| Team model | phase2-retrain-1.0 `champion_recipe`, stage `walk_forward` only. Standard `run_20261110T203611Z`, New-Format `run_20261110T203319Z` (local `p2-rebuild-1110`). p_over15/25/35, p_btts. A match scored by two folds keeps the earlier fold. | out of sample, fitted before each fold |
| Props model | The **served recipe** `props_harness.HGBNaNNative` (`hgb_nan_native`, promoted from run 42855107089), feature rule `all_prior_verified`, the 128 `PLAYER_FEATURE_COLS`. **Refitted on the first day of every month** 2025-01..2026-03 on all eligible rows dated before that day (train 462,671 → 770,325 rows). It is the served recipe, not the served pickle: the pickle was fitted through 2026-08-31 and is in-sample on these dates. | out of sample, monthly |
| Pro model | `wowzaV9-Pro` data-truth-100 `9cdfa207`, `src/specials/study.py` rule SPC-1 code unchanged (`_fit`, `predict_rows`, `flag_candidates`). Run on v9's repaired history exported from this v9 commit (`8c66793a3`, 71,618 matches). It was restricted to the 20 leagues that have both team and props data and to months 2023-07..2026-03 (the full 32-league run needed hours). The 6+ flag cutoffs therefore come from earlier months starting 2023-07, not 2019. | out of sample, monthly |
| Pro 1X2 | Not used. Its per-match walk-forward predictions are not stored; the Dixon-Coles lambdas carry the side strengths. | - |
| Outcomes | props labels from the harness frame (verified counts); team O2.5 from the team frame. These were cross-checked against the v9 scores: 0 disagreements in 41,052 fixtures. | after the match |
| Prices | `player_prop_odds_history.csv` + archive: **row counts only**, for the forward volume. | - |

**Development window: matches 2025-01-01..2026-03-31.** The team holdout (2026-04-01..11-08) and the props holdout and spent window (2026-09-01..) are never scored.

**Join.** v9 team names were mapped to API-Football names with a per-league crosswalk, learned from exact normalised names and score-proven same-date fixtures (543 pairs, `crosswalk` stage).
* 99.8-100% of development props rows in the 20 leagues found their fixture.
* The props rows of the Big-5 leagues, the UEFA cups and internationals have **no team or Pro model**. They are out of scope. Their rows count, but they are never joined.
* While building the crosswalk, two data issues turned up:
  * player_history spells some clubs two ways across seasons. Examples: `FC Nurnberg` 2023/24, then `1. FC Nürnberg`; also Kaiserslautern, Magdeburg, Fürth, Düsseldorf, Osnabrück, Heidenheim, Bochum, Mönchengladbach, plus one `León` row. The props features key teams by API-Football id, so this is cosmetic for the model. The crosswalk accepts two spellings only when their date ranges do not overlap.
  * The GK-derived score disagreed with the real score in 1,183 fixtures (issue (a)). 1,170 were kept because each side's player goals fit inside the real score (the gap is own goals). **13 were dropped.**

## 1. Inventory (development months, rows usable per question)

Full table: `inventory_by_league.csv`. Only the 20 leagues with team and Pro models are usable.

| league | SoT player rows | fixtures | Q1 rows (team terciles) | Q1 rows (side terciles) | Q2 team-high fixtures with props volume | Q2 SoT tips | 6+ flagged fixtures |
|---|---|---|---|---|---|---|---|
| Argentina Primera Division | 14,703 | 509 | 14,045 | 9,693 | 1 | 1,952 | 36 |
| Austrian Bundesliga | 6,778 | 243 | 5,253 | 3,921 | 32 | 809 | 7 |
| Brazil Serie A | 11,219 | 380 | 9,818 | 7,644 | 18 | 1,265 | 37 |
| Bundesliga 2 | 11,323 | 398 | 5,238 | 6,880 | 67 | 738 | 11 |
| Championship | 21,444 | 740 | 8,339 | 13,546 | 100 | 1,162 | 50 |
| China Super League | 6,925 | 240 | 6,325 | 4,325 | 65 | 971 | 42 |
| Denmark Superliga | 6,754 | 235 | 5,927 | 4,261 | 75 | 914 | 72 |
| Finland Veikkausliiga | 1,507 | 67 | 1,427 | 856 | 0 | 196 | 12 |
| Ireland Premier Division | 5,282 | 190 | 4,228 | 3,306 | 9 | 594 | 19 |
| Japan J-League | 11,096 | 380 | 8,964 | 7,219 | 30 | 1,233 | 17 |
| La Liga 2 | 17,431 | 594 | 9,814 | 10,872 | 144 | 1,431 | 86 |
| League One | 21,741 | 759 | 8,852 | 13,295 | 102 | 1,176 | 73 |
| League Two | 21,568 | 765 | 8,177 | 13,886 | 98 | 1,160 | 24 |
| Ligue 2 | 11,426 | 414 | 4,967 | 7,223 | 64 | 657 | 52 |
| Mexico Liga MX | 12,942 | 444 | 9,242 | 8,899 | 107 | 1,324 | 58 |
| Norway Eliteserien | 6,796 | 240 | 5,681 | 4,444 | 68 | 792 | 18 |
| Romanian Superliga | 11,414 | 402 | 9,394 | 7,658 | 40 | 1,467 | 29 |
| Serie B | 14,799 | 508 | 5,972 | 9,738 | 102 | 916 | 59 |
| Sweden Allsvenskan | 6,703 | 241 | 5,797 | 4,876 | 58 | 799 | 22 |
| USA MLS | 15,528 | 540 | 12,268 | 9,909 | 171 | 1,919 | 71 |

**Priced rows on development data: 0** (prices begin 2026-10-14).

Tercile groups use cutoffs from the league's **earlier** months only. Where a league's prediction distribution drifted, the groups are unbalanced, and that is reported, never re-cut. Example: the team model's P(O2.5) in Argentina fell below every earlier-month cutoff, leaving 1 top-tercile fixture.

## 2. Did the props model already use team strength? (double-counting check)

The props model's 128 features include these team or opponent inputs: `is_home`, `team_goals_pg_roll`, `opp_goals_conceded_pg`, `opp_sot_conceded_pg`, `opp_strength_index`, `opp_goals/sot_conceded_vs_fwd/mid_pg`, `prev_season_team_goals_pg`, `prev_season_team_conceded_pg`, `prev_season_opp_goals_pg`, `prev_season_opp_conceded_pg`, `team_corners_pg`, `box_threat_vs_leaky_defense`, `efficiency_vs_leaky_keeper`, `team_momentum_forward_matchup`, `quality_adj_goals_pg`, `quality_adj_sot_pg`.

It has **no match-level goal expectation and no market input.** Spearman correlations with Pro's side lambda (`props_team_features_vs_dc.csv`, 16,111 team-fixtures):

| feature | vs own side λ | vs opponent λ | reading |
|---|---|---|---|
| opp_goals_conceded_pg | 0.32 | 0.00 | opponent defence: as named |
| prev_season_team_goals_pg | 0.43 | 0.07 | own attack (last season): as named |
| prev_season_opp_conceded_pg | 0.39 | 0.05 | as named |
| **team_goals_pg_roll** | **-0.03** | **0.35** | **named "team", measures the OPPONENT's scoring** |
| team_momentum_forward_matchup | 0.01 | 0.02 | built on the line above: carries almost nothing |

The code shows why. `feature_engineering._TEAM_ROLL` maps `goals_scored -> team_goals_pg_roll`, and every `_TEAM_ROLL` feature is merged on `_okey` (the opponent), at `feature_engineering.py` line ~1316. That is right for the `opp_*` features and wrong for this one.

So **match-total conditioning double-counts** what the props model already has (the opponent's defence and the opponent's attack). **Own-side conditioning does not**: the model's only own-attack input is last season's.

## 3. Q1: residual vs the props model (y - p_props), top minus bottom tercile

One Holm family for Q1 + Q2: **340 per-league cells** (180 Q1 + 160 Q2). 239 are readable (≥ 30 fixtures in each group); unreadable cells enter with p = 1. 95% fixture-cluster bootstrap CIs (2,000 draws).

**Correction during the run.** The first run used the empirical bootstrap p-value. Its floor is 1/2,001, and 340 x 0.0005 > 0.05, so no cell could ever pass Holm. The p-values now come from the bootstrap standard error (normal form). The empirical share is kept beside it (`p_emp_one`), and a test covers this. The CIs and estimates did not change. The change made one cell pass: Serie B anytime, empirical p 0.0005 → SE-form p 0.0001.

### 3a. Conditioner = own side's expected goals (Pro Dixon-Coles), effect in percentage points, [95% CI], Holm p

| league | fixtures top/bottom | SoT 1+ | SoT 2+ | anytime scorer |
|---|---|---|---|---|
| Argentina Primera Division | 283/327 | +0.3 [-1.2, +2.0] H=1.00 | +1.0 [+0.2, +1.7] H=1.00 | +0.7 [-0.4, +1.6] H=1.00 |
| Austrian Bundesliga | 130/143 | -1.4 [-3.8, +0.9] H=1.00 | +0.2 [-1.3, +1.6] H=1.00 | +0.0 [-1.7, +1.7] H=1.00 |
| Brazil Serie A | 237/251 | +2.6 [+0.6, +4.5] H=1.00 | +1.2 [+0.1, +2.2] H=1.00 | +2.0 [+0.9, +3.1] H=0.22 |
| Bundesliga 2 | 188/242 | +0.0 [-1.9, +2.0] H=1.00 | -0.1 [-1.1, +1.0] H=1.00 | -0.1 [-1.4, +1.3] H=1.00 |
| Championship | 414/452 | +1.2 [-0.1, +2.5] H=1.00 | +1.0 [+0.3, +1.8] H=1.00 | +0.9 [+0.0, +1.7] H=1.00 |
| China Super League | 166/112 | +2.6 [+0.3, +5.1] H=1.00 | +0.1 [-1.3, +1.5] H=1.00 | -0.2 [-1.6, +1.4] H=1.00 |
| Denmark Superliga | 190/85 | +2.6 [+0.3, +5.0] H=1.00 | +1.5 [+0.1, +2.9] H=1.00 | +0.7 [-1.0, +2.4] H=1.00 |
| Finland Veikkausliiga | 46/26 | too thin | too thin | too thin |
| Ireland Premier Division | 121/103 | +3.9 [+1.2, +6.7] H=1.00 | +1.7 [+0.0, +3.4] H=1.00 | +1.4 [-0.3, +3.1] H=1.00 |
| Japan J-League | 180/238 | +1.2 [-0.7, +3.0] H=1.00 | +0.3 [-0.6, +1.2] H=1.00 | +1.1 [-0.2, +2.3] H=1.00 |
| La Liga 2 | 397/238 | +1.0 [-0.5, +2.4] H=1.00 | +0.8 [+0.1, +1.5] H=1.00 | +0.8 [-0.3, +1.7] H=1.00 |
| League One | 444/404 | +1.3 [-0.1, +2.7] H=1.00 | +0.9 [+0.2, +1.6] H=1.00 | +1.1 [+0.2, +2.0] H=1.00 |
| League Two | 370/494 | -0.3 [-1.7, +1.2] H=1.00 | +0.4 [-0.3, +1.2] H=1.00 | +0.5 [-0.4, +1.5] H=1.00 |
| Ligue 2 | 236/232 | +0.2 [-1.7, +2.0] H=1.00 | +0.8 [-0.2, +1.8] H=1.00 | +0.2 [-1.0, +1.4] H=1.00 |
| Mexico Liga MX | 303/278 | +1.9 [+0.2, +3.7] H=1.00 | +0.7 [-0.2, +1.6] H=1.00 | +1.2 [+0.2, +2.2] H=1.00 |
| Norway Eliteserien | 139/153 | +3.2 [+0.7, +5.8] H=1.00 | +1.6 [+0.3, +3.0] H=1.00 | +2.7 [+1.0, +4.5] H=0.93 |
| Romanian Superliga | 224/278 | +1.9 [-0.1, +4.1] H=1.00 | +1.2 [+0.1, +2.3] H=1.00 | +0.7 [-0.4, +1.9] H=1.00 |
| **Serie B** | 312/306 | +2.6 [+0.9, +4.2] H=0.80 | +0.6 [-0.2, +1.4] H=1.00 | **+1.8 [+0.9, +2.7] H=0.02** |
| Sweden Allsvenskan | 158/160 | +3.2 [+1.0, +5.5] H=1.00 | +0.3 [-1.0, +1.5] H=1.00 | +1.7 [+0.1, +3.3] H=1.00 |
| USA MLS | 341/282 | +0.3 [-1.1, +1.8] H=1.00 | -0.3 [-1.3, +0.6] H=1.00 | -0.2 [-1.2, +0.8] H=1.00 |
| ALL (context) | 4,879/4,804 | +1.3 [+0.9, +1.7] | +0.7 [+0.5, +0.9] | +0.9 [+0.7, +1.2] |

Context (never a decision; `context.json`):
* **Logistic recalibration slope per SD of side λ:** SoT b = +0.039 (SE 0.006), SoT 2+ +0.063 (0.010), anytime +0.049 (0.008). All p < 1e-9. The same for match total: P(O2.5) b = -0.017 (SoT), Pro total -0.002. The match total carries nothing extra, and side λ does.
* **Venue:** the effect is present for home and away players alike (SoT home +1.1, away +1.8 pp). It is not an `is_home` artefact.
* **Position** (pooled SoT; anytime scorer): F +0.3 [-0.6, +1.3]; M +1.2 [+0.5, +1.9]; D +2.3 [+1.7, +2.9]; anytime M +1.7, D +1.0, F -0.3 [-1.0, +0.5]. In Serie B anytime: M +2.4 [+1.0, +3.9], F +2.0 [-0.6, +4.6], starters only +2.3 [+1.1, +3.4].

### 3b. Conditioner = match goal expectation (the owner's idea)

| conditioner | readable per-league cells | positive | raw CI above 0 | Holm-surviving | pooled SoT | pooled SoT 2+ | pooled anytime |
|---|---|---|---|---|---|---|---|
| team P(O2.5) | 45 of 60 | 15 | 0 | 0 | -0.5 [-0.9, -0.1] | -0.1 [-0.3, +0.1] | +0.1 [-0.2, +0.3] |
| Pro λ home + λ away | 54 of 60 | 23 | 0 | 0 | -0.2 [-0.6, +0.2] | +0.0 [-0.2, +0.2] | -0.1 [-0.3, +0.2] |

Per-league rows: `tests.csv` (`Q1_team_p_over25`, `Q1_dc_lam_total`). **Answer to the owner's question: no.** A high-scoring match on its own does not make SoT more likely than the props model says.

### 3c. Versus the bookmaker

**Not testable on development data:** 0 priced development rows. On the spent window (2026-10-14..11-08) prices exist but outcomes there are spent. They are only counted, for the forward plan (section 6).

## 4. Q2: coherence (agreement defined in advance, per league)

Definitions (protocol, before results):
* **team_high / team_low:** P(O2.5) ≥ the league's earlier-month 2/3 quantile, or ≤ the 1/3 quantile.
* **props volume V:** the mean served-recipe P(SoT) of the fixture's recorded starters, with ≥ 18 starters required. V high/low uses the same earlier-month terciles.
* **6+ flag:** Pro SPC-1 `cand_ge6`.
* **Props tip:** a starter above the league's earlier-month 0.8 quantile of p_props.

| test | readable / searched | Holm-surviving | best cell | pooled context |
|---|---|---|---|---|
| T2a team O2.5 tips: props volume high vs low | 2 / 20 | 0 | USA MLS +21.7 pp [+5.7, +37.6], Holm 1.00 | +9.5 [+3.6, +15.4] |
| T2c team O2.5 tips: 6+ flag on vs off | 6 / 20 | 0 | **La Liga 2 +26.6 pp [+11.5, +41.0], Holm 0.074** (52/154 fixtures) | +5.1 [-0.2, +10.2] |
| T2b props tips: team high vs low (sot / sot2 / anytime) | 42 / 60 | 0 | Japan SoT +4.2 [-3.7, +11.9] | -0.4 / -1.2 / -0.3 |
| T2d props tips: 6+ flag on vs off (sot / sot2 / anytime) | 33 / 60 | 0 | Ligue 2 SoT +6.6 [-0.6, +14.2] | +1.7 [+0.1, +3.3] / -0.6 / -0.6 |

Reading:
* Agreement helps the **team** side's O2.5 tips a little in the pooled view, but never in a single league after Holm.
* It does **not** help props tips.
* Most per-league Q2 cells are thin, because team_high x props_high is about a third of a third of the fixtures.
* Side note (context): props tips have a mean residual of +3 to +5 pp in almost every league. The served recipe is under-confident at the top of its range on these months. That is a calibration point for the props owner, not a combined-model effect.

## 5. Q3: contradictions as a bug-finder

**Agreement between the models is low.** Correlations per league, fixture level, development months (`q3_model_agreement_by_league.csv`):
* team total vs Pro total: 0.17-0.53
* team total vs props volume: 0.22-0.53
* Pro total vs props volume: 0.08-0.53

Two models of the same expected goals agreeing at about 0.3 is weak.

**Strong contradictions** (|z difference| ≥ 2.5 within league): 482 of 7,946 fixtures (`q3_patterns.csv`). The realised totals sit near the league average, so in a strong contradiction the extreme side is usually wrong:

| pattern | fixtures | goals | O2.5 rate | team P(O2.5) | Pro total | props V |
|---|---|---|---|---|---|---|
| none | 7,464 | 2.64 | 0.503 | 0.493 | 2.60 | 0.218 |
| team HIGH / Pro LOW | 90 | 2.38 | 0.467 | 0.538 | 2.02 | 0.227 |
| team LOW / Pro HIGH | 125 | 2.74 | 0.512 | 0.456 | 3.47 | 0.214 |
| team HIGH / props LOW | 167 | 2.66 | 0.497 | 0.557 | 2.76 | 0.192 |
| team LOW / props HIGH | 100 | 2.50 | 0.490 | 0.453 | 2.70 | 0.250 |

Patterns and what they point to:

1. **Goalkeeper `goals_conceded` = 0 against the real score (data bug, serving).**

   | month | fixtures | GK side shows 0 while the real score shows goals |
   |---|---|---|
   | 2025-10..2026-05 | 3,951 | 0-1.3% |
   | 2026-07 / 08 | 153 / 591 | 7.2% / 6.9% |
   | **2026-09** | 653 | **14.1%** |
   | **2026-10** | 751 | **46.6%** |
   | **2026-11** (to the 9th) | 91 | **83.5%** |

   * The counts are measured after `history_refetch.with_refetched`, so they are the counts the props features read (`gk_goals_conceded_defect_by_month_after_refetch.csv`). The raw parquet is similar (`..._by_month.csv`).
   * 1,038 of the 1,045 defective GK rows since September are `refetched`, so the re-fetch itself returns 0.
   * The value feeds `gk_save_rate = saves / (saves + goals_conceded)` → the served feature **`opp_gk_save_rate`** (inflated toward 1.0), and `goals_conceded_gk_pg`.
   * It barely touches the development window used here, but it touches every current serving row.
   * Suggested fix (owner): take conceded goals from the fixture score (`af_ht_scores` / v9 history), never from the GK row. A 0 that contradicts the score should become missing (no-defaults rule).
2. **`team_goals_pg_roll` keyed to the opponent (feature bug, serving).** See section 2. It explains the side-λ residual in section 3a.
3. **Promoted and relegated teams in August.** "team HIGH / props LOW" jumps to 6.3% of August fixtures (0.5-3.1% in other months). 35% of those August cases involve a team with < 10 league matches in the previous year, vs 11% of normal August fixtures. The team model's last-5 form carries the other division's results across the season boundary. The props model's per-player features do not.
4. **Team model O2.5 calibration on its own walk-forward months** (mean P vs actual O2.5, 2025-01..2026-03):
   * Finland: 0.554 vs 0.670 (-11.7 pp)
   * China: -8.3 pp
   * Bundesliga 2: -5.4 pp
   * Denmark, Liga MX, Eredivisie: about -5 pp
   * Argentina: 0.396 vs 0.324 (+7.2 pp). In 2023-24 it was 0.49 vs 0.30-0.33.

   This is a leak-free check, and the labels match the scores exactly (0 / 41,052 disagree), so it is a model-level miscalibration, not a label bug. It also unbalances the earlier-month tercile groups (Argentina: 1 top-tercile fixture).
5. **Names.** Some API-Football clubs are spelled two ways across seasons in player_history (section "Join"). This is cosmetic for the features (they use team ids), but it breaks any join by name.

## 6. Forward paper protocol (pre-registered, per league)

File: `config/experiments/combined_models/forward_protocol.json` (COMBINED-MODELS-FWD-1). Start: kickoffs from 2026-11-12. **Void if the served props champion changes.** Fixing pattern 1 or 2 changes it, and I recommend fixing them first.

* **Family A (Serie B, anytime scorer), single look 2027-02-28, Holm over 3 cells.**
  * A1: flat 1u on every priced row whose own side λ ≥ 1.332 (frozen: the dev 2/3 quantile of the 1,974 Serie B side lambdas). ROI.
  * A2: the same rows, only when p_served x price ≥ 1. ROI.
  * A3: replication of the residual, top (≥ 1.332) minus bottom (≤ 1.0445) tercile. No price needed.
  * **PROVEN** (A1 or A2 only): ≥ 150 settled bets, 95% fixture-cluster CI of ROI above 0, and Holm p < 0.05.
* **Family B (replication), single look 2027-06-30, Holm over 56 cells:** the side-λ residual per league x market for every league whose development cell was readable. The frozen cutoffs per league are in `forward_cutoffs.csv`.

**What is testable now vs later** (counts only; `forward_price_volume.csv`):

| item | volume | when |
|---|---|---|
| A1 (Serie B priced anytime rows, top-tercile sides) | ~650 distinct priced anytime rows/week in Serie B in 2026-10-14..11-08, about one third of them in top-tercile sides | 150 bets in about **1-2 match weeks**; read at 2027-02-28 |
| A2 (the same, p_served x price ≥ 1) | at the ledger's Serie B anytime rate (~9 priced tips/week, all sides) about 3-5/week in top-tercile sides | **30-50 match weeks** → expected **INSUFFICIENT** at 2027-02-28, reported as such |
| A3 / family B residual cells | need ≥ 30 fixtures per tercile group (~60-90 fixtures per league) | Serie B, Championship, League One/Two, La Liga 2: ~2-3 months. The Nordic leagues, Finland, Ireland, China, Japan, Brazil, Argentina and MLS are out of season for much of the window and may stay unreadable at 2027-06-30 |
| Big-5 leagues, UEFA cups | no team or Pro model | not testable |

**Blockers at registration** (also in the protocol):
* No pre-kickoff served-props log is running. `props_r2_shadow.csv` was written only on 2026-11-01/02.
* Pro's specials tracker has 2 logged rows, on data-truth-100 only.

Both inputs can be recomputed leak-free afterwards, and the protocol says how: the frozen champion pickle on pre-match features, and SPC-1 refitted on earlier months. Each row records which source it came from. Pattern 1 contaminates every forward served probability until it is fixed.

## Holm accounting

* Development family: **340 cells** (Q1 180 + Q2 160; per league; unreadable cells enter with p = 1). One cell survives: **Q1 side λ, Serie B, anytime scorer (Holm p 0.019).**
* Not in the family (context only, stated as such): the pooled rows, the logistic slopes, the venue and position splits, the Serie B position split, and the Q3 descriptives.
* Configurations tried outside the registered family:
  * the empirical bootstrap p-value (replaced, see section 3);
  * a first crosswalk that refused season-separated spellings (Bundesliga 2: 59% of names). It was fixed before any test was run.

No other variant was run.

## What is NOT claimed

* Nothing here is PROVEN, and no development row has a price.
* The Serie B cell is one survivor out of 340, with a plausible mechanical cause (section 2). It earns a forward paper check, not a tip.
* The props probabilities are the served **recipe** refitted monthly, not the served pickle.
* Pro's 6+ cutoffs use predictions from 2023-07 on (a shorter run than SPC-1 live).
