# Wowza System Audit — Ranked Findings Report

> **Read-only audit under CODE FREEZE (2026/27 season).** No live changes were made and none are recommended for immediate application. Every fix below is staged for a **v10/staging copy**, validated there, and only then considered for promotion to live v9 — per the v10 staging-process and code-freeze rules.

## Executive Summary

- **1 Critical, 9 High** confirmed findings across 6 areas; 12 Medium, 9 Low; **1 finding unverified** (parked in its own section, not ranked).
- **The single most important fix:** the Live Scanner's `STRONG_STUCK`/`COMEBACK` logic recommends backing **OVER 2.5 exactly when the model's own live P(over) is 1–8%** — an open-ended gate with no upper bound on fair odds. It is actively misleading and near-guaranteed to lose.
- **Recurring theme #1 — measurement is corrupted:** the closing-odds archives that feed CLV are cross-contaminated/line-mislabelled (~1/3 of NF fixtures), so the CLV "edge" this audit exists to verify is partly an artifact.
- **Recurring theme #2 — in-sample claims vs OOS reality:** Standard SNIPER, side-market edges, and prop ROI headlines all rest on in-sample or synthetic numbers that honest OOS/live data contradicts; no code gate ties real money to the validated (`approved`) subset.
- **Reassuring:** the feared new-format scaler-collapse bug is **NOT active** in any current model (all confirmed dead-at-train), calibration is broadly sound, model isolation holds, and the CLV formula/units are correct — the corruption is in the inputs, not the math.

---

## CRITICAL

### C1 — Live Scanner recommends OVER 2.5 when the model says OVER is nearly impossible
- **Area:** Live Scanner (in-play) · `src/live_scanner.py`
- **Claim:** `STRONG_STUCK`/`COMEBACK` fire an OVER 2.5 tip whenever `fair_over_odds >= 2.00`, which only requires `p_over <= 0.50` with **no ceiling** — so it fires hardest at p_over near zero.
- **Evidence:** `live_scanner.py:552-556`. Across all 7 `STRONG_STUCK` rows in `live_signals_history.csv`, `live_p_over` mean = 0.0845 (min 0.0066); 5 of 7 have `fair_over_odds >= 10` (model P(over) < 10%). E.g. Yokohama F Marinos 0-0@77min → fair_over = 100.0 (P = 0.66%); Gamba Osaka 0-0@72 → 53.45; Hartberg 0-0@70 → 47.67. Independently recomputed: `_live_probs(0,77,lam)` = 0.0066 exactly. Snapshot-graded CA Tigre vs River Plate (`STRONG_STUCK` OVER@61, fair 26.41) finished 1-0 → **LOST**.
- **Failure scenario:** A 0-0 game at 70-78min with a strong-attack team triggers a Telegram "OVER 2.5" tip. The model assigns 1-8% to OVER; no book offers the ~50-100 fair price needed for value, so every realistic price is -EV. Backing these is a near-certain loss and the alert misleads.
- **Staged fix:** On v10, bound OVER-type signals to a plausible value band (e.g. fire only when `p_over` ∈ 0.35-0.55, i.e. `2.00 <= fair_over <= ~2.9`) instead of open-ended `>= 2.00`. Re-validate on collected snapshots before promoting. Do **not** hot-patch live under freeze.

---

## HIGH

### H1 — No code gate ties real-money staking to OOS-`approved` standard leagues
- **Area:** Standard O/U 2.5 · `config.py`, `src/betting.py`, `models/best_params_standard.json`
- **Claim:** The claimed SNIPER edge fails walk-forward OOS in 6 of 8 configured leagues; config comments cite in-sample ROI, and nothing in code restricts real stakes to `approved==true` leagues.
- **Evidence:** Honest walk-forward OOS (`backtest.py:552-589`): only **Ligue 2** (roi_oos +1.72, n93) and **Serie B** (+13.01, n99) are `approved=true`. Negative/insufficient: Championship -7.9 (n311), La Liga 2 -12.88 (n290), League Two -19.07 (n42), League One -27.75 (n8), Bundesliga 2 +4.57 (n7<min). Yet `config.py:267-277` comments claim "League Two +22.6%", "La Liga 2 +53.5%", "Ligue 2 +45.2%", "League One +21.4%" (in-sample). `betting.py:153-158` falls back to exactly these hand-set thresholds for non-approved leagues, so live still emits SNIPER there. `grep` shows `approved` is referenced only in `backtest.py`/`betting.py` — no paper-vs-real stake gate keys on it. Live track record: 40 std bets since 2026-05-01, 26 settled, winrate 0.423, ROI -0.15%, only 5 SNIPER.
- **Failure scenario:** Under the season-start plan ($30/bet real money on Std O/U SNIPER), a SNIPER tip fires in League Two / La Liga 2 / Championship (all negative OOS). With no code gate, real money rides in-sample noise; historical OOS ROI there is -8% to -28%.
- **Staged fix:** Gate real-money staking to `approved==true` (currently Ligue 2, Serie B) **in code, not discipline**; correct/remove the in-sample ROI claims in `config.py` comments to cite OOS. Validate before promoting.

### H2 — Side models pool std+NF with no per-league intercept → every live tip concentrated in the most-overconfident league
- **Area:** Side markets (BTTS / O1.5 / O3.5) · side model pkls, `pipeline.py`
- **Claim:** Pooled training with no per-league intercept systematically over-predicts overs/BTTS in low-scoring new-format leagues; all 12 live side tips are Over1.5 on Argentina — the exact league where the model is most overconfident — inflating edges beyond backtest support.
- **Evidence:** 14 Argentina fixtures, model mean vs `af_history` base rate: p_over15 0.722 vs 0.655 (+0.067), p_btts 0.535 vs 0.454 (+0.081), p_over35 0.230 vs 0.200, p_over25 0.403 vs 0.389 — all biased high. `side_bets_ledger.csv`: all 12 live tips over15, all Argentina, edges 8.76-18.82% (reproduces `pipeline.py:100-101`). But `backtest_results_over15.csv` (standard leagues) ROI +2.39%, winrate 0.730 == base 0.722 (edge at base rate). NF leagues have **no** optimizer validation (`best_params_side_markets.json` covers 7 standard leagues only), and `pipeline.py:108-116` ignores the drop flag and emits anyway.
- **Failure scenario:** Nightly predict emits over1.5 "SNIPER" at 12-19% edge on the lowest-scoring leagues because the pooled model reports ~0.72 where the true rate is ~0.655 and books price ~0.58 — the "edge" is overconfidence. If side markets ever came off paper, these become the top-ranked bets and systematically lose.
- **Staged fix:** Add per-league base-rate/intercept correction (or league dummies) or per-league calibration; re-run walk-forward side backtest **including** NF leagues and gate live tips on it. Keep side markets paper per staking memo until then.

### H3 — Closing-odds archives are cross-contaminated / line-mislabelled → CLV captured WRONG for NF and side markets
- **Area:** Data & CLV integrity · `newformat_odds_history.csv`, `standard_sidemarket_odds_history.csv`, `update_results.py`
- **Claim:** A large minority of closing lines are mathematically impossible or copied from a different market, so the CLV metric this audit exists to verify is corrupted for new-format and side markets.
- **Evidence:** Within a single `snapshot_ts`, **673/2064 = 32.6%** of NF fixtures violate O/U monotonicity (over15>=over25 or over25>=over35). **618/4831 = 12.8%** have over25 exactly equal to btts_yes, 640/4831 = 13.2% have under25==btts_no (distinct markets sharing identical floats). Concrete: Gremio vs Sao Paulo 2026-08-08 stores over25=5.5/under25=1.14 (== btts). These feed `update_results.py` `_closing_from_csv` (l.225-255) → 13 ledger rows with |clv_pct|>40% (up to +132.64% NY Red Bulls; -62.91% Gremio). NF settled-CLV is +7.54% over 48 rows but **+4.69%** over the 37 plausible-closing rows — corruption inflates headline CLV by ~2.85pp. `standard_sidemarket` shows 21.9% over25>=over35. (CLV formula itself: 0/72 recompute mismatches — the math is fine.)
- **Failure scenario:** Operator reads "NF CLV +7.5% → SHARP, beating the close" and scales real money onto NF O/U per the staking plan; in reality ~1/3 of NF closings are garbage and true CLV is materially lower and noise-dominated.
- **Staged fix:** Harden `_parse_all_odds` in the capture scripts: (a) accept only the FT goals O/U market (exclude Corners/Cards/Home/Away; exact "Over 2.5" token not substring); (b) reject a fixture's O/U set unless it passes a monotonicity+overround gate (over15<over25<over35, 1.00<overround<1.15, over25!=btts_yes); (c) apply the same gate in `_closing_from_csv` so an implausible closing is treated as missing (CLV blank) not a fake +132%. Backfill-clean both CSVs; confirm the |CLV|>40% tail disappears before promoting.

### H4 — Player-props pipeline is effectively dead and every current tip is un-actionable
- **Area:** Player Props · `player_model/`, `predict.py`
- **Claim:** No automated props run/commit since 2026-07-26 (15 days), and all 250 current tips are AVOID, have no odds, and sit in non-prop leagues.
- **Evidence:** Last "auto: player props" commit 2026-07-26 15:29 UTC. `player_tips.csv`: 250/250 tier=AVOID, market_odds non-null 0/250. Leagues = MLS(130), Brazil(54), Finland(30), Norway(17), Argentina(15), Ireland(4) — **none** in `config.PROP_LEAGUES` (PL/Bundesliga/La Liga/Serie A/Ligue1/Champ/L1/Bund2/CL/EL/Conf/WC). These entered only via the `bets.csv` fallback (`predict.py:303-315`); the PROP_LEAGUES fetch (`predict.py:282-298`) returned zero. Club prop odds never captured (odds history 100% World Cup).
- **Failure scenario:** Anyone assuming props are live sees a stale CSV of AVOID tips on summer leagues with no bookmaker prop markets; when club props open mid-Aug the season-rolled `PROP_SEASONS='2026'` config is untested against a live fetch.
- **Staged fix:** Confirm the props workflow schedule is still enabled; before club-season open, dry-run the PROP_LEAGUES fixture+odds capture on API-Football season 2026 and assert non-empty; treat `bets.csv`-fallback (non-PROP_LEAGUES) rows as data-only so they don't masquerade as tips.

### H5 — Live Scanner v2 SOT-blend (`SOT_TO_GOAL=0.311`) never executes and can never be fit
- **Area:** Live Scanner · `src/live_scanner.py`, `inplay_snapshots.csv`
- **Claim:** The headline "live-adjusted λ from in-play SOT" never runs in production, and the Phase-1 collection dataset lacks the two fields (fixture_id, sot) needed to ever fit it.
- **Evidence:** `inplay_snapshots.csv`: 128 rows, fixture_id non-null = 0, sot non-null = 0. `_detect_signals` (`:448-451`) computes `_lam_rem` only when `_sot_live` is not None, and `_sot_live = sot_by_fixture.get(fixture_id)`; with fixture_id always empty the override is always None → naive `lam_total*remaining_frac` path. Proof: Yokohama 0-0@77 p_over 0.0066/fair 100 and Kashiwa 2-0@64 p_under 0.4755/fair 2.1 reproduce **exactly** from the naive path. The scanner runs the OddsAPI fallback (`_fetch_live_scores_oddsapi` sets no fixture_id); the SOT/v2 machinery only works on the API-Football path.
- **Failure scenario:** The team believes live prices are SOT-adjusted; they are the old naive time-decay. And because fixture_id+sot are never logged, the Phase-2 plan to fit K and game-state multipliers can never run.
- **Staged fix:** On v10, confirm the live fetch path; if OddsAPI is primary, document the v2 SOT blend as **inert** rather than implied-active. Populate fixture_id+sot (API-Football path) before claiming v2 is live or fitting multipliers; validate the blend improves calibration on real snapshots first.

### H6 — Live signals have no measured edge: grader matches almost nothing, no market price ever captured
- **Area:** Live Scanner · `grade_live_signals()`, `live_scanner.py`
- **Claim:** No hit-rate edge can be shown — the grader matches ~2 of 23 signals and the module captures no bookmaker price, so fair-vs-market CLV is uncomputable.
- **Evidence:** `grade_live_signals()` returned n=2, hit 50%. Root cause: `player_history.parquet` ends 2026-07-04 but 18 of 23 signals are dated 2026-07-06..08-09 (ungradable) + WC name mismatches. Independent snapshot grade (elapsed>=85): UNDER_RECOVERY 0/3, STRONG_STUCK 0/1, UNDER_HOLD 1/1 → 1/5. By design no book price is captured (`live_scanner.py:7-8` "calculate the FAIR live price").
- **Failure scenario:** Any edge claim is unfalsifiable; the lone "50% on n=2" is noise that could be used to justify real staking on a zero-validated-edge strategy.
- **Staged fix:** On v10: point the grader at a current-date results source (or backfill finals); normalize WC/team names; log the actual live O/U price at signal time so realized CLV/EV is measurable. Treat live signals as PAPER until a real OOS hit rate with market prices exists.

### H7 — HT O/U models have essentially zero discriminative power (base-rate predictors)
- **Area:** Half-Time O/U · `model_ht_over05.pkl`, `model_ht_over15.pkl`
- **Claim:** The HT models are near-random rank-orderers — a WEAK/low-signal model (not the NF scaler-collapse bug).
- **Evidence:** Test AUC over05 = 0.5418/0.5304/0.5277, over15 = 0.5433/0.521/0.5178 (all near 0.5 on fully-populated test). All three base models report identical accuracy (0.7075 over05, 0.6486 over15) = pure majority-class. On 43 predicted std fixtures: p_ht_over05 std 0.0062 (0.6923-0.7218), p_ht_over15 std 0.0071 (0.3324-0.3659). Means match base rate almost exactly (0.7068 vs 0.7075; 0.3482 vs ~0.3514) → unbiased but flat.
- **Failure scenario:** Any downstream logic treating p_ht as an informative per-fixture signal is reading noise around the base rate; there is no fixture-level edge.
- **Staged fix:** Treat HT as validated no-signal (like props). Retire the HT models from the live predict path or flag output non-actionable. Any revival must clear an honest OOS AUC bar on v10 first.

### H8 — HT market emits ZERO actionable tips; `ht_ledger.csv` has never been written
- **Area:** Half-Time O/U · `src/ledger.py`
- **Claim:** The flat HT output can never cross the tip thresholds, so the HT ledger is never even created.
- **Evidence:** `ledger.py:341-348` thresholds: over05 needs p>=0.75 or <=0.30; over15 needs p>=0.60 or <=0.25. On all 43 fixtures: 0/43 cross over05 (max 0.7218, 0.028 short), 0/43 cross over15 (max 0.3659, 0.234 short). `git log --all -- output/ht_ledger.csv` returns nothing; file absent from origin/main.
- **Failure scenario:** HT consumes retrain compute, feature-engineering surface, and predict-time work every run but has produced zero tips and cannot given its output range — dead weight and maintenance surface with no upside.
- **Staged fix:** On v10, either fix the model (H7) so output can plausibly reach thresholds, or remove HT tip generation. Do **not** merely lower thresholds on the current flat model — that fires tips with no signal.

### H9 — HT CLV + grading is entirely unimplemented
- **Area:** Half-Time O/U · `src/ledger.py`, odds snapshotter
- **Claim:** No HT odds are captured, no settlement code exists, and ledger CLV/result/pnl are never populated.
- **Evidence:** `HT_LEDGER_COLS` (`ledger.py:313-317`) includes entry_odds/closing_odds/clv_pct/result/pnl, but `append_ht_tips` (`:357-362`) writes them all empty. Docstring claims they settle against `standard_sidemarket_odds_history` ht_* markets — but that file's 12,909 rows contain markets {btts_yes/no, over25/under25, over35/under35, h2h_*, over15/under15} and **zero ht_* markets**. Grep finds only the tip-writer + label definitions; no settlement path.
- **Failure scenario:** If the model were ever fixed to fire a tip, it would stay permanently open — no entry/closing odds, no CLV, no P&L — so HT can never be validated by the season-long CLV/ROI test.
- **Staged fix:** Before reviving HT: (1) add ht_* market capture to the odds snapshotter; (2) implement HT settlement (grade vs HTHG+HTAG, resolve entry vs closing, compute clv_pct) matching the ledger's PERCENT×100 units. Stage on v10 with a coverage check.

---

## MEDIUM

### M1 — Standard model is a marginal discriminator (compressed toward base rate)
- **Area:** Standard O/U · `model_v9_standard.pkl`
- Output on 12,187 backtest rows: mean 0.5075, std 0.0249, span 0.416-0.618. Held-out AUC 0.546. Reliability curve monotonic but compressed (pred spans 0.44-0.56 while actual spans 0.19-0.62). Current predictions: best_edge max 0.071, **zero** reach SNIPER (needs 0.12-0.25), only 4 VALUABLE.
- **Failure scenario:** O/U 2.5 is near-efficient; the model rank-orders weakly and edges are understated, so real money on this thin separation is unlikely +EV net of vig.
- **Staged fix:** Treat standard SNIPER as paper until a larger live sample or stronger holdout shows edge. Do **not** "fix" by widening — the compression matches genuine uncertainty. Revisit whether Platt calibration over-shrinks, on v10 only.

### M2 — `bets_ledger.csv` standard aggregates dominated by 4,115 stale backtest rows
- **Area:** Standard O/U / Data · `bets_ledger.csv`
- 4,155 standard rows = 4,115 backtest + 40 live. Naive aggregate: winrate 0.406, ROI -6.9%, SNIPER 45% — but the current backtest file shows SNIPER only 4% and current best_edge maxes at 0.28, so the ledger's 45%/edge-up-to-57 rows are from retired wider generations. Live-only: 40 rows, ROI -0.15%.
- **Failure scenario:** Any dashboard computing ROI/winrate/CLV without filtering `source=='live'` reports a -6.9%/45%-SNIPER picture from retired models → wrong go/no-go.
- **Staged fix:** Separate backtest from live rows (distinct file or always filter `source=='live'`) and stamp each row with the model/config version.

### M3 — Side models retain the exact features that collapsed the NF model (latent NF-class risk)
- **Area:** Side markets · side pkls, `pipeline.py`
- All three side models carry the full 42-col set incl. xg/insidebox and HT rates (100% / 89% NaN at predict). Benign **now** (scaler mean_=0.0/scale_=1.0/coef=0.0 → dead-at-train too). NF model explicitly drops these via `_NF_DROP` (feature_cols=27); side models pass `feature_cols=None` → full.
- **Failure scenario:** On the next retrain after xg/insidebox populate in the training window, side models learn a non-zero coef; at predict xg stays all-NaN → imputed 0.0 → StandardScaler extreme z → collapsed prob, silently reproducing the NF 0.36-vs-0.51 failure.
- **Staged fix:** Apply `_NF_DROP` to the side models, or persist TRAIN-time medians in the payload and impute from those. Add a predict-time guard asserting no column is 100% NaN.

### M4 — `newformat_odds_history.csv` contains per-row over/under label swaps
- **Area:** Side markets / Data · `newformat_odds_history.csv`
- over35<=over25 in 155/1099 rows (14.1%); over15>=over35 in 110/363 (30%). E.g. Botafogo vs Santos O2.5=4.33 but O3.5=1.62 (impossible). Correct rows also exist → per-row swap, not global. (Argentina over15 is NOT swapped — initial suspicion retracted.)
- **Failure scenario:** Feeds `data_loader.py:64 _nf_real_odds()`; when `af_odds_history.parquet` is regenerated or the NF over35 backtest is enabled, swapped prices inject spurious edges. Currently contained (parquet missing; over35 backtest fires 0 bets).
- **Staged fix:** Add ingestion validation rejecting rows that violate implied(over15)<over25<over35 ordering; re-derive swapped rows from source; re-run the forward backfill. Don't consume for edges until the check passes. *(Overlaps H3 — fix together.)*

### M5 — BTTS model near-random and Over 3.5 inactive → 2 of 3 side markets produce no usable output
- **Area:** Side markets
- BTTS AUC 0.5302; p_btts std 0.0124 (0.522-0.579); backtest ROI -1.34%. BTTS can never fire live: `predict.py:66` requests `markets='totals'` only, so odds_btts=0/129 and the ledger shows only over15 (12). Over3.5: odds_over35=14/129, backtest 0 bets, optimizer drop=True in all 7 standard leagues.
- **Failure scenario:** p_btts is dead weight that could mislead if read as a signal; three markets appear live when only Over1.5 emits tips.
- **Staged fix:** Either wire the OddsAPI per-event BTTS endpoint or retire the BTTS/Over3.5 live paths as research-only; document that only Over1.5 emits. Labelling/expectation fix — no live change under freeze.

### M6 — HT partial train/predict feature mismatch (batch-median imputation flattens further)
- **Area:** Half-Time O/U · `model.py:128`, `pipeline.py`
- On 43 predict rows: home/away ht rates 67-79% NaN, season venue stats 67-79% NaN, rest_days 60-63% NaN — yet training requires them non-null (`pipeline.py:220`). `col_medians = X.median().fillna(0.0)` recomputes the fill from the current batch, so 67-79% of rows get an identical median → cross-fixture variance destroyed, fill center differs from train. Low-importance features → no blow-up, calibration unbiased.
- **Failure scenario:** Season-boundary predictions (few current-season games) collapse these features to one batch-median value exactly when the season starts.
- **Staged fix:** Persist train-time `col_medians` in the payload and impute predict rows from those; gate features >X% NaN at predict. Validate model isolation.

### M7 — HT feature set carries xg/insidebox the NF model dropped (latent NF collapse)
- **Area:** Half-Time O/U · HT pkls, `pipeline.py:203-226`
- HT models trained with full default FEATURE_COLS (42) incl. xg/insidebox; NF drops these via `_NF_DROP` for the collapse reason. On predict all four xg/insidebox cols 100% NaN, importance 0.0. Not firing today (xg constant→zero-variance at train).
- **Failure scenario:** If the xG cache is backfilled for NF leagues carrying HT labels (making xg non-constant at train) while std predict fixtures stay cold (xg all-NaN→0.0→extreme z), the logistic base shifts → NF-style collapse in the HT model.
- **Staged fix:** Mirror `_NF_DROP` into the HT training call (drop xg/insidebox at minimum), or adopt the persisted-median fix from M6. Add a 100%-NaN-drop guard/test.

### M8 — `enrich_no_odds_markets` (assists tiering) is dead code from a JSON key mismatch
- **Area:** Player Props · `predict.py:645`, `player_props_calibration.json`
- `predict.py:645` reads `cal.get("base_rate")` but the file only has `base_rate_%`. Reproduced: `.get('base_rate')`=None → `if not base_rate: continue` no-ops the whole assists loop. The value is also a PERCENT (5.9) needing /100.
- **Failure scenario:** Assists has no OddsAPI coverage, so this is its only tiering path; it can never assign a tier → assists tips permanently AVOID even when the model (AUC 0.684) is confident.
- **Staged fix:** Read `base_rate_%` and divide by 100 (or regenerate the file with a `base_rate` fraction). Add a unit test that a confident assists row gets a non-AVOID tier. Stays paper-only.

### M9 — `player_props_calibration.json` advertises +140% to +263% ROI (synthetic, contradicts "no edge")
- **Area:** Player Props · `player_props_calibration.json`
- goals roi_%=140.36, assists 133.06, sot2 128.91, sot3 263.51, goals2 9656 (n=1). `implied_odds` ≈ 1/base_rate → ROI = win_rate × fair-odds − 1, i.e. profit vs **synthetic** odds, not real book prices. No live `.py` writes these; contradicted by the honest OOS ledger (−233u, every tier/market negative).
- **Failure scenario:** A human or the Model-Info dashboard reads +140-263% ROI, concludes props have huge edge, and pushes them off paper into real staking — the exact rushed-to-live mistake the process guards against.
- **Staged fix:** Drop roi_%/win_rate_% (keep only base_rate as a fraction), or clearly label the file an in-sample fair-odds diagnostic. Never surface these ROI numbers where they inform staking.

### M10 — Live Scanner history has no de-dup → fixtures double-counted, contradictory bets on one match
- **Area:** Live Scanner · `_append_to_history` (`:786-800`), `grade_live_signals`
- The full tips df is appended every scan with no unique key. 5 fixtures appear >1×: Kashiwa Reysol twice as the same UNDER_RECOVERY (2-0@51 and 2-0@64); Gamba Osaka shows HT_UNDER_0.5 (UNDER) then STRONG_STUCK (OVER) — contradictory sides. The grader iterates rows with no dedup.
- **Failure scenario:** A signal persisting across the 2-min rescan is appended repeatedly, inflating n and biasing the hit-rate; contradictory UNDER-then-OVER rows make per-type stats incoherent.
- **Staged fix:** Dedup history on (date, match, signal_type) before appending, and dedup in the grader before aggregating. Consider one-signal-per-fixture-per-side locking.

### M11 — Live Scanner game clock is a coarse wall-clock estimate on the active path
- **Area:** Live Scanner · `_fetch_live_scores_oddsapi` (`:270-274`)
- elapsed = wall_mins − 15 when wall_mins>60, capped 95, fixed 15-min HT assumption, no real match clock. Snapshot values consistent (Beijing FC 22→95 over 89-min wall gap). elapsed feeds `remaining_frac` and lam_remaining in `_live_probs`, and the MIN_ELAPSED/HT_LOCK gates.
- **Failure scenario:** Near HT or with stoppage time, elapsed can be off 10+ min; a gate at "elapsed>=70" can fire when the true clock is ~60, materially changing p_under/p_over and triggering UNDER locks early.
- **Staged fix:** Prefer the API-Football real elapsed (also needed for H5's fixture_id/SOT); when only wall-clock is available, widen time-gate margins and flag the price approximate.

### M12 — `PERFORMANCE_CUTOFF_DATE` is date-granular → ~74 pre-fix NF tips leak into post-fix aggregates
- **Area:** Data & CLV · `config.py:359`, `telegram_bot/notifier.py`
- Cutoff compares `generated_at[:10] >= '2026-08-09'` (config self-comment admits "~64 pre-fix tips will also count"). 78 rows have generated_at date 2026-08-09 (74 new_format), but generated_at carries full timestamps (sub-day precision unused). The 74 NF tips are UNDER-skewed 55:10 — the signature of the pre-fix P(over)=0.36 collapse.
- **Failure scenario:** Once those UNDER-biased pre-fix tips settle, the "since 2026-08-09" clean post-fix track record is polluted by the exact bug the cutoff was meant to exclude, biasing the NF go/no-go read.
- **Staged fix:** Compare against the full fix-commit timestamp on 2026-08-09 (or bump the cutoff to '2026-08-10' per the config's own note); confirm the 74 rows drop out before trusting post-fix numbers.

### M13 — CLV coverage is thin and model-imbalanced; standard forward capture only just began
- **Area:** Data & CLV · `bets_ledger.csv`, `standard_odds_history.csv`
- 72/585 settled rows (12%) have clv_pct: new_format 48/296 (16%), standard 21/286 (7%). `standard_odds_history.csv` holds only 99 rows spanning 2026-08-09..08-16. After removing corrupt closings (H3), usable NF CLV = 37 rows, standard = 21.
- **Failure scenario:** A CLV/ROI "edge" on 20-40 rows per model is within noise; treating it as validated repeats the small-sample artifact pattern the mandate warns against.
- **Staged fix:** Report CLV coverage % and n beside any CLV figure; suppress the SHARP/SOFT verdict until n>=100 and coverage>=50%. Reporting guardrail only — no model change.

---

## LOW

### L1 — Standard CLV captured for only ~half of live bets (headline 0.5% is a denominator artifact)
- **Area:** Standard O/U · `bets_ledger.csv`
- clv_pct on 21/4,155 standard rows (0.5%) — but 4,115 are backtest rows with no closing odds. Of the 40 live rows, 21 have clv_pct (52.5%), mean +0.595. **Fix:** improve closing-odds capture for live standard bets; report CLV over live rows only.

### L2 — `src/model.py` FEATURE_COLS (65) diverges from deployed pkl (42) — latent maintenance trap
- **Area:** Standard O/U · `model.py`
- The 23 extras (incl. `api_implied_over25`, h2h_*, formation, possession) are absent from the standard training frame, so `_prep` drops them at train; serving is internally consistent (payload uses 42). Notably `api_implied_over25` IS populated for all 43 std fixtures but excluded from the model. **Fix:** pin the standard model to an explicit feature list (as NF does); separately evaluate adding `api_implied_over25`.

### L3 — `side_bets_ledger` CLV/closing/result 0/12 populated (settlement path unverified)
- **Area:** Side markets · `side_bets_ledger.csv`
- All 12 rows are future/unsettled (signal 2026-08-09, match 2026-08-15), so NaN is expected — but no settled historical side bet exists to prove the closing-capture/CLV path runs. **Fix:** re-check after 2026-08-15 settles; if still NaN, audit `ledger.append_side_market_tips` + settlement. No code change now.

### L4 — HT saved `payload['target']` == 'over25' (stale metadata, not the real target)
- **Area:** Half-Time O/U · `model.py:263`, `pipeline.py:143`
- Both HT pkls report target='over25'; training was correct (`ht_over05`/`ht_over15`), but `save_models` defaults target='over25' and the call omits it. `predict_proba` doesn't read the field → output unaffected. **Fix:** pass target through `save_models` or drop the field. Cosmetic; via staging.

### L5 — Prop odds history & CLV are World-Cup-only, ungraded, partly degenerate
- **Area:** Player Props · `player_prop_odds_history.csv`, `clv_records.csv`
- 11,972 rows all league=World Cup (Jun-Jul); 4 rows odds=1.0 for anytime-scorer defenders (impossible), 303 rows odds<1.5, no over/under side column. `clv_records.csv`: 25 rows all PAPER, 0 graded, clv_pct mean -0.0057 over n=20. **Fix:** filter odds<=1.0 and implausible sub-1.5 lines at ingest; add a side column; grade results; re-enable capture for PROP_LEAGUES. Keep paper.

### L6 — Prop `_prep` uses per-batch median then 0.0 (NF root-cause pattern) — bounded impact
- **Area:** Player Props · `model.py:37`
- Same all-NaN→0.0→extreme-z mechanism as NF, but empirically bounded: forcing `sot_pg` all-NaN shifts the ensemble mean only +0.015 because the ensemble averages a scale-invariant GradientBoosting alongside LogReg. **Fix:** persist train-time medians and impute from those; warn on all-NaN features. Low priority given bounded impact and paper-only status.

### L7 — `home_advantage` variable at train but pinned to 1.0 for every upcoming fixture
- **Area:** Data & CLV · `predictions.csv`
- All 129 rows =1.0 (train mean~0.48-0.52). z(1.0)=+1.99 std / +1.76 NF but coef ~+0.026/+0.022 → logit shift only +0.052/+0.040 (~1pt). Uniform offset → no discrimination harm. **Fix:** populate a genuine per-fixture value at predict or drop it from FEATURE_COLS if structurally constant. Not urgent.

### L8 — `bets_ledger.csv` polluted with thousands of 2020-2025 backtest rows that can never settle
- **Area:** Data & CLV · `bets_ledger.csv`
- 4,034/4,619 past-dated tips unresolved; 3,865 standard span 2020-09..2025-05, generated_at = bulk-import 2026-04-26 (so PERFORMANCE_CUTOFF **does** exclude them from headline stats). But `update_results.py` "LEDGER TOTALS" print (l.1021-1031) and any un-cutoff reader mix eras; `_FD_STD_URL` hardcoded to 2526 so pre-2025 can never resolve. **Fix:** move backtest rows to `arch_bets` or tag `source='backtest'` and filter everywhere. Housekeeping. *(Overlaps M2.)*

---

## Unverified / Needs Confirmation (NOT ranked — do not treat as confirmed)

### U1 — HT models may be trained on out-of-scope leagues while applied only to standard leagues
- **Area:** Half-Time O/U · `af_ht_history.parquet`, `pipeline.py:220`
- **Status:** `verified_with_code=false` — the auditor could **not** trace offline whether `af_ht_history` is actually merged into the training frame (network-gated), so the training league mix is unconfirmed.
- **What was seen:** `af_ht_history.parquet` (6,977 rows) contains only non-scope leagues (Argentina, Brazil, MLS, K-League, Saudi) — 0 rows for any `STANDARD_FORMAT_LEAGUES` — yet HT is predicted only for standard fixtures. `ht_valid` (`pipeline.py:220`) is not league-filtered, so it *could* pull out-of-scope HT rows in.
- **If confirmed:** the HT model would learn South American/Asian/MLS patterns and apply them to English lower divisions — a distribution mismatch violating the ONLY-OUR-LEAGUES rule (adds to H7's weak-signal picture).
- **Next step:** Confirm the `ht_valid` league mix from a training-run log; if contaminated, league-filter `ht_valid` to `STANDARD_FORMAT_LEAGUES` on v10. Do not act until confirmed.

---

## Confirmed Sound (verified healthy — do not re-litigate)

**Cross-cutting (Data & CLV audit):**
- **The feared NF scaler-collapse bug is NOT active in any current model.** Every fitted StandardScaler was extracted: no feature is alive at train (mean≠0/scale≠1) AND all-NaN at predict. The 4 all-NaN std features (xg/insidebox) are dead-at-train too (mean 0.0, scale 1.0 → z=0). Current models use 42 (std) / 27 (NF) features — reduced in the Aug-3/Aug-9 retrains.
- **CLV math & units are correct.** Formula `(odds−close)/close×100`, settles at entry, PERCENT×100, consistent across bets/HT ledgers; 0/72 recompute mismatches. Extreme values trace to bad closing **inputs** (H3), not the formula.
- **Over/under labelling is sound.** De-vig implied P(over) tracks league base rates; "over is shorter" occurs exactly in high-scoring leagues (MLS/Norway/China) and "over longer" in low-scoring (Argentina/Brazil). No systematic swap.
- **Live calibration matches the fix having landed:** NF p_over25 mean 0.508 vs implied 0.500; standard 0.504 vs 0.500 — no NF +0.20 bias.
- **Dead columns don't leak:** `api_implied_over25`/`implied_prob_over` are not in any current model's feature_cols. Dedup logic in capture scripts and `drift.py` is correct.

**Standard O/U:** Calibration bias sound (+0.027 vs base, monotonic reliability, no label swap). Train/predict feature mismatch NOT at risk (all-NaN features are inert at both train and predict; partial-NaN features impute within ~0.3 sd of train mean). Model isolation sound (self-contained pkl; xG/HT drop applied only to NF). The OOS/`approved` optimizer methodology is legitimate walk-forward — its honesty is what exposed H1.

**Side markets:** Global calibration sound across 129 fixtures (no NF-style gap). Current saved side scalers have coef=0.0 on the dangerous features (dead-at-train). Live over-odds ordering internally consistent. Model isolation holds (`data_loader.py:347` excludes odds_over25 from the side fill — std money-market untouched). `standard_sidemarket_odds_history.csv` well-formed; standard over15 backtest internally consistent (winrate 0.730 == model mean).

**Half-Time O/U:** Calibration sound and unbiased — means within ~0.003 of base rate (positively rules out the NF collapse for HT). Label definitions correct (`ht_over05`=HT≥1, `ht_over15`=HT≥2); predicted ordering sane. No 100%-NaN populated-at-train feature blows up in current data. Both feature builders use the same `half_avg` formula (no train/predict divergence). HT correctly restricted to standard leagues at predict.

**Player Props:** Calibration sound on a temporal holdout — all 7 markets within ~0.008 of base rate, no NF-style bias. Discrimination sound (goals 0.747, sot 0.735, sot3 0.847; not flat). The "no edge / paper-only" decision is confirmed OOS (503-bet WC ledger −233u, every tier and market negative). Model isolation sound (7 independent pkls; unused goals3/sot4 can't leak). Correct guards observed (WC card gate, GK skip, per-market caps, quality-mismatch discount, PAPER labelling). The "cards near-constant" config comment is a WC-imputation artifact — cards spans 0.061-0.552 on real club data.

**Live Scanner:** Poisson core mathematically sound — `_poisson_prob`/`_poisson_cdf` and the `_lambda_from_p_over` binary-search inversion reproduce recorded fair prices exactly. UNDER-side signals (UNDER_HOLD/RECOVERY/SLEEPING_GAME) are directionally consistent with their fair prices (only the OVER branch is inverted — C1). No ML model of its own (rule-based on `predictions.csv`), so NF-style feature-mismatch and calibration dimensions don't apply. Model isolation intact (reads predictions, writes only its own CSVs; no training feedback). Telegram dedup via `live_notified.json` functions. `SOT_TO_GOAL=0.311` and the heuristic multipliers are honestly labelled as empirical/placeholder.