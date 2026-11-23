# P1 item 8: Clean Record evaluation (honest post-repair performance record)

Owner deep-repair brief, 2026-11-22, P1 item 8. Repo `NevixAA/wowza-betting`, built on `origin/main` @ `61e869877`
(auto: predict 2026-11-22 19:26 UTC). Data as of the last settled match **2026-11-22**.

## Verdict

**NO PROVEN EDGE YET.** The Clean Record (from `2026-11-19T06:41:48Z`, `config/trustworthy_performance.json`) holds
**12 settled bets** across all forward records: TOP-3 staked 1, TOP-3 no stake 0, O/U 2.5 tips 4, side-market tips 6,
HT tips 1, props tips 0. Every one of the 9 forward bet cells is `NOT_ENOUGH_DATA` (minimums: 30 settled, 8 matchdays,
150 certified CLV). A cell needs 8 matchdays and 30 settled bets, so no forward cell can be read before January 2027
at the earliest (weekly leagues; winter breaks push many later).

`performance_claims_enabled` stays `false` (unchanged). No bar, threshold or stake was touched. The summary says
"EVALUATION ONLY - NOT A PERFORMANCE CLAIM".

## What was built

| File | Role |
|---|---|
| `src/clean_record_eval.py` | the reproducible evaluation: `python -m src.clean_record_eval --write` |
| `output/clean_record_evaluation.json` | committed summary: one total per segment / part / source, owner figures, method, immutability fingerprint |
| `output/clean_record_evaluation_cells.csv` | committed cells: one row per league × market × model version (2,124 cells at build) |
| `dashboard_data/clean_record_eval.py` | read model + page section (reads the committed files only) |
| `pages/24_🧾_Clean_Record.py` | new section "Post-repair evaluation (every record apart)" |
| `pages/29_🧭_Admin_Board.py` | new tab "Clean Record evaluation" (adds every cell of history, research and the learning rows) |
| `tests/test_clean_record_immutability.py` | the immutability test (12 checks, below) |

No price is written: the files carry counts, units, percentages and probability scores only (no odds value, no mean
price, no mean market probability). A test enforces the key names.

## Method

* **Segments, never pooled:** pre-repair historical; post-repair forward (TOP-3 staked, TOP-3 no stake, tips, props
  tips, prediction-ledger learning rows); research only; shadow experiments; real-price backtests; synthetic-price
  diagnostics (quarantined).
* **Split:** CLEAN = generated AND played on or after the start; the same rule as `dashboard_data/clean_record.assign`
  (a test proves the two agree row for row). Pre-repair tips come from `src/trustworthy_performance` view A, so every
  corrections sidecar is applied (post-kickoff tips, first-tip-only, CLV contamination, unpriced HT).
* **Cell:** league × market × model version; O/U 2.5 and HT split by side.
* **Per cell:** settled, matchdays, real-odds coverage, flat-1u P/L and ROI, ROI 95% CI (matchday block bootstrap,
  2,000 draws, `config/trustworthy_performance.json`), staked n and ROI (src/staked.py; TOP-3 by the frozen
  `stake_reason`) apart from not staked, max drawdown, log loss and Brier of the model, the same of the de-vigged
  bookmaker probability on the same rows (power de-vig, `edge_engine_shadow.csv` + its pre-re-seed archive, latest
  quote at or before the tip), calibration (mean p vs hit rate; bins from 30 rows), certified CLV eligibility and
  coverage (stored ledger CLV shown apart as uncertified), registered forward-test status (`config/track_registry.json`
  + `registry/preregistered_hypotheses.json`), verdict.
* **Verdicts:** NOT_ENOUGH_DATA / NEGATIVE (CI < 0) / NO_PROVEN_EDGE (CI spans 0) / POSITIVE_ROI_UNCONFIRMED (CI > 0
  but no certified CLV, or not forward) / EDGE_CANDIDATE_FOR_OWNER_REVIEW (forward only, ROI CI > 0 AND certified CLV
  CI > 0 at its minimum). Nothing is ever labelled PROVEN.

## Numbers (segment totals)

| Segment | Record | Settled | Matchdays | ROI | ROI 95% CI | P/L (1u) | Staked n | Max DD (u) | Log loss model − market (rows) | Certified CLV | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Before the repairs | O/U 2.5 tips | 1,716 | 145 | −6.8% | −12.1% … −1.4% | −115.99 | 122 | 131.7 | +0.0271 (205) | 0 | NEGATIVE |
| Before the repairs | Side-market tips | 736 | 84 | +4.7% | −2.7% … +12.7% | +34.91 | 21 | 22.2 | +0.0031 (86) | 0 | NO PROVEN EDGE |
| Before the repairs | HT tips | 14 | 3 | +39.8% | −11.3% … +88.8% | +5.57 | 0 | 1.8 | — | 0 | NOT ENOUGH |
| Before the repairs | Prediction ledger (learning) | 32,504 | 5 | — | — | — | — | — | −0.0033 (530) | — | NOT ENOUGH |
| Clean Record | TOP-3 staked | 1 | 1 | +48.0% | — | +0.48 | 1 | 0 | −0.1186 (1) | 0 | NOT ENOUGH |
| Clean Record | TOP-3 no stake | 0 | 0 | — | — | — | — | — | — | — | — |
| Clean Record | O/U 2.5 tips | 4 | 1 | +1.2% | — | +0.05 | 0 | 2.0 | +0.0243 (4) | 0 | NOT ENOUGH |
| Clean Record | Side-market tips | 6 | 2 | −46.0% | −61.8% … −14.5% | −2.76 | 1 | 3.3 | +0.0124 (6) | 0 | NOT ENOUGH |
| Clean Record | HT tips | 1 | 1 | +25.0% | — | +0.25 | 0 | 0 | — | 0 | NOT ENOUGH |
| Clean Record | Props tips | 0 | 0 | — | — | — | — | — | — | — | NOT ENOUGH |
| Clean Record | Prediction ledger, first sight | 52,005 | 4 | — | — | — | — | — | +0.0020 (884) | — | NOT ENOUGH |
| Clean Record | Prediction ledger, final | 25,782 | 4 | — | — | — | — | — | +0.0469 (212) | — | NOT ENOUGH |
| Research only | Weekly league panel | 765 | 24 | −9.6% | −18.3% … −1.7% | −73.78 | 0 | 79.2 | +0.0091 (765) | 544 (71.1%), mean −1.39% [−1.86, −0.95] | NEGATIVE |
| Research only | Sharp tracker | 2,010 | 122 | −7.9% | −13.1% … −2.7% | −159.15 | 0 | 165.6 | — | 0 | NEGATIVE |

Log loss model − market: positive = the bookmaker's de-vigged probability scored better on the same rows. On every
record with more than a handful of market-matched rows the market is at least as good as the model.

**Shadow (v91_shadow, O/U 2.5 OVER, 1,179 settled, 46 matchdays), log loss minus market on the same rows:** v9 +0.0136,
Pro HGB +0.0116 (513 rows), goal distribution +0.1236. No shadow model beats the market.

**Pre-repair cells (league × market × model version) with enough data:** 25 of 460: 21 NO_PROVEN_EDGE, 1 NEGATIVE
(League One O/U 2.5 UNDER, −47.2% [−71.4, −22.0], n 39), 3 POSITIVE_ROI_UNCONFIRMED: Finland O/U 2.5 UNDER +42.4%
[+10.2, +72.3] (n 48), Argentina BTTS +31.1% [+13.7, +49.6] (n 122; preregistered as H-BTTS-01, confirmation window
from 2026-10-02 needs 150 post-registration bets), Brazil Serie A BTTS +25.1% [+2.5, +43.9] (n 41). All three were
found by slicing history over 460 cells (discoveries under `registry/preregistered_hypotheses.json`, not evidence), on
older models, without certified CLV. They are hypotheses, not edges.

**Research cells with enough data:** 22 of 196 (weekly panel: none readable; sharp tracker: 20 NO_PROVEN_EDGE,
1 NEGATIVE: League Two 1X2 AWAY −69.2% [−87.0, −48.7], 1 POSITIVE_ROI_UNCONFIRMED: Liga MX 1X2 AWAY +61.2%
[+0.6, +123.9], n 39). Research is never staked and never enters the Clean Record.

## Historical figures (older models), verified

Definition that reproduces them: live tips generated before the Clean Record start, `signal_tier` SNIPER + MARKSMAN (the
retired tier basis of "staked" before 2026-11-04), `match_date` ≥ `config.PERFORMANCE_CUTOFF_DATE` (2026-08-10),
settled WIN/LOSS/PUSH/HALF (VOID out), flat 1u.

| Figure | Owner | Raw ledger now | Corrections applied now | 95% CI (corrected) | Exact match |
|---|---|---|---|---|---|
| Std O/U 2.5 | 164, −16.0% | 167, −13.98% | 165, −15.25% | −31.6% … +0.7% | at commit `0dc7e3326` (results 2026-11-21 04:14 UTC): 164, −16.00% |
| NF O/U 2.5 | 501, −8.8% | 503, −8.68% | 502, −8.89% | −19.1% … +1.0% | at `0dc7e3326`: 501, −8.80% |
| NF BTTS | 155, +16.1% | 155, +16.05% | 155, +16.05% | −2.2% … +32.7% | yes, now |
| NF O1.5 | 115, +8.7% | 115, +8.69% | 115, +8.69% | −4.1% … +21.3% | yes, now |

All four verified. The two O/U figures moved after the 2026-11-22 03:29 results run settled 3 more pre-start Std rows
(2026-11-21 matches) and 2 more NF rows; the corrections remove 2 Std rows and 1 NF row (post-kickoff / later tip of an
opposite pair). Every CI includes 0: none of the four is evidence of an edge or of a loss.

## Immutability test (`tests/test_clean_record_immutability.py`)

1. Rows generated before the start never become Clean Record under any of the three splits (synthetic, 400 rows).
2. Settling or appending rows never moves a pre-start row.
3. A date-only generation on the start day is never Clean Record.
4. The identity fingerprint ignores settlement and new rows but catches a rewritten frozen field and a back-dated row.
5. The committed ledgers still hash to the committed pre-start fingerprint (bets 1,936, side 765, HT 80, player 44,761,
   TOP-3 0 pre-start rows), and 0 pre-start rows are counted as Clean Record. The fingerprint was stable over every
   ledger commit since go-live (checked at 7 commits from 6848a3fc8 to 7ddbf4f57).
6. Every `output/*.first_sight.json` sidecar (TOP-3 ledger, weekly research, ledger tier first sight, quality tier
   shadow) still matches its file: frozen first-sight fields never changed.
7. The summary has no price key and echoes `performance_claims_enabled` unchanged; verdicts are never PROVEN and
   history is never a review candidate.

## Blockers and gaps

1. **Sample.** 12 settled forward bets on 1 to 2 matchdays. Nothing can be read before 30 bets over 8 matchdays per cell.
2. **No certified CLV on any live record.** Team ledgers carry stored (uncertified) CLV only; TOP-3 rows carry no
   close at all. Certified closes exist only in the weekly research panel and v91_shadow. Without a certified close on
   TOP-3 and tips, no forward cell can ever reach EDGE_CANDIDATE. Needs: write a certified close (pre-kickoff, ≤30 min)
   onto TOP-3 and tip rows at settlement.
3. **Model version missing on most pre-repair rows** (`unrecorded`): older ledger rows have no `model_version`, so the
   historical cells group them as one version.
4. **Market comparison coverage is thin.** The de-vigged market probability exists only where the edge shadow quoted
   the same fixture (from 2026-10-25): 205 of 1,716 pre-repair O/U rows, 86 of 736 side rows, 884 of 52,005 forward
   learning rows (team markets only; props have no market quote).
5. **`bets_ledger.csv` has no model probability before November** (`cell_p_used`); its log loss uses the shadow's
   `p_model_raw` where joined.
6. **Summary refresh.** The committed summary is built by hand (`python -m src.clean_record_eval --write`, about 2
   minutes). No workflow refreshes it yet; the dashboard shows its "data as of" date. Wiring it into the results
   workflow is a follow-up (a workflow change, not done here).
7. **Real-price backtests** are listed (7 files + `bars_opt/`) but not re-scored: they are older models, and a backtest
   proves nothing under the owner's live-proof rule.
8. **Synthetic-price diagnostics:** `output/player_props_calibration.json` stays quarantined (marker present); no
   figure from it is carried anywhere.
