# Opening-price edge: does the clean model beat the early market?

**Protocol** `opening-price-edge/1`: `config/opening_price_edge_protocol.json`, sha256 `683b4e534385eff80a1da823c2e63843ef5918590f3efe0eb10b94b81892b288`.
It was registered in `config/opening_price_edge_protocol_ledger.json` and pushed in commit `0319be19f` **before** any ROI, CLV or p-value was computed.
**Forward protocol** `opening-price-edge-forward/1`: `config/opening_price_edge_forward_protocol.json`, sha256 `8498811f9819f0d7d637e6ffce3040002d8ece16b258909fa7da1dea1792240e`. Its status is **NO_ELIGIBLE_CELLS**.
**Owner approval:** 2026-11-15 "yes". This is research only. Nothing in production changes, and the owner decides.
**Licence:** this report holds counts, ROI, CLV, CI and p-values only. No OddsAPI price appears here. Priced intermediates stay local, outside the repository.

## Answer in one paragraph

**No.** On the clean champion_recipe walk-forward and model-holdout predictions dated before 2026-09-01, priced through the shared adapter, the model does **not** beat the OPEN (T-48h) price in any league × market × side. It also does not beat the BET-tier (T-125h / T-153h) price or the football-data pre-match price.

- **Verdicts:** of 466 registered cells (932 Holm tests: ROI and CLV), there are **0 PROVEN** and **0 PROMISING**. 20 cells are UNPROVEN and 446 are INSUFFICIENT (fewer than 150 bets).
- **CLV:** every cell with at least 150 bets has negative CLV against the close, and in all but one of them the whole 95% CI is below zero.
- **Where the model "likes" a side,** that side's de-vigged probability does **not** move toward the model by the close. The selected-minus-unselected drift is between -0.35 and +0.30 percentage points.
- **After the OPEN snapshot,** the price of OPEN-selected Over 2.5 bets moves slightly **against** us, losing 0.3 (New-Format) to 1.1 (Standard) points of edge by kickoff.

There is no early-market information edge to harvest. Therefore no forward paper track is proposed.

## Data and rules (as registered)

| Item | Value |
|---|---|
| Predictions | champion_recipe, Standard `run_20261110T203611Z`, New-Format `run_20261110T203319Z` (sha256 pinned); stages walk_forward + model holdout; over15 / over25 / over35 / btts; dated < 2026-09-01 |
| OddsAPI store | `NevixAA/wowza-exec` `odds-history` @ `7bbda7d`; read only through `src/oddsapi_history_adapter.py` (blob `dd4278dd`: attach / prices_for / attach_to_universe with explicit purpose; post-kickoff quotes and prices <= 1.01 dropped; adapter's proven fixture mapping) |
| OPEN | C1 'T-48h' (event odds; btts + alternate_totals), purpose `open` |
| BET | BT 'bet' (T-125h Standard / T-153h New-Format), purpose `bet`. **In time, BT is earlier than OPEN.** |
| FD_PRE | football-data pre-match O/U 2.5 (Bet365 else Pinnacle, one book per pair) through the certified one-to-one join `src/fd_prematch.py`, Standard only; its closing pair for CLV only |
| CLOSE | B1 'close' (~9 min before kickoff), purpose `close`: **CLV only, never a bet price** |
| MOVEMENT | B4 featured totals (8 day-timed snapshots), placed by hours before the fixture's own kickoff: edge-decay description only (Over 2.5, before 2026-06-14) |
| Price | per side the median across books; fair = power de-vig of the median pair |
| Calibration | R2 slope of `research/top3-ranking` `src/research/top3_ranking.py` (blob `b7fec7c2`, `_lams`): p_cal = fair + lam·(p − fair), lam fitted walk-forward per (track, role) on rows dated before each bet's month; cells with no lam yet use raw p, flagged RAW |
| Bet rule | ONE configuration: bet when p_cal × odds − 1 > 0; flat 1u |
| CI / p | `study.max_se` (largest of iid / match-day / ISO-week SEs); 95% CI = mean ± 1.96·SE; one-sided normal p |
| Holm family | 466 cells × {ROI, CLV} = 932 tests, enumerated before computing; one-sided α 0.025 |
| PROVEN | n ≥ 150 AND Holm p < 0.025 AND CI lower bound > 0 |

Coverage (counts only; `output/research/opening_price_edge/coverage.json`):

- **C1 (OPEN) is still being collected at the pinned store commit.** There are no OPEN rows yet for Sweden Allsvenskan, Turkish Super Lig, USA MLS or Portuguese Primeira Liga, and Serie B has only 514 of about 1,200 fixtures.
- **Portugal has BT only.** It has no B1 close, so its CLV is missing.
- **National League and the three lower Scottish divisions have no OddsAPI sport key.** They enter through FD_PRE only.
- **The football-data join rate is 0.968-1.000 per league.**

## Q1: ROI at OPEN vs ROI at BT (pooled context, no verdict)

Bets are taken at the median price, with the R2-calibrated model (43% of bets RAW-flagged, i.e. no lam yet).

| Track | Market | Side | OPEN n | OPEN ROI [95% CI] | BET n | BET ROI [95% CI] | FD_PRE n | FD_PRE ROI [95% CI] |
|---|---|---|---:|---|---:|---|---:|---|
| NF | btts | No | 159 | -9.0% [-26.6, +8.6] | 266 | -6.0% [-20.6, +8.7] | - | - |
| NF | btts | Yes | 358 | -7.4% [-17.8, +2.9] | 386 | +2.0% [-8.3, +12.4] | - | - |
| NF | over15 | Under | 225 | -3.1% [-28.7, +22.5] | 364 | -17.6% [-36.9, +1.7] | - | - |
| NF | over15 | Over | 464 | -8.1% [-14.0, -2.1] | 420 | -3.9% [-10.7, +2.9] | - | - |
| NF | over25 | Under | 359 | -2.2% [-14.6, +10.2] | 442 | -6.0% [-16.6, +4.6] | - | - |
| NF | over25 | Over | 626 | -7.4% [-16.1, +1.2] | 571 | -8.5% [-17.4, +0.4] | - | - |
| NF | over35 | Under | 560 | -7.6% [-13.9, -1.4] | 719 | -4.5% [-9.9, +1.0] | - | - |
| NF | over35 | Over | 444 | -14.0% [-30.5, +2.5] | 377 | -10.4% [-27.5, +6.7] | - | - |
| Std | btts | No | 304 | -2.0% [-14.6, +10.6] | 364 | -0.1% [-11.7, +11.5] | - | - |
| Std | btts | Yes | 326 | +0.5% [-12.5, +13.4] | 575 | -4.3% [-14.3, +5.6] | - | - |
| Std | over15 | Under | 339 | -7.7% [-27.6, +12.2] | 458 | -4.2% [-21.5, +13.2] | - | - |
| Std | over15 | Over | 483 | +3.5% [-2.6, +9.7] | 756 | +2.1% [-2.4, +6.7] | - | - |
| Std | over25 | Under | 374 | -17.4% [-29.0, -5.9] | 578 | -11.1% [-20.6, -1.7] | 1140 | -5.6% [-11.8, +0.6] |
| Std | over25 | Over | 571 | -6.2% [-15.4, +3.0] | 1099 | -2.9% [-9.4, +3.5] | 860 | -4.3% [-11.5, +3.0] |
| Std | over35 | Under | 482 | -4.9% [-12.5, +2.8] | 733 | +0.2% [-5.5, +5.8] | - | - |
| Std | over35 | Over | 418 | -8.0% [-25.4, +9.3] | 771 | -8.1% [-21.0, +5.0] | - | - |

Across the 16 OPEN pooled rows, three have a CI that excludes zero, and all three are **negative**: New-Format Over 1.5, New-Format Under 3.5 and Standard Under 2.5. Standard Over 1.5 at OPEN, +3.5% [-2.6, +9.7], is the best OPEN row and is not significant. BET-tier and FD_PRE rows look the same.

## Q2: CLV — does our side's price shorten toward the close when the model likes it?

**No.**

**Pooled CLV of the selected bets against the de-vigged close is negative for every track × market × side.**

| Role | CLV range |
|---|---|
| OPEN | -3.3% to -18.5% |
| BET | -3.1% to -18.1% |
| FD_PRE | -5.7% and -8.2% |

The 95% CI is below zero in every pooled row.

Part of the CLV level is the bookmaker margin: a median price against a vig-free close gives about -4 to -6% even with no movement. So the information test is the **drift** of the selected side's fair probability from the bet snapshot to the close, compared with the rows the model did not select:

| Role | Track | Bets | Mean calibrated EV at selection | Fair drift to close, selected | Fair drift to close, not selected | Share of bets whose side shortened |
|---|---|---:|---:|---:|---:|---:|
| OPEN | Standard | 3,297 | +9.1% | -0.33 pp | +0.02 pp | 42.9% |
| OPEN | New-Format | 3,195 | +7.3% | -0.08 pp | -0.05 pp | 45.1% |
| BET | Standard | 5,334 | +8.7% | -0.03 pp | 0.00 pp | 49.0% |
| BET | New-Format | 3,545 | +7.5% | +0.19 pp | -0.11 pp | 50.8% |
| FD_PRE | Standard | 2,000 | +7.8% | -0.10 pp | 0.00 pp | 37.9% |

The model's selected sides move the same way as everything else, or slightly against us. Fewer than half of the bets shorten.

**Only one cell has a CLV CI above zero:** Dutch Eredivisie Over 2.5 at BET, with **2 bets**. That is noise.

## Q3: line movement — how much edge disappears, and by when?

**No cell beats OPEN,** so there is no edge whose decay could be timed.

The descriptive path below uses Over 2.5 only, the one market with the B4 intermediate snapshots. It shows the mean of odds_selected × fair_t − 1 for the selected bets: "how good was the price we took, valued at the market's view at time t". The B4 featured-totals book set differs from the event-odds tiers, so compare within a row.

| Selected at | Track | vs BT fair (T-125/153h) | vs OPEN fair (T-48h) | B4 >36h | B4 18-36h | B4 8-18h | B4 3-8h | B4 0.5-3h | vs CLOSE fair (T-9 min) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OPEN | Standard | -5.5% | -6.3% | -5.9% | -6.5% | -6.5% | -6.8% | -6.9% | -7.4% |
| OPEN | New-Format | -5.3% | -6.8% | -6.3% | -6.4% | -6.0% | -5.9% | -5.8% | -7.1% |
| BET | Standard | -6.6% | -6.1% | -6.0% | -6.3% | -6.1% | -6.3% | -6.3% | -6.8% |
| BET | New-Format | -6.7% | -6.3% | -6.0% | -6.0% | -5.9% | -5.6% | -5.6% | -6.6% |

How to read the path:

- **OPEN picks sit on sides the market had already moved away from between T-125/153h and T-48h** (Standard -5.5% to -6.3%, New-Format -5.3% to -6.8%).
- **The move continues by the close:** -7.4% Standard, -7.1% New-Format, i.e. about **0.3 to 1.1 points of edge lost after OPEN**.
- **BET picks are essentially flat from bet time to close** (-6.6% to -6.8% Standard, -6.7% to -6.6% New-Format).

The model tends to like prices the market is leaving, not prices the market is about to move to. Betting earlier than T-48h would not have helped either: the BT-tier cells show no edge.

## Q4: practical check — when are our live tips first generated?

Source: the live ledgers on `origin/main` @ `ab48dec99`, `output/bets_ledger.csv` and `output/side_bets_ledger.csv` (source live, first sight). Kickoff is the provider's actual kickoff from the store listings through the adapter's fixture mapping. Of 2,657 first sights, 2,420 have a proven kickoff and a non-negative lead time. 230 are unmatched or have no kickoff, and 7 were generated after kickoff; none of these is guessed.

| Track | Market | First sights with kickoff | Median hours before KO | ≥ 48h (OPEN reachable) | ≥ 125h | ≥ 153h | < 24h |
|---|---|---:|---:|---:|---:|---:|---:|
| NF | btts | 306 | 156 | 86% | 63% | 51% | 8% |
| NF | over15 | 134 | 164 | 83% | 68% | 60% | 13% |
| NF | over25 | 1261 | 149 | 85% | 64% | 46% | 10% |
| NF | over35 | 28 | 81 | 64% | 21% | 14% | 29% |
| Std | btts | 185 | 119 | 79% | 46% | 25% | 16% |
| Std | over15 | 17 | 71 | 65% | 35% | 6% | 35% |
| Std | over25 | 478 | 126 | 82% | 51% | 35% | 13% |
| Std | over35 | 11 | 50 | 55% | 9% | 9% | 45% |
| **All** | | **2420** | **145** | **83%** | **60%** | **43%** | |

- **Timing makes OPEN prices reachable.** 83% of live first sights come at least 48 h before kickoff, and the median is about 6 days.
- **There is no live T-48h capture today.** The live tip carries whatever price the board reads at generation, and the C1 OPEN snapshot exists only in the bought history. An OPEN-price strategy would need a scheduled live snapshot at T-48h.
- **Given Q1-Q3, that capture is not worth credits for this purpose.**

## Cells with at least 150 bets (all UNPROVEN)

Holm-adjusted p is 1.000 for every one of these cells.

| Track | League | Market | Side | Role | n | ROI [95% CI] | CLV [95% CI] |
|---|---|---|---|---|---:|---|---|
| NF | Austrian Bundesliga | over15 | Under | BET | 151 | -12.6% [-42.7, +17.6] | -11.1% [-14.2, -8.0] |
| NF | Austrian Bundesliga | over25 | Under | BET | 167 | +12.3% [-5.5, +30.1] | -1.7% [-3.6, +0.2] |
| NF | Austrian Bundesliga | over25 | Over | OPEN | 156 | +1.1% [-15.2, +17.4] | -7.0% [-8.0, -5.9] |
| NF | Austrian Bundesliga | over35 | Under | BET | 219 | +1.1% [-8.8, +10.9] | -0.8% [-1.9, +0.3] |
| NF | Austrian Bundesliga | over35 | Under | OPEN | 224 | +0.0% [-9.8, +9.9] | -2.2% [-3.1, -1.4] |
| NF | Mexico Liga MX | over35 | Under | BET | 153 | -2.1% [-13.6, +9.3] | -3.4% [-4.6, -2.2] |
| Std | Championship | over25 | Over | BET | 179 | -6.0% [-24.2, +12.2] | -5.6% [-6.8, -4.3] |
| Std | Championship | over25 | Over | FD_PRE | 186 | -6.6% [-22.0, +8.8] | -7.7% [-8.9, -6.5] |
| Std | Championship | over25 | Over | OPEN | 155 | -8.1% [-26.1, +9.9] | -6.3% [-7.4, -5.3] |
| Std | League One | over25 | Under | FD_PRE | 172 | +9.7% [-6.7, +26.1] | -2.8% [-3.9, -1.8] |
| Std | Scottish League Two | over25 | Under | FD_PRE | 235 | +7.4% [-6.2, +21.1] | -4.2% [-5.4, -3.0] |
| Std | Scottish League Two | over25 | Over | FD_PRE | 193 | -1.2% [-16.0, +13.5] | -6.2% [-7.3, -5.2] |
| Std | Serie B | btts | Yes | BET | 273 | +7.1% [-4.8, +19.0] | -7.1% [-7.9, -6.3] |
| Std | Serie B | over15 | Over | BET | 435 | +3.2% [-2.7, +9.2] | -3.2% [-3.5, -2.8] |
| Std | Serie B | over15 | Over | OPEN | 258 | +2.5% [-5.5, +10.4] | -3.8% [-4.4, -3.2] |
| Std | Serie B | over25 | Over | BET | 504 | +3.3% [-6.7, +13.2] | -6.9% [-7.6, -6.2] |
| Std | Serie B | over25 | Over | OPEN | 160 | +2.4% [-16.0, +20.8] | -8.6% [-9.7, -7.4] |
| Std | Serie B | over35 | Over | BET | 404 | -3.8% [-23.9, +16.4] | -15.5% [-16.7, -14.3] |
| Std | Turkish Super Lig | over25 | Under | BET | 154 | -3.7% [-23.3, +15.8] | -8.7% [-10.5, -6.8] |
| Std | Turkish Super Lig | over35 | Under | BET | 174 | -0.1% [-12.3, +12.2] | -4.0% [-5.4, -2.7] |

### Near misses (for the record; none qualifies)

- **New-Format Austrian Bundesliga Under 2.5 at OPEN:** 148 bets, ROI +20.7% (unadjusted 95% LB +0.1%). This is below the 150-bet floor. Its CLV is -4.2% [-5.6, -2.8]: the price moved against these bets, which points to outcome variance, not information.
- **Saudi Pro League Over 2.5 OPEN** (34 bets), **Denmark Over 1.5 OPEN** (5 bets) and **Bundesliga 2 Over 1.5 OPEN** (3 bets) have a positive unadjusted LB only because they have very few bets.

## Which leagues / markets look promising for early betting?

**None.**

| Verdict | Count |
|---|---:|
| PROVEN | 0 |
| PROMISING | 0 |
| EARLY_ONLY (beats OPEN where it does not beat BT) | 0 |

The forward paper protocol (`opening-price-edge-forward/1`) is therefore registered with **NO_ELIGIBLE_CELLS**. It opens no paper track, and it says that only a new registered protocol on new data could name cells.

## Limitations (stated, not corrected after the fact)

1. **Development evidence only.** Rows before 2026-09-01 include stretches whose outcomes earlier studies saw (the bars fresh-holdout windows). No historical holdout is claimed.
2. **Many bets are RAW.** 43% of bets (bet-weighted) had no walk-forward lam yet, so they used the raw model probability. 203 of the 415 cells with bets are entirely RAW, mostly early months and thin cells. The calibrated bets do no better: every pooled row is at or below zero either way.
3. **The bet rule is strict.** With lam shrunk toward 0, calibrated EV > 0 is rare, so most cells have fewer than 150 bets (INSUFFICIENT). That was the single registered configuration; no other bar was searched.
4. **Tiny-n p-values.** A cell with 2-5 bets that all won gets a near-zero p-value from the block SE. The registered rule gives p = 1 only for n < 2 or SE 0. These tiny cells are INSUFFICIENT by the 150-bet floor, so no verdict changes.
5. **C1 is incomplete at the pinned commit.** Sweden, Turkey, MLS and Portugal have no OPEN rows, and Serie B is partial. Those cells counted in the Holm family with no bets, which is conservative.
6. **CLV level includes the margin,** because a median price is compared with a vig-free close. The selected-versus-unselected drift in Q2 is the margin-free comparison.

## Reproduce

```
python scripts/opening_price_edge.py prepare --local <dir outside the repo>   # prices local; coverage counts
python scripts/opening_price_edge.py run --local <dir>                        # refuses unless the protocol hash is registered
python scripts/opening_price_edge.py timing                                   # Q4
python scripts/opening_price_edge_report.py                                   # the findings table below
```

## Full findings table (every cell with at least one bet)

There are 466 registered cells. 51 cells with no bets are omitted here; they are listed in `output/research/opening_price_edge/results.json`.

| Track | League | Market | Side | Role | n bets | ROI | ROI 95% CI | Holm p (ROI) | CLV vs close | CLV 95% CI | RAW share | Verdict |
|---|---|---|---|---|---:|---:|---|---:|---:|---|---:|---|
| NF | Argentina Primera Division | btts | Yes | OPEN | 38 | -39.0% | [-69.8, -8.1] | 1.000 | -8.7% | [-10.7, -6.8] | +100.0% | INSUFFICIENT |
| NF | Argentina Primera Division | btts | Yes | BET | 41 | -12.1% | [-44.6, +20.4] | 1.000 | -10.7% | [-12.9, -8.4] | +100.0% | INSUFFICIENT |
| NF | Argentina Primera Division | over15 | Over | OPEN | 54 | -21.6% | [-41.2, -1.9] | 1.000 | -3.6% | [-4.7, -2.6] | +100.0% | INSUFFICIENT |
| NF | Argentina Primera Division | over15 | Over | BET | 43 | -2.8% | [-29.3, +23.8] | 1.000 | -5.5% | [-6.8, -4.1] | +100.0% | INSUFFICIENT |
| NF | Argentina Primera Division | over25 | Over | OPEN | 52 | -30.8% | [-61.4, -0.2] | 1.000 | -7.7% | [-9.8, -5.6] | +100.0% | INSUFFICIENT |
| NF | Argentina Primera Division | over25 | Over | BET | 42 | -28.7% | [-74.2, +16.7] | 1.000 | -11.0% | [-13.3, -8.7] | +100.0% | INSUFFICIENT |
| NF | Argentina Primera Division | over35 | Over | OPEN | 52 | -40.1% | [-83.4, +3.3] | 1.000 | -16.9% | [-19.6, -14.3] | +100.0% | INSUFFICIENT |
| NF | Argentina Primera Division | over35 | Over | BET | 42 | -7.4% | [-69.7, +54.9] | 1.000 | -21.1% | [-24.2, -17.9] | +100.0% | INSUFFICIENT |
| NF | Austrian Bundesliga | btts | No | OPEN | 80 | +9.9% | [-15.0, +34.8] | 1.000 | -6.4% | [-8.2, -4.5] | +45.0% | INSUFFICIENT |
| NF | Austrian Bundesliga | btts | No | BET | 102 | -7.8% | [-29.3, +13.7] | 1.000 | -4.9% | [-6.6, -3.2] | +21.6% | INSUFFICIENT |
| NF | Austrian Bundesliga | btts | Yes | OPEN | 47 | +8.3% | [-19.3, +35.9] | 1.000 | -7.0% | [-9.0, -5.0] | +6.4% | INSUFFICIENT |
| NF | Austrian Bundesliga | btts | Yes | BET | 58 | +10.5% | [-13.7, +34.7] | 1.000 | -4.2% | [-6.2, -2.1] | +6.9% | INSUFFICIENT |
| NF | Austrian Bundesliga | over15 | Under | OPEN | 109 | -10.9% | [-47.8, +26.0] | 1.000 | -15.8% | [-18.1, -13.4] | +32.1% | INSUFFICIENT |
| NF | Austrian Bundesliga | over15 | Under | BET | 151 | -12.6% | [-42.7, +17.6] | 1.000 | -11.1% | [-14.2, -8.0] | +17.9% | UNPROVEN |
| NF | Austrian Bundesliga | over15 | Over | OPEN | 71 | -1.1% | [-16.0, +13.9] | 1.000 | -4.5% | [-5.2, -3.9] | +5.6% | INSUFFICIENT |
| NF | Austrian Bundesliga | over15 | Over | BET | 56 | -4.9% | [-21.5, +11.7] | 1.000 | -3.7% | [-5.1, -2.3] | +7.1% | INSUFFICIENT |
| NF | Austrian Bundesliga | over25 | Under | OPEN | 148 | +20.7% | [+0.1, +41.4] | 1.000 | -4.2% | [-5.6, -2.7] | +28.4% | INSUFFICIENT |
| NF | Austrian Bundesliga | over25 | Under | BET | 167 | +12.3% | [-5.5, +30.1] | 1.000 | -1.7% | [-3.6, +0.2] | +16.2% | UNPROVEN |
| NF | Austrian Bundesliga | over25 | Over | OPEN | 156 | +1.1% | [-15.2, +17.4] | 1.000 | -7.0% | [-8.0, -5.9] | +2.6% | UNPROVEN |
| NF | Austrian Bundesliga | over25 | Over | BET | 143 | +4.4% | [-12.1, +20.9] | 1.000 | -6.4% | [-8.0, -4.9] | +2.8% | INSUFFICIENT |
| NF | Austrian Bundesliga | over35 | Under | OPEN | 224 | +0.0% | [-9.8, +9.9] | 1.000 | -2.2% | [-3.1, -1.4] | +24.6% | UNPROVEN |
| NF | Austrian Bundesliga | over35 | Under | BET | 219 | +1.1% | [-8.8, +10.9] | 1.000 | -0.8% | [-1.9, +0.3] | +26.0% | UNPROVEN |
| NF | Austrian Bundesliga | over35 | Over | OPEN | 104 | -4.4% | [-37.0, +28.1] | 1.000 | -15.0% | [-16.7, -13.2] | +1.0% | INSUFFICIENT |
| NF | Austrian Bundesliga | over35 | Over | BET | 92 | -4.2% | [-36.9, +28.6] | 1.000 | -10.3% | [-13.3, -7.3] | +0.0% | INSUFFICIENT |
| NF | Brazil Serie A | btts | No | OPEN | 1 | -100.0% | - | 1.000 | +3.1% | - | +100.0% | INSUFFICIENT |
| NF | Brazil Serie A | btts | Yes | OPEN | 59 | -5.3% | [-32.0, +21.4] | 1.000 | -6.8% | [-8.7, -4.9] | +44.1% | INSUFFICIENT |
| NF | Brazil Serie A | btts | Yes | BET | 32 | -18.0% | [-53.0, +17.1] | 1.000 | -9.4% | [-11.5, -7.3] | +100.0% | INSUFFICIENT |
| NF | Brazil Serie A | over15 | Under | OPEN | 1 | -100.0% | - | 1.000 | -28.6% | - | +100.0% | INSUFFICIENT |
| NF | Brazil Serie A | over15 | Over | OPEN | 71 | +2.7% | [-17.1, +22.5] | 1.000 | -4.2% | [-5.3, -3.2] | +56.3% | INSUFFICIENT |
| NF | Brazil Serie A | over15 | Over | BET | 57 | +0.8% | [-19.1, +20.6] | 1.000 | -4.6% | [-6.2, -3.0] | +100.0% | INSUFFICIENT |
| NF | Brazil Serie A | over25 | Under | OPEN | 5 | -19.4% | [-132.2, +93.4] | 1.000 | -5.9% | [-12.8, +1.0] | +20.0% | INSUFFICIENT |
| NF | Brazil Serie A | over25 | Under | BET | 1 | -100.0% | - | 1.000 | +1.9% | - | +100.0% | INSUFFICIENT |
| NF | Brazil Serie A | over25 | Over | OPEN | 67 | -15.7% | [-41.3, +10.0] | 1.000 | -9.5% | [-11.3, -7.7] | +50.7% | INSUFFICIENT |
| NF | Brazil Serie A | over25 | Over | BET | 51 | -22.0% | [-50.1, +6.1] | 1.000 | -10.5% | [-13.2, -7.9] | +100.0% | INSUFFICIENT |
| NF | Brazil Serie A | over35 | Under | OPEN | 9 | +10.8% | [-30.3, +51.9] | 1.000 | -0.7% | [-5.8, +4.5] | +44.4% | INSUFFICIENT |
| NF | Brazil Serie A | over35 | Under | BET | 5 | -12.0% | [-82.8, +58.8] | 1.000 | +0.2% | [-6.2, +6.7] | +80.0% | INSUFFICIENT |
| NF | Brazil Serie A | over35 | Over | OPEN | 56 | -34.6% | [-77.2, +8.1] | 1.000 | -14.3% | [-17.9, -10.7] | +50.0% | INSUFFICIENT |
| NF | Brazil Serie A | over35 | Over | BET | 26 | -20.3% | [-85.0, +44.4] | 1.000 | -15.6% | [-19.2, -11.9] | +100.0% | INSUFFICIENT |
| NF | China Super League | btts | No | OPEN | 6 | -100.0% | - | 1.000 | -5.2% | [-9.2, -1.1] | +100.0% | INSUFFICIENT |
| NF | China Super League | btts | No | BET | 12 | -8.2% | [-74.1, +57.8] | 1.000 | -3.4% | [-8.0, +1.1] | +100.0% | INSUFFICIENT |
| NF | China Super League | btts | Yes | OPEN | 26 | +6.3% | [-35.7, +48.4] | 1.000 | -5.2% | [-10.0, -0.5] | +100.0% | INSUFFICIENT |
| NF | China Super League | btts | Yes | BET | 10 | -35.2% | [-100.2, +29.8] | 1.000 | -6.8% | [-17.8, +4.2] | +100.0% | INSUFFICIENT |
| NF | China Super League | over15 | Under | OPEN | 3 | -100.0% | - | 1.000 | -4.9% | [-7.0, -2.9] | +100.0% | INSUFFICIENT |
| NF | China Super League | over15 | Under | BET | 21 | -17.0% | [-92.2, +58.2] | 1.000 | -18.6% | [-26.5, -10.6] | +100.0% | INSUFFICIENT |
| NF | China Super League | over15 | Over | OPEN | 17 | -29.0% | [-72.5, +14.5] | 1.000 | -5.3% | [-8.4, -2.3] | +100.0% | INSUFFICIENT |
| NF | China Super League | over15 | Over | BET | 5 | -17.6% | [-83.6, +48.4] | 1.000 | -4.8% | [-8.2, -1.3] | +100.0% | INSUFFICIENT |
| NF | China Super League | over25 | Under | OPEN | 18 | -52.1% | [-94.8, -9.3] | 1.000 | -16.2% | [-23.4, -9.0] | +100.0% | INSUFFICIENT |
| NF | China Super League | over25 | Under | BET | 21 | -23.9% | [-71.5, +23.7] | 1.000 | -11.3% | [-19.0, -3.6] | +100.0% | INSUFFICIENT |
| NF | China Super League | over25 | Over | OPEN | 15 | -0.5% | [-68.8, +67.8] | 1.000 | -6.9% | [-14.6, +0.8] | +100.0% | INSUFFICIENT |
| NF | China Super League | over25 | Over | BET | 7 | -68.0% | [-130.7, -5.3] | 1.000 | -9.3% | [-13.4, -5.2] | +100.0% | INSUFFICIENT |
| NF | China Super League | over35 | Under | OPEN | 10 | +0.7% | [-42.4, +43.8] | 1.000 | -3.8% | [-8.4, +0.8] | +100.0% | INSUFFICIENT |
| NF | China Super League | over35 | Under | BET | 22 | -17.9% | [-50.2, +14.2] | 1.000 | -6.8% | [-12.3, -1.3] | +100.0% | INSUFFICIENT |
| NF | China Super League | over35 | Over | OPEN | 7 | -44.6% | [-153.2, +64.1] | 1.000 | -8.3% | [-32.6, +15.9] | +100.0% | INSUFFICIENT |
| NF | China Super League | over35 | Over | BET | 7 | -41.7% | [-156.0, +72.5] | 1.000 | -18.3% | [-23.8, -12.8] | +100.0% | INSUFFICIENT |
| NF | Denmark Superliga | btts | No | OPEN | 19 | -15.8% | [-72.3, +40.8] | 1.000 | -10.1% | [-13.8, -6.4] | +100.0% | INSUFFICIENT |
| NF | Denmark Superliga | btts | No | BET | 19 | +4.3% | [-71.0, +79.7] | 1.000 | -6.0% | [-8.5, -3.6] | +100.0% | INSUFFICIENT |
| NF | Denmark Superliga | btts | Yes | OPEN | 4 | -52.0% | [-159.8, +55.8] | 1.000 | -8.1% | [-12.1, -4.2] | +100.0% | INSUFFICIENT |
| NF | Denmark Superliga | btts | Yes | BET | 4 | -53.2% | [-144.9, +38.4] | 1.000 | +0.2% | [-12.8, +13.2] | +100.0% | INSUFFICIENT |
| NF | Denmark Superliga | over15 | Under | OPEN | 13 | -26.8% | [-145.7, +92.1] | 1.000 | -22.1% | [-33.2, -11.1] | +100.0% | INSUFFICIENT |
| NF | Denmark Superliga | over15 | Under | BET | 20 | -39.2% | [-154.5, +76.0] | 1.000 | -18.6% | [-23.4, -13.7] | +100.0% | INSUFFICIENT |
| NF | Denmark Superliga | over15 | Over | OPEN | 5 | +32.8% | [+29.4, +36.2] | 6.65e-79 | -2.4% | [-6.8, +2.1] | +100.0% | INSUFFICIENT |
| NF | Denmark Superliga | over15 | Over | BET | 3 | +30.7% | [+28.3, +33.0] | 6.03e-141 | +2.0% | [-4.2, +8.3] | +100.0% | INSUFFICIENT |
| NF | Denmark Superliga | over25 | Under | OPEN | 25 | -33.3% | [-76.2, +9.6] | 1.000 | -6.0% | [-10.5, -1.5] | +68.0% | INSUFFICIENT |
| NF | Denmark Superliga | over25 | Under | BET | 19 | -43.5% | [-120.0, +32.9] | 1.000 | -6.0% | [-9.6, -2.4] | +100.0% | INSUFFICIENT |
| NF | Denmark Superliga | over25 | Over | OPEN | 9 | -11.9% | [-96.9, +73.0] | 1.000 | -6.5% | [-12.3, -0.6] | +77.8% | INSUFFICIENT |
| NF | Denmark Superliga | over25 | Over | BET | 5 | -24.5% | [-115.1, +66.1] | 1.000 | -5.1% | [-13.5, +3.2] | +100.0% | INSUFFICIENT |
| NF | Denmark Superliga | over35 | Under | OPEN | 41 | -12.4% | [-35.3, +10.5] | 1.000 | -4.4% | [-6.5, -2.4] | +73.2% | INSUFFICIENT |
| NF | Denmark Superliga | over35 | Under | BET | 35 | -11.7% | [-36.9, +13.6] | 1.000 | -4.5% | [-6.3, -2.6] | +100.0% | INSUFFICIENT |
| NF | Denmark Superliga | over35 | Over | OPEN | 2 | -100.0% | - | 1.000 | -16.7% | [-28.6, -4.8] | +100.0% | INSUFFICIENT |
| NF | Finland Veikkausliiga | btts | No | OPEN | 1 | -100.0% | - | 1.000 | -9.9% | - | +0.0% | INSUFFICIENT |
| NF | Finland Veikkausliiga | btts | No | BET | 6 | -62.0% | [-136.5, +12.5] | 1.000 | -8.2% | [-17.8, +1.4] | +0.0% | INSUFFICIENT |
| NF | Finland Veikkausliiga | btts | Yes | OPEN | 39 | -15.3% | [-49.2, +18.7] | 1.000 | -5.9% | [-8.2, -3.6] | +76.9% | INSUFFICIENT |
| NF | Finland Veikkausliiga | btts | Yes | BET | 34 | +13.6% | [-24.6, +51.7] | 1.000 | -3.8% | [-6.1, -1.6] | +73.5% | INSUFFICIENT |
| NF | Finland Veikkausliiga | over15 | Under | OPEN | 5 | -8.0% | [-188.3, +172.3] | 1.000 | -20.6% | [-29.0, -12.1] | +0.0% | INSUFFICIENT |
| NF | Finland Veikkausliiga | over15 | Under | BET | 16 | -47.7% | [-121.7, +26.3] | 1.000 | -22.8% | [-28.4, -17.3] | +6.2% | INSUFFICIENT |
| NF | Finland Veikkausliiga | over15 | Over | OPEN | 43 | +6.7% | [-9.9, +23.3] | 1.000 | -3.5% | [-5.0, -1.9] | +100.0% | INSUFFICIENT |
| NF | Finland Veikkausliiga | over15 | Over | BET | 24 | -2.8% | [-28.3, +22.8] | 1.000 | -0.6% | [-2.5, +1.3] | +91.7% | INSUFFICIENT |
| NF | Finland Veikkausliiga | over25 | Under | OPEN | 6 | -64.4% | [-134.2, +5.3] | 1.000 | -11.4% | [-16.1, -6.7] | +33.3% | INSUFFICIENT |
| NF | Finland Veikkausliiga | over25 | Under | BET | 13 | -49.6% | [-103.0, +3.9] | 1.000 | -7.8% | [-13.7, -1.9] | +0.0% | INSUFFICIENT |
| NF | Finland Veikkausliiga | over25 | Over | OPEN | 38 | -6.7% | [-52.4, +39.1] | 1.000 | -7.1% | [-10.1, -4.1] | +86.8% | INSUFFICIENT |
| NF | Finland Veikkausliiga | over25 | Over | BET | 32 | -1.5% | [-42.5, +39.5] | 1.000 | -3.6% | [-6.6, -0.6] | +71.9% | INSUFFICIENT |
| NF | Finland Veikkausliiga | over35 | Under | OPEN | 23 | -49.0% | [-81.7, -16.2] | 1.000 | -3.3% | [-5.5, -1.1] | +21.7% | INSUFFICIENT |
| NF | Finland Veikkausliiga | over35 | Under | BET | 33 | -19.2% | [-46.2, +7.7] | 1.000 | -3.8% | [-5.8, -1.7] | +27.3% | INSUFFICIENT |
| NF | Finland Veikkausliiga | over35 | Over | OPEN | 32 | -3.9% | [-62.7, +55.0] | 1.000 | -11.2% | [-15.1, -7.2] | +87.5% | INSUFFICIENT |
| NF | Finland Veikkausliiga | over35 | Over | BET | 28 | +35.5% | [-33.2, +104.3] | 1.000 | -10.7% | [-15.9, -5.5] | +71.4% | INSUFFICIENT |
| NF | Ireland Premier Division | btts | No | OPEN | 6 | +9.9% | [-86.5, +106.3] | 1.000 | -6.8% | [-11.6, -1.9] | +16.7% | INSUFFICIENT |
| NF | Ireland Premier Division | btts | No | BET | 20 | +50.9% | [+6.2, +95.7] | 1.000 | -8.5% | [-10.8, -6.1] | +5.0% | INSUFFICIENT |
| NF | Ireland Premier Division | btts | Yes | OPEN | 46 | +17.4% | [-11.4, +46.2] | 1.000 | -7.6% | [-9.2, -6.0] | +34.8% | INSUFFICIENT |
| NF | Ireland Premier Division | btts | Yes | BET | 88 | +27.7% | [+6.0, +49.4] | 1.000 | -6.4% | [-7.8, -5.1] | +26.1% | INSUFFICIENT |
| NF | Ireland Premier Division | over15 | Under | OPEN | 4 | +101.2% | [-126.7, +329.2] | 1.000 | -14.8% | [-22.9, -6.6] | +100.0% | INSUFFICIENT |
| NF | Ireland Premier Division | over15 | Under | BET | 10 | -9.2% | [-132.3, +114.0] | 1.000 | -12.8% | [-19.5, -6.2] | +10.0% | INSUFFICIENT |
| NF | Ireland Premier Division | over15 | Over | OPEN | 23 | -12.8% | [-42.8, +17.2] | 1.000 | -5.1% | [-6.9, -3.4] | +60.9% | INSUFFICIENT |
| NF | Ireland Premier Division | over15 | Over | BET | 46 | +7.8% | [-9.9, +25.5] | 1.000 | -3.2% | [-4.5, -1.9] | +45.6% | INSUFFICIENT |
| NF | Ireland Premier Division | over25 | Under | OPEN | 5 | +22.1% | [-100.0, +144.2] | 1.000 | -8.8% | [-14.8, -2.8] | +100.0% | INSUFFICIENT |
| NF | Ireland Premier Division | over25 | Under | BET | 24 | -0.2% | [-49.8, +49.4] | 1.000 | -4.9% | [-8.0, -1.8] | +8.3% | INSUFFICIENT |
| NF | Ireland Premier Division | over25 | Over | OPEN | 28 | -20.0% | [-61.6, +21.6] | 1.000 | -9.0% | [-11.7, -6.2] | +60.7% | INSUFFICIENT |
| NF | Ireland Premier Division | over25 | Over | BET | 52 | -18.0% | [-49.2, +13.1] | 1.000 | -5.7% | [-7.9, -3.5] | +51.9% | INSUFFICIENT |
| NF | Ireland Premier Division | over35 | Under | OPEN | 8 | +6.4% | [-39.1, +52.0] | 1.000 | -3.7% | [-6.8, -0.6] | +87.5% | INSUFFICIENT |
| NF | Ireland Premier Division | over35 | Under | BET | 28 | -5.6% | [-33.3, +22.1] | 1.000 | -2.3% | [-4.1, -0.6] | +7.1% | INSUFFICIENT |
| NF | Ireland Premier Division | over35 | Over | OPEN | 25 | -1.5% | [-75.5, +72.5] | 1.000 | -16.5% | [-20.8, -12.2] | +68.0% | INSUFFICIENT |
| NF | Ireland Premier Division | over35 | Over | BET | 37 | -40.0% | [-85.1, +5.1] | 1.000 | -13.0% | [-16.8, -9.1] | +70.3% | INSUFFICIENT |
| NF | Japan J-League | btts | No | OPEN | 12 | -36.8% | [-101.5, +27.9] | 1.000 | -12.5% | [-15.3, -9.7] | +83.3% | INSUFFICIENT |
| NF | Japan J-League | btts | No | BET | 44 | -5.5% | [-44.0, +33.0] | 1.000 | -9.0% | [-12.3, -5.8] | +15.9% | INSUFFICIENT |
| NF | Japan J-League | btts | Yes | OPEN | 30 | -10.8% | [-48.0, +26.5] | 1.000 | -8.0% | [-9.5, -6.5] | +50.0% | INSUFFICIENT |
| NF | Japan J-League | btts | Yes | BET | 66 | +0.0% | [-25.7, +25.7] | 1.000 | -7.6% | [-9.0, -6.2] | +15.2% | INSUFFICIENT |
| NF | Japan J-League | over15 | Under | OPEN | 11 | -6.0% | [-129.7, +117.7] | 1.000 | -25.5% | [-28.6, -22.4] | +72.7% | INSUFFICIENT |
| NF | Japan J-League | over15 | Under | BET | 45 | +8.2% | [-49.0, +65.4] | 1.000 | -17.0% | [-22.8, -11.2] | +8.9% | INSUFFICIENT |
| NF | Japan J-League | over15 | Over | OPEN | 80 | -13.2% | [-29.1, +2.8] | 1.000 | -4.5% | [-5.4, -3.7] | +45.0% | INSUFFICIENT |
| NF | Japan J-League | over15 | Over | BET | 109 | -8.1% | [-21.1, +4.9] | 1.000 | -4.2% | [-5.1, -3.4] | +22.0% | INSUFFICIENT |
| NF | Japan J-League | over25 | Under | OPEN | 17 | -28.7% | [-82.9, +25.5] | 1.000 | -12.2% | [-16.8, -7.6] | +41.2% | INSUFFICIENT |
| NF | Japan J-League | over25 | Under | BET | 69 | -16.5% | [-43.4, +10.3] | 1.000 | -6.8% | [-9.4, -4.2] | +4.3% | INSUFFICIENT |
| NF | Japan J-League | over25 | Over | OPEN | 83 | -13.9% | [-37.6, +9.9] | 1.000 | -9.4% | [-11.1, -7.8] | +41.0% | INSUFFICIENT |
| NF | Japan J-League | over25 | Over | BET | 140 | -6.6% | [-25.2, +12.1] | 1.000 | -7.9% | [-9.4, -6.5] | +17.1% | INSUFFICIENT |
| NF | Japan J-League | over35 | Under | OPEN | 44 | -11.5% | [-35.6, +12.6] | 1.000 | -5.9% | [-7.6, -4.1] | +34.1% | INSUFFICIENT |
| NF | Japan J-League | over35 | Under | BET | 98 | +1.3% | [-13.2, +15.7] | 1.000 | -4.0% | [-5.5, -2.6] | +17.3% | INSUFFICIENT |
| NF | Japan J-League | over35 | Over | OPEN | 63 | -26.5% | [-67.1, +14.2] | 1.000 | -17.9% | [-20.5, -15.2] | +31.8% | INSUFFICIENT |
| NF | Japan J-League | over35 | Over | BET | 92 | -21.4% | [-58.4, +15.5] | 1.000 | -15.1% | [-17.7, -12.6] | +10.9% | INSUFFICIENT |
| NF | K-League 1 | btts | No | OPEN | 8 | +4.3% | [-73.2, +81.8] | 1.000 | -6.8% | [-11.5, -2.0] | +12.5% | INSUFFICIENT |
| NF | K-League 1 | btts | No | BET | 1 | -100.0% | - | 1.000 | -7.0% | - | +100.0% | INSUFFICIENT |
| NF | K-League 1 | btts | Yes | OPEN | 32 | -7.3% | [-40.0, +25.3] | 1.000 | -5.5% | [-7.0, -3.9] | +21.9% | INSUFFICIENT |
| NF | K-League 1 | btts | Yes | BET | 11 | -13.0% | [-72.1, +46.2] | 1.000 | -3.8% | [-8.1, +0.4] | +100.0% | INSUFFICIENT |
| NF | K-League 1 | over15 | Under | OPEN | 6 | -100.0% | - | 1.000 | -18.5% | [-21.7, -15.3] | +16.7% | INSUFFICIENT |
| NF | K-League 1 | over15 | Under | BET | 2 | -100.0% | - | 1.000 | -26.8% | [-30.5, -23.1] | +100.0% | INSUFFICIENT |
| NF | K-League 1 | over15 | Over | OPEN | 71 | -9.9% | [-25.3, +5.6] | 1.000 | -3.6% | [-4.4, -2.9] | +39.4% | INSUFFICIENT |
| NF | K-League 1 | over15 | Over | BET | 22 | -25.7% | [-55.9, +4.5] | 1.000 | -1.3% | [-3.7, +1.1] | +100.0% | INSUFFICIENT |
| NF | K-League 1 | over25 | Under | OPEN | 34 | -3.9% | [-38.0, +30.2] | 1.000 | -6.3% | [-8.5, -4.2] | +2.9% | INSUFFICIENT |
| NF | K-League 1 | over25 | Under | BET | 2 | -7.5% | [-188.8, +173.8] | 1.000 | -7.0% | [-21.9, +7.9] | +50.0% | INSUFFICIENT |
| NF | K-League 1 | over25 | Over | OPEN | 113 | -8.1% | [-26.9, +10.8] | 1.000 | -6.3% | [-7.6, -5.0] | +26.6% | INSUFFICIENT |
| NF | K-League 1 | over25 | Over | BET | 23 | -36.5% | [-76.6, +3.6] | 1.000 | -4.9% | [-9.0, -0.8] | +95.7% | INSUFFICIENT |
| NF | K-League 1 | over35 | Under | OPEN | 48 | -15.4% | [-35.4, +4.7] | 1.000 | -3.7% | [-4.8, -2.6] | +12.5% | INSUFFICIENT |
| NF | K-League 1 | over35 | Under | BET | 11 | +4.5% | [-38.8, +47.7] | 1.000 | -8.2% | [-11.6, -4.7] | +100.0% | INSUFFICIENT |
| NF | K-League 1 | over35 | Over | OPEN | 64 | +3.8% | [-45.5, +53.0] | 1.000 | -14.1% | [-16.1, -12.1] | +28.1% | INSUFFICIENT |
| NF | K-League 1 | over35 | Over | BET | 11 | +1.4% | [-101.4, +104.1] | 1.000 | -14.4% | [-20.0, -8.8] | +100.0% | INSUFFICIENT |
| NF | Mexico Liga MX | btts | No | OPEN | 9 | -25.6% | [-98.5, +47.4] | 1.000 | -5.9% | [-10.4, -1.4] | +100.0% | INSUFFICIENT |
| NF | Mexico Liga MX | btts | No | BET | 19 | -53.2% | [-95.7, -10.6] | 1.000 | -8.2% | [-12.0, -4.4] | +100.0% | INSUFFICIENT |
| NF | Mexico Liga MX | btts | Yes | OPEN | 5 | -61.8% | [-142.9, +19.3] | 1.000 | -7.5% | [-16.1, +1.1] | +100.0% | INSUFFICIENT |
| NF | Mexico Liga MX | btts | Yes | BET | 5 | +16.6% | [-92.0, +125.2] | 1.000 | -2.0% | [-9.1, +5.1] | +100.0% | INSUFFICIENT |
| NF | Mexico Liga MX | over15 | Under | OPEN | 20 | -31.3% | [-106.7, +44.1] | 1.000 | -15.5% | [-20.5, -10.4] | +100.0% | INSUFFICIENT |
| NF | Mexico Liga MX | over15 | Under | BET | 31 | -54.2% | [-104.4, -3.9] | 1.000 | -20.9% | [-26.0, -15.9] | +96.8% | INSUFFICIENT |
| NF | Mexico Liga MX | over15 | Over | OPEN | 7 | -40.7% | [-95.5, +14.1] | 1.000 | -6.7% | [-9.4, -4.0] | +100.0% | INSUFFICIENT |
| NF | Mexico Liga MX | over15 | Over | BET | 3 | +37.7% | [+35.7, +39.6] | 1.74e-299 | -3.9% | [-5.3, -2.5] | +100.0% | INSUFFICIENT |
| NF | Mexico Liga MX | over25 | Under | OPEN | 23 | -2.1% | [-60.7, +56.4] | 1.000 | -4.6% | [-8.7, -0.5] | +100.0% | INSUFFICIENT |
| NF | Mexico Liga MX | over25 | Under | BET | 40 | -16.9% | [-56.2, +22.4] | 1.000 | -10.3% | [-13.7, -6.9] | +72.5% | INSUFFICIENT |
| NF | Mexico Liga MX | over25 | Over | OPEN | 10 | -55.6% | [-113.7, +2.5] | 1.000 | -11.4% | [-15.2, -7.7] | +100.0% | INSUFFICIENT |
| NF | Mexico Liga MX | over25 | Over | BET | 7 | -15.1% | [-93.7, +63.5] | 1.000 | -7.6% | [-10.3, -5.0] | +71.4% | INSUFFICIENT |
| NF | Mexico Liga MX | over35 | Under | OPEN | 24 | -4.2% | [-34.6, +26.3] | 1.000 | -2.2% | [-4.6, +0.2] | +100.0% | INSUFFICIENT |
| NF | Mexico Liga MX | over35 | Under | BET | 153 | -2.1% | [-13.6, +9.3] | 1.000 | -3.4% | [-4.6, -2.2] | +20.3% | UNPROVEN |
| NF | Mexico Liga MX | over35 | Over | OPEN | 9 | -51.7% | [-150.8, +47.4] | 1.000 | -16.5% | [-22.3, -10.8] | +100.0% | INSUFFICIENT |
| NF | Mexico Liga MX | over35 | Over | BET | 11 | +15.1% | [-93.2, +123.4] | 1.000 | -1.6% | [-10.1, +7.0] | +27.3% | INSUFFICIENT |
| NF | Norway Eliteserien | btts | No | OPEN | 14 | -34.0% | [-105.4, +37.4] | 1.000 | -7.7% | [-10.9, -4.6] | +100.0% | INSUFFICIENT |
| NF | Norway Eliteserien | btts | No | BET | 27 | +5.2% | [-40.1, +50.6] | 1.000 | -11.6% | [-14.2, -8.9] | +77.8% | INSUFFICIENT |
| NF | Norway Eliteserien | btts | Yes | OPEN | 13 | -10.9% | [-65.4, +43.6] | 1.000 | -4.5% | [-7.6, -1.3] | +100.0% | INSUFFICIENT |
| NF | Norway Eliteserien | btts | Yes | BET | 11 | +22.1% | [-35.1, +79.4] | 1.000 | -1.1% | [-4.5, +2.4] | +100.0% | INSUFFICIENT |
| NF | Norway Eliteserien | over15 | Under | OPEN | 17 | +20.1% | [-86.6, +126.7] | 1.000 | -24.6% | [-29.1, -20.0] | +94.1% | INSUFFICIENT |
| NF | Norway Eliteserien | over15 | Under | BET | 38 | -33.6% | [-89.7, +22.5] | 1.000 | -29.1% | [-33.0, -25.2] | +84.2% | INSUFFICIENT |
| NF | Norway Eliteserien | over15 | Over | OPEN | 14 | -22.7% | [-65.0, +19.6] | 1.000 | -4.8% | [-6.2, -3.2] | +100.0% | INSUFFICIENT |
| NF | Norway Eliteserien | over15 | Over | BET | 12 | -10.3% | [-47.9, +27.2] | 1.000 | -2.4% | [-5.5, +0.6] | +100.0% | INSUFFICIENT |
| NF | Norway Eliteserien | over25 | Under | OPEN | 32 | -36.4% | [-75.7, +3.0] | 1.000 | -8.7% | [-11.3, -6.2] | +68.8% | INSUFFICIENT |
| NF | Norway Eliteserien | over25 | Under | BET | 44 | -14.9% | [-50.7, +20.8] | 1.000 | -12.4% | [-15.0, -9.7] | +77.3% | INSUFFICIENT |
| NF | Norway Eliteserien | over25 | Over | OPEN | 21 | +14.9% | [-40.7, +70.5] | 1.000 | -8.3% | [-11.1, -5.6] | +76.2% | INSUFFICIENT |
| NF | Norway Eliteserien | over25 | Over | BET | 21 | +30.8% | [-15.9, +77.4] | 1.000 | -3.9% | [-7.9, +0.1] | +100.0% | INSUFFICIENT |
| NF | Norway Eliteserien | over35 | Under | OPEN | 84 | -11.2% | [-29.2, +6.8] | 1.000 | -4.0% | [-5.2, -2.7] | +40.5% | INSUFFICIENT |
| NF | Norway Eliteserien | over35 | Under | BET | 48 | -19.1% | [-42.4, +4.3] | 1.000 | -6.4% | [-8.2, -4.6] | +77.1% | INSUFFICIENT |
| NF | Norway Eliteserien | over35 | Over | OPEN | 8 | +34.2% | [-143.0, +211.5] | 1.000 | -13.5% | [-22.0, -5.1] | +87.5% | INSUFFICIENT |
| NF | Norway Eliteserien | over35 | Over | BET | 6 | -28.7% | [-168.5, +111.1] | 1.000 | -6.6% | [-17.7, +4.6] | +100.0% | INSUFFICIENT |
| NF | Saudi Pro League | btts | No | OPEN | 3 | -23.7% | [-173.3, +126.0] | 1.000 | -1.9% | [-12.2, +8.3] | +100.0% | INSUFFICIENT |
| NF | Saudi Pro League | btts | No | BET | 3 | -16.7% | [-180.0, +146.7] | 1.000 | -11.5% | [-16.4, -6.7] | +100.0% | INSUFFICIENT |
| NF | Saudi Pro League | btts | Yes | OPEN | 19 | -21.7% | [-82.9, +39.4] | 1.000 | -4.9% | [-6.9, -2.8] | +100.0% | INSUFFICIENT |
| NF | Saudi Pro League | btts | Yes | BET | 9 | -58.7% | [-121.3, +3.9] | 1.000 | -5.5% | [-9.4, -1.6] | +100.0% | INSUFFICIENT |
| NF | Saudi Pro League | over15 | Under | OPEN | 36 | +50.8% | [-25.0, +126.7] | 1.000 | -15.4% | [-19.3, -11.6] | +58.3% | INSUFFICIENT |
| NF | Saudi Pro League | over15 | Under | BET | 9 | +111.8% | [-53.3, +276.8] | 1.000 | -22.0% | [-28.2, -15.8] | +100.0% | INSUFFICIENT |
| NF | Saudi Pro League | over15 | Over | OPEN | 8 | +0.4% | [-42.6, +43.3] | 1.000 | -2.0% | [-3.7, -0.3] | +87.5% | INSUFFICIENT |
| NF | Saudi Pro League | over15 | Over | BET | 3 | -9.0% | [-127.9, +109.9] | 1.000 | -2.6% | [-6.9, +1.7] | +100.0% | INSUFFICIENT |
| NF | Saudi Pro League | over25 | Under | OPEN | 46 | +2.4% | [-38.0, +42.9] | 1.000 | -8.2% | [-12.6, -3.8] | +52.2% | INSUFFICIENT |
| NF | Saudi Pro League | over25 | Under | BET | 10 | +7.6% | [-80.4, +95.6] | 1.000 | -12.3% | [-18.5, -6.1] | +100.0% | INSUFFICIENT |
| NF | Saudi Pro League | over25 | Over | OPEN | 34 | +30.9% | [+1.5, +60.4] | 1.000 | -1.0% | [-3.5, +1.5] | +73.5% | INSUFFICIENT |
| NF | Saudi Pro League | over25 | Over | BET | 10 | +28.8% | [-41.0, +98.6] | 1.000 | -0.8% | [-5.1, +3.5] | +100.0% | INSUFFICIENT |
| NF | Saudi Pro League | over35 | Under | OPEN | 45 | -11.7% | [-37.7, +14.4] | 1.000 | -4.0% | [-7.3, -0.8] | +62.2% | INSUFFICIENT |
| NF | Saudi Pro League | over35 | Under | BET | 11 | -10.7% | [-61.8, +40.4] | 1.000 | -6.8% | [-10.3, -3.2] | +100.0% | INSUFFICIENT |
| NF | Saudi Pro League | over35 | Over | OPEN | 22 | +25.5% | [-41.6, +92.6] | 1.000 | -5.8% | [-10.0, -1.7] | +95.5% | INSUFFICIENT |
| NF | Saudi Pro League | over35 | Over | BET | 9 | -26.7% | [-134.2, +80.9] | 1.000 | -5.7% | [-13.4, +2.0] | +100.0% | INSUFFICIENT |
| NF | Sweden Allsvenskan | btts | No | BET | 4 | -42.2% | [-155.4, +70.9] | 1.000 | -10.8% | [-13.3, -8.4] | +100.0% | INSUFFICIENT |
| NF | Sweden Allsvenskan | btts | Yes | BET | 10 | -23.6% | [-84.8, +37.6] | 1.000 | -7.0% | [-10.7, -3.2] | +100.0% | INSUFFICIENT |
| NF | Sweden Allsvenskan | over15 | Under | BET | 19 | -53.2% | [-146.2, +39.7] | 1.000 | -23.1% | [-30.4, -15.8] | +100.0% | INSUFFICIENT |
| NF | Sweden Allsvenskan | over15 | Over | BET | 25 | +2.1% | [-23.3, +27.5] | 1.000 | -3.8% | [-5.7, -1.8] | +100.0% | INSUFFICIENT |
| NF | Sweden Allsvenskan | over25 | Under | BET | 22 | -4.8% | [-53.8, +44.2] | 1.000 | -9.8% | [-14.8, -4.8] | +100.0% | INSUFFICIENT |
| NF | Sweden Allsvenskan | over25 | Over | BET | 23 | -29.3% | [-79.9, +21.2] | 1.000 | -5.0% | [-8.5, -1.4] | +100.0% | INSUFFICIENT |
| NF | Sweden Allsvenskan | over35 | Under | BET | 28 | -9.3% | [-37.8, +19.2] | 1.000 | -4.8% | [-6.4, -3.3] | +100.0% | INSUFFICIENT |
| NF | Sweden Allsvenskan | over35 | Over | BET | 10 | -23.8% | [-123.6, +76.0] | 1.000 | -10.8% | [-18.7, -3.0] | +100.0% | INSUFFICIENT |
| NF | USA MLS | btts | No | BET | 9 | +1.0% | [-81.7, +83.7] | 1.000 | -2.9% | [-10.9, +5.1] | +100.0% | INSUFFICIENT |
| NF | USA MLS | btts | Yes | BET | 7 | -72.7% | [-137.5, -7.9] | 1.000 | -3.9% | [-11.8, +4.0] | +100.0% | INSUFFICIENT |
| NF | USA MLS | over15 | Under | BET | 2 | +132.8% | [-323.4, +588.9] | 1.000 | -23.8% | [-31.8, -15.9] | +100.0% | INSUFFICIENT |
| NF | USA MLS | over15 | Over | BET | 12 | -12.4% | [-49.0, +24.2] | 1.000 | -2.3% | [-6.2, +1.6] | +100.0% | INSUFFICIENT |
| NF | USA MLS | over25 | Under | BET | 10 | -10.3% | [-82.2, +61.6] | 1.000 | -4.7% | [-11.5, +2.1] | +40.0% | INSUFFICIENT |
| NF | USA MLS | over25 | Over | BET | 15 | +3.2% | [-55.0, +61.4] | 1.000 | -5.7% | [-10.6, -0.8] | +100.0% | INSUFFICIENT |
| NF | USA MLS | over35 | Under | BET | 28 | -11.8% | [-38.6, +15.0] | 1.000 | -4.5% | [-6.5, -2.4] | +75.0% | INSUFFICIENT |
| NF | USA MLS | over35 | Over | BET | 6 | +86.7% | [-106.4, +279.7] | 1.000 | -13.8% | [-20.9, -6.6] | +100.0% | INSUFFICIENT |
| Std | Belgian First Division A | btts | No | OPEN | 75 | +2.4% | [-25.8, +30.6] | 1.000 | -7.7% | [-9.7, -5.6] | +57.3% | INSUFFICIENT |
| Std | Belgian First Division A | btts | No | BET | 53 | -8.9% | [-42.1, +24.1] | 1.000 | -9.7% | [-11.7, -7.6] | +79.2% | INSUFFICIENT |
| Std | Belgian First Division A | btts | Yes | OPEN | 5 | +23.1% | [-76.2, +122.4] | 1.000 | -0.2% | [-6.7, +6.3] | +80.0% | INSUFFICIENT |
| Std | Belgian First Division A | btts | Yes | BET | 4 | -51.2% | [-146.8, +44.3] | 1.000 | -5.1% | [-8.0, -2.1] | +75.0% | INSUFFICIENT |
| Std | Belgian First Division A | over15 | Under | OPEN | 58 | +22.8% | [-44.5, +90.1] | 1.000 | -21.6% | [-25.2, -18.1] | +82.8% | INSUFFICIENT |
| Std | Belgian First Division A | over15 | Under | BET | 60 | +43.0% | [-27.8, +113.8] | 1.000 | -21.1% | [-24.5, -17.6] | +75.0% | INSUFFICIENT |
| Std | Belgian First Division A | over15 | Over | OPEN | 3 | -11.2% | [-127.2, +104.9] | 1.000 | -1.9% | [-5.0, +1.2] | +100.0% | INSUFFICIENT |
| Std | Belgian First Division A | over15 | Over | BET | 7 | -41.5% | [-105.3, +22.3] | 1.000 | -2.9% | [-5.4, -0.5] | +42.9% | INSUFFICIENT |
| Std | Belgian First Division A | over25 | Under | OPEN | 51 | -15.3% | [-46.0, +15.3] | 1.000 | -7.8% | [-10.3, -5.4] | +64.7% | INSUFFICIENT |
| Std | Belgian First Division A | over25 | Under | BET | 64 | -17.1% | [-46.8, +12.5] | 1.000 | -7.2% | [-10.0, -4.5] | +50.0% | INSUFFICIENT |
| Std | Belgian First Division A | over25 | Under | FD_PRE | 93 | -8.4% | [-31.6, +14.8] | 1.000 | -5.7% | [-7.8, -3.5] | +47.3% | INSUFFICIENT |
| Std | Belgian First Division A | over25 | Over | OPEN | 29 | -40.3% | [-70.9, -9.7] | 1.000 | -2.8% | [-5.3, -0.3] | +72.4% | INSUFFICIENT |
| Std | Belgian First Division A | over25 | Over | BET | 35 | -28.8% | [-58.3, +0.8] | 1.000 | -4.3% | [-6.6, -2.1] | +60.0% | INSUFFICIENT |
| Std | Belgian First Division A | over25 | Over | FD_PRE | 2 | +5.0% | [-200.8, +210.8] | 1.000 | -18.0% | [-29.7, -6.3] | +100.0% | INSUFFICIENT |
| Std | Belgian First Division A | over35 | Under | OPEN | 87 | -7.5% | [-25.4, +10.4] | 1.000 | -4.5% | [-5.8, -3.2] | +55.2% | INSUFFICIENT |
| Std | Belgian First Division A | over35 | Under | BET | 123 | +3.0% | [-11.2, +17.2] | 1.000 | -4.3% | [-5.7, -3.0] | +38.2% | INSUFFICIENT |
| Std | Belgian First Division A | over35 | Over | OPEN | 6 | +0.2% | [-163.4, +163.8] | 1.000 | -7.5% | [-17.8, +2.9] | +100.0% | INSUFFICIENT |
| Std | Belgian First Division A | over35 | Over | BET | 6 | +59.9% | [-101.7, +221.5] | 1.000 | -5.6% | [-17.8, +6.6] | +83.3% | INSUFFICIENT |
| Std | Bundesliga 2 | btts | No | OPEN | 58 | -3.6% | [-40.6, +33.3] | 1.000 | -11.2% | [-13.2, -9.1] | +100.0% | INSUFFICIENT |
| Std | Bundesliga 2 | btts | No | BET | 46 | -7.7% | [-52.1, +36.7] | 1.000 | -11.7% | [-14.4, -9.1] | +100.0% | INSUFFICIENT |
| Std | Bundesliga 2 | over15 | Under | OPEN | 36 | -3.8% | [-69.7, +62.0] | 1.000 | -23.4% | [-29.3, -17.4] | +100.0% | INSUFFICIENT |
| Std | Bundesliga 2 | over15 | Under | BET | 44 | +9.8% | [-51.5, +71.0] | 1.000 | -24.1% | [-29.8, -18.4] | +100.0% | INSUFFICIENT |
| Std | Bundesliga 2 | over15 | Over | OPEN | 3 | +37.2% | [+36.0, +38.3] | 0 | -5.6% | [-7.8, -3.4] | +100.0% | INSUFFICIENT |
| Std | Bundesliga 2 | over15 | Over | BET | 1 | +36.0% | - | 1.000 | -7.8% | - | +100.0% | INSUFFICIENT |
| Std | Bundesliga 2 | over25 | Under | OPEN | 36 | -26.7% | [-63.7, +10.3] | 1.000 | -12.4% | [-15.1, -9.7] | +100.0% | INSUFFICIENT |
| Std | Bundesliga 2 | over25 | Under | BET | 42 | -10.2% | [-53.1, +32.7] | 1.000 | -11.1% | [-15.2, -7.0] | +100.0% | INSUFFICIENT |
| Std | Bundesliga 2 | over25 | Under | FD_PRE | 32 | -21.5% | [-62.3, +19.3] | 1.000 | -3.5% | [-6.5, -0.4] | +100.0% | INSUFFICIENT |
| Std | Bundesliga 2 | over25 | Over | OPEN | 1 | -100.0% | - | 1.000 | -16.3% | - | +100.0% | INSUFFICIENT |
| Std | Bundesliga 2 | over25 | Over | BET | 7 | +47.5% | [-2.6, +97.6] | 1.000 | -4.7% | [-12.3, +3.0] | +100.0% | INSUFFICIENT |
| Std | Bundesliga 2 | over25 | Over | FD_PRE | 1 | -100.0% | - | 1.000 | -9.5% | - | +100.0% | INSUFFICIENT |
| Std | Bundesliga 2 | over35 | Under | OPEN | 46 | -5.5% | [-28.9, +18.1] | 1.000 | -6.8% | [-8.7, -4.9] | +78.3% | INSUFFICIENT |
| Std | Bundesliga 2 | over35 | Under | BET | 44 | +1.7% | [-20.3, +23.8] | 1.000 | -6.2% | [-8.9, -3.4] | +97.7% | INSUFFICIENT |
| Std | Bundesliga 2 | over35 | Over | OPEN | 1 | -100.0% | - | 1.000 | -24.0% | - | +100.0% | INSUFFICIENT |
| Std | Bundesliga 2 | over35 | Over | BET | 2 | +40.2% | [-234.6, +315.1] | 1.000 | -17.2% | [-30.6, -3.8] | +100.0% | INSUFFICIENT |
| Std | Championship | btts | No | OPEN | 59 | +9.9% | [-19.1, +38.8] | 1.000 | -9.9% | [-11.4, -8.3] | +18.6% | INSUFFICIENT |
| Std | Championship | btts | No | BET | 51 | +9.1% | [-21.8, +40.0] | 1.000 | -10.7% | [-12.5, -8.8] | +19.6% | INSUFFICIENT |
| Std | Championship | btts | Yes | OPEN | 31 | +1.0% | [-35.7, +37.8] | 1.000 | -6.8% | [-8.7, -4.9] | +29.0% | INSUFFICIENT |
| Std | Championship | btts | Yes | BET | 31 | -17.1% | [-55.5, +21.3] | 1.000 | -3.9% | [-6.3, -1.5] | +29.0% | INSUFFICIENT |
| Std | Championship | over15 | Under | OPEN | 84 | +3.5% | [-35.4, +42.4] | 1.000 | -15.7% | [-17.8, -13.6] | +25.0% | INSUFFICIENT |
| Std | Championship | over15 | Under | BET | 60 | +5.7% | [-39.3, +50.8] | 1.000 | -15.9% | [-20.3, -11.5] | +31.7% | INSUFFICIENT |
| Std | Championship | over15 | Over | OPEN | 88 | +6.7% | [-6.3, +19.8] | 1.000 | -3.2% | [-3.9, -2.5] | +10.2% | INSUFFICIENT |
| Std | Championship | over15 | Over | BET | 100 | +3.4% | [-9.1, +16.0] | 1.000 | -2.8% | [-3.7, -1.9] | +13.0% | INSUFFICIENT |
| Std | Championship | over25 | Under | OPEN | 83 | -9.1% | [-32.3, +14.1] | 1.000 | -7.3% | [-8.8, -5.9] | +22.9% | INSUFFICIENT |
| Std | Championship | over25 | Under | BET | 81 | +6.2% | [-17.6, +30.0] | 1.000 | -7.5% | [-9.4, -5.7] | +21.0% | INSUFFICIENT |
| Std | Championship | over25 | Under | FD_PRE | 82 | -3.0% | [-25.4, +19.4] | 1.000 | -5.7% | [-7.3, -4.1] | +6.1% | INSUFFICIENT |
| Std | Championship | over25 | Over | OPEN | 155 | -8.1% | [-26.1, +9.9] | 1.000 | -6.3% | [-7.4, -5.3] | +13.6% | UNPROVEN |
| Std | Championship | over25 | Over | BET | 179 | -6.0% | [-24.2, +12.2] | 1.000 | -5.6% | [-6.8, -4.3] | +11.7% | UNPROVEN |
| Std | Championship | over25 | Over | FD_PRE | 186 | -6.6% | [-22.0, +8.8] | 1.000 | -7.7% | [-8.9, -6.5] | +8.6% | UNPROVEN |
| Std | Championship | over35 | Under | OPEN | 115 | +2.0% | [-10.7, +14.8] | 1.000 | -3.7% | [-4.5, -3.0] | +17.4% | INSUFFICIENT |
| Std | Championship | over35 | Under | BET | 117 | +5.6% | [-6.8, +18.0] | 1.000 | -4.3% | [-5.3, -3.2] | +15.4% | INSUFFICIENT |
| Std | Championship | over35 | Over | OPEN | 122 | -1.1% | [-32.7, +30.5] | 1.000 | -12.6% | [-14.3, -10.8] | +18.0% | INSUFFICIENT |
| Std | Championship | over35 | Over | BET | 118 | -3.0% | [-34.2, +28.3] | 1.000 | -12.3% | [-14.9, -9.6] | +21.2% | INSUFFICIENT |
| Std | Dutch Eredivisie | btts | No | OPEN | 46 | -1.7% | [-40.3, +36.9] | 1.000 | -8.2% | [-11.2, -5.2] | +100.0% | INSUFFICIENT |
| Std | Dutch Eredivisie | btts | No | BET | 39 | -11.6% | [-57.5, +34.3] | 1.000 | -7.2% | [-11.6, -2.8] | +100.0% | INSUFFICIENT |
| Std | Dutch Eredivisie | btts | Yes | OPEN | 2 | -4.5% | [-191.7, +182.7] | 1.000 | -9.1% | [-15.8, -2.4] | +100.0% | INSUFFICIENT |
| Std | Dutch Eredivisie | btts | Yes | BET | 3 | +27.7% | [-97.5, +152.8] | 1.000 | -4.3% | [-14.1, +5.5] | +100.0% | INSUFFICIENT |
| Std | Dutch Eredivisie | over15 | Under | OPEN | 58 | -43.5% | [-87.5, +0.5] | 1.000 | -19.7% | [-23.6, -15.7] | +100.0% | INSUFFICIENT |
| Std | Dutch Eredivisie | over15 | Under | BET | 48 | -53.0% | [-98.4, -7.5] | 1.000 | -20.5% | [-26.7, -14.4] | +100.0% | INSUFFICIENT |
| Std | Dutch Eredivisie | over25 | Under | OPEN | 53 | -30.7% | [-68.9, +7.5] | 1.000 | -6.3% | [-9.2, -3.3] | +100.0% | INSUFFICIENT |
| Std | Dutch Eredivisie | over25 | Under | BET | 45 | -46.1% | [-80.4, -11.8] | 1.000 | -6.5% | [-11.4, -1.6] | +100.0% | INSUFFICIENT |
| Std | Dutch Eredivisie | over25 | Under | FD_PRE | 119 | -12.7% | [-42.1, +16.7] | 1.000 | -7.4% | [-9.2, -5.6] | +27.7% | INSUFFICIENT |
| Std | Dutch Eredivisie | over25 | Over | OPEN | 3 | +10.3% | [-97.9, +118.5] | 1.000 | -1.1% | [-5.2, +3.1] | +100.0% | INSUFFICIENT |
| Std | Dutch Eredivisie | over25 | Over | BET | 2 | +62.0% | [+60.0, +64.0] | 0 | +2.8% | [+2.8, +2.8] | +100.0% | INSUFFICIENT |
| Std | Dutch Eredivisie | over25 | Over | FD_PRE | 4 | +62.5% | [-43.7, +168.7] | 1.000 | -10.5% | [-16.1, -4.8] | +25.0% | INSUFFICIENT |
| Std | Dutch Eredivisie | over35 | Under | OPEN | 56 | -17.1% | [-42.7, +8.6] | 1.000 | -3.0% | [-5.0, -1.0] | +100.0% | INSUFFICIENT |
| Std | Dutch Eredivisie | over35 | Under | BET | 46 | -26.3% | [-53.0, +0.4] | 1.000 | -3.1% | [-6.3, +0.1] | +100.0% | INSUFFICIENT |
| Std | Dutch Eredivisie | over35 | Over | OPEN | 2 | +37.0% | [-231.5, +305.5] | 1.000 | -7.8% | [-10.5, -5.0] | +100.0% | INSUFFICIENT |
| Std | Greek Super League | btts | No | OPEN | 1 | +103.5% | - | 1.000 | -10.5% | - | +100.0% | INSUFFICIENT |
| Std | Greek Super League | btts | No | BET | 2 | +95.5% | [+91.6, +99.4] | 0 | -7.9% | [-18.2, +2.4] | +100.0% | INSUFFICIENT |
| Std | Greek Super League | btts | Yes | OPEN | 107 | +3.8% | [-17.9, +25.5] | 1.000 | -7.5% | [-9.3, -5.7] | +33.6% | INSUFFICIENT |
| Std | Greek Super League | btts | Yes | BET | 147 | -13.0% | [-30.6, +4.6] | 1.000 | -8.5% | [-10.0, -7.0] | +20.4% | INSUFFICIENT |
| Std | Greek Super League | over15 | Under | OPEN | 19 | -100.0% | - | 1.000 | -18.8% | [-24.8, -12.7] | +68.4% | INSUFFICIENT |
| Std | Greek Super League | over15 | Under | BET | 40 | -13.9% | [-85.7, +58.0] | 1.000 | -12.2% | [-18.5, -5.8] | +27.5% | INSUFFICIENT |
| Std | Greek Super League | over15 | Over | OPEN | 41 | -1.4% | [-26.1, +23.3] | 1.000 | -4.9% | [-6.3, -3.6] | +34.2% | INSUFFICIENT |
| Std | Greek Super League | over15 | Over | BET | 64 | +9.2% | [-6.5, +24.8] | 1.000 | -4.4% | [-5.5, -3.2] | +26.6% | INSUFFICIENT |
| Std | Greek Super League | over25 | Under | OPEN | 50 | -21.7% | [-53.1, +9.7] | 1.000 | -6.7% | [-10.1, -3.3] | +28.0% | INSUFFICIENT |
| Std | Greek Super League | over25 | Under | BET | 66 | -10.4% | [-44.2, +23.5] | 1.000 | -3.7% | [-6.5, -1.0] | +19.7% | INSUFFICIENT |
| Std | Greek Super League | over25 | Under | FD_PRE | 15 | -59.3% | [-102.0, -16.7] | 1.000 | -5.6% | [-10.5, -0.6] | +86.7% | INSUFFICIENT |
| Std | Greek Super League | over25 | Over | OPEN | 86 | -4.5% | [-29.3, +20.2] | 1.000 | -8.1% | [-9.7, -6.5] | +30.2% | INSUFFICIENT |
| Std | Greek Super League | over25 | Over | BET | 135 | -3.1% | [-24.0, +17.8] | 1.000 | -6.8% | [-8.4, -5.2] | +23.7% | INSUFFICIENT |
| Std | Greek Super League | over25 | Over | FD_PRE | 18 | -33.1% | [-84.6, +18.5] | 1.000 | -16.5% | [-20.4, -12.5] | +100.0% | INSUFFICIENT |
| Std | Greek Super League | over35 | Under | OPEN | 56 | +11.5% | [-9.5, +32.4] | 1.000 | -3.4% | [-5.4, -1.4] | +17.9% | INSUFFICIENT |
| Std | Greek Super League | over35 | Under | BET | 74 | +6.8% | [-10.0, +23.6] | 1.000 | -2.0% | [-3.7, -0.2] | +18.9% | INSUFFICIENT |
| Std | Greek Super League | over35 | Over | OPEN | 36 | +38.1% | [-41.5, +117.7] | 1.000 | -20.7% | [-24.2, -17.2] | +55.6% | INSUFFICIENT |
| Std | Greek Super League | over35 | Over | BET | 66 | -7.1% | [-51.2, +37.0] | 1.000 | -18.0% | [-20.5, -15.6] | +28.8% | INSUFFICIENT |
| Std | La Liga 2 | btts | No | OPEN | 4 | -100.0% | - | 1.000 | -10.5% | [-17.0, -3.9] | +100.0% | INSUFFICIENT |
| Std | La Liga 2 | btts | No | BET | 2 | -100.0% | - | 1.000 | -8.0% | [-9.5, -6.5] | +100.0% | INSUFFICIENT |
| Std | La Liga 2 | btts | Yes | OPEN | 41 | -11.3% | [-62.7, +40.1] | 1.000 | -7.4% | [-10.1, -4.8] | +46.3% | INSUFFICIENT |
| Std | La Liga 2 | btts | Yes | BET | 43 | -17.7% | [-68.2, +32.8] | 1.000 | -6.9% | [-9.5, -4.3] | +100.0% | INSUFFICIENT |
| Std | La Liga 2 | over15 | Under | OPEN | 1 | -100.0% | - | 1.000 | +6.4% | - | +0.0% | INSUFFICIENT |
| Std | La Liga 2 | over15 | Under | BET | 2 | +78.0% | [-270.9, +426.9] | 1.000 | -15.4% | [-58.0, +27.2] | +100.0% | INSUFFICIENT |
| Std | La Liga 2 | over15 | Over | OPEN | 51 | +5.9% | [-14.9, +26.7] | 1.000 | -2.9% | [-4.5, -1.3] | +56.9% | INSUFFICIENT |
| Std | La Liga 2 | over15 | Over | BET | 50 | +1.3% | [-21.4, +24.1] | 1.000 | -2.1% | [-4.5, +0.3] | +100.0% | INSUFFICIENT |
| Std | La Liga 2 | over25 | Under | OPEN | 4 | +42.1% | [-50.8, +135.0] | 1.000 | -12.4% | [-27.4, +2.5] | +100.0% | INSUFFICIENT |
| Std | La Liga 2 | over25 | Under | BET | 2 | +85.8% | [+71.5, +100.0] | 1.3e-29 | -24.9% | [-27.4, -22.5] | +100.0% | INSUFFICIENT |
| Std | La Liga 2 | over25 | Over | OPEN | 53 | -14.0% | [-51.4, +23.4] | 1.000 | -6.3% | [-9.4, -3.1] | +49.1% | INSUFFICIENT |
| Std | La Liga 2 | over25 | Over | BET | 50 | -11.2% | [-61.7, +39.2] | 1.000 | -5.0% | [-10.0, -0.1] | +100.0% | INSUFFICIENT |
| Std | La Liga 2 | over25 | Over | FD_PRE | 59 | -5.5% | [-37.2, +26.1] | 1.000 | -12.9% | [-15.3, -10.6] | +100.0% | INSUFFICIENT |
| Std | La Liga 2 | over35 | Under | OPEN | 7 | -4.8% | [-78.6, +69.0] | 1.000 | -7.1% | [-13.7, -0.6] | +100.0% | INSUFFICIENT |
| Std | La Liga 2 | over35 | Under | BET | 2 | +33.0% | [+30.1, +35.9] | 1.33e-104 | -15.4% | [-16.1, -14.6] | +100.0% | INSUFFICIENT |
| Std | La Liga 2 | over35 | Over | OPEN | 49 | -39.2% | [-94.6, +16.2] | 1.000 | -14.0% | [-17.4, -10.6] | +46.9% | INSUFFICIENT |
| Std | La Liga 2 | over35 | Over | BET | 37 | -23.9% | [-91.8, +43.9] | 1.000 | -15.9% | [-19.8, -12.0] | +100.0% | INSUFFICIENT |
| Std | League One | btts | No | OPEN | 12 | -11.3% | [-73.5, +50.8] | 1.000 | -6.2% | [-9.2, -3.3] | +100.0% | INSUFFICIENT |
| Std | League One | btts | No | BET | 3 | +43.3% | [-97.2, +183.9] | 1.000 | +2.9% | [-13.7, +19.6] | +100.0% | INSUFFICIENT |
| Std | League One | btts | Yes | OPEN | 6 | -100.0% | - | 1.000 | -7.2% | [-10.3, -4.1] | +100.0% | INSUFFICIENT |
| Std | League One | btts | Yes | BET | 8 | -100.0% | - | 1.000 | -7.2% | [-10.5, -3.9] | +100.0% | INSUFFICIENT |
| Std | League One | over15 | Under | OPEN | 20 | -4.8% | [-79.0, +69.5] | 1.000 | -10.9% | [-21.0, -0.9] | +100.0% | INSUFFICIENT |
| Std | League One | over15 | Under | BET | 8 | -11.3% | [-125.2, +102.6] | 1.000 | -3.1% | [-14.6, +8.4] | +100.0% | INSUFFICIENT |
| Std | League One | over15 | Over | OPEN | 7 | -3.0% | [-52.1, +46.1] | 1.000 | -2.4% | [-3.2, -1.6] | +100.0% | INSUFFICIENT |
| Std | League One | over15 | Over | BET | 9 | +6.3% | [-33.1, +45.8] | 1.000 | -1.6% | [-2.7, -0.6] | +100.0% | INSUFFICIENT |
| Std | League One | over25 | Under | OPEN | 15 | -34.5% | [-83.3, +14.3] | 1.000 | -2.2% | [-11.3, +6.8] | +100.0% | INSUFFICIENT |
| Std | League One | over25 | Under | BET | 4 | -6.2% | [-113.7, +101.2] | 1.000 | +4.2% | [-13.5, +21.9] | +100.0% | INSUFFICIENT |
| Std | League One | over25 | Under | FD_PRE | 172 | +9.7% | [-6.7, +26.1] | 1.000 | -2.8% | [-3.9, -1.8] | +15.1% | UNPROVEN |
| Std | League One | over25 | Over | OPEN | 36 | -13.0% | [-59.4, +33.4] | 1.000 | -6.7% | [-8.0, -5.3] | +100.0% | INSUFFICIENT |
| Std | League One | over25 | Over | BET | 29 | -31.1% | [-68.0, +5.9] | 1.000 | -4.9% | [-7.8, -2.0] | +100.0% | INSUFFICIENT |
| Std | League One | over25 | Over | FD_PRE | 103 | -7.4% | [-27.5, +12.7] | 1.000 | -7.9% | [-9.7, -6.1] | +6.8% | INSUFFICIENT |
| Std | League One | over35 | Under | OPEN | 20 | -16.9% | [-53.7, +20.0] | 1.000 | -3.0% | [-6.2, +0.2] | +100.0% | INSUFFICIENT |
| Std | League One | over35 | Under | BET | 5 | +39.8% | [+37.1, +42.5] | 6.42e-177 | +0.2% | [-8.1, +8.5] | +100.0% | INSUFFICIENT |
| Std | League One | over35 | Over | OPEN | 23 | -40.4% | [-94.7, +14.0] | 1.000 | -15.9% | [-18.1, -13.8] | +100.0% | INSUFFICIENT |
| Std | League One | over35 | Over | BET | 23 | -55.1% | [-103.6, -6.7] | 1.000 | -14.7% | [-19.5, -9.9] | +100.0% | INSUFFICIENT |
| Std | League Two | btts | No | OPEN | 8 | +4.4% | [-73.1, +82.0] | 1.000 | -9.0% | [-12.9, -5.2] | +100.0% | INSUFFICIENT |
| Std | League Two | btts | No | BET | 8 | -24.2% | [-96.8, +48.4] | 1.000 | -9.6% | [-14.3, -4.8] | +100.0% | INSUFFICIENT |
| Std | League Two | btts | Yes | OPEN | 10 | -1.0% | [-82.2, +80.2] | 1.000 | -7.3% | [-8.9, -5.7] | +100.0% | INSUFFICIENT |
| Std | League Two | btts | Yes | BET | 13 | -8.8% | [-64.6, +46.9] | 1.000 | -5.8% | [-8.5, -3.2] | +100.0% | INSUFFICIENT |
| Std | League Two | over15 | Under | OPEN | 21 | +4.7% | [-68.2, +77.7] | 1.000 | -14.4% | [-19.0, -9.8] | +100.0% | INSUFFICIENT |
| Std | League Two | over15 | Under | BET | 14 | -49.4% | [-125.3, +26.5] | 1.000 | -18.0% | [-24.5, -11.6] | +100.0% | INSUFFICIENT |
| Std | League Two | over15 | Over | OPEN | 13 | -3.9% | [-57.8, +50.0] | 1.000 | -3.4% | [-4.4, -2.4] | +100.0% | INSUFFICIENT |
| Std | League Two | over15 | Over | BET | 17 | -2.0% | [-35.8, +31.7] | 1.000 | -2.4% | [-4.0, -0.8] | +100.0% | INSUFFICIENT |
| Std | League Two | over25 | Under | OPEN | 20 | -4.2% | [-50.0, +41.6] | 1.000 | -9.0% | [-13.5, -4.5] | +100.0% | INSUFFICIENT |
| Std | League Two | over25 | Under | BET | 20 | +5.2% | [-37.8, +48.3] | 1.000 | -9.5% | [-14.7, -4.4] | +85.0% | INSUFFICIENT |
| Std | League Two | over25 | Under | FD_PRE | 12 | -19.0% | [-75.7, +37.7] | 1.000 | -1.6% | [-4.2, +0.9] | +58.3% | INSUFFICIENT |
| Std | League Two | over25 | Over | OPEN | 30 | +14.5% | [-29.4, +58.5] | 1.000 | -5.9% | [-7.4, -4.5] | +100.0% | INSUFFICIENT |
| Std | League Two | over25 | Over | BET | 42 | +12.1% | [-24.1, +48.3] | 1.000 | -4.8% | [-6.1, -3.5] | +73.8% | INSUFFICIENT |
| Std | League Two | over25 | Over | FD_PRE | 52 | -4.6% | [-36.1, +26.9] | 1.000 | -11.0% | [-12.7, -9.3] | +55.8% | INSUFFICIENT |
| Std | League Two | over35 | Under | OPEN | 20 | +3.8% | [-31.8, +39.3] | 1.000 | -4.4% | [-7.2, -1.5] | +100.0% | INSUFFICIENT |
| Std | League Two | over35 | Under | BET | 21 | +1.2% | [-38.6, +41.1] | 1.000 | -5.1% | [-7.8, -2.3] | +57.1% | INSUFFICIENT |
| Std | League Two | over35 | Over | OPEN | 20 | -46.2% | [-120.6, +28.1] | 1.000 | -13.4% | [-15.3, -11.5] | +100.0% | INSUFFICIENT |
| Std | League Two | over35 | Over | BET | 17 | -13.8% | [-90.2, +62.6] | 1.000 | -12.7% | [-16.6, -8.8] | +100.0% | INSUFFICIENT |
| Std | Ligue 2 | btts | No | OPEN | 15 | -21.0% | [-92.3, +50.3] | 1.000 | -9.1% | [-12.3, -5.9] | +100.0% | INSUFFICIENT |
| Std | Ligue 2 | btts | No | BET | 18 | -30.8% | [-106.8, +45.2] | 1.000 | -11.5% | [-16.1, -6.9] | +100.0% | INSUFFICIENT |
| Std | Ligue 2 | btts | Yes | OPEN | 3 | -100.0% | - | 1.000 | -9.3% | [-20.2, +1.6] | +100.0% | INSUFFICIENT |
| Std | Ligue 2 | btts | Yes | BET | 2 | -100.0% | - | 1.000 | -5.8% | [-20.3, +8.7] | +100.0% | INSUFFICIENT |
| Std | Ligue 2 | over15 | Under | OPEN | 11 | -32.5% | [-121.4, +56.5] | 1.000 | -24.0% | [-31.7, -16.3] | +100.0% | INSUFFICIENT |
| Std | Ligue 2 | over15 | Under | BET | 18 | +5.8% | [-75.4, +87.0] | 1.000 | -26.8% | [-32.6, -21.0] | +100.0% | INSUFFICIENT |
| Std | Ligue 2 | over15 | Over | OPEN | 10 | +15.5% | [-28.4, +59.4] | 1.000 | -2.7% | [-4.2, -1.1] | +90.0% | INSUFFICIENT |
| Std | Ligue 2 | over15 | Over | BET | 11 | +4.1% | [-39.9, +48.0] | 1.000 | -3.0% | [-6.1, +0.1] | +63.6% | INSUFFICIENT |
| Std | Ligue 2 | over25 | Under | OPEN | 22 | -13.7% | [-74.7, +47.3] | 1.000 | -11.5% | [-14.9, -8.1] | +100.0% | INSUFFICIENT |
| Std | Ligue 2 | over25 | Under | BET | 25 | -4.1% | [-59.0, +50.8] | 1.000 | -13.2% | [-19.0, -7.4] | +100.0% | INSUFFICIENT |
| Std | Ligue 2 | over25 | Under | FD_PRE | 2 | -10.0% | [-186.4, +166.4] | 1.000 | -10.5% | [-24.7, +3.7] | +100.0% | INSUFFICIENT |
| Std | Ligue 2 | over25 | Over | OPEN | 8 | -8.4% | [-118.5, +101.7] | 1.000 | -4.5% | [-8.0, -1.1] | +75.0% | INSUFFICIENT |
| Std | Ligue 2 | over25 | Over | BET | 9 | -75.3% | [-126.0, -24.6] | 1.000 | -7.7% | [-14.0, -1.5] | +55.6% | INSUFFICIENT |
| Std | Ligue 2 | over25 | Over | FD_PRE | 52 | -20.9% | [-52.2, +10.5] | 1.000 | -10.1% | [-11.9, -8.4] | +75.0% | INSUFFICIENT |
| Std | Ligue 2 | over35 | Under | OPEN | 28 | -10.8% | [-36.0, +14.5] | 1.000 | -5.9% | [-8.1, -3.8] | +82.1% | INSUFFICIENT |
| Std | Ligue 2 | over35 | Under | BET | 31 | -4.8% | [-28.5, +18.8] | 1.000 | -7.2% | [-10.6, -3.8] | +90.3% | INSUFFICIENT |
| Std | Ligue 2 | over35 | Over | OPEN | 3 | -100.0% | - | 1.000 | -18.6% | [-22.0, -15.1] | +100.0% | INSUFFICIENT |
| Std | Ligue 2 | over35 | Over | BET | 16 | -31.6% | [-103.7, +40.5] | 1.000 | -17.7% | [-22.7, -12.8] | +100.0% | INSUFFICIENT |
| Std | National League | over25 | Under | FD_PRE | 79 | -6.8% | [-30.8, +17.2] | 1.000 | -7.6% | [-9.3, -5.9] | +48.1% | INSUFFICIENT |
| Std | National League | over25 | Over | FD_PRE | 5 | -57.0% | [-148.3, +34.3] | 1.000 | -9.7% | [-14.0, -5.5] | +60.0% | INSUFFICIENT |
| Std | Portuguese Primeira Liga | btts | No | BET | 6 | -1.2% | [-87.9, +85.4] | 1.000 | - | - | +100.0% | INSUFFICIENT |
| Std | Portuguese Primeira Liga | btts | Yes | BET | 27 | +30.8% | [-12.2, +73.7] | 1.000 | - | - | +92.6% | INSUFFICIENT |
| Std | Portuguese Primeira Liga | over15 | Under | BET | 20 | -36.1% | [-109.0, +36.8] | 1.000 | - | - | +100.0% | INSUFFICIENT |
| Std | Portuguese Primeira Liga | over15 | Over | BET | 14 | +10.8% | [-22.2, +43.8] | 1.000 | - | - | +100.0% | INSUFFICIENT |
| Std | Portuguese Primeira Liga | over25 | Under | BET | 19 | -37.7% | [-84.4, +9.1] | 1.000 | - | - | +100.0% | INSUFFICIENT |
| Std | Portuguese Primeira Liga | over25 | Under | FD_PRE | 75 | -6.1% | [-34.7, +22.6] | 1.000 | -9.5% | [-11.6, -7.3] | +21.3% | INSUFFICIENT |
| Std | Portuguese Primeira Liga | over25 | Over | BET | 15 | +38.1% | [-17.9, +94.2] | 1.000 | - | - | +100.0% | INSUFFICIENT |
| Std | Portuguese Primeira Liga | over25 | Over | FD_PRE | 33 | -11.4% | [-47.5, +24.8] | 1.000 | -7.0% | [-9.5, -4.5] | +36.4% | INSUFFICIENT |
| Std | Portuguese Primeira Liga | over35 | Under | BET | 28 | -23.0% | [-52.1, +6.2] | 1.000 | - | - | +100.0% | INSUFFICIENT |
| Std | Portuguese Primeira Liga | over35 | Over | BET | 14 | +17.4% | [-106.6, +141.5] | 1.000 | - | - | +100.0% | INSUFFICIENT |
| Std | Scottish Championship | over25 | Under | FD_PRE | 78 | -7.3% | [-30.4, +15.8] | 1.000 | -5.9% | [-7.5, -4.4] | +44.9% | INSUFFICIENT |
| Std | Scottish Championship | over25 | Over | FD_PRE | 21 | -23.1% | [-67.3, +21.0] | 1.000 | -5.4% | [-8.5, -2.4] | +14.3% | INSUFFICIENT |
| Std | Scottish League One | over25 | Under | FD_PRE | 47 | -12.2% | [-47.9, +23.5] | 1.000 | -6.0% | [-7.5, -4.5] | +100.0% | INSUFFICIENT |
| Std | Scottish League Two | over25 | Under | FD_PRE | 235 | +7.4% | [-6.2, +21.1] | 1.000 | -4.2% | [-5.4, -3.0] | +20.0% | UNPROVEN |
| Std | Scottish League Two | over25 | Over | FD_PRE | 193 | -1.2% | [-16.0, +13.5] | 1.000 | -6.2% | [-7.3, -5.2] | +0.0% | UNPROVEN |
| Std | Scottish Premiership | btts | No | OPEN | 14 | +2.8% | [-53.3, +58.9] | 1.000 | -8.3% | [-11.5, -5.0] | +100.0% | INSUFFICIENT |
| Std | Scottish Premiership | btts | No | BET | 16 | +7.8% | [-65.7, +81.4] | 1.000 | -6.2% | [-10.6, -1.8] | +100.0% | INSUFFICIENT |
| Std | Scottish Premiership | btts | Yes | OPEN | 6 | -27.6% | [-137.3, +82.1] | 1.000 | -7.0% | [-11.4, -2.5] | +100.0% | INSUFFICIENT |
| Std | Scottish Premiership | btts | Yes | BET | 16 | -38.2% | [-98.8, +22.4] | 1.000 | -5.0% | [-8.2, -1.7] | +100.0% | INSUFFICIENT |
| Std | Scottish Premiership | over15 | Under | OPEN | 18 | +20.9% | [-94.2, +136.0] | 1.000 | -24.8% | [-30.6, -19.0] | +100.0% | INSUFFICIENT |
| Std | Scottish Premiership | over15 | Under | BET | 25 | +37.9% | [-54.8, +130.6] | 1.000 | -19.2% | [-24.9, -13.5] | +100.0% | INSUFFICIENT |
| Std | Scottish Premiership | over15 | Over | OPEN | 9 | +8.4% | [-31.8, +48.6] | 1.000 | -3.7% | [-5.3, -2.1] | +100.0% | INSUFFICIENT |
| Std | Scottish Premiership | over15 | Over | BET | 8 | -13.2% | [-67.4, +41.0] | 1.000 | -6.5% | [-8.1, -4.9] | +100.0% | INSUFFICIENT |
| Std | Scottish Premiership | over25 | Under | OPEN | 19 | -9.4% | [-72.5, +53.7] | 1.000 | -9.2% | [-13.4, -5.1] | +100.0% | INSUFFICIENT |
| Std | Scottish Premiership | over25 | Under | BET | 21 | -18.7% | [-70.9, +33.5] | 1.000 | -6.1% | [-10.1, -2.1] | +95.2% | INSUFFICIENT |
| Std | Scottish Premiership | over25 | Under | FD_PRE | 32 | -18.5% | [-79.1, +42.1] | 1.000 | -9.2% | [-12.0, -6.3] | +75.0% | INSUFFICIENT |
| Std | Scottish Premiership | over25 | Over | OPEN | 10 | -19.5% | [-86.9, +47.9] | 1.000 | -6.3% | [-9.0, -3.7] | +100.0% | INSUFFICIENT |
| Std | Scottish Premiership | over25 | Over | BET | 20 | -40.1% | [-81.7, +1.5] | 1.000 | -9.7% | [-12.4, -7.0] | +100.0% | INSUFFICIENT |
| Std | Scottish Premiership | over25 | Over | FD_PRE | 13 | +38.1% | [-23.8, +100.1] | 1.000 | -13.5% | [-17.9, -9.0] | +53.8% | INSUFFICIENT |
| Std | Scottish Premiership | over35 | Under | OPEN | 20 | -14.9% | [-58.3, +28.5] | 1.000 | -5.1% | [-7.9, -2.3] | +100.0% | INSUFFICIENT |
| Std | Scottish Premiership | over35 | Under | BET | 26 | +2.1% | [-34.4, +38.5] | 1.000 | -2.7% | [-5.4, +0.0] | +73.1% | INSUFFICIENT |
| Std | Scottish Premiership | over35 | Over | OPEN | 12 | -42.9% | [-118.3, +32.5] | 1.000 | -11.7% | [-16.3, -7.1] | +100.0% | INSUFFICIENT |
| Std | Scottish Premiership | over35 | Over | BET | 18 | -1.2% | [-77.2, +74.7] | 1.000 | -17.8% | [-21.1, -14.5] | +100.0% | INSUFFICIENT |
| Std | Serie B | btts | No | OPEN | 12 | -33.6% | [-89.1, +21.9] | 1.000 | -7.4% | [-9.7, -5.1] | +100.0% | INSUFFICIENT |
| Std | Serie B | btts | No | BET | 20 | +6.7% | [-43.0, +56.4] | 1.000 | -6.1% | [-9.0, -3.3] | +25.0% | INSUFFICIENT |
| Std | Serie B | btts | Yes | OPEN | 115 | +10.0% | [-8.4, +28.4] | 1.000 | -8.1% | [-9.5, -6.8] | +13.0% | INSUFFICIENT |
| Std | Serie B | btts | Yes | BET | 273 | +7.1% | [-4.8, +19.0] | 1.000 | -7.1% | [-7.9, -6.3] | +14.3% | UNPROVEN |
| Std | Serie B | over15 | Under | OPEN | 13 | +30.5% | [-63.6, +124.7] | 1.000 | -10.4% | [-18.7, -2.1] | +84.6% | INSUFFICIENT |
| Std | Serie B | over15 | Under | BET | 26 | +10.3% | [-57.1, +77.7] | 1.000 | -11.2% | [-15.6, -6.8] | +11.5% | INSUFFICIENT |
| Std | Serie B | over15 | Over | OPEN | 258 | +2.5% | [-5.5, +10.4] | 1.000 | -3.8% | [-4.4, -3.2] | +7.8% | UNPROVEN |
| Std | Serie B | over15 | Over | BET | 435 | +3.2% | [-2.7, +9.2] | 1.000 | -3.2% | [-3.5, -2.8] | +9.4% | UNPROVEN |
| Std | Serie B | over25 | Under | OPEN | 21 | -18.6% | [-59.9, +22.8] | 1.000 | -9.4% | [-13.0, -5.8] | +66.7% | INSUFFICIENT |
| Std | Serie B | over25 | Under | BET | 35 | -32.0% | [-63.8, -0.2] | 1.000 | -5.9% | [-8.6, -3.2] | +14.3% | INSUFFICIENT |
| Std | Serie B | over25 | Under | FD_PRE | 12 | -14.3% | [-74.7, +46.0] | 1.000 | -6.8% | [-13.1, -0.5] | +50.0% | INSUFFICIENT |
| Std | Serie B | over25 | Over | OPEN | 160 | +2.4% | [-16.0, +20.8] | 1.000 | -8.6% | [-9.7, -7.4] | +12.5% | UNPROVEN |
| Std | Serie B | over25 | Over | BET | 504 | +3.3% | [-6.7, +13.2] | 1.000 | -6.9% | [-7.6, -6.2] | +9.9% | UNPROVEN |
| Std | Serie B | over25 | Over | FD_PRE | 118 | +11.0% | [-10.3, +32.4] | 1.000 | -7.0% | [-8.8, -5.2] | +20.3% | INSUFFICIENT |
| Std | Serie B | over35 | Under | OPEN | 27 | -17.1% | [-43.7, +9.5] | 1.000 | -5.9% | [-9.2, -2.6] | +59.3% | INSUFFICIENT |
| Std | Serie B | over35 | Under | BET | 42 | +4.8% | [-14.4, +23.9] | 1.000 | -4.5% | [-5.8, -3.1] | +16.7% | INSUFFICIENT |
| Std | Serie B | over35 | Over | OPEN | 144 | +0.1% | [-33.9, +34.2] | 1.000 | -17.2% | [-19.2, -15.2] | +15.3% | INSUFFICIENT |
| Std | Serie B | over35 | Over | BET | 404 | -3.8% | [-23.9, +16.4] | 1.000 | -15.5% | [-16.7, -14.3] | +10.2% | UNPROVEN |
| Std | Turkish Super Lig | btts | No | BET | 100 | +11.5% | [-11.6, +34.5] | 1.000 | -7.4% | [-10.2, -4.5] | +57.0% | INSUFFICIENT |
| Std | Turkish Super Lig | btts | Yes | BET | 8 | -26.9% | [-96.8, +43.1] | 1.000 | -4.5% | [-12.7, +3.6] | +0.0% | INSUFFICIENT |
| Std | Turkish Super Lig | over15 | Under | BET | 93 | -22.9% | [-59.3, +13.5] | 1.000 | -17.4% | [-22.1, -12.6] | +50.5% | INSUFFICIENT |
| Std | Turkish Super Lig | over15 | Over | BET | 40 | -16.1% | [-39.5, +7.3] | 1.000 | -2.9% | [-4.0, -1.8] | +2.5% | INSUFFICIENT |
| Std | Turkish Super Lig | over25 | Under | BET | 154 | -3.7% | [-23.3, +15.8] | 1.000 | -8.7% | [-10.5, -6.8] | +24.0% | UNPROVEN |
| Std | Turkish Super Lig | over25 | Under | FD_PRE | 55 | -46.2% | [-70.9, -21.5] | 1.000 | -9.6% | [-13.2, -6.1] | +100.0% | INSUFFICIENT |
| Std | Turkish Super Lig | over25 | Over | BET | 72 | -13.1% | [-36.4, +10.3] | 1.000 | -5.5% | [-6.9, -4.1] | +9.7% | INSUFFICIENT |
| Std | Turkish Super Lig | over35 | Under | BET | 174 | -0.1% | [-12.3, +12.2] | 1.000 | -4.0% | [-5.4, -2.7] | +27.6% | UNPROVEN |
| Std | Turkish Super Lig | over35 | Over | BET | 50 | -32.4% | [-77.3, +12.5] | 1.000 | -13.1% | [-15.6, -10.7] | +8.0% | INSUFFICIENT |
