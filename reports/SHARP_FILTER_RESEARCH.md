# Sharp-agreement filter: research report (protocol sharp-filter/1)

**Owner approval:** 2026-11-15 ("yes"). **Status:** research only. Nothing in production changes; the owner decides.

**Protocol:** [`reports/sharp_filter/protocol.json`](sharp_filter/protocol.json), sha256 `aeb0ab955b42bc8f1e282f7c969dd241e2c264c41b4936eae721026791badee6`. It was registered in commit `ea704b933` and pushed **before** any effect was computed: `evaluate` refuses to run unless the protocol file is committed and clean.

**Results:** [`reports/sharp_filter/results.json`](sharp_filter/results.json) holds every cell, every contrast and every timing statistic. **Code:** `scripts/sharp_filter_study.py`, tables from `scripts/sharp_filter_tables.py`, tests in `tests/test_sharp_filter_study.py` (6 pass).

**Licence:** this report and results.json hold counts, ROI, CI, p-values, CLV percentages and lead-time statistics only. No OddsAPI price, event id or per-row frame is in the repo or in any log; per-row frames stay in a private work directory outside the repo.

**Label:** DEVELOPMENT evidence. Every row is dated before 2026-09-01. The approved bars of TP-2026-11-15-bade2c0b were also chosen on rows before 2026-09-01, so each cell's tip set is in-sample for its own bar.

## Answer in one paragraph

- **No configuration is PROVEN:** 0 of 1,463 tested cells.
- **Following the sharp move without the model loses about the margin:**
  - O/U 2.5: -4.6% over 1,813 bets.
  - 1X2: -2.7% over 6,903 bets.
  - CLV is about 0 (-0.3%). This is the T-72 h to T-6 h Pinnacle signal, betting Pinnacle's own price at T-6 h. Pinnacle's later moves do not continue in a way that pays.
- **The model with a multi-book consensus AGREE looks better, but the evidence is weak:**
  - At T-6 h it returns +10.8% over 103 bets, against -2.0% for NONE and -5.2% for DISAGREE. These are descriptive sums across cells.
  - It shrinks to +7.2% when the bet is placed at the next snapshot (median T-2 h).
  - It turns negative (-10.6% over 99) when the signal is read at T-24 h.
  - None of the 287 pre-registered contrasts survives Holm. The smallest Holm p is 0.79.
- **Coverage limits the test more than anything else:**
  - Pinnacle quotes the exact 2.5 line in only about 27% of the bet snapshots, because its single featured line is often 2.25 or 2.75.
  - No model cell reaches 150 bets in any one class. The largest has 151 (Belgium Standard Under, NONE).
- **1X2 cannot be tested on the model side:** the clean Phase-2 retrain produced no 1X2 predictions. 1X2 appears only in the market-only baseline.

## 1. The sharp definition (fixed in advance)

**Why the existing v9 definition (src/sharp_tracker.py) is not reused:**
- It flags a 5% / 10% drift in the *median decimal price across books* since the first capture. Those are global constants, which the owner's no-global rule forbids.
- It measures drift in decimal odds, so the same probability move reads differently at 1.5 and at 4.0.
- Its "consensus" is a time-weighted share of snapshots, not of books.
- It has no per-league learning.

The ledger (output/sharp_ledger.csv, live since 2026-06-14) is therefore context only and is not used here.

**New definition:**
- **Snapshots.**
  - B4 featured snapshots are timed off the league-day's *first* kickoff, and one event can sit in two league-day units. Labels are therefore not a match's lead time.
  - Each fixture's snapshots are chosen by their actual lead (the provider's latest kickoff minus snapshot_ts). The open snapshot is the one closest to 72 h within [60, 84] h. The signal snapshot is the one closest to 6 h within [4, 10] h, or closest to 24 h within [18, 30] h.
  - Measured leads: open median 72.1 h; signal median 6.1 h (IQR 5.6-6.1); strict-execution bet median 2.1 h.
  - Only MOVEMENT-role snapshots are inputs. CLOSE (B4's last snapshot, and B1) is read only for CLV.
- **PIN.** The change in Pinnacle's fair probability (power de-vig for O/U, proportional de-vig for 1X2) from open to signal.
  - For O/U the change is measured on the line Pinnacle quotes at both snapshots.
  - If Pinnacle's main line itself moved, the direction of that line move counts as a move beyond the threshold (up means toward Over).
- **CONS.** The median of the same change over every other book that quotes the exact 2.5 pair (or the full 1X2 triple) at both snapshots, with at least 3 such books.
- **BOTH.** PIN and CONS must give the same class; otherwise the bet is MIXED (reported, not tested).
- **Threshold, per league × market (× side for 1X2) × signal × window.** It is the 75th percentile of |change| over the same league's fixtures dated **strictly earlier**. It uses prices only, never outcomes. A fixture with fewer than 100 earlier measured fixtures is in WARMUP and gets no class.
- **Classes for a bet on side s.**
  - AGREE: the move toward s exceeds the threshold.
  - DISAGREE: the move against s exceeds it.
  - NONE: the move is within it.
  - NO_DATA: the move cannot be measured (counted, not tested).
- **Bet price.** Pinnacle's own exact-line pair (or triple) in the bet snapshot, which is a MOVEMENT snapshot at or after the signal. It is one book's real pre-kickoff price, never the close and never a best-of-books price. Without a pair there is no bet, and nothing stands in.
- **Windows** (Holm counts all three):
  - W72_6: T-72 h to T-6 h, bet at the signal snapshot. This is the primary window.
  - W72_6_next: same signal, bet at the first snapshot strictly after it.
  - W72_24: T-72 h to T-24 h.
- **Model tips.** A tip is a fixture that clears the approved cell's own bar of TP-2026-11-15-bade2c0b, with that cell's fitted source, measure and odds range (`src.bars_opt.serve.score`, unchanged), at the bet price.
  - Only MODEL-mode cells are tested.
  - The 4 LEAGUE_WIDE cells bet every match and are listed, not tested: NF O2.5 in Argentina, Austria, Ireland and K-League.
- **Statistics.** These follow `src.bars_opt.study.score`: a block bootstrap by match date, and the max-of-three-SE CI. 13 cells whose p sat at the 2,000-draw floor were re-drawn with 585,200 resamples.
  - **F1** (the decision family, 1,463 cells): Holm over every cell with at least one bet. That covers the model arm (AGREE, DISAGREE, NONE and ALL) and the market-only FOLLOW arm, across every window and signal.
  - **F2** (287 contrasts): AGREE-NONE, AGREE-DISAGREE and AGREE minus market FOLLOW, with Holm applied separately.
- **PROVEN** means at least 150 bets, a Holm p below 0.025 and a CI above 0.

**Pre-registration changes, based on coverage counts only (no outcome read):** consensus_min_books was drafted at 5, which left 53% of fixtures without a consensus, and was set to 3. The coverage counts are recorded in the protocol.

**Data:**
- 44,719 clean fixtures (both tracks, over15/25/35, dated before 2026-09-01). Of these, 27,666 matched a provider event (the adapter's own rule).
- The adapter dropped 140,622 post-kickoff quotes and 12,184 quotes at or below 1.01.
- 5,725 fixtures are in leagues without an OddsAPI key (National League, the Scottish lower divisions and Romania).
- W72_6 is readable on 21,574 fixtures.
- 1X2 results come from an exact join to fd_history: 38,073 of 44,719 fixtures are settleable. 2 rows contradicted the universe's own over25 outcome and were excluded.

## 2. Findings per league × market: model tips by sharp class (primary window W72_6)

How to read the tables:
- Each row is one approved MODEL cell. "ALL" is every tip of the cell at the T-6 h Pinnacle price, whatever the signal.
- ROI is per 1-unit bet. The CI is the 95% interval in percentage points. The Holm p is the F1-adjusted p.
- win-implied is the win rate minus the mean of 1/odds.
- A Holm p of 0.0025 on 3-8 bets is a bootstrap artefact: a handful of all-winning bets has no resample at or below 0. The 150-bet rule exists for exactly this case.
- The other two windows are in results.json.

### Signal PIN (Pinnacle)
| Track | Market-side | League | ALL n / ROI | AGREE n / ROI / CI / Holm p | DISAGREE n / ROI | NONE n / ROI | NO_DATA+MIXED n | AGREE win-implied | AGREE verdict |
|---|---|---|---|---|---|---|---|---|---|
| new_format | over15_over | Argentina Primera Division | 56 / -19.5% | 0 / - / - / - | 37 / -18.7% | 16 / -41.2% | 3 | - | - |
| new_format | over25_over | Denmark Superliga | 106 / -3.7% | 6 / -67.3% / [-131.4, -3.3] / 1 | 29 / -5.5% | 26 / +32.6% | 45 | -35.5% | UNPROVEN |
| new_format | over25_over | Finland Veikkausliiga | 134 / +10.1% | 7 / +11.9% / [-65.8, +89.5] / 1 | 24 / +4.5% | 28 / -17.8% | 75 | +6.7% | UNPROVEN |
| new_format | over25_over | Mexico Liga MX | 188 / -11.4% | 14 / -17.1% / [-71.3, +35.0] / 1 | 53 / -26.7% | 74 / -2.6% | 47 | -8.9% | UNPROVEN |
| new_format | over25_over | Sweden Allsvenskan | 168 / +3.4% | 21 / +19.3% / [-24.9, +63.5] / 1 | 30 / +29.3% | 71 / -18.7% | 46 | +9.5% | PROMISING |
| new_format | over25_over | USA MLS | 140 / +1.9% | 4 / -6.8% / [-112.3, +98.8] / 1 | 49 / +2.2% | 67 / -1.9% | 20 | -3.5% | UNPROVEN |
| new_format | over25_under | Austrian Bundesliga | 10 / +16.0% | 3 / -100.0% / [-100.0, -100.0] / 1 | 0 / - | 6 / +93.3% | 1 | -52.4% | UNPROVEN |
| new_format | over25_under | Brazil Serie A | 31 / -5.2% | 1 / -100.0% / - / - | 8 / +23.1% | 15 / -8.5% | 7 | -49.8% | - |
| new_format | over25_under | Finland Veikkausliiga | 33 / -21.1% | 8 / -24.2% / [-107.3, +72.7] / 1 | 2 / -8.5% | 9 / -10.3% | 14 | -13.6% | UNPROVEN |
| new_format | over25_under | Japan J-League | 81 / -1.1% | 18 / +10.1% / [-43.0, +63.1] / 1 | 16 / +34.9% | 38 / -18.7% | 9 | +5.2% | PROMISING |
| new_format | over25_under | Sweden Allsvenskan | 42 / -16.1% | 7 / -72.1% / [-126.7, -16.4] / 1 | 8 / -26.9% | 16 / +34.4% | 11 | -37.5% | UNPROVEN |
| new_format | over25_under | USA MLS | 7 / +42.6% | 5 / +20.4% / [-76.0, +116.8] / 1 | 1 / +103.0% | 1 / +93.0% | 0 | +10.3% | UNPROVEN |
| new_format | over35_over | Denmark Superliga | 10 / +61.1% | 8 / +76.0% / [+26.6, +125.4] / 0.0025 | 0 / - | 1 / +103.0% | 1 | +37.5% | UNPROVEN |
| new_format | over35_over | Finland Veikkausliiga | 16 / +11.0% | 8 / -1.5% / [-74.6, +71.6] / 1 | 2 / -4.0% | 5 / +59.2% | 1 | -0.4% | UNPROVEN |
| new_format | over35_over | Norway Eliteserien | 52 / -1.8% | 23 / -31.8% / [-72.0, +7.9] / 1 | 2 / +0.0% | 9 / +29.9% | 18 | -16.2% | UNPROVEN |
| new_format | over35_over | Sweden Allsvenskan | 12 / +47.8% | 5 / +98.2% / [+90.6, +105.8] / 0.0025 | 0 / - | 4 / +48.8% | 3 | +49.5% | PROMISING |
| standard | over15_over | Ligue 2 | 2 / +88.5% | 0 / - / - / - | 2 / +88.5% | 0 / - | 0 | - | - |
| standard | over25_over | Greek Super League | 3 / -32.3% | 0 / - / - / - | 0 / - | 1 / -100.0% | 2 | - | - |
| standard | over25_over | League One | 10 / +40.2% | 0 / - / - / - | 0 / - | 2 / -0.5% | 8 | - | - |
| standard | over25_over | Serie B | 5 / +63.0% | 0 / - / - / - | 0 / - | 2 / +0.0% | 3 | - | - |
| standard | over25_under | Belgian First Division A | 228 / -0.4% | 77 / +11.2% / [-11.5, +34.0] / 1 | 17 / +3.6% | 111 / -12.6% | 23 | +5.6% | UNPROVEN |
| standard | over35_over | Championship | 4 / -100.0% | 2 / -100.0% / [-100.0, -100.0] / - | 0 / - | 2 / -100.0% | 0 | -49.3% | - |
| standard | over35_over | Dutch Eredivisie | 6 / -35.8% | 2 / -100.0% / [-100.0, -100.0] / - | 1 / +84.0% | 2 / -100.0% | 1 | -50.1% | - |
### Signal CONS (median of the other books)
| Track | Market-side | League | ALL n / ROI | AGREE n / ROI / CI / Holm p | DISAGREE n / ROI | NONE n / ROI | NO_DATA+MIXED n | AGREE win-implied | AGREE verdict |
|---|---|---|---|---|---|---|---|---|---|
| new_format | over15_over | Argentina Primera Division | 56 / -19.5% | 0 / - / - / - | 0 / - | 0 / - | 56 | - | - |
| new_format | over25_over | Denmark Superliga | 106 / -3.7% | 5 / -23.8% / [-115.3, +67.7] / 1 | 26 / -24.5% | 45 / +19.4% | 30 | -12.1% | UNPROVEN |
| new_format | over25_over | Finland Veikkausliiga | 134 / +10.1% | 4 / -1.8% / [-113.2, +109.7] / 1 | 17 / -17.7% | 58 / +5.9% | 55 | -1.9% | UNPROVEN |
| new_format | over25_over | Mexico Liga MX | 188 / -11.4% | 8 / -51.0% / [-113.9, +11.9] / 1 | 29 / -18.3% | 86 / -3.2% | 65 | -26.8% | UNPROVEN |
| new_format | over25_over | Sweden Allsvenskan | 168 / +3.4% | 10 / +13.9% / [-71.1, +98.9] / 1 | 22 / +7.0% | 97 / -6.8% | 39 | +7.0% | PROMISING |
| new_format | over25_over | USA MLS | 140 / +1.9% | 4 / -6.5% / [-112.3, +99.3] / 1 | 32 / -4.0% | 91 / +1.7% | 13 | -3.5% | UNPROVEN |
| new_format | over25_under | Austrian Bundesliga | 10 / +16.0% | 3 / -37.3% / [-160.2, +88.0] / 1 | 0 / - | 6 / +62.0% | 1 | -20.5% | UNPROVEN |
| new_format | over25_under | Brazil Serie A | 31 / -5.2% | 0 / - / - / - | 6 / +62.3% | 22 / -10.7% | 3 | - | - |
| new_format | over25_under | Finland Veikkausliiga | 33 / -21.1% | 5 / +21.2% / [-75.8, +118.2] / 1 | 0 / - | 15 / -34.0% | 13 | +8.8% | UNPROVEN |
| new_format | over25_under | Japan J-League | 81 / -1.1% | 9 / +29.2% / [-35.1, +92.7] / 1 | 13 / +20.7% | 51 / -8.0% | 8 | +15.2% | PROMISING |
| new_format | over25_under | Sweden Allsvenskan | 42 / -16.1% | 4 / -51.2% / [-146.8, +46.2] / 1 | 7 / +12.7% | 21 / +1.9% | 10 | -27.8% | UNPROVEN |
| new_format | over25_under | USA MLS | 7 / +42.6% | 1 / +94.0% / - / - | 0 / - | 6 / +34.0% | 0 | +48.4% | - |
| new_format | over35_over | Denmark Superliga | 10 / +61.1% | 2 / +96.5% / [+79.8, +113.2] / - | 0 / - | 0 / - | 8 | +49.0% | - |
| new_format | over35_over | Finland Veikkausliiga | 16 / +11.0% | 0 / - / - / - | 0 / - | 0 / - | 16 | - | - |
| new_format | over35_over | Norway Eliteserien | 52 / -1.8% | 0 / - / - / - | 0 / - | 2 / +93.5% | 50 | - | - |
| new_format | over35_over | Sweden Allsvenskan | 12 / +47.8% | 1 / +96.0% / - / - | 0 / - | 0 / - | 11 | +49.0% | - |
| standard | over15_over | Ligue 2 | 2 / +88.5% | 0 / - / - / - | 0 / - | 0 / - | 2 | - | - |
| standard | over25_over | Greek Super League | 3 / -32.3% | 0 / - / - / - | 0 / - | 1 / -100.0% | 2 | - | - |
| standard | over25_over | League One | 10 / +40.2% | 0 / - / - / - | 0 / - | 3 / +32.7% | 7 | - | - |
| standard | over25_over | Serie B | 5 / +63.0% | 0 / - / - / - | 0 / - | 2 / +0.0% | 3 | - | - |
| standard | over25_under | Belgian First Division A | 228 / -0.4% | 47 / +23.5% / [-4.1, +51.0] / 1 | 9 / -13.2% | 151 / -8.5% | 21 | +11.9% | UNPROVEN |
| standard | over35_over | Championship | 4 / -100.0% | 0 / - / - / - | 0 / - | 0 / - | 4 | - | - |
| standard | over35_over | Dutch Eredivisie | 6 / -35.8% | 0 / - / - / - | 0 / - | 1 / -100.0% | 5 | - | - |
### Signal BOTH (PIN and CONS agree)
| Track | Market-side | League | ALL n / ROI | AGREE n / ROI / CI / Holm p | DISAGREE n / ROI | NONE n / ROI | NO_DATA+MIXED n | AGREE win-implied | AGREE verdict |
|---|---|---|---|---|---|---|---|---|---|
| new_format | over15_over | Argentina Primera Division | 56 / -19.5% | 0 / - / - / - | 0 / - | 0 / - | 56 | - | - |
| new_format | over25_over | Denmark Superliga | 106 / -3.7% | 3 / -34.7% / [-162.7, +93.4] / - | 22 / -19.4% | 25 / +30.3% | 56 | -18.7% | - |
| new_format | over25_over | Finland Veikkausliiga | 134 / +10.1% | 3 / +31.0% / [-100.0, +159.8] / 1 | 11 / -26.8% | 27 / -14.8% | 93 | +15.2% | UNPROVEN |
| new_format | over25_over | Mexico Liga MX | 188 / -11.4% | 7 / -44.0% / [-114.9, +26.9] / 1 | 23 / -22.5% | 57 / +5.9% | 101 | -22.9% | UNPROVEN |
| new_format | over25_over | Sweden Allsvenskan | 168 / +3.4% | 8 / +17.8% / [-63.5, +99.0] / 1 | 19 / +23.9% | 68 / -18.0% | 73 | +9.1% | PROMISING |
| new_format | over25_over | USA MLS | 140 / +1.9% | 2 / -7.0% / [-189.3, +175.3] / - | 29 / -13.2% | 67 / -1.9% | 42 | -3.5% | - |
| new_format | over25_under | Austrian Bundesliga | 10 / +16.0% | 2 / -100.0% / [-100.0, -100.0] / - | 0 / - | 5 / +94.4% | 3 | -54.2% | - |
| new_format | over25_under | Brazil Serie A | 31 / -5.2% | 0 / - / - / - | 2 / +99.5% | 14 / -15.4% | 15 | - | - |
| new_format | over25_under | Finland Veikkausliiga | 33 / -21.1% | 5 / +21.2% / [-75.8, +118.2] / 1 | 0 / - | 9 / -10.3% | 19 | +8.8% | UNPROVEN |
| new_format | over25_under | Japan J-League | 81 / -1.1% | 7 / +10.3% / [-70.6, +86.9] / 1 | 12 / +30.8% | 37 / -16.5% | 25 | +5.6% | PROMISING |
| new_format | over25_under | Sweden Allsvenskan | 42 / -16.1% | 4 / -51.2% / [-146.8, +46.2] / 1 | 5 / -19.4% | 15 / +30.5% | 18 | -27.8% | UNPROVEN |
| new_format | over25_under | USA MLS | 7 / +42.6% | 1 / +94.0% / - / - | 0 / - | 1 / +93.0% | 5 | +48.4% | - |
| new_format | over35_over | Denmark Superliga | 10 / +61.1% | 2 / +96.5% / [+79.8, +113.2] / - | 0 / - | 0 / - | 8 | +49.0% | - |
| new_format | over35_over | Finland Veikkausliiga | 16 / +11.0% | 0 / - / - / - | 0 / - | 0 / - | 16 | - | - |
| new_format | over35_over | Norway Eliteserien | 52 / -1.8% | 0 / - / - / - | 0 / - | 2 / +93.5% | 50 | - | - |
| new_format | over35_over | Sweden Allsvenskan | 12 / +47.8% | 1 / +96.0% / - / - | 0 / - | 0 / - | 11 | +49.0% | - |
| standard | over15_over | Ligue 2 | 2 / +88.5% | 0 / - / - / - | 0 / - | 0 / - | 2 | - | - |
| standard | over25_over | Greek Super League | 3 / -32.3% | 0 / - / - / - | 0 / - | 1 / -100.0% | 2 | - | - |
| standard | over25_over | League One | 10 / +40.2% | 0 / - / - / - | 0 / - | 2 / -0.5% | 8 | - | - |
| standard | over25_over | Serie B | 5 / +63.0% | 0 / - / - / - | 0 / - | 2 / +0.0% | 3 | - | - |
| standard | over25_under | Belgian First Division A | 228 / -0.4% | 43 / +21.7% / [-9.9, +50.9] / 1 | 9 / -13.2% | 109 / -12.8% | 67 | +10.9% | UNPROVEN |
| standard | over35_over | Championship | 4 / -100.0% | 0 / - / - / - | 0 / - | 0 / - | 4 | - | - |
| standard | over35_over | Dutch Eredivisie | 6 / -35.8% | 0 / - / - / - | 0 / - | 0 / - | 6 | - | - |

### Pooled view (a descriptive sum over cells, not a decision unit)

| Window | Signal | AGREE n / ROI | DISAGREE n / ROI | NONE n / ROI | ALL tips n / ROI |
|---|---|---|---|---|---|
| W72_6 | PIN | 219 / -0.8% | 281 / -0.9% | 506 / -5.4% | 1,344 / -0.9% |
| W72_6 | CONS | 103 / +10.8% | 161 / -5.2% | 658 / -2.0% | |
| W72_6 | BOTH | 88 / +10.2% | 132 / -6.2% | 441 / -4.2% | |
| W72_6_next | PIN | 154 / -1.6% | 247 / -2.0% | 447 / -3.3% | 1,187 / +1.2% |
| W72_6_next | CONS | 74 / +7.2% | 150 / -7.7% | 557 / -1.2% | |
| W72_6_next | BOTH | 60 / +3.3% | 124 / -8.4% | 383 / -2.8% | |
| W72_24 | PIN | 117 / -10.2% | 193 / -6.9% | 626 / -4.1% | 1,230 / -2.0% |
| W72_24 | CONS | 99 / -10.6% | 152 / -5.0% | 612 / -3.0% | |
| W72_24 | BOTH | 55 / -15.8% | 105 / -6.7% | 500 / -2.0% | |

**F2 contrasts:** 287 tested, and 0 have a Holm p below 0.025 (the smallest Holm p is 0.79). The strongest raw contrast is Belgium Standard O/U 2.5 Under:
- CONS AGREE beats NONE by +31.9 points (47 vs 151 bets, raw p 0.025).
- PIN AGREE returns +11.2% over 77 bets, against -12.6% for NONE over 111.

It is one cell among 287 and is not significant after Holm.

## 3. Market-only baseline (no model): follow every sharp move

The bet is the side the signal moved toward, at Pinnacle's own price in the bet snapshot.

| Window | Signal | O/U 2.5: n / ROI / CLV | 1X2: n / ROI / CLV |
|---|---|---|---|
| W72_6 | PIN | 1,813 / -4.6% / -0.29% | 6,903 / -2.7% / -0.26% |
| W72_6 | CONS | 1,139 / -4.2% / -0.18% | 7,057 / -2.2% / -0.11% |
| W72_6 | BOTH | 855 / -5.6% / -0.30% | 5,010 / -1.1% / -0.18% |
| W72_6_next | PIN | 1,670 / -2.5% / -0.18% | 6,414 / -3.2% / -0.15% |
| W72_6_next | CONS | 1,054 / -2.2% / -0.14% | 6,564 / -2.1% / -0.16% |
| W72_6_next | BOTH | 794 / -2.6% / -0.20% | 4,644 / -1.1% / -0.14% |
| W72_24 | PIN | 1,467 / -4.9% / -0.92% | 6,508 / -3.4% / -1.50% |
| W72_24 | CONS | 1,226 / -4.1% / -0.73% | 6,855 / -4.9% / -1.21% |
| W72_24 | BOTH | 722 / -5.1% / -1.01% | 4,239 / -4.0% / -1.63% |

- **Overall:** following the move alone loses roughly the bookmaker margin.
- **CLV:** CLV against Pinnacle's own close is about zero at T-6 h and negative at T-24 h. A Pinnacle move from T-72 h to T-24 h tends to partly reverse by kickoff.
- **Per cell:** of the 69 F1 cells with at least 150 bets (61 market-only, 8 model), none has a CI above 0. The best lower bound is -9.4%.
- **Question 3:** the model + CONS AGREE pattern (+10.8% / +7.2% across the two T-6 h windows) is not explained by the move itself, because following the same move without the model loses. But the AGREE-minus-market contrasts are not significant either (F2).

Full per-league market-only table for the primary window and signal (W72_6, PIN). Every other window and signal is in results.json:

| Side | League | n | ROI | ROI 95% CI | win-implied | CLV (n) | Holm p | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1x2_away | Argentina Primera Division | 94 | +2.3% | [-29.3, +33.9] | -0.6% | -0.8% (64) | 1 | UNPROVEN |
| 1x2_away | Austrian Bundesliga | 50 | -1.5% | [-38.0, +35.3] | +1.6% | -0.1% (50) | 1 | UNPROVEN |
| 1x2_away | Belgian First Division A | 93 | -5.1% | [-33.2, +22.6] | -1.4% | -1.0% (93) | 1 | UNPROVEN |
| 1x2_away | Brazil Serie A | 43 | +19.4% | [-27.8, +66.4] | +5.4% | -0.8% (36) | 1 | UNPROVEN |
| 1x2_away | Bundesliga 2 | 70 | +2.3% | [-28.9, +33.5] | +0.1% | -0.9% (62) | 1 | UNPROVEN |
| 1x2_away | Championship | 163 | -1.6% | [-23.4, +20.3] | -1.1% | -0.3% (163) | 1 | UNPROVEN |
| 1x2_away | China Super League | 61 | -2.3% | [-33.6, +28.9] | +0.6% | -1.5% (61) | 1 | UNPROVEN |
| 1x2_away | Denmark Superliga | 52 | -8.1% | [-48.8, +32.6] | -3.1% | -2.0% (52) | 1 | UNPROVEN |
| 1x2_away | Dutch Eredivisie | 76 | -18.5% | [-40.3, +3.4] | -3.5% | -0.8% (74) | 1 | UNPROVEN |
| 1x2_away | Finland Veikkausliiga | 48 | -18.2% | [-53.8, +17.4] | -8.5% | -1.4% (48) | 1 | UNPROVEN |
| 1x2_away | Greek Super League | 42 | +7.3% | [-26.5, +41.1] | +5.1% | +2.0% (41) | 1 | PROMISING |
| 1x2_away | Ireland Premier Division | 32 | -14.9% | [-59.5, +29.7] | -5.2% | -2.7% (32) | 1 | UNPROVEN |
| 1x2_away | Japan J-League | 77 | +0.7% | [-30.3, +32.4] | +1.0% | +1.7% (77) | 1 | PROMISING |
| 1x2_away | K-League 1 | 56 | +17.1% | [-20.0, +54.0] | +5.6% | +2.0% (56) | 1 | PROMISING |
| 1x2_away | La Liga 2 | 112 | -19.8% | [-46.2, +7.6] | -5.6% | -1.1% (109) | 1 | UNPROVEN |
| 1x2_away | League One | 191 | -2.1% | [-20.6, +16.4] | +0.3% | -1.7% (191) | 1 | UNPROVEN |
| 1x2_away | League Two | 170 | +3.6% | [-17.2, +25.3] | +2.1% | -0.1% (170) | 1 | UNPROVEN |
| 1x2_away | Ligue 2 | 105 | -3.5% | [-27.7, +21.8] | -0.8% | -0.9% (105) | 1 | UNPROVEN |
| 1x2_away | Mexico Liga MX | 71 | -11.6% | [-39.9, +16.8] | -3.5% | +1.8% (66) | 1 | UNPROVEN |
| 1x2_away | Norway Eliteserien | 53 | -3.9% | [-33.7, +26.0] | +2.1% | +0.9% (53) | 1 | UNPROVEN |
| 1x2_away | Scottish Premiership | 54 | -13.3% | [-39.9, +13.8] | -3.5% | -0.6% (54) | 1 | UNPROVEN |
| 1x2_away | Serie B | 93 | -21.3% | [-45.0, +2.4] | -6.3% | +0.4% (93) | 1 | UNPROVEN |
| 1x2_away | Sweden Allsvenskan | 61 | -7.1% | [-41.5, +27.4] | -1.5% | +0.3% (61) | 1 | UNPROVEN |
| 1x2_away | Turkish Super Lig | 61 | -37.3% | [-64.0, -10.5] | -10.0% | -1.2% (55) | 1 | UNPROVEN |
| 1x2_away | USA MLS | 108 | +9.0% | [-17.7, +35.8] | +3.2% | -0.6% (100) | 1 | UNPROVEN |
| 1x2_draw | Argentina Primera Division | 118 | -0.3% | [-25.7, +25.0] | +0.1% | +0.3% (90) | 1 | UNPROVEN |
| 1x2_draw | Austrian Bundesliga | 79 | +2.9% | [-35.1, +41.0] | -0.4% | -0.4% (78) | 1 | UNPROVEN |
| 1x2_draw | Belgian First Division A | 130 | -3.4% | [-31.2, +24.7] | -0.4% | -0.2% (130) | 1 | UNPROVEN |
| 1x2_draw | Brazil Serie A | 73 | -7.3% | [-43.3, +28.7] | -1.5% | +0.2% (61) | 1 | UNPROVEN |
| 1x2_draw | Bundesliga 2 | 103 | -12.1% | [-42.3, +18.1] | -2.9% | -0.1% (89) | 1 | UNPROVEN |
| 1x2_draw | Championship | 207 | -7.1% | [-30.3, +16.1] | -2.6% | -0.3% (207) | 1 | UNPROVEN |
| 1x2_draw | China Super League | 54 | -26.0% | [-72.5, +24.2] | -8.9% | -0.1% (54) | 1 | UNPROVEN |
| 1x2_draw | Denmark Superliga | 55 | -12.1% | [-55.0, +31.2] | -3.2% | -0.9% (55) | 1 | UNPROVEN |
| 1x2_draw | Dutch Eredivisie | 113 | +29.1% | [-8.8, +67.0] | +7.3% | -0.2% (112) | 1 | UNPROVEN |
| 1x2_draw | Finland Veikkausliiga | 53 | -10.6% | [-57.4, +37.2] | -2.3% | +0.0% (53) | 1 | UNPROVEN |
| 1x2_draw | Greek Super League | 92 | +8.7% | [-24.0, +41.4] | +2.0% | -0.8% (92) | 1 | UNPROVEN |
| 1x2_draw | Ireland Premier Division | 35 | +9.1% | [-41.9, +60.1] | +3.6% | -1.2% (35) | 1 | UNPROVEN |
| 1x2_draw | Japan J-League | 79 | -28.4% | [-59.0, +2.1] | -9.0% | -0.3% (79) | 1 | UNPROVEN |
| 1x2_draw | K-League 1 | 55 | -4.7% | [-45.5, +36.0] | -1.7% | -0.3% (55) | 1 | UNPROVEN |
| 1x2_draw | La Liga 2 | 176 | +2.7% | [-19.2, +24.6] | +1.0% | -0.3% (169) | 1 | UNPROVEN |
| 1x2_draw | League One | 188 | -6.0% | [-29.7, +17.8] | -1.2% | -0.1% (188) | 1 | UNPROVEN |
| 1x2_draw | League Two | 209 | +7.9% | [-13.9, +29.6] | +2.2% | -0.4% (209) | 1 | UNPROVEN |
| 1x2_draw | Ligue 2 | 154 | -5.6% | [-31.8, +20.5] | -2.2% | -0.7% (154) | 1 | UNPROVEN |
| 1x2_draw | Mexico Liga MX | 72 | -0.6% | [-36.9, +35.7] | +1.0% | +0.4% (68) | 1 | UNPROVEN |
| 1x2_draw | Norway Eliteserien | 85 | -5.7% | [-42.8, +33.6] | -2.1% | -0.2% (84) | 1 | UNPROVEN |
| 1x2_draw | Scottish Premiership | 70 | +12.5% | [-29.2, +54.3] | +4.1% | -1.5% (70) | 1 | UNPROVEN |
| 1x2_draw | Serie B | 148 | -2.5% | [-26.5, +21.4] | -0.3% | -0.1% (148) | 1 | UNPROVEN |
| 1x2_draw | Sweden Allsvenskan | 55 | -51.8% | [-86.0, -15.0] | -13.8% | -0.4% (55) | 1 | UNPROVEN |
| 1x2_draw | Turkish Super Lig | 124 | -7.0% | [-35.3, +21.4] | -2.2% | -0.7% (112) | 1 | UNPROVEN |
| 1x2_draw | USA MLS | 97 | -6.0% | [-38.6, +27.5] | -1.8% | -0.2% (89) | 1 | UNPROVEN |
| 1x2_home | Argentina Primera Division | 85 | -3.3% | [-25.6, +19.5] | -1.8% | +0.7% (64) | 1 | UNPROVEN |
| 1x2_home | Austrian Bundesliga | 43 | -39.1% | [-67.1, -10.7] | -14.6% | -2.2% (43) | 1 | UNPROVEN |
| 1x2_home | Belgian First Division A | 114 | +17.1% | [-5.8, +40.0] | +4.8% | -0.0% (114) | 1 | UNPROVEN |
| 1x2_home | Brazil Serie A | 69 | +6.9% | [-18.1, +30.9] | +1.6% | -0.3% (60) | 1 | UNPROVEN |
| 1x2_home | Bundesliga 2 | 90 | -8.4% | [-29.5, +12.8] | -5.0% | -1.0% (76) | 1 | UNPROVEN |
| 1x2_home | Championship | 140 | -1.0% | [-17.5, +15.4] | +0.3% | -0.1% (140) | 1 | UNPROVEN |
| 1x2_home | China Super League | 74 | +13.8% | [-13.4, +41.0] | +2.1% | +1.5% (74) | 1 | PROMISING |
| 1x2_home | Denmark Superliga | 49 | +1.0% | [-31.4, +33.7] | -1.1% | -0.2% (49) | 1 | UNPROVEN |
| 1x2_home | Dutch Eredivisie | 127 | +4.2% | [-17.0, +25.3] | +1.8% | +0.8% (126) | 1 | PROMISING |
| 1x2_home | Finland Veikkausliiga | 42 | +9.0% | [-25.7, +43.7] | +5.1% | -0.8% (42) | 1 | UNPROVEN |
| 1x2_home | Greek Super League | 47 | +0.3% | [-29.4, +30.4] | +1.3% | -0.8% (47) | 1 | UNPROVEN |
| 1x2_home | Ireland Premier Division | 39 | +18.4% | [-24.3, +61.0] | +2.2% | -0.4% (39) | 1 | UNPROVEN |
| 1x2_home | Japan J-League | 77 | -1.9% | [-26.3, +22.9] | -0.5% | +0.3% (77) | 1 | UNPROVEN |
| 1x2_home | K-League 1 | 53 | -11.0% | [-39.9, +17.8] | -4.7% | -0.3% (53) | 1 | UNPROVEN |
| 1x2_home | La Liga 2 | 134 | -1.4% | [-18.6, +15.8] | -0.4% | +0.0% (131) | 1 | UNPROVEN |
| 1x2_home | League One | 161 | +1.5% | [-14.2, +16.9] | +1.1% | -0.2% (161) | 1 | UNPROVEN |
| 1x2_home | League Two | 166 | -19.9% | [-35.8, -4.0] | -10.2% | -0.8% (166) | 1 | UNPROVEN |
| 1x2_home | Ligue 2 | 108 | +13.0% | [-5.9, +31.8] | +6.8% | +0.4% (108) | 1 | PROMISING |
| 1x2_home | Mexico Liga MX | 88 | +1.0% | [-18.3, +20.7] | +0.8% | +0.1% (82) | 1 | PROMISING |
| 1x2_home | Norway Eliteserien | 104 | +16.8% | [-5.0, +39.3] | +8.1% | +0.3% (104) | 1 | PROMISING |
| 1x2_home | Scottish Premiership | 67 | -14.5% | [-36.4, +8.6] | -4.6% | -1.2% (67) | 1 | UNPROVEN |
| 1x2_home | Serie B | 100 | -1.9% | [-23.0, +19.7] | -1.6% | +0.2% (100) | 1 | UNPROVEN |
| 1x2_home | Sweden Allsvenskan | 47 | +5.8% | [-25.0, +36.6] | +2.4% | +0.1% (47) | 1 | PROMISING |
| 1x2_home | Turkish Super Lig | 98 | -14.2% | [-36.6, +10.0] | -5.2% | +1.1% (88) | 1 | UNPROVEN |
| 1x2_home | USA MLS | 121 | -2.4% | [-18.1, +13.2] | -1.6% | +0.3% (114) | 1 | UNPROVEN |
| over25_over | Argentina Primera Division | 6 | +0.2% | [-87.8, +88.1] | -0.3% | -0.5% (6) | 1 | UNPROVEN |
| over25_over | Austrian Bundesliga | 9 | -12.4% | [-80.3, +55.5] | -6.6% | +1.6% (9) | 1 | UNPROVEN |
| over25_over | Belgian First Division A | 18 | -13.7% | [-59.6, +32.5] | -7.5% | +0.7% (18) | 1 | UNPROVEN |
| over25_over | Brazil Serie A | 32 | +5.4% | [-35.2, +44.3] | +2.5% | +1.0% (32) | 1 | PROMISING |
| over25_over | Bundesliga 2 | 5 | -20.2% | [-116.0, +75.6] | -10.0% | +1.8% (5) | 1 | UNPROVEN |
| over25_over | Championship | 60 | -7.3% | [-32.6, +18.0] | -3.7% | +0.2% (60) | 1 | UNPROVEN |
| over25_over | China Super League | 5 | -62.8% | [-135.7, +11.6] | -32.6% | +3.8% (5) | 1 | UNPROVEN |
| over25_over | Denmark Superliga | 6 | -67.3% | [-131.4, -3.3] | -35.5% | -0.2% (6) | 1 | UNPROVEN |
| over25_over | Dutch Eredivisie | 6 | +28.5% | [-51.4, +108.4] | +15.4% | -3.7% (6) | 1 | UNPROVEN |
| over25_over | Finland Veikkausliiga | 7 | +11.9% | [-65.8, +89.5] | +6.7% | -1.6% (7) | 1 | UNPROVEN |
| over25_over | Greek Super League | 33 | +2.1% | [-32.9, +36.4] | +0.5% | +0.1% (33) | 1 | PROMISING |
| over25_over | Ireland Premier Division | 25 | -28.2% | [-66.6, +10.1] | -14.6% | -0.1% (25) | 1 | UNPROVEN |
| over25_over | Japan J-League | 59 | -12.4% | [-37.9, +13.0] | -7.0% | +1.1% (59) | 1 | UNPROVEN |
| over25_over | K-League 1 | 46 | -31.2% | [-58.8, -3.6] | -15.5% | -2.1% (46) | 1 | UNPROVEN |
| over25_over | La Liga 2 | 53 | -14.2% | [-40.9, +12.5] | -7.4% | -0.9% (53) | 1 | UNPROVEN |
| over25_over | League One | 101 | -7.0% | [-27.1, +13.1] | -3.5% | -1.2% (101) | 1 | UNPROVEN |
| over25_over | League Two | 92 | -8.9% | [-28.9, +11.1] | -4.8% | -1.0% (92) | 1 | UNPROVEN |
| over25_over | Ligue 2 | 58 | -2.6% | [-28.2, +23.0] | -1.2% | -0.2% (57) | 1 | UNPROVEN |
| over25_over | Mexico Liga MX | 20 | -12.6% | [-56.1, +34.6] | -6.9% | +0.6% (20) | 1 | UNPROVEN |
| over25_over | Norway Eliteserien | 10 | -61.7% | [-111.8, -4.4] | -31.9% | -2.6% (10) | 1 | UNPROVEN |
| over25_over | Scottish Premiership | 14 | -1.3% | [-69.3, +66.7] | -0.7% | +0.6% (14) | 1 | UNPROVEN |
| over25_over | Serie B | 62 | +0.1% | [-25.9, +26.2] | -0.4% | +0.9% (62) | 1 | PROMISING |
| over25_over | Sweden Allsvenskan | 21 | +19.3% | [-24.9, +63.5] | +9.5% | +1.0% (21) | 1 | PROMISING |
| over25_over | Turkish Super Lig | 28 | +3.8% | [-32.7, +40.3] | +2.5% | -0.9% (28) | 1 | UNPROVEN |
| over25_over | USA MLS | 13 | +34.4% | [-16.4, +85.2] | +17.4% | +0.6% (13) | 1 | PROMISING |
| over25_under | Argentina Primera Division | 1 | +88.0% | - | +46.8% | +1.6% (1) | - | - |
| over25_under | Austrian Bundesliga | 28 | -3.9% | [-40.3, +32.5] | -2.2% | +0.9% (28) | 1 | UNPROVEN |
| over25_under | Belgian First Division A | 82 | +14.1% | [-8.2, +35.7] | +7.0% | -1.0% (82) | 1 | UNPROVEN |
| over25_under | Brazil Serie A | 4 | -4.8% | [-112.5, +103.0] | -2.0% | -1.4% (4) | 1 | UNPROVEN |
| over25_under | Bundesliga 2 | 48 | -5.2% | [-34.5, +23.7] | -2.6% | -0.8% (48) | 1 | UNPROVEN |
| over25_under | Championship | 85 | -0.0% | [-22.2, +22.1] | +0.0% | -0.7% (85) | 1 | UNPROVEN |
| over25_under | China Super League | 26 | -10.0% | [-51.3, +31.3] | -4.9% | +2.1% (26) | 1 | UNPROVEN |
| over25_under | Denmark Superliga | 29 | +1.2% | [-37.8, +40.2] | +0.2% | -1.1% (29) | 1 | UNPROVEN |
| over25_under | Dutch Eredivisie | 52 | +16.9% | [-9.6, +43.3] | +8.6% | -0.3% (52) | 1 | UNPROVEN |
| over25_under | Finland Veikkausliiga | 25 | -5.5% | [-49.7, +38.7] | -2.9% | -0.1% (25) | 1 | UNPROVEN |
| over25_under | Greek Super League | 9 | -35.7% | [-102.7, +31.4] | -18.3% | +1.7% (9) | 1 | UNPROVEN |
| over25_under | Ireland Premier Division | 14 | -18.6% | [-69.8, +32.5] | -10.0% | -0.9% (14) | 1 | UNPROVEN |
| over25_under | Japan J-League | 40 | -3.5% | [-35.9, +28.9] | -1.5% | +0.2% (38) | 1 | UNPROVEN |
| over25_under | K-League 1 | 21 | +0.8% | [-41.4, +43.0] | +1.3% | +0.2% (21) | 1 | PROMISING |
| over25_under | La Liga 2 | 21 | -17.4% | [-66.2, +31.4] | -8.6% | -0.2% (21) | 1 | UNPROVEN |
| over25_under | League One | 93 | +2.0% | [-19.3, +22.4] | +1.1% | -0.3% (93) | 1 | UNPROVEN |
| over25_under | League Two | 115 | +2.1% | [-16.5, +20.7] | +1.0% | +0.7% (115) | 1 | PROMISING |
| over25_under | Ligue 2 | 32 | -16.6% | [-51.5, +18.5] | -8.4% | -3.2% (32) | 1 | UNPROVEN |
| over25_under | Mexico Liga MX | 71 | +19.3% | [-2.6, +41.2] | +10.2% | -0.6% (71) | 1 | UNPROVEN |
| over25_under | Norway Eliteserien | 32 | -9.0% | [-44.4, +28.6] | -4.4% | +0.9% (32) | 1 | UNPROVEN |
| over25_under | Scottish Premiership | 40 | -17.7% | [-48.7, +13.3] | -8.8% | -0.4% (40) | 1 | UNPROVEN |
| over25_under | Serie B | 18 | -45.2% | [-89.2, -0.4] | -24.3% | -1.8% (18) | 1 | UNPROVEN |
| over25_under | Sweden Allsvenskan | 30 | -35.0% | [-73.9, +3.9] | -18.4% | +0.1% (30) | 1 | UNPROVEN |
| over25_under | Turkish Super Lig | 38 | +7.5% | [-23.8, +38.7] | +3.8% | -1.1% (38) | 1 | UNPROVEN |
| over25_under | USA MLS | 70 | -13.4% | [-36.3, +10.4] | -6.6% | -0.6% (70) | 1 | UNPROVEN |

## 4. Timing: what the filter would need in practice

### 4a. When Pinnacle moves (O/U, the same line as the T-72 h open)

How to read the table:
- The (0,4] h bucket holds the final pre-kickoff movement snapshot, so its "same sign" is 1 by construction.
- The (60,84] h bucket is the open itself.

| Lead bucket (h) | Fixtures | Mean abs change of Pinnacle fair Over since T-72 h (pp) | Same sign as the final pre-kickoff movement |
|---|---|---|---|
| (60,84] | 20181 | 0.011 | 0.018 |
| (36,60] | 18500 | 0.478 | 0.4511 |
| (18,36] | 17114 | 0.817 | 0.6854 |
| (10,18] | 15807 | 1.083 | 0.7486 |
| (4,10] | 14906 | 1.402 | 0.8644 |
| (0,4] | 12617 | 1.605 | 1.0 |
| close (CLV snapshot) | 13758 | 1.589 | - |

- **When the information arrives:** most of the move comes late. At T-24 h Pinnacle has made about half of its final movement (0.82 of 1.6 points), and its direction matches the final direction 69% of the time. At T-6 h these figures are 1.40 points and 86%.
- **What that means for a signal read at T-24 h:** it is a much noisier read of the eventual sharp direction. This matches its negative results above.
- **The alternative, betting at the close:** the CLOSE snapshot may never be a bet price (owner rule), so that alternative was not studied.

### 4b. Live tips: first sight is T-125 h (Standard) / T-153 h (New-Format); the filter needs T-6 h

The tier BT snapshot is taken at the measured live lead time and priced by Pinnacle only. For each approved O/U 2.5 MODEL cell, the table shows the tips at the BT price on fixtures where W72_6 is readable, and their PIN class at T-6 h. These are prices only; no outcome is used.

| Track | Market-side | League | Active weeks | Tips at BT | with a PIN reading | PIN AGREE | DISAGREE | NONE | Tips/active week | PIN AGREE/active week | BT tips with a T-6 h Pinnacle 2.5 pair | still a tip at T-6 h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| new_format | over25_under | Austrian Bundesliga | 94 | 122 | 70 | 20 | 15 | 35 | 1.30 | 0.21 | 23 | 10 |
| standard | over25_under | Belgian First Division A | 116 | 60 | 56 | 8 | 13 | 35 | 0.52 | 0.07 | 31 | 31 |
| new_format | over25_under | Brazil Serie A | 108 | 30 | 29 | 8 | 8 | 13 | 0.28 | 0.07 | 9 | 8 |
| new_format | over25_over | Denmark Superliga | 96 | 297 | 216 | 50 | 47 | 119 | 3.09 | 0.52 | 60 | 60 |
| new_format | over25_over | Finland Veikkausliiga | 82 | 188 | 152 | 41 | 34 | 77 | 2.29 | 0.50 | 50 | 50 |
| new_format | over25_under | Finland Veikkausliiga | 82 | 93 | 81 | 17 | 23 | 41 | 1.13 | 0.21 | 14 | 9 |
| standard | over25_over | Greek Super League | 101 | 37 | 27 | 6 | 4 | 17 | 0.37 | 0.06 | 4 | 3 |
| new_format | over25_under | Japan J-League | 106 | 155 | 116 | 21 | 22 | 73 | 1.46 | 0.20 | 34 | 26 |
| standard | over25_over | La Liga 2 | 128 | 191 | 148 | 15 | 46 | 87 | 1.49 | 0.12 | 2 | 0 |
| standard | over25_over | League One | 127 | 28 | 23 | 4 | 2 | 17 | 0.22 | 0.03 | 2 | 2 |
| standard | over25_over | Ligue 2 | 105 | 34 | 28 | 3 | 12 | 13 | 0.32 | 0.03 | 0 | 0 |
| new_format | over25_over | Mexico Liga MX | 122 | 152 | 135 | 38 | 38 | 59 | 1.25 | 0.31 | 54 | 54 |
| new_format | over25_under | Norway Eliteserien | 92 | 37 | 14 | 7 | 3 | 4 | 0.40 | 0.08 | 0 | 0 |
| new_format | over25_under | Saudi Pro League | 11 | 3 | 0 | 0 | 0 | 0 | 0.27 | 0.00 | 0 | 0 |
| standard | over25_over | Serie B | 107 | 192 | 150 | 19 | 30 | 101 | 1.79 | 0.18 | 7 | 3 |
| new_format | over25_over | Sweden Allsvenskan | 83 | 252 | 193 | 49 | 35 | 109 | 3.04 | 0.59 | 68 | 68 |
| new_format | over25_under | Sweden Allsvenskan | 83 | 139 | 97 | 20 | 25 | 52 | 1.68 | 0.24 | 16 | 14 |
| new_format | over25_over | USA MLS | 113 | 220 | 217 | 68 | 31 | 118 | 1.95 | 0.60 | 59 | 57 |
| new_format | over25_under | USA MLS | 113 | 168 | 157 | 47 | 36 | 74 | 1.49 | 0.42 | 12 | 5 |

- **Bet timing the filter needs:** hold every tip from first sight (T-125/153 h) until about **T-6 h**, read the signal, and bet then (or within the next hour or two). That is about 5-6 days later than tips are made today.
- **Do tips survive the wait?** Of the BT tips that also have a Pinnacle 2.5 pair at T-6 h, 400 of 445 (90%) still clear their bar at the T-6 h price. The model's view mostly survives the wait.
- **Passing tips per week:**
  - The 19 approved O/U 2.5 MODEL cells together make about **24.3 tips per active league-week** at BT.
  - About **4.4 per active league-week** would pass PIN AGREE: 23% of the BT tips with a PIN reading, or 18% of all BT tips.
  - These figures sum each league's rate over its own active weeks, so they are an upper bound for a calendar week in which every league plays. The over15/over35 cells are in results.json.
- **What live operation would need:** our own captures at T-72 h and T-6 h across Pinnacle and at least 3 other books, for every league with an approved cell.
  - The current sharp tracker polls every ~2 h but stores the median price across books, not per-book pairs.
  - CONS needs per-book exact-2.5 pairs.

## 5. Forward paper protocol (proposal; the owner decides)

**Nothing qualifies by the strict rule.** The only pattern worth watching is "model tip + consensus AGREE at T-6 h":
- It is positive in both T-6 h windows and negative at T-24 h.
- Every DISAGREE bucket is negative.
- It is small (n ≈ 100) and not significant.

Proposed paper track `SF-PAPER-1`, from 2026-11-21, if the owner agrees:
- **Universe.** Every tip of an approved O/U 2.5 MODEL cell, with the bars of TP-2026-11-15-bade2c0b frozen and no change to the live tip itself.
- **Signal.** Exactly sharp-filter/1, W72_6:
  - PIN and CONS from our own captures. The snapshots closest to 72 h (within [60, 84] h) and to 6 h (within [4, 10] h) before the match's own kickoff.
  - Thresholds: the 75th percentile of |change| from all earlier fixtures of the league. History first (B4 to 2026-06-13), then our own live captures, with at least 100 earlier fixtures.
- **Paper bet.**
  - Placed at the first captured real pre-kickoff price at or after the T-6 h snapshot, one book's pair (Pinnacle when it quotes the exact line).
  - Recorded with its class (AGREE / DISAGREE / NONE / NO_DATA) beside the tip's own first-sight paper result.
  - CLV against the match's own close.
- **Review.** At 150 AGREE paper bets per cell, or at the 2-month Clean Record milestone, whichever comes first.
  - Test: AGREE ROI > 0 and AGREE - NONE > 0, Holm over the cells reviewed.
  - No stake without a new owner decision.
- **Prerequisite (an owner decision):** per-book exact-line O/U captures at T-72 h and T-6 h. The sharp tracker stores medians today, so a small capped collector change is needed. Test it before deploying it.

**Not proposed:** market-only following (it loses about the margin, with CLV about 0), the T-24 h signal (negative), and 1X2 with the model (no clean 1X2 model exists; Phase 2 would need one first).

## 6. Limits stated plainly

- **Pinnacle coverage.** The featured endpoint gives Pinnacle one main line, so only about 27% of fixtures have the exact 2.5 bet price. One-book pricing was kept, as the bars study requires. Using best-of-books or consensus prices would have raised n but broken the real-price rule.
- **Bars are in-sample.** Bars were chosen on the same pre-2026-09-01 rows, so the ALL-tip ROI is optimistic. The AGREE / NONE / DISAGREE comparison within the same tips is less affected.
- **Line moves count as beyond the threshold.** A main-line move (kind LINE, 5,847 W72_6 fixtures) always counts as beyond the threshold. That is a pre-registered choice.
- **Over 1.5 / Over 3.5.** Pinnacle rarely features 1.5 or 3.5, so these cells are nearly empty.
- **CLV coverage.** The B1 close (the match's own kickoff minus 5 min) is used when it exists, else B4's last snapshot. In both cases it must be later than the bet.
