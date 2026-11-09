# P6 — Per-league bars for owner approval (November 2026)

Proposal **TP-2026-11-09-c2414fcf**, proposal_hash `c2414fcf64f15aac1310a35f491dac37ccd0e4f3ec46ddf1768b730e27ed90df`, file `output/threshold_proposals/2026-11_c2414fcf.json`. Status AWAITING_OWNER: 71 league × market entries. Nothing is applied until you approve; `src/threshold_apply` writes exactly the entries you name.

Built 2026-11-09 by `python scripts/bars_opt.py serving-cells --month 2026-11` (src/bars_opt/cells.py): every family re-chosen by the pre-registered protocol (reports/P4_BARS_OPTIMIZATION.md) on all priced rows before 2026-11-01, with your decision **"Learned blend, own number"**: a league with thin own data keeps its OWN bars, its ROI curve pulled toward the all-league curve by a weight learned from the data (reported per cell as the shrink weight; 1 = all own data, 0 = no own weight at that bar). No global bar anywhere: a league × market without an approved cell tips nothing (NO_LEAGUE_BAR).

**Nothing here is PROVEN.** Verdicts come from the study's untouched FINAL window (2026-09-01..11-08); a cell served only through the learned blend, or under NO_SELECTION_EVIDENCE, is UNPROVEN. This season is the live test: approve to collect clean evidence month by month.

## Go-live day: what you approve, in order

1. Merge `prep/bars-serving` into data-truth-100 together with go-live (it contains prep/no-global-bars, prep/bars-opt and prep/top3-league). On its own, before step 3, it silences every tip (no league has an applied cell yet).
2. **Re-approve the research lock** (it is RED on the branch, by design): the only PROTECTED real-money track, Argentina Primera Division BTTS YES, loses its old global gates (0.10/0.08/0.04 → none until its cell is applied); predict.yml no longer sets LEAGUE_SNIPER_CAP / MARKSMAN_THRESHOLD / VALUABLE_THRESHOLD (not read by serving); and config/bankroll_policy.json stakes per CELL (section Staking). Prepared note (runbook step U3a): `python -m src.research_lock --approve --by "Nevo (owner)" --note "2026-11 go-live: bars_opt serving (no global bars, NO_LEAGUE_BAR), predict.yml global threshold variables removed, per-cell staking 'Paper + PROMISING small' (10 staked cells, reports/P6_BARS_FOR_APPROVAL.md)"`. threshold_apply refuses while the lock is RED.
3. **Approve the entries** you want in `config/threshold_approvals.json` → `approvals`, by entry_id + entry_hash only (hashes are listed per league below), with proposal_id / proposal_hash above, approved_by and approved_at. Then `python -m src.threshold_apply --proposal output/threshold_proposals/2026-11_c2414fcf.json --dry-run`, then without --dry-run.
4. Argentina Primera Division BTTS YES (PROTECTED, real money) keeps tipping only if `btts|Argentina Primera Division` is approved: its cell is LEAGUE_WIDE (every match up to odds 2.50), UNPROVEN, 350 selection bets at +5.4%.
5. **TOP-3 on** (owner 2026-11-09, runbook step U3c): repository variables `TOP3_FREEZE_LEAD_MIN = 120` (freeze 2 h before each league-week's first kickoff) and `TOP3_LEAGUE_ENABLED = true`, both AT GO-LIVE. TOP-3 reads the same approved cells, ranks by the cell's expected value, and its picks follow their cell's staking state (no extra stake).

**Standard Over 1.5** (3 learned-blend cells under RAW|EDGE, NO_SELECTION_EVIDENCE): **owner-approved in principle 2026-11-09 as PAPER / UNPROVEN.** They are not staked (no cell_staking entry); the owner still approves their entries by id + hash on go-live day.

## Staking this season: "Paper + PROMISING small" (owner 2026-11-09)

`config/bankroll_policy.json` → `cell_staking` (enabled): staking is per CELL (league × market × side, key `<track>|<league>`). Only the cells below stake; **every other cell is PAPER** (shown and recorded, stake 0, reason NO_REAL_STAKE_PAPER_NO_CELL_ENTRY), never inherited from its track. A BLOCKED track (Over 3.5, player props) stays blocked for its cells. Portfolio caps (single 2%, fixture 2%, daily 6%, league 3%), the drawdown throttle, paper leagues (USA MLS) and the direction rule are unchanged. Cloudbet stays PAPER_ONLY: the stake plan only sizes and records.

| Staked cell | State | Base fraction | Max odds | Tier multipliers S / M / V |
|---|---|---|---|---|
| `new_format|BTTS|Argentina Primera Division` | ACTIVE | 0.0125 | 2.6 | 1.1 / 0.9 / 1.0 |
| `new_format|BTTS|Brazil Serie A` | ACTIVE_LOW | 0.005 | 2.5 | 1.0 / 1.0 / 1.0 |
| `new_format|BTTS|Ireland Premier Division` | ACTIVE_LOW | 0.005 | 2.5 | 1.0 / 1.0 / 1.0 |
| `new_format|BTTS|Norway Eliteserien` | ACTIVE_LOW | 0.005 | 2.5 | 1.0 / 1.0 / 1.0 |
| `new_format|OU25_OVER|Brazil Serie A` | ACTIVE_LOW | 0.005 | 2.5 | 1.0 / 1.0 / 1.0 |
| `new_format|OU25_OVER|Sweden Allsvenskan` | ACTIVE_LOW | 0.005 | 2.5 | 1.0 / 1.0 / 1.0 |
| `standard|BTTS|La Liga 2` | ACTIVE_LOW | 0.005 | 2.5 | 1.0 / 1.0 / 1.0 |
| `standard|BTTS|Serie B` | ACTIVE_LOW | 0.005 | 2.5 | 1.0 / 1.0 / 1.0 |
| `standard|OU25_OVER|Serie B` | ACTIVE_LOW | 0.005 | HARD ceiling 2.75 | 1.0 / 1.0 / 1.0 |
| `standard|OU25_UNDER|Scottish League Two` | ACTIVE_LOW | 0.005 | 2.5 | 1.0 / 1.0 / 1.0 |

Tier multipliers per cell (a decision for you): the 9 PROMISING cells stake every tier flat (1.0), the way their evidence was measured (flat stakes on every tip at or above VALUABLE). Argentina BTTS keeps SNIPER 1.1 / MARKSMAN 0.9 as today **plus VALUABLE 1.0**: its cell is LEAGUE_WIDE and every league-wide tip is served as VALUABLE, so without it the protected track would stop staking. Scottish League Two is not a tracked registry league (the registry shows 8 ACTIVE_LOW cells), but its cell stakes like the others.

**Stake replay, 2026-10-16..2026-11-14** (`scripts/stake_cells_replay.py`, notional bankroll 1, P/L on settled stakes; `output/threshold_proposals/bars_opt_serving/2026-11_stake_replay.json`):

| View | Selections | Staked | Total fraction | Settled staked | P/L (bankroll units) | ROI on staked |
|---|---|---|---|---|---|---|
| A: same live selections, CURRENT per-track policy | 731 | 47 | 0.2983 | 41 | -0.03641 | -13.0% |
| A: same live selections, NEW per-cell policy | 731 | 93 | 0.5725 | 86 | -0.13207 | -24.3% |
| B: tips the proposed cells produce, NEW per-cell policy | 106 | 91 | 0.53 | 69 | 0.00013 | +0.0% |

View A isolates the staking change on the tips the OLD bars produced (pre-retrain engines); view B sizes what the new cells would have tipped. Both are short samples on old-model probabilities: they show what gets staked, not future profit. Staked cells in B: `new_format|BTTS|Argentina Primera Division` 39; `new_format|BTTS|Brazil Serie A` 9; `new_format|BTTS|Norway Eliteserien` 1; `new_format|OU25_OVER|Brazil Serie A` 2; `standard|BTTS|La Liga 2` 37; `standard|BTTS|Serie B` 1; `standard|OU25_OVER|Serie B` 2.

## Cell counts

| Track | Market | Chosen configuration | Basis | PROVEN | PROMISING | UNPROVEN (own) | UNPROVEN (blend) | OFF no own estimate | OFF negative | OFF no value | OFF thin | NO_PRICE_DATA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| standard | O/U 2.5 Over | `SHRINK|EDGE|max-|min-` | SELECTED | 0 | 1 | 0 | 0 | 16 | 0 | 0 | 0 | 0 |
| standard | O/U 2.5 Under | `RAW|EV|max2.5|min-` | SELECTED | 0 | 1 | 1 | 0 | 0 | 0 | 15 | 0 | 0 |
| standard | BTTS Yes | `RAW|EV|max2.5|min-` | SELECTED | 0 | 2 | 4 | 4 | 7 | 0 | 0 | 0 | 0 |
| standard | Over 1.5 | `RAW|EDGE|max-|min-` | NO_SELECTION_EVIDENCE | 0 | 0 | 0 | 3 | 0 | 0 | 14 | 0 | 0 |
| standard | Over 3.5 | `LEARNED|EDGE|max-|min-` | SELECTED | 0 | 0 | 3 | 4 | 10 | 0 | 0 | 0 | 0 |
| new_format | O/U 2.5 Over | `RAW|EV|max2.5|min-` | SELECTED | 0 | 2 | 8 | 5 | 0 | 0 | 0 | 0 | 0 |
| new_format | O/U 2.5 Under | `RAW|EDGE|max3.5|min-` | SELECTED | 0 | 0 | 0 | 0 | 4 | 0 | 11 | 0 | 0 |
| new_format | BTTS Yes | `RAW|EDGE|max2.5|min-` | SELECTED | 0 | 3 | 6 | 6 | 0 | 0 | 0 | 0 | 0 |
| new_format | Over 1.5 | `BLEND|EV|max-|min-` | SELECTED | 0 | 0 | 1 | 3 | 11 | 0 | 0 | 0 | 0 |
| new_format | Over 3.5 | `NONE|ALL|max-|min-` | SELECTED | 0 | 0 | 0 | 0 | 0 | 0 | 15 | 0 | 0 |
| **all** | | | | **0** | **9** | **23** | **25** | **48** | **0** | **55** | **0** | **0** |

Configuration = probability source | edge measure | max odds | min odds. RAW = the model's probability; BLEND = model and fair price combined by a fit on earlier rows; SHRINK = the model pulled toward the fair price by the league's learned weight; LEARNED = Pro's learned-tiers logistic; NONE|ALL = LEAGUE_WIDE (every match in the odds range). EDGE = p − fair; EV = p × odds − 1. Standard Over 1.5 had too few selection bets for the protocol to choose: it uses production's own definition RAW|EDGE, labelled NO_SELECTION_EVIDENCE.

OFF reasons. OFF no own estimate: a learned-blend league with fewer than 2 own bets at its VALUABLE bar - its bars would be the pooled curve (weight 0), a stand-in, never served (the same 2-bet own-variance rule src.league_bars.choose_bars uses). OFF negative: the study's FINAL check put the cell's whole 95% CI below zero (none in November's refit; Ireland New-Format Under 2.5 had it in the study and is OFF no value now). OFF no value: no bar has a positive (blended) selection ROI, so there is nothing to tip. OFF thin: too few bets own AND pooled. NO_PRICE_DATA: no real two-sided price before the month.

## Per league

Bars: VALUABLE / MARKSMAN / SNIPER on the cell's measure. Selection = rows before 2026-11-01 at the VALUABLE bar (own bets, own ROI, blended ROI). FINAL = the study's untouched 2026-09-01..11-08 check of the pre-September choice (bets, ROI, 95% CI).

### Argentina Primera Division (new_format)

| Market | Config | Mode | Bars V / M / S | Odds | Verdict | Basis | Shrink w | Selection bets / ROI / blended | FINAL bets / ROI / CI |
|---|---|---|---|---|---|---|---|---|---|
| BTTS Yes | `NONE|ALL|max2.5|min-` | LEAGUE_WIDE | 0.020 / 0.020 / 0.300 | 1.01–2.5 | **UNPROVEN** | OWN_BETS | 0.00 | 350 / +5.4% / +0.3% | 68 / +9.2% / [-17.5%, +36.8%] |
| Over 1.5 | `BLEND|EV|max-|min-` | MODEL | 0.020 / 0.040 / 0.040 | any | **UNPROVEN** | OWN_BETS | 0.98 | 193 / +6.4% / +6.4% | — |
| O/U 2.5 Over | `RAW|EV|max2.5|min-` | MODEL | 0.030 / 0.030 / 0.030 | 1.01–2.5 | **UNPROVEN** | OWN_BETS | 0.00 | 40 / -2.5% / +8.4% | — |

Entries: `btts|Argentina Primera Division` hash `9e3fa8f8b8e3a952…`; `ou25|Argentina Primera Division` hash `ed3d6ff5603c4439…`; `over15|Argentina Primera Division` hash `9cdfb5357a32ee8d…`

OFF: O/U 2.5 Under = OFF_NO_OWN_ESTIMATE; Over 3.5 = OFF_NO_VALUE

### Austrian Bundesliga (new_format)

| Market | Config | Mode | Bars V / M / S | Odds | Verdict | Basis | Shrink w | Selection bets / ROI / blended | FINAL bets / ROI / CI |
|---|---|---|---|---|---|---|---|---|---|
| BTTS Yes | `RAW|EDGE|max2.5|min-` | MODEL | 0.020 / 0.020 / 0.020 | 1.01–2.5 | **UNPROVEN** | OWN_BETS | 0.00 | 104 / +9.9% / +6.5% | 3 / -40.0% / [-157.6%, +80.0%] |
| O/U 2.5 Over | `RAW|EV|max2.5|min-` | MODEL | 0.020 / 0.140 / 0.150 | 1.01–2.5 | **UNPROVEN** | OWN_BETS | 0.98 | 124 / -0.1% / +0.1% | 1 / +90.0% / [—, —] |

Entries: `btts|Austrian Bundesliga` hash `3853321134448187…`; `ou25|Austrian Bundesliga` hash `db9c120b3620f862…`

OFF: Over 1.5 = OFF_NO_OWN_ESTIMATE; O/U 2.5 Under = OFF_NO_VALUE; Over 3.5 = OFF_NO_VALUE

### Brazil Serie A (new_format)

| Market | Config | Mode | Bars V / M / S | Odds | Verdict | Basis | Shrink w | Selection bets / ROI / blended | FINAL bets / ROI / CI |
|---|---|---|---|---|---|---|---|---|---|
| BTTS Yes | `RAW|EDGE|max2.5|min-` | MODEL | 0.020 / 0.020 / 0.060 | 1.01–2.5 | **PROMISING** | OWN_BETS | 0.00 | 246 / +14.3% / +6.5% | 22 / +28.3% / [-13.4%, +69.9%] |
| Over 1.5 | `BLEND|EV|max-|min-` | MODEL | 0.020 / 0.030 / 0.030 | any | **UNPROVEN** | LEARNED_BLEND | 0.88 | 23 / +20.4% / +18.7% | — |
| O/U 2.5 Over | `RAW|EV|max2.5|min-` | MODEL | 0.020 / 0.020 / 0.020 | 1.01–2.5 | **PROMISING** | OWN_BETS | 0.98 | 162 / +20.3% / +20.1% | 11 / +30.0% / [-31.3%, +91.3%] |

Entries: `btts|Brazil Serie A` hash `6ca9609c7782da82…`; `ou25|Brazil Serie A` hash `85a434c38830d0ad…`; `over15|Brazil Serie A` hash `c5888eb74a03405a…`

OFF: O/U 2.5 Under = OFF_NO_VALUE; Over 3.5 = OFF_NO_VALUE

### China Super League (new_format)

| Market | Config | Mode | Bars V / M / S | Odds | Verdict | Basis | Shrink w | Selection bets / ROI / blended | FINAL bets / ROI / CI |
|---|---|---|---|---|---|---|---|---|---|
| BTTS Yes | `RAW|EDGE|max2.5|min-` | MODEL | 0.020 / 0.020 / 0.020 | 1.01–2.5 | **UNPROVEN** | OWN_BETS | 0.00 | 76 / -8.9% / +6.5% | 3 / -40.0% / [-157.6%, +80.0%] |
| O/U 2.5 Over | `RAW|EV|max2.5|min-` | MODEL | 0.020 / 0.070 / 0.210 | 1.01–2.5 | **UNPROVEN** | LEARNED_BLEND | 0.95 | 38 / +4.1% / +4.3% | — |

Entries: `btts|China Super League` hash `980cd09ceb95920b…`; `ou25|China Super League` hash `0718f0ff7270b617…`

OFF: Over 1.5 = OFF_NO_OWN_ESTIMATE; O/U 2.5 Under = OFF_NO_VALUE; Over 3.5 = OFF_NO_VALUE

### Denmark Superliga (new_format)

| Market | Config | Mode | Bars V / M / S | Odds | Verdict | Basis | Shrink w | Selection bets / ROI / blended | FINAL bets / ROI / CI |
|---|---|---|---|---|---|---|---|---|---|
| BTTS Yes | `RAW|EDGE|max2.5|min-` | MODEL | 0.020 / 0.060 / 0.060 | 1.01–2.5 | **UNPROVEN** | LEARNED_BLEND | 0.00 | 22 / +12.8% / +6.5% | — |
| O/U 2.5 Over | `RAW|EV|max2.5|min-` | MODEL | 0.020 / 0.020 / 0.020 | 1.01–2.5 | **UNPROVEN** | OWN_BETS | 0.97 | 54 / +10.1% / +10.1% | — |

Entries: `btts|Denmark Superliga` hash `19ce592d8c7b5e18…`; `ou25|Denmark Superliga` hash `cf55140b1b51927f…`

OFF: Over 1.5 = OFF_NO_OWN_ESTIMATE; O/U 2.5 Under = OFF_NO_VALUE; Over 3.5 = OFF_NO_VALUE

### Finland Veikkausliiga (new_format)

| Market | Config | Mode | Bars V / M / S | Odds | Verdict | Basis | Shrink w | Selection bets / ROI / blended | FINAL bets / ROI / CI |
|---|---|---|---|---|---|---|---|---|---|
| BTTS Yes | `RAW|EDGE|max2.5|min-` | MODEL | 0.020 / 0.060 / 0.060 | 1.01–2.5 | **UNPROVEN** | LEARNED_BLEND | 0.00 | 28 / +9.8% / +6.5% | — |
| O/U 2.5 Over | `RAW|EV|max2.5|min-` | MODEL | 0.030 / 0.030 / 0.230 | 1.01–2.5 | **UNPROVEN** | LEARNED_BLEND | 0.00 | 28 / -11.5% / +8.4% | — |

Entries: `btts|Finland Veikkausliiga` hash `b33d230422d54aa7…`; `ou25|Finland Veikkausliiga` hash `0b6e809b24476f2f…`

OFF: Over 1.5 = OFF_NO_OWN_ESTIMATE; O/U 2.5 Under = OFF_NO_VALUE; Over 3.5 = OFF_NO_VALUE

### Ireland Premier Division (new_format)

| Market | Config | Mode | Bars V / M / S | Odds | Verdict | Basis | Shrink w | Selection bets / ROI / blended | FINAL bets / ROI / CI |
|---|---|---|---|---|---|---|---|---|---|
| BTTS Yes | `RAW|EDGE|max2.5|min-` | MODEL | 0.020 / 0.020 / 0.020 | 1.01–2.5 | **PROMISING** | OWN_BETS | 0.00 | 113 / +10.8% / +6.5% | 6 / +25.8% / [-69.5%, +121.2%] |
| Over 1.5 | `BLEND|EV|max-|min-` | MODEL | 0.020 / 0.020 / 0.020 | any | **UNPROVEN** | LEARNED_BLEND | 1.00 | 2 / +64.5% / +64.4% | — |
| O/U 2.5 Over | `RAW|EV|max2.5|min-` | MODEL | 0.020 / 0.020 / 0.020 | 1.01–2.5 | **UNPROVEN** | OWN_BETS | 0.96 | 65 / +11.0% / +10.9% | 1 / +75.0% / [—, —] |

Entries: `btts|Ireland Premier Division` hash `0f17a1b67302ad79…`; `ou25|Ireland Premier Division` hash `2d8fce3e3fae7e82…`; `over15|Ireland Premier Division` hash `50144d7623096e42…`

OFF: O/U 2.5 Under = OFF_NO_VALUE; Over 3.5 = OFF_NO_VALUE

### Japan J-League (new_format)

| Market | Config | Mode | Bars V / M / S | Odds | Verdict | Basis | Shrink w | Selection bets / ROI / blended | FINAL bets / ROI / CI |
|---|---|---|---|---|---|---|---|---|---|
| BTTS Yes | `RAW|EDGE|max2.5|min-` | MODEL | 0.020 / 0.020 / 0.020 | 1.01–2.5 | **UNPROVEN** | OWN_BETS | 0.00 | 41 / -34.1% / +6.5% | — |
| Over 1.5 | `BLEND|EV|max-|min-` | MODEL | 0.040 / 0.070 / 0.070 | any | **UNPROVEN** | LEARNED_BLEND | 0.01 | 4 / -14.8% / +11.9% | — |
| O/U 2.5 Over | `RAW|EV|max2.5|min-` | MODEL | 0.020 / 0.030 / 0.030 | 1.01–2.5 | **UNPROVEN** | OWN_BETS | 0.95 | 49 / +3.1% / +3.3% | 1 / -100.0% / [—, —] |

Entries: `btts|Japan J-League` hash `595fb7a3bfc30dc5…`; `ou25|Japan J-League` hash `e4fbc36efe59029f…`; `over15|Japan J-League` hash `42c5d6f977c69458…`

OFF: O/U 2.5 Under = OFF_NO_VALUE; Over 3.5 = OFF_NO_VALUE

### K-League 1 (new_format)

| Market | Config | Mode | Bars V / M / S | Odds | Verdict | Basis | Shrink w | Selection bets / ROI / blended | FINAL bets / ROI / CI |
|---|---|---|---|---|---|---|---|---|---|
| BTTS Yes | `RAW|EDGE|max2.5|min-` | MODEL | 0.020 / 0.020 / 0.090 | 1.01–2.5 | **UNPROVEN** | LEARNED_BLEND | 0.00 | 4 / +2.0% / +6.5% | — |
| O/U 2.5 Over | `RAW|EV|max2.5|min-` | MODEL | 0.020 / 0.130 / 0.150 | 1.01–2.5 | **UNPROVEN** | LEARNED_BLEND | 0.61 | 4 / +68.8% / +45.4% | — |

Entries: `btts|K-League 1` hash `5afa08e6f3868126…`; `ou25|K-League 1` hash `47447929729de599…`

OFF: Over 1.5 = OFF_NO_OWN_ESTIMATE; O/U 2.5 Under = OFF_NO_OWN_ESTIMATE; Over 3.5 = OFF_NO_VALUE

### Mexico Liga MX (new_format)

| Market | Config | Mode | Bars V / M / S | Odds | Verdict | Basis | Shrink w | Selection bets / ROI / blended | FINAL bets / ROI / CI |
|---|---|---|---|---|---|---|---|---|---|
| BTTS Yes | `RAW|EDGE|max2.5|min-` | MODEL | 0.020 / 0.020 / 0.020 | 1.01–2.5 | **UNPROVEN** | OWN_BETS | 0.00 | 48 / +15.7% / +6.5% | 6 / +43.2% / [-14.7%, +99.4%] |
| O/U 2.5 Over | `RAW|EV|max2.5|min-` | MODEL | 0.020 / 0.060 / 0.070 | 1.01–2.5 | **UNPROVEN** | OWN_BETS | 0.97 | 67 / +16.8% / +16.5% | 4 / -53.8% / [-144.4%, +38.8%] |

Entries: `btts|Mexico Liga MX` hash `f65309a49ddb73d5…`; `ou25|Mexico Liga MX` hash `d4819e548602dd88…`

OFF: Over 1.5 = OFF_NO_OWN_ESTIMATE; O/U 2.5 Under = OFF_NO_VALUE; Over 3.5 = OFF_NO_VALUE

### Norway Eliteserien (new_format)

| Market | Config | Mode | Bars V / M / S | Odds | Verdict | Basis | Shrink w | Selection bets / ROI / blended | FINAL bets / ROI / CI |
|---|---|---|---|---|---|---|---|---|---|
| BTTS Yes | `RAW|EDGE|max2.5|min-` | MODEL | 0.020 / 0.020 / 0.020 | 1.01–2.5 | **PROMISING** | OWN_BETS | 0.00 | 63 / +9.2% / +6.5% | 6 / +46.2% / [-11.3%, +103.6%] |
| O/U 2.5 Over | `RAW|EV|max2.5|min-` | MODEL | 0.020 / 0.030 / 0.030 | 1.01–2.5 | **UNPROVEN** | OWN_BETS | 0.98 | 72 / +2.1% / +2.3% | 2 / +64.0% / [+32.6%, +95.4%] |

Entries: `btts|Norway Eliteserien` hash `168dfeada7a4a02b…`; `ou25|Norway Eliteserien` hash `69d4f3dd86c206de…`

OFF: Over 1.5 = OFF_NO_OWN_ESTIMATE; O/U 2.5 Under = OFF_NO_VALUE; Over 3.5 = OFF_NO_VALUE

### Romanian Superliga (new_format)

| Market | Config | Mode | Bars V / M / S | Odds | Verdict | Basis | Shrink w | Selection bets / ROI / blended | FINAL bets / ROI / CI |
|---|---|---|---|---|---|---|---|---|---|
| BTTS Yes | `RAW|EDGE|max2.5|min-` | MODEL | 0.020 / 0.020 / 0.090 | 1.01–2.5 | **UNPROVEN** | LEARNED_BLEND | 0.00 | 19 / +29.6% / +6.5% | — |
| O/U 2.5 Over | `RAW|EV|max2.5|min-` | MODEL | 0.030 / 0.080 / 0.230 | 1.01–2.5 | **UNPROVEN** | LEARNED_BLEND | 0.00 | 6 / -32.5% / +8.4% | — |

Entries: `btts|Romanian Superliga` hash `0037f8cd2537de30…`; `ou25|Romanian Superliga` hash `172ca647fd9ee157…`

OFF: Over 1.5 = OFF_NO_OWN_ESTIMATE; O/U 2.5 Under = OFF_NO_OWN_ESTIMATE; Over 3.5 = OFF_NO_VALUE

### Saudi Pro League (new_format)

| Market | Config | Mode | Bars V / M / S | Odds | Verdict | Basis | Shrink w | Selection bets / ROI / blended | FINAL bets / ROI / CI |
|---|---|---|---|---|---|---|---|---|---|
| BTTS Yes | `RAW|EDGE|max2.5|min-` | MODEL | 0.020 / 0.020 / 0.090 | 1.01–2.5 | **UNPROVEN** | LEARNED_BLEND | 0.00 | 6 / -10.2% / +6.5% | — |
| O/U 2.5 Over | `RAW|EV|max2.5|min-` | MODEL | 0.020 / 0.020 / 0.020 | 1.01–2.5 | **UNPROVEN** | LEARNED_BLEND | 1.00 | 3 / +109.3% / +109.3% | — |

Entries: `btts|Saudi Pro League` hash `8b4717a587ffba95…`; `ou25|Saudi Pro League` hash `e87fb4bb38d631d0…`

OFF: Over 1.5 = OFF_NO_OWN_ESTIMATE; O/U 2.5 Under = OFF_NO_OWN_ESTIMATE; Over 3.5 = OFF_NO_VALUE

### Sweden Allsvenskan (new_format)

| Market | Config | Mode | Bars V / M / S | Odds | Verdict | Basis | Shrink w | Selection bets / ROI / blended | FINAL bets / ROI / CI |
|---|---|---|---|---|---|---|---|---|---|
| BTTS Yes | `RAW|EDGE|max2.5|min-` | MODEL | 0.020 / 0.020 / 0.020 | 1.01–2.5 | **UNPROVEN** | OWN_BETS | 0.00 | 47 / -2.4% / +6.5% | 3 / -36.3% / [-161.1%, +88.4%] |
| O/U 2.5 Over | `RAW|EV|max2.5|min-` | MODEL | 0.020 / 0.020 / 0.020 | 1.01–2.5 | **PROMISING** | OWN_BETS | 0.97 | 69 / +24.6% / +24.1% | 2 / +81.5% / [+64.8%, +98.2%] |

Entries: `btts|Sweden Allsvenskan` hash `481d43644b9fc488…`; `ou25|Sweden Allsvenskan` hash `3f0d355253570691…`

OFF: Over 1.5 = OFF_NO_OWN_ESTIMATE; O/U 2.5 Under = OFF_NO_VALUE; Over 3.5 = OFF_NO_VALUE

### USA MLS (new_format)

| Market | Config | Mode | Bars V / M / S | Odds | Verdict | Basis | Shrink w | Selection bets / ROI / blended | FINAL bets / ROI / CI |
|---|---|---|---|---|---|---|---|---|---|
| BTTS Yes | `RAW|EDGE|max2.5|min-` | MODEL | 0.020 / 0.020 / 0.090 | 1.01–2.5 | **UNPROVEN** | LEARNED_BLEND | 0.00 | 23 / +14.4% / +6.5% | — |
| O/U 2.5 Over | `RAW|EV|max2.5|min-` | MODEL | 0.020 / 0.030 / 0.030 | 1.01–2.5 | **UNPROVEN** | OWN_BETS | 0.97 | 66 / +2.2% / +2.3% | — |

Entries: `btts|USA MLS` hash `b98b750d8c0735b2…`; `ou25|USA MLS` hash `00d1ec6c3c289763…`

OFF: Over 1.5 = OFF_NO_OWN_ESTIMATE; O/U 2.5 Under = OFF_NO_VALUE; Over 3.5 = OFF_NO_VALUE

### Belgian First Division A (standard)

| Market | Config | Mode | Bars V / M / S | Odds | Verdict | Basis | Shrink w | Selection bets / ROI / blended | FINAL bets / ROI / CI |
|---|---|---|---|---|---|---|---|---|---|
| O/U 2.5 Under | `RAW|EV|max2.5|min-` | MODEL | 0.020 / 0.020 / 0.020 | 1.01–2.5 | **UNPROVEN** | OWN_BETS | 0.57 | 422 / +8.8% / +1.2% | 10 / -14.0% / [-98.4%, +70.4%] |

Entries: `ou25|Belgian First Division A` hash `ab6e58de59e5d75f…`

OFF: BTTS Yes = OFF_NO_OWN_ESTIMATE; Over 1.5 = OFF_NO_VALUE; O/U 2.5 Over = OFF_NO_OWN_ESTIMATE; Over 3.5 = OFF_NO_OWN_ESTIMATE

### Bundesliga 2 (standard)

OFF: BTTS Yes = OFF_NO_OWN_ESTIMATE; Over 1.5 = OFF_NO_VALUE; O/U 2.5 Over = OFF_NO_OWN_ESTIMATE; O/U 2.5 Under = OFF_NO_VALUE; Over 3.5 = OFF_NO_OWN_ESTIMATE

### Championship (standard)

| Market | Config | Mode | Bars V / M / S | Odds | Verdict | Basis | Shrink w | Selection bets / ROI / blended | FINAL bets / ROI / CI |
|---|---|---|---|---|---|---|---|---|---|
| BTTS Yes | `RAW|EV|max2.5|min-` | MODEL | 0.080 / 0.100 / 0.140 | 1.01–2.5 | **UNPROVEN** | OWN_BETS | 0.43 | 163 / +4.8% / +1.4% | — |
| Over 1.5 | `RAW|EDGE|max-|min-` | MODEL | 0.020 / 0.030 / 0.030 | any | **UNPROVEN** | LEARNED_BLEND | 0.66 | 5 / +7.0% / +1.1% | — |
| Over 3.5 | `LEARNED|EDGE|max-|min-` | MODEL | 0.020 / 0.040 / 0.050 | any | **UNPROVEN** | OWN_BETS | 0.00 | 86 / +7.8% / +8.8% | — |

Entries: `btts|Championship` hash `513864c48afbfb88…`; `over15|Championship` hash `2612cf596c63c501…`; `over35|Championship` hash `43633b4953bddc3f…`

OFF: O/U 2.5 Over = OFF_NO_OWN_ESTIMATE; O/U 2.5 Under = OFF_NO_VALUE

### Dutch Eredivisie (standard)

| Market | Config | Mode | Bars V / M / S | Odds | Verdict | Basis | Shrink w | Selection bets / ROI / blended | FINAL bets / ROI / CI |
|---|---|---|---|---|---|---|---|---|---|
| Over 3.5 | `LEARNED|EDGE|max-|min-` | MODEL | 0.020 / 0.020 / 0.020 | any | **UNPROVEN** | OWN_BETS | 0.00 | 40 / +7.8% / +8.8% | — |

Entries: `over35|Dutch Eredivisie` hash `0b82c9a4214deec3…`

OFF: BTTS Yes = OFF_NO_OWN_ESTIMATE; Over 1.5 = OFF_NO_VALUE; O/U 2.5 Over = OFF_NO_OWN_ESTIMATE; O/U 2.5 Under = OFF_NO_VALUE

### Greek Super League (standard)

| Market | Config | Mode | Bars V / M / S | Odds | Verdict | Basis | Shrink w | Selection bets / ROI / blended | FINAL bets / ROI / CI |
|---|---|---|---|---|---|---|---|---|---|
| BTTS Yes | `RAW|EV|max2.5|min-` | MODEL | 0.100 / 0.200 / 0.240 | 1.01–2.5 | **UNPROVEN** | LEARNED_BLEND | 0.02 | 10 / -14.5% / +1.2% | — |

Entries: `btts|Greek Super League` hash `fb9d8fdc34536731…`

OFF: Over 1.5 = OFF_NO_VALUE; O/U 2.5 Over = OFF_NO_OWN_ESTIMATE; O/U 2.5 Under = OFF_NO_VALUE; Over 3.5 = OFF_NO_OWN_ESTIMATE

### La Liga 2 (standard)

| Market | Config | Mode | Bars V / M / S | Odds | Verdict | Basis | Shrink w | Selection bets / ROI / blended | FINAL bets / ROI / CI |
|---|---|---|---|---|---|---|---|---|---|
| BTTS Yes | `NONE|ALL|max2.5|min-` | LEAGUE_WIDE | 0.020 / 0.020 / 0.300 | 1.01–2.5 | **PROMISING** | OWN_BETS | 0.60 | 932 / +5.3% / +2.3% | 5 / +33.0% / [-73.4%, +139.4%] |
| Over 3.5 | `LEARNED|EDGE|max-|min-` | MODEL | 0.020 / 0.050 / 0.060 | any | **UNPROVEN** | OWN_BETS | 0.00 | 92 / +13.5% / +8.8% | — |

Entries: `btts|La Liga 2` hash `7614a423ed58377e…`; `over35|La Liga 2` hash `2488ea4436d6bc1e…`

OFF: Over 1.5 = OFF_NO_VALUE; O/U 2.5 Over = OFF_NO_OWN_ESTIMATE; O/U 2.5 Under = OFF_NO_VALUE

### League One (standard)

| Market | Config | Mode | Bars V / M / S | Odds | Verdict | Basis | Shrink w | Selection bets / ROI / blended | FINAL bets / ROI / CI |
|---|---|---|---|---|---|---|---|---|---|
| BTTS Yes | `RAW|EV|max2.5|min-` | MODEL | 0.110 / 0.130 / 0.130 | 1.01–2.5 | **UNPROVEN** | OWN_BETS | 0.00 | 87 / -15.8% / +1.3% | — |
| Over 1.5 | `RAW|EDGE|max-|min-` | MODEL | 0.030 / 0.030 / 0.040 | any | **UNPROVEN** | LEARNED_BLEND | 0.97 | 3 / +37.3% / +35.8% | — |
| Over 3.5 | `LEARNED|EDGE|max-|min-` | MODEL | 0.020 / 0.020 / 0.070 | any | **UNPROVEN** | LEARNED_BLEND | 0.00 | 12 / +20.8% / +8.8% | — |

Entries: `btts|League One` hash `520aa8c0019ff5b4…`; `over15|League One` hash `828d61dec4a9126b…`; `over35|League One` hash `f5bec9910abc5403…`

OFF: O/U 2.5 Over = OFF_NO_OWN_ESTIMATE; O/U 2.5 Under = OFF_NO_VALUE

### League Two (standard)

| Market | Config | Mode | Bars V / M / S | Odds | Verdict | Basis | Shrink w | Selection bets / ROI / blended | FINAL bets / ROI / CI |
|---|---|---|---|---|---|---|---|---|---|
| BTTS Yes | `RAW|EV|max2.5|min-` | MODEL | 0.080 / 0.100 / 0.130 | 1.01–2.5 | **UNPROVEN** | OWN_BETS | 0.35 | 115 / +5.8% / +1.4% | — |
| Over 3.5 | `LEARNED|EDGE|max-|min-` | MODEL | 0.020 / 0.020 / 0.070 | any | **UNPROVEN** | LEARNED_BLEND | 0.00 | 5 / -40.0% / +8.8% | — |

Entries: `btts|League Two` hash `b617b04dd372bdfa…`; `over35|League Two` hash `2b1e8147e1923ccb…`

OFF: Over 1.5 = OFF_NO_VALUE; O/U 2.5 Over = OFF_NO_OWN_ESTIMATE; O/U 2.5 Under = OFF_NO_VALUE

### Ligue 2 (standard)

| Market | Config | Mode | Bars V / M / S | Odds | Verdict | Basis | Shrink w | Selection bets / ROI / blended | FINAL bets / ROI / CI |
|---|---|---|---|---|---|---|---|---|---|
| BTTS Yes | `RAW|EV|max2.5|min-` | MODEL | 0.110 / 0.130 / 0.130 | 1.01–2.5 | **UNPROVEN** | OWN_BETS | 0.00 | 65 / -21.8% / +1.3% | — |
| Over 1.5 | `RAW|EDGE|max-|min-` | MODEL | 0.020 / 0.020 / 0.020 | any | **UNPROVEN** | LEARNED_BLEND | 1.00 | 2 / +42.0% / +41.9% | — |

Entries: `btts|Ligue 2` hash `7db26c4454e53f0c…`; `over15|Ligue 2` hash `f6a0a86724178dd7…`

OFF: O/U 2.5 Over = OFF_NO_OWN_ESTIMATE; O/U 2.5 Under = OFF_NO_VALUE; Over 3.5 = OFF_NO_OWN_ESTIMATE

### National League (standard)

OFF: BTTS Yes = OFF_NO_OWN_ESTIMATE; Over 1.5 = OFF_NO_VALUE; O/U 2.5 Over = OFF_NO_OWN_ESTIMATE; O/U 2.5 Under = OFF_NO_VALUE; Over 3.5 = OFF_NO_OWN_ESTIMATE

### Portuguese Primeira Liga (standard)

| Market | Config | Mode | Bars V / M / S | Odds | Verdict | Basis | Shrink w | Selection bets / ROI / blended | FINAL bets / ROI / CI |
|---|---|---|---|---|---|---|---|---|---|
| BTTS Yes | `RAW|EV|max2.5|min-` | MODEL | 0.100 / 0.170 / 0.200 | 1.01–2.5 | **UNPROVEN** | LEARNED_BLEND | 0.01 | 8 / +11.6% / +1.7% | — |
| Over 3.5 | `LEARNED|EDGE|max-|min-` | MODEL | 0.020 / 0.070 / 0.070 | any | **UNPROVEN** | LEARNED_BLEND | 0.00 | 35 / +23.7% / +8.8% | — |

Entries: `btts|Portuguese Primeira Liga` hash `ba9bae4f32f984e9…`; `over35|Portuguese Primeira Liga` hash `587cd7dd59f0f52b…`

OFF: Over 1.5 = OFF_NO_VALUE; O/U 2.5 Over = OFF_NO_OWN_ESTIMATE; O/U 2.5 Under = OFF_NO_VALUE

### Scottish Championship (standard)

| Market | Config | Mode | Bars V / M / S | Odds | Verdict | Basis | Shrink w | Selection bets / ROI / blended | FINAL bets / ROI / CI |
|---|---|---|---|---|---|---|---|---|---|
| BTTS Yes | `RAW|EV|max2.5|min-` | MODEL | 0.100 / 0.240 / 0.240 | 1.01–2.5 | **UNPROVEN** | LEARNED_BLEND | 0.01 | 4 / +5.0% / +1.6% | — |

Entries: `btts|Scottish Championship` hash `a0c2050f39cf30bf…`

OFF: Over 1.5 = OFF_NO_VALUE; O/U 2.5 Over = OFF_NO_OWN_ESTIMATE; O/U 2.5 Under = OFF_NO_VALUE; Over 3.5 = OFF_NO_OWN_ESTIMATE

### Scottish League One (standard)

OFF: BTTS Yes = OFF_NO_OWN_ESTIMATE; Over 1.5 = OFF_NO_VALUE; O/U 2.5 Over = OFF_NO_OWN_ESTIMATE; O/U 2.5 Under = OFF_NO_VALUE; Over 3.5 = OFF_NO_OWN_ESTIMATE

### Scottish League Two (standard)

| Market | Config | Mode | Bars V / M / S | Odds | Verdict | Basis | Shrink w | Selection bets / ROI / blended | FINAL bets / ROI / CI |
|---|---|---|---|---|---|---|---|---|---|
| O/U 2.5 Under | `RAW|EV|max2.5|min-` | MODEL | 0.030 / 0.030 / 0.030 | 1.01–2.5 | **PROMISING** | OWN_BETS | 0.44 | 205 / +11.7% / +0.2% | 9 / +4.0% / [-76.6%, +84.6%] |

Entries: `ou25|Scottish League Two` hash `656ceae1cf0f0a03…`

OFF: BTTS Yes = OFF_NO_OWN_ESTIMATE; Over 1.5 = OFF_NO_VALUE; O/U 2.5 Over = OFF_NO_OWN_ESTIMATE; Over 3.5 = OFF_NO_OWN_ESTIMATE

### Scottish Premiership (standard)

OFF: BTTS Yes = OFF_NO_OWN_ESTIMATE; Over 1.5 = OFF_NO_VALUE; O/U 2.5 Over = OFF_NO_OWN_ESTIMATE; O/U 2.5 Under = OFF_NO_VALUE; Over 3.5 = OFF_NO_OWN_ESTIMATE

### Serie B (standard)

| Market | Config | Mode | Bars V / M / S | Odds | Verdict | Basis | Shrink w | Selection bets / ROI / blended | FINAL bets / ROI / CI |
|---|---|---|---|---|---|---|---|---|---|
| BTTS Yes | `RAW|EV|max2.5|min-` | MODEL | 0.020 / 0.050 / 0.050 | 1.01–2.5 | **PROMISING** | OWN_BETS | 0.28 | 393 / +6.1% / +0.0% | 6 / +32.5% / [-74.2%, +139.2%] |
| O/U 2.5 Over | `SHRINK|EDGE|max-|min-` | MODEL | 0.020 / 0.020 / 0.020 | any | **PROMISING** | OWN_BETS | 0.00 | 157 / +5.6% / +5.6% | 2 / +132.5% / [+117.8%, +147.2%] |

Entries: `btts|Serie B` hash `a9b68dd31009710f…`; `ou25|Serie B` hash `946bd72b3723ac9c…`

OFF: Over 1.5 = OFF_NO_VALUE; O/U 2.5 Under = OFF_NO_VALUE; Over 3.5 = OFF_NO_OWN_ESTIMATE

### Turkish Super Lig (standard)

| Market | Config | Mode | Bars V / M / S | Odds | Verdict | Basis | Shrink w | Selection bets / ROI / blended | FINAL bets / ROI / CI |
|---|---|---|---|---|---|---|---|---|---|
| BTTS Yes | `RAW|EV|max2.5|min-` | MODEL | 0.100 / 0.240 / 0.240 | 1.01–2.5 | **UNPROVEN** | LEARNED_BLEND | 0.01 | 3 / -30.0% / +1.4% | — |
| Over 3.5 | `LEARNED|EDGE|max-|min-` | MODEL | 0.020 / 0.020 / 0.070 | any | **UNPROVEN** | LEARNED_BLEND | 0.00 | 14 / -44.1% / +8.8% | — |

Entries: `btts|Turkish Super Lig` hash `cddf3904fb74d73a…`; `over35|Turkish Super Lig` hash `5f7a0afa41b20889…`

OFF: Over 1.5 = OFF_NO_VALUE; O/U 2.5 Over = OFF_NO_OWN_ESTIMATE; O/U 2.5 Under = OFF_NO_VALUE

## Entries that remove or replace an existing production entry

These leagues have a production entry today (from the pre-repair optimiser; none carries a VALUABLE bar, so none serves under the no-global rule). Approving them removes the stale entry (REMOVE: the league is OFF in November) or replaces an `approved: true` legacy entry with an UNPROVEN cell (APPROVAL_REVOKED).

- `ou25|Bundesliga 2`: REMOVE
- `ou25|Championship`: REMOVE
- `ou25|La Liga 2`: REMOVE
- `ou25|League One`: REMOVE
- `ou25|League Two`: REMOVE
- `ou25|Ligue 2`: REMOVE
- `ou25|Serie B`: APPROVAL_REVOKED
- `btts|Bundesliga 2`: REMOVE
- `btts|Championship`: APPROVAL_REVOKED
- `over15|Bundesliga 2`: REMOVE
- `over15|La Liga 2`: REMOVE
- `over15|League Two`: REMOVE
- `over15|Serie B`: REMOVE
- `over35|Bundesliga 2`: REMOVE
- `over35|Ligue 2`: REMOVE
- `over35|Serie B`: REMOVE

## Replay on the live board (in memory, nothing applied)

`scripts/bars_serving_replay.py` scored every live first-sight candidate (2026-10-25..11-15, both O/U 2.5 sides) with the proposed cells as if all were approved. Output: `output/threshold_proposals/bars_opt_serving/2026-11_replay.json`.

- Last 30 days: 2,345 side-rows → **106 tips** (84 settled, ROI -0.2%).
- Reasons: NO_PROPOSED_CELL 1,286, BELOW_LEAGUE_BAR 605, NO_TWO_SIDED_PRICE 333, TIPPED 106, OUTSIDE_ODDS_RANGE 15.
- Board at 2026-11-09 02:35 UTC (international break): 190 side-rows → **11 tips**.
- TOP-3 per league, the last 3 gameweeks with results (2026-W43, 2026-W44, 2026-W45): **40 picks, ROI -3.0%** (2026-W43: 8 picks -58.8%, 2026-W44: 15 picks -3.3%, 2026-W45: 17 picks +23.5%).

**Caveat:** the live candidates come from the engines serving when they were logged (the PRE-RETRAIN models until go-live), while the cells were fitted on the clean retrained models' predictions. The replay therefore shows how the cells filter the board (volume per league x market, the TOP-3 mechanics), not how the clean models will perform. The panel's file holds only W43–W45.

## Decisions for you

1. Approve the entries per league (or all), then apply (steps above).
2. Real money vs paper per league: every cell is UNPROVEN or PROMISING. Recommendation: paper everywhere except the existing PROTECTED Argentina BTTS track, until the Clean Record proves a cell.
3. LEAGUE_WIDE bets are served with tier **VALUABLE** and `bar_mode = LEAGUE_WIDE` (the tip contract knows SNIPER / MARKSMAN / VALUABLE only). Keep, or add a LEAGUE_WIDE tier to the contract.
4. Standard Over 1.5 has no selection evidence: approve its 3 blend cells under production's RAW|EDGE definition (labelled NO_SELECTION_EVIDENCE) or leave Over 1.5 Standard OFF.
5. The exec repo still has a global execution minimum `min_edge_vs_consensus` (src/control/policy.py:64/175, used at bot.py:293 and selection.py:72): not edited here.
