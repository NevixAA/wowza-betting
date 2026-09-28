# Wowza Fantasy — product redesign

*What was wrong, what changed, what it now measures, and what is still missing.*

Written 2026-09-27/28 against v9 HEAD. Betting system untouched throughout: no O/U logic, no
thresholds, no staking, no Telegram tips, no collectors, no retraining, no v11.

---

## 1. Current-state audit — the numbers, re-measured

The brief supplied figures and said not to trust them. Re-measured at HEAD, **they were exact**:

| | brief | measured |
|---|---|---|
| projection rows | 170 | **170** |
| clubs | 20 | **20** |
| FPL matched | 170 | **170 (100%)** |
| available | 97 | **97** |
| doubtful | 8 | **8** |
| injured / unavailable | 65 | **65** (33 injured + 30 unavailable + 2 suspended) |
| `p_start == 0` | 66 | **66** |

### Where the brief was wrong

**§19 said the projection log was empty.** It was not: **8,366 rows across 31 days**
(2026-08-27 → 09-26). The real defect was quieter and much worse — `gw` was non-null on **zero**
of them. `fantasy_log.append()` takes `gw` as a keyword defaulting to `None` and the caller
passed none, so every row was written with `""`.

A projection is only a forecast if you know *which* gameweek it forecast. Without that tag there
is no key to join it to the points actually scored, so settling forecasts — and therefore
answering *"is Wowza better than FPL's own ep_next?"* — was impossible however long the log ran.

**§1.a: no non-PL players were found.** All 170 are current FPL players across the correct 20
clubs, and `fantasy.py` already drops non-matches (`pl = pl[pl["fpl_matched"]]`).

Two false alarms were raised during this audit and both were the auditor's fault, not the data's:
Coventry was called a non-PL club (it is in this season's FPL universe), and E. Le Fée was called
absent (he is in Sunderland's squad with 438 minutes — the matcher split the two-word surname
"Le Fée" and compared against "fée"). **Ad-hoc name matching is exactly what invariant 11 warns
about for clubs; it applies equally to players.**

The genuine §1.a risk is different: a **stale** FPL snapshot keeps a departed player alive,
because the squad filter is only as current as the snapshot behind it.

---

## 2. Decision-logic problems found

### The board ranked players who could not play

`overall_rank` sorted by `fantasy_pts` — **conditional** points, what a player scores *if* he
starts. Consequences on the 2026-09-27 board:

- **8 of the top 20 could not play.** Ekitiké #2 and Romero #3, both `p_start = 0.00`;
  de Ligt #11; Richarlison #12.
- **65 of 170** players with `p_start = 0` outranked someone fully available.

The correct number already existed and nothing used it: `xpts_rot` is exactly
`fantasy_pts × p_start`. This was a change of *which column sorts*, not new modelling.

### Captaincy was not a decision

It excluded only `injured`, so doubtful players were suggested. Both picks that day were
**João Pedro** and **Pedro Porro**, each at a 75% chance of playing — precisely the example the
brief gave. A captain is doubled, so a blank costs twice.

### The page silently overrode the CSV

`pages/13_⚽_Fantasy.py` recomputed its own ranking and captain picks from conditional points,
so fixing `fantasy.py` alone would have changed nothing on screen. One shared rule now serves
both.

### "Dream XI" was not a legal FPL team

`best_xi()`'s own docstring admitted it: *"budget is reported (not hard-constrained)"*. There
was **no per-club limit at all**. It landed at £60.4m with ≤3 per club purely by luck, while
starting two doubtful players and a keeper with a 30% start probability.

### Transfers ignored everything that decides a transfer

`transfer_suggestions()` ranked by conditional points, checked no budget, had no horizon, and
never mentioned the 4-point hit.

---

## 3. What changed

### Data correctness (Phase A)

| | |
|---|---|
| `fpl_api.gameweek_context()` | resolves the target gameweek from the bootstrap and stamps it on every logged projection |
| `output/fantasy_health.json` | feed ages, coverage, availability split, whether settlement is possible |
| freshness banner | CURRENT → AGING → STALE on the page |

The gameweek target is **`is_next`, not `is_current`**: once a deadline passes the current
gameweek is locked, so a projection made now is advice for the next one. Season-complete is
reported explicitly rather than silently tagging rows with the last gameweek.

**Ages never come from file mtime.** `git checkout` resets mtime every CI run, a trap this
estate has hit three times. They come from the recorded-fetch sidecar, and an unknown age is
treated as infinitely old, never as fresh.

### Decision logic (Phase B)

**`rank_and_captains()`** — ranks by unconditional points; captains require a status that is not
out-or-doubtful **and** `p_start ≥ 0.60`. Adds `start_confidence`: Nailed / Likely starter /
Rotation risk / Doubtful / Unavailable, derived from data rather than an uncalibrated 0–100
score. Current split: 49 / 24 / 23 / 8 / 66.

*Result: 0 of the top 20 cannot play (was 8). All three captain picks are Nailed at 100% start.*

**`fantasy_optimizer.optimal_squad()`** — a real MILP (scipy/HiGHS, ~0.03s). 15 players, 2-5-5-3,
legal XI split, ≤£100m, ≤3 per club, maximising XI points + captain again + a small bench weight.

Budget and the three-per-club rule are **coupling** constraints — whether a £12m forward belongs
depends on what the other fourteen cost — which no per-position ranking can see.

**`fantasy_transfers.transfer_options()`** — gain next gameweek and over each player's own
fixture window, price delta, affordability, club legality, and the hit maths:
`net = gain − 4 × hits`.

### Visual (Phase C)

The XI is drawn as a **pitch**, keeper at the bottom, rows centred whatever the formation, with
captain and vice marked and a minutes dot from start probability. A lineup is a shape and a table
cannot show one.

### Performance (Phase D)

**`fantasy_settle.py`** reconstructs a settled history from data already on disk — see §4.

---

## 4. Validation methodology

```
projection = last snapshot strictly BEFORE the deadline   (what you could have acted on)
actual     = cumulative FPL points at the NEXT deadline − cumulative before
```

Bounding by the **next** deadline is exact rather than heuristic: by then the gameweek is fully
scored including bonus, and no following-gameweek points can have crept in.

### Two ways the actuals came out wrong first

Both produced confident, plausible-looking numbers:

1. Looking for where the cumulative total "stopped moving" picked the **earliest** post-deadline
   snapshot — `diff()` makes row one NaN and `fillna(0)` reads as no-movement. Actuals came out
   near zero and the model appeared to over-predict by 4 points a week.
2. `fpl_total_points` contains a **season rollover on 2026-09-10** (Khusanov reads 67 then 9; a
   cumulative total cannot fall). Windows spanning it gave GW3 a mean of **−73** points per
   player. Averaged in, this produced a believable *"FPL beats Wowza by 2.4 MAE"* that was mostly
   artefact.

Both now **reject a gameweek with a stated reason** rather than estimating through it.

### The first real result — GW4 and GW5, 575 player-gameweeks

| | MAE | RMSE | Spearman | bias |
|---|---|---|---|---|
| Wowza (unconditional) | 2.164 | 2.966 | 0.543 | +0.99 |
| Wowza (conditional) | 3.430 | 3.867 | 0.195 | +2.44 |
| **FPL `ep_next`** | **1.535** | **2.588** | **0.705** | −0.12 |

**FPL's free number is still better**, and that line stays on the page until it flips. A model
that cannot beat the number the platform gives away is not earning its keep.

The two Wowza rows matter as much: the unconditional number **halves the error** and **triples
rank correlation**. 46% of real gameweek scores are zero because the player did not play, and a
conditional projection never claimed to predict those. Both are reported so the choice is not a
silent one in our own favour.

Not settleable, each with its reason: **GW1** no snapshot on one side · **GW2** every player on
exactly zero · **GW3** spans the rollover.

---

## 5. New fields and artifacts

| artifact | holds |
|---|---|
| `output/fantasy_health.json` | feed ages, coverage, availability, settlement readiness |
| `output/fantasy_ledger.csv` | one row per (gameweek, player): forecast, `fpl_ep_next`, actual |
| `output/fantasy_performance.json` | MAE / RMSE / Spearman / bias, Wowza vs FPL, per gameweek and overall |

New columns: `xpts_uncond`, `start_confidence`, and a populated `gw` on every new log row.

All three are produced by `fantasy_refresh.yml` (`continue-on-error`, so a failure here can
never cost the refresh that already succeeded) and committed per-file.

---

## 6. Remaining limitations

**Honest headline: Wowza does not yet beat FPL's free number.** Until it does, the product's
value is the decision surface — legality, availability, hit maths — not forecast superiority.

- **Only two settled gameweeks.** 575 player-gameweeks is enough to see a gap of 0.63 MAE but
  not to track a trend. The rollover cost GW2 and GW3.
- **Transfers are not on the page.** The engine is built and tested; wiring needs the live FPL
  entry API, which the development machine cannot reach (`CERTIFICATE_VERIFY_FAILED`, local TLS
  inspection — not an FPL outage, CI is fine).
- **Fixture adjustment is still unfitted.** The FDR step (~0.10) and home edge (~0.05) remain
  conventional constants. §18 asks for them to be fitted chronologically once a real projection
  → actual history exists. That history now exists and is growing.
- **Not built:** attack/defence split fixture difficulty (§17), the uncertainty and ceiling
  distribution (§34), an improved minutes model (§35), clean-sheet and bonus models trained
  against real BPS (§37, §38), effective ownership (§42), and the chip advisor's incremental EV
  (§44).
- **`best_xi()` is deliberately unchanged** where it picks an XI from a user's own 15: that
  squad is already legal, so choosing who starts from it is a different and legitimate problem.

---

## 7. What would prove this worked

One question, and the ledger now answers it every week:

> Does Wowza's pre-deadline projection beat FPL's `ep_next` on the same players over the same
> gameweeks?

Today: **no**, by 0.63 MAE. The number is on the page, it updates itself, and it will say so
until it changes.
