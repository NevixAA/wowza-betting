# League × market × model tier-gate study

Date **2026-10-05**. HEAD `wowza-betting 2c2380af`, `wowzaV9-Pro ef8c7b9`, `wowza_v11 6d3d699`.

**`PRODUCTION_SELECTION_CHANGED = NO`.** No threshold, stake, tier, notifier or betting
behaviour was modified. Everything below is measurement and infrastructure.

Regime `0fc4f3056ea1` — the hash of every input that determines a gate. Results from a different
regime are not comparable to these.

---

## A. What gates are actually running today

**Not what `config.py` says.** `src/betting.py` prefers an **approved** value from
`models/best_params_standard.json`, and `pipeline.py::_generate_side_bets` runs a separate
ladder entirely. The resolution order, reproduced in `src/gate_resolver.py` and asserted against
production's own `_base_tier` at every boundary:

```
main O/U   approved best_params_standard  ->  LEAGUE_SNIPER_THRESHOLDS (capped)
                                          ->  side-specific global
           EDGE_CEILING demotes to MARKSMAN only when NO per-league value was found
           VALUABLE = config.VALUABLE_THRESHOLD

side mkts  approved best_params_side_markets[target]  ->  fixed 0.10 / 0.08
           VALUABLE is HARD-CODED 0.04 and ignores VALUABLE_THRESHOLD
```

Under the production environment (`LEAGUE_SNIPER_CAP=0.12`, `MARKSMAN_THRESHOLD=0.08`,
`VALUABLE_THRESHOLD=0.03`):

| league | market | VAL | MM | SNIP | source |
|---|---|---|---|---|---|
| Championship | ou25 | 0.03 | **0.05** | **0.07** | approved optimiser |
| Serie B | ou25 | 0.03 | **0.10** | **0.12** | approved optimiser |
| all other ou25 | ou25 | 0.03 | 0.08 | 0.12 | global / capped |
| Bundesliga 2 | ou25 | 0.03 | **0.12** | **0.12** | capped — **MARKSMAN disabled** |
| League Two | ou25 | 0.03 | **0.12** | **0.12** | capped — **MARKSMAN disabled** |
| Serie B | btts | 0.04 | 0.07 | 0.09 | approved optimiser |
| Serie B | over15 | 0.04 | 0.05 | 0.07 | approved optimiser |
| Championship | btts | 0.04 | 0.10 | 0.12 | approved optimiser |
| all new-format side markets | btts/over15/over35 | 0.04 | 0.08 | 0.10 | fixed global |

Full 84-row table in `output/league_market_gate_research.csv`.

**Three defects this exposed:**

1. **The published study of 2026-10-05 was wrong.** It read `current_threshold` from `config.py`
   and reported Championship at 0.15 and Serie B at 0.15. Production runs **0.07** and **0.12**.
   Every "current" value for an approved league was wrong. Fixed by routing all research through
   one resolver.
2. **`LEAGUE_SNIPER_CAP=0.12` flattens the whole calibrated table.** Eight hand-calibrated
   thresholds (0.14–0.25) all collapse to 0.12. Distinct effective O/U SNIPER gates across the
   estate: **two** (0.07 Championship, 0.12 everything else).
3. **Bundesliga 2 and League Two have MARKSMAN == SNIPER**, so the MARKSMAN tier cannot fire in
   those leagues. Nothing reports this; it is visible only by resolving both gates together.

---

## B–E. Evidence by cell

57 cells, 1,653 hypotheses examined (cells × threshold grid). Windows: fit 50% / validation 20%
/ holdout 30% of each cell's backtest by date, then live settled bets since 2026-08-10.

| grade | n | meaning |
|---|---|---|
| **EDGE_NOT_RANKING_OUTCOMES** | **17** | ROI *falls* as the threshold rises |
| NO_DATA | 13 | no backtest and too few live bets |
| DISCOVERY | 10 | fitted; no OOS window has enough bets |
| INSUFFICIENT | 9 | |
| CANDIDATE | 7 | survives ≥1 OOS window; live CI spans zero |
| **FORWARD_TEST** | **1** | positive live with a CI excluding zero |
| **CONFIRMED** | **0** | unreachable by retrospective code, by construction |

### C. Cells where claimed edge does NOT rank outcomes — 17

`USA MLS` (104 live bets), `Championship`, `League Two`, `Bundesliga 2`, `Norway`, `Mexico`,
`Brazil`, `China`, `Denmark`, `Finland`, `Sweden` on O/U; `La Liga 2`, `League One`,
`League Two`, `Ligue 2` on side markets.

In these, raising the bar makes results **worse**. No threshold repairs an edge measure that
does not rank, so the correct answer is **not a different number** — it is that the gate is not
the lever. This is the single largest group in the study.

### D. Candidates

| cell | fit | validation | holdout | live | note |
|---|---|---|---|---|---|
| **Serie B · btts · standard** | +0.139 | **+0.077** | **+0.072** | n=29 | **the only cell positive in all three backtest windows** |
| Serie B · ou25 · standard | +0.201 | +0.276 | −0.061 | n=36 | strong fit and validation, fails holdout |
| Championship · over15 | +0.066 | — | +0.086 | — | |
| Championship · btts | +0.090 | −0.097 | +0.063 | n=12 | |
| Ireland · ou25 | +0.038 | +0.361 | −0.004 | n=17 | validation driven by a thin window |
| Austrian Bundesliga · ou25 | +0.237 | −0.143 | +0.039 | n=21 | |
| Argentina · over15 | — | — | — | +0.039, CI [−0.11, +0.19] | 74% of profit from one month |

### E. Do not promote

Every `EDGE_NOT_RANKING_OUTCOMES` cell, and `Argentina · over15` — its CI spans zero and
removing its top five winners turns ROI **negative** (+0.039 → −0.016).

---

## The one cell that cleared everything: Argentina · BTTS · new_format

```
live n = 54 over 21 matchdays
ROI    +0.4029   CI [+0.1534, +0.6627]   excludes zero
hit    61.1%     break-even 44.1%        excess +17.0pp
BH-FDR q=0.10 across all cells with a live CI:  p=0.0019  SURVIVES
robustness: ROI excluding the top 5 winners still +0.285
```

**Two warnings that belong next to that number, not in a footnote:**

- **CLV is NEGATIVE: −2.04%**, CI [−4.94, +0.11] on 51 graded bets. The market moved *against*
  these selections while they won. Beating the result while losing to the close is the signature
  of variance rather than pricing edge — and it is exactly what preregistered hypothesis
  **H-BTTS-03** was written to test.
- **59.5% of the profit comes from one month** (2026-09).

So: the strongest cell in the estate, and still not evidence of a durable edge. It is
`FORWARD_TEST`, not `CONFIRMED`, and the preregistered Argentina window (4 of 150 bets as of
2026-10-04) is what decides it.

---

## F–G. Not done in this pass

Stated plainly rather than implied:

- **§5 prospective gate dataset** — the shadow log went live 2026-10-04 and holds ~191 rows. The
  extended per-market schema (book count, dispersion, movement-at-first-sight, production tier
  at first sight) is **not built**. Until it is, "what if I lowered the bar?" is unanswerable for
  any cell, which is why every candidate below a production gate is reported `LEDGER_CENSORED`.
- **§9 movement as a tier upgrade** — not tested per cell in this pass. Prior estate measurement
  stands unrefuted: `Confirmed` −10.8% vs `Neutral` +8.0%, i.e. the rule points the wrong way.
- **§10 Pro / V11 cross-check** — not performed. V11 artifacts are 13 days stale.
- **§12 BTTS leave-one-league-out** — partial (Argentina isolated; no LOLO sweep).
- **§19 dashboard** — not built.

---

## H. Best current scientific answer per cell

For **56 of 57 cells: `UNKNOWN`, `NO_SIGNAL` or `INSUFFICIENT`.** No numeric threshold is
manufactured where the evidence does not support one.

For **Argentina · BTTS · new_format**: the production gate (0.04 / 0.08 / 0.10) is the only one
with live evidence behind it, and that evidence is one month old, negative on CLV, and awaiting
its preregistered window.

**Nothing has earned the right to be called SNIPER on evidence.** The operational label and a
scientifically confirmed signal remain different things, exactly as §4 anticipated.
