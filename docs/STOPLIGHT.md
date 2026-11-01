# Data Truth stoplight

Owner rule, 2026-11-01: **we can't go back from 100% data. Only keep collecting true and good data.**

## The milestone

GitHub milestone **Data Truth 100%** (NevixAA/wowza-betting) holds every known remaining data-truth
gap. Each gap is an issue labelled `data-truth`.

| Stoplight | Meaning |
|---|---|
| 🔴 RED: milestone open | No upgrades or new ideas. Work goes only to closing milestone issues. A PR labelled `upgrade` fails the `upgrades-wait` check. |
| 🟢 GREEN: milestone closed | Upgrades may proceed, one at a time, each passing the ratchet below. |

New ideas are recorded as issues labelled `upgrade` and **without** the milestone, so nothing is
lost while the light is red.

## The ratchet (always on, also after the milestone closes)

`.github/workflows/data_truth_gate.yml` runs on every code or config push to `main` and on every
PR. The commit is RED (the run fails) when any of these is true:

1. **Naming (rule 1) is RED.** A league value in a board, ledger or history bank is not a registry
   league.
2. **Defaults (rule 2) is RED.** A new invented default (`fillna(<non-zero literal>)`, a literal
   season fallback or season list, a pinned football-data folder) appears in production code.
3. **League relations (rule 4) is RED.** There are misfiled league rows, the registry is invalid,
   or joins are not league-keyed.
4. **The known-defaults baseline grows.** `config/data_truth_baseline.json` holds more entries than
   `max_known_defaults` in `config/data_truth_ratchet.json`.
5. **The ceiling is raised.** `max_known_defaults` is higher than in the previous commit.

The ceiling only goes down. When you remove a baselined default, run
`python -m src.integrity.data_truth --baseline` and lower `max_known_defaults` in the same commit.
The milestone target is **0**.

Rule 3 (missing data: prices, serving features, fresh banks) is operational: an API outage must
not block code. It turns system health RED daily via `evidence_refresh.yml` and is watched there.

## Local commands

    python -m src.integrity.data_truth           # the four checks -> output/data_truth_health.json
    python -m src.integrity.data_truth --gate    # the stoplight, exactly as CI runs it
