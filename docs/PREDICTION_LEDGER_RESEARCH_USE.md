# How research uses the prediction ledger: the forward cross-model study (spec only)

**Scope.** This is a specification. No study runs and nothing is trained by this document or the code that ships with it. The rules in memory still apply: no training on unclean data, and no refit until everything is wired.

**Study.** COMBINED-MODELS-DEV-2, branch `research/combined-models-clean`.

**Inputs.** The ledger built by `src/prediction_ledger.py`:
- raw: `output/prediction_ledger/<repo>/<model_id>/<market>/<league_slug>/<season>.csv`
- settled view: `output/prediction_ledger_settled/...`
- registry: `output/prediction_ledger/registry.json`

Pro's ledger lives in NevixAA/wowzaV9-Pro under the same layout.

## 1. Question

For each league and each model × market (the **target**):
- Does any other model × market (a **helper**) add out-of-sample information to the target's own probability?
- Is a TOP-3 bar on the target too strict or too loose, judged on **every** fixture rather than only on the ones that were tipped?

Every active model × market of `config/prediction_ledger_markets.json` is both a target and a helper. Examples:
- v9 O1.5 helps v9 O2.5.
- Pro 1X2 helps BTTS.
- A player SoT model helps team O2.5 through the sum over the lineup.
- Correct score helps the extreme-goals markets.

## 2. Why the ledger makes this out-of-sample by construction

- **Every row is written before kickoff.** The writer refuses a row at or after kickoff, and Data truth 12 fails a file that holds one. Every row also carries `predicted_at` and `minutes_to_kickoff`.
- **First write wins.** A stored opinion is never revised. A new model version writes new rows under its own `model_version` and never touches the old ones.
- **A month's study only uses rows that existed before that month's pre-registration was committed** (section 4). Outcomes arrive later, through the settled view. That view is rebuilt from the history banks and joined 1:1 on the ledger key; the raw file is only read.
- **Helpers are taken at the same snapshot as the target, or earlier**, on the same fixture (`fixture_ref`). A helper written later than the target snapshot is never used. There are two snapshots: `first_sight`, the TOP-3 freeze convention, and `final`, the first run inside the last 120 minutes.

## 3. Unit of analysis

**One study cell** = (league_slug, target model_id × market × selection, snapshot, model_version).

- **Leagues are never pooled** (owner: every league its own bars, never global).
- A cell with fewer settled rows than its pre-registered minimum is reported as `INSUFFICIENT`. It is never extrapolated from other leagues.

**Join rules:**
- **v9 team ↔ Pro:** `fixture_ref` = `oa:<OddsAPI event id>`. Pro maps its own fixture_key to v9's board id. A fixture the board dropped falls back to the shared natural key `nk:<league>|<date>|<home>|<away>`.
- **Props ↔ team:**
  - Props rows carry `af:<API-Football fixture id>`.
  - They map to team rows through the API-Football fixture index (`output/collect/upcoming_fixtures.parquet`, on main from the go-live merge) on (league id, kickoff, team ids). This is an exact-id join only.
  - A fixture that does not map is excluded and counted, never matched on names.

## 4. Pre-registration (monthly)

On the **first Monday of each month**, before any settled row of the coming month exists, commit `research/combined_models/prereg/<YYYY-MM>.json` on the research branch. It holds:

1. **The cells under study.** League × target × selection × snapshot, plus the minimum n per cell. The default minimum is 150 settled rows with ≥30 events per side.
2. **The helper set for each target**, frozen. A helper may only be a model × market that already has rows for the same league and snapshot.
3. **The method, frozen for the month.** It is fixed by section 5. Any change takes effect the next month and gets a new prereg file.
4. **The decision rule** in section 6, verbatim.
5. **The sha256 of each raw partition used**, and the git commit. The study reads exactly those bytes, and a re-run reproduces them bit for bit.

**The evaluation window** is the calendar month after the prereg commit. Only rows whose `predicted_at` falls in that window **and** whose fixture settled by the evaluation date are used. Rows predicted earlier serve only as the rolling training window of the combiner (section 5).

## 5. Method (frozen per month)

**Baseline.** The target's own probability, scored by:

| Market | Score |
|---|---|
| Binary | Log loss and Brier |
| 1X2 | RPS |
| Correct score | Log score |

**Combiner.**
- **Form:** a per-league logistic stack: logit(target p) plus logit(helper p) for each helper.
- **Regularization:** L2, with a fixed λ from the prereg.
- **Training:** fit **only** on rows settled before the window opens. Use a rolling window of the previous 3 months, or everything available while the ledger is younger than that.
- **No tuning inside the window.**
- **Multi-class targets:** 1X2 and correct score use the multinomial analogue.

**Helpers that are aggregates:**
- A props helper is aggregated per team **before** joining:
  - expected SoT = Σ over the lineup of P(player SoT ≥ 1);
  - expected scorers = Σ P(goal ≥ 1).
- Only players with a `final` snapshot row count, so the confirmed-lineup information is used. A `first_sight` study uses `first_sight` props.

**Missing values stay missing.** If a helper's value is missing on a fixture, that fixture is dropped from that helper's comparison and counted. It is never imputed.

**Bars check (the owner's selection-bias question).** For each league × target, compute calibration (slope and intercept) and hit rate by probability decile on **all** settled rows. Show the TOP-3 bar band against the deciles above and below it. The question it answers: "do non-tipped fixtures in the band below the bar hit better than the tipped ones?"

## 6. Decision rule (pre-registered, per cell)

A helper is **PROMISING** for a cell when both hold:
- The combiner's mean log loss beats the baseline on the window.
- The match-date block-bootstrap 95% CI of the difference is entirely below 0, with n ≥ the cell minimum.

A helper is **PROVEN** only after **two consecutive monthly windows** are PROMISING on disjoint fixtures, with the same frozen method. This is the strict PROVEN rule.

**Output:** a proposal only. Nothing promotes automatically and nothing changes a bar, a tip or a stake. A PROVEN cell becomes an owner decision, recorded in `docs/` with its prereg files.

## 7. Outputs (research branch only, never main)

| File | Content |
|---|---|
| `research/combined_models/results/<YYYY-MM>/cells.csv` | One row per cell: n, events, baseline score, combined score, delta, CI, verdict |
| `research/combined_models/results/<YYYY-MM>/bars.csv` | Calibration and decile hit rates per league × target (the bars check) |
| `research/combined_models/results/<YYYY-MM>/REPORT.md` | Plain-language summary; every number cites its cell |

## 8. Guards the study inherits from the ledger

- **Data truth 12 is green on the commit used**: no duplicate key, every row in its own model/market/league/season partition, no unknown league, every snapshot strictly before kickoff, no price column.
- **Registry counts** (`registry.json`: rows, predicted, settled per model × market × league) are copied into the prereg so the study's n can be audited against the source.
- **Prices never enter.** The ledger stores none. A market-anchored helper (closing price) comes from the private store only, under its own entry and licence, and never from this ledger.
- **Pre-/post-go-live rows separate by `model_version`.** The Clean Record (from 2026-11-21) studies only the clean-retrain versions.

## 9. Cadence

| When | What runs |
|---|---|
| Monthly | Prereg on the first Monday; evaluation after the window's last fixture settles (the settled view is rebuilt by `update_results.yml`). The report is due by the 10th of the next month |
| Ad hoc | Only on the owner's request, using the latest frozen prereg, and labelled exploratory (never PROMISING or PROVEN) |
