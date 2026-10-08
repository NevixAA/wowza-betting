# Wowza v9 — Football Betting Intelligence System

**Status:** Production — fully automated via GitHub Actions
**Live since:** April 2026
**Last doc update:** 2026-10-08

> **2026-10-08 update.**
>
> * **Player props named players at clubs they had left.** Benzema was tipped for Real Madrid
>   (last appearance 2023), Verratti for PSG, Brozović and Taremi for Inter — 400 of 2,581 board
>   rows. A player is now tipped only if he is in his club's live squad list, or (where no list
>   exists) appeared within `STALE_DAYS` = 120. Separately, Sporting CP players were tipped in
>   Cádiz v Sporting Gijón because a club inherited every competition its signings had played in;
>   a club's competitions now come from its own rows (`player_model/predict.py`).
> * **Ligue 2 was the wrong API-Football league.** `API_FOOTBALL_IDS["Ligue 2"]` was 65 (the
>   defunct Coupe de la Ligue); it is 62. Ligue 2 had no per-book quotes and no side-market odds
>   from 2026-05-09 to 2026-10-08. That gap is permanent — odds cannot be backfilled.
> * **New research, in Pro** (`wowzaV9-Pro/output/studies/REPORT.md`, refreshed twice a week):
>   Argentina BTTS is a **league scoring regime** the market prices slowly (BTTS 38% in 2025 →
>   56% since August; blind YES matched Wowza's picks), not model selection. v9's O/U 2.5
>   probability is **compressed near 52%** — its UNDER "edges" are matches the market rates
>   high-scoring, and the market is right. Adding the model to the market improves no market
>   out of sample. A market-anchored O/U challenger is being recorded forward; **owner decision:
>   no change to v9's O/U probability before ~2026-10-29.**
> * **New repos and pages.** `wowza-exec` (private) holds the Cloudbet execution layer — PAPER_ONLY,
>   fail-closed; v9's own `src/cloudbet/` is dormant and must never run outside DRY. Pro now runs a
>   **League Scout** over ~200 leagues Wowza does not bet, and a **tip scoreboard** for the 1X2 and
>   Bet Builder tips it sends. Dashboard: new 🔭 League Scout page; the 🔬 Pro page gained
>   *Tips sent* and *Studies* tabs.

> **2026-10-07 update.** The previous version of this file was from 2026-08-09 and several of
> its numbers had stopped being true. Corrected below, with the wrong claims named rather than
> quietly replaced:
>
> * **Tier thresholds.** This file said SNIPER was "per-league 14–25%". It is not: production
>   sets `LEAGUE_SNIPER_CAP=0.12`, which flattens every calibrated value to 0.12, and
>   `src/betting.py` prefers an *approved* value from `models/best_params_standard.json` over
>   both. **Never read a threshold from `config.py`** — use
>   `src.gate_resolver.resolve_effective_tier_gates()`.
> * **`MAX_OU_ODDS` does not exist.** This file claimed main O/U odds were "bounded by
>   `MAX_OU_ODDS` to reject stale/fringe prices". There are **zero** occurrences in the
>   codebase. What exists is `MIN_OVER_ODDS = MIN_UNDER_ODDS = 1.75`, a *lower* bound only.
> * **Retrain cadence.** Was "Sun 03:00"; ran daily 2026-09-22 → 2026-10-05 by design and is
>   now weekly again, via a date-expiring guard rather than a cron edit.
> * **Predict cadence.** Was "every 5 min"; it is 15-minute Fri–Sun and 30-minute Mon–Thu. Note
>   that **cron minutes do not survive GitHub's dispatcher** — never rely on a run landing near
>   its slot.
> * **USA MLS is a paper league** (owner decision, evidence in `config.py`): fully collected —
>   predictions, tips, movement, CLV, shadow log — but never sent, never counted in any KPI and
>   never bet. See *Paper leagues* below.

---

## What It Does

A football betting + prediction platform with several independent signal families, all
running automatically in the cloud (no server/PC needed).

| Module | What it does | Money? |
|---|---|---|
| **Standard O/U 2.5** | Over/Under 2.5 for our second-division leagues — SNIPER/MARKSMAN/VALUABLE tiers | **Real-money candidate** — but the 2026-10-08 studies found no evidence the model adds to the market (see update above) |
| **Side Markets** | BTTS, O/U 1.5, O/U 3.5 — per-league walk-forward thresholds | Real-money candidate |
| **HT Model** | Half-time O/U 0.5 and 1.5 | Paper — **no edge** (see below) |
| **New-Format Model** | O/U for goals-only leagues — separate model, never mixed with standard | Paper |
| **Sharp Money Tracker** | Odds-drift detection — STEAM / STRONG / SHARP | Info |
| **Live Scanner** | In-play Poisson signals during match hours | Info/alert |
| **Player Props** | 7-market ML ensemble | **Paper only — no betting edge** |
| **Fantasy (FPL)** | FPL point projections | Prediction product |
| **Health Monitor** | Alerts if predict fetches 0 fixtures / all leagues error | Reliability |

### Two closed questions — do not re-open without new evidence

* **Player props have no betting edge.** The model is accurate (AUC 0.62–0.85, calibrated) and
  the market matches or beats it. Confirmed across 8 tests on the bug-fixed model. Accuracy is
  monetised through **Fantasy**, where there is no vig.
* **The HT models have no edge.** 74% of HT training labels were fabricated; on honest labels
  AUC is 0.512. A deep-feature rebuild doubled measured skill and still lost to the price.

---

## Signal Tiers

**The effective gate is not in `config.py`.** Resolution order, as `src/betting.py` actually
implements it:

```
main O/U   approved best_params_standard  →  LEAGUE_SNIPER_THRESHOLDS (capped)
                                         →  side-specific global
           EDGE_CEILING demotes to MARKSMAN only when NO per-league value was found
           VALUABLE = config.VALUABLE_THRESHOLD

side mkts  approved best_params_side_markets[target]  →  fixed 0.10 / 0.08
           VALUABLE is HARD-CODED 0.04 and ignores VALUABLE_THRESHOLD
```

Under the production environment (`LEAGUE_SNIPER_CAP=0.12`, `MARKSMAN_THRESHOLD=0.08`,
`VALUABLE_THRESHOLD=0.03`):

| league | market | VAL | MM | SNIP | source |
|---|---|---|---|---|---|
| Serie B | ou25 | 0.03 | 0.10 | 0.12 | approved optimiser |
| League Two | ou25 | 0.03 | 0.11 | 0.13 | approved optimiser |
| all other ou25 | ou25 | 0.03 | 0.08 | 0.12 | global / capped |
| new-format side markets | btts/o15/o35 | 0.04 | 0.08 | 0.10 | fixed global |

**VALUABLE is a collection layer, not money.** It is half-stake/monitor and must never be
pooled into a P/L headline — doing so overstated the standard track sevenfold until 2026-10-06.
Real-money evaluation is SNIPER + MARKSMAN only.

Regenerate the live table with:

```bash
LEAGUE_SNIPER_CAP=0.12 MARKSMAN_THRESHOLD=0.08 VALUABLE_THRESHOLD=0.03 \
  python -c "import src.gate_resolver as g; [print(x) for x in g.all_gates()]"
```

---

## Architecture

```
pipeline.py                ← team models: train / predict / backtest
retrain.py                 ← full download + retrain, and (since 2026-10-06) threshold refit
update_results.py          ← fill WIN/LOSS/PnL from football-data.co.uk
config.py                  ← paths, leagues, threshold TABLES (not the effective gates)

src/
  data_loader.py           ← Excel workbook + CSV + CI web download
  feature_engineering.py   ← rolling form, HT, home advantage, per-format isolation
  model.py                 ← LogReg + GradientBoosting + Platt calibration
  predict.py               ← fixtures → model → tiers  (PRE-MATCH ONLY)
  betting.py               ← 3-tier signal logic
  backtest.py              ← walk-forward backtest + threshold optimiser
  gate_resolver.py         ← THE effective gate per league × market × model  ← read this
  promotion_gate.py        ← champion/challenger: 10 checks, block bootstrap, canary
  model_validation.py      ← leak-free FIT | CAL | META | HOLDOUT
  shadow_compare.py        ← prospective per-fixture log + BH-FDR
  book_quotes.py           ← per-bookmaker quotes on a kickoff ladder
  clv_types.py             ← same-book / sharp-reference / consensus CLV, kept apart
  market_anchor.py         ← exchange → cross-book median → single book → INSUFFICIENT
  allocation.py            ← bankroll fraction per cell + why losing cells lose
  cloudbet/                ← automated execution (DRY by default — cannot bet)
  sharp_tracker.py · live_scanner.py · health_check.py

player_model/              ← 7 prop markets, 139 features, Fantasy projections
telegram_bot/notifier.py   ← all Telegram sends
pages/                     ← Streamlit dashboard (14 pages, incl. 🔭 League Scout)
tests/                     ← 306 tests
```

---

## Research + promotion infrastructure (added Sept–Oct 2026)

The part that decides whether a model or a threshold is allowed to change.

| Piece | What it guarantees |
|---|---|
| **No-harm promotion rule** | A retrained model that scores *worse* on the incumbent's own holdout no longer replaces it. Replayed over 91 historical decisions, the old tolerance promoted 88 and **29 were measurably worse**. Revert with `TRAIN_BLOCK_WORSE_CANDIDATE=0`. |
| **Strict gate, log-only** | The 10-check gate records what it *would* decide (`v91_gate` in `retrain_log.json`). Not enforced: on the same 91 decisions its material floor would have blocked 62 of 88, and there is no evidence that is good. |
| **`gate_resolver`** | One place that answers "what gate is really running", asserted against production's own `_base_tier` at every boundary. Built because a study read `config.py` and published wrong thresholds for every approved league. |
| **Prospective shadow log** | Every scored fixture, frozen at first sight, with the gate in force. The **only** thing that can ever answer "what if the bar were lower" — the ledger holds no fixture below the live floor. Started 2026-10-04; cannot be backfilled. |
| **Preregistered hypotheses** | `registry/preregistered_hypotheses.json`. An entry may not be edited once its confirmation window opens; supersede instead. |
| **Allocation registry** | `scripts/allocation_registry.py` — quarter-Kelly on the *lower bound* of a matchday-block CI. A cell whose interval includes zero gets exactly 0%. |

### Rules these encode

1. **Never fit and score on the same rows.** Windows are chronological, never random.
2. **A closing price may grade a signal, never select one.**
3. **Block-bootstrap by matchday.** Twenty bets on one Saturday are not twenty observations.
4. **Break-even is `mean(1/odds)`, never `1/mean(odds)`.** The wrong one flattered SNIPER by 1.5pp.
5. **Never pool** markets, model types or tiers. Pooling has reversed the sign of a headline
   three separate times.
6. **Retrospective code can never award `CONFIRMED`.** That needs a preregistered forward window.

---

## Automation (22 workflows)

| Workflow | Schedule (UTC) | Does |
|---|---|---|
| **predict** | 15-min Fri–Sun, 30-min Mon–Thu | O/U 2.5 + side markets → Telegram. Commits each file individually so one missing output can't abort the commit |
| **player_props** | 2-hourly at weekends | props predict / club retrain |
| **live_scanner** | ~10-min during match hours | in-play Poisson signals |
| **sharp_tracker** | every 2 h | drift signals → Telegram |
| **update_results** | every 2 h | WIN/LOSS + closing odds → CLV |
| **std/nf odds capture** | hourly + near-kickoff loop | forward odds curve, **per-bookmaker** |
| **daily_summary** | five slots 01–09, sends once | Telegram digest, deduped on `DIGEST\|<date>` |
| **retrain** | Sundays (daily 09-22→10-05, date-expiring guard) | retrain + **threshold refit** |
| **backtest_matrix** | monthly, 1st | walk-forward per model + threshold optimiser |

**Two scheduling facts that invalidate a whole class of design:**

* **Cron minutes do not survive the dispatcher.** `predict` asks for specific minutes and starts
  spread across all 60. Never make anything depend on a run landing near its slot — use an
  in-run adaptive loop instead.
* **Delivery and punctuality trade off.** A once-daily workflow runs ~100% of the time but
  **4–5 hours late**. High-frequency schedules land ~13% of requested slots. Make a
  time-sensitive job idempotent, then over-schedule it.

---

## Reliability / monitoring

- **Pre-match filter** — already-started matches are skipped; live odds on an in-play match
  would otherwise manufacture a false SNIPER.
- **Outage health alert** — pings Telegram if every league errors or the system goes stale.
- **Team-name matching** — league-scoped identity tokens via `src/team_names.resolve`; refuses
  ambiguous matches rather than guessing. A naive prefix match once left 46% of standard
  fixtures with no form data.
- **No form data → no bet.** `evaluate_value` forces AVOID when rolling-form features are
  missing, because median imputation turns a blind fixture into a *confident* wrong edge.
- **Odds bounds.** `MIN_OVER_ODDS = MIN_UNDER_ODDS = 1.75`, a **lower** bound only. There is no
  upper bound on main O/U. Measured cost: odds > 3.0 are 6.1% of settled bets and 35.5% of the
  net loss, all of it new-format. Adding a bound is a live selection change and needs its own
  evidence.
- **Enrichment must be cached.** Any workflow running `WOWZA_FULL_ENRICH=1` needs an
  `actions/cache` step; a test fails CI otherwise. Without it jobs start cold, refetch the whole
  history and die on their timeout — which silently froze every per-league threshold for five
  weeks in Sept–Oct 2026.

---

## Leagues

19 in `ENABLED_LEAGUES` (bet + tip), from 17 standard-format and 15 new-format in the training
pools. **`STANDARD_FORMAT_LEAGUES` is deliberately a superset of `ENABLED_LEAGUES`** — several
leagues stay in training because they improve the model while being excluded from prediction on
measured negative-ROI grounds, each with its evidence in a comment beside it.

### Standard O/U — bet
Championship · League One · League Two · Bundesliga 2 · La Liga 2 · Serie B · Ligue 2

### New-Format O/U (goals only) — bet
Denmark · Austria · Sweden · Norway · Finland · Ireland · Argentina · Brazil · Japan ·
Mexico · China · Romania

**Training-only** (in the training pools, not predicted): Saudi Pro League, K-League 1, and the
training-only standard leagues (Dutch, Portuguese, Greek, Turkish, Belgian, Scottish, National
League).

### Paper leagues — collect everything, send nothing, count nothing, bet nothing

`config.PAPER_LEAGUES` (currently **USA MLS**). A paper league stays in `ENABLED_LEAGUES` so the
whole collection machinery keeps running for it — predictions, the prospective shadow log, odds
movement and drift, per-book quotes, sharp tracking, the live scanner, settlement and CLV. Its
tips are generated and written to the ledgers like any other league's. It never reaches Telegram
(stripped from `bets.csv` / `side_bets.csv` and filtered from every notifier read), never counts
in a KPI (digest, weekly summary, every dashboard page), and never reaches the Cloudbet bot.

**Do not remove a league from `ENABLED_LEAGUES` to stop betting it.** That was tried on
2026-10-06 and silently stopped its sharp and in-play collection as well, because
`sharp_tracker` and `live_scanner` iterate that set. Add it to `PAPER_LEAGUES` instead.

All filtering goes through one function, `config.drop_paper_leagues()`, so the rule cannot be
re-implemented slightly differently somewhere and leak. To upgrade a paper league later, remove
it from the set — its paper-period rows are already ledgered and will start counting from then.

---

## Data Sources

| Source | Used for | Notes |
|---|---|---|
| football-data.co.uk | Historical match data | CI download = last 4 seasons |
| `England_Leagues_4_Seasons_With_Summary.xlsx` | Deep English history | **Local-only, not in git** |
| OddsAPI | Live odds, player props | credit-metered |
| API-Football | Player stats, lineups, injuries, referees, live scores, per-book odds | |

> ⚠️ **CI trains on less data than a local run**, because the deep Excel is local-only. Treat
> per-league backtest return as re-validated each retrain, never as a fixed headline.
>
> ⚠️ **`/odds` is PRE-MATCH ONLY.** Historical odds cannot be backfilled — proved over ~830
> calls, 0 of 3 FT fixtures returned odds in every season 2019–2025. A closing price not
> captured before kickoff is gone permanently. This is why forward capture cadence matters and
> why `scripts/backfill_af_odds.py` must never be run.
>
> ⚠️ **`player_history.parquet` is a DIRECTORY** of one part per season, not a file. Git stores a
> whole new copy of a changed binary; 219 versions took `.git` to 3.4 GB and crossed GitHub's
> 100 MB limit. `pd.read_parquet` reads a directory identically.

---

## Setup

**GitHub secrets:** `ODDS_API_KEY`, `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`, `APIFOOTBALL_KEY`
(+ `CLOUDBET_API_KEY` if using the execution bot). Local secrets live in gitignored
`.env` / `.api_keys` — never committed, never on a command line.

```bash
python pipeline.py --mode predict          # O/U 2.5 + side markets
python retrain.py                          # full team retrain
streamlit run app.py                       # dashboard
python -m pytest tests/ -q                 # 306 tests

# research (none of these touch production)
python scripts/gate_study.py               # evidence per league × market × model
python scripts/allocation_registry.py --diagnose
python scripts/replay_promotions.py        # every retrain decision through both gates
python -m src.cloudbet.bot                 # DRY by default — builds requests, sends nothing
```

Always set the production environment before reasoning about a threshold:
`LEAGUE_SNIPER_CAP=0.12 MARKSMAN_THRESHOLD=0.08 VALUABLE_THRESHOLD=0.03`.

---

## The three repos

| Repo | Role |
|---|---|
| `NevixAA/wowza-betting` | **this one.** Production: predict, tips, notifications, dashboard |
| `NevixAA/wowzaV9-Pro` | canonical evidence store, independent validation, Bet Builder |
| `NevixAA/wowza_v11` | market-first / microstructure / CLV research |
| `NevixAA/wowza-exec` | **private.** Cloudbet execution control: fail-closed modes, execution policy (PAPER_ONLY), market-anchored eligibility, candidate log. Reads v9's output; never writes into it |

What lives where in Pro, for the things v9's dashboard shows:

| Pro path | What |
|---|---|
| `src/scout/` · `output/scout/` | League Scout: odds, results and a baseline model for ~200 leagues; `league_status.json` per league |
| `src/studies/` · `output/studies/REPORT.md` | Argentina BTTS controls, UNDER diagnosis, model-vs-market residual test, per-cell evidence (shrinkage, FDR, sequential CLV), O/U challenger |
| `src/pipelines/tip_scoreboard.py` · `output/TIP_SCOREBOARD.md` | how the 1X2 and Bet Builder tips Pro sends are doing; daily Telegram recap |

Pro, v11 and wowza-exec **read v9's committed output and never write into it.** New ideas go
through Pro and v11 — chronological evidence, a challenger, repeated validation, owner approval
— then a deliberate v9 upgrade.

`v9/config.py` sets `DATA_DIR = BASE_DIR.parent`, so shared historical data lives in the parent
folder. Clone into the same layout or those files will not be found.

**Routine retraining is allowed and expected.** Architectural change — a new target, feature
family, algorithm, league routing, threshold system, staking or notification semantics — needs
explicit owner approval. The authoritative version of this is
`wowzaV9-Pro/docs/WOWZA_SYSTEM_CONTRACT.md`; where it and any doc disagree, the contract wins.

---

## Disclaimer

Informational & entertainment purposes only — **not** financial, betting, legal, or
investment advice. No guarantees; past results do not predict future outcomes. 18+, bet
responsibly, and only wager what you can afford to lose. Full terms: **[DISCLAIMER.md](DISCLAIMER.md)**
(also shown on every dashboard page).
