# Per-league weekly research system (owner brief §34) — verified state, 2026-10-26

Owner principle 2026-10-25: *"Every league generates evidence every week. Not every league generates bets
every week. Do not break a working production strategy."*

This document checks each item of the brief against the code and the committed artifacts, then answers
the acceptance questions A–E from the evidence as generated today. **Research only**: nothing below is a
bet, a threshold, a stake or a production state.

**Evidence base (all read-only):**

| artifact | generated_at | what it is |
|---|---|---|
| `config/league_market_regimes.json` | generated_for 2026-10-25, evidence cutoff 2026-10-20 | regime registry, 100 entries |
| `config/track_registry.json` | generated_for 2026-10-25 | canonical track read-model (same 100 keys) |
| `output/league_weekly_research_summary.json` | 2026-10-26T08:20:54Z | panel summary + evaluation |
| `output/league_market_health.json` | 2026-10-26T08:20:54Z | per league × market coverage |
| `output/weekly_league_report.json` | 2026-10-26T08:20:54Z | latest gameweek top-5 per league (frozen, else provisional) |
| `output/league_weekly_candidates.csv` | rows first seen 2026-10-25T09:25:36Z → 2026-10-26T07:58:36Z | 900 candidate rows |
| `output/league_weekly_research.csv` | first `created_at` 2026-10-25T13:12:19Z | 136 frozen panel rows, all settled |
| `config/research_lock_snapshot.json` | approved_at 2026-10-25T12:09:26Z | non-regression fingerprint |

The modules were also re-run read-only at 2026-10-26T08:41Z (`provisional()`, `league_weekly_eval.*`,
`research_lock.check()` on the stored files; the SHA-256 of all five panel artifacts was identical before
and after, and the panel store's sidecar `verify()` returned `ok: True`).

---

## 1. Tracked leagues

20 leagues: `config/track_registry.json` `entries[].league`, read by
`src/research/league_weekly.py:241` (`tracked_leagues`). The same 20 are in
`config/league_market_regimes.json` `tracked_leagues` (built from `config.ENABLED_LEAGUES`,
`scripts/build_league_regimes.py:11-12`).

Standard (European 2nd divisions): Bundesliga 2, Championship, La Liga 2, League One, League Two,
Ligue 2, Serie B. New-format: Argentina Primera Division, Austrian Bundesliga, Brazil Serie A, China
Super League, Denmark Superliga, Finland Veikkausliiga, Ireland Premier Division, Japan J-League, Mexico
Liga MX, Norway Eliteserien, Romanian Superliga, Sweden Allsvenskan, USA MLS (paper league,
`config/bankroll_policy.json` `paper_leagues`).

Health at 08:20Z: 18 leagues have current-week data, `missing_leagues` = [], `no_fixture_leagues` =
China Super League and Romanian Superliga (`NO_FIXTURES`, nothing expected — distinguished from
`MISSING_ROWS` at `league_weekly.py:668-672`). Expected candidate rows 760 = actual 760.

## 2. Tracked markets

`PANEL_MARKETS`, `league_weekly.py:116-119`:

| panel market | engine family | side | note |
|---|---|---|---|
| OU25 | OU25 | engine's better side (max EV lower bound, ties → OVER) | OVER and UNDER analysed separately |
| BTTS | BTTS | YES (the registry side) | NO side reconstructed exactly when the engine chose it (`side_view`, `:251-305`) |
| OVER15 | OU15 | OVER | |
| OVER35 | OU35 | OVER | **observe-only** (`OBSERVE_ONLY`, `:119`): blocked market, ranked and graded for evidence, never a candidate |

The regime registry has five entries per league (OU25 OVER, OU25 UNDER, BTTS YES, OVER15 OVER, OVER35
OVER; `build_league_regimes.py:13-14`). Sides and markets are never pooled (invariant 1).

## 3. Regime classification (rules REGIME-1)

Code: `scripts/build_league_regimes.py:36-64` (rules text), `:88-93` (constants), `:277-306`
(`classify`). Output: `config/league_market_regimes.json` (`rules.version = "REGIME-1"`).

* Adequate evidence = n_signals ≥ 30 **and** ≥ 8 independent matchdays. Adequate market = ≥ 100
  fixtures with both sides priced over ≥ 8 matchdays.
* CI = 90% matchday-block bootstrap, 2,000 resamples, seed = crc32(entry key).
* **PROTECTED** (checked first): real-money production state, not a paper league, adequate evidence,
  and (ROI CI lower > 0 **or** model-minus-market log-loss CI upper < 0). `owner_locked = true`.
* **REPAIR**: (adequate evidence and ROI CI upper < 0) **or** (adequate market and dLL CI lower > 0,
  i.e. the market clearly beats the model).
* **LEARNING**: everything else, including no evidence. No rule fires on fewer than 8 matchdays, so one
  losing week can never move an entry.

Evidence (docstring `:16-34`): Pro `experiments/valuable_2026_10/all_fixture_settled.parquet` (first
sight, immune to the ledger's in-place tier upgrades), certified CLV (close ≤ 30 min before kickoff),
model vs de-vigged first-sight market on every settled fixture (OU25, BTTS only; OVER15/OVER35 have no
opposite price → `UNAVAILABLE`), ledger cross-check informational only.

Current counts: **PROTECTED 1, REPAIR 6, LEARNING 93**; `repair_priority_high` = [] (none ≥ 0.40).

## 4. Protected combos

| key | production | evidence | reason (registry) |
|---|---|---|---|
| Argentina Primera Division · BTTS · YES · new_format | ACTIVE (track `new_format|BTTS`, base 0.0125, max odds 2.6) | n 81, 35 matchdays, ROI +36.4% CI90 [+15.8%, +56.4%]; certified CLV n 22, mean +0.26%; dLL −0.0547 CI [−0.0901, −0.0195] on 89 fixtures | credible positive evidence: ROI 90% CI lower bound > 0 |

It is the only entry in `research_lock_snapshot.json` `fingerprint.protected_tracks`
(`n_protected_tracks = 1`). Caveat already on record (MEMORY: Upgrade Studies Oct 2026): Pro's study
found the Argentina BTTS profit is league-wide (a BTTS-yes regime), not model selection skill — the
protection protects the production track, it does not claim the model is the reason.

## 5. Repair-priority combos

| key | production | trigger | dLL (model − market) CI90 | repair_priority_score |
|---|---|---|---|---|
| League One · OU25 · OVER · standard | PAPER | market beats model | +0.0303 [+0.0146, +0.0480], 114 fx | 0.265 |
| League One · OU25 · UNDER · standard | PAPER | market beats model | same fixture-level number | 0.351 |
| League One · BTTS · YES · standard | PAPER | market beats model | +0.0141 [+0.0008, +0.0261], 101 fx | 0.110 |
| League Two · OU25 · OVER · standard | PAPER | market beats model | +0.0196 [+0.0060, +0.0333], 115 fx | 0.159 |
| League Two · OU25 · UNDER · standard | PAPER | market beats model | same fixture-level number | 0.165 |
| USA MLS · BTTS · YES · new_format | PAPER_LEAGUE | market beats model | +0.0293 [+0.0063, +0.0515], 133 fx | 0.196 |

All six are REPAIR on the **market** rule; none on the ROI rule (no entry has adequate signals and an ROI
CI entirely below zero). None reaches REPAIR_PRIORITY_HIGH (score ≥ 0.40, REPAIR-1 weights
`build_league_regimes.py:52-64`, `:309-330`). Highest LEARNING scores, for reference: Bundesliga 2 OU25
UNDER 0.579, USA MLS OU25 UNDER 0.439, La Liga 2 OU25 OVER 0.430 — scored, but not REPAIR, because their
evidence is below the adequacy bar.

## 6. Learning combos

93 entries. They split into:

* **No evidence at all** (n 0 and no market metric): every Japan J-League and Romanian Superliga entry,
  every OVER35 entry, and most OVER15 entries (no opposite price → market metric UNAVAILABLE).
* **Below the adequacy bar** (n < 30 or < 8 matchdays): the large majority, e.g. Argentina OU25 OVER
  n 24 / 19 md; Serie B OU25 OVER n 28 / 14 md (ROI CI90 [+1.0%, +64.1%] — positive but under n 30, and
  PAPER, so it could not be PROTECTED anyway).
* **Adequate but inconclusive**: Argentina OVER15 OVER (n 69, 34 md, ROI CI [−14.3%, +14.9%]),
  Bundesliga 2 OU25 UNDER (n 34, 16 md, CI [−53.6%, +19.8%]), La Liga 2 OU25 OVER (n 34, 22 md), Mexico
  Liga MX OU25 UNDER (n 30, 19 md, CI [−5.4%, +47.7%]), USA MLS OU25 UNDER (n 56, 23 md).

## 7. Weekly ranking methodology

**RS-1 ranking score** (`league_weekly.py:42-49`, key `rs1_key` `:431-436`, applied by `rank_frame`
`:409-422`). Sort, first key first: (1) price tier — a two-sided price (`MEDIAN_TWO_SIDED_BOOKS` or
`SINGLE_TWO_SIDED_BOOK`) before INSUFFICIENT; (2) `ev_lower_bound` desc; (3) `ev_point` desc;
(4) `p_lower_bound` desc; (5) `n_books` desc; (6) kickoff asc; (7) fixture_id asc. NaN sorts last; the
order is total and deterministic. `legacy_rank` = the same candidates ranked by legacy raw edge, copied
for comparison, never recomputed. Every score comes from `output/edge_engine_shadow.csv` (market_edge_v2
first-sight rows); the panel never re-prices a fixture (`:23-25`).

**WL-1 week membership / lock** (`:53-65`; `week_locks` `:439-444`, `freeze_rows` `:461-488`,
`provisional` `:491-504`). Each fixture × market is frozen at first sight. The league-week LOCK is the
earliest kickoff among that league-week's candidates. The top-5 = RS-1 ranking of candidates first seen
**strictly before the lock**, written once at the first run at/after the lock and never again. Late
fixtures (first seen at/after the lock) stay in the candidates log as `late_after_lock` and are never
members. Before the lock the ranking is PROVISIONAL (in `weekly_league_report.json`, never in the
canonical panel).

**GW-1 gameweek rule** (`src/gameweek.py:8-28`, `league_week_id` `:45-57`): `'<league>|<ISO
year>-W<ISO week>'` of the **tip day** (`src/tip_day.py:33-39`; the tip day rolls over at 10:00 Israel
time, `:28-29`, so a Sunday-night Americas kickoff at 01:00–03:00 UTC Monday stays in its weekend).
The official-round branch exists but is dormant (no committed artifact carries an API-Football round).
Documented edge cases: midweek rounds share the preceding weekend's ISO week; Monday-evening EFL
fixtures belong to the next ISO week; a rearranged fixture takes the week it is played in.

## 8. Canonical schema

* `output/league_weekly_research.csv` — key `PANEL_KEY` (`selection_engine_version, league_week_id,
  market, league_market_rank`, `league_weekly.py:144`); 56 frozen columns `PANEL_FROZEN` (`:145-156`)
  and 19 grading columns `PANEL_GRADING` (`:157-162`). `selection_mode = WEEKLY_LEAGUE_RESEARCH`.
* `output/league_weekly_candidates.csv` — key `CAND_KEY` (`selection_engine_version, fixture_id,
  market`, `:132`), columns `CAND_COLS` (`:133-142`), append-only first sight.
* Contract `LEAGUE_WEEKLY_RESEARCH_ROW` v1, `src/contracts/schemas.py:320` (unique key, rank 1..5,
  `selection_engine_version = market_edge_v2`, `price_quality_state` ∈ {TWO_SIDED_MULTI_BOOK,
  TWO_SIDED_SINGLE_BOOK, INSUFFICIENT}; rules: first seen before kickoff, first seen strictly before the
  week lock (WL-1), INSUFFICIENT never carries a P/L, QUALIFIED needs EV_LB > 0), registered for the file
  at `:499`; checked in shadow mode on every freeze (`league_weekly.py:829-833`) and by
  `tests/test_league_weekly.py::test_contract_for_panel_rows`.

## 9. First-sight freeze

`src/research/first_sight.py` (`FirstSightStore`, class at `:81`; rules `:10-21`): earliest row per key
wins against the file and within the batch; rows observed at/after kickoff are refused; survivors are
**appended** (existing bytes never rewritten); `digest=True` keeps a sidecar
(`league_weekly_research.csv.first_sight.json`) and fails closed — nothing appended — when the frozen
columns' digest no longer matches (`frozen_digest`, `:65-78`). The panel store uses `digest=True` with
`PANEL_FROZEN` as the frozen set (`league_weekly.py:402-405`); the candidates store has the
key/kickoff refusal but no digest (`:397-399`). Concurrency: one writer per file, serialised by the
`predict` concurrency group (`first_sight.py:28-37`). Today's `verify()`: `ok: True`.

## 10. Settlement

`settle_panel`, `league_weekly.py:544-619`, run by `update_results.yml:130-145`
(`python -m src.research.league_weekly settle`). Waits `SETTLE_DELAY_MIN = 240` min after kickoff
(`:103`). Score from `output/v91_shadow.csv` settled rows (fixture_id, else date + teams) — never a new
network call. Fills grading columns once; the frozen columns are hashed before and after and a mismatch
raises (`:617-618`); a tampered sidecar stops settlement entirely (`settle`, `:853-855`). `flat_pnl`
only on rows with a two-sided price; INSUFFICIENT rows carry the result but no P/L (`:589-591`).
`production_selected_final` / `actually_bet_final` / `stake_final` REPORT what the ledgers and stake
audit did; they never set it.

Today: 136 / 136 panel rows settled (`settle_status = SETTLED`), 0 pending.

## 11. CLV

Certified only: the close must be strictly pre-kickoff and within `CLOSE_MAX_MIN = 30` minutes
(`league_weekly.py:104`, check `:583-588`); otherwise `clv_pct` is NaN and `close_status =
STALE_CLOSE_GT_30MIN`. `clv_pct = (odds / closing_odds − 1) × 100`; `closing_market_fair_prob` = power
de-vig of both sides' certified closes; `clv_prob = closing_market_fair_prob − p_market_fair` (`:592-608`).

Today: 53 of 136 settled rows have a certified close — **coverage 39.0%** (`clv_coverage 0.3897` in the
summary). Example: the only USA MLS W43 row (LAFC v LA Galaxy) is `STALE_CLOSE_GT_30MIN`, so it has a
result but no CLV.

## 12. Dashboard changes

* **League Repair Lab, page 18** (`pages/18_🔧_League_Research.py`, admin tier,
  `dashboard_access.py:145`): leagues REPAIR → LEARNING → PROTECTED (`:79`), league profile (`:90`),
  this gameweek's top-5 (`:112`), TOP 1/2/3/5 strip (`:122`), **weekly rank heatmap** gameweek × rank
  with result / CLV / decision state per cell (`:126-127`; `dashboard_data/league_research_view.py`
  `heatmap_frame` `:293`, `heatmap_chart` `:338`, `heatmap` `:362`), production vs market-first shadow
  (`:133`). Every aggregate comes from `src/research/league_weekly_eval.py` via
  `dashboard_data/league_research.py` (column-projected, cached).
* **Leagues, page 17** (`pages/17_🏟️_Leagues.py`, pro tier, `dashboard_access.py:130`): per-league
  results/tips/rating/averages from the ledgers (same functions as the Monday Telegram picture), plus a
  "Research status" section showing each market's research regime (`:193`) — labelled as where research
  effort goes, never a stake.

## 13. Tests

Run 2026-10-26 in this worktree: `tests/test_league_weekly.py` (18), `test_research_lock.py` (9),
`test_league_regimes.py` (8), `test_gameweek.py` (4), `test_leagues_page.py` (8),
`test_first_sight.py`, `test_track_registry.py` — **76 passed**. They cover rows per league ≤ 5,
fewer fixtures never manufactured, RS-1 determinism, byte-identical reruns, no post-kickoff or late
members, provisional-before-lock only, missing price fails closed, legacy fields distinct, settlement
leaves frozen columns intact, stale close not certified, exact side reconstruction, health NO_FIXTURES
vs MISSING_ROWS, eval metrics, page = eval, contract; regime rules (one losing week never moves an
entry; sustained loss → REPAIR; credible staked profit → PROTECTED); lock detection of a changed
threshold/state/stake, missing snapshot fails closed.

## 14. Production non-regression proof

`python -m src.research_lock --check` (2026-10-26T08:39Z, this worktree at `f6770fdf`):

```
protected_tracks_modified=False (approved by Nevo (owner) - approved in chat 2026-10-25 at 2026-10-25T12:09:26Z)
exit=0
```

`check()` detail: `changes: []`, `threshold_files_unapproved: []`, `unapproved_new_protected_tracks:
[]`, `n_protected_tracks: 1`, approved digest = current digest = `cbd42d57…4a7803`. The fingerprint
(`src/research_lock.py:25-46`, `fingerprint` `:175`) covers the bankroll policy, every env var
`predict.yml` sets for `config.py`, thresholds resolved under that env in a subprocess, eligibility
lists, Cloudbet mode, the threshold files' content digests (`:16-23`, since `8b46b37f`), and every
PROTECTED track's state / base fraction / tier multipliers / odds caps / resolved gates.

Static and dynamic tests that research never writes production files:

* `tests/test_league_weekly.py::test_research_rows_never_touch_production` (`:287`): the module source
  contains none of `bankroll_policy`, `league_market_regimes`, `notified.json`, `gate_resolver`,
  `LEAGUE_MARKSMAN`, `THRESHOLD`, `telegram`; its only writes are its own tmp-then-replace files and the
  store; after `run` + `settle`, the SHA-256 of `stake_plan.csv`, both ledgers, `stake_audit.csv`,
  `predictions.csv`, `bankroll_policy.json`, `league_market_regimes.json` and `track_registry.json` is
  unchanged.
* `tests/test_research_lock.py::test_research_code_never_imports_production_decision_modules` (`:80`) and
  `::test_research_code_path_never_writes_production_files` (`:97`): the regime builder + lock write only
  the snapshot path and refuse to write any other `config/` file.
* Workflow isolation: the panel runs as a separate tolerant step after the notifier
  (`predict.yml:241`, `|| echo ::warning::`), staging only its own files per file (`:264-265`).

Commit-level non-regression evidence: `9f68f862` and `33eb6ebe` ran `pipeline.mode_predict` on committed
predictions before/after with the deployed env — bets, side_bets, every ledger, `stake_audit`, `v91_shadow`
byte-identical (only wall-clock columns differ, identically in a before-vs-before control); `8b46b37f`
resolved all 192 league × market × side gates under the `predict.yml` env byte-identical before/after
(sha256 `232d3e58…`).

## 15. Sample-accumulation requirements

What the code enforces today:

| gate | rule | where |
|---|---|---|
| panel status OK | ≥ 30 **priced** settled rows **and** ≥ 8 independent gameweeks, per league × market × model_type × side × subset | `league_weekly_eval.py:27-29, 42, 147` |
| regime change | ≥ 30 signals and ≥ 8 matchdays (or ≥ 100 two-sided fixtures and ≥ 8 matchdays) | `build_league_regimes.py:90` |
| CLV term counts | ≥ 10 certified closes | `build_league_regimes.py:93` |
| INSUFFICIENT price | can rank, never certifies performance (no P/L) | `league_weekly.py:43-45, 589-591` |

Lifecycle. The brief's chain DISCOVERY → SHADOW → ACCRUING → ELIGIBLE_FOR_REVIEW → OWNER_REVIEW is **not
one state machine in code** (already recorded as gap G4 in `docs/WOWZA_PLATFORM_ARCHITECTURE_2026_10.md`
§7). Mapped onto what exists for this panel:

| stage | meaning here | today |
|---|---|---|
| DISCOVERY | candidates logged, provisional ranking | every tracked league with fixtures |
| SHADOW | frozen top-5 rows, graded, never staked | 136 rows / 44 league-week × market groups (W43) |
| ACCRUING | `NOT_ENOUGH_DATA` (below 30 priced settled or 8 GW) | all 294 evaluation rows |
| ELIGIBLE_FOR_REVIEW | status OK on TOP-k with certified-CLV coverage reported | none |
| OWNER_REVIEW | a human decision; no recorded state | — |

When can a league first leave ACCRUING? Computed (script, two ways) from the W44 candidate counts and the
W43 rows already frozen, assuming every week from W44 yields min(5, fixtures) **priced** settled TOP-5
rows: the 8-gameweek bar binds before n 30 everywhere. Leagues with a W43 gameweek (Argentina, Austria,
Bundesliga 2, Denmark, Finland, La Liga 2, Liga MX, Norway, Serie B, Sweden, MLS) reach 8 GW at **W50
(2026-12-07 → 12-13)**; the rest (Brazil, Championship, Ireland, Japan, League One, League Two, Ligue 2)
at **W51 (2026-12-14 → 12-20)**. That is a best case: OVER15 rows are mostly INSUFFICIENT (see §16), a
winter break or a season end stops a league's clock (several calendar-year leagues may finish before
W50), and Rank-1 alone needs 30 gameweeks (a full season). CLV coverage must also be read with each
figure: 39% today.

## 16. Unresolved issues

1. **OVER15 pricing gap.** 214 of 225 OVER15 candidates are INSUFFICIENT (W44: 180 of 190) because the
   engine rarely has a young two-sided O/U 1.5 price at first sight. The capture change addressing it —
   in-loop WIDE passes + FAR price refresh (`eed1f1af`, merged in `9c0a3220`) and the 5.5 h FAR heartbeat
   + new-format NEAR 04–08 UTC (`f6770fdf`), owner-approved 2026-10-26 — is committed but not yet
   visible in the panel: its replay estimated OU15 two-sided coverage 48–96 h out ~40% → ~75%. Re-check
   the INSUFFICIENT share on W45 candidates.
2. **First full week is W44 (2026-10-26 → 11-01).** W43's 44 frozen groups are partial: the panel began
   2026-10-25T09:25Z (a Sunday), so W43 holds only Sunday/Monday-early fixtures first seen before each
   lock (1–5 per league; USA MLS one fixture). W43 still counts as a gameweek in the evaluation.
3. **No market-first candidate has a positive EV lower bound.** 0 of 535 priced candidates; every
   decision_state is NO_BET. The panel ranks the "least negative" fixtures, which is why ranking quality
   (relative) and threshold (absolute) are evaluated separately (`league_weekly_eval.absolute_vs_relative`,
   `:211`); the ABSOLUTE_EV_LB_POSITIVE subset is empty everywhere.
4. **First-sight stake vs final stake differ.** E.g. Argentina BTTS W43 rank 3 (Independiente v Boca):
   `actually_bet` True, stake 1.13 at first sight; `actually_bet_final` False at settlement. The panel
   reports both and changes neither; the difference comes from the stake plan being rebuilt between runs.
5. **Regime registry is a 2026-10-20 snapshot with no scheduled rebuild.** `evidence_refresh.yml`
   (06:45 UTC) regenerates `config/track_registry.json` (`:77`) but not `league_market_regimes.json`;
   `scripts/build_league_regimes.py` is run by hand and needs a Pro checkout. A rebuild that newly
   classifies a track PROTECTED is reported by the lock as `unapproved_new_protected_tracks` until an
   owner `--approve`. The REGIME-1 labels do not include the panel's own forward evidence (by design).
6. **Lifecycle not one state machine** (G4, §15) and the official-round GW branch is dormant (§7).
7. **No fixtures:** China Super League and Romanian Superliga currently `NO_FIXTURES`; Japan J-League and
   Romanian Superliga have no regime evidence at all.

---

## Acceptance questions

### A. Which league × market are PROTECTED / REPAIR / LEARNING, and why?

* **PROTECTED (1):** Argentina Primera Division BTTS YES (new_format, ACTIVE): ROI +36.4%, CI90
  [+15.8%, +56.4%] on 81 signals over 35 matchdays; the model also beats the de-vigged market there
  (dLL −0.0547, CI upper −0.0195). §4.
* **REPAIR (6):** League One OU25 OVER + UNDER and BTTS YES; League Two OU25 OVER + UNDER; USA MLS BTTS
  YES — in every case because the de-vigged market beats the model with a CI excluding zero on ≥ 100
  fixtures (§5). No entry is REPAIR on ROI and none is REPAIR_PRIORITY_HIGH.
* **LEARNING (93):** all others — no evidence, below 30 signals / 8 matchdays, or adequate but
  inconclusive (§6).

### B. Proof that PROTECTED tracks behave identically

* `research_lock --check` today: `protected_tracks_modified=False`, no changes, approved digest =
  current digest (`cbd42d57…`). For Argentina BTTS YES that pins production state ACTIVE, track state
  ACTIVE, base fraction 0.0125, tier multipliers SNIPER 1.1 / MARKSMAN 0.9, max odds 2.6, and the
  resolved gates VALUABLE 0.04 / MARKSMAN 0.08 / SNIPER 0.10 (pipeline.py side-market globals) — i.e.
  the selection rule, the side (YES), the odds cap, the stake and the policy state.
* The panel only REPORTS production: on Argentina BTTS W43 its `production_selected` / `actually_bet`
  columns are copied from the legacy tier and the stake plan (three legacy selections: SNIPER @2.10,
  MARKSMAN @2.10, SNIPER @2.70; the @2.70 one carries no stake, consistent with the 2.6 track cap). The
  panel writes none of those files (§14).
* Commit-level byte-identity of `mode_predict` outputs (`9f68f862`, `33eb6ebe`), 192 gate cells
  identical (`8b46b37f`), and the static/dynamic no-write tests (§14).

### C. First weekly top-3/5 panel for every REPAIR league

**No REPAIR league has a frozen weekly panel yet.** League One, League Two and USA MLS have no W43
fixtures in the panel except one MLS match; their first full week is W44 and its lock has not passed at
the time of writing (08:41Z). The panels below are **PROVISIONAL** (`weekly_league_report.json`
08:20Z, identical on a read-only re-run at 08:41Z) and can still change until the lock. Every row is
NO_BET with a negative EV lower bound; `legacy` = the legacy tier at first sight.

*League One W44 — lock 2026-10-27T19:45Z, 17 candidates*

| rank | OU25 (side, EV_LB, odds, legacy) | BTTS YES (EV_LB, odds, legacy) |
|---|---|---|
| 1 | Bradford v Huddersfield UNDER −0.094 @1.91 AVOID | Stockport v Wimbledon −0.119 @1.80 AVOID |
| 2 | Leyton Orient v Cambridge OVER −0.094 @1.83 AVOID | Leicester v Stevenage −0.120 @1.69 AVOID |
| 3 | Leicester v Stevenage OVER −0.100 @1.76 AVOID | Sheffield Wed v Cambridge −0.120 @1.54 AVOID |
| 4 | Plymouth v Reading OVER −0.102 @1.60 AVOID | Wycombe v Burton −0.121 @1.71 AVOID |
| 5 | MK Dons v Bromley OVER −0.104 @1.61 AVOID | Luton v Doncaster −0.123 @1.64 AVOID |

*League Two W44 — lock 2026-10-26T20:00Z (tonight), 14 candidates*

| rank | OU25 | BTTS YES |
|---|---|---|
| 1 | Barnet v York OVER −0.082 @1.54 AVOID | Accrington v Northampton −0.116 @1.83 AVOID |
| 2 | Port Vale v Rochdale UNDER −0.091 @1.82 AVOID | Fleetwood v Bristol Rovers −0.119 @1.66 AVOID |
| 3 | Newport v Chesterfield OVER −0.095 @1.82 AVOID | Grimsby v York −0.123 @1.53 AVOID |
| 4 | Fleetwood v Bristol Rovers UNDER −0.095 @1.97 AVOID | Shrewsbury v Gillingham −0.123 @1.81 VALUABLE |
| 5 | Salford v Exeter OVER −0.097 @1.80 AVOID | Newport v Chesterfield −0.124 @1.67 AVOID |

*USA MLS W44 — lock 2026-10-28T23:30Z, 17 candidates (paper league)*

| rank | OU25 | BTTS YES (the REPAIR entry) |
|---|---|---|
| 1 | Portland v Houston OVER −0.099 @1.56 AVOID | Charlotte v Montreal −0.104 @1.53 AVOID |
| 2 | Seattle v Houston OVER −0.102 @1.67 AVOID | Minnesota v Sporting KC −0.107 @1.48 AVOID |
| 3 | Colorado v Chicago OVER −0.109 @1.55 AVOID | Austin v Real Salt Lake −0.114 @1.50 AVOID |
| 4 | NYCFC v Atlanta UNDER −0.111 @2.34 VALUABLE | NYCFC v Atlanta −0.115 @1.54 AVOID |
| 5 | Minnesota v Sporting KC OVER −0.112 @1.35 VALUABLE | Seattle v Houston −0.116 @1.57 AVOID |

The only frozen REPAIR-league row is USA MLS W43 rank 1 (LAFC v LA Galaxy): BTTS YES @1.50 WIN +0.50u,
OU25 OVER @1.40 WIN +0.40u, both with a stale close (no certified CLV). One row is not evidence.

### D. "Ranking works but threshold doesn't" vs "model does not beat market" vs "insufficient evidence"

* **Ranking works but threshold doesn't — no league can be placed here yet.** It needs the panel's
  TOP-k to show positive ROI/CLV with status OK while the production-selected subset does not. All 294
  evaluation rows are `NOT_ENOUGH_DATA`; 85 priced settled rows exist in total, at most 5 per league ×
  market, one (partial) gameweek. The pooled rank-monotonicity table is labelled POOLED and is not
  readable at n 0–8 per rank.
* **Model does not beat market — six entries, from the regime registry's market-relative dLL** (not from
  the panel): League One OU25 (+0.0303, CI90 [+0.0146, +0.0480], 114 fixtures) and BTTS (+0.0141
  [+0.0008, +0.0261], 101); League Two OU25 (+0.0196 [+0.0060, +0.0333], 115); USA MLS BTTS (+0.0293
  [+0.0063, +0.0515], 133). Trending the same way but below the 100-fixture bar, so not classifying:
  Norway OU25 (+0.0354 [+0.0084, +0.0588], 56) and Ireland BTTS (+0.0290 [+0.0098, +0.0481], 32).
* **Insufficient evidence — everything else**, which on forward panel evidence is every league ×
  market. Retrospectively the market-first replay says the same at track level: no track's model beats
  the de-vigged market with a CI excluding zero, and standard OU25 is beaten by it (Brier gain −0.0064,
  CI [−0.0115, −0.0005], `output/edge_engine_comparison.json`).

### E. Working leagues tracked in shadow without production change

* **Argentina Primera Division BTTS YES** (PROTECTED, ACTIVE): tracked in the panel every week (W43 frozen,
  5 rows settled: 1 win, ROI −46% on n 5, certified CLV coverage 80%, mean +1.9%; W44 provisional), with its
  production fingerprint unchanged (B).
* Point-positive but not adequate (LEARNING, so neither protected nor repaired, all tracked weekly with
  no production change): Serie B OU25 OVER (PAPER, ROI CI90 [+1.0%, +64.1%] on n 28), Brazil Serie A OU25
  OVER (ACTIVE_LOW, n 10, model beats market dLL CI [−0.047, −0.002] on 75 fixtures), Mexico Liga MX OU25
  UNDER (ACTIVE_LOW, n 30, CI [−5.4%, +47.7%]), Sweden Allsvenskan BTTS (ACTIVE, model beats market dLL
  CI [−0.054, −0.008] on 56 fixtures).
* All 20 leagues are in the panel; none of their thresholds, stakes, eligibility or states changed
  (`research_lock`, §14).
