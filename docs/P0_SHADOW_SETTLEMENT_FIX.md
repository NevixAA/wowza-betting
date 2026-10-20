# P0 — Shadow settlement fix (2026-10-20)

**Scope:** observability only. No production threshold, staking, league routing, model
champion, BTTS preregistration or selection rule was changed.

## Root cause

`output/v91_shadow.csv` had 641 prospective rows (logged 2026-10-04 → 2026-10-20) and **0
settled**. There were two causes, and either one alone was enough to block settlement:

1. **No writer existed.** `pipeline._log_shadow` writes `closing_odds`, `clv` and `result` as
   empty (`pipeline.py:264-266` at `71271bb7`), and the comment says they are "filled at
   settlement". No code in the repo ever wrote them. A `git grep` at `71271bb7` finds
   `v91_shadow.csv` only in its writer (`src/shadow_compare.py:29`) and in predict's commit
   list (`.github/workflows/predict.yml:258`). `update_results.py` never opened it.
2. **The rows could not be joined to a result.** A row held only `fixture_id`, the OddsAPI
   event id (`pipeline.py:222`), plus league and date. It held **no team names**. Every results
   source (football-data, OddsAPI scores, API-Football) is keyed by team names and a date, so a
   settler could not have matched a single row.

The effect was that `prospective_window()` (`src/shadow_compare.py:133`) returns WIN/LOSS rows
only, so the forward period was always empty.

## What changed

| File | Change |
|---|---|
| `src/shadow_settlement.py` (new) | Pure settlement logic: eligibility, result matching, grading, pre-kickoff close, CLV, `apply_updates` with a frozen-column hash guard, health numbers. No network access and no file paths. |
| `update_results.py` | `settle_shadow_predictions()` runs **last** in `main()` inside `try/except`, the same way as `ht_observations`. Also adds `reapply_shadow_settlement()` and the flags `--shadow-journal`, `--shadow-reapply` and `--shadow-only`. `fetch_scores()` now also returns the OddsAPI `event_id`. This is additive and no ledger reads it. |
| `src/shadow_compare.py` | New columns: identity (`home_team`, `away_team`) and grading detail (`outcome`, `home_goals`, `away_goals`, `result_source`, `close_ts`, `close_minutes_to_kickoff`, `close_band`, `close_source`, `close_status`, `settle_status`, `settled_at`). `GRADING_ONLY` is extended. New `IDENTITY` and `FROZEN` sets. `append_shadow` now reads the old rows with `dtype=str`, so predict copies them through verbatim instead of re-rendering settled cells. |
| `pipeline.py` | `_log_shadow` also records `home_team` / `away_team` at first sight. |
| `.github/workflows/update_results.yml` | The settling run writes a journal to `$RUNNER_TEMP`. The ledger step drops its local edit of the shadow file before rebasing. A new step, **"Commit shadow settlement"** (`if: always()`), handles the shadow file. Timeout raised from 10 to 15 minutes. |
| `dashboard_data/v9.py`, `pages/0_🏠_Overview.py` | Read-only shadow-settlement health chip on the Overview "Feed health" strip. |
| `tests/test_shadow_settlement.py` (new) | 36 tests. |
| `output/v91_shadow.csv`, `output/shadow_settlement_health.json` | One-time identity backfill, then the first real settlement. |

### Result hierarchy (reused, not re-implemented)

Settlement uses the same sources, in the same order, as the ledgers:

1. `fetch_scores_fd`, for leagues in `_FD_SOURCES`.
2. `fetch_scores`, the OddsAPI scores endpoint. It is only asked about rows inside its 3-day
   window, so a stranded row cannot spend OddsAPI credits on every run.
3. `fetch_scores_apifootball`, as the fallback.

Each source is called lazily and cached per league for the run.

Matching tries the following, in order:
1. The exact OddsAPI event id.
2. `src/team_names.resolve` on both teams.
3. The ledgers' own `_names_match`.

`_names_match` is wrapped in a guard because its first-word rule pairs "Manchester City" with
"Manchester United". The guard rejects any pair whose descriptor tokens (city / united / town /
rovers …) differ. A date tier counts only when **exactly one** event matches, so ambiguous cases
are refused rather than guessed.

### Closing price

The close is the **last price strictly before kickoff**, taken from the ledgers' own archives
(`_ARCHIVE_FILES`, densest first: `newformat_odds_dense.csv`, `standard_odds_history.csv`, …).
These are the same OddsAPI price series that `entry_odds` comes from.

- A post-kickoff quote is never used.
- When there are quotes but all of them are at or after kickoff, the status is
  `POST_KICKOFF_ONLY`.
- When there is no quote at all, the status is `NO_CLOSE`.
- In both cases the result is still settled, with `closing_odds` and `clv` set to NaN. No price
  is ever invented or defaulted.

CLV is `(entry_odds / closing_odds − 1) × 100`, the same percent convention as `bets_ledger`.

**Change-log caveat:** the archives store consecutive-distinct prices. `close_ts` is therefore
when the closing price was *first* seen, and that price then stayed in force until kickoff
(as far as the archive knows). `close_band` (FAR / T-6h / … / T-10m) shows how far from
kickoff that observation was, so a far-out close is labelled rather than silently called a
closing line. Consumers that need a certified near-kickoff close should filter on
`close_band in {T-30m, T-10m}`.

### Why the workflow does not use `pull --rebase -X theirs` for this file

`v91_shadow.csv` has two writers:
- predict **appends** rows every 5–15 minutes;
- update_results **edits** rows in place.

When predict's appended lines sit next to a row this run settled, git reports a conflict. With
`-X theirs` the replayed commit (this run) wins that hunk, which silently deletes predict's new
rows. Predict would then re-log them later with a newer `logged_at`, which breaks the
first-sight freeze without anyone noticing.

The new step avoids merging entirely. Each attempt does the following:
1. `git fetch` and `git reset --hard origin/main`.
2. Re-apply the journal onto that fresh file. This runs offline, is idempotent, and fills only
   blank grading cells.
3. Stage the two files individually and push.
4. If the push loses a race, retry.

The reverse race is harmless: if predict's own rebase reverts a settled cell, that row is blank
again and the next run re-settles it with the same values.

## Test evidence

- New file `tests/test_shadow_settlement.py`: **36 passed**. It covers:
  - an unsettled row becoming WIN, and one becoming LOSS;
  - UNDER grading and OVER grading at the 2.5 boundary;
  - BTTS / O1.5 / O3.5 grading, plus a BTTS row closing on its own `btts_yes` key;
  - future, in-progress and logged-after-kickoff rows staying unsettled;
  - frozen columns byte-identical after settlement, and `apply_updates` refusing a frozen
    column;
  - the close being the last pre-kickoff quote, a post-kickoff quote being rejected, and a
    post-kickoff-only case returning no close;
  - a missing close giving a result with NaN CLV, and the close never coming from the other
    side;
  - an already-settled row staying idempotent, and repeated settlement leaving the file
    byte-identical;
  - re-applying the journal onto a file predict has appended to, keeping both writers' rows;
  - CRLF and untouched rows surviving byte for byte;
  - team-name variants (QPR, 1. FC Kaiserslautern, Cádiz CF, Sheffield Weds) resolving, Man City
    never matching Man United, and two candidate events being refused rather than guessed;
  - a row with no identity being resolved by OddsAPI event id;
  - OddsAPI not being asked about rows older than its window;
  - `prospective_window()` seeing settled rows;
  - an end-to-end run: `_log_shadow` → result → settlement → `prospective_window()`;
  - the health artifact being written;
  - the workflow YAML staging `output/v91_shadow.csv` per file in an `always()` step with
    `--shadow-reapply`, and the journal env being passed;
  - settlement running last in `main()`.
- Existing shadow tests (`test_shadow_compare`, `test_shadow_logging`, `test_shadow_gate`) all
  pass unchanged.
- Full suite: **427 passed, 22 skipped**.

## Local run on the real file (2026-10-20 ~11:48 UTC)

Keys were loaded silently from the local `.env`. football-data.co.uk **failed locally** with a
TLS certificate error (corporate network), so this run used OddsAPI scores and API-Football
only. In CI, football-data is reachable and stays the primary source for the leagues it covers.

| | Before | After |
|---|---|---|
| Shadow rows | 641 | 641 (same order, same rows) |
| Rows with team identity | 0 | 641 (backfilled once from 100 recorded `predictions.csv` versions in git history; every `fixture_id` matched exactly one event, and league and date agreed on 641/641) |
| Finished + prospective (settleable) rows | — | **420** |
| Settled (WIN/LOSS) | **0** | **375**: 186 WIN / 189 LOSS (240 UNDER, 135 OVER) |
| Result coverage of settleable rows | 0% | **89.3%** (58.5% of all rows; the other 221 rows have not been played yet) |
| Valid pre-kickoff closing price | 0% | **100%** of settled |
| CLV populated | 0% | **100%** of settled |
| Result source | — | API-Football 242, OddsAPI 133 |
| Close source | — | `newformat_odds_dense.csv` 207, `standard_odds_history.csv` 168 |

### Close distance from kickoff (settled rows)

| Band | FAR | T-6h | T-3h | T-1h | T-30m | T-10m |
|---|---|---|---|---|---|---|
| Rows | 170 | 84 | 75 | 18 | 20 | 8 |

The median close is 319 minutes before kickoff. Only 28 of 375 closes (7.5%) fall inside T-30m.
"100% closing-price coverage" therefore means "a valid pre-kickoff price exists", **not** "a
certified closing line exists". That is a property of the change-log archive and of predict's
delivery rate, not of this fix.

### Second-way check

84 settled shadow rows overlap with settled `bets_ledger.csv` tips: same teams, date and side.

- Results agree on **84/84**.
- Closing odds equal the ledger's on **84/84**.

### Prospective freeze proof

A sha256 hash covers all 29 first-sight columns, row by row, in order. The columns are
`fixture_id`, `market`, `logged_at`, `match_date`, `kickoff_utc`, `league`, `model_type`,
`p_v9`, `p_v91`, `p_pro_hgb`, `p_goal_distribution`, `p_market`, `v9_side`, `v91_side`,
`v9_tier`, `v91_tier`, `v9_edge`, `v91_edge`, `entry_odds`, the 3 production gates,
`threshold_regime_id`, `opening_odds`, `book_count`, `market_dispersion`, `v9_model_sha`,
`v91_model_sha` and `dataset_id`.

| Version | sha256 |
|---|---|
| `HEAD:output/v91_shadow.csv` before settlement | `db5a82774128017f57d023d1a133ebde07da2f915b00134babe30dc1a5207f5e` |
| After settlement | `db5a82774128017f57d023d1a133ebde07da2f915b00134babe30dc1a5207f5e` |

The hashes are **identical**. `apply_updates` enforces the same check on every write and raises
instead of writing if it ever fails.

## Remaining unsettled or unsupported cases

There are 45 finished rows with no result found locally:
- Japan J-League 18
- Denmark Superliga 7
- China Super League 5
- USA MLS 4
- Brazil Serie A 4
- Austrian Bundesliga 3
- Finland Veikkausliiga 2
- Ireland Premier Division 2

- **39 of them are in football-data leagues** (Japan, Denmark, China, MLS, Austria, Ireland).
  Only the local TLS failure stopped them settling, and the first CI run should settle them.
- **API-Football fallback season config is stale for two leagues.** This is pre-existing and
  also affects the ledgers' fallback.
  - `Denmark Superliga` resolves to season **2025**, whose results end on 2026-05-17.
  - `Japan J-League` season 2026 returns the spring tournament, whose results end on 2026-06-06.
  - When football-data lags, these leagues therefore have no working fallback. CLAUDE.md
    already warns that season config must be derived, never hardcoded. This is an owner
    decision because it touches v9 config.
- **Name and club-identity gaps in API-Football** (Brazil `Bragantino-SP`, Finland `TPS Turku`,
  China renamed clubs such as `Zhejiang` → `Hangzhou Greentown`). These are refused rather than
  guessed. They need `team_names._ALIASES` entries if the owner wants them.
- **Postponed or abandoned matches** never appear in a completed-results feed, so they stay
  unsettled. There is deliberately no VOID inference. The health JSON counts them under
  `NO_RESULT_FOUND`.
- **Rows with no `v9_side`** settle the `outcome` but leave `result` empty (`SETTLED_NO_SIDE`).
  There are none today.
- **Only `ou25` rows exist.** BTTS / O1.5 / O3.5 grading and closing keys are implemented and
  tested, but predict does not log those markets yet.
- **`v91_side` / `p_v91` / `p_pro_hgb` are empty in every row** (challengers are not wired), so
  only v9 is graded. `outcome` plus the goal columns let any future challenger side be graded
  without re-fetching.

## What this does NOT establish

375 settled rows over 16 days is a coverage milestone, **not** a valid prospective experiment.
The sample is small, it spans one regime, and most closes are far from kickoff. Nothing here is
evidence of edge, and the promotion gate's thresholds still apply.
