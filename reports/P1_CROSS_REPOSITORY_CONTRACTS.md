# P1 - Cross-repository contracts (Phase 2, Part 10)

Auditor run: 2026-11-03. Independent re-derivation: every figure below was recomputed by
`scripts/audit_cross_repo_contracts.py` from committed files. Its raw output is in
`reports/crossrepo_contracts_evidence.json`. No earlier report's number was copied.

## Verdict

**AMBER.** Every displayed P&L I sampled matches its source ledger exactly, to the cent. That
covers 4 figure families on 3 dashboard pages. **I found no case where the dashboard shows a signal
as profitable when its ledger cannot back it up.** League names are canonical everywhere, and every
price is a valid decimal price. Four contract problems remain:

1. The word **"Staked"** means two different things on two pages. Neither meaning is "money the
   policy actually staked".
2. There are **five incompatible fixture identities** across the repos, and v9's ledgers carry none.
3. Every consumer reads **v9 `main`**, not `data-truth-100`. Pro's branch already needs v9's newer
   league registry, so the merge order matters.
4. Cross-repo reads swallowed failures silently. I fixed the two that mattered most, writing the
   failing test first.

| Contract | v9 | Pro | v11 | exec | dashboard | Status |
|---|---|---|---|---|---|---|
| Canonical league names | 100% canonical (team ledgers) | 100% (`paper_1x2`, bet builder, ratings) | 100% (`shadow_log`, `graded`) | resolves via v9 registry, unknown = dropped | reads v9 | GREEN (1 alias "World Cup" in props + Pro registry) |
| Fixture identity | **none in ledgers** (names only) | `fixture_key` v2 (resolver hash) | `fixture_id` = sha1(date\|league\|match) | `date\|league\|home\|away` string | name-based | **RED** - 5 schemes, not joinable |
| Market keys / sides | `ou25` implied, `btts/over15/over35` lower, policy `OU25/BTTS/OVER15/OVER35` upper; side `AVOID` (122 rows) | uppercase track keys | O/U 2.5 only | lowercase markets; drops non-OVER/UNDER/YES/NO | reads v9 | AMBER (case-sensitive `track_key`) |
| Decimal odds | all > 1, none unparseable | all > 1 | all > 1 | rejects <= 1 | - | GREEN |
| Stake / profit units | flat 1u; 0 mismatches in 2,346 priced settled team rows (1,709 + 623 + 14) | flat 1u | flat 1u at **v11 snapshot** prices | USDT notional, PAPER_ONLY | flat 1u | GREEN (v11 "v9" line is not v9's P&L) |
| Model versions | stamped on 100% of rows since 2026-10-25; 1,590 + 546 older rows null; `ht_ledger` has no column | `calc_version` | `v9_model_sha` null 154/2,251 | - | - | AMBER |
| Signal time vs match time | 0 rows generated after match day; **kickoff not in ledgers** | 0 logged/sent at or after kickoff | **446 snapshots at/after kickoff, graded** | kickoff joined from predictions | - | AMBER |
| Settlement status | WIN/LOSS/VOID/blank | 0/1/2 (1X2), WON/LOST/UNKNOWN | `over25_result` mixes 1/0 with True/False (81 rows) | reads v9 | reads v9 | AMBER |
| PAPER vs ACTIVE | per-track policy (NF Unders PAPER) | excludes paper leagues | SHADOW label | PAPER_ONLY; cells have **no side** | "Staked" = tier, not track state | **RED** (definition split) |

## Repos and commits inspected

| Repo | Branch inspected | Commit | `origin/main` | Accessible |
|---|---|---|---|---|
| wowza-betting (v9) | `data-truth-100` (worktree `phase2/crossrepo`) | `e07a45915ba5` | `f6c329a54711` | yes |
| wowzaV9-Pro | `data-truth-100` | `78580b4286f3` | `5b29395b55dc` | yes |
| wowza_v11 | `data-truth-100` | `27acd24ae2d3` | `e31e1824b65c` | yes |
| wowza-exec | `data-truth-100` | `a18bf544624` | `56670768db0f` | code yes; **runtime outputs UNIDENTIFIABLE** (`output/` holds only `.gitkeep` on both branches; the candidate log and bets are not committed) |
| wowza-dashboard | `data-truth-100` (detached worktree) | `ff53514e70eb` | `aa4e6a26f2e9` | yes; generated copy of v9 code (`sync_from_v9.yml`); dt100 vs main differ only in `src/feature_engineering.py` + build stamps |

## Fixes made (each with its test written first)

| # | Repo | Change | Test (failed before, passes after) |
|---|---|---|---|
| F1 | v9 / dashboard | `dashboard_data/pro.py`: `tip_scoreboard()` now checks Pro's payload against its own definition (flat 1u, `roi = mean(units)`, `hit_rate = won/settled`, `0 <= won <= settled`, units >= -settled, numeric types, object shapes). An incompatible or non-reproducible payload comes back as `{"schema_error": why}` with **no figures**. The Pro page (`pages/7_🔬_Pro.py`) then shows an ERROR card, "SCHEMA_MISMATCH ...", instead of P&L or "No tip scoreboard yet". | `tests/test_crossrepo_contracts.py` (11 cases; 8 failed before the fix). Pro's live payloads pass: dt100 (generated 2026-11-01T18:00:53Z) and main (2026-11-02T13:43:04Z) |
| F2 | v9 / dashboard | `dashboard_data/research_view.load_all` used to turn a raising Pro read into `{}`, which the page shows as "no data yet". The failure is now recorded in `out["_errors"][name]` and the Pro page shows it. | `tests/test_crossrepo_contracts.py::test_load_all_records_a_raising_read`; `tests/test_dashboard_stage3.py::test_research_load_all_returns_the_same_call_values` now **also** asserts that the error is recorded. This makes the test stricter; its existing value checks are unchanged. |
| F3 | Pro | `src/importers/current_wowza.from_ledgers` used `d.get(col, <empty Series>)`. If v9 renamed `edge_pct`, `odds` or `pnl`, every settlement row was imported with that field silently all-NaN. It now raises `SchemaIncompatible`, a subclass of `SourceUnavailable`, naming the missing columns. `pro_collect` already records a raising importer in `collect_health.json` and does not advance watermarks. | `tests/test_v9_ledger_schema_contract.py` (18 cases; all failed before). The existing `tests/test_side_market_settlements.py` still passes against the real v9 ledgers. |

None of these changes touches a threshold, a stake, a track state or a ledger.

## Was any displayed P&L not reproducible from its ledger?

Each figure was recomputed independently from the raw ledgers, not through the dashboard code.

| Displayed figure (page) | Displayed | Recomputed from source | Source | Result |
|---|---|---|---|---|
| Home/Overview "Staked" (`today_view.staked_settled`) | 805 bets, -8.44u | 805, -8.44u (recorded pnl); -8.44u (flat 1u from result x odds) | v9 `bets_ledger` + `side_bets_ledger` @ `e07a4591` | **reproduced** |
| Overview by track (`v9.performance`) staked_only | new_format 601 / +27.155u; standard 204 / -35.60u | sum 805 / -8.445u, identical to the line above | same | **reproduced** |
| Research page "1X2 tips sent" (Pro `tip_scoreboard.json`) | 39 settled, 10 won, -6.40u, ROI -16.41% | 39, 10, -6.40u, -16.41% | Pro `output/paper_1x2.csv` @ `78580b42` | **reproduced** |
| Bankroll page review figures (Pro `track_evidence.json`, cutoff 2026-10-25) | 11 tracks | 9 of 11 exact from the current v9 ledger, using the review's tier rule (O/U 2.5 = SNIPER + MARKSMAN + VALUABLE, others = SNIPER + MARKSMAN). `standard\|BTTS` 162 / -0.42u is exact at v9 `dfcdf289` (the ledger as of the evidence timestamp); the current ledger gives 163 / +0.35u because one row was added or settled later. `new_format\|OU25_UNKNOWN_SIDE` 1 / 0.00u is a zero-P&L row whose side is not OVER/UNDER. | v9 ledgers | **reproduced** |
| v11 scoreboard "v9 overall" | 688 picks, -68.56u | **not v9's P&L.** v11 re-grades its own snapshots at snapshot prices (262 of 262 wins = snapshot odds - 1). 159 picks (-17.24u) come from snapshots at or after kickoff, and 36 have unknown kickoff status. v9's own ledger for the same period: staked O/U 507 / -36.94u; all tiers 1,038 / -63.10u. | v11 `v11_graded.csv`, `v11_shadow_log.csv` | not reproducible, **but not displayed**: no dashboard page calls `v11.scoreboard()` |

The ledger P&L itself is internally consistent. On every priced settled row, recorded pnl equals
flat 1u from (result, odds):

| Ledger | Settled rows | Recorded sum | Recomputed sum | Mismatches |
|---|---|---|---|---|
| `bets_ledger.csv` | 1,709 | -127.91u | -127.91u | 0 |
| `side_bets_ledger.csv` | 623 | +39.58u | +39.58u | 0 |
| `ht_ledger.csv` | 61 (47 have **no market price**; pnl = model fair odds - 1, verified on 27 of 27 wins) | -14.43u | -14.43u on the 14 priced rows | 0 |
| `player_ledger.csv` (props, PAPER) | 29,573 (16,886 without pnl, 5,465 without a price) | -1,119.77u | -1,119.77u | 0 |

The unpriced HT P&L is already kept out of the staked figures: `ht_ledger` has no
`signal_tier`, and Success Rates skips it. A sidecar exists at
`output/corrections/ht_ledger_unpriced.csv`.

## Findings in detail

### 1. "Staked" has two definitions (P1)

- Home, Overview and Tips use the owner definition from 2026-11-02 (`today_view.is_staked`):
  SNIPER + MARKSMAN on **every** track. Of its 805 bets (-8.44u), **495 bets (-61.32u) sit on
  tracks the current policy (`config/bankroll_policy.json`) has as PAPER**. Another **4 (-2.05u)
  sit on BLOCKED** O/U 3.5 tracks. The tracks: `standard|*`, and `new_format|OU25_UNDER` since
  2026-11-02.
- The Bankroll page uses Pro's review rule, where O/U 2.5 includes VALUABLE. For NF O/U 2.5 Under
  up to 2026-10-25, Home's basis is 259 bets / -16.95u. The Bankroll page shows 402 / -1.53u for
  the same track.
- Which historical rows the policy actually staked is **UNIDENTIFIABLE at HEAD**.
  `output/stake_plan.csv` keeps only the latest run: 70 rows from 2026-11-02 on dt100, 72 rows
  from 2026-11-03 on main. The answer exists only in git history.
- Owner action: decide whether "Staked" should mean "tier-qualified tip" (today) or "policy
  stake > 0". If the latter, keep a cumulative, append-only stake-plan log. Until then, both pages
  should say which definition they use.

### 2. Fixture identity (P2)

- Five schemes are in use:
  - v9 ledgers: none, names only.
  - exec: `date|league|home|away`.
  - v9 research (`src/research/edge_replay.py`): `league|date|home|away`.
  - v11: `sha1(date|league|"home vs away")[:16]`.
  - Pro: `fixture_key` v2, a resolver-based hash. Pro's `src/architecture/entity.py` also has
    `date|home|away`, with no league.
- In `bets_ledger`, 139 rows share a name key x market, and 67 such keys carry more than one side
  (AVOID included). exec keeps the first by `generated_at`; Pro imports all of them.
- Owner action: choose one canonical id (Pro `fixture_key` v2 is the most complete) and have v9
  write it into its ledgers as a new column, beside the names.

### 3. Every consumer reads v9 `main`, and the registries differ (P1, merge order)

- The raw URLs are hard-coded to `main`: Pro `config/pro_config.py`, v11 `config.py` and
  `src/league_registry.py`, exec `config.py`, and the dashboard's `core.RAW_BASE` and
  `remote_mirror.BRANCH`.
- The CI checkouts of wowza-betting use the default branch: 32 Pro workflow steps, plus v11 and
  exec.
- v9 `main` has registry `LR-1.0-2026-11-01`: 48 leagues, 3 aliases, no `competition_type`. v9
  dt100 has `LR-1.2-2026-11-02`: 53 leagues, 11 aliases. The 8 aliases missing on main include
  "Netherlands Eredivisie", "Turkey Super Lig" and "Primeira Liga".
- Pro dt100 `src/data/v9_config.leagues_of_type` raises on LR-1.0. It does so loudly, which is
  correct, and it is used by `src/pipelines/tourney.py`.
- Owner action: **merge v9 `data-truth-100` before, or together with, Pro's**.
- On this laptop, v11 resolves `../wowza-betting`, which is checked out at dt100. Local runs and
  CI therefore read different registries.

### 4. Silent exception handlers on cross-repo reads (inventory)

| Where | Behaviour | Status |
|---|---|---|
| v9 `dashboard_data/research_view.load_all` | a raising Pro read became `{}` and showed as "no data" | **FIXED (F2)** |
| v9 `dashboard_data/pro.tip_scoreboard` | a reshaped or non-reproducible payload rendered, or crashed on `h["units"]` | **FIXED (F1)** |
| Pro `importers/current_wowza.from_ledgers` | a renamed v9 column became all-NaN | **FIXED (F3)** |
| v9 `dashboard_data/core.read_csv/read_json` | a parse error looks the same as a missing file (empty frame) | OPEN P2 |
| v9 `dashboard_data/core.remote_listing` + `pro.canonical_stores` | a listing failure (GitHub 60/h limit) returns `[]`, and the page says "Pro is reachable but has no data/season_* store" | OPEN P2 (wrong claim) |
| Pro `data/v9_source.fetch_csv_all` | an archive-listing failure falls back to the live file only (recent matches), with a printed warning; downstream studies see truncated history | OPEN P2 |
| exec `cloudbet/candidates.team_candidates` | `d.get("model_type")` / `d.get("generated_at")` turn a missing column into None; the cell lookup then falls to the default state. PAPER_ONLY today, so no money risk | OPEN P2 |
| v9 `dashboard_data/pro.model_lineage/price_trader/valuable/tier_rules` | `except -> empty DataFrame` | OPEN P3 |
| v9 `dashboard_data/core.resolve_file` | network error -> stale cached copy (freshness still from content) | acceptable, P3 |
| exec `control/league_rating`, `control/policy`; v11/exec `league_registry` | fail closed or raise | OK |

### 5. Other contract notes

- **NF O/U Unders stay PAPER.** On main, `stake_plan.csv` (2026-11-03 06:44 UTC) carries the new
  policy version and PAPER / `NO_REAL_STAKE_PAPER` for `new_format|OU25_UNDER`. The dt100 copy is
  a snapshot from 2026-11-02 05:59 UTC, taken before commit `26ded833` (07:13 UTC). It still shows
  29 ACTIVE_LOW rows, 11 of them with stake > 0, under `balanced-2026-10-13`. That is branch
  staleness, not a defect. The `stake_plan.csv` headers also differ between main and dt100
  (`league_rating` and `rating_scale` exist only on dt100), so consumers must read it by column
  name.
- **exec cells have no side** (league x market x model_type). exec never reads v9's track states,
  so if a cell is ever promoted it cannot respect "NF Unders PAPER". exec is PAPER_ONLY today with
  default cell PAPER, so there is no risk now. This is a P2 prerequisite before any promotion.
- **Pooled rating sizing is off** in all three places: v9 policy `league_rating_sizing.enabled =
  false`, exec `league_rating_sizing: false`, and the exec gap is tracked by its own strict xfail.
- **Timestamps.** v9 `generated_at` is naive (1,832 of 1,833 rows). One row has a `Z` suffix:
  2026-08-08 Wolfsburg v Kaiserslautern, before the cutoff. Ledgers carry the match date only, so
  "signal before kickoff" on match day is **UNIDENTIFIABLE** from the ledgers. 0 rows were
  generated after the match day.
- **v11** grades 159 picks priced at or after kickoff and mixes `1/0` with `True/False` in
  `over25_result`. The v11 scoreboard's "v9" line should be relabelled "v9 tips re-graded at v11
  snapshot prices". Do not wire it to a page until both issues are fixed.

## Open issues

| Priority | Issue | Owner action |
|---|---|---|
| P1 | "Staked" means "SNIPER + MARKSMAN on any track" on Home/Overview (495 of 805 bets are on tracks PAPER now) but the review tier rule on Bankroll. Historical stake decisions cannot be recovered at HEAD. | Choose the definition. If it should be "policy stake > 0", approve an append-only stake-plan history. Label both pages. |
| P1 | Pro / v11 / exec / dashboard read v9 **main**; the registry on main is LR-1.0, and Pro dt100 needs LR-1.1+. | Merge v9 `data-truth-100` first (or together). |
| P2 | No canonical fixture id in v9 ledgers; 5 schemes in use. | Approve Pro `fixture_key` v2 as canonical, written as a new v9 ledger column. |
| P2 | `core.read_csv/read_json`, `remote_listing`, Pro `fetch_csv_all`, exec `d.get(...)` still degrade silently. | Approve the same pattern as F1-F3 (error state instead of empty). |
| P2 | v11 grades post-kickoff snapshots; its `over25_result` vocabulary is mixed. | v11 fix: exclude `is_post_kickoff` from grading, normalise the result to 0/1. |
| P2 | exec cells cannot express side-level PAPER. | Add `side` to exec cells or read v9 track states before any promotion from PAPER_ONLY. |
| P3 | `ht_ledger` has no `model_version`; the `bets_ledger` side vocabulary includes `AVOID`; `track_key` is case-sensitive (`btts` gives a key no policy defines); "World Cup" alias in props / Pro registry. | Low priority clean-ups. |
| P3 | exec runtime outputs are not committed, so exec's candidate-log contract is UNIDENTIFIABLE. | Commit or publish a daily candidate-log artifact. |

## How to reproduce

```
python scripts/audit_cross_repo_contracts.py --dev C:\Users\nevo\dev --dash <dashboard dt100 checkout>
python -m pytest tests/test_crossrepo_contracts.py -q                     # v9, branch phase2/crossrepo
python -m pytest tests/test_v9_ledger_schema_contract.py -q               # Pro, branch phase2/crossrepo
```

The script reads the committed outputs with light, column-projected reads. It does not train,
backtest, write to a ledger or send anything.
