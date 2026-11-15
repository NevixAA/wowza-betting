# P0 - Half-time history repair (Phase 2, Part 3)

Inspected: `wowza-betting` branch `origin/data-truth-100` at commit **e07a4591** (sidecar data commit
**0efc078a**, "data truth collect (htscores) 2026-11-03 00:16 UTC"). Work branch: `phase2/ht`.
Every number below was recomputed on 2026-11-03 by `scripts/audit/ht_history_repair_audit.py`
from committed files only. No API calls, no retrain, no backtest, nothing sent.

## Verdict

- **The half-time fill is sound.** All 33,259 filled rows match a stored API-Football fixture in the
  same league, on the same or the next day. None crosses a league, none repeats a key, none has a
  half-time score above full time, none was worked out from the full-time score, and none turns a
  missing provider score into 0. An independent re-run of the matching rule reproduces the file row
  for row.
- **The model was not reading all of it. Fixed.** The loader renames some clubs before it adds
  half-time scores (for example "Atletico Paranaense" becomes "Athletico-PR"). After the rename it
  could not place 2,303 fill rows. With the fix, the matches that training and serving read carry a
  half-time score in **95.9%** of new-format matches. That was 88.9% before the fix and 12.4% before
  the fill.
- **"Has half-time history" now works the same way in training and serving.** It is true for 97.8%
  of training rows, up from 31% without the fill. The serving check showed 0 disagreements on the 15
  real fixtures compared.
- **Still missing:** 1,328 of 32,303 loader matches (4.1%). Most of them are in Argentina (497), MLS
  (353) and China (220), for the three reasons listed under P1/P2 below. Each needs an owner
  decision, not a guess.
- **Wrong-match check:** 0 full-time mismatches among the 1,489 filled rows where we hold the
  provider's full-time score. For the other 31,770 rows the provider full-time score is not stored,
  so a direct check is **UNIDENTIFIABLE**. The indirect bound below caps wrong matches at about 10
  rows (95%).

| League | Matches (loader) | HT before fill | After fill, before fix | **After fix** | Still missing |
|---|---:|---:|---:|---:|---:|
| Argentina Primera Division | 3,624 | 7.6% | 86.3% | **86.3%** | 497 |
| Austrian Bundesliga | 1,424 | 16.0% | 96.0% | **99.6%** | 6 |
| Brazil Serie A | 2,987 | 8.8% | 89.2% | **96.4%** | 108 |
| China Super League | 1,833 | 12.5% | 39.5% | **88.0%** | 220 |
| Denmark Superliga | 1,471 | 7.0% | 99.9% | **99.9%** | 1 |
| Finland Veikkausliiga | 1,322 | 6.3% | 98.1% | **99.4%** | 8 |
| Ireland Premier Division | 1,350 | 11.3% | 99.9% | **99.9%** | 1 |
| Japan J-League | 2,637 | 8.1% | 92.1% | **98.9%** | 28 |
| K-League 1 | 1,472 | 27.0% | 100.0% | **100.0%** | 0 |
| Mexico Liga MX | 2,438 | 7.3% | 97.7% | **97.7%** | 56 |
| Norway Eliteserien | 1,881 | 7.3% | 92.0% | **99.9%** | 1 |
| Romanian Superliga | 2,290 | 22.1% | 68.1% | **98.2%** | 42 |
| Saudi Pro League | 1,920 | 25.9% | 100.0% | **100.0%** | 0 |
| Sweden Allsvenskan | 1,902 | 17.9% | 95.6% | **99.6%** | 7 |
| USA MLS | 3,752 | 10.3% | 90.6% | **90.6%** | 353 |
| **Total** | **32,303** | **12.4%** (3,991) | **88.9%** (28,707) | **95.9%** (30,975) | **1,328** |

"Loader" is `output/fd_history.parquet` passed through `data_loader.one_club_one_match` then
`data_loader._enrich_ht_scores`, the same order both loader paths use (`src/data_loader.py`
`load_all_matches` -> `_enrich_with_api_shots` step 1b). Source: `reports/ht_repair/checks.json` ->
`loader_level`.

---

## 1. Coverage at the raw history level (league x season x month)

The full table, one row per league x history season x month (1,254 rows), is
`reports/ht_repair/coverage_league_season_month.csv`. Its columns: completed fixtures, with FT, HT from the
history, HT from the fill, valid HT, coverage % before and after the fill, source provider, fill rows
with a provider fixture id, missing count, missing reasons, last collection (UTC) and FT-mismatch rows.
Per row: `reports/ht_repair/row_status.csv.gz` (42,084 rows; fixture id, rule and missing reason).

Raw `fd_history` rows of the 15 registry `model_type: new_format` leagues: **42,084**. All are played
(FT present). `fd_history` holds no unplayed fixtures, so "completed" = rows, and postponed or never-
recorded matches cannot be counted from this file.

| | Rows |
|---|---:|
| HT from the history (football-data.co.uk) | 4,011 (9.5%) |
| HT from the fill (API-Football `score.halftime`, by fixture id) | 33,259 |
| Valid HT after the fill | 37,270 (**88.6%**) |
| Missing | 4,814 |

The raw level counts the history's duplicate rows. The same match is often filed twice, under two club
spellings or one day apart. The loader merges these, which is why the loader table above is the one
the model sees.

Missing reasons (raw rows), re-derived row by row:

| Reason | Rows | Where |
|---|---:|---|
| `club_not_resolved`: a history spelling `team_names.resolve_result` cannot prove among that day's provider clubs | 3,197 | China 1,072, Romania 1,009, Brazil 289, Austria 231, Japan 201, Sweden 118, Finland 111, Norway 91, Ireland 74, Denmark 1 |
| `refused_fixture_claimed_by_two_dates`: one provider fixture claimed by history rows of two different dates (refused by rule) | 1,198 | MLS 706, Brazil 200, Argentina 180, Mexico 112 |
| `no_provider_fixture_on_date_pm1`: the provider has no fixture of that league within +-1 day | 410 | Argentina 407 (all 2023-08 to 2024-05), Romania 2, Japan 1 |
| `provider_ht_null`: fixture found, provider half-time is null, kept missing (not 0) | 9 | China 6, Romania 2, Ireland 1 |

Per league and season: `reports/ht_repair/coverage_by_league_season.csv`. Reasons by league:
`missing_reasons_by_league.csv`. Unresolved spellings with row counts: `unresolved_club_spellings.csv`
(top: Romania "FC Rapid Bucuresti" 191, "U. Cluj" 174, "Petrolul" 169; China "Shandong Taishan" 155,
"Shanghai Port" 155). The refused two-date claims are listed in `refused_claimed_by_two_dates.csv`.

**Source provider / fixture id / last collection.** History HT comes from football-data.co.uk, with no
fixture id. Fill HT comes from API-Football `/fixtures?league&season`, with a fixture id on all 33,259
fill rows. The journal `output/collect/ht_seasons.parquet` holds 120 league-seasons, all fetched
2026-11-02 15:52 UTC. The 2026-11-03 00:16 run made 0 calls and only re-matched
(`ht_fill_audit.json`: `calls: 0`).

## 2. Verification of the fill

Recomputed from `fd_history_ht_fill.parquet`, `af_ht_scores.parquet`, `fd_history.parquet` and
`af_history.parquet` (`reports/ht_repair/checks.json` -> `checks`).

| Check | Result | Fails when |
|---|---|---|
| Matched by league + date (or +-1 day) + both clubs resolved home->home, away->away; no name-only join | PASS. Rule read in `scripts/collect_ht_scores.py::match` and **re-implemented independently**: re-derived 33,259 keys = sidecar 33,259 keys; 0 differ in fixture id / HT / rule | any key or value differs |
| Ambiguity refused | PASS. >1 candidate fixture -> refused; a fixture claimed from 2 dates -> all claims refused (1,198 rows) | - |
| No fixture id claimed by history rows of two dates | PASS, 0 | > 0 |
| Duplicate fixture ids | 4,110 ids appear on 2 rows each (8,220 rows). **All are same-date copies of one match under two club spellings** that the history itself holds. Both carry the same score, and the loader merges them | a shared id with two dates (0) |
| Provider fixture ids unique; every fill id is a stored provider fixture | PASS, 0 / 0 | missing or duplicate id |
| No cross-league joins | PASS, 0. Fill league = provider league, and registry `api_football_id` = provider `league_id` | > 0 |
| HT is the provider's `score.halftime`, not inferred from FT | PASS, 0 differ | > 0 |
| Missing never written as 0 | PASS. 9 provider fixtures with null HT, 0 of them in the fill; 0 null / 0 negative HT in the fill | > 0 |
| HT <= FT (fill vs history FT) | PASS, 0 of 33,259 | > 0 |
| HT <= FT (history's own HT) | PASS, 0 of 4,011 | > 0 |
| Date gap fill vs provider | max 1 day (same day 32,551, adjacent day 708) | > 1 |
| Collector's audit JSON vs recount | PASS. Per-league `filled` and total 33,259 agree | any differ |

### FT mismatch (possible wrong match)

`af_ht_scores.parquet` stores **no full-time score**, so the provider FT for a fill row can only come
from `output/af_history.parquet` by fixture id. That covers **1,489 of 33,259** fill rows: **0
mismatches** (`reports/ht_repair/ft_mismatch.csv`, header only). For the other **31,770** rows the
direct FT check is **UNIDENTIFIABLE**: the provider FT was never stored.

Indirect evidence (the placebo row in `checks.json`). If each fill's half-time score were attached to a
random match of the same league, 30.3% of rows would break HT <= FT (20 seeded shuffles, min 29.9%).
The real rate is 0 of 33,259. If wrong matches behaved like random ones, the 95% upper bound on wrong
matches is about 3 / (0.303 x 33,259) = 0.03%, about **10 rows**. This bound is weaker for a wrong
match between the same two clubs on another date, whose scores are correlated.

## 3. The feature path reads the fill

- `src/data_loader.py::HT_SOURCES` = `af_ht_history`, `fd_history_ht_fill`. `_enrich_ht_scores` runs
  inside `_enrich_with_api_shots` on **both** loader paths, CI download and local workbook
  (lines 2071 and 1857 at e07a4591). It is league-keyed through `integrity.joins.safe_left_enrich`,
  fills missing values only, and leaves conflicting duplicates missing.
- `output/af_ht_history.parquet` **has never existed in git history**, so that source is always
  skipped (see P2).
- **Gap found and fixed (P0).** `one_club_one_match` runs *before* the HT step and renames 87 club
  spellings by same-day evidence. Of these, 47 are evidence-only aliases that `team_names.resolve`
  cannot spell-match. The sidecar is keyed on the history's original spellings, so
  `_resolve_onto_history` dropped **2,303 of 33,259** fill rows as "not provably a club of the league":
  China 901, Romania 688, Brazil 216, Japan 181, Norway 149, Sweden 77, Austria 51, Argentina 23,
  Finland 17. Effect: +2,268 loader matches with HT (28,707 -> 30,975), with 0 HT > FT, 0 half-filled
  rows, 0 ambiguous key groups, and labels consistent (`ht_over05` mismatch 0, label present where HT
  is missing 0).
- **has_ht_history.** In training, `build_features` gives a home share of 0.9783 and an away share of
  0.9784 with the fill, against 0.3108 / 0.3132 without it. The flag disagrees with
  `ht_scored_last5`-present on 0 rows. For serving parity, each league's latest match was served by
  `build_upcoming_features` from the history before it: 15 fixtures, 0 disagreeing cells across
  `home/away_has_ht_history` and `home/away_ht_scored_last5`.

## 4. Fixes made (failing test first)

| Fix | File | Test (failed before, passes after) |
|---|---|---|
| The HT step places sidecar rows on clubs the identity pipeline renamed. It uses the identity map the loader itself applied (`IDENTITY_ALIASES`, recorded from `identity_repair.canonicalise_and_dedupe`), only after `resolve()` fails, only onto a club in **that league's** pool, and refuses when ambiguous | `src/data_loader.py` (`_resolve_onto_history(aliases=)`, `_canonical_identity_and_seasons`, `_enrich_ht_scores`) | `tests/test_phase2_ht_repair.py`: `test_identity_pipeline_renames_the_old_spelling`, `test_ht_fill_keyed_on_the_history_spelling_reaches_the_renamed_club`, `test_resolve_onto_history_without_aliases_is_unchanged`, `test_ambiguous_identity_alias_is_refused`; guard `test_identity_alias_is_league_scoped_and_needs_the_club_in_the_pool` |
| Collector re-match: a history row re-evaluated this run now carries this run's answer. Before, a row that a later run *refuses* kept its stale score, and an empty match result left the whole stored file untouched. Also: `lacking_rows` on a frame with nothing lacking dropped every column (empty object mask), and `plan()` crashed | `scripts/collect_ht_scores.py` (`run`, `lacking_rows`) | `test_rematch_drops_a_stored_answer_that_is_now_refused`, `test_rematch_keeps_rows_not_re_evaluated` |
| Guard on the committed data | - | `test_committed_fill_integrity`: unique key, no null or negative HT, one date per fixture id, ids are stored provider fixtures of the same league, HT equals the provider value, HT <= history FT |

Collector fix on today's data: `run(match_only=True)` against a copy reproduces the committed sidecar
exactly (33,259 = 33,259 rows, 0 stored rows dropped, 0 calls). It is a no-op today and guards future
runs. Focused run (`test_phase2_ht_repair`, `test_collect_ht_scores`, `test_data5_truth`,
`test_resolver_joins`, `test_history_identity`, `test_team_frame_health`, `test_team6_round6`):
**77 passed, 1 failed**. The one failure is not related to this work and fails on e07a4591 without
these changes (P3).

## 5. Open issues

| P | Issue | Owner action |
|---|---|---|
| P1 | **Argentina: 407 rows, 2023-08 to 2024-05, have no fixture in API-Football league 128.** The dates match the Copa de la Liga Profesional, which is likely but **not verified** to be a separate API-Football competition. If so, `fd_history` files a domestic cup under the league, which is a wrong-league question. These rows stay missing (no cross-league join). | Decide whether these matches belong in the Argentina league model at all. If yes, approve fetching the cup competition's HT (one call per season), with the league identity kept separate. |
| P1 | **1,198 raw rows refused because one fixture is claimed from two dates** (MLS 706, Brazil 200, Argentina 180, Mexico 112). These are the history's own duplicates one day apart with the same score, and the loader merges them. Refusal is the stated rule (`test_a_fixture_claimed_by_two_history_rows_fills_neither`). It accounts for most of the 497 Argentina / 353 MLS / 56 Mexico loader matches still missing. | Approve or reject a narrower rule: accept only when every claiming row has the **same FT** and the dates are within +-1 day (the loader's own one-match rule). Better still, approve storing the provider FT (next row) so each claim can be checked against it. |
| P2 | **Provider FT is not stored** (`af_ht_scores` has no FT), so the direct wrong-match check covers 4.5% of the fill. | Approve adding `score.fulltime` to `parse()`. It is free on future fetches; re-fetching the stored seasons costs about 120 calls. |
| P2 | **3,197 raw rows with a club spelling no rule proves** at collector level (see `unresolved_club_spellings.csv`). After the loader fix most of these matches get HT through their other-spelling copy. What remains is China 220 and small counts elsewhere. The loader logs 47 aliases proven by same-day evidence only. | Add alias-store entries with fixture-level evidence for the listed spellings. Never by name similarity. |
| P2 | `output/af_ht_history.parquet` is listed in `HT_SOURCES` and read by `update_results.py` and the Live page, but has never been committed. | Produce it, or remove it from the source lists. |
| P2 | The same identity-rename gap probably affects other sidecars keyed on a source's own spelling (`af_stats_fill`, `af_history`). This was not measured here, and the fix is limited to the HT step. | Approve a measurement and the same fix for those joins. |
| P2 | HT features change materially: has_ht_history goes from 31% to 97.8%, and HT labels rise from 3,991 to 30,975 (7.8x). Any model trained before this fill cannot be compared on HT inputs. | Retrain only through GitHub Actions (owner decision). Nothing was retrained here. |
| P3 | `tests/test_team6_round6.py::test_workflow_has_a_guarded_insidebox_step_and_keeps_the_run_day_override` fails on e07a4591. It still expects the one-day override that commit 12864791 removed as scheduled. | Approve updating the test to the post-override expectation. I did not change or disable it. |
| P3 | "Completed fixtures" can only be counted from history rows. `fd_history` holds played matches only, so an unrecorded or postponed match is invisible here. | None now. Use the fixtures calendar if schedule completeness is needed. |

## Reproduce

```
python scripts/audit/ht_history_repair_audit.py          # writes reports/ht_repair/*, ~2 min
python -m pytest -p no:cacheprovider -q tests/test_phase2_ht_repair.py tests/test_collect_ht_scores.py
```

Sources: `output/fd_history.parquet` (c9211b1f), `output/collect/fd_history_ht_fill.parquet`,
`af_ht_scores.parquet`, `ht_seasons.parquet`, `ht_fill_audit.json` (0efc078a), `output/af_history.parquet`,
`src/data_loader.py`, `src/feature_engineering.py`, `scripts/collect_ht_scores.py` at e07a4591.
