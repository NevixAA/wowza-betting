# Restored af_history provider rows: the frame change (2026-11-16)

> **For the bars agent.** The clean-model bars re-run that is in progress uses the frame **without** these rows. The Nov 20 confirm-diff must account for the hashes below. **Standard is unchanged on both branches; only New-Format changed.**

**What changed**

- **Owner decisions, 2026-11-16**
  - "yes, make the models use them".
  - Cells with no independent second source are **excluded** ("Exclude them").
- **The source.** The loader reads the af_history provider rows that the old write-back replaced, restored from git (PR #35/#36) and verified here. It is a separate, labelled file: `output/corrections/af_history_restored_provider_rows_2026-11_verified.parquet`. Every row carries `provenance`, `verified_by` and `restored_from_commit`.
- **What it fills.** Only cells that are missing (`nan_only`). An existing value is never overwritten. A cell without a confirmed value stays missing.
- **Where it runs**
  - data-truth-100: `src/data_loader._enrich_af_restored_rows`, the last af fill (after `--insidebox` / `--fullstats`).
  - main: right after step 1.
- **Not touched.** `af_history.parquet` is not rewritten. Nothing was retrained or promoted; the Nov 19 refit picks this up.

## Verification (`scripts/verify_af_history_restored_rows.py`, no API call)

**Rows**

- **Restored:** 28,412.
- **Verified:** 21,686. Every verified row has an FT score equal to API-Football's FT for the fixture, matched by fixture id or by provider names and date.
- **Excluded:** 6,726.
  - 6,724 are copies removed by the owner's af_history duplicate rule.
  - 2 have a score that differs from API-Football:

| match | restored score | API-Football score |
|---|---|---|
| China 2020-11-10 Dalian–Guangzhou R&F | 0-4 | 0-3 |
| Ireland 2022-08-29 Sligo–Dundalk | 2-0 | 0-3 |

**Cells.** A statistic is kept only when API-Football's fixture-id-proven `/fixtures/statistics` answer (`--fullstats`, same key) holds the **same value**.

- Cells where the values differ are excluded: **437**, reason `!= API-Football`.
- Cells with no API-Football value are excluded: **2,950**, reason `no second source`.
- Impossible values: **0**.
- Every excluded cell is listed in `..._excluded.csv`.
- football-data is reported, not counted: for these leagues its statistic columns share the API-Football lineage, so it is not independent. Only its scores are independent: it was checked on 21,505 of the served rows and disagreed on 1.

Counts below are on the 21,686 verified rows:

| col | present | API-Football checked | exact match | `!= API-Football` | `no second source` | served |
|---|---:|---:|---:|---:|---:|---:|
| HS / AS | 20,580 | 19,846 | 99.51% / 99.55% | 97 / 90 | 734 / 734 | 19,749 / 19,756 |
| HST / AST | 20,572 | 19,838 | 99.76% / 99.74% | 47 / 52 | 734 / 734 | 19,791 / 19,786 |
| HC / AC | 19,820 | 19,820 | 99.91% / 99.93% | 18 / 14 | 0 | 19,802 / 19,806 |
| HF / AF | 19,823 | 19,823 | 99.86% / 99.83% | 28 / 34 | 0 | 19,795 / 19,789 |
| HY / AY | 19,221 | 19,221 | 99.87% / 99.86% | 25 / 26 | 0 | 19,196 / 19,195 |
| HR / AR | 4,895 / 4,899 | all | 100% | 0 | 0 | 4,895 / 4,899 |
| HXG / AXG | 289 | 285 | 100% | 0 | 4 / 4 | 285 / 285 |
| HIB / AIB | 434 | 431 | 99.30% | 3 / 3 | 3 / 3 | 428 / 428 |

**Loader copies.** The 27,909 loader-copy cells (xG, inside-box and shots in `..._writeback_cells_2026-11.parquet`) are **never served**.

- 2,129 equal an API-Football value. That value already fills the cell.
- 6,101 differ from the API-Football value.
- 19,679 have no provider value.

## Effect on the frames

**How the snapshot was taken:** offline, network blocked, `load_all_matches(force=True)` then `build_features`. Each hash is the sha256 of `pd.util.hash_pandas_object` over the frame with its columns sorted. Hashes depend on the committed data at the time; to reproduce, compare the same commit with and without the verified file (an absent file means no fill).

### data-truth-100

Commit 7d74553a4 (after #36), plus this change; 71,583 rows.

| frame | before | after |
|---|---|---|
| matches, Standard | `6480e2af6c4db2cadf75d956b6363598791f4c13ad38cd2d45b16e55f978179c` | **unchanged** |
| matches, New-Format | `8dcf6664bd3309a5024c92b3bff1914e7a6a6feb1b249ebf03d8ad61c7c88f6b` | `c66d8939db14de6162f6fb731403caeb9dbe4263d71a93dd3cf78e9e64a28180` |
| features, Standard | `74f51aa8e3faffc29424697015c3c092e0d332c6ee79577ac2b05ebed0e0ded1` | **unchanged** |
| features, New-Format | `e8fe6b5853ccfcb9d4645cd173acf8cb5ff96b440bc5abb6b7ef3d63010a3615` | `96db70d0f0b43910d087a6eada354443c8e1118ca28c16553d8a31db5770e50c` |
| props source / frame | `10c8a5a89bb7…` / `37bf21c68d4f…` | **unchanged** (the props frame does not read the team loader) |

**Matches cell diff**

- **0** cells overwritten, **0** erased, **0** columns added or removed.
- Filled (missing before):

| column | home | away |
|---|---:|---:|
| yellows | 19,194 | 19,193 |
| inside-box | 394 | 394 |
| xG | 258 | 258 |

- Shots, SoT, corners and fouls: **0**. Every confirmed value is the one `--fullstats` already serves.
- Fills by league: USA MLS 5,076 at the top, Ireland 939 at the bottom, across all 15 New-Format leagues; seasons 2019 to 2026.

**Features.** Derived windows changed:

| feature | home | away |
|---|---:|---:|
| xg_last5 | 395 | 399 |
| insidebox_last5 | 607 | 600 |

Yellows feed no team feature today.

### main

Commit e4f437be7, plus PR #37; 74,997 rows. main serves the written-back rows' own shots and has no `--fullstats` fill, so the fills land elsewhere.

| frame | before | after |
|---|---|---|
| matches, Standard | `fd5c4b6ae9ea3d90bfc89f8d4eabbd4f7fdd881a1f002bf8efb18fda03270b0b` | **unchanged** |
| matches, New-Format | `022a17b79c7b4e4a8332ccb4300b1a8ebaac49e945b2a95a6fa105ff5b9c9483` | `9308880905fd9f847503c5f04ca6bf53d49645fe71c423efea0a3f0efb681a66` |
| features, Standard | `39a0ced5754f38ba081888584fa614f605fc241aabdffaabd253eb1ad9ffbb6b` | **unchanged** |
| features, New-Format | `c3153f549e379a299e8e7e4f356f19993d987410dfa355424e78cf2f40113fbd` | `56f733a40814c5c0dd2f4d9ad4e81948565156befd322e31ebe1c331639d6684` |

Filled (missing before); **0** overwritten or erased:

| column | home | away |
|---|---:|---:|
| corners | 5,301 | 5,300 |
| fouls | 5,297 | 5,295 |
| yellows | 16,374 | 16,377 |
| shots and SoT | 38 | 38 |
| xG | 4 | 4 |
| inside-box | 6 | 7 |

## Train/serve parity (data-truth-100)

- **Team** (`scripts/team_train_serve_parity.py <out> model_v9_{standard,newformat,btts,over15,over35}.pkl 6`):
  - Before: 0 mismatched cells; 125/125 probabilities exact per payload.
  - After: 0 mismatched cells; 125/125 exact per payload.
- **Props:** frame hashes are identical before and after. `tests/test_props_train_serve_parity_2026_11_09.py` and `tests/test_props_owner_rules_2026_11_08.py` pass.
