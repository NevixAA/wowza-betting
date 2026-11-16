# Restored af_history provider rows: the frame change (2026-11-16)

> **For the bars agent.** The clean-model bars re-run that is in progress uses the frame **without** these rows. The Nov 20 confirm-diff must account for the hashes below. **Standard is unchanged. Only New-Format changed.**

**What changed.** The owner decided on 2026-11-16: "yes, make the models use them".
- **The source.** The loader now reads the af_history provider rows that the old write-back replaced. PR #35 restored them from git, and this change verifies them before use. They are a separate, labelled source: `output/corrections/af_history_restored_provider_rows_2026-11_verified.parquet`. Every row carries `provenance`, `verified_by` and `restored_from_commit`.
- **The fill.** It only touches cells that are missing (`nan_only`). An existing value is never overwritten, and a cell with no verified value stays missing. `af_history.parquet` is not rewritten.
- **Where it runs.**
  - On data-truth-100, `src/data_loader._enrich_af_restored_rows` is the last af fill, after `--insidebox` and `--fullstats`. A fixture-id-proven provider answer always wins.
  - On main, the fill runs right after step 1.
- **No training.** Nothing was retrained or promoted. The Nov 19 refit picks this up.

## Verification (`scripts/verify_af_history_restored_rows.py`, no API call)

**Rows**
- 28,412 rows were restored and 21,686 were verified (served).
- 6,726 rows are excluded:
  - 6,724 are copies that the owner's af_history duplicate rule removed (`af_history_removed_duplicates.csv`);
  - 2 have an FT score that disagrees with API-Football: China 2020-11-10 Dalian–Guangzhou R&F (restored 0-4, API-Football 0-3) and Ireland 2022-08-29 Sligo–Dundalk (restored 2-0, API-Football 0-3).
- **Row identity.** All 21,686 served rows have an FT score equal to API-Football's FT for the fixture, either by fixture id or by provider names and date.
  - Every row had a second source; no row lacked one.
  - football-data was also checked on 21,505 rows and disagreed on 1.

**Cells checked against API-Football `/fixtures/statistics`.** The comparison uses the fixture-id-proven `--fullstats` answer on the same key. A cell that differs is excluded, and impossible values were checked too (0 found). 439 cells are excluded in total; they are listed in `..._excluded.csv`.

| col | present | API-Football checked | exact match | mismatch (excluded) | no 2nd source |
|---|---:|---:|---:|---:|---:|
| HS / AS | 27,138 | 19,994 | 99.51% / 99.55% | 97 / 90 | 1,470 |
| HST / AST | 27,128 | 19,986 | 99.76% / 99.74% | 47 / 52 | 1,469 |
| HC / AC | 26,320 | 19,968 | 99.90% / 99.93% | 19 / 14 | 680 |
| HF / AF | 26,326 | 19,971 | 99.86% / 99.82% | 28 / 35 | 681 |
| HY / AY | 25,539 | 19,364 | 99.87% / 99.87% | 25 / 26 | 6,175 |
| HR / AR | 6,274 / 6,285 | 4,923 / 4,927 | 100% / 100% | 0 | 1,351 / 1,358 |
| HXG / AXG | 333 | 287 | 100% | 0 | 46 |
| HIB / AIB | 492 | 433 | 99.31% | 3 / 3 | 59 |

- **The football-data column.** It agreed 100% on every stat it holds, but it is **not independent** for New-Format statistics: football-data's own CSVs for these leagues carry no corners, fouls or shots, so those `fd_history` columns share the API-Football lineage. Only its score is independent.
- **Cells with no second source.** These are kept because the row's identity is proven. They are the provider's own original answer, written by `backfill_af_history`.
- **The loader-copy cells** (27,909 xG, inside-box and shots cells in `..._writeback_cells_2026-11.parquet`) are **not served**. Only 2,129 of them equal an API-Football value, and that value already fills the cell. The rest differ (6,101) or have no provider value (19,679), so they stay excluded.

## Effect on the frames

The snapshot was taken offline with the network blocked: `load_all_matches(force=True)` and then `build_features`. Each sha256 is `pd.util.hash_pandas_object` over the frame with its columns sorted.

**data-truth-100 @ 258abac30 + this change** (71,583 rows)

| frame | before | after |
|---|---|---|
| matches, Standard | `6480e2af6c4db2cadf75d956b6363598791f4c13ad38cd2d45b16e55f978179c` | **unchanged** |
| matches, New-Format | `8dcf6664bd3309a5024c92b3bff1914e7a6a6feb1b249ebf03d8ad61c7c88f6b` | `cf144aba91bf26950a1cdc12014fa26c9f24251e18d2f5c75b36d5175f47acff` |
| features, Standard | `74f51aa8e3faffc29424697015c3c092e0d332c6ee79577ac2b05ebed0e0ded1` | **unchanged** |
| features, New-Format | `e8fe6b5853ccfcb9d4645cd173acf8cb5ff96b440bc5abb6b7ef3d63010a3615` | `df13e2b72049f25ba3c2d011bdaee347dcbaf51c8e32743e7d501040ad8f0f42` |
| props source / frame | `10c8a5a8…` / `37bf21c6…` | **unchanged** (the props frame does not read the team loader) |

- **Matches frame cell diff:** **0 cells overwritten, 0 erased, 0 columns added or removed.** Cells filled, missing before:
  - yellows: home 19,194, away 19,193;
  - shots: home 732, away 731;
  - SoT: home 732, away 731, plus their row SoT ratios;
  - inside-box: 397 each;
  - xG: 262 each;
  - corners and fouls: 0, because `--fullstats` already serves them.
- **Fills by league** run from USA MLS (5,094) down to Sweden (1,122), across all 15 New-Format leagues, and cover seasons 2019 through 2026.
- **Features frame.** Raw changes come only from fills. The derived form windows move with them:
  - shots_last5: about 520 changed and 880 newly present per side;
  - xg_last5: 403 changed;
  - insidebox_last5: about 615 changed;
  - sot ratios: 859 combined.

**main @ d73977e3b + this change** (74,944 rows). main serves the written-back rows' own shots, so the fills fall elsewhere.
- **Hashes.**
  - matches New-Format: `da7bfffd…0e` → `3375ad93…359`;
  - features New-Format: `51054131…841` → `5f9f49fb…a27`;
  - Standard: unchanged.
- **Filled cells:**
  - corners: 5,301 / 5,300;
  - fouls: 5,297 / 5,295;
  - yellows: 16,374 / 16,377;
  - shots and SoT: 38 each;
  - xG: 7 each;
  - inside-box: 8 / 9.
- **Overwritten or erased: 0.**

## Train/serve parity (data-truth-100)

- **Team.** `scripts/team_train_serve_parity.py <out> model_v9_{standard,newformat,btts,over15,over35}.pkl 6`
  - Before: 0 mismatched cells; 125/125 probabilities exact per payload.
  - After: 0 mismatched cells; 125/125 probabilities exact per payload.
- **Props.** The props frame hashes are identical before and after. `tests/test_props_train_serve_parity_2026_11_09.py` and `tests/test_props_owner_rules_2026_11_08.py` pass: 42 passed.

Hashes depend on the committed data at the time they were taken. To compare later, run the same snapshot with and without the verified file. If the file is absent, the loader skips the fill.
