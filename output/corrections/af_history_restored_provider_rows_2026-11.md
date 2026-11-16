# af_history provider rows restored from git (2026-11-16)

**What broke.** `src/data_loader._enrich_with_api_shots` ended with a write-back ("BANK THE ENRICHMENT", added 2026-09-22) that `retrain.yml` committed. The write-back ran `concat([stored, new]).drop_duplicates(key, keep="last")`, and `new` carried only HS/AS/HST/AST/HXG/AXG/HIB/AIB. Every stored provider row whose key the loader touched was replaced by a shots-only row. That row's full-time score, corners, fouls and cards went from present to missing. From PR #34 on it would also have dropped `fixture_id`.

**The fix.**
- On main, the write-back merges by key and only fills missing cells of the columns it owns (`src/integrity/preserving_upsert.py`, `data_loader.bank_af_enrichment`).
- On data-truth-100, the loader no longer writes af_history at all (fix round 9).
- On both branches, the other writers (`backfill_af_history`, the results sync from main, `collect_common.append_parquet` and the sidecar collectors) no longer drop columns or rows they do not own.
- `retrain.yml` stays disabled; it was not run.

**Damage found.** The scan walked every committed version of `output/af_history.parquet` and looked for cells that were present in an earlier version and missing in the current file. Both branches give the same counts:

| column | cells lost | restored here |
|---|---:|---:|
| FTHG / FTAG / FTR (score) | 28,412 each | 28,412 rows |
| HC / AC (corners) | 26,320 each | 26,320 |
| HF / AF (fouls) | 26,326 each | 26,326 |
| HY / AY (yellows) | 25,539 each | 25,539 |
| HR / AR (reds) | 6,274 / 6,285 | 6,274 / 6,285 |
| HXG / AXG | 7,714 each | 4 in provider rows + 7,710 listed cell by cell |
| HIB / AIB | 6,216 each | 2 in provider rows + 6,214 listed cell by cell |
| HS / AS / HST / AST | 20 / 17 / 17 / 7 | listed cell by cell |

- **Rows affected:** 28,412 provider rows across 15 new-format leagues. The largest are USA MLS (3,105), Argentina (2,634) and Brazil (2,559). 26,177 of the rows come from the original backfill commit f7000733b.
- **Writers:** every loss is in one of the 19 `auto: main-model retrain` commits, from 6bf25b233 (2026-09-23, 26,177 rows at once) to 268b6826a (2026-11-01). No other writer lost a cell.

**Where the restored data is (beside the original, per the corrections rule).** `output/af_history.parquet` is **not modified**. On data-truth-100, `src/af_history_provenance` marks a row as written back by "no FT score". Rewriting scores in place would silently re-label 28,412 rows and serve their unproven shot values.
- `af_history_restored_provider_rows_2026-11.parquet` holds each provider row exactly as last committed before the write-back replaced it, with `restored_from_commit` and its date.
- `af_history_restored_writeback_cells_2026-11.parquet` holds the cells the write-back erased from rows that were never provider rows. These are earlier loader copies of xG, inside-box and shots, which are unproven. Each cell has its key, column, value and source commit.
- `af_history_restored_provider_rows_2026-11_audit.json` holds the counts on this branch's file.

**Not restored, on purpose.**
- The 3,841 misfiled-league rows removed by fc6b8bab6 (2026-11-01).
- The duplicates removed under the owner's rule (fix14).

**Not yet served.** No loader reads these files yet. Wiring them into the history, for example as a nan-only fill like the `--fullstats` sidecar, changes training inputs. That is the owner's decision.

Rebuild or verify with `python scripts/restore_af_history_provider_rows.py [--check]`. It reads git only and makes no API call.
