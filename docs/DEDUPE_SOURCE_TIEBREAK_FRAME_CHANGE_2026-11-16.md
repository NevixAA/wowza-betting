# Duplicate-fixture tie-break: the frame change (2026-11-16)

> **For the bars agent.** The Nov 20 confirm-diff must account for the hashes below. **No cell value changed in either frame.** Only the order of rows that share a match date changed. Hashes taken over the frame as served (row order included) therefore differ; hashes taken after sorting by fixture key do not.

This is separate from the restored af_history rows change (`docs/AF_RESTORED_ROWS_FRAME_CHANGE_2026-11-16.md`). The two changes are independent.

**What changed.** This is a port of main PR #38 (931c7671d) to data-truth-100, in `src/data_loader.py`:
- **`_canonicalise_clubs` tie-break.** It keeps the most complete copy of a same-day duplicate fixture. On a completeness tie it used to keep whichever copy sorted last, so the survivor depended on input order. Now the copy whose `row_source` ranks first wins. The ranking is the one-match rule data-truth-100 already applies, `src.integrity.duplicate_matches.SOURCE_PRIORITY`: football-data, then its cache, then API-Football, then the backtest archive, with an unknown source last. Completeness still decides first. With no `row_source` column (direct callers), the old rule holds.
- **Stable date sort.** `load_all_matches` now sorts the merged frame by date with a stable sort (`kind="mergesort"`) before the identity pipeline. Rows on the same date keep their concat order, so the order never depends on unrelated rows.
- **No helper column is left behind.** data-truth-100 already tags every row with `row_source`, so the main-only `_source_priority` column is not needed.

**Why main's bug applies here.** Before the fix, the input order came from an unstable (quicksort) date sort over the whole merged frame. `tests/test_dedupe_source_tiebreak.py` reproduces it on the unfixed code: 4 of its 7 tests failed (the archive copy won when it came second), and all 7 pass after the fix.

## Ties in the current data

The counts come from the frame `_canonicalise_clubs` de-duplicates, captured in an offline `load_all_matches(force=True)`.

| | count |
|---|---:|
| rows into the de-duplication | 105,446 |
| overlapping fixtures (same date, league, clubs) | 23,696 |
| of which cross-source (fd_cache + backtest archive, incl. 983 with two fd_cache copies) | 17,210 |
| completeness ties, cross-source | **1**: Antwerp v Beerschot VA, 2024-09-29 (fd_cache vs backtest archive, identical values) |
| completeness ties, fd_cache + fd_cache | 2: Preston v Sheffield United 2026-11-03, Kortrijk v Mechelen 2026-11-04 (identical values) |
| completeness ties, archive + archive | 3,589: they differ only in the stored `season` label, which `_canonical_identity_and_seasons` re-derives from the date right after |

**Surviving copy changed: 0 fixtures.** Before the fix, Antwerp v Beerschot VA already resolved to the fd_cache copy by input-order luck. That is the copy main lost on 2026-11-16, when unrelated rows flipped it. Now the tie is decided by source and cannot flip. The same-source ties carry the same values, or a season label that is re-derived, so they cannot change a served cell.

## Effect on the frames

The snapshot was taken offline with the network blocked: `load_all_matches(force=True)`, then `build_features`. Each hash is the sha256 of `pd.util.hash_pandas_object(index=True)` over the frame with its columns sorted. This is the same method as the af-restored note.

The before frame is data-truth-100 @ 7d74553a4, and the before hashes were reproduced bit for bit in a second, clean worktree. The after frame is the same commit plus this change. Both have 71,583 rows, the same fixture keys, and the same date sequence.

**Hashes as served (row order included):**

| frame | before | after |
|---|---|---|
| matches, Standard | `6480e2af6c4db2cadf75d956b6363598791f4c13ad38cd2d45b16e55f978179c` | `7b5c3653fd3750a1d32fdad21bbf9280f5158852e0a29ff3b303c8d3bda26420` |
| matches, New-Format | `8dcf6664bd3309a5024c92b3bff1914e7a6a6feb1b249ebf03d8ad61c7c88f6b` | `f66e4500ef2398c21d8a857b28649412008f50292f6e2fc121b80e9e5312b640` |
| features, Standard | `74f51aa8e3faffc29424697015c3c092e0d332c6ee79577ac2b05ebed0e0ded1` | `34bc47ae6d6f6a0c5ba53d88ecbf990d4c58125fd0d8069df4e350e319ffa9e3` |
| features, New-Format | `e8fe6b5853ccfcb9d4645cd173acf8cb5ff96b440bc5abb6b7ef3d63010a3615` | `d045131fde03baca2581d3952890dd09c2d8931108addbb2a40e2c27873756b8` |

The before hashes equal the "before" column of the af-restored note (taken @ 258abac30), so both notes share one baseline.

**Hashes after sorting by (date, league, home_team, away_team): unchanged.**

| frame | before = after |
|---|---|
| matches, Standard | `124025f0e62c96541b699b516dffe0750f1f86bd0d8e546786b5a57a83574e45` |
| matches, New-Format | `d9715034d3be440d5540fbc2e0bbd5857f2df6566fe09f6f42004bec0ea3737f` |
| features, Standard | `d8d6531073ef9711d77920e5b36f679672ed6a624955beadcb6c3e4ef02af1e3` |
| features, New-Format | `47273942a482636371d740f74a21e87eed3a362a1886edf79bd9fda0961e8f53` |

- **Key-aligned cell diff:** 0 cells changed in either frame, and that includes `row_source`.
- **Row positions that moved within their date:**
  - matches: 36,426 Standard and 25,646 New-Format;
  - features: 36,429 and 25,648.
- **For the confirm-diff.** Compare key-sorted frames. A raw-order hash will differ even though nothing a model reads per row has changed. A fit that depends on row order within a date may still differ slightly, for example through a seeded subsample or a positional split. Attribute that to this change, not to data.
- **Combining with the af-restored change.** If both changes are merged, the New-Format content change is the af-restored fill alone. This change adds only the within-date reorder.

## Train/serve parity (data-truth-100 + this change)

`scripts/team_train_serve_parity.py <out> model_v9_{standard,newformat,btts,over15,over35}.pkl 6`:
- **Cells.** The check covered 125 fixtures over the last 6 dates (2026-11-01 to 2026-11-06) and 5,375 raw feature cells. **0 cells mismatched.**
- **Probabilities.** 125 of 125 were exact for each of the 5 payloads (max |diff| 0.0 pp).

No API call, no training, and no promotion.
