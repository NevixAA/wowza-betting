"""
history_refetch.py - the props pipeline's reader of the re-fetched player counts (owner 2026-11-04,
fix round 8, props item 3).

WHY. player_history.parquet and the UEFA cup rows were parsed by `int(x or 0)`: every per-player
count holds 0 where API-Football sent null, so a real 0 and a missing value cannot be told apart in
those files. scripts/collect_player_refetch.py re-reads /fixtures/players per fixture and stores the
counts under the owner's null rule (player_model.player_stat_nulls) BESIDE the history, in
output/collect/player_history_refetch.parquet, with the original values kept there as orig_<col>.

THE RULE HERE. For a row whose (fixture_id, player_id) was re-fetched with player and fixture proven,
the re-fetched value REPLACES the stored one - including a missing value replacing a suspect 0. This
is not a nan-only fill on purpose: the stored 0 is the suspect value, so filling only blanks would
keep every wrong 0. A row the collect parsed from a raw provider answer WITH the rule (count_source
'provider_rule', written by player_model.data_fetcher._parse_fixture_player since fix round 9,
2026-11-04) is verified as stored: it keeps its values and its 'provider_rule' label and is never
re-fetched. Every other row - written by the old `or 0` parser before the fix - keeps its stored
values and is marked 'zeros_unverified' in `count_source`, so the Phase 2 props retrain can exclude
it (owner 2026-11-04). Nothing is written back: player_history.parquet is never rewritten here.

NOT HERE. Stored derived columns (rolling per-game rates, target_* labels) were computed from the
old values; serving recomputes its rolling features from the raw rows, and the Phase 2 retrain must
rebuild features and labels from these raw counts. This reader changes raw counts only.
"""
from __future__ import annotations

import pandas as pd

from . import config
from .player_stat_nulls import (HISTORY_COUNT_COLS, PARSED_WITH_RULE, REFETCHED, SOURCE_COL,
                                UNVERIFIED)

REFETCH_FILE = config.OUTPUT_DIR / "collect" / "player_history_refetch.parquet"
#: the replaced columns: every per-player count player_history stores + rating (not a count; an
#: unrated appearance was stored as 0 by the same bug)
REPLACED = list(HISTORY_COUNT_COLS) + ["rating"]
STATUS_COL = SOURCE_COL


def load_refetch(path=None) -> pd.DataFrame:
    p = REFETCH_FILE if path is None else path
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(p)
    except Exception as e:                                              # noqa: BLE001
        print(f"[refetch] {p.name} unreadable ({e}): not used, every row stays {UNVERIFIED}")
        return pd.DataFrame()


def with_refetched(history: pd.DataFrame, refetch: pd.DataFrame | None = None) -> pd.DataFrame:
    """history with the re-fetched counts in place of the stored ones where (fixture_id, player_id)
    was re-fetched, and `count_source` on every row: 'refetched' | 'provider_rule' (parsed with the
    rule, kept as stored) | 'zeros_unverified'."""
    if history is None or history.empty:
        return history
    rf = load_refetch() if refetch is None else refetch
    out = history.copy()
    stored = out[STATUS_COL] if STATUS_COL in out.columns else pd.Series(None, index=out.index, dtype=object)
    ruled = stored.eq(PARSED_WITH_RULE)
    out[STATUS_COL] = UNVERIFIED
    out.loc[ruled, STATUS_COL] = PARSED_WITH_RULE
    if rf is None or rf.empty or not {"fixture_id", "player_id"} <= set(out.columns):
        return out
    rf = rf.assign(fixture_id=pd.to_numeric(rf["fixture_id"], errors="coerce"),
                   player_id=pd.to_numeric(rf["player_id"], errors="coerce"))
    rf = rf.dropna(subset=["fixture_id", "player_id"]).drop_duplicates(["fixture_id", "player_id"], keep="last")
    rf = rf.set_index(["fixture_id", "player_id"])
    key = pd.MultiIndex.from_arrays([pd.to_numeric(out["fixture_id"], errors="coerce"),
                                     pd.to_numeric(out["player_id"], errors="coerce")])
    # a row parsed with the rule is the newer reading of the same answer: a re-fetch never overrides it
    hit = key.isin(rf.index) & ~ruled.to_numpy()
    if not hit.any():
        return out
    sel = rf.reindex(key[hit])
    for c in REPLACED:
        if c not in rf.columns:
            continue
        col = pd.to_numeric(out[c], errors="coerce").astype(float) if c in out.columns \
            else pd.Series(float("nan"), index=out.index)
        col.loc[hit] = pd.to_numeric(sel[c], errors="coerce").astype(float).to_numpy()
        out[c] = col
    out.loc[hit, STATUS_COL] = REFETCHED
    print(f"[refetch] {int(hit.sum()):,} of {len(out):,} row(s) carry re-fetched counts; "
          f"{int((out[STATUS_COL] == PARSED_WITH_RULE).sum()):,} parsed with the rule; "
          f"{int((out[STATUS_COL] == UNVERIFIED).sum()):,} marked {UNVERIFIED}")
    return out


# ── proof (read-only): does the reader apply the re-fetch exactly? ───────────────────────────────

#: the props markets as labels (props_harness.LABEL_RULE, the same reading build_features uses)
PROOF_MARKETS = {"sot": ("shots_on_target", 1), "sot2": ("shots_on_target", 2), "goals": ("goals", 1),
                 "assists": ("assists", 1), "cards": ("yellow_cards", 1)}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def _key(df: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays([pd.to_numeric(df["fixture_id"], errors="coerce"),
                                      pd.to_numeric(df["player_id"], errors="coerce")])


def _same(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a == b) | (a.isna() & b.isna())


def refetch_proof(before: pd.DataFrame, refetch: pd.DataFrame, journal: pd.DataFrame | None = None) -> dict:
    """PROOF that with_refetched() puts the re-fetched counts in place of the stored ones, computed
    on `before` (the history rows as stored, `count_source` as stored) and the re-fetch sidecar.
    Nothing is written. Per field: how the stored value changed (stored 0 -> a value, 0 -> 0
    confirmed, 0 -> missing under the null rule, a value -> another value / missing / the same);
    per props market: how the label changed; and the checks behind `ok`:
      * every re-fetched row carries exactly the sidecar's value (`applied_exactly`);
      * the sidecar's orig_<col> equals the stored value (the original is kept beside, unchanged);
      * every sidecar row has its history row; every row NOT re-fetched keeps its stored values and
        is 'provider_rule' or 'zeros_unverified';
      * every row of a journaled fixture that the answer refused stays 'zeros_unverified', unchanged."""
    after = with_refetched(before, refetch)
    st = after[STATUS_COL]
    hit = st.eq(REFETCHED).to_numpy()
    rf = refetch.assign(fixture_id=pd.to_numeric(refetch["fixture_id"], errors="coerce"),
                        player_id=pd.to_numeric(refetch["player_id"], errors="coerce")) \
        .drop_duplicates(["fixture_id", "player_id"], keep="last").set_index(["fixture_id", "player_id"])
    key = _key(before)
    sel = rf.reindex(key[hit])
    out: dict = {"rows": int(len(before)), "refetch_rows": int(len(rf)), "rows_replaced": int(hit.sum()),
                 "refetch_rows_without_history_row": int((~rf.index.isin(key)).sum()),
                 "count_source_before": (before[STATUS_COL].fillna("(none: old parser)").value_counts().to_dict()
                                         if STATUS_COL in before.columns else {}),
                 "count_source_after": st.value_counts().to_dict(), "fields": {}, "markets": {}}
    played = (_num(sel["minutes_refetch"]) > 0).to_numpy() if "minutes_refetch" in sel.columns else None
    for c in REPLACED:
        if c not in rf.columns:
            continue
        old = _num(before[c])[hit] if c in before.columns else pd.Series(float("nan"), index=before.index[hit])
        new = _num(after[c])[hit]
        side = pd.Series(_num(sel[c]).to_numpy(), index=new.index)
        orig = (pd.Series(_num(sel[f"orig_{c}"]).to_numpy(), index=new.index)
                if f"orig_{c}" in sel.columns else None)
        z = old == 0
        f = {"rows": int(hit.sum()),
             "applied_exactly": int(_same(new, side).sum()),
             "orig_kept_equal_to_stored": None if orig is None else int(_same(orig, old).sum()),
             "old0_now_value": int((z & (new > 0)).sum()),
             "old0_now_0_confirmed": int((z & (new == 0)).sum()),
             "old0_now_missing": int((z & new.isna()).sum()),
             "oldvalue_now_same": int(((old > 0) & (new == old)).sum()),
             "oldvalue_now_different_value": int(((old > 0) & new.notna() & (new != old)).sum()),
             "oldvalue_now_missing": int(((old > 0) & new.isna()).sum()),
             "old_missing": int(old.isna().sum())}
        if played is not None:
            f["now_missing_player_played"] = int((new.isna().to_numpy() & played).sum())
            f["now_missing_player_did_not_play"] = int((new.isna().to_numpy() & ~played).sum())
        out["fields"][c] = f
    for m, (c, k) in PROOF_MARKETS.items():
        old = _num(before[c])[hit]
        new = _num(after[c])[hit]
        lo = (old >= k).astype(float).where(old.notna())
        ln = (new >= k).astype(float).where(new.notna())
        out["markets"][m] = {"rule": f"{c} >= {k}", "rows": int(hit.sum()),
                             "0_to_1": int(((lo == 0) & (ln == 1)).sum()),
                             "1_to_0": int(((lo == 1) & (ln == 0)).sum()),
                             "0_stays_0": int(((lo == 0) & (ln == 0)).sum()),
                             "1_stays_1": int(((lo == 1) & (ln == 1)).sum()),
                             "label_now_missing": int(ln.isna().sum()),
                             "positives_before": int((lo == 1).sum()), "positives_after": int((ln == 1).sum())}
    rest = ~hit
    unchanged = {c: int((~_same(_num(after[c])[rest], _num(before[c])[rest])).sum())
                 for c in REPLACED if c in before.columns}
    out["not_refetched_rows"] = {"rows": int(rest.sum()), "values_changed": unchanged,
                                 "count_source": st[rest].value_counts().to_dict()}
    if journal is not None and len(journal):
        jf = set(pd.to_numeric(journal["fixture_id"], errors="coerce").dropna().astype(int))
        fid = pd.to_numeric(before["fixture_id"], errors="coerce")
        stored_src = before[STATUS_COL] if STATUS_COL in before.columns \
            else pd.Series(None, index=before.index, dtype=object)
        in_j = fid.isin(jf).to_numpy() & ~stored_src.eq(PARSED_WITH_RULE).to_numpy()
        refused = in_j & ~hit
        out["journaled_fixtures"] = {
            "fixtures": len(jf), "rows_to_verify": int(in_j.sum()), "rows_replaced": int((in_j & hit).sum()),
            "rows_refused": int(refused.sum()),
            "journal_n_refused": (int(pd.to_numeric(journal["n_refused"], errors="coerce").sum())
                                  if "n_refused" in journal.columns else None),
            "refused_rows_count_source": st[refused].value_counts().to_dict(),
            "refused_rows_values_changed": int(sum(int((~_same(_num(after[c])[refused],
                                                               _num(before[c])[refused])).sum())
                                                   for c in REPLACED if c in before.columns))}
    j = out.get("journaled_fixtures")
    out["ok"] = bool(
        all(f["applied_exactly"] == f["rows"] for f in out["fields"].values())
        and all(f["orig_kept_equal_to_stored"] in (None, f["rows"]) for f in out["fields"].values())
        and not any(unchanged.values())
        and set(out["not_refetched_rows"]["count_source"]) <= {PARSED_WITH_RULE, UNVERIFIED}
        and out["refetch_rows_without_history_row"] == 0
        and (j is None or (set(j["refused_rows_count_source"]) <= {UNVERIFIED}
                           and j["refused_rows_values_changed"] == 0
                           and j["rows_refused"] == j["journal_n_refused"])))
    return out


def main(argv=None) -> int:
    """python -m player_model.history_refetch --dry-run [--out FILE]: the proof on the committed data
    (player_history + the collected cup rows, read as props_harness reads them). Writes nothing
    except --out, which may not point under output/."""
    import argparse
    import json
    from pathlib import Path
    ap = argparse.ArgumentParser(description="prove the re-fetched counts replace the stored ones (read-only)")
    ap.add_argument("--dry-run", action="store_true", required=True)
    ap.add_argument("--out", help="also write the proof JSON here (never under output/)")
    a = ap.parse_args(argv)
    if a.out and config.OUTPUT_DIR.resolve() in Path(a.out).resolve().parents:
        raise SystemExit("refusing to write under output/ (the proof is read-only)")
    from . import props_harness as PH
    cols = list(dict.fromkeys(["date", "fixture_id", "player_id", "minutes", "fixture_status", STATUS_COL]
                              + REPLACED))
    before = PH.source_frame(columns=cols, refetch=pd.DataFrame())       # empty sidecar: stored values
    # source_frame labelled every row; the stored provenance is 'provider_rule' or nothing
    before[STATUS_COL] = before[STATUS_COL].where(before[STATUS_COL].eq(PARSED_WITH_RULE))
    jpath = REFETCH_FILE.parent / "player_refetch_fixtures.parquet"
    journal = pd.read_parquet(jpath) if jpath.exists() else None
    proof = refetch_proof(before, load_refetch(), journal)
    text = json.dumps(proof, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if proof["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
