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

HOME / AWAY SIDE (owner 2026-11-09, P7 item 6). For 3,566 rows the history's is_home was wrong: the
writer compared the row's club name with the fixture's home name by spelling ('Pachuca' vs 'CF
Pachuca'). The re-fetch proves the provider side when the row's club resolves to the list club on
that side (scripts/collect_player_refetch.py: is_home, is_home_orig, is_home_corrected). Where
is_home_corrected is true the proven side REPLACES is_home and the opponent name follows it; the
stored values stay beside as is_home_orig / opponent_orig, and is_home_source says which rows carry
the provider side. Every feature that reads the side (opponent, team and opponent keys, home / away
splits, strength, set-piece mapping) reads the corrected one, in training and serving alike.

NOT HERE. Stored derived columns (rolling per-game rates, target_* labels) were computed from the
old values; serving recomputes its rolling features from the raw rows, and the Phase 2 retrain must
rebuild features and labels from these raw counts. This reader changes raw counts only.
"""
from __future__ import annotations

import pandas as pd

from . import config
from .player_stat_nulls import (HISTORY_COUNT_COLS, NULL_IS_MISSING, PARSED_WITH_RULE, REFETCHED,
                                SOURCE_COL, UNVERIFIED)

REFETCH_FILE = config.OUTPUT_DIR / "collect" / "player_history_refetch.parquet"
#: the replaced columns: every per-player count player_history stores + rating (not a count; an
#: unrated appearance was stored as 0 by the same bug)
REPLACED = list(HISTORY_COUNT_COLS) + ["rating"]
#: per row, the NULL_IS_MISSING columns the reader read as missing (provider null), comma-separated
NULL_BLANKED_COL = "null_is_missing_cols"
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
        return _with_proven_side(out, rf, key)
    sel = rf.reindex(key[hit])
    # A null the collector read as 0 in a NULL_IS_MISSING column (goals_conceded, fix 2026-11-11;
    # penalty_missed, owner 2026-11-11 "Fix penalties too, then promote") is
    # missing: the stored re-fetch row keeps its 0 and filled_cols beside (corrections beside the
    # originals); the reader takes the provider's null.
    filled = (sel["filled_cols"].fillna("").astype(str).str.split(",") if "filled_cols" in sel.columns
              else pd.Series([[]] * len(sel), index=sel.index))
    blanked = pd.Series("", index=out.index, dtype=object)
    for c in REPLACED:
        if c not in rf.columns:
            continue
        col = pd.to_numeric(out[c], errors="coerce").astype(float) if c in out.columns \
            else pd.Series(float("nan"), index=out.index)
        new = pd.to_numeric(sel[c], errors="coerce").astype(float)
        if c in NULL_IS_MISSING:
            null_read_as_0 = filled.map(lambda xs, c=c: c in xs).to_numpy(dtype=bool)
            new = new.where(~null_read_as_0)
            hit_idx = out.index[hit][null_read_as_0]
            blanked.loc[hit_idx] = blanked.loc[hit_idx] + c + ","
            print(f"[refetch] {c}: {int(null_read_as_0.sum()):,} stored null-read-as-0 value(s) read as "
                  "missing (the provider sent null)")
        col.loc[hit] = new.to_numpy()
        out[c] = col
    # provenance: the NULL_IS_MISSING columns read as missing on this row because the provider sent null
    # (props_feature_truth 6c checks that they stay missing)
    out[NULL_BLANKED_COL] = blanked.str.rstrip(",")
    out.loc[hit, STATUS_COL] = REFETCHED
    print(f"[refetch] {int(hit.sum()):,} of {len(out):,} row(s) carry re-fetched counts; "
          f"{int((out[STATUS_COL] == PARSED_WITH_RULE).sum()):,} parsed with the rule; "
          f"{int((out[STATUS_COL] == UNVERIFIED).sum()):,} marked {UNVERIFIED}")
    return _with_proven_side(out, rf, key)


# ── goalkeeper goals_conceded against the match score (fix/props-feature-bugs 2026-11-11) ──────────
#
# WHY. API-Football's goals.conceded disagrees with the match score on a growing share of keeper rows
# (re-fetched answers: provider null on 47% of 2026-10 and 90% of 2026-11 keeper rows, read as 0 by
# the null rule until NULL_IS_MISSING; plus a sent 0 beside a real score on ~40 rows since 2026-09).
# The score is the match fact: where a keeper was his side's ONLY keeper for the full 90 minutes of a
# match finished in 90 minutes, his goals conceded ARE the opponent's 90-minute goals. A value that
# says otherwise is wrong and is read as missing - never 0 and never replaced by the score (no
# defaults: the rule removes a contradicted value, it does not invent one).

#: gk_conceded_check values
GK_AGREES = "agrees_with_score"
GK_CONTRADICTS = "contradicts_score"   # value blanked; the provider's value kept beside
GK_NOT_CHECKABLE = "not_checkable"     # not the side's only keeper for 90 minutes, no 90-minute
                                       # score, a match decided after extra time, or no value
GK_CHECK_COL = "gk_conceded_check"
GK_ORIG_COL = "goals_conceded_provider"


def gk_conceded_frame(raw: pd.DataFrame, fx: pd.DataFrame | None = None) -> pd.DataFrame:
    """The keeper rows of `raw` (index kept) with gc (goals_conceded as read), conc (the opponent's
    90-minute goals from team_identity.fixture_results, the provider's score.fulltime) and checkable:
    the side's ONLY keeper with minutes > 0, >= 90 minutes, fixture status FT, score stored."""
    from . import team_identity as TI
    cols = ["fixture_id", "is_home", "minutes", "gc", "conc", "checkable"]
    need = {"fixture_id", "position", "minutes", "is_home", "goals_conceded"}
    if raw is None or raw.empty or not need <= set(raw.columns):
        return pd.DataFrame(columns=cols)
    fx = TI.fixture_results() if fx is None else fx
    pos = raw["position"].astype(str).str.strip().str.upper().str[:1]
    mins = pd.to_numeric(raw["minutes"], errors="coerce")
    g = pd.DataFrame({"fixture_id": pd.to_numeric(raw["fixture_id"], errors="coerce"),
                      "is_home": pd.to_numeric(raw["is_home"], errors="coerce"),
                      "minutes": mins, "gc": pd.to_numeric(raw["goals_conceded"], errors="coerce")},
                     index=raw.index)[((pos == "G") & (mins > 0)).to_numpy()]
    g = g.dropna(subset=["fixture_id", "is_home"])
    if g.empty or fx is None or fx.empty:
        return g.assign(conc=float("nan"), checkable=False).reindex(columns=cols)
    n_gk = g.groupby(["fixture_id", "is_home"])["minutes"].transform("size")
    f = (fx[["fixture_id", "ft_home", "ft_away", "status"]]
         .assign(fixture_id=lambda x: pd.to_numeric(x["fixture_id"], errors="coerce"))
         .drop_duplicates("fixture_id").set_index("fixture_id").reindex(g["fixture_id"].to_numpy()))
    home = g["is_home"].to_numpy() == 1
    conc = pd.Series(pd.to_numeric(f["ft_away"], errors="coerce").where(home, pd.to_numeric(
        f["ft_home"], errors="coerce")).to_numpy(dtype=float), index=g.index)
    ft = pd.Series(f["status"].eq("FT").to_numpy(dtype=bool), index=g.index)
    ok = n_gk.eq(1) & g["minutes"].ge(90) & ft & conc.notna()
    return g.assign(conc=conc, checkable=ok.astype(bool))


def with_gk_conceded_checked(raw: pd.DataFrame, fx: pd.DataFrame | None = None) -> pd.DataFrame:
    """A keeper's goals_conceded that CONTRADICTS the match score (gk_conceded_frame: the side's only
    keeper, full 90 minutes, match finished in 90 minutes) is missing. The value as read before the
    check is kept beside in goals_conceded_provider; gk_conceded_check says agrees / contradicts /
    not checkable per keeper row (None on other rows). props_harness.source_frame applies it, and
    training and serving both read that frame."""
    if raw is None or raw.empty or "goals_conceded" not in raw.columns:
        return raw
    g = gk_conceded_frame(raw, fx)
    out = raw.copy()
    out[GK_ORIG_COL] = pd.to_numeric(out["goals_conceded"], errors="coerce").astype(float)
    out[GK_CHECK_COL] = None
    if g.empty:
        return out
    chk = g["checkable"].astype(bool) & g["gc"].notna()
    bad = chk & (g["gc"] != g["conc"])
    out.loc[g.index, GK_CHECK_COL] = GK_NOT_CHECKABLE
    out.loc[g.index[(chk & ~bad).to_numpy()], GK_CHECK_COL] = GK_AGREES
    out.loc[g.index[bad.to_numpy()], GK_CHECK_COL] = GK_CONTRADICTS
    gc = out[GK_ORIG_COL].copy()
    gc.loc[g.index[bad.to_numpy()]] = float("nan")
    out["goals_conceded"] = gc
    print(f"[gk-check] {len(g):,} keeper row(s): {int(chk.sum()):,} checkable against the 90-minute "
          f"score, {int(bad.sum()):,} contradicted it -> goals_conceded missing (value kept in {GK_ORIG_COL})")
    return out


#: is_home_source values
SIDE_FROM_HISTORY = "history"
SIDE_PROVIDER_PROVEN = "provider_side_proven"


def _with_proven_side(out: pd.DataFrame, rf: pd.DataFrame, key: pd.MultiIndex) -> pd.DataFrame:
    """is_home (and the opponent name) from the provider side where the re-fetch proved it
    (is_home_corrected), the stored values kept beside (is_home_orig, opponent_orig). Applies to any
    row of a proven (fixture, player), whatever its count_source: the side is identity, not a count."""
    if not {"is_home", "is_home_corrected"} <= set(rf.columns) or "is_home" not in out.columns:
        return out
    proven = rf[rf["is_home_corrected"] == True]                       # noqa: E712
    m = key.isin(proven.index)
    out["is_home_orig"] = out["is_home"]
    out["is_home_source"] = SIDE_FROM_HISTORY
    if not m.any():
        return out
    side = pd.to_numeric(proven.reindex(key[m])["is_home"], errors="coerce").to_numpy()
    was = pd.to_numeric(out.loc[m, "is_home"], errors="coerce").to_numpy()
    out["is_home"] = pd.to_numeric(out["is_home"], errors="coerce")
    out.loc[m, "is_home"] = side
    out.loc[m, "is_home_source"] = SIDE_PROVIDER_PROVEN
    if {"home_team", "away_team"} <= set(out.columns):
        out["opponent_orig"] = out["opponent"] if "opponent" in out.columns else pd.Series(None, index=out.index)
        opp = out.loc[m, "away_team"].where(out.loc[m, "is_home"].astype(bool), out.loc[m, "home_team"])
        if "opponent" not in out.columns:
            out["opponent"] = out["opponent_orig"]
        out.loc[m, "opponent"] = opp
    print(f"[refetch] home/away side proven by the provider on {int(m.sum()):,} row(s); "
          f"{int((side != was).sum()):,} changed side (is_home_orig kept beside)")
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
