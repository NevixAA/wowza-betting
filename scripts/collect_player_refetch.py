"""
collect_player_refetch.py - re-read API-Football /fixtures/players for every fixture behind
player_history and store the per-player counts BESIDE it, under the owner's null rule
(owner decision 2026-11-04, fix round 8).

    python scripts/collect_player_refetch.py [--limit N] [--dry-run]

WHY. The committed player_history.parquet (876,130 rows on 2026-11-04) and the UEFA cup rows beside
it (output/collect/cup_player_rows.parquet, 73,120 rows) were parsed by `int(x or 0)`: every count
holds 0 where the provider sent null, and not one count is blank. A real 0 ("played, no shot") and a
missing value ("the provider has no such statistic for this fixture") cannot be told apart in those
files. Only the provider's answer can tell them apart, so it is read again, once per fixture.

WHICH FIXTURES. Every fixture_id in player_history (all 876,130 rows carry the provider's fixture id)
and in the cup rows. A row WITHOUT a fixture id is mapped through the collected fixture lists
(output/strength/competition_fixtures.parquet): same date, both clubs the same by src.team_names
identity (club_rule.same_fixture_sides), exactly one fixture - otherwise refused and counted. A fixture
is FETCHED only when its identity is proven against the fixture lists:
  * the id is in the lists (finished: FT / AET / PEN);
  * the history's match date equals the list's date;
  * the history's home and away clubs are the list's home and away clubs (team_names identity).
Anything else is counted by reason in the audit (`refused`) and never fetched: UNIDENTIFIABLE
(e.g. national-team fixtures, which the registry's fixture lists do not hold). ONE call per proven
fixture, newest first (the props serving window first), so a budget stop leaves the oldest seasons.

A CORRECTION IS APPLIED ONLY WHEN PLAYER AND FIXTURE ARE PROVEN. Per history row: the player's
provider id (> 0) appears exactly once in the answer, in the team block that is the row's own side
(side by API-Football team id against the list's home_id / away_id; a cup row's team_id must equal
the block's team id). Otherwise the row is counted (`rows_refused`) and gets no correction.

THE NULL RULE (player_model.player_stat_nulls, the one shared reading; refined by the owner
2026-11-04, fix round 9): a null count is 0 when the player played (minutes > 0) AND the provider
returned player statistics for the fixture, even if nobody in the match has that statistic; otherwise
it stays missing. `filled_cols` names, per row, the counts read from a null.

ROWS ALREADY VERIFIED ARE NOT ASKED AGAIN (owner 2026-11-04, fix round 9). A row a collector parsed
from the raw provider answer with the rule carries count_source 'provider_rule' (history or cup rows)
and is left out of the plan: its fixture is fetched only if another row of it is still unverified.
Only rows written before the fix (no count_source) are re-fetched. The plan counts both.

OUTPUT (output/collect/ only; player_history.parquet and the cup rows are never rewritten)
  player_history_refetch.parquet   one row per corrected (fixture_id, player_id): source ('history' |
        'cup_rows'), team_id, is_home, minutes_refetch, the re-fetched counts + rating (nullable),
        orig_<col> = the value the original file holds (kept beside it), filled_cols, fetched_at.
  player_refetch_fixtures.parquet  the journal: fixture_id, source, status ('ok' | 'empty' - an empty
        answer is the provider's real answer and is not asked again), n_teams, n_players, n_rows,
        n_refused, fetched_at. A failed request records nothing and is asked again next run.
  player_history_refetch_audit.json  the plan (fixtures, proofs, refusals by reason, calls), the run,
        and per field the MEASURED provider behaviour on the stored rows (sent value / sent 0 / null
        read as 0 / missing) and the change against the original (0 -> 0, 0 -> missing, 0 -> value).
Readers: player_model.history_refetch.with_refetched() replaces the suspect values with these and
marks every row not yet re-fetched 'zeros_unverified'.

Budget: scripts/collect_common.AFClient (checked before every call; a stop is a clean exit 0 and the
next run resumes from the journal). Workflow step 'playerrefetch' runs AFTER lineups (owner).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import collect_af_lists as AL  # noqa: E402
from collect_common import (COLLECT_DIR, FLUSH_EVERY, AFClient, BudgetStop,  # noqa: E402
                            append_parquet, history_frame, read_parquet)
from player_model import player_stat_nulls as PSN  # noqa: E402

OUT = COLLECT_DIR / "player_history_refetch.parquet"
JOURNAL = COLLECT_DIR / "player_refetch_fixtures.parquet"
AUDIT = COLLECT_DIR / "player_history_refetch_audit.json"
CUP_ROWS = COLLECT_DIR / "cup_player_rows.parquet"
#: provider-proven fixtures outside the collected lists (scripts/collect_refetch_extra_fixtures.py)
EXTRA_FIXTURES = COLLECT_DIR / "refetch_extra_fixtures.parquet"

#: the columns that are re-read and replaced: every per-player count player_history stores + rating
VALUE_COLS = list(PSN.HISTORY_COUNT_COLS) + ["rating"]
KEY_COLS = ["fixture_id", "player_id", "date", "home_team", "away_team", "team", "is_home"]
COLS = (["fixture_id", "player_id", "source", "team_id", "is_home", "is_home_orig", "is_home_corrected",
         "minutes_refetch"] + VALUE_COLS
        + [f"orig_{c}" for c in VALUE_COLS] + ["filled_cols", "fetched_at"])
J_COLS = ["fixture_id", "source", "status", "n_teams", "n_players", "n_rows", "n_refused", "fetched_at"]


# ── the rows behind the history ──────────────────────────────────────────────────────────────────

def source_rows(history: pd.DataFrame | None = None, cups: pd.DataFrame | None = None) -> pd.DataFrame:
    """player_history rows + cup rows (the columns the plan and the correction need), `source` set."""
    want = KEY_COLS + ["team_id"] + VALUE_COLS + [PSN.SOURCE_COL]
    h = history_frame(want) if history is None else history
    c = read_parquet(CUP_ROWS) if cups is None else cups
    parts = []
    for name, f in (("history", h), ("cup_rows", c)):
        if f is None or f.empty:
            continue
        f = f[[x for x in want if x in f.columns]].copy()
        f["source"] = name
        parts.append(f)
    if not parts:
        return pd.DataFrame(columns=want + ["source"])
    r = pd.concat(parts, ignore_index=True, sort=False)
    r["fixture_id"] = pd.to_numeric(r["fixture_id"], errors="coerce")
    r["player_id"] = pd.to_numeric(r["player_id"], errors="coerce")
    r["_day"] = pd.to_datetime(r["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return r


def map_fixture_ids(rows: pd.DataFrame, fx: pd.DataFrame) -> tuple[pd.Series, int]:
    """fixture_id for rows WITHOUT one, from the fixture lists: same date, both clubs the same
    (team_names identity), exactly one fixture. Returns (ids, n_refused). Ids stay NaN when unproven."""
    from player_model.club_rule import same_fixture_sides
    ids = rows["fixture_id"].copy()
    miss = rows[ids.isna()]
    if miss.empty or fx.empty:
        return ids, int(len(miss))
    day = pd.to_datetime(fx["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    by_day = {d: g for d, g in fx.assign(_day=day).groupby("_day")}
    refused = 0
    for (d, hm, aw), g in miss.groupby(["_day", "home_team", "away_team"], dropna=False):
        pool = by_day.get(d)
        hits = [] if pool is None else [int(f) for f, h2, a2 in zip(pool["fixture_id"], pool["home_name"],
                                                                     pool["away_name"])
                                        if same_fixture_sides(hm, aw, h2, a2)]
        if len(hits) == 1:
            ids.loc[g.index] = hits[0]
        else:
            refused += len(g)
    return ids, refused


def plan(rows: pd.DataFrame | None = None, fx: pd.DataFrame | None = None,
         journal: pd.DataFrame | None = None, redo_refused: bool = False) -> dict:
    """The proven fixtures still to fetch (newest first) + every refusal, counted by reason."""
    from player_model.club_rule import same_fixture_sides
    rows = source_rows() if rows is None else rows
    if fx is None:
        fx = AL.fixture_list()
        # P7 item 5: fixtures outside the collected lists (national teams etc.), proven by the
        # provider's own /fixtures?ids answer (scripts/collect_refetch_extra_fixtures.py). Read HERE
        # only, never by collect_af_lists.fixture_list (the team-id collectors must not see them).
        extra = AL.fixture_list(EXTRA_FIXTURES) if EXTRA_FIXTURES.exists() else pd.DataFrame()
        if len(extra):
            fx = pd.concat([fx, extra[[c for c in fx.columns if c in extra.columns]]],
                           ignore_index=True).drop_duplicates("fixture_id", keep="first")
    journal = read_parquet(JOURNAL) if journal is None else journal
    out = {"rows": int(len(rows)), "rows_by_source": rows["source"].value_counts().to_dict() if len(rows) else {},
           "refused_fixtures": {}, "refused_rows": {}}
    # rows parsed with the rule are verified as stored: never part of the re-fetch (fix round 9)
    verified = (rows[PSN.SOURCE_COL].isin(PSN.VERIFIED_SOURCES) if PSN.SOURCE_COL in rows.columns
                else pd.Series(False, index=rows.index))
    out["rows_verified_at_parse"] = int(verified.sum())
    rows = rows[~verified.to_numpy()]
    out["rows_to_verify"] = int(len(rows))
    if rows.empty:                                    # nothing left to verify: no call at all
        out.update(rows_without_fixture_id=0, rows_mapped_through_lists=0, fixtures=0, proven_fixtures=0,
                   journaled_fixtures=0, planned_calls=0,
                   todo=pd.DataFrame(columns=["fixture_id", "date", "source"]),
                   fx=fx.drop_duplicates("fixture_id").set_index("fixture_id"), rows_frame=rows)
        return out
    out["rows_without_fixture_id"] = int(rows["fixture_id"].isna().sum())
    ids, n_unmapped = map_fixture_ids(rows, fx)
    rows = rows.assign(fixture_id=ids)
    out["rows_mapped_through_lists"] = out["rows_without_fixture_id"] - int(n_unmapped)
    if n_unmapped:
        out["refused_rows"]["no_fixture_id_unmapped"] = int(n_unmapped)
    rows = rows[rows["fixture_id"].notna()]
    fx = fx.drop_duplicates("fixture_id").set_index("fixture_id")
    fx_day = pd.to_datetime(fx["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    g = rows.groupby("fixture_id").agg(day=("_day", "unique"), home=("home_team", "unique"),
                                       away=("away_team", "unique"), source=("source", "first"),
                                       n=("player_id", "size"))
    out["fixtures"] = int(len(g))
    reason = pd.Series(None, index=g.index, dtype=object)
    in_list = g.index.isin(fx.index)
    reason[~in_list] = "fixture_not_in_lists"
    multi = g["day"].map(len).gt(1) | g["home"].map(len).gt(1) | g["away"].map(len).gt(1)
    reason[in_list & multi.to_numpy()] = "history_disagrees_with_itself"
    ok = reason.isna()
    day = g.loc[ok, "day"].map(lambda a: a[0])
    reason[ok & (day != fx_day.reindex(day.index)).reindex(g.index, fill_value=False)] = "date_differs_from_list"
    ok = reason.isna()
    pairs = pd.DataFrame({"h": g.loc[ok, "home"].map(lambda a: a[0]), "a": g.loc[ok, "away"].map(lambda a: a[0]),
                          "lh": fx.loc[g.index[ok], "home_name"].to_numpy(),
                          "la": fx.loc[g.index[ok], "away_name"].to_numpy()}, index=g.index[ok])
    memo: dict = {}
    for fid, h, a, lh, la in pairs.itertuples():
        k = (h, a, lh, la)
        if k not in memo:
            memo[k] = bool(same_fixture_sides(h, a, lh, la))
        if not memo[k]:
            reason[fid] = "clubs_unproven"
    for r, n in reason.dropna().value_counts().items():
        out["refused_fixtures"][r] = int(n)
        out["refused_rows"][r] = int(g.loc[reason == r, "n"].sum())
    proven = g[reason.isna()].copy()
    out["proven_fixtures"] = int(len(proven))
    done = set(pd.to_numeric(journal["fixture_id"], errors="coerce").dropna().astype(int)) if len(journal) else set()
    if redo_refused and done:
        # P7 item 6: re-read the journaled fixtures that still hold a source row with no stored
        # result (refused at the time, e.g. side_unproven before the is_home proof existed)
        st_rows = read_parquet(OUT)
        stored = set()
        if len(st_rows):
            keys = pd.DataFrame({"f": pd.to_numeric(st_rows["fixture_id"], errors="coerce"),
                                 "p": pd.to_numeric(st_rows["player_id"], errors="coerce")}).dropna()
            stored = set(zip(keys["f"].astype(int), keys["p"].astype(int)))
        src = rows[rows["fixture_id"].notna() & rows["player_id"].notna()]
        unstored = {int(f) for f, p_ in zip(src["fixture_id"], src["player_id"])
                    if (int(f), int(p_)) not in stored}
        out["redo_refused_fixtures"] = int(len(done & unstored))
        done = done - unstored
    proven["_done"] = [int(f) in done for f in proven.index]
    out["journaled_fixtures"] = int(proven["_done"].sum())
    todo = proven[~proven["_done"]].copy()
    todo["date"] = fx_day.reindex(todo.index)
    todo = todo.reset_index().sort_values(["date", "fixture_id"], ascending=[False, False], kind="mergesort")
    out["planned_calls"] = int(len(todo))
    out["todo"] = todo[["fixture_id", "date", "source"]].reset_index(drop=True)
    out["fx"] = fx
    out["rows_frame"] = rows
    return out


def _club_on_side(team, fxrow, home: bool) -> bool:
    """True only when `team` (the history row's club) is proven to be the fixture-list club on the
    given side and NOT the other one (team_names.resolve_identity over the two list names)."""
    from src.team_names import resolve_identity
    if team is None or (isinstance(team, float) and pd.isna(team)):
        return False
    h, a = str(fxrow.get("home_name")), str(fxrow.get("away_name"))
    hit = resolve_identity(str(team), [h, a])
    return hit == (h if home else a)


# ── one answer ───────────────────────────────────────────────────────────────────────────────────

def parse_answer(body: dict | None, fid: int, fxrow, rows: pd.DataFrame, now: str) -> tuple[list[dict], dict]:
    """Corrections for the history rows `rows` of fixture `fid` from one /fixtures/players answer.
    Returns (rows to store, journal row)."""
    from player_model.data_fetcher import fixture_side
    resp = (body or {}).get("response") or []
    j = {"fixture_id": int(fid), "source": str(rows["source"].iloc[0]) if len(rows) else None,
         "status": "ok" if resp else "empty", "n_teams": len(resp), "n_players": 0, "n_rows": 0,
         "n_refused": 0, "is_home_corrected": 0, "fetched_at": now, "refused": {}}
    if not resp:
        j["n_refused"] = int(len(rows))
        j["refused"]["empty_answer"] = int(len(rows))
        return [], j
    answered = PSN.fixture_answered(resp)
    # side by team id (the list's ids); a list row without ids falls back to club identity by name
    hid, aid = AL.int_or_none(fxrow.get("home_id")), AL.int_or_none(fxrow.get("away_id"))
    fixture = {"teams": {"home": {"id": hid}, "away": {"id": aid}}}
    seen: dict[int, list] = {}
    for block in resp:
        side = fixture_side(block, fixture, fxrow["home_name"], fxrow["away_name"])
        tid = AL.int_or_none((block.get("team") or {}).get("id"))
        for pe in block.get("players") or []:
            pid = AL.int_or_none((pe.get("player") or {}).get("id"))
            j["n_players"] += 1
            if pid is not None and pid > 0:
                seen.setdefault(pid, []).append((side, tid, pe))
    out = []

    def _refuse(why):
        j["refused"][why] = j["refused"].get(why, 0) + 1
        j["n_refused"] += 1
    for r in rows.itertuples(index=False):
        pid = AL.int_or_none(r.player_id)
        if pid is None or pid <= 0:
            _refuse("player_id_invalid")
            continue
        hits = seen.get(pid, [])
        if not hits:
            _refuse("player_not_in_answer")
            continue
        if len(hits) > 1:
            _refuse("player_twice_in_answer")
            continue
        side, tid, pe = hits[0]
        row_home = AL.int_or_none(r.is_home)
        home_corrected = False
        if side is None or row_home is None:
            _refuse("side_unproven")
            continue
        if bool(row_home) != bool(side):
            # P7 item 6 (2026-11-09): the history's is_home disagrees with the provider. 3,566 such rows
            # were measured, all of one shape: the row's own club IS the list's home club while the row
            # says away (e.g. VfL Bochum, 1. FC Koln home matches stored as away). Accepted ONLY when
            # the row's club resolves (team_names.resolve_identity) to the club on the provider's side
            # and not to the other one; the provider side is stored and the original kept beside it
            # (is_home_orig, is_home_corrected). Anything else stays refused.
            if not _club_on_side(getattr(r, "team", None), fxrow, bool(side)):
                _refuse("side_unproven")
                continue
            home_corrected = True
            j["is_home_corrected"] += 1
        own_tid = AL.int_or_none(getattr(r, "team_id", None))
        if own_tid is not None and own_tid != tid:
            _refuse("team_id_differs")
            continue
        st = PSN._stats0(pe)
        mins = PSN.minutes_of(st)
        counts = PSN.player_counts(st, answered)
        filled = [c for c in PSN.HISTORY_COUNT_COLS
                  if counts[c] == 0 and PSN.as_count(PSN.raw_value(st, c)) is None]
        rec = {"fixture_id": int(fid), "player_id": pid, "source": r.source, "team_id": tid,
               "is_home": int(bool(side)), "is_home_orig": int(bool(row_home)),
               "is_home_corrected": bool(home_corrected), "minutes_refetch": mins,
               **{c: counts[c] for c in PSN.HISTORY_COUNT_COLS}, "rating": PSN.rating_of(st),
               **{f"orig_{c}": getattr(r, c, None) for c in VALUE_COLS},
               "filled_cols": ",".join(filled), "fetched_at": now}
        out.append(rec)
    j["n_rows"] = len(out)
    return out, j


# ── audit ────────────────────────────────────────────────────────────────────────────────────────

def field_audit(stored: pd.DataFrame) -> dict:
    """Per field, on the stored re-fetched rows of players who played: what the provider SENT (value
    > 0 / 0), what was a null read as 0, what stays missing; and against the original file: 0 -> 0,
    0 -> missing, 0 -> a value, value -> other value."""
    if stored.empty:
        return {}
    played = pd.to_numeric(stored["minutes_refetch"], errors="coerce") > 0
    s = stored[played]
    filled = s["filled_cols"].fillna("").astype(str).str.split(",")
    out = {}
    for c in VALUE_COLS:
        new = pd.to_numeric(s[c], errors="coerce")
        old = pd.to_numeric(s[f"orig_{c}"], errors="coerce")
        was_filled = filled.map(lambda xs: c in xs)
        out[c] = {"rows": int(len(s)), "sent_value": int((new > 0).sum()),
                  "sent_zero": int(((new == 0) & ~was_filled).sum()),
                  "null_read_as_zero": int(was_filled.sum()), "missing": int(new.isna().sum()),
                  "orig_zero_now_zero": int(((old == 0) & (new == 0)).sum()),
                  "orig_zero_now_missing": int(((old == 0) & new.isna()).sum()),
                  "orig_zero_now_value": int(((old == 0) & (new > 0)).sum()),
                  "orig_value_now_different": int(((old > 0) & (new != old)).sum())}
        if c == "rating":
            out[c] = {k: v for k, v in out[c].items() if k not in ("null_read_as_zero", "sent_zero")}
    return out


def write_audit(p: dict, st: dict) -> dict:
    stored = read_parquet(OUT)
    journal = read_parquet(JOURNAL)
    a = {"generated_at": AL.now_iso(), "owner_decision": "2026-11-04 fix round 8 (props item 3)",
         "plan": {k: v for k, v in p.items() if k not in ("todo", "fx", "rows_frame")},
         "run": st,
         "stored_rows": int(len(stored)),
         "stored_rows_by_source": stored["source"].value_counts().to_dict() if len(stored) else {},
         "journal_fixtures": int(len(journal)),
         "journal_by_status": journal["status"].value_counts().to_dict() if len(journal) else {},
         "rows_refused_in_answers": (pd.to_numeric(journal["n_refused"], errors="coerce").sum().item()
                                     if len(journal) else 0),
         "fields": field_audit(stored),
         "not_counts": sorted(PSN.NOT_COUNTS)}
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(a, indent=1, default=str), encoding="utf-8")
    return a


# ── run ──────────────────────────────────────────────────────────────────────────────────────────

def _flush(buf, jbuf):
    if buf:                                   # rows first, journal last: the journal decides what is asked
        append_parquet(OUT, buf, ["fixture_id", "player_id"], COLS)
    if jbuf:
        append_parquet(JOURNAL, [{k: v for k, v in j.items() if k in J_COLS} for j in jbuf],
                       ["fixture_id"], J_COLS)


def run(limit: int | None = None, dry_run: bool = False, client: AFClient | None = None,
        rows: pd.DataFrame | None = None, fx: pd.DataFrame | None = None, redo_refused: bool = False) -> dict:
    p = plan(rows, fx, redo_refused=redo_refused)
    pub = {k: v for k, v in p.items() if k not in ("todo", "fx", "rows_frame")}
    print(f"[playerrefetch] {json.dumps(pub, default=str)}")
    print(f"[playerrefetch] plan: exactly {p['planned_calls']:,} /fixtures/players call(s) "
          f"(1 per proven fixture not yet journaled; a failed call is retried next run)")
    st = {"calls": 0, "fetched": 0, "rows_stored": 0, "rows_refused": {}, "empty_answers": 0}
    if dry_run:
        return {**pub, **st}
    if p["planned_calls"] == 0:
        write_audit(p, st)
        return {**pub, **st}
    client = client or AFClient()
    by_fid = {int(f): g for f, g in p["rows_frame"].groupby("fixture_id")}
    buf, jbuf = [], []
    try:
        for n, t in enumerate(p["todo"].itertuples(index=False)):
            if limit is not None and n >= limit:
                break
            fid = int(t.fixture_id)
            body = client.get("fixtures/players", {"fixture": fid})
            if body is None:
                continue                                  # failed: nothing recorded, asked next run
            st["calls"] += 1
            recs, j = parse_answer(body, fid, p["fx"].loc[fid], by_fid[fid], AL.now_iso())
            buf.extend(recs)
            jbuf.append(j)
            st["fetched"] += 1
            st["rows_stored"] += len(recs)
            st["empty_answers"] += j["status"] == "empty"
            for k, v in j["refused"].items():
                st["rows_refused"][k] = st["rows_refused"].get(k, 0) + v
            if len(jbuf) >= FLUSH_EVERY:
                _flush(buf, jbuf)
                buf, jbuf = [], []
    except BudgetStop as e:
        print(f"[playerrefetch] budget stop: {e} (resumable)")
        st["budget_stop"] = str(e)
    finally:
        _flush(buf, jbuf)
        client.flush()
    write_audit(p, st)
    print(f"[playerrefetch] {json.dumps(st)}")
    return {**pub, **st}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="re-read /fixtures/players for every player_history fixture")
    ap.add_argument("--limit", type=int, default=None, help="at most N fixture calls")
    ap.add_argument("--dry-run", action="store_true", help="print the exact planned call count; no call")
    ap.add_argument("--redo-refused", action="store_true",
                    help="also re-read journaled fixtures that still hold refused rows (P7 item 6)")
    a = ap.parse_args(argv)
    run(limit=a.limit, dry_run=a.dry_run, redo_refused=a.redo_refused)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
