"""One match, one row - duplicate-match detection and removal for the team training frame (2026-11-02).

THE DEFECT (CLAUDE_PROMPT_2 P0/P1 #1, re-measured at data-truth-100 HEAD). The merged history
(football-data download + fd_history cache + backtest archive + API-Football) still carried the same
match twice after both merge steps in src/data_loader.py:
  * SAME DAY, ONE CLUB UNDER TWO SPELLINGS: 'Tirol' / 'WSG Wattens', 'U. Cluj' / 'Universitatea
    Cluj', and the mojibake 'PreuÃ\\x9fen MÃ¼nster' (UTF-8 bytes read as Latin-1) next to
    'Preußen Münster';
  * 1-3 DAYS APART: the same fixture filed under the local date by one source and the UTC date by
    another (MLS, Argentina, Brazil, Mexico) - an exact-date key can never see it.
A copy dated EARLIER than its twin sits inside the twin's own last-5 form window, so the form
feature contains the very result it is meant to predict.

THE RULE (owner rule: canonical names, no invented values). Two rows are ONE match when, in the
same league,
  * both clubs are the same canonical club (src.team_names - normalised equality, the alias store,
    the identity-token rule; a name pair that has ever played EACH OTHER is never one club), and
  * the score is identical, and
  * the dates are at most `window_days` (3) apart.
A league never schedules the same home/away fixture twice within three days, so this cannot merge
two real matches. Also one match: the SAME DAY, the same score and ONE side the same canonical club
(a club does not play twice in a day). That pair is merged as a match, but neither club is renamed:
naming stays with src.team_names / src/data_loader._one_side_aliases.

SURVIVOR (stated source rule): the row from the highest-priority source in SOURCE_PRIORITY
(football-data > its cache > API-Football > backtest archive; a row without a source ranks last),
then the most populated row, then the earliest. Its empty cells are filled from the dropped copy of
the SAME match (real values, never invented); the dropped copy's date and spellings are kept in the
report. Copies whose scores disagree are not merged here (not evidence of one match); same-day score
conflicts are quarantined upstream by data_loader._canonicalise_clubs.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

WINDOW_DAYS = 3
#: lower = preferred survivor
SOURCE_PRIORITY = {"football_data": 0, "fd_cache": 1, "api_football": 2, "backtest_archive": 3}
_MOJIBAKE_MARKS = ("Ã", "Â", "Å", "Ä", "Æ", "Ð", "Ñ", "â€")


def fix_mojibake(s):
    """'PreuÃ\\x9fen MÃ¼nster' -> 'Preußen Münster': UTF-8 bytes that were decoded as Latin-1/cp1252
    are re-decoded. Only a string that round-trips exactly is changed; anything else is returned
    as-is (no guessing)."""
    if not isinstance(s, str) or not any(m in s for m in _MOJIBAKE_MARKS):
        return s
    for enc in ("latin-1", "cp1252"):
        try:
            fixed = s.encode(enc).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if fixed != s:
            return fixed
    return s


def decode_csv_bytes(content: bytes) -> str:
    """Text of a downloaded CSV: UTF-8 (BOM stripped) when the bytes are valid UTF-8, else Latin-1.
    requests' `r.text` guesses ISO-8859-1 when the server sends no charset, which is how UTF-8
    club names became mojibake in the banks."""
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return content.decode("latin-1")


def club_identity(df: pd.DataFrame) -> dict:
    """{(league, spelling): canonical id} - spellings of ONE club within a league share an id.

    Proof only (src.team_names): resolve() of a spelling among the league's OTHER spellings (alias
    store, normalised equality, identity tokens - exactly one hit) after the mojibake repair. Two
    spellings that ever met each other are two clubs, whatever a rule says."""
    from src.team_names import resolve
    out: dict = {}
    d = df[["league", "home_team", "away_team"]].astype(str)
    for lg, g in d.groupby("league", sort=False):
        names = sorted(set(g["home_team"]) | set(g["away_team"]))
        met = set(map(frozenset, zip(g["home_team"], g["away_team"])))
        parent = {n: n for n in names}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        fixed = {n: fix_mojibake(n) for n in names}
        by_fixed = defaultdict(list)
        for n in names:
            by_fixed[fixed[n]].append(n)
        for n in names:
            pool = [m for m in names if m != n]
            hits = set(by_fixed[fixed[n]]) - {n}
            r = resolve(fixed[n], [fixed[m] for m in pool])
            if r is not None:
                hits |= {m for m in pool if fixed[m] == r}
            for m in hits:
                if frozenset((n, m)) in met:
                    continue
                parent[find(n)] = find(m)
        for n in names:
            out[(lg, n)] = f"{lg}|{find(n)}"
    return out


def _groups(k: pd.DataFrame, keys: list):
    """(first key value, group frame as plain arrays) in the order and row order of
    `k.groupby(keys, sort=False)`, without pandas' per-group frame slicing (the data-truth rule 5
    stoplight spent most of its time there, 2026-11-02)."""
    if k.empty:
        return
    gid = k.groupby(keys, sort=False, dropna=False).ngroup().to_numpy()
    order = np.argsort(gid, kind="stable")
    gid = gid[order]
    cols = {c: k[c].to_numpy()[order] for c in k.columns}
    cuts = np.flatnonzero(np.diff(gid)) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(gid)]
    first = keys[0]
    for s, e in zip(starts, ends):
        yield cols[first][s], {c: v[s:e] for c, v in cols.items()}


def duplicate_pairs(df: pd.DataFrame, window_days: int = WINDOW_DAYS,
                    identity: dict | None = None) -> pd.DataFrame:
    """Every pair of rows that are one match (see module rule). Columns: i, j (df index labels,
    i dated first), league, gap_days, kind ('both_sides' | 'one_side_same_day'). Rows without a
    score are never paired."""
    cols = ["i", "j", "league", "gap_days"]
    if df.empty:
        return pd.DataFrame(columns=cols + ["kind"])
    ident = identity if identity is not None else club_identity(df)
    d = df[df["home_goals"].notna() & df["away_goals"].notna()]
    lg = d["league"].astype(str)
    hid = [ident.get((a, b), f"{a}|{b}") for a, b in zip(lg, d["home_team"].astype(str))]
    aid = [ident.get((a, b), f"{a}|{b}") for a, b in zip(lg, d["away_team"].astype(str))]
    k = pd.DataFrame({"lg": lg.to_numpy(), "h": hid, "a": aid,
                      "hg": d["home_goals"].to_numpy(), "ag": d["away_goals"].to_numpy(),
                      "dt": pd.to_datetime(d["date"], errors="coerce").dt.normalize().to_numpy(),
                      "idx": d.index.to_numpy()})
    k = k[k["dt"].notna()]
    dup = k.duplicated(["lg", "h", "a", "hg", "ag"], keep=False)
    rows, seen = [], set()
    for _lg, g in _groups(k[dup].sort_values("dt", kind="mergesort"), ["lg", "h", "a", "hg", "ag"]):
        dts, ids = g["dt"], g["idx"]
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                gap = int((dts[y] - dts[x]) / np.timedelta64(1, "D"))
                if gap > window_days:
                    break
                rows.append((ids[x], ids[y], _lg, gap, "both_sides"))
                seen.add(frozenset((ids[x], ids[y])))
    # SAME DAY, SAME SCORE, ONE SIDE THE SAME CLUB: a club does not play twice in a day, so the two
    # rows are one match whatever the other side is called (match-level evidence; no club is renamed).
    same = k.duplicated(["lg", "dt", "hg", "ag"], keep=False)
    for _lg, g in _groups(k[same], ["lg", "dt", "hg", "ag"]):
        r = list(zip(g["idx"], g["h"], g["a"]))
        for x in range(len(r)):
            for y in range(x + 1, len(r)):
                (i1, h1, a1), (i2, h2, a2) = r[x], r[y]
                if (h1 == h2) != (a1 == a2) and frozenset((i1, i2)) not in seen:
                    rows.append((i1, i2, _lg, 0, "one_side_same_day"))
    return pd.DataFrame(rows, columns=cols + ["kind"])


def self_containing_rows(df: pd.DataFrame, pairs: pd.DataFrame, n: int = 5) -> pd.Series:
    """True for a row whose home or away club's last-`n` form window (src/feature_engineering:
    per (league, team) in date order, shift(1)) contains a copy of the row's OWN match."""
    flag = pd.Series(False, index=df.index)
    if pairs.empty:
        return flag
    twin = defaultdict(set)
    for i, j in zip(pairs["i"], pairs["j"]):
        twin[i].add(j)
        twin[j].add(i)
    s = df.assign(_ix=df.index, _dt=pd.to_datetime(df["date"], errors="coerce")) \
          .sort_values("_dt", kind="mergesort")
    long = pd.concat([s[["league", "home_team", "_dt", "_ix"]].rename(columns={"home_team": "team"}),
                      s[["league", "away_team", "_dt", "_ix"]].rename(columns={"away_team": "team"})],
                     ignore_index=True).sort_values(["league", "team", "_dt"], kind="mergesort")
    hit = set()
    for _, g in long.groupby(["league", "team"], sort=False):
        ix = g["_ix"].tolist()
        for p, r in enumerate(ix):
            t = twin.get(r)
            if t and any(w in t for w in ix[max(0, p - n):p]):
                hit.add(r)
    flag.loc[list(hit)] = True
    return flag


def over25_form_auc(df: pd.DataFrame, mask: pd.Series | None = None, n: int = 5) -> float | None:
    """AUC of the over-2.5 form signal (mean of home/away over25 rate over the last n, computed
    exactly as src/feature_engineering does it) against the over25 label, on `mask` rows."""
    from sklearn.metrics import roc_auc_score
    from src.feature_engineering import _build_team_centric, _rolling
    m = df.sort_values("date", kind="mergesort")
    tc = _build_team_centric(m)
    tc["roll"] = _rolling(tc, "over25", n)
    h = tc[tc["is_home"] == 1].set_index("_src_idx")["roll"]
    a = tc[tc["is_home"] == 0].set_index("_src_idx")["roll"]
    form = (h.reindex(m.index) + a.reindex(m.index)) / 2
    y = m["over25"]
    ok = form.notna() & y.notna()
    if mask is not None:
        ok &= mask.reindex(m.index).fillna(False).astype(bool)
    if ok.sum() < 50 or y[ok].nunique() < 2:
        return None
    return float(roc_auc_score(y[ok], form[ok]))


def dedupe_matches(df: pd.DataFrame, window_days: int = WINDOW_DAYS, source_col: str = "source"):
    """(frame with one row per match, report). See the module docstring for the rule."""
    rep = {"rule": f"same league, same canonical clubs (src.team_names), same score, dates <= "
                   f"{window_days} d apart - or same day, same score, one side the same club; survivor by SOURCE_PRIORITY {SOURCE_PRIORITY}, then most "
                   f"populated, then earliest; survivor NaNs filled from the dropped copy",
           "pairs": 0, "same_day_pairs": 0, "near_date_pairs": 0, "rows_removed": 0,
           "cells_filled_from_copy": 0, "examples": []}
    if df.empty:
        return df, rep
    pairs = duplicate_pairs(df, window_days)
    rep["pairs"] = int(len(pairs))
    rep["same_day_pairs"] = int((pairs["gap_days"] == 0).sum())
    rep["near_date_pairs"] = int((pairs["gap_days"] > 0).sum())
    rep["one_side_same_day_pairs"] = int((pairs["kind"] == "one_side_same_day").sum())
    if pairs.empty:
        return df, rep
    # connected components: a match held three times is one group
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(pairs["i"], pairs["j"]):
        parent[find(i)] = find(j)
    groups = defaultdict(list)
    for x in list(parent):
        groups[find(x)].append(x)
    src = df[source_col].map(SOURCE_PRIORITY) if source_col in df.columns \
        else pd.Series(np.nan, index=df.index)
    prio = src.fillna(len(SOURCE_PRIORITY)).astype(int)
    filled_n = df.notna().sum(axis=1)
    dts = pd.to_datetime(df["date"], errors="coerce")
    out = df.copy()
    drop = []
    for members in groups.values():
        members = sorted(members, key=lambda r: (prio[r], -filled_n[r], dts[r], str(r)))
        keep, rest = members[0], members[1:]
        for r in rest:
            gaps = out.loc[keep].isna() & df.loc[r].notna()
            gaps = gaps[gaps].index.difference(["date", "league", "season", "home_team", "away_team",
                                                source_col])
            if len(gaps):
                out.loc[keep, gaps] = df.loc[r, gaps].to_numpy()
                rep["cells_filled_from_copy"] += int(len(gaps))
            if len(rep["examples"]) < 10:
                rep["examples"].append({
                    "league": str(df.at[keep, "league"]),
                    "kept": f"{str(dts[keep])[:10]} {df.at[keep, 'home_team']} v {df.at[keep, 'away_team']}"
                            f" ({df.at[keep, source_col] if source_col in df.columns else '?'})",
                    "dropped": f"{str(dts[r])[:10]} {df.at[r, 'home_team']} v {df.at[r, 'away_team']}"
                               f" ({df.at[r, source_col] if source_col in df.columns else '?'})",
                    "score": f"{df.at[r, 'home_goals']:.0f}-{df.at[r, 'away_goals']:.0f}"})
        drop += rest
    out = out.drop(index=drop)
    rep["rows_removed"] = int(len(drop))
    return out, rep
