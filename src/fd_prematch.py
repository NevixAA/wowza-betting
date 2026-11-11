"""Football-data PRE-MATCH O/U 2.5 prices joined ONE-TO-ONE to the served history (owner 2026-11-11: a closing price
may never be a bet price; the study's Standard bet prices come from football-data pre-match, through a certified
join).

    rows = load_fd_raw(fd_raw_dir)                       # every fd_raw row: league, date, clubs, score, prices
    j, report = join(rows, served, aliases, crosswalk)   # served row -> its fd_raw row (or none), per-league rates

PRICES (the study's single-book rule, src.league_bars.fd_ou25_pairs): pre-match = Bet365 (B365>2.5 / B365<2.5),
else Pinnacle (P>2.5 / P<2.5), both sides from the SAME book; closing (CLV only) = B365C, else PC. Average / max
columns are never a bet price.

THE JOIN. Within ONE league: both clubs resolved by NAME onto the served club pool (src.team_names.resolve_identity
with the loader's identity map, plus the PROVEN crosswalk config/fd_raw_crosswalk.json), on the SAME date - or one
day apart when that pair of clubs is unique within +-1 day (a provider / UK date difference). ONE-TO-ONE: a served
row claimed by two fd rows, or an fd row that fits two served rows, is dropped (counted). The score is a CHECK
only: a matched pair whose full-time scores disagree is rejected (counted) - a score never pairs anything (no
league + date + score fallback).

THE CROSSWALK is proven, never typed: prove_crosswalk() accepts an fd spelling X -> served club Y in a league only
when, on at least MIN_DATES different dates, an fd row with X and a resolved opponent sits on the same date as a
served row with Y and that opponent in the same home/away roles, with the same score - and X fits no other club.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CROSSWALK = ROOT / "config" / "fd_raw_crosswalk.json"
PREMATCH = (("B365", "B365>2.5", "B365<2.5"), ("P", "P>2.5", "P<2.5"))
CLOSE = (("B365C", "B365C>2.5", "B365C<2.5"), ("PC", "PC>2.5", "PC<2.5"))
MIN_DATES = 3                       # the OddsAPI alias proof's own rule (src.oddsapi_history.MIN_ALIAS_DATES)


def _pair(raw: pd.DataFrame, books) -> tuple[pd.Series, pd.Series, pd.Series]:
    o = pd.Series(np.nan, index=raw.index, dtype=float)
    u = pd.Series(np.nan, index=raw.index, dtype=float)
    s = pd.Series("", index=raw.index, dtype=object)
    for name, oc, uc in books:
        if oc not in raw.columns or uc not in raw.columns:
            continue
        a, b = pd.to_numeric(raw[oc], errors="coerce"), pd.to_numeric(raw[uc], errors="coerce")
        take = o.isna() & (a > 1) & (b > 1)
        o[take], u[take], s[take] = a[take], b[take], name
    return o, u, s


def load_fd_raw(fd_raw_dir) -> pd.DataFrame:
    from src.standard_repair import FD_CODES
    inv = {v: k for k, v in FD_CODES.items()}
    frames = []
    for p in sorted(Path(fd_raw_dir).glob("*_*.csv")):
        lg = inv.get(p.stem.rsplit("_", 1)[0])
        if not lg:
            continue
        try:
            raw = pd.read_csv(p, encoding="utf-8-sig", on_bad_lines="skip", low_memory=False)
        except UnicodeDecodeError:
            raw = pd.read_csv(p, encoding="cp1252", encoding_errors="replace", on_bad_lines="skip", low_memory=False)
        if not {"Date", "HomeTeam", "AwayTeam"} <= set(raw.columns):
            continue
        o, u, s = _pair(raw, PREMATCH)
        oc, uc, sc = _pair(raw, CLOSE)
        frames.append(pd.DataFrame({
            "league": lg, "date": pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce").dt.normalize(),
            "fd_home": raw["HomeTeam"].astype(str).str.strip(), "fd_away": raw["AwayTeam"].astype(str).str.strip(),
            "fd_hg": pd.to_numeric(raw.get("FTHG"), errors="coerce"),
            "fd_ag": pd.to_numeric(raw.get("FTAG"), errors="coerce"),
            "pm_over": o, "pm_under": u, "pm_source": s, "close_over": oc, "close_under": uc, "close_source": sc,
            "file": p.name}))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out[out["date"].notna()].reset_index(drop=True) if len(out) else out


def certified_ou25(pred: pd.DataFrame, aliases: dict, fd_raw, crosswalk: dict | None = None
                   ) -> tuple[pd.DataFrame, dict]:
    """The study's Standard O/U 2.5 prices through the CERTIFIED join (protocol v5): `pred` rows (row_id
    'date|league|home|away') + yes / no = the matched fd row's PRE-MATCH pair (Bet365, else Pinnacle) and
    yes_c / no_c = its CLOSING pair (CLV only). Unjoined rows keep NaN. Returns (rows, {league: report})."""
    p = pred.copy()
    parts = p["row_id"].astype(str).str.split("|", expand=True)
    sv = pd.DataFrame({"league": p["league"].to_numpy(), "date": pd.to_datetime(p["date"]).dt.normalize().to_numpy(),
                       "home_team": parts[2].to_numpy(), "away_team": parts[3].to_numpy()})
    fx = sv.drop_duplicates().reset_index(drop=True)
    j, rep = join(load_fd_raw(fd_raw), fx, aliases, crosswalk if crosswalk is not None else load_crosswalk())
    key = ["league", "date", "home_team", "away_team"]
    m = sv.merge(j[key + ["pm_over", "pm_under", "close_over", "close_under", "pm_source"]], on=key, how="left")
    p["yes"], p["no"] = m["pm_over"].to_numpy(), m["pm_under"].to_numpy()
    p["yes_c"], p["no_c"] = m["close_over"].to_numpy(), m["close_under"].to_numpy()
    p["price_book"] = m["pm_source"].to_numpy()
    p["home"], p["away"] = parts[2].to_numpy(), parts[3].to_numpy()
    return p, {r["league"]: r for r in rep}


CERTIFIED = "FD_CERTIFIED_JOIN"


def crosswalk_sha256(path: Path = CROSSWALK) -> str:
    """sha256 of the crosswalk file's text with LF line ends (the same on every checkout)."""
    import hashlib
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def universe_source(certified: bool) -> dict:
    """The stamp `bars_opt.py universe` writes beside universe_standard.parquet (protocol v5 checks it)."""
    if not certified:
        return {"over25": "LEGACY_NAME_JOIN"}
    return {"over25": CERTIFIED, "crosswalk_sha256": crosswalk_sha256(),
            "bet_books": [b for b, _, _ in PREMATCH], "close_books_clv_only": [b for b, _, _ in CLOSE]}


def load_crosswalk(path: Path = CROSSWALK) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")).get("map", {})
    except FileNotFoundError:
        return {}


def _resolver(pool, aliases: dict | None, cross: dict | None):
    from src import team_names as TN
    cache: dict = {}

    def res(n):
        if n not in cache:
            c = (cross or {}).get(n)
            cache[n] = c if c in pool else (TN.resolve_identity(n, sorted(pool), aliases) if isinstance(n, str) else None)
        return cache[n]
    return res


def join(fd: pd.DataFrame, served: pd.DataFrame, aliases: dict | None = None, crosswalk: dict | None = None
         ) -> tuple[pd.DataFrame, list[dict]]:
    """served: league, date, home_team, away_team (+ home_goals / away_goals for the check). Returns (served rows +
    fd prices of their ONE matched fd row, per-league report)."""
    aliases, crosswalk = aliases or {}, crosswalk or {}
    sv = served.copy()
    sv["date"] = pd.to_datetime(sv["date"]).dt.normalize()
    sv["row_ix"] = np.arange(len(sv))
    out_cols = ["pm_over", "pm_under", "pm_source", "close_over", "close_under", "close_source", "fd_file",
                "fd_home", "fd_away"]
    for c in out_cols:
        sv[c] = np.nan if c not in ("pm_source", "close_source", "fd_file", "fd_home", "fd_away") else ""
    report = []
    for lg, g in sv.groupby("league"):
        f = fd[fd["league"] == lg]
        rec = Counter(served=len(g), fd_rows=len(f))
        if f.empty:
            report.append({"league": lg, **rec, "joined": 0, "join_rate": 0.0})
            continue
        pool = set(g["home_team"].astype(str)) | set(g["away_team"].astype(str))
        res = _resolver(pool, aliases.get(lg), crosswalk.get(lg))
        idx = defaultdict(list)
        for r in g.itertuples(index=False):
            idx[(str(r.home_team), str(r.away_team))].append((r.date, r.row_ix))
        claims = defaultdict(list)
        for fi, r in enumerate(f.itertuples(index=False)):
            h, a = res(r.fd_home), res(r.fd_away)
            if not h or not a or h == a:
                rec["fd_unresolved_club"] += 1
                continue
            cands = idx.get((h, a), [])
            exact = [i for d, i in cands if d == r.date]
            near = [i for d, i in cands if abs((d - r.date).days) <= 1]
            hit = exact if exact else near
            if len(hit) != 1:
                rec["fd_no_or_ambiguous_served_row" if not hit else "fd_fits_two_served_rows"] += 1
                continue
            claims[hit[0]].append(f.index[fi])
        joined = 0
        for si, fis in claims.items():
            if len(fis) != 1:
                rec["served_claimed_twice"] += 1
                continue
            fr = fd.loc[fis[0]]
            srow = sv.iloc[si]
            hg, ag = srow.get("home_goals"), srow.get("away_goals")
            if pd.notna(fr["fd_hg"]) and pd.notna(hg) and (fr["fd_hg"] != hg or fr["fd_ag"] != ag):
                rec["score_check_failed"] += 1
                continue
            for c, src in (("pm_over", "pm_over"), ("pm_under", "pm_under"), ("pm_source", "pm_source"),
                           ("close_over", "close_over"), ("close_under", "close_under"),
                           ("close_source", "close_source"), ("fd_file", "file"), ("fd_home", "fd_home"),
                           ("fd_away", "fd_away")):
                sv.iat[si, sv.columns.get_loc(c)] = fr[src]
            joined += 1
        report.append({"league": lg, **rec, "joined": joined,
                       "join_rate": round(joined / len(g), 4) if len(g) else 0.0,
                       "fd_row_join_rate": round(joined / len(f), 4) if len(f) else 0.0})
    return sv.drop(columns="row_ix"), report


def prove_crosswalk(fd: pd.DataFrame, served: pd.DataFrame, aliases: dict | None = None) -> dict:
    """{league: {fd spelling: served club}} PROVEN on >= MIN_DATES dates (same date, same roles, a resolved
    opponent, the same score); a spelling that fits two clubs is refused."""
    aliases = aliases or {}
    sv = served.copy()
    sv["date"] = pd.to_datetime(sv["date"]).dt.normalize()
    out = {}
    for lg, g in sv.groupby("league"):
        f = fd[fd["league"] == lg]
        if f.empty:
            continue
        pool = set(g["home_team"].astype(str)) | set(g["away_team"].astype(str))
        res = _resolver(pool, aliases.get(lg), None)
        by_day = defaultdict(list)
        for r in g.itertuples(index=False):
            by_day[r.date].append(r)
        votes = defaultdict(Counter)
        for r in f.itertuples(index=False):
            h, a = res(r.fd_home), res(r.fd_away)
            for unk, known, role in ((r.fd_home, a, "home"), (r.fd_away, h, "away")):
                if (role == "home" and h) or (role == "away" and a) or not known:
                    continue
                for s in by_day.get(r.date, []):
                    opp = s.away_team if role == "home" else s.home_team
                    me = s.home_team if role == "home" else s.away_team
                    if str(opp) == known and s.home_goals == r.fd_hg and s.away_goals == r.fd_ag:
                        votes[unk][str(me)] += 1
        m = {}
        for unk, c in votes.items():
            (best, n), *rest = c.most_common() + [(None, 0)]
            if n >= MIN_DATES and (not rest or rest[0][1] == 0):
                m[unk] = best
        if m:
            out[lg] = m
    return out


def used_names(joined: pd.DataFrame) -> dict:
    """{league: {fd spelling: {served, joined_rows}}} the certified join used where the spellings differ - each fd
    spelling to exactly ONE served club on >= MIN_DATES joined rows (the crosswalk's evidence)."""
    j = joined[joined["fd_file"].astype(str) != ""]
    votes: dict = {}
    for r in j.itertuples(index=False):
        for fn, sn in ((r.fd_home, r.home_team), (r.fd_away, r.away_team)):
            if fn != sn:
                votes.setdefault(r.league, {}).setdefault(fn, Counter())[sn] += 1
    out: dict = {}
    for lg, m in votes.items():
        for n, c in m.items():
            if len(c) == 1:
                (sn, k), = c.items()
                if k >= MIN_DATES:
                    out.setdefault(lg, {})[n] = {"served": sn, "joined_rows": int(k)}
    return out
