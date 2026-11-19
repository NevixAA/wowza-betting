"""Independent match-result lookup and settlement (no production imports, no src/team_names).

Sources: the committed output/fd_history.parquet (football-data.co.uk) and
output/af_history.parquet (API-Football) at the audited ref. Production settles from LIVE
fetches (football-data website, OddsAPI scores, API-Football) inside update_results.py, so
these are a genuinely different read path.

Matching is league-scoped, date window +/- 1 day (UTC vs local calendar day), and requires BOTH
team names to match. It refuses ambiguous matches (two candidates within 0.02 similarity) and
reports orientation (a fixture found only with home/away swapped is flagged, never silently used
for a side-specific market).
"""
from __future__ import annotations

import io
import re
import unicodedata
from difflib import SequenceMatcher

import pandas as pd

from .common import REF, show

_STOP = {"fc", "cf", "afc", "sc", "ac", "cd", "ud", "sd", "fk", "if", "bk", "sk", "club", "de", "the",
         "calcio", "football", "futbol", "clube", "esporte", "sv", "tsg", "vfl", "vfb", "1", "ii",
         "rc", "ca", "cs", "ec",
         "se", "ss", "us", "as", "kv", "krc", "ff", "jk", "aik", "and", "&"}


def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return " ".join(s.split())


def core(name: str) -> str:
    toks = [t for t in norm(name).split() if t not in _STOP]
    return " ".join(toks) or norm(name)


def sim(a: str, b: str) -> float:
    na, nb = norm(a), norm(b)
    if na == nb:
        return 1.0
    ca, cb = core(a), core(b)
    if ca == cb:
        return 0.97
    if ca and cb and (ca in cb or cb in ca) and min(len(ca), len(cb)) >= 4:
        return 0.9
    ta, tb = set(ca.split()), set(cb.split())
    jac = len(ta & tb) / max(1, len(ta | tb))
    score = max(SequenceMatcher(None, ca, cb).ratio() * 0.85, 0.5 + 0.4 * jac if ta & tb else 0.0)
    # abbreviations: initials ("QPR" = Queens Park Rangers) and a shared >=4-char first-token
    # prefix ("Wolves" / "Wolverhampton", "Argentinos JRS" / "Argentinos Juniors")
    la, lb = norm(a).split(), norm(b).split()
    for x, y in ((la, lb), (lb, la)):
        if len(x) == 1 and len(y) >= 2 and len(x[0]) == len(y) and x[0] == "".join(w[0] for w in y):
            score = max(score, 0.85)
    xa, xb = ca.split(), cb.split()
    fa, fb = xa[0], xb[0]
    k = 0
    while k < min(len(fa), len(fb)) and fa[k] == fb[k]:
        k += 1
    if k >= 4 and _rest_compatible(xa[1:], xb[1:]):
        score = max(score, 0.8)
    return score


def _subseq(short: str, long: str) -> bool:
    it = iter(long)
    return all(ch in it for ch in short)


def _rest_compatible(ra: list, rb: list) -> bool:
    """Tokens after the shared first token must not CONFLICT: one side may omit them ("Wolves" vs
    "Wolverhampton Wanderers"), or one may abbreviate the other ("jrs" in "juniors"). "united" vs
    "city" conflicts, so Manchester United never matches Manchester City."""
    if not ra or not rb:
        return True
    for x, y in zip(ra, rb):
        if x == y:
            continue
        s_, l_ = (x, y) if len(x) <= len(y) else (y, x)
        if not (s_[0] == l_[0] and _subseq(s_, l_)):
            return False
    return True


def load_results(ref: str = REF) -> pd.DataFrame:
    frames = []
    fd = pd.read_parquet(io.BytesIO(show(ref, "output/fd_history.parquet")))
    fd = fd.rename(columns={"home_goals": "hg", "away_goals": "ag", "ht_home_goals": "hthg",
                            "ht_away_goals": "htag"})
    fd["src"] = "fd"
    frames.append(fd[["league", "date", "home_team", "away_team", "hg", "ag", "hthg", "htag", "src"]])
    af = pd.read_parquet(io.BytesIO(show(ref, "output/af_history.parquet")))
    af = af.rename(columns={"FTHG": "hg", "FTAG": "ag"})
    af["hthg"] = pd.NA
    af["htag"] = pd.NA
    af["src"] = "af"
    frames.append(af[["league", "date", "home_team", "away_team", "hg", "ag", "hthg", "htag", "src"]])
    r = pd.concat(frames, ignore_index=True)
    r["date"] = pd.to_datetime(r["date"], errors="coerce").dt.normalize()
    r = r[r["date"] >= "2026-03-01"].dropna(subset=["hg", "ag"])
    return r.reset_index(drop=True)


class ResultIndex:
    def __init__(self, results: pd.DataFrame):
        self.by_league = {lg: g.reset_index(drop=True) for lg, g in results.groupby("league")}

    def find(self, league: str, match_date: str, home: str, away: str) -> dict:
        """Return {'status': MATCHED|SWAPPED|AMBIGUOUS|CONFLICT|NOT_FOUND, hg, ag, src, score, n_sources}."""
        g = self.by_league.get(league)
        if g is None:
            return {"status": "NOT_FOUND", "why": "league_absent"}
        d = pd.Timestamp(str(match_date)[:10])
        w = g[(g["date"] >= d - pd.Timedelta(days=1)) & (g["date"] <= d + pd.Timedelta(days=1))]
        if w.empty:
            return {"status": "NOT_FOUND", "why": "no_fixture_in_window"}
        cands = []
        for r in w.itertuples():
            sh, sa = sim(home, r.home_team), sim(away, r.away_team)
            xh, xa = sim(home, r.away_team), sim(away, r.home_team)
            straight, swapped = min(sh, sa), min(xh, xa)
            if straight >= 0.75:
                cands.append((straight, False, r))
            elif swapped >= 0.75:
                cands.append((swapped, True, r))
        if not cands:
            return {"status": "NOT_FOUND", "why": "no_name_match"}
        cands.sort(key=lambda c: -c[0])
        best = cands[0]
        # same fixture from two sources is not ambiguity; different scores is a CONFLICT
        same_fx = [c for c in cands if c[0] >= best[0] - 0.02]
        scores = {(int(c[2].hg), int(c[2].ag)) if not c[1] else (int(c[2].ag), int(c[2].hg)) for c in same_fx}
        teams = {(core(c[2].home_team), core(c[2].away_team)) for c in same_fx}
        srcs = {c[2].src for c in same_fx}
        if len(same_fx) > 1 and len(scores) > 1:
            # two sources disagree on the score, or two different fixtures match equally
            status = "CONFLICT" if len(srcs) > 1 and len({c[2].date for c in same_fx}) == 1 else "AMBIGUOUS"
            return {"status": status, "why": f"{len(same_fx)} candidates scores={sorted(scores)} teams={sorted(teams)}"}
        sc, swapped, r = best
        hg, ag = (int(r.hg), int(r.ag)) if not swapped else (int(r.ag), int(r.hg))
        return {"status": "SWAPPED" if swapped else "MATCHED", "hg": hg, "ag": ag, "src": "+".join(sorted(srcs)),
                "score": round(sc, 3), "res_home": r.home_team, "res_away": r.away_team,
                "res_date": str(r.date.date())}


# ── settlement rules (written from the market definitions, not from production code) ──
LINES = {"ou25": 2.5, "over15": 1.5, "over35": 3.5}


def outcome(market: str, side: str, hg: int, ag: int) -> str:
    """WIN / LOSS for a full-time 90-minute market. Raises on an unknown market/side."""
    total = hg + ag
    m, s = str(market).lower(), str(side).upper()
    if m in LINES:
        line = LINES[m]
        if s == "OVER":
            return "WIN" if total > line else "LOSS"
        if s == "UNDER":
            return "WIN" if total < line else "LOSS"
        raise ValueError(f"invalid side {side!r} for {market}")
    if m == "btts":
        both = hg > 0 and ag > 0
        if s in ("YES", "BTTS_YES"):
            return "WIN" if both else "LOSS"
        if s in ("NO", "BTTS_NO"):
            return "LOSS" if both else "WIN"
        raise ValueError(f"invalid side {side!r} for btts")
    raise ValueError(f"unknown market {market!r}")


def pnl(result: str, odds: float, stake: float = 1.0):
    if result == "WIN":
        return round(stake * (float(odds) - 1.0), 6)
    if result == "LOSS":
        return -float(stake)
    if result == "VOID":
        return 0.0
    return None  # unsettled: never 0, never a loss
