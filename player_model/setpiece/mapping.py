"""
Understat -> our ids (owner decision B, 2026-11-02).

MATCH. An Understat match is our fixture when, among the fixtures we hold within +-1 day of its
date (kick-off dates differ by time zone), its home club resolves (src.team_names.resolve) onto a
home club of that window, its away club onto an away club, and EXACTLY ONE fixture has that pair.
Anything else (no hit, ambiguity) is unmapped and counted. No league mapping is needed: the fixture
carries our league.

PLAYER. Inside a mapped fixture and side, an Understat roster player is our player when exactly one
of our players of that side has a COMPATIBLE name and that player is compatible with no other
Understat roster player of the side (ambiguity refused both ways). Compatible: after accent
stripping and letter folding, every full token of our (API-Football) name is a token of the Understat
name and every initial ('M.') is the first letter of one of its tokens; at least one full token is
required. Each such fixture is one vote; an Understat player maps to our player_id only when ALL its
votes agree (and no other Understat id claims the same player_id). Conflicts are refused and counted.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

import pandas as pd

# Latin letters NFKD does not decompose (character folding, not name aliases).
_FOLD = str.maketrans({"ø": "o", "Ø": "o", "æ": "ae", "Æ": "ae", "œ": "oe", "Œ": "oe", "ß": "ss",
                       "đ": "d", "Đ": "d", "ð": "d", "ł": "l", "Ł": "l", "ı": "i", "þ": "th"})

MATCH_WINDOW_DAYS = 1


def name_tokens(s) -> list[str]:
    if not isinstance(s, str):
        return []
    s = unicodedata.normalize("NFKD", s.translate(_FOLD))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return [t for t in re.split(r"[^a-z0-9]+", s) if t]


def compatible(ours, theirs) -> bool:
    a, u = name_tokens(ours), name_tokens(theirs)
    full = [t for t in a if len(t) > 1]
    inits = [t for t in a if len(t) == 1]
    if not full or not u:
        return False
    us = set(u)
    firsts = {t[0] for t in u}
    return all(t in us for t in full) and all(i in firsts for i in inits)


def _resolver():
    from src.team_names import resolve
    return resolve


def map_matches(matches: pd.DataFrame, history: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """(match_id, fixture_id, league) for every Understat match that maps; plus a report."""
    resolve = _resolver()
    fx = (history[["fixture_id", "date", "league", "home_team", "away_team"]]
          .drop_duplicates("fixture_id").dropna(subset=["fixture_id", "date"]))
    fx = fx.assign(_day=pd.to_datetime(fx["date"], errors="coerce").dt.normalize())
    by_day = {d: g for d, g in fx.groupby("_day")}
    rep = {"matches": 0, "mapped": 0, "no_window": 0, "unresolved": 0, "ambiguous": 0}
    out = []
    m = matches[matches["is_result"].astype(bool)] if "is_result" in matches.columns else matches
    for mid, dt, ht, at in zip(m["match_id"], m["datetime"], m["h_title"], m["a_title"]):
        rep["matches"] += 1
        day = pd.to_datetime(dt, errors="coerce")
        if day != day:
            rep["no_window"] += 1
            continue
        day = day.normalize()
        win = [by_day[d] for d in (day + pd.Timedelta(days=k) for k in range(-MATCH_WINDOW_DAYS, MATCH_WINDOW_DAYS + 1))
               if d in by_day]
        if not win:
            rep["no_window"] += 1
            continue
        w = pd.concat(win)
        hh = resolve(str(ht), sorted(w["home_team"].dropna().astype(str).unique()))
        aa = resolve(str(at), sorted(w["away_team"].dropna().astype(str).unique()))
        if not hh or not aa:
            rep["unresolved"] += 1
            continue
        hit = w[(w["home_team"] == hh) & (w["away_team"] == aa)]     # names from the pool itself
        if hit["fixture_id"].nunique() != 1:
            rep["ambiguous"] += 1
            continue
        out.append({"match_id": int(mid), "fixture_id": int(hit["fixture_id"].iloc[0]),
                    "league": hit["league"].iloc[0]})
        rep["mapped"] += 1
    res = pd.DataFrame(out, columns=["match_id", "fixture_id", "league"])
    dup = res["fixture_id"].duplicated(keep=False)        # two Understat matches on one fixture
    rep["ambiguous"] += int(dup.sum())
    return res[~dup].reset_index(drop=True), rep


def map_players(rosters: pd.DataFrame, match_map: pd.DataFrame, history: pd.DataFrame
                ) -> tuple[pd.DataFrame, dict]:
    """(us_player_id, player_id, n_votes) for every Understat player whose votes all agree."""
    rep = {"roster_rows": 0, "voted": 0, "no_candidate": 0, "ambiguous": 0,
           "conflicting_us_ids": 0, "shared_player_ids": 0}
    if match_map.empty or rosters.empty:
        return pd.DataFrame(columns=["us_player_id", "player_id", "n_votes"]), rep
    fix = dict(zip(match_map["match_id"], match_map["fixture_id"]))
    h = history[history["fixture_id"].isin(set(fix.values()))]
    ours = {(int(f), int(bool(ih))): g[["player_id", "player_name"]].drop_duplicates("player_id")
            for (f, ih), g in h.groupby(["fixture_id", "is_home"])}
    votes: dict = defaultdict(lambda: defaultdict(int))
    r = rosters[rosters["match_id"].isin(fix.keys())]
    for (mid, side), g in r.groupby(["match_id", "h_a"]):
        cand = ours.get((int(fix[mid]), int(side == "h")))
        rep["roster_rows"] += len(g)
        if cand is None or cand.empty:
            rep["no_candidate"] += len(g)
            continue
        us = list(zip(g["us_player_id"], g["player"]))
        comp = {uid: [pid for pid, pn in zip(cand["player_id"], cand["player_name"]) if compatible(pn, un)]
                for uid, un in us}
        claimed = defaultdict(int)
        for pids in comp.values():
            for p in pids:
                claimed[p] += 1
        for uid, pids in comp.items():
            if not pids:
                rep["no_candidate"] += 1
            elif len(pids) > 1 or claimed[pids[0]] > 1:
                rep["ambiguous"] += 1
            else:
                votes[int(uid)][int(pids[0])] += 1
                rep["voted"] += 1
    rows = []
    for uid, d in votes.items():
        if len(d) != 1:
            rep["conflicting_us_ids"] += 1
            continue
        (pid, n), = d.items()
        rows.append({"us_player_id": uid, "player_id": pid, "n_votes": int(n)})
    res = pd.DataFrame(rows, columns=["us_player_id", "player_id", "n_votes"])
    shared = res["player_id"].duplicated(keep=False)
    rep["shared_player_ids"] = int(shared.sum())
    return res[~shared].reset_index(drop=True), rep
