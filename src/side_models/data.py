"""Verified corners / cards match table: a fixture's count is used only when TWO sources agree on it.
=================================================================================================

Sources (both committed, provenance in reports/R_SIDE_DATA_TRUTH_LIVE_2026-11-16.md):
  * API-Football /fixtures/statistics  output/collect/af_history_fullstats_fill.parquet (by fixture_id),
    with the fixture list output/strength/competition_fixtures.parquet (date, team ids, FT score).
  * football-data raw season CSVs        output/collect/fd_raw/<CODE>_<season>.csv (+ manifest.json
    with the sha256 of every file): HC/AC corners, HY/AY yellows, HR/AR reds, FTHG/FTAG.

JOIN WITHOUT NAME GUESSING. football-data names a club its own way. The crosswalk is learned from the
data, per league, with no string similarity:
  1. ANCHORS: a (league, date, FT home goals, FT away goals) group that holds exactly one fd match and
     exactly one API-Football fixture pairs those two matches.
  2. VOTES: every anchor votes fd home name -> AF home team id and fd away name -> AF away team id.
  3. MAP: an fd name maps to an AF team id only when that id has >= MIN_VOTES votes and >= MIN_SHARE of
     the name's votes, and the map is one-to-one inside the league. Anything else stays unmapped.
  4. JOIN: fd row -> AF fixture on (league, AF home id, AF away id) with the date equal, else +-1 day
     when exactly one candidate exists (kickoff-date convention). Unmatched rows stay unmatched.

VERIFIED (per fixture, per family):
  * corners: AF home/away corners == fd HC/AC exactly.
  * cards:   the count is API-Football's (yellow + red per team; the settlement source on main). fd must
             agree on the RED count of each team exactly, and on the YELLOW count exactly for a team
             without a red card. For a team WITH a red card the two providers count a second-yellow
             dismissal differently (AF lists both yellows, fd does not), so AF yellow - fd yellow must
             lie in [0, 2 x reds] - the documented convention band (audit G9), never a wider tolerance.
A fixture failing its check is NOT used for that family (missing stays missing; nothing is corrected
or averaged). Rows where every AF corner and card value is 0 are refused (suspected zero-fill, audit).
The cards rule above is the owner's (2026-11-18; closes audit G9).

SINGLE SOURCE (owner 2026-11-18). A league football-data publishes no corners/cards for (the New-Format
leagues) uses API-Football as its single certified source. A fixture's counts are used only when they
pass the sanity rules (`SANITY`: every count present, none negative or impossible, not all-zero);
nothing is filled. Such rows carry basis 'single_source' (two-source rows: 'two_source').

API-FOOTBALL'S SIDE has two stores: the fullstats fill (by fixture_id) and, for fixtures the fill does
not hold, the side-stats sidecar output/collect/af_side_stats.parquet (scripts/collect_side_stats.py,
the capped daily refresh). Both are parsed under the same owner null rule (src/af_team_stats).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FILL = ROOT / "output" / "collect" / "af_history_fullstats_fill.parquet"
FIXTURES = ROOT / "output" / "strength" / "competition_fixtures.parquet"
FD_RAW = ROOT / "output" / "collect" / "fd_raw"
SIDECAR = ROOT / "output" / "collect" / "af_side_stats.parquet"
STAT_COLS = ["af_hc", "af_ac", "af_hy", "af_ay", "af_hr", "af_ar"]
#: impossible-value rules of the side-data audit (2026-11-16): a count beyond these is not real
SANITY = {"max_match_corners": 30, "max_match_cards": 22, "max_team_red": 5, "max_team_yellow": 16}

MIN_VOTES = 5
MIN_SHARE = 0.95
FT_STATUSES = ("FT", "AET", "PEN")


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def load_fd_raw(leagues: list, fd_dir: Path | None = None) -> pd.DataFrame:
    """Every football-data raw row of `leagues` (the manifest's valid files only)."""
    fd_dir = Path(fd_dir or FD_RAW)
    man = json.loads((fd_dir / "manifest.json").read_text(encoding="utf-8"))
    parts = []
    for fn, meta in sorted(man["files"].items()):
        if meta.get("league") not in leagues or not meta.get("valid", False):
            continue
        d = pd.read_csv(fd_dir / fn, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        d = d[d.get("HomeTeam", pd.Series(dtype=str)).astype(str).str.strip() != ""]
        out = pd.DataFrame({
            "league": meta["league"],
            "fd_file": fn,
            "date": pd.to_datetime(d["Date"], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d"),
            "fd_home": d["HomeTeam"].str.strip(),
            "fd_away": d["AwayTeam"].str.strip(),
            "fd_fthg": _num(d["FTHG"]), "fd_ftag": _num(d["FTAG"]),
            "fd_hc": _num(d.get("HC")), "fd_ac": _num(d.get("AC")),
            "fd_hy": _num(d.get("HY")), "fd_ay": _num(d.get("AY")),
            "fd_hr": _num(d.get("HR")), "fd_ar": _num(d.get("AR")),
        })
        parts.append(out)
    if not parts:
        return pd.DataFrame(columns=["league", "fd_file", "date", "fd_home", "fd_away", "fd_fthg", "fd_ftag",
                                     "fd_hc", "fd_ac", "fd_hy", "fd_ay", "fd_hr", "fd_ar"])
    fd = pd.concat(parts, ignore_index=True)
    return fd[fd["date"].notna()].reset_index(drop=True)


SIDE_COLS = ["fixture_id", "home_id", "away_id", "home_corners", "away_corners", "home_yellow",
             "away_yellow", "home_red", "away_red"]


def load_stats(fill_path: Path | None = None, sidecar_path: Path | None = None) -> pd.DataFrame:
    """API-Football per-fixture counts: the fullstats fill, plus the side-stats sidecar for fixtures the
    fill does not hold (the fill wins on an overlap; nothing is averaged or combined)."""
    fill = pd.read_parquet(fill_path or FILL, columns=SIDE_COLS)
    fill = fill.drop_duplicates("fixture_id").assign(stats_source="fullstats_fill")
    sp = Path(sidecar_path or SIDECAR)
    if sp.exists():
        side = pd.read_parquet(sp)
        side = side[side["answered"].eq(True)].sort_values("fetched_at").drop_duplicates("fixture_id", keep="last")
        side = side[~side["fixture_id"].isin(fill["fixture_id"])][SIDE_COLS].assign(stats_source="side_stats")
        fill = pd.concat([fill, side], ignore_index=True)
    return fill.rename(columns={
        "home_id": "fill_home_id", "away_id": "fill_away_id",
        "home_corners": "af_hc", "away_corners": "af_ac", "home_yellow": "af_hy", "away_yellow": "af_ay",
        "home_red": "af_hr", "away_red": "af_ar"})


def load_af(leagues: list, fixtures_path: Path | None = None, fill_path: Path | None = None,
            sidecar_path: Path | None = None) -> pd.DataFrame:
    """API-Football finished fixtures of `leagues` with their /fixtures/statistics counts (NaN when no
    store holds an answer for the fixture)."""
    fx = pd.read_parquet(fixtures_path or FIXTURES)
    fx = fx[fx["league"].isin(leagues) & fx["status"].isin(FT_STATUSES)].copy()
    fx = fx.drop_duplicates("fixture_id")
    af = fx.merge(load_stats(fill_path, sidecar_path), on="fixture_id", how="left")
    # a stats row whose team ids disagree with the fixture list is not this fixture's answer
    bad = af["fill_home_id"].notna() & ((af["fill_home_id"] != af["home_id"]) | (af["fill_away_id"] != af["away_id"]))
    af.loc[bad, STAT_COLS] = np.nan
    cols = ["fixture_id", "league", "season", "date", "home_id", "away_id", "home_name", "away_name",
            "ft_home", "ft_away", *STAT_COLS, "stats_source"]
    return af[cols].reset_index(drop=True)


def sane(af: pd.DataFrame) -> pd.Series:
    """Every count present, none negative or impossible (SANITY), not all zero (suspected zero-fill)."""
    v = af[STAT_COLS]
    present = v.notna().all(axis=1)
    nonneg = ~(v < 0).any(axis=1)
    ok_c = (af["af_hc"] + af["af_ac"]) <= SANITY["max_match_corners"]
    ok_k = (af["af_hy"] + af["af_ay"] + af["af_hr"] + af["af_ar"]) <= SANITY["max_match_cards"]
    ok_r = (af["af_hr"] <= SANITY["max_team_red"]) & (af["af_ar"] <= SANITY["max_team_red"])
    ok_y = (af["af_hy"] <= SANITY["max_team_yellow"]) & (af["af_ay"] <= SANITY["max_team_yellow"])
    return present & nonneg & ok_c & ok_k & ok_r & ok_y & ~v.eq(0).all(axis=1)


def verify_single(af: pd.DataFrame) -> pd.DataFrame:
    """Single-source leagues: API-Football counts that pass `sane` are used, nothing else."""
    out = af.copy()
    ok = sane(out)
    has = out[STAT_COLS].notna().all(axis=1)
    out["zero_fill_suspect"] = has & out[STAT_COLS].eq(0).all(axis=1)
    out["corners_both"] = has
    out["cards_both"] = has
    out["corners_verified"] = ok
    out["cards_verified"] = ok
    out["home_corners"] = out["af_hc"].where(ok)
    out["away_corners"] = out["af_ac"].where(ok)
    out["home_cards"] = (out["af_hy"] + out["af_hr"]).where(ok)
    out["away_cards"] = (out["af_ay"] + out["af_ar"]).where(ok)
    return out


def crosswalk(fd: pd.DataFrame, af: pd.DataFrame) -> tuple[dict, dict]:
    """{league: {fd name: AF team id}} learned from score anchors (see module doc) + an audit dict."""
    a = af.rename(columns={"ft_home": "g_h", "ft_away": "g_a"})
    f = fd.rename(columns={"fd_fthg": "g_h", "fd_ftag": "g_a"}).dropna(subset=["g_h", "g_a"])
    k = ["league", "date", "g_h", "g_a"]
    a_n = a.groupby(k).size().rename("n_af")
    f_n = f.groupby(k).size().rename("n_fd")
    a1 = a.set_index(k).join(a_n)
    a1 = a1[a1["n_af"] == 1].reset_index()
    f1 = f.set_index(k).join(f_n)
    f1 = f1[f1["n_fd"] == 1].reset_index()
    anchors = f1.merge(a1, on=k)
    votes = pd.concat([
        anchors[["league", "fd_home", "home_id"]].set_axis(["league", "name", "team_id"], axis=1),
        anchors[["league", "fd_away", "away_id"]].set_axis(["league", "name", "team_id"], axis=1),
    ])
    cnt = votes.groupby(["league", "name", "team_id"]).size().rename("v").reset_index()
    tot = cnt.groupby(["league", "name"])["v"].transform("sum")
    cnt["share"] = cnt["v"] / tot
    best = cnt.sort_values("v", ascending=False).drop_duplicates(["league", "name"])
    ok = best[(best["v"] >= MIN_VOTES) & (best["share"] >= MIN_SHARE)]
    # one-to-one inside the league: an AF id claimed by two fd names maps neither
    dup = ok.duplicated(["league", "team_id"], keep=False)
    ok = ok[~dup]
    out: dict = {}
    for r in ok.itertuples(index=False):
        out.setdefault(r.league, {})[r.name] = int(r.team_id)
    names = f.groupby("league")["fd_home"].apply(lambda s: set(s)).to_dict()
    audit = {lg: {"anchors": int((anchors["league"] == lg).sum()),
                  "fd_names": len(names.get(lg, set())),
                  "mapped": len(out.get(lg, {})),
                  "unmapped": sorted(names.get(lg, set()) - set(out.get(lg, {})))}
             for lg in sorted(set(fd["league"]))}
    return out, audit


def join(fd: pd.DataFrame, af: pd.DataFrame, xw: dict) -> pd.DataFrame:
    """fd rows -> AF fixtures through the crosswalk (exact date, else a unique +-1 day candidate)."""
    f = fd.copy()
    f["home_id"] = [xw.get(lg, {}).get(n) for lg, n in zip(f["league"], f["fd_home"])]
    f["away_id"] = [xw.get(lg, {}).get(n) for lg, n in zip(f["league"], f["fd_away"])]
    f = f.dropna(subset=["home_id", "away_id"]).copy()
    f[["home_id", "away_id"]] = f[["home_id", "away_id"]].astype("int64")
    f["d"] = pd.to_datetime(f["date"])
    a = af.copy()
    a["d"] = pd.to_datetime(a["date"])
    m = f.merge(a, on=["league", "home_id", "away_id"], suffixes=("_fd", ""))
    m["dd"] = (m["d"] - m["d_fd"]).dt.days.abs()
    m = m[m["dd"] <= 1]
    exact = m[m["dd"] == 0]
    near = m[m["dd"] == 1]
    # a +-1 day candidate counts only when the fd row has no exact match and exactly one near match
    key = ["league", "fd_file", "fd_home", "fd_away", "date_fd"]
    near = near[~near.set_index(key).index.isin(exact.set_index(key).index)]
    near = near[near.groupby(key)["fixture_id"].transform("size") == 1]
    j = pd.concat([exact, near], ignore_index=True)
    # one fd row <-> one fixture, both ways
    j = j[j.groupby(key)["fixture_id"].transform("size") == 1]
    j = j[j.groupby("fixture_id")["fixture_id"].transform("size") == 1]
    return j.drop(columns=["d", "d_fd", "dd"]).reset_index(drop=True)


def verify(j: pd.DataFrame) -> pd.DataFrame:
    """Two-source leagues: per-fixture verification flags + the verified counts (NaN where not verified).
    The API-Football side must also pass `sane`."""
    out = j.copy()
    has_af = out[STAT_COLS].notna().all(axis=1)
    out["zero_fill_suspect"] = has_af & out[STAT_COLS].eq(0).all(axis=1)
    is_sane = sane(out)
    c_ok = is_sane & out[["fd_hc", "fd_ac"]].notna().all(axis=1) & \
        (out["af_hc"] == out["fd_hc"]) & (out["af_ac"] == out["fd_ac"])
    fd_cards = out[["fd_hy", "fd_ay", "fd_hr", "fd_ar"]].notna().all(axis=1)

    def team_ok(afy, fdy, afr, fdr):
        red_eq = afr == fdr
        diff = afy - fdy
        no_red = (fdr == 0) & (diff == 0)
        with_red = (fdr > 0) & (diff >= 0) & (diff <= 2 * fdr)
        return red_eq & (no_red | with_red)

    k_ok = is_sane & fd_cards & \
        team_ok(out["af_hy"], out["fd_hy"], out["af_hr"], out["fd_hr"]) & \
        team_ok(out["af_ay"], out["fd_ay"], out["af_ar"], out["fd_ar"])
    out["corners_both"] = has_af & out[["fd_hc", "fd_ac"]].notna().all(axis=1)
    out["cards_both"] = has_af & fd_cards
    out["corners_verified"] = c_ok
    out["cards_verified"] = k_ok
    out["home_corners"] = out["af_hc"].where(c_ok)
    out["away_corners"] = out["af_ac"].where(c_ok)
    out["home_cards"] = (out["af_hy"] + out["af_hr"]).where(k_ok)
    out["away_cards"] = (out["af_ay"] + out["af_ar"]).where(k_ok)
    return out


FLAGS = ("corners_both", "cards_both", "corners_verified", "cards_verified", "zero_fill_suspect")
KEEP = ["fixture_id", *FLAGS, "home_corners", "away_corners", "home_cards", "away_cards"]


def build(leagues: list, *, single_source: dict | None = None, fixtures_path=None, fill_path=None,
          fd_dir=None, sidecar_path=None) -> tuple[pd.DataFrame, dict]:
    """(verified match table, audit). `leagues` are two-source (football-data + API-Football);
    `single_source` maps a league to its window start ('YYYY-MM-DD') under the API-Football single
    certified source rule. The table holds every finished AF fixture of those leagues (also ones without
    usable stats, with NaN counts) so the as-of features see a team's real fixture sequence and count
    the matches whose statistics are not usable. Column `basis`: 'two_source' / 'single_source'."""
    single_source = dict(single_source or {})
    two = [lg for lg in leagues if lg not in single_source]
    allg = two + list(single_source)
    af = load_af(allg, fixtures_path, fill_path, sidecar_path)
    parts, audit = [], {}
    if two:
        fd = load_fd_raw(two, fd_dir)
        af2 = af[af["league"].isin(two)]
        xw, xaudit = crosswalk(fd, af2)
        j = verify(join(fd, af2, xw))
        t2 = af2.merge(j[KEEP], on="fixture_id", how="left")
        for c in FLAGS:
            t2[c] = t2[c].eq(True)                 # a fixture football-data does not hold is not verified
        t2["fd_matched"] = t2["fixture_id"].isin(j["fixture_id"])
        t2["basis"] = "two_source"
        parts.append(t2)
        first_fd = fd.groupby("league")["date"].min().to_dict()
        for lg in two:
            t = t2[t2["league"] == lg]
            w = t[t["date"] >= first_fd.get(lg, "9999")]          # the fd_raw window (2022/23 on)
            m = w[w["fd_matched"]]
            audit[lg] = {
                "basis": "two_source",
                "fd_window_from": first_fd.get(lg),
                "af_fixtures_in_window": int(len(w)),
                "fd_rows": int((fd["league"] == lg).sum()),
                "matched": int(len(m)),
                "matched_pct": round(100 * len(m) / len(w), 2) if len(w) else None,
                "corners_both": int(m["corners_both"].sum()),
                "corners_verified": int(m["corners_verified"].sum()),
                "cards_both": int(m["cards_both"].sum()),
                "cards_verified": int(m["cards_verified"].sum()),
                "zero_fill_suspect": int(m["zero_fill_suspect"].sum()),
                "crosswalk": xaudit.get(lg, {}),
            }
    if single_source:
        afs = verify_single(af[af["league"].isin(list(single_source))])
        afs["fd_matched"] = False
        afs["basis"] = "single_source"
        parts.append(afs[t2.columns] if two else afs)
        for lg, since in single_source.items():
            w = afs[(afs["league"] == lg) & (afs["date"] >= since)]
            by_season = (w.assign(y=w["date"].str[:4]).groupby("y")
                         .apply(lambda g: round(100 * g["corners_both"].mean(), 1), include_groups=False).to_dict())
            audit[lg] = {
                "basis": "single_source",
                "fd_window_from": since,
                "af_fixtures_in_window": int(len(w)),
                "with_stats": int(w["corners_both"].sum()),
                "with_stats_pct": round(100 * w["corners_both"].mean(), 2) if len(w) else None,
                "with_stats_pct_by_year": by_season,
                "corners_both": int(w["corners_both"].sum()),
                "corners_verified": int(w["corners_verified"].sum()),
                "cards_both": int(w["cards_both"].sum()),
                "cards_verified": int(w["cards_verified"].sum()),
                "failed_sanity": int((w["corners_both"] & ~w["corners_verified"]).sum()),
                "zero_fill_suspect": int(w["zero_fill_suspect"].sum()),
            }
    table = pd.concat(parts, ignore_index=True) if parts else af.iloc[0:0]
    table = table.sort_values(["league", "date", "fixture_id"]).reset_index(drop=True)
    return table, audit
