"""Team-information features for the EARLY team-goals tip, from the data the project collected
(owner 2026-11-15: "we collect it already... this is why we collect live data").

ONE function computes the block for training AND serving:

    info_features(rows, store) -> DataFrame (same index as rows) with INFO_FEATURE_COLS

`rows` needs date, league, home_team, away_team (canonical history names); an optional `kickoff_utc`
column wins over the fixture index. `store` is an InfoStore of the collected tables. Training calls it
on the whole frame with the full store; serving calls it on the upcoming fixtures with the live store;
scripts/team_info_parity.py checks that a store TRUNCATED to what was finished before each match gives
the same numbers (0 differences).

TIP TIME. The early tip is made 125..153 h before kickoff. Every feature is computed as of
    T = kickoff - EARLY_TIP_MAX_HOURS (153 h), the EARLIEST moment of the window,
so whatever the feature uses was known at any tip in the window. Kickoff comes from the provider's
fixture index (output/collect/af_ht_scores.parquet, real UTC kickoff). A match not in the index has
no proven kickoff; its lower bound date 00:00 UTC - KICKOFF_UNKNOWN_SLACK_H is used (any time zone).

LEAK RULES (each is tested in tests/test_team_info_features.py by truncation):
  (a) INJURIES. output/collect/injuries.parquet was fetched in hindsight (fetched_at 2026-11-05..09),
      so a list for match F carries additions made after F's kickoff (about 15-19% per Pro). The
      list of F itself is NEVER read. An absence is used only when the player is listed 'Missing
      Fixture' for an EARLIER fixture P of the same team with kickoff(P) + INJURY_AMEND_MARGIN_H <= T
      (P = the team's newest such fixture in an injury-covered league-season, at most
      INJURY_MAX_AGE_DAYS before T). 'Questionable' entries are not absences. A team whose newest
      such fixture is not covered by the provider (injuries_seasons coverage False/empty) gets
      NaN, never 0. Residual risk: an amendment to P's list made more than the margin after P's
      kickoff cannot be detected in a hindsight backfill; live serving reads P's list as fetched
      before T, which is the same definition.
  (b) LINEUPS. Only lineups of fixtures with kickoff < T (finished before the tip). The fixture's
      own confirmed XI (about 1 h before kickoff) belongs to a LATE model, not built here.
  (c) xG. Rolling per team per league from fixtures with kickoff < T (stricter than "before the
      match date").
  Coaches: a career counts from the END of its start month (the provider gives month dates), so a
  coach appointed later in the month of T is not seen early. Transfers: dated moves in [T-120 d, T).

Missing stays NaN: no lineup history, unmapped club, uncovered injuries, no xG -> NaN. A real zero
(covered list, nobody missing) is 0.
"""
from __future__ import annotations

import glob
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLLECT = ROOT / "output" / "collect"

INFO_VERSION = "team-info-1.0"
EARLY_TIP_MAX_HOURS = 153          # the early tip window is 125..153 h before kickoff
INJURY_AMEND_MARGIN_H = 24         # P's list is used only if P kicked off >= 24 h before T
INJURY_MAX_AGE_DAYS = 21           # an older list is not "the team's current list" -> NaN
KICKOFF_UNKNOWN_SLACK_H = 14       # no kickoff in the index: date 00:00 UTC - 14 h (UTC+14 bound)
IMPORTANCE_WINDOW = 10             # player importance from the team's last 10 lineups before T
XG_WINDOW = 5
TRANSFER_WINDOW_DAYS = 120
CONGESTION_DAYS = 14
DEF_POS = ("G", "D")

BASE = ["info_xi_absent_w", "info_def_absent_w", "info_gk_absent", "info_goal_share_absent",
        "info_xi_strength_ratio", "info_xi_continuity", "info_xg_for5", "info_xg_against5",
        "info_coach_tenure_days", "info_transfers_in_120d", "info_matches_14d"]
INFO_FEATURE_COLS = [f"{s}_{b}" for s in ("home", "away") for b in BASE]
#: columns that depend on each source (coverage audit, live-availability table)
SOURCE_OF = {"info_xi_absent_w": "injuries+lineups", "info_def_absent_w": "injuries+lineups",
             "info_gk_absent": "injuries+lineups", "info_goal_share_absent": "injuries+player_goals",
             "info_xi_strength_ratio": "injuries+lineups", "info_xi_continuity": "lineups",
             "info_xg_for5": "xg", "info_xg_against5": "xg", "info_coach_tenure_days": "coaches",
             "info_transfers_in_120d": "transfers", "info_matches_14d": "fixtures"}


# ── the store ───────────────────────────────────────────────────────────────────────────────

@dataclass
class InfoStore:
    fixtures: pd.DataFrame        # fixture_id, league, league_id, season, kickoff(UTC ts), home_id, away_id
    lineups: pd.DataFrame         # fixture_id, team_id, player_id, pos   (starters only)
    injuries: pd.DataFrame        # fixture_id, team_id, player_id        ('Missing Fixture' only)
    injury_cover: set             # {(league_id, season)} the provider covers
    xg: pd.DataFrame              # fixture_id, hxg, axg
    goals: pd.DataFrame           # fixture_id, team_id, player_id, goals
    coaches: pd.DataFrame         # team_id, start(ts), end(ts or NaT)
    transfers: pd.DataFrame       # team_in_id, date(ts)
    team_ids: dict                # {(league, history name): team_id}
    transfer_cover: set = field(default_factory=set)   # team ids whose transfer list the provider answered
    info: dict = field(default_factory=dict)

    def truncated(self, as_of: pd.Timestamp) -> "InfoStore":
        """What a live collection holds at `as_of`: fixtures finished (kicked off) before it, their
        lineups / injury lists / xG / goals, careers and transfers dated before it. Identity
        (team_ids) and the provider's coverage flags are static."""
        as_of = pd.Timestamp(as_of)
        fx = self.fixtures[self.fixtures["kickoff"] < as_of]
        ids = set(fx["fixture_id"])
        co = self.coaches[self.coaches["start"] < as_of].copy()
        co.loc[co["end"] >= as_of, "end"] = pd.NaT          # not ended yet as far as `as_of` knows
        return InfoStore(fixtures=fx, lineups=self.lineups[self.lineups["fixture_id"].isin(ids)],
                         injuries=self.injuries[self.injuries["fixture_id"].isin(ids)],
                         injury_cover=self.injury_cover, xg=self.xg[self.xg["fixture_id"].isin(ids)],
                         goals=self.goals[self.goals["fixture_id"].isin(ids)], coaches=co,
                         transfers=self.transfers[self.transfers["date"] < as_of],
                         team_ids=self.team_ids, transfer_cover=self.transfer_cover,
                         info=dict(self.info, truncated_at=str(as_of)))


def _utc(s) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def coach_known_from(start: pd.Series) -> pd.Series:
    """The provider dates a career by its month ('2026-04-01' = some day in April): it is known from
    the first day of the NEXT month, so a coach appointed late in the month of T is never seen early."""
    return start + pd.offsets.MonthBegin(1)


def build_team_ids(frame_names: pd.DataFrame | None = None, collect: Path = COLLECT) -> tuple[dict, dict]:
    """{(league, history name): provider team_id} and an audit.

    1. af_history_fullstats_fill.parquet: history rows (history names) already matched to the
       provider fixture (home_id / away_id).
    2. for clubs still unmapped and present in `frame_names` (date, league, home_team, away_team):
       the provider fixture index (af_ht_scores) on the SAME league and date, both clubs resolved by
       src.team_names.resolve inside that day's history names, both onto the same history row.
    A name that ends up with two ids is dropped (never guessed)."""
    pairs = []
    fs = pd.read_parquet(collect / "af_history_fullstats_fill.parquet",
                         columns=["league", "home_team", "away_team", "home_id", "away_id"])
    pairs += [fs[["league", "home_team", "home_id"]].set_axis(["league", "name", "id"], axis=1),
              fs[["league", "away_team", "away_id"]].set_axis(["league", "name", "id"], axis=1)]
    n_resolved = 0
    if frame_names is not None and len(frame_names):
        from src.team_names import resolve
        known = set(zip(pairs[0]["league"], pairs[0]["name"])) | set(zip(pairs[1]["league"], pairs[1]["name"]))
        fr = frame_names[["date", "league", "home_team", "away_team"]].copy()
        fr["d"] = pd.to_datetime(fr["date"]).dt.strftime("%Y-%m-%d")
        need = fr[[(lg, h) not in known or (lg, a) not in known
                   for lg, h, a in zip(fr["league"], fr["home_team"], fr["away_team"])]]
        if len(need):
            ht = pd.read_parquet(collect / "af_ht_scores.parquet",
                                 columns=["league", "date", "home_id", "home_name", "away_id", "away_name"])
            ht["d"] = ht["date"].astype(str).str[:10]
            day = {k: g for k, g in fr.groupby(["league", "d"])}
            got = []
            for (lg, d), g in ht[ht.set_index(["league", "d"]).index.isin(
                    set(zip(need["league"], need["d"])))].groupby(["league", "d"]):
                f = day.get((lg, d))
                if f is None:
                    continue
                hp, ap = sorted(set(f["home_team"])), sorted(set(f["away_team"]))
                rows = set(zip(f["home_team"], f["away_team"]))
                for r in g.itertuples(index=False):
                    h, a = resolve(str(r.home_name), hp), resolve(str(r.away_name), ap)
                    if h and a and (h, a) in rows:
                        got += [(lg, h, int(r.home_id)), (lg, a, int(r.away_id))]
            if got:
                n_resolved = len(got) // 2
                pairs.append(pd.DataFrame(got, columns=["league", "name", "id"]))
    m = pd.concat(pairs, ignore_index=True).dropna()
    m["id"] = m["id"].astype("int64")
    nun = m.groupby(["league", "name"])["id"].nunique()
    amb = nun[nun > 1]
    m = m.drop_duplicates(["league", "name"])
    m = m[~m.set_index(["league", "name"]).index.isin(amb.index)]
    out = {(lg, n): int(i) for lg, n, i in zip(m["league"], m["name"], m["id"])}
    return out, {"names_mapped": len(out), "ambiguous_dropped": int(len(amb)),
                 "fixtures_resolved_from_index": n_resolved}


def load_store(frame_names: pd.DataFrame | None = None, collect: Path = COLLECT) -> InfoStore:
    """Every collected table the block reads (committed files only, no network)."""
    collect = Path(collect)
    fx = [pd.read_parquet(collect / "af_ht_scores.parquet",
                          columns=["fixture_id", "league", "league_id", "season", "kickoff", "home_id", "away_id"])]
    for f, cols in (("national_cup_fixtures.parquet", ["fixture_id", "league_id", "season", "kickoff", "home_id", "away_id"]),
                    ("refetch_extra_fixtures.parquet", ["fixture_id", "league_id", "season", "date", "home_id", "away_id"])):
        p = collect / f
        if p.exists():
            d = pd.read_parquet(p, columns=cols).rename(columns={"date": "kickoff"})
            d["league"] = None
            fx.append(d)
    fixtures = pd.concat(fx, ignore_index=True)
    fixtures["kickoff"] = _utc(fixtures["kickoff"])
    fixtures["season"] = fixtures["season"].astype(str)
    fixtures = (fixtures.dropna(subset=["kickoff", "home_id", "away_id"])
                .drop_duplicates("fixture_id", keep="first").reset_index(drop=True))
    for c in ("home_id", "away_id", "league_id", "fixture_id"):
        fixtures[c] = fixtures[c].astype("int64")

    parts = []
    for p in sorted(glob.glob(str(collect / "lineups" / "players_*.parquet"))):
        d = pd.read_parquet(p, columns=["fixture_id", "team_id", "role", "player_id", "pos"])
        parts.append(d[d["role"] == "start"].drop(columns="role"))
    lineups = pd.concat(parts, ignore_index=True).dropna(subset=["player_id"])
    lineups["player_id"] = lineups["player_id"].astype("int64")

    inj = pd.read_parquet(collect / "injuries.parquet", columns=["fixture_id", "team_id", "player_id", "type"])
    inj = inj[inj["type"] == "Missing Fixture"].drop(columns="type").drop_duplicates()
    cov = pd.read_parquet(collect / "injuries_seasons.parquet")
    cover = {(int(l), str(s)) for l, s, ok, st in zip(cov["league_id"], cov["season"], cov["coverage_injuries"],
                                                       cov["status"]) if bool(ok) and st == "ok"}

    xg = pd.read_parquet(collect / "af_fixture_stats.parquet", columns=["fixture_id", "HXG", "AXG"])
    fill = pd.read_parquet(collect / "af_history_fullstats_fill.parquet", columns=["fixture_id", "home_xg", "away_xg"])
    parts = [xg.rename(columns={"HXG": "hxg", "AXG": "axg"}),
             fill.rename(columns={"home_xg": "hxg", "away_xg": "axg"})]
    # the LIVE path: af_history_extend.yml (daily on main) writes provider rows with fixture_id and
    # xG into output/af_history.parquet (2026-11-15: 1,486 fixtures, every one also in the tables
    # above and equal there); rows without a fixture_id are not provider rows and are not read
    afh = collect.parent / "af_history.parquet"
    if afh.exists():
        h = pd.read_parquet(afh, columns=["fixture_id", "HXG", "AXG"]).dropna(subset=["fixture_id"])
        h["fixture_id"] = h["fixture_id"].astype("int64")
        parts.append(h.rename(columns={"HXG": "hxg", "AXG": "axg"}))
    xg = pd.concat(parts, ignore_index=True)
    xg = xg.dropna(subset=["hxg", "axg"]).drop_duplicates("fixture_id", keep="first")

    goals = pd.read_parquet(collect / "player_history_refetch.parquet",
                            columns=["fixture_id", "team_id", "player_id", "goals"]).dropna(subset=["team_id"])
    goals["team_id"] = goals["team_id"].astype("int64")

    co = pd.read_parquet(collect / "coach_careers.parquet", columns=["team_id", "start", "end"])
    co["start"] = _utc(co["start"])
    co["end"] = _utc(co["end"])
    # a month-dated start is known from the END of that month (rule above)
    co["start"] = coach_known_from(co["start"])
    co = co.dropna(subset=["start"])

    tr = pd.read_parquet(collect / "transfers.parquet", columns=["team_in_id", "date"]).dropna()
    tr["team_in_id"] = tr["team_in_id"].astype("int64")
    tr["date"] = _utc(tr["date"])
    tr = tr.dropna()

    tt = pd.read_parquet(collect / "transfers_teams.parquet", columns=["team_id"])
    team_ids, audit = build_team_ids(frame_names, collect)
    return InfoStore(fixtures=fixtures, lineups=lineups, injuries=inj, injury_cover=cover, xg=xg,
                     goals=goals, coaches=co, transfers=tr, team_ids=team_ids,
                     transfer_cover=set(tt["team_id"].astype("int64").tolist()),
                     info={"version": INFO_VERSION, "team_ids": audit})


# ── the features ────────────────────────────────────────────────────────────────────────────

class _Slices:
    """(fixture_id, team_id) -> the rows of `df` as numpy slices (a sort + one pass; no groupby)."""

    def __init__(self, df: pd.DataFrame, cols: tuple):
        d = df.sort_values(["fixture_id", "team_id"], kind="mergesort")
        f = d["fixture_id"].to_numpy()
        t = d["team_id"].to_numpy()
        self.cols = {c: d[c].to_numpy() for c in cols}
        if not len(d):
            self.span = {}
            return
        brk = np.flatnonzero((f[1:] != f[:-1]) | (t[1:] != t[:-1])) + 1
        starts = np.r_[0, brk]
        ends = np.r_[brk, len(d)]
        self.span = dict(zip(zip(f[starts].tolist(), t[starts].tolist()), zip(starts.tolist(), ends.tolist())))

    def __contains__(self, key) -> bool:
        return key in self.span

    def get(self, key, col):
        a, b = self.span[key]
        return self.cols[col][a:b]


class _Index:
    """Per-team timelines of the store, built once per info_features call."""

    def __init__(self, s: InfoStore):
        fx = s.fixtures
        side = pd.concat([fx[["fixture_id", "kickoff", "league_id", "season", "home_id"]].rename(columns={"home_id": "team"}).assign(home=True),
                          fx[["fixture_id", "kickoff", "league_id", "season", "away_id"]].rename(columns={"away_id": "team"}).assign(home=False)])
        side = side.sort_values(["team", "kickoff", "fixture_id"], kind="mergesort")
        self.kick = {}
        for t, g in side.groupby("team", sort=False):
            self.kick[t] = (g["kickoff"].to_numpy(dtype="datetime64[ns]"), g["fixture_id"].to_numpy(),
                            g["home"].to_numpy(), g["league_id"].to_numpy(), g["season"].to_numpy())
        self.lu = _Slices(s.lineups, ("player_id", "pos"))
        self.inj = _Slices(s.injuries, ("player_id",))
        self.cover = s.injury_cover
        self.xg = dict(zip(s.xg["fixture_id"], zip(s.xg["hxg"].astype(float), s.xg["axg"].astype(float))))
        self.goals = _Slices(s.goals, ("player_id", "goals"))
        self.goal_fx = set(s.goals["fixture_id"])
        self.co = {t: (g["start"].to_numpy(dtype="datetime64[ns]"), g["end"].to_numpy(dtype="datetime64[ns]"))
                   for t, g in s.coaches.sort_values("start").groupby("team_id")}
        self.transfer_cover = s.transfer_cover
        self.tr = {t: np.sort(g["date"].to_numpy(dtype="datetime64[ns]")) for t, g in s.transfers.groupby("team_in_id")}
        fxk = s.fixtures[["fixture_id", "kickoff"]]
        self.fx_kick = dict(zip(fxk["fixture_id"], fxk["kickoff"].to_numpy(dtype="datetime64[ns]")))
        self.fx_by_pair = {}
        f = s.fixtures.assign(d=s.fixtures["kickoff"].dt.strftime("%Y-%m-%d"))
        for fid, h, a, d, k in zip(f["fixture_id"], f["home_id"], f["away_id"], f["d"],
                                   f["kickoff"].to_numpy(dtype="datetime64[ns]")):
            self.fx_by_pair[(h, a, d)] = k


def _team_block(ix: _Index, team, T: np.datetime64) -> dict:
    out = dict.fromkeys(BASE, np.nan)
    if team is None:
        return out
    _club_facts(ix, team, T, out)                 # coach / transfers: dated, no fixture needed
    if team not in ix.kick:
        return out
    kick, fids, home, lids, seasons = ix.kick[team]
    n_before = int(np.searchsorted(kick, T, side="left"))          # fixtures with kickoff < T
    # congestion: matches played in the 14 days before T (the team's fixture index is complete for
    # its league and the collected cups; a team with no fixture in the index at all stays NaN)
    lo = int(np.searchsorted(kick, T - np.timedelta64(CONGESTION_DAYS, "D"), side="left"))
    out["info_matches_14d"] = float(n_before - lo) if n_before > 0 else np.nan

    # (b) lineups of finished fixtures: importance = share of the last IMPORTANCE_WINDOW XIs started
    lus = [(fids[i], (ix.lu.get((fids[i], team), "player_id").tolist(), ix.lu.get((fids[i], team), "pos").tolist()))
           for i in range(n_before - 1, -1, -1) if (fids[i], team) in ix.lu]
    lus = lus[:IMPORTANCE_WINDOW]
    w, pos_of = {}, {}
    if lus:
        for _f, (pids, pos) in lus:
            for p, q in zip(pids, pos):
                w[p] = w[p] + 1.0 / len(lus) if p in w else 1.0 / len(lus)    # start share, counted
                pos_of.setdefault(p, []).append(q)
        pos_of = {p: max(set(v), key=lambda x: (v.count(x), str(x))) for p, v in pos_of.items()}
        reg = sorted(w, key=lambda p: (-w[p], p))[:11]
        last = set(lus[0][1][0])
        out["info_xi_continuity"] = len(last & set(reg)) / 11.0

    # (a) the newest injury list of a covered earlier fixture, kicked off >= margin before T
    lim = T - np.timedelta64(INJURY_AMEND_MARGIN_H, "h")
    j = int(np.searchsorted(kick, lim, side="right")) - 1
    absent = None
    while j >= 0 and kick[j] >= T - np.timedelta64(INJURY_MAX_AGE_DAYS, "D"):
        if (int(lids[j]), str(seasons[j])) in ix.cover:
            absent = set(ix.inj.get((fids[j], team), "player_id").tolist()) if (fids[j], team) in ix.inj else set()
            break
        j -= 1
    if absent is not None and w:
        reg = sorted(w, key=lambda p: (-w[p], p))[:11]
        out["info_xi_absent_w"] = float(sum(w[p] for p in absent if p in w))
        out["info_def_absent_w"] = float(sum(w[p] for p in absent if p in w and pos_of.get(p) in DEF_POS))
        gks = [p for p in reg if pos_of.get(p) == "G"]
        out["info_gk_absent"] = float(gks[0] in absent) if gks else np.nan
        avail = sorted((w[p] for p in w if p not in absent), reverse=True)[:11]
        out["info_xi_strength_ratio"] = float(sum(avail) / sum(w[p] for p in reg))
        gfx = [f for f, _ in lus if f in ix.goal_fx]
        if gfx:
            tot = {}
            for f in gfx:
                if (f, team) not in ix.goals:
                    continue
                for p, g in zip(ix.goals.get((f, team), "player_id").tolist(), ix.goals.get((f, team), "goals").astype(float).tolist()):
                    if np.isfinite(g):
                        tot[p] = tot[p] + g if p in tot else g
            s = sum(tot.values())
            # an absentee with no goal row in the window scored none of the team's goals there
            out["info_goal_share_absent"] = float(sum(tot[p] for p in absent if p in tot) / s) if s > 0 else np.nan

    # (c) rolling xG for / against, fixtures with kickoff < T
    xf, xa = [], []
    for i in range(n_before - 1, -1, -1):
        v = ix.xg.get(fids[i])
        if v is None:
            continue
        xf.append(v[0] if home[i] else v[1])
        xa.append(v[1] if home[i] else v[0])
        if len(xf) == XG_WINDOW:
            break
    if xf:
        out["info_xg_for5"] = float(np.mean(xf))
        out["info_xg_against5"] = float(np.mean(xa))

    return out


def _club_facts(ix: _Index, team, T: np.datetime64, out: dict) -> None:
    # coach tenure at T: the newest career known before T (start, from the end of its start month,
    # < T) that has not ended before T. Order-free: two careers of the same month (an interim and
    # the appointment) give the same start, so no tie can change the value.
    if team in ix.co:
        st, en = ix.co[team]
        run = (st < T) & (np.isnat(en) | (en >= T))
        if run.any():
            out["info_coach_tenure_days"] = float((T - st[run].max()) / np.timedelta64(1, "D"))
    # transfers in, dated [T - 120 d, T)
    if team in ix.transfer_cover:                 # the provider answered for this club: 0 is real
        d = ix.tr.get(team, np.array([], dtype="datetime64[ns]"))
        out["info_transfers_in_120d"] = float(np.searchsorted(d, T, side="left")
                                              - np.searchsorted(d, T - np.timedelta64(TRANSFER_WINDOW_DAYS, "D"), side="left"))


def kickoff_of(rows: pd.DataFrame, store: InfoStore, ix: _Index | None = None) -> np.ndarray:
    """Proven kickoff (provider fixture index, or rows.kickoff_utc) else the conservative bound."""
    ix = ix or _Index(store)
    d = pd.to_datetime(rows["date"]).dt.strftime("%Y-%m-%d")
    out = []
    ko = _utc(rows["kickoff_utc"]) if "kickoff_utc" in rows.columns else None
    for i, (lg, h, a, dd) in enumerate(zip(rows["league"], rows["home_team"], rows["away_team"], d)):
        if ko is not None and pd.notna(ko.iat[i]):
            out.append(ko.iat[i].tz_convert(None).to_datetime64())
            continue
        hid, aid = store.team_ids.get((lg, h)), store.team_ids.get((lg, a))
        k = None
        if hid is not None and aid is not None:
            # the history date is the LOCAL match date; the index is keyed on the UTC date, which
            # is the same day or one day either side (the same pairing never recurs within days)
            day = np.datetime64(dd)
            for off in (0, 1, -1):
                k = ix.fx_by_pair.get((hid, aid, str(day + np.timedelta64(off, "D"))))
                if k is not None:
                    break
        if k is None:
            k = np.datetime64(dd) - np.timedelta64(KICKOFF_UNKNOWN_SLACK_H, "h")
        out.append(k)
    return np.array(out, dtype="datetime64[ns]")


def info_features(rows: pd.DataFrame, store: InfoStore, kickoffs: np.ndarray | None = None) -> pd.DataFrame:
    """The block for `rows` (index kept). See the module docstring for every rule."""
    ix = _Index(store)
    k = kickoff_of(rows, store, ix) if kickoffs is None else np.asarray(kickoffs, dtype="datetime64[ns]")
    T = k - np.timedelta64(EARLY_TIP_MAX_HOURS, "h")
    recs = []
    for i, (lg, h, a) in enumerate(zip(rows["league"], rows["home_team"], rows["away_team"])):
        r = {}
        for side, name in (("home", h), ("away", a)):
            b = _team_block(ix, store.team_ids.get((lg, name)), T[i])
            r.update({f"{side}_{c}": v for c, v in b.items()})
        recs.append(r)
    return pd.DataFrame(recs, index=rows.index, columns=INFO_FEATURE_COLS).astype(float)


def attach_info_features(feat: pd.DataFrame, store: InfoStore | None = None) -> tuple[pd.DataFrame, dict]:
    store = store or load_store(feat)
    blk = info_features(feat, store)
    out = feat.drop(columns=[c for c in INFO_FEATURE_COLS if c in feat.columns]).join(blk)
    return out, dict(store.info)


# ── serving hook (src/predict.py) ───────────────────────────────────────────────────────────

def _feature_cols(obj) -> set:
    """Every feature_cols list anywhere in a payload (track-split payloads nest them)."""
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "feature_cols" and isinstance(v, (list, tuple)):
                out |= set(v)
            elif isinstance(v, dict):
                out |= _feature_cols(v)
    return out


def payloads_read_info(payloads) -> bool:
    cols = set()
    for p in payloads:
        cols |= _feature_cols(p)
    return bool({c.removesuffix("__missing") for c in cols} & set(INFO_FEATURE_COLS))


def attach_for_payloads(feat: pd.DataFrame, payloads, store: InfoStore | None = None) -> pd.DataFrame:
    """predict.py: the block on the upcoming fixtures, ONLY when a loaded payload reads it (no
    production payload does on 2026-11-15, so this is a no-op until the owner installs one). Same
    function as training; the store is the live collection as committed. Kickoffs come from the
    fixtures' kickoff_utc. A failure leaves the block missing (NaN), never a stand-in."""
    if not payloads_read_info([p for p in payloads if p]):
        return feat
    out = feat.drop(columns=[c for c in INFO_FEATURE_COLS if c in feat.columns])
    try:
        blk = info_features(out, store or load_store(out))
    except Exception as e:                                             # noqa: BLE001
        import logging
        logging.getLogger(__name__).error(f"team info block not built ({type(e).__name__}: {e}): missing")
        blk = pd.DataFrame(np.nan, index=out.index, columns=INFO_FEATURE_COLS)
    return out.join(blk)
